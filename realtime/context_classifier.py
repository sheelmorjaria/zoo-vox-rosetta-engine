#!/usr/bin/env python3
"""
Context Classifier - Direction 4: Semantic Alignment

This module implements a supervised classifier for behavioral context inference,
replacing the brittle rule-based system in InteractionAgent.

The classifier uses the full 112D feature vector to predict context labels
with confidence scores, enabling more accurate context detection.

Key Features:
- MLP neural network for non-linear decision boundaries
- Support for binary and multi-class classification
- Confidence scoring for predictions
- Model persistence (pickle/joblib)
- Integration with InteractionAgent

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextClassifier:
    """
    Supervised classifier for behavioral context inference.

    Uses an MLP (Multi-Layer Perceptron) to learn non-linear boundaries
    between behavioral contexts using 112D feature vectors.

    Usage:
        classifier = ContextClassifier(model_type="mlp")
        classifier.train(features, labels)  # labels can be strings or integers
        context, confidence = classifier.predict(feature_vector_112d)
    """

    def __init__(
        self,
        model_type: str = "mlp",
        hidden_layers: Tuple[int, ...] = (64, 32),
        random_state: Optional[int] = None,
    ):
        """
        Initialize the ContextClassifier.

        Args:
            model_type: Type of model ("mlp" for multi-layer perceptron)
            hidden_layers: Hidden layer sizes for MLP
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.hidden_layers = hidden_layers
        self.random_state = random_state

        self.model: Optional[BaseEstimator] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.class_names: List[str] = []

        # Training metadata
        self.training_timestamp: Optional[float] = None
        self.n_samples: int = 0
        self.feature_dim: int = 112
        self.data_hash: Optional[str] = None

    def train(
        self,
        features: np.ndarray,
        labels: Union[np.ndarray, List[str]],
    ) -> None:
        """
        Train the classifier on 112D features with context labels.

        Args:
            features: Feature matrix (n_samples, 112)
            labels: Context labels (strings or integers)

        Raises:
            ValueError: If features don't have 112 dimensions
        """
        if features.shape[1] != 112:
            raise ValueError(f"Expected 112D features, got {features.shape[1]}D")

        self.n_samples = len(features)
        self.feature_dim = features.shape[1]

        # Compute data hash for tracking
        self.data_hash = self._compute_hash(features, labels)

        # Encode labels
        self.label_encoder = LabelEncoder()
        if isinstance(labels, list):
            labels = np.array(labels)

        # Store class names for later prediction
        unique_labels = np.unique(labels)
        self.class_names = [str(l) for l in unique_labels]

        y_encoded = self.label_encoder.fit_transform(labels)

        # Validate class counts and determine if early stopping is safe
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_class_count = int(np.min(counts))

        # Early stopping with validation split requires:
        # - At least 10 samples total (for validation_fraction=0.2)
        # - Each class must have at least 2 samples (for stratified split)
        use_early_stopping = (
            self.n_samples >= 10 and
            min_class_count >= 2
        )

        if not use_early_stopping:
            if self.n_samples < 10:
                logger.warning(
                    f"Dataset too small ({self.n_samples} < 10 samples) for "
                    f"early stopping. Disabling validation split."
                )
            if min_class_count < 2:
                logger.warning(
                    f"Some classes have <2 samples (min={min_class_count}). "
                    f"Disabling early stopping to avoid validation split failure."
                )

        # Create and train model
        if self.model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layers,
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size=32,
                learning_rate="adaptive",
                max_iter=200,
                random_state=self.random_state,
                early_stopping=use_early_stopping,
                validation_fraction=0.2 if use_early_stopping else 0.0,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Train
        logger.info(f"Training {self.model_type} on {self.n_samples} samples...")
        self.model.fit(features, y_encoded)

        self.training_timestamp = time.time()

        # Log training results
        train_accuracy = self.model.score(features, y_encoded)
        logger.info(
            f"Training complete: {len(self.class_names)} classes, "
            f"accuracy={train_accuracy:.2%}"
        )

    def predict(self, features_112d: np.ndarray) -> Tuple[str, float]:
        """
        Predict context and confidence for a single 112D feature vector.

        Args:
            features_112d: 112D feature vector

        Returns:
            Tuple of (context_label, confidence)

        Raises:
            RuntimeError: If model is not trained
            ValueError: If features don't have 112 dimensions
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if features_112d.shape != (112,):
            raise ValueError(f"Expected shape (112,), got {features_112d.shape}")

        # Reshape for sklearn
        features_2d = features_112d.reshape(1, -1)

        # Get prediction probabilities
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features_2d)[0]
            predicted_idx = int(self.model.predict(features_2d)[0])
            confidence = float(proba[predicted_idx])
        else:
            # Fallback for models without predict_proba
            predicted_idx = int(self.model.predict(features_2d)[0])
            confidence = 1.0  # No confidence available

        # Convert to class name
        context = self.class_names[predicted_idx]

        return context, confidence

    def predict_batch(self, features: np.ndarray) -> List[str]:
        """
        Predict contexts for multiple feature vectors.

        Args:
            features: Feature matrix (n_samples, 112)

        Returns:
            List of predicted context labels
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if features.shape[1] != 112:
            raise ValueError(f"Expected 112D features, got {features.shape[1]}D")

        indices = self.model.predict(features)
        return [self.class_names[int(i)] for i in indices]

    def predict_with_confidence_batch(
        self, features: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Predict contexts and confidences for multiple feature vectors.

        Args:
            features: Feature matrix (n_samples, 112)

        Returns:
            List of (context_label, confidence) tuples
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if features.shape[1] != 112:
            raise ValueError(f"Expected 112D features, got {features.shape[1]}D")

        results = []
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(features)
            indices = self.model.predict(features)

            for idx, proba in zip(indices, probas):
                context = self.class_names[int(idx)]
                confidence = float(proba[int(idx)])
                results.append((context, confidence))
        else:
            indices = self.model.predict(features)
            for idx in indices:
                context = self.class_names[int(idx)]
                results.append((context, 1.0))

        return results

    def save(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: File path to save the model
        """
        if self.model is None:
            raise RuntimeError("Cannot save untrained model")

        model_data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "class_names": self.class_names,
            "model_type": self.model_type,
            "hidden_layers": self.hidden_layers,
            "random_state": self.random_state,
            "training_timestamp": self.training_timestamp,
            "n_samples": self.n_samples,
            "feature_dim": self.feature_dim,
            "data_hash": self.data_hash,
        }

        path_obj = Path(path)

        # Use joblib for sklearn models (more efficient)
        if path_obj.suffix == ".joblib":
            joblib.dump(model_data, path)
        else:
            # Fallback to pickle
            with open(path, "wb") as f:
                pickle.dump(model_data, f)

        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str) -> "ContextClassifier":
        """
        Load a trained model from disk.

        Args:
            path: File path to load the model from

        Returns:
            Loaded ContextClassifier instance
        """
        path_obj = Path(path)

        # Try joblib first
        if path_obj.suffix == ".joblib":
            model_data = joblib.load(path)
        else:
            with open(path, "rb") as f:
                model_data = pickle.load(f)

        # Create new instance and restore state
        classifier = cls(
            model_type=model_data["model_type"],
            hidden_layers=model_data["hidden_layers"],
            random_state=model_data["random_state"],
        )

        classifier.model = model_data["model"]
        classifier.label_encoder = model_data["label_encoder"]
        classifier.class_names = model_data["class_names"]
        classifier.training_timestamp = model_data.get("training_timestamp")
        classifier.n_samples = model_data.get("n_samples", 0)
        classifier.feature_dim = model_data.get("feature_dim", 112)
        classifier.data_hash = model_data.get("data_hash")

        logger.info(f"Loaded model from {path}")

        return classifier

    def get_metadata(self) -> Dict:
        """
        Get model metadata.

        Returns:
            Dictionary with model information
        """
        return {
            "model_type": self.model_type,
            "hidden_layers": self.hidden_layers,
            "n_classes": len(self.class_names),
            "class_names": self.class_names,
            "feature_dim": self.feature_dim,
            "n_samples": self.n_samples,
            "training_timestamp": self.training_timestamp,
            "data_hash": self.data_hash,
        }

    def _compute_hash(
        self, features: np.ndarray, labels: np.ndarray
    ) -> str:
        """Compute hash of training data for tracking."""
        data_bytes = features.tobytes() + np.array(labels).tobytes()
        return hashlib.md5(data_bytes).hexdigest()[:8]


# =============================================================================
# Context Dataset (Weak Supervision)
# =============================================================================


class ContextDataset:
    """
    Dataset for weak supervision and training.

    Provides methods for generating pseudo-labels from temporal co-occurrence
    patterns and loading manually annotated data.
    """

    @staticmethod
    def from_temporal_cooccurrence(
        segments: List,
        window_ms: float = 500.0,
    ) -> "ContextDataset":
        """
        Generate pseudo-labels from temporal clustering.

        Segments occurring within the specified window are assigned
        the same pseudo-label, assuming they share context.

        Args:
            segments: List of segments with timestamp and features
            window_ms: Time window for co-occurrence (milliseconds)

        Returns:
            ContextDataset with pseudo-labels
        """
        if not segments:
            raise ValueError("No segments provided")

        # Sort by timestamp
        sorted_segments = sorted(segments, key=lambda s: s.get("timestamp", 0))

        # Group by temporal proximity
        groups = []
        current_group = [sorted_segments[0]]

        for seg in sorted_segments[1:]:
            seg_time = seg.get("timestamp", 0)
            group_time = current_group[0].get("timestamp", 0)

            if seg_time - group_time <= window_ms / 1000.0:  # Convert ms to seconds
                current_group.append(seg)
            else:
                groups.append(current_group)
                current_group = [seg]

        if current_group:
            groups.append(current_group)

        # Assign pseudo-labels
        pseudo_labels = []
        for group in groups:
            label = f"context_{len(pseudo_labels)}"
            for _ in group:
                pseudo_labels.append(label)

        # Extract features
        features = np.array([s.get("features", np.zeros(112)) for s in sorted_segments])

        return ContextDataset(features=features, labels=np.array(pseudo_labels))

    @staticmethod
    def from_manual_labels(manifest_path: str) -> "ContextDataset":
        """
        Load Level 1 annotated data from JSON manifest.

        Args:
            manifest_path: Path to JSON file with manual labels

        Returns:
            ContextDataset with manual labels
        """
        import json

        with open(manifest_path, "r") as f:
            data = json.load(f)

        features = np.array([s["features_112d"] for s in data["segments"]])
        labels = np.array([s["context"] for s in data["segments"]])

        return ContextDataset(features=features, labels=labels)

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.

        Args:
            features: Feature matrix (n_samples, 112)
            labels: Context labels
        """
        self.features = features
        self.labels = labels

    def train_test_split(
        self, test_size: float = 0.2, random_state: Optional[int] = None
    ) -> Tuple:
        """
        Split dataset into train and test sets.

        Args:
            test_size: Fraction of data for testing
            random_state: Random seed

        Returns:
            Tuple of (train_features, test_features, train_labels, test_labels)
        """
        from sklearn.model_selection import train_test_split

        # Check if all classes have at least 2 samples for stratified split
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        singleton_classes = unique_labels[counts < 2]

        if len(singleton_classes) > 0:
            # Fall back to unstratified split with logging
            logger.warning(
                f"Cannot stratify: {len(singleton_classes)} class(es) have "
                f"fewer than 2 samples. Falling back to unstratified split. "
                f"Singleton classes: {list(singleton_classes)}"
            )
            return train_test_split(
                self.features,
                self.labels,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )

        # All classes have sufficient samples - use stratified split
        return train_test_split(
            self.features,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels,
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def train_context_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    model_path: Optional[str] = None,
) -> ContextClassifier:
    """
    Convenience function to train and optionally save a classifier.

    Args:
        features: 112D feature matrix
        labels: Context labels
        model_path: Optional path to save the trained model

    Returns:
        Trained ContextClassifier
    """
    classifier = ContextClassifier(model_type="mlp", random_state=42)
    classifier.train(features, labels)

    if model_path:
        classifier.save(model_path)

    return classifier


def main():
    """Demo CLI for ContextClassifier."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a context classifier for behavioral inference"
    )
    parser.add_argument(
        "--features", "-f", required=True, help="Path to features .npy file"
    )
    parser.add_argument(
        "--labels", "-l", required=True, help="Path to labels .npy file"
    )
    parser.add_argument(
        "--output", "-o", help="Save trained model to this path"
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading features from {args.features}")
    features = np.load(args.features)

    logger.info(f"Loading labels from {args.labels}")
    labels = np.load(args.labels)

    # Train
    classifier = train_context_classifier(features, labels, args.output)

    metadata = classifier.get_metadata()

    print("\nTraining complete!")
    print(f"  Classes: {metadata['n_classes']}")
    print(f"  Class names: {metadata['class_names']}")
    print(f"  Samples: {metadata['n_samples']}")

    if args.output:
        print(f"  Model saved to: {args.output}")


if __name__ == "__main__":
    main()
