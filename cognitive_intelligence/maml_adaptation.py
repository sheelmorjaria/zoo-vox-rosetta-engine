#!/usr/bin/env python3
"""
MAML Adaptation - Model-Agnostic Meta-Learning
==============================================

Rapid cross-species adaptation using few-shot meta-learning.

This module implements:
- MAML optimizer for meta-learning
- Few-shot classification with rapid adaptation
- Cross-species task distribution
- Species-specific encoders
- Transfer learning across animal species

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetaParameters:
    """Container for meta-learned parameters."""
    weights: List[np.ndarray]  # List of weight matrices
    biases: Optional[List[np.ndarray]] = None  # List of bias vectors


class MAMLOptimizer:
    """
    MAML optimizer for meta-learning.

    Implements Model-Agnostic Meta-Learning for rapid
    adaptation to new species with few examples.
    """

    def __init__(
        self,
        input_dim: int = 112,
        hidden_dim: int = 64,
        output_dim: int = 4,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        n_inner_steps: int = 5,
    ):
        """
        Initialize MAML optimizer.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            inner_lr: Learning rate for inner loop (task-specific)
            meta_lr: Learning rate for meta update
            n_inner_steps: Number of inner loop steps
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.n_inner_steps = n_inner_steps

        # Initialize meta-parameters
        scale = 1.0 / math.sqrt(input_dim)
        self.w1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, output_dim) * scale
        self.b2 = np.zeros(output_dim)

        self.meta_parameters = MetaParameters(
            weights=[self.w1, self.w2],
            biases=[self.b1, self.b2],
        )

    def inner_loop_update(
        self, support_x: np.ndarray, support_y: np.ndarray
    ) -> MetaParameters:
        """
        Perform inner loop update for a specific task.

        Args:
            support_x: Support set features (n_samples, input_dim)
            support_y: Support set labels (n_samples,)

        Returns:
            Adapted parameters for this task
        """
        # Copy meta-parameters
        w1 = self.w1.copy()
        b1 = self.b1.copy()
        w2 = self.w2.copy()
        b2 = self.b2.copy()

        # Perform gradient steps on support set
        for _ in range(self.n_inner_steps):
            # Forward pass
            hidden = np.maximum(0, support_x @ w1 + b1)  # ReLU
            logits = hidden @ w2 + b2

            # Compute loss (cross-entropy gradient approximation)
            probs = self._softmax(logits)
            one_hot = np.zeros_like(probs)
            for i, label in enumerate(support_y):
                if 0 <= label < self.output_dim:
                    one_hot[i, label] = 1.0

            grad_logits = probs - one_hot

            # Backward pass
            grad_w2 = hidden.T @ grad_logits
            grad_b2 = np.sum(grad_logits, axis=0)

            grad_hidden = grad_logits @ w2.T
            grad_hidden[hidden <= 0] = 0  # ReLU gradient

            grad_w1 = support_x.T @ grad_hidden
            grad_b1 = np.sum(grad_hidden, axis=0)

            # Gradient descent update
            w1 -= self.inner_lr * grad_w1
            b1 -= self.inner_lr * grad_b1
            w2 -= self.inner_lr * grad_w2
            b2 -= self.inner_lr * grad_b2

        return MetaParameters(
            weights=[w1, w2],
            biases=[b1, b2],
        )

    def meta_update(self, tasks: List[Tuple]) -> float:
        """
        Perform meta-update across multiple tasks.

        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples

        Returns:
            Meta-loss value
        """
        meta_grad_w1 = np.zeros_like(self.w1)
        meta_grad_b1 = np.zeros_like(self.b1)
        meta_grad_w2 = np.zeros_like(self.w2)
        meta_grad_b2 = np.zeros_like(self.b2)
        total_loss = 0.0

        for support_x, support_y, query_x, query_y in tasks:
            # Get task-specific parameters
            task_params = self.inner_loop_update(support_x, support_y)

            w1, w2 = task_params.weights
            b1, b2 = task_params.biases

            # Compute loss on query set
            hidden = np.maximum(0, query_x @ w1 + b1)
            logits = hidden @ w2 + b2
            probs = self._softmax(logits)

            # Cross-entropy loss
            loss = 0.0
            for i, label in enumerate(query_y):
                if 0 <= label < self.output_dim:
                    loss -= np.log(probs[i, label] + 1e-8)

            loss /= len(query_y)
            total_loss += loss

            # Compute gradients through adaptation
            one_hot = np.zeros_like(probs)
            for i, label in enumerate(query_y):
                if 0 <= label < self.output_dim:
                    one_hot[i, label] = 1.0

            grad_logits = probs - one_hot

            grad_w2_task = hidden.T @ grad_logits
            grad_b2_task = np.sum(grad_logits, axis=0)

            grad_hidden = grad_logits @ w2.T
            grad_hidden[hidden <= 0] = 0

            grad_w1_task = query_x.T @ grad_hidden
            grad_b1_task = np.sum(grad_hidden, axis=0)

            meta_grad_w1 += grad_w1_task
            meta_grad_b1 += grad_b1_task
            meta_grad_w2 += grad_w2_task
            meta_grad_b2 += grad_b2_task

        # Average gradients
        n_tasks = len(tasks)
        meta_grad_w1 /= n_tasks
        meta_grad_b1 /= n_tasks
        meta_grad_w2 /= n_tasks
        meta_grad_b2 /= n_tasks

        # Meta-update
        self.w1 -= self.meta_lr * meta_grad_w1
        self.b1 -= self.meta_lr * meta_grad_b1
        self.w2 -= self.meta_lr * meta_grad_w2
        self.b2 -= self.meta_lr * meta_grad_b2

        # Update meta-parameters
        self.meta_parameters = MetaParameters(
            weights=[self.w1, self.w2],
            biases=[self.b1, self.b2],
        )

        return total_loss / n_tasks

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax with numerical stability."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class FewShotClassifier:
    """
    Few-shot classifier using MAML.

    Adapts to new tasks with only k examples per class.
    """

    def __init__(
        self,
        input_dim: int = 112,
        num_classes: int = 5,
        k_shot: int = 5,
        inner_lr: float = 0.01,
    ):
        """
        Initialize few-shot classifier.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of classes
            k_shot: Number of examples per class
            inner_lr: Learning rate for adaptation
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.k_shot = k_shot
        self.inner_lr = inner_lr

        # Initialize MAML optimizer
        self.maml = MAMLOptimizer(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=num_classes,
            inner_lr=inner_lr,
        )

        self.adapted_params = None

    def pretrain(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Pre-train on base data.

        Args:
            features: Training features
            labels: Training labels
        """
        # Create synthetic tasks from base data
        tasks = []
        n_samples = len(features)

        for _ in range(10):
            # Random split
            indices = np.random.permutation(n_samples)
            support_idx = indices[: self.k_shot * self.num_classes]
            query_idx = indices[self.k_shot * self.num_classes :]

            support_x = features[support_idx]
            support_y = labels[support_idx]
            query_x = features[query_idx]
            query_y = labels[query_idx]

            tasks.append((support_x, support_y, query_x, query_y))

        # Meta-update
        self.maml.meta_update(tasks)

    def adapt(self, support_x: np.ndarray, support_y: np.ndarray) -> None:
        """
        Adapt to a new task with support set.

        Args:
            support_x: Support set features
            support_y: Support set labels
        """
        self.adapted_params = self.maml.inner_loop_update(support_x, support_y)

    def predict(self, query_x: np.ndarray) -> np.ndarray:
        """
        Predict on query set.

        Args:
            query_x: Query set features

        Returns:
            Predictions
        """
        if self.adapted_params is None:
            # Use meta-parameters
            w1, w2 = self.maml.w1, self.maml.w2
            b1, b2 = self.maml.b1, self.maml.b2
        else:
            w1, w2 = self.adapted_params.weights[0], self.adapted_params.weights[1]
            b1, b2 = self.adapted_params.biases[0], self.adapted_params.biases[1]

        # Forward pass
        hidden = np.maximum(0, query_x @ w1 + b1)
        logits = hidden @ w2 + b2
        predictions = np.argmax(logits, axis=-1)

        return predictions


class TaskDistribution:
    """
    Task distribution for meta-learning.

    Generates tasks for training and evaluation.
    """

    def __init__(
        self,
        n_classes: int = 5,
        n_support: int = 5,
        n_query: int = 10,
        species: Optional[List[str]] = None,
    ):
        """
        Initialize task distribution.

        Args:
            n_classes: Number of classes per task
            n_support: Number of support examples per class
            n_query: Number of query examples per class
            species: List of species for cross-species tasks
        """
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query
        self.species = species or []

        # Simulated feature generator
        self.feature_dim = 112

    def sample_task(self) -> Dict:
        """
        Sample a single task.

        Returns:
            Dictionary with support_x, support_y, query_x, query_y
        """
        # Generate random features for each class
        support_x = []
        support_y = []
        query_x = []
        query_y = []

        for class_id in range(self.n_classes):
            # Class-specific mean
            class_mean = np.random.randn(self.feature_dim) * 0.5

            # Support examples
            for _ in range(self.n_support):
                sample = class_mean + np.random.randn(self.feature_dim) * 0.1
                support_x.append(sample)
                support_y.append(class_id)

            # Query examples
            for _ in range(self.n_query):
                sample = class_mean + np.random.randn(self.feature_dim) * 0.1
                query_x.append(sample)
                query_y.append(class_id)

        return {
            "support_x": np.array(support_x, dtype=np.float32),
            "support_y": np.array(support_y, dtype=np.int32),
            "query_x": np.array(query_x, dtype=np.float32),
            "query_y": np.array(query_y, dtype=np.int32),
        }

    def sample_cross_species_task(self, species: str) -> Dict:
        """
        Sample a task for a specific species.

        Args:
            species: Species name

        Returns:
            Task dictionary
        """
        task = self.sample_task()

        # Add species-specific bias to features
        species_seed = hash(species) % 1000
        species_bias = np.random.RandomState(species_seed).randn(self.feature_dim) * 0.2

        task["support_x"] += species_bias
        task["query_x"] += species_bias
        task["species"] = species

        return task

    def sample_batch(self, n_tasks: int = 4) -> List[Dict]:
        """
        Sample a batch of tasks.

        Args:
            n_tasks: Number of tasks to sample

        Returns:
            List of task dictionaries
        """
        tasks = []
        for _ in range(n_tasks):
            if self.species:
                species = np.random.choice(self.species)
                task = self.sample_cross_species_task(species)
            else:
                task = self.sample_task()
            tasks.append(task)

        return tasks


class SpeciesEncoder:
    """
    Species-specific encoder for cross-species adaptation.

    Encodes features with species conditioning.
    """

    def __init__(
        self,
        input_dim: int = 112,
        latent_dim: int = 32,
        num_species: int = 4,
    ):
        """
        Initialize species encoder.

        Args:
            input_dim: Input feature dimension
            latent_dim: Latent representation dimension
            num_species: Number of species
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_species = num_species

        # Shared encoder
        scale = 1.0 / math.sqrt(input_dim)
        self.shared_encoder = np.random.randn(input_dim, latent_dim) * scale

        # Species-specific embeddings
        self.species_embeddings = np.random.randn(num_species, latent_dim) * scale

    def encode(self, features: np.ndarray, species_id: int) -> np.ndarray:
        """
        Encode features with species conditioning.

        Args:
            features: Input features (n_samples, input_dim)
            species_id: Species ID

        Returns:
            Encoded features (n_samples, latent_dim)
        """
        # Shared encoding
        encoded = features @ self.shared_encoder

        # Add species embedding
        species_emb = self.species_embeddings[species_id]
        encoded = encoded + species_emb

        return encoded


class MetaLearner:
    """
    Meta-learner for cross-species adaptation.

    Combines MAML with species-specific conditioning.
    """

    def __init__(
        self,
        input_dim: int = 112,
        num_classes: int = 4,
        species: Optional[List[str]] = None,
    ):
        """
        Initialize meta-learner.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of classes
            species: List of species names
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.species_list = species or []

        # MAML optimizer (uses encoded dimension)
        self.encoded_dim = 32  # Latent dimension from species encoder
        self.maml = MAMLOptimizer(
            input_dim=self.encoded_dim,
            hidden_dim=64,
            output_dim=num_classes,
        )

        # Species encoder
        self.num_species = len(species) if species else 1
        self.species_encoder = SpeciesEncoder(
            input_dim=input_dim,
            latent_dim=self.encoded_dim,
            num_species=self.num_species,
        )

        # Store species data
        self.species_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def add_species_data(
        self, species: str, features: np.ndarray, labels: np.ndarray
    ) -> None:
        """
        Add training data for a species.

        Args:
            species: Species name
            features: Feature array
            labels: Label array
        """
        self.species_data[species] = (features, labels)

        if species not in self.species_list:
            self.species_list.append(species)
            self.num_species = len(self.species_list)

    def meta_train(
        self, n_epochs: int = 10, n_tasks_per_epoch: int = 8
    ) -> List[float]:
        """
        Perform meta-training across species.

        Args:
            n_epochs: Number of meta-training epochs
            n_tasks_per_epoch: Tasks per epoch

        Returns:
            List of meta-losses
        """
        losses = []

        for epoch in range(n_epochs):
            epoch_tasks = []
            epoch_loss = 0.0

            for _ in range(n_tasks_per_epoch):
                # Sample species
                if self.species_list:
                    species = np.random.choice(self.species_list)
                    species_id = self.species_list.index(species)

                    if species in self.species_data:
                        base_features, base_labels = self.species_data[species]

                        # Encode with species conditioning
                        encoded_features = self.species_encoder.encode(
                            base_features, species_id
                        )

                        # Sample task from species data
                        n_samples = min(20, len(encoded_features))
                        indices = np.random.choice(len(encoded_features), n_samples, replace=False)

                        split = n_samples // 2
                        support_idx = indices[:split]
                        query_idx = indices[split:]

                        epoch_tasks.append((
                            encoded_features[support_idx],
                            base_labels[support_idx],
                            encoded_features[query_idx],
                            base_labels[query_idx],
                        ))

            if epoch_tasks:
                meta_loss = self.maml.meta_update(epoch_tasks)
                losses.append(meta_loss)
                epoch_loss = meta_loss

            logger.info(f"Epoch {epoch + 1}/{n_epochs}, meta-loss: {epoch_loss:.4f}")

        return losses

    def meta_update_task(
        self, features: np.ndarray, labels: np.ndarray
    ) -> float:
        """
        Single task meta-update.

        Args:
            features: Task features (will be encoded)
            labels: Task labels

        Returns:
            Task loss
        """
        # Encode features (use species 0 as default for meta-training)
        encoded_features = self.species_encoder.encode(features, species_id=0)

        # Simple train/val split
        n = len(encoded_features)
        split = n // 2

        support_x, query_x = encoded_features[:split], encoded_features[split:]
        support_y, query_y = labels[:split], labels[split:]

        loss = self.maml.meta_update([(support_x, support_y, query_x, query_y)])

        return loss

    def adapt(self, support_x: np.ndarray, support_y: np.ndarray) -> None:
        """
        Adapt to a new task.

        Args:
            support_x: Support features (will be encoded)
            support_y: Support labels
        """
        # Encode features (use species 0 as default)
        encoded_features = self.species_encoder.encode(support_x, species_id=0)
        self.current_adapted_params = self.maml.inner_loop_update(encoded_features, support_y)

    def adapt_to_species(
        self, species: str, features: np.ndarray, labels: np.ndarray
    ) -> None:
        """
        Adapt to a new species.

        Args:
            species: Species name
            features: Species features
            labels: Species labels
        """
        # Add to species list if new
        if species not in self.species_list:
            self.species_list.append(species)
            self.num_species = len(self.species_list)

            # Expand species embeddings if needed
            if self.num_species > self.species_encoder.species_embeddings.shape[0]:
                # Add new embedding
                new_embedding = np.random.randn(1, self.species_encoder.latent_dim) * 0.1
                self.species_encoder.species_embeddings = np.vstack([
                    self.species_encoder.species_embeddings,
                    new_embedding
                ])

        species_id = self.species_list.index(species)

        # Encode with species conditioning
        encoded_features = self.species_encoder.encode(features, species_id)

        # Adapt (pass encoded features directly to MAML)
        self.current_adapted_params = self.maml.inner_loop_update(encoded_features, labels)

    def predict(
        self, query_x: np.ndarray, species: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict on query set.

        Args:
            query_x: Query features
            species: Optional species name

        Returns:
            Predictions
        """
        # Encode features (always encode since MAML uses encoded dimension)
        if species is not None and species in self.species_list:
            species_id = self.species_list.index(species)
            query_x = self.species_encoder.encode(query_x, species_id)
        else:
            # Use default species encoding
            query_x = self.species_encoder.encode(query_x, species_id=0)

        # Use adapted parameters if available
        if hasattr(self, "current_adapted_params") and self.current_adapted_params is not None:
            w1, w2 = self.current_adapted_params.weights[0], self.current_adapted_params.weights[1]
            b1, b2 = self.current_adapted_params.biases[0], self.current_adapted_params.biases[1]
        else:
            w1, w2 = self.maml.w1, self.maml.w2
            b1, b2 = self.maml.b1, self.maml.b2

        # Forward pass
        hidden = np.maximum(0, query_x @ w1 + b1)
        logits = hidden @ w2 + b2
        predictions = np.argmax(logits, axis=-1)

        return predictions


class MAMLAdaptationWrapper:
    """
    Wrapper for adapting existing models with MAML.

    Provides MAML capabilities to pre-trained models.
    """

    def __init__(self, base_model):
        """
        Initialize wrapper.

        Args:
            base_model: Pre-trained model to adapt
        """
        self.base_model = base_model
        self.adapted_weights = None

    def adapt(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        n_steps: int = 5,
        lr: float = 0.01,
    ) -> None:
        """
        Adapt model to new task.

        Args:
            support_x: Support features
            support_y: Support labels
            n_steps: Number of adaptation steps
            lr: Learning rate
        """
        # Get base weights
        if hasattr(self.base_model, "weights"):
            weights = self.base_model.weights.copy()
        else:
            raise ValueError("Base model must have 'weights' attribute")

        # Gradient steps on support set
        for _ in range(n_steps):
            # Compute predictions
            logits = support_x @ weights

            # Compute gradient (simplified)
            probs = self._softmax(logits)
            one_hot = np.zeros_like(probs)
            for i, label in enumerate(support_y):
                if 0 <= label < probs.shape[1]:
                    one_hot[i, label] = 1.0

            grad = support_x.T @ (probs - one_hot)
            weights -= lr * grad

        self.adapted_weights = weights

    def predict(self, query_x: np.ndarray) -> np.ndarray:
        """
        Predict with adapted model.

        Args:
            query_x: Query features

        Returns:
            Predictions
        """
        if self.adapted_weights is not None:
            weights = self.adapted_weights
        elif hasattr(self.base_model, "weights"):
            weights = self.base_model.weights
        else:
            raise ValueError("No weights available for prediction")

        logits = query_x @ weights
        predictions = np.argmax(logits, axis=-1)

        return predictions

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax with numerical stability."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("MAML Adaptation - Model-Agnostic Meta-Learning")
    print("=" * 60)

    # Test MAML optimizer
    maml = MAMLOptimizer(input_dim=112, hidden_dim=64, output_dim=4)

    support_x = np.random.randn(5, 112).astype(np.float32)
    support_y = np.array([0, 1, 2, 3, 0], dtype=np.int32)

    adapted = maml.inner_loop_update(support_x, support_y)

    print(f"Adapted weights shape: {adapted.weights.shape}")

    # Test few-shot classifier
    classifier = FewShotClassifier(input_dim=112, num_classes=5, k_shot=5)

    base_features = np.random.randn(50, 112).astype(np.float32)
    base_labels = np.random.randint(0, 5, 50).astype(np.int32)

    classifier.pretrain(base_features, base_labels)

    few_shot_x = np.random.randn(5, 112).astype(np.float32)
    few_shot_y = np.array([0, 1, 2, 3, 4], dtype=np.int32)

    classifier.adapt(few_shot_x, few_shot_y)

    query_x = np.random.randn(3, 112).astype(np.float32)
    predictions = classifier.predict(query_x)

    print(f"Predictions: {predictions}")

    # Test meta-learner
    meta_learner = MetaLearner(
        input_dim=112,
        num_classes=4,
        species=["marmoset", "bat", "dolphin"],
    )

    for species in ["marmoset", "bat", "dolphin"]:
        features = np.random.randn(50, 112).astype(np.float32)
        labels = np.random.randint(0, 4, 50).astype(np.int32)
        meta_learner.add_species_data(species, features, labels)

    meta_learner.meta_train(n_epochs=2, n_tasks_per_epoch=4)

    print("Meta-training complete")
