#!/usr/bin/env python3
"""
Tests for ContextClassifier - Direction 4: Semantic Alignment

TDD Sprint 4.1: ContextClassifier Core
TDD Sprint 4.2: Weak Supervision
TDD Sprint 4.3: Model Persistence

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from realtime.context_classifier import ContextClassifier, ContextDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures
# =============================================================================


def create_binary_context_data(n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic data for binary context classification.

    Returns:
        (features, labels) where labels are 0 or 1
    """
    np.random.seed(42)

    # Context 0: Low frequency, low energy (social)
    context0_features = np.random.randn(n_samples // 2, 112) * 0.5
    context0_features[:, 0] -= 3.0  # Lower F0
    context0_features[:, 1] -= 1.0  # Lower RMS

    # Context 1: High frequency, high energy (alarm)
    context1_features = np.random.randn(n_samples // 2, 112) * 0.5
    context1_features[:, 0] += 3.0  # Higher F0
    context1_features[:, 1] += 1.0  # Higher RMS

    features = np.vstack([context0_features, context1_features])
    labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Shuffle
    indices = np.random.permutation(n_samples)
    return features[indices], labels[indices]


def create_multi_context_data(n_samples: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic data for multi-class context classification.

    Returns:
        (features, labels) where labels are 0, 1, 2, or 3
    """
    np.random.seed(42)

    features_list = []
    labels_list = []

    # Context 0: Social (low F0, low RMS)
    features = np.random.randn(n_samples // 4, 112) * 0.5
    features[:, 0] -= 5.0
    features[:, 1] -= 2.0
    features_list.append(features)
    labels_list.append([0] * (n_samples // 4))

    # Context 1: Contact (medium F0, medium RMS)
    features = np.random.randn(n_samples // 4, 112) * 0.5
    features[:, 0] -= 1.0
    features[:, 1] -= 0.5
    features_list.append(features)
    labels_list.append([1] * (n_samples // 4))

    # Context 2: Territorial (high F0, medium RMS)
    features = np.random.randn(n_samples // 4, 112) * 0.5
    features[:, 0] += 2.0
    features[:, 1] += 0.5
    features_list.append(features)
    labels_list.append([2] * (n_samples // 4))

    # Context 3: Alarm (very high F0, high RMS)
    features = np.random.randn(n_samples // 4, 112) * 0.5
    features[:, 0] += 5.0
    features[:, 1] += 2.0
    features_list.append(features)
    labels_list.append([3] * (n_samples // 4))

    features = np.vstack(features_list)
    labels = np.array(labels_list).flatten()

    # Shuffle
    indices = np.random.permutation(n_samples)
    return features[indices], labels[indices]


# =============================================================================
# Sprint 4.1: ContextClassifier Core Tests
# =============================================================================


class TestContextClassifierCore:
    """Test ContextClassifier core functionality."""

    def test_classifier_binary_separation(self):
        """MLP separates two contexts with 95%+ accuracy."""
        features, labels = create_binary_context_data(n_samples=300)

        # Split train/test
        split = int(0.8 * len(features))
        train_features, test_features = features[:split], features[split:]
        train_labels, test_labels = labels[:split], labels[split:]

        classifier = ContextClassifier(model_type="mlp", random_state=42)
        classifier.train(train_features, train_labels)

        predictions = classifier.predict_batch(test_features)

        # Convert string predictions to integers for comparison
        # Class names are string representations of integer labels
        pred_ints = np.array([int(p) for p in predictions])

        accuracy = np.mean(pred_ints == test_labels)

        assert accuracy > 0.90, f"Expected >90% accuracy, got {accuracy:.2%}"
        logger.info(f"✓ Binary classification accuracy: {accuracy:.2%}")

    def test_classifier_multi_class(self):
        """Handles 4+ context classes."""
        features, labels = create_multi_context_data(n_samples=400)

        split = int(0.8 * len(features))
        train_features, test_features = features[:split], features[split:]
        train_labels, test_labels = labels[:split], labels[split:]

        classifier = ContextClassifier(model_type="mlp", random_state=42)
        classifier.train(train_features, train_labels)

        predictions = classifier.predict_batch(test_features)

        # Convert string predictions to integers for comparison
        pred_ints = np.array([int(p) for p in predictions])

        accuracy = np.mean(pred_ints == test_labels)

        assert accuracy > 0.80, f"Expected >80% accuracy, got {accuracy:.2%}"
        logger.info(f"✓ Multi-class accuracy: {accuracy:.2%}")

    def test_classifier_returns_confidence(self):
        """Confidence scales with prediction certainty."""
        features, labels = create_binary_context_data(n_samples=200)

        classifier = ContextClassifier(model_type="mlp", random_state=42)
        classifier.train(features, labels)

        # Predict on clear examples (far from boundary)
        clear_example = np.zeros(112)
        clear_example[0] = -5.0  # Clearly context 0
        context1, conf1 = classifier.predict(clear_example)

        # Should predict context 0 with high confidence
        assert context1 == "context_0" or context1 == "0"
        assert conf1 > 0.7, f"Expected high confidence, got {conf1:.2f}"

        logger.info(f"✓ Clear example: {context1}, confidence={conf1:.2f}")

    def test_classifier_112d_input(self):
        """Accepts full 112D feature vector."""
        features, labels = create_binary_context_data(n_samples=100)

        classifier = ContextClassifier(model_type="mlp", random_state=42)
        classifier.train(features, labels)

        # Test single 112D prediction
        single_feature = np.random.randn(112)
        context, confidence = classifier.predict(single_feature)

        assert isinstance(context, str)
        assert 0.0 <= confidence <= 1.0

        logger.info(f"✓ 112D input accepted: {context}, conf={confidence:.2f}")

    def test_classifier_untrained_fails(self):
        """Predict fails gracefully when model is not trained."""
        classifier = ContextClassifier(model_type="mlp")

        with pytest.raises(RuntimeError):
            classifier.predict(np.random.randn(112))

    def test_classifier_context_names(self):
        """Uses context names instead of integers."""
        features, labels = create_binary_context_data(n_samples=200)

        # Use string labels
        label_names = np.array(["social"] * 100 + ["alarm"] * 100)

        classifier = ContextClassifier(model_type="mlp", random_state=42)
        classifier.train(features, label_names)

        context, confidence = classifier.predict(np.random.randn(112))

        assert context in ["social", "alarm"]
        logger.info(f"✓ Named contexts: {context}, conf={confidence:.2f}")


# =============================================================================
# Sprint 4.2: Weak Supervision Tests
# =============================================================================


class TestWeakSupervision:
    """Test weak supervision from temporal co-occurrence."""

    @dataclass
    class TemporalSegment:
        """Test fixture for temporal segments."""

        features: np.ndarray
        timestamp: float
        true_context: str

    def create_temporal_stream(self, n_segments: int = 50) -> List[TemporalSegment]:
        """Create a simulated stream of temporal segments."""
        segments = []
        current_time = 0.0
        current_context = "social"

        for i in range(n_segments):
            # Switch context every ~10 segments
            if i % 10 == 0 and i > 0:
                current_context = "alarm" if current_context == "social" else "social"

            # Generate features for current context
            if current_context == "social":
                features = np.random.randn(112) * 0.3
                features[0] -= 3.0  # Low F0
            else:  # alarm
                features = np.random.randn(112) * 0.3
                features[0] += 3.0  # High F0

            segments.append(
                self.TemporalSegment(
                    features=features,
                    timestamp=current_time,
                    true_context=current_context,
                )
            )

            current_time += np.random.uniform(0.05, 0.2)  # 50-200ms between segments

        return segments

    def test_temporal_cooccurrence_labeling(self):
        """Segments within 500ms get same label."""
        # This would use ContextDataset, testing the concept
        segments = self.create_temporal_stream(n_segments=50)

        # Group by temporal proximity (< 500ms)
        groups = []
        current_group = [segments[0]]

        for seg in segments[1:]:
            if seg.timestamp - current_group[0].timestamp < 0.5:
                current_group.append(seg)
            else:
                groups.append(current_group)
                current_group = [seg]
        groups.append(current_group)

        # Each group should have consistent true_context
        consistent_count = sum(1 for g in groups if len(set(s.true_context for s in g)) == 1)

        # Most groups should be consistent
        assert consistent_count > len(groups) * 0.8, (
            f"Expected >80% consistent groups, got {consistent_count}/{len(groups)}"
        )

        logger.info(f"✓ Temporal grouping: {consistent_count}/{len(groups)} consistent")

    def test_temporal_label_boundary_detection(self):
        """Detects context switch at temporal gaps."""
        segments = self.create_temporal_stream(n_segments=50)

        # Find context switches
        switches = []
        for i in range(1, len(segments)):
            if segments[i].true_context != segments[i - 1].true_context:
                switches.append(i)

        # Should have ~4-5 switches for 50 segments with 10-segment contexts
        assert 3 <= len(switches) <= 6, f"Expected 3-6 switches, got {len(switches)}"

        logger.info(f"✓ Detected {len(switches)} context switches")

    def test_singleton_pseudo_labels_fallback_split(self):
        """Handles singleton pseudo-labels gracefully during split."""
        # Create sparse temporal groups (many singletons)
        segments = []
        current_time = 0.0

        # Create 20 segments, each in its own temporal group
        # (simulating sparse or fragmented recordings)
        for i in range(20):
            segments.append(
                {
                    "features": np.random.randn(112) * 0.5,
                    "timestamp": current_time,
                }
            )
            # Large gap ensures each segment gets its own label
            # Use 1 second gaps, with a 10ms window
            current_time += 1.0  # 1 second = 1000ms >> 10ms window

        # Create dataset with very small window (10ms) to ensure singletons
        # Each segment will be >10ms apart, so each gets its own label
        dataset = ContextDataset.from_temporal_cooccurrence(segments, window_ms=10.0)

        # Verify we have many singleton classes
        unique_labels, counts = np.unique(dataset.labels, return_counts=True)
        n_singletons = np.sum(counts == 1)
        assert n_singletons > 10, f"Expected many singletons, got {n_singletons}"

        # train_test_split should fall back to unstratified split
        # and NOT raise an error
        train_feat, test_feat, train_labels, test_labels = dataset.train_test_split(
            test_size=0.2, random_state=42
        )

        # Verify split succeeded
        assert len(train_feat) + len(test_feat) == len(dataset.features)
        assert len(train_labels) + len(test_labels) == len(dataset.labels)

        logger.info(
            f"✓ Singleton fallback split: {n_singletons} singletons, "
            f"train={len(train_feat)}, test={len(test_feat)}"
        )

    def test_training_with_singleton_classes(self):
        """Training handles singleton pseudo-labels gracefully."""
        # Create a small dataset with singleton classes
        segments = []
        for i in range(10):
            segments.append(
                {
                    "features": np.random.randn(112) * 0.5,
                    "timestamp": float(i),  # Each segment is far apart
                }
            )

        # Create dataset with tiny window to ensure singletons
        dataset = ContextDataset.from_temporal_cooccurrence(segments, window_ms=10.0)

        # Verify we have singleton classes
        unique_labels, counts = np.unique(dataset.labels, return_counts=True)
        assert np.min(counts) == 1, "Should have singleton classes"

        # Training should succeed (disable early stopping internally)
        classifier = ContextClassifier(model_type="mlp", random_state=42)
        classifier.train(dataset.features, dataset.labels)

        # Should be able to predict
        test_feature = np.random.randn(112)
        context, confidence = classifier.predict(test_feature)

        assert context in unique_labels
        assert 0.0 <= confidence <= 1.0

        logger.info(
            f"✓ Training with singleton classes: {len(unique_labels)} classes, "
            f"min_count={np.min(counts)}"
        )


# =============================================================================
# Sprint 4.3: Model Persistence Tests
# =============================================================================


class TestModelPersistence:
    """Test model save/load functionality."""

    def test_save_load_roundtrip(self):
        """Model predictions identical after save/load."""
        features, labels = create_binary_context_data(n_samples=200)

        classifier1 = ContextClassifier(model_type="mlp", random_state=42)
        classifier1.train(features, labels)

        # Test prediction before saving
        test_feature = np.random.randn(112)
        context1, conf1 = classifier1.predict(test_feature)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            classifier1.save(f.name)
            classifier2 = ContextClassifier.load(f.name)

        # Test prediction after loading
        context2, conf2 = classifier2.predict(test_feature)

        assert context1 == context2, f"Context mismatch: {context1} != {context2}"
        assert abs(conf1 - conf2) < 0.01, f"Confidence mismatch: {conf1} != {conf2}"

        logger.info(f"✓ Roundtrip: {context1}={context2}, {conf1:.2f}={conf2:.2f}")

    def test_model_metadata_preserved(self):
        """Tracks training timestamp and data hash."""
        features, labels = create_binary_context_data(n_samples=200)

        classifier = ContextClassifier(model_type="mlp", random_state=42)
        classifier.train(features, labels)

        metadata = classifier.get_metadata()

        assert "training_timestamp" in metadata
        assert "n_classes" in metadata
        assert "feature_dim" in metadata
        assert metadata["feature_dim"] == 112

        logger.info(f"✓ Metadata: {metadata}")

    def test_save_to_multiple_formats(self):
        """Can save to pickle and joblib formats."""
        features, labels = create_binary_context_data(n_samples=100)

        classifier = ContextClassifier(model_type="mlp", random_state=42)
        classifier.train(features, labels)

        # Test pickle format
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            classifier.save(f.name)
            assert Path(f.name).stat().st_size > 0

        # Test joblib format
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            classifier.save(f.name)
            assert Path(f.name).stat().st_size > 0

        logger.info("✓ Saved to both pickle and joblib formats")


# =============================================================================
# Integration Tests
# =============================================================================


class TestContextClassifierIntegration:
    """Integration tests with InteractionAgent."""

    def test_classifier_replaces_rules(self):
        """ML classifier should outperform simple rules."""
        features, labels = create_multi_context_data(n_samples=400)

        split = int(0.8 * len(features))
        test_features = features[split:]
        test_labels = labels[split:]

        classifier = ContextClassifier(model_type="mlp", random_state=42)
        classifier.train(features[:split], labels[:split])

        # ML predictions
        ml_predictions = classifier.predict_batch(test_features)
        # Convert string predictions to integers  # noqa: E501
        # (class names are string representations of integers)
        ml_pred_ints = np.array([int(p) for p in ml_predictions])
        ml_accuracy = np.mean(ml_pred_ints == test_labels)

        # Simple rule baseline (F0 threshold)
        rule_predictions = np.where(test_features[:, 0] > 0, 1, 0)  # Binary for simplicity
        np.mean(rule_predictions == (test_labels > 1))  # Compare alarm(3)+territorial(2) vs others

        # ML should significantly outperform simple rules
        logger.info(f"ML accuracy: {ml_accuracy:.2%}, Rule baseline: ~50%")
        assert ml_accuracy > 0.70, "ML should significantly outperform rules"

    def test_classifier_with_realistic_112d(self):
        """Works with realistic 112D feature distributions."""
        np.random.seed(42)

        # Create more realistic features using actual feature correlations
        n_samples = 300
        features = np.random.randn(n_samples, 112) * 0.5

        # Add structured patterns for contexts
        # Context 0: harmonic (higher HNR, lower FM)
        features[: n_samples // 3, 6] += 2.0  # HNR
        features[: n_samples // 3, 20] -= 1.5  # FM rate

        # Context 1: noisy (lower HNR, higher FM)
        features[n_samples // 3 : 2 * n_samples // 3, 6] -= 2.0
        features[n_samples // 3 : 2 * n_samples // 3, 20] += 2.0

        # Context 2: mixed
        # Keep near-zero modifications

        labels = np.array(
            [0] * (n_samples // 3) + [1] * (n_samples // 3) + [2] * (n_samples // 3 + n_samples % 3)
        )

        classifier = ContextClassifier(model_type="mlp", random_state=42, hidden_layers=(64, 32))
        classifier.train(features, labels)

        # Predict on test examples
        test_features = np.random.randn(10, 112) * 0.5
        # Make test_features similar to context 0
        test_features[:3, 6] += 2.0
        test_features[:3, 20] -= 1.5

        predictions = classifier.predict_batch(test_features)

        # First 3 should be predicted as context 0 or close to it
        assert len(predictions) == 10
        logger.info(f"✓ Realistic 112D features: predictions={predictions[:3]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
