#!/usr/bin/env python3
"""
Tests for Graded Clustering Support

Tests for soft clustering probabilities and graded phrase detection
in FeatureEvent and the clustering pipeline.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest

import numpy as np

from realtime.feature_subscriber import FeatureEvent


class TestFeatureEventGradedClustering(unittest.TestCase):
    """Test soft clustering support in FeatureEvent"""

    def test_feature_event_supports_soft_clusters(self):
        """FeatureEvent should support cluster probabilities dictionary"""
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.zeros(112),
            timestamp=0.0,
            sequence=1,
            cluster_probabilities={42: 0.6, 89: 0.4},
        )

        self.assertIsNotNone(event.cluster_probabilities)
        self.assertEqual(event.cluster_probabilities[42], 0.6)
        self.assertEqual(event.cluster_probabilities[89], 0.4)

    def test_feature_event_is_graded_single_cluster(self):
        """Single cluster with high probability should not be graded"""
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.zeros(112),
            timestamp=0.0,
            sequence=1,
            cluster_probabilities={42: 0.95},
        )

        self.assertFalse(event.is_graded())

    def test_feature_event_is_graded_secondary_cluster(self):
        """Significant secondary probability should be graded"""
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.zeros(112),
            timestamp=0.0,
            sequence=1,
            cluster_probabilities={42: 0.6, 89: 0.4},
        )

        self.assertTrue(event.is_graded())

    def test_feature_event_is_graded_threshold(self):
        """Should respect custom threshold"""
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.zeros(112),
            timestamp=0.0,
            sequence=1,
            cluster_probabilities={42: 0.7, 89: 0.3},
        )

        # Default threshold 0.3
        self.assertFalse(event.is_graded(0.31))
        self.assertTrue(event.is_graded(0.29))
        self.assertTrue(event.is_graded())  # Uses default 0.3

    def test_feature_event_is_graded_no_probabilities(self):
        """Missing cluster probabilities should not be graded"""
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.zeros(112),
            timestamp=0.0,
            sequence=1,
        )

        self.assertFalse(event.is_graded())

    def test_feature_event_json_serialization_with_probabilities(self):
        """Cluster probabilities should serialize to JSON"""
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.zeros(112),
            timestamp=0.0,
            sequence=1,
            cluster_probabilities={42: 0.6, 89: 0.4},
        )

        json_dict = event.to_json_dict()
        self.assertIn("cluster_probabilities", json_dict)
        self.assertEqual(json_dict["cluster_probabilities"], {42: 0.6, 89: 0.4})

    def test_feature_event_json_deserialization_with_probabilities(self):
        """Cluster probabilities should deserialize from JSON"""
        json_data = {
            "event_type": "feature_extraction",
            "cluster_id": 42,
            "features_112d": [0.0] * 112,
            "timestamp": 0.0,
            "sequence": 1,
            "cluster_probabilities": {42: 0.6, 89: 0.4},
        }

        event = FeatureEvent.from_json(json_data)
        self.assertIsNotNone(event.cluster_probabilities)
        self.assertEqual(event.cluster_probabilities[42], 0.6)
        self.assertEqual(event.cluster_probabilities[89], 0.4)

    def test_feature_event_json_roundtrip_with_probabilities(self):
        """JSON roundtrip should preserve cluster probabilities"""
        original = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.random.randn(112).astype(np.float32),
            timestamp=123.456,
            sequence=42,
            cluster_probabilities={42: 0.6, 89: 0.3, 15: 0.1},
        )

        # Serialize
        json_dict = original.to_json_dict()

        # Deserialize
        restored = FeatureEvent.from_json(json_dict)

        # Verify
        self.assertEqual(restored.cluster_probabilities, original.cluster_probabilities)
        self.assertTrue(restored.is_graded())


class TestGradedClusteringPipeline(unittest.TestCase):
    """Test the graded clustering pipeline"""

    def test_umap_reduces_dimensionality(self):
        """UMAP should reduce 112D to lower dimensional space"""
        try:
            import umap
        except ImportError:
            self.skipTest("umap-learn not installed")

        # Create synthetic data
        np.random.seed(42)
        features = np.random.randn(100, 112).astype(np.float32)

        # Fit UMAP
        reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, random_state=42)
        embedding = reducer.fit_transform(features)

        self.assertEqual(embedding.shape, (100, 10))
        self.assertLess(embedding.shape[1], features.shape[1])

    def test_hdbscan_on_umap_embedding(self):
        """HDBSCAN should cluster UMAP embedding efficiently"""
        try:
            import hdbscan
            import umap
        except ImportError:
            self.skipTest("umap-learn or hdbscan not installed")

        # Create synthetic data with structure
        np.random.seed(42)
        features = np.random.randn(100, 112).astype(np.float32)

        # UMAP reduction
        reducer = umap.UMAP(n_components=10, random_state=42)
        embedding = reducer.fit_transform(features)

        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, prediction_data=True)
        labels = clusterer.fit_predict(embedding)

        # Should produce labels (noise = -1)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), 100)

    def test_soft_clustering_probabilities(self):
        """HDBSCAN should produce soft clustering probabilities"""
        try:
            import hdbscan
            import umap
        except ImportError:
            self.skipTest("umap-learn or hdbscan not installed")

        # Create synthetic data
        np.random.seed(42)
        features = np.random.randn(50, 112).astype(np.float32)

        # UMAP + HDBSCAN
        reducer = umap.UMAP(n_components=5, random_state=42)
        embedding = reducer.fit_transform(features)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
        clusterer.fit(embedding)

        # Get soft probabilities
        soft_clusters = hdbscan.all_points_membership_vectors(clusterer)

        self.assertEqual(soft_clusters.shape[0], 50)
        self.assertTrue(np.allclose(soft_clusters.sum(axis=1), 1.0, atol=0.01))

    def test_graded_boundary_detection(self):
        """Should detect graded boundary segments"""
        try:
            import hdbscan  # noqa: F401
        except ImportError:
            self.skipTest("hdbscan not installed")

        # Simulate soft clustering with graded boundary
        soft_probs = np.array(
            [
                [0.9, 0.1, 0.0],  # Clear cluster 0
                [0.7, 0.3, 0.0],  # Graded boundary
                [0.1, 0.9, 0.0],  # Clear cluster 1
                [0.4, 0.4, 0.2],  # Highly graded
            ]
        )

        # Detect graded (threshold = 0.3)
        graded = []
        for probs in soft_probs:
            sorted_probs = sorted(probs, reverse=True)
            is_graded = len(sorted_probs) > 1 and sorted_probs[1] > 0.3
            graded.append(is_graded)

        self.assertEqual(graded, [False, True, False, True])


class TestGradedContextIntegration(unittest.TestCase):
    """Test integration of graded clustering with context inference"""

    def test_graded_event_creates_blended_context(self):
        """Graded event should blend context probabilities"""

        # This would be implemented in the InteractionAgent
        # For now, test the concept
        alarm_prob = 0.5
        territorial_prob = 0.5

        # Blended F0 delta (alarm: +500, territorial: +200)
        alarm_delta = 500.0
        territorial_delta = 200.0
        blended_delta = alarm_delta * alarm_prob + territorial_delta * territorial_prob

        self.assertAlmostEqual(blended_delta, 350.0, places=1)

    def test_graded_threshold_configurable(self):
        """Graded threshold should be configurable"""
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.zeros(112),
            timestamp=0.0,
            sequence=1,
            cluster_probabilities={42: 0.7, 89: 0.3},
        )

        # Strict threshold
        self.assertFalse(event.is_graded(threshold=0.35))

        # Lenient threshold
        self.assertTrue(event.is_graded(threshold=0.25))


if __name__ == "__main__":
    unittest.main()
