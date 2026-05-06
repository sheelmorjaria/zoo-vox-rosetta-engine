#!/usr/bin/env python3
"""
Tests for Cluster Benchmark Suite

Tests for comparing clustering algorithms on Zoo Vox metrics.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
import numpy as np
from typing import Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.cluster_benchmark_suite import (
    ClusterBenchmarkSuite,
    ClusterResult,
    BenchmarkMetrics
)


class TestClusterBenchmarkSuite(unittest.TestCase):
    """Test the benchmark suite infrastructure."""

    def setUp(self):
        """Create synthetic test data."""
        np.random.seed(42)
        self.n_samples = 1000
        self.n_dims = 112

        # Create synthetic features with 3 clusters
        self.features = np.random.randn(self.n_samples, self.n_dims).astype(np.float32)

        # Add cluster structure
        self.features[:300] += 2.0  # Cluster 0
        self.features[300:600] += 4.0  # Cluster 1
        self.features[600:] += 6.0  # Cluster 2

        # Create sequences
        self.sequences = [
            [0, 0, 0, 1, 1, 0],
            [1, 1, 2, 2, 1],
            [2, 2, 0, 0, 2, 2, 1],
        ]

        self.suite = ClusterBenchmarkSuite()

    def test_benchmark_suite_runs_all_methods(self):
        """Should run all specified methods and return results."""
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            self.skipTest("sklearn not available")

        results = self.suite.run(
            self.features,
            self.sequences,
            methods=["kmeans"]
        )

        self.assertIn("kmeans", results)
        self.assertIsInstance(results["kmeans"], BenchmarkMetrics)

    def test_benchmark_result_contains_all_metrics(self):
        """Results should contain all required metrics."""
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            self.skipTest("sklearn not available")

        results = self.suite.run(self.features, self.sequences, methods=["kmeans"])
        metrics = results["kmeans"]

        # Computational metrics
        self.assertGreater(metrics.fit_time_seconds, 0)
        self.assertGreater(metrics.peak_ram_mb, 0)

        # Cluster statistics
        self.assertGreater(metrics.n_clusters, 0)
        self.assertGreaterEqual(metrics.noise_rate, 0)
        self.assertLessEqual(metrics.noise_rate, 1)

        # Zoo Vox metrics
        self.assertGreaterEqual(metrics.shared_vocabulary_score, 0)
        self.assertGreaterEqual(metrics.lrn_depth, 0)
        self.assertGreater(metrics.vocabulary_utilization, 0)

        # Graded continuity
        self.assertGreaterEqual(metrics.neighborhood_consistency, 0)
        self.assertLessEqual(metrics.neighborhood_consistency, 1)

    def test_benchmark_compares_multiple_methods(self):
        """Should compare multiple methods and print comparison table."""
        try:
            from sklearn.cluster import MiniBatchKMeans
            from sklearn.mixture import BayesianGaussianMixture
        except ImportError:
            self.skipTest("sklearn not available")

        results = self.suite.run(
            self.features,
            self.sequences,
            methods=["kmeans", "bgmm"]
        )

        self.assertEqual(len(results), 2)
        self.assertIn("kmeans", results)
        self.assertIn("bgmm", results)

    def test_svs_calculation(self):
        """Should calculate Shared Vocabulary Score."""
        # Test with simple case
        svs = self.suite._calculate_svs(None)
        self.assertGreaterEqual(svs, 0.0)
        self.assertLessEqual(svs, 1.0)

    def test_neighborhood_consistency(self):
        """Should calculate neighborhood consistency."""
        labels = np.array([0] * 300 + [1] * 300 + [2] * 400)
        consistency = self.suite._calculate_neighborhood_consistency(
            self.features, labels
        )

        self.assertGreater(consistency, 0.5)  # Should be fairly consistent
        self.assertLessEqual(consistency, 1.0)

    def test_neighbor_kl_divergence(self):
        """Should calculate KL divergence between neighbors."""
        # Create soft labels
        soft_labels = np.random.dirichlet([1, 1, 1], size=self.n_samples)

        kl_div = self.suite._calculate_neighbor_kl_divergence(
            self.features, soft_labels
        )

        self.assertGreaterEqual(kl_div, 0.0)


class TestBenchmarkWithRealData(unittest.TestCase):
    """Test benchmark with actual extracted features."""

    @unittest.skip("Requires full extraction - enable manually")
    def test_benchmark_on_112d_bat_data(self):
        """Run full benchmark on 112D bat features."""
        import json

        feature_path = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction_112d_no_cluster.json"

        if not Path(feature_path).exists():
            self.skipTest("Feature file not found")

        # Load features
        with open(feature_path, 'r') as f:
            data = json.load(f)

        # Sample
        n_samples = 10000
        features_list = [seg['features_112d'] for seg in data['segments'][:n_samples]]
        features_112d = np.array(features_list, dtype=np.float32)

        # Dummy sequences
        sequences = [[0, 1, 0], [1, 1, 2]]

        suite = ClusterBenchmarkSuite()
        results = suite.run(features_112d, sequences)

        # Should have results
        self.assertGreater(len(results), 0)


class TestSoftClusteringMetrics(unittest.TestCase):
    """Test soft clustering specific metrics."""

    def test_soft_clustering_has_neighbor_kl_divergence(self):
        """Soft clustering should compute neighbor KL divergence."""
        suite = ClusterBenchmarkSuite()

        # Create synthetic data with smooth transitions
        np.random.seed(42)
        features = np.random.randn(100, 10).astype(np.float32)

        # Create soft labels with gradual change
        soft_labels = np.zeros((100, 3))
        for i in range(100):
            # Gradual transition from cluster 0 to cluster 1
            soft_labels[i, 0] = max(0, 1 - i / 100)
            soft_labels[i, 1] = min(1, i / 100)
            soft_labels[i, 2] = 0.1

        # Normalize
        soft_labels = soft_labels / soft_labels.sum(axis=1, keepdims=True)

        kl_div = suite._calculate_neighbor_kl_divergence(features, soft_labels)

        # Should be relatively low due to smooth transitions
        self.assertLess(kl_div, 2.0)

    def test_graded_boundary_detection(self):
        """Should detect graded boundary segments."""
        suite = ClusterBenchmarkSuite(graded_threshold=0.3)

        # Create soft probabilities with graded boundary
        soft_labels = np.array([
            [0.9, 0.1, 0.0],  # Clear cluster 0
            [0.6, 0.4, 0.0],  # Graded boundary
            [0.1, 0.9, 0.0],  # Clear cluster 1
            [0.4, 0.4, 0.2],  # Highly graded
        ])

        # Detect graded segments
        graded = []
        for probs in soft_labels:
            sorted_probs = sorted(probs, reverse=True)
            is_graded = len(sorted_probs) > 1 and sorted_probs[1] >= suite.graded_threshold
            graded.append(is_graded)

        self.assertEqual(graded, [False, True, False, True])


class TestClusterResult(unittest.TestCase):
    """Test ClusterResult dataclass."""

    def test_cluster_result_creation(self):
        """Should create cluster result with all fields."""
        labels = np.array([0, 1, 0, 1, 2])
        soft_labels = np.array([
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 0.95, 0.05],
            [0.0, 0.1, 0.9],
        ])

        result = ClusterResult(
            method_name="test",
            labels=labels,
            soft_labels=soft_labels,
            fit_time=10.5,
            peak_ram_mb=256.0
        )

        self.assertEqual(result.method_name, "test")
        self.assertEqual(len(result.labels), 5)
        self.assertIsNotNone(result.soft_labels)


class TestBenchmarkMetrics(unittest.TestCase):
    """Test BenchmarkMetrics dataclass."""

    def test_benchmark_metrics_creation(self):
        """Should create metrics with all fields."""
        metrics = BenchmarkMetrics(
            method_name="test",
            fit_time_seconds=10.5,
            peak_ram_mb=256.0,
            n_clusters=10,
            noise_rate=0.05,
            shared_vocabulary_score=0.75,
            lrn_depth=4,
            vocabulary_utilization=0.8,
            neighborhood_consistency=0.85,
            avg_neighbor_kl_divergence=0.3,
            noise_classification_rate=0.05
        )

        self.assertEqual(metrics.method_name, "test")
        self.assertEqual(metrics.fit_time_seconds, 10.5)
        self.assertEqual(metrics.n_clusters, 10)


if __name__ == "__main__":
    unittest.main()
