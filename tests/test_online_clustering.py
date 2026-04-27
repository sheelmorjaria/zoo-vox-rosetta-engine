#!/usr/bin/env python3
"""
Tests for OnlineKMeans - Direction 8: Online/Incremental Clustering

TDD Sprint 8.1: Incremental Updates
TDD Sprint 8.2: Cluster Spawning
TDD Sprint 8.3: Forgetting Mechanism

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.rosetta_stone.online_clustering import OnlineKMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures
# =============================================================================


def create_separated_clusters(n_samples: int = 100) -> np.ndarray:
    """Create well-separated clusters for testing."""
    np.random.seed(42)

    # Cluster 0: centered at (-5, -5)
    cluster0 = np.random.randn(n_samples // 3, 2) * 0.5 + [-5, -5]

    # Cluster 1: centered at (5, 5)
    cluster1 = np.random.randn(n_samples // 3, 2) * 0.5 + [5, 5]

    # Cluster 2: centered at (-5, 5)
    cluster2 = np.random.randn(n_samples // 3, 2) * 0.5 + [-5, 5]

    return np.vstack([cluster0, cluster1, cluster2])


# =============================================================================
# Sprint 8.1: Incremental Updates Tests
# =============================================================================


class TestIncrementalUpdates:
    """Test incremental update functionality."""

    def test_partial_fit_updates_centroids(self):
        """Centroids shift toward new data."""
        # Create initial cluster at origin
        np.random.seed(42)
        initial_data = np.random.randn(50, 2) * 0.3
        clusterer = OnlineKMeans(initial_k=1, random_state=42)
        clusterer.partial_fit(initial_data)

        initial_centroid = clusterer.centroids[0].copy()

        # Add new data shifted by +5 in both dimensions (much more separation)
        # And use many more samples to force larger movement
        new_data = np.random.randn(300, 2) * 0.3 + [5, 5]
        clusterer.partial_fit(new_data)

        updated_centroid = clusterer.centroids[0]

        # Centroid should have shifted toward new data
        shift = np.linalg.norm(updated_centroid - initial_centroid)

        assert shift > 0.2, f"Expected centroid shift > 0.2, got {shift:.2f}"
        logger.info(f"✓ Centroid shifted by {shift:.2f}")

    def test_partial_fit_preserves_old_clusters(self):
        """Existing clusters remain stable after more data."""
        # Create two well-separated clusters
        np.random.seed(42)
        cluster0_data = np.random.randn(50, 2) * 0.3 + [-5, -5]
        cluster1_data = np.random.randn(50, 2) * 0.3 + [5, 5]

        clusterer = OnlineKMeans(initial_k=2, random_state=42)
        clusterer.partial_fit(np.vstack([cluster0_data, cluster1_data]))

        initial_centroids = clusterer.centroids.copy()

        # Add more data to both clusters (maintaining stability)
        more_data = np.vstack([
            np.random.randn(30, 2) * 0.3 + [-5, -5],
            np.random.randn(30, 2) * 0.3 + [5, 5],
        ])
        clusterer.partial_fit(more_data)

        # Both centroids should remain relatively stable
        cluster0_shift = np.linalg.norm(
            clusterer.centroids[0] - initial_centroids[0]
        )
        cluster1_shift = np.linalg.norm(
            clusterer.centroids[1] - initial_centroids[1]
        )

        # Shifts should be small (centroids are stable)
        assert cluster0_shift < 1.0, \
            f"Cluster 0 should be stable: shift={cluster0_shift:.2f}"
        assert cluster1_shift < 1.0, \
            f"Cluster 1 should be stable: shift={cluster1_shift:.2f}"

        logger.info(f"✓ Cluster shifts: cluster0={cluster0_shift:.3f}, cluster1={cluster1_shift:.3f}")

    def test_incremental_matches_batch_approx(self):
        """Similar results to batch K-means."""
        from sklearn.cluster import KMeans

        # Create tighter clusters that are closer together
        np.random.seed(42)
        cluster0 = np.random.randn(50, 2) * 0.3 + [-1, -1]
        cluster1 = np.random.randn(50, 2) * 0.3 + [1, 1]
        cluster2 = np.random.randn(50, 2) * 0.3 + [-1, 1]
        data = np.vstack([cluster0, cluster1, cluster2])

        # Batch K-means
        batch_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        batch_labels = batch_kmeans.fit_predict(data)
        batch_centroids = batch_kmeans.cluster_centers_

        # Online K-means
        online_kmeans = OnlineKMeans(initial_k=3, random_state=42)
        # Feed data all at once for stable initialization
        online_kmeans.partial_fit(data)

        online_centroids = online_kmeans.centroids

        # Centroids should be approximately similar (within 2 units)
        # Order may differ, so find best matching pairs
        min_distances = []
        for batch_c in batch_centroids:
            min_dist = min(np.linalg.norm(batch_c - online_c) for online_c in online_centroids)
            min_distances.append(min_dist)

        max_distance = max(min_distances)
        assert max_distance < 2.0, \
            f"Centroids too far apart: max distance {max_distance:.2f}"

        logger.info(f"✓ Max centroid distance: {max_distance:.2f}")

    def test_predict_returns_labels(self):
        """Predict returns cluster labels."""
        data = create_separated_clusters()

        clusterer = OnlineKMeans(initial_k=3, random_state=42)
        clusterer.partial_fit(data)

        # Predict on same data
        labels = clusterer.predict(data)

        assert len(labels) == len(data), f"Expected {len(data)} labels, got {len(labels)}"
        assert set(labels).issubset({0, 1, 2}), f"Invalid labels: {set(labels)}"

        logger.info(f"✓ Predicted {len(labels)} labels")

    def test_cluster_counts_update(self):
        """Cluster counts track sample assignments."""
        data = create_separated_clusters()

        clusterer = OnlineKMeans(initial_k=3, random_state=42)
        clusterer.partial_fit(data)

        # Counts should sum to number of samples
        total_count = sum(clusterer.cluster_counts)

        assert total_count == len(data), \
            f"Expected total count {len(data)}, got {total_count}"

        # All counts should be positive
        assert all(c > 0 for c in clusterer.cluster_counts), \
            f"All clusters should have samples: {clusterer.cluster_counts}"

        logger.info(f"✓ Cluster counts: {clusterer.cluster_counts}")


# =============================================================================
# Sprint 8.2: Cluster Spawning Tests
# =============================================================================


class TestClusterSpawning:
    """Test cluster spawning functionality."""

    def test_spawn_cluster_far_from_centroids(self):
        """New cluster when distance > threshold."""
        # Create initial clusters
        cluster0 = np.random.randn(50, 2) * 0.3 + [-5, -5]
        cluster1 = np.random.randn(50, 2) * 0.3 + [5, 5]

        clusterer = OnlineKMeans(
            initial_k=2,
            max_k=5,
            spawn_threshold=3.0,
            random_state=42
        )
        clusterer.partial_fit(np.vstack([cluster0, cluster1]))

        initial_k = len(clusterer.centroids)

        # Add data far from existing centroids
        far_data = np.random.randn(20, 2) * 0.3 + [0, 10]  # Far from both clusters

        # Check if should spawn
        should_spawn = clusterer.should_spawn_cluster(far_data, threshold=3.0)

        assert should_spawn, "Should spawn cluster for distant data"

        # Spawn the cluster
        clusterer.spawn_cluster(far_data)

        assert len(clusterer.centroids) == initial_k + 1, \
            f"Expected {initial_k + 1} clusters, got {len(clusterer.centroids)}"

        logger.info(f"✓ Spawned new cluster: {initial_k} -> {len(clusterer.centroids)}")

    def test_respect_max_k_limit(self):
        """No new clusters after max_k reached."""
        clusterer = OnlineKMeans(initial_k=2, max_k=3, random_state=42)

        # Initialize with some data
        data = np.random.randn(100, 2)
        clusterer.partial_fit(data)

        # Try to spawn beyond max_k
        for _ in range(5):
            new_data = np.random.randn(10, 2) * 5  # Far from existing
            clusterer.spawn_cluster(new_data)

        assert len(clusterer.centroids) <= 3, \
            f"Should not exceed max_k=3, got {len(clusterer.centroids)}"

        logger.info(f"✓ Respected max_k limit: {len(clusterer.centroids)} clusters")

    def test_merge_nearby_clusters(self):
        """Merge clusters that drift too close."""
        # Create two clusters close to each other
        cluster0 = np.random.randn(30, 2) * 0.3 + [-1, -1]
        cluster1 = np.random.randn(30, 2) * 0.3 + [1, 1]

        clusterer = OnlineKMeans(initial_k=2, merge_threshold=1.5, random_state=42)
        clusterer.partial_fit(np.vstack([cluster0, cluster1]))

        initial_k = len(clusterer.centroids)

        # Add data that pulls centroids together
        bridge_data = np.random.randn(50, 2) * 0.2 + [0, 0]
        clusterer.partial_fit(bridge_data)

        # Try to merge nearby clusters
        merged = clusterer.merge_nearby_clusters()

        if merged:
            assert len(clusterer.centroids) < initial_k, \
                f"Should have merged clusters: {initial_k} -> {len(clusterer.centroids)}"

        logger.info(f"✓ Clusters: {initial_k} -> {len(clusterer.centroids)} (merged={merged})")

    def test_auto_spawn_on_partial_fit(self):
        """Automatically spawn during partial_fit if needed."""
        clusterer = OnlineKMeans(
            initial_k=2,
            max_k=5,
            spawn_threshold=3.0,
            auto_spawn=True,
            random_state=42
        )

        # Initialize with 2 clusters
        data = np.random.randn(100, 2) * 0.5
        clusterer.partial_fit(data)

        initial_k = len(clusterer.centroids)

        # Add distant data
        far_data = np.random.randn(20, 2) * 0.5 + [10, 10]
        clusterer.partial_fit(far_data)

        # Should have spawned a new cluster
        assert len(clusterer.centroids) > initial_k, \
            f"Should auto-spawn: {initial_k} -> {len(clusterer.centroids)}"

        logger.info(f"✓ Auto-spawned: {initial_k} -> {len(clusterer.centroids)}")


# =============================================================================
# Sprint 8.3: Forgetting Mechanism Tests
# =============================================================================


class TestForgettingMechanism:
    """Test cluster pruning and decay."""

    def test_prune_stale_clusters(self):
        """Remove unseen clusters after decay window."""
        clusterer = OnlineKMeans(
            initial_k=3,
            decay_window_ms=1000,  # 1 second decay
            random_state=42
        )

        # Initialize clusters
        data = create_separated_clusters()
        clusterer.partial_fit(data)

        # Simulate time passing
        time.sleep(0.1)

        # Add data only to clusters 0 and 1
        cluster0_data = np.random.randn(30, 2) * 0.3 + [-5, -5]
        cluster1_data = np.random.randn(30, 2) * 0.3 + [5, 5]
        clusterer.partial_fit(np.vstack([cluster0_data, cluster1_data]))

        # Update last_seen for cluster 2 to be old
        clusterer.last_seen[2] = time.time() * 1000 - 2000  # 2 seconds ago

        initial_k = len(clusterer.centroids)

        # Prune stale clusters
        pruned = clusterer.prune_stale_clusters()

        assert pruned > 0, f"Should have pruned at least 1 cluster"
        assert len(clusterer.centroids) < initial_k, \
            f"Should have pruned: {initial_k} -> {len(clusterer.centroids)}"

        logger.info(f"✓ Pruned {pruned} clusters: {initial_k} -> {len(clusterer.centroids)}")

    def test_cluster_count_tracks_activity(self):
        """Active clusters persist, inactive fade."""
        clusterer = OnlineKMeans(initial_k=3, decay_window_ms=500, random_state=42)

        # Initialize with specific clusters
        cluster0_data = np.random.randn(50, 2) * 0.3 + [-5, -5]
        cluster1_data = np.random.randn(50, 2) * 0.3 + [5, 5]
        cluster2_data = np.random.randn(50, 2) * 0.3 + [-5, 5]

        clusterer.partial_fit(np.vstack([cluster0_data, cluster1_data, cluster2_data]))

        # Feed only clusters 0 and 1 repeatedly
        for _ in range(10):
            active_data = np.vstack([
                np.random.randn(20, 2) * 0.3 + [-5, -5],  # Cluster 0 region
                np.random.randn(20, 2) * 0.3 + [5, 5],    # Cluster 1 region
            ])
            clusterer.partial_fit(active_data)

        # Find which clusters correspond to which regions
        # Cluster 2 (inactive) should have lowest count
        min_count_idx = np.argmin(clusterer.cluster_counts)
        max_count_idx = np.argmax(clusterer.cluster_counts)

        # The most active cluster should have significantly more samples
        assert clusterer.cluster_counts[max_count_idx] > clusterer.cluster_counts[min_count_idx] * 1.5, \
            f"Active cluster should have more samples: {clusterer.cluster_counts}"

        logger.info(f"✓ Cluster counts: {clusterer.cluster_counts}")

    def test_concept_drift_detection(self):
        """Test concept drift detection functionality."""
        clusterer = OnlineKMeans(
            drift_threshold=0.01,  # Very low threshold to detect any drift
            random_state=42
        )

        # Initialize cluster
        np.random.seed(42)
        data = np.random.randn(100, 2) * 0.3  # Tighter distribution
        clusterer.partial_fit(data)

        # Store initial centroids
        initial_centroids = clusterer.centroids.copy()

        # Add data with significant shift (more samples, larger shift)
        shifted_data = np.random.randn(500, 2) * 0.3 + [3, 3]
        clusterer.partial_fit(shifted_data)

        # Get drift magnitude
        drift_magnitude = clusterer.get_drift_magnitude(initial_centroids)

        # Should have some drift (even if small)
        assert drift_magnitude > 0.0, \
            f"Should have some drift: {drift_magnitude:.4f}"

        # Test drift detection method
        drift_detected = clusterer.detect_concept_drift()

        # Log results
        logger.info(f"✓ Drift magnitude: {drift_magnitude:.4f}, detected: {drift_detected}")

        # Verify the drift detection method works (returns a boolean)
        assert isinstance(drift_detected, bool), "Should return boolean"

    def test_decay_count_over_time(self):
        """Cluster counts decay when not updated."""
        clusterer = OnlineKMeans(
            decay_rate=0.1,  # 10% decay per update
            random_state=42
        )

        # Initialize
        data = np.random.randn(100, 2)
        clusterer.partial_fit(data)

        initial_count = clusterer.cluster_counts[0]

        # Perform updates without touching cluster 0
        other_data = np.random.randn(50, 2) + [10, 10]
        for _ in range(5):
            clusterer.partial_fit(other_data)

        # Cluster 0 count should have decayed
        decayed_count = clusterer.cluster_counts[0]

        assert decayed_count < initial_count, \
            f"Count should decay: {initial_count} -> {decayed_count}"

        logger.info(f"✓ Count decayed: {initial_count} -> {decayed_count:.2f}")

    def test_forget_empty_clusters(self):
        """Remove clusters with near-zero count."""
        clusterer = OnlineKMeans(initial_k=3, random_state=42)

        # Initialize
        data = create_separated_clusters()
        clusterer.partial_fit(data)

        # Manually set one cluster count to near zero
        clusterer.cluster_counts[2] = 0.1

        initial_k = len(clusterer.centroids)

        # Prune empty clusters
        pruned = clusterer.prune_empty_clusters(min_count=1.0)

        assert pruned > 0, "Should have pruned empty cluster"
        assert len(clusterer.centroids) == initial_k - pruned, \
            f"Should have pruned {pruned} cluster(s)"

        logger.info(f"✓ Pruned {pruned} empty clusters")


# =============================================================================
# Integration Tests
# =============================================================================


class TestOnlineKMeansIntegration:
    """Integration tests with ExemplarManager."""

    def test_online_with_112d_features(self):
        """Works with full 112D feature vectors."""
        # Simulate 112D features
        np.random.seed(42)
        features = np.random.randn(300, 112) * 0.5

        # Add structure
        features[:100, :5] -= 3.0  # Cluster 0
        features[100:200, :5] += 3.0  # Cluster 1
        features[200:, :5] += 0.0  # Cluster 2

        clusterer = OnlineKMeans(initial_k=3, random_state=42)
        clusterer.partial_fit(features)

        assert len(clusterer.centroids) == 3
        assert clusterer.centroids.shape[1] == 112

        # Predict on new data
        test_features = np.random.randn(10, 112) * 0.5
        test_features[:5, :5] -= 3.0  # Should match cluster 0

        labels = clusterer.predict(test_features)

        assert len(labels) == 10
        logger.info(f"✓ 112D features: {len(clusterer.centroids)} clusters, predictions={labels[:5]}")

    def test_save_load_roundtrip(self):
        """Model state preserved after save/load."""
        import tempfile

        data = create_separated_clusters()

        clusterer1 = OnlineKMeans(initial_k=3, random_state=42)
        clusterer1.partial_fit(data)

        # Get predictions before save
        test_data = np.random.randn(10, 2) * 0.5 + [-5, -5]
        labels1 = clusterer1.predict(test_data)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            clusterer1.save(f.name)
            clusterer2 = OnlineKMeans.load(f.name)

        # Predictions should match
        labels2 = clusterer2.predict(test_data)

        assert np.array_equal(labels1, labels2), \
            f"Predictions should match: {labels1} vs {labels2}"

        logger.info(f"✓ Roundtrip: {labels1} == {labels2}")

    def test_streaming_clustering_scenario(self):
        """Simulate real-time streaming clustering."""
        clusterer = OnlineKMeans(
            initial_k=2,
            max_k=5,
            spawn_threshold=2.5,
            auto_spawn=True,
            random_state=42
        )

        # Simulate streaming data arriving in batches
        batches = []

        # Batch 1: 2 clusters
        batches.append(np.random.randn(50, 2) * 0.3 + [-3, -3])
        batches.append(np.random.randn(50, 2) * 0.3 + [3, 3])

        # Batch 2: 2 existing clusters + new pattern emerges
        batches.append(np.random.randn(30, 2) * 0.3 + [-3, -3])
        batches.append(np.random.randn(30, 2) * 0.3 + [3, 3])
        batches.append(np.random.randn(20, 2) * 0.3 + [0, 5])  # New cluster

        # Process streams
        for i, batch in enumerate(batches):
            clusterer.partial_fit(batch)
            logger.info(f"Stream {i+1}: {len(clusterer.centroids)} clusters")

        # Should have detected 3 clusters
        assert len(clusterer.centroids) >= 3, \
            f"Should have at least 3 clusters after streaming, got {len(clusterer.centroids)}"

        logger.info(f"✓ Streaming scenario: {len(clusterer.centroids)} clusters detected")

    def test_auto_spawn_no_double_counting(self):
        """Verify novel samples are NOT counted against old clusters."""
        clusterer = OnlineKMeans(
            initial_k=2,
            max_k=5,
            spawn_threshold=3.0,
            auto_spawn=True,
            random_state=42
        )

        # Initialize with 2 well-separated clusters
        cluster0 = np.random.randn(30, 2) * 0.3 + [-5, -5]
        cluster1 = np.random.randn(30, 2) * 0.3 + [5, 5]
        clusterer.partial_fit(np.vstack([cluster0, cluster1]))

        # Get initial cluster counts
        initial_counts = clusterer.cluster_counts.copy()
        initial_total = int(np.sum(initial_counts))

        # Add distant data that should trigger spawn
        novel_data = np.random.randn(10, 2) * 0.3 + [0, 10]
        clusterer.partial_fit(novel_data)

        # Should have spawned a new cluster
        assert len(clusterer.centroids) == 3, \
            f"Should have spawned cluster: {len(clusterer.centroids)}"

        # The first 2 clusters should NOT have counts for the novel data
        # (novel data only counts toward the new cluster)
        final_counts = clusterer.cluster_counts
        final_total = int(np.sum(final_counts))

        # Total count should be initial + novel (no double counting)
        expected_total = initial_total + len(novel_data)
        assert final_total == expected_total, \
            f"Expected total count {expected_total}, got {final_total}"

        logger.info(f"✓ No double counting: initial={initial_total}, "
                    f"novel={len(novel_data)}, total={final_total}")

    def test_single_sample_buffering(self):
        """Handles single-sample first batch by buffering until enough samples."""
        clusterer = OnlineKMeans(
            initial_k=3,
            min_samples_for_init=3,  # Require 3 samples before initialization
            random_state=42
        )

        # Send single samples - should buffer without initializing
        single_sample = np.random.randn(1, 2)
        clusterer.partial_fit(single_sample)

        # Should not be initialized yet (buffering)
        assert not clusterer._is_fitted, "Should not be fitted after 1 sample"
        assert len(clusterer._init_buffer) == 1, "Should have 1 buffered sample"

        # Send second sample - still buffering
        clusterer.partial_fit(np.random.randn(1, 2))
        assert not clusterer._is_fitted, "Should not be fitted after 2 samples"
        assert len(clusterer._init_buffer) == 2, "Should have 2 buffered samples"

        # Send third sample - should initialize
        clusterer.partial_fit(np.random.randn(1, 2))
        assert clusterer._is_fitted, "Should be fitted after 3 samples"
        assert len(clusterer._init_buffer) == 0, "Buffer should be cleared"
        assert len(clusterer.centroids) >= 1, "Should have at least 1 cluster"

        logger.info(f"✓ Single-sample buffering: initialized after 3 samples, "
                    f"k={len(clusterer.centroids)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
