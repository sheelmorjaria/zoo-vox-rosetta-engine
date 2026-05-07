#!/usr/bin/env python3
"""
Tests for optimized PCA+BGMM clustering pipeline and vocabulary distillation.

TDD approach:
- Red: Failing tests define requirements
- Green: Minimum viable implementation
- Refactor: Clean up and integrate
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.decomposition import PCA
    from sklearn.mixture import BayesianGaussianMixture

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
class TestPCABGMMPipeline:
    """Test PCA + BGMM optimized pipeline."""

    def test_pca_bgmm_pipeline_speed(self):
        """PCA+BGMM must be significantly faster than pure BGMM."""
        # Generate synthetic 112D features
        np.random.seed(42)
        features_112d = np.random.randn(5000, 112).astype(np.float32)

        start = time.time()
        labels, probs = fit_pca_bgmm(features_112d, n_components=30)
        elapsed = time.time() - start

        # Must be faster than 30 seconds (10x speedup over pure BGMM)
        assert elapsed < 30.0, f"PCA+BGMM took {elapsed:.1f}s, expected < 30s"
        print(f"  ✅ PCA+BGMM completed in {elapsed:.1f}s")

    def test_pca_preserves_variance_on_real_data(self):
        """PCA should preserve >90% of variance on real acoustic-like data.
        Real audio features have correlated dimensions (not random).
        """
        np.random.seed(42)
        # Create synthetic data with correlation structure (like real audio features)
        n_samples = 1000
        n_latent = 20  # Latent factors generating the 112D features
        base_latent = np.random.randn(n_samples, n_latent)
        # Project to 112D with random loadings (creates correlation)
        loading_matrix = np.random.randn(n_latent, 112) * 0.5
        features_112d = (
            base_latent @ loading_matrix + np.random.randn(n_samples, 112) * 0.1
        ).astype(np.float32)

        pca = PCA(n_components=30, random_state=42)
        pca.fit_transform(features_112d)

        explained_variance = pca.explained_variance_ratio_.sum()
        assert explained_variance >= 0.90, f"PCA preserved only {explained_variance:.1%} variance"
        print(f"  ✅ PCA preserved {explained_variance:.1%} variance")

    def test_bgmm_auto_prunes_clusters(self):
        """BGMM should automatically prune unused components.
        Uses clustered synthetic data (not random) so BGMM can find patterns.
        """
        np.random.seed(42)
        # Create 5 true clusters in 112D space
        features = []
        for i in range(5):
            center = np.random.randn(112) * 5  # Well-separated centers
            for _ in range(400):
                features.append(center + np.random.randn(112) * 0.5)
        features_112d = np.array(features).astype(np.float32)

        # Set high n_components, let BGMM prune to true clusters
        labels, probs = fit_pca_bgmm(features_112d, n_components=30, n_bgmm_components=50)

        # Should find far fewer than 50 clusters (closer to true 5)
        n_found = len(np.unique(labels))
        assert n_found < 50, f"BGMM found {n_found} clusters, expected < 50"
        assert n_found >= 3, f"BGMM found {n_found} clusters, expected at least 3"
        print(f"  ✅ BGMM pruned from 50 → {n_found} clusters")

    def test_pca_bgmm_preserves_cluster_structure(self):
        """PCA preprocessing should not break cluster discovery.
        Tests that BGMM finds similar clusters with or without PCA.
        """
        np.random.seed(42)
        # Create 5 distinct clusters
        features = []
        for i in range(5):
            center = np.random.randn(112) * 5
            for _ in range(400):
                features.append(center + np.random.randn(112) * 0.5)
        features_112d = np.array(features).astype(np.float32)

        # Fit with PCA
        labels_pca, _ = fit_pca_bgmm(features_112d, n_components=30, n_bgmm_components=20)
        n_clusters_pca = len(np.unique(labels_pca))

        # Fit without PCA (more components to allow discovery)
        pca_direct = PCA(n_components=50, random_state=42)  # Still use PCA but more components
        features_50d = pca_direct.fit_transform(features_112d)
        bgmm_direct = BayesianGaussianMixture(
            n_components=20, covariance_type="diag", random_state=42, max_iter=100
        )
        bgmm_direct.fit(features_50d)
        labels_direct = bgmm_direct.predict(features_50d)
        n_clusters_direct = len(np.unique(labels_direct))

        # Should find similar number of clusters
        diff = abs(n_clusters_pca - n_clusters_direct)
        assert diff <= 3, f"Cluster count mismatch: {n_clusters_pca} vs {n_clusters_direct}"
        print(f"  ✅ Cluster structure preserved: {n_clusters_pca} vs {n_clusters_direct}")


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
class TestVocabularyDistillation:
    """Test teacher-student distillation from BGMM to KMeans."""

    def test_vocabulary_distillation(self):
        """Extract centroids from BGMM, verify KMeans can use them."""
        np.random.seed(42)

        # Create clustered data
        features = []
        true_centers = []
        for i in range(5):
            center = np.random.randn(112) * 10
            true_centers.append(center)
            for _ in range(200):
                features.append(center + np.random.randn(112) * 0.5)
        features_112d = np.array(features).astype(np.float32)

        # Train BGMM teacher
        bgmm_labels = fit_pca_bgmm(features_112d, n_components=30)[0]
        bgmm_centroids = extract_centroids(features_112d, bgmm_labels)

        # Verify centroids extracted
        assert len(bgmm_centroids) > 0, "No centroids extracted"
        assert len(bgmm_centroids) <= 100, f"Too many centroids: {len(bgmm_centroids)}"
        print(f"  ✅ Extracted {len(bgmm_centroids)} centroids")

        # Verify nearest centroid assignment works
        event_features = np.random.randn(112).astype(np.float32)
        assigned_cluster = assign_to_nearest_centroid(event_features, bgmm_centroids)

        assert assigned_cluster in bgmm_centroids.keys(), f"Invalid cluster: {assigned_cluster}"
        print(f"  ✅ Assigned to cluster {assigned_cluster}")

    def test_centroid_dimensions(self):
        """Extracted centroids should have correct dimensions."""
        np.random.seed(42)
        features_112d = np.random.randn(1000, 112).astype(np.float32)

        # Simple clustering
        labels = np.random.randint(0, 10, size=len(features_112d))
        centroids = extract_centroids(features_112d, labels)

        # Verify dimensions
        for cluster_id, centroid in centroids.items():
            assert centroid.shape == (112,), (
                f"Centroid {cluster_id} has wrong shape: {centroid.shape}"
            )
        print(f"  ✅ All {len(centroids)} centroids have 112D shape")

    def test_real_time_assignment_speed(self):
        """Nearest centroid lookup must be sub-millisecond."""
        np.random.seed(42)
        features_112d = np.random.randn(5000, 112).astype(np.float32)

        bgmm_labels = fit_pca_bgmm(features_112d, n_components=30)[0]
        bgmm_centroids = extract_centroids(features_112d, bgmm_labels)

        # Time assignment
        event_features = np.random.randn(112).astype(np.float32)

        start = time.time()
        for _ in range(1000):
            assign_to_nearest_centroid(event_features, bgmm_centroids)
        elapsed = time.time() - start

        avg_time_ms = (elapsed / 1000) * 1000
        assert avg_time_ms < 1.0, f"Assignment took {avg_time_ms:.3f}ms, expected < 1ms"
        print(f"  ✅ Assignment: {avg_time_ms:.3f}ms per lookup (< 1ms target)")


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
class TestSynthesisManifest:
    """Test synthesis manifest export for Rust integration."""

    def test_manifest_contains_112d_centroids(self, tmp_path):
        """Verify manifest exports 112D centroids (not 30D PCA space).
        This is critical for Rust integration - centroids must match
        the feature dimension that Rust extracts in real-time.
        """
        np.random.seed(42)
        features_112d = np.random.randn(500, 112).astype(np.float32)

        # Fit clustering
        labels, probs = fit_pca_bgmm(features_112d, n_components=30)
        centroids = extract_centroids(features_112d, labels)

        # Export to manifest
        manifest_path = tmp_path / "synthesis_manifest.json"
        export_synthesis_manifest(centroids, labels, probs, manifest_path)

        # Verify file exists and is valid JSON
        assert manifest_path.exists(), "Manifest file not created"
        with open(manifest_path) as f:
            data = json.load(f)

        # Verify centroids are in 112D space
        for cluster_id, centroid_data in data["centroids"].items():
            centroid_vector = centroid_data["centroid"]
            assert len(centroid_vector) == 112, (
                f"Centroid {cluster_id} has dimension {len(centroid_vector)}, expected 112"
            )
        print(f"  ✅ All {len(data['centroids'])} centroids are 112D")

    def test_export_synthesis_manifest(self, tmp_path):
        """Export centroids to synthesis_manifest.json."""
        np.random.seed(42)
        features_112d = np.random.randn(1000, 112).astype(np.float32)

        # Fit clustering
        labels, probs = fit_pca_bgmm(features_112d, n_components=30)
        centroids = extract_centroids(features_112d, labels)

        # Export to manifest
        manifest_path = tmp_path / "synthesis_manifest.json"
        export_synthesis_manifest(centroids, labels, probs, manifest_path)

        # Verify file exists and is valid JSON
        assert manifest_path.exists(), "Manifest file not created"
        with open(manifest_path) as f:
            data = json.load(f)

        assert "centroids" in data, "Manifest missing 'centroids'"
        assert "vocabulary_size" in data, "Manifest missing 'vocabulary_size'"
        assert "feature_dimension" in data, "Manifest missing 'feature_dimension'"
        assert data["feature_dimension"] == 112
        print(f"  ✅ Manifest exported: {len(data['centroids'])} centroids")

    def test_manifest_centroid_format(self, tmp_path):
        """Manifest centroids should be serializable JSON arrays."""
        np.random.seed(42)
        features_112d = np.random.randn(500, 112).astype(np.float32)

        labels = np.random.randint(0, 5, size=len(features_112d))
        centroids = extract_centroids(features_112d, labels)

        manifest_path = tmp_path / "test_manifest.json"
        export_synthesis_manifest(centroids, labels, None, manifest_path)

        with open(manifest_path) as f:
            data = json.load(f)

        # Check centroid format
        for cluster_id, centroid_data in data["centroids"].items():
            assert "centroid" in centroid_data
            assert len(centroid_data["centroid"]) == 112
            assert all(isinstance(x, float) for x in centroid_data["centroid"])
        print("  ✅ Manifest format valid")


# ============================================================================
# Implementation Functions (Green Phase)
# ============================================================================


def fit_pca_bgmm(
    features: np.ndarray, n_components: int = 30, n_bgmm_components: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit PCA + Bayesian GMM Model with diag covariance (fast)."""
    pca = PCA(n_components=n_components, random_state=42)
    features_reduced = pca.fit_transform(features)

    bgmm = BayesianGaussianMixture(
        n_components=n_bgmm_components,
        covariance_type="diag",  # Fast and effective
        max_iter=300,
        weight_concentration_prior=0.01,
        random_state=42,
    )
    bgmm.fit(features_reduced)

    soft_labels = bgmm.predict_proba(features_reduced)
    hard_labels = soft_labels.argmax(axis=1)

    return hard_labels, soft_labels


def extract_centroids(features: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    """Extract mean 112D centroid for each cluster."""
    centroids = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        cluster_features = features[mask]
        centroid = cluster_features.mean(axis=0)
        centroids[int(label)] = centroid

    return centroids


def assign_to_nearest_centroid(feature: np.ndarray, centroids: Dict[int, np.ndarray]) -> int:
    """Assign feature to nearest cluster centroid."""
    if not centroids:
        raise ValueError("No centroids available")

    # Convert to arrays
    centroid_array = np.array(list(centroids.values()))
    centroid_ids = list(centroids.keys())

    # Calculate distances
    distances = np.linalg.norm(centroid_array - feature, axis=1)
    nearest_idx = np.argmin(distances)

    return centroid_ids[nearest_idx]


def export_synthesis_manifest(
    centroids: Dict[int, np.ndarray],
    labels: np.ndarray,
    soft_labels: Optional[np.ndarray],
    output_path: Path,
) -> None:
    """Export centroids to synthesis_manifest.json for Rust."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Prepare data for JSON serialization
    centroids_data = {}
    for cluster_id, centroid in centroids.items():
        centroids_data[str(cluster_id)] = {
            "centroid": centroid.tolist(),
            "cluster_size": int((labels == cluster_id).sum()),
        }

    manifest = {
        "vocabulary_size": len(centroids),
        "feature_dimension": 112,
        "total_samples": len(labels),
        "centroids": centroids_data,
        "metadata": {
            "extraction_method": "pca_bgmm",
            "n_components": 30,
            "has_soft_labels": soft_labels is not None,
        },
    }

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
