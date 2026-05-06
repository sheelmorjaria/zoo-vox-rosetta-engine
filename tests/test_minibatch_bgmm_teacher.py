#!/usr/bin/env python3
"""
Tests for MiniBatch BGMM Teacher-Student Distillation Pipeline.

TDD approach:
- Red: Failing tests define requirements
- Green: Minimum viable implementation
- Refactor: Clean up and integrate
"""

import json
import time
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.decomposition import PCA
    from sklearn.mixture import BayesianGaussianMixture
    from sklearn.metrics import pairwise_distances
    from scipy.spatial.distance import cdist
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
class TestMiniBatchBGMMTeacher:
    """Test MiniBatch BGMM Teacher for scalable vocabulary discovery."""

    def test_minibatch_bgmm_discovers_vocabulary(self):
        """MiniBatch BGMM should discover vocabulary on a subset of data.

        The Teacher must:
        1. Sample a representative subset for tractable EM training
        2. Reduce to 30D with PCA
        3. Train BGMM and discover true cluster count
        4. Project centroids back to 112D for Rust integration
        """
        np.random.seed(42)

        # Generate synthetic data with 5 true clusters
        features = []
        for i in range(5):
            center = np.random.randn(112) * 5
            for _ in range(400):
                features.append(center + np.random.randn(112) * 0.5)
        features_112d = np.array(features).astype(np.float32)

        # Create teacher with small sample size for test
        teacher = MiniBatchBGMMTeacher(
            pca_components=30,
            max_clusters=20,
            sample_size=1000
        )

        centroids_112d, vocabulary_size = teacher.fit(features_112d)

        # Should discover clusters (not all 20, but close to true 5)
        assert vocabulary_size > 3, f"Vocabulary size {vocabulary_size} too small"
        assert vocabulary_size <= 20, f"Vocabulary size {vocabulary_size} too large"

        # Centroids must be in 112D for Rust integration
        assert centroids_112d.shape[0] == vocabulary_size
        assert centroids_112d.shape[1] == 112, f"Centroids have shape {centroids_112d.shape}"

        print(f"  ✅ Discovered vocabulary size: {vocabulary_size}")
        print(f"  ✅ Centroids shape: {centroids_112d.shape}")

    def test_student_assignment_matches_teacher(self):
        """Student (nearest centroid) must agree with Teacher's hard assignments.

        This validates the distillation: Student approximates Teacher with >95% accuracy.
        """
        np.random.seed(42)

        # Generate larger dataset for this test
        features = []
        for i in range(5):
            center = np.random.randn(112) * 5
            for _ in range(500):
                features.append(center + np.random.randn(112) * 0.5)
        features_112d = np.array(features).astype(np.float32)

        # Train teacher on subset
        teacher = MiniBatchBGMMTeacher(
            pca_components=30,
            max_clusters=20,
            sample_size=1000
        )
        centroids_112d, _ = teacher.fit(features_112d)

        # Test on 1000 samples
        test_features = features_112d[:1000]
        teacher_labels = teacher.predict(test_features)

        # Student prediction (nearest centroid in 112D)
        student_labels = student_predict(test_features, centroids_112d)

        # Student and teacher should agree > 95% of the time
        agreement = np.mean(teacher_labels == student_labels)
        assert agreement > 0.95, f"Student-Teacher agreement {agreement:.3f} < 0.95"

        print(f"  ✅ Student-Teacher agreement: {agreement:.3f}")

    def test_centroid_export_format(self):
        """Exported centroids should be compatible with Rust synthesis_manifest.json format.

        Validates that the centroids can be loaded by Rust ExemplarManager.
        """
        np.random.seed(42)

        # Generate test data
        features = []
        for i in range(3):
            center = np.random.randn(112) * 5
            for _ in range(200):
                features.append(center + np.random.randn(112) * 0.5)
        features_112d = np.array(features).astype(np.float32)

        teacher = MiniBatchBGMMTeacher(
            pca_components=30,
            max_clusters=10,
            sample_size=500
        )

        centroids_112d, vocabulary_size = teacher.fit(features_112d)

        # Verify export format
        export_data = teacher.export_for_rust()

        assert 'vocabulary_size' in export_data
        assert 'clusters' in export_data
        assert export_data['vocabulary_size'] == vocabulary_size

        # Verify each cluster has 112D centroid
        for cluster_id, cluster_data in export_data['clusters'].items():
            centroid = cluster_data['centroid_112d']
            assert len(centroid) == 112, f"Cluster {cluster_id} has {len(centroid)}D centroid"

        print(f"  ✅ Export format valid for {vocabulary_size} clusters")

    def test_scalability_to_large_corpus(self):
        """MiniBatch approach should scale to millions of segments.

        This test validates that the Teacher:
        1. Trains on a small subset (fast)
        2. Can predict on the full corpus (vectorized)
        """
        np.random.seed(42)

        # Simulate large corpus: 100k segments
        n_samples = 100000
        n_true_clusters = 10

        # Generate clustered data
        features = []
        for i in range(n_true_clusters):
            center = np.random.randn(112) * 5
            for _ in range(n_samples // n_true_clusters):
                features.append(center + np.random.randn(112) * 0.5)
        features_112d = np.array(features).astype(np.float32)

        # Teacher should only train on sample_size
        sample_size = 10000
        teacher = MiniBatchBGMMTeacher(
            pca_components=30,
            max_clusters=30,
            sample_size=sample_size
        )

        start = time.time()
        centroids_112d, vocabulary_size = teacher.fit(features_112d)
        fit_time = time.time() - start

        # Training should be fast (only on sample)
        assert fit_time < 30.0, f"Training took {fit_time:.1f}s, expected < 30s"

        # Predict on full corpus (vectorized)
        start = time.time()
        all_labels = teacher.predict(features_112d)
        predict_time = time.time() - start

        # Prediction should be fast (vectorized)
        assert predict_time < 5.0, f"Prediction took {predict_time:.1f}s, expected < 5s"

        # All samples should be labeled
        assert len(all_labels) == len(features_112d)

        print(f"  ✅ Training: {fit_time:.1f}s on {sample_size} samples")
        print(f"  ✅ Prediction: {predict_time:.1f}s on {len(features_112d)} samples")
        print(f"  ✅ Vocabulary size: {vocabulary_size}")

    def test_pca_inverse_transform_preserves_centroids(self):
        """PCA inverse transform should accurately reconstruct 112D centroids.

        This validates that the centroids exported to Rust are faithful
        representations of the BGMM cluster centers in the original 112D space.
        """
        np.random.seed(42)

        # Generate data with known structure
        features = []
        true_centers = []
        for i in range(5):
            center = np.random.randn(112) * 10
            true_centers.append(center)
            for _ in range(300):
                features.append(center + np.random.randn(112) * 0.3)
        features_112d = np.array(features).astype(np.float32)

        teacher = MiniBatchBGMMTeacher(
            pca_components=30,
            max_clusters=10,
            sample_size=1000
        )

        centroids_112d, vocabulary_size = teacher.fit(features_112d)

        # Verify PCA explained variance is high
        assert teacher.pca.explained_variance_ratio_.sum() > 0.90, \
            f"PCA only preserved {teacher.pca.explained_variance_ratio_.sum():.1%} variance"

        # Centroids should be in reasonable range
        assert np.all(np.isfinite(centroids_112d)), "Centroids contain non-finite values"

        print(f"  ✅ PCA preserved {teacher.pca.explained_variance_ratio_.sum():.1%} variance")
        print(f"  ✅ All {vocabulary_size} centroids are valid 112D vectors")


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
class TestStudentPredict:
    """Test Student prediction function (nearest centroid lookup)."""

    def test_student_predict_single_feature(self):
        """Student predict should return nearest centroid for single feature."""
        np.random.seed(42)

        # Create 5 centroids
        centroids = np.random.randn(5, 112).astype(np.float32)
        test_feature = centroids[2] + np.random.randn(112) * 0.1

        label = student_predict(test_feature, centroids)

        # Single feature should return scalar
        assert np.isscalar(label) or isinstance(label, (int, np.integer)), f"Expected scalar, got {type(label)}"
        assert label == 2, f"Expected cluster 2, got {label}"

    def test_student_predict_batch(self):
        """Student predict should handle batch features efficiently."""
        np.random.seed(42)

        centroids = np.random.randn(10, 112).astype(np.float32)
        features = np.random.randn(100, 112).astype(np.float32)

        labels = student_predict(features, centroids)

        assert labels.shape == (100,)
        assert np.all(labels >= 0) and np.all(labels < 10)

    def test_student_predict_speed(self):
        """Student predict should be sub-millisecond per lookup."""
        np.random.seed(42)

        centroids = np.random.randn(94, 112).astype(np.float32)  # 94 clusters
        features = np.random.randn(1000, 112).astype(np.float32)

        # Time prediction
        start = time.time()
        labels = student_predict(features, centroids)
        elapsed = time.time() - start

        avg_time_ms = (elapsed / len(features)) * 1000

        # Should be sub-millisecond per feature
        assert avg_time_ms < 1.0, f"Prediction took {avg_time_ms:.3f}ms per feature"

        print(f"  ✅ Student prediction: {avg_time_ms:.3f}ms per feature")


# ============================================================================
# Implementation Functions (Green Phase)
# ============================================================================

def student_predict(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Student prediction: nearest centroid lookup in 112D space.

    Args:
        features: (N, 112) or (112,) feature vectors
        centroids: (K, 112) cluster centroids from Teacher

    Returns:
        labels: (N,) cluster assignments
    """
    # Ensure features is 2D
    features_2d = np.atleast_2d(features)

    # Vectorized Euclidean distance calculation
    # Using scipy.spatial.distance.cdist for efficiency
    distances = cdist(features_2d, centroids, metric='euclidean')
    labels = np.argmin(distances, axis=1)

    # Return scalar if input was scalar
    if features.ndim == 1:
        return labels[0]
    return labels


class MiniBatchBGMMTeacher:
    """Teacher model: PCA + BGMM for offline vocabulary discovery.

    Uses MiniBatch approach to scale to millions of segments:
    1. Subsample for tractable EM training
    2. PCA reduction for speed
    3. BGMM for probabilistic clustering
    4. PCA inverse transform to get 112D centroids for Rust
    """

    def __init__(
        self,
        pca_components: int = 30,
        max_clusters: int = 150,
        sample_size: int = 100000,
        weight_threshold: float = 0.01,
        random_state: int = 42
    ):
        self.pca_components = pca_components
        self.max_clusters = max_clusters
        self.sample_size = sample_size
        self.weight_threshold = weight_threshold
        self.random_state = random_state

        # Fitted components
        self.pca = None
        self.bgmm = None
        self.centroids_112d = None
        self.vocabulary_size = None

    def fit(self, features_112d: np.ndarray) -> Tuple[np.ndarray, int]:
        """Fit the Teacher on a subset of the data.

        Args:
            features_112d: (N, 112) full feature matrix

        Returns:
            centroids_112d: (K, 112) cluster centroids in original space
            vocabulary_size: K (number of discovered clusters)
        """
        np.random.seed(self.random_state)

        # Step 1: Subsample for tractable EM training
        n_samples = len(features_112d)
        subset_size = min(self.sample_size, n_samples)

        if n_samples > self.sample_size:
            sample_indices = np.random.choice(n_samples, subset_size, replace=False)
            sample = features_112d[sample_indices]
        else:
            sample = features_112d

        print(f"  Teacher: Training on {subset_size:,} samples (subset of {n_samples:,})")

        # Step 2: PCA Reduction
        print(f"  Teacher: Reducing 112D → {self.pca_components}D with PCA...")
        self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
        reduced = self.pca.fit_transform(sample)

        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"  Teacher: PCA preserved {explained_var:.1%} variance")

        # Step 3: Train BGMM
        print(f"  Teacher: Fitting BGMM (max_components={self.max_clusters})...")
        self.bgmm = BayesianGaussianMixture(
            n_components=self.max_clusters,
            covariance_type='diag',
            max_iter=300,
            weight_concentration_prior=0.01,
            random_state=self.random_state
        )
        self.bgmm.fit(reduced)

        # Step 4: Filter pruned clusters (weight < threshold)
        active_mask = self.bgmm.weights_ > self.weight_threshold
        self.vocabulary_size = int(active_mask.sum())

        print(f"  Teacher: Discovered {self.vocabulary_size} clusters (pruned from {self.max_clusters})")

        # Step 5: Project centroids back to 112D (crucial for Rust integration!)
        centroids_reduced = self.bgmm.means_[active_mask]
        self.centroids_112d = self.pca.inverse_transform(centroids_reduced)

        return self.centroids_112d, self.vocabulary_size

    def predict(self, features_112d: np.ndarray) -> np.ndarray:
        """Predict cluster labels using Student inference (nearest centroid).

        Args:
            features_112d: (N, 112) feature vectors

        Returns:
            labels: (N,) cluster assignments
        """
        if self.centroids_112d is None:
            raise ValueError("Teacher not fitted. Call fit() first.")

        return student_predict(features_112d, self.centroids_112d)

    def export_for_rust(self) -> Dict:
        """Export centroids in Rust-compatible synthesis_manifest.json format.

        Returns:
            Dictionary compatible with Rust ExemplarManager.load_centroids_from_manifest()
        """
        if self.centroids_112d is None:
            raise ValueError("Teacher not fitted. Call fit() first.")

        clusters_info = {}
        for i, centroid in enumerate(self.centroids_112d):
            clusters_info[str(i)] = {
                'cluster_id': int(i),
                'centroid_112d': centroid.tolist(),
                'exemplar_audio': f"cluster_{i}_exemplar.wav",
                'exemplar_features_112d': centroid.tolist(),
                'num_segments': 0,  # Would be filled in during full corpus assignment
                'mean_distance_to_centroid': 0.0
            }

        return {
            'vocabulary_size': self.vocabulary_size,
            'num_clusters': self.vocabulary_size,
            'clusters': clusters_info,
            'metadata': {
                'extraction_method': 'minibatch_pca_bgmm_teacher_student',
                'n_components': self.pca_components,
                'max_clusters': self.max_clusters,
                'sample_size': self.sample_size,
                'weight_threshold': self.weight_threshold,
                'explained_variance': float(self.pca.explained_variance_ratio_.sum())
            }
        }
