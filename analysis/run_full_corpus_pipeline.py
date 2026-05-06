#!/usr/bin/env python3
"""
Full Corpus Teacher-Student Distillation Pipeline.

This script implements the complete Teacher-Student clustering pipeline:
1. Teacher (Offline): PCA + BGMM discovers true vocabulary on a subset
2. Student (Online): Nearest centroid lookup assigns all segments

Pipeline:
- Loads 112D features from extraction
- Trains Teacher on 100k subset (for speed)
- Assigns all 8.9M segments using Student
- Exports synthesis_manifest.json for Rust

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.decomposition import PCA
    from sklearn.mixture import BayesianGaussianMixture
    from scipy.spatial.distance import cdist
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Error: sklearn not available")
    sys.exit(1)


def student_predict(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Student prediction: nearest centroid lookup in 112D space."""
    features_2d = np.atleast_2d(features)
    distances = cdist(features_2d, centroids, metric='euclidean')
    labels = np.argmin(distances, axis=1)
    if features.ndim == 1:
        return labels[0]
    return labels


class MiniBatchBGMMTeacher:
    """Teacher model: PCA + BGMM for offline vocabulary discovery."""

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
        """Fit the Teacher on a subset of the data."""
        np.random.seed(self.random_state)

        # Step 1: Subsample for tractable EM training
        n_samples = len(features_112d)
        subset_size = min(self.sample_size, n_samples)

        if n_samples > self.sample_size:
            sample_indices = np.random.choice(n_samples, subset_size, replace=False)
            sample = features_112d[sample_indices]
        else:
            sample = features_112d

        print(f'  Teacher: Training on {subset_size:,} samples (subset of {n_samples:,})')

        # Step 2: PCA Reduction
        print(f'  Teacher: Reducing 112D → {self.pca_components}D with PCA...')
        self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
        reduced = self.pca.fit_transform(sample)

        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f'  Teacher: PCA preserved {explained_var:.1%} variance')

        # Step 3: Train BGMM
        print(f'  Teacher: Fitting BGMM (max_components={self.max_clusters})...')
        bgmm_start = time.time()

        self.bgmm = BayesianGaussianMixture(
            n_components=self.max_clusters,
            covariance_type='diag',
            max_iter=300,
            weight_concentration_prior=0.01,
            random_state=self.random_state
        )
        self.bgmm.fit(reduced)

        bgmm_time = time.time() - bgmm_start
        print(f'  Teacher: BGMM fit in {bgmm_time:.1f}s')

        # Step 4: Filter pruned clusters
        active_mask = self.bgmm.weights_ > self.weight_threshold
        self.vocabulary_size = int(active_mask.sum())

        print(f'  Teacher: Discovered {self.vocabulary_size} clusters (pruned from {self.max_clusters})')

        # Step 5: Project centroids back to 112D
        centroids_reduced = self.bgmm.means_[active_mask]
        self.centroids_112d = self.pca.inverse_transform(centroids_reduced)

        return self.centroids_112d, self.vocabulary_size

    def predict(self, features_112d: np.ndarray) -> np.ndarray:
        """Predict cluster labels using Student inference."""
        if self.centroids_112d is None:
            raise ValueError("Teacher not fitted. Call fit() first.")
        return student_predict(features_112d, self.centroids_112d)

    def export_for_rust(self, cluster_stats: Optional[Dict] = None) -> Dict:
        """Export centroids in Rust-compatible format."""
        if self.centroids_112d is None:
            raise ValueError("Teacher not fitted. Call fit() first.")

        clusters_info = {}
        for i, centroid in enumerate(self.centroids_112d):
            stats = cluster_stats.get(i, {}) if cluster_stats else {}
            clusters_info[str(i)] = {
                'cluster_id': int(i),
                'centroid_112d': centroid.tolist(),
                'exemplar_audio': f"cluster_{i}_exemplar.wav",
                'exemplar_features_112d': centroid.tolist(),
                'num_segments': stats.get('count', 0),
                'mean_distance_to_centroid': float(stats.get('mean_distance', 0.0))
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


def compute_cluster_stats(
    features_112d: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray
) -> Dict[int, Dict]:
    """Compute statistics for each cluster."""
    cluster_stats = {}

    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        cluster_features = features_112d[mask]
        centroid = centroids[cluster_id]

        # Mean distance to centroid
        distances = np.linalg.norm(cluster_features - centroid, axis=1)

        cluster_stats[cluster_id] = {
            'count': int(mask.sum()),
            'mean_distance': float(distances.mean()),
            'std_distance': float(distances.std())
        }

    return cluster_stats


def main():
    """Run the full Teacher-Student pipeline."""
    print('╔═══════════════════════════════════════════════════════════════════════════╗')
    print('║     Teacher-Student Distillation Pipeline                                 ║')
    print('║     MiniBatch BGMM Teacher → Rust Student                                 ║')
    print('╚═══════════════════════════════════════════════════════════════════════════╝')
    print()

    # Configuration
    feature_path = Path('/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction_112d_no_cluster.json')
    output_dir = Path('/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d')
    manifest_path = output_dir / 'synthesis_manifest.json'
    labeled_output_path = output_dir / 'extraction_112d_labeled.json'

    if not feature_path.exists():
        print(f'Error: Feature file not found: {feature_path}')
        return

    # Load features
    print(f'Loading features from {feature_path}...')
    with open(feature_path) as f:
        data = json.load(f)

    all_features = np.array([seg['features_112d'] for seg in data['segments']], dtype=np.float32)
    print(f'Loaded {len(all_features):,} segments × 112D')
    print()

    # Configuration
    SAMPLE_SIZE = 100000  # Teacher trains on this many samples
    MAX_CLUSTERS = 150    # BGMM starts with this many

    print('=' * 70)
    print('PHASE 1: Teacher Training (Offline, Batch)')
    print('=' * 70)
    print(f'Training on {min(SAMPLE_SIZE, len(all_features)):,} samples')
    print()

    teacher_start = time.time()

    # Create and fit teacher
    teacher = MiniBatchBGMMTeacher(
        pca_components=30,
        max_clusters=MAX_CLUSTERS,
        sample_size=SAMPLE_SIZE,
        random_state=42
    )

    centroids_112d, vocabulary_size = teacher.fit(all_features)

    teacher_time = time.time() - teacher_start

    print()
    print('=' * 70)
    print('PHASE 2: Student Assignment (Online, Vectorized)')
    print('=' * 70)
    print(f'Assigning {len(all_features):,} segments to {vocabulary_size} clusters')
    print()

    # Assign all segments using Student
    student_start = time.time()
    all_labels = teacher.predict(all_features)
    student_time = time.time() - student_start

    # Compute cluster statistics
    print('Computing cluster statistics...')
    cluster_stats = compute_cluster_stats(all_features, all_labels, centroids_112d)

    print()
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'Total segments: {len(all_features):,}')
    print(f'Vocabulary size: {vocabulary_size}')
    print(f'Teacher training time: {teacher_time:.1f}s')
    print(f'Student assignment time: {student_time:.1f}s')
    print(f'Total pipeline time: {teacher_time + student_time:.1f}s')
    print(f'Student throughput: {len(all_features) / student_time:,.0f} segments/second')
    print()

    # Per-cluster stats
    print('Cluster Statistics:')
    print(f'  {"ID":<6} {"Count":<10} {"Mean Dist":<12} {"Std Dist":<12}')
    print('-' * 45)
    for cluster_id in sorted(cluster_stats.keys()):
        stats = cluster_stats[cluster_id]
        print(f'  {cluster_id:<6} {stats["count"]:<10,} {stats["mean_distance"]:<12.3f} {stats["std_distance"]:<12.3f}')
    print()

    # Export manifest
    print('=' * 70)
    print('EXPORTING FOR RUST INTEGRATION')
    print('=' * 70)

    manifest = teacher.export_for_rust(cluster_stats)

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'Exported synthesis_manifest.json to: {manifest_path}')
    print()

    # Optionally export labeled data
    print('=' * 70)
    print('EXPORTING LABELED DATA')
    print('=' * 70)

    labeled_data = {
        'metadata': data.get('metadata', {}),
        'vocabulary_size': vocabulary_size,
        'num_segments': len(all_labels),
        'segments': []
    }

    for i, (seg, label) in enumerate(zip(data['segments'], all_labels)):
        labeled_seg = seg.copy()
        labeled_seg['cluster_id'] = int(label)
        labeled_seg['cluster_probability'] = 1.0  # Hard assignment
        labeled_data['segments'].append(labeled_seg)

        if i > 0 and i % 100000 == 0:
            print(f'  Processed {i:,} segments...')

    with open(labeled_output_path, 'w') as f:
        json.dump(labeled_data, f, indent=2)

    print(f'Exported labeled data to: {labeled_output_path}')
    print()

    print('╔═══════════════════════════════════════════════════════════════════════════╗')
    print('║     Teacher-Student Pipeline Complete!                                     ║')
    print('╚═══════════════════════════════════════════════════════════════════════════╝')
    print()
    print('Next steps:')
    print('  1. Rust: ExemplarManager.load_from_manifest()')
    print('  2. Rust: Implement real-time nearest centroid lookup')
    print(f'  3. Rust: Update vocabulary_size → {vocabulary_size}')


if __name__ == '__main__':
    main()
