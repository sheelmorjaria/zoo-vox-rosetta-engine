#!/usr/bin/env python3
"""
PCA+BGMM optimized clustering pipeline with teacher-student distillation.

This script:
1. Loads 112D features from bat extraction
2. Runs PCA dimensionality reduction (112D → 30D)
3. Runs Bayesian GMM to discover true vocabulary size
4. Extracts centroids for real-time inference
5. Exports synthesis_manifest.json for Rust integration
"""

import json
import time
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture


def main():
    # Path configuration
    feature_path = Path('/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction_112d_no_cluster.json')
    output_dir = Path('/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d')
    manifest_path = output_dir / 'synthesis_manifest.json'

    print(f'Loading features from {feature_path}...')

    with open(feature_path) as f:
        data = json.load(f)

    all_features = np.array([seg['features_112d'] for seg in data['segments']], dtype=np.float32)
    print(f'Loaded {len(all_features):,} segments × 112D')

    # Sample for BGMM (use 100k segments for speed)
    MAX_SAMPLES = 100000
    if len(all_features) > MAX_SAMPLES:
        print(f'Sampling {MAX_SAMPLES:,} segments for BGMM training...')
        indices = np.random.choice(len(all_features), MAX_SAMPLES, replace=False)
        features_112d = all_features[indices]
    else:
        features_112d = all_features

    print(f'Using {len(features_112d):,} segments for clustering')
    print()
    print('=' * 70)
    print('PCA + BGMM Pipeline on Real Bat Data')
    print('=' * 70)
    print()

    start = time.time()

    # Step 1: PCA dimensionality reduction
    print('Step 1: PCA (112D → 30D)')
    pca = PCA(n_components=30, random_state=42)
    features_reduced = pca.fit_transform(features_112d)
    explained_variance = pca.explained_variance_ratio_.sum()
    pca_time = time.time() - start

    print(f'  Completed in {pca_time:.1f}s')
    print(f'  Variance preserved: {explained_variance:.1%}')
    print()

    # Step 2: Bayesian GMM clustering
    print('Step 2: Bayesian GMM (diag covariance)')
    bgmm_start = time.time()
    bgmm = BayesianGaussianMixture(
        n_components=100,
        covariance_type='diag',  # Fast and effective
        max_iter=300,
        weight_concentration_prior=0.01,
        random_state=42
    )
    bgmm.fit(features_reduced)
    bgmm_time = time.time() - bgmm_start

    soft_labels = bgmm.predict_proba(features_reduced)
    hard_labels = soft_labels.argmax(axis=1)

    total_time = time.time() - start

    # Results
    unique_clusters = len(np.unique(hard_labels))
    print(f'  Completed in {bgmm_time:.1f}s')
    print(f'  Clusters discovered: {unique_clusters}')
    print()

    print('=' * 70)
    print('Summary')
    print('=' * 70)
    print(f'  Total time: {total_time:.1f}s')
    print(f'  PCA time: {pca_time:.1f}s')
    print(f'  BGMM time: {bgmm_time:.1f}s')
    print(f'  Speedup vs pure BGMM: ~{383/total_time:.0f}x')
    print(f'  Vocabulary size: {unique_clusters}')
    print()

    # Step 3: Extract centroids for real-time inference
    print('=' * 70)
    print('Extracting Centroids for Real-Time Inference')
    print('=' * 70)

    centroids = {}
    for label in np.unique(hard_labels):
        mask = hard_labels == label
        centroids[int(label)] = features_112d[mask].mean(axis=0)

    print(f'  Extracted {len(centroids)} centroids in 112D')
    print()

    # Test real-time assignment speed
    print('Testing real-time assignment speed...')
    test_feature = features_112d[0]
    assignment_start = time.time()
    for _ in range(1000):
        centroid_array = np.array(list(centroids.values()))
        distances = np.linalg.norm(centroid_array - test_feature, axis=1)
        _ = np.argmin(distances)
    assignment_time = (time.time() - assignment_start) / 1000 * 1000

    print(f'  Assignment speed: {assignment_time:.3f}ms per lookup')
    print(f'  Target: < 1ms ✅' if assignment_time < 1.0 else f'  Target: < 1ms ❌')
    print()

    # Step 4: Find exemplar audio for each cluster
    print('=' * 70)
    print('Selecting Exemplar Audio for Each Cluster')
    print('=' * 70)

    # For each cluster, find the segment closest to centroid
    clusters_info = {}
    for cluster_id, centroid in centroids.items():
        mask = hard_labels == cluster_id
        cluster_features = features_112d[mask]

        # Calculate distances to centroid
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        nearest_idx = np.argmin(distances)

        # Find the original segment index for this exemplar
        cluster_indices = np.where(mask)[0]
        exemplar_sample_idx = cluster_indices[nearest_idx]

        # For now, use a placeholder audio path
        # In production, this would reference the actual audio file
        exemplar_audio = f"cluster_{cluster_id}_exemplar.wav"

        clusters_info[cluster_id] = {
            'cluster_id': int(cluster_id),
            'centroid_112d': centroid.tolist(),
            'exemplar_audio': exemplar_audio,
            'exemplar_features_112d': centroid.tolist(),  # Same as centroid for now
            'num_segments': int(mask.sum()),
            'mean_distance_to_centroid': float(distances.mean())
        }

    print(f'  Selected {len(clusters_info)} exemplars')

    # Step 5: Export synthesis manifest in Rust-compatible format
    print('=' * 70)
    print('Exporting Synthesis Manifest for Rust (ClustersManifest format)')
    print('=' * 70)

    # Export in format matching Rust ClustersManifest
    manifest = {
        'vocabulary_size': len(clusters_info),
        'num_clusters': len(clusters_info),
        'clusters': {str(k): v for k, v in clusters_info.items()},
        'metadata': {
            'extraction_method': 'pca_bgmm_teacher_student',
            'n_components': 30,
            'covariance_type': 'diag',
            'explained_variance': float(explained_variance),
            'pca_time_seconds': pca_time,
            'bgmm_time_seconds': bgmm_time,
            'total_time_seconds': total_time,
            'assignment_latency_ms': assignment_time
        }
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'  Exported to: {manifest_path}')
    print()
    print('╔═══════════════════════════════════════════════════════════════════════════╗')
    print('║     PCA+BGMM Pipeline Complete!                                            ║')
    print('╚═══════════════════════════════════════════════════════════════════════════╝')
    print()
    print('Summary:')
    print(f'  Vocabulary size: {len(clusters_info)} (BGMM pruned from 100)')
    print(f'  Real-time assignment: {assignment_time:.3f}ms per lookup')
    print(f'  Centroids in 112D space: Ready for Rust integration')
    print()
    print('Next steps:')
    print('  1. Rust: ExemplarManager.load_from_manifest()')
    print('  2. Rust: Implement find_nearest_centroid()')
    print(f'  3. Rust: Update vocabulary_size → {len(clusters_info)}')
    print('  4. Test: End-to-end feature event → cluster ID')


if __name__ == '__main__':
    main()
