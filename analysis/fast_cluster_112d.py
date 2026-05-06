#!/usr/bin/env python3
"""
Fast Clustering for 112D Bat Features

Uses approximate HDBSCAN and efficient data structures to cluster
millions of 112D feature vectors.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import time

# Try to import hdbscan
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not available, will use sklearn")

try:
    from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
    from sklearn.metrics import pairwise_distances_argmin_min
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available")


def load_features(
    path: str = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction_112d_no_cluster.json"
) -> Tuple[np.ndarray, List[Dict]]:
    """Load 112D features from JSON."""
    print(f"Loading features from {path}...")

    start_time = time.time()

    with open(path, 'r') as f:
        data = json.load(f)

    n_segments = data['total_segments']
    n_dims = data['feature_dimension']

    print(f"  Segments: {n_segments:,}")
    print(f"  Dimensions: {n_dims}")

    # Load features in chunks if too large
    features_list = []
    metadata = []

    for i, seg in enumerate(data['segments']):
        features_list.append(seg['features_112d'])
        metadata.append({
            'file_name': seg['file_name'],
            'start_sample': seg['start_sample'],
            'segment_index': seg['segment_index']
        })

        if (i + 1) % 100000 == 0:
            print(f"  Loaded {i+1:,}/{n_segments:,} segments...")

    features = np.array(features_list, dtype=np.float32)

    elapsed = time.time() - start_time
    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  Memory: {features.nbytes / 1024**3:.2f} GB")

    return features, metadata


def cluster_approx_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 10,
    sample_size: int = 100000
) -> np.ndarray:
    """Cluster using approximate HDBSCAN on a sample, then predict labels."""
    print(f"\n{'='*70}")
    print("Approximate HDBSCAN Clustering")
    print(f"{'='*70}")

    n_samples = len(features)

    # Sample for clustering
    if n_samples > sample_size:
        print(f"Sampling {sample_size:,} / {n_samples:,} for clustering...")
        indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_features = features[indices]
    else:
        print(f"Using all {n_samples:,} samples for clustering...")
        sample_features = features
        indices = np.arange(n_samples)

    start_time = time.time()

    # Run HDBSCAN on sample
    print(f"Running HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    clusterer.fit(sample_features)

    n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    n_noise = list(clusterer.labels_).count(-1)

    print(f"  Sample clusters: {n_clusters}")
    print(f"  Sample noise: {n_noise}")

    # Predict labels for all points
    if n_samples > sample_size:
        print(f"Predicting labels for remaining {n_samples - sample_size:,} points...")

        # Get soft clusters
        soft_clusters = hdbscan.all_points_membership_vectors(clusterer)

        # For points not in sample, assign to nearest cluster
        labels = np.full(n_samples, -1, dtype=np.int32)

        # Assign sample labels
        labels[indices] = clusterer.labels_

        # For remaining points, use approximate nearest neighbor
        # Use sklearn's NearestNeighbors for efficiency
        from sklearn.neighbors import NearestNeighbors

        print("  Building nearest neighbor index...")
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        nn.fit(sample_features)

        # Get remaining indices
        remaining_mask = np.ones(n_samples, dtype=bool)
        remaining_mask[indices] = False
        remaining_indices = np.where(remaining_mask)[0]

        # Predict in batches
        batch_size = 10000
        for i in range(0, len(remaining_indices), batch_size):
            batch_end = min(i + batch_size, len(remaining_indices))
            batch_indices = remaining_indices[i:batch_end]
            batch_features = features[batch_indices]

            # Find nearest in sample
            distances, nn_indices = nn.kneighbors(batch_features)
            labels[batch_indices] = clusterer.labels_[nn_indices.flatten()]

            if (i + batch_size) % 50000 == 0:
                print(f"  Predicted {i + batch_size:,}/{len(remaining_indices):,} points...")

    else:
        labels = clusterer.labels_

    elapsed = time.time() - start_time
    print(f"Clustering complete in {elapsed:.1f}s")

    # Final stats
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"  Final clusters: {n_clusters}")
    print(f"  Final noise: {n_noise} ({n_noise/n_samples*100:.1f}%)")

    return labels


def cluster_minibatch_kmeans(
    features: np.ndarray,
    n_clusters: int = 100,
    batch_size: int = 10000
) -> np.ndarray:
    """Cluster using MiniBatch K-Means for very large datasets."""
    print(f"\n{'='*70}")
    print("MiniBatch K-Means Clustering")
    print(f"{'='*70}")

    n_samples = len(features)
    print(f"Clustering {n_samples:,} points into {n_clusters} clusters...")
    print(f"Batch size: {batch_size:,}")

    start_time = time.time()

    clusterer = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=100,
        n_init=3,
        verbose=1,
        random_state=42
    )

    # Fit in batches manually for progress tracking
    print("\nFitting clusters...")
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch = features[i:batch_end]

        if i == 0:
            clusterer.partial_fit(batch)
        else:
            clusterer.partial_fit(batch)

        if (i + batch_size) % 100000 == 0:
            print(f"  Processed {i + batch_size:,}/{n_samples:,} samples")

    # Get final labels
    labels = clusterer.predict(features)

    elapsed = time.time() - start_time
    print(f"\nClustering complete in {elapsed:.1f}s")
    print(f"  Inertia: {clusterer.inertia_:.2e}")

    return labels


def export_clustered_results(
    features: np.ndarray,
    labels: np.ndarray,
    metadata: List[Dict],
    output_path: str
) -> None:
    """Export clustered results to JSON."""
    print(f"\nExporting results to {output_path}...")

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    output_data = {
        'total_segments': len(labels),
        'feature_dimension': features.shape[1],
        'cluster_count': n_clusters,
        'noise_count': int(list(labels).count(-1)),
        'segments': []
    }

    # Export in chunks
    chunk_size = 10000
    for i in range(0, len(labels), chunk_size):
        chunk_end = min(i + chunk_size, len(labels))

        for j in range(i, chunk_end):
            output_data['segments'].append({
                'file_name': metadata[j]['file_name'],
                'start_sample': metadata[j]['start_sample'],
                'segment_index': metadata[j]['segment_index'],
                'features_112d': features[j].tolist(),
                'cluster_id': int(labels[j])
            })

        print(f"  Exported {chunk_end:,}/{len(labels):,} segments...")

    with open(output_path, 'w') as f:
        json.dump(output_data, f)

    print(f"Results exported successfully!")


def main():
    """Main clustering pipeline."""
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║     Fast Clustering for 112D Bat Features                                 ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")

    # Load features
    features, metadata = load_features()

    # Choose clustering method based on data size
    n_samples = len(features)

    if n_samples > 1000000 and HDBSCAN_AVAILABLE:
        # Use approximate HDBSCAN for very large datasets
        labels = cluster_approx_hdbscan(
            features,
            min_cluster_size=500,
            min_samples=50,
            sample_size=100000
        )
    elif SKLEARN_AVAILABLE:
        # Use MiniBatch K-Means
        n_clusters = min(200, n_samples // 1000)
        labels = cluster_minibatch_kmeans(
            features,
            n_clusters=n_clusters,
            batch_size=10000
        )
    else:
        print("Error: No clustering library available")
        return

    # Export results
    output_path = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction_112d_clustered.json"
    export_clustered_results(features, labels, metadata, output_path)

    # Print summary
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"\n{'='*70}")
    print("Clustering Summary")
    print(f"{'='*70}")
    print(f"  Total segments: {n_samples:,}")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise:,} ({n_noise/n_samples*100:.1f}%)")
    print(f"\nOutput: {output_path}")
    print(f"\nNext: Run PCFG analysis with clustered results")


if __name__ == "__main__":
    main()
