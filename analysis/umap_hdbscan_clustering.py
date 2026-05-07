#!/usr/bin/env python3
"""
UMAP + HDBSCAN Clustering for Graded Phrase Mining

This is the standard bioacoustics stack:
1. UMAP reduces 112D → 10D while preserving topological structure
2. HDBSCAN clusters the low-D embedding with soft probabilities

Ideal for graded vocalizations because:
- UMAP preserves continuous "streams" of graded signals
- HDBSCAN assigns soft probabilities to transitional points
- Memory efficient: O(N) UMAP + fast HDBSCAN in low dimensions

Author: Sheel Morjaria
"""

import json
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import hdbscan
    import umap

    UMAP_HDBSCAN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: UMAP/HDBSCAN not available: {e}")
    UMAP_HDBSCAN_AVAILABLE = False
    exit(1)


def load_features(feature_path: str, max_samples: int = None) -> np.ndarray:
    """Load 112D features from extraction output."""
    print(f"Loading features from {feature_path}...")

    with open(feature_path, "r") as f:
        data = json.load(f)

    n_samples = min(max_samples, len(data["segments"])) if max_samples else len(data["segments"])
    print(f"  Loading {n_samples:,} of {len(data['segments']):,} total segments...")

    features_list = []
    for i, seg in enumerate(data["segments"][:n_samples]):
        features_list.append(seg["features_112d"])

    return np.array(features_list, dtype=np.float32)


def run_umap_hdbscan(
    features_112d: np.ndarray,
    n_components: int = 10,
    n_neighbors: int = 30,
    min_cluster_size: int = 50,
    min_samples: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Run UMAP + HDBSCAN clustering pipeline.

    Args:
        features_112d: Input 112D feature vectors
        n_components: UMAP output dimension (5-15 recommended)
        n_neighbors: UMAP neighbors (balances local/global structure)
        min_cluster_size: HDBSCAN minimum cluster size
        min_samples: HDBSCAN minimum samples for core points

    Returns:
        labels: Cluster assignments (-1 = noise)
        soft_labels: Membership probabilities (n_samples, n_clusters)
        metadata: Dictionary with timing and stats
    """
    n_samples = len(features_112d)
    metadata = {
        "n_samples": n_samples,
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_cluster_size": min_cluster_size,
    }

    print(f"\n{'=' * 70}")
    print("UMAP + HDBSCAN Pipeline")
    print(f"{'=' * 70}")
    print(f"Dataset: {n_samples:,} samples × 112D")
    print()

    # ========================================================================
    # Step 1: UMAP Dimensionality Reduction
    # ========================================================================

    print("Step 1: UMAP Dimensionality Reduction")
    print("-" * 70)
    print(f"  Reducing 112D → {n_components}D...")
    print("  Parameters:")
    print(f"    n_neighbors = {n_neighbors} (local vs global balance)")
    print("    min_dist = 0.0 (tight clusters for HDBSCAN)")
    print("    metric = cosine (better for high-dim audio)")

    tracemalloc.start()
    start_time = time.time()

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        # Safety parameters for WSL
        low_memory=True,
        n_jobs=1,  # Disable parallelism for WSL compatibility
    )

    try:
        embedding = reducer.fit_transform(features_112d)
    except Exception as e:
        print(f"  ✗ UMAP failed: {e}")
        print("  Tip: Try reducing n_samples or n_neighbors")
        raise

    umap_time = time.time() - start_time
    _, umap_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  ✅ UMAP complete in {umap_time:.1f}s")
    print(f"  RAM: {umap_ram / 10**6:.1f} MB")
    print(f"  Embedding shape: {embedding.shape}")
    print()

    # ========================================================================
    # Step 2: HDBSCAN Clustering
    # ========================================================================

    print("Step 2: HDBSCAN Clustering")
    print("-" * 70)
    print(f"  Clustering {n_components}D embedding...")
    print("  Parameters:")
    print(f"    min_cluster_size = {min_cluster_size}")
    print(f"    min_samples = {min_samples}")

    tracemalloc.start()
    start_time = time.time()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # Safe in low dimensions
        prediction_data=True,
        cluster_selection_method="eom",
    )

    try:
        clusterer.fit(embedding)
    except Exception as e:
        print(f"  ✗ HDBSCAN failed: {e}")
        raise

    labels = clusterer.labels_
    hdbscan_time = time.time() - start_time
    _, hdbscan_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Get soft clustering probabilities
    print("  Computing soft membership vectors...")
    soft_labels = hdbscan.all_points_membership_vectors(clusterer)

    print(f"  ✅ HDBSCAN complete in {hdbscan_time:.1f}s")
    print(f"  RAM: {hdbscan_ram / 10**6:.1f} MB")
    print()

    # ========================================================================
    # Summary Statistics
    # ========================================================================

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_count = list(labels).count(-1)
    noise_rate = noise_count / len(labels)

    print("Summary")
    print("-" * 70)
    print(f"  Total time: {umap_time + hdbscan_time:.1f}s")
    print(f"  Peak RAM: {max(umap_ram, hdbscan_ram) / 10**6:.1f} MB")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {noise_count:,} ({noise_rate * 100:.1f}%)")
    print(f"  Soft labels shape: {soft_labels.shape}")
    print()

    metadata.update(
        {
            "umap_time_seconds": umap_time,
            "hdbscan_time_seconds": hdbscan_time,
            "total_time_seconds": umap_time + hdbscan_time,
            "peak_ram_mb": max(umap_ram, hdbscan_ram) / 10**6,
            "n_clusters": n_clusters,
            "noise_count": noise_count,
            "noise_rate": noise_rate,
            "soft_labels_shape": list(soft_labels.shape),
        }
    )

    return labels, soft_labels, metadata


def main():
    """Run UMAP + HDBSCAN clustering on extracted bat features."""
    feature_path = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction_112d_no_cluster.json"

    if not Path(feature_path).exists():
        print(f"Error: Feature file not found: {feature_path}")
        return

    # Use smaller sample to avoid WSL memory issues
    MAX_SAMPLES = 30000

    features_112d = load_features(feature_path, max_samples=MAX_SAMPLES)

    # Run UMAP + HDBSCAN
    labels, soft_labels, metadata = run_umap_hdbscan(
        features_112d, n_components=10, n_neighbors=30, min_cluster_size=50, min_samples=10
    )

    # Export results
    output_path = Path("/mnt/c/Users/sheel/Desktop/src/analysis/results/umap_hdbscan_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": metadata,
        "labels": labels.tolist(),
        "soft_labels": soft_labels.tolist(),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results exported to: {output_path}")
    print()
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║     UMAP + HDBSCAN Complete!                                             ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
