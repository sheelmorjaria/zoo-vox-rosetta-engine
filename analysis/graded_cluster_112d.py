#!/usr/bin/env python3
"""
Graded Clustering for 112D Bat Features using UMAP + HDBSCAN

This implementation:
1. Uses UMAP to reduce 112D to a lower-dimensional manifold (preserving graded structure)
2. Applies HDBSCAN on the low-D embedding for efficient clustering
3. Extracts soft clustering probabilities for graded phrase boundaries

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
"""

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import UMAP and HDBSCAN
try:
    import hdbscan
    import umap

    print("✓ UMAP and HDBSCAN available")
except ImportError as e:
    print(f"✗ Missing required library: {e}")
    print("  Install with: pip install umap-learn hdbscan")
    raise

try:
    from sklearn.neighbors import NearestNeighbors

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available, some features limited")


@dataclass
class GradedClusterResult:
    """Result of graded clustering with soft probabilities."""

    segment_index: int
    file_name: str
    start_sample: int
    primary_cluster: int
    cluster_probabilities: Dict[int, float]  # Soft clustering
    is_graded: bool  # True if has significant secondary cluster
    umap_coordinates: Optional[List[float]] = None  # For visualization


@dataclass
class ClusterProfile:
    """Profile of a cluster with acoustic characteristics."""

    cluster_id: int
    n_segments: int
    mean_f0_hz: float
    mean_duration_ms: float
    probability_mass: float  # Total probability mass in this cluster
    boundary_segments: List[int]  # Segments with graded membership


class GradedClusteringPipeline:
    """Pipeline for graded phrase mining using UMAP + HDBSCAN."""

    def __init__(
        self,
        umap_n_components: int = 10,
        umap_n_neighbors: int = 30,
        umap_min_dist: float = 0.0,
        hdbscan_min_cluster_size: int = 50,
        hdbscan_min_samples: int = 10,
        graded_threshold: float = 0.3,  # Secondary prob > this = graded
    ):
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.graded_threshold = graded_threshold

        # Fitted models
        self.reducer = None
        self.clusterer = None
        self.umap_embedding = None

        # Results
        self.results: List[GradedClusterResult] = []
        self.cluster_profiles: Dict[int, ClusterProfile] = {}

    def load_features(
        self,
        path: str = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction_112d_no_cluster.json",
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Load 112D features from JSON."""
        print(f"Loading features from {path}...")

        start_time = time.time()

        with open(path, "r") as f:
            data = json.load(f)

        n_segments = data["total_segments"]
        n_dims = data["feature_dimension"]

        print(f"  Segments: {n_segments:,}")
        print(f"  Dimensions: {n_dims}")

        # Load features
        features_list = []
        metadata = []

        for seg in data["segments"]:
            features_list.append(seg["features_112d"])
            metadata.append(
                {
                    "file_name": seg["file_name"],
                    "start_sample": seg["start_sample"],
                    "segment_index": seg["segment_index"],
                }
            )

        features = np.array(features_list, dtype=np.float32)

        elapsed = time.time() - start_time
        print(f"  Loaded in {elapsed:.1f}s")
        print(f"  Memory: {features.nbytes / 1024**3:.2f} GB")

        return features, metadata

    def fit_umap(self, features: np.ndarray) -> np.ndarray:
        """Fit UMAP and reduce dimensionality."""
        print(f"\n{'=' * 70}")
        print("UMAP Dimensionality Reduction")
        print(f"{'=' * 70}")
        print(f"  {features.shape[1]}D → {self.umap_n_components}D")
        print(f"  n_neighbors: {self.umap_n_neighbors}")
        print(f"  min_dist: {self.umap_min_dist}")
        print("  metric: cosine")

        start_time = time.time()

        self.reducer = umap.UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric="cosine",  # Better for high-dim audio features
            random_state=42,
            n_jobs=-1,  # Use all cores
        )

        print("  Fitting UMAP...")
        self.umap_embedding = self.reducer.fit_transform(features)

        elapsed = time.time() - start_time
        print(f"  ✓ UMAP complete in {elapsed:.1f}s")
        print(f"  Embedding shape: {self.umap_embedding.shape}")
        print(f"  Memory: {self.umap_embedding.nbytes / 1024**2:.1f} MB")

        return self.umap_embedding

    def fit_hdbscan(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit HDBSCAN on UMAP embedding."""
        print(f"\n{'=' * 70}")
        print("HDBSCAN Clustering on UMAP Embedding")
        print(f"{'=' * 70}")
        print(f"  Input dimensions: {embedding.shape[1]}D")
        print(f"  min_cluster_size: {self.hdbscan_min_cluster_size}")
        print(f"  min_samples: {self.hdbscan_min_samples}")

        start_time = time.time()

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            metric="euclidean",  # Safe to use on low-D UMAP embedding
            cluster_selection_method="eom",
            prediction_data=True,  # Enable soft clustering
        )

        print("  Fitting HDBSCAN...")
        self.clusterer.fit(embedding)

        # Get hard labels
        labels = self.clusterer.labels_

        # Get soft clustering probabilities (membership vectors)
        soft_clusters = hdbscan.all_points_membership_vectors(self.clusterer)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        elapsed = time.time() - start_time
        print(f"  ✓ HDBSCAN complete in {elapsed:.1f}s")
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise} ({n_noise / len(labels) * 100:.1f}%)")

        return labels, soft_clusters

    def build_results(
        self,
        features: np.ndarray,
        metadata: List[Dict],
        labels: np.ndarray,
        soft_clusters: np.ndarray,
    ) -> List[GradedClusterResult]:
        """Build graded cluster results with soft probabilities."""
        print(f"\n{'=' * 70}")
        print("Building Graded Cluster Results")
        print(f"{'=' * 70}")

        self.results = []

        for i, (meta, label, soft_prob) in enumerate(zip(metadata, labels, soft_clusters)):
            # Get cluster probabilities for this point
            # HDBSCAN returns probabilities for each cluster (including noise)
            cluster_probs = {}

            # Filter to actual clusters (noise = -1)
            for cluster_id, prob in enumerate(soft_prob):
                if cluster_id < len(set(labels)) - (1 if -1 in labels else 0):
                    cluster_probs[int(cluster_id)] = float(prob)

            # Normalize probabilities
            total_prob = sum(cluster_probs.values())
            if total_prob > 0:
                cluster_probs = {k: v / total_prob for k, v in cluster_probs.items()}

            # Find primary cluster
            primary_cluster = int(label) if label >= 0 else -1

            # Check if graded (has significant secondary probability)
            sorted_probs = sorted(cluster_probs.items(), key=lambda x: x[1], reverse=True)
            is_graded = False

            if len(sorted_probs) > 1:
                secondary_prob = sorted_probs[1][1]
                is_graded = secondary_prob > self.graded_threshold

            result = GradedClusterResult(
                segment_index=i,
                file_name=meta["file_name"],
                start_sample=meta["start_sample"],
                primary_cluster=primary_cluster,
                cluster_probabilities=cluster_probs,
                is_graded=is_graded,
                umap_coordinates=self.umap_embedding[i].tolist()
                if self.umap_embedding is not None
                else None,
            )

            self.results.append(result)

        # Print graded statistics
        n_graded = sum(1 for r in self.results if r.is_graded)
        print(f"  Total segments: {len(self.results):,}")
        print(f"  Graded segments: {n_graded:,} ({n_graded / len(self.results) * 100:.1f}%)")

        return self.results

    def build_cluster_profiles(self, features: np.ndarray) -> Dict[int, ClusterProfile]:
        """Build profiles for each cluster."""
        print(f"\n{'=' * 70}")
        print("Building Cluster Profiles")
        print(f"{'=' * 70}")

        # Group segments by primary cluster
        cluster_segments = defaultdict(list)
        for i, result in enumerate(self.results):
            if result.primary_cluster >= 0:
                cluster_segments[result.primary_cluster].append((i, result))

        # Compute cluster statistics
        self.cluster_profiles = {}

        for cluster_id, segments in cluster_segments.items():
            indices = [idx for idx, _ in segments]
            cluster_features = features[indices]

            # Extract acoustic features (first 3 dimensions of 112D)
            # mean_f0_hz, duration_ms, f0_range_hz
            mean_f0 = float(np.mean(cluster_features[:, 0]))
            mean_duration = float(np.mean(cluster_features[:, 1]))

            # Compute probability mass
            prob_mass = sum(r.cluster_probabilities.get(cluster_id, 0.0) for _, r in segments)

            # Find boundary segments (graded)
            boundary_segs = [
                idx
                for idx, r in segments
                if r.is_graded
                and cluster_id in r.cluster_probabilities
                and r.cluster_probabilities[cluster_id] < 0.8
            ]

            self.cluster_profiles[cluster_id] = ClusterProfile(
                cluster_id=cluster_id,
                n_segments=len(segments),
                mean_f0_hz=mean_f0,
                mean_duration_ms=mean_duration,
                probability_mass=prob_mass,
                boundary_segments=boundary_segs,
            )

        print(f"  Built {len(self.cluster_profiles)} cluster profiles")

        # Print top clusters
        top_clusters = sorted(
            self.cluster_profiles.values(), key=lambda x: x.n_segments, reverse=True
        )[:10]

        print("\n  Top 10 clusters by size:")
        for i, profile in enumerate(top_clusters):
            graded_pct = len(profile.boundary_segments) / profile.n_segments * 100
            print(
                f"    {i + 1}. Cluster {profile.cluster_id}: "
                f"{profile.n_segments:,} segments, "
                f"F0={profile.mean_f0_hz:.0f}Hz, "
                f"{graded_pct:.1f}% graded"
            )

        return self.cluster_profiles

    def export_results(self, output_path: str) -> None:
        """Export graded clustering results."""
        print(f"\n{'=' * 70}")
        print("Exporting Results")
        print(f"{'=' * 70}")

        output_data = {
            "metadata": {
                "umap_n_components": self.umap_n_components,
                "umap_n_neighbors": self.umap_n_neighbors,
                "hdbscan_min_cluster_size": self.hdbscan_min_cluster_size,
                "hdbscan_min_samples": self.hdbscan_min_samples,
                "graded_threshold": self.graded_threshold,
                "n_clusters": len(self.cluster_profiles),
                "total_segments": len(self.results),
            },
            "cluster_profiles": {str(k): asdict(v) for k, v in self.cluster_profiles.items()},
            "segments": [],
        }

        # Export segments in chunks
        print(f"  Exporting {len(self.results):,} segments...")

        for i, result in enumerate(self.results):
            output_data["segments"].append(asdict(result))

            if (i + 1) % 10000 == 0:
                print(f"  {i + 1:,}/{len(self.results):,}...")

        # Write to file
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"  ✓ Exported to {output_path}")

        # Also export just UMAP embedding for visualization
        if self.umap_embedding is not None:
            viz_path = output_path.replace(".json", "_umap.json")
            with open(viz_path, "w") as f:
                json.dump(
                    {
                        "embedding": self.umap_embedding.tolist(),
                        "labels": [r.primary_cluster for r in self.results],
                    },
                    f,
                )
            print(f"  ✓ UMAP embedding exported to {viz_path}")


def main():
    """Main graded clustering pipeline."""
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║     Graded Clustering for 112D Bat Features                              ║")
    print("║     (UMAP + HDBSCAN with Soft Clustering)                               ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")

    pipeline = GradedClusteringPipeline(
        umap_n_components=10,  # Reduce 112D → 10D
        umap_n_neighbors=30,  # Balance local/global structure
        umap_min_dist=0.0,  # Tighter clusters for HDBSCAN
        hdbscan_min_cluster_size=50,
        hdbscan_min_samples=10,
        graded_threshold=0.3,  # Secondary prob > 30% = graded
    )

    # Load features
    features, metadata = pipeline.load_features()

    # Step 1: UMAP dimensionality reduction
    embedding = pipeline.fit_umap(features)

    # Step 2: HDBSCAN clustering on embedding
    labels, soft_clusters = pipeline.fit_hdbscan(embedding)

    # Step 3: Build graded results
    pipeline.build_results(features, metadata, labels, soft_clusters)

    # Step 4: Build cluster profiles
    pipeline.build_cluster_profiles(features)

    # Step 5: Export results
    output_path = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/graded_clustering_112d.json"
    pipeline.export_results(output_path)

    print(f"\n{'=' * 70}")
    print("Graded Clustering Complete!")
    print(f"{'=' * 70}")
    print(f"  Output: {output_path}")
    print("\nNext steps:")
    print("  1. Analyze graded boundaries between clusters")
    print("  2. Run PCFG analysis on primary cluster sequences")
    print("  3. Investigate soft-cluster transitions for graded syntax")


if __name__ == "__main__":
    main()
