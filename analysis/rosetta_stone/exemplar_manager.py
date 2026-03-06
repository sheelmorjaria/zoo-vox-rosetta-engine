#!/usr/bin/env python3
"""
Exemplar Manager - Stage 3 of Synthesis Pipeline
=================================================

Selects the "best" audio segment for each Cluster ID from a corpus of
segmented audio features. The best exemplar is the segment whose 112D
feature vector is closest to the cluster centroid.

Usage:
    from analysis.rosetta_stone.exemplar_manager import ExemplarManager

    manager = ExemplarManager()
    manager.load_manifest("segments_manifest.json")
    manager.cluster_features(k=1020)  # vocabulary size
    manager.save_exemplars("clusters.json")

Architecture:
    Rust (NBD + Feature Extraction) --> segments_manifest.json
    Python (ExemplarManager) --> clusters.json
    Rust (Granular Synthesizer) <-- clusters.json

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SegmentInfo:
    """Information about a segmented audio file."""
    file_path: str
    features_112d: List[float]
    duration_ms: float
    mean_f0_hz: float
    cluster_id: Optional[int] = None


@dataclass
class ClusterInfo:
    """Information about a cluster and its best exemplar."""
    cluster_id: int
    centroid_112d: List[float]
    exemplar_audio: str  # Path to best audio file for this cluster
    exemplar_features_112d: List[float]
    num_segments: int
    mean_distance_to_centroid: float


class ExemplarManager:
    """
    Manages audio segment clustering and exemplar selection.

    The ExemplarManager takes segmented audio features (from Rust NBD + Feature Extraction),
    clusters them into a vocabulary of k symbols, and selects the best exemplar
    (closest to centroid) for each cluster.

    This enables the synthesis pipeline to reconstruct vocalizations by:
    1. Converting audio to cluster IDs (symbolic representation)
    2. Looking up the best exemplar audio for each cluster
    3. Using granular synthesis to reconstruct from exemplars
    """

    def __init__(self, vocabulary_size: int = 1020):
        """
        Initialize the ExemplarManager.

        Args:
            vocabulary_size: Number of clusters (k) for k-means. Default 1020.
        """
        self.vocabulary_size = vocabulary_size
        self.segments: List[SegmentInfo] = []
        self.clusters: Dict[int, ClusterInfo] = {}
        self.scaler: Optional[StandardScaler] = None
        self.kmeans: Optional[MiniBatchKMeans] = None

    def load_manifest(self, manifest_path: str) -> int:
        """
        Load segment features from a JSON manifest file.

        The manifest should be produced by the Rust feature extraction pipeline
        and contain entries like:
        {
            "segments": [
                {
                    "file_path": "seg_001.wav",
                    "features_112d": [...],
                    "duration_ms": 150.5,
                    "mean_f0_hz": 8500.0
                },
                ...
            ]
        }

        Args:
            manifest_path: Path to the JSON manifest file.

        Returns:
            Number of segments loaded.
        """
        with open(manifest_path, 'r') as f:
            data = json.load(f)

        self.segments = []
        for seg_data in data.get('segments', []):
            segment = SegmentInfo(
                file_path=seg_data['file_path'],
                features_112d=seg_data['features_112d'],
                duration_ms=seg_data.get('duration_ms', 0.0),
                mean_f0_hz=seg_data.get('mean_f0_hz', 0.0),
                cluster_id=None
            )
            self.segments.append(segment)

        logger.info(f"Loaded {len(self.segments)} segments from {manifest_path}")
        return len(self.segments)

    def add_segment(self, file_path: str, features_112d: List[float],
                    duration_ms: float = 0.0, mean_f0_hz: float = 0.0) -> None:
        """
        Add a single segment to the manager.

        Args:
            file_path: Path to the audio segment file.
            features_112d: 112D feature vector from RosettaFeatures.
            duration_ms: Duration of the segment in milliseconds.
            mean_f0_hz: Mean fundamental frequency in Hz.
        """
        segment = SegmentInfo(
            file_path=file_path,
            features_112d=features_112d,
            duration_ms=duration_ms,
            mean_f0_hz=mean_f0_hz,
            cluster_id=None
        )
        self.segments.append(segment)

    def cluster_features(self, k: Optional[int] = None, batch_size: int = 1000) -> None:
        """
        Cluster all segments using MiniBatchKMeans.

        Args:
            k: Number of clusters. If None, uses vocabulary_size from constructor.
            batch_size: Batch size for MiniBatchKMeans.
        """
        if not self.segments:
            raise ValueError("No segments loaded. Call load_manifest() first.")

        k = k or self.vocabulary_size

        # Extract feature matrix
        X = np.array([seg.features_112d for seg in self.segments], dtype=np.float32)

        # Handle any NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize features
        logger.info("Normalizing features...")
        self.scaler = StandardScaler()
        X_normalized = self.scaler.fit_transform(X)

        # Cluster using MiniBatchKMeans (efficient for large datasets)
        logger.info(f"Clustering {len(self.segments)} segments into {k} clusters...")
        self.kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=batch_size,
            random_state=42,
            max_iter=300,
            n_init=10
        )
        cluster_ids = self.kmeans.fit_predict(X_normalized)

        # Assign cluster IDs to segments
        for segment, cluster_id in zip(self.segments, cluster_ids):
            segment.cluster_id = int(cluster_id)

        logger.info(f"Clustering complete. Found {len(set(cluster_ids))} unique clusters.")

    def select_exemplars(self) -> Dict[int, ClusterInfo]:
        """
        Select the best exemplar (closest to centroid) for each cluster.

        For each cluster, we find the segment whose features are closest
        to the cluster centroid. This segment becomes the "exemplar" that
        will be used for granular synthesis.

        Returns:
            Dictionary mapping cluster_id to ClusterInfo.
        """
        if self.kmeans is None:
            raise ValueError("Clustering not performed. Call cluster_features() first.")

        self.clusters = {}

        # Get centroids in original feature space
        centroids_normalized = self.kmeans.cluster_centers_
        centroids_original = self.scaler.inverse_transform(centroids_normalized)

        # Group segments by cluster
        cluster_segments: Dict[int, List[SegmentInfo]] = {}
        for segment in self.segments:
            if segment.cluster_id is not None:
                if segment.cluster_id not in cluster_segments:
                    cluster_segments[segment.cluster_id] = []
                cluster_segments[segment.cluster_id].append(segment)

        # Select best exemplar for each cluster
        for cluster_id, segments in cluster_segments.items():
            centroid = centroids_original[cluster_id]

            # Find segment closest to centroid
            best_segment = None
            best_distance = float('inf')
            total_distance = 0.0

            for segment in segments:
                seg_features = np.array(segment.features_112d)
                distance = np.linalg.norm(seg_features - centroid)
                total_distance += distance

                if distance < best_distance:
                    best_distance = distance
                    best_segment = segment

            if best_segment is not None:
                cluster_info = ClusterInfo(
                    cluster_id=cluster_id,
                    centroid_112d=centroid.tolist(),
                    exemplar_audio=best_segment.file_path,
                    exemplar_features_112d=best_segment.features_112d,
                    num_segments=len(segments),
                    mean_distance_to_centroid=total_distance / len(segments)
                )
                self.clusters[cluster_id] = cluster_info

        logger.info(f"Selected exemplars for {len(self.clusters)} clusters")
        return self.clusters

    def get_exemplar_for_cluster(self, cluster_id: int) -> Optional[ClusterInfo]:
        """
        Get the exemplar information for a specific cluster.

        Args:
            cluster_id: The cluster ID to look up.

        Returns:
            ClusterInfo if found, None otherwise.
        """
        return self.clusters.get(cluster_id)

    def get_exemplar_audio_path(self, cluster_id: int) -> Optional[str]:
        """
        Get the audio file path for a cluster's exemplar.

        Args:
            cluster_id: The cluster ID to look up.

        Returns:
            Path to the exemplar audio file, or None if not found.
        """
        cluster_info = self.clusters.get(cluster_id)
        return cluster_info.exemplar_audio if cluster_info else None

    def save_exemplars(self, output_path: str) -> None:
        """
        Save cluster and exemplar information to JSON.

        Output format:
        {
            "vocabulary_size": 1020,
            "clusters": {
                "0": {
                    "cluster_id": 0,
                    "centroid_112d": [...],
                    "exemplar_audio": "seg_042.wav",
                    "exemplar_features_112d": [...],
                    "num_segments": 15,
                    "mean_distance_to_centroid": 0.85
                },
                ...
            }
        }

        Args:
            output_path: Path to save the JSON file.
        """
        output_data = {
            "vocabulary_size": self.vocabulary_size,
            "num_clusters": len(self.clusters),
            "clusters": {
                str(cid): asdict(info)
                for cid, info in self.clusters.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved {len(self.clusters)} cluster exemplars to {output_path}")

    def load_exemplars(self, input_path: str) -> None:
        """
        Load pre-computed cluster and exemplar information from JSON.

        Args:
            input_path: Path to the JSON file.
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        self.vocabulary_size = data.get('vocabulary_size', 1020)
        self.clusters = {}

        for cid_str, cluster_data in data.get('clusters', {}).items():
            cluster_info = ClusterInfo(
                cluster_id=cluster_data['cluster_id'],
                centroid_112d=cluster_data['centroid_112d'],
                exemplar_audio=cluster_data['exemplar_audio'],
                exemplar_features_112d=cluster_data['exemplar_features_112d'],
                num_segments=cluster_data['num_segments'],
                mean_distance_to_centroid=cluster_data['mean_distance_to_centroid']
            )
            self.clusters[cluster_info.cluster_id] = cluster_info

        logger.info(f"Loaded {len(self.clusters)} cluster exemplars from {input_path}")

    def create_synthesis_manifest(self, output_path: str) -> None:
        """
        Create a synthesis-ready manifest for the Rust granular synthesizer.

        Output format (optimized for Rust consumption):
        {
            "exemplars": [
                {
                    "cluster_id": 0,
                    "audio_path": "seg_042.wav",
                    "metadata": {
                        "mean_f0_hz": 8500.0,
                        "duration_ms": 150.5,
                        ...
                    }
                },
                ...
            ]
        }

        Args:
            output_path: Path to save the synthesis manifest.
        """
        exemplars = []
        for cluster_id in sorted(self.clusters.keys()):
            cluster_info = self.clusters[cluster_id]

            # Extract key metadata from features
            features = cluster_info.exemplar_features_112d
            metadata = {
                "mean_f0_hz": features[0] if len(features) > 0 else 5000.0,
                "duration_ms": features[1] if len(features) > 1 else 100.0,
                "f0_range_hz": features[2] if len(features) > 2 else 500.0,
                "rms_energy": features[3] if len(features) > 3 else 0.5,
                "harmonic_to_noise_ratio": features[6] if len(features) > 6 else 15.0,
                "attack_time_ms": features[9] if len(features) > 9 else 10.0,
                "decay_time_ms": features[10] if len(features) > 10 else 50.0,
            }

            exemplars.append({
                "cluster_id": cluster_id,
                "audio_path": cluster_info.exemplar_audio,
                "metadata": metadata
            })

        output_data = {
            "vocabulary_size": len(exemplars),
            "exemplars": exemplars
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Created synthesis manifest with {len(exemplars)} exemplars at {output_path}")


def main():
    """Command-line interface for ExemplarManager."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Exemplar Manager for Synthesis Pipeline Stage 3"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON manifest from Rust feature extraction"
    )
    parser.add_argument(
        "--output", "-o",
        default="clusters.json",
        help="Output JSON file for cluster exemplars"
    )
    parser.add_argument(
        "--synthesis-manifest", "-s",
        default="synthesis_manifest.json",
        help="Output synthesis-ready manifest for Rust"
    )
    parser.add_argument(
        "--k", "-k",
        type=int,
        default=1020,
        help="Vocabulary size (number of clusters)"
    )

    args = parser.parse_args()

    # Create manager and process
    manager = ExemplarManager(vocabulary_size=args.k)

    # Load segments
    manager.load_manifest(args.input)

    # Cluster features
    manager.cluster_features()

    # Select best exemplars
    manager.select_exemplars()

    # Save results
    manager.save_exemplars(args.output)
    manager.create_synthesis_manifest(args.synthesis_manifest)

    print(f"\nPipeline complete!")
    print(f"  Clusters: {args.output}")
    print(f"  Synthesis manifest: {args.synthesis_manifest}")


if __name__ == "__main__":
    main()
