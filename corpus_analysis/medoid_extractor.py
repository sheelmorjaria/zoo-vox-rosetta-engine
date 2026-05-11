#!/usr/bin/env python3
"""
Quality-Weighted Medoid Extraction

Replaces centroid averaging with medoid selection + SNR quality weighting.
Preserves rare calls (long-tail) instead of pruning them.

Key features:
- HDBSCAN for density-based clustering (no cluster count specification)
- Preserves rare calls as "noise" points instead of deleting them
- Quality-weighted medoid selection based on SNR
- Exemplar bank generation for synthesis manifest

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Literal, Callable

import numpy as np
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


@dataclass
class ExemplarMetadata:
    """Metadata for a single exemplar."""
    exemplar_id: str
    latent_coord: List[float]  # 16D VAE coordinate
    audio_path: str
    snr: float
    exemplar_type: Literal["dense_zone", "rare"]
    cluster_size: int = 0
    cluster_label: int = -1
    description: str = ""


@dataclass
class MedoidConfig:
    """Configuration for medoid extraction."""
    # HDBSCAN parameters
    min_cluster_size: int = 50
    min_samples: Optional[int] = None
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"  # Excess of Mass
    prediction_data: bool = True

    # Quality weighting
    snr_threshold: float = 20.0  # dB
    snr_percentile: float = 0.10  # Top 10% near medoid
    min_snr: float = 5.0  # Minimum acceptable SNR

    # Rare call handling
    preserve_all_rare: bool = True  # Keep ALL noise points
    max_rare_per_snr_tier: Optional[int] = None  # Limit rare calls if needed

    # Audio path resolution
    audio_path_template: str = "audio/segment_{idx:06d}.wav"


class MedoidExtractor:
    """
    Extracts quality-weighted medoids from VAE latent space.

    Process:
    1. Run HDBSCAN to find dense regions
    2. For each cluster, find the medoid (min total distance)
    3. Check SNR of medoid; find nearby high-SNR alternative if needed
    4. Preserve rare calls (noise points) as individual exemplars
    """

    def __init__(
        self,
        config: Optional[MedoidConfig] = None,
    ):
        if config is None:
            config = MedoidConfig()

        self.config = config
        self.clusterer = None
        self._init_clusterer()

    def _init_clusterer(self):
        """Initialize HDBSCAN clusterer."""
        try:
            import hdbscan
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples,
                metric=self.config.metric,
                cluster_selection_method=self.config.cluster_selection_method,
                prediction_data=self.config.prediction_data,
            )
        except ImportError:
            raise ImportError(
                "hdbscan is required. Install with: pip install hdbscan"
            )

    def extract_exemplars(
        self,
        latent_coords_16d: np.ndarray,
        original_audio_snrs: np.ndarray,
        audio_paths: Optional[List[str]] = None,
    ) -> Dict[str, ExemplarMetadata]:
        """
        Extract exemplars from VAE latent space.

        Args:
            latent_coords_16d: (N, 16) VAE latent coordinates
            original_audio_snrs: (N,) Signal-to-Noise Ratio for each audio file
            audio_paths: Optional list of audio file paths

        Returns:
            exemplars: Dict mapping exemplar IDs to metadata
        """
        logger.info(f"Extracting exemplars from {len(latent_coords_16d)} points")

        # Validate inputs
        if len(latent_coords_16d) != len(original_audio_snrs):
            raise ValueError(
                f"Mismatch: {len(latent_coords_16d)} coords vs {len(original_audio_snrs)} SNRs"
            )

        # Generate audio paths if not provided
        if audio_paths is None:
            audio_paths = [
                self.config.audio_path_template.format(idx=i)
                for i in range(len(latent_coords_16d))
            ]

        # Run HDBSCAN clustering
        logger.info("Running HDBSCAN clustering...")
        labels = self.clusterer.fit_predict(latent_coords_16d)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        logger.info(f"HDBSCAN found {n_clusters} clusters, {n_noise} noise points")

        # Extract exemplars
        exemplars = {}
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                # Handle rare calls (noise points)
                rare_exemplars = self._extract_rare_exemplars(
                    latent_coords_16d,
                    original_audio_snrs,
                    audio_paths,
                    labels,
                )
                exemplars.update(rare_exemplars)
            else:
                # Handle dense clusters
                cluster_exemplar = self._extract_cluster_medoid(
                    latent_coords_16d,
                    original_audio_snrs,
                    audio_paths,
                    labels,
                    label,
                )
                exemplars[cluster_exemplar.exemplar_id] = cluster_exemplar

        logger.info(f"Extracted {len(exemplars)} exemplars total")
        return exemplars

    def _extract_cluster_medoid(
        self,
        latent_coords: np.ndarray,
        snrs: np.ndarray,
        audio_paths: List[str],
        labels: np.ndarray,
        cluster_label: int,
    ) -> ExemplarMetadata:
        """
        Extract medoid exemplar for a single cluster.

        Args:
            latent_coords: All latent coordinates
            snrs: All SNR values
            audio_paths: All audio paths
            labels: All cluster labels
            cluster_label: The cluster to process

        Returns:
            ExemplarMetadata for the cluster medoid
        """
        cluster_mask = (labels == cluster_label)
        cluster_points = latent_coords[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        cluster_size = len(cluster_indices)

        # Calculate distance matrix within cluster
        dist_matrix = pairwise_distances(cluster_points, metric=self.config.metric)

        # Find medoid (point with minimum total distance)
        medoid_local_idx = np.argmin(dist_matrix.sum(axis=1))
        medoid_global_idx = cluster_indices[medoid_local_idx]

        # Quality-weighted selection
        best_idx = self._find_pristine_exemplar(
            cluster_points,
            cluster_indices,
            medoid_global_idx,
            snrs,
        )

        # Get SNR score
        final_snr = snrs[best_idx]

        # Check if we substituted the medoid
        is_substituted = (best_idx != medoid_global_idx)
        if is_substituted:
            logger.debug(
                f"Zone {cluster_label}: Substituted medoid (SNR={snrs[medoid_global_idx]:.1f}) "
                f"with higher-SNR point (SNR={final_snr:.1f})"
            )

        return ExemplarMetadata(
            exemplar_id=f"zone_{cluster_label}",
            latent_coord=latent_coords[best_idx].tolist(),
            audio_path=audio_paths[best_idx],
            snr=float(final_snr),
            exemplar_type="dense_zone",
            cluster_size=cluster_size,
            cluster_label=int(cluster_label),
            description=f"Dense zone medoid (n={cluster_size})",
        )

    def _extract_rare_exemplars(
        self,
        latent_coords: np.ndarray,
        snrs: np.ndarray,
        audio_paths: List[str],
        labels: np.ndarray,
    ) -> Dict[str, ExemplarMetadata]:
        """
        Extract rare calls (HDBSCAN noise points) as individual exemplars.

        This implements the "Long-Tail Rescue" - preserving rare calls
        instead of pruning them.

        Args:
            latent_coords: All latent coordinates
            snrs: All SNR values
            audio_paths: All audio paths
            labels: All cluster labels

        Returns:
            Dict of rare exemplar metadata
        """
        rare_mask = (labels == -1)
        rare_indices = np.where(rare_mask)[0]

        exemplars = {}

        for idx in rare_indices:
            # Skip if SNR is too low
            if snrs[idx] < self.config.min_snr:
                continue

            exemplar_id = f"rare_{idx}"

            # Optional: Group by SNR tiers to limit very large rare sets
            if self.config.max_rare_per_snr_tier is not None:
                # Implement SNR tier logic if needed
                pass

            exemplars[exemplar_id] = ExemplarMetadata(
                exemplar_id=exemplar_id,
                latent_coord=latent_coords[idx].tolist(),
                audio_path=audio_paths[idx],
                snr=float(snrs[idx]),
                exemplar_type="rare",
                cluster_size=1,
                cluster_label=-1,
                description="Rare call (long-tail preserved)",
            )

        logger.info(f"Preserved {len(exemplars)} rare calls (long-tail rescue)")
        return exemplars

    def _find_pristine_exemplar(
        self,
        cluster_points: np.ndarray,
        cluster_indices: np.ndarray,
        medoid_idx: int,
        snrs: np.ndarray,
    ) -> int:
        """
        Find the highest-SNR exemplar near the medoid.

        If the mathematical medoid has low SNR, we find a nearby point
        with better audio quality.

        Args:
            cluster_points: Points in the cluster
            cluster_indices: Global indices of cluster points
            medoid_idx: Global index of the medoid
            snrs: All SNR values

        Returns:
            Best exemplar index (global)
        """
        # If the medoid has high enough SNR, use it
        if snrs[medoid_idx] >= self.config.snr_threshold:
            return medoid_idx

        # Find medoid's position within cluster
        medoid_local_idx = np.where(cluster_indices == medoid_idx)[0][0]

        # Calculate distances to medoid
        medoid_point = cluster_points[medoid_local_idx:medoid_local_idx+1]
        dists_to_medoid = pairwise_distances(
            cluster_points,
            medoid_point,
            metric=self.config.metric,
        ).flatten()

        # Select points in top percentile closest to medoid
        percentile_threshold = np.percentile(
            dists_to_medoid,
            self.config.snr_percentile * 100
        )
        close_mask = dists_to_medoid <= percentile_threshold
        close_local_indices = np.where(close_mask)[0]

        # Find highest SNR among close points
        close_global_indices = cluster_indices[close_local_indices]
        close_snrs = snrs[close_global_indices]
        best_close_local_idx = np.argmax(close_snrs)
        best_idx = close_global_indices[best_close_local_idx]

        logger.debug(
            f"Medoid SNR {snrs[medoid_idx]:.1f} < threshold {self.config.snr_threshold} "
            f"→ substituted with {snrs[best_idx]:.1f}"
        )

        return best_idx

    def get_cluster_statistics(
        self,
        labels: np.ndarray,
    ) -> Dict[str, any]:
        """
        Compute statistics about the clustering.

        Args:
            labels: HDBSCAN cluster labels

        Returns:
            Dictionary with clustering statistics
        """
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        total_points = len(labels)

        # Cluster sizes
        cluster_sizes = []
        for label in unique_labels:
            if label != -1:
                cluster_sizes.append(list(labels).count(label))

        return {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_fraction": n_noise / total_points,
            "total_points": total_points,
            "cluster_sizes": {
                "min": min(cluster_sizes) if cluster_sizes else 0,
                "max": max(cluster_sizes) if cluster_sizes else 0,
                "mean": np.mean(cluster_sizes) if cluster_sizes else 0,
                "median": np.median(cluster_sizes) if cluster_sizes else 0,
            },
        }


def compute_snr(
    audio: np.ndarray,
    noise_floor: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Signal-to-Noise Ratio for audio.

    Args:
        audio: Audio samples
        noise_floor: Optional pre-computed noise floor

    Returns:
        SNR in dB
    """
    # Power of signal
    signal_power = np.mean(audio ** 2)

    if noise_floor is None:
        # Use quietest 10% as noise floor
        sorted_power = np.sort(audio ** 2)
        noise_power = np.mean(sorted_power[:len(sorted_power) // 10])
    else:
        noise_power = np.mean(noise_floor ** 2)

    if noise_power < 1e-10:
        return 100.0  # Very high SNR

    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear + 1e-10)

    return float(snr_db)


def batch_compute_snr(
    audio_files: List[str],
    max_workers: int = 4,
) -> np.ndarray:
    """
    Compute SNR for multiple audio files in parallel.

    Args:
        audio_files: List of audio file paths
        max_workers: Number of parallel workers

    Returns:
        Array of SNR values in dB
    """
    from concurrent.futures import ThreadPoolExecutor
    import torchaudio

    def compute_single(path: str) -> float:
        try:
            audio, sr = torchaudio.load(path)
            audio = audio.numpy().flatten()
            return compute_snr(audio)
        except Exception as e:
            logger.warning(f"Failed to compute SNR for {path}: {e}")
            return 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        snrs = list(executor.map(compute_single, audio_files))

    return np.array(snrs)


def get_audio_path(idx: int, template: str = None) -> str:
    """Get audio file path from index."""
    if template is None:
        template = "audio/segment_{idx:06d}.wav"
    return template.format(idx=idx)


# Preset configurations

CONSERVATIVE_CONFIG = MedoidConfig(
    min_cluster_size=100,
    snr_threshold=25.0,  # Higher quality threshold
    snr_percentile=0.05,  # Stricter proximity
    preserve_all_rare=True,
)

AGGRESSIVE_CONFIG = MedoidConfig(
    min_cluster_size=30,
    snr_threshold=15.0,  # Lower quality threshold
    snr_percentile=0.20,  # Broader search
    preserve_all_rare=True,
)


def create_medoid_extractor(
    config: Optional[MedoidConfig] = None,
) -> MedoidExtractor:
    """
    Factory function to create medoid extractor.

    Args:
        config: Medoid configuration

    Returns:
        Configured MedoidExtractor
    """
    return MedoidExtractor(config)


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic latent coordinates
    np.random.seed(42)
    n_dense = 5000
    n_rare = 100

    # Create 3 dense clusters
    cluster1 = np.random.randn(n_dense // 3, 16) + np.array([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster2 = np.random.randn(n_dense // 3, 16) + np.array([-2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster3 = np.random.randn(n_dense // 3, 16) + np.array([0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Create rare calls (scattered)
    rare = np.random.randn(n_rare, 16) * 3

    latent_coords = np.vstack([cluster1, cluster2, cluster3, rare])

    # Generate SNR values (rare calls have variable SNR)
    snrs = np.concatenate([
        np.random.uniform(20, 50, n_dense),  # Dense: good quality
        np.random.uniform(10, 40, n_rare),    # Rare: variable quality
    ])

    # Extract exemplars
    extractor = create_medoid_extractor()
    exemplars = extractor.extract_exemplars(latent_coords, snrs)

    # Print summary
    n_dense = sum(1 for e in exemplars.values() if e.exemplar_type == "dense_zone")
    n_rare = sum(1 for e in exemplars.values() if e.exemplar_type == "rare")

    print(f"\nExemplar Summary:")
    print(f"  Dense zones: {n_dense}")
    print(f"  Rare calls: {n_rare}")
    print(f"  Total: {len(exemplars)}")

    # Sample exemplars
    for exemplar_id, meta in list(exemplars.items())[:5]:
        print(f"  {exemplar_id}: SNR={meta.snr:.1f}dB, type={meta.exemplar_type}")


if __name__ == '__main__':
    main()
