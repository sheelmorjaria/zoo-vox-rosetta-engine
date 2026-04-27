#!/usr/bin/env python3
"""
VocabOptimizer - Direction 1: Adaptive Vocabulary Optimization

This module implements vocabulary size (k) optimization using Silhouette Validation
Score (SVS) maximization. Each species has unique acoustic characteristics that
require different vocabulary granularity.

Key Features:
- SVS computation for cluster quality assessment
- Automatic k optimization within specified range
- Species-specific vocabulary configuration
- Integration with ExemplarManager

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SpeciesVocabConfig:
    """Species-specific vocabulary configuration."""

    species: str
    optimal_k: int
    svs_score: float
    discovery_timestamp: int

    def __init__(self, species: str, optimal_k: int, svs_score: float):
        self.species = species
        self.optimal_k = optimal_k
        self.svs_score = svs_score
        self.discovery_timestamp = int(time.time())

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "species": self.species,
            "optimal_k": self.optimal_k,
            "svs_score": self.svs_score,
            "discovery_timestamp": self.discovery_timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SpeciesVocabConfig":
        """Create from dictionary."""
        config = cls(
            species=data["species"],
            optimal_k=data["optimal_k"],
            svs_score=data["svs_score"],
        )
        config.discovery_timestamp = data.get("discovery_timestamp", int(time.time()))
        return config

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved vocab config for {self.species} to {path}")

    @classmethod
    def load(cls, path: str) -> "SpeciesVocabConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __eq__(self, other) -> bool:
        """Check equality."""
        if not isinstance(other, SpeciesVocabConfig):
            return False
        return (
            self.species == other.species
            and self.optimal_k == other.optimal_k
            and self.svs_score == other.svs_score
        )


class VocabOptimizer:
    """
    Optimizes vocabulary size (k) using Silhouette Validation Score.

    The optimizer searches for the k that maximizes the mean silhouette score,
    which measures how well-separated the clusters are. Higher SVS indicates
    better-defined clusters.

    Usage:
        optimizer = VocabOptimizer(k_range=(100, 2000))
        optimal_k = optimizer.optimize_k(features, species="egyptian_fruit_bat")
    """

    def __init__(
        self,
        k_range: Tuple[int, int] = (100, 2000),
        step_size: int = 50,
        batch_size: int = 1000,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the VocabOptimizer.

        Args:
            k_range: (min_k, max_k) range to search for optimal k
            step_size: Step size for k search (tests every Nth value)
            batch_size: Batch size for MiniBatchKMeans
            random_state: Random seed for reproducibility
        """
        self.k_range = k_range
        self.step_size = step_size
        self.batch_size = batch_size
        self.random_state = random_state

    def compute_svs(self, features: np.ndarray, k: int) -> float:
        """
        Compute Silhouette Validation Score for given k.

        Args:
            features: Feature matrix (n_samples, n_dimensions)
            k: Number of clusters

        Returns:
            Mean silhouette score (0 = poor, 1 = excellent)
        """
        # Handle edge cases
        n_samples = features.shape[0]

        if k <= 1:
            return 0.0

        if k >= n_samples:
            return 0.0

        # Handle NaN values
        features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize features
        scaler = StandardScaler()
        try:
            features_normalized = scaler.fit_transform(features_clean)
        except Exception:
            # Fallback if normalization fails
            features_normalized = features_clean

        # Cluster with MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(self.batch_size, n_samples),
            random_state=self.random_state,
            max_iter=100,
            n_init=3,
        )

        try:
            labels = kmeans.fit_predict(features_normalized)
        except Exception:
            return 0.0

        # Handle cases where all points go to one cluster
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 0.0

        # Compute silhouette score
        try:
            score = silhouette_score(features_normalized, labels)
            return float(score)
        except Exception:
            return 0.0

    def optimize_k(self, features: np.ndarray, species: str) -> int:
        """
        Find optimal k that maximizes mean silhouette score.

        Args:
            features: Feature matrix (n_samples, n_dimensions)
            species: Species name for logging

        Returns:
            Optimal k value
        """
        optimal_k, _ = self.optimize_k_with_history(features, species)
        return optimal_k

    def optimize_k_with_history(
        self, features: np.ndarray, species: str
    ) -> Tuple[int, Dict[int, float]]:
        """
        Find optimal k and return SVS history.

        Args:
            features: Feature matrix (n_samples, n_dimensions)
            species: Species name for logging

        Returns:
            Tuple of (optimal_k, svs_history_dict)
        """
        min_k, max_k = self.k_range
        n_samples = features.shape[0]

        # Handle edge case: too few samples for clustering (need at least 2)
        if n_samples < 2:
            logger.warning(
                f"Insufficient samples (n={n_samples}) for clustering. "
                f"Returning k=1 as fallback."
            )
            return 1, {1: 0.0}

        # Handle edge case: fewer samples than min_k
        if n_samples < min_k:
            # Use n_samples - 1 as the only viable k (k must be < n_samples)
            viable_k = n_samples - 1
            svs = self.compute_svs(features, viable_k)
            logger.info(
                f"Small dataset (n={n_samples}): using k={viable_k}, SVS={svs:.4f}"
            )
            return viable_k, {viable_k: svs}

        # Adjust max_k if it exceeds number of samples
        # max_k is exclusive (standard Python range behavior)
        effective_max_k = min(max_k, n_samples)

        # Validate that at least one legal k exists
        # When n_samples == min_k, range(min_k, effective_max_k) is empty
        # In this case, use k = n_samples - 1 as the only viable option
        if effective_max_k <= min_k:
            viable_k = n_samples - 1
            svs = self.compute_svs(features, viable_k)
            logger.info(
                f"Boundary case (n_samples={n_samples} == min_k={min_k}): "
                f"using k={viable_k}, SVS={svs:.4f}"
            )
            return viable_k, {viable_k: svs}

        svs_history: Dict[int, float] = {}
        best_k = min_k
        best_svs = -1.0

        # Test k values in range with step_size
        # max_k is exclusive, so we test [min_k, max_k)
        k_values = range(min_k, effective_max_k, self.step_size)

        logger.info(
            f"Optimizing k for {species}: testing {len(list(k_values))} values "
            f"from {min_k} to {effective_max_k}"
        )

        for k in k_values:
            svs = self.compute_svs(features, k)
            svs_history[k] = svs

            if svs > best_svs:
                best_svs = svs
                best_k = k

            logger.debug(f"  k={k}: SVS={svs:.4f}")

        logger.info(
            f"Optimal k for {species}: k={best_k} (SVS={best_svs:.4f})"
        )

        return best_k, svs_history

    def optimize_and_create_config(
        self, features: np.ndarray, species: str
    ) -> SpeciesVocabConfig:
        """
        Optimize k and create a SpeciesVocabConfig.

        Args:
            features: Feature matrix (n_samples, n_dimensions)
            species: Species name

        Returns:
            SpeciesVocabConfig with optimal k and SVS score
        """
        optimal_k, svs_history = self.optimize_k_with_history(features, species)
        svs_score = svs_history[optimal_k]

        config = SpeciesVocabConfig(
            species=species, optimal_k=optimal_k, svs_score=svs_score
        )

        logger.info(
            f"Created config for {species}: k={optimal_k}, SVS={svs_score:.4f}"
        )

        return config


# =============================================================================
# Species Vocab Registry
# =============================================================================


class SpeciesVocabRegistry:
    """Registry for species-specific vocabulary configurations."""

    def __init__(self):
        """Initialize an empty registry."""
        self.configs: Dict[str, SpeciesVocabConfig] = {}

    def register(self, config: SpeciesVocabConfig) -> None:
        """
        Register a vocabulary configuration.

        Args:
            config: SpeciesVocabConfig to register
        """
        self.configs[config.species] = config
        logger.info(f"Registered vocab config for {config.species}: k={config.optimal_k}")

    def get(self, species: str) -> Optional[SpeciesVocabConfig]:
        """
        Get vocabulary configuration for a species.

        Args:
            species: Species name

        Returns:
            SpeciesVocabConfig if found, None otherwise
        """
        return self.configs.get(species)

    def get_optimal_k(self, species: str, default: int = 1020) -> int:
        """
        Get optimal k for a species.

        Args:
            species: Species name
            default: Default k if not found

        Returns:
            Optimal k value
        """
        config = self.get(species)
        return config.optimal_k if config else default

    def has_species(self, species: str) -> bool:
        """Check if registry has configuration for species."""
        return species in self.configs

    def list_species(self) -> list:
        """List all species in the registry."""
        return list(self.configs.keys())

    def save(self, path: str) -> None:
        """Save registry to JSON file."""
        data = {
            species: config.to_dict()
            for species, config in self.configs.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved registry with {len(self.configs)} species to {path}")

    @classmethod
    def load(cls, path: str) -> "SpeciesVocabRegistry":
        """Load registry from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        registry = cls()
        for species, config_data in data.items():
            config = SpeciesVocabConfig.from_dict(config_data)
            registry.register(config)

        logger.info(f"Loaded registry with {len(registry.configs)} species")
        return registry


# =============================================================================
# Convenience Functions
# =============================================================================


def optimize_vocab_for_species(
    features: np.ndarray,
    species: str,
    k_range: Tuple[int, int] = (100, 2000),
    registry: Optional[SpeciesVocabRegistry] = None,
) -> int:
    """
    Convenience function to optimize vocabulary for a species.

    Args:
        features: Feature matrix (n_samples, n_dimensions)
        species: Species name
        k_range: Range of k values to search
        registry: Optional registry to store result

    Returns:
        Optimal k value
    """
    optimizer = VocabOptimizer(k_range=k_range)
    config = optimizer.optimize_and_create_config(features, species)

    if registry:
        registry.register(config)

    return config.optimal_k


def main():
    """Demo CLI for VocabOptimizer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize vocabulary size using Silhouette Validation Score"
    )
    parser.add_argument(
        "--features", "-f", required=True, help="Path to features .npy file"
    )
    parser.add_argument(
        "--species", "-s", required=True, help="Species name"
    )
    parser.add_argument(
        "--k-min", type=int, default=100, help="Minimum k to test"
    )
    parser.add_argument(
        "--k-max", type=int, default=2000, help="Maximum k to test"
    )
    parser.add_argument(
        "--step", type=int, default=50, help="Step size for k search"
    )
    parser.add_argument(
        "--output", "-o", help="Save config to this path"
    )

    args = parser.parse_args()

    # Load features
    logger.info(f"Loading features from {args.features}")
    features = np.load(args.features)

    # Optimize
    optimizer = VocabOptimizer(
        k_range=(args.k_min, args.k_max),
        step_size=args.step,
        random_state=42,
    )

    config = optimizer.optimize_and_create_config(features, args.species)

    print(f"\nResults for {args.species}:")
    print(f"  Optimal k: {config.optimal_k}")
    print(f"  SVS Score: {config.svs_score:.4f}")
    print(f"  Timestamp: {config.discovery_timestamp}")

    # Save if requested
    if args.output:
        config.save(args.output)
        print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
