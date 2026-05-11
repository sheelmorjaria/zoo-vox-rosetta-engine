#!/usr/bin/env python3
"""
Mahalanobis OOD Detection: Statistical Outlier Detection

Replaces L2 distance thresholding with Mahalanobis distance, which accounts
for the covariance structure of the latent space. This is critical for
high-dimensional spaces where L2 distance suffers from the curse of
dimensionality.

Key improvements:
- Accounts for variance differences across dimensions
- Uses Chi-squared distribution for statistically sound thresholds
- Per-token cluster modeling (each VQ-VAE token has its own distribution)

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Literal

import numpy as np
from scipy.stats import chi2

logger = logging.getLogger(__name__)


@dataclass
class OODStatistics:
    """Per-token cluster statistics for Mahalanobis distance."""
    token_id: int
    count: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(16))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(16))
    inv_covariance: np.ndarray = field(default_factory=lambda: np.eye(16))
    is_fitted: bool = False


@dataclass
class OODCalibrationConfig:
    """Configuration for OOD calibration."""
    num_tokens: int = 64
    latent_dim: int = 16
    regularization: float = 1e-6  # Prevent singular covariance matrices
    min_samples: int = 17  # Minimum samples to fit covariance (dim + 1)
    chi2_confidence: float = 0.99  # 99% confidence interval


class OODCalibrator:
    """
    Offline tool to calculate multivariate Gaussian parameters
    for Mahalanobis distance calculation.

    Process:
    1. Collect affect vectors grouped by their VQ-VAE token
    2. Calculate mean and covariance for each token cluster
    3. Compute inverse covariance with regularization
    4. Export to JSON for runtime use
    """

    def __init__(
        self,
        config: Optional[OODCalibrationConfig] = None,
    ):
        if config is None:
            config = OODCalibrationConfig()

        self.config = config
        self.statistics: Dict[int, OODStatistics] = {
            t: OODStatistics(token_id=t) for t in range(config.num_tokens)
        }

        # Chi-squared threshold for OOD detection
        self.chi2_threshold = chi2.ppf(
            config.chi2_confidence,
            df=config.latent_dim
        )

    def fit(
        self,
        token_ids: np.ndarray,
        affect_vectors: np.ndarray,
    ) -> None:
        """
        Calculate mean and inverse covariance for each token cluster.

        Args:
            token_ids: (N,) array of VQ-VAE token IDs
            affect_vectors: (N, latent_dim) array of 16D affect vectors
        """
        if len(token_ids) != len(affect_vectors):
            raise ValueError(
                f"Token IDs ({len(token_ids)}) and vectors ({len(affect_vectors)}) "
                "must have same length"
            )

        if affect_vectors.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"Expected {self.config.latent_dim}D vectors, got {affect_vectors.shape[1]}D"
            )

        logger.info(f"Fitting OOD statistics on {len(token_ids)} samples...")

        for token_id in range(self.config.num_tokens):
            mask = (token_ids == token_id)
            vectors = affect_vectors[mask]

            if len(vectors) < self.config.min_samples:
                logger.warning(
                    f"Token {token_id}: insufficient samples ({len(vectors)}), "
                    f"skipping covariance fit"
                )
                # Use default identity covariance
                self.statistics[token_id].count = len(vectors)
                self.statistics[token_id].mean = np.zeros(self.config.latent_dim)
                self.statistics[token_id].covariance = np.eye(self.config.latent_dim)
                self.statistics[token_id].inv_covariance = np.eye(self.config.latent_dim)
                self.statistics[token_id].is_fitted = False
                continue

            # Calculate mean
            mean = np.mean(vectors, axis=0)

            # Calculate covariance with regularization
            cov = np.cov(vectors, rowvar=False)
            cov += np.eye(self.config.latent_dim) * self.config.regularization

            # Invert covariance for Mahalanobis distance
            try:
                inv_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                logger.warning(f"Token {token_id}: singular covariance, using pseudo-inverse")
                inv_cov = np.linalg.pinv(cov)

            self.statistics[token_id].count = len(vectors)
            self.statistics[token_id].mean = mean
            self.statistics[token_id].covariance = cov
            self.statistics[token_id].inv_covariance = inv_cov
            self.statistics[token_id].is_fitted = True

        n_fitted = sum(1 for s in self.statistics.values() if s.is_fitted)
        logger.info(f"Fitted {n_fitted}/{self.config.num_tokens} token clusters")

    def get_token_statistics(self, token_id: int) -> OODStatistics:
        """Get statistics for a specific token."""
        return self.statistics.get(token_id, OODStatistics(token_id=token_id))

    def export_to_json(
        self,
        output_path: str = "models/ood_statistics.json",
    ) -> Path:
        """
        Export statistics to JSON for runtime use.

        Args:
            output_path: Path to output JSON file

        Returns:
            Path to exported file
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = {
            "num_tokens": self.config.num_tokens,
            "latent_dim": self.config.latent_dim,
            "chi2_threshold": float(self.chi2_threshold),
            "chi2_confidence": self.config.chi2_confidence,
            "statistics": {}
        }

        for token_id, stats in self.statistics.items():
            data["statistics"][str(token_id)] = {
                "token_id": int(stats.token_id),
                "count": int(stats.count),
                "mean": stats.mean.tolist(),
                "covariance": stats.covariance.tolist(),
                "inv_covariance": stats.inv_covariance.tolist(),
                "is_fitted": bool(stats.is_fitted),
            }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported OOD statistics to {output_path}")
        return output_path

    @classmethod
    def load_from_json(
        cls,
        path: str,
    ) -> "OODCalibrator":
        """
        Load OOD statistics from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Configured OODCalibrator
        """
        import json

        with open(path) as f:
            data = json.load(f)

        config = OODCalibrationConfig(
            num_tokens=data["num_tokens"],
            latent_dim=data["latent_dim"],
            chi2_confidence=data["chi2_confidence"],
        )

        calibrator = cls(config)
        calibrator.chi2_threshold = data["chi2_threshold"]

        for token_id_str, stats_data in data["statistics"].items():
            token_id = int(token_id_str)
            stats = OODStatistics(token_id=token_id)
            stats.count = stats_data["count"]
            stats.mean = np.array(stats_data["mean"])
            stats.covariance = np.array(stats_data["covariance"])
            stats.inv_covariance = np.array(stats_data["inv_covariance"])
            stats.is_fitted = stats_data["is_fitted"]
            calibrator.statistics[token_id] = stats

        logger.info(f"Loaded OOD statistics from {path}")
        return calibrator


class MahalanobisOOD:
    """
    Real-time OOD detector using Mahalanobis distance.

    Replaces L2 distance thresholding with statistically sound
    outlier detection that respects the covariance structure.

    D^2 = (x - μ)^T * Σ^(-1) * (x - μ)

    Uses Chi-squared distribution to determine OOD threshold.
    """

    def __init__(
        self,
        statistics: Dict[int, OODStatistics],
        chi2_threshold: float,
    ):
        """
        Initialize with pre-fitted statistics.

        Args:
            statistics: Per-token cluster statistics
            chi2_threshold: Chi-squared threshold for OOD detection
        """
        self.statistics = statistics
        self.chi2_threshold = chi2_threshold
        self.latent_dim = next(iter(statistics.values())).mean.shape[0]

    def is_ood(
        self,
        affect_vector: np.ndarray,
        token_id: int,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Calculate Mahalanobis distance and compare to chi-squared threshold.

        Args:
            affect_vector: (latent_dim,) affect vector from VAE encoder
            token_id: Associated VQ-VAE token ID

        Returns:
            (is_ood, md_squared, reason)
            - is_ood: True if vector is out-of-distribution
            - md_squared: Mahalanobis distance squared
            - reason: Human-readable explanation
        """
        affect_vector = np.asarray(affect_vector)

        if affect_vector.shape != (self.latent_dim,):
            return (
                True,
                float('inf'),
                f"Invalid vector shape: {affect_vector.shape}, expected ({self.latent_dim},)"
            )

        if token_id not in self.statistics:
            return (
                True,
                float('inf'),
                f"Unknown token ID: {token_id}"
            )

        stats = self.statistics[token_id]

        if not stats.is_fitted:
            return (
                True,
                float('inf'),
                f"Token {token_id} has insufficient statistics for OOD detection"
            )

        # Calculate Mahalanobis distance
        # D^2 = (x - μ)^T * Σ^(-1) * (x - μ)
        delta = affect_vector - stats.mean

        # More numerically stable: solve Σ * x = delta, then dot with delta
        try:
            # Use solve instead of explicit inverse for better numerical stability
            solved = np.linalg.solve(stats.covariance, delta)
            md_squared = float(np.dot(delta, solved))
        except np.linalg.LinAlgError:
            # Fallback to explicit inverse
            md_squared = float(np.dot(np.dot(delta.T, stats.inv_covariance), delta))

        # Compare to chi-squared threshold
        is_ood = md_squared > self.chi2_threshold

        # Generate reason
        if is_ood:
            confidence = 1 - chi2.cdf(md_squared, df=self.latent_dim)
            reason = (
                f"Mahalanobis D²={md_squared:.2f} > threshold={self.chi2_threshold:.2f} "
                f"(p={confidence:.4f})"
            )
        else:
            confidence = chi2.cdf(md_squared, df=self.latent_dim)
            reason = (
                f"Mahalanobis D²={md_squared:.2f} < threshold={self.chi2_threshold:.2f} "
                f"(p={confidence:.4f})"
            )

        return is_ood, md_squared, reason

    def compute_confidence(
        self,
        affect_vector: np.ndarray,
        token_id: int,
    ) -> float:
        """
        Compute confidence score (0-1) based on Mahalanobis distance.

        Returns 1 for in-distribution, 0 for extreme outliers.

        Args:
            affect_vector: Affect vector
            token_id: Associated token ID

        Returns:
            Confidence score between 0 and 1
        """
        is_ood, md_squared, _ = self.is_ood(affect_vector, token_id)

        # Convert Mahalanobis distance to confidence using CDF
        # Lower D² = higher confidence
        confidence = 1 - chi2.cdf(md_squared, df=self.latent_dim)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    @classmethod
    def from_calibrator(
        cls,
        calibrator: OODCalibrator,
    ) -> "MahalanobisOOD":
        """Create MahalanobisOOD from fitted OODCalibrator."""
        return cls(
            statistics=calibrator.statistics,
            chi2_threshold=calibrator.chi2_threshold,
        )

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "MahalanobisOOD":
        """Load MahalanobisOOD from JSON statistics file."""
        calibrator = OODCalibrator.load_from_json(path)
        return cls.from_calibrator(calibrator)


# Preset configurations

STANDARD_OOD_CONFIG = OODCalibrationConfig(
    num_tokens=64,
    latent_dim=16,
    regularization=1e-6,
    min_samples=17,
    chi2_confidence=0.99,  # 99% confidence
)

STRICT_OOD_CONFIG = OODCalibrationConfig(
    num_tokens=64,
    latent_dim=16,
    regularization=1e-6,
    min_samples=17,
    chi2_confidence=0.999,  # 99.9% confidence (more strict)
)


def create_ood_calibrator(
    config: Optional[OODCalibrationConfig] = None,
) -> OODCalibrator:
    """
    Factory function to create OOD calibrator.

    Args:
        config: OOD configuration (uses STANDARD_OOD_CONFIG if None)

    Returns:
        Configured OODCalibrator
    """
    return OODCalibrator(config)


def create_mahalanobis_ood(
    statistics_path: str,
) -> MahalanobisOOD:
    """
    Factory function to create MahalanobisOOD from file.

    Args:
        statistics_path: Path to OOD statistics JSON file

    Returns:
        Configured MahalanobisOOD
    """
    return MahalanobisOOD.load(statistics_path)


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    n_tokens = 64
    latent_dim = 16

    # Generate token IDs
    token_ids = np.random.randint(0, n_tokens, n_samples)

    # Generate affect vectors with cluster structure
    affect_vectors = np.zeros((n_samples, latent_dim))
    for t in range(n_tokens):
        mask = (token_ids == t)
        n_cluster = mask.sum()
        if n_cluster > 0:
            # Each token has its own cluster center
            center = np.random.randn(latent_dim) * 2
            # Add covariance (different variances per dimension)
            cluster = np.random.randn(n_cluster, latent_dim) * np.linspace(0.1, 2.0, latent_dim)
            affect_vectors[mask] = center + cluster

    # Calibrate OOD detector
    calibrator = create_ood_calibrator()
    calibrator.fit(token_ids, affect_vectors)

    # Export statistics
    calibrator.export_to_json("models/ood_statistics_demo.json")

    # Create OOD detector
    ood_detector = MahalanobisOOD.from_calibrator(calibrator)

    # Test with in-distribution sample
    in_dist_sample = affect_vectors[0]
    token_id = token_ids[0]
    is_ood, md, reason = ood_detector.is_ood(in_dist_sample, token_id)
    print(f"In-distribution: is_ood={is_ood}, D²={md:.2f}, {reason}")

    # Test with out-of-distribution sample
    ood_sample = np.random.randn(latent_dim) * 10
    is_ood, md, reason = ood_detector.is_ood(ood_sample, token_id)
    print(f"Out-of-distribution: is_ood={is_ood}, D²={md:.2f}, {reason}")


if __name__ == '__main__':
    main()
