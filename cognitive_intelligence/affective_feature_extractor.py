#!/usr/bin/env python3
"""
Affective Feature Extractor (Stream 1) - Sprint 1-2

Extracts ~30D continuous affective features from 112D Rosetta vector.
These features capture graded continuum of internal state, arousal, and affect.

Feature Selection Rationale:
- Layer 1 (0-45): F0, HNR, Jitter, Shimmer, Vibrato → continuous affective cues
- Layer 2 (46-75): GLCM texture → continuous prosodic variation
- Layer 3 (76-111): All micro texture → continuous temporal dynamics

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AffectiveFeatureConfig:
    """Configuration for affective feature extraction."""

    # Total input dimension (112D Rosetta vector)
    input_dim: int = 112

    # Output dimension (affective features)
    output_dim: int = 30

    # Feature indices (see FEATURE_INDICES below)
    feature_indices: List[int] = None

    # Whether to normalize features
    normalize: bool = True


class AffectiveFeatureExtractor:
    """
    Extract continuous affective features from 112D Rosetta vector.

    Stream 1 (Ventral/"How") features:
    - Continuous physics: F0, HNR, Jitter, Shimmer, Vibrato
    - GLCM texture: prosodic variation
    - Micro texture: temporal dynamics

    These features are ideal for β-VAE encoding to 16D latent space.
    """

    # Feature indices selected from 112D Rosetta vector
    # Based on acoustic algebra hierarchy:
    # Layer 1 (0-45): Base Physics
    # Layer 2 (46-75): Macro Texture
    # Layer 3 (76-111): Micro Texture

    FEATURE_INDICES = [
        # Layer 1: Base Physics - Continuous affective cues
        0,    # F0 (fundamental frequency) - primary pitch
        7,    # HNR (harmonic-to-noise ratio) - breathiness/tension
        35,   # Jitter (frequency perturbation) - instability
        36,   # Shimmer (amplitude perturbation) - roughness
        37,   # APQ (amplitude perturbation quotient) - tremor
        38,   # PPQ (period perturbation quotient) - rhythmic instability
        39,   # RAP (relative average perturbation) - micro-instability

        # Layer 1: MFCCs (spectral envelope) - timbre/affect
        8, 9, 10, 11, 12, 13,  # MFCC 1-6 (spectral shape, brightness)

        # Layer 1: ADSR envelope dynamics
        40,   # Attack time - onset sharpness
        41,   # Decay time - initial fade
        42,   # Sustain level - stability
        43,   # Release time - offset fade

        # Layer 2: GLCM texture (continuous prosodic variation)
        59,   # GLCM Contrast - texture variation
        60,   # GLCM Dissimilarity - local variation
        61,   # GLCM Homogeneity - smoothness
        62,   # GLCM Energy - intensity
        63,   # GLCM Correlation - periodicity
        64,   # GLCM ASM - uniformity

        # Layer 2: Spectral flux (rate of spectral change)
        65,   # Spectral Flux - dynamics
        66,   # Spectral Rolloff - brightness
        67,   # Spectral Centroid - brightness center
        68,   # Spectral Bandwidth - spectral spread

        # Layer 2: Harmonic features
        50,   # Harmonic ratio - harmonic content
        51,   # Inharmonicity - dissonance

        # Layer 2: Pitch geometry (continuous pitch variation)
        52,   # F0 range - pitch variability
        53,   # F0 std - pitch instability
        54,   # F0 CV - coefficient of variation

        # Layer 3: Micro texture (temporal dynamics)
        # All 36 micro features for continuous temporal modeling
        76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
        88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    ]

    def __init__(self, config: Optional[AffectiveFeatureConfig] = None):
        self.config = config or AffectiveFeatureConfig()

        if self.config.feature_indices is None:
            self.config.feature_indices = self.FEATURE_INDICES

        self.output_dim = len(self.config.feature_indices)

        # Normalization statistics (computed from training data)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

        logger.info(
            f"AffectiveFeatureExtractor initialized: "
            f"{self.config.input_dim}D → {self.output_dim}D"
        )

    def extract(self, features_112d: np.ndarray) -> np.ndarray:
        """
        Extract affective features from 112D Rosetta vector.

        Args:
            features_112d: Input vector of shape (112,) or (batch, 112)

        Returns:
            Affective features of shape (output_dim,) or (batch, output_dim)
        """
        if features_112d.ndim == 1:
            features_112d = features_112d.reshape(1, -1)

        batch_size = features_112d.shape[0]
        extracted = np.zeros((batch_size, self.output_dim), dtype=np.float32)

        for i, idx in enumerate(self.config.feature_indices):
            if idx < self.config.input_dim:
                extracted[:, i] = features_112d[:, idx]
            else:
                logger.warning(f"Feature index {idx} out of bounds")
                extracted[:, i] = 0.0

        # Normalize if statistics available
        if self.config.normalize and self.mean is not None and self.std is not None:
            extracted = (extracted - self.mean) / (self.std + 1e-8)

        return squeezed(extracted)

    def compute_normalization_stats(
        self,
        features: List[np.ndarray],
    ) -> None:
        """
        Compute mean and std from training data for normalization.

        Args:
            features: List of 112D feature vectors
        """
        all_extracted = []

        for feat in features:
            extracted = self.extract(feat)
            all_extracted.append(extracted)

        stacked = np.vstack(all_extracted)
        self.mean = stacked.mean(axis=0)
        self.std = stacked.std(axis=0)

        logger.info(
            f"Computed normalization stats: mean shape {self.mean.shape}, "
            f"std shape {self.std.shape}"
        )

    def save_normalization_stats(self, path: str) -> None:
        """Save normalization statistics to file."""
        if self.mean is None or self.std is None:
            raise ValueError("No normalization stats to save")

        np.savez(
            path,
            mean=self.mean,
            std=self.std,
            feature_indices=np.array(self.config.feature_indices),
        )
        logger.info(f"Saved normalization stats to {path}")

    def load_normalization_stats(self, path: str) -> None:
        """Load normalization statistics from file."""
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]

        # Optionally load feature indices
        if "feature_indices" in data:
            self.config.feature_indices = data["feature_indices"].tolist()
            self.output_dim = len(self.config.feature_indices)

        logger.info(f"Loaded normalization stats from {path}")

    def get_feature_names(self) -> List[str]:
        """Get human-readable names for extracted features."""
        names = {
            0: "F0",
            7: "HNR",
            35: "Jitter",
            36: "Shimmer",
            37: "APQ",
            38: "PPQ",
            39: "RAP",
            8: "MFCC1", 9: "MFCC2", 10: "MFCC3", 11: "MFCC4",
            12: "MFCC5", 13: "MFCC6",
            40: "Attack", 41: "Decay", 42: "Sustain", 43: "Release",
            59: "GLCM_Contrast", 60: "GLCM_Dissimilarity",
            61: "GLCM_Homogeneity", 62: "GLCM_Energy",
            63: "GLCM_Correlation", 64: "GLCM_ASM",
            65: "Spectral_Flux", 66: "Spectral_Rolloff",
            67: "Spectral_Centroid", 68: "Spectral_Bandwidth",
            50: "Harmonic_Ratio", 51: "Inharmonicity",
            52: "F0_Range", 53: "F0_Std", 54: "F0_CV",
        }

        # Micro features
        for i in range(76, 112):
            names[i] = f"Micro_{i-76}"

        return [names.get(idx, f"Feature_{idx}") for idx in self.config.feature_indices]


def create_affective_feature_extractor(
    config: Optional[AffectiveFeatureConfig] = None,
) -> AffectiveFeatureExtractor:
    """Factory function to create affective feature extractor."""
    return AffectiveFeatureExtractor(config)


def squeezed(arr: np.ndarray) -> np.ndarray:
    """Remove singleton dimensions if present."""
    if arr.shape[0] == 1 and arr.ndim == 2:
        return arr[0]
    return arr


# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    # Test the extractor
    extractor = create_affective_feature_extractor()

    # Create dummy 112D input
    dummy_input = np.random.randn(112).astype(np.float32)

    # Extract features
    extracted = extractor.extract(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {extracted.shape}")
    print(f"Feature names (first 10): {extractor.get_feature_names()[:10]}")
