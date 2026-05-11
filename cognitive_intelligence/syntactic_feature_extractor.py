#!/usr/bin/env python3
"""
Syntactic Feature Extractor (Stream 2) - Sprint 1-2

Extracts ~44D syntactic features from 112D Rosetta vector.
These features capture discrete inter-call sequencing and call categories.

Feature Selection Rationale:
- Layer 1 (0-45): MFCCs → spectral envelope (syntactic categories)
- Layer 2 (46-75): Harmonic features, pitch geometry → call types
- Excludes: Layer 3 micro texture (too continuous for syntactic discretization)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyntacticFeatureConfig:
    """Configuration for syntactic feature extraction."""

    # Total input dimension (112D Rosetta vector)
    input_dim: int = 112

    # Output dimension (syntactic features)
    output_dim: int = 44

    # Feature indices (see FEATURE_INDICES below)
    feature_indices: List[int] = None

    # Whether to normalize features
    normalize: bool = True


class SyntacticFeatureExtractor:
    """
    Extract syntactic features from 112D Rosetta vector.

    Stream 2 (Dorsal/"What-When") features:
    - MFCCs: spectral envelope for call type classification
    - Harmonic features: harmonic structure patterns
    - Pitch geometry: F0 patterns defining call categories

    These features are ideal for VQ-VAE encoding to discrete tokens.
    """

    # Feature indices selected from 112D Rosetta vector
    # Selected for syntactic/discrete representation:
    # - Spectral envelope (MFCCs) → call types
    # - Harmonic structure → phoneme categories
    # - Pitch geometry → prosodic categories

    FEATURE_INDICES = [
        # Layer 1: MFCCs (spectral envelope) - primary syntactic features
        # MFCC 1-13 capture spectral shape for call type classification
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,

        # Layer 1: Delta MFCCs (spectral change) - transitional cues
        14, 15, 16, 17,  # Delta MFCC 1-4

        # Layer 1: Spectral features for category discrimination
        18,  # Spectral centroid - brightness category
        19,  # Spectral bandwidth - spread category
        20,  # Spectral flatness - tonal vs noise category
        21,  # Spectral rolloff - spectral shape category
        22,  # Spectral contrast - texture category

        # Layer 2: Harmonic features - phoneme-like categories
        46,  # Harmonic mean - harmonic quality
        47,  # Harmonic std - harmonic variation
        48,  # Harmonic entropy - harmonic complexity
        49,  # Harmonic slope - harmonic tilt

        # Layer 2: Harmonic ratio (already in affective, but crucial for syntax)
        50,  # Harmonic ratio - voiced/unvoiced category

        # Layer 2: Pitch geometry - prosodic categories
        55,  # F0 mean - pitch register category
        56,  # F0 median - pitch center category
        57,  # F0 mode - modal pitch category
        58,  # F0 range - pitch span category

        # Layer 2: Additional pitch features
        52,  # F0 range (duplicate - ensures coverage)
        53,  # F0 std - pitch variation category
        54,  # F0 CV - pitch stability category

        # Layer 2: Formant-like features (from harmonic peaks)
        69,  # Formant 1 frequency - vowel quality
        70,  # Formant 2 frequency - vowel quality
        71,  # Formant 3 frequency - vowel quality

        # Layer 2: Temporal features for sequence timing
        72,  # Onset rate - call rate category
        73,  # Offset rate - offset pattern
        74,  # Duration ratio - temporal pattern

        # Layer 2: Energy distribution
        75,  # Energy envelope - temporal shape category

        # Additional MFCC-derived features for finer categories
        23, 24, 25, 26,  # Additional spectral descriptors
    ]

    def __init__(self, config: Optional[SyntacticFeatureConfig] = None):
        self.config = config or SyntacticFeatureConfig()

        if self.config.feature_indices is None:
            self.config.feature_indices = self.FEATURE_INDICES

        self.output_dim = len(self.config.feature_indices)

        # Normalization statistics (computed from training data)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

        logger.info(
            f"SyntacticFeatureExtractor initialized: "
            f"{self.config.input_dim}D → {self.output_dim}D"
        )

    def extract(self, features_112d: np.ndarray) -> np.ndarray:
        """
        Extract syntactic features from 112D Rosetta vector.

        Args:
            features_112d: Input vector of shape (112,) or (batch, 112)

        Returns:
            Syntactic features of shape (output_dim,) or (batch, output_dim)
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
            1: "MFCC1", 2: "MFCC2", 3: "MFCC3", 4: "MFCC4", 5: "MFCC5",
            6: "MFCC6", 7: "MFCC7", 8: "MFCC8", 9: "MFCC9", 10: "MFCC10",
            11: "MFCC11", 12: "MFCC12", 13: "MFCC13",
            14: "Delta_MFCC1", 15: "Delta_MFCC2", 16: "Delta_MFCC3", 17: "Delta_MFCC4",
            18: "Spectral_Centroid", 19: "Spectral_Bandwidth", 20: "Spectral_Flatness",
            21: "Spectral_Rolloff", 22: "Spectral_Contrast",
            46: "Harmonic_Mean", 47: "Harmonic_Std", 48: "Harmonic_Entropy",
            49: "Harmonic_Slope", 50: "Harmonic_Ratio",
            52: "F0_Range", 53: "F0_Std", 54: "F0_CV",
            55: "F0_Mean", 56: "F0_Median", 57: "F0_Mode", 58: "F0_Range2",
            69: "Formant1", 70: "Formant2", 71: "Formant3",
            72: "Onset_Rate", 73: "Offset_Rate", 74: "Duration_Ratio", 75: "Energy_Envelope",
            23: "Spectral1", 24: "Spectral2", 25: "Spectral3", 26: "Spectral4",
        }

        return [names.get(idx, f"Feature_{idx}") for idx in self.config.feature_indices]


def create_syntactic_feature_extractor(
    config: Optional[SyntacticFeatureConfig] = None,
) -> SyntacticFeatureExtractor:
    """Factory function to create syntactic feature extractor."""
    return SyntacticFeatureExtractor(config)


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
    extractor = create_syntactic_feature_extractor()

    # Create dummy 112D input
    dummy_input = np.random.randn(112).astype(np.float32)

    # Extract features
    extracted = extractor.extract(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {extracted.shape}")
    print(f"Feature names (first 10): {extractor.get_feature_names()[:10]}")
