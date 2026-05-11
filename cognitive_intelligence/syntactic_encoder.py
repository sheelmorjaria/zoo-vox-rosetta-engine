#!/usr/bin/env python3
"""
Syntactic Feature Extractor for Stream 2 (Dorsal/"What-When")

This module extracts discrete syntactic features from the 112D
Rosetta feature vector for input to the VQ-VAE tokenizer.

Feature selection focuses on categorical, phoneme-like features:
- Layer 1 (0-45): MFCCs, ADSR envelope (phonetic content)
- Layer 2 (46-75): Harmonic structure, pitch geometry (call type)
- Excludes: Micro dynamics (Stream 1 continuous affect)

This results in ~40D features suitable for syntactic tokenization.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import numpy as np
from typing import List


class SyntacticFeatureExtractor:
    """
    Extract discrete syntactic features from 112D Rosetta vector.

    The selected features capture categorical information for call typing:
    - Phonemic content: MFCCs (spectral envelope)
    - Call type: Harmonic structure, pitch geometry
    - Temporal pattern: ADSR envelope, rhythm

    Stream 2 feeds the VQ-VAE for discrete token encoding and syntax graph.
    """

    # Indices for discrete syntactic features from 112D Rosetta
    SYNTACTIC_INDICES: List[int] = [
        # Layer 1: Base Physics - MFCCs (phonetic content)
        9,   # MFCC_1 (spectral envelope coarse)
        10,  # MFCC_2
        11,  # MFCC_3
        12,  # MFCC_4
        13,  # MFCC_5
        14,  # MFCC_6
        15,  # MFCC_7 (fixed from duplicate 14)
        16,  # MFCC_8
        17,  # MFCC_9
        18,  # MFCC_10
        19,  # MFCC_11
        20,  # MFCC_12
        21,  # MFCC_13

        # Layer 1: ADSR Envelope (temporal pattern)
        22,  # Attack_time_ms
        23,  # Decay_time_ms
        24,  # Sustain_time_ms
        25,  # Release_time_ms
        26,  # Attack_level
        27,  # Sustain_level

        # Layer 1: Spectral shape (call type discriminators)
        28,  # Spectral_centroid_mean
        29,  # Spectral_centroid_std
        30,  # Spectral_bandwidth_mean
        31,  # Spectral_bandwidth_std
        32,  # Spectral_rolloff_mean
        33,  # Spectral_rolloff_std
        34,  # Spectral_skewness

        # Layer 2: Macro Texture - Harmonic structure
        46,  # Harmonic_mean
        47,  # Harmonic_std
        48,  # Harmonic_ratio_mean
        49,  # Harmonic_ratio_std
        50,  # Inharmonicity_mean
        51,  # Inharmonicity_std

        # Layer 2: Pitch geometry (melodic contour)
        52,  # Pitch_contour_mean
        53,  # Pitch_contour_std
        54,  # Pitch_contour_range
        55,  # Pitch_slope_mean
        56,  # Pitch_slope_std

        # Layer 2: Spectral coherence
        57,  # Coherence_mean
        58,  # Coherence_std

        # Layer 2: Fundamental statistics
        4,   # F0_mode (most common F0)
        5,   # F0_range (max - min)
        6,   # F0_quartile1
        39,  # RMS_mean
        40,  # RMS_std
    ]

    # Unique indices after removing duplicates
    _UNIQUE_INDICES = sorted(set(SYNTACTIC_INDICES))
    OUTPUT_DIM: int = len(_UNIQUE_INDICES)  # 44 dimensions

    @classmethod
    def extract_syntactic_features(cls, features_112d: np.ndarray) -> np.ndarray:
        """
        Extract discrete syntactic features from 112D Rosetta vector.

        Args:
            features_112d: Input feature vector of shape (112,) or (batch, 112)

        Returns:
            Syntactic features of shape (OUTPUT_DIM,) or (batch, OUTPUT_DIM)

        Raises:
            ValueError: If input shape is incorrect
        """
        features = np.asarray(features_112d, dtype=np.float32)

        if features.ndim == 1:
            if features.shape[0] != 112:
                raise ValueError(
                    f"Expected 112D input vector, got shape {features.shape}"
                )
            return features[cls._UNIQUE_INDICES]

        elif features.ndim == 2:
            if features.shape[1] != 112:
                raise ValueError(
                    f"Expected 112D input features, got shape {features.shape}"
                )
            return features[:, cls._UNIQUE_INDICES]

        else:
            raise ValueError(
                f"Expected 1D or 2D input, got {features.ndim}D"
            )

    @classmethod
    def extract_syntactic_features_batch(cls, features_112d: np.ndarray) -> np.ndarray:
        """
        Batch extraction of syntactic features.

        Convenience alias for extract_syntactic_features with 2D input.

        Args:
            features_112d: Input features of shape (N, 112)

        Returns:
            Syntactic features of shape (N, OUTPUT_DIM)
        """
        return cls.extract_syntactic_features(features_112d)

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Return human-readable names for each syntactic feature dimension."""
        return [
            # MFCCs (phonetic content)
            "MFCC_1", "MFCC_2", "MFCC_3", "MFCC_4", "MFCC_5",
            "MFCC_6", "MFCC_7", "MFCC_8", "MFCC_9", "MFCC_10",
            "MFCC_11", "MFCC_12", "MFCC_13",

            # ADSR Envelope (temporal pattern)
            "Attack_time_ms", "Decay_time_ms", "Sustain_time_ms", "Release_time_ms",
            "Attack_level", "Sustain_level",

            # Spectral shape (call type)
            "Spectral_centroid_mean", "Spectral_centroid_std",
            "Spectral_bandwidth_mean", "Spectral_bandwidth_std",
            "Spectral_rolloff_mean", "Spectral_rolloff_std",
            "Spectral_skewness",

            # Harmonic structure
            "Harmonic_mean", "Harmonic_std",
            "Harmonic_ratio_mean", "Harmonic_ratio_std",
            "Inharmonicity_mean", "Inharmonicity_std",

            # Pitch geometry
            "Pitch_contour_mean", "Pitch_contour_std", "Pitch_contour_range",
            "Pitch_slope_mean", "Pitch_slope_std",

            # Spectral coherence
            "Coherence_mean", "Coherence_std",

            # Fundamental statistics
            "F0_mode", "F0_range", "F0_quartile1",
            "RMS_mean", "RMS_std",
        ]

    @classmethod
    def validate_feature_vector(cls, features_112d: np.ndarray) -> bool:
        """
        Validate that input is a properly formed 112D Rosetta vector.

        Args:
            features_112d: Input feature vector to validate

        Returns:
            True if valid, False otherwise
        """
        features = np.asarray(features_112d, dtype=np.float32)

        # Check shape
        if features.shape != (112,):
            return False

        # Check for NaN or Inf
        if not np.all(np.isfinite(features)):
            return False

        return True


def extract_syntactic_features(features_112d: np.ndarray) -> np.ndarray:
    """
    Convenience function for syntactic feature extraction.

    Args:
        features_112d: Input 112D Rosetta feature vector

    Returns:
        Syntactic feature vector (OUTPUT_DIM,)
    """
    return SyntacticFeatureExtractor.extract_syntactic_features(features_112d)
