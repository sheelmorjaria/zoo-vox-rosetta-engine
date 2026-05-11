#!/usr/bin/env python3
"""
Affective Feature Extractor for Stream 1 (Ventral/"How")

This module extracts continuous affect/prosodic features from the 112D
Rosetta feature vector for input to the β-VAE encoder.

Feature selection is based on the three-layer Rosetta hierarchy:
- Layer 1 (0-45): Base physics - F0, HNR, Jitter, Shimmer, Vibrato
- Layer 2 (46-75): Macro texture - GLCM entropy, contrast, correlation
- Layer 3 (76-111): Micro texture - All 36 dimensions

This results in ~30D continuous features suitable for affect encoding.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import numpy as np
from typing import List, Optional


class AffectiveFeatureExtractor:
    """
    Extract continuous affect features from 112D Rosetta vector.

    The selected features capture graded continuum of internal state:
    - Arousal: F0, RMS, HNR, spectral dynamics
    - Tension: Jitter, Shimmer, micro-variations
    - Valence: Harmonic-to-noise ratio, spectral entropy

    Stream 1 feeds the β-VAE for disentangled 16D latent encoding.
    """

    # Indices for continuous affect features from 112D Rosetta
    AFFECTIVE_INDICES: List[int] = [
        # Layer 1: Base Physics (continuous measures)
        0,   # F0_mean - Fundamental frequency (arousal)
        1,   # F0_std - Pitch variation (expressiveness)
        2,   # F0_min - Pitch range low
        3,   # F0_max - Pitch range high
        7,   # HNR - Harmonic-to-noise ratio (valence/tension)
        8,   # HNR_std - HNR variation
        35,  # Jitter - Pitch perturbation (tension/stress)
        36,  # Shimmer - Amplitude perturbation (tension/stress)
        37,  # Vibrato_rate - Oscillation frequency
        38,  # Vibrato_depth - Oscillation depth (expressiveness)

        # Layer 2: Macro Texture - GLCM (affective texture)
        59,  # GLCM_entropy - Texture complexity
        60,  # GLCM_entropy_std - Texture variation
        62,  # GLCM_contrast - Texture contrast (tension)
        63,  # GLCM_contrast_std
        65,  # GLCM_homogeneity - Texture smoothness (calmness)
        66,  # GLCM_homogeneity_std
        67,  # GLCM_correlation - Texture structure
        68,  # GLCM_correlation_std

        # Layer 3: Micro Texture (all 36 dimensions)
        # These capture fine-grained prosodic and affective information
        76,  # Spectral_derivative_mean
        77,  # Spectral_derivative_std
        78,  # Spectral_flux_mean
        79,  # Spectral_flux_std
        80,  # FM_rate_mean - Frequency modulation rate
        81,  # FM_rate_std
        82,  # FM_depth_mean - Frequency modulation depth (arousal)
        83,  # FM_depth_std
        84,  # AM_rate_mean - Amplitude modulation rate
        85,  # AM_rate_std
        86,  # AM_depth_mean - Amplitude modulation depth
        87,  # AM_depth_std
        88,  # Onset_rate - Note/attack rate (excitement)
        89,  # Onset_strength
        90,  # Micro_dynamics_mean
        91,  # Micro_dynamics_std
        92,  # Rhythmic_density
        93,  # Rhythmic_regularity
        94,  # Temporal_entropy
        95,  # Temporal_entropy_std
        96,  # Spectral_centroid_flux
        97,  # Spectral_rolloff_flux
        98,  # Zero_crossing_rate_mean
        99,  # Zero_crossing_rate_std
        100, # Energy_delta_mean
        101, # Energy_delta_std
        102, # Spectral_slope_delta
        103, # Coherence_flux
        104, # Formant_tracking_mean
        105, # Formant_tracking_std
        106, # Pitch_contour_variability
        107, # Jitter_local
        108, # Shimmer_local
        109, # Voice_quality_index
        110, # Breathiness_index
        111, # Tension_index
    ]

    # Remove duplicates and sort
    _UNIQUE_INDICES = sorted(set(AFFECTIVE_INDICES))

    # Expected output dimension (unique indices only)
    OUTPUT_DIM: int = 54  # len(_UNIQUE_INDICES)

    @classmethod
    def extract_affective_features(cls, features_112d: np.ndarray) -> np.ndarray:
        """
        Extract continuous affect features from 112D Rosetta vector.

        Args:
            features_112d: Input feature vector of shape (112,) or (batch, 112)

        Returns:
            Affective features of shape (OUTPUT_DIM,) or (batch, OUTPUT_DIM)

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
    def extract_affective_features_batch(cls, features_112d: np.ndarray) -> np.ndarray:
        """
        Batch extraction of affective features.

        Convenience alias for extract_affective_features with 2D input.

        Args:
            features_112d: Input features of shape (N, 112)

        Returns:
            Affective features of shape (N, OUTPUT_DIM)
        """
        return cls.extract_affective_features(features_112d)

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Return human-readable names for each affective feature dimension."""
        return [
            # Layer 1: Base Physics
            "F0_mean", "F0_std", "F0_min", "F0_max",
            "HNR", "HNR_std",
            "Jitter", "Shimmer", "Vibrato_rate", "Vibrato_depth",

            # Layer 2: GLCM Texture
            "GLCM_entropy", "GLCM_entropy_std",
            "GLCM_contrast", "GLCM_contrast_std",
            "GLCM_homogeneity", "GLCM_homogeneity_std",
            "GLCM_correlation", "GLCM_correlation_std",

            # Layer 3: Micro Texture
            "Spectral_derivative_mean", "Spectral_derivative_std",
            "Spectral_flux_mean", "Spectral_flux_std",
            "FM_rate_mean", "FM_rate_std",
            "FM_depth_mean", "FM_depth_std",
            "AM_rate_mean", "AM_rate_std",
            "AM_depth_mean", "AM_depth_std",
            "Onset_rate", "Onset_strength",
            "Micro_dynamics_mean", "Micro_dynamics_std",
            "Rhythmic_density", "Rhythmic_regularity",
            "Temporal_entropy", "Temporal_entropy_std",
            "Spectral_centroid_flux", "Spectral_rolloff_flux",
            "Zero_crossing_rate_mean", "Zero_crossing_rate_std",
            "Energy_delta_mean", "Energy_delta_std",
            "Spectral_slope_delta", "Coherence_flux",
            "Formant_tracking_mean", "Formant_tracking_std",
            "Pitch_contour_variability", "Jitter_local", "Shimmer_local",
            "Voice_quality_index", "Breathiness_index", "Tension_index",
        ]

    @classmethod
    def validate_feature_vector(cls, features_112d: np.ndarray) -> bool:
        """
        Validate that input is a properly formed 112D Rosetta vector.

        Checks:
        - Correct shape (112,)
        - Finite values (no NaN or Inf)
        - Reasonable ranges for key features

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

        # Check reasonable ranges for key features
        # F0 should be positive and within animal vocalization range
        f0_mean = features[0]
        if f0_mean <= 0 or f0_mean > 150000:  # 150 kHz for bats
            return False

        # HNR should be between 0 and 1 (or slightly above due to measurement)
        hnr = features[7]
        if hnr < 0 or hnr > 50:  # Allow some margin
            return False

        # Jitter and Shimmer should be non-negative
        if features[35] < 0 or features[36] < 0:
            return False

        return True


class AffectiveNormalization:
    """
    Normalization utilities for affective features.

    Ensures features are appropriately scaled for VAE training.
    """

    # Default normalization parameters (learned from training data)
    DEFAULT_MEAN: Optional[np.ndarray] = None
    DEFAULT_STD: Optional[np.ndarray] = None

    @classmethod
    def compute_normalization(cls, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std for z-score normalization.

        Args:
            features: Array of shape (n_samples, OUTPUT_DIM)

        Returns:
            (mean, std) arrays for normalization
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        # Avoid division by zero
        std = np.where(std < 1e-8, 1.0, std)
        return mean, std

    @classmethod
    def normalize(cls, features: np.ndarray,
                  mean: np.ndarray,
                  std: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization.

        Args:
            features: Input features (n_samples, OUTPUT_DIM)
            mean: Mean vector (OUTPUT_DIM,)
            std: Std vector (OUTPUT_DIM,)

        Returns:
            Normalized features
        """
        return (features - mean) / std

    @classmethod
    def denormalize(cls, features: np.ndarray,
                    mean: np.ndarray,
                    std: np.ndarray) -> np.ndarray:
        """
        Reverse z-score normalization.

        Args:
            features: Normalized features (n_samples, OUTPUT_DIM)
            mean: Mean vector (OUTPUT_DIM,)
            std: Std vector (OUTPUT_DIM,)

        Returns:
            Denormalized features
        """
        return features * std + mean


def extract_affective_features(features_112d: np.ndarray) -> np.ndarray:
    """
    Convenience function for affective feature extraction.

    Args:
        features_112d: Input 112D Rosetta feature vector

    Returns:
        Affective feature vector (OUTPUT_DIM,)
    """
    return AffectiveFeatureExtractor.extract_affective_features(features_112d)
