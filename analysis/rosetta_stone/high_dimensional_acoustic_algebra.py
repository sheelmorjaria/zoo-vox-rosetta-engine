"""
High-Dimensional Acoustic Algebra
===================================

Enhanced Acoustic Algebra using all 30 micro-dynamics features for
true timbral interpolation, not just pitch shifting.

Key Innovation:
- Normalizes all features to Z-scores before interpolation
- Denormalizes back to physical units for synthesis
- Enables "Phonetic Constraints" - discovering physical limits of vocal production

30D Feature Vector:
- Fundamental (3): mean_f0_hz, f0_range_hz, duration_ms
- Grit Factors (3): harmonic_to_noise_ratio, spectral_flatness, harmonicity
- Motion Factors (7): attack_time_ms, decay_time_ms, sustain_level,
  vibrato_rate_hz, vibrato_depth, jitter, shimmer
- Fingerprint Factors (13 MFCCs): mfcc_1 through mfcc_13
- Spectral Dynamics (1): spectral_flux
- Rhythm Factors (3): median_ici_ms, onset_rate_hz, ici_coefficient_of_variation

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# 30-Dimensional Acoustic Feature Vector
# ============================================================================


@dataclass
class AcousticFeatureVector30:
    """
    Complete 30-dimensional acoustic feature vector for high-dimensional
    interpolation and algebra.

    Features are organized into logical groups matching the micro-dynamics
    extraction pipeline:

    1. Fundamental (3): mean_f0_hz, f0_range_hz, duration_ms
    2. Grit Factors (3): harmonic_to_noise_ratio, spectral_flatness, harmonicity
    3. Motion Factors (7): attack_time_ms, decay_time_ms, sustain_level,
                       vibrato_rate_hz, vibrato_depth, jitter, shimmer
    4. Fingerprint Factors (13 MFCCs): mfcc_1 through mfcc_13
    5. Spectral Dynamics (1): spectral_flux
    6. Rhythm Factors (3): median_ici_ms, onset_rate_hz, ici_coefficient_of_variation
    """

    # Fundamental (3)
    mean_f0_hz: float
    f0_range_hz: float
    duration_ms: float

    # Grit Factors (3)
    harmonic_to_noise_ratio: float
    spectral_flatness: float
    harmonicity: float

    # Motion Factors (7)
    attack_time_ms: float
    decay_time_ms: float
    sustain_level: float
    vibrato_rate_hz: float
    vibrato_depth: float
    jitter: float
    shimmer: float

    # Fingerprint Factors (13 MFCCs)
    mfcc_1: float
    mfcc_2: float
    mfcc_3: float
    mfcc_4: float
    mfcc_5: float
    mfcc_6: float
    mfcc_7: float
    mfcc_8: float
    mfcc_9: float
    mfcc_10: float
    mfcc_11: float
    mfcc_12: float
    mfcc_13: float

    # Spectral Dynamics (1)
    spectral_flux: float

    # Rhythm Factors (3)
    median_ici_ms: float
    onset_rate_hz: float
    ici_coefficient_of_variation: float

    def to_numpy(self) -> np.ndarray:
        """Convert to 1D numpy array (30-dim vector)."""
        return np.array(
            [
                # Fundamental (3)
                self.mean_f0_hz,
                self.f0_range_hz,
                self.duration_ms,
                # Grit Factors (3)
                self.harmonic_to_noise_ratio,
                self.spectral_flatness,
                self.harmonicity,
                # Motion Factors (7)
                self.attack_time_ms,
                self.decay_time_ms,
                self.sustain_level,
                self.vibrato_rate_hz,
                self.vibrato_depth,
                self.jitter,
                self.shimmer,
                # Fingerprint Factors (13 MFCCs)
                self.mfcc_1,
                self.mfcc_2,
                self.mfcc_3,
                self.mfcc_4,
                self.mfcc_5,
                self.mfcc_6,
                self.mfcc_7,
                self.mfcc_8,
                self.mfcc_9,
                self.mfcc_10,
                self.mfcc_11,
                self.mfcc_12,
                self.mfcc_13,
                # Spectral Dynamics (1)
                self.spectral_flux,
                # Rhythm Factors (3)
                self.median_ici_ms,
                self.onset_rate_hz,
                self.ici_coefficient_of_variation,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_numpy(cls, vec: np.ndarray) -> "AcousticFeatureVector30":
        """Create from 1D numpy array."""
        return cls(
            # Fundamental (3)
            mean_f0_hz=vec[0],
            f0_range_hz=vec[1],
            duration_ms=vec[2],
            # Grit Factors (3)
            harmonic_to_noise_ratio=vec[3],
            spectral_flatness=vec[4],
            harmonicity=vec[5],
            # Motion Factors (7)
            attack_time_ms=vec[6],
            decay_time_ms=vec[7],
            sustain_level=vec[8],
            vibrato_rate_hz=vec[9],
            vibrato_depth=vec[10],
            jitter=vec[11],
            shimmer=vec[12],
            # Fingerprint Factors (13 MFCCs)
            mfcc_1=vec[13],
            mfcc_2=vec[14],
            mfcc_3=vec[15],
            mfcc_4=vec[16],
            mfcc_5=vec[17],
            mfcc_6=vec[18],
            mfcc_7=vec[19],
            mfcc_8=vec[20],
            mfcc_9=vec[21],
            mfcc_10=vec[22],
            mfcc_11=vec[23],
            mfcc_12=vec[24],
            mfcc_13=vec[25],
            # Spectral Dynamics (1)
            spectral_flux=vec[26],
            # Rhythm Factors (3)
            median_ici_ms=vec[27],
            onset_rate_hz=vec[28],
            ici_coefficient_of_variation=vec[29],
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (for feature extraction)."""
        return {
            # Fundamental (3)
            "mean_f0_hz": self.mean_f0_hz,
            "f0_range_hz": self.f0_range_hz,
            "duration_ms": self.duration_ms,
            # Grit Factors (3)
            "harmonic_to_noise_ratio": self.harmonic_to_noise_ratio,
            "spectral_flatness": self.spectral_flatness,
            "harmonicity": self.harmonicity,
            # Motion Factors (7)
            "attack_time_ms": self.attack_time_ms,
            "decay_time_ms": self.decay_time_ms,
            "sustain_level": self.sustain_level,
            "vibrato_rate_hz": self.vibrato_rate_hz,
            "vibrato_depth": self.vibrato_depth,
            "jitter": self.jitter,
            "shimmer": self.shimmer,
            # Fingerprint Factors (13 MFCCs)
            "mfcc_1": self.mfcc_1,
            "mfcc_2": self.mfcc_2,
            "mfcc_3": self.mfcc_3,
            "mfcc_4": self.mfcc_4,
            "mfcc_5": self.mfcc_5,
            "mfcc_6": self.mfcc_6,
            "mfcc_7": self.mfcc_7,
            "mfcc_8": self.mfcc_8,
            "mfcc_9": self.mfcc_9,
            "mfcc_10": self.mfcc_10,
            "mfcc_11": self.mfcc_11,
            "mfcc_12": self.mfcc_12,
            "mfcc_13": self.mfcc_13,
            # Spectral Dynamics (1)
            "spectral_flux": self.spectral_flux,
            # Rhythm Factors (3)
            "median_ici_ms": self.median_ici_ms,
            "onset_rate_hz": self.onset_rate_hz,
            "ici_coefficient_of_variation": self.ici_coefficient_of_variation,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "AcousticFeatureVector30":
        """Create from dictionary (with defaults for missing features)."""
        return cls(
            # Fundamental (3)
            mean_f0_hz=d.get("mean_f0_hz", 7000.0),
            f0_range_hz=d.get("f0_range_hz", 400.0),
            duration_ms=d.get("duration_ms", 50.0),
            # Grit Factors (3)
            harmonic_to_noise_ratio=d.get("harmonic_to_noise_ratio", 20.0),
            spectral_flatness=d.get("spectral_flatness", 0.3),
            harmonicity=d.get("harmonicity", 0.8),
            # Motion Factors (7)
            attack_time_ms=d.get("attack_time_ms", 5.0),
            decay_time_ms=d.get("decay_time_ms", 20.0),
            sustain_level=d.get("sustain_level", 0.7),
            vibrato_rate_hz=d.get("vibrato_rate_hz", 7.0),
            vibrato_depth=d.get("vibrato_depth", 0.02),
            jitter=d.get("jitter", 0.01),
            shimmer=d.get("shimmer", 0.03),
            # Fingerprint Factors (13 MFCCs)
            mfcc_1=d.get("mfcc_1", -10.0),
            mfcc_2=d.get("mfcc_2", -5.0),
            mfcc_3=d.get("mfcc_3", -2.0),
            mfcc_4=d.get("mfcc_4", -1.0),
            mfcc_5=d.get("mfcc_5", -0.5),
            mfcc_6=d.get("mfcc_6", -0.3),
            mfcc_7=d.get("mfcc_7", -0.2),
            mfcc_8=d.get("mfcc_8", -0.1),
            mfcc_9=d.get("mfcc_9", 0.0),
            mfcc_10=d.get("mfcc_10", 0.1),
            mfcc_11=d.get("mfcc_11", 0.2),
            mfcc_12=d.get("mfcc_12", 0.3),
            mfcc_13=d.get("mfcc_13", 0.4),
            # Spectral Dynamics (1)
            spectral_flux=d.get("spectral_flux", 0.5),
            # Rhythm Factors (3)
            median_ici_ms=d.get("median_ici_ms", 15.0),
            onset_rate_hz=d.get("onset_rate_hz", 8.0),
            ici_coefficient_of_variation=d.get("ici_coefficient_of_variation", 0.3),
        )

    def __repr__(self) -> str:
        return (
            f"AcousticFeatureVector30(F0={self.mean_f0_hz:.0f}Hz, "
            f"Dur={self.duration_ms:.1f}ms, Range={self.f0_range_hz:.0f}Hz, "
            f"HNR={self.harmonic_to_noise_ratio:.1f}dB, Flatness={self.spectral_flatness:.2f})"
        )

    def feature_names(self) -> List[str]:
        """Get list of all 30 feature names in order."""
        return [
            # Fundamental (3)
            "mean_f0_hz",
            "f0_range_hz",
            "duration_ms",
            # Grit Factors (3)
            "harmonic_to_noise_ratio",
            "spectral_flatness",
            "harmonicity",
            # Motion Factors (7)
            "attack_time_ms",
            "decay_time_ms",
            "sustain_level",
            "vibrato_rate_hz",
            "vibrato_depth",
            "jitter",
            "shimmer",
            # Fingerprint Factors (13 MFCCs)
            "mfcc_1",
            "mfcc_2",
            "mfcc_3",
            "mfcc_4",
            "mfcc_5",
            "mfcc_6",
            "mfcc_7",
            "mfcc_8",
            "mfcc_9",
            "mfcc_10",
            "mfcc_11",
            "mfcc_12",
            "mfcc_13",
            # Spectral Dynamics (1)
            "spectral_flux",
            # Rhythm Factors (3)
            "median_ici_ms",
            "onset_rate_hz",
            "ici_coefficient_of_variation",
        ]


# ============================================================================
# Z-Score Normalization for 30D Features
# ============================================================================


class ZScoreNormalizer:
    """
    Z-score normalization for 30-dimensional acoustic feature vectors.

    Normalizes each feature to zero mean and unit variance based on corpus
    statistics. This enables meaningful interpolation and algebra in 30D space.

    Typical ranges for each feature (for marmoset calls):
    - F0: 5000-10000 Hz
    - Duration: 20-100 ms
    - F0 Range: 100-800 Hz
    - HNR: 5-30 dB
    - Spectral Flatness: 0.05-0.5
    - Harmonicity: 0.5-1.0
    - Attack: 2-20 ms
    - Decay: 10-50 ms
    - Sustain: 0.3-0.9
    - Vibrato Rate: 5-15 Hz
    - Vibrato Depth: 0.01-0.1
    - Jitter: 0.005-0.05
    - Shimmer: 0.01-0.1
    - MFCCs: -20 to +20
    - Spectral Flux: 0.1-1.0
    - Median ICI: 5-30 ms
    - Onset Rate: 2-20 Hz
    - ICI CV: 0.1-0.6
    """

    def __init__(
        self,
        mean_vector: Optional[np.ndarray] = None,
        std_vector: Optional[np.ndarray] = None,
        species: str = "marmoset",
    ):
        """
        Initialize normalizer with corpus statistics.

        Parameters:
        - mean_vector: Mean of each feature across corpus (30-dim)
        - std_vector: Standard deviation of each feature across corpus (30-dim)
        - species: Species for default statistics (marmoset, bat, etc.)
        """
        if mean_vector is not None and std_vector is not None:
            self.mean = mean_vector
            self.std = std_vector
        else:
            # Use species-specific defaults
            if species == "marmoset":
                self.mean = self._default_marmoset_mean()
                self.std = self._default_marmoset_std()
            else:
                # Generic defaults
                self.mean = np.zeros(30)
                self.std = np.ones(30)

    def _default_marmoset_mean(self) -> np.ndarray:
        """Default mean vector for marmoset calls (30-dim)."""
        return np.array(
            [
                # Fundamental (3)
                7000.0,  # mean_f0_hz
                400.0,  # f0_range_hz
                50.0,  # duration_ms
                # Grit Factors (3)
                20.0,  # harmonic_to_noise_ratio (dB)
                0.3,  # spectral_flatness
                0.8,  # harmonicity
                # Motion Factors (7)
                5.0,  # attack_time_ms
                20.0,  # decay_time_ms
                0.7,  # sustain_level
                7.0,  # vibrato_rate_hz
                0.02,  # vibrato_depth
                0.01,  # jitter
                0.03,  # shimmer
                # Fingerprint Factors (13 MFCCs)
                -10.0,  # mfcc_1
                -5.0,  # mfcc_2
                -2.0,  # mfcc_3
                -1.0,  # mfcc_4
                -0.5,  # mfcc_5
                -0.3,  # mfcc_6
                -0.2,  # mfcc_7
                -0.1,  # mfcc_8
                0.0,  # mfcc_9
                0.1,  # mfcc_10
                0.2,  # mfcc_11
                0.3,  # mfcc_12
                0.4,  # mfcc_13
                # Spectral Dynamics (1)
                0.5,  # spectral_flux
                # Rhythm Factors (3)
                15.0,  # median_ici_ms
                8.0,  # onset_rate_hz
                0.3,  # ici_coefficient_of_variation
            ],
            dtype=np.float64,
        )

    def _default_marmoset_std(self) -> np.ndarray:
        """Default std vector for marmoset calls (30-dim)."""
        return np.array(
            [
                # Fundamental (3)
                1500.0,  # mean_f0_hz (±1500Hz)
                200.0,  # f0_range_hz (±200Hz)
                30.0,  # duration_ms (±30ms)
                # Grit Factors (3)
                10.0,  # harmonic_to_noise_ratio (±10dB)
                0.2,  # spectral_flatness (±0.2)
                0.15,  # harmonicity (±0.15)
                # Motion Factors (7)
                5.0,  # attack_time_ms (±5ms)
                15.0,  # decay_time_ms (±15ms)
                0.2,  # sustain_level (±0.2)
                3.0,  # vibrato_rate_hz (±3Hz)
                0.03,  # vibrato_depth
                0.015,  # jitter
                0.02,  # shimmer
                # Fingerprint Factors (13 MFCCs)
                5.0,  # mfcc_1 (±5)
                5.0,  # mfcc_2 (±5)
                5.0,  # mfcc_3 (±5)
                5.0,  # mfcc_4 (±5)
                5.0,  # mfcc_5 (±5)
                5.0,  # mfcc_6 (±5)
                5.0,  # mfcc_7 (±5)
                5.0,  # mfcc_8 (±5)
                5.0,  # mfcc_9 (±5)
                5.0,  # mfcc_10 (±5)
                5.0,  # mfcc_11 (±5)
                5.0,  # mfcc_12 (±5)
                5.0,  # mfcc_13 (±5)
                # Spectral Dynamics (1)
                0.3,  # spectral_flux (±0.3)
                # Rhythm Factors (3)
                10.0,  # median_ici_ms (±10ms)
                5.0,  # onset_rate_hz (±5Hz)
                0.15,  # ici_coefficient_of_variation (±0.15)
            ],
            dtype=np.float64,
        )

    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize feature vector to Z-scores.

        Parameters:
            vector: Raw feature vector (30-dim)

        Returns:
            Normalized vector (30-dim Z-scores)
        """
        return (vector - self.mean) / (self.std + 1e-8)

    def denormalize(self, zscore_vector: np.ndarray) -> np.ndarray:
        """
        Denormalize Z-score vector back to physical units.

        Parameters:
            zscore_vector: Normalized vector (30-dim Z-scores)

        Returns:
            Physical feature vector (30-dim)
        """
        return zscore_vector * self.std + self.mean

    def compute_corpus_statistics(self, vectors: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std from corpus of feature vectors.

        Parameters:
            vectors: List of feature vectors (30-dim each)

        Returns:
            (mean_vector, std_vector) - Statistics for each dimension
        """
        matrix = np.stack(vectors)  # (N, 30)
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        return mean, std


# ============================================================================
# 30D Acoustic Algebra Engine
# ============================================================================


class AcousticAlgebraEngine30D:
    """
    High-dimensional acoustic algebra engine using 30 micro-dynamics features.

    Key operations:
    1. Vector addition: v1 + v2 (combine features)
    2. Scalar multiplication: alpha * v (scale features)
    3. Interpolation: (1-alpha) * v1 + alpha * v2 (blend features)
    4. Delta calculation: v2 - v1 (get difference)
    5. Z-score normalization: Enable meaningful interpolation

    All operations are performed in normalized Z-score space to ensure
    each feature contributes equally to the interpolation.
    """

    def __init__(self, normalizer: Optional[ZScoreNormalizer] = None):
        """
        Initialize algebra engine.

        Parameters:
            normalizer: Z-score normalizer (uses marmoset defaults if None)
        """
        self.normalizer = normalizer or ZScoreNormalizer(species="marmoset")

    def interpolate(
        self,
        v1: AcousticFeatureVector30,
        v2: AcousticFeatureVector30,
        alpha: float,
    ) -> AcousticFeatureVector30:
        """
        Interpolate between two 30D vectors.

        Parameters:
            v1: Start vector
            v2: End vector
            alpha: Interpolation factor (0.0 = v1, 1.0 = v2)

        Returns:
            Interpolated vector: (1-alpha) * v1 + alpha * v2
        """
        # Convert to numpy
        vec1 = v1.to_numpy()
        vec2 = v2.to_numpy()

        # Normalize to Z-scores
        z1 = self.normalizer.normalize(vec1)
        z2 = self.normalizer.normalize(vec2)

        # Interpolate in Z-score space
        z_interp = (1 - alpha) * z1 + alpha * z2

        # Denormalize back to physical units
        vec_interp = self.normalizer.denormalize(z_interp)

        return AcousticFeatureVector30.from_numpy(vec_interp)

    def add(
        self,
        v1: AcousticFeatureVector30,
        v2: AcousticFeatureVector30,
    ) -> AcousticFeatureVector30:
        """Add two 30D vectors (v1 + v2)."""
        vec1 = v1.to_numpy()
        vec2 = v2.to_numpy()
        vec_sum = vec1 + vec2
        return AcousticFeatureVector30.from_numpy(vec_sum)

    def subtract(
        self,
        v1: AcousticFeatureVector30,
        v2: AcousticFeatureVector30,
    ) -> AcousticFeatureVector30:
        """Subtract two 30D vectors (v1 - v2)."""
        vec1 = v1.to_numpy()
        vec2 = v2.to_numpy()
        vec_diff = vec1 - vec2
        return AcousticFeatureVector30.from_numpy(vec_diff)

    def scale(
        self,
        v: AcousticFeatureVector30,
        factor: float,
    ) -> AcousticFeatureVector30:
        """Scale 30D vector by factor."""
        vec = v.to_numpy()
        vec_scaled = vec * factor
        return AcousticFeatureVector30.from_numpy(vec_scaled)

    def compute_delta(
        self,
        target: AcousticFeatureVector30,
        source: AcousticFeatureVector30,
    ) -> np.ndarray:
        """
        Compute delta vector (target - source) for synthesis.

        Parameters:
            target: Desired target features
            source: Current source features

        Returns:
            Delta vector (30-dim) for warp calculations
        """
        vec_target = target.to_numpy()
        vec_source = source.to_numpy()
        return vec_target - vec_source


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# Alias for backward compatibility
AcousticFeatureVector = AcousticFeatureVector30
AcousticAlgebraEngine = AcousticAlgebraEngine30D
