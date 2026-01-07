"""
High-Dimensional Acoustic Algebra
===================================

Enhanced Acoustic Algebra using all 17 micro-dynamics features for
true timbral interpolation, not just pitch shifting.

Key Innovation:
- Normalizes all features to Z-scores before interpolation
- Denormalizes back to physical units for synthesis
- Enables "Phonetic Constraints" - discovering physical limits of vocal production

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# 17-Dimensional Acoustic Feature Vector
# ============================================================================


@dataclass
class AcousticFeatureVector17:
    """
    Complete 17-dimensional acoustic feature vector for high-dimensional
    interpolation and algebra.

    Features are organized into logical groups:
    1. Fundamental Frequency (1): mean_f0_hz
    2. Temporal (3): duration_ms, attack_ms, decay_ms
    3. Frequency Modulation (3): f0_range_hz, vibrato_rate_hz, vibrato_depth_hz
    4. Perturbation (2): jitter, shimmer
    5. Harmonic/Noise Balance (2): harmonicity_hnr, spectral_flatness
    6. Spectral Shape (4): spectral_centroid_hz, spectral_rolloff_hz, bandwidth_hz, slope_db_per_octave
    7. Energy (2): rms_db, peak_amplitude
    """

    # Fundamental frequency
    mean_f0_hz: float

    # Temporal features
    duration_ms: float
    attack_ms: float
    decay_ms: float

    # Frequency modulation
    f0_range_hz: float
    vibrato_rate_hz: float
    vibrato_depth_hz: float

    # Perturbation
    jitter: float
    shimmer: float

    # Harmonic/noise balance
    harmonicity_hnr: float
    spectral_flatness: float

    # Spectral shape
    spectral_centroid_hz: float
    spectral_rolloff_hz: float
    bandwidth_hz: float
    slope_db_per_octave: float

    # Energy
    rms_db: float
    peak_amplitude: float

    def to_numpy(self) -> np.ndarray:
        """Convert to 1D numpy array (17-dim vector)."""
        return np.array(
            [
                self.mean_f0_hz,
                self.duration_ms,
                self.attack_ms,
                self.decay_ms,
                self.f0_range_hz,
                self.vibrato_rate_hz,
                self.vibrato_depth_hz,
                self.jitter,
                self.shimmer,
                self.harmonicity_hnr,
                self.spectral_flatness,
                self.spectral_centroid_hz,
                self.spectral_rolloff_hz,
                self.bandwidth_hz,
                self.slope_db_per_octave,
                self.rms_db,
                self.peak_amplitude,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_numpy(cls, vec: np.ndarray) -> "AcousticFeatureVector17":
        """Create from 1D numpy array."""
        return cls(
            mean_f0_hz=vec[0],
            duration_ms=vec[1],
            attack_ms=vec[2],
            decay_ms=vec[3],
            f0_range_hz=vec[4],
            vibrato_rate_hz=vec[5],
            vibrato_depth_hz=vec[6],
            jitter=vec[7],
            shimmer=vec[8],
            harmonicity_hnr=vec[9],
            spectral_flatness=vec[10],
            spectral_centroid_hz=vec[11],
            spectral_rolloff_hz=vec[12],
            bandwidth_hz=vec[13],
            slope_db_per_octave=vec[14],
            rms_db=vec[15],
            peak_amplitude=vec[16],
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (for feature extraction)."""
        return {
            "mean_f0_hz": self.mean_f0_hz,
            "duration_ms": self.duration_ms,
            "attack_ms": self.attack_ms,
            "decay_ms": self.decay_ms,
            "f0_range_hz": self.f0_range_hz,
            "vibrato_rate_hz": self.vibrato_rate_hz,
            "vibrato_depth_hz": self.vibrato_depth_hz,
            "jitter": self.jitter,
            "shimmer": self.shimmer,
            "harmonicity_hnr": self.harmonicity_hnr,
            "spectral_flatness": self.spectral_flatness,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "spectral_rolloff_hz": self.spectral_rolloff_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "slope_db_per_octave": self.slope_db_per_octave,
            "rms_db": self.rms_db,
            "peak_amplitude": self.peak_amplitude,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "AcousticFeatureVector17":
        """Create from dictionary (with defaults for missing features)."""
        return cls(
            mean_f0_hz=d.get("mean_f0_hz", 0.0),
            duration_ms=d.get("duration_ms", 0.0),
            attack_ms=d.get("attack_ms", 0.01),
            decay_ms=d.get("decay_ms", 0.05),
            f0_range_hz=d.get("f0_range_hz", 0.0),
            vibrato_rate_hz=d.get("vibrato_rate_hz", 0.0),
            vibrato_depth_hz=d.get("vibrato_depth_hz", 0.0),
            jitter=d.get("jitter", 0.01),
            shimmer=d.get("shimmer", 0.02),
            harmonicity_hnr=d.get("harmonicity_hnr", 10.0),
            spectral_flatness=d.get("spectral_flatness", 0.5),
            spectral_centroid_hz=d.get("spectral_centroid_hz", 7000.0),
            spectral_rolloff_hz=d.get("spectral_rolloff_hz", 12000.0),
            bandwidth_hz=d.get("bandwidth_hz", 5000.0),
            slope_db_per_octave=d.get("slope_db_per_octave", -6.0),
            rms_db=d.get("rms_db", -20.0),
            peak_amplitude=d.get("peak_amplitude", 0.1),
        )

    def __repr__(self) -> str:
        return (
            f"AcousticFeatureVector17(F0={self.mean_f0_hz:.0f}Hz, "
            f"Dur={self.duration_ms:.1f}ms, Attack={self.attack_ms * 1000:.1f}ms, "
            f"HNR={self.harmonicity_hnr:.1f}dB, Flatness={self.spectral_flatness:.2f})"
        )

    def feature_names(self) -> List[str]:
        """Return list of feature names (in order)."""
        return [
            "mean_f0_hz",
            "duration_ms",
            "attack_ms",
            "decay_ms",
            "f0_range_hz",
            "vibrato_rate_hz",
            "vibrato_depth_hz",
            "jitter",
            "shimmer",
            "harmonicity_hnr",
            "spectral_flatness",
            "spectral_centroid_hz",
            "spectral_rolloff_hz",
            "bandwidth_hz",
            "slope_db_per_octave",
            "rms_db",
            "peak_amplitude",
        ]


# ============================================================================
# Z-Score Normalization
# ============================================================================


class ZScoreNormalizer:
    """
    Normalizes acoustic feature vectors to Z-scores for interpolation.

    CRITICAL: You cannot interpolate raw values!
    - F0 ranges 5000-8000 Hz
    - Attack ranges 0.001-0.050 seconds
    - HNR ranges -10 to +30 dB

    Without normalization, small-magnitude features dominate the interpolation.
    """

    def __init__(
        self, mean_vector: Optional[np.ndarray] = None, std_vector: Optional[np.ndarray] = None
    ):
        """
        Initialize normalizer with corpus statistics.

        Args:
            mean_vector: Mean of each feature across corpus (17-dim)
            std_vector: Standard deviation of each feature across corpus (17-dim)
        """
        if mean_vector is not None and std_vector is not None:
            self.mean = mean_vector
            self.std = std_vector
        else:
            # Default to marmoset statistics
            self.mean = self._default_marmoset_mean()
            self.std = self._default_marmoset_std()

        # Avoid division by zero
        self.std[self.std < 1e-6] = 1.0

    def _default_marmoset_mean(self) -> np.ndarray:
        """Default mean vector for marmoset vocalizations."""
        return np.array(
            [
                6500.0,  # mean_f0_hz
                70.0,  # duration_ms
                0.015,  # attack_ms (15ms)
                0.040,  # decay_ms (40ms)
                500.0,  # f0_range_hz
                8.0,  # vibrato_rate_hz
                50.0,  # vibrato_depth_hz
                0.02,  # jitter
                0.03,  # shimmer
                15.0,  # harmonicity_hnr (dB)
                0.2,  # spectral_flatness
                7500.0,  # spectral_centroid_hz
                13000.0,  # spectral_rolloff_hz
                6000.0,  # bandwidth_hz
                -8.0,  # slope_db_per_octave
                -25.0,  # rms_db
                0.15,  # peak_amplitude
            ],
            dtype=np.float64,
        )

    def _default_marmoset_std(self) -> np.ndarray:
        """Default std vector for marmoset vocalizations."""
        return np.array(
            [
                1000.0,  # mean_f0_hz (±1000Hz)
                30.0,  # duration_ms (±30ms)
                0.010,  # attack_ms (±10ms)
                0.020,  # decay_ms (±20ms)
                800.0,  # f0_range_hz
                3.0,  # vibrato_rate_hz
                30.0,  # vibrato_depth_hz
                0.01,  # jitter
                0.02,  # shimmer
                8.0,  # harmonicity_hnr
                0.15,  # spectral_flatness
                1500.0,  # spectral_centroid_hz
                2000.0,  # spectral_rolloff_hz
                2000.0,  # bandwidth_hz
                3.0,  # slope_db_per_octave
                5.0,  # rms_db
                0.05,  # peak_amplitude
            ],
            dtype=np.float64,
        )

    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize feature vector to Z-scores.

        Z = (X - μ) / σ

        Args:
            vector: Raw feature vector (17-dim)

        Returns:
            Normalized vector (17-dim Z-scores)
        """
        return (vector - self.mean) / self.std

    def denormalize(self, zscore_vector: np.ndarray) -> np.ndarray:
        """
        Denormalize Z-score vector back to physical units.

        X = Z * σ + μ

        Args:
            zscore_vector: Normalized vector (17-dim Z-scores)

        Returns:
            Physical feature vector (17-dim)
        """
        return zscore_vector * self.std + self.mean

    def compute_corpus_statistics(self, vectors: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std from corpus of feature vectors.

        Args:
            vectors: List of feature vectors (17-dim each)

        Returns:
            (mean_vector, std_vector)
        """
        matrix = np.stack(vectors)  # (N, 17)
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)

        self.mean = mean
        self.std = std

        return mean, std


# ============================================================================
# High-Dimensional Acoustic Algebra
# ============================================================================


class HighDimensionalAcousticAlgebra:
    """
    High-dimensional acoustic algebra using all 17 features.

    This enables "Phonetic Constraints" - discovering physical limits
    of vocal production through multi-dimensional interpolation.
    """

    def __init__(self, normalizer: Optional[ZScoreNormalizer] = None):
        """
        Initialize high-dimensional algebra engine.

        Args:
            normalizer: Z-score normalizer (creates default if None)
        """
        self.normalizer = normalizer or ZScoreNormalizer()

    def interpolate(
        self, vector_a: AcousticFeatureVector17, vector_b: AcousticFeatureVector17, alpha: float
    ) -> AcousticFeatureVector17:
        """
        Interpolate between two 17-dimensional vectors.

        Uses Z-score normalization to ensure all features contribute
        equally to the interpolation (not dominated by small-magnitude features).

        Math:
            Z_A = (V_A - μ) / σ
            Z_B = (V_B - μ) / σ
            Z_target = Z_A * (1-α) + Z_B * α
            V_target = Z_target * σ + μ

        Args:
            vector_a: Source vector A
            vector_b: Source vector B
            alpha: Interpolation factor (0.0 = A, 1.0 = B)

        Returns:
            Interpolated vector
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        # Convert to numpy
        vec_a = vector_a.to_numpy()
        vec_b = vector_b.to_numpy()

        # Normalize to Z-scores
        z_a = self.normalizer.normalize(vec_a)
        z_b = self.normalizer.normalize(vec_b)

        # Interpolate in Z-score space (linear)
        z_target = z_a * (1.0 - alpha) + z_b * alpha

        # Denormalize back to physical units
        vec_target = self.normalizer.denormalize(z_target)

        return AcousticFeatureVector17.from_numpy(vec_target)

    def extrapolate(
        self,
        vector_base: AcousticFeatureVector17,
        vector_direction: AcousticFeatureVector17,
        alpha: float,
    ) -> AcousticFeatureVector17:
        """
        Extrapolate beyond a base vector in the direction of another.

        Math:
            V_target = V_base + α * (V_direction - V_base)

        Args:
            vector_base: Base vector (starting point)
            vector_direction: Direction vector (where to go)
            alpha: Extrapolation factor (> 1.0 goes beyond direction)

        Returns:
            Extrapolated vector
        """
        vec_base = vector_base.to_numpy()
        vec_dir = vector_direction.to_numpy()

        # Normalize
        z_base = self.normalizer.normalize(vec_base)
        z_dir = self.normalizer.normalize(vec_dir)

        # Extrapolate in Z-score space
        z_target = z_base + alpha * (z_dir - z_base)

        # Denormalize
        vec_target = self.normalizer.denormalize(z_target)

        return AcousticFeatureVector17.from_numpy(vec_target)

    def add(
        self, vector_a: AcousticFeatureVector17, vector_b: AcousticFeatureVector17
    ) -> AcousticFeatureVector17:
        """
        Add two vectors (vector composition).

        Useful for applying context vectors to base phrases.

        Args:
            vector_a: Base vector
            vector_b: Context/modulation vector

        Returns:
            Composed vector
        """
        vec_a = vector_a.to_numpy()
        vec_b = vector_b.to_numpy()

        # Normalize
        z_a = self.normalizer.normalize(vec_a)
        z_b = self.normalizer.normalize(vec_b)

        # Add in Z-score space
        z_target = z_a + z_b

        # Denormalize
        vec_target = self.normalizer.denormalize(z_target)

        return AcousticFeatureVector17.from_numpy(vec_target)

    def subtract(
        self, vector_a: AcousticFeatureVector17, vector_b: AcousticFeatureVector17
    ) -> AcousticFeatureVector17:
        """
        Subtract two vectors (compute difference).

        Useful for extracting context delta between phrases.

        Args:
            vector_a: Vector A
            vector_b: Vector B

        Returns:
            Difference vector (A - B)
        """
        vec_a = vector_a.to_numpy()
        vec_b = vector_b.to_numpy()

        # Normalize
        z_a = self.normalizer.normalize(vec_a)
        z_b = self.normalizer.normalize(vec_b)

        # Subtract in Z-score space
        z_target = z_a - z_b

        # Denormalize
        vec_target = self.normalizer.denormalize(z_target)

        return AcousticFeatureVector17.from_numpy(vec_target)

    def scalar_multiply(
        self, vector: AcousticFeatureVector17, scalar: float
    ) -> AcousticFeatureVector17:
        """
        Multiply vector by scalar.

        Args:
            vector: Input vector
            scalar: Multiplication factor

        Returns:
            Scaled vector
        """
        vec = vector.to_numpy()

        # Normalize
        z = self.normalizer.normalize(vec)

        # Scale in Z-score space
        z_target = z * scalar

        # Denormalize
        vec_target = self.normalizer.denormalize(z_target)

        return AcousticFeatureVector17.from_numpy(vec_target)

    def check_phonetic_constraints(self, vector: AcousticFeatureVector17) -> Dict[str, any]:
        """
        Check if vector violates "phonetic constraints" (physical limits).

        Example constraints:
        - HNR < 0 dB: Silence (no harmonic content)
        - Attack < 0 ms: Physically impossible
        - Duration <= 0: Physically impossible

        Args:
            vector: Feature vector to check

        Returns:
            Dict with 'valid' bool and 'violations' list
        """
        violations = []

        # Check harmonic-to-noise ratio
        if vector.harmonicity_hnr < 0:
            violations.append(f"HNR < 0 dB: {vector.harmonicity_hnr:.1f} dB (Silence)")

        # Check temporal features
        if vector.attack_ms < 0:
            violations.append(f"Attack < 0 ms: {vector.attack_ms * 1000:.1f} ms (Impossible)")

        if vector.decay_ms < 0:
            violations.append(f"Decay < 0 ms: {vector.decay_ms * 1000:.1f} ms (Impossible)")

        if vector.duration_ms <= 0:
            violations.append(f"Duration <= 0 ms: {vector.duration_ms:.1f} ms (Impossible)")

        # Check F0
        if vector.mean_f0_hz < 100 or vector.mean_f0_hz > 20000:
            violations.append(f"F0 out of range: {vector.mean_f0_hz:.0f} Hz")

        # Check perturbation (should be positive)
        if vector.jitter < 0:
            violations.append(f"Jitter < 0: {vector.jitter:.3f}")

        if vector.shimmer < 0:
            violations.append(f"Shimmer < 0: {vector.shimmer:.3f}")

        # Check energy
        if vector.rms_db < -60:
            violations.append(f"RMS too low: {vector.rms_db:.1f} dB (Silence)")

        return {"valid": len(violations) == 0, "violations": violations}


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HIGH-DIMENSIONAL ACOUSTIC ALGEBRA DEMONSTRATION")
    print("=" * 80)

    # Create algebra engine
    algebra = HighDimensionalAcousticAlgebra()

    # Example vectors (Phee vs Alarm)
    phee = AcousticFeatureVector17(
        mean_f0_hz=6526,
        duration_ms=76.5,
        attack_ms=0.010,
        decay_ms=0.050,
        f0_range_hz=427,
        vibrato_rate_hz=8.0,
        vibrato_depth_hz=50.0,
        jitter=0.02,
        shimmer=0.03,
        harmonicity_hnr=20.0,
        spectral_flatness=0.1,
        spectral_centroid_hz=7000.0,
        spectral_rolloff_hz=13000.0,
        bandwidth_hz=5000.0,
        slope_db_per_octave=-8.0,
        rms_db=-20.0,
        peak_amplitude=0.15,
    )

    alarm = AcousticFeatureVector17(
        mean_f0_hz=6020,
        duration_ms=58.1,
        attack_ms=0.005,
        decay_ms=0.030,
        f0_range_hz=3722,
        vibrato_rate_hz=12.0,
        vibrato_depth_hz=150.0,
        jitter=0.08,
        shimmer=0.05,
        harmonicity_hnr=5.0,
        spectral_flatness=0.3,
        spectral_centroid_hz=8000.0,
        spectral_rolloff_hz=15000.0,
        bandwidth_hz=8000.0,
        slope_db_per_octave=-4.0,
        rms_db=-15.0,
        peak_amplitude=0.25,
    )

    print("\n--- Vector A (Phee) ---")
    print(phee)

    print("\n--- Vector B (Alarm) ---")
    print(alarm)

    # Interpolate at midpoint
    print("\n--- Interpolation at α=0.5 ---")
    midpoint = algebra.interpolate(phee, alarm, alpha=0.5)
    print(midpoint)

    # Check constraints
    print("\n--- Phonetic Constraints ---")
    constraints = algebra.check_phonetic_constraints(midpoint)
    if constraints["valid"]:
        print("✅ All constraints satisfied")
    else:
        print("❌ Constraint violations:")
        for v in constraints["violations"]:
            print(f"  - {v}")

    # Extrapolate beyond alarm
    print("\n--- Extrapolation (1.5x beyond Alarm) ---")
    extreme = algebra.extrapolate(phee, alarm, alpha=1.5)
    print(extreme)

    constraints = algebra.check_phonetic_constraints(extreme)
    if constraints["valid"]:
        print("✅ All constraints satisfied")
    else:
        print("❌ Constraint violations:")
        for v in constraints["violations"]:
            print(f"  - {v}")

    # Vector subtraction (context delta)
    print("\n--- Context Delta (Alarm - Phee) ---")
    delta = algebra.subtract(alarm, phee)
    print(delta)

    print("\n" + "=" * 80)
    print("\n🎯 High-Dimensional Acoustic Algebra enables:")
    print("   ✓ 17-dimensional interpolation (not just pitch)")
    print("   ✓ Z-score normalization (equal feature weighting)")
    print("   ✓ Phonetic constraint detection (physical limits)")
    print("   ✓ Context extraction (delta between phrases)")
    print()
