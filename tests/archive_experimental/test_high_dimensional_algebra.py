"""
Tests for High-Dimensional Acoustic Algebra and Grammar Discovery
=================================================================

Tests for:
- 17-dimensional feature vectors
- Z-score normalization
- High-dimensional interpolation/extrapolation
- Phonetic constraint checking
- Grain-based phrase discovery
- DBSCAN clustering
- Transition entropy analysis

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings

from analysis.rosetta_stone.grain_based_grammar_discovery import (
    AtomicPhraseDiscoverer,
    Grain,
    GrainExtractor,
    GrammarDiscoveryPipeline,
    SentenceReconstructor,
    SentenceStructure,
    TransitionEntropyAnalyzer,
)

from analysis.rosetta_stone.high_dimensional_acoustic_algebra import (
    AcousticFeatureVector17,
    HighDimensionalAcousticAlgebra,
    ZScoreNormalizer,
)

warnings.filterwarnings("ignore")


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def phee_vector():
    """Standard phee call vector."""
    return AcousticFeatureVector17(
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


@pytest.fixture
def alarm_vector():
    """Alarm call vector."""
    return AcousticFeatureVector17(
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


@pytest.fixture
def algebra():
    """Create algebra engine."""
    return HighDimensionalAcousticAlgebra()


# ============================================================================
# Test Suite 1: AcousticFeatureVector17
# ============================================================================


class TestAcousticFeatureVector17:
    """Test 17-dimensional feature vector."""

    def test_vector_creation(self):
        """Create vector with all 17 features."""
        vec = AcousticFeatureVector17(
            mean_f0_hz=6500,
            duration_ms=70,
            attack_ms=0.01,
            decay_ms=0.04,
            f0_range_hz=500,
            vibrato_rate_hz=8,
            vibrato_depth_hz=50,
            jitter=0.02,
            shimmer=0.03,
            harmonicity_hnr=15,
            spectral_flatness=0.2,
            spectral_centroid_hz=7000,
            spectral_rolloff_hz=13000,
            bandwidth_hz=5000,
            slope_db_per_octave=-8,
            rms_db=-20,
            peak_amplitude=0.15,
        )

        assert vec.mean_f0_hz == 6500
        assert vec.duration_ms == 70
        assert len(vec.to_numpy()) == 17

    def test_numpy_conversion(self, phee_vector):
        """Convert to numpy array and back."""
        vec = phee_vector
        arr = vec.to_numpy()

        assert arr.shape == (17,)
        assert arr[0] == 6526  # F0
        assert arr[1] == 76.5  # Duration

        # Round-trip
        vec2 = AcousticFeatureVector17.from_numpy(arr)
        assert vec2.mean_f0_hz == vec.mean_f0_hz

    def test_dict_conversion(self, phee_vector):
        """Convert to dictionary."""
        d = phee_vector.to_dict()

        assert d["mean_f0_hz"] == 6526
        assert d["duration_ms"] == 76.5
        assert len(d) == 17

    def test_from_dict_with_defaults(self):
        """Create from dict with missing features (uses defaults)."""
        d = {"mean_f0_hz": 6500, "duration_ms": 70}
        vec = AcousticFeatureVector17.from_dict(d)

        assert vec.mean_f0_hz == 6500
        assert vec.duration_ms == 70
        assert vec.attack_ms == 0.01  # Default

    def test_feature_names(self, phee_vector):
        """Get list of feature names."""
        names = phee_vector.feature_names()

        assert len(names) == 17
        assert "mean_f0_hz" in names
        assert "duration_ms" in names
        assert "harmonicity_hnr" in names


# ============================================================================
# Test Suite 2: Z-Score Normalization
# ============================================================================


class TestZScoreNormalizer:
    """Test Z-score normalization for 17-dim vectors."""

    def test_normalize_and_denormalize(self):
        """Normalize and denormalize preserves values."""
        normalizer = ZScoreNormalizer()

        vec = np.array(
            [
                6500,
                70,
                0.01,
                0.04,
                500,
                8,
                50,
                0.02,
                0.03,
                15,
                0.2,
                7000,
                13000,
                5000,
                -8,
                -20,
                0.15,
            ]
        )

        # Normalize
        z = normalizer.normalize(vec)

        # Should have non-zero z-scores
        assert np.any(z != 0)

        # Denormalize
        vec_reconstructed = normalizer.denormalize(z)

        # Should match original
        np.testing.assert_array_almost_equal(vec, vec_reconstructed, decimal=10)

    def test_normalization_scales_differently(self):
        """Different features scale differently based on variance."""
        normalizer = ZScoreNormalizer()

        vec_a = np.array(
            [
                7000,
                100,
                0.02,
                0.05,
                1000,
                10,
                60,
                0.03,
                0.04,
                20,
                0.3,
                8000,
                14000,
                6000,
                -6,
                -18,
                0.2,
            ]
        )
        vec_b = np.array(
            [
                6000,
                40,
                0.01,
                0.03,
                200,
                6,
                40,
                0.01,
                0.02,
                10,
                0.1,
                6000,
                12000,
                4000,
                -10,
                -22,
                0.1,
            ]
        )

        z_a = normalizer.normalize(vec_a)
        z_b = normalizer.normalize(vec_b)

        # Both should be on similar scale (roughly -3 to +3)
        assert np.max(np.abs(z_a)) < 10
        assert np.max(np.abs(z_b)) < 10

    def test_corpus_statistics(self):
        """Compute statistics from corpus."""
        normalizer = ZScoreNormalizer()

        # Create synthetic corpus
        vectors = [
            np.random.randn(17) * 100 + 6500,  # High F0
            np.random.randn(17) * 50 + 6000,  # Low F0
            np.random.randn(17) * 75 + 6250,  # Mid F0
        ]

        mean, std = normalizer.compute_corpus_statistics(vectors)

        assert len(mean) == 17
        assert len(std) == 17
        assert np.all(std > 0)  # All features have variance


# ============================================================================
# Test Suite 3: High-Dimensional Interpolation
# ============================================================================


class TestHighDimensionalInterpolation:
    """Test 17-dimensional interpolation."""

    def test_interpolate_midpoint(self, algebra, phee_vector, alarm_vector):
        """Interpolate at alpha=0.5 (midpoint)."""
        result = algebra.interpolate(phee_vector, alarm_vector, alpha=0.5)

        # Should be halfway between
        assert result.mean_f0_hz > 6000 and result.mean_f0_hz < 6526
        assert result.duration_ms > 58.1 and result.duration_ms < 76.5
        assert result.harmonicity_hnr > 5 and result.harmonicity_hnr < 20

    def test_interpolate_at_bounds(self, algebra, phee_vector, alarm_vector):
        """Interpolate at alpha=0.0 and alpha=1.0."""
        # Alpha=0 should return A
        result_a = algebra.interpolate(phee_vector, alarm_vector, alpha=0.0)
        assert result_a.mean_f0_hz == pytest.approx(phee_vector.mean_f0_hz, rel=0.01)

        # Alpha=1 should return B
        result_b = algebra.interpolate(phee_vector, alarm_vector, alpha=1.0)
        assert result_b.mean_f0_hz == pytest.approx(alarm_vector.mean_f0_hz, rel=0.01)

    def test_interpolate_all_features(self, algebra, phee_vector, alarm_vector):
        """All 17 features should interpolate."""
        result = algebra.interpolate(phee_vector, alarm_vector, alpha=0.3)

        # Check that all features moved toward B
        vec_a = phee_vector.to_numpy()
        vec_b = alarm_vector.to_numpy()
        vec_result = result.to_numpy()

        for i in range(17):
            # Result should be between A and B (for alpha=0.3)
            min_val = min(vec_a[i], vec_b[i])
            max_val = max(vec_a[i], vec_b[i])

            # Allow small tolerance for numerical errors
            assert min_val - 1 <= vec_result[i] <= max_val + 1, f"Feature {i} failed"

    def test_interpolate_invalid_alpha(self, algebra, phee_vector, alarm_vector):
        """Alpha outside [0, 1] should raise error."""
        with pytest.raises(ValueError):
            algebra.interpolate(phee_vector, alarm_vector, alpha=1.5)

        with pytest.raises(ValueError):
            algebra.interpolate(phee_vector, alarm_vector, alpha=-0.1)


# ============================================================================
# Test Suite 4: Phonetic Constraints
# ============================================================================


class TestPhoneticConstraints:
    """Test phonetic constraint checking."""

    def test_valid_vector_passes(self, algebra, phee_vector):
        """Valid vector should pass constraints."""
        result = algebra.check_phonetic_constraints(phee_vector)

        assert result["valid"]
        assert len(result["violations"]) == 0

    def test_negative_hnr_fails(self, algebra):
        """Negative HNR violates constraint (silence)."""
        vec = AcousticFeatureVector17(
            mean_f0_hz=6500,
            duration_ms=70,
            attack_ms=0.01,
            decay_ms=0.04,
            f0_range_hz=500,
            vibrato_rate_hz=8,
            vibrato_depth_hz=50,
            jitter=0.02,
            shimmer=0.03,
            harmonicity_hnr=-5.0,  # Violation!
            spectral_flatness=0.2,
            spectral_centroid_hz=7000,
            spectral_rolloff_hz=13000,
            bandwidth_hz=5000,
            slope_db_per_octave=-8,
            rms_db=-20,
            peak_amplitude=0.15,
        )

        result = algebra.check_phonetic_constraints(vec)

        assert not result["valid"]
        assert any("HNR < 0" in v for v in result["violations"])

    def test_negative_attack_fails(self, algebra):
        """Negative attack time is impossible."""
        vec = AcousticFeatureVector17(
            mean_f0_hz=6500,
            duration_ms=70,
            attack_ms=-0.01,
            decay_ms=0.04,  # Violation!
            f0_range_hz=500,
            vibrato_rate_hz=8,
            vibrato_depth_hz=50,
            jitter=0.02,
            shimmer=0.03,
            harmonicity_hnr=15,
            spectral_flatness=0.2,
            spectral_centroid_hz=7000,
            spectral_rolloff_hz=13000,
            bandwidth_hz=5000,
            slope_db_per_octave=-8,
            rms_db=-20,
            peak_amplitude=0.15,
        )

        result = algebra.check_phonetic_constraints(vec)

        assert not result["valid"]
        assert any("Attack < 0" in v for v in result["violations"])


# ============================================================================
# Test Suite 5: Grain-Based Discovery
# ============================================================================


class TestGrainBasedDiscovery:
    """Test grain extraction and phrase discovery."""

    def test_grain_extraction(self):
        """Extract grains from feature matrix."""
        extractor = GrainExtractor(grain_duration_ms=10.0)

        # Create synthetic features (100 frames, 17 dims)
        features = np.random.randn(100, 17)

        grains = extractor.extract_grains_from_features(features, sample_rate=48000)

        # Should have ~100 grains (one per frame)
        assert len(grains) == 100
        assert grains[0].duration_ms == 10.0

    def test_dbscan_discovers_phrases(self):
        """DBSCAN should discover phrases in synthetic data."""
        # Create synthetic data with 2 clusters
        np.random.seed(42)
        cluster_a = np.random.randn(20, 17) * 0.1 + np.array([0] * 17)
        cluster_b = np.random.randn(30, 17) * 0.1 + np.array([2] * 17)
        features = np.vstack([cluster_a, cluster_b])

        discoverer = AtomicPhraseDiscoverer(eps=0.5, min_samples=3)

        grains = []
        for i, feat in enumerate(features):
            grain = Grain(start_time_ms=i * 10, duration_ms=10, features=feat)
            grains.append(grain)

        phrases, labels = discoverer.discover_phrases(grains)

        # Should discover 2 phrases
        assert len(phrases) >= 1
        assert len(set(labels)) >= 2

    def test_sentence_reconstruction(self):
        """Reconstruct sentence from grain labels."""
        # Create grains with known pattern: A, A, B, B, A
        grains = [
            Grain(0, 10, np.zeros(17), cluster_label=0),
            Grain(10, 10, np.zeros(17), cluster_label=0),
            Grain(20, 10, np.zeros(17), cluster_label=1),
            Grain(30, 10, np.zeros(17), cluster_label=1),
            Grain(40, 10, np.zeros(17), cluster_label=0),
        ]

        reconstructor = SentenceReconstructor()
        structure = reconstructor.reconstruct(grains)

        # Compressed sequence should be [0, 1, 0]
        assert structure.phrase_sequence == [0, 1, 0]
        assert structure.n_phrases == 2


# ============================================================================
# Test Suite 6: Transition Entropy Analysis
# ============================================================================


class TestTransitionEntropyAnalysis:
    """Test entropy-based grammar discovery."""

    def test_deterministic_grammar_zero_entropy(self):
        """Deterministic transitions (A->B always) have zero entropy."""
        # Create structure: 0 always goes to 1, 1 always goes to 0
        structure = SentenceStructure(
            phrase_sequence=[0, 1, 0, 1, 0, 1],
            grain_labels=np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]),
            n_phrases=2,
            transitions={(0, 1): 3, (1, 0): 3},
        )

        analyzer = TransitionEntropyAnalyzer()
        stats = analyzer.analyze(structure)

        # Should have zero entropy (deterministic)
        assert stats.grammar_rigidity == 1.0
        assert stats.mean_entropy < 0.1

    def test_random_grammar_high_entropy(self):
        """Random transitions have high entropy."""
        # Create structure with uniform transitions
        structure = SentenceStructure(
            phrase_sequence=[0, 1, 2],
            grain_labels=np.array([0, 1, 2, 1, 0, 2]),
            n_phrases=3,
            transitions={(0, 1): 1, (0, 2): 1, (1, 0): 1, (1, 2): 1, (2, 0): 1, (2, 1): 1},
        )

        analyzer = TransitionEntropyAnalyzer()
        stats = analyzer.analyze(structure)

        # Should have high entropy (random)
        assert stats.grammar_rigidity < 0.5
        assert stats.mean_entropy >= 1.0  # 1.0 bits for uniform distribution over 2 options


# ============================================================================
# Test Suite 7: Complete Pipeline
# ============================================================================


class TestGrammarDiscoveryPipeline:
    """Test complete grammar discovery pipeline."""

    def test_end_to_end_discovery(self):
        """Test complete pipeline with synthetic data."""
        pipeline = GrammarDiscoveryPipeline(
            grain_duration_ms=10.0, dbscan_eps=0.5, dbscan_min_samples=3
        )

        # Create synthetic sentence: A-A-B-B-A
        np.random.seed(42)
        phrase_a = np.random.randn(10, 17) * 0.1
        phrase_a[:, 0] = 0  # F0 center
        phrase_b = np.random.randn(10, 17) * 0.1
        phrase_b[:, 0] = 3  # Different F0 center
        phrase_a2 = np.random.randn(5, 17) * 0.1
        phrase_a2[:, 0] = 0  # Same as first A

        features = np.vstack([phrase_a, phrase_a, phrase_b, phrase_b, phrase_a2])

        phrases, structure, grammar_stats = pipeline.discover_from_features(
            features, sample_rate=48000
        )

        # Should discover at least 1 phrase
        assert len(phrases) >= 1
        assert structure.n_phrases >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
