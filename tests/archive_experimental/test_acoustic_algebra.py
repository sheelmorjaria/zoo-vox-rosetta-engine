#!/usr/bin/env python3
"""
Test Suite for Acoustic Algebra System

Tests the "Continuous Acoustic Field" concept where vocalizations exist
as mathematical vectors in a multi-dimensional feature space.

Operations tested:
- Identity: Phrase retrieval
- Addition: Phrase + Context Vector
- Subtraction: Phrase A - Phrase B
- Scalar Multiplication: Phrase * intensity
- Average: Interpolation between phrases
- Composition: Sentence-level extrapolation

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.rosetta_stone.acoustic_algebra import (
    GRAMMAR_CONTEXT,
    SENTENCE_TEMPLATE,
    AcousticAlgebraEngine,
    AcousticVector,
    ContextualExtrapolator,
    ContextVector,
    PhraseInterpolator,
    SentenceExtrapolator,
)

# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_rate():
    return 48000


@pytest.fixture
def neutral_phrase_vector():
    """Neutral marmoset phee call vector."""
    return AcousticVector(
        f0_hz=6000.0,
        duration_ms=50.0,
        f0_range_hz=300.0,
        harmonicity=0.95,
        spectral_flatness=0.1,
        jitter=0.0,
        shimmer=0.0
    )


@pytest.fixture
def excited_phrase_vector():
    """Excited marmoset call vector."""
    return AcousticVector(
        f0_hz=7000.0,
        duration_ms=30.0,
        f0_range_hz=500.0,
        harmonicity=0.90,
        spectral_flatness=0.15,
        jitter=0.05,
        shimmer=0.02
    )


@pytest.fixture
def aggression_context():
    """Aggression context vector (direction in acoustic space)."""
    return ContextVector(
        name="aggression",
        f0_multiplier=1.2,  # +20% pitch
        duration_ratio=0.8,  # -20% duration
        f0_range_multiplier=1.5,  # +50% pitch range
        harmonicity_delta=-0.1,  # Less tonal
        spectral_flatness_delta=0.2,  # More noisy
        jitter_add=0.1,  # Add instability
        shimmer_add=0.05
    )


@pytest.fixture
def urgency_context():
    """Urgency context vector."""
    return ContextVector(
        name="urgency",
        f0_multiplier=1.15,
        duration_ratio=0.7,  # -30% duration
        f0_range_multiplier=2.0,
        harmonicity_delta=-0.15,
        spectral_flatness_delta=0.3,
        jitter_add=0.15,
        shimmer_add=0.08
    )


# ============================================================================
# TEST CLASS: AcousticVector
# ============================================================================

class TestAcousticVector:
    """Test acoustic vector representation and operations."""

    def test_vector_creation(self, neutral_phrase_vector):
        """Should create vector with all features."""
        assert neutral_phrase_vector.f0_hz == 6000.0
        assert neutral_phrase_vector.duration_ms == 50.0
        assert neutral_phrase_vector.f0_range_hz == 300.0

    def test_vector_to_tuple(self, neutral_phrase_vector):
        """Should convert to tuple for mathematical operations."""
        tup = neutral_phrase_vector.to_tuple()
        assert len(tup) == 7  # 7 features
        assert tup[0] == 6000.0  # f0_hz

    def test_vector_from_tuple(self):
        """Should reconstruct vector from tuple."""
        tup = (6000.0, 50.0, 300.0, 0.95, 0.1, 0.0, 0.0)
        vec = AcousticVector.from_tuple(tup)
        assert vec.f0_hz == 6000.0
        assert vec.duration_ms == 50.0

    def test_vector_addition(self, neutral_phrase_vector):
        """Should add two vectors (feature-wise)."""
        other = AcousticVector(1000.0, 10.0, 100.0, 0.0, 0.0, 0.0, 0.0)
        result = neutral_phrase_vector + other

        assert result.f0_hz == 7000.0  # 6000 + 1000
        assert result.duration_ms == 60.0  # 50 + 10

    def test_vector_subtraction(self, neutral_phrase_vector):
        """Should subtract two vectors."""
        other = AcousticVector(1000.0, 10.0, 100.0, 0.0, 0.0, 0.0, 0.0)
        result = neutral_phrase_vector - other

        assert result.f0_hz == 5000.0  # 6000 - 1000
        assert result.duration_ms == 40.0  # 50 - 10

    def test_vector_scalar_multiply(self, neutral_phrase_vector):
        """Should multiply vector by scalar."""
        result = neutral_phrase_vector * 1.5

        assert result.f0_hz == 9000.0  # 6000 * 1.5
        assert result.duration_ms == 75.0  # 50 * 1.5

    def test_vector_division(self, neutral_phrase_vector):
        """Should divide vector by scalar."""
        result = neutral_phrase_vector / 2.0

        assert result.f0_hz == 3000.0  # 6000 / 2
        assert result.duration_ms == 25.0  # 50 / 2


# ============================================================================
# TEST CLASS: ContextVector
# ============================================================================

class TestContextVector:
    """Test context vector for directional extrapolation."""

    def test_context_creation(self, aggression_context):
        """Should create context with transformation parameters."""
        assert aggression_context.name == "aggression"
        assert aggression_context.f0_multiplier == 1.2
        assert aggression_context.duration_ratio == 0.8

    def test_apply_to_neutral(self, neutral_phrase_vector, aggression_context):
        """Should transform neutral vector by context."""
        result = aggression_context.apply(neutral_phrase_vector)

        # F0: 6000 * 1.2 = 7200
        assert abs(result.f0_hz - 7200.0) < 0.1

        # Duration: 50 * 0.8 = 40
        assert abs(result.duration_ms - 40.0) < 0.1

        # Range: 300 * 1.5 = 450
        assert abs(result.f0_range_hz - 450.0) < 0.1

        # Harmonicity: 0.95 - 0.1 = 0.85
        assert abs(result.harmonicity - 0.85) < 0.01

        # Jitter: 0.0 + 0.1 = 0.1
        assert abs(result.jitter - 0.1) < 0.01

    def test_urgency_more_extreme(self, neutral_phrase_vector, urgency_context):
        """Urgency should compress duration more than aggression."""
        result = urgency_context.apply(neutral_phrase_vector)

        # Duration: 50 * 0.7 = 35 (more compression than aggression's 40ms)
        assert abs(result.duration_ms - 35.0) < 0.1

        # Jitter increase should be larger
        assert abs(result.jitter - 0.15) < 0.01


# ============================================================================
# TEST CLASS: PhraseInterpolator
# ============================================================================

class TestPhraseInterpolator:
    """Test interpolation between two phrase vectors."""

    def test_interpolate_midpoint(self, neutral_phrase_vector, excited_phrase_vector):
        """50% interpolation should average features."""
        interpolator = PhraseInterpolator()
        result = interpolator.interpolate(
            neutral_phrase_vector,
            excited_phrase_vector,
            alpha=0.5
        )

        # F0: (6000 + 7000) / 2 = 6500
        assert abs(result.f0_hz - 6500.0) < 0.1

        # Duration: (50 + 30) / 2 = 40
        assert abs(result.duration_ms - 40.0) < 0.1

        # Range: (300 + 500) / 2 = 400
        assert abs(result.f0_range_hz - 400.0) < 0.1

    def test_interpolate_favor_a(self, neutral_phrase_vector, excited_phrase_vector):
        """Alpha=0.2 should favor neutral phrase (80% A, 20% B)."""
        interpolator = PhraseInterpolator()
        result = interpolator.interpolate(
            neutral_phrase_vector,
            excited_phrase_vector,
            alpha=0.2
        )

        # Should be closer to neutral
        expected_f0 = 6000 * 0.8 + 7000 * 0.2  # 6200
        assert abs(result.f0_hz - expected_f0) < 0.1

    def test_interpolate_favor_b(self, neutral_phrase_vector, excited_phrase_vector):
        """Alpha=0.8 should favor excited phrase (20% A, 80% B)."""
        interpolator = PhraseInterpolator()
        result = interpolator.interpolate(
            neutral_phrase_vector,
            excited_phrase_vector,
            alpha=0.8
        )

        # Should be closer to excited
        expected_f0 = 6000 * 0.2 + 7000 * 0.8  # 6800
        assert abs(result.f0_hz - expected_f0) < 0.1

    def test_interpolate_identity(self, neutral_phrase_vector):
        """Alpha=0 should return original vector A."""
        interpolator = PhraseInterpolator()
        result = interpolator.interpolate(
            neutral_phrase_vector,
            neutral_phrase_vector,
            alpha=0.0
        )

        assert result.f0_hz == neutral_phrase_vector.f0_hz

    def test_interpolate_full_b(self, neutral_phrase_vector, excited_phrase_vector):
        """Alpha=1 should return vector B."""
        interpolator = PhraseInterpolator()
        result = interpolator.interpolate(
            neutral_phrase_vector,
            excited_phrase_vector,
            alpha=1.0
        )

        assert result.f0_hz == excited_phrase_vector.f0_hz

    def test_interpolate_clipped_alpha(self, neutral_phrase_vector, excited_phrase_vector):
        """Alpha should be clipped to [0, 1] range."""
        interpolator = PhraseInterpolator()

        # Alpha > 1 should clip to 1.0
        result = interpolator.interpolate(
            neutral_phrase_vector,
            excited_phrase_vector,
            alpha=1.5
        )
        assert result.f0_hz == excited_phrase_vector.f0_hz

        # Alpha < 0 should clip to 0.0
        result = interpolator.interpolate(
            neutral_phrase_vector,
            excited_phrase_vector,
            alpha=-0.5
        )
        assert result.f0_hz == neutral_phrase_vector.f0_hz


# ============================================================================
# TEST CLASS: ContextualExtrapolator
# ============================================================================

class TestContextualExtrapolator:
    """Test extrapolation beyond known data using context vectors."""

    def test_extrapolate_from_neutral(self, neutral_phrase_vector, aggression_context):
        """Should extrapolate aggressive phrase from neutral."""
        extrapolator = ContextualExtrapolator()
        result = extrapolator.extrapolate(
            neutral_phrase_vector,
            aggression_context
        )

        # F0 should increase
        assert result.f0_hz > neutral_phrase_vector.f0_hz

        # Duration should decrease
        assert result.duration_ms < neutral_phrase_vector.duration_ms

        # Should match expected transformation
        assert abs(result.f0_hz - 7200.0) < 0.1  # 6000 * 1.2
        assert abs(result.duration_ms - 40.0) < 0.1  # 50 * 0.8

    def test_extrapolate_beyond_range(self, neutral_phrase_vector):
        """Should extrapolate to values not in training data."""
        extreme_context = ContextVector(
            name="extreme",
            f0_multiplier=2.0,  # 2x pitch (outside typical range)
            duration_ratio=0.5,  # Half duration
            f0_range_multiplier=3.0,
            harmonicity_delta=-0.3,
            spectral_flatness_delta=0.5,
            jitter_add=0.3,
            shimmer_add=0.2
        )

        extrapolator = ContextualExtrapolator()
        result = extrapolator.extrapolate(neutral_phrase_vector, extreme_context)

        # Should create values outside typical range
        assert result.f0_hz == 12000.0  # 2x neutral
        assert result.duration_ms == 25.0  # 0.5x neutral

    def test_chained_extrapolation(self, neutral_phrase_vector):
        """Should support chaining contexts (neutral -> aggression -> extreme)."""
        aggression = ContextVector(
            name="aggression", f0_multiplier=1.2, duration_ratio=0.8,
            f0_range_multiplier=1.5, harmonicity_delta=-0.1,
            spectral_flatness_delta=0.2, jitter_add=0.1, shimmer_add=0.05
        )

        extrapolator = ContextualExtrapolator()

        # First extrapolation
        result1 = extrapolator.extrapolate(neutral_phrase_vector, aggression)

        # Second extrapolation on top of first
        result2 = extrapolator.extrapolate(result1, aggression)

        # Should be cumulative
        # F0: 6000 -> 7200 -> 8640
        expected_f0 = 6000 * 1.2 * 1.2
        assert abs(result2.f0_hz - expected_f0) < 1.0

    def test_extrapolation_with_preservation(self, neutral_phrase_vector):
        """Should preserve modality and other metadata."""
        context = ContextVector(
            name="test", f0_multiplier=1.1, duration_ratio=0.9,
            f0_range_multiplier=1.1, harmonicity_delta=-0.05,
            spectral_flatness_delta=0.1, jitter_add=0.02, shimmer_add=0.01
        )

        extrapolator = ContextualExtrapolator()
        result = extrapolator.extrapolate(neutral_phrase_vector, context)

        # Features should transform
        assert result.f0_hz != neutral_phrase_vector.f0_hz

        # But base characteristics should be recognizable
        assert result.harmonicity > 0.7  # Still tonal


# ============================================================================
# TEST CLASS: SentenceExtrapolator
# ============================================================================

@pytest.fixture
def neutral_sentence():
    """Neutral sentence: [Phrase A] -> [Pause 50ms] -> [Phrase B]."""
    return SENTENCE_TEMPLATE(
        phrases=[
            AcousticVector(f0_hz=6000.0, duration_ms=40.0, f0_range_hz=300.0,
                          harmonicity=0.95, spectral_flatness=0.1, jitter=0.0, shimmer=0.0),
            AcousticVector(f0_hz=6200.0, duration_ms=45.0, f0_range_hz=350.0,
                          harmonicity=0.93, spectral_flatness=0.12, jitter=0.01, shimmer=0.0)
        ],
        pauses_ms=[50.0],  # Pause between phrases
        modality="harmonic"
    )


@pytest.fixture
def urgent_context():
    """Urgency context that compresses timing."""
    return GRAMMAR_CONTEXT(
        name="urgency",
        phrase_f0_multiplier=1.15,
        phrase_duration_ratio=0.7,
        phrase_pause_ratio=0.2,  # Compress pauses heavily
        f0_range_multiplier=1.8,
        harmonicity_delta=-0.1,
        spectral_flatness_delta=0.2,
        jitter_add=0.1,
        shimmer_add=0.05
    )


class TestSentenceExtrapolator:
    """Test sentence-level extrapolation (grammar + context)."""

    def test_extrapolate_sentence_timing(self, neutral_sentence, urgent_context):
        """Should warp timing (phrases + pauses) based on context."""
        extrapolator = SentenceExtrapolator()
        result = extrapolator.extrapolate_sentence(neutral_sentence, urgent_context)

        # Check phrase durations
        # Phrase 1: 40ms * 0.7 = 28ms
        assert abs(result.phrases[0].duration_ms - 28.0) < 1.0

        # Phrase 2: 45ms * 0.7 = 31.5ms
        assert abs(result.phrases[1].duration_ms - 31.5) < 1.0

        # Check pause compression
        # Pause: 50ms * 0.2 = 10ms
        assert len(result.pauses_ms) == 1
        assert abs(result.pauses_ms[0] - 10.0) < 1.0

    def test_extrapolate_sentence_pitch(self, neutral_sentence, urgent_context):
        """Should raise pitch across all phrases."""
        extrapolator = SentenceExtrapolator()
        result = extrapolator.extrapolate_sentence(neutral_sentence, urgent_context)

        # Phrase 1: 6000 * 1.15 = 6900
        assert abs(result.phrases[0].f0_hz - 6900.0) < 1.0

        # Phrase 2: 6200 * 1.15 = 7130
        assert abs(result.phrases[1].f0_hz - 7130.0) < 1.0

    def test_preserve_sentence_structure(self, neutral_sentence, urgent_context):
        """Should preserve modality and phrase count."""
        extrapolator = SentenceExtrapolator()
        result = extrapolator.extrapolate_sentence(neutral_sentence, urgent_context)

        # Same number of phrases
        assert len(result.phrases) == len(neutral_sentence.phrases)

        # Same modality
        assert result.modality == neutral_sentence.modality

        # Same number of pauses
        assert len(result.pauses_ms) == len(neutral_sentence.pauses_ms)

    def test_extrapolate_with_progressive_intensity(self, neutral_sentence):
        """Should apply context with increasing intensity across phrases."""
        progressive_context = GRAMMAR_CONTEXT(
            name="progressive",
            phrase_f0_multiplier=1.0,  # Base
            phrase_duration_ratio=1.0,
            phrase_pause_ratio=0.5,
            f0_range_multiplier=1.0,
            harmonicity_delta=0.0,
            spectral_flatness_delta=0.0,
            jitter_add=0.02,  # Slight jitter increase per phrase
            shimmer_add=0.01,
            progressive_intensity=True  # Enable ramp
        )

        extrapolator = SentenceExtrapolator()
        result = extrapolator.extrapolate_sentence(neutral_sentence, progressive_context)

        # Jitter should increase progressively
        # Phrase 1: 0.0 + 0.02 * 1 = 0.02
        # Phrase 2: 0.01 + 0.02 * 2 = 0.05
        assert result.phrases[1].jitter > result.phrases[0].jitter


# ============================================================================
# TEST CLASS: AcousticAlgebraEngine
# ============================================================================

class TestAcousticAlgebraEngine:
    """Test the main acoustic algebra orchestrator."""

    def test_identity_operation(self, neutral_phrase_vector):
        """Identity: Return the phrase unchanged."""
        engine = AcousticAlgebraEngine()
        result = engine.identity(neutral_phrase_vector)

        assert result.f0_hz == neutral_phrase_vector.f0_hz
        assert result.duration_ms == neutral_phrase_vector.duration_ms

    def test_addition_operation(self, neutral_phrase_vector, aggression_context):
        """Addition: Phrase + Context Vector."""
        engine = AcousticAlgebraEngine()
        result = engine.add(neutral_phrase_vector, aggression_context)

        # Should match context application
        expected = aggression_context.apply(neutral_phrase_vector)
        assert abs(result.f0_hz - expected.f0_hz) < 0.1

    def test_subtraction_operation(self, neutral_phrase_vector, excited_phrase_vector):
        """Subtraction: Phrase A - Phrase B (feature delta)."""
        engine = AcousticAlgebraEngine()
        result = engine.subtract(neutral_phrase_vector, excited_phrase_vector)

        # F0 difference: 6000 - 7000 = -1000
        assert result.f0_hz == -1000.0

        # Duration difference: 50 - 30 = 20
        assert result.duration_ms == 20.0

    def test_scalar_multiply(self, neutral_phrase_vector):
        """Scalar Multiplication: Phrase * Intensity."""
        engine = AcousticAlgebraEngine()
        result = engine.multiply(neutral_phrase_vector, 1.5)

        assert result.f0_hz == 9000.0  # 6000 * 1.5
        assert result.duration_ms == 75.0  # 50 * 1.5

    def test_average_operation(self, neutral_phrase_vector, excited_phrase_vector):
        """Average: (Phrase A + Phrase B) / 2."""
        engine = AcousticAlgebraEngine()
        result = engine.average(neutral_phrase_vector, excited_phrase_vector)

        # Should equal 50% interpolation
        assert abs(result.f0_hz - 6500.0) < 0.1
        assert abs(result.duration_ms - 40.0) < 0.1

    def test_composition_operation(self, neutral_sentence, urgent_context):
        """Composition: Sentence + Context (grammar warp)."""
        engine = AcousticAlgebraEngine()
        result = engine.compose(neutral_sentence, urgent_context)

        # Should extrapolate sentence
        assert isinstance(result, SENTENCE_TEMPLATE)
        assert len(result.phrases) == len(neutral_sentence.phrases)


# ============================================================================
# TEST CLASS: SemanticContinuitySimulation
# ============================================================================

class TestSemanticContinuitySimulation:
    """Test semantic continuity through gradual transformation."""

    def test_neutral_to_aggressive_continuum(self, neutral_phrase_vector, aggression_context):
        """Should generate continuum from neutral to aggressive."""
        engine = AcousticAlgebraEngine()

        # Generate 10 steps from neutral to aggressive
        continuum = engine.generate_continuum(
            neutral_phrase_vector,
            aggression_context,
            num_steps=10
        )

        assert len(continuum) == 10

        # First step should be close to neutral
        assert abs(continuum[0].f0_hz - neutral_phrase_vector.f0_hz) < 100

        # Last step should show full aggression effect
        # F0: 6000 * 1.2 = 7200
        assert abs(continuum[-1].f0_hz - 7200.0) < 100

        # Should be monotonic increase in F0
        for i in range(len(continuum) - 1):
            assert continuum[i].f0_hz < continuum[i + 1].f0_hz

    def test_extrapolation_beyond_known(self, neutral_phrase_vector):
        """Should extrapolate beyond known dataset range."""
        engine = AcousticAlgebraEngine()

        extreme_context = ContextVector(
            name="extreme_beyond_data",
            f0_multiplier=3.0,  # 3x pitch (far outside training range)
            duration_ratio=0.3,  # 30% duration
            f0_range_multiplier=5.0,
            harmonicity_delta=-0.5,
            spectral_flatness_delta=0.8,
            jitter_add=0.5,
            shimmer_add=0.3
        )

        result = engine.extrapolate(neutral_phrase_vector, extreme_context)

        # Should generate values never seen in training
        assert result.f0_hz == 18000.0  # 6000 * 3
        assert result.duration_ms == 15.0  # 50 * 0.3

        # But should still be physically valid
        assert result.f0_hz > 0
        assert result.duration_ms > 0

    def test_categorical_perception_boundary(self, neutral_phrase_vector, excited_phrase_vector):
        """Test interpolation to find perceptual boundary."""
        engine = AcousticAlgebraEngine()

        # Generate 100 interpolated steps
        continuum = engine.generate_interpolation_continuum(
            neutral_phrase_vector,
            excited_phrase_vector,
            num_steps=100
        )

        assert len(continuum) == 100

        # Should find approximate midpoint at index 50
        # With 100 steps (indices 0-99), index 50 has alpha = 50/99 ≈ 0.505
        midpoint = continuum[50]
        # Expected F0 at index 50: 6000 + (7000-6000) * 50/99 ≈ 6505 Hz
        expected_f0 = 6000.0 + 1000.0 * 50.0 / 99.0
        assert abs(midpoint.f0_hz - expected_f0) < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
