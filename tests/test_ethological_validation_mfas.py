#!/usr/bin/env python3
"""
Test Suite for Ethological Validation Module

Comprehensive tests for the Multi-Factor Acceptance Score (MFAS) system,
including temporal gating, acoustic convergence, prosodic DTW, and
fused scoring.

Test Categories:
1. Temporal Gating Tests - Species-specific timing constraints
2. Acoustic Convergence Tests - Dialect matching metrics
3. Prosodic DTW Tests - Temporal prosody comparison
4. MFAS Integration Tests - Fused scoring system
5. Cross-Species Tests - Multi-species validation

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch

from ethological_validation import (
    # Taxa Profiles
    TaxaTemporalProfile,
    TemporalGate,
    get_temporal_gate,
    SPECIES_PROFILES,
    create_custom_profile,
    analyze_corpus_latencies,
    # Acoustic Convergence
    AcousticConvergenceEngine,
    ConvergenceResult,
    MultiDimensionalConvergence,
    compute_convergence_from_affect_vectors,
    compute_batch_convergence,
    # Prosodic DTW
    FastDTW,
    ProsodicDTW,
    ProsodicFeature,
    ProsodicFeatureExtractor,
    # MFAS
    InteractionEvent,
    MFASResult,
    MultiFactorAcceptanceScore,
    MFASComparator,
    create_mfas_for_species,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_baseline_contours():
    """Sample F0 contours for prosodic DTW baselines."""
    np.random.seed(42)
    return [
        np.linspace(5000, 7000, 50) + np.random.randn(50) * 100,
        np.ones(50) * 6000 + np.random.randn(50) * 50,
        6000 + 1000 * np.sin(np.linspace(0, 2 * np.pi, 50)),
    ]


@pytest.fixture
def sample_16d_vectors():
    """Sample 16D affect vectors for testing."""
    np.random.seed(42)
    return {
        "neutral": np.zeros(16),
        "high_arousal": np.ones(16) * 0.8,
        "medium_arousal": np.ones(16) * 0.5,
        "low_arousal": np.ones(16) * 0.2,
    }


@pytest.fixture
def sample_interaction_events(sample_16d_vectors):
    """Sample interaction events for batch testing."""
    events = []

    # High acceptance event (valid timing, convergence)
    events.append(InteractionEvent(
        species="rousettus_aegyptiacus",
        ai_output_state=sample_16d_vectors["medium_arousal"],
        animal_pre_state=sample_16d_vectors["neutral"],
        animal_post_state=sample_16d_vectors["medium_arousal"] * 0.9,
        animal_f0_contour=np.linspace(5000, 7000, 45),
        ai_end_time_ms=1000,
        animal_response_time_ms=1090,
    ))

    # Low acceptance event (divergence)
    events.append(InteractionEvent(
        species="rousettus_aegyptiacus",
        ai_output_state=sample_16d_vectors["medium_arousal"],
        animal_pre_state=sample_16d_vectors["neutral"],
        animal_post_state=sample_16d_vectors["neutral"],  # No convergence
        animal_f0_contour=np.concatenate([
            np.ones(20) * 9000,
            np.zeros(15),
            np.ones(10) * 9000,
        ]),
        ai_end_time_ms=1000,
        animal_response_time_ms=1090,
    ))

    # Invalid timing event
    events.append(InteractionEvent(
        species="rousettus_aegyptiacus",
        ai_output_state=sample_16d_vectors["medium_arousal"],
        animal_pre_state=sample_16d_vectors["neutral"],
        animal_post_state=sample_16d_vectors["medium_arousal"] * 0.9,
        animal_f0_contour=np.linspace(5000, 7000, 45),
        ai_end_time_ms=1000,
        animal_response_time_ms=2000,  # Invalid: >150ms
    ))

    return events


# =============================================================================
# Test Category 1: Temporal Gating Tests
# =============================================================================

class TestTaxaTemporalProfile:
    """Tests for TaxaTemporalProfile dataclass."""

    def test_profile_creation(self):
        """Test creating a valid temporal profile."""
        profile = TaxaTemporalProfile(
            species_name="Test Species",
            min_response_ms=50,
            max_response_ms=500,
            debounce_ms=20,
        )
        assert profile.species_name == "Test Species"
        assert profile.min_response_ms == 50
        assert profile.max_response_ms == 500
        assert profile.debounce_ms == 20

    def test_profile_validation_invalid_bounds(self):
        """Test that invalid bounds raise ValueError."""
        with pytest.raises(ValueError, match="min_response_ms.*must be < max_response_ms"):
            TaxaTemporalProfile(
                species_name="Invalid",
                min_response_ms=500,
                max_response_ms=100,
                debounce_ms=20,
            )

    def test_profile_validation_negative_debounce(self):
        """Test that negative debounce raises ValueError."""
        with pytest.raises(ValueError, match="debounce_ms cannot be negative"):
            TaxaTemporalProfile(
                species_name="Invalid",
                min_response_ms=50,
                max_response_ms=500,
                debounce_ms=-10,
            )


class TestSpeciesProfiles:
    """Tests for predefined species profiles."""

    def test_bat_profile_exists(self):
        """Test that Egyptian Fruit Bat profile exists."""
        assert "rousettus_aegyptiacus" in SPECIES_PROFILES
        profile = SPECIES_PROFILES["rousettus_aegyptiacus"]
        assert profile.min_response_ms == 30
        assert profile.max_response_ms == 150
        assert profile.species_name == "Egyptian Fruit Bat"

    def test_marmoset_profile_exists(self):
        """Test that Marmoset profile exists."""
        assert "callithrix_jacchus" in SPECIES_PROFILES
        profile = SPECIES_PROFILES["callithrix_jacchus"]
        assert profile.min_response_ms == 50
        assert profile.max_response_ms == 800

    def test_all_profiles_valid(self):
        """Test that all predefined profiles are valid."""
        for key, profile in SPECIES_PROFILES.items():
            assert profile.min_response_ms < profile.max_response_ms
            assert profile.debounce_ms >= 0


class TestTemporalGate:
    """Tests for TemporalGate class."""

    def test_gate_creation_known_species(self):
        """Test creating gate for known species."""
        gate = get_temporal_gate("rousettus_aegyptiacus")
        assert gate.species == "rousettus_aegyptiacus"
        assert gate.profile.species_name == "Egyptian Fruit Bat"

    def test_gate_creation_unknown_species(self):
        """Test that unknown species raises ValueError."""
        with pytest.raises(ValueError, match="Unknown species profile"):
            get_temporal_gate("unknown_species")

    def test_case_insensitive_lookup(self):
        """Test case-insensitive species lookup."""
        gate1 = get_temporal_gate("Rousettus_Aegyptiacus")
        gate2 = get_temporal_gate("egyptian fruit bat")
        assert gate1.profile.species_name == gate2.profile.species_name

    def test_is_valid_response_within_window(self):
        """Test validation of response within valid window."""
        gate = get_temporal_gate("rousettus_aegyptiacus")
        # 90ms is within 30-150ms window
        assert gate.is_valid_response(1000, 1090) is True

    def test_is_valid_response_below_min(self):
        """Test rejection of response below minimum."""
        gate = get_temporal_gate("rousettus_aegyptiacus")
        # 10ms is below 30ms minimum
        assert gate.is_valid_response(1000, 1010) is False

    def test_is_valid_response_above_max(self):
        """Test rejection of response above maximum."""
        gate = get_temporal_gate("rousettus_aegyptiacus")
        # 200ms is above 150ms maximum
        assert gate.is_valid_response(1000, 1200) is False

    def test_is_rapid_turn(self):
        """Test rapid turn detection."""
        gate = get_temporal_gate("rousettus_aegyptiacus")
        # 40ms is below 50ms threshold
        assert gate.is_rapid_turn(40) is True
        assert gate.is_rapid_turn(100) is False

    def test_get_latency_score_optimal(self):
        """Test latency score at optimal timing."""
        gate = get_temporal_gate("rousettus_aegyptiacus")
        # Optimal is (30+150)/2 = 90ms
        score = gate.get_latency_score(90)
        assert score > 0.9  # Near maximum

    def test_get_latency_score_boundary(self):
        """Test latency score at window boundary."""
        gate = get_temporal_gate("rousettus_aegyptiacus")
        # At boundary (150ms), score should approach 0
        score = gate.get_latency_score(150)
        assert score > 0  # Still valid, just lower score

    def test_get_latency_score_invalid(self):
        """Test latency score for invalid timing."""
        gate = get_temporal_gate("rousettus_aegyptiacus")
        score = gate.get_latency_score(200)  # Above max
        assert score == 0.0


class TestAnalyzeCorpusLatencies:
    """Tests for corpus latency analysis."""

    def test_empty_corpus(self):
        """Test analysis with empty corpus."""
        stats = analyze_corpus_latencies("rousettus_aegyptiacus", [])
        assert stats["count"] == 0
        assert stats["mean_latency_ms"] == 0

    def test_valid_and_invalid_responses(self):
        """Test analysis with mix of valid/invalid responses."""
        log = [
            {"ai_end_ms": 1000, "animal_start_ms": 1090},  # Valid
            {"ai_end_ms": 2000, "animal_start_ms": 2100},  # Valid
            {"ai_end_ms": 3000, "animal_start_ms": 3500},  # Invalid (500ms)
        ]
        stats = analyze_corpus_latencies("rousettus_aegyptiacus", log)
        assert stats["count"] == 3
        assert stats["valid_rate"] == 2/3


# =============================================================================
# Test Category 2: Acoustic Convergence Tests
# =============================================================================

class TestAcousticConvergenceEngine:
    """Tests for AcousticConvergenceEngine."""

    def test_cosine_distance_computation(self):
        """Test cosine distance calculation."""
        engine = AcousticConvergenceEngine(distance_metric='cosine')
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        # Orthogonal vectors -> distance = 1
        dist = engine._compute_distance(v1, v2)
        assert abs(dist - 1.0) < 0.01

    def test_euclidean_distance_computation(self):
        """Test Euclidean distance calculation."""
        engine = AcousticConvergenceEngine(distance_metric='euclidean')
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        # 3-4-5 triangle
        dist = engine._compute_distance(v1, v2)
        assert abs(dist - 5.0) < 0.01

    def test_convergence_toward_ai(self):
        """Test convergence when animal moves toward AI."""
        engine = AcousticConvergenceEngine(distance_metric='cosine')
        # Pre is orthogonal to AI (distance ~1)
        pre = np.array([1.0, 0.0] + [0.0] * 14)
        # AI is in a different direction
        ai = np.array([0.0, 1.0] + [0.0] * 14)
        # Post moves closer to AI's direction
        post = np.array([0.3, 0.9] + [0.0] * 14)

        result = engine.calculate_convergence(pre, ai, post)
        assert result.direction == "toward"
        assert result.convergence_score > 0.5

    def test_convergence_away_from_ai(self):
        """Test divergence when animal moves away from AI."""
        engine = AcousticConvergenceEngine(distance_metric='cosine')
        # Pre is close to AI's direction
        pre = np.array([0.1, 0.9] + [0.0] * 14)
        # AI target direction
        ai = np.array([0.0, 1.0] + [0.0] * 14)
        # Post moves away from AI
        post = np.array([1.0, 0.0] + [0.0] * 14)

        result = engine.calculate_convergence(pre, ai, post)
        assert result.direction == "away"
        assert result.convergence_score < 0.5

    def test_convergence_neutral(self):
        """Test neutral convergence when no change."""
        engine = AcousticConvergenceEngine(distance_metric='cosine')
        state = np.ones(16) * 0.5

        result = engine.calculate_convergence(state, state, state)
        # No movement in any direction
        assert result.convergence_score > 0.4  # Baseline similarity


class TestMultiDimensionalConvergence:
    """Tests for MultiDimensionalConvergence."""

    def test_dimensional_extraction(self):
        """Test extraction of specific dimensions."""
        mdc = MultiDimensionalConvergence()
        state_112d = np.random.randn(112)

        results = mdc.calculate_dimensional_convergence(
            state_112d, state_112d * 1.1, state_112d * 0.9
        )

        assert "f0" in results
        assert "harmonics" in results
        assert "noise" in results
        assert "affect" in results


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compute_convergence_from_affect_vectors(self):
        """Test simplified convergence calculation."""
        pre = np.zeros(16)
        ai = np.ones(16) * 0.5
        post = np.ones(16) * 0.4

        score = compute_convergence_from_affect_vectors(pre, ai, post)
        assert 0 <= score <= 1

    def test_compute_batch_convergence(self):
        """Test batch convergence calculation."""
        interactions = [
            {
                "animal_pre": np.zeros(16),
                "ai_output": np.ones(16) * 0.5,
                "animal_post": np.ones(16) * 0.4,
            }
        ]

        stats = compute_batch_convergence(interactions)
        assert stats["count"] == 1
        assert "mean_score" in stats


# =============================================================================
# Test Category 3: Prosodic DTW Tests
# =============================================================================

class TestFastDTW:
    """Tests for FastDTW implementation."""

    def test_identical_sequences(self):
        """Test DTW on identical sequences."""
        dtw = FastDTW()
        seq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = dtw.compute_distance(seq, seq)
        # Identical sequences should have zero distance
        assert dist < 0.01

    def test_similar_sequences(self):
        """Test DTW on similar sequences."""
        dtw = FastDTW()
        seq1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        seq2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        dist = dtw.compute_distance(seq1, seq2)
        # Similar sequences should have small distance
        assert dist < 5.0

    def test_dissimilar_sequences(self):
        """Test DTW on dissimilar sequences."""
        dtw = FastDTW()
        seq1 = np.array([1.0, 2.0, 3.0])
        seq2 = np.array([100.0, 200.0, 300.0])
        dist = dtw.compute_distance(seq1, seq2)
        # Dissimilar sequences should have large distance
        assert dist > 1000

    def test_compute_distance_with_path(self):
        """Test DTW with path reconstruction."""
        dtw = FastDTW()
        seq1 = np.array([1.0, 2.0, 3.0])
        seq2 = np.array([1.0, 2.0, 3.0])

        dist, path = dtw.compute_distance_with_path(seq1, seq2)
        assert dist < 0.01
        assert len(path) > 0
        # Path should contain indices
        assert path.shape[1] == 2  # (i, j) pairs


class TestProsodicDTW:
    """Tests for ProsodicDTW."""

    def test_score_response_no_baselines(self):
        """Test scoring with no baselines."""
        dtw = ProsodicDTW(baseline_contours=[])
        result = dtw.score_response(np.array([1.0, 2.0, 3.0]))
        # Should return default score
        assert result.similarity_score == 0.5
        assert result.best_match_idx == -1

    def test_score_response_with_baselines(self, sample_baseline_contours):
        """Test scoring with baselines."""
        dtw = ProsodicDTW(baseline_contours=sample_baseline_contours)
        # Test with a contour similar to baseline 0
        test_contour = np.linspace(5000, 7000, 45)
        result = dtw.score_response(test_contour)
        assert 0 <= result.similarity_score <= 1
        assert result.best_match_idx >= 0

    def test_add_baseline(self):
        """Test adding baselines dynamically."""
        dtw = ProsodicDTW(baseline_contours=[])
        assert len(dtw.baselines) == 0

        dtw.add_baseline(np.array([1.0, 2.0, 3.0]))
        assert len(dtw.baselines) == 1

    def test_set_baselines(self):
        """Test replacing all baselines."""
        dtw = ProsodicDTW(baseline_contours=[np.array([1.0])])
        dtw.set_baselines([np.array([2.0]), np.array([3.0])])
        assert len(dtw.baselines) == 2


class TestProsodicFeatureExtractor:
    """Tests for ProsodicFeatureExtractor."""

    def test_extract_amplitude_envelope(self):
        """Test amplitude envelope extraction."""
        extractor = ProsodicFeatureExtractor(sample_rate=48000)
        # 100ms of audio at 48kHz
        audio = np.random.randn(4800).astype(np.float32) * 0.1
        envelope = extractor._extract_amplitude_envelope(audio)
        assert len(envelope) > 0
        assert all(np.isfinite(envelope))

    def test_extract_f0_contour(self):
        """Test F0 contour extraction."""
        extractor = ProsodicFeatureExtractor(sample_rate=48000)
        # Generate 100ms of 9kHz tone
        t = np.linspace(0, 0.1, 4800)
        audio = np.sin(2 * np.pi * 9000 * t).astype(np.float32)
        f0 = extractor._extract_f0_contour(audio)
        assert len(f0) > 0
        # Most frames should detect ~9kHz F0
        detected_f0 = f0[f0 > 0]
        if len(detected_f0) > 0:
            assert np.mean(detected_f0) > 8000  # Should detect high pitch

    def test_extract_spectral_centroid(self):
        """Test spectral centroid extraction."""
        extractor = ProsodicFeatureExtractor(sample_rate=48000)
        audio = np.random.randn(4800).astype(np.float32)
        centroid = extractor._extract_spectral_centroid(audio)
        assert len(centroid) > 0
        assert all(np.isfinite(centroid))


# =============================================================================
# Test Category 4: MFAS Integration Tests
# =============================================================================

class TestInteractionEvent:
    """Tests for InteractionEvent dataclass."""

    def test_event_creation(self, sample_16d_vectors):
        """Test creating an interaction event."""
        event = InteractionEvent(
            species="rousettus_aegyptiacus",
            ai_output_state=sample_16d_vectors["medium_arousal"],
            animal_pre_state=sample_16d_vectors["neutral"],
            animal_post_state=sample_16d_vectors["medium_arousal"] * 0.9,
            animal_f0_contour=np.array([1000, 2000, 3000]),
            ai_end_time_ms=1000,
            animal_response_time_ms=1090,
        )
        assert event.species == "rousettus_aegyptiacus"
        assert event.ai_end_time_ms == 1000
        assert event.animal_response_time_ms == 1090


class TestMultiFactorAcceptanceScore:
    """Tests for MultiFactorAcceptanceScore."""

    @pytest.fixture
    def mfas_calculator(self, sample_baseline_contours):
        """Create MFAS calculator for testing."""
        gate = get_temporal_gate("rousettus_aegyptiacus")
        convergence = AcousticConvergenceEngine(distance_metric='cosine')
        dtw = ProsodicDTW(baseline_contours=sample_baseline_contours)
        return MultiFactorAcceptanceScore(
            temporal_gate=gate,
            convergence_engine=convergence,
            dtw_engine=dtw,
            w_convergence=0.4,
            w_prosody=0.6,
        )

    def test_weight_normalization(self, sample_baseline_contours):
        """Test that weights are normalized."""
        gate = get_temporal_gate("rousettus_aegyptiacus")
        convergence = AcousticConvergenceEngine()
        dtw = ProsodicDTW(baseline_contours=sample_baseline_contours)

        # Weights that don't sum to 1
        mfas = MultiFactorAcceptanceScore(
            temporal_gate=gate,
            convergence_engine=convergence,
            dtw_engine=dtw,
            w_convergence=0.7,
            w_prosody=0.7,
        )
        # Should be normalized
        assert np.isclose(mfas.w_convergence + mfas.w_prosody, 1.0)

    def test_evaluate_high_acceptance(self, mfas_calculator, sample_16d_vectors):
        """Test evaluation of high-acceptance interaction."""
        event = InteractionEvent(
            species="rousettus_aegyptiacus",
            ai_output_state=sample_16d_vectors["medium_arousal"],
            animal_pre_state=sample_16d_vectors["neutral"],
            animal_post_state=sample_16d_vectors["medium_arousal"] * 0.9,
            animal_f0_contour=np.linspace(5000, 7000, 45),
            ai_end_time_ms=1000,
            animal_response_time_ms=1090,
        )

        result = mfas_calculator.evaluate_interaction(event)
        assert result.temporal_valid is True
        assert result.mfas_score > 0
        assert result.convergence_result.direction in ["toward", "neutral"]

    def test_evaluate_invalid_timing(self, mfas_calculator, sample_16d_vectors):
        """Test evaluation of interaction with invalid timing."""
        event = InteractionEvent(
            species="rousettus_aegyptiacus",
            ai_output_state=sample_16d_vectors["medium_arousal"],
            animal_pre_state=sample_16d_vectors["neutral"],
            animal_post_state=sample_16d_vectors["medium_arousal"] * 0.9,
            animal_f0_contour=np.linspace(5000, 7000, 45),
            ai_end_time_ms=1000,
            animal_response_time_ms=2000,  # Invalid: >150ms
        )

        result = mfas_calculator.evaluate_interaction(event)
        assert result.temporal_valid is False
        assert result.mfas_score == 0.0
        assert result.rejected_reason is not None

    def test_evaluate_acoustic_divergence(self, mfas_calculator, sample_16d_vectors):
        """Test evaluation with acoustic divergence."""
        # Create vectors that show clear divergence
        ai_state = np.array([0.0, 1.0] + [0.0] * 14)  # AI direction
        pre_state = np.array([0.1, 0.9] + [0.0] * 14)  # Close to AI
        post_state = np.array([1.0, 0.0] + [0.0] * 14)  # Moved away

        event = InteractionEvent(
            species="rousettus_aegyptiacus",
            ai_output_state=ai_state,
            animal_pre_state=pre_state,
            animal_post_state=post_state,  # Divergence
            animal_f0_contour=np.linspace(5000, 7000, 45),
            ai_end_time_ms=1000,
            animal_response_time_ms=1090,
        )

        result = mfas_calculator.evaluate_interaction(event)
        # Should have lower score due to divergence
        assert result.convergence_result.direction == "away"

    def test_evaluate_batch(self, mfas_calculator, sample_interaction_events):
        """Test batch evaluation."""
        stats = mfas_calculator.evaluate_batch(sample_interaction_events)
        assert stats["count"] == 3
        assert 0 <= stats["mean_mfas"] <= 1
        assert stats["valid_rate"] == 2/3  # 2 valid, 1 invalid


class TestMFASComparator:
    """Tests for MFASComparator."""

    @pytest.fixture
    def comparator(self, sample_baseline_contours):
        """Create comparator for testing."""
        mfas = create_mfas_for_species(
            "rousettus_aegyptiacus",
            baseline_contours=sample_baseline_contours,
        )
        return MFASComparator(mfas)

    def test_compare_conditions(self, comparator, sample_interaction_events):
        """Test comparing two conditions."""
        # Split events into two conditions
        mid = len(sample_interaction_events) // 2
        condition_a = sample_interaction_events[:mid]
        condition_b = sample_interaction_events[mid:]

        result = comparator.compare_conditions(
            condition_a,
            condition_b,
            "Condition A",
            "Condition B",
        )

        assert "Condition A" in result
        assert "Condition B" in result
        assert "comparison" in result
        assert "p_value" in result["comparison"]


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_mfas_for_species(self, sample_baseline_contours):
        """Test factory function for MFAS creation."""
        mfas = create_mfas_for_species(
            "rousettus_aegyptiacus",
            baseline_contours=sample_baseline_contours,
        )
        assert isinstance(mfas, MultiFactorAcceptanceScore)
        assert mfas.gate.species == "rousettus_aegyptiacus"


# =============================================================================
# Test Category 5: Cross-Species Tests
# =============================================================================

class TestCrossSpeciesValidation:
    """Tests for multi-species validation."""

    @pytest.mark.parametrize("species,valid_latencies,invalid_latencies", [
        ("rousettus_aegyptiacus", [50, 100, 140], [10, 20, 200, 500]),
        ("callithrix_jacchus", [100, 400, 700], [20, 30, 1000, 2000]),
        ("taeniopygia_guttata", [100, 250, 450], [50, 600, 800]),
    ])
    def test_species_specific_windows(self, species, valid_latencies, invalid_latencies):
        """Test that different species have different valid windows."""
        gate = get_temporal_gate(species)

        for latency in valid_latencies:
            assert gate.is_valid_response(0, latency), \
                f"{species}: {latency}ms should be valid"

        for latency in invalid_latencies:
            assert not gate.is_valid_response(0, latency), \
                f"{species}: {latency}ms should be invalid"

    def test_custom_profile_creation(self):
        """Test creating custom species profile."""
        profile = create_custom_profile(
            "Test Species",
            min_response_ms=100,
            max_response_ms=1000,
        )
        assert profile.species_name == "Test Species"
        assert profile in SPECIES_PROFILES.values()


# =============================================================================
# Integration Tests
# =============================================================================

class TestFullPipelineIntegration:
    """Integration tests for the complete MFAS pipeline."""

    @pytest.fixture
    def full_pipeline(self, sample_baseline_contours):
        """Create complete MFAS pipeline."""
        return create_mfas_for_species(
            "rousettus_aegyptiacus",
            baseline_contours=sample_baseline_contours,
        )

    def test_acceptance_rejection_classification(
        self, full_pipeline, sample_16d_vectors
    ):
        """Test correct classification of acceptance vs rejection."""
        # Create high-acceptance event
        accept_event = InteractionEvent(
            species="rousettus_aegyptiacus",
            ai_output_state=sample_16d_vectors["medium_arousal"],
            animal_pre_state=sample_16d_vectors["neutral"],
            animal_post_state=sample_16d_vectors["medium_arousal"] * 0.9,
            animal_f0_contour=np.linspace(5000, 7000, 45),
            ai_end_time_ms=1000,
            animal_response_time_ms=1090,
        )

        # Create rejection event (invalid timing)
        reject_event = InteractionEvent(
            species="rousettus_aegyptiacus",
            ai_output_state=sample_16d_vectors["medium_arousal"],
            animal_pre_state=sample_16d_vectors["neutral"],
            animal_post_state=sample_16d_vectors["medium_arousal"] * 0.9,
            animal_f0_contour=np.linspace(5000, 7000, 45),
            ai_end_time_ms=1000,
            animal_response_time_ms=2000,
        )

        accept_result = full_pipeline.evaluate_interaction(accept_event)
        reject_result = full_pipeline.evaluate_interaction(reject_event)

        # Acceptance should have higher score than rejection
        assert accept_result.mfas_score > reject_result.mfas_score

    def test_score_bounds(self, full_pipeline, sample_16d_vectors):
        """Test that MFAS scores are always in [0, 1]."""
        for _ in range(10):
            event = InteractionEvent(
                species="rousettus_aegyptiacus",
                ai_output_state=np.random.randn(16),
                animal_pre_state=np.random.randn(16),
                animal_post_state=np.random.randn(16),
                animal_f0_contour=np.random.randn(50) * 1000 + 6000,
                ai_end_time_ms=1000,
                animal_response_time_ms=np.random.randint(50, 140),
            )
            result = full_pipeline.evaluate_interaction(event)
            assert 0 <= result.mfas_score <= 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
