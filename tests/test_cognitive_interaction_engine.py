"""
TDD Tests for Cognitive Interaction Engine (Phase 1: The "Brain")

This test suite validates the cognitive layer that:
1. Calculates virtual targets from intents (Acoustic Algebra)
2. Finds nearest neighbors (Phrase Database)
3. Applies safety clamping (Delta Clamping)
4. Converts deltas to Rust synthesis parameters

Architecture: Python Logic Layer → Rust Execution Layer

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
from dataclasses import dataclass
from unittest.mock import Mock

# =============================================================================
# Mock Dependencies
# =============================================================================


@dataclass
class Vector17D:
    """29D acoustic vector (expanded from 17D/20D, matches Rust implementation)"""

    mean_f0_hz: float
    duration_ms: float
    f0_range_hz: float
    harmonic_to_noise_ratio: float
    spectral_flatness: float
    harmonicity: float  # NEW
    attack_time_ms: float
    decay_time_ms: float
    sustain_level: float
    vibrato_rate_hz: float
    vibrato_depth: float
    jitter: float
    shimmer: float  # NEW
    mfcc_1: float
    mfcc_2: float
    mfcc_3: float
    mfcc_4: float
    mfcc_5: float = 0.0  # NEW
    mfcc_6: float = 0.0  # NEW
    mfcc_7: float = 0.0  # NEW
    mfcc_8: float = 0.0  # NEW
    mfcc_9: float = 0.0  # NEW
    mfcc_10: float = 0.0  # NEW
    mfcc_11: float = 0.0  # NEW
    mfcc_12: float = 0.0  # NEW
    mfcc_13: float = 0.0  # NEW
    spectral_contrast: float = 0.0
    spectral_flux: float = 0.0  # NEW
    median_ici_ms: float = 0.0
    onset_rate_hz: float = 0.0
    ici_coefficient_of_variation: float = 0.0

    def __sub__(self, other):
        """Calculate delta between two vectors"""
        return VectorDelta(
            delta_mean_f0_hz=self.mean_f0_hz - other.mean_f0_hz,
            delta_duration_ms=self.duration_ms - other.duration_ms,
            delta_f0_range_hz=self.f0_range_hz - other.f0_range_hz,
            delta_harmonic_to_noise_ratio=self.harmonic_to_noise_ratio
            - other.harmonic_to_noise_ratio,
            delta_spectral_flatness=self.spectral_flatness - other.spectral_flatness,
            delta_harmonicity=self.harmonicity - other.harmonicity,
            delta_attack_time_ms=self.attack_time_ms - other.attack_time_ms,
            delta_decay_time_ms=self.decay_time_ms - other.decay_time_ms,
            delta_sustain_level=self.sustain_level - other.sustain_level,
            delta_vibrato_rate_hz=self.vibrato_rate_hz - other.vibrato_rate_hz,
            delta_vibrato_depth=self.vibrato_depth - other.vibrato_depth,
            delta_jitter=self.jitter - other.jitter,
            delta_shimmer=self.shimmer - other.shimmer,
            delta_mfcc_1=self.mfcc_1 - other.mfcc_1,
            delta_mfcc_2=self.mfcc_2 - other.mfcc_2,
            delta_mfcc_3=self.mfcc_3 - other.mfcc_3,
            delta_mfcc_4=self.mfcc_4 - other.mfcc_4,
            delta_mfcc_5=self.mfcc_5 - other.mfcc_5,  # NEW
            delta_mfcc_6=self.mfcc_6 - other.mfcc_6,  # NEW
            delta_mfcc_7=self.mfcc_7 - other.mfcc_7,  # NEW
            delta_mfcc_8=self.mfcc_8 - other.mfcc_8,  # NEW
            delta_mfcc_9=self.mfcc_9 - other.mfcc_9,  # NEW
            delta_mfcc_10=self.mfcc_10 - other.mfcc_10,  # NEW
            delta_mfcc_11=self.mfcc_11 - other.mfcc_11,  # NEW
            delta_mfcc_12=self.mfcc_12 - other.mfcc_12,  # NEW
            delta_mfcc_13=self.mfcc_13 - other.mfcc_13,  # NEW
            delta_spectral_contrast=self.spectral_contrast - other.spectral_contrast,
            delta_spectral_flux=self.spectral_flux - other.spectral_flux,
            delta_median_ici_ms=self.median_ici_ms - other.median_ici_ms,
            delta_onset_rate_hz=self.onset_rate_hz - other.onset_rate_hz,
            delta_ici_coefficient_of_variation=self.ici_coefficient_of_variation
            - other.ici_coefficient_of_variation,
        )


@dataclass
class VectorDelta:
    """Delta between two vectors"""

    delta_mean_f0_hz: float
    delta_duration_ms: float
    delta_f0_range_hz: float
    delta_harmonic_to_noise_ratio: float
    delta_spectral_flatness: float
    delta_harmonicity: float  # NEW
    delta_attack_time_ms: float
    delta_decay_time_ms: float
    delta_sustain_level: float
    delta_vibrato_rate_hz: float
    delta_vibrato_depth: float
    delta_jitter: float
    delta_shimmer: float  # NEW
    delta_mfcc_1: float
    delta_mfcc_2: float
    delta_mfcc_3: float
    delta_mfcc_4: float
    delta_mfcc_5: float = 0.0  # NEW
    delta_mfcc_6: float = 0.0  # NEW
    delta_mfcc_7: float = 0.0  # NEW
    delta_mfcc_8: float = 0.0  # NEW
    delta_mfcc_9: float = 0.0  # NEW
    delta_mfcc_10: float = 0.0  # NEW
    delta_mfcc_11: float = 0.0  # NEW
    delta_mfcc_12: float = 0.0  # NEW
    delta_mfcc_13: float = 0.0  # NEW
    delta_spectral_contrast: float = 0.0
    delta_spectral_flux: float = 0.0  # NEW
    delta_median_ici_ms: float = 0.0
    delta_onset_rate_hz: float = 0.0
    delta_ici_coefficient_of_variation: float = 0.0


@dataclass
class AudioPhrase:
    """Audio phrase from database"""

    key: str
    features: Vector17D
    species: str


# =============================================================================
# Test 1.1: Virtual Target Calculation
# =============================================================================


class TestVirtualTargetCalculation(unittest.TestCase):
    """Test 1.1: Engine calculates virtual target from intent"""

    def test_generate_response_calculates_virtual_target(self):
        """
        RED TEST: Engine should calculate virtual target using Acoustic Algebra

        Scenario: Request "Aggression" at 0.5 intensity
        Expected: Engine calls algebra.generate_graded_vector with correct parameters
        """
        # Arrange
        virtual_target = Vector17D(
            mean_f0_hz=7500.0,
            duration_ms=40.0,
            f0_range_hz=500.0,
            harmonic_to_noise_ratio=22.5,
            spectral_flatness=0.4,
            attack_time_ms=4.0,
            decay_time_ms=17.5,
            sustain_level=0.8,
            vibrato_rate_hz=8.5,
            vibrato_depth=0.035,
            jitter=0.02,
            mfcc_1=-9.0,
            mfcc_2=-4.0,
            mfcc_3=-1.5,
            mfcc_4=-0.5,
            spectral_contrast=25.0,
            harmonicity=0.75,
            shimmer=0.015,
            spectral_flux=1.5,
            median_ici_ms=14.0,
            onset_rate_hz=55.0,
            ici_coefficient_of_variation=0.32,
        )

        phrase = AudioPhrase(
            key="test_phrase",
            features=Vector17D(
                mean_f0_hz=7000.0,
                duration_ms=50.0,
                f0_range_hz=400.0,
                harmonic_to_noise_ratio=20.0,
                spectral_flatness=0.3,
                attack_time_ms=5.0,
                decay_time_ms=20.0,
                sustain_level=0.7,
                vibrato_rate_hz=7.0,
                vibrato_depth=0.02,
                jitter=0.01,
                mfcc_1=-10.0,
                mfcc_2=-5.0,
                mfcc_3=-2.0,
                mfcc_4=-1.0,
                spectral_contrast=20.0,
                harmonicity=0.75,
                shimmer=0.015,
                spectral_flux=1.5,
                median_ici_ms=15.0,
                onset_rate_hz=50.0,
                ici_coefficient_of_variation=0.3,
            ),
            species="marmoset",
        )

        mock_algebra = Mock()
        mock_algebra.generate_graded_vector = Mock(return_value=virtual_target)

        mock_db = Mock()
        mock_db.find_nearest = Mock(return_value=phrase)

        mock_synthesizer = Mock()

        # Import after defining mocks to avoid circular import
        from realtime.cognitive_interaction_engine import CognitiveInteractionEngine

        engine = CognitiveInteractionEngine(
            algebra_map=mock_algebra, phrase_db=mock_db, synthesizer=mock_synthesizer
        )

        # Act
        result = engine.generate_response(intent="aggression", intensity=0.5)

        # Assert
        mock_algebra.generate_graded_vector.assert_called_once_with(
            intent="aggression", intensity=0.5
        )
        self.assertIsNotNone(result)


# =============================================================================
# Test 1.2: Nearest Neighbor Lookup
# =============================================================================


class TestNearestNeighborLookup(unittest.TestCase):
    """Test 1.2: Engine finds nearest phrase in database"""

    def test_generate_response_finds_nearest_neighbor(self):
        """
        RED TEST: Engine should find nearest neighbor to virtual target

        Scenario: Virtual target is [0.5, 1.0, ...]
        Database has Phrase A [0.4, 1.0, ...] and Phrase B [0.9, 0.0, ...]
        Expected: Engine selects Phrase A (closest distance)
        """
        # Arrange
        virtual_target = Vector17D(
            mean_f0_hz=7500.0,
            duration_ms=40.0,
            f0_range_hz=500.0,
            harmonic_to_noise_ratio=22.5,
            spectral_flatness=0.4,
            attack_time_ms=4.0,
            decay_time_ms=17.5,
            sustain_level=0.8,
            vibrato_rate_hz=8.5,
            vibrato_depth=0.035,
            jitter=0.02,
            mfcc_1=-9.0,
            mfcc_2=-4.0,
            mfcc_3=-1.5,
            mfcc_4=-0.5,
            spectral_contrast=25.0,
            harmonicity=0.75,
            shimmer=0.015,
            spectral_flux=1.5,
            median_ici_ms=14.0,
            onset_rate_hz=55.0,
            ici_coefficient_of_variation=0.32,
        )

        phrase_a = AudioPhrase(
            key="neutral",
            features=Vector17D(
                mean_f0_hz=7000.0,
                duration_ms=50.0,
                f0_range_hz=400.0,
                harmonic_to_noise_ratio=20.0,
                spectral_flatness=0.3,
                attack_time_ms=5.0,
                decay_time_ms=20.0,
                sustain_level=0.7,
                vibrato_rate_hz=7.0,
                vibrato_depth=0.02,
                jitter=0.01,
                mfcc_1=-10.0,
                mfcc_2=-5.0,
                mfcc_3=-2.0,
                mfcc_4=-1.0,
                spectral_contrast=20.0,
                harmonicity=0.75,
                shimmer=0.015,
                spectral_flux=1.5,
                median_ici_ms=15.0,
                onset_rate_hz=50.0,
                ici_coefficient_of_variation=0.3,
            ),
            species="marmoset",
        )

        _ = AudioPhrase(  # Unused: test placeholder phrase
            key="aggressive",
            features=Vector17D(
                mean_f0_hz=9000.0,
                duration_ms=30.0,
                f0_range_hz=700.0,
                harmonic_to_noise_ratio=30.0,
                spectral_flatness=0.6,
                attack_time_ms=2.0,
                decay_time_ms=10.0,
                sustain_level=0.9,
                vibrato_rate_hz=12.0,
                vibrato_depth=0.06,
                jitter=0.05,
                mfcc_1=-5.0,
                mfcc_2=-2.0,
                mfcc_3=-0.5,
                mfcc_4=0.0,
                spectral_contrast=35.0,
                harmonicity=0.75,
                shimmer=0.015,
                spectral_flux=1.5,
                median_ici_ms=10.0,
                onset_rate_hz=70.0,
                ici_coefficient_of_variation=0.5,
            ),
            species="marmoset",
        )

        mock_algebra = Mock()
        mock_algebra.generate_graded_vector = Mock(return_value=virtual_target)

        mock_db = Mock()
        mock_db.find_nearest = Mock(return_value=phrase_a)

        mock_synthesizer = Mock()

        from realtime.cognitive_interaction_engine import CognitiveInteractionEngine

        engine = CognitiveInteractionEngine(
            algebra_map=mock_algebra, phrase_db=mock_db, synthesizer=mock_synthesizer
        )

        # Act
        result = engine.generate_response(intent="aggression", intensity=0.5)

        # Assert
        mock_db.find_nearest.assert_called_once_with(virtual_target)
        # Verify phrase_a was returned (it's closer to 7500 than phrase_b's 9000)
        self.assertEqual(result["source_phrase"], "neutral")


# =============================================================================
# Test 1.3: The Safety Valve (Delta Clamping)
# =============================================================================


class TestDeltaClamping(unittest.TestCase):
    """Test 1.3: Engine applies safety clamping to prevent over-warping"""

    def test_generate_response_clamps_extrapolation(self):
        """
        RED TEST: Engine should clamp when delta exceeds MAX_WARP (0.2)

        Scenario: Virtual target is very far from nearest neighbor
        Expected: Engine detects distance > 0.2 and applies clamping
        """
        # Arrange
        virtual_target = Vector17D(
            mean_f0_hz=9000.0,
            duration_ms=20.0,
            f0_range_hz=700.0,  # Very far
            harmonic_to_noise_ratio=30.0,
            spectral_flatness=0.6,
            attack_time_ms=2.0,
            decay_time_ms=10.0,
            sustain_level=0.9,
            vibrato_rate_hz=12.0,
            vibrato_depth=0.06,
            jitter=0.05,
            mfcc_1=-5.0,
            mfcc_2=-2.0,
            mfcc_3=-0.5,
            mfcc_4=0.0,
            spectral_contrast=35.0,
            harmonicity=0.75,
            shimmer=0.015,
            spectral_flux=1.5,
            median_ici_ms=5.0,
            onset_rate_hz=100.0,
            ici_coefficient_of_variation=1.0,
        )

        nearest_phrase = AudioPhrase(
            key="neutral",
            features=Vector17D(
                mean_f0_hz=7000.0,
                duration_ms=50.0,
                f0_range_hz=400.0,
                harmonic_to_noise_ratio=20.0,
                spectral_flatness=0.3,
                attack_time_ms=5.0,
                decay_time_ms=20.0,
                sustain_level=0.7,
                vibrato_rate_hz=7.0,
                vibrato_depth=0.02,
                jitter=0.01,
                mfcc_1=-10.0,
                mfcc_2=-5.0,
                mfcc_3=-2.0,
                mfcc_4=-1.0,
                spectral_contrast=20.0,
                harmonicity=0.75,
                shimmer=0.015,
                spectral_flux=1.5,
                median_ici_ms=15.0,
                onset_rate_hz=50.0,
                ici_coefficient_of_variation=0.3,
            ),
            species="marmoset",
        )

        mock_algebra = Mock()
        mock_algebra.generate_graded_vector = Mock(return_value=virtual_target)

        mock_db = Mock()
        mock_db.find_nearest = Mock(return_value=nearest_phrase)

        mock_synthesizer = Mock()

        from realtime.cognitive_interaction_engine import CognitiveInteractionEngine

        engine = CognitiveInteractionEngine(
            algebra_map=mock_algebra,
            phrase_db=mock_db,
            synthesizer=mock_synthesizer,
            max_safe_warp=0.2,
        )

        # Act
        with self.assertLogs(engine.logger, level="WARNING") as log:
            _ = engine.generate_response(intent="aggression", intensity=0.8)

        # Assert
        # Should log warning about clamping
        self.assertTrue(any("clamp" in msg.lower() for msg in log.output))
        # Should have called synthesizer with clamped parameters
        self.assertTrue(mock_synthesizer.set_warp_delta.called)


# =============================================================================
# Test 1.4: Delta to Rust Parameter Conversion
# =============================================================================


class TestDeltaConversion(unittest.TestCase):
    """Test 1.4: Engine correctly converts delta to Rust synthesis parameters"""

    def test_generate_response_sends_correct_warp(self):
        """
        RED TEST: Engine should convert delta to correct Rust parameters

        Scenario: Delta is [+0.05 (F0), +0.1 (Roughness), 0.0 (Duration)]
        Expected: set_warp_delta called with pitch_shift ~5% and roughness ~10%
        """
        # Arrange
        anchor_features = Vector17D(
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.3,
            attack_time_ms=5.0,
            decay_time_ms=20.0,
            sustain_level=0.7,
            vibrato_rate_hz=7.0,
            vibrato_depth=0.02,
            jitter=0.01,
            mfcc_1=-10.0,
            mfcc_2=-5.0,
            mfcc_3=-2.0,
            mfcc_4=-1.0,
            spectral_contrast=20.0,
            harmonicity=0.75,
            shimmer=0.015,
            spectral_flux=1.5,
            median_ici_ms=15.0,
            onset_rate_hz=50.0,
            ici_coefficient_of_variation=0.3,
        )

        virtual_target = Vector17D(
            mean_f0_hz=7350.0,  # +350 Hz (5% increase)
            duration_ms=50.0,  # Same
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.4,  # +0.1 increase
            attack_time_ms=5.0,
            decay_time_ms=20.0,
            sustain_level=0.7,
            vibrato_rate_hz=7.0,
            vibrato_depth=0.02,
            jitter=0.01,
            mfcc_1=-10.0,
            mfcc_2=-5.0,
            mfcc_3=-2.0,
            mfcc_4=-1.0,
            spectral_contrast=20.0,
            harmonicity=0.75,
            shimmer=0.015,
            spectral_flux=1.5,
            median_ici_ms=15.0,
            onset_rate_hz=50.0,
            ici_coefficient_of_variation=0.3,
        )

        nearest_phrase = AudioPhrase(key="neutral", features=anchor_features, species="marmoset")

        mock_algebra = Mock()
        mock_algebra.generate_graded_vector = Mock(return_value=virtual_target)

        mock_db = Mock()
        mock_db.find_nearest = Mock(return_value=nearest_phrase)

        mock_synthesizer = Mock()

        from realtime.cognitive_interaction_engine import CognitiveInteractionEngine

        engine = CognitiveInteractionEngine(
            algebra_map=mock_algebra,
            phrase_db=mock_db,
            synthesizer=mock_synthesizer,
            max_safe_warp=0.3,
        )

        # Act
        _ = engine.generate_response(intent="aggression", intensity=0.5)

        # Assert
        self.assertTrue(mock_synthesizer.set_warp_delta.called)

        # Get the warp_params dict that was passed to set_warp_delta
        call_args = mock_synthesizer.set_warp_delta.call_args[0]
        warp_params = call_args[0] if call_args else {}

        # Verify pitch_shift_ratio (should be ~1.05 for 5% F0 increase)
        pitch_shift = warp_params.get("pitch_shift_ratio", warp_params.get("pitch_shift"))
        if isinstance(pitch_shift, dict):
            pitch_shift = pitch_shift.get("value", pitch_shift)
        self.assertAlmostEqual(
            pitch_shift, 1.05, delta=0.01, msg="Pitch shift should be ~1.05 (5% increase)"
        )

        # Verify roughness_amount (should be ~0.4)
        roughness = warp_params.get("roughness_amount", warp_params.get("roughness"))
        if isinstance(roughness, dict):
            roughness = roughness.get("value", roughness)
        self.assertAlmostEqual(roughness, 0.4, delta=0.01, msg="Roughness should be ~0.4")


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
