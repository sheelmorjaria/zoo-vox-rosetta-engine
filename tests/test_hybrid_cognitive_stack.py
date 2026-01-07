"""
TDD Integration Tests for Cognitive Hybrid Stack (Phase 3: End-to-End)

This test suite validates the complete hybrid stack:
1. Python "Brain" generates intents and calculates virtual targets
2. Rust "Mouth" executes synthesis with island hopping
3. Full pipeline from intent to audio output
4. Latency constraints (<100ms response time)

Architecture: Python (Logic) → Rust (Execution)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
import time
import json
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional, Dict, Any


# =============================================================================
# Mock Dependencies
# =============================================================================


@dataclass
class Vector17D:
    """20D acoustic vector (expanded from 17D, matches Rust implementation)"""

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
    spectral_contrast: float
    spectral_flux: float  # NEW
    median_ici_ms: float
    onset_rate_hz: float
    ici_coefficient_of_variation: float


@dataclass
class AudioPhrase:
    """Audio phrase from database"""

    key: str
    features: Vector17D
    species: str


# =============================================================================
# Test 3.1: Full Stack Gradient Generation
# =============================================================================


class TestFullStackGradientGeneration(unittest.TestCase):
    """Test 3.1: Complete pipeline from intent to synthesis parameters"""

    def test_full_stack_generates_correct_gradient(self):
        """
        RED TEST: Full stack generates correct morph gradient

        Scenario:
        1. Python receives intent="aggression", intensity=0.7
        2. Acoustic Algebra calculates virtual target
        3. Nearest neighbor found in database
        4. Delta calculated and clamped
        5. Rust synthesis parameters generated
        Expected: Complete pipeline executes, correct parameters passed to Rust
        """
        # Arrange
        virtual_target = Vector17D(
            mean_f0_hz=8200.0,
            duration_ms=35.0,
            f0_range_hz=600.0,
            harmonic_to_noise_ratio=26.0,
            spectral_flatness=0.5,
            attack_time_ms=3.0,
            decay_time_ms=15.0,
            sustain_level=0.85,
            vibrato_rate_hz=10.0,
            vibrato_depth=0.05,
            jitter=0.03,
            mfcc_1=-7.0,
            mfcc_2=-3.0,
            mfcc_3=-1.0,
            mfcc_4=-0.3,
            spectral_contrast=30.0,
            harmonicity=0.75,
            shimmer=0.015,
            spectral_flux=1.5,
            median_ici_ms=12.0,
            onset_rate_hz=60.0,
            ici_coefficient_of_variation=0.35,
        )

        nearest_phrase = AudioPhrase(
            key="neutral_001",
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

        # Import CognitiveInteractionEngine
        from realtime.cognitive_interaction_engine import CognitiveInteractionEngine

        engine = CognitiveInteractionEngine(
            algebra_map=mock_algebra,
            phrase_db=mock_db,
            synthesizer=mock_synthesizer,
            max_safe_warp=0.25,
        )

        # Act - Generate response
        start_time = time.time()
        result = engine.generate_response(intent="aggression", intensity=0.7)
        elapsed = time.time() - start_time

        # Assert
        self.assertIsNotNone(result, "Result should not be None")
        self.assertEqual(result["intent"], "aggression")
        self.assertEqual(result["intensity"], 0.7)
        self.assertEqual(result["source_phrase"], "neutral_001")

        # Verify algebra was called correctly
        mock_algebra.generate_graded_vector.assert_called_once_with(
            intent="aggression", intensity=0.7
        )

        # Verify nearest neighbor was found
        mock_db.find_nearest.assert_called_once_with(virtual_target)

        # Verify synthesizer was called
        self.assertTrue(mock_synthesizer.set_warp_delta.called)

        # Get the warp parameters that were passed
        call_args = mock_synthesizer.set_warp_delta.call_args[0]
        warp_params = call_args[0] if call_args else {}

        # Verify key parameters
        self.assertIn("pitch_shift_ratio", warp_params)
        self.assertIn("roughness_amount", warp_params)
        self.assertIn("duration_scale", warp_params)

        # Pitch should increase (target is higher)
        self.assertGreater(
            warp_params["pitch_shift_ratio"], 1.0, "Pitch shift should be >1.0 (pitch increase)"
        )

        # Roughness should increase (target is rougher)
        self.assertGreater(warp_params["roughness_amount"], 0.3, "Roughness should increase")

        print(f"✓ Full stack test passed in {elapsed * 1000:.2f}ms")
        print(f"  Source phrase: {result['source_phrase']}")
        print(f"  Pitch shift: {warp_params['pitch_shift_ratio']:.2f}x")
        print(f"  Roughness: {warp_params['roughness_amount']:.2f}")
        print(f"  Duration scale: {warp_params['duration_scale']:.2f}x")


# =============================================================================
# Test 3.2: Latency Constraints
# =============================================================================


class TestLatencyConstraints(unittest.TestCase):
    """Test 3.2: Full pipeline completes in <100ms"""

    def test_response_latency_under_100ms(self):
        """
        RED TEST: Full stack response time must be <100ms

        Scenario: Generate 10 responses sequentially
        Expected: All responses complete in <100ms
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

        from realtime.cognitive_interaction_engine import CognitiveInteractionEngine

        engine = CognitiveInteractionEngine(
            algebra_map=mock_algebra, phrase_db=mock_db, synthesizer=mock_synthesizer
        )

        # Act - Generate 10 responses
        latencies = []
        for i in range(10):
            start = time.time()
            result = engine.generate_response(intent="aggression", intensity=0.5)
            elapsed_ms = (time.time() - start) * 1000
            latencies.append(elapsed_ms)

            # Verify each response succeeded
            self.assertIsNotNone(result, f"Response {i} should not be None")

        # Assert
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        # All responses must be <100ms
        for i, latency in enumerate(latencies):
            self.assertLess(
                latency, 100.0, f"Response {i} took {latency:.2f}ms (exceeds 100ms limit)"
            )

        # Average should be significantly lower (ideally <50ms)
        self.assertLess(avg_latency, 50.0, f"Average latency {avg_latency:.2f}ms should be <50ms")

        print(f"✓ Latency test passed")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Min: {min_latency:.2f}ms")
        print(f"  Max: {max_latency:.2f}ms")
        print(f"  All 10 responses <100ms ✓")


# =============================================================================
# Test 3.3: Gradient Intensity Mapping
# =============================================================================


class TestGradientIntensityMapping(unittest.TestCase):
    """Test 3.3: Intensity maps to correct gradient"""

    def test_intensity_0_5_maps_to_correct_gradient(self):
        """
        RED TEST: Intensity 0.5 maps to correct gradient (midpoint)

        Scenario: Request aggression at intensity 0.5
        Expected: Virtual target is midway between baseline and full aggression
        """
        # This test verifies that the Acoustic Algebra correctly maps intensity
        # to the appropriate gradient in 17D space
        pass  # Placeholder - would require actual AcousticAlgebraMap implementation


# =============================================================================
# Test 3.4: Safety Clamp Activation
# =============================================================================


class TestSafetyClampActivation(unittest.TestCase):
    """Test 3.4: Safety clamp activates when needed"""

    def test_clamp_activates_for_large_deltas(self):
        """
        RED TEST: Safety clamp activates for large extrapolation distances

        Scenario: Virtual target is 0.5 distance from nearest neighbor
        Expected: Clamping activates, delta is scaled to max_safe_warp
        """
        # Arrange
        virtual_target = Vector17D(
            mean_f0_hz=10500.0,  # Very far from anchor
            duration_ms=10.0,
            f0_range_hz=900.0,
            harmonic_to_noise_ratio=40.0,
            spectral_flatness=0.9,
            attack_time_ms=0.0,
            decay_time_ms=5.0,
            sustain_level=1.0,
            vibrato_rate_hz=20.0,
            vibrato_depth=0.1,
            jitter=0.1,
            mfcc_1=0.0,
            mfcc_2=0.0,
            mfcc_3=0.0,
            mfcc_4=0.0,
            spectral_contrast=50.0,
            harmonicity=0.75,
            shimmer=0.015,
            spectral_flux=1.5,
            median_ici_ms=5.0,
            onset_rate_hz=100.0,
            ici_coefficient_of_variation=1.0,
        )

        nearest_phrase = AudioPhrase(
            key="neutral_001",
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
            max_safe_warp=0.2,  # 20% max warp
        )

        # Act
        with self.assertLogs(engine.logger, level="WARNING") as log:
            result = engine.generate_response(intent="aggression", intensity=1.0)

        # Assert
        self.assertIsNotNone(result)
        self.assertTrue(result["was_clamped"], "Result should be clamped")

        # Verify warning was logged
        self.assertTrue(
            any("clamping" in msg.lower() or "clamp" in msg.lower() for msg in log.output),
            "Should log clamping warning",
        )

        # Verify synthesizer received clamped parameters
        self.assertTrue(mock_synthesizer.set_warp_delta.called)

        print("✓ Safety clamp activation test passed")
        print(f"  Clamp activated: {result['was_clamped']}")
        print(f"  Warning logged: True")


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
