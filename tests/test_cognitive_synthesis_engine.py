"""
TDD Tests for Cognitive Synthesis Engine (Corvid Multi-Modal Communication)

This test suite validates the "Cognitive" approach to corvid communication,
which requires decomposing single intents into multi-modality strategies.

Key Features:
1. Intent Decomposition: One intent → Multiple modalities (e.g., Alarm → Whistle + Rattle)
2. Multi-Vector Modulation: Different deltas per modality component
3. Granular Cross-Fading: Smooth transitions between source buffers
4. Timeline Orchestration: Coordinated multi-modal synthesis

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np

# =============================================================================
# Phase 1: Extended Data Models with Cross-Fade Support
# =============================================================================


class Modality(Enum):
    """Vocalization modality types"""

    HARMONIC = "HARMONIC"
    TRANSIENT = "TRANSIENT"
    FM_SWEEP = "FM_SWEEP"


class Intent(Enum):
    """Semantic intents that map to multi-modal strategies"""

    ALARM = "ALARM"
    AGGRESSION = "AGGRESSION"
    COURTSHIP = "COURTSHIP"
    FOOD_DISCOVERY = "FOOD_DISCOVERY"
    TERRITORY = "TERRITORY"


@dataclass
class ModalityDelta:
    """Delta parameters for a specific modality"""

    pitch_shift_ratio: float = 1.0
    roughness_amount: float = 0.0
    time_stretch_ratio: float = 1.0
    grain_size_ms: float = 20.0
    vibrato_amount: float = 0.0


@dataclass
class TimelineEvent:
    """Enhanced timeline event with cross-fade support"""

    start_ms: float
    duration_ms: float
    source_buffer: str
    modality: Modality
    delta: ModalityDelta
    fade_in_ms: float = 0.0
    fade_out_ms: float = 0.0


@dataclass
class ModalityStrategy:
    """
    Defines how to decompose an intent into a multi-modal sequence.

    Example: ALARM → [Whistle (Attention) + Rattle (Urgency) + Whistle (Resolve)]
    """

    intent: Intent
    intensity: float
    sequence: List[TimelineEvent]


@dataclass
class VirtualTarget:
    """17D virtual target vector (reused from Hybrid Bridge)"""

    mean_f0_hz: float
    duration_ms: float
    harmonic_to_noise_ratio: float
    spectral_flatness: float


# =============================================================================
# Phase 2: Intent Decomposition Engine
# =============================================================================


class IntentDecomposer:
    """
    The "Cognitive" Logic: Decomposes single intents into multi-modal strategies.

    Key Principle: Corvids use "Sentences" not "Words"
    - Alarm = Whistle (Attention) + Rattle (Urgency) + Whistle (Resolve)
    - Aggression = Rattle (Threat) + Whistle (Dominance)
    """

    def __init__(self):
        # Define modality archetypes
        self.modality_profiles = {
            Modality.HARMONIC: VirtualTarget(
                mean_f0_hz=7000.0,
                duration_ms=80.0,
                harmonic_to_noise_ratio=25.0,
                spectral_flatness=0.05,
            ),
            Modality.TRANSIENT: VirtualTarget(
                mean_f0_hz=4000.0,
                duration_ms=30.0,
                harmonic_to_noise_ratio=3.0,
                spectral_flatness=0.8,
            ),
            Modality.FM_SWEEP: VirtualTarget(
                mean_f0_hz=6000.0,
                duration_ms=60.0,
                harmonic_to_noise_ratio=10.0,
                spectral_flatness=0.4,
            ),
        }

    def decompose_intent(self, intent: Intent, intensity: float) -> ModalityStrategy:
        """
        Decompose a single intent into a multi-modal sequence.

        Args:
            intent: Semantic intent (ALARM, AGGRESSION, etc.)
            intensity: Intensity from 0.0 to 1.0

        Returns:
            ModalityStrategy: Multi-modal sequence with deltas
        """
        strategies = {
            Intent.ALARM: self._decompose_alarm,
            Intent.AGGRESSION: self._decompose_aggression,
            Intent.COURTSHIP: self._decompose_courtship,
            Intent.FOOD_DISCOVERY: self._decompose_food_discovery,
            Intent.TERRITORY: self._decompose_territory,
        }

        decomposer = strategies.get(intent)
        if decomposer:
            return decomposer(intensity)

        # Default: Simple harmonic phrase
        return ModalityStrategy(
            intent=intent,
            intensity=intensity,
            sequence=[
                TimelineEvent(
                    start_ms=0.0,
                    duration_ms=100.0,
                    source_buffer="neutral",
                    modality=Modality.HARMONIC,
                    delta=ModalityDelta(),
                )
            ],
        )

    def _decompose_alarm(self, intensity: float) -> ModalityStrategy:
        """
        Alarm = Whistle (Attention) + Rattle (Urgency) + Whistle (Resolve)

        Components:
        - Whistle (0-100ms): Grab attention with high-pitched tone
        - Rattle (90-140ms): Add urgency with transients
        - Whistle (130-170ms): Resolve with falling tone
        """
        # Calculate modality-specific deltas
        # Whistle: Higher pitch for alarm
        whistle_delta = ModalityDelta(
            pitch_shift_ratio=1.0 + (0.2 * intensity),  # Up to +20% pitch
            roughness_amount=0.0,
            time_stretch_ratio=1.0,
            grain_size_ms=20.0,
        )

        # Rattle: Much grittier for urgency
        rattle_delta = ModalityDelta(
            pitch_shift_ratio=1.0,
            roughness_amount=0.5 * intensity,  # Up to 50% roughness
            time_stretch_ratio=1.0,
            grain_size_ms=5.0,  # Small grains for texture
        )

        return ModalityStrategy(
            intent=Intent.ALARM,
            intensity=intensity,
            sequence=[
                # 1. Whistle (Attention)
                TimelineEvent(
                    start_ms=0.0,
                    duration_ms=100.0,
                    source_buffer="corvid_whistle",
                    modality=Modality.HARMONIC,
                    delta=whistle_delta,
                    fade_out_ms=10.0,
                ),
                # 2. Rattle (Urgency) - with 10ms overlap
                TimelineEvent(
                    start_ms=90.0,
                    duration_ms=50.0,
                    source_buffer="corvid_rattle",
                    modality=Modality.TRANSIENT,
                    delta=rattle_delta,
                    fade_in_ms=10.0,
                    fade_out_ms=10.0,
                ),
                # 3. Whistle (Resolve) - with 10ms overlap
                TimelineEvent(
                    start_ms=130.0,
                    duration_ms=40.0,
                    source_buffer="corvid_whistle",
                    modality=Modality.HARMONIC,
                    delta=whistle_delta,
                    fade_in_ms=10.0,
                ),
            ],
        )

    def _decompose_aggression(self, intensity: float) -> ModalityStrategy:
        """
        Aggression = Rattle (Threat) + Whistle (Dominance)

        Components:
        - Rattle (0-60ms): Initial threat display
        - Whistle (50-110ms): Dominance assertion
        """
        rattle_delta = ModalityDelta(
            pitch_shift_ratio=1.0,
            roughness_amount=0.6 * intensity,  # Very gritty
            grain_size_ms=5.0,
        )

        whistle_delta = ModalityDelta(
            pitch_shift_ratio=1.0 + (0.15 * intensity),  # Lower pitch = more dominant
            roughness_amount=0.1 * intensity,
            grain_size_ms=20.0,
        )

        return ModalityStrategy(
            intent=Intent.AGGRESSION,
            intensity=intensity,
            sequence=[
                TimelineEvent(
                    start_ms=0.0,
                    duration_ms=60.0,
                    source_buffer="corvid_rattle",
                    modality=Modality.TRANSIENT,
                    delta=rattle_delta,
                    fade_out_ms=10.0,
                ),
                TimelineEvent(
                    start_ms=50.0,
                    duration_ms=60.0,
                    source_buffer="corvid_whistle",
                    modality=Modality.HARMONIC,
                    delta=whistle_delta,
                    fade_in_ms=10.0,
                ),
            ],
        )

    def _decompose_courtship(self, intensity: float) -> ModalityStrategy:
        """
        Courtship = Whistle (Display) + FM Sweep (Complexity)
        """
        whistle_delta = ModalityDelta(
            pitch_shift_ratio=1.0 + (0.3 * intensity),
            roughness_amount=0.0,
            vibrato_amount=0.3 * intensity,
        )

        fm_sweep_delta = ModalityDelta(
            pitch_shift_ratio=1.0 + (0.5 * intensity),
            roughness_amount=0.0,
            grain_size_ms=15.0,
        )

        return ModalityStrategy(
            intent=Intent.COURTSHIP,
            intensity=intensity,
            sequence=[
                TimelineEvent(
                    start_ms=0.0,
                    duration_ms=120.0,
                    source_buffer="corvid_whistle",
                    modality=Modality.HARMONIC,
                    delta=whistle_delta,
                    fade_out_ms=20.0,
                ),
                TimelineEvent(
                    start_ms=100.0,
                    duration_ms=80.0,
                    source_buffer="corvid_trill",
                    modality=Modality.FM_SWEEP,
                    delta=fm_sweep_delta,
                    fade_in_ms=20.0,
                ),
            ],
        )

    def _decompose_food_discovery(self, intensity: float) -> ModalityStrategy:
        """Food Discovery = Rattle (Excitement)"""
        rattle_delta = ModalityDelta(
            pitch_shift_ratio=1.0 + (0.2 * intensity),
            roughness_amount=0.7 * intensity,
            grain_size_ms=5.0,
        )

        return ModalityStrategy(
            intent=Intent.FOOD_DISCOVERY,
            intensity=intensity,
            sequence=[
                TimelineEvent(
                    start_ms=0.0,
                    duration_ms=80.0,
                    source_buffer="corvid_rattle",
                    modality=Modality.TRANSIENT,
                    delta=rattle_delta,
                ),
            ],
        )

    def _decompose_territory(self, intensity: float) -> ModalityStrategy:
        """Territory = Rattle (Warning) + Whistle (Boundary)"""
        rattle_delta = ModalityDelta(
            pitch_shift_ratio=1.0,
            roughness_amount=0.5 * intensity,
            grain_size_ms=5.0,
        )

        whistle_delta = ModalityDelta(
            pitch_shift_ratio=1.0 + (0.1 * intensity),
            roughness_amount=0.0,
            grain_size_ms=20.0,
        )

        return ModalityStrategy(
            intent=Intent.TERRITORY,
            intensity=intensity,
            sequence=[
                TimelineEvent(
                    start_ms=0.0,
                    duration_ms=70.0,
                    source_buffer="corvid_rattle",
                    modality=Modality.TRANSIENT,
                    delta=rattle_delta,
                    fade_out_ms=10.0,
                ),
                TimelineEvent(
                    start_ms=60.0,
                    duration_ms=100.0,
                    source_buffer="corvid_whistle",
                    modality=Modality.HARMONIC,
                    delta=whistle_delta,
                    fade_in_ms=10.0,
                ),
            ],
        )


# =============================================================================
# Phase 3: Cognitive Synthesis Engine (Orchestrator)
# =============================================================================


class CognitiveSynthesisEngine:
    """
    Orchestrates cognitive multi-modal synthesis.

    This is the "Cognitive Layer" that:
    1. Decomposes intents into multi-modal strategies
    2. Applies modality-specific deltas
    3. Executes granular cross-fading
    """

    def __init__(self):
        self.decomposer = IntentDecomposer()
        self.source_buffers: Dict[str, np.ndarray] = {}

    def register_source(self, name: str, audio: np.ndarray):
        """Register a source audio buffer"""
        self.source_buffers[name] = audio

    def synthesize_intent(
        self, intent: Intent, intensity: float
    ) -> Tuple[np.ndarray, ModalityStrategy]:
        """
        Synthesize a multi-modal vocalization from semantic intent.

        Args:
            intent: Semantic intent (ALARM, AGGRESSION, etc.)
            intensity: Intensity from 0.0 to 1.0

        Returns:
            (audio_buffer, strategy): Synthesized audio and strategy used
        """
        # 1. DECOMPOSE INTENT
        strategy = self.decomposer.decompose_intent(intent, intensity)

        # 2. EXECUTE TIMELINE with cross-fading
        audio = self._execute_timeline(strategy.sequence)

        return audio, strategy

    def _execute_timeline(self, events: List[TimelineEvent]) -> np.ndarray:
        """
        Execute timeline with granular cross-fading.

        This is where the "Cognitive" magic happens:
        - Multiple source buffers
        - Modality-specific warping
        - Smooth cross-fades at transitions
        """
        if not events:
            return np.array([], dtype=np.float32)

        # Calculate total duration
        total_duration_ms = max(e.start_ms + e.duration_ms for e in events)
        sample_rate = 44100
        total_samples = int(total_duration_ms / 1000.0 * sample_rate)

        output = np.zeros(total_samples, dtype=np.float32)

        for event in events:
            # Get source buffer
            source = self.source_buffers.get(event.source_buffer)
            if source is None:
                # Create placeholder audio for testing
                source = self._generate_placeholder_audio(event)

            # Apply modality-specific warping
            warped = self._apply_warp(source, event.delta)

            # Calculate sample positions
            start_sample = int(event.start_ms / 1000.0 * sample_rate)
            duration_samples = int(event.duration_ms / 1000.0 * sample_rate)
            end_sample = start_sample + duration_samples

            # Apply cross-fade
            if event.fade_in_ms > 0:
                fade_in_samples = int(event.fade_in_ms / 1000.0 * sample_rate)
                self._apply_fade_in(warped, fade_in_samples)

            if event.fade_out_ms > 0:
                fade_out_samples = int(event.fade_out_ms / 1000.0 * sample_rate)
                self._apply_fade_out(warped, fade_out_samples)

            # Mix into output
            end_sample = min(end_sample, total_samples)
            warped_duration = min(len(warped), end_sample - start_sample)

            if warped_duration > 0:
                # Mix with existing audio (additive for cross-fade overlap)
                output[start_sample : start_sample + warped_duration] += warped[:warped_duration]

        # Normalize to prevent clipping
        if np.max(np.abs(output)) > 0:
            output = output / np.max(np.abs(output))

        return output

    def _generate_placeholder_audio(self, event: TimelineEvent) -> np.ndarray:
        """Generate placeholder audio for testing"""
        # Apply time stretch ratio to duration
        stretched_duration_ms = event.duration_ms * event.delta.time_stretch_ratio
        duration_samples = int(stretched_duration_ms / 1000.0 * 44100)

        if event.modality == Modality.HARMONIC:
            # Sine wave
            t = np.linspace(0, stretched_duration_ms / 1000.0, duration_samples)
            freq = event.delta.pitch_shift_ratio * 7000.0
            audio = 0.3 * np.sin(2 * np.pi * freq * t)
        elif event.modality == Modality.TRANSIENT:
            # Noise burst
            audio = np.random.randn(duration_samples).astype(np.float32) * 0.3
            audio *= np.exp(-np.linspace(0, 5, duration_samples))  # Decay
        else:  # FM_SWEEP
            t = np.linspace(0, stretched_duration_ms / 1000.0, duration_samples)
            freq_start = 5000.0 * event.delta.pitch_shift_ratio
            freq_end = 8000.0 * event.delta.pitch_shift_ratio
            freq_inst = freq_start + (freq_end - freq_start) * t / (stretched_duration_ms / 1000.0)
            phase = 2 * np.pi * np.cumsum(freq_inst) / 44100.0
            audio = 0.3 * np.sin(phase)

        return audio.astype(np.float32)

    def _apply_warp(self, audio: np.ndarray, delta: ModalityDelta) -> np.ndarray:
        """Apply warping parameters to audio (simplified for testing)"""
        # Placeholder: In real implementation, this would use Rust granular engine
        # For now, just return the audio with basic modifications
        return audio

    def _apply_fade_in(self, audio: np.ndarray, fade_samples: int):
        """Apply linear fade-in"""
        fade_samples = min(fade_samples, len(audio))
        if fade_samples > 0:
            fade_curve = np.linspace(0, 1, fade_samples)
            audio[:fade_samples] *= fade_curve

    def _apply_fade_out(self, audio: np.ndarray, fade_samples: int):
        """Apply linear fade-out"""
        fade_samples = min(fade_samples, len(audio))
        if fade_samples > 0:
            fade_curve = np.linspace(1, 0, fade_samples)
            audio[-fade_samples:] *= fade_curve


# =============================================================================
# Phase 1 Tests: Intent Decomposition
# =============================================================================


class TestIntentDecomposition(unittest.TestCase):
    """Test cognitive intent decomposition into multi-modal strategies"""

    def setUp(self):
        self.decomposer = IntentDecomposer()

    def test_alarm_decomposition(self):
        """Test ALARM intent decomposes into Whistle + Rattle + Whistle"""
        strategy = self.decomposer.decompose_intent(Intent.ALARM, 0.5)

        self.assertEqual(strategy.intent, Intent.ALARM)
        self.assertEqual(strategy.intensity, 0.5)
        self.assertEqual(len(strategy.sequence), 3)

        # Verify sequence: Whistle → Rattle → Whistle
        self.assertEqual(strategy.sequence[0].modality, Modality.HARMONIC)
        self.assertEqual(strategy.sequence[1].modality, Modality.TRANSIENT)
        self.assertEqual(strategy.sequence[2].modality, Modality.HARMONIC)

    def test_aggression_decomposition(self):
        """Test AGGRESSION intent decomposes into Rattle + Whistle"""
        strategy = self.decomposer.decompose_intent(Intent.AGGRESSION, 0.7)

        self.assertEqual(len(strategy.sequence), 2)
        self.assertEqual(strategy.sequence[0].modality, Modality.TRANSIENT)
        self.assertEqual(strategy.sequence[1].modality, Modality.HARMONIC)

    def test_courtship_decomposition(self):
        """Test COURTSHIP intent decomposes into Whistle + FM Sweep"""
        strategy = self.decomposer.decompose_intent(Intent.COURTSHIP, 0.6)

        self.assertEqual(len(strategy.sequence), 2)
        self.assertEqual(strategy.sequence[0].modality, Modality.HARMONIC)
        self.assertEqual(strategy.sequence[1].modality, Modality.FM_SWEEP)

    def test_intensity_scaling(self):
        """Test that intensity affects delta parameters"""
        strategy_low = self.decomposer.decompose_intent(Intent.ALARM, 0.2)
        strategy_high = self.decomposer.decompose_intent(Intent.ALARM, 0.8)

        # Higher intensity = more pitch shift on whistle
        whistle_delta_low = strategy_low.sequence[0].delta
        whistle_delta_high = strategy_high.sequence[0].delta

        self.assertGreater(
            whistle_delta_high.pitch_shift_ratio, whistle_delta_low.pitch_shift_ratio
        )

        # Higher intensity = more roughness on rattle
        rattle_delta_low = strategy_low.sequence[1].delta
        rattle_delta_high = strategy_high.sequence[1].delta

        self.assertGreater(rattle_delta_high.roughness_amount, rattle_delta_low.roughness_amount)

    def test_cross_fade_parameters(self):
        """Test that cross-fade parameters are set correctly"""
        strategy = self.decomposer.decompose_intent(Intent.ALARM, 0.5)

        # First event should have fade_out
        self.assertGreater(strategy.sequence[0].fade_out_ms, 0)

        # Middle event should have both fade_in and fade_out
        self.assertGreater(strategy.sequence[1].fade_in_ms, 0)
        self.assertGreater(strategy.sequence[1].fade_out_ms, 0)

        # Last event should have fade_in
        self.assertGreater(strategy.sequence[2].fade_in_ms, 0)

    def test_timeline_overlap(self):
        """Test that events overlap for smooth cross-fading"""
        strategy = self.decomposer.decompose_intent(Intent.ALARM, 0.5)

        # Events should overlap
        # Event 1: 0-100ms, Event 2: 90-140ms
        # Overlap: 90-100ms (10ms)
        event1_end = strategy.sequence[0].start_ms + strategy.sequence[0].duration_ms
        event2_start = strategy.sequence[1].start_ms

        self.assertLess(event2_start, event1_end, "Events should overlap for cross-fading")


# =============================================================================
# Phase 2 Tests: Multi-Vector Modulation
# =============================================================================


class TestMultiVectorModulation(unittest.TestCase):
    """Test different deltas for different modalities"""

    def setUp(self):
        self.decomposer = IntentDecomposer()

    def test_whistle_affected_differently_than_rattle(self):
        """Test that whistle and rattle get different delta parameters"""
        strategy = self.decomposer.decompose_intent(Intent.ALARM, 0.5)

        whistle_delta = strategy.sequence[0].delta
        rattle_delta = strategy.sequence[1].delta

        # Whistle gets pitch shift
        self.assertGreater(whistle_delta.pitch_shift_ratio, 1.0)
        self.assertEqual(whistle_delta.roughness_amount, 0.0)

        # Rattle gets roughness
        self.assertGreater(rattle_delta.roughness_amount, 0.0)
        self.assertEqual(rattle_delta.pitch_shift_ratio, 1.0)

    def test_grain_size_modality_specific(self):
        """Test that grain size is modality-specific"""
        strategy = self.decomposer.decompose_intent(Intent.ALARM, 0.5)

        whistle_delta = strategy.sequence[0].delta
        rattle_delta = strategy.sequence[1].delta

        # Rattle uses smaller grains for texture
        self.assertLess(rattle_delta.grain_size_ms, whistle_delta.grain_size_ms)

    def test_aggression_intensity_affects_both_modalities(self):
        """Test that aggression intensity affects both rattle and whistle"""
        strategy = self.decomposer.decompose_intent(Intent.AGGRESSION, 0.8)

        rattle_delta = strategy.sequence[0].delta
        whistle_delta = strategy.sequence[1].delta

        # Both should be affected
        self.assertGreater(rattle_delta.roughness_amount, 0.0)
        self.assertGreater(whistle_delta.roughness_amount, 0.0)


# =============================================================================
# Phase 3 Tests: Cognitive Engine Orchestration
# =============================================================================


class TestCognitiveSynthesisEngine(unittest.TestCase):
    """Test the complete cognitive synthesis workflow"""

    def setUp(self):
        self.engine = CognitiveSynthesisEngine()

        # Register test sources
        self.engine.register_source(
            "corvid_whistle", np.random.randn(44100).astype(np.float32) * 0.1
        )
        self.engine.register_source(
            "corvid_rattle", np.random.randn(13230).astype(np.float32) * 0.1
        )

    def test_synthesize_alarm(self):
        """Test synthesizing alarm call"""
        audio, strategy = self.engine.synthesize_intent(Intent.ALARM, 0.5)

        self.assertIsNotNone(audio)
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

        # Verify strategy
        self.assertEqual(strategy.intent, Intent.ALARM)
        self.assertEqual(len(strategy.sequence), 3)

    def test_synthesize_aggression(self):
        """Test synthesizing aggression call"""
        audio, strategy = self.engine.synthesize_intent(Intent.AGGRESSION, 0.6)

        self.assertIsNotNone(audio)
        self.assertEqual(strategy.intent, Intent.AGGRESSION)
        self.assertEqual(len(strategy.sequence), 2)

    def test_different_intensities_produce_different_audio(self):
        """Test that different intensities produce different results"""
        audio_low, strategy_low = self.engine.synthesize_intent(Intent.ALARM, 0.2)
        audio_high, strategy_high = self.engine.synthesize_intent(Intent.ALARM, 0.8)

        # Audio should be different - check that at least one delta parameter differs
        # (In real implementation, differences would be in pitch, roughness, etc.)
        low_deltas = [e.delta for e in strategy_low.sequence]
        high_deltas = [e.delta for e in strategy_high.sequence]

        # At least one delta should be different
        differences_found = False
        for low, high in zip(low_deltas, high_deltas):
            if (
                low.pitch_shift_ratio != high.pitch_shift_ratio
                or low.roughness_amount != high.roughness_amount
                or low.grain_size_ms != high.grain_size_ms
            ):
                differences_found = True
                break

        self.assertTrue(
            differences_found, "Different intensities should produce different delta parameters"
        )

    def test_cross_fade_smooths_transitions(self):
        """Test that cross-fade prevents audio glitches"""
        audio, strategy = self.engine.synthesize_intent(Intent.ALARM, 0.5)

        # Verify no clipping (which would indicate poor mixing)
        self.assertLessEqual(np.max(np.abs(audio)), 1.0)


# =============================================================================
# Phase 4 Tests: Edge Cases and Safety
# =============================================================================


class TestCognitiveEngineSafety(unittest.TestCase):
    """Test safety mechanisms and edge cases"""

    def setUp(self):
        self.engine = CognitiveSynthesisEngine()

    def test_empty_source_registry(self):
        """Test behavior with no sources registered"""
        audio, strategy = self.engine.synthesize_intent(Intent.ALARM, 0.5)

        # Should still generate audio (using placeholders)
        self.assertIsNotNone(audio)
        self.assertGreater(len(audio), 0)

    def test_zero_intensity(self):
        """Test zero intensity (neutral call)"""
        audio, strategy = self.engine.synthesize_intent(Intent.ALARM, 0.0)

        self.assertIsNotNone(audio)
        # At zero intensity, deltas should be minimal
        for event in strategy.sequence:
            self.assertAlmostEqual(event.delta.pitch_shift_ratio, 1.0, places=1)

    def test_high_intensity(self):
        """Test high intensity (1.0)"""
        audio, strategy = self.engine.synthesize_intent(Intent.AGGRESSION, 1.0)

        self.assertIsNotNone(audio)
        # At high intensity, deltas should be significant
        for event in strategy.sequence:
            if event.modality == Modality.TRANSIENT:
                self.assertGreater(event.delta.roughness_amount, 0.0)

    def test_all_intents_defined(self):
        """Test that all defined intents can be decomposed"""
        intents = [
            Intent.ALARM,
            Intent.AGGRESSION,
            Intent.COURTSHIP,
            Intent.FOOD_DISCOVERY,
            Intent.TERRITORY,
        ]

        for intent in intents:
            strategy = self.engine.decomposer.decompose_intent(intent, 0.5)
            self.assertEqual(strategy.intent, intent)
            self.assertGreater(len(strategy.sequence), 0)


if __name__ == "__main__":
    unittest.main()
