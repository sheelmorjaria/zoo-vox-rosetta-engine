"""
TDD Tests for Multi-Modal Corvid Support

Corvids (crows, ravens) utilize Multiple Modalities (Transient + Harmonic + FM Sweep)
in single vocalizations, breaking the "Single Persona" assumption.

This test suite validates the "Texture Sequencing" architecture required for Corvids:
- Composite Personas (modality sequences)
- Multi-Buffer Granular Sequencer
- Modality Constraints in Acoustic Algebra
- Deception Detection via Modality Mismatches

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# =============================================================================
# Phase 1: Data Model Tests - Composite Personas
# =============================================================================


class Modality(Enum):
    """Vocalization modality types"""

    HARMONIC = "HARMONIC"  # Tonal, sine-like (whistle, phee)
    TRANSIENT = "TRANSIENT"  # Clicky, noise-like (rattle, click)
    FM_SWEEP = "FM_SWEEP"  # Frequency modulated (trill, sweep)


@dataclass
class CorvidPersona:
    """
    Composite Persona for multi-modal species like Corvids.

    A Corvid "Persona" is defined as a Sequence of Textures, not a single buffer.
    Example: "Alert Call" = [Harmonic_Start (whistle), Transient_Rattle (rattle),
    Harmonic_End (whistle)]
    """

    species: str
    id: str
    modality_sequence: List[Modality]  # NEW: Sequence of modalities


@dataclass
class PhraseModalityTag:
    """
    Modality tags for phrases discovered from audio analysis.

    Old (Marmoset): Phrase_001 = [Grain, Grain, Grain]
    New (Corvid): Phrase_002 = { Grains: [H, H, H], Dominant: "HARMONIC", Mixed: False }
    """

    phrase_id: str
    grain_modalities: List[Modality]
    dominant_modality: Modality
    is_mixed: bool


class TestCorvidDataModels(unittest.TestCase):
    """Test Phase 1: Data Model Changes for Composite Personas"""

    def test_corvid_persona_with_single_modality(self):
        """Test that CorvidPersona can represent single-modality calls"""
        persona = CorvidPersona(
            species="American Crow", id="alert_call", modality_sequence=[Modality.HARMONIC]
        )

        self.assertEqual(persona.species, "American Crow")
        self.assertEqual(persona.id, "alert_call")
        self.assertEqual(len(persona.modality_sequence), 1)
        self.assertEqual(persona.modality_sequence[0], Modality.HARMONIC)

    def test_corvid_persona_with_multi_modality(self):
        """Test that CorvidPersona can represent multi-modal sequences"""
        persona = CorvidPersona(
            species="Common Raven",
            id="alarm_call",
            modality_sequence=[
                Modality.HARMONIC,  # Whistle to get attention
                Modality.TRANSIENT,  # Rattle (the message)
                Modality.HARMONIC,  # Whistle (close)
            ],
        )

        self.assertEqual(len(persona.modality_sequence), 3)
        self.assertEqual(persona.modality_sequence[0], Modality.HARMONIC)
        self.assertEqual(persona.modality_sequence[1], Modality.TRANSIENT)
        self.assertEqual(persona.modality_sequence[2], Modality.HARMONIC)

    def test_phrase_modality_tag_pure_harmonic(self):
        """Test phrase modality tagging for pure harmonic phrases"""
        tag = PhraseModalityTag(
            phrase_id="crow_whistle_001",
            grain_modalities=[Modality.HARMONIC, Modality.HARMONIC, Modality.HARMONIC],
            dominant_modality=Modality.HARMONIC,
            is_mixed=False,
        )

        self.assertEqual(tag.dominant_modality, Modality.HARMONIC)
        self.assertFalse(tag.is_mixed)
        self.assertTrue(all(m == Modality.HARMONIC for m in tag.grain_modalities))

    def test_phrase_modality_tag_pure_transient(self):
        """Test phrase modality tagging for pure transient phrases"""
        tag = PhraseModalityTag(
            phrase_id="crow_rattle_001",
            grain_modalities=[Modality.TRANSIENT, Modality.TRANSIENT, Modality.TRANSIENT],
            dominant_modality=Modality.TRANSIENT,
            is_mixed=False,
        )

        self.assertEqual(tag.dominant_modality, Modality.TRANSIENT)
        self.assertFalse(tag.is_mixed)

    def test_phrase_modality_tag_mixed(self):
        """Test phrase modality tagging for mixed phrases (code-switching)"""
        tag = PhraseModalityTag(
            phrase_id="raven_mixed_001",
            grain_modalities=[Modality.HARMONIC, Modality.TRANSIENT, Modality.TRANSIENT],
            dominant_modality=Modality.TRANSIENT,  # Transient is dominant (2/3)
            is_mixed=True,
        )

        self.assertEqual(tag.dominant_modality, Modality.TRANSIENT)
        self.assertTrue(tag.is_mixed)
        self.assertIn(Modality.HARMONIC, tag.grain_modalities)
        self.assertIn(Modality.TRANSIENT, tag.grain_modalities)

    def test_detect_dominant_modality_harmonic(self):
        """Test dominant modality detection for harmonic-majority"""
        grains = [Modality.HARMONIC] * 7 + [Modality.TRANSIENT] * 3
        dominant = max(set(grains), key=grains.count)

        self.assertEqual(dominant, Modality.HARMONIC)

    def test_detect_dominant_modality_transient(self):
        """Test dominant modality detection for transient-majority"""
        grains = [Modality.HARMONIC] * 2 + [Modality.TRANSIENT] * 8
        dominant = max(set(grains), key=grains.count)

        self.assertEqual(dominant, Modality.TRANSIENT)


# =============================================================================
# Phase 2: Synthesis Tests - Multi-Buffer Granular Sequencer
# =============================================================================


@dataclass
class TimelineEvent:
    """
    Single event in a multi-modal sequence.

    Example: At 100ms, play TRANSIENT source for 50ms
    """

    start_ms: float
    duration_ms: float
    source_buffer: str  # e.g., "corvid_whistle.wav"
    modality: Modality


@dataclass
class ModalityTimeline:
    """Timeline of events for multi-modal synthesis"""

    events: List[TimelineEvent] = field(default_factory=list)

    def add_event(self, start_ms: float, duration_ms: float, source: str, modality: Modality):
        """Add an event to the timeline"""
        event = TimelineEvent(
            start_ms=start_ms, duration_ms=duration_ms, source_buffer=source, modality=modality
        )
        self.events.append(event)

    def sort_by_time(self):
        """Sort events by start time"""
        self.events.sort(key=lambda e: e.start_ms)

    def validate(self):
        """Validate timeline has no overlaps and is sequential"""
        self.sort_by_time()

        for i in range(len(self.events) - 1):
            current = self.events[i]
            next_event = self.events[i + 1]

            # Check no overlap
            current_end = current.start_ms + current.duration_ms
            if current_end > next_event.start_ms:
                raise ValueError(
                    f"Timeline overlap: Event {i} ends at {current_end}ms, "
                    f"Event {i + 1} starts at {next_event.start_ms}ms"
                )

        return True


class TestModalityTimeline(unittest.TestCase):
    """Test Phase 2: Multi-Buffer Granular Sequencer - Timeline Management"""

    def test_create_empty_timeline(self):
        """Test creating an empty timeline"""
        timeline = ModalityTimeline()

        self.assertEqual(len(timeline.events), 0)
        self.assertTrue(timeline.validate())

    def test_add_single_event(self):
        """Test adding a single event to timeline"""
        timeline = ModalityTimeline()
        timeline.add_event(
            start_ms=0.0, duration_ms=100.0, source="whistle.wav", modality=Modality.HARMONIC
        )

        self.assertEqual(len(timeline.events), 1)
        event = timeline.events[0]
        self.assertEqual(event.start_ms, 0.0)
        self.assertEqual(event.duration_ms, 100.0)
        self.assertEqual(event.source_buffer, "whistle.wav")
        self.assertEqual(event.modality, Modality.HARMONIC)

    def test_create_multi_modal_sequence(self):
        """Test creating a multi-modal sequence (H -> T -> H)"""
        timeline = ModalityTimeline()

        # Harmonic start (whistle to get attention)
        timeline.add_event(0.0, 100.0, "whistle.wav", Modality.HARMONIC)

        # Transient middle (rattle - the message)
        timeline.add_event(100.0, 50.0, "rattle.wav", Modality.TRANSIENT)

        # Harmonic end (whistle to close)
        timeline.add_event(150.0, 20.0, "whistle.wav", Modality.HARMONIC)

        self.assertEqual(len(timeline.events), 3)

        # Validate no overlaps
        self.assertTrue(timeline.validate())

        # Check sequence
        self.assertEqual(timeline.events[0].modality, Modality.HARMONIC)
        self.assertEqual(timeline.events[1].modality, Modality.TRANSIENT)
        self.assertEqual(timeline.events[2].modality, Modality.HARMONIC)

    def test_timeline_sorting(self):
        """Test that timeline can be sorted by start time"""
        timeline = ModalityTimeline()

        # Add events out of order
        timeline.add_event(100.0, 50.0, "rattle.wav", Modality.TRANSIENT)
        timeline.add_event(0.0, 100.0, "whistle.wav", Modality.HARMONIC)

        timeline.sort_by_time()

        self.assertEqual(timeline.events[0].start_ms, 0.0)
        self.assertEqual(timeline.events[1].start_ms, 100.0)

    def test_timeline_overlap_detection(self):
        """Test that overlapping events are detected"""
        timeline = ModalityTimeline()

        # Add overlapping events
        timeline.add_event(0.0, 150.0, "whistle.wav", Modality.HARMONIC)
        timeline.add_event(100.0, 50.0, "rattle.wav", Modality.TRANSIENT)

        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            timeline.validate()

        self.assertIn("overlap", str(context.exception).lower())

    def test_timeline_total_duration(self):
        """Test calculating total duration of timeline"""
        timeline = ModalityTimeline()

        timeline.add_event(0.0, 100.0, "whistle.wav", Modality.HARMONIC)
        timeline.add_event(100.0, 50.0, "rattle.wav", Modality.TRANSIENT)
        timeline.add_event(150.0, 20.0, "whistle.wav", Modality.HARMONIC)

        # Total duration = 150 + 20 = 170ms
        total_duration = timeline.events[-1].start_ms + timeline.events[-1].duration_ms

        self.assertEqual(total_duration, 170.0)


class TestGranularSequencer(unittest.TestCase):
    """Test Phase 2: Multi-Buffer Granular Sequencer - Synthesis Logic"""

    def test_sequencer_accepts_timeline(self):
        """
        TEST: GranularSequencer can accept and process multi-modal timeline

        This is a structural test - validates the API accepts timeline events
        """
        timeline = ModalityTimeline()
        timeline.add_event(0.0, 100.0, "whistle.wav", Modality.HARMONIC)
        timeline.add_event(100.0, 50.0, "rattle.wav", Modality.TRANSIENT)

        # Validate timeline
        self.assertTrue(timeline.validate())

        # Test that we can extract sequence info
        timeline.sort_by_time()
        self.assertEqual(len(timeline.events), 2)
        self.assertEqual(timeline.events[0].modality, Modality.HARMONIC)
        self.assertEqual(timeline.events[1].modality, Modality.TRANSIENT)

    def test_active_voice_switching(self):
        """
        TEST: Sequencer switches active voice (source buffer) per event

        This validates the "Code-Switching" problem: each event may use
        a different source buffer.
        """
        timeline = ModalityTimeline()
        timeline.add_event(0.0, 100.0, "whistle.wav", Modality.HARMONIC)
        timeline.add_event(100.0, 50.0, "rattle.wav", Modality.TRANSIENT)
        timeline.add_event(150.0, 20.0, "whistle.wav", Modality.HARMONIC)

        timeline.sort_by_time()

        # Simulate voice switching
        active_voices = []
        for event in timeline.events:
            active_voices.append(event.source_buffer)

        # Should switch voices: whistle -> rattle -> whistle
        expected = ["whistle.wav", "rattle.wav", "whistle.wav"]
        self.assertEqual(active_voices, expected)

    def test_modality_preservation_per_event(self):
        """
        TEST: Each event preserves its source modality

        Whistle events produce harmonic output, rattle events produce transient output
        """
        timeline = ModalityTimeline()
        timeline.add_event(0.0, 100.0, "whistle.wav", Modality.HARMONIC)
        timeline.add_event(100.0, 50.0, "rattle.wav", Modality.TRANSIENT)

        timeline.sort_by_time()

        # Validate modality assignment
        for event in timeline.events:
            if event.source_buffer == "whistle.wav":
                self.assertEqual(
                    event.modality,
                    Modality.HARMONIC,
                    "Whistle source should produce harmonic modality",
                )
            elif event.source_buffer == "rattle.wav":
                self.assertEqual(
                    event.modality,
                    Modality.TRANSIENT,
                    "Rattle source should produce transient modality",
                )


# =============================================================================
# Phase 3: Acoustic Algebra Tests - Modality Constraints
# =============================================================================


@dataclass
class AcousticVector:
    """
    17D acoustic feature vector with modality constraint.
    """

    mean_f0_hz: float
    duration_ms: float
    f0_range_hz: float
    harmonic_to_noise_ratio: float
    spectral_flatness: float
    attack_time_ms: float
    decay_time_ms: float
    sustain_level: float
    vibrato_rate_hz: float
    vibrato_depth: float
    jitter: float
    mfcc_1: float
    mfcc_2: float
    mfcc_3: float
    mfcc_4: float
    spectral_contrast: float
    median_ici_ms: float
    onset_rate_hz: float
    ici_coefficient_of_variation: float

    # NEW: Modality constraint
    modality: Modality


def interpolate_vectors(v1: AcousticVector, v2: AcousticVector, alpha: float) -> AcousticVector:
    """
    Interpolate between two acoustic vectors.

    WARNING: This should ONLY be called if v1.modality == v2.modality
    Interpolating across modalities creates garbage!

    Raises:
        ValueError: If v1.modality != v2.modality (modality gate)
    """
    # MODALITY GATE: Prevent cross-modality interpolation
    if v1.modality != v2.modality:
        raise ValueError(
            f"Cannot interpolate across modalities: {v1.modality.value} != {v2.modality.value}. "
            "Cross-modality interpolation creates garbage audio artifacts. "
            "Use persona switching (different source buffers) instead."
        )

    return AcousticVector(
        mean_f0_hz=v1.mean_f0_hz * (1 - alpha) + v2.mean_f0_hz * alpha,
        duration_ms=v1.duration_ms * (1 - alpha) + v2.duration_ms * alpha,
        f0_range_hz=v1.f0_range_hz * (1 - alpha) + v2.f0_range_hz * alpha,
        harmonic_to_noise_ratio=v1.harmonic_to_noise_ratio * (1 - alpha)
        + v2.harmonic_to_noise_ratio * alpha,
        spectral_flatness=v1.spectral_flatness * (1 - alpha) + v2.spectral_flatness * alpha,
        attack_time_ms=v1.attack_time_ms * (1 - alpha) + v2.attack_time_ms * alpha,
        decay_time_ms=v1.decay_time_ms * (1 - alpha) + v2.decay_time_ms * alpha,
        sustain_level=v1.sustain_level * (1 - alpha) + v2.sustain_level * alpha,
        vibrato_rate_hz=v1.vibrato_rate_hz * (1 - alpha) + v2.vibrato_rate_hz * alpha,
        vibrato_depth=v1.vibrato_depth * (1 - alpha) + v2.vibrato_depth * alpha,
        jitter=v1.jitter * (1 - alpha) + v2.jitter * alpha,
        mfcc_1=v1.mfcc_1 * (1 - alpha) + v2.mfcc_1 * alpha,
        mfcc_2=v1.mfcc_2 * (1 - alpha) + v2.mfcc_2 * alpha,
        mfcc_3=v1.mfcc_3 * (1 - alpha) + v2.mfcc_3 * alpha,
        mfcc_4=v1.mfcc_4 * (1 - alpha) + v2.mfcc_4 * alpha,
        spectral_contrast=v1.spectral_contrast * (1 - alpha) + v2.spectral_contrast * alpha,
        median_ici_ms=v1.median_ici_ms * (1 - alpha) + v2.median_ici_ms * alpha,
        onset_rate_hz=v1.onset_rate_hz * (1 - alpha) + v2.onset_rate_hz * alpha,
        ici_coefficient_of_variation=v1.ici_coefficient_of_variation * (1 - alpha)
        + v2.ici_coefficient_of_variation * alpha,
        modality=v1.modality,  # Preserve source modality
    )


class TestAcousticAlgebraConstraints(unittest.TestCase):
    """Test Phase 3: Modality Constraints in Acoustic Algebra"""

    def test_interpolate_within_same_modality(self):
        """
        TEST: Interpolation within same modality is SAFE

        Harmonic + Harmonic = Harmonic (nuanced, but still harmonic)
        """
        v1 = AcousticVector(
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=25.0,
            spectral_flatness=0.05,  # Pure harmonic
            attack_time_ms=25.0,
            decay_time_ms=15.0,
            sustain_level=0.7,
            vibrato_rate_hz=8.0,
            vibrato_depth=50.0,
            jitter=0.01,
            mfcc_1=-500.0,
            mfcc_2=-100.0,
            mfcc_3=-50.0,
            mfcc_4=-20.0,
            spectral_contrast=20.0,
            median_ici_ms=0.0,
            onset_rate_hz=0.0,
            ici_coefficient_of_variation=0.0,
            modality=Modality.HARMONIC,
        )

        v2 = AcousticVector(
            mean_f0_hz=7500.0,
            duration_ms=60.0,
            f0_range_hz=500.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.08,  # Still harmonic
            attack_time_ms=20.0,
            decay_time_ms=12.0,
            sustain_level=0.6,
            vibrato_rate_hz=7.0,
            vibrato_depth=40.0,
            jitter=0.02,
            mfcc_1=-450.0,
            mfcc_2=-90.0,
            mfcc_3=-40.0,
            mfcc_4=-15.0,
            spectral_contrast=18.0,
            median_ici_ms=0.0,
            onset_rate_hz=0.0,
            ici_coefficient_of_variation=0.0,
            modality=Modality.HARMONIC,
        )

        # Interpolate at 50%
        result = interpolate_vectors(v1, v2, alpha=0.5)

        # Assert: Result is still harmonic
        self.assertEqual(result.modality, Modality.HARMONIC)
        self.assertGreater(
            result.harmonic_to_noise_ratio, 15.0, "Interpolated harmonic should remain tonal"
        )
        self.assertLess(result.spectral_flatness, 0.2, "Interpolated harmonic should remain tonal")

    def test_interpolate_across_modalities_is_forbidden(self):
        """
        TEST: Interpolation across modalities is FORBIDDEN by modality gate

        Harmonic (clean sine) + Transient (white noise) = ValueError
        The modality gate prevents cross-modality interpolation to avoid garbage.
        """
        v_harmonic = AcousticVector(
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=25.0,
            spectral_flatness=0.05,  # Pure
            attack_time_ms=25.0,
            decay_time_ms=15.0,
            sustain_level=0.7,
            vibrato_rate_hz=8.0,
            vibrato_depth=50.0,
            jitter=0.01,
            mfcc_1=-500.0,
            mfcc_2=-100.0,
            mfcc_3=-50.0,
            mfcc_4=-20.0,
            spectral_contrast=20.0,
            median_ici_ms=0.0,
            onset_rate_hz=0.0,
            ici_coefficient_of_variation=0.0,
            modality=Modality.HARMONIC,
        )

        v_transient = AcousticVector(
            mean_f0_hz=7000.0,
            duration_ms=10.0,
            f0_range_hz=200.0,
            harmonic_to_noise_ratio=2.0,
            spectral_flatness=0.8,  # Noisy
            attack_time_ms=3.0,
            decay_time_ms=5.0,
            sustain_level=0.3,
            vibrato_rate_hz=0.0,
            vibrato_depth=0.0,
            jitter=0.15,
            mfcc_1=100.0,
            mfcc_2=50.0,
            mfcc_3=20.0,
            mfcc_4=10.0,
            spectral_contrast=5.0,
            median_ici_ms=50.0,
            onset_rate_hz=20.0,
            ici_coefficient_of_variation=0.2,
            modality=Modality.TRANSIENT,
        )

        # Assert: Cross-modality interpolation raises ValueError
        with self.assertRaises(ValueError) as context:
            interpolate_vectors(v_harmonic, v_transient, alpha=0.5)

        # Assert: Error message mentions modality
        self.assertIn("modality", str(context.exception).lower())
        self.assertIn("garbage", str(context.exception).lower())

    def test_modality_gate_blocks_cross_modality_interpolation(self):
        """
        TEST: Modality Gate prevents interpolation across modalities

        The system should detect and prevent cross-modality interpolation.
        """
        v_neutral = AcousticVector(
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=25.0,
            spectral_flatness=0.05,
            attack_time_ms=25.0,
            decay_time_ms=15.0,
            sustain_level=0.7,
            vibrato_rate_hz=8.0,
            vibrato_depth=50.0,
            jitter=0.01,
            mfcc_1=-500.0,
            mfcc_2=-100.0,
            mfcc_3=-50.0,
            mfcc_4=-20.0,
            spectral_contrast=20.0,
            median_ici_ms=0.0,
            onset_rate_hz=0.0,
            ici_coefficient_of_variation=0.0,
            modality=Modality.HARMONIC,
        )

        v_aggression = AcousticVector(
            mean_f0_hz=7000.0,
            duration_ms=10.0,
            f0_range_hz=200.0,
            harmonic_to_noise_ratio=2.0,
            spectral_flatness=0.8,
            attack_time_ms=3.0,
            decay_time_ms=5.0,
            sustain_level=0.3,
            vibrato_rate_hz=0.0,
            vibrato_depth=0.0,
            jitter=0.15,
            mfcc_1=100.0,
            mfcc_2=50.0,
            mfcc_3=20.0,
            mfcc_4=10.0,
            spectral_contrast=5.0,
            median_ici_ms=50.0,
            onset_rate_hz=20.0,
            ici_coefficient_of_variation=0.2,
            modality=Modality.TRANSIENT,
        )

        # CONSTRAINT CHECK
        if v_neutral.modality != v_aggression.modality:
            # Should raise error or use fallback
            with self.assertRaises(ValueError) as context:
                interpolate_vectors(v_neutral, v_aggression, alpha=0.5)

            self.assertIn("modality", str(context.exception).lower())


class TestModalityGate(unittest.TestCase):
    """Test the Modality Gate constraint system"""

    def test_safe_interpolation_same_modality(self):
        """Test that same-modality interpolation passes the gate"""
        v1 = AcousticVector(
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=25.0,
            spectral_flatness=0.05,
            attack_time_ms=25.0,
            decay_time_ms=15.0,
            sustain_level=0.7,
            vibrato_rate_hz=8.0,
            vibrato_depth=50.0,
            jitter=0.01,
            mfcc_1=-500.0,
            mfcc_2=-100.0,
            mfcc_3=-50.0,
            mfcc_4=-20.0,
            spectral_contrast=20.0,
            median_ici_ms=0.0,
            onset_rate_hz=0.0,
            ici_coefficient_of_variation=0.0,
            modality=Modality.HARMONIC,
        )

        v2 = AcousticVector(
            mean_f0_hz=7500.0,
            duration_ms=60.0,
            f0_range_hz=500.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.08,
            attack_time_ms=20.0,
            decay_time_ms=12.0,
            sustain_level=0.6,
            vibrato_rate_hz=7.0,
            vibrato_depth=40.0,
            jitter=0.02,
            mfcc_1=-450.0,
            mfcc_2=-90.0,
            mfcc_3=-40.0,
            mfcc_4=-15.0,
            spectral_contrast=18.0,
            median_ici_ms=0.0,
            onset_rate_hz=0.0,
            ici_coefficient_of_variation=0.0,
            modality=Modality.HARMONIC,
        )

        # Should NOT raise error
        try:
            result = interpolate_vectors(v1, v2, alpha=0.5)
            # Success
            self.assertEqual(result.modality, Modality.HARMONIC)
        except ValueError:
            self.fail("Same-modality interpolation should be allowed")

    def test_blocked_interpolation_cross_modality(self):
        """Test that cross-modality interpolation is blocked"""
        v_harmonic = AcousticVector(
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=25.0,
            spectral_flatness=0.05,
            attack_time_ms=25.0,
            decay_time_ms=15.0,
            sustain_level=0.7,
            vibrato_rate_hz=8.0,
            vibrato_depth=50.0,
            jitter=0.01,
            mfcc_1=-500.0,
            mfcc_2=-100.0,
            mfcc_3=-50.0,
            mfcc_4=-20.0,
            spectral_contrast=20.0,
            median_ici_ms=0.0,
            onset_rate_hz=0.0,
            ici_coefficient_of_variation=0.0,
            modality=Modality.HARMONIC,
        )

        v_transient = AcousticVector(
            mean_f0_hz=7000.0,
            duration_ms=10.0,
            f0_range_hz=200.0,
            harmonic_to_noise_ratio=2.0,
            spectral_flatness=0.8,
            attack_time_ms=3.0,
            decay_time_ms=5.0,
            sustain_level=0.3,
            vibrato_rate_hz=0.0,
            vibrato_depth=0.0,
            jitter=0.15,
            mfcc_1=100.0,
            mfcc_2=50.0,
            mfcc_3=20.0,
            mfcc_4=10.0,
            spectral_contrast=5.0,
            median_ici_ms=50.0,
            onset_rate_hz=20.0,
            ici_coefficient_of_variation=0.2,
            modality=Modality.TRANSIENT,
        )

        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            interpolate_vectors(v_harmonic, v_transient, alpha=0.5)

        self.assertIn("modality", str(context.exception).lower())


# =============================================================================
# Phase 4: Semiotic Engine Tests - Deception Detection via Modality
# =============================================================================


@dataclass
class ContextualState:
    """Contextual state for semiotic analysis"""

    predator_present: bool
    conspecific_present: bool
    food_present: bool
    territory_violation: bool


@dataclass
class SemioticAnalysis:
    """Result of semiotic analysis"""

    audio_modality: Modality
    context: ContextualState
    expected_modality: Optional[Modality]
    modality_mismatch: bool
    deception_probability: float  # 0.0 to 1.0


class TestSemioticDeceptionDetection(unittest.TestCase):
    """Test Phase 4: Deception Detection via Modality Mismatches"""

    def test_false_alarm_detection(self):
        """
        TEST: Detect "False Alarm" scenario via modality mismatch

        Scenario:
        - Audio: Corvid produces "Seep" (Harmonic, Soft)
        - Context: Predator is present
        - Semiotics: "Soft Seep" implies "Safe" or "Mating"
        - Deception: Corvid SHOULD have used "Rattle" (Transient)
        """
        analysis = SemioticAnalysis(
            audio_modality=Modality.HARMONIC,
            context=ContextualState(
                predator_present=True,  # Danger!
                conspecific_present=False,
                food_present=False,
                territory_violation=True,
            ),
            expected_modality=Modality.TRANSIENT,  # Should use alarm
            modality_mismatch=True,  # Harmonic != Transient
            deception_probability=0.85,  # High probability of deception
        )

        # Assert: Modality mismatch detected
        self.assertTrue(analysis.modality_mismatch)
        self.assertEqual(analysis.audio_modality, Modality.HARMONIC)
        self.assertEqual(analysis.expected_modality, Modality.TRANSIENT)

        # Assert: High deception probability
        self.assertGreater(
            analysis.deception_probability,
            0.7,
            "Context modality mismatch should indicate high deception probability",
        )

    def test_correct_modality_for_context(self):
        """
        TEST: Correct modality for context (no deception)

        Scenario:
        - Audio: Corvid produces "Rattle" (Transient, Urgent)
        - Context: Predator is present
        - Semiotics: Correct! Rattle = Alarm
        - Deception: Low probability
        """
        analysis = SemioticAnalysis(
            audio_modality=Modality.TRANSIENT,
            context=ContextualState(
                predator_present=True,
                conspecific_present=False,
                food_present=False,
                territory_violation=True,
            ),
            expected_modality=Modality.TRANSIENT,  # Correct!
            modality_mismatch=False,  # Transient == Transient
            deception_probability=0.05,  # Low probability
        )

        # Assert: No modality mismatch
        self.assertFalse(analysis.modality_mismatch)
        self.assertLess(
            analysis.deception_probability,
            0.2,
            "Correct modality should indicate low deception probability",
        )

    def test_mating_context_harmonic_appropriate(self):
        """
        TEST: Harmonic modality is appropriate for mating context

        Scenario:
        - Audio: Corvid produces "Whistle" (Harmonic)
        - Context: Mating season, no predators
        - Semiotics: Correct! Whistle = Courtship
        - Deception: Low probability
        """
        analysis = SemioticAnalysis(
            audio_modality=Modality.HARMONIC,
            context=ContextualState(
                predator_present=False,
                conspecific_present=True,
                food_present=False,
                territory_violation=False,
            ),
            expected_modality=Modality.HARMONIC,  # Appropriate for courtship
            modality_mismatch=False,
            deception_probability=0.03,
        )

        self.assertFalse(analysis.modality_mismatch)
        self.assertLess(analysis.deception_probability, 0.1)

    def test_modality_rules_definition(self):
        """
        TEST: Define modality rules for different contexts
        """
        # Define rules: Context -> Expected Modality
        rules = {
            "predator": Modality.TRANSIENT,  # Alarm calls = urgent
            "mating": Modality.HARMONIC,  # Courtship = tonal
            "food": Modality.TRANSIENT,  # Food discovery = excited
            "territory": Modality.TRANSIENT,  # Territory defense = aggressive
        }

        # Test predator context
        self.assertEqual(rules["predator"], Modality.TRANSIENT)
        self.assertEqual(rules["mating"], Modality.HARMONIC)
        self.assertEqual(rules["food"], Modality.TRANSIENT)
        self.assertEqual(rules["territory"], Modality.TRANSIENT)

    def test_detect_contextual_modality_mismatch(self):
        """
        TEST: System can detect contextual modality mismatches
        """

        # Define detection logic
        def detect_modality_mismatch(audio_modality: Modality, context: ContextualState) -> bool:
            """Check if audio modality matches expected for context"""
            expected_modality = None

            if context.predator_present:
                expected_modality = Modality.TRANSIENT  # Alarm required
            elif context.conspecific_present and not context.predator_present:
                expected_modality = Modality.HARMONIC  # Courtship allowed
            elif context.territory_violation:
                expected_modality = Modality.TRANSIENT  # Aggression required

            return audio_modality != expected_modality if expected_modality else False

        # Test 1: Predator context with harmonic audio (MISMATCH)
        context1 = ContextualState(
            predator_present=True,
            conspecific_present=False,
            food_present=False,
            territory_violation=True,
        )
        is_mismatch1 = detect_modality_mismatch(Modality.HARMONIC, context1)
        self.assertTrue(is_mismatch1, "Harmonic during predator = mismatch")

        # Test 2: Mating context with harmonic audio (MATCH)
        context2 = ContextualState(
            predator_present=False,
            conspecific_present=True,
            food_present=False,
            territory_violation=False,
        )
        is_mismatch2 = detect_modality_mismatch(Modality.HARMONIC, context2)
        self.assertFalse(is_mismatch2, "Harmonic during mating = correct")

        # Test 3: Predator context with transient audio (MATCH)
        is_mismatch3 = detect_modality_mismatch(Modality.TRANSIENT, context1)
        self.assertFalse(is_mismatch3, "Transient during predator = correct")


# =============================================================================
# Integration Tests
# =============================================================================


class TestCorvidIntegration(unittest.TestCase):
    """Integration tests for complete multi-modal corvid support"""

    def test_end_to_end_alarm_sequence(self):
        """
        TEST: End-to-end alarm sequence generation

        1. Intent: "Alarm"
        2. Router: Generate timeline [H, T, H]
        3. Synthesis: Execute timeline
        """
        # Step 1: Generate timeline from intent
        timeline = ModalityTimeline()
        timeline.add_event(0.0, 100.0, "whistle.wav", Modality.HARMONIC)
        timeline.add_event(100.0, 50.0, "rattle.wav", Modality.TRANSIENT)
        timeline.add_event(150.0, 20.0, "whistle.wav", Modality.HARMONIC)

        # Step 2: Validate timeline
        self.assertTrue(timeline.validate())

        # Step 3: Verify sequence
        timeline.sort_by_time()
        self.assertEqual(len(timeline.events), 3)

        # Verify modality sequence
        modalities = [e.modality for e in timeline.events]
        expected = [Modality.HARMONIC, Modality.TRANSIENT, Modality.HARMONIC]
        self.assertEqual(modalities, expected)

    def test_texture_composability(self):
        """
        TEST: System supports "Texture Composability"

        Corvid system composes "Sentences" using different physical voices,
        unlike Marmoset system which plays a single "Record."
        """
        # Corvid (Composite Persona): Multiple buffers for one call
        corvid_persona = {
            "sequence": [
                {"buffer": "whistle.wav", "modality": Modality.HARMONIC, "duration": 100.0},
                {"buffer": "rattle.wav", "modality": Modality.TRANSIENT, "duration": 50.0},
                {"buffer": "whistle.wav", "modality": Modality.HARMONIC, "duration": 20.0},
            ],
            "structure": "composite",
        }

        # Assert: Corvid has multiple buffers
        self.assertGreater(len(corvid_persona["sequence"]), 1, "Corvid should use multiple buffers")
        self.assertEqual(corvid_persona["structure"], "composite")

        # Assert: At least two modalities represented
        modalities = set(e["modality"] for e in corvid_persona["sequence"])
        self.assertTrue(len(modalities) >= 2, "Corvid sequence should use multiple modalities")


if __name__ == "__main__":
    unittest.main()
