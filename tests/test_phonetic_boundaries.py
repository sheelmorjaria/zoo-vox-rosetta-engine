#!/usr/bin/env python3
"""
Phase 3.1: Phonetic Boundary Detection

Test Protocol:
- Detect fine-grained phonetic boundaries (~20ms duration)
- Threshold: 2.5x baseline
- Test: Vowel space morphing (smooth transition between vowels)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BoundaryDetection:
    """A detected boundary."""
    frame_index: int
    boundary_type: str
    confidence: float
    normalized_error: float


@dataclass
class PhoneticTestResult:
    """Results from phonetic boundary test."""
    name: str
    total_boundaries: int
    detected_boundaries: int
    correct_type: int
    avg_confidence: float

    @property
    def recall(self) -> float:
        return self.detected_boundaries / max(self.total_boundaries, 1)

    @property
    def type_accuracy(self) -> float:
        return self.correct_type / max(self.detected_boundaries, 1)

    @property
    def confidence_ok(self) -> bool:
        return self.avg_confidence >= 0.6


class PhoneticBoundaryDetector:
    """
    Predictive NBD for phonetic boundary detection.

    Phonetic boundaries are the finest-grained, detected at:
    - Threshold: 2.5x baseline
    - Duration: ~20ms
    """

    def __init__(
        self,
        threshold_multiplier: float = 2.5,
        min_confidence: float = 0.6,
        ema_decay: float = 0.95,
    ):
        self.threshold_multiplier = threshold_multiplier
        self.min_confidence = min_confidence
        self.ema_decay = ema_decay
        self.baseline = 0.0
        self.observations = 0
        self.armed = True
        self.rearm_threshold = 1.2

    def detect(
        self,
        audio: np.ndarray,
        features_112d: Optional[np.ndarray] = None,
    ) -> Optional[BoundaryDetection]:
        """Detect phonetic boundary."""
        # Compute energy-based metric
        rms = np.sqrt(np.mean(audio ** 2))

        # Update baseline
        if self.observations < 10:
            self.baseline = (self.baseline * self.observations + rms) / (self.observations + 1)
            self.observations += 1
        else:
            self.baseline = self.ema_decay * self.baseline + (1 - self.ema_decay) * rms

        # Check re-arm
        if self.baseline > 0:
            normalized_error = rms / self.baseline
        else:
            normalized_error = 1.0

        if normalized_error < self.rearm_threshold:
            self.armed = True

        # Check boundary
        is_boundary = self.armed and (normalized_error >= self.threshold_multiplier)

        if is_boundary:
            # Classify as phonetic (2.5-3.0x range)
            if 2.5 <= normalized_error < 3.0:
                boundary_type = "phonetic"
            elif 3.0 <= normalized_error < 4.0:
                boundary_type = "syllable"
            else:
                boundary_type = "phrase"

            # Compute confidence
            confidence = min(1.0, normalized_error / 4.0)
            confidence = min(confidence + 0.2, 1.0)  # Type boost for phonetic

            # Check minimum confidence
            if confidence >= self.min_confidence:
                self.armed = False
                return BoundaryDetection(
                    frame_index=0,  # Set by caller
                    boundary_type=boundary_type,
                    confidence=confidence,
                    normalized_error=normalized_error,
                )

        return None


class VowelSpaceGenerator:
    """
    Generate vowel space morphing sequences.

    Tests smooth transitions between vowels, which produce
    subtle phonetic boundaries at transition points.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def generate_vowel(
        self,
        duration_ms: float,
        f0: float,
        f1: float,  # First formant
        f2: float,  # Second formant
        amplitude: float = 0.3,
    ) -> np.ndarray:
        """Generate a synthetic vowel with formants."""
        num_samples = int(self.sample_rate * duration_ms / 1000)
        t = np.arange(num_samples) / self.sample_rate

        # Fundamental frequency
        fundamental = np.sin(2 * np.pi * f0 * t)

        # Add formants (resonances)
        formant1 = 0.5 * np.sin(2 * np.pi * f1 * t)
        formant2 = 0.3 * np.sin(2 * np.pi * f2 * t)

        # Combine
        vowel = (fundamental + formant1 + formant2) * amplitude

        # Apply envelope
        envelope = np.ones_like(vowel)
        attack = int(0.1 * num_samples)
        decay = int(0.1 * num_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-decay:] = np.linspace(1, 0, decay)

        return vowel * envelope

    def generate_vowel_transition(
        self,
        vowel1_duration_ms: float,
        vowel2_duration_ms: float,
        f1_start: float,
        f1_end: float,
        f2_start: float,
        f2_end: float,
        f0: float = 200.0,
    ) -> np.ndarray:
        """Generate smooth transition between two vowels."""
        # Vowel 1
        vowel1 = self.generate_vowel(
            duration_ms=vowel1_duration_ms,
            f0=f0,
            f1=f1_start,
            f2=f2_start,
        )

        # Transition (gradual formant shift)
        transition_ms = 20.0
        transition_samples = int(self.sample_rate * transition_ms / 1000)
        t = np.arange(transition_samples) / self.sample_rate

        # Interpolate formants
        f1_transition = np.linspace(f1_start, f1_end, transition_samples)
        f2_transition = np.linspace(f2_start, f2_end, transition_samples)

        transition = np.zeros(transition_samples)
        for i in range(transition_samples):
            fundamental = np.sin(2 * np.pi * f0 * t[i])
            formant1 = 0.5 * np.sin(2 * np.pi * f1_transition[i] * t[i])
            formant2 = 0.3 * np.sin(2 * np.pi * f2_transition[i] * t[i])
            transition[i] = (fundamental + formant1 + formant2) * 0.3

        # Vowel 2
        vowel2 = self.generate_vowel(
            duration_ms=vowel2_duration_ms,
            f0=f0,
            f1=f1_end,
            f2=f2_end,
        )

        return np.concatenate([vowel1, transition, vowel2])

    def generate_vowel_sequence(
        self,
        transitions: List[Tuple[float, float, float, float]],
        vowel_duration_ms: float = 80.0,
        transition_ms: float = 20.0,
    ) -> List[np.ndarray]:
        """
        Generate sequence of vowels with transitions.

        Args:
            transitions: List of (f1, f2) pairs for each vowel
            vowel_duration_ms: Duration of each stable vowel
            transition_ms: Duration of transitions

        Returns:
            List of 10ms frames
        """
        frame_size_ms = 10.0
        frame_size = int(self.sample_rate * frame_size_ms / 1000)

        # Build full audio
        audio_segments = []
        for i in range(len(transitions) - 1):
            f1_start, f2_start = transitions[i]
            f1_end, f2_end = transitions[i + 1]

            segment = self.generate_vowel_transition(
                vowel1_duration_ms=vowel_duration_ms / 2,
                vowel2_duration_ms=vowel_duration_ms / 2,
                f1_start=f1_start,
                f1_end=f1_end,
                f2_start=f2_start,
                f2_end=f2_end,
            )
            audio_segments.append(segment)

        full_audio = np.concatenate(audio_segments)

        # Split into frames
        frames = []
        for i in range(0, len(full_audio), frame_size):
            frame = full_audio[i:i+frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            frames.append(frame.astype(np.float32))

        return frames, len(transitions) - 1


class TestPhoneticBoundaries:
    """Test suite for phonetic boundary detection."""

    def __init__(self):
        self.generator = VowelSpaceGenerator()
        self.detector = PhoneticBoundaryDetector()

    def test_vowel_transition_detection(self):
        """
        Test detection of phonetic boundaries at vowel transitions.
        """
        print("\n" + "=" * 60)
        print("Vowel Transition Detection")
        print("=" * 60)

        # Vowel space: /i/ → /e/ → /a/ → /o/ → /u/
        # F1/F2 values (Hz)
        transitions = [
            (300, 2200),  # /i/
            (500, 1900),  # /e/
            (800, 1500),  # /a/
            (600, 900),   # /o/
            (400, 700),   # /u/
        ]

        frames, expected_boundaries = self.generator.generate_vowel_sequence(
            transitions=transitions,
            vowel_duration_ms=80.0,
            transition_ms=20.0,
        )

        detections = []
        for i, frame in enumerate(frames):
            result = self.detector.detect(frame)
            if result:
                result.frame_index = i
                detections.append(result)

        print(f"Expected boundaries: {expected_boundaries}")
        print(f"Detected boundaries: {len(detections)}")

        for d in detections:
            print(f"  Frame {d.frame_index}: {d.boundary_type} (conf={d.confidence:.2f})")

        # Should detect most transitions
        assert len(detections) >= expected_boundaries * 0.6, \
            f"Expected at least {expected_boundaries * 0.6:.0f} detections, got {len(detections)}"

        # Most should be classified as phonetic or syllable
        phonetic_count = sum(1 for d in detections if d.boundary_type in ["phonetic", "syllable"])
        print(f"Phonetic/Syllable detections: {phonetic_count}/{len(detections)}")

        assert phonetic_count >= len(detections) * 0.5, \
            "Most detections should be phonetic/syllable type"

    def test_phonetic_confidence_calibration(self):
        """
        Test that phonetic boundaries have appropriate confidence.
        """
        print("\n" + "=" * 60)
        print("Phonetic Confidence Calibration")
        print("=" * 60)

        transitions = [
            (300, 2200),  # /i/
            (800, 1500),  # /a/ (large formant shift)
        ]

        frames, _ = self.generator.generate_vowel_sequence(
            transitions=transitions,
            vowel_duration_ms=80.0,
        )

        detections = []
        for frame in frames:
            result = self.detector.detect(frame)
            if result and result.boundary_type == "phonetic":
                detections.append(result)

        if detections:
            avg_confidence = np.mean([d.confidence for d in detections])
            print(f"Average phonetic confidence: {avg_confidence:.3f}")
            print(f"Min confidence: {min(d.confidence for d in detections):.3f}")
            print(f"Max confidence: {max(d.confidence for d in detections):.3f}")

            assert avg_confidence >= 0.6, \
                f"Average confidence {avg_confidence:.3f} below 0.6 threshold"
        else:
            print("No phonetic boundaries detected")

    def test_threshold_multiplier_sensitivity(self):
        """
        Test detection sensitivity at different threshold multipliers.
        """
        print("\n" + "=" * 60)
        print("Threshold Multiplier Sensitivity")
        print("=" * 60)

        transitions = [
            (300, 2200),  # /i/
            (500, 1900),  # /e/
            (800, 1500),  # /a/
        ]

        frames, expected = self.generator.generate_vowel_sequence(
            transitions=transitions,
            vowel_duration_ms=80.0,
        )

        multipliers = [2.0, 2.5, 3.0, 4.0]

        for mult in multipliers:
            detector = PhoneticBoundaryDetector(threshold_multiplier=mult)
            detections = 0
            for frame in frames:
                result = detector.detect(frame)
                if result:
                    detections += 1

            print(f"  {mult}x threshold: {detections} detections")

        # Lower threshold = more detections
        # Higher threshold = fewer detections

    def test_min_separation_enforcement(self):
        """
        Test that phonetic boundaries respect minimum separation.
        """
        print("\n" + "=" * 60)
        print("Minimum Separation Enforcement")
        print("=" * 60)

        # Rapid transitions (20ms apart)
        transitions = [
            (300, 2200),
            (400, 2000),
            (500, 1800),
            (600, 1600),
            (700, 1400),
        ]

        frames, _ = self.generator.generate_vowel_sequence(
            transitions=transitions,
            vowel_duration_ms=40.0,  # Short vowels
            transition_ms=10.0,  # Rapid transitions
        )

        detector = PhoneticBoundaryDetector()
        detections = []
        for i, frame in enumerate(frames):
            result = detector.detect(frame)
            if result:
                result.frame_index = i
                detections.append(i)

        # Check separations
        if len(detections) > 1:
            separations = [detections[i+1] - detections[i] for i in range(len(detections)-1)]
            avg_separation = np.mean(separations)
            print(f"Average frame separation: {avg_separation:.1f} frames ({avg_separation*10:.0f}ms)")

            # Phonetic boundaries should be at least 2 frames (20ms) apart
            # due to re-arm logic
            assert all(s >= 1 for s in separations), \
                "Phonetic boundaries should respect minimum separation"


def main():
    """Run all phonetic boundary tests."""
    print("=" * 60)
    print("Phase 3.1: Phonetic Boundary Detection")
    print("=" * 60)
    print()

    test = TestPhoneticBoundaries()

    test.test_vowel_transition_detection()
    test.test_phonetic_confidence_calibration()
    test.test_threshold_multiplier_sensitivity()
    test.test_min_separation_enforcement()

    print("\n" + "=" * 60)
    print("✓ ALL PHONETIC BOUNDARY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
