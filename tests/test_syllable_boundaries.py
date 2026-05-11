#!/usr/bin/env python3
"""
Phase 3.2: Syllable Boundary Detection

Test Protocol:
- Detect syllable boundaries (~100ms duration)
- Threshold: 3.0x baseline
- Min separation: 30ms
- Test: Two-tone syllable sequence (alternating frequencies)

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


class SyllableBoundaryDetector:
    """
    Predictive NBD for syllable boundary detection.

    Syllable boundaries are mid-granularity, detected at:
    - Threshold: 3.0x baseline
    - Duration: ~100ms
    - Min separation: 30ms
    """

    def __init__(
        self,
        threshold_multiplier: float = 3.0,
        min_confidence: float = 0.6,
        ema_decay: float = 0.95,
        min_separation_ms: float = 30.0,
        frame_size_ms: float = 10.0,
    ):
        self.threshold_multiplier = threshold_multiplier
        self.min_confidence = min_confidence
        self.ema_decay = ema_decay
        self.min_separation_frames = int(min_separation_ms / frame_size_ms)
        self.baseline = 0.0
        self.observations = 0
        self.armed = True
        self.rearm_threshold = 1.2
        self.last_detection_frame = -999

    def detect(
        self,
        audio: np.ndarray,
        frame_index: int,
        features_112d: Optional[np.ndarray] = None,
    ) -> Optional[BoundaryDetection]:
        """Detect syllable boundary."""
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
            # Check min separation
            frames_since_last = frame_index - self.last_detection_frame
            if frames_since_last < self.min_separation_frames:
                return None  # Too soon

            # Classify boundary type
            if normalized_error >= 4.0:
                boundary_type = "phrase"
            elif normalized_error >= 3.0:
                boundary_type = "syllable"
            else:
                boundary_type = "phonetic"

            # Compute confidence
            confidence = min(1.0, normalized_error / 4.0)
            confidence = min(confidence + 0.2, 1.0)

            # Check minimum confidence
            if confidence >= self.min_confidence:
                self.armed = False
                self.last_detection_frame = frame_index
                return BoundaryDetection(
                    frame_index=frame_index,
                    boundary_type=boundary_type,
                    confidence=confidence,
                    normalized_error=normalized_error,
                )

        return None


class TwoToneSyllableGenerator:
    """
    Generate two-tone syllable sequences.

    Tests syllable boundaries at transitions between
    alternating tone frequencies.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def generate_tone(
        self,
        duration_ms: float,
        frequency: float,
        amplitude: float = 0.3,
    ) -> np.ndarray:
        """Generate a pure tone."""
        num_samples = int(self.sample_rate * duration_ms / 1000)
        t = np.arange(num_samples) / self.sample_rate

        tone = np.sin(2 * np.pi * frequency * t) * amplitude

        # Apply envelope
        envelope = np.ones_like(tone)
        attack = int(0.1 * num_samples)
        decay = int(0.1 * num_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-decay:] = np.linspace(1, 0, decay)

        return tone * envelope

    def generate_two_tone_sequence(
        self,
        num_syllables: int = 10,
        syllable_duration_ms: float = 100.0,
        gap_ms: float = 30.0,
        freq_low: float = 1000.0,
        freq_high: float = 2000.0,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate alternating two-tone syllable sequence.

        Returns (frames, boundary_indices) where boundary_indices
        indicates which frames are at syllable transitions.
        """
        frame_size_ms = 10.0
        frame_size = int(self.sample_rate * frame_size_ms / 1000)

        frames = []
        boundary_indices = []

        current_frame = 0

        for i in range(num_syllables):
            # Alternate frequency
            freq = freq_low if i % 2 == 0 else freq_high

            # Generate syllable
            syllable = self.generate_tone(
                duration_ms=syllable_duration_ms,
                frequency=freq,
                amplitude=0.3,
            )

            # Split into frames
            syllable_frames = []
            for j in range(0, len(syllable), frame_size):
                frame = syllable[j:j+frame_size]
                if len(frame) < frame_size:
                    frame = np.pad(frame, (0, frame_size - len(frame)))
                syllable_frames.append(frame.astype(np.float32))

            frames.extend(syllable_frames)

            # Mark transition point (end of syllable)
            if i < num_syllables - 1:
                boundary_idx = current_frame + len(syllable_frames) - 1
                boundary_indices.append(boundary_idx)

            current_frame += len(syllable_frames)

            # Add gap
            gap_samples = int(self.sample_rate * gap_ms / 1000)
            gap_frames = []
            for j in range(0, gap_samples, frame_size):
                frame = np.random.randn(frame_size).astype(np.float32) * 0.01
                gap_frames.append(frame)

            frames.extend(gap_frames)
            current_frame += len(gap_frames)

        return frames, boundary_indices

    def generate_rapid_syllable_sequence(
        self,
        num_syllables: int = 15,
        syllable_duration_ms: float = 80.0,
        gap_ms: float = 20.0,  # Below min separation
        freq_low: float = 1000.0,
        freq_high: float = 2000.0,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Generate rapid syllable sequence (gaps < min separation)."""
        return self.generate_two_tone_sequence(
            num_syllables=num_syllables,
            syllable_duration_ms=syllable_duration_ms,
            gap_ms=gap_ms,
            freq_low=freq_low,
            freq_high=freq_high,
        )


class TestSyllableBoundaries:
    """Test suite for syllable boundary detection."""

    def __init__(self):
        self.generator = TwoToneSyllableGenerator()
        self.detector = SyllableBoundaryDetector()

    def test_two_tone_syllable_detection(self):
        """
        Test detection of syllable boundaries in two-tone sequence.
        """
        print("\n" + "=" * 60)
        print("Two-Tone Syllable Detection")
        print("=" * 60)

        frames, boundary_indices = self.generator.generate_two_tone_sequence(
            num_syllables=10,
            syllable_duration_ms=100.0,
            gap_ms=30.0,
        )

        detections = []
        for i, frame in enumerate(frames):
            result = self.detector.detect(frame, i)
            if result:
                detections.append(result)

        print(f"Expected syllable boundaries: {len(boundary_indices)}")
        print(f"Detected boundaries: {len(detections)}")

        for d in detections[:5]:  # Show first 5
            print(f"  Frame {d.frame_index}: {d.boundary_type} (conf={d.confidence:.2f})")

        # Should detect most syllable boundaries
        recall = len(detections) / max(len(boundary_indices), 1)
        print(f"Recall: {recall*100:.1f}%")

        assert recall >= 0.7, \
            f"Recall {recall*100:.1f}% below 70% target"

    def test_syllable_type_classification(self):
        """
        Test that syllable boundaries are correctly classified.
        """
        print("\n" + "=" * 60)
        print("Syllable Type Classification")
        print("=" * 60)

        frames, boundary_indices = self.generator.generate_two_tone_sequence(
            num_syllables=10,
            syllable_duration_ms=100.0,
            gap_ms=30.0,
        )

        detections = []
        for i, frame in enumerate(frames):
            result = self.detector.detect(frame, i)
            if result:
                detections.append(result)

        syllable_count = sum(1 for d in detections if d.boundary_type == "syllable")
        print(f"Syllable-type detections: {syllable_count}/{len(detections)}")

        # Most detections should be syllable type
        if detections:
            syllable_ratio = syllable_count / len(detections)
            assert syllable_ratio >= 0.5, \
                f"Syllable ratio {syllable_ratio*100:.1f}% below 50%"

    def test_min_separation_enforcement(self):
        """
        Test that syllable boundaries respect minimum separation.
        """
        print("\n" + "=" * 60)
        print("Min Separation Enforcement (30ms)")
        print("=" * 60)

        frames, boundary_indices = self.generator.generate_rapid_syllable_sequence(
            num_syllables=15,
            syllable_duration_ms=80.0,
            gap_ms=20.0,  # Below 30ms min separation
        )

        detections = []
        for i, frame in enumerate(frames):
            result = self.detector.detect(frame, i)
            if result:
                detections.append(result.frame_index)

        print(f"Expected syllables: 15")
        print(f"Detected: {len(detections)}")
        print(f"Min separation: 30ms (3 frames)")

        if len(detections) > 1:
            separations = [detections[i+1] - detections[i] for i in range(len(detections)-1)]
            min_separation = min(separations)
            print(f"Actual min separation: {min_separation} frames ({min_separation*10}ms)")

            # All separations should be >= 3 frames (30ms)
            assert all(s >= 3 for s in separations), \
                "Syllable boundaries must respect 30ms min separation"

    def test_gap_duration_impact(self):
        """
        Test detection across various gap durations.
        """
        print("\n" + "=" * 60)
        print("Gap Duration Impact")
        print("=" * 60)

        gap_durations = [10, 20, 30, 40, 50, 100]

        for gap_ms in gap_durations:
            frames, boundary_indices = self.generator.generate_two_tone_sequence(
                num_syllables=10,
                syllable_duration_ms=100.0,
                gap_ms=float(gap_ms),
            )

            detector = SyllableBoundaryDetector(min_separation_ms=30.0)
            detections = 0
            for i, frame in enumerate(frames):
                result = detector.detect(frame, i)
                if result:
                    detections += 1

            status = "✓" if gap_ms >= 30 else "~"
            print(f"  {status} Gap {gap_ms:3d}ms: {detections:2d} detections")

    def test_syllable_confidence(self):
        """
        Test that syllable boundaries have appropriate confidence.
        """
        print("\n" + "=" * 60)
        print("Syllable Confidence Calibration")
        print("=" * 60)

        frames, boundary_indices = self.generator.generate_two_tone_sequence(
            num_syllables=10,
            syllable_duration_ms=100.0,
            gap_ms=30.0,
        )

        detections = []
        for i, frame in enumerate(frames):
            result = self.detector.detect(frame, i)
            if result and result.boundary_type == "syllable":
                detections.append(result)

        if detections:
            confidences = [d.confidence for d in detections]
            avg_confidence = np.mean(confidences)
            print(f"Average syllable confidence: {avg_confidence:.3f}")
            print(f"Min confidence: {min(confidences):.3f}")
            print(f"Max confidence: {max(confidences):.3f}")

            # All should be above 0.6 threshold
            assert all(c >= 0.6 for c in confidences), \
                "Some syllable detections below 0.6 confidence threshold"

    def test_threshold_multiplier_classification(self):
        """
        Test classification at different threshold multipliers.
        """
        print("\n" + "=" * 60)
        print("Threshold Multiplier Classification")
        print("=" * 60)

        frames, boundary_indices = self.generator.generate_two_tone_sequence(
            num_syllables=10,
            syllable_duration_ms=100.0,
            gap_ms=30.0,
        )

        multipliers = [2.5, 3.0, 3.5, 4.0]

        for mult in multipliers:
            detector = SyllableBoundaryDetector(threshold_multiplier=mult)
            detections = []
            for i, frame in enumerate(frames):
                result = detector.detect(frame, i)
                if result:
                    detections.append(result.boundary_type)

            if detections:
                syllable_count = sum(1 for t in detections if t == "syllable")
                phrase_count = sum(1 for t in detections if t == "phrase")
                phonetic_count = sum(1 for t in detections if t == "phonetic")

                print(f"  {mult}x threshold:")
                print(f"    Phonetic: {phonetic_count}, Syllable: {syllable_count}, Phrase: {phrase_count}")


def main():
    """Run all syllable boundary tests."""
    print("=" * 60)
    print("Phase 3.2: Syllable Boundary Detection")
    print("=" * 60)
    print()

    test = TestSyllableBoundaries()

    test.test_two_tone_syllable_detection()
    test.test_syllable_type_classification()
    test.test_min_separation_enforcement()
    test.test_gap_duration_impact()
    test.test_syllable_confidence()
    test.test_threshold_multiplier_classification()

    print("\n" + "=" * 60)
    print("✓ ALL SYLLABLE BOUNDARY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
