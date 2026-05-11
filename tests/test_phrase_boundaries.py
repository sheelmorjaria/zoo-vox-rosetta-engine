#!/usr/bin/env python3
"""
Phase 3.3: Phrase Boundary Detection

Test Protocol:
- Detect phrase boundaries (~350ms duration)
- Threshold: 4.0x baseline
- Test: Whistle-to-noise transition (clear acoustic change)

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


class PhraseBoundaryDetector:
    """
    Predictive NBD for phrase boundary detection.

    Phrase boundaries are the coarsest granularity, detected at:
    - Threshold: 4.0x baseline
    - Duration: ~350ms
    """

    def __init__(
        self,
        threshold_multiplier: float = 4.0,
        min_confidence: float = 0.6,
        ema_decay: float = 0.95,
        min_separation_ms: float = 100.0,
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
        """Detect phrase boundary."""
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

            # Classify as phrase (highest threshold)
            boundary_type = "phrase"

            # Compute confidence
            confidence = min(1.0, normalized_error / 5.0)  # Higher scale for phrase

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


class WhistleNoiseGenerator:
    """
    Generate whistle-to-noise transition sequences.

    Tests phrase boundaries at clear acoustic transitions
    between tonal whistles and noise bursts.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def generate_whistle(
        self,
        duration_ms: float,
        frequency: float,
        amplitude: float = 0.4,
        fm_depth: float = 0.0,
        fm_rate: float = 0.0,
    ) -> np.ndarray:
        """Generate a whistle (tonal vocalization)."""
        num_samples = int(self.sample_rate * duration_ms / 1000)
        t = np.arange(num_samples) / self.sample_rate

        # Base tone
        phase = 2 * np.pi * frequency * t

        # Add frequency modulation if specified
        if fm_depth > 0 and fm_rate > 0:
            phase += 2 * np.pi * fm_depth * np.sin(2 * np.pi * fm_rate * t) / fm_rate

        whistle = np.sin(phase) * amplitude

        # Apply envelope
        envelope = np.ones_like(whistle)
        attack = int(0.05 * num_samples)
        decay = int(0.1 * num_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-decay:] = np.linspace(1, 0, decay)

        return whistle * envelope

    def generate_noise_burst(
        self,
        duration_ms: float,
        amplitude: float = 0.2,
        color: str = "white",
    ) -> np.ndarray:
        """Generate a noise burst."""
        num_samples = int(self.sample_rate * duration_ms / 1000)

        if color == "white":
            noise = np.random.randn(num_samples).astype(np.float32)
        elif color == "pink":
            # Simple approximation of pink noise
            noise = np.random.randn(num_samples).astype(np.float32)
            # Apply 1/f filtering (simplified)
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            from scipy import signal
            noise = signal.lfilter(b, a, noise)
        else:
            noise = np.random.randn(num_samples).astype(np.float32)

        noise = noise / np.max(np.abs(noise)) * amplitude

        # Apply envelope
        envelope = np.ones_like(noise)
        attack = int(0.1 * num_samples)
        decay = int(0.1 * num_samples)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-decay:] = np.linspace(1, 0, decay)

        return noise * envelope

    def generate_phrase_sequence(
        self,
        num_phrases: int = 5,
        whistle_duration_ms: float = 300.0,
        noise_duration_ms: float = 50.0,
        gap_ms: float = 100.0,
        whistle_freq: float = 5000.0,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate sequence of alternating whistles and noise bursts.

        Returns (frames, boundary_indices) where boundary_indices
        indicates which frames are at phrase transitions.
        """
        frame_size_ms = 10.0
        frame_size = int(self.sample_rate * frame_size_ms / 1000)

        frames = []
        boundary_indices = []

        current_frame = 0

        for i in range(num_phrases):
            # Generate whistle (tonal)
            whistle = self.generate_whistle(
                duration_ms=whistle_duration_ms,
                frequency=whistle_freq,
                amplitude=0.4,
                fm_depth=200.0,
                fm_rate=5.0,
            )

            # Split into frames
            for j in range(0, len(whistle), frame_size):
                frame = whistle[j:j+frame_size]
                if len(frame) < frame_size:
                    frame = np.pad(frame, (0, frame_size - len(frame)))
                frames.append(frame.astype(np.float32))
                current_frame += 1

            # Mark transition point (end of whistle = phrase boundary)
            if i < num_phrases - 1:
                boundary_indices.append(current_frame - 1)

            # Add noise burst (transition)
            noise = self.generate_noise_burst(
                duration_ms=noise_duration_ms,
                amplitude=0.2,
                color="pink",
            )

            for j in range(0, len(noise), frame_size):
                frame = noise[j:j+frame_size]
                if len(frame) < frame_size:
                    frame = np.pad(frame, (0, frame_size - len(frame)))
                frames.append(frame.astype(np.float32))
                current_frame += 1

            # Add gap (silence)
            gap_samples = int(self.sample_rate * gap_ms / 1000)
            for j in range(0, gap_samples, frame_size):
                frame = np.random.randn(frame_size).astype(np.float32) * 0.005
                frames.append(frame.astype(np.float32))
                current_frame += 1

        return frames, boundary_indices

    def generate_long_phrase_sequence(
        self,
        num_phrases: int = 8,
        phrase_duration_ms: float = 350.0,
        gap_ms: float = 150.0,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Generate long phrase sequence (>300ms per phrase)."""
        return self.generate_phrase_sequence(
            num_phrases=num_phrases,
            whistle_duration_ms=phrase_duration_ms,
            noise_duration_ms=50.0,
            gap_ms=gap_ms,
        )


class TestPhraseBoundaries:
    """Test suite for phrase boundary detection."""

    def __init__(self):
        self.generator = WhistleNoiseGenerator()
        self.detector = PhraseBoundaryDetector()

    def test_whistle_noise_phrase_detection(self):
        """
        Test detection of phrase boundaries at whistle-to-noise transitions.
        """
        print("\n" + "=" * 60)
        print("Whistle-to-Noise Phrase Detection")
        print("=" * 60)

        frames, boundary_indices = self.generator.generate_phrase_sequence(
            num_phrases=5,
            whistle_duration_ms=300.0,
            noise_duration_ms=50.0,
            gap_ms=100.0,
        )

        detections = []
        for i, frame in enumerate(frames):
            result = self.detector.detect(frame, i)
            if result:
                detections.append(result)

        print(f"Expected phrase boundaries: {len(boundary_indices)}")
        print(f"Detected boundaries: {len(detections)}")

        for d in detections:
            print(f"  Frame {d.frame_index}: {d.boundary_type} (conf={d.confidence:.2f})")

        # Should detect phrase boundaries
        recall = len(detections) / max(len(boundary_indices), 1)
        print(f"Recall: {recall*100:.1f}%")

        assert recall >= 0.7, \
            f"Recall {recall*100:.1f}% below 70% target"

    def test_phrase_type_classification(self):
        """
        Test that phrase boundaries are correctly classified.
        """
        print("\n" + "=" * 60)
        print("Phrase Type Classification")
        print("=" * 60)

        frames, boundary_indices = self.generator.generate_phrase_sequence(
            num_phrases=5,
            whistle_duration_ms=300.0,
        )

        detections = []
        for i, frame in enumerate(frames):
            result = self.detector.detect(frame, i)
            if result:
                detections.append(result)

        phrase_count = sum(1 for d in detections if d.boundary_type == "phrase")
        print(f"Phrase-type detections: {phrase_count}/{len(detections)}")

        # Most detections at 4.0x threshold should be phrase type
        if detections:
            phrase_ratio = phrase_count / len(detections)
            assert phrase_ratio >= 0.8, \
                f"Phrase ratio {phrase_ratio*100:.1f}% below 80%"

    def test_long_phrase_detection(self):
        """
        Test detection of longer phrases (>300ms).
        """
        print("\n" + "=" * 60)
        print("Long Phrase Detection (>300ms)")
        print("=" * 60)

        frames, boundary_indices = self.generator.generate_long_phrase_sequence(
            num_phrases=6,
            phrase_duration_ms=350.0,
            gap_ms=150.0,
        )

        detector = PhraseBoundaryDetector()
        detections = []
        for i, frame in enumerate(frames):
            result = detector.detect(frame, i)
            if result:
                detections.append(result)

        print(f"Expected phrase boundaries: {len(boundary_indices)}")
        print(f"Detected: {len(detections)}")

        # Should still detect boundaries
        assert len(detections) >= len(boundary_indices) * 0.6, \
            "Long phrases should still be detected"

    def test_phrase_confidence(self):
        """
        Test that phrase boundaries have appropriate confidence.
        """
        print("\n" + "=" * 60)
        print("Phrase Confidence Calibration")
        print("=" * 60)

        frames, boundary_indices = self.generator.generate_phrase_sequence(
            num_phrases=5,
            whistle_duration_ms=300.0,
        )

        detections = []
        for i, frame in enumerate(frames):
            result = self.detector.detect(frame, i)
            if result and result.boundary_type == "phrase":
                detections.append(result)

        if detections:
            confidences = [d.confidence for d in detections]
            avg_confidence = np.mean(confidences)
            print(f"Average phrase confidence: {avg_confidence:.3f}")
            print(f"Min confidence: {min(confidences):.3f}")
            print(f"Max confidence: {max(confidences):.3f}")

            # Phrase boundaries should have high confidence
            assert all(c >= 0.6 for c in confidences), \
                "Some phrase detections below 0.6 confidence threshold"

            # Average should be higher than syllable/phonetic
            assert avg_confidence >= 0.7, \
                f"Average phrase confidence {avg_confidence:.3f} below 0.7"

    def test_threshold_sensitivity(self):
        """
        Test detection at different threshold multipliers.
        """
        print("\n" + "=" * 60)
        print("Threshold Multiplier Sensitivity")
        print("=" * 60)

        frames, boundary_indices = self.generator.generate_phrase_sequence(
            num_phrases=5,
            whistle_duration_ms=300.0,
        )

        multipliers = [3.0, 3.5, 4.0, 4.5, 5.0]

        print("Phrase detection at various thresholds:")
        for mult in multipliers:
            detector = PhraseBoundaryDetector(threshold_multiplier=mult)
            detections = 0
            for i, frame in enumerate(frames):
                result = detector.detect(frame, i)
                if result:
                    detections += 1

            print(f"  {mult}x threshold: {detections} detections")

        # Higher threshold = fewer detections (more selective)

    def test_min_separation_enforcement(self):
        """
        Test that phrase boundaries respect minimum separation.
        """
        print("\n" + "=" * 60)
        print("Min Separation Enforcement (100ms)")
        print("=" * 60)

        frames, boundary_indices = self.generator.generate_phrase_sequence(
            num_phrases=8,
            whistle_duration_ms=200.0,
            noise_duration_ms=30.0,
            gap_ms=50.0,  # Short gap
        )

        detector = PhraseBoundaryDetector(min_separation_ms=100.0)
        detections = []
        for i, frame in enumerate(frames):
            result = detector.detect(frame, i)
            if result:
                detections.append(result.frame_index)

        if len(detections) > 1:
            separations = [detections[i+1] - detections[i] for i in range(len(detections)-1)]
            min_separation = min(separations)
            print(f"Actual min separation: {min_separation} frames ({min_separation*10}ms)")

            # All separations should be >= 10 frames (100ms)
            assert all(s >= 10 for s in separations), \
                "Phrase boundaries must respect 100ms min separation"


def main():
    """Run all phrase boundary tests."""
    print("=" * 60)
    print("Phase 3.3: Phrase Boundary Detection")
    print("=" * 60)
    print()

    test = TestPhraseBoundaries()

    test.test_whistle_noise_phrase_detection()
    test.test_phrase_type_classification()
    test.test_long_phrase_detection()
    test.test_phrase_confidence()
    test.test_threshold_sensitivity()
    test.test_min_separation_enforcement()

    print("\n" + "=" * 60)
    print("✓ ALL PHRASE BOUNDARY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
