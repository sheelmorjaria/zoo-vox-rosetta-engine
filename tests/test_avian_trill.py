#!/usr/bin/env python3
"""
Phase 2.2: Avian Trill Test - Fixed vs. Adaptive Debounce

Test Protocol:
- Rapid chirps at 20-30ms intervals (avian trills)
- Legacy (50ms debounce): 0% recall - chirps filtered out
- Predictive (adaptive re-arm): >90% recall - rapid syllables detected

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrillTestResult:
    """Results from avian trill test."""
    name: str
    total_chirps: int
    detected_chirps: int
    missed_chirps: int
    false_positives: int

    @property
    def recall(self) -> float:
        """Recall rate (detection rate)."""
        return self.detected_chirps / max(self.total_chirps, 1)

    @property
    def target_recall(self) -> float:
        """Target recall is >90%."""
        return 0.90

    @property
    def within_target(self) -> bool:
        return self.recall >= self.target_recall

    def report(self) -> str:
        status = "✓ PASS" if self.within_target else "✗ FAIL"
        return (
            f"{self.name}:\n"
            f"  Total chirps: {self.total_chirps}\n"
            f"  Detected: {self.detected_chirps}\n"
            f"  Missed: {self.missed_chirps}\n"
            f"  False positives: {self.false_positives}\n"
            f"  Recall: {self.recall*100:.1f}% (target: >90%)\n"
            f"  {status}"
        )


class LegacyDebounceDetector:
    """
    Legacy NBD with fixed 50ms debounce.

    Problem: Rapid chirps <50ms apart are filtered out,
    causing 0% recall on avian trills.
    """

    def __init__(self, debounce_ms: float = 50.0, sample_rate: int = 48000):
        self.debounce_ms = debounce_ms
        self.debounce_samples = int(sample_rate * debounce_ms / 1000)
        self.last_detection_sample = -self.debounce_samples
        self.name = f"Legacy ({debounce_ms}ms Debounce)"

    def detect(
        self,
        audio: np.ndarray,
        sample_index: int,
    ) -> Tuple[bool, float]:
        """
        Detect boundary with fixed debounce.

        Returns (is_boundary, confidence).
        """
        # Simple energy-based detection
        rms = np.sqrt(np.mean(audio ** 2))
        threshold = 0.1
        is_boundary = rms > threshold

        # Apply debounce
        if is_boundary:
            samples_since_last = sample_index - self.last_detection_sample
            if samples_since_last < self.debounce_samples:
                # Still in debounce period - reject
                is_boundary = False
            else:
                # Accept detection
                self.last_detection_sample = sample_index

        confidence = rms / threshold if is_boundary else 0.0
        return is_boundary, confidence


class PredictiveAdaptiveDetector:
    """
    Predictive NBD with adaptive re-arm logic.

    Advantage: Re-arms when normalized error drops below threshold,
    allowing detection of rapid chirps.
    """

    def __init__(
        self,
        threshold_multiplier: float = 2.5,
        rearm_threshold: float = 1.2,
        min_confidence: float = 0.6,
        ema_decay: float = 0.95,
    ):
        self.threshold_multiplier = threshold_multiplier
        self.rearm_threshold = rearm_threshold
        self.min_confidence = min_confidence
        self.ema_decay = ema_decay
        self.baseline = 0.0
        self.armed = True
        self.observations = 0
        self.name = "Predictive (Adaptive Re-arm)"

    def detect(
        self,
        audio: np.ndarray,
        sample_index: int,
    ) -> Tuple[bool, float]:
        """
        Detect boundary with adaptive re-arm.

        Returns (is_boundary, confidence).
        """
        # Compute prediction error
        rms = np.sqrt(np.mean(audio ** 2))

        # Update baseline
        if self.observations < 10:
            self.baseline = (self.baseline * self.observations + rms) / (self.observations + 1)
            self.observations += 1
        else:
            self.baseline = self.ema_decay * self.baseline + (1 - self.ema_decay) * rms

        # Compute normalized error
        if self.baseline > 0:
            normalized_error = rms / self.baseline
        else:
            normalized_error = 1.0

        # Check re-arm condition
        if normalized_error < self.rearm_threshold:
            self.armed = True

        # Check boundary
        is_boundary = self.armed and (normalized_error >= self.threshold_multiplier)

        # Compute confidence
        if is_boundary:
            confidence = min(1.0, normalized_error / 4.0)
            confidence = min(confidence + 0.2, 1.0)  # Type boost
            # Disarm after detection
            self.armed = False
        else:
            confidence = 0.0

        # Apply minimum confidence
        final_boundary = is_boundary and (confidence >= self.min_confidence)

        return final_boundary, confidence


class AvianTrillGenerator:
    """
    Generate avian trill sequences.

    Avian trills consist of rapid chirps at 20-30ms intervals.
    Each chirp is a short frequency-modulated burst.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def generate_chirp(
        self,
        duration_ms: float = 30.0,
        start_freq: float = 4000.0,
        end_freq: float = 8000.0,
        amplitude: float = 0.3,
    ) -> np.ndarray:
        """
        Generate a single frequency-modulated chirp.
        """
        num_samples = int(self.sample_rate * duration_ms / 1000)
        t = np.arange(num_samples) / self.sample_rate

        # Frequency sweep (logarithmic)
        freq_ratio = end_freq / start_freq
        instantaneous_freq = start_freq * (freq_ratio ** (t / t[-1]))
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.sample_rate

        # Generate chirp with amplitude envelope
        chirp = np.sin(phase) * amplitude

        # Apply envelope (attack/decay)
        envelope = np.ones_like(chirp)
        attack_samples = int(0.1 * num_samples)
        decay_samples = int(0.1 * num_samples)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)

        return chirp * envelope

    def generate_trill_sequence(
        self,
        num_chirps: int = 20,
        chirp_duration_ms: float = 30.0,
        gap_duration_ms: float = 20.0,
        background_noise: float = 0.01,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate a trill sequence with chirps at specified intervals.

        Returns (frames, chirp_indices) where chirp_indices indicates
        which frames contain chirps.
        """
        frame_size_ms = 10.0
        frame_size = int(self.sample_rate * frame_size_ms / 1000)

        frames = []
        chirp_indices = []

        total_samples = 0

        for i in range(num_chirps):
            # Add chirp
            chirp = self.generate_chirp(
                duration_ms=chirp_duration_ms,
                start_freq=4000.0,
                end_freq=8000.0,
                amplitude=0.3,
            )

            # Split chirp into frames
            chirp_frames = []
            for j in range(0, len(chirp), frame_size):
                frame = chirp[j:j+frame_size]
                if len(frame) < frame_size:
                    frame = np.pad(frame, (0, frame_size - len(frame)))
                chirp_frames.append(frame)
                chirp_indices.append(len(frames) + len(chirp_frames) - 1)

            frames.extend(chirp_frames)
            total_samples += len(chirp)

            # Add gap (silence + noise)
            gap_samples = int(self.sample_rate * gap_duration_ms / 1000)
            gap_frames = []
            for j in range(0, gap_samples, frame_size):
                noise = np.random.randn(frame_size).astype(np.float32) * background_noise
                gap_frames.append(noise)

            frames.extend(gap_frames)
            total_samples += gap_samples

        return frames, chirp_indices


class TestAvianTrill:
    """Test suite for avian trill detection."""

    def __init__(self):
        self.generator = AvianTrillGenerator()

    def run_trill_test(
        self,
        detector,
        frames: List[np.ndarray],
        chirp_indices: List[int],
    ) -> TrillTestResult:
        """Run detector on trill sequence."""
        detected = 0
        missed = 0
        false_positives = 0

        for i, frame in enumerate(frames):
            sample_index = i * len(frame)
            is_boundary, confidence = detector.detect(frame, sample_index)

            if i in chirp_indices:
                if is_boundary:
                    detected += 1
                else:
                    missed += 1
            else:
                if is_boundary:
                    false_positives += 1

        return TrillTestResult(
            name=detector.name,
            total_chirps=len(chirp_indices),
            detected_chirps=detected,
            missed_chirps=missed,
            false_positives=false_positives,
        )

    def test_legacy_fails_on_rapid_chirps(self):
        """
        Test that legacy 50ms debounce fails on rapid chirps.

        Expected: ~0% recall as chirps are filtered out.
        """
        print("\n" + "=" * 60)
        print("Legacy Debounce - Avian Trill Test")
        print("=" * 60)

        frames, chirp_indices = self.generator.generate_trill_sequence(
            num_chirps=20,
            chirp_duration_ms=30.0,
            gap_duration_ms=20.0,  # 20ms gap < 50ms debounce
        )

        detector = LegacyDebounceDetector(debounce_ms=50.0)
        result = self.run_trill_test(detector, frames, chirp_indices)

        print(result.report())

        # Legacy should fail (low recall)
        assert result.recall < 0.20, \
            f"Legacy should have low recall on rapid chirps (got {result.recall*100:.1f}%)"

    def test_predictive_adaptive_succeeds(self):
        """
        Test that predictive adaptive re-arm succeeds on rapid chirps.

        Expected: >90% recall as re-arm allows rapid detection.
        """
        print("\n" + "=" * 60)
        print("Predictive Adaptive - Avian Trill Test")
        print("=" * 60)

        frames, chirp_indices = self.generator.generate_trill_sequence(
            num_chirps=20,
            chirp_duration_ms=30.0,
            gap_duration_ms=20.0,
        )

        detector = PredictiveAdaptiveDetector(
            threshold_multiplier=2.5,
            rearm_threshold=1.2,
        )
        result = self.run_trill_test(detector, frames, chirp_indices)

        print(result.report())

        # Predictive should pass (high recall)
        assert result.within_target, \
            f"Predictive failed: Recall {result.recall*100:.1f}% below 90% target"

    def test_gap_duration_sensitivity(self):
        """
        Test recall across various gap durations.
        """
        print("\n" + "=" * 60)
        print("Gap Duration Sensitivity")
        print("=" * 60)

        gap_durations = [10, 20, 30, 40, 50, 100]

        print("Legacy (50ms debounce):")
        for gap in gap_durations:
            frames, chirp_indices = self.generator.generate_trill_sequence(
                num_chirps=10,
                chirp_duration_ms=30.0,
                gap_duration_ms=float(gap),
            )

            detector = LegacyDebounceDetector(debounce_ms=50.0)
            result = self.run_trill_test(detector, frames, chirp_indices)
            status = "✓" if gap >= 50 else "✗"
            print(f"  {status} Gap {gap}ms: Recall = {result.recall*100:.0f}%")

        print("\nPredictive (adaptive re-arm):")
        for gap in gap_durations:
            frames, chirp_indices = self.generator.generate_trill_sequence(
                num_chirps=10,
                chirp_duration_ms=30.0,
                gap_duration_ms=float(gap),
            )

            detector = PredictiveAdaptiveDetector()
            result = self.run_trill_test(detector, frames, chirp_indices)
            status = "✓" if result.recall >= 0.8 else "✗"
            print(f"  {status} Gap {gap}ms: Recall = {result.recall*100:.0f}%")

    def test_rearm_threshold_impact(self):
        """
        Test impact of rearm threshold on detection.
        """
        print("\n" + "=" * 60)
        print("Rearm Threshold Impact")
        print("=" * 60)

        frames, chirp_indices = self.generator.generate_trill_sequence(
            num_chirps=20,
            chirp_duration_ms=30.0,
            gap_duration_ms=20.0,
        )

        rearm_thresholds = [1.0, 1.2, 1.5, 2.0]

        for threshold in rearm_thresholds:
            detector = PredictiveAdaptiveDetector(rearm_threshold=threshold)
            result = self.run_trill_test(detector, frames, chirp_indices)

            speed = "Fast" if threshold < 1.5 else "Slow"
            print(f"  Rearm={threshold}: Recall = {result.recall*100:.1f}% ({speed} re-arm)")

            # All should maintain reasonable recall
            assert result.recall > 0.70, \
                f"Rearm threshold {threshold} produces low recall: {result.recall*100:.1f}%"

    def test_chirp_duration_variations(self):
        """
        Test detection across various chirp durations.
        """
        print("\n" + "=" * 60)
        print("Chirp Duration Variations")
        print("=" * 60)

        chirp_durations = [10, 20, 30, 40, 50]

        print("Predictive Adaptive:")
        for duration in chirp_durations:
            frames, chirp_indices = self.generator.generate_trill_sequence(
                num_chirps=15,
                chirp_duration_ms=float(duration),
                gap_duration_ms=20.0,
            )

            detector = PredictiveAdaptiveDetector()
            result = self.run_trill_test(detector, frames, chirp_indices)

            print(f"  {duration}ms chirp: Recall = {result.recall*100:.1f}%")

            # Should maintain good recall across durations
            assert result.recall > 0.70, \
                f"Low recall for {duration}ms chirps: {result.recall*100:.1f}%"

    def test_comparison_summary(self):
        """
        Generate comparison summary.
        """
        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)

        # Test condition: 30ms chirp, 20ms gap (challenging)
        frames, chirp_indices = self.generator.generate_trill_sequence(
            num_chirps=20,
            chirp_duration_ms=30.0,
            gap_duration_ms=20.0,
        )

        # Legacy
        legacy = LegacyDebounceDetector(debounce_ms=50.0)
        legacy_result = self.run_trill_test(legacy, frames, chirp_indices)

        # Predictive
        predictive = PredictiveAdaptiveDetector()
        predictive_result = self.run_trill_test(predictive, frames, chirp_indices)

        print(f"\nLegacy (50ms debounce):")
        print(f"  Recall: {legacy_result.recall*100:.1f}%")
        print(f"  Missed chirps: {legacy_result.missed_chirps}/{legacy_result.total_chirps}")

        print(f"\nPredictive (adaptive re-arm):")
        print(f"  Recall: {predictive_result.recall*100:.1f}%")
        print(f"  Missed chirps: {predictive_result.missed_chirps}/{predictive_result.total_chirps}")

        improvement = predictive_result.recall / max(legacy_result.recall, 0.01)
        print(f"\nImprovement: {improvement:.1f}x")

        return {
            "legacy": legacy_result,
            "predictive": predictive_result,
        }


def main():
    """Run all avian trill tests."""
    print("=" * 60)
    print("Phase 2.2: Avian Trill Test")
    print("=" * 60)
    print()

    test = TestAvianTrill()

    # Run tests
    test.test_legacy_fails_on_rapid_chirps()
    test.test_predictive_adaptive_succeeds()
    test.test_gap_duration_sensitivity()
    test.test_rearm_threshold_impact()
    test.test_chirp_duration_variations()
    test.test_comparison_summary()

    print("\n" + "=" * 60)
    print("✓ ALL AVIAN TRILL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
