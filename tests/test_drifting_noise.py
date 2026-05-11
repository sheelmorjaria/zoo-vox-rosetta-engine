#!/usr/bin/env python3
"""
Phase 2.1: Drifting Noise Test - Static vs. Dynamic Thresholding

Test Protocol:
- Inject audio with gradually increasing noise (60 seconds)
- Legacy (fixed thresholds): False positives trigger as noise increases
- Predictive (EMA baseline): Adapts to drifting baseline, maintains <5% FP rate

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftTestResult:
    """Results from drifting noise test."""
    name: str
    total_frames: int
    false_positives: int
    true_negatives: int
    baseline_values: List[float]
    threshold_values: List[float]

    @property
    def fp_rate(self) -> float:
        """False positive rate."""
        total = self.false_positives + self.true_negatives
        return self.false_positives / max(total, 1)

    @property
    def target_fp_rate(self) -> float:
        """Target FP rate is <5%."""
        return 0.05

    @property
    def within_target(self) -> bool:
        return self.fp_rate <= self.target_fp_rate

    def report(self) -> str:
        status = "✓ PASS" if self.within_target else "✗ FAIL"
        return (
            f"{self.name}:\n"
            f"  Total frames: {self.total_frames}\n"
            f"  False positives: {self.false_positives}\n"
            f"  True negatives: {self.true_negatives}\n"
            f"  FP rate: {self.fp_rate*100:.2f}% (target: <5%)\n"
            f"  {status}"
        )


class LegacyThresholdDetector:
    """
    Legacy NBD with fixed RMS energy threshold.

    Problem: Fixed threshold doesn't adapt to changing noise floor,
    causing false positives as noise increases.
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.name = "Legacy (Fixed Threshold)"

    def detect(self, audio: np.ndarray) -> Tuple[bool, float]:
        """Detect boundary based on fixed RMS threshold."""
        rms = np.sqrt(np.mean(audio ** 2))
        is_boundary = rms > self.threshold
        return is_boundary, rms


class PredictiveNBDDetector:
    """
    Predictive NBD with EMA baseline adaptation.

    Advantage: Baseline tracks drifting noise floor,
    maintaining stable false positive rate.
    """

    def __init__(
        self,
        threshold_multiplier: float = 2.5,
        ema_decay: float = 0.95,
        min_observations: int = 10,
    ):
        self.threshold_multiplier = threshold_multiplier
        self.ema_decay = ema_decay
        self.min_observations = min_observations
        self.baseline = 0.0
        self.observations = 0
        self.name = "Predictive (EMA Baseline)"

    def detect(self, audio: np.ndarray) -> Tuple[bool, float]:
        """Detect boundary with adaptive baseline."""
        # Compute prediction error
        rms = np.sqrt(np.mean(audio ** 2))

        # Update baseline using EMA
        if self.observations < self.min_observations:
            # Initialization phase
            self.baseline = (self.baseline * self.observations + rms) / (self.observations + 1)
            self.observations += 1
        else:
            # EMA update
            self.baseline = self.ema_decay * self.baseline + (1 - self.ema_decay) * rms

        # Compute normalized error
        if self.baseline > 0:
            normalized_error = rms / self.baseline
        else:
            normalized_error = 1.0

        # Check boundary with adaptive threshold
        threshold = self.threshold_multiplier
        is_boundary = normalized_error >= threshold

        return is_boundary, normalized_error


class DriftingNoiseGenerator:
    """
    Generate audio with gradually increasing noise floor.

    Simulates 60 seconds of audio where background noise
    gradually increases from quiet to loud.
    """

    def __init__(self, sample_rate: int = 48000, frame_size_ms: float = 10.0):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_size_ms / 1000)

    def generate_drift_sequence(
        self,
        duration_sec: float = 60.0,
        initial_noise_level: float = 0.01,
        final_noise_level: float = 0.15,
    ) -> List[np.ndarray]:
        """
        Generate sequence of audio frames with drifting noise.

        Noise level increases linearly from initial to final.
        """
        num_frames = int(duration_sec * 1000 / 10)  # 10ms frames
        frames = []

        for i in range(num_frames):
            # Linear drift
            progress = i / num_frames
            noise_level = initial_noise_level + (final_noise_level - initial_noise_level) * progress

            # Generate noise frame (no real vocalizations - should all be negatives)
            noise = np.random.randn(self.frame_size).astype(np.float32) * noise_level
            frames.append(noise)

        return frames

    def add_vocalization_spikes(
        self,
        frames: List[np.ndarray],
        spike_indices: List[int],
        spike_amplitude: float = 0.5,
    ) -> List[np.ndarray]:
        """
        Add occasional vocalization spikes (true positives).

        These should be detected by both systems.
        """
        result = []
        for i, frame in enumerate(frames):
            if i in spike_indices:
                # Add a tonal spike at 8kHz
                t = np.arange(len(frame)) / self.sample_rate
                spike = np.sin(2 * np.pi * 8000 * t) * spike_amplitude
                frame = frame + spike.astype(np.float32)
            result.append(frame)
        return result


class TestDriftingNoise:
    """Test suite for drifting noise robustness."""

    def __init__(self):
        self.generator = DriftingNoiseGenerator()

    def run_drift_test(
        self,
        detector,
        frames: List[np.ndarray],
    ) -> DriftTestResult:
        """Run detector on drifting noise sequence."""
        baseline_values = []
        threshold_values = []
        false_positives = 0
        true_negatives = 0

        for frame in frames:
            is_boundary, metric = detector.detect(frame)

            if hasattr(detector, 'baseline'):
                baseline_values.append(detector.baseline)
                threshold_values.append(detector.threshold_multiplier)
            else:
                baseline_values.append(0.0)
                threshold_values.append(detector.threshold)

            # In this test, all frames are noise (no real boundaries)
            if is_boundary:
                false_positives += 1
            else:
                true_negatives += 1

        return DriftTestResult(
            name=detector.name,
            total_frames=len(frames),
            false_positives=false_positives,
            true_negatives=true_negatives,
            baseline_values=baseline_values,
            threshold_values=threshold_values,
        )

    def test_legacy_fixed_threshold_fails(self):
        """
        Test that legacy fixed threshold fails on drifting noise.

        Expected: FP rate > 20% as noise crosses fixed threshold.
        """
        print("\n" + "=" * 60)
        print("Legacy Fixed Threshold - Drifting Noise Test")
        print("=" * 60)

        # Generate drifting noise
        frames = self.generator.generate_drift_sequence(
            duration_sec=60.0,
            initial_noise_level=0.01,
            final_noise_level=0.15,
        )

        detector = LegacyThresholdDetector(threshold=0.1)
        result = self.run_drift_test(detector, frames)

        print(result.report())

        # Legacy should fail (high FP rate)
        assert result.fp_rate > 0.10, \
            f"Legacy should have high FP rate on drifting noise (got {result.fp_rate*100:.1f}%)"

    def test_predictive_ema_adapts(self):
        """
        Test that predictive EMA baseline adapts to drifting noise.

        Expected: FP rate < 5% as baseline tracks noise floor.
        """
        print("\n" + "=" * 60)
        print("Predictive EMA Baseline - Drifting Noise Test")
        print("=" * 60)

        # Generate drifting noise
        frames = self.generator.generate_drift_sequence(
            duration_sec=60.0,
            initial_noise_level=0.01,
            final_noise_level=0.15,
        )

        detector = PredictiveNBDDetector(
            threshold_multiplier=2.5,
            ema_decay=0.95,
        )
        result = self.run_drift_test(detector, frames)

        print(result.report())

        # Predictive should pass (low FP rate)
        assert result.within_target, \
            f"Predictive EMA failed: FP rate {result.fp_rate*100:.2f}% exceeds 5% target"

    def test_baseline_tracking(self):
        """
        Test that EMA baseline correctly tracks drifting noise floor.
        """
        print("\n" + "=" * 60)
        print("Baseline Tracking Analysis")
        print("=" * 60)

        frames = self.generator.generate_drift_sequence(
            duration_sec=60.0,
            initial_noise_level=0.01,
            final_noise_level=0.15,
        )

        detector = PredictiveNBDDetector(ema_decay=0.95)
        result = self.run_drift_test(detector, frames)

        # Check that baseline increases over time
        initial_baseline = np.mean(result.baseline_values[:100])
        final_baseline = np.mean(result.baseline_values[-100:])

        print(f"Initial baseline: {initial_baseline:.4f}")
        print(f"Final baseline:   {final_baseline:.4f}")
        print(f"Baseline growth:  {final_baseline/initial_baseline:.2f}x")

        # Baseline should track noise growth (~15x from 0.01 to 0.15)
        assert final_baseline > initial_baseline * 5, \
            "Baseline should increase significantly with noise floor"

    def test_threshold_multiplier_calibration(self):
        """
        Test various threshold multipliers for optimal FP rate.
        """
        print("\n" + "=" * 60)
        print("Threshold Multiplier Calibration")
        print("=" * 60)

        frames = self.generator.generate_drift_sequence(
            duration_sec=60.0,
            initial_noise_level=0.01,
            final_noise_level=0.15,
        )

        multipliers = [2.0, 2.5, 3.0, 4.0]
        results = []

        for mult in multipliers:
            detector = PredictiveNBDDetector(threshold_multiplier=mult)
            result = self.run_drift_test(detector, frames)
            results.append((mult, result.fp_rate))
            print(f"  {mult}.0x: FP rate = {result.fp_rate*100:.2f}%")

        # Find optimal multiplier (closest to 5% FP rate without exceeding)
        optimal = min(results, key=lambda x: abs(x[1] - 0.05) if x[1] <= 0.05 else 1.0)
        print(f"\nOptimal multiplier: {optimal[0].0}x (FP rate: {optimal[1]*100:.2f}%)")

    def test_ema_decay_sensitivity(self):
        """
        Test EMA decay parameter sensitivity.
        """
        print("\n" + "=" * 60)
        print("EMA Decay Sensitivity Analysis")
        print("=" * 60)

        frames = self.generator.generate_drift_sequence(
            duration_sec=60.0,
            initial_noise_level=0.01,
            final_noise_level=0.15,
        )

        decay_values = [0.90, 0.95, 0.98, 0.99]

        for decay in decay_values:
            detector = PredictiveNBDDetector(ema_decay=decay)
            result = self.run_drift_test(detector, frames)

            adaptation_speed = "Fast" if decay < 0.95 else "Slow"
            print(f"  Decay={decay}: FP rate = {result.fp_rate*100:.2f}% ({adaptation_speed} adaptation)")

            # All should maintain reasonable FP rate
            assert result.fp_rate < 0.10, \
                f"Decay {decay} produces excessive FP rate: {result.fp_rate*100:.2f}%"

    def test_comparison_plot_data(self):
        """
        Generate data for comparison visualization.
        """
        print("\n" + "=" * 60)
        print("Generating Comparison Data")
        print("=" * 60)

        frames = self.generator.generate_drift_sequence(
            duration_sec=60.0,
            initial_noise_level=0.01,
            final_noise_level=0.15,
        )

        # Legacy
        legacy = LegacyThresholdDetector(threshold=0.1)
        legacy_result = self.run_drift_test(legacy, frames)

        # Predictive
        predictive = PredictiveNBDDetector(ema_decay=0.95)
        predictive_result = self.run_drift_test(predictive, frames)

        print("\nComparison Summary:")
        print(f"  Legacy FP rate:     {legacy_result.fp_rate*100:.2f}%")
        print(f"  Predictive FP rate: {predictive_result.fp_rate*100:.2f}%")
        print(f"  Improvement:        {legacy_result.fp_rate/predictive_result.fp_rate:.1f}x")

        return {
            "legacy": legacy_result,
            "predictive": predictive_result,
        }


def main():
    """Run all drifting noise tests."""
    print("=" * 60)
    print("Phase 2.1: Drifting Noise Test")
    print("=" * 60)
    print()

    test = TestDriftingNoise()

    # Run tests
    test.test_legacy_fixed_threshold_fails()
    test.test_predictive_ema_adapts()
    test.test_baseline_tracking()
    test.test_threshold_multiplier_calibration()
    test.test_ema_decay_sensitivity()
    test.test_comparison_plot_data()

    print("\n" + "=" * 60)
    print("✓ ALL DRIFTING NOISE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
