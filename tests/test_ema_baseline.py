#!/usr/bin/env python3
"""
Phase 4.1c: EMA Baseline Validation

Tests for Exponential Moving Average baseline tracking.
Validates adaptive threshold behavior and noise robustness.

Key Requirements:
- EMA decay: 0.95
- Baseline adapts to changing noise floor
- Smooth tracking without oscillation

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
class EMABaselineConfig:
    """Configuration for EMA baseline tracker."""
    decay: float = 0.95
    min_observations: int = 10
    min_baseline: float = 0.001
    max_baseline: float = 1.0


class EMABaselineTracker:
    """
    Exponential Moving Average baseline tracker.

    Formula:
    baseline_t = decay * baseline_{t-1} + (1 - decay) * value_t

    Properties:
    - Adapts to changing noise floor
    - Smooth tracking (decay > 0.9)
    - Bounded to prevent extreme values
    """

    def __init__(self, config: EMABaselineConfig):
        self.config = config
        self.baseline = 0.0
        self.observations = 0

    def update(self, value: float) -> float:
        """
        Update baseline with new value.

        Args:
            value: New observation

        Returns:
            Current baseline after update
        """
        if self.observations < self.config.min_observations:
            # Initialization phase: simple average
            self.baseline = (
                self.baseline * self.observations + value
            ) / (self.observations + 1)
            self.observations += 1
        else:
            # EMA update
            self.baseline = (
                self.config.decay * self.baseline +
                (1 - self.config.decay) * value
            )

        # Clamp to valid range
        self.baseline = np.clip(
            self.baseline,
            self.config.min_baseline,
            self.config.max_baseline,
        )

        return self.baseline

    def reset(self):
        """Reset baseline to initial state."""
        self.baseline = 0.0
        self.observations = 0

    def get_normalized_error(self, value: float) -> float:
        """
        Get normalized error (value / baseline).

        Args:
            value: Current observation

        Returns:
            Normalized error ratio
        """
        if self.baseline > 0:
            return value / self.baseline
        return 1.0

    def is_above_threshold(
        self,
        value: float,
        threshold_multiplier: float,
    ) -> bool:
        """
        Check if value exceeds threshold.

        Args:
            value: Current observation
            threshold_multiplier: Multiplier for baseline

        Returns:
            True if value > baseline * threshold_multiplier
        """
        if self.baseline > 0:
            return value >= (self.baseline * threshold_multiplier)
        return False


class TestEMABaseline:
    """Test suite for EMA baseline validation."""

    def __init__(self):
        self.config = EMABaselineConfig(decay=0.95)

    def test_ema_initialization(self):
        """Test EMA initialization phase."""
        print("\n" + "=" * 60)
        print("EMA Initialization Phase")
        print("=" * 60)

        tracker = EMABaselineTracker(self.config)

        # Feed initial values
        values = [0.1, 0.12, 0.09, 0.11, 0.1, 0.13, 0.08, 0.1, 0.11, 0.1]

        for v in values:
            baseline = tracker.update(v)
            print(f"  Value: {v:.3f} -> Baseline: {baseline:.3f} (obs: {tracker.observations})")

        # After initialization, baseline should be near average
        expected = np.mean(values)
        assert abs(tracker.baseline - expected) < 0.02, \
            f"Initial baseline {tracker.baseline:.3f} should be near {expected:.3f}"

    def test_ema_decay_tracking(self):
        """Test EMA decay parameter behavior."""
        print("\n" + "=" * 60)
        print("EMA Decay Tracking")
        print("=" * 60)

        tracker = EMABaselineTracker(self.config)

        # Initialize
        for _ in range(10):
            tracker.update(0.1)

        initial_baseline = tracker.baseline
        print(f"Initial baseline: {initial_baseline:.4f}")

        # Step change
        new_level = 0.2
        for i in range(50):
            tracker.update(new_level)
            if i % 10 == 0:
                print(f"  Step {i}: baseline = {tracker.baseline:.4f}")

        final_baseline = tracker.baseline
        print(f"Final baseline: {final_baseline:.4f}")

        # With decay=0.95, should converge toward new level
        # After 50 steps: expected ≈ 0.1 * 0.95^50 + 0.2 * (1 - 0.95^50)
        # ≈ 0.2 * (1 - 0.077) ≈ 0.185
        expected_convergence = new_level * (1 - 0.95 ** 50)

        print(f"Expected convergence: {expected_convergence:.4f}")
        assert abs(final_baseline - expected_convergence) < 0.02, \
            f"Baseline {final_baseline:.4f} should converge toward {expected_convergence:.4f}"

    def test_ema_decay_comparison(self):
        """Test different decay values."""
        print("\n" + "=" * 60)
        print("EMA Decay Comparison")
        print("=" * 60)

        initial_level = 0.1
        new_level = 0.2

        decay_values = [0.90, 0.95, 0.98, 0.99]

        for decay in decay_values:
            config = EMABaselineConfig(decay=decay)
            tracker = EMABaselineTracker(config)

            # Initialize
            for _ in range(10):
                tracker.update(initial_level)

            # Step change
            for _ in range(50):
                tracker.update(new_level)

            speed = "Fast" if decay < 0.95 else "Slow"
            print(f"  Decay {decay}: baseline = {tracker.baseline:.4f} ({speed} adaptation)")

        # Lower decay = faster adaptation

    def test_baseline_clamping(self):
        """Test baseline clamping to valid range."""
        print("\n" + "=" * 60)
        print("Baseline Clamping")
        print("=" * 60)

        config = EMABaselineConfig(
            decay=0.95,
            min_baseline=0.01,
            max_baseline=0.5,
        )
        tracker = EMABaselineTracker(config)

        # Initialize
        for _ in range(10):
            tracker.update(0.1)

        # Try very low value
        tracker.update(0.001)
        assert tracker.baseline >= config.min_baseline, \
            f"Baseline {tracker.baseline} should be clamped to min {config.min_baseline}"
        print(f"  Low value (0.001) -> baseline clamped to {tracker.baseline:.4f}")

        # Try very high value
        tracker.update(10.0)
        assert tracker.baseline <= config.max_baseline, \
            f"Baseline {tracker.baseline} should be clamped to max {config.max_baseline}"
        print(f"  High value (10.0) -> baseline clamped to {tracker.baseline:.4f}")

    def test_normalized_error_computation(self):
        """Test normalized error computation."""
        print("\n" + "=" * 60)
        print("Normalized Error Computation")
        print("=" * 60)

        tracker = EMABaselineTracker(self.config)

        # Initialize
        for _ in range(10):
            tracker.update(0.1)

        # Test various values
        test_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.4]

        print(f"Baseline: {tracker.baseline:.4f}")
        for v in test_values:
            normalized = tracker.get_normalized_error(v)
            print(f"  Value {v:.2f} -> normalized error: {normalized:.2f}x")

        # Normalized error = value / baseline
        # Value = baseline -> error = 1.0
        # Value = 2 * baseline -> error = 2.0

    def test_threshold_detection(self):
        """Test threshold-based detection."""
        print("\n" + "=" * 60)
        print("Threshold Detection")
        print("=" * 60)

        tracker = EMABaselineTracker(self.config)

        # Initialize
        for _ in range(10):
            tracker.update(0.1)

        baseline = tracker.baseline
        print(f"Baseline: {baseline:.4f}")

        # Test various multipliers
        multipliers = [1.5, 2.0, 2.5, 3.0, 4.0]

        for mult in multipliers:
            threshold = baseline * mult
            test_value = threshold * 1.1  # Just above threshold

            is_above = tracker.is_above_threshold(test_value, mult)
            normalized = tracker.get_normalized_error(test_value)

            status = "✓ DETECTED" if is_above else "✗ NOT DETECTED"
            print(f"  {mult}x threshold ({threshold:.4f}): value {test_value:.4f} -> {status}")

    def test_noise_robustness(self):
        """Test baseline tracking with noisy input."""
        print("\n" + "=" * 60)
        print("Noise Robustness")
        print("=" * 60)

        tracker = EMABaselineTracker(self.config)

        # Signal with noise
        signal_level = 0.1
        noise_std = 0.02

        baselines = []
        for i in range(100):
            noisy_value = signal_level + np.random.randn() * noise_std
            baseline = tracker.update(noisy_value)
            if i >= 10:  # After initialization
                baselines.append(baseline)

        # Baseline should be stable despite noise
        baseline_std = np.std(baselines)
        baseline_mean = np.mean(baselines)

        print(f"Signal level: {signal_level:.4f}")
        print(f"Noise std: {noise_std:.4f}")
        print(f"Baseline mean: {baseline_mean:.4f}")
        print(f"Baseline std: {baseline_std:.4f}")

        # Baseline variation should be much less than input noise
        assert baseline_std < noise_std * 0.5, \
            f"Baseline std {baseline_std:.4f} should be much less than noise std {noise_std:.4f}"

    def test_drift_adaptation(self):
        """Test adaptation to drifting baseline."""
        print("\n" + "=" * 60)
        print("Drift Adaptation (Simulating Drifting Noise)")
        print("=" * 60)

        tracker = EMABaselineTracker(self.config)

        # Gradually increasing signal (simulating drift)
        baselines = []
        signal_levels = np.linspace(0.05, 0.15, 100)

        for level in signal_levels:
            noisy_value = level + np.random.randn() * 0.005
            baseline = tracker.update(noisy_value)
            baselines.append(baseline)

        # Plot data points
        print("Tracking drift from 0.05 to 0.15:")
        print(f"  Initial baseline: {baselines[10]:.4f}")
        print(f"  Final baseline: {baselines[-1]:.4f}")

        # Final baseline should be close to final signal level
        final_signal = signal_levels[-1]
        final_baseline = baselines[-1]

        assert abs(final_baseline - final_signal) < 0.02, \
            f"Baseline {final_baseline:.4f} should track signal {final_signal:.4f}"

    def test_oscillation_prevention(self):
        """Test that EMA prevents oscillation."""
        print("\n" + "=" * 60)
        print("Oscillation Prevention")
        print("=" * 60)

        # Compare EMA vs. simple moving average
        ema_tracker = EMABaselineTracker(self.config)

        # Alternating signal
        values = [0.05, 0.15] * 25  # 50 alternating values

        ema_baselines = []
        for v in values:
            baseline = ema_tracker.update(v)
            ema_baselines.append(baseline)

        # Calculate oscillation (adjacent difference)
        ema_oscillation = np.mean(np.abs(np.diff(ema_baselines[10:])))
        input_oscillation = np.mean(np.abs(np.diff(values)))

        print(f"Input oscillation: {input_oscillation:.4f}")
        print(f"EMA baseline oscillation: {ema_oscillation:.4f}")
        print(f"Damping factor: {input_oscillation / max(ema_oscillation, 1e-6):.2f}x")

        # EMA should significantly damp oscillations
        assert ema_oscillation < input_oscillation * 0.3, \
            "EMA should damp oscillations"

    def test_reset_functionality(self):
        """Test baseline reset capability."""
        print("\n" + "=" * 60)
        print("Reset Functionality")
        print("=" * 60)

        tracker = EMABaselineTracker(self.config)

        # Build up baseline
        for _ in range(20):
            tracker.update(0.1)

        baseline_before = tracker.baseline
        print(f"Baseline before reset: {baseline_before:.4f}")

        # Reset
        tracker.reset()

        assert tracker.baseline == 0.0, "Baseline should be 0 after reset"
        assert tracker.observations == 0, "Observations should be 0 after reset"
        print(f"Baseline after reset: {tracker.baseline:.4f}")

        # Rebuild
        tracker.update(0.2)
        assert tracker.observations == 1, "Should count from 1 after reset"


def main():
    """Run all EMA baseline validation tests."""
    print("=" * 60)
    print("Phase 4.1c: EMA Baseline Validation")
    print("=" * 60)
    print()

    test = TestEMABaseline()

    test.test_ema_initialization()
    test.test_ema_decay_tracking()
    test.test_ema_decay_comparison()
    test.test_baseline_clamping()
    test.test_normalized_error_computation()
    test.test_threshold_detection()
    test.test_noise_robustness()
    test.test_drift_adaptation()
    test.test_oscillation_prevention()
    test.test_reset_functionality()

    print("\n" + "=" * 60)
    print("✓ ALL EMA BASELINE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
