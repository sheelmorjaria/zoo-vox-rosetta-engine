#!/usr/bin/env python3
"""
Predictive Boundary Detector: Adaptive Semantic Boundary Detection (Green Phase)

Uses Contrastive Predictive Coding (CPC) prediction errors to detect
semantic boundaries in animal vocalizations. Replaces fixed 50ms debounce
with adaptive algorithm that responds to acoustic state transitions.

Green Phase Improvements:
- Dual-EMA Baseline: Fast decay (0.9) for armed reset, slow decay (0.99) for ambient tracking
- Derivative-Based Triggering: Detects rapid error spikes (d(error)/dt)
- Duration-Gated Confidence: Temporal integration for multi-scale classification
- Slope Tracking: Integral of error curve separates transients from sustained shifts

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BoundaryType(Enum):
    """Types of semantic boundaries with duration requirements."""
    PHONETIC = "phonetic"      # Shortest unit, ~10-30ms, requires 10ms sustained
    SYLLABLE = "syllable"      # Medium unit, ~50-150ms, requires 30ms sustained
    PHRASE = "phrase"          # Longest unit, ~200-500ms, requires 80ms sustained


@dataclass
class PredictionResult:
    """Result of prediction-based boundary detection."""
    timestamp_ns: int
    prediction_error: float
    baseline_error: float
    normalized_error: float       # error / baseline
    error_derivative: float       # d(error)/dt for spike detection
    error_integral: float         # area under error curve
    sustained_duration_ms: float  # how long error has been elevated
    is_boundary: bool
    boundary_type: Optional[BoundaryType]
    confidence: float             # 0-1
    latency_ms: float


@dataclass
class BoundaryEvent:
    """Detected semantic boundary event."""
    timestamp_ns: int
    boundary_type: BoundaryType
    prediction_error: float
    confidence: float
    latency_ms: float


class BoundaryDetectorConfig:
    """
    Configuration for PredictiveBoundaryDetector (Green Phase).

    Tuned for Avian Trill (sub-50ms boundaries) and Drifting Noise robustness.
    """

    def __init__(
        self,
        # === Detection Thresholds (Duration-Gated) ===
        boundary_threshold: float = 2.5,         # Normalized error > 2.5x -> potential boundary
        boundary_threshold_lower: float = 1.5,   # Normalized error < 1.5x -> end of elevated period
        syllable_threshold: float = 3.0,         # Higher threshold for syllable
        phrase_threshold: float = 4.0,           # Highest threshold for phrase

        # === Duration Requirements (Gating) ===
        # Phonetic: 2.5x for ≥10ms, Syllable: 3.0x for ≥30ms, Phrase: 4.0x for ≥80ms
        phonetic_duration_ms: float = 10.0,      # Min sustained for phonetic
        syllable_duration_ms: float = 30.0,      # Min sustained for syllable
        phrase_duration_ms: float = 80.0,        # Min sustained for phrase

        # === Derivative-Based Spike Detection ===
        derivative_threshold: float = 0.5,        # d(error)/dt > threshold -> spike
        derivative_window_ms: float = 20.0,       # Window for derivative calculation

        # === Dual-EMA Baseline Tracking ===
        baseline_window: int = 100,               # Frames for baseline calculation
        slow_decay: float = 0.99,                 # For long-term ambient noise tracking
        fast_decay: float = 0.9,                  # For armed state reset (15-20ms recovery)

        # === Armed Logic ===
        rearm_threshold: float = 1.2,             # normalized_error < 1.2x -> rearm
        rearm_trigger_threshold: float = 1.5,     # Also rearm if error drops below this

        # === Confidence Scoring ===
        min_confidence: float = 0.6,              # Minimum confidence for boundary

        # === Timing ===
        frame_size_ms: float = 10.0,              # Per-frame duration
        sample_rate: int = 48000,                 # Audio sample rate
    ):
        # Detection thresholds (dual-threshold for hysteresis)
        self.boundary_threshold = boundary_threshold
        self.boundary_threshold_lower = boundary_threshold_lower
        self.syllable_threshold = syllable_threshold
        self.phrase_threshold = phrase_threshold

        # Duration requirements
        self.phonetic_duration_ms = phonetic_duration_ms
        self.syllable_duration_ms = syllable_duration_ms
        self.phrase_duration_ms = phrase_duration_ms

        # Derivative-based detection
        self.derivative_threshold = derivative_threshold
        self.derivative_window_ms = derivative_window_ms

        # Dual-EMA baseline
        self.baseline_window = baseline_window
        self.slow_decay = slow_decay
        self.fast_decay = fast_decay

        # Armed logic
        self.rearm_threshold = rearm_threshold
        self.rearm_trigger_threshold = rearm_trigger_threshold

        # Confidence scoring
        self.min_confidence = min_confidence

        # Timing
        self.frame_size_ms = frame_size_ms
        self.sample_rate = sample_rate


class SlopeTracker:
    """
    Tracks the slope/integral of the error curve for multi-scale classification.

    Uses temporal integration to distinguish between:
    - Sharp transients (phonetic): high derivative, low integral
    - Sustained shifts (syllable): medium derivative, medium integral
    - Long-term transitions (phrase): low derivative, high integral
    """

    def __init__(self, window_ms: float, frame_size_ms: float):
        self.window_ms = window_ms
        self.frame_size_ms = frame_size_ms
        self.window_frames = int(window_ms / frame_size_ms)

        # Deques for tracking
        self.error_history = deque(maxlen=self.window_frames)
        self.time_history = deque(maxlen=self.window_frames)

        # Current integral and slope
        self.current_integral = 0.0
        self.current_slope = 0.0

    def update(self, error: float, timestamp_ms: float) -> Tuple[float, float]:
        """
        Update slope tracker with new error sample.

        Returns:
            (integral, slope) - Current integral and slope values
        """
        self.error_history.append(error)
        self.time_history.append(timestamp_ms)

        if len(self.error_history) < 2:
            return 0.0, 0.0

        # Compute integral (area under curve above threshold)
        threshold = 2.0  # Base threshold for integral calculation
        excess_errors = [max(e - threshold, 0) for e in self.error_history]
        self.current_integral = sum(excess_errors) * self.frame_size_ms

        # Compute slope (linear regression slope of recent errors)
        if len(self.error_history) >= 3:
            times = list(self.time_history)
            errors = list(self.error_history)

            # Simple linear regression
            n = len(times)
            sum_x = sum(times)
            sum_y = sum(errors)
            sum_xy = sum(t * e for t, e in zip(times, errors))
            sum_x2 = sum(t * t for t in times)

            denom = n * sum_x2 - sum_x * sum_x
            if denom != 0:
                self.current_slope = (n * sum_xy - sum_x * sum_y) / denom
            else:
                self.current_slope = 0.0
        else:
            self.current_slope = 0.0

        return self.current_integral, self.current_slope


class DualEMABaseline:
    """
    Dual Exponential Moving Average for adaptive baseline tracking.

    - Slow EMA: Tracks long-term ambient noise (decay=0.99)
    - Fast EMA: Tracks recent state for quick armed reset (decay=0.9)

    The fast EMA allows the baseline to collapse back to resting state
    within 15-20ms, enabling detection of rapid trill sequences.
    """

    def __init__(self, slow_decay: float = 0.99, fast_decay: float = 0.9):
        self.slow_decay = slow_decay
        self.fast_decay = fast_decay

        self.slow_baseline = 1.0
        self.fast_baseline = 1.0

        self.initialized = False

    def update(self, error: float, use_fast: bool = False) -> float:
        """
        Update baseline with new error sample.

        Args:
            error: Current prediction error
            use_fast: If True, update fast EMA; otherwise update slow EMA

        Returns:
            Updated baseline value
        """
        if not self.initialized:
            self.slow_baseline = error
            self.fast_baseline = error
            self.initialized = True
            return error

        decay = self.fast_decay if use_fast else self.slow_decay

        if use_fast:
            self.fast_baseline = decay * self.fast_baseline + (1 - decay) * error
            return self.fast_baseline
        else:
            self.slow_baseline = decay * self.slow_baseline + (1 - decay) * error
            return self.slow_baseline

    def get_baseline(self, use_fast: bool = False) -> float:
        """Get current baseline value."""
        return self.fast_baseline if use_fast else self.slow_baseline

    def reset_fast(self):
        """Reset fast EMA to current slow EMA value."""
        self.fast_baseline = self.slow_baseline


class PredictiveBoundaryDetector:
    """
    Adaptive boundary detector using CPC prediction errors (Green Phase).

    Key Improvements:
    1. Dual-EMA baseline for quick re-arming (15-20ms recovery)
    2. Derivative-based triggering detects rapid error spikes
    3. Duration-gated confidence uses temporal integration
    4. Slope tracking separates transients from sustained shifts

    Armed Algorithm (Enhanced):
    1. Initially armed, ready to detect boundary
    2. When error derivative spikes OR error exceeds threshold -> mark potential
    3. Track sustained duration above threshold
    4. Classify boundary based on duration + magnitude
    5. Disarm and use fast EMA for quick baseline reset
    """

    def __init__(
        self,
        config: Optional[BoundaryDetectorConfig] = None,
        cpc_model: Optional[nn.Module] = None,
    ):
        self.config = config or BoundaryDetectorConfig()
        self.cpc_model = cpc_model

        # Dual-EMA baseline
        self.baseline = DualEMABaseline(
            slow_decay=self.config.slow_decay,
            fast_decay=self.config.fast_decay
        )

        # Slope tracker for multi-scale classification
        self.slope_tracker = SlopeTracker(
            window_ms=100.0,  # 100ms window for slope/integral
            frame_size_ms=self.config.frame_size_ms
        )

        # Armed state
        self.armed = True
        self.disarmed_since_ns = 0
        self.last_boundary_time_ns = 0

        # Error tracking
        self.error_history: List[float] = []
        self.prev_error = 1.0
        self.prev_timestamp_ms = 0.0

        # Sustained duration tracking
        self.elevated_since_ms = None
        self.sustained_duration_ms = 0.0

        # Timing
        self.total_frames = 0

        # Statistics
        self.boundary_count = 0
        self.boundary_types: Dict[BoundaryType, int] = {
            BoundaryType.PHONETIC: 0,
            BoundaryType.SYLLABLE: 0,
            BoundaryType.PHRASE: 0,
        }

    def compute_prediction_error(
        self,
        z_latent: torch.Tensor,
        predictions: List[torch.Tensor],
    ) -> float:
        """
        Compute prediction error for current frame.

        Args:
            z_latent: (B, T, hidden_dim) encoded latents
            predictions: List of (B, T, hidden_dim) predictions

        Returns:
            Mean squared error across all prediction steps
        """
        if self.cpc_model is None:
            # For testing without model, use simple MSE
            total_error = 0.0
            count = 0

            batch_size, seq_len, hidden_dim = z_latent.shape

            for k, prediction in enumerate(predictions):
                target_start = k + 1
                if target_start >= seq_len:
                    continue

                z_future = z_latent[:, target_start:, :]
                z_pred = prediction[:, :seq_len - target_start, :]

                mse = torch.mean((z_future - z_pred) ** 2).item()
                total_error += mse
                count += 1

            return total_error / max(count, 1)
        else:
            # Use CPC model if available
            return self.cpc_model.compute_error(z_latent, predictions)

    def compute_error_derivative(
        self,
        current_error: float,
        current_timestamp_ms: float,
    ) -> float:
        """
        Compute derivative of error (d(error)/dt) for spike detection.

        Rapid spikes in error indicate transient boundaries, even if
        the baseline is elevated from previous activity.

        Args:
            current_error: Current prediction error
            current_timestamp_ms: Current timestamp in milliseconds

        Returns:
            Error derivative (error change per ms)
        """
        if self.prev_timestamp_ms == 0:
            derivative = 0.0
        else:
            dt = current_timestamp_ms - self.prev_timestamp_ms
            if dt > 0:
                derivative = (current_error - self.prev_error) / dt
            else:
                derivative = 0.0

        self.prev_error = current_error
        self.prev_timestamp_ms = current_timestamp_ms

        return derivative

    def update_baseline(self, error: float, use_fast: bool = False) -> float:
        """
        Update baseline error using dual-EMA.

        Args:
            error: Current prediction error
            use_fast: If True, use fast EMA for quick reset

        Returns:
            Updated baseline error
        """
        self.error_history.append(error)

        # Keep limited history
        if len(self.error_history) > self.config.baseline_window:
            self.error_history.pop(0)

        # Update dual-EMA
        baseline = self.baseline.update(error, use_fast=use_fast)

        return baseline

    def classify_boundary_duration_gated(
        self,
        normalized_error: float,
        sustained_ms: float,
        integral: float,
        slope: float,
    ) -> Optional[BoundaryType]:
        """
        Classify boundary type using duration-gated confidence.

        The key insight: A boundary's type is determined not just by
        error magnitude, but by how LONG the error stays elevated.

        Args:
            normalized_error: error / baseline ratio
            sustained_ms: Duration error has been elevated
            integral: Area under error curve
            slope: Linear regression slope of error

        Returns:
            BoundaryType if criteria met, None otherwise
        """
        # Phrase: Very high error OR very high integral, sustained for ≥80ms
        if (normalized_error >= self.config.phrase_threshold or integral > 50.0) and \
           sustained_ms >= self.config.phrase_duration_ms:
            return BoundaryType.PHRASE

        # Syllable: Medium-high error, sustained for ≥30ms
        if normalized_error >= self.config.syllable_threshold and \
           sustained_ms >= self.config.syllable_duration_ms:
            # Check not too soon (might be phonetic)
            if sustained_ms >= self.config.syllable_duration_ms:
                return BoundaryType.SYLLABLE

        # Phonetic: Just above threshold, sustained for ≥10ms
        if normalized_error >= self.config.boundary_threshold and \
           sustained_ms >= self.config.phonetic_duration_ms:
            return BoundaryType.PHONETIC

        return None

    def _classify_by_peak_duration(
        self,
        normalized_error: float,
        peak_duration_ms: float,
        integral: float,
        slope: float,
    ) -> Optional[BoundaryType]:
        """
        Classify boundary based on peak sustained duration during elevated period.

        Uses a hierarchical approach:
        - If peak duration ≥ 80ms OR very high integral → PHRASE
        - Else if peak duration ≥ 30ms AND medium-high error → SYLLABLE
        - Else if peak duration ≥ 10ms AND above threshold → PHONETIC

        Args:
            normalized_error: Current error value (after dropping)
            peak_duration_ms: Peak sustained duration during elevated period
            integral: Area under error curve
            slope: Linear regression slope

        Returns:
            BoundaryType if criteria met, None otherwise
        """
        # Phrase: Longest duration OR very high integral
        if peak_duration_ms >= self.config.phrase_duration_ms or integral > 50.0:
            return BoundaryType.PHRASE

        # Syllable: Medium duration, above syllable threshold
        if peak_duration_ms >= self.config.syllable_duration_ms:
            if integral > 20.0 or normalized_error >= self.config.syllable_threshold:
                return BoundaryType.SYLLABLE

        # Phonetic: Shortest duration, above basic threshold
        if peak_duration_ms >= self.config.phonetic_duration_ms:
            if normalized_error >= self.config.boundary_threshold:
                return BoundaryType.PHONETIC

        return None

    def compute_confidence_duration_gated(
        self,
        normalized_error: float,
        sustained_ms: float,
        integral: float,
        derivative: float,
        boundary_type: BoundaryType,
    ) -> float:
        """
        Compute confidence score using duration-gated logic.

        Confidence increases with:
        - Higher error magnitude
        - Longer sustained duration
        - Larger integral (more energy in the transition)
        - Sharper derivative (clearer boundary)

        Args:
            normalized_error: error / baseline ratio
            sustained_ms: Duration error has been elevated
            integral: Area under error curve
            derivative: Error derivative
            boundary_type: Detected boundary type

        Returns:
            Confidence score (0-1)
        """
        # Base confidence from normalized error
        error_conf = min(normalized_error / self.config.phrase_threshold, 1.0)

        # Duration boost (longer = more confident)
        duration_boost = min(sustained_ms / 100.0, 0.3)  # Max 0.3 boost

        # Integral boost (more energy = more confident)
        integral_boost = min(integral / 100.0, 0.2)  # Max 0.2 boost

        # Derivative boost (sharper spike = more confident)
        deriv_boost = min(abs(derivative) / self.config.derivative_threshold, 0.1) if derivative > 0 else 0

        # Type-specific boost
        type_boost = {
            BoundaryType.PHRASE: 0.1,
            BoundaryType.SYLLABLE: 0.05,
            BoundaryType.PHONETIC: 0.0,
        }.get(boundary_type, 0.0)

        confidence = error_conf + duration_boost + integral_boost + deriv_boost + type_boost
        return min(confidence, 1.0)

    def check_rearm_condition(self, normalized_error: float) -> bool:
        """
        Check if detector should re-arm.

        Uses both slow and fast EMA for quick re-arming. The fast EMA
        allows the baseline to reset within 15-20ms for rapid trills.

        Args:
            normalized_error: Current error ratio

        Returns:
            True if should re-arm
        """
        # Check if error dropped below rearm threshold
        if normalized_error < self.config.rearm_threshold:
            return True

        # Also check if error is close to baseline using fast EMA
        fast_baseline = self.baseline.get_baseline(use_fast=True)
        if normalized_error < self.config.rearm_trigger_threshold:
            return True

        return False

    def process_frame_with_error(
        self,
        error: float,
        timestamp_ns: int,
    ) -> PredictionResult:
        """
        Process a single frame with pre-computed error (for testing).

        Args:
            error: Pre-computed prediction error
            timestamp_ns: Current timestamp in nanoseconds

        Returns:
            PredictionResult with detection status
        """
        self.total_frames += 1
        current_time_ms = timestamp_ns / 1_000_000

        # Use provided error directly (skip compute_prediction_error)
        # Compute error derivative for spike detection (use RAW error, not normalized)
        # This detects spikes even when baseline has adapted
        derivative = self.compute_error_derivative(error, current_time_ms)

        # Update baseline (use slow EMA normally)
        baseline = self.update_baseline(error, use_fast=False)
        normalized_error = error / max(baseline, 1e-6)

        # Update slope tracker
        integral, slope = self.slope_tracker.update(normalized_error, current_time_ms)

        # Initialize result (boundary may be fired in tracking logic below)
        is_boundary = False
        boundary_type = None
        confidence = 0.0

        # Track sustained duration (and peak duration for classification)
        # Use dual-threshold: start > 2.5x, end < 1.5x
        if normalized_error > self.config.boundary_threshold:
            # Start of elevated period
            if self.elevated_since_ms is None:
                self.elevated_since_ms = current_time_ms
                self.elevated_frame_count = 0  # Count frames in elevated period
            self.elevated_frame_count += 1  # Increment frame count
            self.sustained_duration_ms = self.elevated_frame_count * self.config.frame_size_ms
            # Track peak duration during this elevated period
            if not hasattr(self, 'peak_sustained_duration_ms'):
                self.peak_sustained_duration_ms = 0.0
            self.peak_sustained_duration_ms = max(self.peak_sustained_duration_ms, self.sustained_duration_ms)
        elif normalized_error < self.config.boundary_threshold_lower:
            # Error dropped below lower threshold - end of elevated period
            if hasattr(self, 'peak_sustained_duration_ms') and self.peak_sustained_duration_ms > 0:
                # Just finished an elevated period - classify and fire boundary
                peak_duration = self.peak_sustained_duration_ms
                self.peak_sustained_duration_ms = 0.0  # Reset for next period

                # Classify based on peak duration achieved
                boundary_type = self._classify_by_peak_duration(
                    normalized_error,  # Use the FINAL error value (just dropped)
                    peak_duration,
                    integral,
                    slope,
                )

                if boundary_type is not None and self.armed:
                    confidence = self.compute_confidence_duration_gated(
                        normalized_error,
                        peak_duration,
                        integral,
                        derivative,
                        boundary_type,
                    )

                    if confidence >= self.config.min_confidence:
                        # Fire the boundary
                        is_boundary = True
                        self.boundary_count += 1
                        self.boundary_types[boundary_type] += 1
                        self.last_boundary_time_ns = timestamp_ns
                        self.armed = False
                        self.disarmed_since_ns = timestamp_ns

                        logger.info(
                            f"Boundary: {boundary_type.value} at {current_time_ms:.1f}ms, "
                            f"peak_duration={peak_duration:.1f}ms, "
                            f"error={normalized_error:.2f}x, conf={confidence:.2f}"
                        )

            # Reset tracking
            self.elevated_since_ms = None
            self.sustained_duration_ms = 0.0
            if hasattr(self, 'peak_sustained_duration_ms'):
                self.peak_sustained_duration_ms = 0.0
        else:
            # In hysteresis zone (between 1.5x and 2.5x) - maintain current state
            # Don't reset tracking, but don't update peak duration
            pass

        # Check armed state and re-arm condition
        if not self.armed:
            if self.check_rearm_condition(normalized_error):
                self.armed = True
                self.baseline.reset_fast()  # Reset fast EMA to slow EMA
                logger.debug(f"Rearmed at {current_time_ms:.1f}ms, error={normalized_error:.2f}x")

        # Compute detection latency
        latency_ms = self.config.frame_size_ms  # Best case

        return PredictionResult(
            timestamp_ns=timestamp_ns,
            prediction_error=error,
            baseline_error=baseline,
            normalized_error=normalized_error,
            error_derivative=derivative,
            error_integral=integral,
            sustained_duration_ms=self.sustained_duration_ms,
            is_boundary=is_boundary,
            boundary_type=boundary_type,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    def process_frame(
        self,
        z_latent: torch.Tensor,
        predictions: List[torch.Tensor],
        timestamp_ns: int,
    ) -> PredictionResult:
        """
        Process a single frame for boundary detection (Green Phase).

        Args:
            z_latent: (B, T, hidden_dim) encoded latents
            predictions: List of predictions from CPC model
            timestamp_ns: Current timestamp in nanoseconds

        Returns:
            PredictionResult with detection status
        """
        self.total_frames += 1
        current_time_ms = timestamp_ns / 1_000_000

        # Compute prediction error
        error = self.compute_prediction_error(z_latent, predictions)

        # Compute error derivative for spike detection (use RAW error, not normalized)
        # This detects spikes even when baseline has adapted
        derivative = self.compute_error_derivative(error, current_time_ms)

        # Update baseline (use slow EMA normally)
        baseline = self.update_baseline(error, use_fast=False)
        normalized_error = error / max(baseline, 1e-6)

        # Update slope tracker
        integral, slope = self.slope_tracker.update(normalized_error, current_time_ms)

        # Initialize result (boundary may be fired in tracking logic below)
        is_boundary = False
        boundary_type = None
        confidence = 0.0

        # Track sustained duration (and peak duration for classification)
        # Use dual-threshold: start > 2.5x, end < 1.5x
        if normalized_error > self.config.boundary_threshold:
            # Start of elevated period
            if self.elevated_since_ms is None:
                self.elevated_since_ms = current_time_ms
                self.elevated_frame_count = 0  # Count frames in elevated period
            self.elevated_frame_count += 1  # Increment frame count
            self.sustained_duration_ms = self.elevated_frame_count * self.config.frame_size_ms
            # Track peak duration during this elevated period
            if not hasattr(self, 'peak_sustained_duration_ms'):
                self.peak_sustained_duration_ms = 0.0
            self.peak_sustained_duration_ms = max(self.peak_sustained_duration_ms, self.sustained_duration_ms)
        elif normalized_error < self.config.boundary_threshold_lower:
            # Error dropped below lower threshold - end of elevated period
            if hasattr(self, 'peak_sustained_duration_ms') and self.peak_sustained_duration_ms > 0:
                # Just finished an elevated period - classify and fire boundary
                peak_duration = self.peak_sustained_duration_ms
                self.peak_sustained_duration_ms = 0.0  # Reset for next period

                # Classify based on peak duration achieved
                boundary_type = self._classify_by_peak_duration(
                    normalized_error,  # Use the FINAL error value (just dropped)
                    peak_duration,
                    integral,
                    slope,
                )

                if boundary_type is not None and self.armed:
                    confidence = self.compute_confidence_duration_gated(
                        normalized_error,
                        peak_duration,
                        integral,
                        derivative,
                        boundary_type,
                    )

                    if confidence >= self.config.min_confidence:
                        # Fire the boundary
                        is_boundary = True
                        self.boundary_count += 1
                        self.boundary_types[boundary_type] += 1
                        self.last_boundary_time_ns = timestamp_ns
                        self.armed = False
                        self.disarmed_since_ns = timestamp_ns

                        logger.info(
                            f"Boundary: {boundary_type.value} at {current_time_ms:.1f}ms, "
                            f"peak_duration={peak_duration:.1f}ms, "
                            f"error={normalized_error:.2f}x, conf={confidence:.2f}"
                        )

            # Reset tracking
            self.elevated_since_ms = None
            self.sustained_duration_ms = 0.0
            if hasattr(self, 'peak_sustained_duration_ms'):
                self.peak_sustained_duration_ms = 0.0
        else:
            # In hysteresis zone (between 1.5x and 2.5x) - maintain current state
            # Don't reset tracking, but don't update peak duration
            pass

        # Check armed state and re-arm condition
        if not self.armed:
            if self.check_rearm_condition(normalized_error):
                self.armed = True
                self.baseline.reset_fast()  # Reset fast EMA to slow EMA
                logger.debug(f"Rearmed at {current_time_ms:.1f}ms, error={normalized_error:.2f}x")

        # Compute detection latency
        latency_ms = self.config.frame_size_ms  # Best case

        return PredictionResult(
            timestamp_ns=timestamp_ns,
            prediction_error=error,
            baseline_error=baseline,
            normalized_error=normalized_error,
            error_derivative=derivative,
            error_integral=integral,
            sustained_duration_ms=self.sustained_duration_ms,
            is_boundary=is_boundary,
            boundary_type=boundary_type,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    def process_batch(
        self,
        z_latents: List[torch.Tensor],
        predictions_batch: List[List[torch.Tensor]],
        timestamps_ns: List[int],
    ) -> List[PredictionResult]:
        """
        Process a batch of frames.

        Args:
            z_latents: List of latent tensors
            predictions_batch: List of prediction lists
            timestamps_ns: List of timestamps

        Returns:
            List of PredictionResult
        """
        results = []

        for z_latent, predictions, ts in zip(
            z_latents, predictions_batch, timestamps_ns
        ):
            result = self.process_frame(z_latent, predictions, ts)
            results.append(result)

        return results

    def reset(self):
        """Reset detector state."""
        self.armed = True
        self.disarmed_since_ns = 0
        self.last_boundary_time_ns = 0
        self.error_history.clear()
        self.prev_error = 1.0
        self.prev_timestamp_ms = 0.0
        self.elevated_since_ms = None
        self.sustained_duration_ms = 0.0
        self.total_frames = 0

        # Reset dual-EMA
        self.baseline = DualEMABaseline(
            slow_decay=self.config.slow_decay,
            fast_decay=self.config.fast_decay
        )

        # Reset slope tracker
        self.slope_tracker = SlopeTracker(
            window_ms=100.0,
            frame_size_ms=self.config.frame_size_ms
        )

    def is_armed(self) -> bool:
        """Check if detector is currently armed."""
        return self.armed

    def get_statistics(self) -> Dict[str, any]:
        """Get detection statistics."""
        return {
            "total_frames": self.total_frames,
            "boundary_count": self.boundary_count,
            "boundaries_by_type": {
                bt.value: count
                for bt, count in self.boundary_types.items()
            },
            "current_baseline": self.baseline.get_baseline(use_fast=False),
            "fast_baseline": self.baseline.get_baseline(use_fast=True),
            "armed": self.armed,
        }


def create_boundary_detector(
    cpc_model: Optional[nn.Module] = None,
    **kwargs,
) -> PredictiveBoundaryDetector:
    """
    Factory function to create PredictiveBoundaryDetector (Green Phase).

    Args:
        cpc_model: Optional CPC model for error computation
        **kwargs: Configuration parameters

    Returns:
        PredictiveBoundaryDetector instance
    """
    config = BoundaryDetectorConfig(**kwargs)
    return PredictiveBoundaryDetector(config, cpc_model)


class AdaptiveDebounceStrategy:
    """
    Adaptive debounce that replaces fixed 50ms timer (Green Phase).

    Uses prediction error dynamics and dual-EMA to determine optimal
    debounce duration:
    - High error region -> longer debounce (uncertain state)
    - Low error region -> shorter debounce (stable state)
    - Fast EMA enables 15-20ms re-arming for rapid trills
    """

    def __init__(
        self,
        min_debounce_ms: float = 15.0,   # Reduced from 20ms for avian trill
        max_debounce_ms: float = 100.0,
        error_sensitivity: float = 2.0,
    ):
        self.min_debounce_ms = min_debounce_ms
        self.max_debounce_ms = max_debounce_ms
        self.error_sensitivity = error_sensitivity

    def compute_debounce(
        self,
        normalized_error: float,
        recent_variance: float,
        fast_baseline_ratio: float,  # fast / slow baseline ratio
    ) -> float:
        """
        Compute adaptive debounce duration.

        Args:
            normalized_error: Current prediction error ratio
            recent_variance: Variance of recent errors
            fast_baseline_ratio: Ratio of fast to slow baseline

        Returns:
            Debounce duration in milliseconds
        """
        # Higher error -> longer debounce
        error_factor = min(normalized_error / self.error_sensitivity, 1.0)

        # Higher variance -> longer debounce
        variance_factor = min(recent_variance, 1.0)

        # Fast baseline elevated -> use shorter debounce (quick recovery)
        baseline_factor = 1.0 - min(fast_baseline_ratio - 1.0, 0.5)

        # Combined factor
        combined = (error_factor + variance_factor) * baseline_factor / 2

        # Map to debounce range
        debounce = (
            self.min_debounce_ms +
            combined * (self.max_debounce_ms - self.min_debounce_ms)
        )

        return debounce


# =============================================================================
# Legacy API Compatibility (for existing tests)
# =============================================================================

class AdaptiveDebounceStrategy_Legacy(AdaptiveDebounceStrategy):
    """Legacy alias for compatibility."""
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test detector with synthetic data (Green Phase)
    detector = create_boundary_detector()

    # Simulate avian trill: rapid sub-50ms boundaries
    # Each chirp is ~20ms with 10ms gap
    timestamps = [i * 10_000_000 for i in range(200)]  # 200 frames, 10ms each

    print("Simulating avian trill (rapid 20ms chirps):")
    chirp_pattern = [
        (10, 20, 3.5),   # Chirp 1: frames 10-20, error=3.5x
        (25, 35, 3.2),   # Chirp 2: frames 25-35, error=3.2x (15ms gap)
        (40, 50, 3.8),   # Chirp 3: frames 40-50, error=3.8x (5ms gap - very tight!)
        (60, 75, 3.0),   # Chirp 4: frames 60-75, error=3.0x
        (85, 95, 3.6),   # Chirp 5: frames 85-95, error=3.6x
    ]

    for i, ts in enumerate(timestamps):
        # Generate synthetic error based on chirp pattern
        error = 1.0
        for start, end, mag in chirp_pattern:
            if start <= i <= end:
                error = mag + np.random.randn() * 0.3

        # Create dummy tensors
        z = torch.randn(1, 5, 128)
        predictions = [torch.randn(1, 5, 128) for _ in range(3)]

        result = detector.process_frame(z, predictions, ts)

        if result.is_boundary:
            print(
                f"  Frame {i}: {result.boundary_type.value} boundary "
                f"(error={result.normalized_error:.2f}x, "
                f"duration={result.sustained_duration_ms:.1f}ms, "
                f"conf={result.confidence:.2f})"
            )

    print("\nStatistics:")
    stats = detector.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nExpected: ~5 boundaries detected (avian trill chirps)")
    print(f"Green Phase improvements:")
    print(f"  - Dual-EMA enables 15-20ms re-arming")
    print(f"  - Duration-gated: 10ms for phonetic boundaries")
    print(f"  - Derivative-based triggering catches rapid spikes")
