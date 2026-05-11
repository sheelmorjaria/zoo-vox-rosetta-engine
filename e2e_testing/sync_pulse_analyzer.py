#!/usr/bin/env python3
"""
Sync Pulse Analyzer

Correlates ultrasonic sync pulses between injection and detection to validate
round-trip latency measurements. Handles pulse matching and outlier detection.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PulseMatch:
    """Result of matching an injected pulse to a detected pulse."""
    pulse_id: int
    injection_time_ns: int
    detection_time_ns: int
    rtl_ms: float
    confidence: float  # 0-1, based on correlation match quality


@dataclass
class PulseDetection:
    """Detected ultrasonic pulse in output audio."""
    timestamp_ns: int
    frequency_hz: float
    amplitude: float
    snr_db: float


class SyncPulseAnalyzer:
    """
    Analyzes sync pulse correlation for RTL validation.

    Features:
    - Cross-correlation based pulse matching
    - Outlier detection for missed/false detections
    - Confidence scoring for match quality
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        pulse_frequency_hz: int = 80000,
        pulse_duration_ms: float = 1.0,
        correlation_threshold: float = 0.8,
    ):
        """
        Initialize the sync pulse analyzer.

        Args:
            sample_rate: Audio sample rate in Hz
            pulse_frequency_hz: Sync pulse frequency (ultrasonic)
            pulse_duration_ms: Pulse duration in milliseconds
            correlation_threshold: Minimum correlation for valid match
        """
        self.sample_rate = sample_rate
        self.pulse_frequency_hz = pulse_frequency_hz
        self.pulse_duration_samples = int(
            sample_rate * pulse_duration_ms / 1000
        )
        self.correlation_threshold = correlation_threshold

        # Generate reference pulse template
        self._generate_reference_pulse()

    def _generate_reference_pulse(self) -> None:
        """Generate the reference ultrasonic pulse template."""
        import numpy as np

        t = np.linspace(
            0,
            self.pulse_duration_samples / self.sample_rate,
            self.pulse_duration_samples,
            endpoint=False,
        )

        # Generate ultrasonic sine wave with Hann window
        self.reference_pulse = np.sin(2 * np.pi * self.pulse_frequency_hz * t)
        window = np.hanning(len(self.reference_pulse))
        self.reference_pulse = self.reference_pulse * window

        # Normalize
        self.reference_pulse = (
            self.reference_pulse / np.max(np.abs(self.reference_pulse))
        )

    def detect_pulses(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[PulseDetection]:
        """
        Detect ultrasonic sync pulses in audio buffer.

        Args:
            audio: Audio samples (normalized -1 to 1)
            sample_rate: Sample rate of audio

        Returns:
            List of detected pulses
        """
        if sample_rate != self.sample_rate:
            logger.warning(
                f"Sample rate mismatch: expected {self.sample_rate}, got {sample_rate}"
            )

        detections = []

        # Use cross-correlation to find pulses
        correlation = np.correlate(audio, self.reference_pulse, mode="valid")
        correlation = correlation / (np.std(audio) * np.std(self.reference_pulse) * len(self.reference_pulse))

        # Find peaks above threshold
        threshold = np.mean(correlation) + 3 * np.std(correlation)
        peaks = np.where(correlation > threshold)[0]

        # Group nearby peaks (debounce)
        min_distance = self.pulse_duration_samples // 2
        filtered_peaks = []
        for peak in peaks:
            if not filtered_peaks or peak - filtered_peaks[-1] > min_distance:
                filtered_peaks.append(peak)

        # Convert to detections
        for peak_idx in filtered_peaks:
            # Estimate SNR
            signal_region = audio[peak_idx:peak_idx + self.pulse_duration_samples]
            noise_region = audio[max(0, peak_idx - self.pulse_duration_samples):peak_idx]

            if len(signal_region) > 0 and len(noise_region) > 0:
                signal_power = np.mean(signal_region ** 2)
                noise_power = np.mean(noise_region ** 2)
                snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

                # Only include high-SNR detections
                if snr_db > 10:
                    detections.append(PulseDetection(
                        timestamp_ns=int(peak_idx / sample_rate * 1_000_000_000),
                        frequency_hz=self.pulse_frequency_hz,
                        amplitude=float(np.max(np.abs(signal_region))),
                        snr_db=float(snr_db),
                    ))

        logger.debug(f"Detected {len(detections)} sync pulses in {len(audio)/sample_rate:.2f}s audio")
        return detections

    def match_pulse(
        self,
        detection: PulseDetection,
        expected_time_ns: int,
        time_window_ns: int = 100_000_000,  # 100ms window
    ) -> Optional[PulseMatch]:
        """
        Match a detected pulse to an expected injection time.

        Args:
            detection: Detected pulse
            expected_time_ns: Expected injection time
            time_window_ns: Time window for matching

        Returns:
            PulseMatch if within window, None otherwise
        """
        time_diff_ns = abs(detection.timestamp_ns - expected_time_ns)

        if time_diff_ns <= time_window_ns:
            rtl_ms = time_diff_ns / 1_000_000.0
            # Confidence based on SNR
            confidence = min(1.0, (detection.snr_db - 10) / 20.0)

            return PulseMatch(
                pulse_id=0,  # Will be filled by caller
                injection_time_ns=expected_time_ns,
                detection_time_ns=detection.timestamp_ns,
                rtl_ms=rtl_ms,
                confidence=confidence,
            )

        return None

    def calculate_outlier_statistics(
        self,
        rtl_values: List[float],
    ) -> dict:
        """
        Calculate outlier statistics for RTL measurements.

        Args:
            rtl_values: List of RTL measurements in ms

        Returns:
            Dictionary with outlier statistics
        """
        if not rtl_values:
            return {}

        rtl_array = np.array(rtl_values)
        q1 = np.percentile(rtl_array, 25)
        q3 = np.percentile(rtl_array, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = rtl_array[(rtl_array < lower_bound) | (rtl_array > upper_bound)]

        return {
            "count": len(rtl_values),
            "outlier_count": len(outliers),
            "outlier_rate": len(outliers) / len(rtl_values) if rtl_values.size > 0 else 0,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outliers": [float(v) for v in outliers],
        }
