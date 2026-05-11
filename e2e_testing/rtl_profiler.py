#!/usr/bin/env python3
"""
Round-Trip Latency (RTL) Profiler

Measures end-to-end round-trip latency through the Rust→Python→Rust pipeline
using ultrasonic sync pulse injection and detection. Also tracks Predictive NBD
confidence for ONNX/TensorRT optimization validation.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

import numpy as np

logger = logging.getLogger(__name__)


class LatencyBudgetViolation(Exception):
    """Raised when RTL exceeds the target budget."""
    pass


class NBDOptimizationNeeded(Exception):
    """Raised when Predictive NBD confidence indicates optimization needed."""
    pass


@dataclass
class SyncPulseRecord:
    """Record of a sync pulse injection."""
    pulse_id: int
    ptp_timestamp_ns: int
    injection_time_ns: int


@dataclass
class RTLStatistics:
    """Round-trip latency statistics."""
    mean_rtl_ms: float
    p50_rtl_ms: float
    p95_rtl_ms: float
    p99_rtl_ms: float
    max_rtl_ms: float
    sample_count: int

    # Predictive NBD confidence stats
    nbd_confidence_mean: float = 0.0
    nbd_confidence_p5: float = 0.0
    low_confidence_rate: float = 0.0
    low_confidence_count: int = 0


class RoundTripProfiler:
    """
    Measures round-trip latency with ultrasonic sync pulse correlation.

    Tracks:
    - P50/P95/P99/max RTL metrics
    - Predictive NBD confidence under continuous load
    - Latency budget violations
    """

    def __init__(self, target_rtl_ms: float = 50.0):
        """
        Initialize the RTL profiler.

        Args:
            target_rtl_ms: Target round-trip latency in milliseconds
        """
        self.target_rtl_ms = target_rtl_ms

        # Sync pulse tracking - maps PTP timestamp to injection time
        self.sync_pulse_injections: Dict[int, int] = {}
        self.sync_pulse_detections: Dict[int, int] = {}
        self.rtl_history: List[float] = []

        # Predictive NBD confidence tracking
        self.nbd_confidence_history: List[float] = []
        self.low_confidence_count = 0

    def record_injection(
        self,
        ptp_timestamp_ns: int,
        injection_time_ns: int,
    ) -> None:
        """
        Record a sync pulse injection into the audio stream.

        Args:
            ptp_timestamp_ns: PTP timestamp when pulse was injected
            injection_time_ns: Wall-clock time of injection
        """
        self.sync_pulse_injections[ptp_timestamp_ns] = injection_time_ns
        logger.debug(f"Recorded sync pulse injection at PTP={ptp_timestamp_ns}")

    def get_injection_time(self, ptp_timestamp_ns: int) -> int:
        """
        Get the injection time for a given PTP timestamp.

        Args:
            ptp_timestamp_ns: PTP timestamp of the pulse

        Returns:
            Injection time in nanoseconds

        Raises:
            KeyError: If PTP timestamp not found
        """
        if ptp_timestamp_ns not in self.sync_pulse_injections:
            raise KeyError(f"PTP timestamp {ptp_timestamp_ns} not found in injection records")
        return self.sync_pulse_injections[ptp_timestamp_ns]

    def record_detection(self, ptp_timestamp_ns: int, detection_time_ns: int) -> float:
        """
        Record sync pulse detection in synthesized output.

        Args:
            ptp_timestamp_ns: PTP timestamp of the injected pulse
            detection_time_ns: Wall-clock time of detection

        Returns:
            rtl_ms: Round-trip latency in milliseconds

        Raises:
            KeyError: If PTP timestamp not found in injection records
            LatencyBudgetViolation: If RTL exceeds target
        """
        if ptp_timestamp_ns not in self.sync_pulse_injections:
            raise KeyError(f"PTP timestamp {ptp_timestamp_ns} not found in injection records")

        self.sync_pulse_detections[ptp_timestamp_ns] = detection_time_ns

        # Calculate RTL
        injection_time = self.sync_pulse_injections[ptp_timestamp_ns]
        rtl_ns = detection_time_ns - injection_time
        rtl_ms = rtl_ns / 1_000_000.0
        self.rtl_history.append(rtl_ms)

        logger.debug(f"Sync pulse RTL: {rtl_ms:.2f}ms")

        # Check budget violation
        if rtl_ms > self.target_rtl_ms:
            raise LatencyBudgetViolation(
                f"RTL exceeded: {rtl_ms:.2f}ms > {self.target_rtl_ms:.2f}ms"
            )

        return rtl_ms

    def record_nbd_confidence(self, confidence: float, boundary_type: str) -> None:
        """
        Track Predictive NBD confidence during continuous streaming.

        This validates that ONNX/TensorRT execution graphs are optimized
        and EMA baseline tracking is stable under load.

        Args:
            confidence: NBD confidence score (0-1)
            boundary_type: Type of boundary detected

        Raises:
            NBDOptimizationNeeded: If low confidence rate exceeds threshold
        """
        self.nbd_confidence_history.append(confidence)

        # Track low confidence events (exclude Noise boundaries)
        if confidence < 0.6 and boundary_type != "Noise":
            self.low_confidence_count += 1
            logger.warning(
                f"Low NBD confidence: {confidence:.2f} on {boundary_type} "
                f"(count: {self.low_confidence_count})"
            )

        # Check if >10% low confidence in recent window
        if len(self.nbd_confidence_history) > 100:
            recent = self.nbd_confidence_history[-100:]
            low_count = sum(1 for c in recent if c < 0.6)
            if low_count > 10:
                raise NBDOptimizationNeeded(
                    f"High low-confidence rate: {low_count}% in recent window"
                )

    def get_statistics(self) -> RTLStatistics:
        """
        Calculate RTL and NBD confidence statistics.

        Returns:
            RTLStatistics: Current statistics (RTL stats are 0 if no samples)
        """
        # NBD confidence stats (always calculate)
        if self.nbd_confidence_history:
            conf_array = np.array(self.nbd_confidence_history)
            conf_mean = float(np.mean(conf_array))
            conf_p5 = float(np.percentile(conf_array, 5))
            low_rate = self.low_confidence_count / len(self.nbd_confidence_history)
        else:
            conf_mean = 0.0
            conf_p5 = 0.0
            low_rate = 0.0

        # RTL stats (0 if no samples)
        if not self.rtl_history:
            return RTLStatistics(
                mean_rtl_ms=0.0,
                p50_rtl_ms=0.0,
                p95_rtl_ms=0.0,
                p99_rtl_ms=0.0,
                max_rtl_ms=0.0,
                sample_count=0,
                nbd_confidence_mean=conf_mean,
                nbd_confidence_p5=conf_p5,
                low_confidence_rate=low_rate,
                low_confidence_count=self.low_confidence_count,
            )

        rtl_array = np.array(self.rtl_history)

        return RTLStatistics(
            mean_rtl_ms=float(np.mean(rtl_array)),
            p50_rtl_ms=float(np.percentile(rtl_array, 50)),
            p95_rtl_ms=float(np.percentile(rtl_array, 95)),
            p99_rtl_ms=float(np.percentile(rtl_array, 99)),
            max_rtl_ms=float(np.max(rtl_array)),
            sample_count=len(self.rtl_history),
            nbd_confidence_mean=conf_mean,
            nbd_confidence_p5=conf_p5,
            low_confidence_rate=low_rate,
            low_confidence_count=self.low_confidence_count,
        )

    def reset(self) -> None:
        """Reset all tracking state."""
        self.sync_pulse_injections.clear()
        self.sync_pulse_detections.clear()
        self.rtl_history.clear()
        self.nbd_confidence_history.clear()
        self.low_confidence_count = 0
        logger.info("RTL profiler reset")
