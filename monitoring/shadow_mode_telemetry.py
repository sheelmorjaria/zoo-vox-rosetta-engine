#!/usr/bin/env python3
"""
Shadow Mode Telemetry and Monitoring

Monitors the full ALP pipeline in passive listening mode before
engaging in closed-loop interaction with the colony.

Tracks latencies, memory usage, thermal state, and generates
synthetic audio quality metrics without playback.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameTiming:
    """Timing data for a single processed frame."""
    timestamp: float
    nbd_latency_ms: float
    bio_mae_latency_ms: float
    vae_latency_ms: float
    vqvae_latency_ms: float
    agent_latency_ms: float
    ddsp_latency_ms: float
    total_latency_ms: float
    memory_mb: float
    temperature_c: float


@dataclass
class ShadowModeReport:
    """Summary report for shadow mode operation."""
    duration_seconds: float
    frames_processed: int
    frame_rate_hz: float

    # Latency percentiles (ms)
    p50_latency: float
    p95_latency: float
    p99_latency: float
    max_latency: float

    # Component breakdown
    mean_nbd_latency: float
    mean_bio_mae_latency: float
    mean_vae_latency: float
    mean_vqvae_latency: float
    mean_agent_latency: float
    mean_ddsp_latency: float

    # System health
    max_memory_mb: float
    max_temperature_c: float
    memory_leak_detected: bool

    # Quality metrics
    phase_discontinuity_score: float
    spectral_smoothness: float


class ShadowModeTelemetry:
    """
    Telemetry collector for shadow mode operation.

    Tracks all aspects of pipeline performance without
    engaging in actual playback to the colony.
    """

    def __init__(
        self,
        window_size: int = 1000,
        memory_threshold_mb: float = 4096,
        temperature_threshold_c: float = 70,
    ):
        """
        Initialize telemetry collector.

        Args:
            window_size: Number of frames to keep in rolling window
            memory_threshold_mb: Warning threshold for memory usage
            temperature_threshold_c: Warning threshold for temperature
        """
        self.window_size = window_size
        self.memory_threshold = memory_threshold_mb
        self.temp_threshold = temperature_threshold_c

        # Rolling metrics
        self.timings: Deque[FrameTiming] = deque(maxlen=window_size)
        self.audio_outputs: List[np.ndarray] = []

        # Start time
        self.start_time = time.time()
        self.frame_count = 0

        # Alerts
        self.alerts: List[str] = []

        logger.info("ShadowModeTelemetry initialized")

    def record_frame(
        self,
        nbd_latency_ms: float,
        bio_mae_latency_ms: float,
        vae_latency_ms: float,
        vqvae_latency_ms: float,
        agent_latency_ms: float,
        ddsp_latency_ms: float,
        audio_output: Optional[np.ndarray] = None,
    ) -> None:
        """
        Record timing data for a processed frame.

        Args:
            nbd_latency_ms: NBD segmentation latency
            bio_mae_latency_ms: BioMAE embedding latency
            vae_latency_ms: VAE encoding latency
            vqvae_latency_ms: VQ-VAE tokenization latency
            agent_latency_ms: Agent decision latency
            ddsp_latency_ms: DDSP synthesis latency
            audio_output: Synthesized audio (for quality metrics)
        """
        import psutil
        import os

        # Compute total latency
        total_latency = (
            nbd_latency_ms +
            bio_mae_latency_ms +
            vae_latency_ms +
            vqvae_latency_ms +
            agent_latency_ms +
            ddsp_latency_ms
        )

        # Get system metrics
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        # Temperature (Linux-specific)
        temperature_c = self._read_temperature()

        # Create timing record
        timing = FrameTiming(
            timestamp=time.time(),
            nbd_latency_ms=nbd_latency_ms,
            bio_mae_latency_ms=bio_mae_latency_ms,
            vae_latency_ms=vae_latency_ms,
            vqvae_latency_ms=vqvae_latency_ms,
            agent_latency_ms=agent_latency_ms,
            ddsp_latency_ms=ddsp_latency_ms,
            total_latency_ms=total_latency,
            memory_mb=memory_mb,
            temperature_c=temperature_c,
        )

        self.timings.append(timing)
        self.frame_count += 1

        # Store audio for quality analysis (keep last 10)
        if audio_output is not None:
            self.audio_outputs.append(audio_output)
            if len(self.audio_outputs) > 10:
                self.audio_outputs.pop(0)

        # Check for alerts
        self._check_alerts(timing)

    def _read_temperature(self) -> float:
        """Read CPU temperature (Linux-specific)."""
        try:
            # Try reading from thermal zones
            for zone in ["/sys/class/thermal/thermal_zone0/temp",
                        "/sys/class/thermal/thermal_zone1/temp"]:
                try:
                    with open(zone, "r") as f:
                        temp_millidegrees = int(f.read().strip())
                        return temp_millidegrees / 1000.0
                except (FileNotFoundError, ValueError):
                    continue
        except Exception:
            pass
        return 0.0  # Not available

    def _check_alerts(self, timing: FrameTiming) -> None:
        """Check for alert conditions."""
        # Latency alert
        if timing.total_latency_ms > 100:
            self.alerts.append(
                f"[{datetime.now().isoformat()}] "
                f"High latency: {timing.total_latency_ms:.1f}ms"
            )

        # Memory alert
        if timing.memory_mb > self.memory_threshold:
            self.alerts.append(
                f"[{datetime.now().isoformat()}] "
                f"High memory: {timing.memory_mb:.1f}MB"
            )

        # Temperature alert
        if timing.temperature_c > self.temp_threshold and timing.temperature_c > 0:
            self.alerts.append(
                f"[{datetime.now().isoformat()}] "
                f"High temperature: {timing.temperature_c:.1f}°C"
            )

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current rolling metrics."""
        if not self.timings:
            return {}

        timings_list = list(self.timings)

        return {
            "frame_rate": self.frame_count / (time.time() - self.start_time),
            "p50_latency_ms": np.percentile([t.total_latency_ms for t in timings_list], 50),
            "p95_latency_ms": np.percentile([t.total_latency_ms for t in timings_list], 95),
            "p99_latency_ms": np.percentile([t.total_latency_ms for t in timings_list], 99),
            "max_latency_ms": max(t.total_latency_ms for t in timings_list),
            "mean_memory_mb": np.mean([t.memory_mb for t in timings_list]),
            "max_memory_mb": max(t.memory_mb for t in timings_list),
            "max_temperature_c": max(t.temperature_c for t in timings_list),
            "alert_count": len(self.alerts),
        }

    def check_memory_leak(self) -> bool:
        """
        Check for memory leak by comparing memory trend.

        Returns:
            True if leak detected (monotonic increase >10% over window)
        """
        if len(self.timings) < 100:
            return False

        # Compare first 10% to last 10% of window
        timings_list = list(self.timings)
        n = len(timings_list)
        first_quartile = timings_list[:n//10]
        last_quartile = timings_list[-n//10:]

        first_mean = np.mean([t.memory_mb for t in first_quartile])
        last_mean = np.mean([t.memory_mb for t in last_quartile])

        # Check for >10% increase
        leak_detected = (last_mean - first_mean) / first_mean > 0.1

        if leak_detected:
            logger.warning(
                f"Memory leak detected: {first_mean:.1f}MB → {last_mean:.1f}MB"
            )

        return leak_detected

    def generate_report(self) -> ShadowModeReport:
        """Generate comprehensive shadow mode report."""
        if not self.timings:
            return ShadowModeReport(
                duration_seconds=0,
                frames_processed=0,
                frame_rate_hz=0,
                p50_latency=0,
                p95_latency=0,
                p99_latency=0,
                max_latency=0,
                mean_nbd_latency=0,
                mean_bio_mae_latency=0,
                mean_vae_latency=0,
                mean_vqvae_latency=0,
                mean_agent_latency=0,
                mean_ddsp_latency=0,
                max_memory_mb=0,
                max_temperature_c=0,
                memory_leak_detected=False,
                phase_discontinuity_score=0,
                spectral_smoothness=0,
            )

        timings_list = list(self.timings)
        duration = time.time() - self.start_time

        # Compute percentiles
        latencies = [t.total_latency_ms for t in timings_list]
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        max_lat = max(latencies)

        # Component means
        nbd_mean = np.mean([t.nbd_latency_ms for t in timings_list])
        bio_mae_mean = np.mean([t.bio_mae_latency_ms for t in timings_list])
        vae_mean = np.mean([t.vae_latency_ms for t in timings_list])
        vqvae_mean = np.mean([t.vqvae_latency_ms for t in timings_list])
        agent_mean = np.mean([t.agent_latency_ms for t in timings_list])
        ddsp_mean = np.mean([t.ddsp_latency_ms for t in timings_list])

        # System health
        max_mem = max(t.memory_mb for t in timings_list)
        max_temp = max(t.temperature_c for t in timings_list)
        memory_leak = self.check_memory_leak()

        # Audio quality metrics
        phase_score = 0.0
        smooth_score = 0.0
        if self.audio_outputs:
            phase_score = self._compute_phase_discontinuity(
                self.audio_outputs[-1], 48000
            )
            smooth_score = self._compute_spectral_smoothness(
                self.audio_outputs[-1], 48000
            )

        return ShadowModeReport(
            duration_seconds=duration,
            frames_processed=self.frame_count,
            frame_rate_hz=self.frame_count / duration,
            p50_latency=p50,
            p95_latency=p95,
            p99_latency=p99,
            max_latency=max_lat,
            mean_nbd_latency=nbd_mean,
            mean_bio_mae_latency=bio_mae_mean,
            mean_vae_latency=vae_mean,
            mean_vqvae_latency=vqvae_mean,
            mean_agent_latency=agent_mean,
            mean_ddsp_latency=ddsp_mean,
            max_memory_mb=max_mem,
            max_temperature_c=max_temp,
            memory_leak_detected=memory_leak,
            phase_discontinuity_score=phase_score,
            spectral_smoothness=smooth_score,
        )

    def _compute_phase_discontinuity(
        self, audio: np.ndarray, sample_rate: int
    ) -> float:
        """Compute phase discontinuity score."""
        try:
            from scipy import signal
        except ImportError:
            return 0.0

        # Compute analytic signal
        analytic = signal.hilbert(audio)

        # Extract instantaneous phase
        inst_phase = np.unwrap(np.angle(analytic))

        # Compute phase derivative
        phase_diff = np.diff(inst_phase)

        # Detect large jumps (discontinuities)
        threshold = np.pi
        discontinuities = np.sum(np.abs(phase_diff) > threshold)

        # Normalize by audio length
        return discontinuities / len(audio) * 1000

    def _compute_spectral_smoothness(
        self, audio: np.ndarray, sample_rate: int
    ) -> float:
        """Compute spectral smoothness across time."""
        try:
            from scipy import signal
        except ImportError:
            return 0.0

        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(audio, fs=sample_rate, nperseg=512)

        # Compute spectral centroid trajectory
        centroid = []
        for col in range(Sxx.shape[1]):
            freq_weights = f * Sxx[:, col]
            cent = np.sum(freq_weights) / (np.sum(Sxx[:, col]) + 1e-10)
            centroid.append(cent)

        centroid = np.array(centroid)

        # Compute second derivative (curvature)
        if len(centroid) > 2:
            curvature = np.diff(np.diff(centroid))
            smoothness = 1.0 / (1.0 + np.mean(np.abs(curvature)))
            return smoothness

        return 0.0

    def save_report(self, filepath: str) -> None:
        """Save report to file."""
        report = self.generate_report()

        with open(filepath, "w") as f:
            f.write("Shadow Mode Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Duration: {report.duration_seconds:.1f}s\n")
            f.write(f"Frames Processed: {report.frames_processed}\n")
            f.write(f"Frame Rate: {report.frame_rate_hz:.1f} Hz\n\n")

            f.write("Latency (ms):\n")
            f.write(f"  P50: {report.p50_latency:.1f}\n")
            f.write(f"  P95: {report.p95_latency:.1f}\n")
            f.write(f"  P99: {report.p99_latency:.1f}\n")
            f.write(f"  Max: {report.max_latency:.1f}\n\n")

            f.write("Component Breakdown (ms):\n")
            f.write(f"  NBD: {report.mean_nbd_latency:.1f}\n")
            f.write(f"  BioMAE: {report.mean_bio_mae_latency:.1f}\n")
            f.write(f"  VAE: {report.mean_vae_latency:.1f}\n")
            f.write(f"  VQ-VAE: {report.mean_vqvae_latency:.1f}\n")
            f.write(f"  Agent: {report.mean_agent_latency:.1f}\n")
            f.write(f"  DDSP: {report.mean_ddsp_latency:.1f}\n\n")

            f.write("System Health:\n")
            f.write(f"  Max Memory: {report.max_memory_mb:.1f} MB\n")
            f.write(f"  Max Temperature: {report.max_temperature_c:.1f}°C\n")
            f.write(f"  Memory Leak: {report.memory_leak_detected}\n\n")

            f.write("Audio Quality:\n")
            f.write(f"  Phase Discontinuity: {report.phase_discontinuity_score:.2f}\n")
            f.write(f"  Spectral Smoothness: {report.spectral_smoothness:.2f}\n")

        logger.info(f"Report saved to {filepath}")


class ShadowModeRunner:
    """
    Runs the full ALP pipeline in shadow mode.

    Processes audio through the entire pipeline but records
    synthesized output instead of playing it.
    """

    def __init__(
        self,
        nbd_layer,
        bio_mae,
        vae_encoder,
        vqvae_encoder,
        interaction_agent,
        ddsp_synthesizer,
        telemetry: ShadowModeTelemetry,
    ):
        """
        Initialize shadow mode runner.

        Args:
            nbd_layer: Neural Boundary Detector
            bio_mae: BioMAE encoder
            vae_encoder: VAE encoder (Stream 1)
            vqvae_encoder: VQ-VAE encoder (Stream 2)
            interaction_agent: DualStreamInteractionAgent
            ddsp_synthesizer: DDSP synthesizer
            telemetry: Telemetry collector
        """
        self.nbd = nbd_layer
        self.bio_mae = bio_mae
        self.vae = vae_encoder
        self.vqvae = vqvae_encoder
        self.agent = interaction_agent
        self.ddsp = ddsp_synthesizer
        self.telemetry = telemetry

    def process_audio_chunk(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Process audio chunk through full pipeline.

        Args:
            audio: Input audio samples

        Returns:
            Synthesized output audio (recorded, not played)
        """
        import time

        t0 = time.time()

        # Step 1: NBD segmentation
        t_nbd_start = time.time()
        boundaries = self.nbd.segment(audio)
        nbd_latency = (time.time() - t_nbd_start) * 1000

        if not boundaries:
            return None

        # Process first segment (simplified)
        segment = boundaries[0].audio

        # Step 2: BioMAE embedding
        t_mae_start = time.time()
        features_112d = self.bio_mae.encode(segment)
        mae_latency = (time.time() - t_mae_start) * 1000

        # Step 3: VAE encoding (Stream 1)
        t_vae_start = time.time()
        affect_vector = self.vae.encode(features_112d)
        vae_latency = (time.time() - t_vae_start) * 1000

        # Step 4: VQ-VAE encoding (Stream 2)
        t_vqvae_start = time.time()
        syntactic_token = self.vqvae.encode(features_112d)
        vqvae_latency = (time.time() - t_vqvae_start) * 1000

        # Step 5: Agent decision
        t_agent_start = time.time()
        action = self.agent.decide_response(affect_vector, syntactic_token)
        agent_latency = (time.time() - t_agent_start) * 1000

        # Step 6: DDSP synthesis
        t_ddsp_start = time.time()
        synthesized = self.ddsp.synthesize_dual_stream(
            action.syntactic_token,
            action.affect_vector
        )
        ddsp_latency = (time.time() - t_ddsp_start) * 1000

        # Record telemetry
        self.telemetry.record_frame(
            nbd_latency_ms=nbd_latency,
            bio_mae_latency_ms=mae_latency,
            vae_latency_ms=vae_latency,
            vqvae_latency_ms=vqvae_latency,
            agent_latency_ms=agent_latency,
            ddsp_latency_ms=ddsp_latency,
            audio_output=synthesized,
        )

        return synthesized

    def run_shadow_mode(
        self,
        audio_source,
        duration_seconds: float = 3600,
    ) -> ShadowModeReport:
        """
        Run shadow mode for specified duration.

        Args:
            audio_source: Audio source (file or stream)
            duration_seconds: How long to run

        Returns:
            ShadowModeReport with telemetry summary
        """
        logger.info(f"Starting shadow mode for {duration_seconds}s")

        start_time = time.time()
        chunk_count = 0

        while time.time() - start_time < duration_seconds:
            # Get audio chunk
            audio = audio_source.get_chunk()
            if audio is None:
                break

            # Process through pipeline
            self.process_audio_chunk(audio)
            chunk_count += 1

            # Log progress every 100 chunks
            if chunk_count % 100 == 0:
                metrics = self.telemetry.get_current_metrics()
                logger.info(
                    f"Processed {chunk_count} chunks, "
                    f"P99 latency: {metrics.get('p99_latency_ms', 0):.1f}ms"
                )

        # Generate final report
        report = self.telemetry.generate_report()
        logger.info(f"Shadow mode complete: {report.frames_processed} frames processed")

        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    telemetry = ShadowModeTelemetry()

    # Simulate some telemetry
    import time
    import random

    print("Shadow Mode Telemetry Demo")
    print("=" * 50)

    for i in range(100):
        # Simulate varying latencies
        nbd = random.uniform(2, 5)
        mae = random.uniform(5, 10)
        vae = random.uniform(2, 4)
        vqvae = random.uniform(2, 4)
        agent = random.uniform(10, 20)
        ddsp = random.uniform(30, 50)

        telemetry.record_frame(
            nbd_latency_ms=nbd,
            bio_mae_latency_ms=mae,
            vae_latency_ms=vae,
            vqvae_latency_ms=vqvae,
            agent_latency_ms=agent,
            ddsp_latency_ms=ddsp,
        )

        time.sleep(0.01)

    # Generate report
    report = telemetry.generate_report()

    print(f"\nShadow Mode Report:")
    print(f"  Frames: {report.frames_processed}")
    print(f"  Frame Rate: {report.frame_rate_hz:.1f} Hz")
    print(f"  P50 Latency: {report.p50_latency:.1f}ms")
    print(f"  P95 Latency: {report.p95_latency:.1f}ms")
    print(f"  P99 Latency: {report.p99_latency:.1f}ms")
    print(f"  Max Memory: {report.max_memory_mb:.1f}MB")
    print(f"  Memory Leak: {report.memory_leak_detected}")
