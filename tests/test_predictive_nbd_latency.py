#!/usr/bin/env python3
"""
Phase 1.1: Execution Speed Profiling for Predictive NBD

Verifies that the total NBD pipeline latency does not exceed 12ms:
- ONNX 1D Conv Encoder: ≤ 5ms
- Autoregressive Model: ≤ 5ms
- MSE Error Computation: ≤ 1ms
- Adaptive Boundary Logic: < 1ms

Test Protocol: Feed 10,000 continuous audio frames (10ms resolution)
and measure 99th-percentile latency.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """Latency statistics for a component."""
    name: str
    samples: List[float]
    budget_ms: float

    @property
    def avg_ms(self) -> float:
        return np.mean(self.samples) * 1000

    @property
    def p95_ms(self) -> float:
        return np.percentile(self.samples, 95) * 1000

    @property
    def p99_ms(self) -> float:
        return np.percentile(self.samples, 99) * 1000

    @property
    def min_ms(self) -> float:
        return np.min(self.samples) * 1000

    @property
    def max_ms(self) -> float:
        return np.max(self.samples) * 1000

    @property
    def within_budget(self) -> bool:
        return self.p99_ms <= self.budget_ms

    def report(self) -> str:
        status = "✓ PASS" if self.within_budget else "✗ FAIL"
        return (
            f"{self.name}:\n"
            f"  Budget: {self.budget_ms:.2f}ms\n"
            f"  Avg: {self.avg_ms:.3f}ms, "
            f"P95: {self.p95_ms:.3f}ms, "
            f"P99: {self.p99_ms:.3f}ms\n"
            f"  Min: {self.min_ms:.3f}ms, "
            f"Max: {self.max_ms:.3f}ms\n"
            f"  {status}"
        )


class LatencyProfiler:
    """
    Profile latency of Predictive NBD components.

    Measures:
    1. ONNX Encoder latency
    2. Autoregressive model latency
    3. MSE computation latency
    4. Boundary logic latency
    5. Total end-to-end latency
    """

    def __init__(self, sample_rate: int = 48000, frame_size_ms: float = 10.0):
        self.sample_rate = sample_rate
        self.frame_size_ms = frame_size_ms
        self.frame_size = int(sample_rate * frame_size_ms / 1000)
        self.hidden_dim = 128
        self.steps_ahead = 5

        # Budget allocations (from TDD)
        self.budgets = {
            "encoder": 5.0,      # ONNX 1D Conv Encoder
            "ar_model": 5.0,      # Autoregressive Model
            "mse": 1.0,           # MSE Error Computation
            "logic": 1.0,         # Adaptive Boundary Logic
            "total": 12.0,        # Total end-to-end
        }

        # Mock components (replace with real ONNX models)
        self._setup_components()

    def _setup_components(self):
        """Setup mock or real components for profiling."""
        # Mock encoder (1D Conv)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=5),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, self.hidden_dim, kernel_size=4, stride=2, padding=2),
            nn.Flatten(),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim),
        )
        self.encoder.eval()

        # Mock AR model
        self.ar_model = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim),
        )
        self.ar_model.eval()

    def profile_encoder(self, audio: torch.Tensor) -> float:
        """Profile encoder latency."""
        start = time.perf_counter()
        with torch.no_grad():
            audio_reshaped = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
            z = self.encoder(audio_reshaped)
            # Simulate latent reshaping
            z = z.view(1, -1, self.hidden_dim)  # (batch, seq, hidden)
        end = time.perf_counter()
        return end - start

    def profile_ar_model(self, z: torch.Tensor) -> List[torch.Tensor]:
        """Profile autoregressive model latency."""
        predictions = []
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(self.steps_ahead):
                pred = self.ar_model(z)
                predictions.append(pred)
        end = time.perf_counter()
        self._last_ar_latency = end - start
        return predictions

    def profile_mse_computation(
        self,
        z: torch.Tensor,
        predictions: List[torch.Tensor]
    ) -> float:
        """Profile MSE computation latency."""
        start = time.perf_counter()
        with torch.no_grad():
            total_error = 0.0
            count = 0
            for k, prediction in enumerate(predictions):
                # Simulate prediction error computation
                diff = z[:, -1:, :] - prediction[:, :1, :]
                mse = torch.sum(diff ** 2).item()
                total_error += mse
                count += 1
            error = total_error / max(count, 1)
        end = time.perf_counter()
        return end - start

    def profile_boundary_logic(
        self,
        error: float,
        baseline: float,
        armed: bool,
        normalized_error: float,
    ) -> float:
        """Profile boundary detection logic latency."""
        start = time.perf_counter()

        # Simulate boundary logic
        threshold = 2.5
        rearm_threshold = 1.2
        min_confidence = 0.6

        # Check rearm
        new_armed = armed or (normalized_error < rearm_threshold)

        # Check boundary
        is_boundary = new_armed and (normalized_error >= threshold)

        # Classify boundary type
        if is_boundary:
            if normalized_error >= 4.0:
                boundary_type = "phrase"
            elif normalized_error >= 3.0:
                boundary_type = "syllable"
            else:
                boundary_type = "phonetic"
        else:
            boundary_type = None

        # Compute confidence
        if is_boundary:
            confidence = min(1.0, normalized_error / 4.0)
            confidence = min(confidence + 0.2, 1.0)  # Type boost
        else:
            confidence = 0.0

        # Check min confidence
        final_boundary = is_boundary and (confidence >= min_confidence)

        end = time.perf_counter()
        return end - start

    def run_benchmark(
        self,
        num_frames: int = 10000,
        warmup_frames: int = 100,
    ) -> Dict[str, LatencyStats]:
        """
        Run full latency benchmark.

        Args:
            num_frames: Number of frames to process
            warmup_frames: Number of warmup frames (not counted in stats)

        Returns:
            Dictionary of latency statistics for each component
        """
        logger.info(f"Running latency benchmark: {num_frames} frames, {warmup_frames} warmup")

        # Initialize stats storage
        stats = {
            "encoder": LatencyStats("encoder", [], self.budgets["encoder"]),
            "ar_model": LatencyStats("ar_model", [], self.budgets["ar_model"]),
            "mse": LatencyStats("mse", [], self.budgets["mse"]),
            "logic": LatencyStats("logic", [], self.budgets["logic"]),
            "total": LatencyStats("total", [], self.budgets["total"]),
        }

        # Warmup
        logger.info(f"Warming up with {warmup_frames} frames...")
        for i in range(warmup_frames):
            audio = torch.randn(self.frame_size)
            z = torch.randn(1, 1, self.hidden_dim)
            predictions = [torch.randn(1, 1, self.hidden_dim) for _ in range(self.steps_ahead)]
            self.profile_encoder(audio)
            self.profile_ar_model(z)
            self.profile_mse_computation(z, predictions)
            self.profile_boundary_logic(1.0, 1.0, True, 1.0)

        logger.info("Warmup complete. Starting benchmark...")

        # Benchmark
        armed = True
        baseline = 1.0

        for i in range(num_frames):
            # Generate test data
            audio = torch.randn(self.frame_size)
            z = torch.randn(1, 1, self.hidden_dim)

            # Profile each component
            t_start = time.perf_counter()

            # 1. Encoder
            t_enc = self.profile_encoder(audio)

            # 2. AR Model
            predictions = self.profile_ar_model(z)
            t_ar = self._last_ar_latency

            # 3. MSE Computation
            t_mse = self.profile_mse_computation(z, predictions)

            # 4. Boundary Logic
            normalized_error = 1.0 + np.random.randn() * 0.1
            t_logic = self.profile_boundary_logic(1.0, baseline, armed, normalized_error)

            t_total = time.perf_counter() - t_start

            # Store stats
            stats["encoder"].samples.append(t_enc)
            stats["ar_model"].samples.append(t_ar)
            stats["mse"].samples.append(t_mse)
            stats["logic"].samples.append(t_logic)
            stats["total"].samples.append(t_total)

            # Update state
            baseline = 0.95 * baseline + 0.05 * normalized_error

            # Progress
            if (i + 1) % 1000 == 0:
                logger.info(f"  Processed {i+1}/{num_frames} frames")

        return stats


class TestLatencyProfiling:
    """Test suite for latency profiling."""

    def __init__(self):
        self.profiler = LatencyProfiler()

    def test_encoder_latency_budget(self):
        """Test that encoder latency is within 5ms budget (P99)."""
        stats = self.profiler.run_benchmark(num_frames=1000, warmup_frames=100)
        encoder_stats = stats["encoder"]

        print(encoder_stats.report())

        assert encoder_stats.within_budget, \
            f"Encoder P99 latency {encoder_stats.p99_ms:.3f}ms exceeds budget {encoder_stats.budget_ms}ms"

    def test_ar_model_latency_budget(self):
        """Test that AR model latency is within 5ms budget (P99)."""
        stats = self.profiler.run_benchmark(num_frames=1000, warmup_frames=100)
        ar_stats = stats["ar_model"]

        print(ar_stats.report())

        assert ar_stats.within_budget, \
            f"AR model P99 latency {ar_stats.p99_ms:.3f}ms exceeds budget {ar_stats.budget_ms}ms"

    def test_mse_latency_budget(self):
        """Test that MSE computation is within 1ms budget (P99)."""
        stats = self.profiler.run_benchmark(num_frames=1000, warmup_frames=100)
        mse_stats = stats["mse"]

        print(mse_stats.report())

        assert mse_stats.within_budget, \
            f"MSE P99 latency {mse_stats.p99_ms:.3f}ms exceeds budget {mse_stats.budget_ms}ms"

    def test_logic_latency_budget(self):
        """Test that boundary logic is within 1ms budget (P99)."""
        stats = self.profiler.run_benchmark(num_frames=1000, warmup_frames=100)
        logic_stats = stats["logic"]

        print(logic_stats.report())

        assert logic_stats.within_budget, \
            f"Logic P99 latency {logic_stats.p99_ms:.3f}ms exceeds budget {logic_stats.budget_ms}ms"

    def test_total_latency_budget(self):
        """Test that total end-to-end latency is within 12ms budget (P99)."""
        stats = self.profiler.run_benchmark(num_frames=10000, warmup_frames=100)
        total_stats = stats["total"]

        print("\n" + "=" * 60)
        print("LATENCY PROFILING RESULTS")
        print("=" * 60)

        for component, stat in stats.items():
            print(stat.report())
            print()

        assert total_stats.within_budget, \
            f"Total P99 latency {total_stats.p99_ms:.3f}ms exceeds budget {total_stats.budget_ms}ms"

    def test_latency_stability(self):
        """Test that latency remains stable over time (no spikes)."""
        stats = self.profiler.run_benchmark(num_frames=5000, warmup_frames=100)
        total_stats = stats["total"]

        # Check that max is not too far from p99 (indicates stability)
        max_p99_ratio = total_stats.max_ms / total_stats.p99_ms

        print(f"\nLatency stability: Max/P99 ratio = {max_p99_ratio:.2f}")

        assert max_p99_ratio < 2.0, \
            f"Latency unstable: Max ({total_stats.max_ms:.3f}ms) is {max_p99_ratio:.1f}x P99"

    def test_frame_budget_compliance(self):
        """Test that each frame completes within frame time (10ms)."""
        stats = self.profiler.run_benchmark(num_frames=1000, warmup_frames=100)
        total_stats = stats["total"]

        frame_budget_ms = self.profiler.frame_size_ms

        # P99 should be well below frame time
        assert total_stats.p99_ms < frame_budget_ms, \
            f"P99 latency {total_stats.p99_ms:.3f}ms exceeds frame time {frame_budget_ms}ms"

        # Average should be much lower (headroom for processing)
        avg_headroom = frame_budget_ms - total_stats.avg_ms
        assert avg_headroom > frame_budget_ms * 0.5, \
            f"Insufficient headroom: {avg_headroom:.3f}ms (need >50% of frame time)"


def main():
    """Run all latency tests."""
    print("=" * 60)
    print("Phase 1.1: Execution Speed Profiling")
    print("=" * 60)
    print()

    test = TestLatencyProfiling()

    # Run tests
    test.test_encoder_latency_budget()
    test.test_ar_model_latency_budget()
    test.test_mse_latency_budget()
    test.test_logic_latency_budget()
    test.test_total_latency_budget()
    test.test_latency_stability()
    test.test_frame_budget_compliance()

    print("\n" + "=" * 60)
    print("✓ ALL LATENCY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
