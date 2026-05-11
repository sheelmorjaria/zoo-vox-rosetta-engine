#!/usr/bin/env python3
"""
Phase 1.2: Memory Footprint Evaluation for Predictive NBD

Verifies that the model memory usage fits within target budgets:
- TCN: ~330KB (82K parameters)
- Mamba: ~600KB (150K parameters)

Test Protocol:
1. Track VRAM/RAM usage during inference
2. Verify models fit in L2/L3 cache
3. Check for memory leaks over 24-hour soak test

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import gc
import logging
import tracemalloc
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory statistics for a component."""
    name: str
    parameters: int
    memory_bytes: int
    target_kb: float

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)

    @property
    def memory_kb(self) -> float:
        return self.memory_bytes / 1024

    @property
    def within_target(self) -> bool:
        return self.memory_kb <= self.target_kb

    @property
    def parameter_count_str(self) -> str:
        if self.parameters >= 1e6:
            return f"{self.parameters / 1e6:.2f}M"
        elif self.parameters >= 1e3:
            return f"{self.parameters / 1e3:.0f}K"
        return str(self.parameters)

    def report(self) -> str:
        status = "✓ PASS" if self.within_target else "✗ FAIL"
        return (
            f"{self.name}:\n"
            f"  Parameters: {self.parameter_count_str}\n"
            f"  Memory: {self.memory_kb:.1f}KB ({self.memory_mb:.2f}MB)\n"
            f"  Target: {self.target_kb:.0f}KB\n"
            f"  {status}"
        )


class MemoryProfiler:
    """
    Profile memory usage of Predictive NBD components.

    Measures:
    1. Parameter count
    2. Model memory size (in bytes)
    3. Runtime memory allocation
    4. Memory leak detection
    """

    def __init__(self, sample_rate: int = 48000, frame_size_ms: float = 10.0):
        self.sample_rate = sample_rate
        self.frame_size_ms = frame_size_ms
        self.frame_size = int(sample_rate * frame_size_ms / 1000)
        self.hidden_dim = 128

        # Target memory budgets (from TDD)
        self.targets = {
            "tcn": 330.0,      # 82K parameters
            "mamba": 600.0,    # 150K parameters
            "encoder": 500.0,  # ONNX 1D Conv Encoder
            "total": 1500.0,   # Total system memory
        }

        self._setup_components()

    def _setup_components(self):
        """Setup mock or real components for profiling."""
        # Mock TCN model (~82K parameters)
        self.tcn_model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=5),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, self.hidden_dim, kernel_size=4, stride=2, padding=2),
            nn.Flatten(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )

        # Mock Mamba model (~150K parameters)
        self.mamba_model = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim),
        )

        # Mock ONNX encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=5),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(64, self.hidden_dim, kernel_size=4, stride=2, padding=2),
        )

    def count_parameters(self, model: nn.Module) -> int:
        """Count total parameters in a model."""
        return sum(p.numel() for p in model.parameters())

    def estimate_model_size(self, model: nn.Module) -> int:
        """
        Estimate model memory size in bytes.

        Assumes float32 (4 bytes per parameter).
        """
        param_count = self.count_parameters(model)
        return param_count * 4  # 4 bytes per float32

    def measure_runtime_allocation(
        self,
        model: nn.Module,
        audio: torch.Tensor,
        iterations: int = 1000,
    ) -> Dict[str, float]:
        """
        Measure runtime memory allocation during inference.

        Returns peak and baseline memory in MB.
        """
        # Force cleanup
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Start tracing
        tracemalloc.start()

        # Baseline
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(audio.unsqueeze(0).unsqueeze(0))

        # Get baseline snapshot
        baseline_snapshot = tracemalloc.take_snapshot()

        # Run iterations
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(audio.unsqueeze(0).unsqueeze(0))

        # Get peak snapshot
        peak_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Calculate stats
        baseline_mem = baseline_snapshot.compare_to(peak_snapshot, 'lineno')
        peak_stats = tracemalloc.get_traced_memory()

        return {
            "baseline_mb": peak_stats[0] / (1024 * 1024),
            "peak_mb": peak_stats[1] / (1024 * 1024),
            "delta_mb": (peak_stats[1] - peak_stats[0]) / (1024 * 1024),
        }

    def check_cache_fit(self, memory_kb: float) -> Dict[str, bool]:
        """
        Check if memory fits in typical CPU cache sizes.

        L1: ~32KB per core
        L2: ~256KB per core
        L3: ~8-32MB shared
        """
        return {
            "fits_l1": memory_kb <= 32,
            "fits_l2": memory_kb <= 256,
            "fits_l3": memory_kb <= (8 * 1024),  # 8MB
        }

    def profile_model(
        self,
        model: nn.Module,
        name: str,
        target_kb: float,
    ) -> MemoryStats:
        """Profile a single model's memory usage."""
        param_count = self.count_parameters(model)
        memory_bytes = self.estimate_model_size(model)

        return MemoryStats(
            name=name,
            parameters=param_count,
            memory_bytes=memory_bytes,
            target_kb=target_kb,
        )

    def run_memory_leak_test(
        self,
        model: nn.Module,
        iterations: int = 10000,
        threshold_mb: float = 10.0,
    ) -> Dict[str, float]:
        """
        Test for memory leaks over repeated inference.

        Simulates 24-hour operation with accelerated iterations.
        """
        tracemalloc.start()

        # Get initial memory
        initial_mem = tracemalloc.get_traced_memory()[1]

        # Run iterations
        audio = torch.randn(self.frame_size)
        model.eval()

        for i in range(iterations):
            with torch.no_grad():
                _ = model(audio.unsqueeze(0).unsqueeze(0))

            # Periodic cleanup
            if i % 100 == 0:
                gc.collect()

        # Get final memory
        final_mem = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        leak_mb = (final_mem - initial_mem) / (1024 * 1024)

        return {
            "initial_mb": initial_mem / (1024 * 1024),
            "final_mb": final_mem / (1024 * 1024),
            "leak_mb": leak_mb,
            "has_leak": leak_mb > threshold_mb,
        }


class TestMemoryProfiling:
    """Test suite for memory profiling."""

    def __init__(self):
        self.profiler = MemoryProfiler()

    def test_tcn_memory_target(self):
        """Test that TCN model is within 330KB target."""
        stats = self.profiler.profile_model(
            self.profiler.tcn_model,
            "TCN Model",
            self.profiler.targets["tcn"],
        )

        print(stats.report())

        assert stats.within_target, \
            f"TCN memory {stats.memory_kb:.1f}KB exceeds target {stats.target_kb:.0f}KB"

    def test_mamba_memory_target(self):
        """Test that Mamba model is within 600KB target."""
        stats = self.profiler.profile_model(
            self.profiler.mamba_model,
            "Mamba Model",
            self.profiler.targets["mamba"],
        )

        print(stats.report())

        assert stats.within_target, \
            f"Mamba memory {stats.memory_kb:.1f}KB exceeds target {stats.target_kb:.0f}KB"

    def test_encoder_memory_target(self):
        """Test that ONNX encoder is within 500KB target."""
        stats = self.profiler.profile_model(
            self.profiler.encoder,
            "ONNX Encoder",
            self.profiler.targets["encoder"],
        )

        print(stats.report())

        assert stats.within_target, \
            f"Encoder memory {stats.memory_kb:.1f}KB exceeds target {stats.target_kb:.0f}KB"

    def test_cache_fit_l2(self):
        """Test that models fit in L2 cache for optimal performance."""
        print("\n" + "=" * 60)
        print("L2 Cache Fit Analysis")
        print("=" * 60)

        models = [
            (self.profiler.tcn_model, "TCN", self.profiler.targets["tcn"]),
            (self.profiler.mamba_model, "Mamba", self.profiler.targets["mamba"]),
            (self.profiler.encoder, "Encoder", self.profiler.targets["encoder"]),
        ]

        all_fit_l2 = True
        for model, name, _ in models:
            stats = self.profiler.profile_model(model, name, 0)
            cache_fit = self.profiler.check_cache_fit(stats.memory_kb)

            print(f"\n{name}:")
            print(f"  Memory: {stats.memory_kb:.1f}KB")
            print(f"  Fits L1: {cache_fit['fits_l1']}")
            print(f"  Fits L2: {cache_fit['fits_l2']}")
            print(f"  Fits L3: {cache_fit['fits_l3']}")

            if not cache_fit['fits_l2']:
                all_fit_l2 = False

        assert all_fit_l2, "Not all models fit in L2 cache"

    def test_memory_leak_detection(self):
        """Test for memory leaks over extended operation."""
        print("\n" + "=" * 60)
        print("Memory Leak Test (10,000 iterations)")
        print("=" * 60)

        leak_results = self.profiler.run_memory_leak_test(
            self.profiler.tcn_model,
            iterations=10000,
            threshold_mb=5.0,
        )

        print(f"Initial memory: {leak_results['initial_mb']:.2f}MB")
        print(f"Final memory:   {leak_results['final_mb']:.2f}MB")
        print(f"Leak detected:  {leak_results['leak_mb']:.2f}MB")

        assert not leak_results['has_leak'], \
            f"Memory leak detected: {leak_results['leak_mb']:.2f}MB"

    def test_runtime_allocation_stability(self):
        """Test that runtime allocation remains stable."""
        print("\n" + "=" * 60)
        print("Runtime Allocation Stability")
        print("=" * 60)

        audio = torch.randn(self.profiler.frame_size)

        for name, model in [
            ("TCN", self.profiler.tcn_model),
            ("Mamba", self.profiler.mamba_model),
            ("Encoder", self.profiler.encoder),
        ]:
            stats = self.profiler.measure_runtime_allocation(
                model, audio, iterations=1000
            )

            print(f"\n{name}:")
            print(f"  Baseline: {stats['baseline_mb']:.2f}MB")
            print(f"  Peak:     {stats['peak_mb']:.2f}MB")
            print(f"  Delta:    {stats['delta_mb']:.2f}MB")

            # Delta should be small (stable allocation)
            assert stats['delta_mb'] < 1.0, \
                f"{name} shows unstable allocation: {stats['delta_mb']:.2f}MB"

    def test_parameter_efficiency(self):
        """Test parameter efficiency for edge deployment."""
        print("\n" + "=" * 60)
        print("Parameter Efficiency Analysis")
        print("=" * 60)

        models = [
            (self.profiler.tcn_model, "TCN", 82_000),
            (self.profiler.mamba_model, "Mamba", 150_000),
            (self.profiler.encoder, "Encoder", 100_000),
        ]

        for model, name, target_params in models:
            param_count = self.profiler.count_parameters(model)
            memory_kb = self.profiler.estimate_model_size(model) / 1024

            print(f"\n{name}:")
            print(f"  Parameters: {param_count:,} (target: {target_params:,})")
            print(f"  Memory: {memory_kb:.1f}KB")

            assert param_count <= target_params * 1.5, \
                f"{name} parameter count {param_count:,} exceeds 1.5x target"

    def test_total_system_memory(self):
        """Test total system memory budget."""
        print("\n" + "=" * 60)
        print("Total System Memory")
        print("=" * 60)

        total_bytes = sum(
            self.profiler.estimate_model_size(m)
            for m in [
                self.profiler.tcn_model,
                self.profiler.mamba_model,
                self.profiler.encoder,
            ]
        )

        total_kb = total_bytes / 1024
        target_kb = self.profiler.targets["total"]

        print(f"Total memory: {total_kb:.1f}KB ({total_kb/1024:.2f}MB)")
        print(f"Target: {target_kb:.0f}KB")

        assert total_kb <= target_kb, \
            f"Total memory {total_kb:.1f}KB exceeds target {target_kb:.0f}KB"


def main():
    """Run all memory tests."""
    print("=" * 60)
    print("Phase 1.2: Memory Footprint Evaluation")
    print("=" * 60)
    print()

    test = TestMemoryProfiling()

    # Run tests
    test.test_tcn_memory_target()
    test.test_mamba_memory_target()
    test.test_encoder_memory_target()
    test.test_cache_fit_l2()
    test.test_memory_leak_detection()
    test.test_runtime_allocation_stability()
    test.test_parameter_efficiency()
    test.test_total_system_memory()

    print("\n" + "=" * 60)
    print("✓ ALL MEMORY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
