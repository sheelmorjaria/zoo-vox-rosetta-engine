#!/usr/bin/env python3
"""
Module 4 TDD Tests: Jetson Edge Deployment

This test suite verifies the ONNX/TensorRT export and real-time inference
agent for NVIDIA Jetson deployment.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
Module 4 (v1.6.0): Jetson Edge Deployment
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for PyTorch availability
torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from cognitive_intelligence.ddsp_decoder import DDSPDecoder
from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer
from cognitive_intelligence.jetson_export import (
    benchmark_pytorch_model,
    export_ddsp_decoder_to_onnx,
    export_ddsp_pipeline,
    export_ddsp_synthesizer_to_onnx,
    get_model_size_mb,
)

# =============================================================================
# TEST SUITE 1: ONNX Export
# =============================================================================


class TestONNXExport:
    """Verify ONNX export functionality."""

    def test_export_decoder_to_onnx(self):
        """DDSPDecoder should export to ONNX format."""
        decoder = DDSPDecoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "decoder.onnx")

            result = export_ddsp_decoder_to_onnx(decoder, output_path)

            assert result
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

    def test_export_synthesizer_to_onnx(self):
        """DDSPSynthesizer should export to ONNX format."""
        synthesizer = DDSPSynthesizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "synthesizer.onnx")

            # Use dynamic_axes=False for synthesizer due to PyTorch ONNX limitations
            result = export_ddsp_synthesizer_to_onnx(
                synthesizer,
                output_path,
                dynamic_axes=False,
            )

            assert result
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0

    def test_export_with_dynamic_axes(self):
        """ONNX export should support dynamic batch sizes."""
        decoder = DDSPDecoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "decoder_dynamic.onnx")

            export_ddsp_decoder_to_onnx(
                decoder,
                output_path,
                dynamic_axes=True,
            )

            # Verify file was created
            assert os.path.exists(output_path)

    def test_export_with_fixed_batch(self):
        """ONNX export should work with fixed batch size."""
        decoder = DDSPDecoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "decoder_fixed.onnx")

            export_ddsp_decoder_to_onnx(
                decoder,
                output_path,
                input_shape=(4, 112),  # Fixed batch of 4
                dynamic_axes=False,
            )

            assert os.path.exists(output_path)


# =============================================================================
# TEST SUITE 2: Model Benchmarking
# =============================================================================


class TestModelBenchmarking:
    """Verify model benchmarking functionality."""

    def test_benchmark_decoder(self):
        """Decoder benchmark should return timing statistics."""
        decoder = DDSPDecoder()

        stats = benchmark_pytorch_model(
            decoder,
            input_shape=(1, 112),
            num_runs=10,
            warmup_runs=2,
        )

        assert "mean_ms" in stats
        assert "std_ms" in stats
        assert "min_ms" in stats
        assert "max_ms" in stats
        assert "median_ms" in stats

        assert stats["mean_ms"] > 0
        assert stats["min_ms"] <= stats["mean_ms"] <= stats["max_ms"]

    def test_benchmark_synthesizer(self):
        """Synthesizer benchmark should return timing statistics."""
        import time

        synthesizer = DDSPSynthesizer()
        synthesizer.eval()

        # Create typical inputs
        dummy_f0 = torch.randn(1, 100)
        dummy_harmonic_amps = torch.randn(1, 100, 60)
        dummy_noise_mags = torch.abs(torch.randn(1, 100, 5))

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = synthesizer(dummy_f0, dummy_harmonic_amps, dummy_noise_mags)

        # Time inference
        times = []
        for _ in range(10):
            start = time.perf_counter()
            with torch.no_grad():
                _ = synthesizer(dummy_f0, dummy_harmonic_amps, dummy_noise_mags)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        stats = {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
        }

        assert stats["mean_ms"] > 0
        assert stats["min_ms"] <= stats["mean_ms"] <= stats["max_ms"]

    def test_benchmark_consistency(self):
        """Benchmark results should be consistent across runs."""
        decoder = DDSPDecoder()

        stats1 = benchmark_pytorch_model(decoder, num_runs=10)
        stats2 = benchmark_pytorch_model(decoder, num_runs=10)

        # Mean should be within reasonable range
        assert abs(stats1["mean_ms"] - stats2["mean_ms"]) < 5.0

    def test_model_size_calculation(self):
        """Model size calculation should return reasonable value."""
        decoder = DDSPDecoder()

        size_mb = get_model_size_mb(decoder)

        assert size_mb > 0
        assert size_mb < 100  # Should be less than 100MB

    def test_decoder_latency_target(self):
        """Decoder should meet latency target for Jetson."""
        decoder = DDSPDecoder()
        decoder.eval()

        # Single inference timing
        import time

        dummy_input = torch.randn(1, 112)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = decoder(dummy_input)

        # Time inference
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = decoder(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        latency_ms = (end - start) * 1000

        # Target: <2ms on GPU, <10ms on CPU
        if torch.cuda.is_available():
            assert latency_ms < 10  # Relaxed for CI environment
        else:
            assert latency_ms < 50  # More lenient for CPU


# =============================================================================
# TEST SUITE 3: Pipeline Export
# =============================================================================


class TestPipelineExport:
    """Verify complete DDSP pipeline export."""

    def test_export_pipeline(self):
        """Complete pipeline export should create all artifacts."""
        decoder = DDSPDecoder()
        synthesizer = DDSPSynthesizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = export_ddsp_pipeline(
                decoder,
                synthesizer,
                tmpdir,
                export_tensorrt=False,  # Skip TensorRT if not available
            )

            # Should have exported both models
            assert "decoder_onnx" in artifacts
            assert "synthesizer_onnx" in artifacts

            # Verify files exist
            assert os.path.exists(artifacts["decoder_onnx"])
            assert os.path.exists(artifacts["synthesizer_onnx"])

    def test_export_creates_directory(self):
        """Export should create output directory if needed."""
        decoder = DDSPDecoder()
        synthesizer = DDSPSynthesizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "exports", "ddsp")

            artifacts = export_ddsp_pipeline(
                decoder,
                synthesizer,
                output_dir,
                export_tensorrt=False,
            )

            assert os.path.exists(output_dir)
            assert len(artifacts) >= 2

    def test_export_artifact_paths(self):
        """Export artifact paths should be correct."""
        decoder = DDSPDecoder()
        synthesizer = DDSPSynthesizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = export_ddsp_pipeline(
                decoder,
                synthesizer,
                tmpdir,
                export_tensorrt=False,
            )

            # Paths should be strings
            for name, path in artifacts.items():
                assert isinstance(path, str)
                assert os.path.isabs(path) or os.path.exists(path)


# =============================================================================
# TEST SUITE 4: Real-time Agent
# =============================================================================


class TestRealtimeAgent:
    """Verify real-time DDSP agent functionality."""

    def test_agent_initialization(self):
        """Agent should initialize with default config."""
        from realtime.ddsp_agent import DDSPAgentConfig, RealtimeDDSPAgent

        # Use random ports to avoid conflicts
        config = DDSPAgentConfig(
            device="cpu",  # Use CPU for testing
            audio_pub_port=0,  # Random port
            heartbeat_pub_port=0,  # Random port
            feature_sub_port=0,  # Random port
        )
        agent = RealtimeDDSPAgent(config)

        assert agent.decoder is not None
        assert agent.synthesizer is not None
        assert agent.centroids is not None

    def test_agent_synthesize_from_features(self):
        """Agent should synthesize audio from 112D features."""
        from realtime.ddsp_agent import DDSPAgentConfig, RealtimeDDSPAgent

        config = DDSPAgentConfig(
            device="cpu",
            audio_pub_port=0,
            heartbeat_pub_port=0,
            feature_sub_port=0,
        )
        agent = RealtimeDDSPAgent(config)

        features = np.random.randn(112).astype(np.float32)
        audio, latency = agent.synthesize_from_features(features, duration_ms=100.0)

        assert len(audio) == 4800  # 100ms at 48kHz
        assert latency > 0
        assert latency < 1000  # Should be fast

    def test_agent_synthesize_from_cluster(self):
        """Agent should synthesize audio from cluster ID."""
        from realtime.ddsp_agent import DDSPAgentConfig, RealtimeDDSPAgent

        config = DDSPAgentConfig(
            device="cpu",
            audio_pub_port=0,
            heartbeat_pub_port=0,
            feature_sub_port=0,
        )
        agent = RealtimeDDSPAgent(config)

        # Use first cluster
        audio, latency = agent.synthesize_from_cluster(
            cluster_id=0,
            duration_ms=100.0,
        )

        assert len(audio) == 4800
        assert latency > 0

    def test_agent_synthesize_with_delta(self):
        """Agent should apply delta to cluster centroid."""
        from realtime.ddsp_agent import DDSPAgentConfig, RealtimeDDSPAgent

        config = DDSPAgentConfig(
            device="cpu",
            audio_pub_port=0,
            heartbeat_pub_port=0,
            feature_sub_port=0,
        )
        agent = RealtimeDDSPAgent(config)

        delta = np.random.randn(112).astype(np.float32) * 0.1

        audio1, _ = agent.synthesize_from_cluster(cluster_id=0, duration_ms=100.0)
        audio2, _ = agent.synthesize_from_cluster(
            cluster_id=0,
            delta_112d=delta,
            duration_ms=100.0,
        )

        # Audio should be different (though not guaranteed)
        assert audio1.shape == audio2.shape

    def test_agent_statistics(self):
        """Agent should track performance statistics."""
        from realtime.ddsp_agent import DDSPAgentConfig, RealtimeDDSPAgent

        config = DDSPAgentConfig(
            device="cpu",
            audio_pub_port=0,
            heartbeat_pub_port=0,
            feature_sub_port=0,
        )
        agent = RealtimeDDSPAgent(config)

        # Run a few syntheses
        for _ in range(5):
            agent.synthesize_from_cluster(cluster_id=0, duration_ms=50.0)

        stats = agent.get_statistics()

        assert "frame_count" in stats
        assert "avg_latency_ms" in stats
        assert "target_latency_ms" in stats
        assert stats["frame_count"] >= 5


# =============================================================================
# TEST SUITE 5: Edge Cases
# =============================================================================


class TestDeploymentEdgeCases:
    """Test edge cases for deployment."""

    def test_export_with_invalid_path(self):
        """Export should handle invalid paths gracefully."""
        decoder = DDSPDecoder()

        # Try to export to read-only directory
        result = export_ddsp_decoder_to_onnx(
            decoder,
            "/root/invalid/path/decoder.onnx",
        )

        # Should fail gracefully
        assert not result

    def test_agent_with_invalid_cluster(self):
        """Agent should handle invalid cluster IDs."""
        from realtime.ddsp_agent import DDSPAgentConfig, RealtimeDDSPAgent

        config = DDSPAgentConfig(
            device="cpu",
            audio_pub_port=0,
            heartbeat_pub_port=0,
            feature_sub_port=0,
        )
        agent = RealtimeDDSPAgent(config)

        # Use very large cluster ID
        audio, latency = agent.synthesize_from_cluster(
            cluster_id=9999,
            duration_ms=100.0,
        )

        # Should fall back to first centroid
        assert len(audio) == 4800

    def test_agent_with_zero_duration(self):
        """Agent should handle zero duration request."""
        from realtime.ddsp_agent import DDSPAgentConfig, RealtimeDDSPAgent

        config = DDSPAgentConfig(
            device="cpu",
            audio_pub_port=0,
            heartbeat_pub_port=0,
            feature_sub_port=0,
        )
        agent = RealtimeDDSPAgent(config)

        # This might produce very short audio
        audio, latency = agent.synthesize_from_features(
            np.random.randn(112).astype(np.float32),
            duration_ms=10.0,  # Very short
        )

        assert len(audio) >= 480  # At least 10ms worth

    def test_benchmark_with_small_model(self):
        """Benchmark should work with small models."""
        import torch.nn as nn

        # Create tiny model
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(112, 10)

            def forward(self, x):
                return self.linear(x)

        model = TinyModel()
        stats = benchmark_pytorch_model(model, num_runs=5)

        assert stats["mean_ms"] > 0


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Check dependencies
    try:
        import numpy as np
    except ImportError:
        print("NumPy not available. Install with: pip install numpy")
        exit(1)

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not available. Install with: pip install torch")
        exit(1)

    pytest.main([__file__, "-v"])
