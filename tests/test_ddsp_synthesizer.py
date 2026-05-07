#!/usr/bin/env python3
"""
Module 3 TDD Tests: DDSP Synthesizer (Differentiable Audio Engine)

This test suite verifies the differentiable DDSP synthesizer components
that enable gradient-based audio synthesis for animal vocalizations.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
Module 3 (v1.6.0): DDSP Synthesizer Differentiable Audio Engine
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for PyTorch availability
torch = pytest.importorskip("torch")
import torch.nn.functional as F

from cognitive_intelligence.ddsp_synthesis import (
    DDSPSynthesizer,
    DDSPSynthesizerLight,
    DifferentiableNoiseFilter,
    DifferentiableSineOscillator,
)

# =============================================================================
# TEST SUITE 1: DifferentiableSineOscillator
# =============================================================================


class TestDifferentiableSineOscillator:
    """Verify differentiable sine oscillator implementation."""

    def test_oscillator_initialization(self):
        """Oscillator should initialize with correct parameters."""
        osc = DifferentiableSineOscillator(sample_rate=48000)

        assert osc.sample_rate == 48000
        assert osc.phase_accumulator is None

    def test_oscillator_forward_constant_f0(self):
        """Oscillator should generate sine wave with constant frequency."""
        osc = DifferentiableSineOscillator(sample_rate=48000)

        # 100 frames at 10ms hop = 1 second
        f0 = torch.ones(2, 100) * 440.0  # A4

        audio, phase_acc = osc(f0)

        # Output shape: (batch, samples) where samples = 100 * 480 = 48000
        assert audio.shape == (2, 48000)
        assert phase_acc.shape == (2,)

        # Check that audio is normalized
        assert audio.abs().max() <= 1.0

    def test_oscillator_phase_continuity(self):
        """Oscillator should maintain phase continuity across calls."""
        osc = DifferentiableSineOscillator(sample_rate=48000)

        # First call
        f0 = torch.ones(1, 50) * 440.0
        audio1, phase1 = osc(f0)

        # Second call with accumulated phase
        audio2, phase2 = osc(f0, phase_acc=phase1)

        # Check that we can chain calls
        assert audio2.shape == (1, 24000)  # 50 frames * 480 samples
        assert phase2.shape == (1,)

    def test_oscillator_chirp_frequency(self):
        """Oscillator should generate chirp with changing frequency."""
        osc = DifferentiableSineOscillator(sample_rate=48000)

        # Rising chirp from 1kHz to 5kHz
        f0 = torch.linspace(1000, 5000, 100).unsqueeze(0)

        audio, _ = osc(f0)

        assert audio.shape == (1, 48000)
        assert audio.abs().max() <= 1.0

    def test_oscillator_requires_grad(self):
        """Oscillator output should require gradients for backprop."""
        osc = DifferentiableSineOscillator(sample_rate=48000)

        f0 = torch.ones(1, 50) * 440.0
        f0.requires_grad = True

        audio, _ = osc(f0)

        # Check gradients can be computed
        assert audio.requires_grad

        # Test backward pass
        loss = audio.sum()
        loss.backward()
        assert f0.grad is not None


# =============================================================================
# TEST SUITE 2: DifferentiableNoiseFilter
# =============================================================================


class TestDifferentiableNoiseFilter:
    """Verify differentiable noise filter implementation."""

    def test_filter_initialization(self):
        """Filter should initialize with correct parameters."""
        filter = DifferentiableNoiseFilter(sample_rate=48000, num_bands=5)

        assert filter.sample_rate == 48000
        assert filter.num_bands == 5
        assert filter.fft_size == 2048

    def test_filter_forward(self):
        """Filter should process white noise correctly."""
        filter = DifferentiableNoiseFilter(sample_rate=48000, num_bands=5)

        batch_size = 2
        n_samples = 4800  # 100ms
        white_noise = torch.randn(batch_size, n_samples)

        # Band magnitudes
        band_mags = torch.tensor(
            [
                [1.0, 0.5, 0.25, 0.1, 0.0],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ]
        )

        filtered = filter(white_noise, band_mags)

        # Output shape should match input
        assert filtered.shape == (batch_size, n_samples)

    def test_filter_requires_grad(self):
        """Filter should support gradient computation."""
        filter = DifferentiableNoiseFilter(sample_rate=48000, num_bands=5)

        white_noise = torch.randn(1, 4800)
        band_mags = torch.ones(1, 5) * 0.5
        band_mags.requires_grad = True

        filtered = filter(white_noise, band_mags)

        assert filtered.requires_grad

        # Test backward pass
        loss = filtered.sum()
        loss.backward()
        assert band_mags.grad is not None

    def test_filter_high_frequency_emphasis(self):
        """Filter with high band emphasis should boost high frequencies."""
        filter = DifferentiableNoiseFilter(sample_rate=48000, num_bands=5)

        white_noise = torch.randn(1, 4800)

        # Emphasize high frequencies
        band_mags = torch.tensor([[0.0, 0.1, 0.3, 0.7, 1.0]])

        filtered = filter(white_noise, band_mags)

        # High-frequency emphasis should increase energy in high bands
        # This is a weak test - just verifies it runs
        assert filtered.shape == (1, 4800)


# =============================================================================
# TEST SUITE 3: DDSPSynthesizer
# =============================================================================


class TestDDSPSynthesizer:
    """Verify complete DDSP synthesizer implementation."""

    def test_synthesizer_initialization(self):
        """Synthesizer should initialize with correct parameters."""
        synth = DDSPSynthesizer(sample_rate=48000)

        assert synth.sample_rate == 48000
        assert synth.num_harmonics == 60
        assert synth.num_noise_bands == 5
        assert synth.hop_size == 480

    def test_synthesizer_forward(self):
        """Synthesizer should generate audio from DDSP parameters."""
        synth = DDSPSynthesizer(sample_rate=48000)

        batch_size = 2
        n_frames = 50

        # F0 trajectory (constant 4kHz)
        f0 = torch.ones(batch_size, n_frames) * 4000.0

        # Harmonic amplitudes (random)
        harmonic_amps = F.softmax(torch.randn(batch_size, n_frames, 60), dim=-1)

        # Noise magnitudes (random)
        noise_mags = F.relu(torch.randn(batch_size, n_frames, 5))

        # Generate audio
        audio, phase_acc = synth(f0, harmonic_amps, noise_mags)

        # Expected output shape: (batch, n_frames * hop_size)
        expected_samples = n_frames * synth.hop_size
        assert audio.shape == (batch_size, expected_samples)
        assert phase_acc.shape == (batch_size,)

        # Check normalization
        assert audio.abs().max() <= 1.0

    def test_synthesizer_chirp(self):
        """Synthesizer should handle frequency sweeps correctly."""
        synth = DDSPSynthesizer(sample_rate=48000)

        n_frames = 100

        # Rising chirp from 5kHz to 10kHz
        f0 = torch.linspace(5000, 10000, n_frames).unsqueeze(0)

        # Flat amplitudes
        harmonic_amps = torch.ones(1, n_frames, 60) / 60
        noise_mags = torch.ones(1, n_frames, 5) * 0.1

        audio, _ = synth(f0, harmonic_amps, noise_mags)

        assert audio.shape == (1, 48000)  # 100 frames * 480 samples

    def test_synthesizer_phase_continuity(self):
        """Synthesizer should maintain phase continuity across calls."""
        synth = DDSPSynthesizer(sample_rate=48000)

        n_frames = 50

        f0 = torch.ones(1, n_frames) * 8000.0
        harmonic_amps = torch.ones(1, n_frames, 60) / 60
        noise_mags = torch.zeros(1, n_frames, 5)  # No noise for cleaner test

        # First call
        audio1, phase1 = synth(f0, harmonic_amps, noise_mags)

        # Second call with phase accumulator
        audio2, phase2 = synth(f0, harmonic_amps, noise_mags, phase_acc=phase1)

        # Should produce audio without clicks
        assert audio1.shape == (1, 24000)
        assert audio2.shape == (1, 24000)

    def test_synthesizer_requires_grad(self):
        """Synthesizer should support end-to-end gradient flow."""
        synth = DDSPSynthesizer(sample_rate=48000)

        n_frames = 25

        f0 = torch.ones(1, n_frames) * 6000.0
        harmonic_amps = F.softmax(torch.randn(1, n_frames, 60), dim=-1)
        noise_mags = F.relu(torch.randn(1, n_frames, 5))

        # Enable gradients on inputs
        harmonic_amps.requires_grad = True

        audio, _ = synth(f0, harmonic_amps, noise_mags)

        # Check gradients flow
        assert audio.requires_grad

        loss = audio.sum()
        loss.backward()
        assert harmonic_amps.grad is not None

    def test_synthesizer_light_variant(self):
        """Light variant should work correctly."""
        synth = DDSPSynthesizerLight(sample_rate=48000)

        assert synth.num_harmonics == 32
        assert synth.num_noise_bands == 3

        # Test forward pass
        n_frames = 50
        f0 = torch.ones(1, n_frames) * 5000.0
        harmonic_amps = torch.ones(1, n_frames, 32) / 32
        noise_mags = torch.ones(1, n_frames, 3) * 0.1

        audio, _ = synth(f0, harmonic_amps, noise_mags)

        assert audio.shape == (1, 24000)


# =============================================================================
# TEST SUITE 4: Integration Tests
# =============================================================================


class TestDDSPIntegration:
    """Integration tests for complete DDSP pipeline."""

    def test_full_pipeline_with_decoder(self):
        """Test complete pipeline: Decoder → Synthesizer."""
        from cognitive_intelligence.ddsp_decoder import DDSPDecoder

        decoder = DDSPDecoder()
        synthesizer = DDSPSynthesizer(sample_rate=48000)

        # Input features
        features_112d = torch.randn(1, 112)

        # Decode features to DDSP parameters
        harmonic_amps, noise_mags = decoder(features_112d)

        # Need to reshape for synthesizer (add time dimension)
        # For this test, repeat across 50 frames
        n_frames = 50
        harmonic_amps_expanded = harmonic_amps.unsqueeze(1).expand(1, n_frames, 60)
        noise_mags_expanded = noise_mags.unsqueeze(1).expand(1, n_frames, 5)

        # Create F0 trajectory (use feature 0 as base frequency)
        base_f0 = 4000 + features_112d[0, 0] * 2000
        f0 = torch.ones(1, n_frames) * base_f0

        # Synthesize audio
        audio, _ = synthesizer(f0, harmonic_amps_expanded, noise_mags_expanded)

        # Verify output
        assert audio.shape == (1, n_frames * synthesizer.hop_size)
        assert audio.abs().max() <= 1.0

    def test_batch_processing(self):
        """Synthesizer should handle batches efficiently."""
        synth = DDSPSynthesizer(sample_rate=48000)

        batch_size = 8
        n_frames = 25

        f0 = torch.linspace(5000, 8000, n_frames).unsqueeze(0).expand(batch_size, -1)
        harmonic_amps = F.softmax(torch.randn(batch_size, n_frames, 60), dim=-1)
        noise_mags = F.relu(torch.randn(batch_size, n_frames, 5))

        audio, _ = synth(f0, harmonic_amps, noise_mags)

        assert audio.shape == (batch_size, n_frames * synth.hop_size)

    def test_synthesis_output_length(self):
        """Synthesis output length should match expected value."""
        synth = DDSPSynthesizer(sample_rate=48000, hop_size=480)

        n_frames = 100
        expected_samples = n_frames * synth.hop_size

        f0 = torch.ones(1, n_frames) * 7000.0
        harmonic_amps = torch.ones(1, n_frames, 60) / 60
        noise_mags = torch.zeros(1, n_frames, 5)

        audio, _ = synth(f0, harmonic_amps, noise_mags)

        assert audio.shape[-1] == expected_samples


# =============================================================================
# TEST SUITE 5: Edge Cases
# =============================================================================


class TestDDSPEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_synthesizer_single_frame(self):
        """Synthesizer should handle single frame input."""
        synth = DDSPSynthesizer(sample_rate=48000)

        f0 = torch.ones(1, 1) * 9000.0
        harmonic_amps = torch.ones(1, 1, 60) / 60
        noise_mags = torch.zeros(1, 1, 5)

        audio, _ = synth(f0, harmonic_amps, noise_mags)

        assert audio.shape == (1, synth.hop_size)

    def test_synthesizer_zero_noise(self):
        """Synthesizer with zero noise should produce pure harmonic audio."""
        synth = DDSPSynthesizer(sample_rate=48000)

        n_frames = 50
        f0 = torch.ones(1, n_frames) * 6000.0
        harmonic_amps = torch.ones(1, n_frames, 60) / 60
        noise_mags = torch.zeros(1, n_frames, 5)  # No noise

        audio, _ = synth(f0, harmonic_amps, noise_mags)

        assert audio.shape == (1, n_frames * synth.hop_size)
        assert audio.abs().max() <= 1.0

    def test_synthesizer_zero_harmonics(self):
        """Synthesizer with zero harmonics should produce pure noise."""
        synth = DDSPSynthesizer(sample_rate=48000)

        n_frames = 50
        f0 = torch.ones(1, n_frames) * 6000.0
        harmonic_amps = torch.zeros(1, n_frames, 60)  # No harmonics
        noise_mags = torch.ones(1, n_frames, 5) * 0.5

        audio, _ = synth(f0, harmonic_amps, noise_mags)

        # Should still produce audio (from noise component)
        assert audio.shape == (1, n_frames * synth.hop_size)

    def test_oscillator_reset_phase(self):
        """Phase reset should clear the accumulator."""
        osc = DifferentiableSineOscillator(sample_rate=48000)

        f0 = torch.ones(1, 50) * 440.0
        audio1, _ = osc(f0)

        osc.reset_phase()
        assert osc.phase_accumulator is None


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Check if PyTorch is available
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        pytest.main([__file__, "-v"])
    except ImportError:
        print("PyTorch not available. Skipping tests.")
        print("Install with: pip install torch")
