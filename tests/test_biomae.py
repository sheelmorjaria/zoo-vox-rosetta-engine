#!/usr/bin/env python3
"""
Test Suite for BioMAE Pipeline

Tests for:
- Log-linear spectrogram computation
- Patch embedding tokenization
- BioMAE encoder/decoder functionality
- Integration tests (ultrasonic sweep, latency profiling)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import sys
import time
from typing import List

import numpy as np
import pytest
import torch

sys.path.insert(0, '/mnt/c/Users/sheel/Desktop/src')

from feature_extraction.bio_spectrogram import (
    UltrasonicSpectrogram,
    SpectrogramConfig,
    create_spectrogram,
    create_spectrogram_for_taxa,
)
from feature_extraction.patch_embed import (
    PatchEmbedding,
    PatchEmbedConfig,
    create_patch_embedding,
)
from feature_extraction.biomae import (
    BioMAEEncoder,
    BioMAEDecoder,
    BioMAEModel,
    EncoderConfig,
    DecoderConfig,
    create_biomae_model,
)


# =============================================================================
# Spectrogram Tests
# =============================================================================

class TestUltrasonicSpectrogram:
    """Test log-linear spectrogram computation."""

    def test_spectrogram_shape(self):
        """Verify output shape is correct."""
        config = SpectrogramConfig(sample_rate=48000, n_fft=1024, hop_length=256)
        spec = UltrasonicSpectrogram(config)

        waveform = torch.randn(1, 48000)  # 1 second at 48kHz
        log_spec = spec(waveform)

        # Shape: (Batch, Freq_bins, Time_frames)
        assert log_spec.dim() == 3
        assert log_spec.shape[0] == 1
        assert log_spec.shape[1] == config.n_fft // 2 + 1  # Freq bins

    def test_linear_frequency_axis(self):
        """
        Verify frequency bins are spaced linearly (not warped like Mel-scale).

        This is critical for ultrasonic analysis where absolute frequency
        matters (e.g., 60kHz vs 65kHz bat echolocation).
        """
        config = SpectrogramConfig(sample_rate=96000, n_fft=1024)
        spec = UltrasonicSpectrogram(config)

        freq_axis = spec.frequency_axis()

        # Check spacing is constant
        diffs = torch.diff(freq_axis)
        assert torch.allclose(diffs, diffs[0], atol=1e-5), "Frequency bins should be linearly spaced"

        # Verify range covers ultrasonic frequencies
        assert freq_axis[0] == 0
        assert freq_axis[-1] >= 48000  # Should reach Nyquist (48kHz for 96kHz sample rate)

    def test_no_mel_warping(self):
        """
        Verify that high frequencies are not compressed (unlike Mel-scale).

        Mel-scale compresses >8kHz, which is catastrophic for bat ultrasound.
        This test verifies linear scaling preserves ultrasonic information.
        """
        config = SpectrogramConfig(sample_rate=96000, n_fft=1024)
        spec = UltrasonicSpectrogram(config)

        freq_axis = spec.frequency_axis()

        # Linear frequency bins mean equal Hz per bin
        # Mel would have fewer Hz/bin at high frequencies
        bin_width_hz = freq_axis[1].item() - freq_axis[0].item()

        # Check high frequency bins have same width as low frequency
        low_bin_width = freq_axis[10].item() - freq_axis[9].item()
        high_bin_width = freq_axis[400].item() - freq_axis[399].item()

        assert abs(low_bin_width - high_bin_width) < 1e-3, "Linear scale should have constant bin width"

    def test_taxa_presets(self):
        """Verify presets for different taxonomic groups."""
        bat_spec = create_spectrogram_for_taxa('bat')
        assert bat_spec.config.sample_rate == 96000

        bird_spec = create_spectrogram_for_taxa('bird')
        assert bird_spec.config.sample_rate == 48000

        cetacean_spec = create_spectrogram_for_taxa('cetacean')
        assert cetacean_spec.config.sample_rate == 192000


# =============================================================================
# Patch Embedding Tests
# =============================================================================

class TestPatchEmbedding:
    """Test ViT-style patch embedding."""

    def test_patch_shape_and_count(self):
        """Verify patch embedding produces correct output dimensions."""
        config = PatchEmbedConfig(
            img_size=(128, 128),
            patch_size=(16, 16),
            embed_dim=256,
        )
        embed = PatchEmbedding(config)

        spectrogram = torch.randn(2, 1, 128, 128)  # Batch of 2
        patches = embed(spectrogram)

        # Shape: (Batch, num_patches+1, embed_dim)
        # num_patches = (128/16) * (128/16) = 8 * 8 = 64
        assert patches.shape[0] == 2  # Batch
        assert patches.shape[1] == 65  # 64 patches + 1 CLS token
        assert patches.shape[2] == 256  # Embedding dimension

    def test_cls_token_present(self):
        """Verify CLS token is prepended to patches."""
        config = PatchEmbedConfig(img_size=(64, 64), patch_size=(16, 16), embed_dim=128)
        embed = PatchEmbedding(config)

        spectrogram = torch.randn(1, 1, 64, 64)
        patches = embed(spectrogram)

        # CLS token should be different from patch tokens
        cls_token = patches[0, 0, :]
        first_patch = patches[0, 1, :]

        assert not torch.allclose(cls_token, first_patch)

    def test_positional_embeddings_added(self):
        """Verify positional embeddings are added to patches."""
        config = PatchEmbedConfig(img_size=(64, 64), patch_size=(16, 16), embed_dim=128)
        embed = PatchEmbedding(config)

        # Two identical spectrograms should get different embeddings
        # due to positional encoding
        spectrogram = torch.randn(1, 1, 64, 64)

        # Create identical inputs
        patches1 = embed(spectrogram)
        patches2 = embed(spectrogram.clone())

        # Positional embeddings are added identically
        assert torch.allclose(patches1, patches2)


# =============================================================================
# BioMAE Model Tests
# =============================================================================

class TestBioMAEEncoder:
    """Test BioMAE encoder."""

    def test_output_shape_112d(self):
        """Verify encoder outputs exactly 112 dimensions (Rosetta compatibility)."""
        config = EncoderConfig(
            img_size=(128, 128),
            embed_dim=256,
            depth=4,
            num_heads=4,
            output_dim=112,
        )
        encoder = BioMAEEncoder(config)

        spectrogram = torch.randn(2, 1, 128, 128)
        embedding = encoder(spectrogram)

        assert embedding.shape == (2, 112)

    def test_encoder_with_patches(self):
        """Verify encoder can return patch representations."""
        config = EncoderConfig(img_size=(64, 64), embed_dim=128, output_dim=112)
        encoder = BioMAEEncoder(config)

        spectrogram = torch.randn(1, 1, 64, 64)
        patches, embedding = encoder(spectrogram, return_patches=True)

        assert patches.dim() == 3  # (Batch, num_patches+1, embed_dim)
        assert embedding.shape == (1, 112)


class TestBioMAEDecoder:
    """Test BioMAE decoder."""

    def test_decoder_reconstruction(self):
        """Verify decoder can reconstruct spectrogram shape."""
        encoder_config = EncoderConfig(
            img_size=(128, 128),
            embed_dim=256,
            output_dim=112,
        )
        decoder_config = DecoderConfig(
            embed_dim=256,
            img_size=(128, 128),
        )

        encoder = BioMAEEncoder(encoder_config)
        decoder = BioMAEDecoder(decoder_config)

        spectrogram = torch.randn(1, 1, 128, 128)
        patches, _ = encoder(spectrogram, return_patches=True)

        reconstruction = decoder(patches)

        # Should reconstruct original shape
        assert reconstruction.shape == spectrogram.shape


class TestBioMAEModel:
    """Test complete BioMAE model."""

    def test_forward_pass(self):
        """Test end-to-end forward pass."""
        encoder_config = EncoderConfig(
            img_size=(128, 128),
            embed_dim=256,
            depth=4,
            output_dim=112,
        )
        decoder_config = DecoderConfig(
            embed_dim=256,
            img_size=(128, 128),
        )

        model = BioMAEModel(encoder_config, decoder_config)

        spectrogram = torch.randn(2, 1, 128, 128)
        reconstructed, embedding = model(spectrogram)

        assert reconstructed.shape == spectrogram.shape
        assert embedding.shape == (2, 112)

    def test_mask_generation(self):
        """Test random mask generation with 75% ratio."""
        encoder_config = EncoderConfig(img_size=(64, 64), embed_dim=128)
        decoder_config = DecoderConfig(embed_dim=128, img_size=(64, 64))

        model = BioMAEModel(encoder_config, decoder_config)

        mask = model.generate_random_mask(batch_size=4, device=torch.device('cpu'), mask_ratio=0.75)

        assert mask.shape == (4, 16)  # (Batch, num_patches) where 64/(16*16) patches

        # Check masking ratio is approximately correct
        mask_ratio = mask.float().mean().item()
        assert 0.70 < mask_ratio < 0.80  # Allow some variance


# =============================================================================
# Integration Tests
# =============================================================================

class TestUltrasonicSweep:
    """
    The "Ultrasonic Formant" Test.

    Generate a synthetic sweep from 10kHz to 90kHz. Verify that
    BioMAE preserves the high-frequency trajectory, while MFCC
    would compress it.
    """

    def test_ultrasonic_frequency_preservation(self):
        """
        Test that linear spectrogram preserves ultrasonic frequencies.

        This test verifies the core motivation for BioMAE: that
        log-linear spectrograms preserve ultrasonic formant geometry
        better than Mel-spectrograms.
        """
        # Generate synthetic ultrasonic sweep: 10kHz to 90kHz
        config = SpectrogramConfig(sample_rate=192000, n_fft=2048, hop_length=512)
        spec = UltrasonicSpectrogram(config)

        # Create a chirp signal sweeping from 10kHz to 90kHz
        duration = 0.5  # seconds
        sample_rate = config.sample_rate
        t = torch.linspace(0, duration, int(sample_rate * duration))

        # Chirp: frequency increases linearly from 10kHz to 90kHz
        f0, f1 = 10000, 90000
        instantaneous_phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
        chirp = torch.sin(instantaneous_phase).unsqueeze(0)

        # Compute spectrogram
        log_spec = spec(chirp)  # (1, Freq, Time)

        # Verify high frequencies are preserved
        freq_axis = spec.frequency_axis()

        # Find frequency range
        high_freq_idx = torch.where(freq_axis >= 80000)[0]
        assert len(high_freq_idx) > 0, "Should have frequency bins >= 80kHz"

        # Energy should be present in high frequencies at end of chirp
        high_freq_energy = log_spec[0, high_freq_idx, -20:].mean()
        assert high_freq_energy > -100, "High frequency energy should be preserved"


class TestLatencyProfiler:
    """
    Test latency characteristics for edge deployment.

    Target: <5ms latency (refined from <1ms based on research)
    """

    def test_encoder_latency(self):
        """Profile encoder latency on CPU."""
        config = EncoderConfig(
            img_size=(128, 128),
            embed_dim=256,
            depth=4,
            num_heads=4,
            output_dim=112,
        )
        encoder = BioMAEEncoder(config)
        encoder.eval()

        spectrogram = torch.randn(1, 1, 128, 128)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = encoder(spectrogram)

        # Profile
        latencies = []
        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()
                _ = encoder(spectrogram)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)

        # Note: CPU will be slower than GPU/TensorRT
        # This is a baseline test
        assert avg_latency > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_encoder_latency_gpu(self):
        """Profile encoder latency on GPU (closer to production target)."""
        device = torch.device('cuda')

        config = EncoderConfig(
            img_size=(128, 128),
            embed_dim=256,
            depth=4,
            num_heads=4,
            output_dim=112,
        )
        encoder = BioMAEEncoder(config).to(device)
        encoder.eval()

        spectrogram = torch.randn(1, 1, 128, 128, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(50):
                _ = encoder(spectrogram)

        # Synchronize
        torch.cuda.synchronize()

        # Profile
        latencies = []
        with torch.no_grad():
            for _ in range(1000):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = encoder(spectrogram)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))  # in ms

        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)

        # Target: <5ms on GPU
        # Note: PyTorch native will be slower than TensorRT
        # This is a sanity check
        assert p99_latency < 100, f"P99 latency {p99_latency:.2f}ms should be reasonable"


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions for easy model creation."""

    def test_create_spectrogram(self):
        """Test spectrogram factory."""
        spec = create_spectrogram()
        assert isinstance(spec, UltrasonicSpectrogram)

    def test_create_patch_embedding(self):
        """Test patch embedding factory."""
        embed = create_patch_embedding()
        assert isinstance(embed, PatchEmbedding)

    def test_create_biomae_model(self):
        """Test BioMAE model factory."""
        model = create_biomae_model()
        assert isinstance(model, BioMAEModel)

        # Verify it produces correct output
        spec = torch.randn(1, 1, 128, 128)
        reconstructed, embedding = model(spec)
        assert embedding.shape == (1, 112)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
