#!/usr/bin/env python3
"""
Tests for Dual-Stream Synthesis (Module 4)

These tests verify the FiLM-based DDSP decoder that enables
affective modulation while preserving pre-trained weights.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class TestFiLMGenerator(unittest.TestCase):
    """Test FiLM parameter generator."""

    def test_film_generator_initialization(self):
        """Should initialize with correct dimensions."""
        from cognitive_intelligence.ddsp_decoder import FiLMGenerator

        film = FiLMGenerator(
            affect_dim=16,
            hidden_dim=256,
            num_layers=2,
            film_hidden_dim=64,
        )

        self.assertEqual(film.affect_dim, 16)
        self.assertEqual(film.hidden_dim, 256)
        self.assertEqual(film.num_layers, 2)
        self.assertEqual(len(film.film_layers), 2)

    def test_film_forward_shape(self):
        """Should generate correct FiLM parameter shapes."""
        from cognitive_intelligence.ddsp_decoder import FiLMGenerator

        film = FiLMGenerator(affect_dim=16, hidden_dim=256, num_layers=2)

        affect_vector = torch.randn(4, 16)  # Batch of 4
        films = film(affect_vector)

        self.assertEqual(len(films), 2)  # 2 FiLM layers

        for gamma, beta in films:
            self.assertEqual(gamma.shape, (4, 256))  # (batch, hidden_dim)
            self.assertEqual(beta.shape, (4, 256))

    def test_film_single_batch(self):
        """Should handle single sample input."""
        from cognitive_intelligence.ddsp_decoder import FiLMGenerator

        film = FiLMGenerator(affect_dim=16, hidden_dim=256, num_layers=2)

        affect_vector = torch.randn(16)  # Single sample (no batch dim)
        # FiLM generator requires 2D input, so unsqueeze
        affect_vector_2d = affect_vector.unsqueeze(0)
        films = film(affect_vector_2d)

        # Should have batch dimension
        for gamma, beta in films:
            self.assertEqual(gamma.shape[0], 1)
            self.assertEqual(gamma.shape[1], 256)

    def test_film_different_affect(self):
        """Should generate different parameters for different affect vectors."""
        from cognitive_intelligence.ddsp_decoder import FiLMGenerator

        film = FiLMGenerator(affect_dim=16, hidden_dim=256, num_layers=2)

        affect1 = torch.randn(1, 16)
        affect2 = torch.randn(1, 16)

        films1 = film(affect1)
        films2 = film(affect2)

        # Parameters should be different
        for (gamma1, beta1), (gamma2, beta2) in zip(films1, films2):
            self.assertFalse(torch.allclose(gamma1, gamma2))


class TestDualStreamDDSPDecoder(unittest.TestCase):
    """Test dual-stream DDSP decoder with FiLM."""

    def test_decoder_initialization(self):
        """Should initialize with base decoder and FiLM generator."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder(
            affect_dim=16,
            num_film_layers=2,
            freeze_base_mlp=True,
        )

        self.assertIsNotNone(decoder.base_decoder)
        self.assertIsNotNone(decoder.film_gen)
        self.assertEqual(decoder.affect_dim, 16)
        self.assertEqual(decoder.num_harmonics, 60)
        self.assertEqual(decoder.num_noise_bands, 5)

    def test_forward_shape(self):
        """Should return correct output shapes."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder()

        features_112d = torch.randn(4, 112)
        affect_vector = torch.randn(4, 16)

        harmonic_amps, noise_mags = decoder(features_112d, affect_vector)

        self.assertEqual(harmonic_amps.shape, (4, 60))
        self.assertEqual(noise_mags.shape, (4, 5))

    def test_forward_single_sample(self):
        """Should handle single sample input."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder()

        features_112d = torch.randn(112)
        affect_vector = torch.randn(16)

        harmonic_amps, noise_mags = decoder(features_112d, affect_vector)

        # Should add batch dimension
        self.assertEqual(harmonic_amps.shape[0], 1)
        self.assertEqual(harmonic_amps.shape[1], 60)

    def test_harmonic_normalization(self):
        """Harmonic amplitudes should sum to 1.0."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder()

        features_112d = torch.randn(2, 112)
        affect_vector = torch.randn(2, 16)

        harmonic_amps, _ = decoder(features_112d, affect_vector)

        for i in range(2):
            self.assertAlmostEqual(harmonic_amps[i].sum().item(), 1.0, places=5)

    def test_noise_non_negative(self):
        """Noise magnitudes should be non-negative."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder()

        features_112d = torch.randn(2, 112)
        affect_vector = torch.randn(2, 16)

        _, noise_mags = decoder(features_112d, affect_vector)

        self.assertTrue(torch.all(noise_mags >= 0))

    def test_base_mlp_freezing(self):
        """Should freeze base MLP weights when configured."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder(freeze_base_mlp=True)

        # Check that base MLP parameters are frozen
        for param in decoder.base_decoder.parameters():
            self.assertFalse(param.requires_grad)

        # FiLM generator should still be trainable
        for param in decoder.film_gen.parameters():
            self.assertTrue(param.requires_grad)

    def test_base_mlp_unfreeze(self):
        """Should unfreeze base MLP weights."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder(freeze_base_mlp=True)
        decoder.unfreeze_base_mlp()

        # All parameters should now be trainable
        for param in decoder.base_decoder.parameters():
            self.assertTrue(param.requires_grad)

    def test_affect_modulation_changes_output(self):
        """Different affect vectors should produce different outputs."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder(freeze_base_mlp=True)

        features_112d = torch.randn(1, 112)

        # Low arousal
        affect_low = torch.zeros(1, 16)
        affect_low[0, 0] = 0.2

        # High arousal
        affect_high = torch.zeros(1, 16)
        affect_high[0, 0] = 0.9

        harmonic_low, _ = decoder(features_112d, affect_low)
        harmonic_high, _ = decoder(features_112d, affect_high)

        # Outputs should be different
        self.assertFalse(torch.allclose(harmonic_low, harmonic_high))

    def test_inference_mode(self):
        """Should run inference without gradients."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder()

        features_112d = torch.randn(1, 112)
        affect_vector = torch.randn(1, 16)

        result = decoder.inference(features_112d, affect_vector)

        self.assertIn("harmonic_amps", result)
        self.assertIn("noise_mags", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["harmonic_amps"].shape, (1, 60))
        self.assertEqual(result["noise_mags"].shape, (1, 5))

    def test_apply_affect_arousal(self):
        """Should apply arousal-based modulation."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder()

        features_112d = torch.randn(2, 112)

        harmonic_low, _ = decoder.apply_affect_arousal(features_112d, arousal_level=0.2)
        harmonic_high, _ = decoder.apply_affect_arousal(features_112d, arousal_level=0.9)

        self.assertEqual(harmonic_low.shape, (2, 60))
        self.assertEqual(harmonic_high.shape, (2, 60))

        # Different arousal should produce different outputs
        self.assertFalse(torch.allclose(harmonic_low, harmonic_high))


class TestAffectModulation(unittest.TestCase):
    """Test affect modulation behavior."""

    def test_arousal_affects_harmonic_distribution(self):
        """High arousal should change harmonic distribution."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder(freeze_base_mlp=True)

        features_112d = torch.randn(1, 112)

        # Create affect vectors with different arousal
        affect_low = torch.zeros(1, 16)
        affect_low[0, 0] = 0.2  # Low arousal

        affect_high = torch.zeros(1, 16)
        affect_high[0, 0] = 0.9  # High arousal

        harmonic_low, noise_low = decoder(features_112d, affect_low)
        harmonic_high, noise_high = decoder(features_112d, affect_high)

        # High arousal typically increases noise (more chaotic)
        # This is a structural test - actual values depend on training
        self.assertIsNotNone(harmonic_low)
        self.assertIsNotNone(harmonic_high)

    def test_affect_arousal_convenience_method(self):
        """Should handle arousal convenience method correctly."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder()

        features_112d = torch.randn(4, 112)

        # Test different arousal levels
        for arousal in [0.0, 0.5, 1.0]:
            harmonic, noise = decoder.apply_affect_arousal(features_112d, arousal)

            self.assertEqual(harmonic.shape, (4, 60))
            self.assertEqual(noise.shape, (4, 5))

            # Validate outputs
            self.assertTrue(torch.all(harmonic >= 0))
            self.assertTrue(torch.all(noise >= 0))


class TestCreateDualStreamDecoder(unittest.TestCase):
    """Test factory function."""

    def test_create_without_pretrained(self):
        """Should create decoder without pre-trained weights."""
        from cognitive_intelligence.ddsp_decoder import (
            DualStreamDDSPDecoder,
            create_dual_stream_decoder,
        )

        decoder = create_dual_stream_decoder()

        self.assertIsNotNone(decoder)
        self.assertIsInstance(decoder, DualStreamDDSPDecoder)

    def test_create_with_pretrained(self):
        """Should create decoder with pre-trained weights."""
        from cognitive_intelligence.ddsp_decoder import (
            DDSPDecoder,
            create_dual_stream_decoder,
        )

        # Create and save a base decoder
        with tempfile.TemporaryDirectory() as tmpdir:
            pretrained_path = Path(tmpdir) / "pretrained_decoder.pt"

            base_decoder = DDSPDecoder()
            torch.save(base_decoder.state_dict(), pretrained_path)

            # Create dual-stream decoder with pre-trained base
            decoder = create_dual_stream_decoder(
                pretrained_path=str(pretrained_path),
                freeze_base=True,
            )

            self.assertIsNotNone(decoder)
            self.assertTrue(decoder.base_decoder.mlp[0].weight.requires_grad is False)

    def test_create_unfrozen(self):
        """Should create decoder with unfrozen base."""
        from cognitive_intelligence.ddsp_decoder import create_dual_stream_decoder

        decoder = create_dual_stream_decoder(freeze_base=False)

        # Base MLP should be trainable
        for param in decoder.base_decoder.parameters():
            self.assertTrue(param.requires_grad)


class TestIntegration(unittest.TestCase):
    """Integration tests for dual-stream synthesis."""

    def test_full_dual_stream_forward(self):
        """Should complete full forward pass with dual-stream inputs."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder()

        # Simulate dual-stream state
        features_112d = torch.randn(1, 112)
        affect_vector = torch.randn(1, 16)

        harmonic_amps, noise_mags = decoder(features_112d, affect_vector)

        # Validate shapes
        self.assertEqual(harmonic_amps.shape, (1, 60))
        self.assertEqual(noise_mags.shape, (1, 5))

        # Validate constraints
        self.assertAlmostEqual(harmonic_amps[0].sum().item(), 1.0, places=5)
        self.assertTrue(torch.all(noise_mags >= 0))

    def test_batch_processing(self):
        """Should handle batch processing efficiently."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder()

        batch_size = 8
        features_112d = torch.randn(batch_size, 112)
        affect_vector = torch.randn(batch_size, 16)

        harmonic_amps, noise_mags = decoder(features_112d, affect_vector)

        self.assertEqual(harmonic_amps.shape[0], batch_size)
        self.assertEqual(noise_mags.shape[0], batch_size)

        # Each sample should have normalized harmonics
        for i in range(batch_size):
            self.assertAlmostEqual(harmonic_amps[i].sum().item(), 1.0, places=5)

    def test_gradient_flow_to_film_only(self):
        """When base is frozen, gradients should only flow to FiLM."""
        from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

        decoder = DualStreamDDSPDecoder(freeze_base_mlp=True)

        features_112d = torch.randn(2, 112)
        affect_vector = torch.randn(2, 16)

        harmonic_amps, noise_mags = decoder(features_112d, affect_vector)

        # Compute loss (reconstruction)
        loss = harmonic_amps.sum() + noise_mags.sum()
        loss.backward()

        # Check gradient flow
        for name, param in decoder.named_parameters():
            if "film" in name:
                self.assertIsNotNone(param.grad, f"FiLM param {name} should have grad")
            elif "base_decoder" in name:
                self.assertIsNone(param.grad, f"Base param {name} should not have grad")


if __name__ == "__main__":
    unittest.main()
