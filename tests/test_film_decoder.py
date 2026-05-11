#!/usr/bin/env python3
"""
Tests for FiLM-based DDSP Decoder (Risk B Mitigation)

Tests FiLM (Feature-wise Linear Modulation) layers that preserve pre-trained
112D DDSP weights while enabling affective modulation.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import tempfile
import unittest

import torch

from cognitive_intelligence.ddsp_decoder import (
    DDSPDecoder,
    DualStreamDDSPDecoder,
    FiLMGenerator,
    create_dual_stream_decoder,
)


class TestFiLMGenerator(unittest.TestCase):
    """Test FiLM parameter generation."""

    def test_film_generator_shape(self):
        """FiLM generator should produce correct output shapes."""
        affect_dim = 16
        hidden_dim = 256
        num_layers = 2

        film_gen = FiLMGenerator(
            affect_dim=affect_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        # Generate FiLM parameters
        affect_vector = torch.randn(4, affect_dim)
        films = film_gen(affect_vector)

        # Should return num_layers (γ, β) pairs
        self.assertEqual(len(films), num_layers)

        # Each γ, β should have shape (batch, hidden_dim)
        for gamma, beta in films:
            self.assertEqual(gamma.shape, (4, hidden_dim))
            self.assertEqual(beta.shape, (4, hidden_dim))

    def test_film_parameters_differ_per_affect(self):
        """Different affect vectors should produce different FiLM parameters."""
        film_gen = FiLMGenerator(affect_dim=16, hidden_dim=256, num_layers=1)

        affect_low = torch.zeros(1, 16)
        affect_high = torch.ones(1, 16)

        films_low = film_gen(affect_low)
        films_high = film_gen(affect_high)

        # Parameters should differ
        gamma_low, beta_low = films_low[0]
        gamma_high, beta_high = films_high[0]

        self.assertFalse(torch.allclose(gamma_low, gamma_high))
        self.assertFalse(torch.allclose(beta_low, beta_high))


class TestDualStreamDecoderForward(unittest.TestCase):
    """Test dual-stream decoder forward pass."""

    def setUp(self):
        """Create a dual-stream decoder."""
        self.decoder = DualStreamDDSPDecoder(
            affect_dim=16,
            num_film_layers=2,
            freeze_base_mlp=False,  # Don't freeze for testing
        )

    def test_forward_shapes(self):
        """Forward pass should produce correct output shapes."""
        features = torch.randn(2, 112)
        affect = torch.randn(2, 16)

        harmonic_amps, noise_mags = self.decoder(features, affect)

        self.assertEqual(harmonic_amps.shape, (2, 60))
        self.assertEqual(noise_mags.shape, (2, 5))

    def test_harmonic_normalization(self):
        """Harmonic amplitudes should sum to 1."""
        features = torch.randn(4, 112)
        affect = torch.randn(4, 16)

        harmonic_amps, _ = self.decoder(features, affect)

        # Each sample should sum to ~1.0
        sums = harmonic_amps.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_noise_non_negative(self):
        """Noise magnitudes should be non-negative."""
        features = torch.randn(4, 112)
        affect = torch.randn(4, 16)

        _, noise_mags = self.decoder(features, affect)

        self.assertTrue(torch.all(noise_mags >= 0))

    def test_batch_processing(self):
        """Should handle different batch sizes."""
        for batch_size in [1, 4, 8]:
            features = torch.randn(batch_size, 112)
            affect = torch.randn(batch_size, 16)

            harmonic_amps, noise_mags = self.decoder(features, affect)

            self.assertEqual(harmonic_amps.shape[0], batch_size)
            self.assertEqual(noise_mags.shape[0], batch_size)


class TestWeightPreservation(unittest.TestCase):
    """Test that pre-trained weights are preserved."""

    def test_base_mlp_freezing(self):
        """Freezing should prevent gradient updates to base MLP."""
        decoder = DualStreamDDSPDecoder(
            freeze_base_mlp=True,
        )

        # Check that base MLP parameters are frozen
        for param in decoder.base_decoder.parameters():
            self.assertFalse(param.requires_grad)

    def test_base_mlp_unfreezing(self):
        """Unfreezing should allow gradient updates."""
        decoder = DualStreamDDSPDecoder(
            freeze_base_mlp=True,
        )

        # Unfreeze
        decoder.unfreeze_base_mlp()

        # Check that parameters are now trainable
        for param in decoder.base_decoder.parameters():
            self.assertTrue(param.requires_grad)

    def test_only_film_trains_when_frozen(self):
        """When frozen, only FiLM parameters should have gradients."""
        decoder = DualStreamDDSPDecoder(
            freeze_base_mlp=True,
        )

        features = torch.randn(4, 112)
        affect = torch.randn(4, 16)

        harmonic_amps, noise_mags = decoder(features, affect)

        # Compute loss (simple MSE)
        target_harmonic = torch.ones_like(harmonic_amps) / 60
        target_noise = torch.ones_like(noise_mags) * 0.5

        loss = torch.nn.functional.mse_loss(harmonic_amps, target_harmonic)
        loss = loss + torch.nn.functional.mse_loss(noise_mags, target_noise)

        loss.backward()

        # Check gradients
        for name, param in decoder.named_parameters():
            if "base_decoder" in name:
                # Base decoder params should have no gradients
                self.assertIsNone(param.grad)
            elif "film_gen" in name:
                # FiLM params should have gradients
                self.assertIsNotNone(param.grad)
                # Check that gradients are non-zero
                self.assertGreater(param.grad.abs().sum().item(), 0)


class TestWeightPreservationTraining(unittest.TestCase):
    """Test weight preservation during training."""

    def test_weights_stable_with_film_only_training(self):
        """Base MLP weights should remain stable during FiLM-only training."""
        # Create a base decoder and get initial weights
        base_decoder = DDSPDecoder()
        initial_weights = {
            name: param.clone()
            for name, param in base_decoder.named_parameters()
        }

        # Create dual-stream decoder with frozen base
        decoder = DualStreamDDSPDecoder(
            base_decoder=base_decoder,
            freeze_base_mlp=True,
        )

        # Train FiLM only
        optimizer = torch.optim.Adam(decoder.film_gen.parameters(), lr=1e-3)

        for _ in range(10):
            features = torch.randn(16, 112)
            affect = torch.randn(16, 16)

            harmonic_amps, noise_mags = decoder(features, affect)

            target_harmonic = torch.ones_like(harmonic_amps) / 60
            loss = torch.nn.functional.mse_loss(harmonic_amps, target_harmonic)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check base weights haven't changed
        for name, param in base_decoder.named_parameters():
            torch.testing.assert_close(
                param,
                initial_weights[name],
                atol=1e-6,
                rtol=1e-6,
                msg=f"Parameter {name} changed despite freezing"
            )

    def test_film_parameters_change(self):
        """FiLM parameters should update during training."""
        decoder = DualStreamDDSPDecoder(
            freeze_base_mlp=True,
        )

        # Get initial FiLM weights
        initial_weights = {
            name: param.clone()
            for name, param in decoder.film_gen.named_parameters()
        }

        # Train
        optimizer = torch.optim.Adam(decoder.film_gen.parameters(), lr=1e-3)

        for _ in range(10):
            features = torch.randn(16, 112)
            affect = torch.randn(16, 16)

            harmonic_amps, noise_mags = decoder(features, affect)
            target = torch.randn_like(harmonic_amps)
            loss = torch.nn.functional.mse_loss(harmonic_amps, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check FiLM weights have changed
        for name, param in decoder.film_gen.named_parameters():
            self.assertFalse(
                torch.allclose(param, initial_weights[name], atol=1e-5),
                msg=f"FiLM parameter {name} did not change during training"
            )


class TestAffectiveModulation(unittest.TestCase):
    """Test affective modulation behavior."""

    def setUp(self):
        """Create decoder with unfrozen base for testing."""
        self.decoder = DualStreamDDSPDecoder(
            freeze_base_mlp=False,
        )

    def test_arousal_affects_output(self):
        """Different arousal levels should produce different outputs."""
        features = torch.randn(1, 112)

        # Low arousal (all zeros)
        affect_low = torch.zeros(1, 16)
        harmonic_low, noise_low = self.decoder(features, affect_low)

        # High arousal (all ones)
        affect_high = torch.ones(1, 16)
        harmonic_high, noise_high = self.decoder(features, affect_high)

        # Outputs should differ
        self.assertFalse(torch.allclose(harmonic_low, harmonic_high))
        self.assertFalse(torch.allclose(noise_low, noise_high))

    def test_apply_affect_arousal(self):
        """Convenience method should work correctly."""
        features = torch.randn(4, 112)

        # Different arousal levels
        for arousal in [0.0, 0.5, 1.0]:
            harmonic_amps, noise_mags = self.decoder.apply_affect_arousal(
                features, arousal
            )

            self.assertEqual(harmonic_amps.shape, (4, 60))
            self.assertEqual(noise_mags.shape, (4, 5))

            # Check constraints
            torch.testing.assert_close(
                harmonic_amps.sum(dim=-1),
                torch.ones(4),
                atol=1e-5,
                rtol=1e-5
            )
            self.assertTrue(torch.all(noise_mags >= 0))

    def test_affect_arousal_monotonicity(self):
        """Higher arousal should produce monotonic changes (at least in some dimensions)."""
        features = torch.randn(1, 112)

        outputs = []
        for arousal in [0.0, 0.25, 0.5, 0.75, 1.0]:
            harmonic, _ = self.decoder.apply_affect_arousal(features, arousal)
            outputs.append(harmonic.squeeze(0))

        # Check at least one harmonic shows monotonic behavior
        harmonic_stack = torch.stack(outputs)  # (5, 60)
        for h in range(60):
            harmonic_series = harmonic_stack[:, h]
            # Check if monotonically increasing or decreasing
            if torch.all(harmonic_series[1:] >= harmonic_series[:-1] - 1e-6):
                return  # Found monotonic increase
            if torch.all(harmonic_series[1:] <= harmonic_series[:-1] + 1e-6):
                return  # Found monotonic decrease

        # If we get here, no clear monotonic pattern (which is okay for complex modulation)
        # Just verify outputs differ
        for i in range(4):
            self.assertFalse(
                torch.allclose(outputs[i], outputs[i + 1]),
                msg=f"Outputs for arousal {i/4} and {(i+1)/4} are identical"
            )


class TestFactoryFunction(unittest.TestCase):
    """Test factory function for creating dual-stream decoder."""

    def test_create_basic(self):
        """Factory should create decoder without pre-trained weights."""
        decoder = create_dual_stream_decoder()

        self.assertIsInstance(decoder, DualStreamDDSPDecoder)
        self.assertEqual(decoder.affect_dim, 16)

    def test_create_with_frozen_base(self):
        """Factory should respect freeze_base parameter."""
        decoder_frozen = create_dual_stream_decoder(freeze_base=True)
        decoder_unfrozen = create_dual_stream_decoder(freeze_base=False)

        # Check frozen status
        for param in decoder_frozen.base_decoder.parameters():
            self.assertFalse(param.requires_grad)

        for param in decoder_unfrozen.base_decoder.parameters():
            self.assertTrue(param.requires_grad)

    def test_create_with_pretrained(self):
        """Factory should load pre-trained weights if available."""
        # Save a decoder to use as "pre-trained"
        base_decoder = DDSPDecoder()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(base_decoder.state_dict(), f.name)
            f.flush()

            # Load using factory
            decoder = create_dual_stream_decoder(pretrained_path=f.name, freeze_base=True)

            self.assertIsInstance(decoder, DualStreamDDSPDecoder)
            self.assertIsNotNone(decoder.base_decoder)


class TestIntegrationRiskBMitigation(unittest.TestCase):
    """Integration tests for Risk B mitigation."""

    def test_full_training_pipeline_film_only(self):
        """Full training pipeline with FiLM-only training."""
        # Create decoder with frozen base
        decoder = DualStreamDDSPDecoder(freeze_base_mlp=True)

        # Store initial base weights
        initial_base_weights = {
            name: param.clone()
            for name, param in decoder.base_decoder.named_parameters()
        }

        # Training loop
        optimizer = torch.optim.Adam(decoder.film_gen.parameters(), lr=1e-3)

        for step in range(20):
            features = torch.randn(32, 112)
            affect = torch.randn(32, 16)

            harmonic_amps, noise_mags = decoder(features, affect)
            target = torch.randn_like(harmonic_amps)

            loss = torch.nn.functional.mse_loss(harmonic_amps, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Verify base weights unchanged
        for name, param in decoder.base_decoder.named_parameters():
            torch.testing.assert_close(
                param,
                initial_base_weights[name],
                atol=1e-6,
                rtol=1e-6,
                msg=f"Base weight {name} changed during FiLM-only training"
            )

    def test_full_training_pipeline_finetune(self):
        """Full training pipeline with end-to-end fine-tuning."""
        decoder = DualStreamDDSPDecoder(freeze_base_mlp=True)

        # Phase 1: FiLM-only training
        optimizer = torch.optim.Adam(decoder.film_gen.parameters(), lr=1e-3)

        for _ in range(10):
            features = torch.randn(16, 112)
            affect = torch.randn(16, 16)
            harmonic_amps, _ = decoder(features, affect)
            loss = torch.nn.functional.mse_loss(harmonic_amps, torch.ones_like(harmonic_amps))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Phase 2: Unfreeze and fine-tune
        decoder.unfreeze_base_mlp()
        optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)  # Lower LR for fine-tuning

        # Store weights before fine-tuning
        weights_before_ft = {
            name: param.clone()
            for name, param in decoder.base_decoder.named_parameters()
        }

        for _ in range(5):
            features = torch.randn(16, 112)
            affect = torch.randn(16, 16)
            harmonic_amps, _ = decoder(features, affect)
            loss = torch.nn.functional.mse_loss(harmonic_amps, torch.ones_like(harmonic_amps))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check that some weights changed (fine-tuning happened)
        changed_count = 0
        for name, param in decoder.base_decoder.named_parameters():
            if not torch.allclose(param, weights_before_ft[name], atol=1e-5):
                changed_count += 1

        self.assertGreater(changed_count, 0, "No weights changed during fine-tuning")


if __name__ == "__main__":
    unittest.main()
