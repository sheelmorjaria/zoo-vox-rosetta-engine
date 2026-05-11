#!/usr/bin/env python3
"""
Tests for Affective VAE (Stream 1)

These tests verify the β-VAE for continuous affect encoding,
including feature extraction, disentanglement, and modulation.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class TestAffectiveFeatureExtractor(unittest.TestCase):
    """Test affective feature extraction from 112D Rosetta vector."""

    def test_extract_affective_features(self):
        """Should extract 54D affective features from 112D input."""
        from cognitive_intelligence.affective_encoder import AffectiveFeatureExtractor

        # Create valid 112D input
        features_112d = np.random.randn(112).astype(np.float32)

        affective = AffectiveFeatureExtractor.extract_affective_features(features_112d)

        self.assertEqual(len(affective), AffectiveFeatureExtractor.OUTPUT_DIM)
        self.assertEqual(AffectiveFeatureExtractor.OUTPUT_DIM, 54)

    def test_batch_extraction(self):
        """Should handle batch input."""
        from cognitive_intelligence.affective_encoder import AffectiveFeatureExtractor

        batch = np.random.randn(10, 112).astype(np.float32)
        affective = AffectiveFeatureExtractor.extract_affective_features(batch)

        self.assertEqual(affective.shape, (10, 54))

    def test_invalid_shape_raises_error(self):
        """Should raise ValueError for incorrect input shape."""
        from cognitive_intelligence.affective_encoder import AffectiveFeatureExtractor

        with self.assertRaises(ValueError):
            AffectiveFeatureExtractor.extract_affective_features(np.random.randn(100))

    def test_validate_feature_vector(self):
        """Should validate correctly formed 112D vectors."""
        from cognitive_intelligence.affective_encoder import AffectiveFeatureExtractor

        # Valid vector - create with valid values for key features
        valid = np.random.randn(112).astype(np.float32)
        valid[0] = 9000.0  # Valid F0
        valid[7] = 10.0  # Valid HNR
        valid[35] = 0.5  # Valid Jitter
        valid[36] = 0.5  # Valid Shimmer
        self.assertTrue(AffectiveFeatureExtractor.validate_feature_vector(valid))

        # Invalid shape
        self.assertFalse(AffectiveFeatureExtractor.validate_feature_vector(np.random.randn(100)))

        # NaN values
        with_nan = np.random.randn(112).astype(np.float32)
        with_nan[0] = np.nan
        self.assertFalse(AffectiveFeatureExtractor.validate_feature_vector(with_nan))

    def test_feature_names(self):
        """Should return correct number of feature names."""
        from cognitive_intelligence.affective_encoder import AffectiveFeatureExtractor

        names = AffectiveFeatureExtractor.get_feature_names()

        self.assertEqual(len(names), 54)  # Names list matches OUTPUT_DIM
        self.assertIn("F0_mean", names)
        self.assertIn("HNR", names)
        self.assertIn("Jitter", names)


class TestAffectiveNormalization(unittest.TestCase):
    """Test normalization utilities."""

    def test_compute_normalization(self):
        """Should compute mean and std for normalization."""
        from cognitive_intelligence.affective_encoder import AffectiveNormalization

        features = np.random.randn(100, 72).astype(np.float32)

        mean, std = AffectiveNormalization.compute_normalization(features)

        self.assertEqual(mean.shape, (72,))
        self.assertEqual(std.shape, (72,))
        self.assertTrue(np.all(std > 0))  # No division by zero

    def test_normalize_denormalize_roundtrip(self):
        """Should preserve original values through normalize/denormalize."""
        from cognitive_intelligence.affective_encoder import AffectiveNormalization

        original = np.random.randn(10, 72).astype(np.float32)
        mean, std = AffectiveNormalization.compute_normalization(original)

        normalized = AffectiveNormalization.normalize(original, mean, std)
        denormalized = AffectiveNormalization.denormalize(normalized, mean, std)

        np.testing.assert_array_almost_equal(original, denormalized, decimal=5)


class TestBetaVAE(unittest.TestCase):
    """Test β-VAE architecture."""

    def test_vae_initialization(self):
        """Should initialize with correct dimensions."""
        from cognitive_intelligence.affective_vae import BetaVAE

        vae = BetaVAE(input_dim=72, latent_dim=16, hidden_dim=128, beta=2.0)

        self.assertEqual(vae.input_dim, 72)
        self.assertEqual(vae.latent_dim, 16)
        self.assertEqual(vae.beta, 2.0)

    def test_encode_decode(self):
        """Should encode and decode with correct shapes."""
        from cognitive_intelligence.affective_vae import BetaVAE

        vae = BetaVAE(input_dim=72, latent_dim=16, hidden_dim=128)
        vae.eval()

        x = torch.randn(4, 72)
        mu, logvar = vae.encode(x)

        self.assertEqual(mu.shape, (4, 16))
        self.assertEqual(logvar.shape, (4, 16))

        z = vae.reparameterize(mu, logvar)
        recon = vae.decode(z)

        self.assertEqual(recon.shape, (4, 72))

    def test_forward_pass(self):
        """Should return reconstruction, mu, logvar."""
        from cognitive_intelligence.affective_vae import BetaVAE

        vae = BetaVAE(input_dim=72, latent_dim=16)
        vae.eval()

        x = torch.randn(2, 72)
        recon, mu, logvar = vae(x)

        self.assertEqual(recon.shape, (2, 72))
        self.assertEqual(mu.shape, (2, 16))
        self.assertEqual(logvar.shape, (2, 16))

    def test_loss_function(self):
        """Should compute β-VAE loss with weighted KL."""
        from cognitive_intelligence.affective_vae import BetaVAE

        vae = BetaVAE(input_dim=72, latent_dim=16, beta=2.0)

        x = torch.randn(4, 72)
        recon, mu, logvar = vae(x)

        total_loss, recon_loss, kl_loss = vae.loss_function(recon, x, mu, logvar)

        # Total loss should be weighted sum
        expected_total = recon_loss + vae.beta * kl_loss

        self.assertTrue(torch.allclose(total_loss, expected_total, atol=1e-5))

        # KL loss should be non-negative
        self.assertGreaterEqual(kl_loss.item(), 0.0)

    def test_sample_from_prior(self):
        """Should sample from prior distribution."""
        from cognitive_intelligence.affective_vae import BetaVAE

        vae = BetaVAE(input_dim=72, latent_dim=16)
        vae.eval()

        samples = vae.sample(n_samples=10)

        self.assertEqual(samples.shape, (10, 72))

    def test_interpolate(self):
        """Should interpolate between two inputs."""
        from cognitive_intelligence.affective_vae import BetaVAE

        vae = BetaVAE(input_dim=72, latent_dim=16)
        vae.eval()

        x1 = torch.randn(1, 72)
        x2 = torch.randn(1, 72)

        interpolated = vae.interpolate(x1, x2, n_steps=5)

        self.assertEqual(interpolated.shape, (5, 72))


class TestAffectiveResponsePolicy(unittest.TestCase):
    """Test biologically-inspired affective response policy."""

    def test_deescalate_high_arousal(self):
        """Should de-escalate high arousal (>0.8)."""
        from cognitive_intelligence.affective_vae import AffectiveResponsePolicy

        # High arousal input
        high_arousal = np.zeros(16)
        high_arousal[0] = 0.9  # Arousal dimension

        target = AffectiveResponsePolicy.compute_target_affect(high_arousal)

        # Should be scaled down
        self.assertLess(target[0], high_arousal[0])
        self.assertLess(target[0], 0.8)  # Below threshold

    def test_escalate_low_arousal(self):
        """Should escalate low arousal (<0.3)."""
        from cognitive_intelligence.affective_vae import AffectiveResponsePolicy

        low_arousal = np.zeros(16)
        low_arousal[0] = 0.2

        target = AffectiveResponsePolicy.compute_target_affect(low_arousal)

        # Should be scaled up
        self.assertGreater(target[0], low_arousal[0])

    def test_match_medium_arousal(self):
        """Should match medium arousal (0.3-0.8)."""
        from cognitive_intelligence.affective_vae import AffectiveResponsePolicy

        medium_arousal = np.zeros(16)
        medium_arousal[0] = 0.5

        target = AffectiveResponsePolicy.compute_target_affect(medium_arousal)

        # Should remain unchanged (return as-is)
        self.assertEqual(target[0], medium_arousal[0])

    def test_affective_distance(self):
        """Should compute Euclidean distance between affect vectors."""
        from cognitive_intelligence.affective_vae import AffectiveResponsePolicy

        affect1 = np.zeros(16)
        affect2 = np.zeros(16)
        affect2[0] = 1.0

        distance = AffectiveResponsePolicy.compute_affective_distance(affect1, affect2)

        self.assertEqual(distance, 1.0)


class TestAffectVAECheckpoint(unittest.TestCase):
    """Test checkpoint management."""

    def test_save_load_checkpoint(self):
        """Should save and load model checkpoint."""
        from cognitive_intelligence.affective_vae import (
            AffectVAECheckpoint,
            BetaVAE,
            create_affective_vae,
        )

        vae = create_affective_vae()
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            AffectVAECheckpoint.save_checkpoint(
                vae, optimizer, epoch=10, loss=0.5, path=path
            )

            self.assertTrue(path.exists())

            # Load checkpoint
            checkpoint = AffectVAECheckpoint.load_checkpoint(path)

            self.assertEqual(checkpoint['epoch'], 10)
            self.assertEqual(checkpoint['loss'], 0.5)

    def test_save_load_model_only(self):
        """Should save and load model state dict."""
        from cognitive_intelligence.affective_vae import (
            AffectVAECheckpoint,
            create_affective_vae,
        )

        vae = create_affective_vae()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"

            AffectVAECheckpoint.save_model_only(vae, path)

            self.assertTrue(path.exists())

            # Load model
            loaded = AffectVAECheckpoint.load_model_only(path)

            self.assertEqual(loaded.input_dim, vae.input_dim)
            self.assertEqual(loaded.latent_dim, vae.latent_dim)

            # Check weights match
            for p1, p2 in zip(vae.parameters(), loaded.parameters()):
                self.assertTrue(torch.allclose(p1, p2))


class TestIntegration(unittest.TestCase):
    """Integration tests for affective stream."""

    def test_end_to_end_extraction_and_encoding(self):
        """Should extract features and encode to latent space."""
        from cognitive_intelligence.affective_encoder import AffectiveFeatureExtractor
        from cognitive_intelligence.affective_vae import create_affective_vae

        # Create sample 112D features
        features_112d = np.random.randn(112).astype(np.float32)

        # Extract affective features
        affective = AffectiveFeatureExtractor.extract_affective_features(features_112d)

        # Encode to latent space
        vae = create_affective_vae()
        vae.eval()

        with torch.no_grad():
            x = torch.from_numpy(affective).unsqueeze(0)
            mu, logvar = vae.encode(x)

        self.assertEqual(mu.shape, (1, 16))


if __name__ == "__main__":
    unittest.main()
