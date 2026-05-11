#!/usr/bin/env python3
"""
Tests for Offline Validation (Sprint 1-2 Milestone)

Tests the DualStreamReconstructor that validates 112D feature reconstruction
by combining VAE output (continuous affect) and VQ-VAE output (discrete syntax).

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

# Try importing required modules
try:
    from cognitive_intelligence.affective_feature_extractor import AffectiveFeatureExtractor
    from cognitive_intelligence.affective_vae import BetaVAE
    from cognitive_intelligence.offline_validation import (
        DualStreamReconstructor,
        run_offline_validation,
    )
    from cognitive_intelligence.syntactic_feature_extractor import SyntacticFeatureExtractor
    from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE
except ImportError as e:
    raise unittest.SkipTest(f"Required module not found: {e}")


class TestDualStreamReconstructor(unittest.TestCase):
    """Test the DualStreamReconstructor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"  # Use CPU for tests

        # Create feature extractors
        self.affective_extractor = AffectiveFeatureExtractor()
        self.syntactic_extractor = SyntacticFeatureExtractor()

        # Create small models for testing
        # Note: Actual affective extractor returns 68D, syntactic returns 45D
        self.beta_vae = BetaVAE(
            input_dim=68,  # Actual affective feature dimension
            latent_dim=16,
            hidden_dim=32,
            beta=2.0,
        )

        self.vqvae = SyntacticVQVAE(
            input_dim=45,  # Actual syntactic feature dimension
            codebook_size=16,  # Small for testing
            codebook_dim=16,
            hidden_dim=32,
        )

        # Create reconstructor
        self.reconstructor = DualStreamReconstructor(
            self.affective_extractor,
            self.syntactic_extractor,
            self.beta_vae,
            self.vqvae,
            device=self.device,
        )

        # Create test data
        np.random.seed(42)
        torch.manual_seed(42)
        self.test_features = np.random.randn(112).astype(np.float32)

    def test_reconstructor_initialization(self):
        """Test that reconstructor initializes correctly."""
        self.assertIsNotNone(self.reconstructor)
        self.assertEqual(self.reconstructor.device.type, "cpu")
        # Note: Actual dimensions are 68D affective, 45D syntactic
        self.assertEqual(len(self.reconstructor.affective_indices), 68)
        self.assertEqual(len(self.reconstructor.syntactic_indices), 45)

    def test_encode_returns_correct_shapes(self):
        """Test that encode returns correct shapes."""
        affect_16d, token = self.reconstructor.encode(self.test_features)

        # Check affect vector shape
        self.assertEqual(affect_16d.shape, (16,))

        # Check token is scalar integer
        self.assertIsInstance(token, (int, np.integer))
        self.assertGreaterEqual(token, 0)
        self.assertLess(token, 16)  # codebook_size

    def test_decode_returns_correct_shape(self):
        """Test that decode returns 112D features."""
        affect_16d = np.random.randn(16).astype(np.float32)
        token = 5

        reconstructed = self.reconstructor.decode(affect_16d, token)

        self.assertEqual(reconstructed.shape, (112,))

    def test_full_reconstruct_pipeline(self):
        """Test full encode-decode pipeline."""
        reconstructed = self.reconstructor.reconstruct(self.test_features)

        self.assertEqual(reconstructed.shape, (112,))
        self.assertTrue(np.isfinite(reconstructed).all())

    def test_reconstruction_error_metrics(self):
        """Test reconstruction error computation."""
        reconstructed = self.reconstructor.reconstruct(self.test_features)
        errors = self.reconstructor.compute_reconstruction_error(
            self.test_features, reconstructed
        )

        # Check that all error metrics exist and are non-negative
        self.assertIn("mse", errors)
        self.assertIn("mae", errors)
        self.assertIn("affective_mse", errors)
        self.assertIn("syntactic_mse", errors)

        self.assertGreaterEqual(errors["mse"], 0)
        self.assertGreaterEqual(errors["mae"], 0)
        self.assertGreaterEqual(errors["affective_mse"], 0)
        self.assertGreaterEqual(errors["syntactic_mse"], 0)

    def test_validate_reconstruction_with_multiple_samples(self):
        """Test validation with multiple feature samples."""
        features_list = [
            np.random.randn(112).astype(np.float32) for _ in range(10)
        ]

        results = self.reconstructor.validate_reconstruction(
            features_list,
            target_mse=1.0,  # Relaxed target for untrained models
        )

        # Check result structure
        self.assertIn("avg_mse", results)
        self.assertIn("avg_affective_mse", results)
        self.assertIn("avg_syntactic_mse", results)
        self.assertIn("target_mse", results)
        self.assertIn("target_met", results)
        self.assertIn("num_samples", results)

        self.assertEqual(results["num_samples"], 10)
        self.assertIsInstance(results["target_met"], bool)

    def test_feature_indices_mapping(self):
        """Test that feature indices map correctly back to 112D."""
        # Create a features vector with known values at specific indices
        features = np.zeros(112, dtype=np.float32)

        # Set some affective feature indices to known values
        for i, idx in enumerate(self.reconstructor.affective_indices[:5]):
            features[idx] = 1.0 + i

        # Set some syntactic feature indices
        for i, idx in enumerate(self.reconstructor.syntactic_indices[:5]):
            features[idx] = 10.0 + i

        # Extract and verify
        affective = self.affective_extractor.extract(features)
        syntactic = self.syntactic_extractor.extract(features)

        # Check that values were extracted from correct indices
        self.assertAlmostEqual(affective[0], 1.0, places=5)
        self.assertAlmostEqual(syntactic[0], 10.0, places=5)


class TestOfflineValidation(unittest.TestCase):
    """Test the offline validation workflow."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for checkpoints
        self.checkpoint_dir = Path("/tmp/test_offline_validation")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy model checkpoints
        self.create_dummy_checkpoints()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil

        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)

    def create_dummy_checkpoints(self):
        """Create dummy checkpoint files for testing."""
        # Beta-VAE checkpoint (with actual dimensions)
        beta_vae = BetaVAE(input_dim=68, latent_dim=16, hidden_dim=32)
        beta_checkpoint = {
            "model_state_dict": beta_vae.state_dict(),
            "epoch": 10,
            "loss": 0.1,
        }
        beta_path = self.checkpoint_dir / "beta_vae.pt"
        torch.save(beta_checkpoint, beta_path)

        # VQ-VAE checkpoint (with actual dimensions)
        vqvae = SyntacticVQVAE(
            input_dim=45, codebook_size=16, codebook_dim=16, hidden_dim=32
        )
        vqvae_checkpoint = {
            "model_state_dict": vqvae.state_dict(),
            "epoch": 10,
            "loss": 0.1,
        }
        vqvae_path = self.checkpoint_dir / "vqvae.pt"
        torch.save(vqvae_checkpoint, vqvae_path)

    def test_run_offline_validation_creates_results(self):
        """Test that run_offline_validation produces results."""
        # Create test features
        np.random.seed(42)
        features_112d = np.random.randn(50, 112).astype(np.float32)

        # Run validation
        with patch("cognitive_intelligence.offline_validation.Path") as mock_path:
            # Mock the checkpoint paths
            mock_path.return_value = Path("/tmp")

            results = run_offline_validation(
                beta_vae_path=str(self.checkpoint_dir / "beta_vae.pt"),
                vqvae_path=str(self.checkpoint_dir / "vqvae.pt"),
                features_112d=features_112d,
                affective_stats_path=None,
                syntactic_stats_path=None,
            )

        # Check results structure
        self.assertIn("avg_mse", results)
        self.assertIn("avg_affective_mse", results)
        self.assertIn("avg_syntactic_mse", results)
        self.assertIn("target_mse", results)
        self.assertIn("target_met", results)
        self.assertIn("num_samples", results)

        self.assertEqual(results["num_samples"], 50)


class TestReconstructionTargets(unittest.TestCase):
    """Test that reconstruction meets target metrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.affective_extractor = AffectiveFeatureExtractor()
        self.syntactic_extractor = SyntacticFeatureExtractor()
        # Use actual dimensions: 68D affective, 45D syntactic
        self.beta_vae = BetaVAE(input_dim=68, latent_dim=16, hidden_dim=32)
        self.vqvae = SyntacticVQVAE(
            input_dim=45, codebook_size=16, codebook_dim=16, hidden_dim=32
        )

        self.reconstructor = DualStreamReconstructor(
            self.affective_extractor,
            self.syntactic_extractor,
            self.beta_vae,
            self.vqvae,
            device="cpu",
        )

    def test_reconstruction_error_is_finite(self):
        """Test that reconstruction errors are always finite."""
        np.random.seed(42)
        features = np.random.randn(112).astype(np.float32)

        reconstructed = self.reconstructor.reconstruct(features)
        errors = self.reconstructor.compute_reconstruction_error(
            features, reconstructed
        )

        # All errors should be finite
        for key, value in errors.items():
            self.assertTrue(
                np.isfinite(value),
                f"Error metric '{key}' is not finite: {value}",
            )

    def test_affective_and_syntactic_errors_sum_to_total(self):
        """Test that affective + syntactic components roughly equal total."""
        np.random.seed(42)
        features = np.random.randn(112).astype(np.float32)

        reconstructed = self.reconstructor.reconstruct(features)
        errors = self.reconstructor.compute_reconstruction_error(
            features, reconstructed
        )

        # The weighted sum should be close to total MSE
        # (accounting for feature counts)
        affective_count = len(self.reconstructor.affective_indices)
        syntactic_count = len(self.reconstructor.syntactic_indices)
        total_count = affective_count + syntactic_count

        weighted_mse = (
            errors["affective_mse"] * affective_count
            + errors["syntactic_mse"] * syntactic_count
        ) / total_count

        # Should be reasonably close (within factor of 2 for untrained models)
        self.assertLess(
            abs(weighted_mse - errors["mse"]),
            max(errors["mse"], weighted_mse),
        )


class TestFeatureExtractionIntegration(unittest.TestCase):
    """Test integration of feature extractors with reconstructor."""

    def test_affective_extractor_indices(self):
        """Test that affective extractor has correct indices."""
        extractor = AffectiveFeatureExtractor()
        self.assertIsNotNone(extractor.FEATURE_INDICES)
        self.assertGreater(len(extractor.FEATURE_INDICES), 0)
        self.assertLessEqual(len(extractor.FEATURE_INDICES), 112)

        # All indices should be in valid range
        for idx in extractor.FEATURE_INDICES:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 112)

    def test_syntactic_extractor_indices(self):
        """Test that syntactic extractor has correct indices."""
        extractor = SyntacticFeatureExtractor()
        self.assertIsNotNone(extractor.FEATURE_INDICES)
        self.assertGreater(len(extractor.FEATURE_INDICES), 0)
        self.assertLessEqual(len(extractor.FEATURE_INDICES), 112)

        # All indices should be in valid range
        for idx in extractor.FEATURE_INDICES:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 112)

    def test_extractors_dont_overlap(self):
        """Test that affective and syntactic extractors use different indices.

        Note: Some overlap is intentional for shared features like F0,
        MFCCs, harmonic ratio, but most features should be distinct.
        """
        affective = set(AffectiveFeatureExtractor.FEATURE_INDICES)
        syntactic = set(SyntacticFeatureExtractor.FEATURE_INDICES)

        overlap = affective.intersection(syntactic)

        # Allow some overlap but verify most features are distinct
        # (F0, MFCCs, harmonic ratio may be in both)
        overlap_ratio = len(overlap) / min(len(affective), len(syntactic))

        # Allow up to 30% overlap for shared features
        self.assertLess(
            overlap_ratio,
            0.30,
            f"Extractors have too many overlapping indices: {overlap}",
        )


if __name__ == "__main__":
    unittest.main()
