#!/usr/bin/env python3
"""
Tests for Affective β-VAE ONNX Export

Tests the ONNX export functionality for the β-VAE encoder,
ensuring compatibility with Rust ONNX Runtime inference.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cognitive_intelligence.affective_export import (
    AffectiveEncoderONNX,
    export_affective_vae_to_onnx,
    verify_onnx_model,
)
from cognitive_intelligence.affective_vae import BetaVAE, AffectVAEConfig


class TestAffectiveEncoderONNX(unittest.TestCase):
    """Test ONNX export for affective encoder."""

    def setUp(self):
        """Create a test β-VAE model."""
        self.config = AffectVAEConfig(
            input_dim=54,
            latent_dim=16,
            hidden_dim=64,
            beta=2.0,
        )
        self.model = BetaVAE(
            input_dim=self.config.input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
            beta=self.config.beta,
        )
        self.model.eval()

    def test_export_creates_file(self):
        """Should create ONNX file at specified path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "affective_encoder.onnx"
            export_affective_vae_to_onnx(
                self.model,
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())

    def test_export_has_correct_inputs(self):
        """Should have single input node with shape (dynamic, 54)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "affective_encoder.onnx"
            export_affective_vae_to_onnx(
                self.model,
                output_path=output_path,
            )

            metadata = verify_onnx_model(output_path)
            self.assertEqual(metadata["input_names"], ["affective_features"])
            self.assertEqual(metadata["input_shapes"], [["dynamic", 54]])

    def test_export_has_correct_outputs(self):
        """Should have output node with shape (dynamic, 16)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "affective_encoder.onnx"
            export_affective_vae_to_onnx(
                self.model,
                output_path=output_path,
            )

            metadata = verify_onnx_model(output_path)
            self.assertEqual(metadata["output_names"], ["latent_affect"])
            self.assertEqual(metadata["output_shapes"], [["dynamic", 16]])

    def test_opset_version(self):
        """Should use ONNX opset version 17 or higher."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "affective_encoder.onnx"
            export_affective_vae_to_onnx(
                self.model,
                output_path=output_path,
                opset_version=17,
            )

            metadata = verify_onnx_model(output_path)
            self.assertGreaterEqual(metadata["opset_version"], 17)


class TestAffectiveEncoderONNXClass(unittest.TestCase):
    """Test the AffectiveEncoderONNX wrapper class."""

    def setUp(self):
        """Create test model and ONNX export."""
        self.config = AffectVAEConfig(
            input_dim=54,
            latent_dim=16,
            hidden_dim=64,
            beta=2.0,
        )
        self.model = BetaVAE(
            input_dim=self.config.input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
            beta=self.config.beta,
        )
        self.model.eval()

        self.temp_dir = tempfile.mkdtemp()
        self.onnx_path = Path(self.temp_dir) / "affective_encoder.onnx"
        export_affective_vae_to_onnx(
            self.model,
            output_path=self.onnx_path,
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_onnx_model(self):
        """Should load ONNX model without errors."""
        encoder = AffectiveEncoderONNX(self.onnx_path)
        self.assertIsNotNone(encoder.session)

    def test_inference_shape(self):
        """Should produce correct output shape."""
        encoder = AffectiveEncoderONNX(self.onnx_path)
        input_data = np.random.randn(1, 54).astype(np.float32)

        affect = encoder.encode(input_data)

        self.assertEqual(affect.shape, (1, 16))

    def test_inference_batch_processing(self):
        """Should handle batch inference."""
        encoder = AffectiveEncoderONNX(self.onnx_path)
        input_data = np.random.randn(4, 54).astype(np.float32)

        affect = encoder.encode(input_data)

        self.assertEqual(affect.shape, (4, 16))

    def test_output_matches_pytorch(self):
        """Should produce same output as PyTorch model."""
        # Create test input
        input_data = np.random.randn(1, 54).astype(np.float32)
        input_tensor = torch.from_numpy(input_data)

        # PyTorch inference (deterministic)
        with torch.no_grad():
            pytorch_affect = self.model.encode_deterministic(input_tensor)
            pytorch_output = pytorch_affect.cpu().numpy()

        # ONNX inference
        encoder = AffectiveEncoderONNX(self.onnx_path)
        onnx_output = encoder.encode(input_data)

        # Should be very close (some numerical differences expected)
        np.testing.assert_allclose(onnx_output, pytorch_output, rtol=1e-3, atol=1e-5)

    def test_get_metadata(self):
        """Should return model metadata."""
        encoder = AffectiveEncoderONNX(self.onnx_path)
        metadata = encoder.get_metadata()

        self.assertIn("input_names", metadata)
        self.assertIn("output_names", metadata)
        self.assertIn("input_shapes", metadata)
        self.assertIn("output_shapes", metadata)

    def test_output_finite(self):
        """Output should be finite (no NaN or Inf)."""
        encoder = AffectiveEncoderONNX(self.onnx_path)
        input_data = np.random.randn(10, 54).astype(np.float32)

        affect = encoder.encode(input_data)

        self.assertTrue(np.all(np.isfinite(affect)))


class TestAffectiveEncoderIntegration(unittest.TestCase):
    """Integration tests for full export and inference workflow."""

    def test_full_pipeline(self):
        """Test export, load, and inference pipeline."""
        # Create model
        config = AffectVAEConfig(
            input_dim=54,
            latent_dim=16,
            hidden_dim=64,
            beta=2.0,
        )
        model = BetaVAE(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            beta=config.beta,
        )
        model.eval()

        # Export to ONNX
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "affective_encoder.onnx"
            export_affective_vae_to_onnx(model, output_path=onnx_path)

            # Verify export
            metadata = verify_onnx_model(onnx_path)
            self.assertEqual(metadata["input_shapes"], [["dynamic", 54]])

            # Load and infer
            encoder = AffectiveEncoderONNX(onnx_path)
            test_input = np.random.randn(1, 54).astype(np.float32)
            affect = encoder.encode(test_input)

            self.assertEqual(affect.shape, (1, 16))
            self.assertTrue(np.all(np.isfinite(affect)))

    def test_export_to_models_directory(self):
        """Should export to models/dual_stream/ directory."""
        # Create model
        config = AffectVAEConfig(
            input_dim=54,
            latent_dim=16,
            hidden_dim=64,
            beta=2.0,
        )
        model = BetaVAE(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            beta=config.beta,
        )
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models" / "dual_stream"
            models_dir.mkdir(parents=True)

            output_path = models_dir / "affective_encoder.onnx"
            export_affective_vae_to_onnx(
                model,
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
