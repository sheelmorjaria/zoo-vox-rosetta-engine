#!/usr/bin/env python3
"""
Tests for Syntactic VQ-VAE ONNX Export

Tests the ONNX export functionality for the VQ-VAE encoder,
ensuring compatibility with Rust ONNX Runtime inference.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cognitive_intelligence.syntactic_export import (
    SyntacticEncoderONNX,
    export_syntactic_vqvae_to_onnx,
    verify_onnx_model,
)
from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE, VQVAEConfig


class TestSyntacticEncoderONNX(unittest.TestCase):
    """Test ONNX export for syntactic encoder."""

    def setUp(self):
        """Create a test VQ-VAE model."""
        self.config = VQVAEConfig(
            input_dim=44,
            codebook_size=64,
            codebook_dim=32,
            hidden_dim=128,
        )
        self.model = SyntacticVQVAE(
            input_dim=self.config.input_dim,
            codebook_size=self.config.codebook_size,
            codebook_dim=self.config.codebook_dim,
            hidden_dim=self.config.hidden_dim,
            commitment_cost=self.config.commitment_cost,
            decay=self.config.decay,
        )
        self.model.eval()

    def test_export_creates_file(self):
        """Should create ONNX file at specified path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "syntactic_encoder.onnx"
            export_syntactic_vqvae_to_onnx(
                self.model,
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())

    def test_export_has_correct_inputs(self):
        """Should have single input node with shape (dynamic, 44)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "syntactic_encoder.onnx"
            export_syntactic_vqvae_to_onnx(
                self.model,
                output_path=output_path,
            )

            metadata = verify_onnx_model(output_path)
            self.assertEqual(metadata["input_names"], ["syntactic_features"])
            self.assertEqual(metadata["input_shapes"], [["dynamic", 44]])

    def test_export_has_correct_outputs(self):
        """Should have output node with shape (dynamic, 1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "syntactic_encoder.onnx"
            export_syntactic_vqvae_to_onnx(
                self.model,
                output_path=output_path,
            )

            metadata = verify_onnx_model(output_path)
            self.assertEqual(metadata["output_names"], ["token_id"])
            self.assertEqual(metadata["output_shapes"], [["dynamic", 1]])

    def test_opset_version(self):
        """Should use ONNX opset version 17 or higher."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "syntactic_encoder.onnx"
            export_syntactic_vqvae_to_onnx(
                self.model,
                output_path=output_path,
                opset_version=17,
            )

            metadata = verify_onnx_model(output_path)
            self.assertGreaterEqual(metadata["opset_version"], 17)


class TestSyntacticEncoderONNXClass(unittest.TestCase):
    """Test the SyntacticEncoderONNX wrapper class."""

    def setUp(self):
        """Create test model and ONNX export."""
        self.config = VQVAEConfig(
            input_dim=44,
            codebook_size=64,
            codebook_dim=32,
            hidden_dim=128,
        )
        self.model = SyntacticVQVAE(
            input_dim=self.config.input_dim,
            codebook_size=self.config.codebook_size,
            codebook_dim=self.config.codebook_dim,
            hidden_dim=self.config.hidden_dim,
            commitment_cost=self.config.commitment_cost,
            decay=self.config.decay,
        )
        self.model.eval()

        self.temp_dir = tempfile.mkdtemp()
        self.onnx_path = Path(self.temp_dir) / "syntactic_encoder.onnx"
        export_syntactic_vqvae_to_onnx(
            self.model,
            output_path=self.onnx_path,
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_onnx_model(self):
        """Should load ONNX model without errors."""
        encoder = SyntacticEncoderONNX(self.onnx_path)
        self.assertIsNotNone(encoder.session)

    def test_inference_shape(self):
        """Should produce correct output shape."""
        encoder = SyntacticEncoderONNX(self.onnx_path)
        input_data = np.random.randn(1, 44).astype(np.float32)

        token_id = encoder.tokenize(input_data)

        self.assertEqual(token_id.shape, (1,))

    def test_inference_batch_processing(self):
        """Should handle batch inference."""
        encoder = SyntacticEncoderONNX(self.onnx_path)
        input_data = np.random.randn(4, 44).astype(np.float32)

        token_ids = encoder.tokenize(input_data)

        self.assertEqual(token_ids.shape, (4,))

    def test_output_matches_pytorch(self):
        """Should produce same output as PyTorch model."""
        # Create test input
        input_data = np.random.randn(1, 44).astype(np.float32)
        input_tensor = torch.from_numpy(input_data)

        # PyTorch inference
        with torch.no_grad():
            pytorch_token = self.model.tokenize(input_tensor)
            pytorch_output = pytorch_token.cpu().numpy()

        # ONNX inference
        encoder = SyntacticEncoderONNX(self.onnx_path)
        onnx_output = encoder.tokenize(input_data)

        # Should be exactly the same (discrete tokens)
        np.testing.assert_array_equal(onnx_output, pytorch_output)

    def test_get_metadata(self):
        """Should return model metadata."""
        encoder = SyntacticEncoderONNX(self.onnx_path)
        metadata = encoder.get_metadata()

        self.assertIn("input_names", metadata)
        self.assertIn("output_names", metadata)
        self.assertIn("input_shapes", metadata)
        self.assertIn("output_shapes", metadata)

    def test_token_range(self):
        """Tokens should be within valid codebook range."""
        encoder = SyntacticEncoderONNX(self.onnx_path)
        input_data = np.random.randn(10, 44).astype(np.float32)

        token_ids = encoder.tokenize(input_data)

        # All tokens should be in [0, 64)
        self.assertTrue(np.all(token_ids >= 0))
        self.assertTrue(np.all(token_ids < 64))


class TestSyntacticEncoderIntegration(unittest.TestCase):
    """Integration tests for full export and inference workflow."""

    def test_full_pipeline(self):
        """Test export, load, and inference pipeline."""
        # Create model
        config = VQVAEConfig(
            input_dim=44,
            codebook_size=64,
            codebook_dim=32,
            hidden_dim=128,
        )
        model = SyntacticVQVAE(
            input_dim=config.input_dim,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            hidden_dim=config.hidden_dim,
            commitment_cost=config.commitment_cost,
            decay=config.decay,
        )
        model.eval()

        # Export to ONNX
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "syntactic_encoder.onnx"
            export_syntactic_vqvae_to_onnx(model, output_path=onnx_path)

            # Verify export
            metadata = verify_onnx_model(onnx_path)
            self.assertEqual(metadata["input_shapes"], [["dynamic", 44]])

            # Load and infer
            encoder = SyntacticEncoderONNX(onnx_path)
            test_input = np.random.randn(1, 44).astype(np.float32)
            token_id = encoder.tokenize(test_input)

            self.assertEqual(token_id.shape, (1,))
            self.assertGreaterEqual(token_id[0], 0)
            self.assertLess(token_id[0], 64)

    def test_export_to_models_directory(self):
        """Should export to models/dual_stream/ directory."""
        # Create model
        config = VQVAEConfig(
            input_dim=44,
            codebook_size=64,
            codebook_dim=32,
            hidden_dim=128,
        )
        model = SyntacticVQVAE(
            input_dim=config.input_dim,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            hidden_dim=config.hidden_dim,
            commitment_cost=config.commitment_cost,
            decay=config.decay,
        )
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models" / "dual_stream"
            models_dir.mkdir(parents=True)

            output_path = models_dir / "syntactic_encoder.onnx"
            export_syntactic_vqvae_to_onnx(
                model,
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
