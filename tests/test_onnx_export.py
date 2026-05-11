#!/usr/bin/env python3
"""
Tests for ONNX Export (Module 4 - Rust Deployment)

These tests verify that dual-stream models can be exported to ONNX
format for real-time inference in Rust.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from cognitive_intelligence.affective_vae import create_affective_vae
from cognitive_intelligence.export_onnx import ONNXExporter, export_all_dual_stream_models
from cognitive_intelligence.syntactic_vqvae import create_syntactic_vqvae


@unittest.skipIf(not ONNX_AVAILABLE, "ONNX Runtime not installed")
class TestONNXExporter(unittest.TestCase):
    """Test ONNX export functionality."""

    def test_exporter_initialization(self):
        """Should create output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ONNXExporter(tmpdir)
            self.assertTrue(Path(tmpdir).exists())

    def test_export_affective_vae_encoder(self):
        """Should export affective VAE encoder to ONNX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ONNXExporter(tmpdir)

            vae = create_affective_vae()
            vae.eval()

            onnx_path = exporter.export_affective_vae_encoder(vae, filename="test_affective.onnx")

            self.assertTrue(onnx_path.exists())
            self.assertTrue(onnx_path.stat().st_size > 0)

    def test_export_syntactic_vqvae_encoder(self):
        """Should export syntactic VQ-VAE encoder to ONNX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ONNXExporter(tmpdir)

            vqvae = create_syntactic_vqvae()
            vqvae.eval()

            onnx_path = exporter.export_syntactic_vqvae_encoder(vqvae, filename="test_syntactic.onnx")

            self.assertTrue(onnx_path.exists())
            self.assertTrue(onnx_path.stat().st_size > 0)

    def test_verify_onnx_affective_encoder(self):
        """Should verify ONNX output matches PyTorch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ONNXExporter(tmpdir)

            vae = create_affective_vae()
            vae.eval()

            onnx_path = exporter.export_affective_vae_encoder(vae, filename="test_verify.onnx")

            # Verify output matches
            dummy_input = torch.randn(1, 54)
            is_valid = exporter.verify_onnx_model(onnx_path, vae, dummy_input)

            self.assertTrue(is_valid)

    def test_verify_onnx_syntactic_encoder(self):
        """Should verify syntactic encoder ONNX output matches PyTorch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ONNXExporter(tmpdir)

            vqvae = create_syntactic_vqvae()
            vqvae.eval()

            onnx_path = exporter.export_syntactic_vqvae_encoder(vqvae, filename="test_verify.onnx")

            # Verify output matches
            dummy_input = torch.randn(1, 44)
            is_valid = exporter.verify_onnx_model(onnx_path, vqvae, dummy_input)

            self.assertTrue(is_valid)


@unittest.skipIf(not ONNX_AVAILABLE, "ONNX Runtime not installed")
class TestONNXInference(unittest.TestCase):
    """Test ONNX model inference."""

    def test_affective_encoder_onnx_inference(self):
        """Should run affective encoder inference via ONNX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ONNXExporter(tmpdir)

            vae = create_affective_vae()
            vae.eval()
            onnx_path = exporter.export_affective_vae_encoder(vae)

            # Create ONNX session
            session = ort.InferenceSession(str(onnx_path))

            # Run inference
            input_name = session.get_inputs()[0].name
            affective_features = np.random.randn(1, 54).astype(np.float32)

            outputs = session.run(None, {input_name: affective_features})

            # Check output shape
            affect_vector = outputs[0]
            self.assertEqual(affect_vector.shape, (1, 16))

    def test_syntactic_encoder_onnx_inference(self):
        """Should run syntactic encoder inference via ONNX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ONNXExporter(tmpdir)

            vqvae = create_syntactic_vqvae()
            vqvae.eval()
            onnx_path = exporter.export_syntactic_vqvae_encoder(vqvae)

            # Create ONNX session
            session = ort.InferenceSession(str(onnx_path))

            # Run inference
            input_name = session.get_inputs()[0].name
            syntactic_features = np.random.randn(1, 44).astype(np.float32)

            outputs = session.run(None, {input_name: syntactic_features})

            # Check output shape (token_id as float for ONNX compatibility)
            token_id = outputs[0]
            self.assertEqual(token_id.shape, (1,))


@unittest.skipIf(not ONNX_AVAILABLE, "ONNX Runtime not installed")
class TestExportAllModels(unittest.TestCase):
    """Test batch export of all models."""

    def test_export_all_dual_stream_models(self):
        """Should export all dual-stream models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = export_all_dual_stream_models(output_dir=tmpdir)

            self.assertIn("affective_encoder", results)
            self.assertIn("syntactic_encoder", results)

            # Check files exist
            for name, path in results.items():
                self.assertTrue(path.exists(), f"{name} should exist")
                self.assertTrue(path.stat().st_size > 0, f"{name} should have content")


@unittest.skipIf(ONNX_AVAILABLE, "ONNX Runtime is installed - skip placeholder test")
class TestONNXNotAvailable(unittest.TestCase):
    """Test graceful handling when ONNX is not available."""

    def test_onnx_not_available(self):
        """Should indicate ONNX is not installed."""
        self.assertFalse(ONNX_AVAILABLE)


if __name__ == "__main__":
    unittest.main()
