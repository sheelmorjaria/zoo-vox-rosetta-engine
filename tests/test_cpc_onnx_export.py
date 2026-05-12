#!/usr/bin/env python3
"""
Tests for CPC ONNX Export (Module 1: Rust Edge Encoding ONNX Integration)

Tests verify that CPC models for Predictive NBD can be exported to ONNX
format and loaded by Rust tract-onnx runtime for edge deployment.

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

from boundary_detection.cpc_encoder import (
    CPCEncoder,
    LightweightCPCEncoder,
    EncoderConfig,
    create_encoder,
)
from boundary_detection.cpc_autoregressive import create_autoregressive, TCNAutoregressive
from boundary_detection.cpc_onnx_exporter import (
    CPCONNXExporter,
    export_all_cpc_models,
)


class TestCPCONNXExporter(unittest.TestCase):
    """Test ONNX export functionality for CPC models."""

    def test_exporter_initialization(self):
        """Should create output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CPCONNXExporter(tmpdir)
            self.assertTrue(Path(tmpdir).exists())

    def test_export_lightweight_encoder(self):
        """Should export lightweight CPC encoder to ONNX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CPCONNXExporter(tmpdir)

            encoder = LightweightCPCEncoder(hidden_dim=64)
            encoder.eval()

            onnx_path = exporter.export_encoder(
                encoder,
                output_path=Path(tmpdir) / "test_encoder.onnx",
            )

            self.assertTrue(onnx_path.exists())
            self.assertTrue(onnx_path.stat().st_size > 0)

    def test_export_full_encoder(self):
        """Should export full CPC encoder to ONNX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CPCONNXExporter(tmpdir)

            encoder = CPCEncoder(hidden_dim=128)
            encoder.eval()

            onnx_path = exporter.export_encoder(
                encoder,
                output_path=Path(tmpdir) / "test_encoder.onnx",
            )

            self.assertTrue(onnx_path.exists())
            self.assertTrue(onnx_path.stat().st_size > 0)

    def test_export_creates_metadata(self):
        """Should create metadata JSON file alongside ONNX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CPCONNXExporter(tmpdir)

            encoder = LightweightCPCEncoder(hidden_dim=64)
            encoder.eval()

            onnx_path = exporter.export_encoder(
                encoder,
                output_path=Path(tmpdir) / "test_encoder.onnx",
            )

            metadata_path = onnx_path.with_suffix(".json")
            self.assertTrue(metadata_path.exists())

            # Check metadata contents
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)

            self.assertEqual(metadata["model_type"], "cpc_encoder")
            self.assertEqual(metadata["sample_rate"], 48000)
            self.assertEqual(metadata["hidden_dim"], 64)

    def test_export_ar_model(self):
        """Should export AR model to ONNX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CPCONNXExporter(tmpdir)

            ar_model = create_autoregressive(d_model=64, num_layers=2)
            ar_model.eval()

            onnx_path = exporter.export_ar_model(
                ar_model,
                output_path=Path(tmpdir) / "test_ar.onnx",
                input_dim=64,
                hidden_dim=64,
            )

            self.assertTrue(onnx_path.exists())
            self.assertTrue(onnx_path.stat().st_size > 0)

    def test_export_full_pipeline(self):
        """Should export full CPC pipeline to ONNX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CPCONNXExporter(tmpdir)

            encoder = LightweightCPCEncoder(hidden_dim=64)
            ar_model = create_autoregressive(d_model=64, num_layers=2)

            encoder.eval()
            ar_model.eval()

            onnx_path = exporter.export_full_pipeline(
                encoder,
                ar_model,
                output_path=Path(tmpdir) / "test_full.onnx",
            )

            self.assertTrue(onnx_path.exists())
            self.assertTrue(onnx_path.stat().st_size > 0)


@unittest.skipIf(not ONNX_AVAILABLE, "ONNX Runtime not installed")
class TestCPCONNXInference(unittest.TestCase):
    """Test ONNX model inference for CPC models."""

    def test_encoder_onnx_inference(self):
        """Should run encoder inference via ONNX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CPCONNXExporter(tmpdir)

            encoder = LightweightCPCEncoder(hidden_dim=64)
            encoder.eval()

            onnx_path = exporter.export_encoder(
                encoder,
                output_path=Path(tmpdir) / "test_encoder.onnx",
            )

            # Create ONNX session
            session = ort.InferenceSession(str(onnx_path))

            # Run inference
            input_name = session.get_inputs()[0].name
            audio = np.random.randn(1, 1, 480).astype(np.float32)  # 10ms @ 48kHz

            outputs = session.run(None, {input_name: audio})

            # Check output shape
            z = outputs[0]
            self.assertEqual(z.shape[0], 1)  # batch
            self.assertEqual(z.shape[2], 64)  # hidden_dim

    def test_ar_onnx_inference(self):
        """Should run AR model inference via ONNX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CPCONNXExporter(tmpdir)

            ar_model = create_autoregressive(d_model=64, num_layers=2)
            ar_model.eval()

            onnx_path = exporter.export_ar_model(
                ar_model,
                output_path=Path(tmpdir) / "test_ar.onnx",
                input_dim=64,
                hidden_dim=64,
            )

            # Create ONNX session
            session = ort.InferenceSession(str(onnx_path))

            # Run inference
            input_name = session.get_inputs()[0].name
            z = np.random.randn(1, 1, 64).astype(np.float32)

            outputs = session.run(None, {input_name: z})

            # Check output shape
            prediction = outputs[0]
            self.assertEqual(prediction.shape, (1, 1, 64))

    def test_full_pipeline_onnx_inference(self):
        """Should run full pipeline inference via ONNX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CPCONNXExporter(tmpdir)

            encoder = LightweightCPCEncoder(hidden_dim=64)
            ar_model = create_autoregressive(d_model=64, num_layers=2)

            encoder.eval()
            ar_model.eval()

            onnx_path = exporter.export_full_pipeline(
                encoder,
                ar_model,
                output_path=Path(tmpdir) / "test_full.onnx",
            )

            # Create ONNX session
            session = ort.InferenceSession(str(onnx_path))

            # Run inference
            input_name = session.get_inputs()[0].name
            audio = np.random.randn(1, 1, 480).astype(np.float32)

            outputs = session.run(None, {input_name: audio})

            # Check outputs
            prediction, z = outputs
            self.assertEqual(prediction.shape[0], 1)
            self.assertEqual(z.shape[0], 1)

    def test_onnx_output_matches_pytorch(self):
        """Should verify ONNX output matches PyTorch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CPCONNXExporter(tmpdir)

            encoder = LightweightCPCEncoder(hidden_dim=64)
            encoder.eval()

            onnx_path = exporter.export_encoder(
                encoder,
                output_path=Path(tmpdir) / "test_verify.onnx",
            )

            # Create test input
            audio = torch.randn(1, 1, 480)

            # PyTorch output
            with torch.no_grad():
                z_pytorch = encoder(audio).numpy()

            # ONNX output
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            z_onnx = session.run(None, {input_name: audio.numpy()})[0]

            # Check shapes match
            self.assertEqual(z_pytorch.shape, z_onnx.shape)

            # Check values are close (may differ due to ONNX conversion)
            np.testing.assert_allclose(z_pytorch, z_onnx, rtol=1e-3, atol=1e-5)


@unittest.skipIf(not ONNX_AVAILABLE, "ONNX Runtime not installed")
class TestExportAllModels(unittest.TestCase):
    """Test batch export of all CPC models."""

    def test_export_all_cpc_models(self):
        """Should export all CPC models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder_path, ar_path, full_path = export_all_cpc_models(
                output_dir=Path(tmpdir),
                lightweight=True,
                hidden_dim=64,
                ar_hidden_dim=64,  # Changed to match d_model
            )

            # Check files exist
            self.assertTrue(encoder_path.exists())
            self.assertTrue(ar_path.exists())
            self.assertTrue(full_path.exists())

            # Check files have content
            self.assertGreater(encoder_path.stat().st_size, 0)
            self.assertGreater(ar_path.stat().st_size, 0)
            self.assertGreater(full_path.stat().st_size, 0)


@unittest.skipIf(ONNX_AVAILABLE, "ONNX Runtime is installed - skip placeholder test")
class TestONNXNotAvailable(unittest.TestCase):
    """Test graceful handling when ONNX is not available."""

    def test_onnx_not_available(self):
        """Should indicate ONNX is not installed."""
        self.assertFalse(ONNX_AVAILABLE)


class TestEncoderConfigCompatibility(unittest.TestCase):
    """Test encoder configuration for ONNX export compatibility."""

    def test_encoder_config_frame_size(self):
        """Should calculate correct frame size."""
        config = EncoderConfig(sample_rate=48000, frame_size_ms=10)
        self.assertEqual(config.frame_size_samples, 480)

    def test_lightweight_encoder_parameter_count(self):
        """Lightweight encoder should have fewer parameters."""
        lightweight = LightweightCPCEncoder(hidden_dim=64)
        full = CPCEncoder(hidden_dim=128)

        # Count parameters manually
        lightweight_params = sum(p.numel() for p in lightweight.parameters())
        full_params = sum(p.numel() for p in full.parameters())

        # Lightweight should have significantly fewer parameters
        self.assertLess(
            lightweight_params,
            full_params,
            "Lightweight encoder should have fewer parameters"
        )


if __name__ == "__main__":
    unittest.main()
