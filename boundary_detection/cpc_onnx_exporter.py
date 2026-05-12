#!/usr/bin/env python3
"""
ONNX Exporter for CPC Models (Predictive NBD)

Exports the CPC encoder and AR models to ONNX format for Rust tract-onnx inference.
This enables edge deployment on Jetson devices with TensorRT acceleration.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.onnx

from boundary_detection.cpc_encoder import CPCEncoder, LightweightCPCEncoder, EncoderConfig, create_encoder
from boundary_detection.cpc_autoregressive import create_autoregressive, TCNAutoregressive

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CPCONNXExporter:
    """
    Export PyTorch CPC models to ONNX format for Rust inference.

    Features:
    - Dynamic batch size support
    - Fixed audio frame size for streaming
    - TensorRT-compatible opset (17)
    - Metadata embedding for configuration
    """

    def __init__(self, output_dir: Path = Path("models/onnx")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_encoder(
        self,
        model: torch.nn.Module,
        output_path: Optional[Path] = None,
        sample_rate: int = 48000,
        frame_size_ms: float = 10.0,
        opset_version: int = 17,
    ) -> Path:
        """
        Export CPC encoder to ONNX.

        Args:
            model: CPC encoder model (CPCEncoder or LightweightCPCEncoder)
            output_path: Output ONNX file path
            sample_rate: Sample rate in Hz
            frame_size_ms: Frame size in milliseconds
            opset_version: ONNX opset version (17 for TensorRT 8.6+)

        Returns:
            Path to exported ONNX file
        """
        if output_path is None:
            output_path = self.output_dir / "cpc_encoder.onnx"

        # Set model to eval mode
        model.eval()

        # Example input: (batch, channels, samples)
        frame_size = int(sample_rate * frame_size_ms / 1000)
        example_input = torch.randn(1, 1, frame_size)  # (batch=1, channels=1, frame_size)

        # Input/output names
        input_names = ["audio"]
        output_names = ["z"]

        # Dynamic axes (batch dimension only)
        dynamic_axes = {
            "audio": {0: "batch_size"},
            "z": {0: "batch_size"},
        }

        logger.info(f"Exporting encoder to {output_path}")
        logger.info(f"  Input shape: {example_input.shape}")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Frame size: {frame_size} samples ({frame_size_ms}ms)")
        logger.info(f"  Hidden dim: {model.hidden_dim}")

        # Export
        torch.onnx.export(
            model,
            example_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        # Save metadata
        self._save_metadata(
            output_path.with_suffix(".json"),
            {
                "model_type": "cpc_encoder",
                "sample_rate": sample_rate,
                "frame_size_ms": frame_size_ms,
                "frame_size": frame_size,
                "hidden_dim": model.hidden_dim,
                "opset_version": opset_version,
            },
        )

        logger.info(f"✓ Encoder exported to {output_path}")
        return output_path

    def export_ar_model(
        self,
        model: torch.nn.Module,
        output_path: Optional[Path] = None,
        input_dim: int = 128,
        hidden_dim: int = 64,
        steps_ahead: int = 1,
        opset_version: int = 17,
    ) -> Path:
        """
        Export autoregressive model to ONNX.

        Note: We export a single-step prediction model for tract-onnx compatibility.
        The Rust code can call this multiple times for multi-step prediction.

        Args:
            model: AR model (LightweightARModel or similar)
            output_path: Output ONNX file path
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            steps_ahead: Number of steps to predict (for metadata only)
            opset_version: ONNX opset version

        Returns:
            Path to exported ONNX file
        """
        if output_path is None:
            output_path = self.output_dir / "cpc_ar.onnx"

        # Set model to eval mode
        model.eval()

        # Example input: (batch, seq_len, input_dim)
        example_input = torch.randn(1, 1, input_dim)

        # Input/output names
        input_names = ["z"]
        output_names = ["prediction"]

        # Dynamic axes
        dynamic_axes = {
            "z": {0: "batch_size"},
            "prediction": {0: "batch_size"},
        }

        logger.info(f"Exporting AR model to {output_path}")
        logger.info(f"  Input shape: {example_input.shape}")
        logger.info(f"  Hidden dim: {hidden_dim}")

        # Export
        torch.onnx.export(
            model,
            example_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        # Save metadata
        self._save_metadata(
            output_path.with_suffix(".json"),
            {
                "model_type": "cpc_ar",
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "steps_ahead": steps_ahead,
                "opset_version": opset_version,
            },
        )

        logger.info(f"✓ AR model exported to {output_path}")
        return output_path

    def export_full_pipeline(
        self,
        encoder: torch.nn.Module,
        ar_model: torch.nn.Module,
        output_path: Optional[Path] = None,
        sample_rate: int = 48000,
        frame_size_ms: float = 10.0,
        opset_version: int = 17,
    ) -> Path:
        """
        Export full CPC pipeline (encoder + AR) to single ONNX model.

        This combines both models for simpler deployment.

        Args:
            encoder: CPC encoder
            ar_model: AR model
            output_path: Output ONNX file path
            sample_rate: Sample rate
            frame_size_ms: Frame size
            opset_version: ONNX opset version

        Returns:
            Path to exported ONNX file
        """
        if output_path is None:
            output_path = self.output_dir / "cpc_full.onnx"

        # Set models to eval mode
        encoder.eval()
        ar_model.eval()

        # Example input
        frame_size = int(sample_rate * frame_size_ms / 1000)
        example_input = torch.randn(1, 1, frame_size)

        # Create full pipeline wrapper
        class FullPipeline(torch.nn.Module):
            def __init__(self, enc, ar):
                super().__init__()
                self.enc = enc
                self.ar = ar

            def forward(self, audio):
                # Encode: (B, 1, T) -> (B, T', hidden)
                z = self.enc(audio)
                # Take mean over time for single vector
                z = z.mean(dim=1, keepdim=True)  # (B, 1, hidden)
                # Predict: (B, 1, hidden) -> (B, 1, hidden)
                pred = self.ar(z)
                return pred, z

        pipeline = FullPipeline(encoder, ar_model)
        pipeline.eval()

        logger.info(f"Exporting full pipeline to {output_path}")

        # Export
        torch.onnx.export(
            pipeline,
            example_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["audio"],
            output_names=["prediction", "z"],
            dynamic_axes={
                "audio": {0: "batch_size"},
                "prediction": {0: "batch_size"},
                "z": {0: "batch_size"},
            },
        )

        # Save metadata
        self._save_metadata(
            output_path.with_suffix(".json"),
            {
                "model_type": "cpc_full_pipeline",
                "sample_rate": sample_rate,
                "frame_size_ms": frame_size_ms,
                "hidden_dim": encoder.hidden_dim,
                "opset_version": opset_version,
            },
        )

        logger.info(f"✓ Full pipeline exported to {output_path}")
        return output_path

    def _save_metadata(self, path: Path, metadata: dict):
        """Save model metadata to JSON file."""
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  Metadata saved to {path}")


def export_all_cpc_models(
    output_dir: Path = Path("models/onnx"),
    lightweight: bool = True,
    hidden_dim: int = 64 if True else 128,
    ar_hidden_dim: int = 64,
) -> Tuple[Path, Path, Path]:
    """
    Export all CPC models for Predictive NBD.

    Args:
        output_dir: Output directory
        lightweight: If True, use LightweightCPCEncoder for edge deployment
        hidden_dim: Encoder hidden dimension
        ar_hidden_dim: AR model hidden dimension (should match hidden_dim for TCN)

    Returns:
        Tuple of (encoder_path, ar_path, full_path)
    """
    print("=" * 60)
    print("Exporting Predictive NBD (CPC) Models to ONNX")
    print("=" * 60)

    exporter = CPCONNXExporter(output_dir)

    # Create encoder config
    encoder_config = EncoderConfig(
        sample_rate=48000,
        frame_size_ms=10,
        hidden_dim=hidden_dim,
    )

    # Create encoder
    encoder = create_encoder(encoder_config, lightweight=lightweight)

    # Create AR model (TCN uses d_model for both input and output)
    ar_model = create_autoregressive(
        d_model=ar_hidden_dim,
        num_layers=2,
    )

    # Export encoder
    encoder_path = exporter.export_encoder(
        encoder,
        output_dir / "cpc_encoder.onnx",
    )

    # Export AR model
    ar_path = exporter.export_ar_model(
        ar_model,
        output_dir / "cpc_ar.onnx",
        input_dim=ar_hidden_dim,
        hidden_dim=ar_hidden_dim,
    )

    # Export full pipeline
    full_path = exporter.export_full_pipeline(
        encoder,
        ar_model,
        output_dir / "cpc_full.onnx",
    )

    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    print(f"Encoder:        {encoder_path}")
    print(f"AR Model:       {ar_path}")
    print(f"Full Pipeline:  {full_path}")
    print(f"\nAll models exported successfully!")
    print(f"\nFor Rust integration, copy the .onnx files to:")
    print(f"  technical_architecture/models/")

    return encoder_path, ar_path, full_path


def main():
    """Export all models with default parameters (lightweight for edge deployment)."""
    export_all_cpc_models(
        output_dir=Path("models/onnx"),
        lightweight=True,  # Use lightweight encoder for Jetson edge deployment
        hidden_dim=64,
        ar_hidden_dim=64,  # Should match hidden_dim for TCN compatibility
    )


if __name__ == "__main__":
    main()
