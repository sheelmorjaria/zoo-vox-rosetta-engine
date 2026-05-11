#!/usr/bin/env python3
"""
ONNX Export Utilities for Predictive NBD (CPC) Models

Exports PyTorch models to ONNX format for Rust tract-onnx inference.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

import logging
import json
from pathlib import Path
from typing import Optional

import torch
import torch.onnx

from cpc_encoder import CPCEncoder, create_encoder
from cpc_ar_model import TCNARModel, LightweightARModel, create_ar_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CPCONNXExporter:
    """
    Export PyTorch CPC models to ONNX format.

    Features:
    - Dynamic batch size
    - Fixed frame size for audio
    - Simplified outputs for compatibility
    - Metadata embedding
    """

    def __init__(self, output_dir: Path = Path("models/onnx")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_encoder(
        self,
        model: CPCEncoder,
        output_path: Optional[Path] = None,
        sample_rate: int = 48000,
        frame_size_ms: float = 10.0,
        opset_version: int = 14,
    ) -> Path:
        """
        Export CPC encoder to ONNX.

        Args:
            model: CPC encoder model
            output_path: Output ONNX file path
            sample_rate: Sample rate in Hz
            frame_size_ms: Frame size in milliseconds
            opset_version: ONNX opset version

        Returns:
            Path to exported ONNX file
        """
        if output_path is None:
            output_path = self.output_dir / "cpc_encoder.onnx"

        # Set model to eval mode
        model.eval()

        # Example input
        frame_size = int(sample_rate * frame_size_ms / 1000)
        example_input = torch.randn(1, frame_size)  # (batch=1, frame_size)

        # Input names
        input_names = ["audio"]

        # Output names
        output_names = ["z"]

        # Dynamic axes (batch dimension)
        dynamic_axes = {
            "audio": {0: "batch_size"},
            "z": {0: "batch_size"},
        }

        logger.info(f"Exporting encoder to {output_path}")
        logger.info(f"  Input shape: {example_input.shape}")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Frame size: {frame_size} samples ({frame_size_ms}ms)")

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
        steps_ahead: int = 5,
        model_type: str = "tcn",
        opset_version: int = 14,
    ) -> Path:
        """
        Export AR model to ONNX.

        Note: ONNX doesn't support dynamic output lists well.
        We export a single-step prediction model that can be called repeatedly.

        Args:
            model: AR model
            output_path: Output ONNX file path
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            steps_ahead: Steps to predict (for metadata)
            model_type: Type of AR model
            opset_version: ONNX opset version

        Returns:
            Path to exported ONNX file
        """
        if output_path is None:
            output_path = self.output_dir / f"cpc_ar_{model_type}.onnx"

        # Set model to eval mode
        model.eval()

        # Example input: (batch, seq_len, input_dim)
        example_input = torch.randn(1, 1, input_dim)

        # Input names
        input_names = ["z"]

        # Output names
        output_names = ["prediction"]

        # Dynamic axes
        dynamic_axes = {
            "z": {0: "batch_size"},
            "prediction": {0: "batch_size"},
        }

        logger.info(f"Exporting AR model to {output_path}")
        logger.info(f"  Input shape: {example_input.shape}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Steps ahead: {steps_ahead}")

        # Wrap model to return single prediction
        class SingleStepWrapper(torch.nn.Module):
            def __init__(self, ar_model):
                super().__init__()
                self.ar_model = ar_model

            def forward(self, z):
                preds = self.ar_model(z, steps_ahead=1)
                return preds[0]

        wrapper = SingleStepWrapper(model)

        # Export
        torch.onnx.export(
            wrapper,
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
                "model_type": f"cpc_ar_{model_type}",
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
        encoder: CPCEncoder,
        ar_model: torch.nn.Module,
        output_path: Optional[Path] = None,
        sample_rate: int = 48000,
        frame_size_ms: float = 10.0,
        steps_ahead: int = 5,
        opset_version: int = 14,
    ) -> Path:
        """
        Export full CPC pipeline (encoder + AR) to single ONNX model.

        Args:
            encoder: CPC encoder
            ar_model: AR model
            output_path: Output ONNX file path
            sample_rate: Sample rate
            frame_size_ms: Frame size
            steps_ahead: Steps to predict
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
        example_input = torch.randn(1, frame_size)

        # Create full pipeline wrapper
        class FullPipeline(torch.nn.Module):
            def __init__(self, encoder, ar_model, steps_ahead):
                super().__init__()
                self.encoder = encoder
                self.ar_model = ar_model
                self.steps_ahead = steps_ahead

            def forward(self, audio):
                z = self.encoder(audio)
                preds = self.ar_model(z, self.steps_ahead)
                # Concatenate predictions along sequence dimension
                # Shape: (batch, steps_ahead, hidden_dim)
                return torch.cat(preds, dim=1)

        pipeline = FullPipeline(encoder, ar_model, steps_ahead)
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
            output_names=["predictions"],
            dynamic_axes={
                "audio": {0: "batch_size"},
                "predictions": {0: "batch_size"},
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
                "steps_ahead": steps_ahead,
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
    hidden_dim: int = 128,
    ar_hidden_dim: int = 64,
    ar_model_type: str = "lightweight",
):
    """
    Export all CPC models for Predictive NBD.

    Args:
        output_dir: Output directory
        hidden_dim: Encoder hidden dimension
        ar_hidden_dim: AR model hidden dimension
        ar_model_type: AR model type ("tcn" or "lightweight")
    """
    print("=" * 60)
    print("Exporting Predictive NBD (CPC) Models to ONNX")
    print("=" * 60)

    exporter = CPCONNXExporter(output_dir)

    # Create models
    encoder = create_encoder(hidden_dim=hidden_dim)
    ar_model = create_ar_model(
        input_dim=hidden_dim,
        hidden_dim=ar_hidden_dim,
        model_type=ar_model_type,
    )

    # Export encoder
    encoder_path = exporter.export_encoder(
        encoder,
        output_dir / "cpc_encoder.onnx",
    )

    # Export AR model
    ar_path = exporter.export_ar_model(
        ar_model,
        output_dir / f"cpc_ar_{ar_model_type}.onnx",
        model_type=ar_model_type,
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
    """Export all models with default parameters."""
    export_all_cpc_models(
        output_dir=Path("models/onnx"),
        hidden_dim=128,
        ar_hidden_dim=64,
        ar_model_type="lightweight",  # or "tcn"
    )


if __name__ == "__main__":
    main()
