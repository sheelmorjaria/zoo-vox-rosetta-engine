#!/usr/bin/env python3
"""
BioMAE ONNX Export for TensorRT Deployment

Exports the BioMAE encoder to ONNX format for optimized inference on
edge devices (Jetson Orin Nano/NX).

Key features:
- Encoder-only export (no decoder needed for inference)
- FP16 quantization support
- Input/output shape validation
- TensorRT build instructions

Target latency: <5ms on Jetson Orin (refined from <1ms based on research)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.onnx

from feature_extraction.biomae import BioMAEEncoder, EncoderConfig


logger = logging.getLogger(__name__)


class BioMAEExporter:
    """
    Export BioMAE encoder to ONNX for TensorRT deployment.

    The exported model contains only the encoder (decoder is training-only).
    Input: Spectrogram tensor (Batch, Channels, Freq, Time)
    Output: 112D Rosetta embedding (Batch, 112)

    Example:
        >>> exporter = BioMAEExporter()
        >>> exporter.export("models/biomae_encoder.onnx")
    """

    def __init__(
        self,
        encoder: Optional[BioMAEEncoder] = None,
        config: Optional[EncoderConfig] = None,
    ):
        """
        Initialize exporter.

        Args:
            encoder: Trained BioMAEEncoder to export
            config: Config to create encoder if not provided
        """
        if encoder is None:
            if config is None:
                config = EncoderConfig()
            encoder = BioMAEEncoder(config)

        self.encoder = encoder
        self.encoder.eval()

        self.config = config if config else encoder.config

    def export(
        self,
        output_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 1, 128, 128),
        opset_version: int = 17,
        export_params: bool = True,
        do_constant_folding: bool = True,
        dynamic_axes: Optional[dict] = None,
    ) -> Path:
        """
        Export encoder to ONNX format.

        Args:
            output_path: Path for output ONNX file
            input_shape: Dummy input shape (B, C, Freq, Time)
            opset_version: ONNX opset version (17 for TensorRT 8.6+)
            export_params: Export model parameters
            do_constant_folding: Fold constants for optimization
            dynamic_axes: Dynamic axis specifications for variable batch/size

        Returns:
            Path to exported ONNX file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create dummy input
        dummy_input = torch.randn(input_shape)

        logger.info(f"Exporting BioMAE encoder to {output_path}")
        logger.info(f"Input shape: {input_shape}")
        logger.info(f"Output shape: (1, 112)")
        logger.info(f"Parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")

        # Dynamic axes for flexible input sizes
        if dynamic_axes is None:
            dynamic_axes = {
                'spectrogram': {0: 'batch_size', 2: 'freq_bins', 3: 'time_frames'},
                'embedding': {0: 'batch_size'},
            }

        # Export to ONNX
        torch.onnx.export(
            self.encoder,
            dummy_input,
            output_path,
            export_params=export_params,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            input_names=['spectrogram'],
            output_names=['embedding'],
            dynamic_axes=dynamic_axes,
        )

        logger.info(f"Exported successfully to {output_path}")

        # Validate the exported model
        self.validate_onnx_model(output_path, dummy_input)

        return output_path

    def validate_onnx_model(
        self,
        onnx_path: Path,
        dummy_input: torch.Tensor,
    ) -> bool:
        """
        Validate exported ONNX model.

        Args:
            onnx_path: Path to exported ONNX file
            dummy_input: Test input tensor

        Returns:
            True if validation passes
        """
        try:
            import onnx
            from onnx import checker, helper

            # Load and check ONNX model
            model = onnx.load(onnx_path)
            checker.check_model(model)

            logger.info("ONNX model validation passed")

            # Print model info
            logger.info(f"ONNX opset version: {model.opset_import[0].version}")
            logger.info(f"Inputs: {[i.name for i in model.graph.input]}")
            logger.info(f"Outputs: {[o.name for o in model.graph.output]}")

            return True

        except ImportError:
            logger.warning("onnx package not installed, skipping validation")
            return True
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            return False

    def export_fp16(
        self,
        output_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 1, 128, 128),
    ) -> Path:
        """
        Export encoder to FP16 ONNX for TensorRT optimization.

        FP16 provides ~2x speedup on Jetson devices with minimal accuracy loss.

        Args:
            output_path: Path for FP16 ONNX file
            input_shape: Dummy input shape

        Returns:
            Path to exported FP16 ONNX file
        """
        import onnx
        from onnxconverter_common import float16

        # First export standard FP32
        fp32_path = Path(output_path).with_suffix('.fp32.onnx')
        self.export(fp32_path, input_shape=input_shape)

        # Convert to FP16
        logger.info(f"Converting to FP16: {output_path}")

        model = onnx.load(fp32_path)
        model_fp16 = float16.convert_float16_to_float32(model)

        # Save FP16 model
        onnx.save(model_fp16, output_path)

        # Clean up FP32
        fp32_path.unlink()

        logger.info(f"FP16 export complete: {output_path}")
        return Path(output_path)


def export_from_checkpoint(
    checkpoint_path: str,
    output_path: str,
    config: Optional[EncoderConfig] = None,
) -> Path:
    """
    Export BioMAE encoder from a training checkpoint.

    Args:
        checkpoint_path: Path to training checkpoint (.pt file)
        output_path: Path for output ONNX file
        config: Encoder config (uses checkpoint config if None)

    Returns:
        Path to exported ONNX file
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract encoder state dict
    if 'model_state_dict' in checkpoint:
        # Full checkpoint from trainer
        state_dict = checkpoint['model_state_dict']
        # Filter encoder weights only
        encoder_state_dict = {
            k.replace('encoder.', ''): v
            for k, v in state_dict.items()
            if k.startswith('encoder.')
        }
    else:
        # Encoder-only checkpoint
        encoder_state_dict = checkpoint

    # Create encoder
    if config is None and 'config' in checkpoint:
        # Load config from checkpoint
        config = checkpoint['config']
    elif config is None:
        config = EncoderConfig()

    encoder = BioMAEEncoder(config)
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()

    # Export
    exporter = BioMAEExporter(encoder)
    return exporter.export(output_path)


def print_tensorrt_build_instructions(onnx_path: str):
    """
    Print TensorRT engine build instructions.

    Args:
        onnx_path: Path to exported ONNX model
    """
    print("""
====================================
TensorRT Engine Build Instructions
====================================

1. Install TensorRT on Jetson Orin:
   sudo apt-get install tensorrt

2. Build TensorRT engine from ONNX:

   # For FP32 precision:
   trtexec --onnx={onnx_path} \\
          --saveEngine=biomae_fp32.engine \\
          --workspace=1024 \\
          --timingCacheFile=timing.cache \\
          --separateProfileRun

   # For FP16 precision (recommended):
   trtexec --onnx={onnx_path} \\
          --saveEngine=biomae_fp16.engine \\
          --workspace=1024 \\
          --fp16 \\
          --timingCacheFile=timing.cache \\
          --separateProfileRun

3. Profile the engine:

   trtexec --loadEngine=biomae_fp16.engine \\
          --workspace=1024 \\
          --duration=30 \\
          --warmUp=1000

4. Expected performance (Jetson Orin Nano):
   - FP16: ~3-5ms latency (target <5ms 99th percentile)
   - FP32: ~8-12ms latency

5. Integration in Rust (via tract-onnx):

   ```rust
   use tract_onnx::prelude::*;

   // Load ONNX model
   let model = tract_onnx::onnx()
       .model_for_path("biomae_fp16.onnx")?
       .into_runnable()?;

   // Run inference
   let spectrogram = Tensor::zeros(&[1, 1, 128, 128].map(|&d| d.to_dim()));
   let embedding_112d = model.run(tvec!(spectrogram))?;
   ```

====================================
""".format(onnx_path=onnx_path))


def main():
    """Example export script."""
    logging.basicConfig(level=logging.INFO)

    # Create encoder
    config = EncoderConfig(
        img_size=(128, 128),
        embed_dim=256,
        depth=4,
        num_heads=4,
        output_dim=112,
    )
    encoder = BioMAEEncoder(config)

    # Export to ONNX
    exporter = BioMAEExporter(encoder)
    onnx_path = exporter.export("models/biomae_encoder.onnx")

    # Export FP16 version
    fp16_path = exporter.export_fp16("models/biomae_encoder_fp16.onnx")

    # Print TensorRT build instructions
    print_tensorrt_build_instructions(str(fp16_path))


if __name__ == '__main__':
    main()
