#!/usr/bin/env python3
"""
ONNX Export for VQ-VAE Encoder (Risk A Mitigation)

Exports the VQ-VAE encoder to ONNX format for inference in the Rust
Execution Layer via ONNX Runtime.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.onnx

logger = logging.getLogger(__name__)


@dataclass
class SyntacticONNXExportConfig:
    """Configuration for ONNX export."""
    opset_version: int = 17
    batch_size: int = 1
    input_dim: int = 44
    codebook_size: int = 64
    verify: bool = True


class SyntacticEncoderWrapper(torch.nn.Module):
    """
    Wrapper for VQ-VAE encoder to export only tokenization path.

    Implements simplified quantization for ONNX export,
    avoiding side effects in the EMAVectorQuantizer.
    """

    def __init__(self, vqvae_model: torch.nn.Module):
        super().__init__()
        self.encoder = vqvae_model.encoder
        self.codebook = vqvae_model.vq.codebook_ema.clone()
        self.codebook_dim = vqvae_model.vq.codebook_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: input → encoder → quantize → token_id.

        Args:
            x: Input tensor of shape (batch, 44)

        Returns:
            token_id: Discrete token indices of shape (batch, 1)
        """
        # Encode to latent space
        z = self.encoder(x)

        # Compute distances to codebook vectors
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z^T * e
        z_norm = (z ** 2).sum(dim=1, keepdim=True)
        codebook_norm = (self.codebook ** 2).sum(dim=1)
        distances = z_norm + codebook_norm - 2 * torch.matmul(z, self.codebook.t())

        # Find nearest codebook entry
        token_ids = torch.argmin(distances, dim=1)

        # Return as (batch, 1) for ONNX compatibility
        return token_ids.unsqueeze(-1)


def export_syntactic_vqvae_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    config: Optional[SyntacticONNXExportConfig] = None,
    opset_version: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Path:
    """Export VQ-VAE encoder to ONNX format."""
    if config is None:
        config = SyntacticONNXExportConfig()

    if opset_version is not None:
        config.opset_version = opset_version
    if batch_size is not None:
        config.batch_size = batch_size

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrap model to export only encoder → token path
    encoder = SyntacticEncoderWrapper(model)
    encoder.eval()

    # Create dummy input for export
    dummy_input = torch.randn(config.batch_size, config.input_dim)

    # Input/output names
    input_names = ["syntactic_features"]
    output_names = ["token_id"]

    # Dynamic axes
    dynamic_axes = {
        "syntactic_features": {0: "batch_size"},
        "token_id": {0: "batch_size"},
    }

    logger.info(f"Exporting VQ-VAE encoder to ONNX: {output_path}")
    logger.info(f"  Input shape: ({config.batch_size}, {config.input_dim})")
    logger.info(f"  Output: token_id (discrete, 0-{config.codebook_size-1})")
    logger.info(f"  Opset version: {config.opset_version}")

    # Export to ONNX
    torch.onnx.export(
        encoder,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=config.opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Verify export if requested
    if config.verify:
        logger.info("Verifying ONNX export...")
        try:
            import onnx
            model_onnx = onnx.load(str(output_path))
            onnx.checker.check_model(model_onnx)
            logger.info("  Verification passed")
        except ImportError:
            logger.warning("  onnx package not installed, skipping verification")

    logger.info(f"✓ Exported VQ-VAE encoder to {output_path}")
    return output_path


def verify_onnx_model(onnx_path: Path) -> Dict:
    """Verify ONNX model and extract metadata."""
    try:
        import onnx
    except ImportError:
        raise ImportError("ONNX verification requires onnx package")

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    graph = model.graph
    input_names = [inp.name for inp in graph.input]
    output_names = [out.name for out in graph.output]

    input_shapes = []
    for inp in graph.input:
        shape = [d.dim_value if d.dim_value > 0 else "dynamic" for d in inp.type.tensor_type.shape.dim]
        input_shapes.append(shape)

    output_shapes = []
    for out in graph.output:
        shape = [d.dim_value if d.dim_value > 0 else "dynamic" for d in out.type.tensor_type.shape.dim]
        output_shapes.append(shape)

    return {
        "input_names": input_names,
        "output_names": output_names,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "opset_version": model.opset_import[0].version if model.opset_import else 0,
    }


class SyntacticEncoderONNX:
    """ONNX Runtime wrapper for syntactic encoder inference."""

    def __init__(
        self,
        onnx_path: Path,
        providers: Optional[List[str]] = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX Runtime requires onnxruntime package")

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=providers,
        )

        self._metadata = self._extract_metadata()
        logger.info(f"Loaded ONNX encoder from {onnx_path}")

    def _extract_metadata(self) -> Dict:
        """Extract model metadata from ONNX session."""
        input_info = self.session.get_inputs()
        output_info = self.session.get_outputs()

        return {
            "input_names": [inp.name for inp in input_info],
            "output_names": [out.name for out in output_info],
            "input_shapes": [inp.shape for inp in input_info],
            "output_shapes": [out.shape for out in output_info],
        }

    def tokenize(
        self,
        syntactic_features: np.ndarray,
    ) -> np.ndarray:
        """Tokenize syntactic features to discrete token."""
        if syntactic_features.ndim == 1:
            syntactic_features = syntactic_features.reshape(1, -1)

        inputs = {self._metadata["input_names"][0]: syntactic_features.astype(np.float32)}
        outputs = self.session.run(self._metadata["output_names"], inputs)

        # ONNX output shape is (batch, 1), squeeze to (batch,)
        token_ids = outputs[0].squeeze(-1)

        return token_ids

    def get_metadata(self) -> Dict:
        """Get model metadata."""
        return self._metadata.copy()


def main():
    """CLI for exporting VQ-VAE to ONNX."""
    import argparse

    parser = argparse.ArgumentParser(description="Export VQ-VAE encoder to ONNX format")
    parser.add_argument("--output", type=Path, default=Path("models/dual_stream/syntactic_encoder.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--no-verify", action="store_true")

    args = parser.parse_args()

    # Create model
    from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE, VQVAEConfig

    config = VQVAEConfig()
    model = SyntacticVQVAE(
        input_dim=config.input_dim,
        codebook_size=config.codebook_size,
        codebook_dim=config.codebook_dim,
        hidden_dim=config.hidden_dim,
        commitment_cost=config.commitment_cost,
        decay=config.decay,
    )
    model.eval()

    # Export
    export_config = SyntacticONNXExportConfig(
        opset_version=args.opset,
        batch_size=args.batch_size,
        verify=not args.no_verify,
    )

    output_path = export_syntactic_vqvae_to_onnx(model, output_path=args.output, config=export_config)
    logger.info(f"✓ Successfully exported to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
