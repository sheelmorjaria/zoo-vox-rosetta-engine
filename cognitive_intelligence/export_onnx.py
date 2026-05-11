#!/usr/bin/env python3
"""
ONNX Export Utilities for Dual-Stream Models

This module provides utilities to export the dual-stream neural networks
to ONNX format for real-time inference in Rust via ONNX Runtime or TensorRT.

Module 1 (v1.6.0): Added export for affective VAE and syntactic VQ-VAE.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Import dual-stream models
from cognitive_intelligence.affective_encoder import AffectiveFeatureExtractor
from cognitive_intelligence.affective_vae import BetaVAE, create_affective_vae
from cognitive_intelligence.syntactic_encoder import SyntacticFeatureExtractor
from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE, create_syntactic_vqvae

logger = logging.getLogger(__name__)


class ONNXExporter:
    """
    Export PyTorch models to ONNX format for Rust deployment.

    ONNX enables:
    - Cross-platform inference (Rust, C++, Python)
    - Hardware acceleration (TensorRT, ONNX Runtime)
    - Sub-5ms latency for 16D encoding
    """

    def __init__(self, output_dir: str = "models/onnx"):
        """
        Initialize ONNX exporter.

        Args:
            output_dir: Directory to save ONNX models
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_affective_vae_encoder(
        self,
        vae: BetaVAE,
        filename: str = "affective_encoder.onnx",
        opset_version: int = 14,
    ) -> Path:
        """
        Export affective VAE encoder to ONNX.

        Exports only the encoder (54D → 16D) for Rust-side inference.
        The decoder remains in Python for synthesis.

        Args:
            vae: Trained BetaVAE model
            filename: Output filename
            opset_version: ONNX opset version (14 recommended for TensorRT)

        Returns:
            Path to exported ONNX model
        """
        output_path = self.output_dir / filename

        # Create encoder-only model wrapper
        class EncoderWrapper(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae

            def forward(self, x):
                mu, logvar = self.vae.encode(x)
                # Return mu only (16D latent mean)
                return mu

        encoder = EncoderWrapper(vae)
        encoder.eval()

        # Example input (batch_size=1, features=54)
        dummy_input = torch.randn(1, 54)

        # Export to ONNX
        torch.onnx.export(
            encoder,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["affective_features"],
            output_names=["affect_vector"],
            dynamic_axes={
                "affective_features": {0: "batch_size"},
                "affect_vector": {0: "batch_size"},
            },
        )

        logger.info(f"Exported affective VAE encoder to {output_path}")
        return output_path

    def export_syntactic_vqvae_encoder(
        self,
        vqvae: SyntacticVQVAE,
        filename: str = "syntactic_encoder.onnx",
        opset_version: int = 14,
    ) -> Path:
        """
        Export syntactic VQ-VAE encoder to ONNX.

        Exports only the encoder + tokenizer (44D → token_id) for Rust-side inference.

        Args:
            vqvae: Trained SyntacticVQVAE model
            filename: Output filename
            opset_version: ONNX opset version

        Returns:
            Path to exported ONNX model
        """
        output_path = self.output_dir / filename

        # Create tokenizer wrapper
        class TokenizerWrapper(torch.nn.Module):
            def __init__(self, vqvae):
                super().__init__()
                self.vqvae = vqvae

            def forward(self, x):
                # Encode to latent space
                z = self.vqvae.encoder(x)
                # Quantize to get token
                z_q, token_ids, perplexity = self.vqvae.vq(z)
                return token_ids.float()  # Return as float for ONNX compatibility

        tokenizer = TokenizerWrapper(vqvae)
        tokenizer.eval()

        # Example input (batch_size=1, features=44)
        dummy_input = torch.randn(1, 44)

        # Export to ONNX
        torch.onnx.export(
            tokenizer,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["syntactic_features"],
            output_names=["token_id"],
            dynamic_axes={
                "syntactic_features": {0: "batch_size"},
                "token_id": {0: "batch_size"},
            },
        )

        logger.info(f"Exported syntactic VQ-VAE encoder to {output_path}")
        return output_path

    def export_dual_stream_decoder(
        self,
        decoder,
        filename: str = "synthesis_decoder.onnx",
        opset_version: int = 14,
    ) -> Path:
        """
        Export dual-stream DDSP decoder to ONNX.

        Exports the FiLM-based decoder (112D + 16D → 65D) for Rust-side synthesis.

        Args:
            decoder: DualStreamDDSPDecoder with FiLM layers
            filename: Output filename
            opset_version: ONNX opset version

        Returns:
            Path to exported ONNX model
        """
        output_path = self.output_dir / filename
        decoder.eval()

        # Example inputs
        dummy_features = torch.randn(1, 112)
        dummy_affect = torch.randn(1, 16)

        # Export to ONNX
        torch.onnx.export(
            decoder,
            (dummy_features, dummy_affect),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["features_112d", "affect_vector"],
            output_names=["harmonic_amps", "noise_mags"],
            dynamic_axes={
                "features_112d": {0: "batch_size"},
                "affect_vector": {0: "batch_size"},
                "harmonic_amps": {0: "batch_size"},
                "noise_mags": {0: "batch_size"},
            },
        )

        logger.info(f"Exported dual-stream decoder to {output_path}")
        return output_path

    def verify_onnx_model(
        self,
        onnx_path: Path,
        pytorch_model,
        dummy_input,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> bool:
        """
        Verify ONNX export matches PyTorch output.

        Args:
            onnx_path: Path to ONNX model
            pytorch_model: Original PyTorch model
            dummy_input: Test input tensor(s)
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            True if outputs match within tolerance
        """
        try:
            import onnxruntime as ort

            # PyTorch output
            pytorch_model.eval()
            with torch.no_grad():
                if isinstance(dummy_input, tuple):
                    torch_out = pytorch_model(*dummy_input)
                else:
                    torch_out = pytorch_model(dummy_input)

            # ONNX output
            ort_session = ort.InferenceSession(str(onnx_path))

            if isinstance(dummy_input, tuple):
                ort_inputs = {
                    name: inp.numpy() for name, inp in zip(ort_session.get_inputs(), dummy_input)
                }
            else:
                ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}

            ort_outs = ort_session.run(None, ort_inputs)

            # Compare outputs
            if isinstance(torch_out, tuple):
                for i, (torch_tensor, ort_array) in enumerate(zip(torch_out, ort_outs)):
                    np.testing.assert_allclose(
                        torch_tensor.numpy(), ort_array, rtol=rtol, atol=atol
                    )
            else:
                np.testing.assert_allclose(
                    torch_out.numpy(), ort_outs[0], rtol=rtol, atol=atol
                )

            logger.info(f"✓ Verified {onnx_path.name} matches PyTorch output")
            return True

        except Exception as e:
            logger.error(f"✗ Verification failed for {onnx_path.name}: {e}")
            return False


def export_all_dual_stream_models(
    vae_path: Optional[str] = None,
    vqvae_path: Optional[str] = None,
    output_dir: str = "models/onnx",
) -> dict:
    """
    Export all dual-stream models to ONNX.

    Args:
        vae_path: Path to trained affective VAE (creates new if None)
        vqvae_path: Path to trained syntactic VQ-VAE (creates new if None)
        output_dir: Directory to save ONNX models

    Returns:
        Dictionary mapping model names to ONNX paths
    """
    exporter = ONNXExporter(output_dir)
    results = {}

    # Create or load models
    if vae_path:
        vae = BetaVAE(input_dim=54, latent_dim=16, hidden_dim=128, beta=2.0)
        vae.load_state_dict(torch.load(vae_path))
        vae.eval()
    else:
        vae = create_affective_vae()
        vae.eval()

    if vqvae_path:
        vqvae = SyntacticVQVAE(input_dim=44, codebook_size=64, codebook_dim=32)
        vqvae.load_state_dict(torch.load(vqvae_path))
        vqvae.eval()
    else:
        vqvae = create_syntactic_vqvae()
        vqvae.eval()

    # Export affective VAE encoder
    results["affective_encoder"] = exporter.export_affective_vae_encoder(vae)

    # Export syntactic VQ-VAE encoder
    results["syntactic_encoder"] = exporter.export_syntactic_vqvae_encoder(vqvae)

    # Verify exports
    logger.info("Verifying ONNX exports...")

    # Verify affective encoder
    exporter.verify_onnx_model(
        results["affective_encoder"],
        vae,
        torch.randn(1, 54),
    )

    # Verify syntactic encoder
    exporter.verify_onnx_model(
        results["syntactic_encoder"],
        vqvae,
        torch.randn(1, 44),
    )

    logger.info(f"Exported {len(results)} models to {output_dir}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== ONNX Export for Dual-Stream Models ===")

    # Export all models
    results = export_all_dual_stream_models()

    print("\nExported models:")
    for name, path in results.items():
        print(f"  {name}: {path}")
