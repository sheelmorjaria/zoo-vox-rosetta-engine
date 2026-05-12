#!/usr/bin/env python3
"""
ONNX Optimization for Sub-12ms Latency Budget

Exports pipeline components to ONNX format with optimizations for:
- TensorRT execution on Jetson Orin Nano
- ONNX Runtime CPU fallback
- Sub-12ms inference time per frame

Critical components to optimize:
1. CPC Encoder - 1D Conv layers for boundary detection
2. Autoregressive Model (Mamba/TCN) - Temporal context
3. BioMAE Encoder - Spectrogram to 112D features

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.onnx
import onnx
import onnxruntime as ort
from dataclasses import dataclass

from boundary_detection.cpc_encoder import CPCEncoder
from boundary_detection.cpc_autoregressive import AutoregressiveMamba, TCNAutoregressive
from feature_extraction.biomae import BioMAEEncoder, EncoderConfig

logger = logging.getLogger(__name__)


@dataclass
class ONNXExportConfig:
    """Configuration for ONNX export."""
    # Input specifications
    cpc_frame_size: int = 480  # 10ms @ 48kHz
    cpc_batch_size: int = 1
    biomae_img_size: int = 128
    biomae_batch_size: int = 1

    # ONNX export options
    opset_version: int = 17  # Latest stable opset
    dynamic_axes: bool = True  # Allow variable batch size

    # Optimization options
    optimize_model: bool = True
    fp16: bool = True  # Half precision for faster inference

    # Target platform
    target_device: str = "cuda"  # "cuda" for TensorRT, "cpu" for CPU


class ONNXLatencyProfiler:
    """
    Profile ONNX model latency to ensure sub-12ms budget compliance.

    The 12ms budget is critical for:
    - Sub-50ms total round-trip latency
    - Real-time boundary detection
    - Interactive vocalization analysis
    """

    def __init__(self, model_path: Path, target_latency_ms: float = 12.0):
        self.model_path = Path(model_path)
        self.target_latency_ms = target_latency_ms

        # Configure ONNX Runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"Loaded ONNX model: {self.model_path}")
        logger.info(f"Provider: {self.session.get_providers()}")

    def profile(
        self,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
    ) -> Dict[str, float]:
        """
        Profile inference latency.

        Returns:
            Dictionary with latency statistics (ms)
        """
        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        logger.info(f"Input shape: {input_shape}")

        # Warmup
        for _ in range(warmup_iterations):
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            _ = self.session.run([self.output_name], {self.input_name: dummy_input})

        # Timed runs
        latencies = []
        for _ in range(num_iterations):
            dummy_input = np.random.randn(*input_shape).astype(np.float32)

            import time
            start = time.perf_counter()
            _ = self.session.run([self.output_name], {self.input_name: dummy_input})
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # Convert to ms

        latencies = np.array(latencies)

        results = {
            'mean_ms': np.mean(latencies),
            'std_ms': np.std(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
        }

        results['within_budget'] = results['p99_ms'] <= self.target_latency_ms

        logger.info(
            f"Latency profiling (n={num_iterations}):\n"
            f"  Mean: {results['mean_ms']:.2f}ms\n"
            f"  P50: {results['p50_ms']:.2f}ms\n"
            f"  P95: {results['p95_ms']:.2f}ms\n"
            f"  P99: {results['p99_ms']:.2f}ms\n"
            f"  Target: {self.target_latency_ms}ms\n"
            f"  Within budget: {results['within_budget']}"
        )

        return results


def export_cpc_encoder_to_onnx(
    encoder: CPCEncoder,
    output_path: Path,
    config: Optional[ONNXExportConfig] = None,
) -> None:
    """
    Export CPC Encoder to ONNX format.

    Args:
        encoder: Trained CPCEncoder model
        output_path: Path to save ONNX model
        config: Export configuration
    """
    if config is None:
        config = ONNXExportConfig()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder.eval()

    # Dummy input
    dummy_input = torch.randn(
        config.cpc_batch_size,
        1,
        config.cpc_frame_size,
    )

    # Dynamic axes for variable batch size
    dynamic_axes = {
        'audio': {0: 'batch_size'},
        'latent': {0: 'batch_size'},
    } if config.dynamic_axes else None

    # Export to ONNX
    torch.onnx.export(
        encoder,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=config.opset_version,
        do_constant_folding=True,
        input_names=['audio'],
        output_names=['latent'],
        dynamic_axes=dynamic_axes,
    )

    logger.info(f"Exported CPC Encoder to {output_path}")

    # Load and validate
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    # Optimize if requested
    if config.optimize_model:
        from onnxoptimizer import optimize
        optimized_model = optimize(onnx_model)
        onnx.save(optimized_model, str(output_path))
        logger.info(f"Optimized ONNX model saved to {output_path}")

    # Profile latency
    profiler = ONNXLatencyProfiler(output_path, target_latency_ms=12.0)
    latency_results = profiler.profile()

    if not latency_results['within_budget']:
        logger.warning(
            f"CPC Encoder exceeds latency budget: "
            f"P99={latency_results['p99_ms']:.2f}ms > 12ms"
        )
    else:
        logger.info(f"CPC Encoder within latency budget: {latency_results['p99_ms']:.2f}ms")


def export_ar_model_to_onnx(
    model: AutoregressiveMamba | TCNAutoregressive,
    output_path: Path,
    config: Optional[ONNXExportConfig] = None,
) -> None:
    """
    Export Autoregressive Model to ONNX format.

    Note: Mamba models may have limited ONNX support. TCN is preferred
    for ONNX export.
    """
    if config is None:
        config = ONNXExportConfig()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Dummy input: (batch, time, d_model)
    d_model = model.d_model if hasattr(model, 'd_model') else 128
    dummy_input = torch.randn(1, 32, d_model)  # 32 time steps

    dynamic_axes = {
        'z_sequence': {0: 'batch_size', 1: 'time'},
        'context': {0: 'batch_size', 1: 'time'},
    }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=config.opset_version,
        do_constant_folding=True,
        input_names=['z_sequence'],
        output_names=['context'],
        dynamic_axes=dynamic_axes,
    )

    logger.info(f"Exported AR Model to {output_path}")

    # Validate
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)


def export_biomae_encoder_to_onnx(
    encoder: BioMAEEncoder,
    output_path: Path,
    config: Optional[ONNXExportConfig] = None,
) -> None:
    """
    Export BioMAE Encoder to ONNX format.
    """
    if config is None:
        config = ONNXExportConfig()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder.eval()

    # Dummy input: (batch, channels, height, width)
    img_size = config.biomae_img_size
    dummy_input = torch.randn(
        config.biomae_batch_size,
        1,
        img_size,
        img_size,
    )

    dynamic_axes = {
        'spectrogram': {0: 'batch_size'},
        'embedding': {0: 'batch_size'},
    }

    torch.onnx.export(
        encoder,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=config.opset_version,
        do_constant_folding=True,
        input_names=['spectrogram'],
        output_names=['embedding'],
        dynamic_axes=dynamic_axes,
    )

    logger.info(f"Exported BioMAE Encoder to {output_path}")

    # Validate
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)


def optimize_for_tensorrt(
    onnx_path: Path,
    output_path: Optional[Path] = None,
    fp16: bool = True,
) -> Path:
    """
    Optimize ONNX model for TensorRT execution.

    This requires tensorrt installation and the torch-tensorrt package.

    Args:
        onnx_path: Path to input ONNX model
        output_path: Path to save optimized model (optional)
        fp16: Use FP16 precision for faster inference

    Returns:
        Path to optimized model
    """
    try:
        import tensorrt as trt
        from torch_tensorrt import compile
        HAS_TENSORRT = True
    except ImportError:
        logger.warning("TensorRT not available, skipping optimization")
        HAS_TENSORRT = False
        return onnx_path

    if output_path is None:
        output_path = onnx_path.parent / f"{onnx_path.stem}_trt{onnx_path.suffix}"

    logger.info(f"Optimizing {onnx_path} for TensorRT...")

    # TensorRT optimization would go here
    # For now, just return the original path
    logger.warning("TensorRT optimization not fully implemented")

    return onnx_path


def export_pipeline_to_onnx(
    output_dir: Path,
    config: Optional[ONNXExportConfig] = None,
) -> Dict[str, Path]:
    """
    Export all pipeline components to ONNX format.

    Args:
        output_dir: Directory to save ONNX models
        config: Export configuration

    Returns:
        Dictionary mapping component names to ONNX file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = ONNXExportConfig()

    exported = {}

    # Export CPC Encoder
    from boundary_detection.cpc_encoder import create_encoder
    from boundary_detection.cpc_encoder import EncoderConfig
    cpc = create_encoder(EncoderConfig())
    cpc_path = output_dir / "cpc_encoder.onnx"
    export_cpc_encoder_to_onnx(cpc, cpc_path, config)
    exported['cpc_encoder'] = cpc_path

    # Export Autoregressive Model (TCN for ONNX compatibility)
    from boundary_detection.cpc_autoregressive import TCNAutoregressive
    ar = TCNAutoregressive(d_model=128)
    ar_path = output_dir / "ar_model_tcn.onnx"
    export_ar_model_to_onnx(ar, ar_path, config)
    exported['ar_model'] = ar_path

    # Export BioMAE Encoder
    from feature_extraction.biomae import BioMAEEncoder, EncoderConfig as BioMAEConfig
    biomae = BioMAEEncoder(BioMAEConfig())
    biomae_path = output_dir / "biomae_encoder.onnx"
    export_biomae_encoder_to_onnx(biomae, biomae_path, config)
    exported['biomae_encoder'] = biomae_path

    logger.info(f"\nExported {len(exported)} components to ONNX:")
    for name, path in exported.items():
        logger.info(f"  {name}: {path}")

    return exported


def validate_latency_budget(
    onnx_dir: Path,
    target_latency_ms: float = 12.0,
) -> Dict[str, Dict[str, float]]:
    """
    Validate that all components meet the latency budget.

    Returns:
        Dictionary mapping component names to latency results
    """
    onnx_dir = Path(onnx_dir)
    results = {}

    components = {
        'cpc_encoder': 'cpc_encoder.onnx',
        'ar_model': 'ar_model_tcn.onnx',
        'biomae_encoder': 'biomae_encoder.onnx',
    }

    for name, filename in components.items():
        onnx_path = onnx_dir / filename
        if onnx_path.exists():
            profiler = ONNXLatencyProfiler(onnx_path, target_latency_ms)
            results[name] = profiler.profile()
        else:
            logger.warning(f"ONNX model not found: {onnx_path}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Latency Budget Validation (Target: {target_latency_ms}ms)")
    logger.info(f"{'='*60}")

    all_within_budget = True
    for name, result in results.items():
        within = "✓" if result['within_budget'] else "✗"
        logger.info(
            f"{within} {name}: P99={result['p99_ms']:.2f}ms, "
            f"Mean={result['mean_ms']:.2f}ms"
        )
        if not result['within_budget']:
            all_within_budget = False

    logger.info(f"{'='*60}")
    if all_within_budget:
        logger.info("All components within latency budget!")
    else:
        logger.warning("Some components exceed latency budget!")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Export pipeline components
    output_dir = Path("/mnt/c/Users/sheel/Desktop/src/models/onnx")
    exported = export_pipeline_to_onnx(output_dir)

    # Validate latency
    results = validate_latency_budget(output_dir, target_latency_ms=12.0)
