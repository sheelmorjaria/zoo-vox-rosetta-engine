#!/usr/bin/env python3
"""
Jetson Edge Deployment - ONNX and TensorRT Export

Module 4 (v1.6.0): Export PyTorch models to ONNX and optimize with TensorRT
for deployment on NVIDIA Jetson devices with FP16 inference.

This module provides:
- ONNX export for DDSPDecoder and DDSPSynthesizer
- TensorRT engine building with FP16 optimization
- Model validation and benchmarking
- Jetson-compatible deployment artifacts

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Export functionality disabled.")

if TORCH_AVAILABLE:
    from .ddsp_decoder import DDSPDecoder
    from .ddsp_synthesis import DDSPSynthesizer

# Optional TensorRT import (may not be available on all systems)
try:
    import tensorrt as trt

    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.info("TensorRT not available. ONNX export only.")


# =============================================================================
# ONNX Export
# =============================================================================

if TORCH_AVAILABLE:

    def export_ddsp_decoder_to_onnx(
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 112),
        opset_version: int = 18,  # Use higher opset to avoid version conversion
        dynamic_axes: bool = True,
    ) -> bool:
        """
        Export DDSPDecoder to ONNX format.

        Args:
            model: Trained DDSPDecoder model
            output_path: Path to save ONNX model
            input_shape: Input tensor shape (batch_size, feature_dim)
            opset_version: ONNX opset version
            dynamic_axes: Enable dynamic batch size

        Returns:
            True if export succeeded
        """
        try:
            model.eval()

            # Create dummy input
            dummy_input = torch.randn(*input_shape)

            # Define dynamic axes for variable batch size
            if dynamic_axes:
                dynamic_axis_dict = {
                    "features_112d": {0: "batch_size"},
                    "harmonic_amps": {0: "batch_size"},
                    "noise_mags": {0: "batch_size"},
                }
            else:
                dynamic_axis_dict = None

            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=["features_112d"],
                output_names=["harmonic_amps", "noise_mags"],
                dynamic_axes=dynamic_axis_dict,
                opset_version=opset_version,
                export_params=True,
                do_constant_folding=True,
                keep_initializers_as_inputs=False,
            )

            logger.info(f"Exported DDSPDecoder to ONNX: {output_path}")

            # Verify the model
            _verify_onnx_model(output_path, dummy_input.shape)

            return True

        except Exception as e:
            logger.error(f"Failed to export DDSPDecoder to ONNX: {e}")
            return False

    def export_ddsp_synthesizer_to_onnx(
        model: nn.Module,
        output_path: str,
        f0_frames: int = 100,
        opset_version: int = 18,  # Use higher opset to avoid version conversion
        dynamic_axes: bool = False,  # Disabled by default due to PyTorch ONNX issues
    ) -> bool:
        """
        Export DDSPSynthesizer to ONNX format.

        Args:
            model: DDSPSynthesizer model
            output_path: Path to save ONNX model
            f0_frames: Number of F0 frames (variable in practice)
            opset_version: ONNX opset version
            dynamic_axes: Enable dynamic axes (may fail with newer PyTorch)

        Returns:
            True if export succeeded
        """
        try:
            model.eval()

            batch_size = 1
            num_harmonics = model.num_harmonics
            num_noise_bands = model.num_noise_bands

            # Create dummy inputs
            dummy_f0 = torch.randn(batch_size, f0_frames)
            dummy_harmonic_amps = torch.randn(batch_size, f0_frames, num_harmonics)
            dummy_noise_mags = torch.abs(torch.randn(batch_size, f0_frames, num_noise_bands))

            # Define dynamic axes (only if requested and compatible)
            if dynamic_axes:
                try:
                    # Try new-style dynamic shapes first
                    from torch.onnx import Dim  # noqa: F401

                    dynamic_shapes = {
                        "f0": {0: Dim("batch"), 1: Dim("frames")},
                        "harmonic_amps": {0: Dim("batch"), 1: Dim("frames")},
                        "noise_mags": {0: Dim("batch"), 1: Dim("frames")},
                    }
                    export_kwargs = {"dynamic_shapes": dynamic_shapes}
                except (ImportError, TypeError):
                    # Fall back to old-style dynamic_axes
                    dynamic_axis_dict = {
                        "f0": {0: "batch_size", 1: "n_frames"},
                        "harmonic_amps": {0: "batch_size", 1: "n_frames"},
                        "noise_mags": {0: "batch_size", 1: "n_frames"},
                        "audio": {0: "batch_size"},
                        "phase_acc": {0: "batch_size"},
                    }
                    export_kwargs = {"dynamic_axes": dynamic_axis_dict}
            else:
                export_kwargs = {}

            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_f0, dummy_harmonic_amps, dummy_noise_mags),
                output_path,
                input_names=["f0", "harmonic_amps", "noise_mags"],
                output_names=["audio", "phase_acc"],
                opset_version=opset_version,
                export_params=True,
                do_constant_folding=True,
                **export_kwargs,
            )

            logger.info(f"Exported DDSPSynthesizer to ONNX: {output_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to export DDSPSynthesizer to ONNX: {e}")
            return False

    def _verify_onnx_model(onnx_path: str, input_shape: Tuple[int, ...]) -> bool:
        """Verify ONNX model can be loaded and run."""
        try:
            import onnx
            import onnxruntime as ort

            # Load and check model
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)

            # Test inference
            session = ort.InferenceSession(onnx_path)
            dummy_input = np.randn(*input_shape).astype(np.float32)

            outputs = session.run(
                None,
                {"features_112d": dummy_input},
            )

            logger.info(f"ONNX model verified. Outputs: {[o.shape for o in outputs]}")
            return True

        except ImportError:
            logger.warning("onnx or onnxruntime not available. Skipping verification.")
            return True
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            return False


# =============================================================================
# TensorRT Builder
# =============================================================================

if TENSORRT_AVAILABLE:

    def build_tensorrt_engine(
        onnx_path: str,
        engine_path: str,
        fp16: bool = True,
        max_batch_size: int = 4,
        max_workspace_size: int = 1 << 30,  # 1GB
    ) -> bool:
        """
        Build TensorRT engine from ONNX model.

        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            fp16: Enable FP16 optimization
            max_batch_size: Maximum batch size for dynamic shapes
            max_workspace_size: Maximum GPU workspace size

        Returns:
            True if build succeeded
        """
        try:
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)

            # Create builder and network
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)

            # Parse ONNX model
            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    logger.error("Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return False

            # Create builder config
            config = builder.create_builder_config()

            # Set workspace size
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

            # Enable FP16 if supported
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 optimization enabled")
            else:
                logger.info("FP32 mode (FP16 not supported or disabled)")

            # Build engine
            logger.info("Building TensorRT engine...")
            engine = builder.build_serialized_network(network, config)

            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return False

            # Save engine
            with open(engine_path, "wb") as f:
                f.write(engine)

            logger.info(f"TensorRT engine saved: {engine_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to build TensorRT engine: {e}")
            return False

    class TensorRTInference:
        """
        TensorRT inference wrapper for deployed models.

        Provides high-performance inference on Jetson devices.
        """

        def __init__(
            self,
            engine_path: str,
            input_names: List[str],
            output_names: List[str],
        ):
            """
            Initialize TensorRT inference.

            Args:
                engine_path: Path to TensorRT engine file
                input_names: List of input tensor names
                output_names: List of output tensor names
            """
            self.logger = trt.Logger(trt.Logger.WARNING)

            # Load engine
            with open(engine_path, "rb") as f:
                self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())

            if self.engine is None:
                raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

            self.context = self.engine.create_execution_context()
            self.input_names = input_names
            self.output_names = output_names

            # Get I/O shapes
            self.input_shapes = {}
            self.output_shapes = {}

            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                self.engine.get_tensor_mode(name)

                if name in input_names:
                    self.input_shapes[name] = shape
                else:
                    self.output_shapes[name] = shape

            logger.info(f"Loaded TensorRT engine: {engine_path}")
            logger.info(f"Inputs: {self.input_shapes}")
            logger.info(f"Outputs: {self.output_shapes}")

        def infer(self, *inputs: np.ndarray) -> Dict[str, np.ndarray]:
            """
            Run inference.

            Args:
                *inputs: Input arrays matching input_names order

            Returns:
                Dictionary of output arrays
            """
            import pycuda.driver as cuda

            # Allocate GPU memory
            bindings = []
            for i, name in enumerate(self.input_names):
                input_data = inputs[i].astype(np.float32)
                input_mem = cuda.mem_alloc(input_data.nbytes)
                cuda.memcpy_htod(input_mem, input_data)
                bindings.append(int(input_mem))

            # Allocate output memory
            outputs = {}
            for name in self.output_names:
                shape = self.output_shapes[name]
                # Handle dynamic shapes (use -1 as placeholder)
                if any(s < 0 for s in shape):
                    # Use input batch size for dynamic shapes
                    batch_size = inputs[0].shape[0]
                    shape = tuple(batch_size if s < 0 else s for s in shape)

                output_mem = cuda.mem_alloc(np.prod(shape) * 4)  # 4 bytes per float32
                bindings.append(int(output_mem))
                outputs[name] = (output_mem, shape)

            # Set binding addresses
            for i, name in enumerate(self.input_names + self.output_names):
                self.context.set_tensor_address(name, bindings[i])

            # Run inference
            self.context.execute_async_v3(0)

            # Copy outputs back to host
            result = {}
            for name, (mem, shape) in outputs.items():
                output_data = np.empty(shape, dtype=np.float32)
                cuda.memcpy_dtoh(output_data, mem)
                result[name] = output_data

            return result


# =============================================================================
# Model Benchmarking
# =============================================================================

if TORCH_AVAILABLE:

    def benchmark_pytorch_model(
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 112),
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark PyTorch model inference time.

        Args:
            model: PyTorch model to benchmark
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs

        Returns:
            Dictionary with timing statistics
        """
        import time

        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_shape, device=device)

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(dummy_input)

        # Synchronize for accurate timing
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()

            with torch.no_grad():
                _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "median_ms": np.median(times),
        }

    def get_model_size_mb(model: nn.Module) -> float:
        """Get model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


# =============================================================================
# Export Pipeline
# =============================================================================


def export_ddsp_pipeline(
    decoder: nn.Module,
    synthesizer: nn.Module,
    output_dir: str,
    export_tensorrt: bool = False,
) -> Dict[str, str]:
    """
    Export complete DDSP pipeline for deployment.

    Args:
        decoder: Trained DDSPDecoder model
        synthesizer: DDSPSynthesizer model
        output_dir: Directory to save export artifacts
        export_tensorrt: Also build TensorRT engines

    Returns:
        Dictionary mapping component names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    # Export decoder to ONNX
    decoder_onnx = output_path / "ddsp_decoder.onnx"
    if export_ddsp_decoder_to_onnx(decoder, str(decoder_onnx)):
        artifacts["decoder_onnx"] = str(decoder_onnx)

    # Export synthesizer to ONNX
    synthesizer_onnx = output_path / "ddsp_synthesizer.onnx"
    if export_ddsp_synthesizer_to_onnx(synthesizer, str(synthesizer_onnx)):
        artifacts["synthesizer_onnx"] = str(synthesizer_onnx)

    # Optionally build TensorRT engines
    if export_tensorrt and TENSORRT_AVAILABLE:
        decoder_engine = output_path / "ddsp_decoder.trt"
        if build_tensorrt_engine(str(decoder_onnx), str(decoder_engine)):
            artifacts["decoder_engine"] = str(decoder_engine)

        synthesizer_engine = output_path / "ddsp_synthesizer.trt"
        if build_tensorrt_engine(str(synthesizer_onnx), str(synthesizer_engine)):
            artifacts["synthesizer_engine"] = str(synthesizer_engine)

    # Benchmark models
    if artifacts.get("decoder_onnx"):
        decoder_stats = benchmark_pytorch_model(decoder)
        logger.info(f"DDSPDecoder benchmark: {decoder_stats['mean_ms']:.2f}ms mean")

    if artifacts.get("synthesizer_onnx"):
        # Benchmark with typical input shape
        dummy_f0 = torch.randn(1, 100)
        dummy_harmonic = torch.randn(1, 100, synthesizer.num_harmonics)
        dummy_noise = torch.abs(torch.randn(1, 100, synthesizer.num_noise_bands))

        import time

        times = []
        for _ in range(50):
            start = time.perf_counter()
            with torch.no_grad():
                _ = synthesizer(dummy_f0, dummy_harmonic, dummy_noise)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        logger.info(f"DDSPSynthesizer benchmark: {np.mean(times):.2f}ms mean")

    logger.info(f"Export complete. Artifacts: {list(artifacts.keys())}")
    return artifacts


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if not TORCH_AVAILABLE:
        print("PyTorch not available. Install with: pip install torch")
        exit(1)

    print("Jetson Edge Deployment - ONNX and TensorRT Export")
    print("=" * 60)

    # Create sample models for testing
    from .ddsp_decoder import DDSPDecoder
    from .ddsp_synthesis import DDSPSynthesizer

    decoder = DDSPDecoder()
    synthesizer = DDSPSynthesizer()

    print(f"\nDecoder size: {get_model_size_mb(decoder):.2f} MB")
    print(f"Synthesizer size: {get_model_size_mb(synthesizer):.2f} MB")

    # Benchmark PyTorch models
    decoder_stats = benchmark_pytorch_model(decoder)
    print(  # noqa: E501
        f"\nDecoder inference: {decoder_stats['mean_ms']:.2f}ms "
        f"(std: {decoder_stats['std_ms']:.2f}ms)"
    )

    # Export to ONNX
    output_dir = "exports/ddsp_jetson"
    artifacts = export_ddsp_pipeline(
        decoder,
        synthesizer,
        output_dir,
        export_tensorrt=TENSORRT_AVAILABLE,
    )

    print("\nExported artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")

    print("\nExport complete!")
