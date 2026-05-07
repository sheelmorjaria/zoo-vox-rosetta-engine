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
- Auto-detection of Jetson device (Nano/Xavier/Orin)
- Tiered export pipeline for different device capabilities

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# Jetson Device Detection and Tiered Configuration
# =============================================================================


class JetsonDevice(Enum):
    """Detected Jetson device types."""

    NANO = "nano"  # Jetson Nano (Maxwell, no FP16)
    XAVIER = "xavier"  # Jetson Xavier NX (Volta, FP16 support)
    ORIN = "orin"  # Jetson Orin Nano (Ampere, best TensorRT)
    UNKNOWN = "unknown"


@dataclass
class JetsonTierConfig:
    """Configuration for a specific Jetson device tier."""

    device_type: JetsonDevice
    use_tensorrt: bool
    fp16: bool
    num_harmonics: int
    num_noise_bands: int
    enable_post_filter: bool = False
    target_latency_ms: float = 50.0
    description: str = ""


# Tier configurations for each Jetson device
JETSON_TIER_CONFIGS: Dict[JetsonDevice, JetsonTierConfig] = {
    JetsonDevice.NANO: JetsonTierConfig(
        device_type=JetsonDevice.NANO,
        use_tensorrt=False,  # No FP16 hardware support
        fp16=False,
        num_harmonics=40,  # Reduced for 4GB RAM
        num_noise_bands=3,
        enable_post_filter=False,
        target_latency_ms=30.0,
        description="Jetson Nano: PyTorch FP32, reduced model size",
    ),
    JetsonDevice.XAVIER: JetsonTierConfig(
        device_type=JetsonDevice.XAVIER,
        use_tensorrt=True,
        fp16=True,  # Volta Tensor Cores
        num_harmonics=60,  # Full model
        num_noise_bands=5,
        enable_post_filter=False,
        target_latency_ms=12.0,
        description="Jetson Xavier NX: TensorRT FP16, full model",
    ),
    JetsonDevice.ORIN: JetsonTierConfig(
        device_type=JetsonDevice.ORIN,
        use_tensorrt=True,
        fp16=True,  # Ampere Tensor Cores
        num_harmonics=60,
        num_noise_bands=5,
        enable_post_filter=True,  # Orin-only feature
        target_latency_ms=15.0,
        description="Jetson Orin Nano: TensorRT FP16 + neural post-filter",
    ),
    JetsonDevice.UNKNOWN: JetsonTierConfig(
        device_type=JetsonDevice.UNKNOWN,
        use_tensorrt=False,
        fp16=False,
        num_harmonics=40,  # Conservative defaults
        num_noise_bands=3,
        enable_post_filter=False,
        target_latency_ms=50.0,
        description="Unknown device: Conservative FP32 configuration",
    ),
}


def detect_jetson_device() -> JetsonDevice:
    """
    Auto-detect the Jetson device type.

    Detection strategy:
    1. Check /etc/nv_tegra_release for Tegra platform
    2. Check /proc/cpuinfo for chip ID (tegra234=Orin, tegra194=Xavier, tegra210=Nano)

    Returns:
        Detected JetsonDevice type
    """
    # Check if we're on a Jetson at all
    tegra_release = Path("/etc/nv_tegra_release")
    if not tegra_release.exists():
        logger.info("Not a Jetson device (no /etc/nv_tegra_release)")
        return JetsonDevice.UNKNOWN

    try:
        with open(tegra_release, "r") as f:
            release_info = f.read()
        logger.debug(f"Tegra release info: {release_info}")
    except Exception as e:
        logger.warning(f"Could not read tegra release: {e}")
        return JetsonDevice.UNKNOWN

    # Check CPU info for chip ID
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpu_info = f.read().lower()

        # Detect by chip ID
        if "tegra234" in cpu_info:
            logger.info("Detected: Jetson Orin (tegra234)")
            return JetsonDevice.ORIN
        elif "tegra194" in cpu_info:
            logger.info("Detected: Jetson Xavier NX (tegra194)")
            return JetsonDevice.XAVIER
        elif "tegra210" in cpu_info or "tegra186" in cpu_info:
            logger.info("Detected: Jetson Nano (tegra210/186)")
            return JetsonDevice.NANO
    except Exception as e:
        logger.warning(f"Could not read cpuinfo: {e}")

    # Fallback: try to detect from DTS platform
    try:
        with open("/sys/firmware/devicetree/base/model", "r") as f:
            model = f.read().strip("\x00").lower()

        if "orin" in model:
            logger.info(f"Detected: Jetson Orin from model string: {model}")
            return JetsonDevice.ORIN
        elif "xavier" in model:
            logger.info(f"Detected: Jetson Xavier from model string: {model}")
            return JetsonDevice.XAVIER
        elif "nano" in model:
            logger.info(f"Detected: Jetson Nano from model string: {model}")
            return JetsonDevice.NANO
    except Exception:
        pass

    logger.warning("Could not detect specific Jetson device, using UNKNOWN")
    return JetsonDevice.UNKNOWN


def get_tier_config(device: Optional[JetsonDevice] = None) -> JetsonTierConfig:
    """
    Get the tier configuration for a specific device.

    Args:
        device: Device type (auto-detect if None)

    Returns:
        JetsonTierConfig for the device
    """
    if device is None:
        device = detect_jetson_device()

    return JETSON_TIER_CONFIGS.get(device, JETSON_TIER_CONFIGS[JetsonDevice.UNKNOWN])


def get_export_dir_for_device(device: JetsonDevice, base_dir: str = "exports") -> str:
    """Get the export directory for a specific device tier."""
    return {
        JetsonDevice.NANO: f"{base_dir}/nano_fp32",
        JetsonDevice.XAVIER: f"{base_dir}/xavier_fp16",
        JetsonDevice.ORIN: f"{base_dir}/orin_fp16_postfilter",
        JetsonDevice.UNKNOWN: f"{base_dir}/universal_fp32",
    }[device]


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
# Tiered Export Pipeline (Auto-detection for Jetson devices)
# =============================================================================


def export_ddsp_for_jetson_tier(
    decoder: nn.Module,
    synthesizer: nn.Module,
    device: Optional[JetsonDevice] = None,
    base_export_dir: str = "exports",
    save_manifest: bool = True,
) -> Dict[str, str]:
    """
    Export DDSP pipeline with auto-detected or specified Jetson device tier.

    This function automatically configures the export based on the detected
    Jetson device capabilities:
    - Nano: FP32 ONNX only (reduced model size)
    - Xavier: TensorRT FP16 (full model)
    - Orin: TensorRT FP16 + post-filter (full model + quality boost)

    Args:
        decoder: Trained DDSPDecoder model
        synthesizer: DDSPSynthesizer model
        device: Target device (auto-detect if None)
        base_export_dir: Base directory for exports
        save_manifest: Save deployment manifest JSON

    Returns:
        Dictionary mapping component names to file paths
    """
    # Detect or use specified device
    if device is None:
        device = detect_jetson_device()

    config = get_tier_config(device)
    export_dir = get_export_dir_for_device(device, base_export_dir)

    logger.info(f"Exporting for: {config.description}")
    logger.info(f"Target latency: {config.target_latency_ms}ms")
    logger.info(f"Export directory: {export_dir}")

    # Create models with tier-specific configuration
    # Note: For reduced harmonics/noise bands, we would need to create
    # modified synthesizer instances. For now, we use the full model
    # and let the device handle capacity constraints.
    output_path = Path(export_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts = {}

    # Export decoder to ONNX
    decoder_onnx = output_path / "ddsp_decoder.onnx"
    if export_ddsp_decoder_to_onnx(
        decoder,
        str(decoder_onnx),
        dynamic_axes=not config.use_tensorrt,  # Fixed axes for TensorRT
    ):
        artifacts["decoder_onnx"] = str(decoder_onnx)

    # Export synthesizer to ONNX
    synthesizer_onnx = output_path / "ddsp_synthesizer.onnx"
    if export_ddsp_synthesizer_to_onnx(
        synthesizer,
        str(synthesizer_onnx),
        dynamic_axes=False,  # Fixed axes for TensorRT compatibility
    ):
        artifacts["synthesizer_onnx"] = str(synthesizer_onnx)

    # Build TensorRT engines if enabled for this tier
    if config.use_tensorrt and TENSORRT_AVAILABLE:
        decoder_engine = output_path / "ddsp_decoder.trt"
        if build_tensorrt_engine(
            str(decoder_onnx),
            str(decoder_engine),
            fp16=config.fp16,
        ):
            artifacts["decoder_engine"] = str(decoder_engine)

        synthesizer_engine = output_path / "ddsp_synthesizer.trt"
        if build_tensorrt_engine(
            str(synthesizer_onnx),
            str(synthesizer_engine),
            fp16=config.fp16,
        ):
            artifacts["synthesizer_engine"] = str(synthesizer_engine)

    # Create deployment manifest
    if save_manifest:
        manifest = {
            "device_type": device.value,
            "config": {
                "use_tensorrt": config.use_tensorrt,
                "fp16": config.fp16,
                "num_harmonics": config.num_harmonics,
                "num_noise_bands": config.num_noise_bands,
                "enable_post_filter": config.enable_post_filter,
                "target_latency_ms": config.target_latency_ms,
            },
            "artifacts": artifacts,
            "description": config.description,
        }

        manifest_path = output_path / "deployment_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        artifacts["manifest"] = str(manifest_path)
        logger.info(f"Deployment manifest saved to: {manifest_path}")

    # Benchmark summary
    logger.info("=" * 60)
    logger.info(f"Export Summary for {device.value.upper()}:")
    logger.info(f"  Description: {config.description}")
    logger.info(f"  TensorRT: {config.use_tensorrt}")
    logger.info(f"  FP16: {config.fp16}")
    logger.info(f"  Harmonics: {config.num_harmonics}")
    logger.info(f"  Noise Bands: {config.num_noise_bands}")
    logger.info(f"  Post-Filter: {config.enable_post_filter}")
    logger.info(f"  Target Latency: {config.target_latency_ms}ms")
    logger.info("=" * 60)

    return artifacts


def export_all_jets_tiers(
    decoder: nn.Module,
    synthesizer: nn.Module,
    base_export_dir: str = "exports",
) -> Dict[JetsonDevice, Dict[str, str]]:
    """
    Export DDSP pipeline for all Jetson device tiers.

    Useful for pre-building artifacts for deployment on any Jetson device.

    Args:
        decoder: Trained DDSPDecoder model
        synthesizer: DDSPSynthesizer model
        base_export_dir: Base directory for exports

    Returns:
        Dictionary mapping device types to their export artifacts
    """
    all_artifacts = {}

    for device in [JetsonDevice.NANO, JetsonDevice.XAVIER, JetsonDevice.ORIN]:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Exporting for {device.value.upper()} tier...")
        logger.info(f"{'=' * 60}")

        try:
            artifacts = export_ddsp_for_jetson_tier(
                decoder,
                synthesizer,
                device=device,
                base_export_dir=base_export_dir,
            )
            all_artifacts[device] = artifacts
        except Exception as e:
            logger.error(f"Failed to export for {device.value}: {e}")

    logger.info(f"\n{'=' * 60}")
    logger.info("Export Summary (All Tiers):")
    logger.info(f"{'=' * 60}")
    for device, artifacts in all_artifacts.items():
        logger.info(f"  {device.value}: {len(artifacts)} artifacts")

    return all_artifacts


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
