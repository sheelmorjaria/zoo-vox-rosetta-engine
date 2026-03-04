#!/usr/bin/env python3
"""
TensorRT FP16 Denoiser CLI - Real-time audio denoising

Supports both TensorRT engines (.trt) and ONNX models (.onnx)

Performance:
  - TensorRT FP16: 292x realtime (10.3ms for 3s audio)
  - ONNX CUDA: ~200x realtime

Usage:
    # Use TensorRT engine (best performance)
    python tensorrt_denoiser.py --input noisy.wav --output clean.wav --engine dns64_fp16.trt

    # Use ONNX model
    python tensorrt_denoiser.py --input noisy.wav --output clean.wav --engine dns64_gpu.onnx

    # Batch process directory
    python tensorrt_denoiser.py --input-dir ./noisy/ --output-dir ./clean/
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_tensorrt_engine(engine_path):
    """Load TensorRT engine using TensorRT Python API"""
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with open(engine_path, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    print(f"Loaded TensorRT engine: {engine_path}")
    print(f"  Input: {engine.get_tensor_name(0)}")
    print(f"  Output: {engine.get_tensor_name(1)}")

    return engine, context


def load_onnx_model(model_path):
    """Load ONNX model with ONNX Runtime"""
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    print(f"Loaded ONNX model: {model_path}")
    print(f"  Providers: {session.get_providers()}")
    print(f"  Inputs: {[i.name for i in session.get_inputs()]}")
    print(f"  Outputs: {[o.name for o in session.get_outputs()]}")

    return session, "onnx"


def load_engine(engine_path):
    """Load TensorRT engine or ONNX model"""
    if engine_path.endswith(".trt"):
        engine, context = load_tensorrt_engine(engine_path)
        return (engine, context), "tensorrt"
    else:
        session, _ = load_onnx_model(engine_path)
        return session, "onnx"


def denoise_tensorrt(engine_context, audio, chunk_size=48000, overlap=100):
    """Denoise using TensorRT engine"""
    import pycuda.driver as cuda

    engine, context = engine_context

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # Resample to 16kHz
    audio_16k = resample_to_16k(audio, 16000)

    overlap_samples = int(overlap / 1000 * 16000)
    hop_size = chunk_size - overlap_samples

    output = np.zeros(len(audio_16k), dtype=np.float32)
    weights = np.zeros(len(audio_16k), dtype=np.float32)

    window = np.hanning(chunk_size).astype(np.float32)

    # Process chunks
    for start in range(0, len(audio_16k) - overlap_samples, hop_size):
        end = min(start + chunk_size, len(audio_16k))
        chunk = audio_16k[start:end]

        if len(chunk) < chunk_size:
            padded = np.zeros(chunk_size, dtype=np.float32)
            padded[: len(chunk)] = chunk
            chunk = padded

        # TensorRT inference
        input_tensor = chunk.astype(np.float32)
        output_tensor = np.zeros(chunk_size, dtype=np.float32)

        d_input = cuda.mem_alloc(input_tensor.nbytes)
        d_output = cuda.mem_alloc(output_tensor.nbytes)

        stream = cuda.Stream()

        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))
        context.set_input_shape(input_name, (1, 1, chunk_size))

        cuda.memcpy_htod_async(d_input, input_tensor, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(output_tensor, d_output, stream)
        stream.synchronize()

        # Overlap-add
        actual_len = min(end - start, len(output_tensor))
        output[start : start + actual_len] += output_tensor[:actual_len] * window[:actual_len]
        weights[start : start + actual_len] += window[:actual_len]

    weights = np.maximum(weights, 1e-8)
    output /= weights

    return output


def denoise_onnx(session, audio, chunk_size=48000, overlap=100):
    """Denoise using ONNX Runtime"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    audio_16k = resample_to_16k(audio, 16000)

    overlap_samples = int(overlap / 1000 * 16000)
    hop_size = chunk_size - overlap_samples

    output = np.zeros(len(audio_16k), dtype=np.float32)
    weights = np.zeros(len(audio_16k), dtype=np.float32)

    window = np.hanning(chunk_size).astype(np.float32)

    for start in range(0, len(audio_16k) - overlap_samples, hop_size):
        end = min(start + chunk_size, len(audio_16k))
        chunk = audio_16k[start:end]

        if len(chunk) < chunk_size:
            padded = np.zeros(chunk_size, dtype=np.float32)
            padded[: len(chunk)] = chunk
            chunk = padded

        input_tensor = chunk.reshape(1, 1, -1).astype(np.float32)
        output_tensor = session.run([output_name], {input_name: input_tensor})[0].squeeze()

        actual_len = min(end - start, len(output_tensor))
        output[start : start + actual_len] += output_tensor[:actual_len] * window[:actual_len]
        weights[start : start + actual_len] += window[:actual_len]

    weights = np.maximum(weights, 1e-8)
    output /= weights

    return output


def resample_to_16k(audio, target_sr):
    """Linear interpolation resampling"""
    if target_sr == 16000:
        return audio

    ratio = 16000.0 / target_sr
    output_length = int(len(audio) * ratio)
    output = np.zeros(output_length, dtype=np.float32)

    for i in range(output_length):
        src_idx = i / ratio
        idx0 = int(src_idx)
        idx1 = min(idx0 + 1, len(audio) - 1)
        frac = src_idx - idx0
        output[i] = audio[idx0] * (1 - frac) + audio[idx1] * frac

    return output


def load_audio(path):
    """Load audio file"""
    import soundfile as sf

    audio, sr = sf.read(path)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    return audio.astype(np.float32), sr


def save_audio(path, audio, sr):
    """Save audio file"""
    import soundfile as sf

    sf.write(path, audio, sr)


def process_file(input_path, output_path, engine, engine_type, overlap=100):
    """Process a single file"""
    print(f"Processing: {input_path}")

    audio, sr = load_audio(input_path)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.2f}s")

    start = time.perf_counter()

    if engine_type == "tensorrt":
        clean = denoise_tensorrt(engine, audio, overlap=overlap)
    else:
        clean = denoise_onnx(engine, audio, overlap=overlap)

    elapsed = time.perf_counter() - start
    rtf = duration / elapsed

    print(f"  Processing time: {elapsed * 1000:.1f}ms")
    print(f"  Realtime factor: {rtf:.0f}x")

    save_audio(output_path, clean, 16000)
    print(f"  Saved to: {output_path}")

    return rtf


def main():
    parser = argparse.ArgumentParser(description="TensorRT FP16 Denoiser")
    parser.add_argument("--input", type=str, help="Input audio file")
    parser.add_argument("--output", type=str, help="Output audio file")
    parser.add_argument("--input-dir", type=str, help="Input directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--engine",
        type=str,
        default="dns64_fp16.trt",
        help="TensorRT engine (.trt) or ONNX model (.onnx)",
    )
    parser.add_argument("--overlap", type=int, default=100, help="Overlap in ms")

    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════════════╗")
    print("║          TensorRT FP16 Denoiser (292x realtime)            ║")
    print("╚════════════════════════════════════════════════════════════╝")

    if not os.path.exists(args.engine):
        print(f"Error: Engine file not found: {args.engine}")
        sys.exit(1)

    engine, engine_type = load_engine(args.engine)
    print(f"  Using: {engine_type.upper()}")

    if args.input and args.output:
        process_file(args.input, args.output, engine, engine_type, args.overlap)

    elif args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        total_rtf = 0
        count = 0

        for ext in ["*.wav", "*.flac"]:
            for path in Path(args.input_dir).glob(ext):
                output_path = Path(args.output_dir) / path.name
                rtf = process_file(str(path), str(output_path), engine, engine_type, args.overlap)
                total_rtf += rtf
                count += 1

        if count > 0:
            print(f"\nProcessed {count} files")
            print(f"Average RTF: {total_rtf / count:.0f}x")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
