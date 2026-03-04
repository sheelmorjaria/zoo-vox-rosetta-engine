#!/usr/bin/env python3
"""
Denoise BEANS-Zero Detection Datasets using TensorRT FP16

High-noise datasets:
  - rfcx:     10406 samples (Precision 0.045)
  - gibbons:  18560 samples (Precision 0.024)
  - hiceas:    1485 samples (Precision 0.266)
  - enabirds:  4543 samples (Precision 0.890)
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_tensorrt_engine(engine_path):
    """Load TensorRT engine"""
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with open(engine_path, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    context = engine.create_execution_context()

    return engine, context


def denoise_chunk_tensorrt(engine, context, audio, chunk_size=48000):
    """Denoise a single chunk using TensorRT"""
    import pycuda.driver as cuda

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    # Pad if needed
    if len(audio) < chunk_size:
        padded = np.zeros(chunk_size, dtype=np.float32)
        padded[: len(audio)] = audio
        audio = padded

    input_tensor = audio.astype(np.float32)
    output_tensor = np.zeros(chunk_size, dtype=np.float32)

    d_input = cuda.mem_alloc(input_tensor.nbytes)
    d_output = cuda.mem_alloc(output_tensor.nbytes)

    stream = cuda.Stream()

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    context.set_input_shape(input_name, (1, 1, len(audio)))

    cuda.memcpy_htod_async(d_input, input_tensor, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(output_tensor, d_output, stream)
    stream.synchronize()

    return output_tensor[: len(audio)]


def denoise_audio(engine, context, audio, chunk_size=48000, overlap=100):
    """Denoise audio with overlap-add"""
    overlap_samples = int(overlap / 1000 * 16000)
    hop_size = chunk_size - overlap_samples

    output = np.zeros(len(audio), dtype=np.float32)
    weights = np.zeros(len(audio), dtype=np.float32)
    window = np.hanning(chunk_size).astype(np.float32)

    for start in range(0, len(audio) - overlap_samples, hop_size):
        end = min(start + chunk_size, len(audio))
        chunk = audio[start:end]

        if len(chunk) < chunk_size:
            padded = np.zeros(chunk_size, dtype=np.float32)
            padded[: len(chunk)] = chunk
            chunk = padded

        denoised = denoise_chunk_tensorrt(engine, context, chunk, chunk_size)

        actual_len = end - start
        output[start : start + actual_len] += denoised[:actual_len] * window[:actual_len]
        weights[start : start + actual_len] += window[:actual_len]

    weights = np.maximum(weights, 1e-8)
    output /= weights

    return output


def main():
    print("=" * 70)
    print("BEANS-Zero Detection Denoising with TensorRT FP16")
    print("=" * 70)

    # Paths
    engine_path = Path("dns64_fp16.trt")
    manifest_path = Path("beans_zero_cache/beans_audio_manifest.json")
    output_dir = Path("beans_zero_denoised")
    output_dir.mkdir(exist_ok=True)

    # Detection datasets to process
    detection_datasets = {"rfcx", "gibbons", "hiceas", "enabirds"}

    # Load engine
    print(f"\nLoading TensorRT engine: {engine_path}")
    engine, context = load_tensorrt_engine(engine_path)
    print("  ✓ Engine loaded")

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Filter detection samples
    samples = [
        s
        for s in manifest["samples"]
        if s.get("labels", {}).get("task") == "detection"
        and s.get("labels", {}).get("dataset_name") in detection_datasets
    ]

    print(f"\nProcessing {len(samples)} detection samples...")

    # Process samples
    total_rtf = 0
    processed = 0
    errors = 0

    for sample in tqdm(samples, desc="Denoising"):
        try:
            input_path = Path("beans_zero_cache") / sample["audio_file"]
            output_path = output_dir / sample["audio_file"]
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Load raw audio (float32)
            audio = np.fromfile(str(input_path), dtype=np.float32)
            sr = 44100

            # Resample to 16kHz for denoising
            ratio = 16000.0 / sr
            audio_16k = np.zeros(int(len(audio) * ratio), dtype=np.float32)
            for i in range(len(audio_16k)):
                src_idx = i / ratio
                idx0 = int(src_idx)
                idx1 = min(idx0 + 1, len(audio) - 1)
                frac = src_idx - idx0
                audio_16k[i] = audio[idx0] * (1 - frac) + audio[idx1] * frac

            # Denoise
            start = time.perf_counter()
            clean_16k = denoise_audio(engine, context, audio_16k, chunk_size=48000, overlap=100)
            elapsed = time.perf_counter() - start

            # Resample back to 44100 Hz
            ratio_back = 44100.0 / 16000.0
            clean_44k = np.zeros(int(len(clean_16k) * ratio_back), dtype=np.float32)
            for i in range(len(clean_44k)):
                src_idx = i / ratio_back
                idx0 = int(src_idx)
                idx1 = min(idx0 + 1, len(clean_16k) - 1)
                frac = src_idx - idx0
                clean_44k[i] = clean_16k[idx0] * (1 - frac) + clean_16k[idx1] * frac

            # Save as float32
            clean_44k.tofile(str(output_path))

            duration = len(audio) / sr
            rtf = duration / elapsed
            total_rtf += rtf
            processed += 1

        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"\n  Error: {sample['audio_file']}: {e}")

    # Copy manifest
    import shutil

    output_manifest = output_dir / "beans_audio_manifest.json"
    if not output_manifest.exists():
        shutil.copy(manifest_path, output_manifest)

    # Copy model files for evaluation
    for f in ["rosetta_net_model.json", "rf_species_105d.json", "rf_taxonomic_45d_v2.json"]:
        src = Path("beans_zero_cache") / f
        dst = output_dir / f
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)

    # Copy rosetta net
    for f in ["rosetta_net_best.mpk", "rosetta_net_best_config.json"]:
        src = Path(f)
        dst = output_dir / f
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Processed: {processed}")
    print(f"  Errors:    {errors}")
    print(f"  Avg RTF:   {total_rtf / max(processed, 1):.0f}x realtime")
    print(f"\n  Output: {output_dir}")
    print("\nTo evaluate denoised data:")
    print(f"  ./target/release/beans_zero_full_eval {output_dir}/beans_audio_manifest.json")


if __name__ == "__main__":
    main()
