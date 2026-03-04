#!/usr/bin/env python3
"""
Fast BEANS-Zero Manifest Creator
================================

Creates manifest for the full 91,965 sample dataset using parallel WAV extraction.
Uses multiprocessing for 10-24x faster processing.

Usage:
    python scripts/beans_create_manifest_fast.py --dataset beans_zero_data/beans_zero_test --output beans_zero_full_manifest.json

The manifest creation will:
1. Extract audio from HuggingFace format to WAV files (parallel)
2. Create JSON manifest for Rust training pipeline
3. Show progress and ETA
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp

import numpy as np
from scipy.io import wavfile

# Constants
SAMPLE_RATE = 44100


def process_sample(args):
    """Process a single sample - extract to WAV and return manifest entry."""
    idx, sample, audio_dir = args

    try:
        # Get audio data - BEANS-Zero stores as list of floats or dict
        audio_data = sample.get("audio", [])

        if not audio_data:
            return None

        # Handle different audio formats
        if isinstance(audio_data, dict):
            array = audio_data.get("array", [])
            sr = audio_data.get("sampling_rate", 44100)
        elif isinstance(audio_data, (list, np.ndarray)):
            array = audio_data
            sr = 44100
        else:
            return None

        if len(array) == 0:
            return None

        # Create audio filename
        audio_filename = f"sample_{idx:06d}.wav"
        audio_path = audio_dir / audio_filename

        # Convert to numpy array
        audio_np = np.array(array, dtype=np.float32)

        # Normalize if needed
        max_val = max(abs(audio_np.max()), abs(audio_np.min())) if len(audio_np) > 0 else 0
        if max_val > 1.0:
            audio_np = audio_np / max_val

        # Convert to int16 for WAV
        audio_int16 = (audio_np * 32767).astype(np.int16)

        # Resample to 44100 Hz if needed
        if sr != SAMPLE_RATE and len(audio_int16) > 0:
            ratio = SAMPLE_RATE / sr
            new_len = int(len(audio_int16) * ratio)
            if new_len > 0:
                indices = np.linspace(0, len(audio_int16) - 1, new_len)
                audio_int16 = audio_int16[indices.astype(int)]

        # Write WAV file
        wavfile.write(str(audio_path), SAMPLE_RATE, audio_int16)

        # Get label
        output_label = sample.get("output", "unknown")
        if output_label is None:
            output_label = "unknown"

        return {
            "audio_file": str(audio_path),
            "n_samples": len(array),
            "labels": {
                "output": output_label,
                "task": sample.get("dataset_name", "unknown")
            }
        }

    except Exception as e:
        print(f"Error processing sample {idx}: {e}", file=sys.stderr)
        return None


def create_manifest_parallel(dataset_path: Path, output_path: Path, n_workers: int = None):
    """Create manifest using parallel processing."""
    from datasets import load_from_disk

    if n_workers is None:
        n_workers = mp.cpu_count()

    print()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║     BEANS-Zero Manifest Creator (Parallel)                       ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    ds = load_from_disk(str(dataset_path))
    n_samples = len(ds)
    print(f"Total samples: {n_samples}")

    # Create audio output directory
    audio_dir = output_path.parent / "beans_audio_full"
    audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"Audio directory: {audio_dir}")

    # Prepare arguments for parallel processing
    print(f"\nProcessing {n_samples} samples with {n_workers} workers...")
    start_time = time.time()

    # Convert dataset samples to list for parallel processing
    # We do this in chunks to avoid memory issues
    manifest_samples = []
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit in batches to avoid memory issues
        batch_size = 1000
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)

            # Prepare batch
            batch_args = []
            for i in range(batch_start, batch_end):
                batch_args.append((i, ds[i], audio_dir))

            # Submit batch
            futures = {executor.submit(process_sample, arg): arg[0] for arg in batch_args}

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    manifest_samples.append(result)

                completed += 1
                if completed % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (n_samples - completed) / rate
                    print(f"  Processed {completed}/{n_samples} ({completed/n_samples*100:.1f}%) - "
                          f"{rate:.1f} samples/s - ETA: {eta/60:.1f}min", flush=True)

    elapsed = time.time() - start_time
    print(f"\nManifest creation completed in {elapsed:.1f}s ({n_samples/elapsed:.1f} samples/s)")

    # Create final manifest
    manifest = {
        "dataset": "BEANS-Zero",
        "n_samples": len(manifest_samples),
        "samples": manifest_samples
    }

    # Save manifest
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCreated manifest: {output_path}")
    print(f"  Total samples: {manifest['n_samples']}")

    # Print label distribution
    labels = [s["labels"]["output"] for s in manifest["samples"]]
    unique_labels = set(labels)
    print(f"  Unique labels: {len(unique_labels)}")

    # Print label frequency
    from collections import Counter
    label_counts = Counter(labels)
    print(f"\nTop 20 labels by frequency:")
    for label, count in label_counts.most_common(20):
        print(f"    {label}: {count}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Create BEANS-Zero manifest (parallel)")
    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        default=Path("beans_zero_data/beans_zero_test"),
        help="Path to HuggingFace dataset"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("beans_zero_full_manifest.json"),
        help="Output manifest path"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=mp.cpu_count(),
        help=f"Number of parallel workers (default: {mp.cpu_count()})"
    )

    args = parser.parse_args()

    create_manifest_parallel(args.dataset, args.output, args.workers)


if __name__ == "__main__":
    main()
