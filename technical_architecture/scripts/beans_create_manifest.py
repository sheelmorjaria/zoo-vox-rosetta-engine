#!/usr/bin/env python3
"""
Create BEANS-Zero manifest for Rust training pipeline.

Converts HuggingFace dataset format to JSON manifest expected by train_beans_models_112d.rs
"""

import argparse
import json
import os
from pathlib import Path

try:
    from datasets import load_from_disk
    import numpy as np
    from scipy.io import wavfile
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install with: pip install datasets numpy scipy")
    exit(1)


def create_manifest(dataset_path: Path, output_path: Path, max_samples: int = None):
    """Create manifest from HuggingFace dataset."""
    print(f"Loading dataset from: {dataset_path}")
    ds = load_from_disk(str(dataset_path))

    print(f"Dataset: {len(ds)} samples")
    print(f"Columns: {ds.column_names}")

    # Create audio output directory
    audio_dir = output_path.parent / "beans_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "BEANS-Zero",
        "n_samples": 0,
        "samples": []
    }

    n_samples = min(len(ds), max_samples) if max_samples else len(ds)

    print(f"Processing {n_samples} samples...")

    for i in range(n_samples):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{n_samples} samples")

        sample = ds[i]

        # Get audio data - BEANS-Zero stores as list of floats
        audio_data = sample.get("audio", [])

        if not audio_data:
            continue

        # Handle different audio formats
        if isinstance(audio_data, dict):
            # Audio type with 'array' and 'sampling_rate'
            array = audio_data.get("array", [])
            sample_rate = audio_data.get("sampling_rate", 44100)
        elif isinstance(audio_data, (list, np.ndarray)):
            # Direct array
            array = audio_data
            sample_rate = 44100
        else:
            continue

        # Save as WAV file
        audio_filename = f"sample_{i:06d}.wav"
        audio_path = audio_dir / audio_filename

        # Convert to numpy array
        audio_np = np.array(array, dtype=np.float32)

        # Normalize if needed
        max_val = max(abs(audio_np.max()), abs(audio_np.min()))
        if max_val > 1.0:
            audio_np = audio_np / max_val

        # Convert to int16 for WAV
        audio_int16 = (audio_np * 32767).astype(np.int16)

        # Resample to 44100 Hz if needed
        if sample_rate != 44100:
            # Simple linear interpolation resampling
            ratio = 44100 / sample_rate
            new_len = int(len(audio_int16) * ratio)
            indices = np.linspace(0, len(audio_int16) - 1, new_len)
            audio_int16 = audio_int16[indices.astype(int)]
            sample_rate = 44100

        wavfile.write(str(audio_path), sample_rate, audio_int16)

        # Get label
        output_label = sample.get("output", "unknown")
        if output_label is None:
            output_label = "unknown"

        manifest["samples"].append({
            "audio_file": str(audio_path),
            "n_samples": len(array),
            "labels": {
                "output": output_label,
                "task": sample.get("dataset_name", "unknown")
            }
        })

    manifest["n_samples"] = len(manifest["samples"])

    # Save manifest
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCreated manifest: {output_path}")
    print(f"  Total samples: {manifest['n_samples']}")
    print(f"  Audio directory: {audio_dir}")

    # Print label distribution
    labels = [s["labels"]["output"] for s in manifest["samples"]]
    unique_labels = set(labels)
    print(f"  Unique labels: {len(unique_labels)}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Create BEANS-Zero manifest")
    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        default=Path("beans_zero_data/beans_zero_test"),
        help="Path to HuggingFace dataset"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("beans_zero_manifest.json"),
        help="Output manifest path"
    )
    parser.add_argument(
        "--max-samples", "-m",
        type=int,
        default=None,
        help="Maximum samples to process (default: all)"
    )

    args = parser.parse_args()

    create_manifest(args.dataset, args.output, args.max_samples)


if __name__ == "__main__":
    main()
