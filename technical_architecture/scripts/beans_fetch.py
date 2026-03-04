#!/usr/bin/env python3
"""
BEANS-Zero Dataset Fetcher
===========================

Downloads the BEANS-Zero (Benchmark for Animal Sound) dataset from HuggingFace.

BEANS-Zero is a novel benchmark for zero-shot classification of animal sounds,
including unseen species. It's part of the NatureLM-audio project.

Usage:
    python scripts/beans_fetch.py [--output-dir OUTPUT_DIR] [--split SPLIT]

Options:
    --output-dir    Directory to save the dataset (default: ./beans_zero_data)
    --split         Dataset split to download (default: test)
    --stream        Stream dataset instead of downloading (for preview)
    --list          List available component datasets and exit

Source: https://huggingface.co/datasets/EarthSpeciesProject/BEANS-Zero
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from datasets import load_dataset
    import numpy as np
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install with: pip install datasets numpy")
    sys.exit(1)


DATASET_NAME = "EarthSpeciesProject/BEANS-Zero"


def list_component_datasets(ds):
    """List all component datasets in BEANS-Zero."""
    print("\n" + "="*60)
    print("BEANS-Zero Component Datasets")
    print("="*60)

    components, counts = np.unique(ds["dataset_name"], return_counts=True)

    print(f"\n{'Dataset':<40} {'Samples':>10}")
    print("-"*52)

    total = 0
    for component, count in sorted(zip(components, counts), key=lambda x: -x[1]):
        print(f"{component:<40} {count:>10,}")
        total += count

    print("-"*52)
    print(f"{'TOTAL':<40} {total:>10,}")
    print()


def preview_sample(ds, idx=0):
    """Preview a sample from the dataset."""
    print("\n" + "="*60)
    print(f"Sample Preview (index {idx})")
    print("="*60)

    sample = ds[idx]

    print(f"\nDataset:        {sample.get('dataset_name', 'N/A')}")
    print(f"Sample ID:      {sample.get('id', 'N/A')}")
    print(f"Created:        {sample.get('created_at', 'N/A')}")
    print(f"File Name:      {sample.get('file_name', 'N/A')}")

    # Parse metadata
    metadata = sample.get('metadata', '{}')
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}

    print(f"Duration:       {metadata.get('duration', 'N/A')} seconds")
    print(f"Sample Rate:    {metadata.get('sample_rate', 'N/A')} Hz")

    # Audio info
    audio = sample.get('audio', {})
    if isinstance(audio, dict):
        print(f"Audio Array:    Shape {audio.get('array', []).shape if hasattr(audio.get('array', []), 'shape') else 'N/A'}")
    else:
        print(f"Audio Type:     {type(audio).__name__}")

    # Instruction/output
    print(f"\nInstruction:\n  {sample.get('instruction_text', 'N/A')}")
    print(f"\nExpected Output:\n  {sample.get('output', 'N/A')}")

    print()


def download_dataset(output_dir: Path, split: str = "test"):
    """Download the BEANS-Zero dataset."""
    print(f"\nDownloading BEANS-Zero dataset (split: {split})...")
    print(f"Output directory: {output_dir}")
    print(f"Source: {DATASET_NAME}")
    print()

    # Load dataset
    ds = load_dataset(DATASET_NAME, split=split)

    print(f"Loaded {len(ds):,} samples")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save dataset to disk
    save_path = output_dir / f"beans_zero_{split}"
    print(f"\nSaving to: {save_path}")

    ds.save_to_disk(str(save_path))

    # Also save metadata
    metadata = {
        "dataset_name": DATASET_NAME,
        "split": split,
        "total_samples": len(ds),
        "source_url": f"https://huggingface.co/datasets/{DATASET_NAME}",
        "component_datasets": list(set(ds["dataset_name"]))
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")

    return ds


def stream_dataset():
    """Stream dataset for preview without full download."""
    print(f"\nStreaming BEANS-Zero dataset...")
    print(f"Source: {DATASET_NAME}")
    print()

    ds = load_dataset(DATASET_NAME, split="test", streaming=True)

    print("First 3 samples:")
    print("-" * 40)

    for i, sample in enumerate(ds):
        if i >= 3:
            break
        print(f"\nSample {i+1}:")
        print(f"  Dataset: {sample.get('dataset_name', 'N/A')}")
        print(f"  Instruction: {sample.get('instruction_text', 'N/A')[:100]}...")
        print(f"  Output: {sample.get('output', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Download BEANS-Zero benchmark dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download test split to default directory
  python scripts/beans_fetch.py

  # Download to custom directory
  python scripts/beans_fetch.py --output-dir /data/beans_zero

  # List component datasets
  python scripts/beans_fetch.py --list

  # Stream preview without downloading
  python scripts/beans_fetch.py --stream
        """
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./beans_zero_data"),
        help="Directory to save the dataset (default: ./beans_zero_data)"
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        default="test",
        help="Dataset split to download (default: test)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream dataset instead of downloading (for preview)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available component datasets and exit"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview first sample after download"
    )

    args = parser.parse_args()

    print("="*60)
    print("BEANS-Zero Dataset Fetcher")
    print("="*60)

    if args.stream:
        stream_dataset()
        return

    # Download dataset
    ds = download_dataset(args.output_dir, args.split)

    # List components if requested
    if args.list:
        list_component_datasets(ds)

    # Preview sample if requested
    if args.preview:
        preview_sample(ds, 0)

    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print(f"\nDataset saved to: {args.output_dir}")
    print(f"Load with: ds = load_from_disk('{args.output_dir}/beans_zero_{args.split}')")


if __name__ == "__main__":
    main()
