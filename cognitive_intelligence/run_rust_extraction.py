#!/usr/bin/env python3
"""
Run Rust 112D Feature Extraction and Convert to .npy

Uses the Rust MicroDynamicsExtractor to extract proper 112D features,
then saves them in .npy format for training.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_rust_extraction(
    audio_dir: Path,
    output_dir: Path,
    max_files: int = 0,
):
    """
    Run the Rust 112D extraction binary.

    Args:
        audio_dir: Directory containing .wav files
        output_dir: Output directory for features
        max_files: Maximum number of audio files to process (0 = all files)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rust_binary = Path("technical_architecture/target/release/examples/bat_112d_extraction")

    if not rust_binary.exists():
        logger.error(f"Rust binary not found: {rust_binary}")
        logger.info("Build it with: cd technical_architecture && cargo build --release --example bat_112d_extraction")
        sys.exit(1)

    logger.info(f"Running Rust extraction (max_files={'all' if max_files == 0 else max_files})...")

    env = {**subprocess.os.environ}
    if max_files > 0:
        env["MAX_FILES"] = str(max_files)

    result = subprocess.run(
        [str(rust_binary)],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Rust extraction failed: {result.stderr}")
        sys.exit(1)

    logger.info("Rust extraction complete")
    return output_dir


def load_rust_extraction_output(extraction_dir: Path) -> Tuple[np.ndarray, List[dict]]:
    """
    Load Rust extraction results from JSON.

    Args:
        extraction_dir: Directory containing extraction_112d_labeled.json

    Returns:
        features: (N, 112) array of features
        metadata: List of metadata dicts
    """
    json_file = extraction_dir / "extraction_112d_labeled.json"

    if not json_file.exists():
        logger.error(f"Extraction output not found: {json_file}")
        sys.exit(1)

    logger.info(f"Loading features from {json_file}...")

    with open(json_file, 'r') as f:
        data = json.load(f)

    num_segments = data['num_segments']
    logger.info(f"Total segments: {num_segments:,}")

    # Extract features and metadata
    features_list = []
    metadata_list = []

    for segment in data['segments']:
        features_list.append(segment['features_112d'])
        metadata_list.append({
            'file_name': segment['file_name'],
            'start_sample': segment['start_sample'],
            'segment_index': segment['segment_index'],
            'cluster_id': segment.get('cluster_id'),
        })

    features_array = np.array(features_list, dtype=np.float32)

    logger.info(f"Loaded {len(features_array):,} feature vectors")
    logger.info(f"Shape: {features_array.shape}")

    return features_array, metadata_list


def main():
    """Main entry point."""
    data_dir = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats")
    audio_dir = data_dir / "audio"
    extraction_dir = data_dir / "extraction_112d"
    output_dir = Path("data/bat_features_112d_rust")

    # Configuration
    max_audio_files = None  # Max audio files to process (None for all)
    max_samples = 100000  # Max samples for training
    force_reextract = True  # Set to True to force re-extraction

    print("=" * 60)
    print("Rust 112D Feature Extraction + .npy Export")
    print("=" * 60)
    print(f"Audio directory: {audio_dir}")
    print(f"Extraction dir:  {extraction_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Step 1: Check if extraction exists, run Rust if needed
    json_file = extraction_dir / "extraction_112d_labeled.json"

    if force_reextract or not json_file.exists():
        if not json_file.exists():
            print(f"\n⚠ Extraction JSON not found: {json_file}")
            print("Running Rust extraction...")
        else:
            print(f"\n⚠ Force re-extraction requested")

        # Ensure audio directory exists
        if not audio_dir.exists():
            logger.error(f"Audio directory not found: {audio_dir}")
            sys.exit(1)

        # Count audio files
        audio_files = list(audio_dir.glob("*.wav"))
        print(f"Found {len(audio_files):,} audio files")

        if max_audio_files:
            print(f"Limiting to {max_audio_files:,} files (set max_audio_files=None for all)")

        # Run Rust extraction (0 means all files)
        run_rust_extraction(audio_dir, extraction_dir, max_files=max_audio_files or 0)
    else:
        print(f"\n✓ Using existing extraction: {json_file}")
        import os
        file_size_mb = os.path.getsize(json_file) / 1024 / 1024
        print(f"  Size: {file_size_mb:.1f} MB")

    # Step 2: Load and convert to .npy
    features, metadata = load_rust_extraction_output(extraction_dir)

    # Step 3: Sample for training
    if len(features) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(features), max_samples, replace=False)
        features = features[indices]
        metadata = [metadata[i] for i in indices]
        logger.info(f"Sampled to {max_samples:,} features")

    # Step 4: Save to .npy
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = output_dir / "features_112d.npy"
    np.save(features_path, features)
    logger.info(f"Saved features to {features_path}")

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "num_features": len(features),
            "feature_dim": 112,
            "samples": metadata[:100],  # Save first 100 for reference
        }, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    print("\n" + "=" * 60)
    print("✓ Extraction complete!")
    print(f"  Shape: {features.shape}")
    print(f"  Size: {features.nbytes / 1024 / 1024:.1f} MB")
    print(f"\nOutput files:")
    print(f"  - {features_path}")
    print(f"  - {metadata_path}")


if __name__ == "__main__":
    main()
