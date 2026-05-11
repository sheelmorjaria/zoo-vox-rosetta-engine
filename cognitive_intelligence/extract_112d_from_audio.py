#!/usr/bin/env python3
"""
Extract 112D RosettaFeatures from Egyptian Fruit Bat Audio Files

Processes audio files from:
  /mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio/

Outputs:
  - features_112d.npy: (N, 112) array of features
  - metadata.json: File names, sample rates, durations

Features:
  - Parallel processing for speed
  - Resume capability (skips already processed)
  - Incremental saves
  - Progress tracking

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

import json
import logging
import multiprocessing as mp
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.feature_pipeline import RosettaFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AudioMetadata:
    """Metadata for a single audio file."""
    file_name: str
    sample_rate: int
    duration_ms: float
    num_samples: int
    error: Optional[str] = None


def load_and_convert_audio(file_path: Path) -> Tuple[np.ndarray, int, float]:
    """
    Load audio file and convert to float32 mono.

    Returns:
        audio: Normalized audio samples
        sample_rate: Sample rate in Hz
        duration_ms: Duration in milliseconds
    """
    try:
        sample_rate, audio = wavfile.read(file_path)

        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128.0) / 128.0
        else:
            audio = audio.astype(np.float32)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        duration_ms = len(audio) * 1000.0 / sample_rate

        return audio, sample_rate, duration_ms

    except Exception as e:
        logger.warning(f"Error loading {file_path}: {e}")
        raise


def extract_single_audio(args: Tuple[Path, int]) -> Tuple[Optional[np.ndarray], AudioMetadata]:
    """
    Extract features from a single audio file.

    Args:
        args: (file_path, target_sample_rate)

    Returns:
        features_112d: 112D feature vector or None if error
        metadata: AudioMetadata
    """
    file_path, target_sr = args

    try:
        # Load audio
        audio, sample_rate, duration_ms = load_and_convert_audio(file_path)

        # Resample if needed
        if sample_rate != target_sr:
            audio = resample_poly(audio, target_sr, sample_rate)

        # Extract features
        extractor = RosettaFeatureExtractor(sample_rate=target_sr)
        features = extractor.extract(audio, target_sr)

        metadata = AudioMetadata(
            file_name=file_path.name,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            num_samples=len(audio)
        )

        return features, metadata

    except Exception as e:
        metadata = AudioMetadata(
            file_name=file_path.name,
            sample_rate=0,
            duration_ms=0.0,
            num_samples=0,
            error=str(e)
        )
        return None, metadata


def extract_112d_features(
    audio_dir: Path,
    output_dir: Path,
    target_sample_rate: int = 48000,
    num_workers: Optional[int] = None,
    max_files: Optional[int] = None,
    resume_from: Optional[Path] = None,
):
    """
    Extract 112D RosettaFeatures from all audio files.

    Args:
        audio_dir: Directory containing .wav files
        output_dir: Output directory for features and metadata
        target_sample_rate: Target sample rate for extraction
        num_workers: Number of parallel workers (default: CPU count)
        max_files: Maximum number of files to process (for testing)
        resume_from: Path to existing .npy file to resume from
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    audio_files = sorted(audio_dir.glob("*.wav"))
    logger.info(f"Found {len(audio_files):,} audio files")

    if max_files:
        audio_files = audio_files[:max_files]
        logger.info(f"Limited to {max_files:,} files for testing")

    # Check for resume
    existing_features = None
    existing_metadata = []
    start_idx = 0

    if resume_from and resume_from.exists():
        logger.info(f"Resuming from {resume_from}")
        existing_features = np.load(resume_from)
        start_idx = len(existing_features)

        metadata_path = resume_from.parent / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)

        logger.info(f"Already processed {start_idx:,} files")

    # Setup parallel processing
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    logger.info(f"Using {num_workers} workers")

    # Prepare arguments
    args_list = [(f, target_sample_rate) for f in audio_files[start_idx:]]

    # Extract features with progress bar
    features_list = []
    metadata_list = []

    if existing_features is not None:
        features_list = [existing_features[i] for i in range(len(existing_features))]
        metadata_list = existing_metadata

    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(extract_single_audio, args_list),
            total=len(args_list),
            desc="Extracting features",
            unit="files"
        ))

    # Process results - filter to only 112D features
    errors = 0
    shape_mismatch = 0
    valid_features = []

    for features, metadata in results:
        metadata_dict = asdict(metadata)
        metadata_list.append(metadata_dict)

        if features is not None:
            if isinstance(features, np.ndarray) and features.shape == (112,):
                valid_features.append(features)
            else:
                shape_mismatch += 1
                logger.warning(f"Invalid feature shape: {metadata_dict['file_name']} -> {features.shape if features is not None else 'None'}")
        else:
            errors += 1

    # Convert to numpy array
    features_array = np.array(valid_features, dtype=np.float32)
    logger.info(f"Extracted {len(features_array):,} valid feature vectors")
    logger.info(f"  Errors: {errors}, Shape mismatches: {shape_mismatch}")

    # Save features
    features_path = output_dir / "features_112d.npy"
    np.save(features_path, features_array)
    logger.info(f"Saved features to {features_path}")

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "num_files": len(audio_files),
            "num_features": len(features_array),
            "feature_dim": 112,
            "target_sample_rate": target_sample_rate,
            "files": metadata_list,
        }, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    return features_array


def main():
    """Main entry point."""
    data_dir = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats")
    audio_dir = data_dir / "audio"
    output_dir = Path("data/bat_features_112d")

    print("=" * 60)
    print("Egyptian Fruit Bat 112D Feature Extraction")
    print("=" * 60)
    print(f"Audio directory: {audio_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Quick demo: extract 1000 files
    print("\nQuick demo: extracting 1,000 files (from 91,080 available)")
    print("For full extraction, set max_files=None\n")

    features = extract_112d_features(
        audio_dir=audio_dir,
        output_dir=output_dir,
        target_sample_rate=48000,
        num_workers=4,  # Adjust based on your CPU
        max_files=1000,  # Set to None for all files
    )

    print(f"\n✓ Extraction complete!")
    print(f"  Shape: {features.shape}")
    print(f"  Size: {features.nbytes / 1024 / 1024:.1f} MB")
    print(f"\nOutput files:")
    print(f"  - {output_dir}/features_112d.npy")
    print(f"  - {output_dir}/metadata.json")


if __name__ == "__main__":
    main()
