#!/usr/bin/env python3
"""
Convert a representative subset of marmoset FLAC files to WAV format for lexicon_to_syntax analysis.

This script:
1. Reads the marmoset vocalization annotations
2. Samples a representative subset across different call types and dates
3. Converts FLAC files to WAV format using pydub/ffmpeg
"""

import random
from pathlib import Path

import pandas as pd

# Set random seed for reproducibility
random.seed(42)

# Paths
VOCALIZATIONS_DIR = Path.home() / "birdsong_analysis" / "data" / "Vocalizations"
ANNOTATIONS_FILE = VOCALIZATIONS_DIR / "Annotations.tsv"
OUTPUT_DIR = Path.home() / "birdsong_analysis" / "data" / "marmoset_wav_subset"


def load_annotations() -> pd.DataFrame:
    """Load the annotations file."""
    print(f"Loading annotations from {ANNOTATIONS_FILE}...")
    df = pd.read_csv(ANNOTATIONS_FILE, sep="\t")
    print(f"Loaded {len(df)} annotations")
    return df


def get_representative_subset(df: pd.DataFrame, n_files: int = 5000) -> pd.DataFrame:
    """
    Sample a representative subset of vocalizations across:
    - Different call types (Vocalization, Twitter, Tsik, Phee, Trill, Infant, Seep)
    - Different dates (spanning 2019-2023)
    """
    print(f"\nCreating representative subset of {n_files} files...")

    # Group by call type and sample proportionally
    call_types = df["label"].unique()
    print(f"Found {len(call_types)} call types: {list(call_types)}")

    sampled_dfs = []

    for call_type in call_types:
        # Get all files of this call type
        call_type_df = df[df["label"] == call_type].copy()

        # Sample proportionally based on call type frequency
        proportion = len(call_type_df) / len(df)
        n_samples = max(50, int(n_files * proportion))  # At least 50 per call type

        # Limit to available files
        n_samples = min(n_samples, len(call_type_df))

        # Sample evenly across different dates (parent_name)
        call_type_df["date"] = call_type_df["parent_name"].str.extract(r"(\d{4}_\d+)_\d+")

        # Get unique dates and sample from each
        unique_dates = call_type_df["date"].dropna().unique()
        samples_per_date = max(5, n_samples // len(unique_dates))

        for date in unique_dates:
            date_df = call_type_df[call_type_df["date"] == date]
            n_from_date = min(samples_per_date, len(date_df))
            sampled = date_df.sample(n=n_from_date, random_state=42)
            sampled_dfs.append(sampled)

    # Combine all samples
    result = pd.concat(sampled_dfs, ignore_index=True)

    # Shuffle the result
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Selected {len(result)} files")
    print("Call type distribution in subset:")
    print(result["label"].value_counts())

    return result


def convert_flac_to_wav(flac_path: Path, wav_path: Path, sample_rate: int = 48000) -> bool:
    """Convert a FLAC file to WAV format."""
    try:
        from pydub import AudioSegment

        # Load FLAC file
        audio = AudioSegment.from_file(flac_path, format="flac")

        # Set sample rate if needed
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)

        # Export as WAV
        audio.export(wav_path, format="wav")
        return True

    except Exception as e:
        print(f"Error converting {flac_path.name}: {e}")
        return False


def main():
    """Main conversion function."""
    print("=" * 80)
    print("Marmoset FLAC to WAV Converter for Lexicon-to-Syntax Analysis")
    print("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Load annotations
    df = load_annotations()

    # Get representative subset
    subset_df = get_representative_subset(df, n_files=5000)

    # Convert files
    print(f"\nConverting {len(subset_df)} FLAC files to WAV...")

    success_count = 0
    failed_count = 0

    for idx, row in subset_df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(subset_df)} files ({success_count} successful)")

        # Construct paths
        flac_path = VOCALIZATIONS_DIR / row["parent_name"] / row["file_name"]
        wav_path = OUTPUT_DIR / f"{row['parent_name']}_{row['file_name']}".replace(".flac", ".wav")

        # Skip if already converted
        if wav_path.exists():
            success_count += 1
            continue

        # Convert
        if convert_flac_to_wav(flac_path, wav_path):
            success_count += 1
        else:
            failed_count += 1

    print("\nConversion complete!")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Create a manifest file for the Rust example
    manifest_file = OUTPUT_DIR / "manifest.txt"
    with open(manifest_file, "w") as f:
        for wav_file in sorted(OUTPUT_DIR.glob("*.wav")):
            f.write(f"{wav_file}\n")

    print(f"  Manifest file: {manifest_file}")
    print(f"  Total WAV files: {len(list(OUTPUT_DIR.glob('*.wav')))}")


if __name__ == "__main__":
    main()
