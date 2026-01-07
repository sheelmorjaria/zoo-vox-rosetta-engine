#!/usr/bin/env python3
"""
Enhanced Bat Import with Grammar/Syntax Metadata

This script imports Egyptian fruit bat vocalizations AND captures grammar/syntax
metadata from individual audio files, including:
1. Phrase sequences within vocalizations
2. Ascending/descending F0 patterns
3. Phrase transition patterns
4. Repetition patterns
5. Compositional structure metadata
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

# Add URS path
urs_path = str(Path(__file__).parent.parent / "analysis" / "rosetta_stone")
sys.path.insert(0, urs_path)

from universal_rosetta_stone import Modality, PhraseSignature

# Configuration
ANNOTATIONS_PATH = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/annotations.csv"
AUDIO_DIR = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio"
OUTPUT_PATH = "/home/sheel/birdsong_analysis/src/bat_database_with_syntax.json"
AUDIO_LIBRARY_DIR = "/home/sheel/birdsong_analysis/src/audio_library/bat"
AUDIO_INDEX_PATH = "/home/sheel/birdsong_analysis/src/audio_library/bat_audio_index.json"
CHECKPOINT_PATH = "/home/sheel/birdsong_analysis/src/bat_import_checkpoint.json"
SAMPLE_RATE = 22050
NUM_WORKERS = max(1, cpu_count() - 1)  # Parallel processing
BATCH_SIZE = 100  # Increased for speed
MAX_FILES = 5000  # Limit for processing
EXPORT_AUDIO_SEGMENTS = True  # Export individual audio segments
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N batches
ENABLE_CHECKPOINTING = True  # Enable checkpoint/resume


def load_and_segment_audio(file_name: str) -> Tuple[str, Dict, str, List[Tuple]]:
    """
    Load audio file, segment into phrases, extract syntax metadata.

    Args:
        file_name: Audio filename (e.g., "0.wav", "123.wav")

    Returns:
        Tuple of (file_path, syntax_metadata, file_name, audio_segments)
    """
    try:
        # Load audio - file_name already includes .wav extension
        audio_path = Path(AUDIO_DIR) / file_name

        if not audio_path.exists():
            return None

        audio, sr = sf.read(str(audio_path))

        # Convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != SAMPLE_RATE:
            from scipy import signal

            num_samples = int(len(audio) * SAMPLE_RATE / sr)
            audio = signal.resample(audio, num_samples)

        if len(audio) < SAMPLE_RATE * 0.1:  # Too short
            return None

        # Segment into phrases using energy/F0 analysis
        segments = segment_into_phrases(audio)

        if not segments:
            return None

        # Extract syntax metadata
        syntax_metadata = extract_syntax_metadata(segments, audio)

        # Return audio segments for export
        return (str(audio_path), syntax_metadata, file_name, segments)

    except Exception:
        return None


def segment_into_phrases(audio: np.ndarray) -> List[Tuple[np.ndarray, Dict]]:
    """Segment audio into atomic phrases using energy and F0 analysis."""
    segments = []

    # Energy-based segmentation
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks, hilbert

    envelope = np.abs(hilbert(audio))
    smoothed = gaussian_filter1d(envelope, sigma=int(SAMPLE_RATE * 0.002))

    # Find peaks (phrase onsets)
    threshold = np.mean(smoothed) + 0.2 * np.std(smoothed)
    min_distance = int(SAMPLE_RATE * 0.05)  # 50ms minimum

    peaks, _ = find_peaks(smoothed, height=threshold, distance=min_distance)

    # Segment between peaks
    for i in range(len(peaks)):
        onset = peaks[i]

        # Find offset
        if i < len(peaks) - 1:
            offset = peaks[i] + int((peaks[i + 1] - peaks[i]) * 0.7)
        else:
            remaining = smoothed[onset:]
            below_threshold = np.where(remaining < threshold)[0]
            if len(below_threshold) > 0:
                offset = onset + below_threshold[0]
            else:
                offset = len(audio)

        # Minimum duration
        if offset - onset < int(SAMPLE_RATE * 0.05):
            continue

        segment = audio[onset:offset]

        # Extract features - bat uses FM sweep modality
        try:
            sig = PhraseSignature(modality=Modality.FM_SWEEP, data=segment, sample_rate=SAMPLE_RATE)
            features = sig.features
            features["onset_ms"] = onset / SAMPLE_RATE * 1000
            features["offset_ms"] = offset / SAMPLE_RATE * 1000
            segments.append((segment, features))
        except:
            continue

    return segments


def extract_syntax_metadata(
    segments: List[Tuple[np.ndarray, Dict]], full_audio: np.ndarray
) -> Dict:
    """Extract grammar/syntax metadata from segmented vocalization."""

    if not segments:
        return {}

    # Extract phrase sequence
    phrase_sequence = []
    f0_sequence = []

    for audio, features in segments:
        # Generate phrase key for bat (FM sweep)
        f0_mean = int(features.get("f0_mean", 0) / 100) * 100
        f0_range = int(features.get("f0_range", 0) / 100) * 100
        duration_ms = int(features.get("duration_ms", 0) / 5) * 5

        phrase_key = f"FM_{int(f0_mean / 1000)}_{int(f0_range / 1000)}_DUR_{duration_ms}"

        phrase_sequence.append(phrase_key)
        f0_sequence.append(features.get("f0_mean", 0))

    # Analyze F0 contour
    f0_contour = analyze_f0_contour(f0_sequence)

    # Detect repetition
    has_repetition = len(phrase_sequence) != len(set(phrase_sequence))

    # Analyze transitions
    transitions = []
    for i in range(len(phrase_sequence) - 1):
        transitions.append((phrase_sequence[i], phrase_sequence[i + 1]))

    # Determine if compositional
    is_compositional = len(phrase_sequence) > 2 and not has_repetition

    # Overall vocalization features
    total_duration_ms = len(full_audio) / SAMPLE_RATE * 1000
    overall_f0_mean = (
        np.mean([f for f in f0_sequence if f > 0]) if any(f > 0 for f in f0_sequence) else 0
    )
    overall_f0_range = max(f0_sequence) - min(f0_sequence) if f0_sequence else 0

    return {
        "phrase_sequence": phrase_sequence,
        "num_phrases": len(phrase_sequence),
        "f0_sequence": f0_sequence,
        "f0_contour": f0_contour,
        "has_repetition": has_repetition,
        "transitions": transitions,
        "is_compositional": is_compositional,
        "total_duration_ms": total_duration_ms,
        "overall_f0_mean_hz": overall_f0_mean,
        "overall_f0_range_hz": overall_f0_range,
        "segment_details": [
            {
                "phrase_key": phrase_sequence[i],
                "f0_mean": f0_sequence[i],
                "onset_ms": seg[1]["onset_ms"],
                "offset_ms": seg[1]["offset_ms"],
                "duration_ms": seg[1]["duration_ms"],
            }
            for i, seg in enumerate(segments)
        ],
    }


def analyze_f0_contour(f0_sequence: List[float]) -> str:
    """Analyze the overall F0 contour pattern."""
    if len(f0_sequence) < 2:
        return "single"

    # Filter out zero F0 values
    valid_f0 = [f for f in f0_sequence if f > 0]

    if len(valid_f0) < 2:
        return "unmeasured"

    # Calculate trend
    first_half = valid_f0[: len(valid_f0) // 2]
    second_half = valid_f0[len(valid_f0) // 2 :]

    mean_first = np.mean(first_half)
    mean_second = np.mean(second_half)

    diff = mean_second - mean_first
    range_val = max(valid_f0) - min(valid_f0)

    if range_val < 200:
        return "flat"
    elif diff > range_val * 0.3:
        return "ascending"
    elif diff < -range_val * 0.3:
        return "descending"
    else:
        return "complex"


def process_batch(batch: List[int]) -> List[Tuple]:
    """Process a batch of audio files."""
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(load_and_segment_audio, batch)
    return [r for r in results if r is not None]


def export_audio_segments(phrase_segments: Dict[str, List[Dict]]):
    """Export audio segments to WAV files."""
    from pathlib import Path

    output_path = Path(AUDIO_LIBRARY_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_index = {}

    for phrase_key, segments in phrase_segments.items():
        # Create directory for this phrase type
        phrase_dir = output_path / phrase_key
        phrase_dir.mkdir(exist_ok=True)

        # Export each occurrence
        phrase_index = []

        for k, segment_data in enumerate(segments):
            # Generate filename
            filename = f"{k + 1:04d}.wav"
            file_path = phrase_dir / filename

            # Save audio
            sf.write(str(file_path), segment_data["audio"], SAMPLE_RATE)

            # Add to index
            phrase_index.append(
                {
                    "filename": filename,
                    "relative_path": f"bat/{phrase_key}/{filename}",
                    "vocalization_id": segment_data["vocalization_id"],
                    "context": segment_data["context"],
                    "duration_ms": segment_data["duration_ms"],
                    "f0_mean": segment_data["f0_mean"],
                    "num_samples": len(segment_data["audio"]),
                }
            )

        audio_index[phrase_key] = {
            "phrase_key": phrase_key,
            "total_occurrences": len(phrase_index),
            "segments": phrase_index,
        }

    # Save audio index
    with open(AUDIO_INDEX_PATH, "w") as f:
        json.dump(
            {
                "export_date": datetime.now().isoformat(),
                "sample_rate": SAMPLE_RATE,
                "total_phrases": len(audio_index),
                "total_segments": sum(index["total_occurrences"] for index in audio_index.values()),
                "phrases": audio_index,
            },
            f,
            indent=2,
        )

    print(f"✅ Exported {len(audio_index)} phrase types")
    print(f"   Total segments: {sum(index['total_occurrences'] for index in audio_index.values())}")


def save_checkpoint(checkpoint_data: Dict):
    """Save checkpoint data to file."""
    try:
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"💾 Checkpoint saved: {checkpoint_data['processed_count']} files processed")
    except Exception as e:
        print(f"⚠️  Failed to save checkpoint: {e}")


def load_checkpoint() -> Dict:
    """Load checkpoint data from file."""
    try:
        if Path(CHECKPOINT_PATH).exists():
            with open(CHECKPOINT_PATH, "r") as f:
                checkpoint_data = json.load(f)
            print(
                f"📂 Checkpoint loaded: {checkpoint_data['processed_count']} files already processed"
            )
            return checkpoint_data
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
    return None


def import_bat_with_syntax_metadata(max_files: int = MAX_FILES):
    """Import bat data with grammar/syntax metadata."""

    print("=" * 80)
    print("ENHANCED BAT IMPORT WITH GRAMMAR/SYNTAX METADATA")
    print("=" * 80)

    # Load annotations
    print(f"\n📊 Loading annotations from {ANNOTATIONS_PATH}...")
    df = pd.read_csv(ANNOTATIONS_PATH)
    print(f"✅ Loaded {len(df)} annotations")

    # Get unique file names (column is "File Name" - values are like "0.wav", "123.wav")
    file_names = (
        df["File Name"].unique()
        if "File Name" in df.columns
        else [f"{i}.wav" for i in range(max_files)]
    )

    # Sample
    if max_files and len(file_names) > max_files:
        file_names = file_names[:max_files]

    print(f"\n🔍 Processing {len(file_names)} audio files...")

    # Check for existing checkpoint
    if ENABLE_CHECKPOINTING:
        checkpoint = load_checkpoint()
        if checkpoint:
            all_results = checkpoint["results"]
            processed_files = set(checkpoint["processed_files"])
            start_index = checkpoint["last_batch_index"]
            print(
                f"📂 Resuming from batch {start_index + 1}, {len(all_results)} files already processed"
            )
        else:
            all_results = []
            processed_files = set()
            start_index = 0
    else:
        all_results = []
        processed_files = set()
        start_index = 0

    # Process in batches
    total_batches = (len(file_names) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(start_index, total_batches):
        i = batch_idx * BATCH_SIZE
        batch = file_names[i : i + BATCH_SIZE]

        # Skip already processed files
        new_batch = [f for f in batch if f not in processed_files]

        if not new_batch:
            print(f"  Batch {batch_idx + 1}/{total_batches}: skipping (already processed)")
            continue

        batch_results = process_batch(new_batch)
        all_results.extend(batch_results)
        processed_files.update(new_batch)

        print(
            f"  Batch {batch_idx + 1}/{total_batches}: processed {len(batch_results)} new files (total: {len(all_results)})"
        )

        # Save checkpoint periodically
        if ENABLE_CHECKPOINTING and (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
            # Convert results to serializable format (without audio data for checkpoint)
            checkpoint_results = []
            for result in all_results:
                if len(result) >= 4:
                    audio_path, syntax_meta, file_name, _ = result
                    checkpoint_results.append((audio_path, syntax_meta, file_name))

            checkpoint_data = {
                "results": checkpoint_results,
                "processed_files": list(processed_files),
                "last_batch_index": batch_idx + 1,
                "processed_count": len(all_results),
                "timestamp": datetime.now().isoformat(),
            }
            save_checkpoint(checkpoint_data)

    print(f"\n✅ Successfully processed {len(all_results)} vocalizations with syntax metadata")

    # Create export structure
    print("\n📊 Creating database with syntax metadata...")

    vocalizations = []
    phrase_segments = defaultdict(list)  # For audio export

    # Determine if we loaded from checkpoint (no audio segments available)
    from_checkpoint = ENABLE_CHECKPOINTING and Path(CHECKPOINT_PATH).exists() and start_index > 0

    if from_checkpoint:
        print("⚠️  Loaded from checkpoint - audio segments not available for export")
        print("   Delete checkpoint to re-process with audio segment export:")
        print(f"   rm {CHECKPOINT_PATH}")

    for vocalization_id, result in enumerate(all_results):
        # Handle both full results (4-tuple with audio) and checkpoint results (3-tuple without audio)
        if len(result) == 4:
            audio_path, syntax_meta, file_name, audio_segments = result
        else:
            audio_path, syntax_meta, file_name = result
            audio_segments = None

        # Look up context from annotations (column is "File Name")
        context_row = df[df["File Name"] == file_name]

        if not context_row.empty:
            context_code = context_row.iloc[0]["Context"]
            context_name = f"context_{context_code}"
        else:
            context_name = "unknown"

        vocalizations.append(
            {
                "vocalization_id": vocalization_id,
                "file_path": audio_path,
                "context": context_name,
                "syntax_metadata": syntax_meta,
            }
        )

        # Store audio segments for export (only if not from checkpoint)
        if EXPORT_AUDIO_SEGMENTS and audio_segments is not None:
            for seg_idx, (seg_audio, seg_features) in enumerate(audio_segments):
                if seg_idx < len(syntax_meta["segment_details"]):
                    phrase_key = syntax_meta["segment_details"][seg_idx]["phrase_key"]

                    phrase_segments[phrase_key].append(
                        {
                            "audio": seg_audio,
                            "vocalization_id": vocalization_id,
                            "context": context_name,
                            "onset_ms": seg_features["onset_ms"],
                            "offset_ms": seg_features["offset_ms"],
                            "duration_ms": seg_features["duration_ms"],
                            "f0_mean": seg_features.get("f0_mean", 0),
                        }
                    )

    # Build phrase library
    phrase_library = defaultdict(lambda: {"contexts": Counter(), "vocalization_ids": []})

    for vocalization in vocalizations:
        context = vocalization["context"]
        syntax_meta = vocalization["syntax_metadata"]

        for seg in syntax_meta.get("segment_details", []):
            phrase_key = seg["phrase_key"]
            phrase_library[phrase_key]["contexts"][context] += 1
            phrase_library[phrase_key]["vocalization_ids"].append(vocalization["vocalization_id"])

    # Export audio segments if enabled
    if EXPORT_AUDIO_SEGMENTS and phrase_segments:
        print(f"\n💾 Exporting audio segments to {AUDIO_LIBRARY_DIR}...")
        export_audio_segments(phrase_segments)

    # Export
    species_data = {
        "species": "egyptian_bat",
        "analysis_date": datetime.now().isoformat(),
        "total_vocalizations": len(vocalizations),
        "total_phrases": len(phrase_library),
        "vocalizations": vocalizations,
        "phrases": dict(phrase_library),
    }

    # Save
    export_data = {
        "export_date": datetime.now().isoformat(),
        "species_data": {"egyptian_bat": species_data},
    }

    print(f"\n💾 Saving to {OUTPUT_PATH}...")

    # Convert numpy types
    def convert_numpy_types(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, Counter):
            return dict(obj)
        return obj

    export_data = convert_numpy_types(export_data)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(export_data, f, indent=2)

    print("✅ Saved!")

    # Statistics
    print("\n📊 STATISTICS:")
    print(f"   Total vocalizations: {len(vocalizations)}")
    print(f"   Total phrase types: {len(phrase_library)}")

    # Syntax statistics
    f0_contours = Counter()
    compositional_count = 0

    for vocalization in vocalizations:
        syntax = vocalization["syntax_metadata"]
        f0_contours[syntax.get("f0_contour", "unknown")] += 1

        if syntax.get("is_compositional"):
            compositional_count += 1

    print("\n📊 SYNTAX STATISTICS:")
    print(f"   F0 contours: {dict(f0_contours)}")
    if len(vocalizations) > 0:
        print(
            f"   Compositional (3+ phrases): {compositional_count} ({compositional_count / len(vocalizations) * 100:.1f}%)"
        )

    return species_data


if __name__ == "__main__":
    import_bat_with_syntax_metadata(MAX_FILES)

    print("\n" + "=" * 80)
    print("✅ BAT IMPORT WITH SYNTAX METADATA COMPLETE!")
    print("=" * 80)
