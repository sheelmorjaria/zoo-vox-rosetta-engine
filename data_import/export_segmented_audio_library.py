#!/usr/bin/env python3
"""
Export Segmented Audio Library for Concatenative Synthesis
==========================================================

This script processes the syntax-enhanced database and exports individual
phrase audio segments for use in concatenative synthesis.

Extracts:
1. Individual audio segments for each phrase occurrence
2. Organized by phrase type (phrase_key)
3. Stores as WAV files in structured directory
4. Creates index mapping phrase keys to audio files

Usage:
    python export_segmented_audio_library.py

Output:
    audio_library/
        marmoset/
            F0_8000_DUR_100_RANGE_200/
                001.wav  # occurrence 1
                002.wav  # occurrence 2
                ...
        audio_index.json  # mapping file
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'analysis' / 'rosetta_stone'))

# Configuration
SYNTAX_DATABASE_PATH = '/home/sheel/birdsong_analysis/src/vocalization_database_with_syntax.json'
OUTPUT_DIR = '/home/sheel/birdsong_analysis/src/audio_library'
AUDIO_INDEX_PATH = '/home/sheel/birdsong_analysis/src/audio_library/audio_index.json'
SAMPLE_RATE = 22050
EXPORT_LIMIT = None  # Set to limit number of vocalizations processed (None = all)


def load_syntax_database(db_path: str) -> Dict:
    """Load the syntax-enhanced database."""
    print(f"Loading syntax database from {db_path}...")

    with open(db_path, 'r') as f:
        db = json.load(f)

    vocalizations = db['species_data']['marmoset']['vocalizations']

    print(f"✅ Loaded {len(vocalizations)} vocalizations with syntax metadata")

    return vocalizations


def extract_segment_from_file(
    file_path: str,
    onset_ms: float,
    offset_ms: float,
    target_sr: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Extract audio segment from file based on timing.

    Args:
        file_path: Path to audio file
        onset_ms: Start time in milliseconds
        offset_ms: End time in milliseconds
        target_sr: Target sample rate

    Returns:
        Audio segment as numpy array
    """
    try:
        # Load audio
        audio, sr = sf.read(file_path)

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != target_sr:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples)

        # Convert timing to samples
        onset_sample = int(onset_ms / 1000 * target_sr)
        offset_sample = int(offset_ms / 1000 * target_sr)

        # Extract segment
        segment = audio[onset_sample:offset_sample]

        return segment

    except Exception as e:
        print(f"⚠️  Error extracting from {file_path}: {e}")
        return None


def export_segmented_audio_library(
    vocalizations: List[Dict],
    output_dir: str,
    export_limit: int = None
) -> Dict:
    """
    Export segmented audio library from syntax database.

    Creates:
    - Individual WAV files for each phrase segment
    - Organized by phrase type
    - Audio index mapping files to metadata

    Args:
        vocalizations: List of vocalizations with segment_details
        output_dir: Output directory path
        export_limit: Limit number of vocalizations to process

    Returns:
        Audio index dictionary
    """
    print("\n" + "=" * 80)
    print("EXPORTING SEGMENTED AUDIO LIBRARY")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group segments by phrase type
    phrase_segments = defaultdict(list)

    total_vocalizations = len(vocalizations)
    if export_limit:
        total_vocalizations = min(total_vocalizations, export_limit)

    print(f"\n🔍 Processing {total_vocalizations} vocalizations...")

    for i, vocalization in enumerate(vocalizations[:export_limit] if export_limit else vocalizations):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{total_vocalizations}...")

        file_path = vocalization['file_path']
        context = vocalization['context']
        syntax_metadata = vocalization['syntax_metadata']

        # Extract each segment
        for j, segment in enumerate(syntax_metadata.get('segment_details', [])):
            # Extract audio segment
            segment_audio = extract_segment_from_file(
                file_path,
                segment['onset_ms'],
                segment['offset_ms'],
                SAMPLE_RATE
            )

            if segment_audio is not None and len(segment_audio) > 0:
                phrase_key = segment['phrase_key']

                phrase_segments[phrase_key].append({
                    'audio': segment_audio,
                    'vocalization_id': vocalization['vocalization_id'],
                    'context': context,
                    'onset_ms': segment['onset_ms'],
                    'offset_ms': segment['offset_ms'],
                    'duration_ms': segment['duration_ms'],
                    'f0_mean': segment['f0_mean']
                })

    print(f"\n✅ Extracted {sum(len(segments) for segments in phrase_segments.values())} segments")
    print(f"   Unique phrase types: {len(phrase_segments)}")

    # Export audio files
    print(f"\n💾 Exporting audio files to {output_dir}...")

    audio_index = {}
    species_dir = output_path / 'marmoset'
    species_dir.mkdir(exist_ok=True)

    for phrase_key, segments in phrase_segments.items():
        # Create directory for this phrase type
        phrase_dir = species_dir / phrase_key
        phrase_dir.mkdir(exist_ok=True)

        # Export each occurrence
        phrase_index = []

        for k, segment_data in enumerate(segments):
            # Generate filename
            filename = f"{k+1:04d}.wav"
            file_path = phrase_dir / filename

            # Save audio
            sf.write(str(file_path), segment_data['audio'], SAMPLE_RATE)

            # Add to index
            phrase_index.append({
                'filename': filename,
                'relative_path': f'marmoset/{phrase_key}/{filename}',
                'vocalization_id': segment_data['vocalization_id'],
                'context': segment_data['context'],
                'duration_ms': segment_data['duration_ms'],
                'f0_mean': segment_data['f0_mean'],
                'num_samples': len(segment_data['audio'])
            })

        audio_index[phrase_key] = {
            'phrase_key': phrase_key,
            'total_occurrences': len(phrase_index),
            'segments': phrase_index
        }

    # Save audio index
    with open(AUDIO_INDEX_PATH, 'w') as f:
        json.dump({
            'export_date': str(Path(__file__).stat().st_mtime),
            'sample_rate': SAMPLE_RATE,
            'total_phrases': len(audio_index),
            'total_segments': sum(index['total_occurrences'] for index in audio_index.values()),
            'phrases': audio_index
        }, f, indent=2)

    print(f"✅ Exported {len(audio_index)} phrase types")
    print(f"   Total segments: {sum(index['total_occurrences'] for index in audio_index.values())}")
    print(f"   Audio index: {AUDIO_INDEX_PATH}")

    return audio_index


def analyze_audio_library(audio_index: Dict):
    """Analyze the exported audio library."""
    print("\n" + "=" * 80)
    print("AUDIO LIBRARY ANALYSIS")
    print("=" * 80)

    # Statistics
    phrase_counts = {}
    total_segments = 0
    duration_sum = 0

    for phrase_key, phrase_data in audio_index['phrases'].items():
        count = phrase_data['total_occurrences']
        phrase_counts[phrase_key] = count
        total_segments += count

        for segment in phrase_data['segments']:
            duration_sum += segment['duration_ms']

    print(f"\n📊 LIBRARY STATISTICS:")
    print(f"   Unique phrase types: {len(phrase_counts)}")
    print(f"   Total segments: {total_segments}")
    print(f"   Total duration: {duration_sum / 1000:.1f} seconds")
    print(f"   Mean segment duration: {duration_sum / total_segments:.1f} ms")

    # Phrase count distribution
    print(f"\n📊 PHRASE COUNT DISTRIBUTION:")

    count_distribution = defaultdict(int)
    for count in phrase_counts.values():
        if count == 1:
            count_distribution['1 occurrence'] += 1
        elif count <= 5:
            count_distribution['2-5 occurrences'] += 1
        elif count <= 10:
            count_distribution['6-10 occurrences'] += 1
        elif count <= 20:
            count_distribution['11-20 occurrences'] += 1
        else:
            count_distribution['20+ occurrences'] += 1

    for category, count in sorted(count_distribution.items()):
        print(f"   {category}: {count} phrase types")

    # Most common phrases
    print(f"\n📊 MOST COMMON PHRASES:")
    sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
    for phrase_key, count in sorted_phrases[:10]:
        print(f"   {phrase_key}: {count} occurrences")

    # Duration distribution
    print(f"\n📊 DURATION DISTRIBUTION:")

    all_durations = []
    for phrase_data in audio_index['phrases'].values():
        for segment in phrase_data['segments']:
            all_durations.append(segment['duration_ms'])

    all_durations = np.array(all_durations)

    print(f"   Min: {np.min(all_durations):.1f} ms")
    print(f"   Max: {np.max(all_durations):.1f} ms")
    print(f"   Mean: {np.mean(all_durations):.1f} ms")
    print(f"   Median: {np.median(all_durations):.1f} ms")
    print(f"   Std: {np.std(all_durations):.1f} ms")

    # Context distribution
    print(f"\n📊 CONTEXT DISTRIBUTION:")

    context_counts = defaultdict(int)
    for phrase_data in audio_index['phrases'].values():
        for segment in phrase_data['segments']:
            context_counts[segment['context']] += 1

    for context, count in sorted(context_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {context}: {count} segments")

    print("\n" + "=" * 80)


def main():
    """Main export function."""
    print("=" * 80)
    print("SEGMENTED AUDIO LIBRARY EXPORT")
    print("=" * 80)

    # Load syntax database
    vocalizations = load_syntax_database(SYNTAX_DATABASE_PATH)

    # Export segmented audio
    audio_index = export_segmented_audio_library(
        vocalizations,
        OUTPUT_DIR,
        EXPORT_LIMIT
    )

    # Analyze library
    analyze_audio_library({
        'phrases': audio_index
    })

    print("\n" + "=" * 80)
    print("✅ EXPORT COMPLETE!")
    print("=" * 80)
    print(f"\n📂 Audio library location: {OUTPUT_DIR}")
    print(f"📋 Audio index: {AUDIO_INDEX_PATH}")
    print(f"\n🎯 Ready for concatenative synthesis!")
    print(f"   • Load audio segments from library")
    print(f"   • Use phrase_sequence from syntax metadata")
    print(f"   • Concatenate with crossfades")
    print("=" * 80)


if __name__ == "__main__":
    main()
