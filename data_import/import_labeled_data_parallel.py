#!/usr/bin/env python3
"""
Optimized Import of Labeled Vocalizations with Behavioral Contexts

This script uses multiprocessing to speed up the import of labeled vocalizations.
It processes audio files in parallel to extract micro-dynamics features.

Speed improvements:
- Uses multiprocessing for parallel feature extraction
- Processes files in batches to reduce overhead
- Progress tracking with status updates
- Configurable sample size for testing
"""

import json
import pandas as pd
import numpy as np
import sys
import soundfile as sf
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from datetime import datetime
from multiprocessing import Pool, cpu_count
import functools
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

# Add URS path
urs_path = str(Path(__file__).parent.parent / 'analysis' / 'rosetta_stone')
sys.path.insert(0, urs_path)

from universal_rosetta_stone import PhraseSignature, Modality

# Configuration
SAMPLE_RATE = 22050
NUM_WORKERS = max(1, cpu_count() - 1)  # Leave one CPU free
BATCH_SIZE = 100


def load_and_extract_audio(args: Tuple[str, str]) -> Tuple[str, Dict, str]:
    """
    Load audio file and extract features.

    Args:
        args: (audio_path, context_name)

    Returns:
        Tuple of (file_path, features_dict, context_name) or None if error
    """
    audio_path, context_name = args

    try:
        # Load audio
        audio, sr = sf.read(audio_path)

        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != SAMPLE_RATE:
            from scipy import signal
            num_samples = int(len(audio) * SAMPLE_RATE / sr)
            audio = signal.resample(audio, num_samples)

        # Skip if too short
        if len(audio) < SAMPLE_RATE * 0.1:  # Less than 100ms
            return None

        # Extract features
        sig = PhraseSignature(modality=Modality.HARMONIC, data=audio, sample_rate=SAMPLE_RATE)

        return (audio_path, sig.features, context_name)

    except Exception as e:
        return None


def process_batch(batch: List[Tuple[str, str]]) -> List[Tuple]:
    """Process a batch of audio files in parallel."""
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(load_and_extract_audio, batch)
    return [r for r in results if r is not None]


def import_marmoset_data(
    annotations_path: str = '/home/sheel/birdsong_analysis/Annotations.tsv',
    vocalizations_dir: str = '/home/sheel/birdsong_analysis/data/Vocalizations',
    output_path: str = '/home/sheel/birdsong_analysis/src/vocalization_database_with_contexts.json',
    max_files: int = 5000
):
    """Import marmoset vocalizations with behavioral contexts."""

    print("=" * 80)
    print("OPTIMIZED MARMOSET VOCALIZATION IMPORT")
    print("=" * 80)
    print(f"\nWorkers: {NUM_WORKERS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max files: {max_files}")

    # Load annotations
    print(f"\n📊 Loading annotations from {annotations_path}...")
    df = pd.read_csv(annotations_path, sep='\t')
    print(f"✅ Loaded {len(df)} annotations")

    # Sample if needed
    if max_files and len(df) > max_files:
        print(f"⚠️  Sampling to {max_files} files for demonstration...")
        df = df.sample(n=max_files, random_state=42)

    # Prepare file paths
    print(f"\n🔍 Preparing file list...")
    file_tasks = []

    for _, row in df.iterrows():
        parent_name = str(row['parent_name']).replace(' ', '_')
        file_name = str(row['file_name'])
        label = str(row['label'])

        # Normalize label to context
        context_map = {
            'Tsik': 'tsik', 'Trill': 'trill', 'Twitter': 'twitter',
            'Phee': 'phee', 'Seep': 'seep', 'Infant': 'infant',
            'Infant_cry': 'infant', 'Vocalization': 'vocalization'
        }
        context_name = context_map.get(label, label.lower())

        file_path = Path(vocalizations_dir) / parent_name / file_name

        if file_path.exists():
            file_tasks.append((str(file_path), context_name))

    print(f"✅ Found {len(file_tasks)} audio files")

    # Process in batches
    print(f"\n⚙️  Processing in {len(file_tasks)//BATCH_SIZE + 1} batches...")

    all_results = []

    for i in range(0, len(file_tasks), BATCH_SIZE):
        batch = file_tasks[i:i+BATCH_SIZE]
        batch_results = process_batch(batch)
        all_results.extend(batch_results)

        print(f"  Batch {i//BATCH_SIZE + 1}: processed {len(batch_results)} files")

    print(f"\n✅ Successfully processed {len(all_results)} files")

    # Group into phrases
    print(f"\n📊 Grouping into phrases...")

    phrase_library = defaultdict(lambda: {
        'contexts': Counter(),
        'features_list': []
    })

    for audio_path, features, context_name in all_results:
        # Generate phrase key from features
        mean_f0 = int(features.get('f0_mean', 0) / 100) * 100
        f0_range = int(features.get('f0_range', 0) / 100) * 100
        duration_ms = int(features.get('duration_ms', 0) / 5) * 5

        phrase_key = f"F0_{mean_f0}_DUR_{duration_ms}_RANGE_{f0_range}"

        phrase_library[phrase_key]['contexts'][context_name] += 1
        phrase_library[phrase_key]['features_list'].append(features)

    print(f"✅ Created {len(phrase_library)} phrase types")

    # Create export structure
    species_data = {
        'species': 'marmoset',
        'analysis_date': datetime.now().isoformat(),
        'total_phrases': len(phrase_library),
        'phrases': {}
    }

    for phrase_key, phrase_data in phrase_library.items():
        features = phrase_data['features_list'][0]

        # Determine modality
        spectral_flatness = features.get('spectral_flatness', 0)
        if spectral_flatness > 0.5:
            modality = 'transient'
        elif features.get('vibrato_rate_hz', 0) > 5:
            modality = 'rhythmic'
        else:
            modality = 'harmonic'

        # Create contexts list
        contexts = []
        total_ctx = sum(phrase_data['contexts'].values())

        for ctx_name, count in phrase_data['contexts'].most_common():
            contexts.append({
                'context_name': ctx_name,
                'count': count,
                'percentage': (count / total_ctx * 100) if total_ctx > 0 else 0
            })

        species_data['phrases'][phrase_key] = {
            'phrase_key': phrase_key,
            'signature': f"{modality}_{phrase_key}",
            'species': 'marmoset',
            'modality': modality,
            'acoustic_features': {k: float(v) for k, v in features.items()},
            'total_occurrences': len(phrase_data['features_list']),
            'contexts': contexts,
            'social_contexts': {},
            'is_compositional': False,
            'phrase_components': []
        }

    # Show statistics
    print(f"\n📊 STATISTICS:")
    print(f"   Total phrases: {len(species_data['phrases'])}")
    print(f"   Total occurrences: {sum(p['total_occurrences'] for p in species_data['phrases'].values())}")

    # Context distribution
    all_contexts = Counter()
    for phrase in species_data['phrases'].values():
        for ctx in phrase['contexts']:
            all_contexts[ctx['context_name']] += ctx['count']

    print(f"\n📊 CONTEXT DISTRIBUTION:")
    for ctx, count in all_contexts.most_common():
        pct = (count / sum(all_contexts.values())) * 100
        print(f"   {ctx:<20} {count:>8} ({pct:>5.1f}%)")

    # Save
    export_data = {
        'export_date': datetime.now().isoformat(),
        'species_data': {'marmoset': species_data}
    }

    print(f"\n💾 Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"✅ Saved!")

    return species_data


def import_bat_data(
    annotations_path: str = '/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/annotations.csv',
    audio_dir: str = '/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio',
    output_path: str = '/home/sheel/birdsong_analysis/src/vocalization_database_with_bat_contexts.json',
    max_files: int = 5000
):
    """Import Egyptian fruit bat vocalizations with behavioral contexts."""

    print("\n" + "=" * 80)
    print("OPTIMIZED EGYPTIAN FRUIT BAT VOCALIZATION IMPORT")
    print("=" * 80)
    print(f"\nWorkers: {NUM_WORKERS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max files: {max_files}")

    # Load annotations
    print(f"\n📊 Loading annotations from {annotations_path}...")
    df = pd.read_csv(annotations_path)
    print(f"✅ Loaded {len(df)} annotations")

    print(f"\n📊 Columns: {df.columns.tolist()}")

    # Show context distribution
    print(f"\n📊 CONTEXT DISTRIBUTION:")
    context_counts = df['Context'].value_counts()
    for ctx, count in context_counts.head(10).items():
        pct = (count / len(df)) * 100
        print(f"   Context {ctx:<5} {count:>8} ({pct:>5.1f}%)")

    # Sample if needed
    if max_files and len(df) > max_files:
        print(f"\n⚠️  Sampling to {max_files} files for demonstration...")
        df = df.sample(n=max_files, random_state=42)

    # Prepare file paths
    print(f"\n🔍 Preparing file list...")
    file_tasks = []

    for _, row in df.iterrows():
        file_name = str(row['File Name'])
        context_code = int(row['Context'])

        file_path = Path(audio_dir) / file_name

        if file_path.exists():
            file_tasks.append((str(file_path), f"context_{context_code}"))

    print(f"✅ Found {len(file_tasks)} audio files")

    # Process in batches
    print(f"\n⚙️  Processing in {len(file_tasks)//BATCH_SIZE + 1} batches...")

    all_results = []

    for i in range(0, len(file_tasks), BATCH_SIZE):
        batch = file_tasks[i:i+BATCH_SIZE]
        batch_results = process_batch(batch)
        all_results.extend(batch_results)

        print(f"  Batch {i//BATCH_SIZE + 1}: processed {len(batch_results)} files")

    print(f"\n✅ Successfully processed {len(all_results)} files")

    # Group into phrases (for FM sweep modality)
    print(f"\n📊 Grouping into phrases...")

    phrase_library = defaultdict(lambda: {
        'contexts': Counter(),
        'features_list': []
    })

    for audio_path, features, context_name in all_results:
        # Generate phrase key for FM sweeps
        start_freq = int(features.get('start_freq', 0) / 1000) * 1000
        end_freq = int(features.get('end_freq', 0) / 1000) * 1000
        duration_ms = int(features.get('duration_ms', 0) / 10) * 10

        phrase_key = f"FM_{start_freq}_{end_freq}_DUR_{duration_ms}"

        phrase_library[phrase_key]['contexts'][context_name] += 1
        phrase_library[phrase_key]['features_list'].append(features)

    print(f"✅ Created {len(phrase_library)} phrase types")

    # Create export structure
    species_data = {
        'species': 'egyptian_bat',
        'analysis_date': datetime.now().isoformat(),
        'total_phrases': len(phrase_library),
        'phrases': {}
    }

    for phrase_key, phrase_data in phrase_library.items():
        features = phrase_data['features_list'][0]
        modality = 'fm_sweep'  # Bats use FM sweeps

        # Create contexts list
        contexts = []
        total_ctx = sum(phrase_data['contexts'].values())

        for ctx_name, count in phrase_data['contexts'].most_common():
            contexts.append({
                'context_name': ctx_name,
                'count': count,
                'percentage': (count / total_ctx * 100) if total_ctx > 0 else 0
            })

        species_data['phrases'][phrase_key] = {
            'phrase_key': phrase_key,
            'signature': f"{modality}_{phrase_key}",
            'species': 'egyptian_bat',
            'modality': modality,
            'acoustic_features': {k: float(v) for k, v in features.items()},
            'total_occurrences': len(phrase_data['features_list']),
            'contexts': contexts,
            'social_contexts': {},
            'is_compositional': False,
            'phrase_components': []
        }

    # Show statistics
    print(f"\n📊 STATISTICS:")
    print(f"   Total phrases: {len(species_data['phrases'])}")

    # Context distribution
    all_contexts = Counter()
    for phrase in species_data['phrases'].values():
        for ctx in phrase['contexts']:
            all_contexts[ctx['context_name']] += ctx['count']

    print(f"\n📊 CONTEXT DISTRIBUTION:")
    for ctx, count in all_contexts.most_common():
        pct = (count / sum(all_contexts.values())) * 100
        print(f"   {ctx:<20} {count:>8} ({pct:>5.1f}%)")

    # Save
    export_data = {
        'export_date': datetime.now().isoformat(),
        'species_data': {'egyptian_bat': species_data}
    }

    print(f"\n💾 Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"✅ Saved!")

    return species_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Optimized import of labeled vocalizations')
    parser.add_argument('--species', type=str, choices=['marmoset', 'bat'], default='marmoset')
    parser.add_argument('--max-files', type=int, default=5000, help='Max files to process (for testing)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes')

    args = parser.parse_args()

    if args.species == 'marmoset':
        import_marmoset_data(max_files=args.max_files)
    else:
        import_bat_data(max_files=args.max_files)

    print("\n" + "=" * 80)
    print("✅ IMPORT COMPLETE!")
    print("=" * 80)
