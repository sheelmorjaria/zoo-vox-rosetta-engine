#!/usr/bin/env python3
"""
Enhanced Marmoset Import with Grammar/Syntax Metadata

This script imports labeled marmoset vocalizations AND captures grammar/syntax
metadata from individual audio files, including:
1. Phrase sequences within vocalizations
2. Ascending/descending F0 patterns
3. Phrase transition patterns
4. Repetition patterns
5. Compositional structure metadata

Grammar Features Captured:
- phrase_sequence: Ordered list of phrase keys within vocalization
- f0_contour: Overall F0 trajectory (ascending, descending, flat, complex)
- num_phrases: Number of atomic phrases in vocalization
- has_repetition: Whether phrases repeat
- transition_pattern: Phrase-to-phrase transitions
- is_compositional: Evidence of combinatorial structure
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

sys.path.insert(0, str(Path(__file__).parent.parent))

# Add URS path
urs_path = str(Path(__file__).parent.parent / 'analysis' / 'rosetta_stone')
sys.path.insert(0, urs_path)

from universal_rosetta_stone import PhraseSignature, Modality

# Configuration
ANNOTATIONS_PATH = '/home/sheel/birdsong_analysis/Annotations.tsv'
VOCALIZATIONS_DIR = '/home/sheel/birdsong_analysis/data/Vocalizations'
OUTPUT_PATH = '/home/sheel/birdsong_analysis/src/vocalization_database_with_syntax.json'
SAMPLE_RATE = 22050
NUM_WORKERS = max(1, cpu_count() - 1)
BATCH_SIZE = 50
MAX_FILES = 2000


def load_and_segment_audio(args: Tuple[str, str]) -> Tuple[str, Dict, str]:
    """
    Load audio file, segment into phrases, extract syntax metadata.

    Returns:
        Tuple of (file_path, syntax_metadata, context_name)
    """
    audio_path, context_name = args

    try:
        # Load audio
        audio, sr = sf.read(audio_path)

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

        return (audio_path, syntax_metadata, context_name)

    except Exception as e:
        return None


def segment_into_phrases(audio: np.ndarray) -> List[Tuple[np.ndarray, Dict]]:
    """Segment audio into atomic phrases using energy and F0 analysis."""
    segments = []

    # Energy-based segmentation
    from scipy.signal import hilbert, find_peaks
    from scipy.ndimage import gaussian_filter1d

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
            # Go to midpoint before next peak
            offset = peaks[i] + int((peaks[i + 1] - peaks[i]) * 0.7)
        else:
            # Find where energy drops
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

        # Extract features
        try:
            sig = PhraseSignature(modality=Modality.HARMONIC, data=segment, sample_rate=SAMPLE_RATE)
            features = sig.features
            features['onset_ms'] = onset / SAMPLE_RATE * 1000
            features['offset_ms'] = offset / SAMPLE_RATE * 1000
            segments.append((segment, features))
        except:
            continue

    return segments


def extract_syntax_metadata(segments: List[Tuple[np.ndarray, Dict]], full_audio: np.ndarray) -> Dict:
    """Extract grammar/syntax metadata from segmented vocalization."""

    if not segments:
        return {}

    # Extract phrase sequence
    phrase_sequence = []
    f0_sequence = []

    for audio, features in segments:
        # Generate phrase key
        f0_mean = int(features.get('f0_mean', 0) / 100) * 100
        f0_range = int(features.get('f0_range', 0) / 100) * 100
        duration_ms = int(features.get('duration_ms', 0) / 5) * 5

        phrase_key = f"F0_{f0_mean}_DUR_{duration_ms}_RANGE_{f0_range}"

        phrase_sequence.append(phrase_key)
        f0_sequence.append(features.get('f0_mean', 0))

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
    overall_f0_mean = np.mean([f for f in f0_sequence if f > 0]) if any(f > 0 for f in f0_sequence) else 0
    overall_f0_range = max(f0_sequence) - min(f0_sequence) if f0_sequence else 0

    return {
        'phrase_sequence': phrase_sequence,
        'num_phrases': len(phrase_sequence),
        'f0_sequence': f0_sequence,
        'f0_contour': f0_contour,
        'has_repetition': has_repetition,
        'transitions': transitions,
        'is_compositional': is_compositional,
        'total_duration_ms': total_duration_ms,
        'overall_f0_mean_hz': overall_f0_mean,
        'overall_f0_range_hz': overall_f0_range,
        'segment_details': [
            {
                'phrase_key': phrase_sequence[i],
                'f0_mean': f0_sequence[i],
                'onset_ms': seg[1]['onset_ms'],
                'offset_ms': seg[1]['offset_ms'],
                'duration_ms': seg[1]['duration_ms']
            }
            for i, seg in enumerate(segments)
        ]
    }


def analyze_f0_contour(f0_sequence: List[float]) -> str:
    """Analyze the overall F0 contour pattern."""
    if len(f0_sequence) < 2:
        return 'single'

    # Filter out zero F0 values
    valid_f0 = [f for f in f0_sequence if f > 0]

    if len(valid_f0) < 2:
        return 'unmeasured'

    # Calculate trend
    first_half = valid_f0[:len(valid_f0)//2]
    second_half = valid_f0[len(valid_f0)//2:]

    mean_first = np.mean(first_half)
    mean_second = np.mean(second_half)

    diff = mean_second - mean_first
    range_val = max(valid_f0) - min(valid_f0)

    if range_val < 200:  # Less than 200Hz variation
        return 'flat'
    elif diff > range_val * 0.3:  # Strong ascending
        return 'ascending'
    elif diff < -range_val * 0.3:  # Strong descending
        return 'descending'
    else:
        return 'complex'


def process_batch(batch: List[Tuple[str, str]]) -> List[Tuple]:
    """Process a batch of audio files."""
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(load_and_segment_audio, batch)
    return [r for r in results if r is not None]


def import_with_syntax_metadata(max_files: int = MAX_FILES):
    """Import marmoset data with grammar/syntax metadata."""

    print("=" * 80)
    print("ENHANCED MARMOSET IMPORT WITH GRAMMAR/SYNTAX METADATA")
    print("=" * 80)

    # Load annotations
    print(f"\n📊 Loading annotations from {ANNOTATIONS_PATH}...")
    df = pd.read_csv(ANNOTATIONS_PATH, sep='\t')
    print(f"✅ Loaded {len(df)} annotations")

    # Sample
    if max_files and len(df) > max_files:
        df = df.sample(n=max_files, random_state=42)

    # Prepare file tasks
    print(f"\n🔍 Preparing file list...")
    file_tasks = []

    for _, row in df.iterrows():
        parent_name = str(row['parent_name']).replace(' ', '_')
        file_name = str(row['file_name'])
        label = str(row['label'])

        context_map = {
            'Tsik': 'tsik', 'Trill': 'trill', 'Twitter': 'twitter',
            'Phee': 'phee', 'Seep': 'seep', 'Infant': 'infant',
            'Infant_cry': 'infant', 'Vocalization': 'vocalization'
        }
        context_name = context_map.get(label, label.lower())

        file_path = Path(VOCALIZATIONS_DIR) / parent_name / file_name

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

    print(f"\n✅ Successfully processed {len(all_results)} vocalizations with syntax metadata")

    # Build phrase library with syntax data
    print(f"\n📊 Building phrase library...")

    phrase_library = defaultdict(lambda: {
        'contexts': Counter(),
        'features_list': [],
        'syntax_patterns': Counter(),
        'vocalization_ids': []
    })

    vocalizations = []

    for vocalization_id, (audio_path, syntax_meta, context_name) in enumerate(all_results):
        # Store vocalization with syntax
        vocalizations.append({
            'vocalization_id': vocalization_id,
            'file_path': audio_path,
            'context': context_name,
            'syntax_metadata': syntax_meta
        })

        # Add phrase occurrences to library
        for seg in syntax_meta.get('segment_details', []):
            phrase_key = seg['phrase_key']

            phrase_library[phrase_key]['contexts'][context_name] += 1
            phrase_library[phrase_key]['vocalization_ids'].append(vocalization_id)

            # Track syntax patterns
            if syntax_meta.get('f0_contour'):
                phrase_library[phrase_key]['syntax_patterns'][syntax_meta['f0_contour']] += 1

    print(f"✅ Created {len(phrase_library)} phrase types from {len(vocalizations)} vocalizations")

    # Create export structure
    print(f"\n📊 Creating database with syntax metadata...")

    species_data = {
        'species': 'marmoset',
        'analysis_date': datetime.now().isoformat(),
        'total_phrases': len(phrase_library),
        'phrases': {},
        'vocalizations': vocalizations  # NEW: Individual vocalizations with syntax
    }

    # Export phrases with syntax metadata
    for phrase_key, phrase_data in phrase_library.items():
        if phrase_data['features_list']:
            features = phrase_data['features_list'][0]
        else:
            continue

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
            'total_occurrences': len(phrase_data['vocalization_ids']),
            'contexts': contexts,
            'social_contexts': {},
            'is_compositional': False,
            'phrase_components': [],
            # NEW: Syntax metadata
            'syntax_metadata': {
                'appears_in_vocalizations': len(phrase_data['vocalization_ids']),
                'syntax_patterns': dict(phrase_data['syntax_patterns']),
                'common_f0_contours': dict(phrase_data['syntax_patterns'].most_common(3))
            }
        }

    # Show statistics
    print(f"\n📊 STATISTICS:")
    print(f"   Total phrases: {len(species_data['phrases'])}")
    print(f"   Total vocalizations analyzed: {len(vocalizations)}")

    # Syntax statistics
    f0_contours = Counter()
    ascending_count = 0
    descending_count = 0
    compositional_count = 0

    for vocalization in vocalizations:
        syntax = vocalization['syntax_metadata']
        f0_contours[syntax.get('f0_contour', 'unknown')] += 1

        if syntax.get('f0_contour') == 'ascending':
            ascending_count += 1
        elif syntax.get('f0_contour') == 'descending':
            descending_count += 1

        if syntax.get('is_compositional'):
            compositional_count += 1

    print(f"\n📊 SYNTAX STATISTICS:")
    print(f"   F0 contours: {dict(f0_contours)}")
    print(f"   Ascending: {ascending_count} ({ascending_count/len(vocalizations)*100:.1f}%)")
    print(f"   Descending: {descending_count} ({descending_count/len(vocalizations)*100:.1f}%)")
    print(f"   Compositional (3+ phrases): {compositional_count} ({compositional_count/len(vocalizations)*100:.1f}%)")

    # Save
    export_data = {
        'export_date': datetime.now().isoformat(),
        'species_data': {'marmoset': species_data}
    }

    print(f"\n💾 Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"✅ Saved!")

    return species_data


if __name__ == "__main__":
    import_with_syntax_metadata(MAX_FILES)

    print("\n" + "=" * 80)
    print("✅ IMPORT WITH SYNTAX METADATA COMPLETE!")
    print("=" * 80)
    print(f"\n🎯 New metadata includes:")
    print(f"   - phrase_sequence: Ordered phrase list within vocalizations")
    print(f"   - f0_contour: ascending/descending/flat/complex patterns")
    print(f"   - has_repetition: Whether phrases repeat")
    print(f"   - transitions: Phrase-to-phrase transitions")
    print(f"   - is_compositional: Evidence of combinatorial structure")
    print("=" * 80)
