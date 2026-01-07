#!/usr/bin/env python3
"""
Analyze Atomic Phrase Reuse in Individual Vocalizations

This script analyzes how atomic phrases (smallest meaningful units) are reused
within individual vocalizations (sentences/continuous calls).

Key Questions:
1. Do individual vocalizations contain multiple atomic phrases?
2. Are the same phrases reused across different parts of a vocalization?
3. Is there evidence of compositional syntax (phrase combinations)?
4. What patterns of phrase repetition exist?

Method:
1. Load individual audio files (not just pre-segmented phrases)
2. Use URS segmentation to find phrase boundaries within each vocalization
3. Extract atomic phrase signatures for each segment
4. Analyze reuse patterns within and across vocalizations
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

# Add URS path
urs_path = str(Path(__file__).parent.parent / "analysis" / "rosetta_stone")
sys.path.insert(0, urs_path)

from universal_rosetta_stone import Modality, PhraseSignature, UniversalRosettaStone

# Configuration
ANNOTATIONS_PATH = "/home/sheel/birdsong_analysis/Annotations.tsv"
VOCALIZATIONS_DIR = "/home/sheel/birdsong_analysis/data/Vocalizations"
DATABASE_PATH = "/home/sheel/birdsong_analysis/src/vocalization_database_with_contexts.json"
SAMPLE_RATE = 22050
MAX_VOCALIZATIONS_TO_ANALYZE = 100


def load_database(db_path: str) -> Dict:
    """Load phrase database."""
    print(f"Loading database from {db_path}...")

    with open(db_path, "r") as f:
        db = json.load(f)

    phrases = db["species_data"]["marmoset"]["phrases"]

    print(f"✅ Loaded {len(phrases)} phrase types")

    return phrases


def match_phrase_to_library(
    features: Dict, phrase_library: Dict, threshold: float = 0.8
) -> Tuple[str, float]:
    """
    Match extracted features to phrase library.

    Returns:
        Tuple of (phrase_key, similarity_score)
    """
    best_match = None
    best_score = 0

    for phrase_key, phrase_data in phrase_library.items():
        library_features = phrase_data["acoustic_features"]

        # Calculate similarity based on key features
        similarities = []

        # F0 similarity
        f0_diff = abs(features.get("f0_mean", 0) - library_features.get("mean_f0_hz", 0))
        f0_sim = max(0, 1 - f0_diff / 2000)  # Tolerate 2kHz difference
        similarities.append(f0_sim)

        # Duration similarity
        dur_diff = abs(features.get("duration_ms", 0) - library_features.get("mean_duration_ms", 0))
        dur_sim = max(0, 1 - dur_diff / 500)  # Tolerate 500ms difference
        similarities.append(dur_sim)

        # Timbre similarity (if available)
        if library_features.get("spectral_centroid_hz", 0) > 0:
            centroid_diff = abs(
                features.get("spectral_centroid_hz", 0)
                - library_features.get("spectral_centroid_hz", 0)
            )
            centroid_sim = max(0, 1 - centroid_diff / 2000)
            similarities.append(centroid_sim)

        # Overall similarity
        avg_sim = np.mean(similarities)

        if avg_sim > best_score:
            best_score = avg_sim
            best_match = phrase_key

    return (best_match, best_score) if best_score >= threshold else (None, best_score)


def analyze_vocalization_for_phrases(
    audio_path: str, phrase_library: Dict, urs_analyzer: UniversalRosettaStone
) -> List[Dict]:
    """
    Analyze a single vocalization to find atomic phrase segments.

    Returns:
        List of phrase segments found within the vocalization
    """
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

        if len(audio) < SAMPLE_RATE * 0.2:  # Too short
            return []

        # Segment into phrases using URS
        phrases = urs_analyzer.segment_audio(
            audio, modality=Modality.HARMONIC, min_phrase_duration_ms=50, max_gap_ms=200
        )

        segments_found = []

        for phrase in phrases:
            # Extract features
            sig = PhraseSignature(
                modality=Modality.HARMONIC, data=phrase.data, sample_rate=SAMPLE_RATE
            )

            features = sig.features

            # Match to library
            phrase_key, similarity = match_phrase_to_library(features, phrase_library)

            if phrase_key:
                segments_found.append(
                    {
                        "phrase_key": phrase_key,
                        "similarity": similarity,
                        "onset_ms": phrase.timestamp * 1000,
                        "offset_ms": (phrase.timestamp + len(phrase.data) / SAMPLE_RATE) * 1000,
                        "duration_ms": len(phrase.data) / SAMPLE_RATE * 1000,
                        "features": features,
                    }
                )

        return segments_found

    except Exception as e:
        print(f"Error analyzing {audio_path}: {e}")
        return []


def analyze_phrase_reuse_patterns(
    annotations: pd.DataFrame,
    vocalizations_dir: str,
    phrase_library: Dict,
    max_vocalizations: int = MAX_VOCALIZATIONS_TO_ANALYZE,
) -> Dict:
    """
    Analyze phrase reuse patterns within individual vocalizations.

    Returns:
        Dictionary with analysis results
    """
    print("\n" + "=" * 80)
    print("ANALYZING PHRASE REUSE IN INDIVIDUAL VOCALIZATIONS")
    print("=" * 80)

    urs_analyzer = UniversalRosettaStone(sample_rate=SAMPLE_RATE)

    results = {
        "vocalizations_analyzed": 0,
        "total_phrases_found": 0,
        "phrase_reuse_count": 0,
        "vocalizations_with_reuse": 0,
        "phrase_cooccurrence": defaultdict(Counter),  # (phrase1, phrase2) -> count
        "phrase_sequences": [],  # List of phrase sequences in each vocalization
        "context_phrase_patterns": defaultdict(Counter),  # context -> phrase patterns
    }

    # Sample vocalizations
    sample_annotations = annotations.sample(
        n=min(max_vocalizations, len(annotations)), random_state=42
    )

    print(f"\n🔍 Analyzing {len(sample_annotations)} vocalizations...")

    for idx, row in sample_annotations.iterrows():
        parent_name = str(row["parent_name"]).replace(" ", "_")
        file_name = str(row["file_name"])
        label = str(row["label"])

        file_path = Path(vocalizations_dir) / parent_name / file_name

        if not file_path.exists():
            continue

        # Find phrases in this vocalization
        segments = analyze_vocalization_for_phrases(str(file_path), phrase_library, urs_analyzer)

        if not segments:
            continue

        results["vocalizations_analyzed"] += 1
        results["total_phrases_found"] += len(segments)

        # Extract phrase sequence (keys only)
        phrase_sequence = [s["phrase_key"] for s in segments]
        results["phrase_sequences"].append(
            {"label": label, "phrases": phrase_sequence, "num_phrases": len(phrase_sequence)}
        )

        # Track context patterns
        results["context_phrase_patterns"][label].update(phrase_sequence)

        # Check for phrase reuse within vocalization
        phrase_counts = Counter(phrase_sequence)

        if any(count > 1 for count in phrase_counts.values()):
            results["vocalizations_with_reuse"] += 1
            results["phrase_reuse_count"] += sum(
                count - 1 for count in phrase_counts.values() if count > 1
            )

        # Track phrase co-occurrence (adjacent pairs)
        for i in range(len(phrase_sequence) - 1):
            pair = (phrase_sequence[i], phrase_sequence[i + 1])
            results["phrase_cooccurrence"][pair] += 1

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(sample_annotations)} vocalizations...")

    return results


def print_analysis_summary(results: Dict):
    """Print summary of phrase reuse analysis."""
    print("\n" + "=" * 80)
    print("PHRASE REUSE ANALYSIS SUMMARY")
    print("=" * 80)

    print("\n📊 OVERALL STATISTICS:")
    print(f"   Vocalizations analyzed: {results['vocalizations_analyzed']}")
    print(f"   Total phrases found: {results['total_phrases_found']}")
    avg_phrases = results['total_phrases_found'] / results['vocalizations_analyzed']
    print(
        f"   Average phrases per vocalization: {avg_phrases:.2f}"
    )
    print(
        f"   Vocalizations with phrase reuse: {results['vocalizations_with_reuse']} "
        f"({results['vocalizations_with_reuse'] / results['vocalizations_analyzed'] * 100:.1f}%)"
    )
    print(f"   Total phrase reuse events: {results['phrase_reuse_count']}")

    # Phrase sequence length distribution
    print("\n📊 PHRASE SEQUENCE LENGTH DISTRIBUTION:")
    seq_lengths = [s["num_phrases"] for s in results["phrase_sequences"]]
    if seq_lengths:
        print(f"   Min: {min(seq_lengths)} phrases")
        print(f"   Max: {max(seq_lengths)} phrases")
        print(f"   Mean: {np.mean(seq_lengths):.2f} phrases")
        print(f"   Median: {np.median(seq_lengths):.1f} phrases")

        # Distribution
        length_counts = Counter(seq_lengths)
        for length in sorted(length_counts.keys())[:10]:
            count = length_counts[length]
            pct = count / len(seq_lengths) * 100
            print(f"   {length} phrases: {count} ({pct:.1f}%)")

    # Top phrase co-occurrences
    print("\n📊 TOP PHRASE CO-OCCURRENCES (adjacent pairs):")
    top_pairs = results["phrase_cooccurrence"].most_common(10)
    for (phrase1, phrase2), count in top_pairs:
        print(f"   {phrase1} → {phrase2}: {count} occurrences")

    # Context-specific patterns
    print("\n📊 CONTEXT-SPECIFIC PHRASE PATTERNS:")
    for context, phrase_counter in list(results["context_phrase_patterns"].items())[:5]:
        print(f"\n   {context.upper()}:")
        for phrase, count in phrase_counter.most_common(5):
            pct = count / sum(phrase_counter.values()) * 100
            print(f"      {phrase}: {count} ({pct:.1f}%)")

    # Examples of multi-phrase vocalizations
    print("\n📊 EXAMPLE MULTI-PHRASE VOCALIZATIONS:")
    multi_phrase = [s for s in results["phrase_sequences"] if s["num_phrases"] > 1]
    for i, example in enumerate(multi_phrase[:5], 1):
        print(f"\n   Example {i} ({example['label']}):")
        print(f"      Phrases: {', '.join(example['phrases'])}")
        print(f"      Length: {example['num_phrases']} phrases")

    # Scientific interpretation
    print("\n" + "=" * 80)
    print("📚 SCIENTIFIC INTERPRETATION")
    print("=" * 80)

    reuse_rate = (
        results["vocalizations_with_reuse"] / results["vocalizations_analyzed"]
        if results["vocalizations_analyzed"] > 0
        else 0
    )

    if reuse_rate > 0.5:
        print("\n✅ STRONG EVIDENCE OF COMPOSITIONALITY")
        print(f"   - {reuse_rate * 100:.1f}% of vocalizations show phrase reuse")
        print("   - Suggests combinatorial syntax in marmoset communication")
        print("   - Atomic phrases combined to create complex meanings")
    elif reuse_rate > 0.1:
        print("\n✅ MODERATE EVIDENCE OF COMPOSITIONALITY")
        print(f"   - {reuse_rate * 100:.1f}% of vocalizations show phrase reuse")
        print("   - Some combinatorial patterns detected")
        print("   - May indicate emerging syntax in specific contexts")
    else:
        print("\n⚠️  LIMITED EVIDENCE OF COMPOSITIONALITY")
        print(f"   - {reuse_rate * 100:.1f}% of vocalizations show phrase reuse")
        print("   - Most vocalizations are single phrases")
        print("   - May indicate: isolated calls OR need better segmentation")

    print("\n" + "=" * 80)


def main():
    """Main analysis function."""
    print("=" * 80)
    print("ATOMIC PHRASE REUSE ANALYSIS")
    print("=" * 80)

    # Load phrase library
    phrase_library = load_database(DATABASE_PATH)

    # Load annotations
    print(f"\n📊 Loading annotations from {ANNOTATIONS_PATH}...")
    annotations = pd.read_csv(ANNOTATIONS_PATH, sep="\t")
    print(f"✅ Loaded {len(annotations)} annotations")

    # Analyze phrase reuse patterns
    results = analyze_phrase_reuse_patterns(
        annotations,
        VOCALIZATIONS_DIR,
        phrase_library,
        max_vocalizations=MAX_VOCALIZATIONS_TO_ANALYZE,
    )

    # Print summary
    print_analysis_summary(results)

    # Save results
    output_path = "/home/sheel/birdsong_analysis/src/phrase_reuse_analysis.json"
    print(f"\n💾 Saving results to {output_path}...")

    # Convert defaultdicts to regular dicts for JSON serialization
    saveable_results = {
        "vocalizations_analyzed": results["vocalizations_analyzed"],
        "total_phrases_found": results["total_phrases_found"],
        "phrase_reuse_count": results["phrase_reuse_count"],
        "vocalizations_with_reuse": results["vocalizations_with_reuse"],
        "phrase_cooccurrence": {str(k): v for k, v in results["phrase_cooccurrence"].items()},
        "phrase_sequences": results["phrase_sequences"],
        "context_phrase_patterns": {
            k: dict(v) for k, v in results["context_phrase_patterns"].items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(saveable_results, f, indent=2)

    print("✅ Saved!")

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
