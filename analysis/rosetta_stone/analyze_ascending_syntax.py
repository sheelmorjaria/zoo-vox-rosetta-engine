#!/usr/bin/env python3
"""
Analyze Ascending Syntax in Individual Marmoset Vocalizations

This script analyzes individual continuous vocalizations to detect
ascending syntax patterns of "flat tone" (low modulation) atomic phrases.

Background:
- Marmosets may use ascending F0 sequences to convey meaning
- "Flat tone" = low frequency modulation (FM sweeps with small range)
- Ascending syntax: phrase A → phrase B → phrase C where F0 increases

Methods:
1. Load individual audio files
2. Segment into atomic phrases using energy/F0 boundaries
3. Identify "flat tone" phrases (low FM, narrow range)
4. Detect ascending sequences (F0_1 < F0_2 < F0_3 ...)
5. Analyze prevalence across behavioral contexts
"""

import json
import sys
from collections import defaultdict
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
ANNOTATIONS_PATH = "/home/sheel/birdsong_analysis/Annotations.tsv"
VOCALIZATIONS_DIR = "/home/sheel/birdsong_analysis/data/Vocalizations"
DATABASE_PATH = "/home/sheel/birdsong_analysis/src/vocalization_database_with_contexts.json"
SAMPLE_RATE = 22050
MAX_VOCALIZATIONS = 200


def load_database(db_path: str) -> Dict:
    """Load phrase database for reference."""
    print(f"Loading database from {db_path}...")

    with open(db_path, "r") as f:
        db = json.load(f)

    phrases = db["species_data"]["marmoset"]["phrases"]

    print(f"✅ Loaded {len(phrases)} phrase types")

    return phrases


def segment_audio_into_phrases(audio: np.ndarray) -> List[Tuple[np.ndarray, Dict]]:
    """
    Segment audio into atomic phrases using energy and F0 analysis.

    Returns:
        List of (audio_segment, features) tuples
    """
    segments = []

    # Simple energy-based segmentation
    # 1. Compute amplitude envelope
    from scipy.signal import hilbert

    envelope = np.abs(hilbert(audio))

    # Smooth envelope
    from scipy.ndimage import gaussian_filter1d

    smoothed = gaussian_filter1d(envelope, sigma=int(SAMPLE_RATE * 0.002))  # 2ms smoothing

    # Find peaks (potential phrase onsets)
    from scipy.signal import find_peaks

    threshold = np.mean(smoothed) + 0.3 * np.std(smoothed)
    min_distance = int(SAMPLE_RATE * 0.05)  # Minimum 50ms between phrases

    peaks, _ = find_peaks(smoothed, height=threshold, distance=min_distance)

    # Segment between peaks
    for i in range(len(peaks)):
        onset = peaks[i]

        # Find offset (next peak or end of energy)
        if i < len(peaks) - 1:
            offset = peaks[i + 1]
        else:
            # Find where energy drops below threshold
            remaining = smoothed[onset:]
            below_threshold = np.where(remaining < threshold)[0]
            if len(below_threshold) > 0:
                offset = onset + below_threshold[0]
            else:
                offset = len(audio)

        # Minimum duration check (50ms)
        if offset - onset < int(SAMPLE_RATE * 0.05):
            continue

        segment = audio[onset:offset]

        # Extract features
        try:
            sig = PhraseSignature(modality=Modality.HARMONIC, data=segment, sample_rate=SAMPLE_RATE)
            features = sig.features
            features["onset_ms"] = onset / SAMPLE_RATE * 1000
            features["offset_ms"] = offset / SAMPLE_RATE * 1000
            segments.append((segment, features))
        except:
            continue

    return segments


def is_flat_tone_phrase(features: Dict) -> bool:
    """
    Determine if a phrase is a "flat tone" (low frequency modulation).

    Criteria:
    - F0 range < 500 Hz (low modulation)
    - Not a strong FM sweep (start_freq ≈ end_freq)
    """
    f0_mean = features.get("f0_mean", 0)
    f0_range = features.get("f0_range", 0)
    start_freq = features.get("start_freq", f0_mean)
    end_freq = features.get("end_freq", f0_mean)

    # Low modulation range
    if f0_range > 500:
        return False

    # Not a strong sweep
    freq_change = abs(end_freq - start_freq)
    if freq_change > 300:  # More than 300Hz change = sweep
        return False

    return True


def detect_ascending_syntax(segments: List[Tuple[np.ndarray, Dict]]) -> Dict:
    """
    Detect ascending syntax patterns in a vocalization.

    Returns:
        Dictionary with analysis results
    """
    if len(segments) < 2:
        return {
            "has_ascending": False,
            "num_phrases": len(segments),
            "flat_tone_count": 0,
            "sequence": [],
        }

    # Extract F0 sequence
    f0_sequence = [seg[1].get("f0_mean", 0) for seg in segments]

    # Check for ascending pattern
    is_ascending = all(f0_sequence[i] < f0_sequence[i + 1] for i in range(len(f0_sequence) - 1))

    # Count flat tones
    flat_tones = [i for i, seg in enumerate(segments) if is_flat_tone_phrase(seg[1])]

    # Build phrase sequence info
    sequence_info = []
    for i, (audio, features) in enumerate(segments):
        sequence_info.append(
            {
                "index": i,
                "f0_mean": features.get("f0_mean", 0),
                "f0_range": features.get("f0_range", 0),
                "duration_ms": features.get("duration_ms", 0),
                "is_flat_tone": is_flat_tone_phrase(features),
                "onset_ms": features.get("onset_ms", 0),
                "offset_ms": features.get("offset_ms", 0),
            }
        )

    return {
        "has_ascending": is_ascending,
        "num_phrases": len(segments),
        "flat_tone_count": len(flat_tones),
        "flat_tone_indices": flat_tones,
        "f0_sequence": f0_sequence,
        "sequence": sequence_info,
    }


def analyze_vocalizations_for_ascending_syntax(
    annotations: pd.DataFrame, max_vocalizations: int = MAX_VOCALIZATIONS
) -> Dict:
    """Analyze individual vocalizations for ascending syntax patterns."""

    print("\n" + "=" * 80)
    print("ANALYZING ASCENDING SYNTAX IN INDIVIDUAL VOCALIZATIONS")
    print("=" * 80)

    results = {
        "vocalizations_analyzed": 0,
        "multi_phrase_vocalizations": 0,
        "ascending_vocalizations": 0,
        "flat_tone_ascending": 0,
        "context_patterns": defaultdict(
            lambda: {"total": 0, "multi_phrase": 0, "ascending": 0, "flat_ascending": 0}
        ),
        "examples": {"ascending": [], "flat_tone_ascending": [], "multi_phrase_non_ascending": []},
    }

    # Sample annotations
    sample = annotations.sample(n=min(max_vocalizations, len(annotations)), random_state=42)

    print(f"\n🔍 Analyzing {len(sample)} vocalizations...")

    for idx, row in sample.iterrows():
        parent_name = str(row["parent_name"]).replace(" ", "_")
        file_name = str(row["file_name"])
        label = str(row["label"])

        file_path = Path(VOCALIZATIONS_DIR) / parent_name / file_name

        if not file_path.exists():
            continue

        # Load audio
        try:
            audio, sr = sf.read(str(file_path))

            # Convert to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Resample if needed
            if sr != SAMPLE_RATE:
                from scipy import signal

                num_samples = int(len(audio) * SAMPLE_RATE / sr)
                audio = signal.resample(audio, num_samples)

            if len(audio) < SAMPLE_RATE * 0.2:  # Too short
                continue

            # Segment into phrases
            segments = segment_audio_into_phrases(audio)

            if not segments:
                continue

            results["vocalizations_analyzed"] += 1

            # Detect ascending syntax
            syntax_result = detect_ascending_syntax(segments)

            # Track context patterns
            results["context_patterns"][label]["total"] += 1

            if syntax_result["num_phrases"] > 1:
                results["multi_phrase_vocalizations"] += 1
                results["context_patterns"][label]["multi_phrase"] += 1

                if syntax_result["has_ascending"]:
                    results["ascending_vocalizations"] += 1
                    results["context_patterns"][label]["ascending"] += 1

                    # Check if composed of flat tones
                    if syntax_result["flat_tone_count"] >= 2:
                        results["flat_tone_ascending"] += 1
                        results["context_patterns"][label]["flat_ascending"] += 1

                        # Add to examples
                        if len(results["examples"]["flat_tone_ascending"]) < 10:
                            results["examples"]["flat_tone_ascending"].append(
                                {
                                    "label": label,
                                    "file": file_name,
                                    "num_phrases": syntax_result["num_phrases"],
                                    "f0_sequence": syntax_result["f0_sequence"],
                                    "flat_tone_count": syntax_result["flat_tone_count"],
                                    "sequence": syntax_result["sequence"],
                                }
                            )

                # Add to ascending examples
                if syntax_result["has_ascending"] and len(results["examples"]["ascending"]) < 5:
                    results["examples"]["ascending"].append(
                        {
                            "label": label,
                            "file": file_name,
                            "num_phrases": syntax_result["num_phrases"],
                            "f0_sequence": syntax_result["f0_sequence"],
                            "flat_tone_count": syntax_result["flat_tone_count"],
                        }
                    )

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(sample)} vocalizations...")

        except Exception:
            continue

    return results


def print_ascending_syntax_summary(results: Dict):
    """Print summary of ascending syntax analysis."""
    print("\n" + "=" * 80)
    print("ASCENDING SYNTAX ANALYSIS SUMMARY")
    print("=" * 80)

    total = results["vocalizations_analyzed"]

    print("\n📊 OVERALL STATISTICS:")
    print(f"   Vocalizations analyzed: {total}")
    print(
        f"   Multi-phrase vocalizations: {results['multi_phrase_vocalizations']} "
        f"({results['multi_phrase_vocalizations'] / total * 100:.1f}%)"
    )
    print(
        f"   Ascending syntax detected: {results['ascending_vocalizations']} "
        f"({results['ascending_vocalizations'] / total * 100:.1f}%)"
    )
    print(
        f"   Flat-tone ascending: {results['flat_tone_ascending']} "
        f"({results['flat_tone_ascending'] / total * 100:.1f}%)"
    )

    # Context-specific patterns
    print("\n📊 CONTEXT-SPECIFIC PATTERNS:")
    for context, stats in sorted(results["context_patterns"].items()):
        if stats["total"] > 0:
            print(f"\n   {context.upper()}:")
            print(f"      Total: {stats['total']}")
            multi_phrase_pct = stats["multi_phrase"] / stats["total"] * 100
            print(f"      Multi-phrase: {stats['multi_phrase']} ({multi_phrase_pct:.1f}%)")
            if stats["multi_phrase"] > 0:
                ascending_pct = stats["ascending"] / stats["multi_phrase"] * 100
                print(
                    f"      Ascending: {stats['ascending']} ({ascending_pct:.1f}% of multi-phrase)"
                )
                print(f"      Flat-tone ascending: {stats['flat_ascending']}")

    # Examples
    print("\n📊 EXAMPLES OF FLAT-TONE ASCENDING SYNTAX:")
    for i, ex in enumerate(results["examples"]["flat_tone_ascending"][:5], 1):
        print(f"\n   Example {i} ({ex['label']}): {ex['file']}")
        print(f"      Phrases: {ex['num_phrases']}")
        print(f"      F0 sequence: {[f'{f0:.0f}Hz' for f0 in ex['f0_sequence']]}")
        print(f"      Flat tones: {ex['flat_tone_count']}")
        print("      Details:")
        for j, phrase in enumerate(ex["sequence"]):
            flat_marker = " [FLAT]" if phrase["is_flat_tone"] else ""
            print(
                f"         {j + 1}. F0={phrase['f0_mean']:.0f}Hz, "
                f"Range={phrase['f0_range']:.0f}Hz{flat_marker}"
            )

    # Scientific interpretation
    print("\n" + "=" * 80)
    print("📚 SCIENTIFIC INTERPRETATION")
    print("=" * 80)

    ascending_rate = results["ascending_vocalizations"] / total if total > 0 else 0
    flat_ascending_rate = results["flat_tone_ascending"] / total if total > 0 else 0

    if flat_ascending_rate > 0.05:
        print("\n✅ STRONG EVIDENCE OF FLAT-TONE ASCENDING SYNTAX")
        flat_ascending_pct = flat_ascending_rate * 100
        print(f"   - {flat_ascending_pct:.1f}% of vocalizations show flat-tone ascending patterns")
        print("   - Suggests combinatorial syntax using low-modulation tones")
        print("   - Ascending F0 sequences may convey directional or increasing urgency")
    elif ascending_rate > 0.05:
        print("\n✅ MODERATE EVIDENCE OF ASCENDING SYNTAX")
        ascending_pct = ascending_rate * 100
        print(f"   - {ascending_pct:.1f}% of vocalizations show ascending patterns")
        print("   - Some evidence of sequential phrase organization")
    else:
        print("\n⚠️  LIMITED ASCENDING SYNTAX DETECTED")
        ascending_pct = ascending_rate * 100
        flat_ascending_pct = flat_ascending_rate * 100
        print(f"   - {ascending_pct:.1f}% ascending, {flat_ascending_pct:.1f}% flat-tone ascending")
        print("   - Most vocalizations are single phrases or non-ascending")
        print("   - May indicate: limited syntax OR need better segmentation")

    print("\n" + "=" * 80)


def main():
    """Main analysis function."""
    print("=" * 80)
    print("ASCENDING SYNTAX ANALYSIS: FLAT TONE ATOMIC PHRASES")
    print("=" * 80)

    # Load annotations
    print(f"\n📊 Loading annotations from {ANNOTATIONS_PATH}...")
    annotations = pd.read_csv(ANNOTATIONS_PATH, sep="\t")
    print(f"✅ Loaded {len(annotations)} annotations")

    # Analyze
    results = analyze_vocalizations_for_ascending_syntax(annotations, MAX_VOCALIZATIONS)

    # Print summary
    print_ascending_syntax_summary(results)

    # Save results
    output_path = "/home/sheel/birdsong_analysis/src/ascending_syntax_analysis.json"
    print(f"\n💾 Saving results to {output_path}...")

    saveable_results = {
        "vocalizations_analyzed": results["vocalizations_analyzed"],
        "multi_phrase_vocalizations": results["multi_phrase_vocalizations"],
        "ascending_vocalizations": results["ascending_vocalizations"],
        "flat_tone_ascending": results["flat_tone_ascending"],
        "context_patterns": dict(results["context_patterns"]),
        "examples": results["examples"],
    }

    with open(output_path, "w") as f:
        json.dump(saveable_results, f, indent=2)

    print("✅ Saved!")

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
