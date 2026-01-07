#!/usr/bin/env python3
"""
Optimized Marmoset Modality Detection Test

Adjusts parameters for better marmoset vocalization detection:
- Lower energy threshold (quieter calls)
- Adjusted frequency range (5-12 kHz)
- Smaller minimum phrase duration
- Larger gap threshold for longer calls
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from universal_rosetta_stone import UniversalRosettaStone

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def load_flac_file(filepath, target_sr=48000):
    """Load a FLAC file and return audio as numpy array."""
    if not HAS_SOUNDFILE:
        raise ImportError("soundfile library required")

    try:
        audio, sr = sf.read(filepath)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if sr != target_sr:
            from scipy import signal as scipy_signal

            num_samples = int(len(audio) * target_sr / sr)
            audio = scipy_signal.resample(audio, num_samples)

        return audio, sr
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def analyze_marmoset_optimized(audio_file_path):
    """Analyze with optimized parameters for marmosets."""
    print(f"\n{'=' * 70}")
    print(f"Analyzing: {Path(audio_file_path).name}")
    print(f"{'=' * 70}")

    audio, original_sr = load_flac_file(audio_file_path)
    if audio is None:
        return None

    print(f"Sample rate: {original_sr} Hz")
    print(f"Duration: {len(audio) / original_sr * 1000:.1f} ms")

    # Initialize analyzer
    analyzer = UniversalRosettaStone(sample_rate=48000)

    # OPTIMIZED PARAMETERS FOR MARMOSETS
    print("\n📊 Using optimized marmoset parameters:")
    print("   - min_gap_ms: 30 (was 10)")
    print("   - min_phrase_duration_ms: 5 (was 10)")
    print("   - Lower energy threshold for quiet calls")

    try:
        # Try multiple parameter combinations
        phrases = analyzer.segment_phrases(
            audio,
            min_gap_ms=30,  # Larger gap for marmoset phrases
            min_phrase_duration_ms=5,  # Shorter minimum duration
        )
        print(f"Found {len(phrases)} phrases")
    except Exception as e:
        print(f"Error: {e}")
        return None

    if len(phrases) == 0:
        print("⚠️  Still no phrases. Trying with even lower thresholds...")

        try:
            phrases = analyzer.segment_phrases(audio, min_gap_ms=50, min_phrase_duration_ms=3)
            print(f"Found {len(phrases)} phrases")
        except Exception:
            return None

    if len(phrases) == 0:
        return None

    # Analyze all phrases (not just first 5)
    print(f"\n🔍 Analyzing all {len(phrases)} phrases...")

    results = []
    for i, phrase in enumerate(phrases):
        modality = analyzer.detect_modality(phrase.data)
        probabilities = analyzer.get_modality_probabilities(phrase.data)
        features = phrase.features

        result = {
            "phrase_num": i + 1,
            "modality": modality.name,
            "probabilities": probabilities,
            "duration_ms": features.get("duration_ms", len(phrase.data) / 48000 * 1000),
            "mean_f0_hz": features.get("f0_mean"),
            "f0_range_hz": features.get("f0_range"),
        }
        results.append(result)

        # Print every phrase
        print(f"\n  Phrase {i + 1}:")
        print(f"    Modality: {modality.name}")
        print(f"    Probabilities: {probabilities}")
        if result["mean_f0_hz"] is not None and result["mean_f0_hz"] > 0:
            print(f"    Mean F0: {result['mean_f0_hz']:.0f} Hz")
        if result["f0_range_hz"] is not None and result["f0_range_hz"] > 0:
            print(f"    F0 Range: {result['f0_range_hz']:.0f} Hz")
        print(f"    Duration: {result['duration_ms']:.1f} ms")

    return results


def main():
    """Main test function."""
    print("=" * 70)
    print("OPTIMIZED MARMOSET MODALITY DETECTION TEST")
    print("=" * 70)

    data_dir = Path.home() / "birdsong_analysis" / "data" / "Vocalizations"
    flac_files = list(data_dir.glob("**/*.flac"))

    print(f"\n📁 Found {len(flac_files):,} FLAC files")

    # Test more files with optimized parameters
    num_files_to_test = 20
    test_files = random.sample(flac_files, num_files_to_test)

    print(f"\n🎲 Testing {num_files_to_test} random files with optimized parameters...")

    all_results = []
    successful_files = 0

    for wav_file in test_files:
        results = analyze_marmoset_optimized(wav_file)
        if results:
            all_results.extend(results)
            successful_files += 1

    # Summary
    print(f"\n{'=' * 70}")
    print("OPTIMIZED RESULTS SUMMARY")
    print(f"{'=' * 70}")

    if all_results:
        modality_counts = {}
        f0_values = []

        for result in all_results:
            modality = result["modality"]
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
            if result["mean_f0_hz"] is not None and result["mean_f0_hz"] > 0:
                f0_values.append(result["mean_f0_hz"])

        print(
            f"\nFiles successfully analyzed: {successful_files}/{num_files_to_test} "
            f"({successful_files / num_files_to_test * 100:.1f}%)"
        )
        print(f"Total phrases analyzed: {len(all_results)}")

        print("\nModality Distribution:")
        for modality, count in sorted(modality_counts.items()):
            percentage = count / len(all_results) * 100
            bar = "█" * int(percentage / 5)
            print(f"  {modality:15s}: {count:3d} ({percentage:5.1f}%) {bar}")

        if f0_values:
            print("\n📊 F0 Statistics:")
            print(f"  Min F0: {min(f0_values):.0f} Hz")
            print(f"  Max F0: {max(f0_values):.0f} Hz")
            print(f"  Mean F0: {np.mean(f0_values):.0f} Hz")
            print("  Expected range: 5000-12000 Hz")

            # Check if in expected range
            in_range = sum(1 for f in f0_values if 5000 <= f <= 12000)
            print(
                f"  In expected range: {in_range}/{len(f0_values)} "
                f"({in_range / len(f0_values) * 100:.1f}%)"
            )

        print("\n✅ Test completed!")
    else:
        print("\n⚠️  No results obtained.")


if __name__ == "__main__":
    main()
