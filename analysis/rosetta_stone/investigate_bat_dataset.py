#!/usr/bin/env python3
"""
Egyptian Fruit Bat Dataset Investigation

Deep investigation of bat vocalization detection to understand:
1. Why only 2/5 files detected phrases
2. Why only TRANSIENT detected (expected FM_SWEEP)
3. Effect of 250 kHz sample rate on detection
4. Optimal parameters for bat vocalizations
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


def load_wav_file(filepath, target_sr=None):
    """Load a WAV file - preserve native sample rate."""
    if not HAS_SOUNDFILE:
        raise ImportError("soundfile library required")

    try:
        audio, sr = sf.read(filepath)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        return audio, sr
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def analyze_single_file(filepath, params):
    """Analyze a single bat file with detailed diagnostics."""
    print(f"\n{'=' * 70}")
    print(f"File: {Path(filepath).name}")
    print(f"{'=' * 70}")

    audio, sr = load_wav_file(filepath)
    if audio is None:
        return None

    duration_ms = len(audio) / sr * 1000
    rms = np.sqrt(np.mean(audio**2))

    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration_ms:.0f} ms")
    print(f"Samples: {len(audio)}")
    print(f"RMS amplitude: {rms:.6f}")

    # Frequency analysis
    from scipy.fft import fft, fftfreq

    fft_result = fft(audio)
    freqs = fftfreq(len(audio), 1 / sr)
    magnitude = np.abs(fft_result)

    pos_freqs = freqs[: len(freqs) // 2]
    pos_magnitude = magnitude[: len(magnitude) // 2]

    # Energy in different bands
    bands = [
        ("Low (0-10 kHz)", 0, 10000),
        ("Bat range (10-30 kHz)", 10000, 30000),
        ("Mid (30-60 kHz)", 30000, 60000),
        ("High (60-100 kHz)", 60000, 100000),
        ("Ultrasonic (>100 kHz)", 100000, sr // 2),
    ]

    print("\n📊 Energy Distribution:")
    for band_name, low, high in bands:
        mask = (pos_freqs >= low) & (pos_freqs < high)
        band_energy = np.sum(pos_magnitude[mask] ** 2)
        total_energy = np.sum(pos_magnitude**2)
        percentage = band_energy / total_energy * 100
        if percentage > 1:
            bar = "█" * int(percentage / 3)
            print(f"  {band_name:25s}: {percentage:5.1f}% {bar}")

    # Test multiple parameter combinations
    print("\n🔍 Testing phrase segmentation with different parameters:")

    analyzer = UniversalRosettaStone(sample_rate=sr)

    results = {}
    for min_gap in [5, 10, 20, 50]:
        for min_dur in [2, 5, 10]:
            try:
                phrases = analyzer.segment_phrases(
                    audio, min_gap_ms=min_gap, min_phrase_duration_ms=min_dur
                )
                key = f"gap:{min_gap}_dur:{min_dur}"
                results[key] = len(phrases)
                if len(phrases) > 0:
                    print(f"  gap={min_gap:2d}ms, dur={min_dur:2d}ms: {len(phrases):3d} phrases ✓")
            except Exception as e:
                print(f"  gap={min_gap:2d}ms, dur={min_dur:2d}ms: Error - {e}")

    # Find best parameter combination
    best_params = max(results.items(), key=lambda x: x[1])
    best_gap = int(best_params[0].split(":")[1].split("_")[0])
    best_dur = int(best_params[0].split(":")[2].split("_")[0])
    max_phrases = best_params[1]

    if max_phrases == 0:
        print("\n⚠️  No phrases detected with any parameter combination")
        return None

    print(f"\n✓ Best parameters: gap={best_gap}ms, dur={best_dur}ms ({max_phrases} phrases)")

    # Analyze with best parameters
    phrases = analyzer.segment_phrases(audio, min_gap_ms=best_gap, min_phrase_duration_ms=best_dur)

    # Analyze modality for first 20 phrases
    print("\n📊 Modality Analysis (first 20 phrases):")
    modality_counts = {}
    details = []

    for i, phrase in enumerate(phrases[:20]):
        modality = analyzer.detect_modality(phrase.data)
        probabilities = analyzer.get_modality_probabilities(phrase.data)
        features = phrase.features

        modality_counts[modality.name] = modality_counts.get(modality.name, 0) + 1

        details.append(
            {
                "index": i,
                "modality": modality.name,
                "probabilities": probabilities,
                "duration_ms": features.get("duration_ms", len(phrase.data) / sr * 1000),
                "spectral_centroid": features.get("spectral_centroid", 0),
            }
        )

    # Print modality distribution
    print(f"\n  Modality Distribution ({len(phrases)} total phrases):")
    for modality, count in sorted(modality_counts.items()):
        percentage = count / len(phrases) * 100
        bar = "█" * int(percentage / 10)
        print(f"    {modality:15s}: {count:3d} ({percentage:5.1f}%) {bar}")

    # Print detailed analysis for first 10 phrases
    print("\n  Detailed Analysis (first 10 phrases):")
    for d in details[:10]:
        print(f"\n    Phrase {d['index'] + 1}: {d['modality']} ({d['duration_ms']:.1f} ms)")
        print(f"      Probabilities: {d['probabilities']}")
        print(f"      Spectral centroid: {d['spectral_centroid'] / 1000:.1f} kHz")

    return {
        "filepath": filepath,
        "total_phrases": len(phrases),
        "modality_counts": modality_counts,
        "best_params": {"gap": best_gap, "dur": best_dur},
        "details": details,
    }


def main():
    """Investigate Egyptian fruit bat dataset."""
    print("=" * 70)
    print("EGYPTIAN FRUIT BAT DATASET INVESTIGATION")
    print("=" * 70)

    data_dir = Path.home() / "birdsong_analysis" / "data" / "egyptian_fruit_bat_10k" / "audio"

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    wav_files = list(data_dir.glob("*.wav"))
    print(f"\n📁 Found {len(wav_files)} WAV files")

    # Test a representative sample
    num_files_to_test = 20
    test_files = random.sample(wav_files, min(num_files_to_test, len(wav_files)))

    print(f"\n🎲 Testing {len(test_files)} random files...\n")

    all_results = []
    successful = 0

    for filepath in test_files:
        result = analyze_single_file(filepath, None)
        if result:
            all_results.append(result)
            successful += 1

    # Summary
    print(f"\n{'=' * 70}")
    print("INVESTIGATION SUMMARY")
    print(f"{'=' * 70}")

    if all_results:
        success_pct = successful / len(test_files) * 100
        print(f"\nFiles successfully analyzed: {successful}/{len(test_files)} ({success_pct:.1f}%)")

        # Aggregate modality counts
        total_phrases = sum(r["total_phrases"] for r in all_results)
        all_modality_counts = {}

        for result in all_results:
            for modality, count in result["modality_counts"].items():
                all_modality_counts[modality] = all_modality_counts.get(modality, 0) + count

        print(f"\nTotal phrases across all files: {total_phrases}")
        print("\nAggregated Modality Distribution:")
        for modality, count in sorted(all_modality_counts.items()):
            percentage = count / total_phrases * 100
            bar = "█" * int(percentage / 10)
            expected = ""
            if modality == "FM_SWEEP":
                expected = " (expected for bats)"
            elif modality == "HARMONIC":
                expected = " (social calls)"
            print(f"  {modality:15s}: {count:4d} ({percentage:5.1f}%) {bar}{expected}")

        # Parameter analysis
        print("\n📊 Most Common Parameters:")
        param_counts = {}
        for result in all_results:
            key = f"gap={result['best_params']['gap']}, dur={result['best_params']['dur']}"
            param_counts[key] = param_counts.get(key, 0) + 1

        for params, count in sorted(param_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {params}: {count} files")

        print("\n💡 Key Findings:")
        if "FM_SWEEP" in all_modality_counts:
            fm_pct = all_modality_counts["FM_SWEEP"] / total_phrases * 100
            print(f"  ✓ FM_SWEEP detected: {fm_pct:.1f}% of phrases")
        else:
            print("  ⚠️  No FM_SWEEP detected - all TRANSIENT")

        if "HARMONIC" in all_modality_counts:
            harm_pct = all_modality_counts["HARMONIC"] / total_phrases * 100
            print(f"  ✓ HARMONIC (social calls) detected: {harm_pct:.1f}% of phrases")

        print("\n🔬 Interpretation:")
        if "FM_SWEEP" not in all_modality_counts:
            print("  Predominance of TRANSIENT suggests these may be:")
            print("  - Echolocation clicks/pulses")
            print("  - Very short FM sweeps that fall below detection threshold")
            print("  - High-frequency content (>100 kHz) that's hard to analyze")
        else:
            print("  Mixed modalities detected, consistent with bat communication:")

    print(f"\n{'=' * 70}")
    print("✅ Investigation complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
