#!/usr/bin/env python3
"""
Bottlenose Dolphin Whistle Dataset Investigation

Deep investigation of dolphin whistle detection to understand:
1. Why 0/5 files detected phrases in multi-species test
2. Effect of 192 kHz sample rate on detection
3. Optimal parameters for dolphin whistles
4. Whistle characteristics and modality classification
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


def analyze_single_whistle(filepath, params):
    """Analyze a single dolphin whistle with detailed diagnostics."""
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

    # Find dominant frequency
    dom_freq_idx = np.argmax(pos_magnitude)
    dom_freq = pos_freqs[dom_freq_idx]

    print(f"Dominant frequency: {abs(dom_freq) / 1000:.1f} kHz")

    # Energy in different bands
    bands = [
        ("Low (0-5 kHz)", 0, 5000),
        ("Mid (5-10 kHz)", 5000, 10000),
        ("Whistle range (10-20 kHz)", 10000, 20000),
        ("High (20-40 kHz)", 20000, 40000),
        ("Ultrasonic (>40 kHz)", 40000, sr // 2),
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
    print("\n🔍 Testing phrase segmentation:")

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
                    print(f"  gap={min_gap:2d}ms, dur={min_dur:2d}ms: {len(phrases):2d} phrases ✓")
            except Exception as e:
                print(f"  gap={min_gap:2d}ms, dur={min_dur:2d}ms: Error - {e}")

    # Find best parameter combination
    best_params = max(results.items(), key=lambda x: x[1])
    best_gap = int(best_params[0].split(":")[1].split("_")[0])
    best_dur = int(best_params[0].split(":")[2].split("_")[0])
    max_phrases = best_params[1]

    if max_phrases == 0:
        print("\n⚠️  No phrases detected")
        # Try treating the whole file as one phrase
        print("\n🔍 Analyzing entire file as single phrase:")
        try:
            modality = analyzer.detect_modality(audio)
            probabilities = analyzer.get_modality_probabilities(audio)

            print(f"  Modality: {modality.name}")
            print(f"  Probabilities: {probabilities}")

            return {
                "filepath": filepath,
                "total_phrases": 0,
                "full_file_modality": modality.name,
                "probabilities": probabilities,
            }
        except Exception as e:
            print(f"  Error: {e}")
            return None

    print(f"\n✓ Best parameters: gap={best_gap}ms, dur={best_dur}ms ({max_phrases} phrases)")

    # Analyze with best parameters
    phrases = analyzer.segment_phrases(audio, min_gap_ms=best_gap, min_phrase_duration_ms=best_dur)

    # Analyze modality for all phrases
    print(f"\n📊 Modality Analysis ({len(phrases)} total phrases):")
    modality_counts = {}
    details = []

    for i, phrase in enumerate(phrases):
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
                "f0_mean": features.get("f0_mean"),
            }
        )

    # Print modality distribution
    print("\n  Modality Distribution:")
    for modality, count in sorted(modality_counts.items()):
        percentage = count / len(phrases) * 100
        bar = "█" * int(percentage / 10)
        expected = ""
        if modality == "HARMONIC":
            expected = " (expected for whistles)"
        print(f"    {modality:15s}: {count:2d} ({percentage:5.1f}%) {bar}{expected}")

    # Print detailed analysis for first 10 phrases
    print("\n  Detailed Analysis (first 10 phrases):")
    for d in details[:10]:
        print(f"\n    Phrase {d['index'] + 1}: {d['modality']} ({d['duration_ms']:.1f} ms)")
        print(f"      Probabilities: {d['probabilities']}")
        print(f"      Spectral centroid: {d['spectral_centroid'] / 1000:.1f} kHz")
        if d["f0_mean"] and d["f0_mean"] > 0:
            print(f"      F0: {d['f0_mean']:.0f} Hz")

    return {
        "filepath": filepath,
        "total_phrases": len(phrases),
        "modality_counts": modality_counts,
        "best_params": {"gap": best_gap, "dur": best_dur},
        "details": details,
    }


def main():
    """Investigate dolphin whistle dataset."""
    print("=" * 70)
    print("BOTTLENOSE DOLPHIN WHISTLE DATASET INVESTIGATION")
    print("=" * 70)

    data_dir = Path.home() / "birdsong_analysis" / "data" / "Whistle_Signals"

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    # Find all WAV files
    wav_files = list(data_dir.glob("**/*.wav"))
    print(f"\n📁 Found {len(wav_files)} WAV files")

    if len(wav_files) == 0:
        print("❌ No WAV files found")
        return

    # Test a representative sample
    num_files_to_test = 20
    test_files = random.sample(wav_files, min(num_files_to_test, len(wav_files)))

    print(f"\n🎲 Testing {len(test_files)} random files...\n")

    all_results = []
    successful = 0
    full_file_analyses = []

    for filepath in test_files:
        result = analyze_single_whistle(filepath, None)
        if result:
            if result["total_phrases"] > 0:
                all_results.append(result)
                successful += 1
            elif "full_file_modality" in result:
                full_file_analyses.append(result)

    # Summary
    print(f"\n{'=' * 70}")
    print("INVESTIGATION SUMMARY")
    print(f"{'=' * 70}")

    # Files with segmented phrases
    if all_results:
        print(f"\nFiles with phrase segmentation: {successful}/{len(test_files)}")

        # Aggregate modality counts
        total_phrases = sum(r["total_phrases"] for r in all_results)
        all_modality_counts = {}

        for result in all_results:
            for modality, count in result["modality_counts"].items():
                all_modality_counts[modality] = all_modality_counts.get(modality, 0) + 1

        print(f"\nTotal phrases across segmented files: {total_phrases}")
        print("\nAggregated Modality Distribution (by phrase count):")
        for modality, count in sorted(all_modality_counts.items()):
            percentage = count / total_phrases * 100
            bar = "█" * int(percentage / 10)
            expected = ""
            if modality == "HARMONIC":
                expected = " ✓ (expected for whistles)"
            elif modality == "FM_SWEEP":
                expected = " (whistles often have FM)"
            print(f"  {modality:15s}: {count:4d} ({percentage:5.1f}%) {bar}{expected}")

    # Files analyzed as whole
    if full_file_analyses:
        print(f"\nFiles analyzed as whole (no segmentation): {len(full_file_analyses)}")
        modality_counts = {}
        for result in full_file_analyses:
            m = result["full_file_modality"]
            modality_counts[m] = modality_counts.get(m, 0) + 1

        print("\n  Full-file Modality Distribution:")
        for modality, count in sorted(modality_counts.items()):
            percentage = count / len(full_file_analyses) * 100
            bar = "█" * int(percentage / 10)
            print(f"    {modality:15s}: {count:2d} ({percentage:5.1f}%) {bar}")

    print("\n💡 Key Findings:")
    total_analyzed = len(all_results) + len(full_file_analyses)
    analyzed_pct = total_analyzed / len(test_files) * 100
    print(
        f"  - Successfully analyzed: {total_analyzed}/{len(test_files)} ({analyzed_pct:.1f}%)"
    )

    if all_results or full_file_analyses:
        # Count HARMONIC across all
        harmonic_count = sum(1 for r in all_results if "HARMONIC" in r["modality_counts"])
        harmonic_count += sum(
            1 for r in full_file_analyses if r["full_file_modality"] == "HARMONIC"
        )

        if harmonic_count > 0:
            print(f"  - HARMONIC detected in {harmonic_count}/{total_analyzed} files ✓")

    print("\n🔬 Interpretation:")
    if len(all_results) > 0:
        print("  - Dolphin whistles successfully segmented and classified")
        print("  - Whistles are primarily HARMONIC with possible FM components")
    elif len(full_file_analyses) > 0:
        print("  - Whistles detected but not segmented (single continuous signal)")
        print("  - Analyzed as complete file rather than phrases")

    print(f"\n{'=' * 70}")
    print("✅ Investigation complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
