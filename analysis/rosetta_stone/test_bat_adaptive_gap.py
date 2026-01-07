#!/usr/bin/env python3
"""
Egyptian Fruit Bat Adaptive Gap Test

Tests adaptive gap threshold on Egyptian fruit bat dataset.
Uses a representative subset of the 91,080 available files.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from universal_rosetta_stone import UniversalRosettaStone

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def test_bat_file(filepath, duration_sec=2):
    """Test adaptive gap on a single bat file."""
    try:
        audio, sr = sf.read(filepath)
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        # Use first N seconds (bat files are short)
        max_samples = int(duration_sec * sr)
        audio = audio[:max_samples]

        analyzer = UniversalRosettaStone(sample_rate=sr)

        # Detect overall modality
        overall_modality = analyzer._detect_overall_modality(audio)

        # Calculate adaptive threshold
        adaptive_threshold = analyzer._calculate_adaptive_gap_threshold(audio)

        # Test with adaptive gap
        phrases_adaptive = analyzer.segment_phrases(audio, min_gap_ms=50.0, use_adaptive_gap=True)

        # Test without adaptive gap
        phrases_fixed = analyzer.segment_phrases(audio, min_gap_ms=50.0, use_adaptive_gap=False)

        # Get modality distribution
        modality_counts = {}
        for phrase in phrases_adaptive:
            modality_counts[phrase.modality.name] = modality_counts.get(phrase.modality.name, 0) + 1

        # Calculate click rate
        envelope = np.abs(audio)
        from scipy.signal import find_peaks, hilbert

        analytic = hilbert(audio)
        envelope = np.abs(analytic)
        threshold = np.mean(envelope) + 2 * np.std(envelope)
        peaks, _ = find_peaks(envelope, height=threshold, distance=int(0.005 * sr))
        click_rate = len(peaks) / (len(audio) / sr)

        return {
            "filename": Path(filepath).name,
            "duration_sec": len(audio) / sr,
            "sample_rate": sr,
            "overall_modality": overall_modality.name,
            "adaptive_threshold_ms": adaptive_threshold,
            "phrases_adaptive": len(phrases_adaptive),
            "phrases_fixed": len(phrases_fixed),
            "improvement": len(phrases_adaptive) - len(phrases_fixed),
            "modality_distribution": modality_counts,
            "click_rate": click_rate,
        }
    except Exception as e:
        return {"error": str(e), "filename": Path(filepath).name}


def main():
    """Test adaptive gap on representative bat file subset."""
    if not HAS_SOUNDFILE:
        print("soundfile library required")
        return

    # Find bat data directory
    bat_dir = Path("../data/egyptian_fruit_bats/audio")
    if not bat_dir.exists():
        bat_dir = Path.home() / "birdsong_analysis/data/egyptian_fruit_bats/audio"

    if not bat_dir.exists():
        print(f"Bat data directory not found: {bat_dir}")
        return

    # Get all wav files
    all_files = sorted(list(bat_dir.glob("*.wav")))
    print(f"📁 Found {len(all_files):,} bat audio files")

    # Create representative subset
    # Sample evenly across the file range
    num_to_test = 50
    if len(all_files) > num_to_test:
        # Select files evenly distributed across the range
        indices = np.linspace(0, len(all_files) - 1, num_to_test, dtype=int)
        test_files = [all_files[i] for i in indices]
    else:
        test_files = all_files

    print(f"🎲 Testing representative subset of {len(test_files)} files\n")

    print("=" * 80)
    print("EGYPTIAN FRUIT BAT ADAPTIVE GAP TEST")
    print("=" * 80)

    results = []
    errors = []

    for i, filepath in enumerate(test_files):
        result = test_bat_file(filepath, duration_sec=2)

        if "error" in result:
            errors.append(result)
            continue

        results.append(result)

        # Print progress every 10 files
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(test_files)} files processed...")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if not results:
        print("⚠️  No successful results")
        if errors:
            print(f"\nErrors encountered: {len(errors)}")
            for e in errors[:5]:
                print(f"  {e['filename']}: {e['error']}")
        return

    print(f"\n📊 Files successfully analyzed: {len(results)}/{len(test_files)}")

    # Overall statistics
    print("\n📊 MODALITY DISTRIBUTION:")
    modality_counts = {}
    for r in results:
        m = r["overall_modality"]
        modality_counts[m] = modality_counts.get(m, 0) + 1

    for modality, count in sorted(modality_counts.items()):
        percentage = count / len(results) * 100
        print(f"  {modality:15s}: {count:3d} ({percentage:5.1f}%)")

    print("\n📊 CLICK RATE STATISTICS:")
    click_rates = [r["click_rate"] for r in results]
    print(f"  Mean: {np.mean(click_rates):.1f} clicks/second")
    print(f"  Median: {np.median(click_rates):.1f} clicks/second")
    print(f"  Range: {np.min(click_rates):.1f} - {np.max(click_rates):.1f} clicks/second")

    print("\n📊 ADAPTIVE THRESHOLD STATISTICS:")
    thresholds = [r["adaptive_threshold_ms"] for r in results]
    print(f"  Mean: {np.mean(thresholds):.2f} ms")
    print(f"  Median: {np.median(thresholds):.2f} ms")
    print(f"  Range: {np.min(thresholds):.2f} - {np.max(thresholds):.2f} ms")
    print(f"  Std: {np.std(thresholds):.2f} ms")

    print("\n📊 PHRASE DETECTION:")
    total_phrases_adaptive = sum(r["phrases_adaptive"] for r in results)
    total_phrases_fixed = sum(r["phrases_fixed"] for r in results)
    files_with_phrases_adaptive = sum(1 for r in results if r["phrases_adaptive"] > 0)
    files_with_phrases_fixed = sum(1 for r in results if r["phrases_fixed"] > 0)

    print(f"  Total phrases (adaptive): {total_phrases_adaptive}")
    print(f"  Total phrases (fixed 50ms): {total_phrases_fixed}")
    print(
        f"  Files with phrases (adaptive): {files_with_phrases_adaptive}/{len(results)} "
        f"({files_with_phrases_adaptive / len(results) * 100:.1f}%)"
    )
    print(
        f"  Files with phrases (fixed): {files_with_phrases_fixed}/{len(results)} "
        f"({files_with_phrases_fixed / len(results) * 100:.1f}%)"
    )

    # Improvement analysis
    improvements = [r["improvement"] for r in results]
    files_with_improvement = sum(1 for i in improvements if i > 0)
    total_improvement = sum(improvements)

    print("\n📊 IMPROVEMENT:")
    print(
        f"  Files with improvement: {files_with_improvement}/{len(results)} "
        f"({files_with_improvement / len(results) * 100:.1f}%)"
    )
    print(f"  Total additional phrases: +{total_improvement}")

    if files_with_improvement > 0:
        avg_improvement = total_improvement / files_with_improvement
        print(f"  Average improvement per file: +{avg_improvement:.1f} phrases")

    # Modality distribution of detected phrases
    print("\n📊 DETECTED PHRASE MODALITY:")
    phrase_modality_counts = {}
    for r in results:
        for modality, count in r["modality_distribution"].items():
            phrase_modality_counts[modality] = phrase_modality_counts.get(modality, 0) + count

    total_detected_phrases = sum(phrase_modality_counts.values())
    if total_detected_phrases > 0:
        for modality, count in sorted(phrase_modality_counts.items()):
            percentage = count / total_detected_phrases * 100
            print(f"  {modality:15s}: {count:4d} ({percentage:5.1f}%)")

    print("\n" + "=" * 80)
    print("✅ Egyptian fruit bat test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
