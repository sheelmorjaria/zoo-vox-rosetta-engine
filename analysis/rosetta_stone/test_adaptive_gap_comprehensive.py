#!/usr/bin/env python3
"""
Comprehensive Test: Adaptive Gap Threshold Enhancement

Compares phrase detection with and without adaptive gap threshold
across multiple sperm whale files.
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


def compare_adaptive_vs_fixed(filepath, duration_sec=10):
    """Compare adaptive vs fixed gap threshold on a file."""
    audio, sr = sf.read(filepath)
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # Use first N seconds
    audio = audio[: int(duration_sec * sr)]

    analyzer = UniversalRosettaStone(sample_rate=sr)

    # Test with adaptive gap (default max 50ms)
    phrases_adaptive = analyzer.segment_phrases(audio, min_gap_ms=50.0, use_adaptive_gap=True)

    # Test with fixed 50ms gap (no adaptive)
    phrases_fixed_50 = analyzer.segment_phrases(audio, min_gap_ms=50.0, use_adaptive_gap=False)

    # Test with fixed 100ms gap
    phrases_fixed_100 = analyzer.segment_phrases(audio, min_gap_ms=100.0, use_adaptive_gap=False)

    # Get adaptive threshold value
    adaptive_threshold = analyzer._calculate_adaptive_gap_threshold(audio)

    return {
        "filename": Path(filepath).name,
        "duration_sec": duration_sec,
        "adaptive_threshold_ms": adaptive_threshold,
        "phrases_adaptive": len(phrases_adaptive),
        "phrases_fixed_50": len(phrases_fixed_50),
        "phrases_fixed_100": len(phrases_fixed_100),
        "improvement_50": len(phrases_adaptive) - len(phrases_fixed_50),
        "improvement_100": len(phrases_adaptive) - len(phrases_fixed_100),
    }


def main():
    """Run comprehensive comparison test."""
    if not HAS_SOUNDFILE:
        print("soundfile library required")
        return

    base_dir = Path.home() / "birdsong_analysis/data/Dominica_dataset/Signal_parts"

    if not base_dir.exists():
        print(f"Data directory not found: {base_dir}")
        return

    # Test on first 10 files
    files = sorted(list(base_dir.glob("*.wav")))[:10]

    print("=" * 90)
    print("COMPREHENSIVE ADAPTIVE GAP TEST (Sperm Whale Dataset)")
    print("=" * 90)

    results = []
    for filepath in files:
        result = compare_adaptive_vs_fixed(filepath, duration_sec=10)
        results.append(result)

        print(f"\n{result['filename'][:25]:25s} (10s)")
        print(f"  Adaptive threshold: {result['adaptive_threshold_ms']:6.2f} ms")
        print("  Phrases detected:")
        print(f"    Adaptive (max 50ms): {result['phrases_adaptive']:3d}")
        print(
            f"    Fixed 50ms:          {result['phrases_fixed_50']:3d}  "
            f"(improvement: {result['improvement_50']:+3d})"
        )
        print(
            f"    Fixed 100ms:         {result['phrases_fixed_100']:3d}  "
            f"(improvement: {result['improvement_100']:+3d})"
        )

    # Summary statistics
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    total_phrases_adaptive = sum(r["phrases_adaptive"] for r in results)
    total_phrases_fixed_50 = sum(r["phrases_fixed_50"] for r in results)
    total_phrases_fixed_100 = sum(r["phrases_fixed_100"] for r in results)

    files_with_improvement_50 = sum(1 for r in results if r["improvement_50"] > 0)
    files_with_improvement_100 = sum(1 for r in results if r["improvement_100"] > 0)

    print(f"\nTotal phrases detected (across {len(results)} files, 10s each):")
    print(f"  Adaptive gap (max 50ms): {total_phrases_adaptive:3d}")
    print(f"  Fixed 50ms gap:          {total_phrases_fixed_50:3d}")
    print(f"  Fixed 100ms gap:         {total_phrases_fixed_100:3d}")

    print("\nFiles with improvement:")
    print(
        f"  vs Fixed 50ms:  {files_with_improvement_50}/{len(results)} "
        f"({files_with_improvement_50 / len(results) * 100:.1f}%)"
    )
    print(
        f"  vs Fixed 100ms: {files_with_improvement_100}/{len(results)} "
        f"({files_with_improvement_100 / len(results) * 100:.1f}%)"
    )

    if total_phrases_adaptive > 0:
        improvement_pct_50 = (
            (total_phrases_adaptive - total_phrases_fixed_50) / max(total_phrases_fixed_50, 1)
        ) * 100
        improvement_pct_100 = (
            (total_phrases_adaptive - total_phrases_fixed_100) / max(total_phrases_fixed_100, 1)
        ) * 100

        print("\nOverall improvement:")
        print(f"  vs Fixed 50ms:  {improvement_pct_50:+.1f}%")
        print(f"  vs Fixed 100ms: {improvement_pct_100:+.1f}%")

    # Adaptive threshold statistics
    thresholds = [r["adaptive_threshold_ms"] for r in results]
    print("\nAdaptive threshold statistics:")
    print(f"  Mean: {np.mean(thresholds):.2f} ms")
    print(f"  Median: {np.median(thresholds):.2f} ms")
    print(f"  Min: {np.min(thresholds):.2f} ms")
    print(f"  Max: {np.max(thresholds):.2f} ms")
    print(f"  Std: {np.std(thresholds):.2f} ms")

    print("\n" + "=" * 90)
    print("✅ Comprehensive test complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()
