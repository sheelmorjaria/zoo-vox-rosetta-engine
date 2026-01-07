#!/usr/bin/env python3
"""
Comprehensive Sperm Whale Dataset Analysis

Uses adaptive coda detection based on inter-click interval distribution.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, hilbert


def analyze_sperm_whale_file(filepath):
    """Analyze a sperm whale file with adaptive coda detection."""
    audio, sr = sf.read(filepath)
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # Detect clicks
    envelope = np.abs(hilbert(audio))
    threshold = np.mean(envelope) + 2 * np.std(envelope)
    peaks, _ = find_peaks(envelope, height=threshold, distance=int(0.005 * sr))

    if len(peaks) < 2:
        return {
            "filename": Path(filepath).name,
            "duration_s": len(audio) / sr,
            "total_clicks": len(peaks),
            "num_codas": 0,
            "coda_sizes": [],
        }

    # Calculate inter-click intervals
    intervals_ms = np.diff(peaks) / sr * 1000

    # Find adaptive threshold (99th percentile of ICIs)
    adaptive_threshold = np.percentile(intervals_ms, 99)

    # Segment codas using adaptive threshold
    codas = []
    current_coda = [0]

    for i in range(1, len(peaks)):
        gap_ms = intervals_ms[i - 1]
        if gap_ms <= adaptive_threshold:
            current_coda.append(i)
        else:
            if len(current_coda) >= 2:
                codas.append(current_coda)
            current_coda = [i]

    if len(current_coda) >= 2:
        codas.append(current_coda)

    return {
        "filename": Path(filepath).name,
        "duration_s": len(audio) / sr,
        "total_clicks": len(peaks),
        "click_rate": len(peaks) / (len(audio) / sr),
        "num_codas": len(codas),
        "coda_threshold_ms": adaptive_threshold,
        "coda_sizes": [len(c) for c in codas],
        "intervals_ms": intervals_ms,
    }


def main():
    """Comprehensive analysis of sperm whale dataset."""
    base_dir = Path.home() / "birdsong_analysis/data/Dominica_dataset/Signal_parts"
    files = sorted(list(base_dir.glob("*.wav")))[:15]

    print("=" * 100)
    print("SPERM WHALE DATASET COMPREHENSIVE ANALYSIS (Adaptive Coda Detection)")
    print("=" * 100)

    header = (
        f"{'File':<20} {'Dur(s)':>7} {'Clicks':>8} {'Click/s':>8} "
        f"{'Codas':>6} {'Threshold':>10}  {'Coda Size (mean±std)':>20}"
    )
    print(header)
    print("-" * 100)

    all_results = []
    for filepath in files:
        result = analyze_sperm_whale_file(filepath)
        all_results.append(result)

        coda_mean = np.mean(result["coda_sizes"]) if result["coda_sizes"] else 0
        coda_std = np.std(result["coda_sizes"]) if result["coda_sizes"] else 0

        row = (
            f"{result['filename']:<20} {result['duration_s']:7.0f} "
            f"{result['total_clicks']:8d} {result['click_rate']:8.1f} "
            f"{result['num_codas']:6d} {result['coda_threshold_ms']:10.1f}  "
            f"{coda_mean:6.0f}±{coda_std:5.0f}"
        )
        print(row)

    # Summary statistics
    print("=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    total_clicks = sum(r["total_clicks"] for r in all_results)
    total_duration = sum(r["duration_s"] for r in all_results)
    total_codas = sum(r["num_codas"] for r in all_results)

    files_with_codas = [r for r in all_results if r["num_codas"] > 0]
    files_without_codas = [r for r in all_results if r["num_codas"] == 0]

    print("\n📁 File Coverage:")
    print(f"  Total files analyzed: {len(all_results)}")
    with_coda_pct = len(files_with_codas) / len(all_results) * 100
    print(f"  Files with codas: {len(files_with_codas)} ({with_coda_pct:.1f}%)")
    without_coda_pct = len(files_without_codas) / len(all_results) * 100
    print(f"  Files without codas: {len(files_without_codas)} ({without_coda_pct:.1f}%)")

    print("\n📊 Click Statistics:")
    print(f"  Total clicks: {total_clicks:,}")
    print(f"  Total duration: {total_duration:.0f}s ({total_duration / 60:.1f} minutes)")
    print(f"  Average click rate: {total_clicks / total_duration:.1f} clicks/second")

    print("\n📊 Coda Statistics:")
    print(f"  Total codas detected: {total_codas}")
    codas_per_file = total_codas / len(all_results)
    codas_std = np.std([r["num_codas"] for r in all_results])
    print(f"  Codas per file: {codas_per_file:.1f} ± {codas_std:.1f}")

    if files_with_codas:
        # Combine all coda sizes
        all_coda_sizes = []
        for r in files_with_codas:
            all_coda_sizes.extend(r["coda_sizes"])

        print(f"\n  All Coda Sizes (n={len(all_coda_sizes)}):")
        print(f"    Mean: {np.mean(all_coda_sizes):.1f} clicks/coda")
        print(f"    Median: {np.median(all_coda_sizes):.1f} clicks/coda")
        print(f"    Std: {np.std(all_coda_sizes):.1f}")
        print(f"    Min: {np.min(all_coda_sizes)}")
        print(f"    Max: {np.max(all_coda_sizes)}")

        # Coda length distribution
        short_codas = sum(1 for s in all_coda_sizes if s < 10)
        medium_codas = sum(1 for s in all_coda_sizes if 10 <= s < 50)
        long_codas = sum(1 for s in all_coda_sizes if s >= 50)

        print("\n  Coda Length Distribution:")
        short_pct = short_codas / len(all_coda_sizes) * 100
        print(f"    SHORT (<10 clicks): {short_codas} ({short_pct:.1f}%)")
        medium_pct = medium_codas / len(all_coda_sizes) * 100
        print(f"    MEDIUM (10-49): {medium_codas} ({medium_pct:.1f}%)")
        long_pct = long_codas / len(all_coda_sizes) * 100
        print(f"    LONG (50+): {long_codas} ({long_pct:.1f}%)")

        # Inter-click interval analysis
        all_intervals = []
        for r in files_with_codas:
            if "intervals_ms" in r:
                all_intervals.extend(r["intervals_ms"])

        if all_intervals:
            print(f"\n📊 Inter-Click Interval Analysis (n={len(all_intervals)}):")
            print(f"    Mean: {np.mean(all_intervals):.2f} ms")
            print(f"    Median: {np.median(all_intervals):.2f} ms")
            print(f"    Std: {np.std(all_intervals):.2f} ms")

            for p in [50, 75, 90, 95, 99]:
                val = np.percentile(all_intervals, p)
                print(f"    {p}th percentile: {val:.2f} ms")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
