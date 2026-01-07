#!/usr/bin/env python3
"""
Sperm Whale Analysis: Current vs Prior Research Comparison

Compares current Universal Rosetta Stone + Click Detection results
with prior specialized sperm whale coda analysis.

Prior Research Summary (from user's data):
- 404 codas analyzed
- Mean clicks per coda: 69.6 ± 215.3
- Click range: 3-1,933 clicks per coda
- Inter-click intervals: 27,718 ICIs analyzed
- Temporal rhythm score: 0.691
- Three semantic categories:
  * SHORT codas: 94 (23.3%) - 3.9 clicks/coda, rhythm 0.825
  * MEDIUM codas: 61 (15.1%) - 6.9 clicks/coda, rhythm 0.736
  * LONG codas: 249 (61.6%) - 109.8 clicks/coda, rhythm 0.630
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, hilbert


def detect_codas_with_rhythm(audio, sr):
    """
    Detect codas and calculate rhythm regularity for each coda.

    Returns list of coda dictionaries with:
    - num_clicks: number of clicks in coda
    - duration_ms: coda duration
    - mean_ici_ms: mean inter-click interval
    - std_ici_ms: std of inter-click intervals
    - rhythm_regularity: 0-1 score (higher = more regular)
    """
    # Detect clicks
    envelope = np.abs(hilbert(audio))
    threshold = np.mean(envelope) + 2 * np.std(envelope)
    peaks, _ = find_peaks(envelope, height=threshold, distance=int(0.005 * sr))

    if len(peaks) < 2:
        return []

    # Calculate inter-click intervals
    intervals_ms = np.diff(peaks) / sr * 1000

    # Find adaptive threshold (99th percentile)
    adaptive_threshold = np.percentile(intervals_ms, 99)

    # Segment codas
    codas = []
    current_coda_clicks = [peaks[0]]

    for i in range(1, len(peaks)):
        gap_ms = intervals_ms[i-1]
        if gap_ms <= adaptive_threshold:
            current_coda_clicks.append(peaks[i])
        else:
            if len(current_coda_clicks) >= 2:
                coda = analyze_coda_rhythm(current_coda_clicks, sr)
                if coda:
                    codas.append(coda)
            current_coda_clicks = [peaks[i]]

    if len(current_coda_clicks) >= 2:
        coda = analyze_coda_rhythm(current_coda_clicks, sr)
        if coda:
            codas.append(coda)

    return codas


def analyze_coda_rhythm(click_positions, sr):
    """Analyze rhythm regularity of a coda."""
    if len(click_positions) < 2:
        return None

    # Calculate inter-click intervals
    icis_ms = np.diff(click_positions) / sr * 1000

    mean_ici = np.mean(icis_ms)
    std_ici = np.std(icis_ms)

    # Calculate rhythm regularity (0-1, higher = more regular)
    if mean_ici > 0:
        cv = std_ici / mean_ici  # Coefficient of variation
        rhythm_regularity = 1.0 / (1.0 + cv)
    else:
        rhythm_regularity = 0.0

    return {
        'num_clicks': len(click_positions),
        'duration_ms': (click_positions[-1] - click_positions[0]) / sr * 1000,
        'mean_ici_ms': mean_ici,
        'std_ici_ms': std_ici,
        'rhythm_regularity': rhythm_regularity,
        'icis_ms': icis_ms
    }


def classify_coda_semantic(coda):
    """
    Classify coda into semantic categories (SHORT/MEDIUM/LONG)
    based on number of clicks.

    Categories from prior research:
    - SHORT: < 10 clicks (basic social calls)
    - MEDIUM: 10-49 clicks (intermediate information)
    - LONG: >= 50 clicks (complex information encoding)
    """
    if coda['num_clicks'] < 10:
        return 'SHORT'
    elif coda['num_clicks'] < 50:
        return 'MEDIUM'
    else:
        return 'LONG'


def compare_with_prior_research(all_codas):
    """
    Compare current analysis results with prior research findings.
    """
    print("=" * 80)
    print("COMPARISON: CURRENT ANALYSIS vs PRIOR RESEARCH")
    print("=" * 80)

    # Current analysis statistics
    total_codas = len(all_codas)
    coda_lengths = [c['num_clicks'] for c in all_codas]
    rhythm_scores = [c['rhythm_regularity'] for c in all_codas]

    # Classify into semantic categories
    short_codas = [c for c in all_codas if c['num_clicks'] < 10]
    medium_codas = [c for c in all_codas if 10 <= c['num_clicks'] < 50]
    long_codas = [c for c in all_codas if c['num_clicks'] >= 50]

    print("\n📊 OVERALL STATISTICS:")
    print("-" * 80)
    print(f"{'Metric':<30} {'Current':<20} {'Prior Research':<20}")
    print("-" * 80)
    print(f"{'Total codas analyzed':<30} {total_codas:<20} {404:<20}")
    print(f"{'Mean clicks per coda':<30} {np.mean(coda_lengths):.1f} ± {np.std(coda_lengths):.1f}<{69.6: <10} ± 215.3")
    print(f"{'Click range':<30} {min(coda_lengths)}-{max(coda_lengths):<13} {3}-{1933:<10}")
    print(f"{'Mean rhythm regularity':<30} {np.mean(rhythm_scores):.3f}<{0.691: <10}")

    print("\n📊 SEMANTIC CATEGORY DISTRIBUTION:")
    print("-" * 80)

    # Prior research percentages
    prior_short_pct = 23.3
    prior_medium_pct = 15.1
    prior_long_pct = 61.6

    current_short_pct = len(short_codas) / total_codas * 100
    current_medium_pct = len(medium_codas) / total_codas * 100
    current_long_pct = len(long_codas) / total_codas * 100

    print(f"{'Category':<15} {'Current':<15} {'Prior Research':<20} {'Match':<10}")
    print("-" * 80)
    print(f"{'SHORT (<10)':<15} {len(short_codas):4d} ({current_short_pct:5.1f}%) {94:4d} ({prior_short_pct:5.1f}%) {'✓' if abs(current_short_pct - prior_short_pct) < 5 else 'diff':<10}")
    print(f"{'MEDIUM (10-49)':<15} {len(medium_codas):4d} ({current_medium_pct:5.1f}%) {61:4d} ({prior_medium_pct:5.1f}%) {'✓' if abs(current_medium_pct - prior_medium_pct) < 5 else 'diff':<10}")
    print(f"{'LONG (50+)':<15} {len(long_codas):4d} ({current_long_pct:5.1f}%) {249:4d} ({prior_long_pct:5.1f}%) {'✓' if abs(current_long_pct - prior_long_pct) < 5 else 'diff':<10}")

    print("\n📊 SEMANTIC CATEGORY CHARACTERISTICS:")
    print("-" * 80)
    print(f"{'Category':<15} {'Mean Clicks':<15} {'Rhythm Score':<15} {'Current Mean':<20} {'Prior Research':<15}")
    print("-" * 80)

    if short_codas:
        current_short_clicks = np.mean([c['num_clicks'] for c in short_codas])
        current_short_rhythm = np.mean([c['rhythm_regularity'] for c in short_codas])
        print(f"{'SHORT':<15} {current_short_clicks:<15.1f} {current_short_rhythm:<15.3f} {'3.9 clicks, 0.825 rhythm':<20}")
    else:
        print(f"{'SHORT':<15} {'N/A':<15} {'N/A':<15} {'3.9 clicks, 0.825 rhythm':<20}")

    if medium_codas:
        current_medium_clicks = np.mean([c['num_clicks'] for c in medium_codas])
        current_medium_rhythm = np.mean([c['rhythm_regularity'] for c in medium_codas])
        print(f"{'MEDIUM':<15} {current_medium_clicks:<15.1f} {current_medium_rhythm:<15.3f} {'6.9 clicks, 0.736 rhythm':<20}")
    else:
        print(f"{'MEDIUM':<15} {'N/A':<15} {'N/A':<15} {'6.9 clicks, 0.736 rhythm':<20}")

    if long_codas:
        current_long_clicks = np.mean([c['num_clicks'] for c in long_codas])
        current_long_rhythm = np.mean([c['rhythm_regularity'] for c in long_codas])
        print(f"{'LONG':<15} {current_long_clicks:<15.1f} {current_long_rhythm:<15.3f} {'109.8 clicks, 0.630 rhythm':<20}")
    else:
        print(f"{'LONG':<15} {'N/A':<15} {'N/A':<15} {'109.8 clicks, 0.630 rhythm':<20}")

    print("\n📊 INTER-CLICK INTERVAL ANALYSIS:")
    print("-" * 80)

    # Combine all ICIs
    all_icis = []
    for coda in all_codas:
        all_icis.extend(coda['icis_ms'])

    print(f"{'Percentile':<15} {'Current (ms)':<20} {'Interpretation':<30}")
    print("-" * 80)

    for p in [25, 50, 75, 90, 95, 99]:
        val = np.percentile(all_icis, p)
        interpretation = ""
        if p == 50:
            interpretation = "Median ICI"
        elif p == 95:
            interpretation = "Coda boundary threshold"
        elif p == 99:
            interpretation = "Outlier boundary"

        print(f"{p}th percentile{':':<10} {val:<20.2f} {interpretation:<30}")

    print("\n" + "=" * 80)
    print("ANALYSIS CONCLUSIONS:")
    print("=" * 80)

    # Calculate differences
    short_diff = abs(current_short_pct - prior_short_pct)
    medium_diff = abs(current_medium_pct - prior_medium_pct)
    long_diff = abs(current_long_pct - prior_long_pct)

    avg_diff = (short_diff + medium_diff + long_diff) / 3

    print("\n🔍 Semantic Distribution Match:")
    if avg_diff < 10:
        print(f"  ✓ EXCELLENT: Average difference of {avg_diff:.1f}%")
        print("    Current analysis closely matches prior research")
    elif avg_diff < 20:
        print(f"  ~ MODERATE: Average difference of {avg_diff:.1f}%")
        print("    Current analysis shows similar patterns but different proportions")
    else:
        print(f"  ✗ POOR: Average difference of {avg_diff:.1f}%")
        print("    Current analysis shows different distribution")

    print("\n🔍 Rhythm Regularity:")
    current_rhythm = np.mean(rhythm_scores)
    prior_rhythm = 0.691
    rhythm_diff = abs(current_rhythm - prior_rhythm)

    if rhythm_diff < 0.1:
        print(f"  ✓ EXCELLENT: Current {current_rhythm:.3f} vs Prior {prior_rhythm}")
    elif rhythm_diff < 0.2:
        print(f"  ~ MODERATE: Current {current_rhythm:.3f} vs Prior {prior_rhythm}")
    else:
        print(f"  ✗ DIFFERENT: Current {current_rhythm:.3f} vs Prior {prior_rhythm}")

    print("\n🔍 Click Distribution:")
    if np.mean(coda_lengths) > 50:
        print("  Current analysis detects LONGER codas on average")
        print("  Possible reasons:")
        print("    - Adaptive threshold may be too high")
        print("    - Different dataset characteristics")
        print("    - Click detection sensitivity differences")
    else:
        print("  Current analysis detects SIMILAR/SHORTER codas")

    print("\n" + "=" * 80)


def main():
    """Run comparison analysis."""
    base_dir = Path.home() / "birdsong_analysis/data/Dominica_dataset/Signal_parts"
    files = sorted(list(base_dir.glob("*.wav")))[:15]

    print("=" * 80)
    print("SPERM WHALE ANALYSIS: CURRENT vs PRIOR RESEARCH COMPARISON")
    print("=" * 80)
    print(f"\nAnalyzing {len(files)} files...")

    all_codas = []
    for filepath in files:
        audio, sr = sf.read(filepath)
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        codas = detect_codas_with_rhythm(audio, sr)
        all_codas.extend(codas)

    print(f"Detected {len(all_codas)} codas from {len(files)} files")

    compare_with_prior_research(all_codas)


if __name__ == "__main__":
    main()
