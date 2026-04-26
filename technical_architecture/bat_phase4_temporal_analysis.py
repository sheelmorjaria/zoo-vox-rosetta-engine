#!/usr/bin/env python3
"""
Phase 4: Temporal Duration Analysis
====================================

This analysis validates the "Frame Hypothesis" (Openers vs. Closers) and
"Rigid Idiom" hypothesis through temporal duration analysis:

HYPOTHESES:
1. Opener vs. Closer Duration: Openers (Position 0) should be shorter (staccato)
   than Closers (Position 1).
2. Rigid Idiom Timing: LRN-6 segments should have very low variance in duration
   (Coefficient of Variation < 0.5) compared to general population.
3. Contextual Modulation: Duration may vary by behavioral context, indicating
   prosodic modulation.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_cache_data(cache_dirs: list[str]) -> list[dict]:
    """Load all cache data from multiple directories"""
    all_data = []

    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            continue

        for cache_file in sorted(cache_path.glob("*.json")):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            except Exception:
                pass

    return all_data


def extract_durations(cache_data: list[dict]) -> dict[int, list[float]]:
    """
    Extract duration data for each segment ID.

    Returns a dictionary mapping segment_id -> list of durations (in ms)
    """
    segment_durations: dict[int, list[float]] = defaultdict(list)

    for entry in cache_data:
        seg_id = entry.get("segment_idx", entry.get("cluster_id", -1))
        if seg_id < 0:
            continue

        # Try multiple sources for duration
        duration_ms = 0.0

        # 1. From features array (index 1 is duration_ms in RosettaFeatures)
        features = entry.get("features", [])
        if features and len(features) > 1:
            duration_ms = features[1]

        # 2. From start/end_ms as backup
        if duration_ms <= 0:
            start_ms = entry.get("start_ms", 0)
            end_ms = entry.get("end_ms", 0)
            if start_ms > 0 and end_ms > 0:
                duration_ms = end_ms - start_ms

        # 3. From duration_ms directly
        if duration_ms <= 0:
            duration_ms = entry.get("duration_ms", 0)

        if duration_ms > 0:
            segment_durations[seg_id].append(duration_ms)

    return segment_durations


def extract_context_durations(cache_data: list[dict]) -> dict[int, dict[int, list[float]]]:
    """
    Extract duration data grouped by segment ID and context ID.

    Returns: {segment_id: {context_id: [durations]}}
    """
    context_durations: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for entry in cache_data:
        seg_id = entry.get("segment_idx", entry.get("cluster_id", -1))
        context_id = entry.get("context_id", -1)

        if seg_id < 0:
            continue

        # Get duration
        duration_ms = 0.0
        features = entry.get("features", [])
        if features and len(features) > 1:
            duration_ms = features[1]

        if duration_ms <= 0:
            start_ms = entry.get("start_ms", 0)
            end_ms = entry.get("end_ms", 0)
            if start_ms > 0 and end_ms > 0:
                duration_ms = end_ms - start_ms

        if duration_ms > 0:
            context_durations[seg_id][context_id].append(duration_ms)

    return context_durations


def compute_duration_stats(durations: list[float], group_name: str) -> dict:
    """
    Compute duration statistics including Coefficient of Variation (CV).

    CV = std / mean
    - Low CV (< 0.5) = Fixed/stereotyped duration
    - High CV (> 1.0) = Variable/modulated duration
    """
    if not durations:
        return None

    durations = np.array(durations)
    mean_dur = float(np.mean(durations))
    std_dur = float(np.std(durations))
    cv = (std_dur / mean_dur) if mean_dur > 0 else 0.0

    return {
        "Group": group_name,
        "Count": len(durations),
        "Avg_Duration_ms": mean_dur,
        "Std_Dev_ms": std_dur,
        "CV_Stereotypy": cv,
        "Min_ms": float(np.min(durations)),
        "Max_ms": float(np.max(durations)),
        "Median_ms": float(np.median(durations)),
        "Q25_ms": float(np.percentile(durations, 25)),
        "Q75_ms": float(np.percentile(durations, 75)),
    }


def analyze_group_durations(
    segment_durations: dict[int, list[float]],
    groups: dict[str, list[int]],
) -> tuple[list[dict], dict[str, dict]]:
    """
    Analyze duration statistics for each segment group.
    Returns (results_list, per_segment_stats)
    """
    results = []
    per_segment_stats = {}

    for group_name, segment_ids in groups.items():
        # Collect all durations for this group
        all_durations = []
        segment_stats = {}

        for seg_id in segment_ids:
            if seg_id in segment_durations:
                seg_durs = segment_durations[seg_id]
                all_durations.extend(seg_durs)
                segment_stats[seg_id] = compute_duration_stats(seg_durs, f"Seg_{seg_id}")

        per_segment_stats[group_name] = segment_stats

        # Compute aggregate stats
        stats = compute_duration_stats(all_durations, group_name)
        if stats:
            results.append(stats)

    return results, per_segment_stats


def compare_duration_variance(group_results: list[dict]) -> dict:
    """
    Compare variance (CV) across groups to test stereotypy.
    """
    comparison = {}

    for result in group_results:
        group_name = result["Group"]
        cv = result["CV_Stereotypy"]

        if cv < 0.3:
            stereotypy = "HIGHLY FIXED"
            interpretation = "Duration is extremely rigid (fixed code)"
        elif cv < 0.5:
            stereotypy = "FIXED"
            interpretation = "Duration is stereotyped (likely fixed sequence)"
        elif cv < 1.0:
            stereotypy = "MODERATE"
            interpretation = "Duration shows moderate variation"
        else:
            stereotypy = "VARIABLE"
            interpretation = "Duration is highly variable (prosodic modulation)"

        comparison[group_name] = {
            "cv": cv,
            "stereotypy": stereotypy,
            "interpretation": interpretation,
        }

    return comparison


def analyze_context_influence(
    context_durations: dict[int, dict[int, list[float]]],
    target_segments: list[int],
) -> dict:
    """
    Analyze if duration varies by behavioral context.
    """
    results = {}

    for seg_id in target_segments:
        if seg_id not in context_durations:
            continue

        contexts = context_durations[seg_id]
        if len(contexts) < 2:
            continue

        context_stats = {}
        for ctx_id, durations in contexts.items():
            if len(durations) >= 3:  # Need enough samples
                mean_dur = float(np.mean(durations))
                std_dur = float(np.std(durations))
                context_stats[ctx_id] = {
                    "mean_ms": mean_dur,
                    "std_ms": std_dur,
                    "n": len(durations),
                }

        if len(context_stats) >= 2:
            # Check for significant variation across contexts
            means = [s["mean_ms"] for s in context_stats.values()]
            overall_range = max(means) - min(means)
            overall_mean = float(np.mean(means))

            results[seg_id] = {
                "contexts": context_stats,
                "duration_range_ms": overall_range,
                "pct_variation": (overall_range / overall_mean * 100) if overall_mean > 0 else 0,
                "contextually_modulated": overall_range > 10,  # >10ms difference
            }

    return results


def compute_population_baseline(segment_durations: dict[int, list[float]]) -> dict:
    """
    Compute baseline statistics across all segments for comparison.
    """
    all_durations = []
    segment_cvs = []

    for seg_id, durations in segment_durations.items():
        all_durations.extend(durations)
        if len(durations) >= 5:
            cv = (
                float(np.std(durations)) / float(np.mean(durations))
                if np.mean(durations) > 0
                else 0
            )
            segment_cvs.append(cv)

    return {
        "total_segments": len(segment_durations),
        "total_calls": len(all_durations),
        "global_mean_ms": float(np.mean(all_durations)) if all_durations else 0,
        "global_std_ms": float(np.std(all_durations)) if all_durations else 0,
        "global_cv": float(np.std(all_durations)) / float(np.mean(all_durations))
        if all_durations and np.mean(all_durations) > 0
        else 0,
        "mean_segment_cv": float(np.mean(segment_cvs)) if segment_cvs else 0,
        "median_segment_cv": float(np.median(segment_cvs)) if segment_cvs else 0,
    }


def main():
    print("=" * 80)
    print("PHASE 4: TEMPORAL DURATION ANALYSIS")
    print("=" * 80)

    # Define analysis targets from Phase 2 findings
    targets = {
        "Openers": [384, 264, 1014, 484, 454],
        "Closers": [444, 304, 544, 404, 394],
        "LRN-6_Idiom": [114, 464, 604, 324, 94, 714],
        "Top_Bigram": [764, 304],  # From [764, 304]
        "Mid_Position": [114, 464, 604, 324],  # Middle positions in n-grams
        "All_Segments": list(range(0, 100)),  # Baseline comparison
    }

    # Load cache data
    cache_dirs = [
        "bat_nbd_cache_parallel",
        "bat_fm_cache",
        "bat_nbd_cache_full",
    ]

    print("\n[1] Loading cache data...")
    cache_data = load_cache_data(cache_dirs)
    print(f"Total entries loaded: {len(cache_data):,}")

    if not cache_data:
        print("ERROR: No cache data found")
        return

    # Extract durations
    print("\n[2] Extracting temporal duration data...")
    segment_durations = extract_durations(cache_data)
    print(f"Segments with duration data: {len(segment_durations)}")

    # Compute population baseline
    print("\n[3] Computing population baseline...")
    baseline = compute_population_baseline(segment_durations)
    print(f"Total calls analyzed: {baseline['total_calls']:,}")
    print(f"Global mean duration: {baseline['global_mean_ms']:.1f} ms")
    print(f"Global CV: {baseline['global_cv']:.3f}")
    print(f"Median segment CV: {baseline['median_segment_cv']:.3f}")

    # Analyze group durations
    print("\n[4] Analyzing temporal archetypes...")
    group_results, per_segment_stats = analyze_group_durations(segment_durations, targets)

    # Display main results
    print("\n" + "=" * 80)
    print("TEMPORAL ARCHETYPE PROFILE")
    print("=" * 80)

    print(
        f"\n{'Group':<18} {'Count':>8} {'Avg(ms)':>10} {'Std(ms)':>10} "
        f"{'CV':>8} {'Min':>8} {'Max':>8}"
    )
    print("-" * 80)

    for result in group_results:
        print(
            f"{result['Group']:<18} {result['Count']:>8} "
            f"{result['Avg_Duration_ms']:>10.1f} {result['Std_Dev_ms']:>10.1f} "
            f"{result['CV_Stereotypy']:>8.3f} {result['Min_ms']:>8.1f} {result['Max_ms']:>8.1f}"
        )

    # Compare variance across groups
    print("\n" + "=" * 80)
    print("STEREOTYPY ANALYSIS (Variance Comparison)")
    print("=" * 80)

    variance_comparison = compare_duration_variance(group_results)

    for group_name, analysis in variance_comparison.items():
        print(f"\n[{group_name}]")
        print(f"  CV: {analysis['cv']:.3f}")
        print(f"  Stereotypy: {analysis['stereotypy']}")
        print(f"  Interpretation: {analysis['interpretation']}")

    # Openers vs Closers comparison
    print("\n" + "=" * 80)
    print("HYPOTHESIS TEST: OPENERS vs CLOSERS")
    print("=" * 80)

    openers_row = next((r for r in group_results if r["Group"] == "Openers"), None)
    closers_row = next((r for r in group_results if r["Group"] == "Closers"), None)

    findings = []

    if openers_row and closers_row:
        o_dur = openers_row["Avg_Duration_ms"]
        c_dur = closers_row["Avg_Duration_ms"]
        o_cv = openers_row["CV_Stereotypy"]
        c_cv = closers_row["CV_Stereotypy"]

        print("\n[DURATION COMPARISON]")
        print(f"  Openers: {o_dur:.1f} ms (CV={o_cv:.3f})")
        print(f"  Closers: {c_dur:.1f} ms (CV={c_cv:.3f})")
        print(f"  Difference: {c_dur - o_dur:+.1f} ms")

        if o_dur < c_dur:
            pct_shorter = ((c_dur - o_dur) / c_dur) * 100
            print(f"\n  [CONFIRMED] Openers are {pct_shorter:.1f}% SHORTER than Closers")
            print("  -> Supports 'Staccato Alert' hypothesis for Openers")
            findings.append("CONFIRMED: Openers act as short alert signals")
        else:
            print("\n  [UNEXPECTED] Openers are NOT shorter than Closers")
            findings.append("UNEXPECTED: Opener/Closer duration pattern unclear")

        print("\n[VARIANCE COMPARISON]")
        if o_cv < baseline["median_segment_cv"]:
            print(f"  Openers CV ({o_cv:.3f}) < baseline ({baseline['median_segment_cv']:.3f})")
            print("  -> Openers have FIXED timing (stereotyped)")

        if c_cv < baseline["median_segment_cv"]:
            print(f"  Closers CV ({c_cv:.3f}) < baseline ({baseline['median_segment_cv']:.3f})")
            print("  -> Closers have FIXED timing (stereotyped)")

    # LRN-6 Idiom analysis
    print("\n" + "=" * 80)
    print("HYPOTHESIS TEST: RIGID IDIOM (LRN-6)")
    print("=" * 80)

    lrn_row = next((r for r in group_results if r["Group"] == "LRN-6_Idiom"), None)

    if lrn_row:
        cv = lrn_row["CV_Stereotypy"]
        mean_dur = lrn_row["Avg_Duration_ms"]

        print("\n[LRN-6 TEMPORAL PROFILE]")
        print(f"  Mean Duration: {mean_dur:.1f} ms")
        print(f"  Std Deviation: {lrn_row['Std_Dev_ms']:.1f} ms")
        print(f"  CV (Stereotypy): {cv:.3f}")
        print(f"  Range: {lrn_row['Min_ms']:.1f} - {lrn_row['Max_ms']:.1f} ms")

        print("\n[RIGID IDIOM TEST]")
        if cv < 0.5:
            print(f"  [CONFIRMED] CV={cv:.3f} < 0.5 threshold")
            print("  -> LRN-6 segments have LOW VARIANCE (Fixed timing)")
            print("  -> Strong evidence for RIGID IDIOM hypothesis")
            findings.append("CONFIRMED: LRN-6 is a rigid idiom with fixed timing")
        else:
            print(f"  [MIXED] CV={cv:.3f} >= 0.5 threshold")
            print("  -> LRN-6 segments show some temporal variation")
            findings.append("MIXED: LRN-6 shows moderate temporal flexibility")

        # Per-segment breakdown
        print("\n[PER-SEGMENT DURATION BREAKDOWN]")
        for seg_id, stats in per_segment_stats.get("LRN-6_Idiom", {}).items():
            if stats:
                print(
                    f"  {stats['Group']}: {stats['Avg_Duration_ms']:.1f} ms "
                    f"(CV={stats['CV_Stereotypy']:.3f}, n={stats['Count']})"
                )

    # Contextual modulation analysis
    print("\n" + "=" * 80)
    print("CONTEXTUAL MODULATION ANALYSIS")
    print("=" * 80)

    context_durations = extract_context_durations(cache_data)

    # Check specific segments for context influence
    context_target_segments = [764, 304, 384, 444]  # Top bigram + opener/closer examples
    context_analysis = analyze_context_influence(context_durations, context_target_segments)

    if context_analysis:
        print("\n[CONTEXT-DEPENDENT DURATION]")
        for seg_id, analysis in context_analysis.items():
            print(f"\n  Segment {seg_id}:")
            print(f"    Duration range across contexts: {analysis['duration_range_ms']:.1f} ms")
            print(f"    Variation: {analysis['pct_variation']:.1f}%")
            print(f"    Contextually modulated: {analysis['contextually_modulated']}")

            for ctx_id, stats in analysis["contexts"].items():
                print(f"      Context {ctx_id}: {stats['mean_ms']:.1f} ms (n={stats['n']})")

            if analysis["contextually_modulated"]:
                findings.append(
                    f"DISCOVERED: Segment {seg_id} shows temporal modulation by context"
                )
    else:
        print("\n  Insufficient context data for granular analysis")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)

    for i, finding in enumerate(findings, 1):
        print(f"  {i}. {finding}")

    # Save results
    output = {
        "baseline": baseline,
        "group_profiles": {r["Group"]: r for r in group_results},
        "variance_analysis": variance_comparison,
        "context_analysis": {str(k): v for k, v in context_analysis.items()},
        "findings": findings,
        "interpretation": {
            "frame_hypothesis": "CONFIRMED"
            if openers_row
            and closers_row
            and openers_row["Avg_Duration_ms"] < closers_row["Avg_Duration_ms"]
            else "PARTIAL",
            "rigid_idiom_hypothesis": "CONFIRMED"
            if lrn_row and lrn_row["CV_Stereotypy"] < 0.5
            else "MIXED",
            "temporal_modulation": len(context_analysis) > 0,
        },
    }

    output_path = Path(__file__).parent / "bat_phase4_temporal_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")
    print("Phase 4 Analysis Complete.")


if __name__ == "__main__":
    main()
