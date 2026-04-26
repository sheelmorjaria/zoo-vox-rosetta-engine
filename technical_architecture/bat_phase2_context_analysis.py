#!/usr/bin/env python3
"""
Phase 2 Context-Aware Linguistic Analysis for Egyptian Fruit Bat Vocalizations
==============================================================================

Enhanced analysis incorporating context information from NBD cache files.

Key analyses:
1. Context-Specific N-gram Analysis (Territorial vs Social)
2. Modulator Segment Detection (segments that flip context)
3. Mutual Information between segments and contexts
4. LRN-6 Detailed Decomposition with context mapping

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_cache_data(cache_dir: str) -> list[dict]:
    """Load all NBD cache files from directory"""
    cache_path = Path(cache_dir)
    all_data = []

    for cache_file in sorted(cache_path.glob("*.json")):
        try:
            with open(cache_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        except Exception as e:
            print(f"Warning: Could not load {cache_file}: {e}")

    return all_data


def analyze_context_distribution(cache_data: list[dict]) -> dict[str, Any]:
    """
    Analyze segment distribution across contexts.

    Key questions:
    - Which segments appear primarily in territorial (Context 11)?
    - Which appear in social (Context 12)?
    - Which are "context-neutral"?
    """
    print("=" * 70)
    print("CONTEXT-SPECIFIC SEGMENT ANALYSIS")
    print("=" * 70)

    # Count segments by context
    segment_context_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    context_totals: dict[str, int] = defaultdict(int)

    for entry in cache_data:
        context = str(entry.get("context", "unknown"))
        segment_id = entry.get("segment_idx", entry.get("cluster_id", -1))

        segment_context_counts[segment_id][context] += 1
        context_totals[context] += 1

    print("\nContext distribution:")
    for ctx, total in sorted(context_totals.items()):
        print(f"  Context {ctx}: {total:,} segments")

    # Calculate context specificity for each segment
    results = {
        "territorial_markers": [],  # High specificity to Context 11
        "social_markers": [],  # High specificity to Context 12
        "neutral_segments": [],  # Evenly distributed
    }

    total_all = sum(context_totals.values())

    for seg_id, ctx_counts in segment_context_counts.items():
        if seg_id < 0:
            continue

        total_seg = sum(ctx_counts.values())
        if total_seg < 10:  # Skip rare segments
            continue

        # Calculate context ratios
        ctx_11_count = ctx_counts.get("11", ctx_counts.get("11.0", 0))
        ctx_12_count = ctx_counts.get("12", ctx_counts.get("12.0", 0))

        ratio_11 = ctx_11_count / total_seg if total_seg > 0 else 0
        ratio_12 = ctx_12_count / total_seg if total_seg > 0 else 0

        # Overall context prevalence
        overall_11 = (
            context_totals.get("11", context_totals.get("11.0", 0)) / total_all
            if total_all > 0
            else 0.5
        )

        # Specificity score: how much more common in this context vs baseline
        specificity_11 = ratio_11 - overall_11 if overall_11 > 0 else 0

        entry = (seg_id, specificity_11, ratio_11, ratio_12, total_seg)

        if specificity_11 > 0.2:
            results["territorial_markers"].append(entry)
        elif specificity_11 < -0.2:
            results["social_markers"].append(entry)
        else:
            results["neutral_segments"].append(entry)

    # Sort by specificity
    results["territorial_markers"].sort(key=lambda x: x[1], reverse=True)
    results["social_markers"].sort(key=lambda x: x[1])
    results["neutral_segments"].sort(key=lambda x: x[4], reverse=True)

    # Print results
    print("\n" + "-" * 70)
    print("TERRITORIAL MARKERS (Context 11 Enriched):")
    for seg, spec, r11, r12, total in results["territorial_markers"][:10]:
        print(
            f"  Segment {seg}: {r11:.1%} Context 11, {r12:.1%} Context 12 "
            f"(specificity: +{spec:.2f}, n={total})"
        )

    print("\n" + "-" * 70)
    print("SOCIAL MARKERS (Context 12 Enriched):")
    for seg, spec, r11, r12, total in results["social_markers"][:10]:
        print(
            f"  Segment {seg}: {r11:.1%} Context 11, {r12:.1%} Context 12 "
            f"(specificity: {spec:.2f}, n={total})"
        )

    print("\n" + "-" * 70)
    print("NEUTRAL SEGMENTS (Context-Independent):")
    for seg, spec, r11, r12, total in results["neutral_segments"][:10]:
        print(f"  Segment {seg}: {r11:.1%} Context 11, {r12:.1%} Context 12 (n={total})")

    return results


def analyze_lrn6_with_context(cache_data: list[dict]) -> dict[str, Any]:
    """
    Analyze the LRN-6 pattern [114, 464, 604, 324, 94, 714] with context.
    """
    print("\n" + "=" * 70)
    print("LRN-6 CONTEXT DECOMPOSITION")
    print("=" * 70)

    lrn6 = [114, 464, 604, 324, 94, 714]

    # Map segment IDs to cache entries
    segment_entries: dict[int, list[dict]] = defaultdict(list)

    for entry in cache_data:
        seg_id = entry.get("segment_idx", entry.get("cluster_id", -1))
        if seg_id in lrn6:
            segment_entries[seg_id].append(entry)

    print(f"\nLRN-6: {lrn6}")
    print("\nSegment Context Analysis:")

    results = {"lrn6": lrn6, "segments": {}}

    for seg_id in lrn6:
        entries = segment_entries.get(seg_id, [])
        if not entries:
            print(f"  Segment {seg_id}: No cache data found")
            continue

        contexts = Counter(str(e.get("context", "unknown")) for e in entries)
        total = len(entries)

        ctx_11 = contexts.get("11", contexts.get("11.0", 0))
        ctx_12 = contexts.get("12", contexts.get("12.0", 0))

        print(
            f"  Segment {seg_id}: {total} occurrences, "
            f"Context 11: {ctx_11} ({ctx_11 / total * 100:.1f}%), "
            f"Context 12: {ctx_12} ({ctx_12 / total * 100:.1f}%)"
        )

        results["segments"][seg_id] = {
            "total": total,
            "context_11": ctx_11,
            "context_12": ctx_12,
            "context_distribution": dict(contexts),
        }

    # Analyze sub-patterns
    print("\n" + "-" * 70)
    print("SUB-PATTERN HYPOTHESIS:")

    # Check if certain positions are context-determining
    context_changers = []
    for seg_id, data in results["segments"].items():
        if data["total"] > 0:
            ratio_11 = data["context_11"] / data["total"]
            if ratio_11 > 0.6 or ratio_11 < 0.4:
                context_changers.append(seg_id)

    if context_changers:
        print(f"  Context-modulating positions: {context_changers}")
        print("  HYPOTHESIS: These segments carry context-specific meaning")
    else:
        print("  All segments appear context-neutral within LRN-6")
        print("  HYPOTHESIS: Context determined by external factors (emitter, timing)")

    return results


def analyze_segment_transitions_by_context(cache_data: list[dict]) -> dict[str, Any]:
    """
    Analyze how segment transitions differ by context.

    This reveals if the same segment has different "meanings" in different contexts.
    """
    print("\n" + "=" * 70)
    print("CONTEXT-DEPENDENT TRANSITION ANALYSIS")
    print("=" * 70)

    # Group entries by source file and context
    by_source: dict[str, list[dict]] = defaultdict(list)

    for entry in cache_data:
        source = entry.get("source_file", "unknown")
        by_source[source].append(entry)

    # Build transition matrices by context
    transitions_by_context: dict[str, dict[tuple[int, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for source, entries in by_source.items():
        # Sort by time
        sorted_entries = sorted(entries, key=lambda x: x.get("start_ms", 0))

        # Extract context (assume consistent within vocalization)
        if sorted_entries:
            context = str(sorted_entries[0].get("context", "unknown"))
        else:
            continue

        # Build transitions
        for i in range(len(sorted_entries) - 1):
            seg1 = sorted_entries[i].get("segment_idx", sorted_entries[i].get("cluster_id", -1))
            seg2 = sorted_entries[i + 1].get(
                "segment_idx", sorted_entries[i + 1].get("cluster_id", -1)
            )

            if seg1 >= 0 and seg2 >= 0:
                transitions_by_context[context][(seg1, seg2)] += 1

    # Find context-specific transitions
    print("\nTop transitions by context:")

    results = {}

    for context in sorted(transitions_by_context.keys()):
        trans = transitions_by_context[context]
        top = sorted(trans.items(), key=lambda x: x[1], reverse=True)[:10]

        print(f"\n  Context {context}:")
        for (seg1, seg2), count in top[:5]:
            print(f"    {seg1} -> {seg2}: {count}")

        results[context] = {
            "top_transitions": [(list(t), c) for t, c in top],
            "unique_transitions": len(trans),
            "total_transitions": sum(trans.values()),
        }

    # Find divergent transitions (same start, different end by context)
    print("\n" + "-" * 70)
    print("DIVERGENT TRANSITIONS (Context-Dependent Meaning):")

    divergent = []

    # Get all start segments
    all_starts = set()
    for trans in transitions_by_context.values():
        for s1, s2 in trans.keys():
            all_starts.add(s1)

    for start_seg in sorted(all_starts):
        next_by_context = {}

        for context, trans in transitions_by_context.items():
            next_segs = {s2: c for (s1, s2), c in trans.items() if s1 == start_seg}
            if next_segs:
                next_by_context[context] = sorted(
                    next_segs.items(), key=lambda x: x[1], reverse=True
                )[:3]

        if len(next_by_context) > 1:
            # Check if top transitions differ
            contexts = list(next_by_context.keys())
            if len(contexts) >= 2:
                top_11 = (
                    next_by_context[contexts[0]][0][0] if next_by_context[contexts[0]] else None
                )
                top_12 = (
                    next_by_context[contexts[1]][0][0] if next_by_context[contexts[1]] else None
                )

                if top_11 != top_12:
                    divergent.append(
                        {
                            "segment": start_seg,
                            "context_11_top": next_by_context.get(contexts[0], []),
                            "context_12_top": next_by_context.get(contexts[1], []),
                        }
                    )

    for div in divergent[:5]:
        print(f"\n  Segment {div['segment']}:")
        print(f"    Context 11: {div['context_11_top'][:2]}")
        print(f"    Context 12: {div['context_12_top'][:2]}")

    results["divergent_transitions"] = divergent

    return results


def calculate_mutual_information(cache_data: list[dict]) -> dict[str, Any]:
    """
    Calculate Mutual Information between segments and contexts.

    High MI = segment is strongly associated with specific context
    Low MI = segment is context-independent
    """
    print("\n" + "=" * 70)
    print("MUTUAL INFORMATION ANALYSIS")
    print("=" * 70)

    # Count joint distribution
    joint_counts: dict[tuple[int, str], int] = defaultdict(int)
    segment_totals: dict[int, int] = defaultdict(int)
    context_totals: dict[str, int] = defaultdict(int)
    total = 0

    for entry in cache_data:
        seg_id = entry.get("segment_idx", entry.get("cluster_id", -1))
        context = str(entry.get("context", "unknown"))

        if seg_id < 0:
            continue

        joint_counts[(seg_id, context)] += 1
        segment_totals[seg_id] += 1
        context_totals[context] += 1
        total += 1

    if total == 0:
        print("No data available for MI calculation")
        return {}

    # Calculate MI for each segment
    mi_scores = {}

    for seg_id in segment_totals:
        p_seg = segment_totals[seg_id] / total

        mi = 0.0
        for context in context_totals:
            p_ctx = context_totals[context] / total
            p_joint = joint_counts.get((seg_id, context), 0) / total

            if p_joint > 0:
                mi += p_joint * math.log2(p_joint / (p_seg * p_ctx))

        mi_scores[seg_id] = mi

    # Sort by MI
    sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)

    print("\nHigh MI segments (Context-Specific):")
    for seg, mi in sorted_mi[:10]:
        ctx_dist = {ctx: joint_counts.get((seg, ctx), 0) for ctx in context_totals}
        print(f"  Segment {seg}: MI={mi:.4f}, Distribution={ctx_dist}")

    print("\nLow MI segments (Context-Independent):")
    for seg, mi in sorted_mi[-10:]:
        ctx_dist = {ctx: joint_counts.get((seg, ctx), 0) for ctx in context_totals}
        print(f"  Segment {seg}: MI={mi:.4f}, Distribution={ctx_dist}")

    return {
        "mi_scores": dict(sorted_mi),
        "high_mi_segments": [s for s, mi in sorted_mi[:20]],
        "low_mi_segments": [s for s, mi in sorted_mi[-20:]],
    }


def main():
    """Run complete context-aware analysis"""

    # Find cache directory
    cache_dirs = [
        Path(__file__).parent / "bat_nbd_cache_parallel",
        Path(__file__).parent / "bat_fm_cache",
        Path(__file__).parent / "bat_nbd_cache_full",
    ]

    cache_data = []
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            print(f"Loading cache from: {cache_dir}")
            cache_data.extend(load_cache_data(str(cache_dir)))

    if not cache_data:
        print("ERROR: No cache data found")
        print("Please ensure bat_nbd_cache_parallel or bat_fm_cache exists")
        return

    print(f"\nLoaded {len(cache_data):,} total cache entries")

    # Run analyses
    context_dist = analyze_context_distribution(cache_data)
    lrn6_analysis = analyze_lrn6_with_context(cache_data)
    transition_analysis = analyze_segment_transitions_by_context(cache_data)
    _mi_analysis = calculate_mutual_information(cache_data)  # noqa: F841

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_territorial = len(context_dist.get("territorial_markers", []))
    n_social = len(context_dist.get("social_markers", []))
    n_neutral = len(context_dist.get("neutral_segments", []))

    print("\nSegment Classification:")
    print(f"  Territorial markers: {n_territorial}")
    print(f"  Social markers: {n_social}")
    print(f"  Context-neutral: {n_neutral}")

    n_divergent = len(transition_analysis.get("divergent_transitions", []))
    print(f"\nContext-Dependent Transitions: {n_divergent}")

    # Save results
    results = {
        "context_distribution": {
            "n_territorial": n_territorial,
            "n_social": n_social,
            "n_neutral": n_neutral,
        },
        "lrn6_analysis": lrn6_analysis,
        "divergent_transitions_count": n_divergent,
    }

    output_path = Path(__file__).parent / "bat_phase2_context_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
