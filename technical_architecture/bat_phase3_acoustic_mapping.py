#!/usr/bin/env python3
"""
Phase 3: Acoustic Archetype Mapping
====================================

This analysis maps the linguistic roles (Openers, Closers) identified in Phase 2
to their acoustic properties to test the "Frame Hypothesis":

- OPENERS should be acoustically "sharp" (high frequency, short duration)
- CLOSERS should be acoustically "descending" (lower frequency, longer duration)

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


def extract_acoustic_features(cache_data: list[dict]) -> dict[int, dict]:
    """
    Extract acoustic features for each segment ID.

    The 105D feature vector layout (RosettaFeatures):
    - Index 0: mean_f0_hz (fundamental frequency)
    - Index 1: duration_ms
    - Index 2: f0_range_hz
    - Index 3: rms_energy
    - Index 4: zero_crossing_rate
    - Index 5: peak_amplitude
    - Index 6: harmonic_to_noise_ratio
    - Index 7: harmonicity
    - Index 8: spectral_flatness
    - Index 9: attack_time_ms
    - Index 10: decay_time_ms
    - Index 11: sustain_level
    - Index 12: release_time_ms
    - Index 13-25: MFCCs
    """
    segment_features: dict[int, list[dict]] = defaultdict(list)

    for entry in cache_data:
        seg_id = entry.get("segment_idx", entry.get("cluster_id", -1))
        if seg_id < 0:
            continue

        features = entry.get("features", [])
        if not features or len(features) < 14:
            continue

        # Extract key acoustic features from the 105D vector
        extracted = {
            "mean_f0_hz": features[0] if features[0] > 0 else 0,
            "duration_ms": features[1] if features[1] > 0 else 0,
            "f0_range_hz": features[2] if len(features) > 2 else 0,
            "rms_energy": features[3] if len(features) > 3 else 0,
            "zcr": features[4] if len(features) > 4 else 0,
            "peak_amp": features[5] if len(features) > 5 else 0,
            "hnr": features[6] if len(features) > 6 else 0,
            "attack_ms": features[9] if len(features) > 9 else 0,
            "decay_ms": features[10] if len(features) > 10 else 0,
        }

        # Also use start/end_ms as backup for duration
        start_ms = entry.get("start_ms", 0)
        end_ms = entry.get("end_ms", 0)
        if start_ms > 0 and end_ms > 0:
            extracted["duration_ms"] = end_ms - start_ms

        segment_features[seg_id].append(extracted)

    # Aggregate statistics
    segment_stats = {}
    for seg_id, features_list in segment_features.items():
        if not features_list:
            continue

        freqs = [f["mean_f0_hz"] for f in features_list if f["mean_f0_hz"] > 0]
        durations = [f["duration_ms"] for f in features_list if f["duration_ms"] > 0]
        energies = [f["rms_energy"] for f in features_list if f["rms_energy"] > 0]
        hnrs = [f["hnr"] for f in features_list if f["hnr"] != 0]
        attacks = [f["attack_ms"] for f in features_list if f["attack_ms"] != 0]

        segment_stats[seg_id] = {
            "count": len(features_list),
            "avg_freq_hz": float(np.mean(freqs)) if freqs else 0.0,
            "std_freq_hz": float(np.std(freqs)) if freqs else 0.0,
            "avg_duration_ms": float(np.mean(durations)) if durations else 0.0,
            "std_duration_ms": float(np.std(durations)) if durations else 0.0,
            "avg_energy": float(np.mean(energies)) if energies else 0.0,
            "avg_hnr": float(np.mean(hnrs)) if hnrs else 0.0,
            "avg_attack_ms": float(np.mean(attacks)) if attacks else 0.0,
        }

    return segment_stats


def analyze_segment_groups(
    segment_stats: dict[int, dict], groups: dict[str, list[int]]
) -> dict[str, dict]:
    """Analyze acoustic profiles for each segment group"""

    results = {}

    for group_name, segment_ids in groups.items():
        freqs = []
        durations = []
        energies = []
        hnrs = []
        attacks = []
        total_count = 0

        for seg_id in segment_ids:
            if seg_id in segment_stats:
                stats = segment_stats[seg_id]
                weight = stats["count"]
                total_count += weight

                if stats["avg_freq_hz"] > 0:
                    freqs.append((stats["avg_freq_hz"], weight))
                if stats["avg_duration_ms"] > 0:
                    durations.append((stats["avg_duration_ms"], weight))
                if stats["avg_energy"] > 0:
                    energies.append((stats["avg_energy"], weight))
                if stats["avg_hnr"] != 0:
                    hnrs.append((stats["avg_hnr"], weight))
                if stats["avg_attack_ms"] != 0:
                    attacks.append((stats["avg_attack_ms"], weight))

        def weighted_mean(values):
            if not values:
                return 0.0
            total_weight = sum(w for _, w in values)
            return sum(v * w for v, w in values) / total_weight if total_weight > 0 else 0.0

        results[group_name] = {
            "segment_count": len([s for s in segment_ids if s in segment_stats]),
            "total_occurrences": total_count,
            "avg_freq_khz": weighted_mean(freqs) / 1000,
            "avg_duration_ms": weighted_mean(durations),
            "avg_energy": weighted_mean(energies),
            "avg_hnr": weighted_mean(hnrs),
            "avg_attack_ms": weighted_mean(attacks),
        }

    return results


def compare_openers_closers(results: dict[str, dict]) -> dict:
    """Statistical comparison of Openers vs Closers"""

    openers = results.get("Openers", {})
    closers = results.get("Closers", {})

    if not openers or not closers:
        return {"error": "Insufficient data for comparison"}

    freq_open = openers.get("avg_freq_khz", 0)
    freq_close = closers.get("avg_freq_khz", 0)
    dur_open = openers.get("avg_duration_ms", 0)
    dur_close = closers.get("avg_duration_ms", 0)

    comparison = {
        "frequency": {
            "openers_khz": freq_open,
            "closers_khz": freq_close,
            "difference_khz": freq_open - freq_close,
            "openers_higher": freq_open > freq_close,
        },
        "duration": {
            "openers_ms": dur_open,
            "closers_ms": dur_close,
            "difference_ms": dur_open - dur_close,
            "openers_shorter": dur_open < dur_close,
        },
        "energy": {
            "openers": openers.get("avg_energy", 0),
            "closers": closers.get("avg_energy", 0),
        },
        "attack": {
            "openers_ms": openers.get("avg_attack_ms", 0),
            "closers_ms": closers.get("avg_attack_ms", 0),
        },
    }

    # Frame Hypothesis validation
    freq_cond = comparison["frequency"]["openers_higher"]
    dur_cond = comparison["duration"]["openers_shorter"]

    if freq_cond and dur_cond:
        comparison["frame_hypothesis"] = "CONFIRMED"
        comparison["interpretation"] = (
            "Openers act as high-frequency alerts; Closers as lower-frequency termination signals."
        )
    elif freq_cond or dur_cond:
        comparison["frame_hypothesis"] = "PARTIALLY CONFIRMED"
        comparison["interpretation"] = "Physical distinction exists but is less clear-cut."
    else:
        comparison["frame_hypothesis"] = "NOT CONFIRMED"
        comparison["interpretation"] = "Roles may be learned syntactic rather than acoustic."

    return comparison


def main():
    print("=" * 70)
    print("PHASE 3: ACOUSTIC ARCHETYPE MAPPING")
    print("=" * 70)

    # Define segment groups from Phase 2 findings
    groups = {
        "Openers": [384, 264, 1014, 484, 454],
        "Closers": [444, 304, 544, 404, 394],
        "LRN-6_Idiom": [114, 464, 604, 324, 94, 714],
        "Top_Frequent": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
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

    # Extract acoustic features
    print("\n[2] Extracting acoustic features per segment...")
    segment_stats = extract_acoustic_features(cache_data)
    print(f"Segments with acoustic data: {len(segment_stats)}")

    # Analyze group profiles
    print("\n[3] Computing acoustic archetype profiles...")
    group_results = analyze_segment_groups(segment_stats, groups)

    # Display results
    print("\n" + "=" * 70)
    print("ACOUSTIC ARCHETYPE PROFILES")
    print("=" * 70)

    print(f"\n{'Group':<18} {'Segs':>5} {'Freq(kHz)':>10} {'Dur(ms)':>10} {'Energy':>8} {'HNR':>8}")
    print("-" * 70)

    for group_name, stats in group_results.items():
        print(
            f"{group_name:<18} {stats['segment_count']:>5} "
            f"{stats['avg_freq_khz']:>10.2f} "
            f"{stats['avg_duration_ms']:>10.1f} "
            f"{stats['avg_energy']:>8.3f} "
            f"{stats['avg_hnr']:>8.2f}"
        )

    # Compare Openers vs Closers
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON: OPENERS vs CLOSERS")
    print("=" * 70)

    comparison = compare_openers_closers(group_results)

    print("\n[FREQUENCY]")
    print(f"  Openers: {comparison['frequency']['openers_khz']:.2f} kHz")
    print(f"  Closers: {comparison['frequency']['closers_khz']:.2f} kHz")
    print(f"  Difference: {comparison['frequency']['difference_khz']:+.2f} kHz")
    if comparison["frequency"]["openers_higher"]:
        print("  -> FINDING: Openers are HIGHER pitched (Alert Signal)")
    else:
        print("  -> FINDING: Closers are HIGHER pitched")

    print("\n[DURATION]")
    print(f"  Openers: {comparison['duration']['openers_ms']:.1f} ms")
    print(f"  Closers: {comparison['duration']['closers_ms']:.1f} ms")
    print(f"  Difference: {comparison['duration']['difference_ms']:+.1f} ms")
    if comparison["duration"]["openers_shorter"]:
        print("  -> FINDING: Openers are SHORTER (Staccato Burst)")
    else:
        print("  -> FINDING: Closers are SHORTER")

    print("\n[ENERGY]")
    print(f"  Openers: {comparison['energy']['openers']:.3f}")
    print(f"  Closers: {comparison['energy']['closers']:.3f}")

    print("\n[ATTACK TIME]")
    print(f"  Openers: {comparison['attack']['openers_ms']:.2f} ms")
    print(f"  Closers: {comparison['attack']['closers_ms']:.2f} ms")

    print("\n" + "-" * 70)
    print("FRAME HYPOTHESIS VALIDATION:")
    print(f"  Status: {comparison['frame_hypothesis']}")
    print(f"  Interpretation: {comparison['interpretation']}")

    # Detailed segment profiles
    print("\n" + "=" * 70)
    print("DETAILED SEGMENT PROFILES")
    print("=" * 70)

    print("\n[OPENERS]")
    for seg_id in groups["Openers"]:
        if seg_id in segment_stats:
            s = segment_stats[seg_id]
            print(
                f"  Segment {seg_id:>4}: "
                f"Freq={s['avg_freq_hz'] / 1000:.1f}kHz, "
                f"Dur={s['avg_duration_ms']:.1f}ms, "
                f"n={s['count']}"
            )

    print("\n[CLOSERS]")
    for seg_id in groups["Closers"]:
        if seg_id in segment_stats:
            s = segment_stats[seg_id]
            print(
                f"  Segment {seg_id:>4}: "
                f"Freq={s['avg_freq_hz'] / 1000:.1f}kHz, "
                f"Dur={s['avg_duration_ms']:.1f}ms, "
                f"n={s['count']}"
            )

    print("\n[LRN-6 IDIOM COMPONENTS]")
    for seg_id in groups["LRN-6_Idiom"]:
        if seg_id in segment_stats:
            s = segment_stats[seg_id]
            print(
                f"  Segment {seg_id:>4}: "
                f"Freq={s['avg_freq_hz'] / 1000:.1f}kHz, "
                f"Dur={s['avg_duration_ms']:.1f}ms, "
                f"n={s['count']}"
            )
        else:
            print(f"  Segment {seg_id:>4}: NO DATA (rare, LRN-6 specific)")

    # Save results
    output = {
        "group_profiles": group_results,
        "comparison": comparison,
    }

    output_path = Path(__file__).parent / "bat_phase3_acoustic_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")
    print("Phase 3 Analysis Complete.")


if __name__ == "__main__":
    main()
