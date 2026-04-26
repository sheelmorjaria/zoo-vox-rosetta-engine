#!/usr/bin/env python3
"""
Phase 5: Emitter-Idiom Mapping & Dyadic Analysis
=================================================

This analysis determines:
1. Is the "Rigid Idiom" (LRN-6) a universal signal or individual signature?
2. Do specific bats prefer specific "Openers" (Vocal Signature Analysis)?
3. Dyadic patterns: Who talks to whom and with what signals?

CLUSTER ID COMPUTATION:
Segment IDs (384, 764, etc.) are computed via feature quantization:
    hash = (f0 * 1000 + dur * 100 + hnr * 10 + mfcc1) % 1020

HYPOTHESES:
- LRN-6 as Signature: If one bat uses it exclusively, it's an identity marker
- LRN-6 as Syntax: If all bats use it, it's a functional syntactic element
- Opener Preferences: Identity markers at Position 0 in the frame structure

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_annotations(filepath: str) -> pd.DataFrame:
    """Load annotations CSV with Emitter/Addressee info"""
    df = pd.read_csv(filepath)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert Emitter/Addressee to numeric (handle negative values)
    df["Emitter"] = pd.to_numeric(df["Emitter"], errors="coerce").fillna(0).astype(int)
    df["Addressee"] = pd.to_numeric(df["Addressee"], errors="coerce").fillna(0).astype(int)
    df["Context"] = pd.to_numeric(df["Context"], errors="coerce").fillna(0).astype(int)

    # Extract file number from File Name for matching
    # "0.wav" -> 0, "11888.wav" -> 11888
    df["file_num"] = df["File Name"].str.replace(".wav", "", regex=False).astype(int)

    return df


def load_cache_data(cache_dirs: list[str]) -> tuple[list[dict], dict[str, dict]]:
    """
    Load all cache data from multiple directories.
    Returns (all_data, file_info_map) where file_info_map maps source_file to metadata.
    """
    all_data = []
    file_info = {}  # source_file -> {emitter, context, segments: [...]}

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

                        # Build file-level info
                        for entry in data:
                            src = entry.get("source_file", "")
                            if src and src not in file_info:
                                file_info[src] = {
                                    "emitter": entry.get("emitter", 0),
                                    "context": entry.get("context", 0),
                                    "segments": [],
                                }
                            if src:
                                file_info[src]["segments"].append(entry)
                    else:
                        all_data.append(data)
            except Exception:
                pass

    return all_data, file_info


def quantize_features(features: list[float], k: int = 1020) -> int:
    """
    Compute cluster ID from 112D feature vector using hashing quantization.

    This matches the Rust implementation in bat_corpus_analysis_from_cache.rs:
        f0 = features[0] * 100.0
        dur = features[1] * 10.0
        hnr = features[6] (harmonic-to-noise ratio)
        mfcc1 = features[13] (first MFCC)

        hash = (f0 * 1000 + dur * 100 + hnr * 10 + mfcc1) % k
    """
    if len(features) < 14:
        return 0

    try:
        f0 = int(features[0] * 100.0)
        dur = int(features[1] * 10.0)
        hnr = int(features[6]) if len(features) > 6 else 0
        mfcc1 = int(features[13] * 5.0) if len(features) > 13 else 0

        hash_val = abs(f0 * 1000 + dur * 100 + abs(hnr) * 10 + abs(mfcc1))
        return hash_val % k
    except (TypeError, ValueError):
        return 0


def compute_cluster_assignments(cache_data: list[dict], k: int = 1020) -> dict[str, list[int]]:
    """
    Compute cluster IDs for all segments, grouped by source file.
    Returns: {source_file: [cluster_id_0, cluster_id_1, ...]}
    """
    file_clusters: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for entry in cache_data:
        src = entry.get("source_file", "")
        seg_idx = entry.get("segment_idx", 0)
        features = entry.get("features", [])

        if src and features:
            cluster_id = quantize_features(features, k)
            file_clusters[src].append((seg_idx, cluster_id))

    # Sort by segment_idx and extract just cluster IDs
    file_sequences = {}
    for src, segments in file_clusters.items():
        sorted_segments = sorted(segments, key=lambda x: x[0])
        file_sequences[src] = [cluster_id for _, cluster_id in sorted_segments]

    return file_sequences


def detect_lrn6_in_sequence(sequence: list[int], lrn6_ids: list[int]) -> list[int]:
    """
    Detect LRN-6 sequence positions in a sequence.
    Returns list of starting indices where LRN-6 occurs.
    """
    positions = []
    lrn6_len = len(lrn6_ids)

    for i in range(len(sequence) - lrn6_len + 1):
        if sequence[i : i + lrn6_len] == lrn6_ids:
            positions.append(i)

    return positions


def analyze_lrn6_by_emitter(
    file_sequences: dict[str, list[int]],
    file_info: dict[str, dict],
    lrn6_ids: list[int],
) -> dict:
    """
    Analyze which emitters use the LRN-6 idiom.
    """
    emitter_lrn6_count: dict[int, int] = defaultdict(int)
    emitter_total_files: dict[int, int] = defaultdict(int)
    lrn6_files = []

    for src, sequence in file_sequences.items():
        if src not in file_info:
            continue

        emitter = file_info[src].get("emitter", 0)
        emitter_total_files[emitter] += 1

        # Detect LRN-6
        positions = detect_lrn6_in_sequence(sequence, lrn6_ids)

        if positions:
            emitter_lrn6_count[emitter] += len(positions)
            lrn6_files.append(
                {
                    "file": src,
                    "emitter": emitter,
                    "count": len(positions),
                    "positions": positions,
                }
            )

    return {
        "emitter_lrn6_count": dict(emitter_lrn6_count),
        "emitter_total_files": dict(emitter_total_files),
        "lrn6_files": lrn6_files,
        "unique_emitters_using_lrn6": len(emitter_lrn6_count),
    }


def analyze_opener_preferences(
    file_sequences: dict[str, list[int]],
    file_info: dict[str, dict],
    opener_ids: list[int],
    min_samples: int = 10,
) -> dict:
    """
    Analyze if specific bats prefer specific Openers (vocal signatures).
    """
    # Count opener usage by emitter
    emitter_opener_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    emitter_total_openers: dict[int, int] = defaultdict(int)

    for src, sequence in file_sequences.items():
        if src not in file_info:
            continue

        emitter = file_info[src].get("emitter", 0)
        if emitter == 0:
            continue

        # Check first position for opener
        if sequence and sequence[0] in opener_ids:
            opener_id = sequence[0]
            emitter_opener_counts[emitter][opener_id] += 1
            emitter_total_openers[emitter] += 1

    # Find dominant preferences
    signatures = []
    for emitter, opener_counts in emitter_opener_counts.items():
        total = emitter_total_openers[emitter]
        if total < min_samples:
            continue

        # Find most used opener
        top_opener = max(opener_counts.keys(), key=lambda x: opener_counts[x])
        top_count = opener_counts[top_opener]
        pct = (top_count / total) * 100

        signatures.append(
            {
                "emitter": emitter,
                "total_openers": total,
                "dominant_opener": top_opener,
                "dominant_count": top_count,
                "dominant_pct": pct,
                "all_openers": dict(opener_counts),
            }
        )

    # Sort by dominance percentage
    signatures.sort(key=lambda x: x["dominant_pct"], reverse=True)

    return {
        "signatures": signatures,
        "total_emitters_analyzed": len(signatures),
        "strong_signatures": [s for s in signatures if s["dominant_pct"] > 60],
    }


def analyze_closer_preferences(
    file_sequences: dict[str, list[int]],
    file_info: dict[str, dict],
    closer_ids: list[int],
    min_samples: int = 10,
) -> dict:
    """Analyze closer preferences by emitter"""
    emitter_closer_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    emitter_total_closers: dict[int, int] = defaultdict(int)

    for src, sequence in file_sequences.items():
        if src not in file_info:
            continue

        emitter = file_info[src].get("emitter", 0)
        if emitter == 0:
            continue

        # Check last position for closer
        if sequence and sequence[-1] in closer_ids:
            closer_id = sequence[-1]
            emitter_closer_counts[emitter][closer_id] += 1
            emitter_total_closers[emitter] += 1

    signatures = []
    for emitter, closer_counts in emitter_closer_counts.items():
        total = emitter_total_closers[emitter]
        if total < min_samples:
            continue

        top_closer = max(closer_counts.keys(), key=lambda x: closer_counts[x])
        top_count = closer_counts[top_closer]
        pct = (top_count / total) * 100

        signatures.append(
            {
                "emitter": emitter,
                "total_closers": total,
                "dominant_closer": top_closer,
                "dominant_pct": pct,
            }
        )

    return {
        "signatures": signatures,
        "strong_signatures": [s for s in signatures if s["dominant_pct"] > 60],
    }


def analyze_dyadic_patterns(
    file_sequences: dict[str, list[int]],
    file_info: dict[str, dict],
    annotations: pd.DataFrame,
    opener_ids: list[int],
    closer_ids: list[int],
) -> dict:
    """
    Analyze communication patterns: Who talks to whom with what signals?
    """
    dyad_patterns: dict[tuple, dict] = defaultdict(
        lambda: {
            "total_calls": 0,
            "total_segments": 0,
            "openers": defaultdict(int),
            "closers": defaultdict(int),
            "contexts": defaultdict(int),
        }
    )

    for src, sequence in file_sequences.items():
        if src not in file_info:
            continue

        # Extract file number from source_file
        try:
            # "11888.wav" -> 11888
            file_num = int(src.replace(".wav", ""))
        except ValueError:
            continue

        # Find matching annotation
        anno_match = annotations[annotations["file_num"] == file_num]
        if anno_match.empty:
            continue

        emitter = anno_match.iloc[0]["Emitter"]
        addressee = anno_match.iloc[0]["Addressee"]
        context = anno_match.iloc[0]["Context"]

        if emitter == 0:  # Skip unidentified emitters
            continue

        dyad = (emitter, addressee)
        dyad_patterns[dyad]["total_calls"] += 1
        dyad_patterns[dyad]["total_segments"] += len(sequence)
        dyad_patterns[dyad]["contexts"][context] += 1

        # Check opener (first position)
        if sequence and sequence[0] in opener_ids:
            dyad_patterns[dyad]["openers"][sequence[0]] += 1

        # Check closer (last position)
        if sequence and sequence[-1] in closer_ids:
            dyad_patterns[dyad]["closers"][sequence[-1]] += 1

    # Convert defaultdicts to regular dicts
    result = {}
    for dyad, data in dyad_patterns.items():
        result[f"{dyad[0]}->{dyad[1]}"] = {
            "total_calls": data["total_calls"],
            "total_segments": data["total_segments"],
            "openers": dict(data["openers"]),
            "closers": dict(data["closers"]),
            "contexts": {str(k): v for k, v in data["contexts"].items()},
        }

    return result


def analyze_emitter_context_patterns(
    file_sequences: dict[str, list[int]],
    file_info: dict[str, dict],
    lrn6_ids: list[int],
    opener_ids: list[int],
    closer_ids: list[int],
) -> dict:
    """Analyze signal usage by emitter and context"""
    # emitter -> context -> signal_type -> count
    patterns: dict[int, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for src, sequence in file_sequences.items():
        if src not in file_info:
            continue

        emitter = file_info[src].get("emitter", 0)
        context = file_info[src].get("context", 0)

        if emitter == 0:
            continue

        # Check for LRN-6
        if detect_lrn6_in_sequence(sequence, lrn6_ids):
            patterns[emitter][context]["lrn6"] += 1

        # Check opener (first position)
        if sequence and sequence[0] in opener_ids:
            patterns[emitter][context]["openers"] += 1
        elif sequence and sequence[-1] in closer_ids:
            patterns[emitter][context]["closers"] += 1

        patterns[emitter][context]["total_files"] += 1

    # Convert to regular dicts
    return {e: {c: dict(s) for c, s in contexts.items()} for e, contexts in patterns.items()}


def analyze_emitter_vocabulary(
    file_sequences: dict[str, list[int]], file_info: dict[str, dict]
) -> dict:
    """
    Analyze vocabulary usage by emitter.
    """
    emitter_vocab: dict[int, Counter] = defaultdict(Counter)
    emitter_stats: dict[int, dict] = defaultdict(
        lambda: {
            "total_files": 0,
            "total_segments": 0,
            "contexts": Counter(),
        }
    )

    for src, sequence in file_sequences.items():
        if src not in file_info:
            continue

        emitter = file_info[src].get("emitter", 0)
        context = file_info[src].get("context", 0)

        if emitter == 0:
            continue

        emitter_stats[emitter]["total_files"] += 1
        emitter_stats[emitter]["total_segments"] += len(sequence)
        emitter_stats[emitter]["contexts"][context] += 1

        for cluster_id in sequence:
            emitter_vocab[emitter][cluster_id] += 1

    # Compute vocabulary diversity per emitter
    results = []
    for emitter, vocab in emitter_vocab.items():
        stats = emitter_stats[emitter]
        if stats["total_files"] < 10:
            continue

        diversity = len(vocab)
        total_segments = stats["total_segments"]

        # Find top segments
        top_segments = vocab.most_common(5)

        results.append(
            {
                "emitter": emitter,
                "total_files": stats["total_files"],
                "total_segments": total_segments,
                "vocabulary_size": diversity,
                "top_segments": top_segments,
                "top_contexts": dict(stats["contexts"].most_common(3)),
            }
        )

    results.sort(key=lambda x: x["total_segments"], reverse=True)

    return {
        "emitter_profiles": results[:20],
        "total_unique_emitters": len(emitter_vocab),
    }


def main():
    print("=" * 80)
    print("PHASE 5: EMITTER-IDIOM MAPPING & DYADIC ANALYSIS")
    print("=" * 80)

    # Define analysis targets from previous phases
    lrn6_ids = [114, 464, 604, 324, 94, 714]
    opener_ids = [384, 264, 1014, 484, 454]
    closer_ids = [444, 304, 544, 404, 394]

    # Load annotations
    annotations_path = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/annotations.csv")
    print(f"\n[1] Loading annotations from: {annotations_path}")

    try:
        annotations = load_annotations(str(annotations_path))
        print(f"Loaded {len(annotations)} annotation records")
        print(f"Unique emitters: {annotations['Emitter'].nunique()}")
        print(f"Unique contexts: {annotations['Context'].nunique()}")
    except FileNotFoundError:
        print("ERROR: Annotations file not found")
        return

    # Load cache data
    cache_dirs = [
        "bat_nbd_cache_parallel",
        "bat_fm_cache",
        "bat_nbd_cache_full",
    ]

    print("\n[2] Loading cache data...")
    cache_data, file_info = load_cache_data(cache_dirs)
    print(f"Total entries loaded: {len(cache_data):,}")
    print(f"Unique files: {len(file_info):,}")

    if not cache_data:
        print("ERROR: No cache data found")
        return

    # Compute cluster assignments
    print("\n[3] Computing cluster assignments from features...")
    file_sequences = compute_cluster_assignments(cache_data, k=1020)
    print(f"Files with sequences: {len(file_sequences):,}")

    # Sample some sequences to verify
    sample_keys = list(file_sequences.keys())[:3]
    for key in sample_keys:
        seq = file_sequences[key][:6]
        print(f"  {key}: {seq}...")

    # Analyze LRN-6 by emitter
    print("\n" + "=" * 80)
    print("LRN-6 IDIOM: EMITTER ANALYSIS")
    print("=" * 80)

    lrn6_analysis = analyze_lrn6_by_emitter(file_sequences, file_info, lrn6_ids)

    print(f"\nUnique emitters using LRN-6: {lrn6_analysis['unique_emitters_using_lrn6']}")
    print(f"Total files with LRN-6: {len(lrn6_analysis['lrn6_files'])}")

    if lrn6_analysis["emitter_lrn6_count"]:
        print("\n[LRN-6 USAGE BY EMITTER]")
        emitter_counts = lrn6_analysis["emitter_lrn6_count"]
        total_files = lrn6_analysis["emitter_total_files"]

        for emitter in sorted(emitter_counts.keys(), key=lambda x: emitter_counts[x], reverse=True)[
            :10
        ]:
            count = emitter_counts[emitter]
            total = total_files.get(emitter, 1)
            pct = (count / total) * 100 if total > 0 else 0
            print(
                f"  Emitter {emitter:>5}: {count:>4} LRN-6 occurrences "
                f"in {total:>4} files ({pct:.1f}%)"
            )

        # LRN-6 hypothesis test
        print("\n[LRN-6 HYPOTHESIS TEST]")
        if lrn6_analysis["unique_emitters_using_lrn6"] == 1:
            print("  -> SIGNATURE CALL: LRN-6 is used by only ONE bat")
            print("     This suggests it's an identity marker (like a name)")
            lrn6_hypothesis = "SIGNATURE"
        elif lrn6_analysis["unique_emitters_using_lrn6"] <= 3:
            print("  -> SPECIALIZED SIGNAL: LRN-6 used by few bats")
            print("     May indicate social role or subgroup marker")
            lrn6_hypothesis = "SPECIALIZED"
        else:
            print("  -> SHARED SYNTAX: LRN-6 used by many bats")
            print("     This suggests it's a functional syntactic element")
            lrn6_hypothesis = "SHARED_SYNTAX"
    else:
        print("\n  No LRN-6 occurrences detected in the dataset")
        print("  This could indicate the quantization parameters differ from corpus analysis")
        lrn6_hypothesis = "NOT_DETECTED"

    # Analyze opener preferences
    print("\n" + "=" * 80)
    print("OPENER PREFERENCES (VOCAL SIGNATURES)")
    print("=" * 80)

    opener_analysis = analyze_opener_preferences(file_sequences, file_info, opener_ids)

    print(f"\nEmitters analyzed: {opener_analysis['total_emitters_analyzed']}")
    print(f"Strong signatures (>60% preference): {len(opener_analysis['strong_signatures'])}")

    if opener_analysis["signatures"]:
        print("\n[TOP OPENER PREFERENCES]")
        for sig in opener_analysis["signatures"][:10]:
            print(
                f"  Emitter {sig['emitter']:>5}: "
                f"Prefers Opener {sig['dominant_opener']} "
                f"({sig['dominant_pct']:.1f}% of {sig['total_openers']} uses)"
            )
    else:
        print("  No opener preferences detected")

    # Analyze closer preferences
    print("\n" + "=" * 80)
    print("CLOSER PREFERENCES")
    print("=" * 80)

    closer_analysis = analyze_closer_preferences(file_sequences, file_info, closer_ids)

    print(f"Strong closer signatures: {len(closer_analysis['strong_signatures'])}")

    if closer_analysis["signatures"]:
        print("\n[TOP CLOSER PREFERENCES]")
        for sig in closer_analysis["signatures"][:5]:
            print(
                f"  Emitter {sig['emitter']:>5}: "
                f"Prefers Closer {sig['dominant_closer']} "
                f"({sig['dominant_pct']:.1f}%)"
            )

    # Dyadic patterns
    print("\n" + "=" * 80)
    print("DYADIC PATTERNS (Who talks to whom)")
    print("=" * 80)

    dyadic_analysis = analyze_dyadic_patterns(
        file_sequences, file_info, annotations, opener_ids, closer_ids
    )

    # Find most active dyads
    sorted_dyads = sorted(dyadic_analysis.items(), key=lambda x: x[1]["total_calls"], reverse=True)

    print("\n[TOP DYADS BY CALL VOLUME]")
    for dyad, data in sorted_dyads[:10]:
        print(f"  {dyad}: {data['total_calls']} calls, {data['total_segments']} segments")

    # Emitter vocabulary analysis
    print("\n" + "=" * 80)
    print("EMITTER VOCABULARY ANALYSIS")
    print("=" * 80)

    vocab_analysis = analyze_emitter_vocabulary(file_sequences, file_info)

    print(f"\nTotal unique emitters: {vocab_analysis['total_unique_emitters']}")

    if vocab_analysis["emitter_profiles"]:
        print("\n[TOP EMITTERS BY VOCABULARY SIZE]")
        for profile in vocab_analysis["emitter_profiles"][:10]:
            print(
                f"  Emitter {profile['emitter']:>5}: "
                f"{profile['vocabulary_size']} unique segments, "
                f"{profile['total_files']} files, "
                f"top contexts: {profile['top_contexts']}"
            )

    # Summary findings
    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)

    findings = []

    # LRN-6 finding
    findings.append(f"LRN-6 Hypothesis: {lrn6_hypothesis}")
    if lrn6_analysis["unique_emitters_using_lrn6"] > 0:
        findings.append(
            f"  - {lrn6_analysis['unique_emitters_using_lrn6']} unique emitters use LRN-6"
        )

    # Opener signature finding
    strong_sigs = len(opener_analysis["strong_signatures"])
    if strong_sigs > 0:
        findings.append(
            f"VOCAL SIGNATURES DETECTED: {strong_sigs} bats have >60% opener preference"
        )
    else:
        findings.append("No strong vocal signatures detected (openers shared across bats)")

    # Vocabulary finding
    if vocab_analysis["emitter_profiles"]:
        avg_vocab = np.mean([p["vocabulary_size"] for p in vocab_analysis["emitter_profiles"]])
        findings.append(f"Average vocabulary size per emitter: {avg_vocab:.0f} unique segments")

    for i, finding in enumerate(findings, 1):
        print(f"  {i}. {finding}")

    # Save results
    output = {
        "lrn6_analysis": {
            "unique_emitters": lrn6_analysis["unique_emitters_using_lrn6"],
            "hypothesis": lrn6_hypothesis,
            "emitter_counts": lrn6_analysis["emitter_lrn6_count"],
        },
        "opener_analysis": {
            "total_emitters": opener_analysis["total_emitters_analyzed"],
            "strong_signatures": len(opener_analysis["strong_signatures"]),
            "top_signatures": opener_analysis["signatures"][:10],
        },
        "closer_analysis": {
            "strong_signatures": len(closer_analysis["strong_signatures"]),
        },
        "dyadic_patterns": {k: v for k, v in list(dyadic_analysis.items())[:20]},
        "vocabulary_analysis": vocab_analysis,
        "findings": findings,
    }

    output_path = Path(__file__).parent / "bat_phase5_emitter_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")
    print("Phase 5 Analysis Complete.")


if __name__ == "__main__":
    main()
