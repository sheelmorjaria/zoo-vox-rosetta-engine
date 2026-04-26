#!/usr/bin/env python3
"""
Phase 2 Linguistic Analysis for Egyptian Fruit Bat Vocalizations
================================================================

This script performs structural linguistics analysis on the bat corpus
to identify segment roles, contextual meaning, and syntactic structure.

Analysis modules:
1. Segment Role Analysis (Positional Entropy, Transition States)
2. Contextual Discrimination (Mutual Information, Modulator Detection)
3. LRN-6 Decomposition (Syntax Verification)
4. N-gram Combinatorics (Grammar Extraction)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SegmentProfile:
    """Profile for a single segment's linguistic properties"""

    segment_id: int
    total_occurrences: int
    position_distribution: dict[int, int]  # position -> count
    transition_out: dict[int, int]  # next_segment -> count
    transition_in: dict[int, int]  # prev_segment -> count
    positional_entropy: float
    is_opener: bool
    is_closer: bool
    is_function_word: bool


@dataclass
class NgramProfile:
    """Profile for an n-gram's contextual properties"""

    ngram: tuple[int, ...]
    count: int
    context_specificity: float  # -1 to +1
    is_marker: bool
    is_filler: bool


class BatLinguisticAnalyzer:
    """
    Main analyzer for bat vocalization linguistics.

    This class implements the Phase 2 linguistic analysis pipeline,
    focusing on identifying syntactic structure in bat communication.
    """

    def __init__(self, corpus_report_path: str):
        """Initialize with path to corpus analysis report JSON"""
        with open(corpus_report_path) as f:
            self.data = json.load(f)

        self.total_segments = self.data["total_segments"]
        self.unique_segments = self.data["unique_segment_types"]
        self.total_vocalizations = self.data["total_vocalizations"]

        # Extract all n-grams
        self.bigrams = {tuple(ng[0]): ng[1] for ng in self.data["top_bigrams"]}
        self.trigrams = {tuple(ng[0]): ng[1] for ng in self.data["top_trigrams"]}
        self.fourgrams = {tuple(ng[0]): ng[1] for ng in self.data["top_4grams"]}
        self.fivegrams = {tuple(ng[0]): ng[1] for ng in self.data["top_5grams"]}

        # Longest repeated n-gram (LRN-6)
        self.lrn6 = tuple(self.data["longest_repeated_ngram"][0])
        self.lrn6_count = self.data["longest_repeated_ngram"][1]

        # Segment profiles
        self.segment_profiles: dict[int, SegmentProfile] = {}

        print(f"Loaded corpus: {self.total_vocalizations:,} vocalizations")
        print(f"Total segments: {self.total_segments:,}")
        print(f"Unique segment types: {self.unique_segments}")
        print(f"Longest repeated n-gram (LRN-6): {self.lrn6}")

    def analyze_segment_roles(self) -> dict[str, Any]:
        """
        Phase 1: Segment Role Analysis

        Calculate positional entropy for each segment to determine
        if segments act as "operators" (verbs) or "operands" (nouns).

        Returns:
            Dictionary with segment role analysis results
        """
        print("\n" + "=" * 60)
        print("PHASE 1: SEGMENT ROLE ANALYSIS")
        print("=" * 60)

        # Collect positional data from bigrams
        position_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        transition_out: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        transition_in: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        total_occurrences: dict[int, int] = defaultdict(int)

        # Analyze bigrams for position and transitions
        for (seg1, seg2), count in self.bigrams.items():
            # Position 0 = opener, Position 1 = follower
            position_counts[seg1][0] += count
            position_counts[seg2][1] += count

            # Transitions
            transition_out[seg1][seg2] += count
            transition_in[seg2][seg1] += count

            total_occurrences[seg1] += count
            total_occurrences[seg2] += count

        # Calculate positional entropy for each segment
        results = {
            "openers": [],  # High position 0 proportion
            "closers": [],  # High position 1 proportion
            "function_words": [],  # High transition diversity
            "content_words": [],  # Low transition diversity
        }

        for seg_id in sorted(total_occurrences.keys()):
            total = total_occurrences[seg_id]
            pos_dist = position_counts[seg_id]

            # Calculate positional entropy
            entropy = 0.0
            for pos, count in pos_dist.items():
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)

            # Calculate transition diversity
            out_diversity = len(transition_out[seg_id])
            _in_diversity = len(transition_in[seg_id])  # noqa: F841

            # Classify segment role
            pos0_ratio = pos_dist.get(0, 0) / total if total > 0 else 0
            pos1_ratio = pos_dist.get(1, 0) / total if total > 0 else 0

            profile = SegmentProfile(
                segment_id=seg_id,
                total_occurrences=total,
                position_distribution=dict(pos_dist),
                transition_out=dict(transition_out[seg_id]),
                transition_in=dict(transition_in[seg_id]),
                positional_entropy=entropy,
                is_opener=pos0_ratio > 0.7,
                is_closer=pos1_ratio > 0.7,
                is_function_word=out_diversity > 5,
            )
            self.segment_profiles[seg_id] = profile

            # Categorize
            if profile.is_opener:
                results["openers"].append((seg_id, pos0_ratio, total))
            elif profile.is_closer:
                results["closers"].append((seg_id, pos1_ratio, total))

            if out_diversity > 5:
                results["function_words"].append((seg_id, out_diversity, total))
            else:
                results["content_words"].append((seg_id, out_diversity, total))

        # Sort by significance
        results["openers"].sort(key=lambda x: x[2], reverse=True)
        results["closers"].sort(key=lambda x: x[2], reverse=True)
        results["function_words"].sort(key=lambda x: x[1], reverse=True)
        results["content_words"].sort(key=lambda x: x[1])

        # Print summary
        print("\nSegment Role Classification:")
        print(f"  Openers (70%+ at position 0): {len(results['openers'])}")
        for seg, ratio, count in results["openers"][:5]:
            print(f"    Segment {seg}: {ratio:.1%} opener, {count} occurrences")

        print(f"\n  Closers (70%+ at position 1): {len(results['closers'])}")
        for seg, ratio, count in results["closers"][:5]:
            print(f"    Segment {seg}: {ratio:.1%} closer, {count} occurrences")

        print(f"\n  Function Words (5+ transitions): {len(results['function_words'])}")
        for seg, div, count in results["function_words"][:5]:
            print(f"    Segment {seg}: {div} transition types, {count} occurrences")

        print(f"\n  Content Words (<5 transitions): {len(results['content_words'])}")
        for seg, div, count in results["content_words"][:5]:
            print(f"    Segment {seg}: {div} transition types, {count} occurrences")

        return results

    def analyze_transitions(self, top_n: int = 10) -> dict[str, Any]:
        """
        Analyze transition patterns to identify function word behavior.

        A function word is one that changes meaning based on what follows it.
        Example: If 764->304 means "Territorial A" and 764->394 means "Territorial B",
        then 764 is a function word (operator) and 304/394 are arguments.
        """
        print("\n" + "=" * 60)
        print("TRANSITION ANALYSIS (Function Word Detection)")
        print("=" * 60)

        # Build transition matrix
        transitions: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for (seg1, seg2), count in self.bigrams.items():
            transitions[seg1][seg2] += count

        # Find segments with diverse transitions (potential function words)
        results = {}

        for seg_id, next_segs in sorted(
            transitions.items(), key=lambda x: sum(x[1].values()), reverse=True
        )[:top_n]:
            total = sum(next_segs.values())
            unique_next = len(next_segs)

            # Calculate entropy of transition distribution
            entropy = 0.0
            for count in next_segs.values():
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)

            # Top transitions
            top_transitions = sorted(next_segs.items(), key=lambda x: x[1], reverse=True)[:3]

            results[seg_id] = {
                "total_occurrences": total,
                "unique_transitions": unique_next,
                "entropy": entropy,
                "top_transitions": top_transitions,
            }

            print(f"\nSegment {seg_id}:")
            print(f"  Total: {total}, Unique next: {unique_next}, Entropy: {entropy:.3f}")
            for next_seg, count in top_transitions:
                pct = count / total * 100
                print(f"    -> {next_seg}: {count} ({pct:.1f}%)")

            # Interpretation
            if unique_next > 3 and entropy > 1.0:
                print("  [FUNCTION WORD] High diversity suggests operator role")
            elif unique_next <= 2 and entropy < 0.5:
                print("  [CONTENT WORD] Low diversity suggests fixed argument")

        return results

    def decompose_lrn6(self) -> dict[str, Any]:
        """
        Phase 3: Decomposition of LRN-6

        Analyze the longest repeated 6-gram to determine if it's:
        - A rigid idiom (must appear as complete unit)
        - Compositional (sub-parts have independent meaning)

        The LRN-6 is: [114, 464, 604, 324, 94, 714]
        """
        print("\n" + "=" * 60)
        print("PHASE 3: LRN-6 DECOMPOSITION")
        print("=" * 60)
        print(f"LRN-6: {list(self.lrn6)}")
        print(f"Occurrences: {self.lrn6_count}")

        results = {
            "lrn6": list(self.lrn6),
            "count": self.lrn6_count,
            "sub_patterns": {},
            "branching_analysis": {},
        }

        # Check all sub-patterns
        lrn6 = self.lrn6

        # 2-gram sub-patterns
        print("\n2-gram sub-patterns:")
        for i in range(5):
            pattern = (lrn6[i], lrn6[i + 1])
            count = self.bigrams.get(pattern, 0)
            results["sub_patterns"][f"2g_{i}"] = {
                "pattern": list(pattern),
                "count": count,
            }
            status = "INDEPENDENT" if count > 1 else "LRN6-ONLY"
            print(f"  Position {i}: {list(pattern)} -> {count} [{status}]")

        # 3-gram sub-patterns
        print("\n3-gram sub-patterns:")
        for i in range(4):
            pattern = (lrn6[i], lrn6[i + 1], lrn6[i + 2])
            count = self.trigrams.get(pattern, 0)
            results["sub_patterns"][f"3g_{i}"] = {
                "pattern": list(pattern),
                "count": count,
            }
            status = "INDEPENDENT" if count > 0 else "LRN6-ONLY"
            print(f"  Position {i}: {list(pattern)} -> {count} [{status}]")

        # 4-gram sub-patterns
        print("\n4-gram sub-patterns:")
        for i in range(3):
            pattern = (lrn6[i], lrn6[i + 1], lrn6[i + 2], lrn6[i + 3])
            count = self.fourgrams.get(pattern, 0)
            results["sub_patterns"][f"4g_{i}"] = {
                "pattern": list(pattern),
                "count": count,
            }
            status = "INDEPENDENT" if count > 0 else "LRN6-ONLY"
            print(f"  Position {i}: {list(pattern)} -> {count} [{status}]")

        # 5-gram sub-patterns
        print("\n5-gram sub-patterns:")
        for i in range(2):
            pattern = tuple(lrn6[i : i + 5])
            count = self.fivegrams.get(pattern, 0)
            results["sub_patterns"][f"5g_{i}"] = {
                "pattern": list(pattern),
                "count": count,
            }
            status = "INDEPENDENT" if count > 0 else "LRN6-ONLY"
            print(f"  Position {i}: {list(pattern)} -> {count} [{status}]")

        # Determine branching structure
        print("\n" + "-" * 40)
        print("BRANCHING ANALYSIS:")

        # Check prefix vs suffix independence
        prefix_2 = (lrn6[0], lrn6[1])
        suffix_2 = (lrn6[4], lrn6[5])

        prefix_count = self.bigrams.get(prefix_2, 0)
        suffix_count = self.bigrams.get(suffix_2, 0)

        print(f"  Prefix [114,464]: {prefix_count} occurrences")
        print(f"  Suffix [94,714]: {suffix_count} occurrences")

        if prefix_count > suffix_count:
            print("  [LEFT-BRANCHING] Prefix is more productive")
            results["branching_analysis"]["type"] = "left_branching"
        elif suffix_count > prefix_count:
            print("  [RIGHT-BRANCHING] Suffix is more productive")
            results["branching_analysis"]["type"] = "right_branching"
        else:
            print("  [COMPOSITIONAL] Both parts equally productive")
            results["branching_analysis"]["type"] = "compositional"

        # Grammar hypothesis
        print("\n" + "-" * 40)
        print("GRAMMAR HYPOTHESIS:")

        # Check if start and end are independent
        start_independent = prefix_count > 1
        end_independent = suffix_count > 1

        if start_independent and end_independent:
            print("  LRN-6 is COMPOSITIONAL:")
            print("    [114,464] = Subject/Opener")
            print("    [604,324] = Verb/Action")
            print("    [94,714] = Object/Closer")
            results["branching_analysis"]["grammar_type"] = "subject_verb_object"
        elif start_independent:
            print("  LRN-6 is LEFT-HEAVY:")
            print("    [114,464] = Base, rest is modification")
            results["branching_analysis"]["grammar_type"] = "left_heavy"
        elif end_independent:
            print("  LRN-6 is RIGHT-HEAVY:")
            print("    [94,714] = Base, start is modification")
            results["branching_analysis"]["grammar_type"] = "right_heavy"
        else:
            print("  LRN-6 is RIGID IDIOM:")
            print("    No sub-parts occur independently")
            results["branching_analysis"]["grammar_type"] = "rigid_idiom"

        return results

    def analyze_ngram_combinatorics(self) -> dict[str, Any]:
        """
        Analyze the combinatorial structure of the vocabulary.

        Key questions:
        - How many unique bigrams can be formed? (max = 510^2 = 260,100)
        - How many actually appear?
        - What is the combinatorial ratio?
        """
        print("\n" + "=" * 60)
        print("N-GRAM COMBINATORICS")
        print("=" * 60)

        # Calculate combinatorial ratios
        vocab_size = self.unique_segments

        # Bigrams
        max_bigrams = vocab_size * vocab_size
        actual_bigrams = len(self.bigrams)
        bigram_ratio = actual_bigrams / max_bigrams

        print("\nBigrams:")
        print(f"  Maximum possible: {max_bigrams:,}")
        print(f"  Actually observed: {actual_bigrams:,}")
        print(f"  Combinatorial ratio: {bigram_ratio:.4%}")

        # Trigrams
        max_trigrams = vocab_size**3
        actual_trigrams = len(self.trigrams)
        trigram_ratio = actual_trigrams / max_trigrams

        print("\nTrigrams:")
        print(f"  Maximum possible: {max_trigrams:,}")
        print(f"  Actually observed: {actual_trigrams:,}")
        print(f"  Combinatorial ratio: {trigram_ratio:.8%}")

        # Interpretation
        print("\n" + "-" * 40)
        print("INTERPRETATION:")

        if bigram_ratio < 0.01:
            print("  HIGHLY RESTRICTIVE: <1% of possible bigrams used")
            print("  Suggests strong grammatical constraints")
        elif bigram_ratio < 0.05:
            print("  MODERATELY RESTRICTIVE: 1-5% of possible bigrams used")
            print("  Suggests probabilistic grammar")
        else:
            print("  FLEXIBLE: >5% of possible bigrams used")
            print("  Suggests open combinatorial system")

        # Zipf analysis
        print("\n" + "-" * 40)
        print("ZIPF DISTRIBUTION CHECK:")

        bigram_counts = sorted([c for c in self.bigrams.values()], reverse=True)[:50]
        if len(bigram_counts) >= 10:
            # Check if distribution follows Zipf's law
            # (rank * frequency should be approximately constant)
            zipf_products = [(i + 1) * c for i, c in enumerate(bigram_counts[:10])]
            avg_zipf = sum(zipf_products) / len(zipf_products)
            variance = sum((x - avg_zipf) ** 2 for x in zipf_products) / len(zipf_products)

            print(f"  Top 10 bigram Zipf products: {zipf_products[:5]}...")
            print(f"  Average: {avg_zipf:.1f}, Variance: {variance:.1f}")

            if variance < avg_zipf * 0.5:
                print("  [ZIPF-LIKE] Distribution follows power law")
                print("  Suggests natural language-like structure")
            else:
                print("  [NON-ZIPF] Distribution does not follow power law")

        return {
            "vocab_size": vocab_size,
            "bigram_ratio": bigram_ratio,
            "trigram_ratio": trigram_ratio,
            "top_bigrams": sorted(
                [(list(k), v) for k, v in self.bigrams.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:20],
        }

    def detect_modulator_segments(self) -> dict[str, Any]:
        """
        Detect segments that "modulate" meaning based on context.

        A modulator segment is one that, when inserted, changes the
        communicative context of a sequence.

        Since we don't have ground-truth context labels, we use
        bigram co-occurrence patterns as a proxy.
        """
        print("\n" + "=" * 60)
        print("MODULATOR SEGMENT DETECTION")
        print("=" * 60)

        # For each segment, check if it appears with diverse partners
        # but with different frequency distributions

        segment_partners: dict[int, Counter] = defaultdict(Counter)

        for (seg1, seg2), count in self.bigrams.items():
            segment_partners[seg1][seg2] += count
            segment_partners[seg2][seg1] += count  # Bidirectional

        results = {}

        # Find segments that have "clumpy" distributions
        # (some partners much more common than others)
        for seg_id, partners in segment_partners.items():
            if len(partners) < 3:
                continue

            total = sum(partners.values())
            if total < 10:
                continue

            # Calculate Gini coefficient (measure of inequality)
            sorted_counts = sorted(partners.values())
            n = len(sorted_counts)
            cumsum = 0
            for i, c in enumerate(sorted_counts):
                cumsum += (2 * (i + 1) - n - 1) * c
            gini = cumsum / (n * sum(sorted_counts))

            if gini > 0.6:  # High inequality
                # This segment has "preferred" partners
                top_partner = partners.most_common(1)[0]
                results[seg_id] = {
                    "gini": gini,
                    "total": total,
                    "unique_partners": len(partners),
                    "top_partner": top_partner,
                    "top_partner_pct": top_partner[1] / total * 100,
                }

        # Print top modulators
        top_modulators = sorted(results.items(), key=lambda x: x[1]["gini"], reverse=True)[:10]

        print("\nPotential Modulator Segments (Gini > 0.6):")
        for seg_id, data in top_modulators:
            print(f"\n  Segment {seg_id}:")
            print(f"    Gini: {data['gini']:.3f}")
            print(f"    Unique partners: {data['unique_partners']}")
            print(f"    Top partner: {data['top_partner'][0]} ({data['top_partner_pct']:.1f}%)")

            if data["top_partner_pct"] > 50:
                print("    [STRONG MODULATOR] Dominant partner preference")
            elif data["top_partner_pct"] > 30:
                print("    [MODERATE MODULATOR] Preferred partner")
            else:
                print("    [WEAK MODULATOR] Distributed preferences")

        return {"modulators": dict(top_modulators)}

    def generate_summary_report(self) -> dict[str, Any]:
        """Generate comprehensive summary of all analyses"""

        print("\n" + "=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)

        # Run all analyses
        segment_roles = self.analyze_segment_roles()
        _transitions = self.analyze_transitions()  # noqa: F841
        lrn6_analysis = self.decompose_lrn6()
        combinatorics = self.analyze_ngram_combinatorics()
        modulators = self.detect_modulator_segments()

        # Summary
        print("\n" + "=" * 60)
        print("KEY FINDINGS")
        print("=" * 60)

        findings = []

        # Finding 1: Segment classification
        n_openers = len(segment_roles["openers"])
        n_closers = len(segment_roles["closers"])
        n_function = len(segment_roles["function_words"])

        findings.append(
            f"1. SEGMENT ROLES: {n_openers} openers, {n_closers} closers, "
            f"{n_function} function words identified"
        )

        # Finding 2: LRN-6 structure
        grammar_type = lrn6_analysis["branching_analysis"].get("grammar_type", "unknown")
        findings.append(f"2. LRN-6 STRUCTURE: {grammar_type.replace('_', ' ').title()}")

        # Finding 3: Combinatorial ratio
        findings.append(
            f"3. COMBINATORIAL RATIO: {combinatorics['bigram_ratio']:.2%} of possible bigrams used"
        )

        # Finding 4: Modulators
        n_modulators = len(modulators["modulators"])
        findings.append(f"4. MODULATOR SEGMENTS: {n_modulators} detected with Gini > 0.6")

        for finding in findings:
            print(f"\n{finding}")

        # Hypothesis
        print("\n" + "-" * 40)
        print("OVERALL HYPOTHESIS:")

        if grammar_type == "subject_verb_object":
            print("The bat vocalization system shows COMPOSITIONAL GRAMMAR with:")
            print("  - Distinct opener/content/closer segments")
            print("  - LRN-6 decomposable into Subject-Verb-Object-like units")
            print("  - Restricted combinatorial space suggesting grammatical rules")
        elif grammar_type == "rigid_idiom":
            print("The bat vocalization system shows FIXED PATTERN structure with:")
            print("  - LRN-6 as rigid idiom (no sub-parts independent)")
            print("  - Limited combinatorial flexibility")
            print("  - Suggests holophrastic (whole-phrase) communication")
        else:
            print("The bat vocalization system shows MIXED structure with:")
            print("  - Some compositional elements")
            print("  - Some fixed patterns")
            print("  - Requires more data for definitive classification")

        return {
            "segment_roles": {
                "n_openers": n_openers,
                "n_closers": n_closers,
                "n_function_words": n_function,
            },
            "lrn6_grammar_type": grammar_type,
            "combinatorial_ratio": combinatorics["bigram_ratio"],
            "n_modulators": n_modulators,
            "findings": findings,
        }


def main():
    """Run the complete Phase 2 linguistic analysis"""

    report_path = Path(__file__).parent / "bat_corpus_analysis_report.json"

    if not report_path.exists():
        print(f"Error: Report not found at {report_path}")
        return

    analyzer = BatLinguisticAnalyzer(str(report_path))
    summary = analyzer.generate_summary_report()

    # Save results
    output_path = Path(__file__).parent / "bat_phase2_linguistic_results.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
