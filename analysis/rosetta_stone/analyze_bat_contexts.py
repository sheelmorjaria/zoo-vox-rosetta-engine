#!/usr/bin/env python3
"""
Analyze Egyptian Fruit Bat Phrase-Context Patterns

This script analyzes phrase reuse patterns across behavioral contexts
for Egyptian fruit bat vocalizations, investigating:
1. Multi-context phrase usage (phrase flexibility)
2. Context-specific phrase enrichment
3. Acoustic feature distributions by context
4. Evidence of compositional syntax
5. Context overlap and similarity
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import entropy

sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
DATABASE_PATH = "/home/sheel/birdsong_analysis/src/vocalization_database_with_bat_contexts.json"
OUTPUT_PATH = "/home/sheel/birdsong_analysis/src/bat_context_analysis.json"


def load_database(db_path: str) -> Dict:
    """Load bat phrase database."""
    print(f"Loading database from {db_path}...")

    with open(db_path, "r") as f:
        db = json.load(f)

    phrases = db["species_data"]["egyptian_bat"]["phrases"]

    print(f"✅ Loaded {len(phrases)} phrase types")

    return phrases


def analyze_phrase_context_patterns(phrases: Dict) -> Dict:
    """Analyze how phrases are distributed across behavioral contexts."""
    print("\n" + "=" * 80)
    print("ANALYZING PHRASE-CONTEXT PATTERNS")
    print("=" * 80)

    # Build phrase->contexts mapping
    phrase_contexts = {}  # phrase_key -> Counter of contexts
    context_phrases = defaultdict(Counter)  # context -> Counter of phrases
    phrase_total_occurrences = {}  # phrase_key -> total count

    for phrase_key, phrase_data in phrases.items():
        contexts = phrase_data.get("contexts", [])
        total_occurrences = phrase_data.get("total_occurrences", 0)

        phrase_contexts[phrase_key] = Counter()
        phrase_total_occurrences[phrase_key] = total_occurrences

        for ctx in contexts:
            ctx_name = ctx["context_name"]
            count = ctx["count"]

            phrase_contexts[phrase_key][ctx_name] = count
            context_phrases[ctx_name][phrase_key] += count

    # Find phrases that appear in multiple contexts
    multi_context_phrases = {k: v for k, v in phrase_contexts.items() if len(v) > 1}

    single_context_phrases = {k: v for k, v in phrase_contexts.items() if len(v) == 1}

    print("\n📊 CONTEXT DISTRIBUTION:")
    print(f"   Total phrases: {len(phrase_contexts)}")
    print(
        f"   Multi-context phrases: {len(multi_context_phrases)} "
        f"({len(multi_context_phrases) / len(phrase_contexts) * 100:.1f}%)"
    )
    print(
        f"   Single-context phrases: {len(single_context_phrases)} "
        f"({len(single_context_phrases) / len(phrase_contexts) * 100:.1f}%)"
    )

    # Analyze context diversity
    context_diversity = {}
    for ctx, phrase_counter in context_phrases.items():
        total = sum(phrase_counter.values())
        unique_phrases = len(phrase_counter)
        # Simpson's diversity index
        proportions = [count / total for count in phrase_counter.values()]
        simpson_diversity = 1 - sum(p**2 for p in proportions)
        context_diversity[ctx] = {
            "total_occurrences": total,
            "unique_phrases": unique_phrases,
            "simpson_diversity": simpson_diversity,
        }

    print("\n📊 CONTEXT DIVERSITY:")
    for ctx in sorted(context_diversity.keys()):
        stats = context_diversity[ctx]
        print(f"   {ctx}:")
        print(f"      Total occurrences: {stats['total_occurrences']}")
        print(f"      Unique phrases: {stats['unique_phrases']}")
        print(f"      Simpson diversity: {stats['simpson_diversity']:.3f}")

    # Show examples of multi-context phrases
    print("\n📊 MULTI-CONTEXT PHRASE EXAMPLES:")
    for phrase_key, context_counter in list(multi_context_phrases.items())[:10]:
        total = sum(context_counter.values())
        contexts_str = ", ".join(
            [f"{ctx} ({count})" for ctx, count in context_counter.most_common(5)]
        )
        if len(context_counter) > 5:
            contexts_str += "..."
        print(f"      {phrase_key}:")
        print(f"         Contexts: {contexts_str}")
        print(f"         Total: {total}, Diversity: {len(context_counter)} contexts")

    # Analyze phrase specialization vs generalization
    specialized_phrases = []  # Phrases primarily in one context
    generalized_phrases = []  # Phrases evenly distributed across contexts

    for phrase_key, context_counter in phrase_contexts.items():
        if len(context_counter) == 1:
            specialized_phrases.append(phrase_key)
        else:
            # Calculate entropy (higher = more generalized)
            total = sum(context_counter.values())
            proportions = [count / total for count in context_counter.values()]
            phrase_entropy = entropy(proportions)

            # Normalize entropy
            max_entropy = np.log(len(context_counter))
            normalized_entropy = phrase_entropy / max_entropy if max_entropy > 0 else 0

            if normalized_entropy < 0.3:
                specialized_phrases.append(phrase_key)
            else:
                generalized_phrases.append((phrase_key, normalized_entropy, len(context_counter)))

    print("\n📊 SPECIALIZATION vs GENERALIZATION:")
    print(f"   Specialized phrases (single-context or biased): {len(specialized_phrases)}")
    print(f"   Generalized phrases (distributed across contexts): {len(generalized_phrases)}")

    if generalized_phrases:
        print("\n   Most generalized phrases (highest entropy):")
        generalized_phrases.sort(key=lambda x: x[1], reverse=True)
        for phrase_key, ent, num_ctx in generalized_phrases[:15]:
            contexts_str = ", ".join(list(phrase_contexts[phrase_key].keys())[:5])
            if len(phrase_contexts[phrase_key]) > 5:
                contexts_str += "..."
            print(f"      {phrase_key}:")
            print(f"         Entropy: {ent:.3f}, Contexts: {num_ctx}")
            print(f"         Distribution: {contexts_str}")

    return {
        "phrase_contexts": phrase_contexts,
        "context_phrases": dict(context_phrases),
        "multi_context_phrases": multi_context_phrases,
        "single_context_phrases": single_context_phrases,
        "context_diversity": context_diversity,
        "specialized_count": len(specialized_phrases),
        "generalized_count": len(generalized_phrases),
        "generalized_phrases": generalized_phrases,
    }


def analyze_context_overlap(phrases: Dict, context_results: Dict):
    """Analyze overlap between behavioral contexts."""
    print("\n" + "=" * 80)
    print("ANALYZING CONTEXT OVERLAP")
    print("=" * 80)

    context_phrases = context_results["context_phrases"]
    contexts = sorted(context_phrases.keys())

    print("\n📊 CONTEXT PAIR OVERLAP:")
    print(f"   Comparing {len(contexts)} contexts...")

    overlap_matrix = {}
    significant_overlaps = []

    for i, ctx1 in enumerate(contexts):
        for ctx2 in contexts[i + 1 :]:
            phrases1 = set(context_phrases[ctx1].keys())
            phrases2 = set(context_phrases[ctx2].keys())

            shared = phrases1 & phrases2

            if len(shared) > 0:
                jaccard = len(shared) / len(phrases1 | phrases2)
                overlap_pct = len(shared) / min(len(phrases1), len(phrases2)) * 100

                overlap_matrix[f"{ctx1}-{ctx2}"] = {
                    "shared_count": len(shared),
                    "jaccard_index": jaccard,
                    "overlap_percentage": overlap_pct,
                }

                if overlap_pct > 10:  # More than 10% overlap
                    significant_overlaps.append(
                        {
                            "context_pair": f"{ctx1}-{ctx2}",
                            "shared_count": len(shared),
                            "jaccard": jaccard,
                            "overlap_pct": overlap_pct,
                        }
                    )

                print(f"   {ctx1} ↔ {ctx2}:")
                print(f"      Shared phrases: {len(shared)}")
                print(f"      Jaccard index: {jaccard:.3f}")
                print(f"      Overlap: {overlap_pct:.1f}% of smaller context")

    print("\n📊 SIGNIFICANT OVERLAPS (>10%):")
    if significant_overlaps:
        significant_overlaps.sort(key=lambda x: x["overlap_pct"], reverse=True)
        for overlap in significant_overlaps[:20]:
            print(f"      {overlap['context_pair']}:")
            print(f"         {overlap['shared_count']} shared phrases")
            print(
                f"         {overlap['overlap_pct']:.1f}% overlap (Jaccard: {overlap['jaccard']:.3f})"
            )
    else:
        print("   No significant overlaps found")

    return {"overlap_matrix": overlap_matrix, "significant_overlaps": significant_overlaps}


def analyze_context_specific_phrases(phrases: Dict, context_results: Dict):
    """Identify phrases significantly enriched in specific contexts."""
    print("\n" + "=" * 80)
    print("IDENTIFYING CONTEXT-SPECIFIC PHRASES")
    print("=" * 80)

    phrase_contexts = context_results["phrase_contexts"]
    contexts = sorted(
        set(ctx for phrase_ctx in phrase_contexts.values() for ctx in phrase_ctx.keys())
    )

    print("\n📊 CONTEXT-ENRICHED PHRASES:")

    context_enriched = {}

    for ctx in contexts:
        enriched_phrases = []

        for phrase_key, context_counter in phrase_contexts.items():
            if ctx in context_counter:
                total = sum(context_counter.values())
                count_in_ctx = context_counter[ctx]
                proportion = count_in_ctx / total

                # Enriched if >70% in this context
                if proportion > 0.7:
                    enriched_phrases.append(
                        {
                            "phrase_key": phrase_key,
                            "proportion": proportion,
                            "count_in_ctx": count_in_ctx,
                            "total": total,
                        }
                    )

        enriched_phrases.sort(key=lambda x: x["proportion"], reverse=True)

        if enriched_phrases:
            print(f"\n   {ctx.upper()}: {len(enriched_phrases)} enriched phrases")
            context_enriched[ctx] = enriched_phrases

            for phrase in enriched_phrases[:8]:
                print(f"      {phrase['phrase_key']}:")
                print(
                    f"         {phrase['proportion'] * 100:.1f}% in {ctx} "
                    f"({phrase['count_in_ctx']}/{phrase['total']})"
                )

    return context_enriched


def analyze_acoustic_features_by_context(phrases: Dict, context_results: Dict):
    """Analyze acoustic feature distributions for each context."""
    print("\n" + "=" * 80)
    print("ANALYZING ACOUSTIC FEATURES BY CONTEXT")
    print("=" * 80)

    context_results["phrase_contexts"]

    # Group phrases by their primary context
    context_to_primary_phrases = defaultdict(list)

    for phrase_key, phrase_data in phrases.items():
        contexts = phrase_data.get("contexts", [])
        if contexts:
            # Find primary context (highest count)
            primary_ctx = max(contexts, key=lambda x: x["count"])["context_name"]
            context_to_primary_phrases[primary_ctx].append(phrase_key)

    # Calculate mean acoustic features per context
    context_acoustic_profiles = {}

    print("\n📊 ACOUSTIC PROFILES BY CONTEXT:")

    for ctx, phrase_keys in sorted(context_to_primary_phrases.items()):
        if len(phrase_keys) < 5:
            continue

        # Aggregate features
        features_list = []
        for pk in phrase_keys:
            af = phrases[pk]["acoustic_features"]
            features_list.append(af)

        # Calculate means
        mean_f0 = np.mean([f["f0_mean"] for f in features_list])
        mean_duration = np.mean([f["duration_ms"] for f in features_list])
        mean_f0_range = np.mean([f["f0_range"] for f in features_list])
        mean_hnr = np.mean([f.get("harmonic_to_noise_ratio", 0) for f in features_list])
        mean_spectral_centroid = np.mean([f.get("spectral_centroid_hz", 0) for f in features_list])
        mean_attack_time = np.mean([f.get("attack_time_ms", 0) for f in features_list])

        context_acoustic_profiles[ctx] = {
            "num_phrases": len(phrase_keys),
            "mean_f0_hz": mean_f0,
            "mean_duration_ms": mean_duration,
            "mean_f0_range_hz": mean_f0_range,
            "mean_hnr": mean_hnr,
            "mean_spectral_centroid_hz": mean_spectral_centroid,
            "mean_attack_time_ms": mean_attack_time,
        }

        print(f"\n   {ctx.upper()} ({len(phrase_keys)} phrases):")
        print(f"      Mean F0: {mean_f0:.0f} Hz")
        print(f"      Mean duration: {mean_duration:.1f} ms")
        print(f"      Mean F0 range: {mean_f0_range:.0f} Hz")
        print(f"      Mean HNR: {mean_hnr:.1f}")
        print(f"      Mean spectral centroid: {mean_spectral_centroid:.0f} Hz")
        print(f"      Mean attack time: {mean_attack_time:.2f} ms")

    return context_acoustic_profiles


def analyze_compositionality_evidence(phrases: Dict, context_results: Dict):
    """Look for evidence of compositional syntax."""
    print("\n" + "=" * 80)
    print("ANALYZING EVIDENCE OF COMPOSITIONALITY")
    print("=" * 80)

    generalized_phrases = context_results["generalized_phrases"]
    total_phrases = len(phrases)

    # Categorize by flexibility
    highly_flexible = [p for p in generalized_phrases if p[1] > 0.7]  # Entropy > 0.7
    moderately_flexible = [p for p in generalized_phrases if 0.3 <= p[1] <= 0.7]

    print("\n📊 PHRASE FLEXIBILITY ANALYSIS:")
    print(f"   Total phrases: {total_phrases}")
    print(
        f"   Highly flexible (entropy > 0.7): {len(highly_flexible)} "
        f"({len(highly_flexible) / total_phrases * 100:.1f}%)"
    )
    print(
        f"   Moderately flexible (0.3-0.7): {len(moderately_flexible)} "
        f"({len(moderately_flexible) / total_phrases * 100:.1f}%)"
    )
    print(
        f"   Specialized: {context_results['specialized_count']} "
        f"({context_results['specialized_count'] / total_phrases * 100:.1f}%)"
    )

    # Analyze flexible phrases
    if highly_flexible:
        print("\n📊 HIGHLY FLEXIBLE PHRASES (potential building blocks):")
        for phrase_key, ent, num_ctx in highly_flexible[:20]:
            print(f"      {phrase_key}:")
            print(f"         Entropy: {ent:.3f}, Contexts: {num_ctx}")

    # Scientific interpretation
    print("\n" + "=" * 80)
    print("📚 SCIENTIFIC INTERPRETATION")
    print("=" * 80)

    flexible_pct = len(generalized_phrases) / total_phrases if total_phrases > 0 else 0
    multi_context_pct = (
        len(context_results["multi_context_phrases"]) / total_phrases if total_phrases > 0 else 0
    )

    if flexible_pct > 0.20:
        print("\n✅ STRONG EVIDENCE OF COMPOSITIONAL SYSTEM")
        print(
            f"   - {flexible_pct * 100:.1f}% of phrases are flexible (used across multiple contexts)"
        )
        print(f"   - {multi_context_pct * 100:.1f}% appear in multiple contexts")
        print("   - Suggests combinatorial grammar: flexible phrases + context-specific modifiers")
        print("   - Bat vocalizations may use combinatorial syntax for communication")
    elif flexible_pct > 0.05:
        print("\n✅ MODERATE EVIDENCE OF COMPOSITIONALITY")
        print(f"   - {flexible_pct * 100:.1f}% of phrases are flexible")
        print(f"   - {multi_context_pct * 100:.1f}% appear in multiple contexts")
        print("   - Some combinatorial patterns detected")
        print("   - May indicate emerging syntax or contextual flexibility")
    else:
        print("\n⚠️  LIMITED EVIDENCE OF COMPOSITIONALITY")
        print(f"   - {flexible_pct * 100:.1f}% of phrases are flexible")
        print(f"   - {multi_context_pct * 100:.1f}% appear in multiple contexts")
        print("   - Most phrases are context-specific")
        print("   - Suggests: holistic calls OR limited combinatorial ability")

    print("\n" + "=" * 80)

    return {
        "total_phrases": total_phrases,
        "highly_flexible_count": len(highly_flexible),
        "moderately_flexible_count": len(moderately_flexible),
        "specialized_count": context_results["specialized_count"],
        "flexible_percentage": flexible_pct * 100,
        "multi_context_percentage": multi_context_pct * 100,
    }


def main():
    """Main analysis function."""
    print("=" * 80)
    print("EGYPTIAN FRUIT BAT CONTEXT ANALYSIS")
    print("=" * 80)

    # Load database
    phrases = load_database(DATABASE_PATH)

    # Analyze phrase-context patterns
    context_results = analyze_phrase_context_patterns(phrases)

    # Analyze context overlap
    overlap_results = analyze_context_overlap(phrases, context_results)

    # Identify context-specific phrases
    analyze_context_specific_phrases(phrases, context_results)

    # Analyze acoustic features by context
    acoustic_results = analyze_acoustic_features_by_context(phrases, context_results)

    # Analyze compositionality evidence
    compositionality_results = analyze_compositionality_evidence(phrases, context_results)

    # Save results
    print(f"\n💾 Saving results to {OUTPUT_PATH}...")

    export_results = {
        "analysis_date": context_results["context_diversity"],
        "phrase_context_summary": {
            "total_phrases": len(phrases),
            "multi_context_count": len(context_results["multi_context_phrases"]),
            "specialized_count": context_results["specialized_count"],
            "generalized_count": context_results["generalized_count"],
        },
        "context_overlap": {
            "significant_overlaps": len(overlap_results["significant_overlaps"]),
            "overlap_pairs": overlap_results["significant_overlaps"][:50],
        },
        "compositionality": compositionality_results,
        "context_acoustic_profiles": acoustic_results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(export_results, f, indent=2)

    print("✅ Saved!")

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)

    print("\n📊 KEY FINDINGS:")
    print(
        f"   - {len(context_results['multi_context_phrases'])}/{len(phrases)} phrases appear in multiple contexts"
    )
    print(f"   - {compositionality_results['flexible_percentage']:.1f}% flexible phrases")
    print(f"   - {len(overlap_results['significant_overlaps'])} significant context overlaps")
    print(f"   - {len(acoustic_results)} contexts with distinct acoustic profiles")


if __name__ == "__main__":
    main()
