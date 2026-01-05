#!/usr/bin/env python3
"""
Analyze Atomic Phrase Reuse from Imported Database

This script analyzes phrase reuse patterns using the already imported database
with behavioral contexts. Since phrases are already segmented, we analyze:
1. Phrase co-occurrence in the database
2. Context-specific phrase patterns
3. Evidence of compositionality
4. Cross-context phrase reuse
"""

import json
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
DATABASE_PATH = '/home/sheel/birdsong_analysis/src/vocalization_database_with_contexts.json'


def load_database(db_path: str) -> Dict:
    """Load phrase database."""
    print(f"Loading database from {db_path}...")

    with open(db_path, 'r') as f:
        db = json.load(f)

    phrases = db['species_data']['marmoset']['phrases']

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
        contexts = phrase_data.get('contexts', [])
        total_occurrences = phrase_data.get('total_occurrences', 0)

        phrase_contexts[phrase_key] = Counter()
        phrase_total_occurrences[phrase_key] = total_occurrences

        for ctx in contexts:
            ctx_name = ctx['context_name']
            count = ctx['count']

            phrase_contexts[phrase_key][ctx_name] = count
            context_phrases[ctx_name][phrase_key] += count

    # Find phrases that appear in multiple contexts
    multi_context_phrases = {
        k: v for k, v in phrase_contexts.items()
        if len(v) > 1
    }

    print(f"\n📊 MULTI-CONTEXT PHRASES (phrase reuse across contexts):")
    print(f"   Total phrases: {len(phrase_contexts)}")
    print(f"   Multi-context phrases: {len(multi_context_phrases)} "
          f"({len(multi_context_phrases) / len(phrase_contexts) * 100:.1f}%)")

    # Show examples
    print(f"\n   Examples of phrases appearing in multiple contexts:")
    for phrase_key, context_counter in list(multi_context_phrases.items())[:10]:
        total = sum(context_counter.values())
        contexts_str = ", ".join([f"{ctx} ({count})" for ctx, count in context_counter.most_common()])
        print(f"      {phrase_key}: {contexts_str} (total: {total})")

    # Analyze context overlap
    print(f"\n📊 CONTEXT OVERLAP ANALYSIS:")

    contexts = list(context_phrases.keys())

    for i, ctx1 in enumerate(contexts):
        for ctx2 in contexts[i+1:]:
            # Find shared phrases
            phrases1 = set(context_phrases[ctx1].keys())
            phrases2 = set(context_phrases[ctx2].keys())

            shared = phrases1 & phrases2

            if len(shared) > 0:
                overlap_pct = len(shared) / min(len(phrases1), len(phrases2)) * 100
                print(f"   {ctx1} ↔ {ctx2}: {len(shared)} shared phrases ({overlap_pct:.1f}% overlap)")

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
            entropy = -sum(p * np.log(p) for p in proportions if p > 0)

            # Normalize entropy
            max_entropy = np.log(len(context_counter))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            if normalized_entropy < 0.3:
                specialized_phrases.append(phrase_key)
            else:
                generalized_phrases.append((phrase_key, normalized_entropy))

    print(f"\n📊 SPECIALIZATION vs GENERALIZATION:")
    print(f"   Specialized phrases (single context): {len(specialized_phrases)}")
    print(f"   Generalized phrases (multiple contexts): {len(generalized_phrases)}")

    if generalized_phrases:
        print(f"\n   Most generalized phrases (highest entropy):")
        generalized_phrases.sort(key=lambda x: x[1], reverse=True)
        for phrase_key, entropy in generalized_phrases[:10]:
            contexts_str = ", ".join(list(phrase_contexts[phrase_key].keys())[:3])
            if len(phrase_contexts[phrase_key]) > 3:
                contexts_str += "..."
            print(f"      {phrase_key}: entropy={entropy:.3f}, contexts=[{contexts_str}]")

    return {
        'phrase_contexts': phrase_contexts,
        'context_phrases': dict(context_phrases),
        'multi_context_phrases': multi_context_phrases,
        'specialized_count': len(specialized_phrases),
        'generalized_count': len(generalized_phrases)
    }


def analyze_acoustic_similarity_across_contexts(phrases: Dict):
    """Analyze if phrases with similar acoustic features appear in different contexts."""
    print("\n" + "=" * 80)
    print("ANALYZING ACOUSTIC SIMILARITY ACROSS CONTEXTS")
    print("=" * 80)

    # Group phrases by context
    context_phrases = defaultdict(dict)  # context -> {phrase_key: features}

    for phrase_key, phrase_data in phrases.items():
        af = phrase_data['acoustic_features']

        for ctx in phrase_data.get('contexts', []):
            ctx_name = ctx['context_name']
            context_phrases[ctx_name][phrase_key] = af

    # For each pair of contexts, find acoustically similar phrases
    contexts = list(context_phrases.keys())

    print(f"\n📊 CROSS-CONTEXT ACOUSTIC SIMILARITY:")
    print(f"   Comparing phrases across {len(contexts)} contexts...")

    similar_cross_context = []

    for i, ctx1 in enumerate(contexts):
        for ctx2 in contexts[i+1:]:
            phrases1 = context_phrases[ctx1]
            phrases2 = context_phrases[ctx2]

            # Find similar phrases (similar F0 and duration)
            for phrase_key1, af1 in list(phrases1.items())[:20]:  # Sample 20 per context
                f0_1 = af1.get('mean_f0_hz', 0)
                dur_1 = af1.get('mean_duration_ms', 0)

                for phrase_key2, af2 in phrases2.items():
                    f0_2 = af2.get('mean_f0_hz', 0)
                    dur_2 = af2.get('mean_duration_ms', 0)

                    # Check similarity
                    f0_diff = abs(f0_1 - f0_2)
                    dur_diff = abs(dur_1 - dur_2)

                    if f0_diff < 200 and dur_diff < 100:  # Within 200Hz F0, 100ms duration
                        similar_cross_context.append({
                            'context1': ctx1,
                            'phrase1': phrase_key1,
                            'context2': ctx2,
                            'phrase2': phrase_key2,
                            'f0_diff': f0_diff,
                            'dur_diff': dur_diff
                        })

    print(f"   Found {len(similar_cross_context)} similar phrase pairs across contexts")

    if similar_cross_context:
        print(f"\n   Examples of acoustically similar phrases in different contexts:")
        for sim in similar_cross_context[:10]:
            print(f"      {sim['context1']}:{sim['phrase1'][:30]}...")
            print(f"      {sim['context2']}:{sim['phrase2'][:30]}...")
            print(f"      ΔF0: {sim['f0_diff']:.0f}Hz, ΔDur: {sim['dur_diff']:.0f}ms")
            print()

    return similar_cross_context


def analyze_compositionality_evidence(phrases: Dict):
    """Look for evidence of compositional syntax."""
    print("\n" + "=" * 80)
    print("ANALYZING EVIDENCE OF COMPOSITIONALITY")
    print("=" * 80)

    # Since we don't have sentence-level data, we look for:
    # 1. Phrase reuse across contexts (suggests flexible combination)
    # 2. Context-specific phrase combinations
    # 3. Inverse correlations (certain phrases avoid certain contexts)

    # Build context association matrix
    contexts = set()
    for phrase_data in phrases.values():
        for ctx in phrase_data.get('contexts', []):
            contexts.add(ctx['context_name'])

    contexts = sorted(list(contexts))

    phrase_context_matrix = {}  # phrase_key -> {context: proportion}

    for phrase_key, phrase_data in phrases.items():
        total = sum(ctx['count'] for ctx in phrase_data.get('contexts', []))

        if total > 0:
            phrase_context_matrix[phrase_key] = {}
            for ctx in phrase_data.get('contexts', []):
                ctx_name = ctx['context_name']
                count = ctx['count']
                phrase_context_matrix[phrase_key][ctx_name] = count / total

    # Find context-specific phrases (enriched in one context)
    print(f"\n📊 CONTEXT-SPECIFIC PHRASES:")

    for ctx in contexts:
        # Find phrases enriched in this context (>70% of occurrences)
        enriched = []

        for phrase_key, context_dist in phrase_context_matrix.items():
            if ctx in context_dist and context_dist[ctx] > 0.7:
                enriched.append((phrase_key, context_dist[ctx]))

        if enriched:
            print(f"\n   {ctx.upper()}: {len(enriched)} context-specific phrases")
            for phrase_key, proportion in enriched[:5]:
                print(f"      {phrase_key}: {proportion*100:.1f}% in {ctx}")

    # Find "flexible" phrases (evenly distributed across contexts)
    print(f"\n📊 FLEXIBLE PHRASES (potential building blocks):")

    flexible = []
    for phrase_key, context_dist in phrase_context_matrix.items():
        if len(context_dist) >= 3:  # Appears in 3+ contexts
            # Calculate entropy
            proportions = list(context_dist.values())
            entropy = -sum(p * np.log(p) for p in proportions if p > 0)
            max_entropy = np.log(len(context_dist))
            normalized_entropy = entropy / max_entropy

            if normalized_entropy > 0.7:  # High entropy = evenly distributed
                flexible.append((phrase_key, normalized_entropy, len(context_dist)))

    flexible.sort(key=lambda x: x[1], reverse=True)

    if flexible:
        print(f"   Found {len(flexible)} flexible phrases")
        for phrase_key, entropy, num_contexts in flexible[:10]:
            print(f"      {phrase_key}: entropy={entropy:.3f}, {num_contexts} contexts")

    # Scientific interpretation
    print("\n" + "=" * 80)
    print("📚 SCIENTIFIC INTERPRETATION")
    print("=" * 80)

    total_phrases = len(phrases)
    num_flexible = len(flexible)
    flexible_pct = (num_flexible / total_phrases) * 100 if total_phrases > 0 else 0

    if flexible_pct > 20:
        print(f"\n✅ STRONG EVIDENCE OF COMPOSITIONAL SYSTEM")
        print(f"   - {flexible_pct:.1f}% of phrases are flexible (used across multiple contexts)")
        print(f"   - Suggests combinatorial grammar: flexible phrases + context-specific modifiers")
        print(f"   - Analogy: function words (flexible) + content words (context-specific)")
    elif flexible_pct > 5:
        print(f"\n✅ MODERATE EVIDENCE OF COMPOSITIONALITY")
        print(f"   - {flexible_pct:.1f}% of phrases are flexible")
        print(f"   - Some combinatorial patterns detected")
        print(f"   - May indicate emerging syntax or contextual flexibility")
    else:
        print(f"\n⚠️  LIMITED EVIDENCE OF COMPOSITIONALITY")
        print(f"   - {flexible_pct:.1f}% of phrases are flexible")
        print(f"   - Most phrases are context-specific")
        print(f"   - Suggests: holistic calls OR limited combinatorial ability")

    print("\n" + "=" * 80)


def main():
    """Main analysis function."""
    print("=" * 80)
    print("PHRASE REUSE AND COMPOSITIONALITY ANALYSIS")
    print("=" * 80)

    # Load database
    phrases = load_database(DATABASE_PATH)

    # Analyze phrase-context patterns
    context_results = analyze_phrase_context_patterns(phrases)

    # Analyze acoustic similarity across contexts
    similarity_results = analyze_acoustic_similarity_across_contexts(phrases)

    # Analyze compositionality evidence
    analyze_compositionality_evidence(phrases)

    # Save results
    output_path = '/home/sheel/birdsong_analysis/src/phrase_reuse_analysis.json'
    print(f"\n💾 Saving results to {output_path}...")

    results = {
        'total_phrases': len(phrases),
        'multi_context_count': len(context_results['multi_context_phrases']),
        'specialized_count': context_results['specialized_count'],
        'generalized_count': context_results['generalized_count'],
        'similar_cross_context_pairs': len(similarity_results)
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved!")

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
