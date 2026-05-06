#!/usr/bin/env python3
"""
Egyptian Fruit Bat PCFG Syntax Analysis - 112D Features

This script uses Probabilistic Context-Free Grammar induction to discover
syntactic structure in Egyptian fruit bat vocalizations using fresh 112D
RosettaFeatures.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from semiotics.pcfg_induction import (
    GrammarRule,
    PCFG,
    PCFGInducer,
    VocalizationGrammar
)


def load_bat_sequences(
    data_path: str = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/sequences_by_context.json"
) -> Dict[str, List[List[int]]]:
    """Load bat vocalization sequences from 112D extraction."""
    print("Loading bat vocalization sequences (112D-based)...")
    print(f"  Path: {data_path}")

    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"  Error: File not found. Run extraction first.")
        print(f"  Run: python3 src/analysis/monitor_and_analyze.py")
        sys.exit(1)

    total_sequences = sum(len(seq) for seq in data.values())
    print(f"  Loaded {len(data)} contexts with {total_sequences} total sequences")

    return data


def convert_sequences_to_symbols(sequences: List[List[int]]) -> List[List[str]]:
    """Convert integer cluster IDs to string symbols for PCFG."""
    return [[f"C{cid}" for cid in seq] for seq in sequences]


def analyze_context_grammar(
    context_id: str,
    sequences: List[List[int]],
    n_samples: int = 5000
) -> Dict:
    """Analyze grammar for a specific context."""
    print(f"\n{'=' * 70}")
    print(f"Analyzing Context {context_id}")
    print(f"{'=' * 70}")

    # Sample sequences if too many
    if len(sequences) > n_samples:
        sampled = sequences[:n_samples]
        print(f"  Using {n_samples}/{len(sequences)} sequences (sampled)")
    else:
        sampled = sequences
        print(f"  Using all {len(sequences)} sequences")

    # Convert to symbols
    symbol_sequences = convert_sequences_to_symbols(sampled)

    # Get unique vocabulary
    unique_symbols = sorted(set(s for seq in symbol_sequences for s in seq))
    print(f"  Vocabulary size: {len(unique_symbols)} symbols")

    # Analyze sequence lengths
    lengths = [len(seq) for seq in symbol_sequences]
    print(f"  Sequence length: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    # Analyze transitions
    transition_counts = defaultdict(int)
    for seq in symbol_sequences:
        for i in range(len(seq) - 1):
            transition = (seq[i], seq[i + 1])
            transition_counts[transition] += 1

    print(f"  Unique transitions: {len(transition_counts)}")

    # Find most common transitions
    top_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Top transitions:")
    for (src, dst), count in top_transitions:
        print(f"    {src} -> {dst}: {count} occurrences")

    # Induce PCFG
    print(f"\n  Inducing PCFG from sequences...")
    inducer = PCFGInducer(
        max_rule_length=5,
        min_frequency=2
    )

    try:
        learned_pcfg = inducer.induce(symbol_sequences)
        n_rules = len(learned_pcfg.rules)
        print(f"  Learned {n_rules} grammar rules")

        # Show top rules
        print(f"\n  Top grammar rules:")
        rules_by_lhs = defaultdict(list)
        for rule in learned_pcfg.rules:
            rules_by_lhs[rule.lhs].append(rule)

        all_rules = []
        for lhs, rules in rules_by_lhs.items():
            for rule in rules:
                all_rules.append((rule, lhs))

        all_rules.sort(key=lambda x: x[0].prob, reverse=True)

        for i, (rule, lhs) in enumerate(all_rules[:10]):
            rhs_str = " ".join(rule.rhs)
            print(f"    {i+1}. {lhs} -> {rhs_str} (p={rule.prob:.3f})")

        # Compute entropy
        entropy = learned_pcfg.compute_entropy()
        print(f"\n  Grammar entropy: {entropy:.3f} bits")

        # Return results
        return {
            "context_id": context_id,
            "n_sequences": len(sampled),
            "vocabulary_size": len(unique_symbols),
            "avg_length": float(np.mean(lengths)),
            "n_rules": n_rules,
            "entropy": float(entropy),
            "top_transitions": [(f"{s}->{d}", c) for (s, d), c in top_transitions[:5]],
            "pcfg": learned_pcfg
        }

    except Exception as e:
        print(f"  Error inducing PCFG: {e}")
        import traceback
        traceback.print_exc()

        return {
            "context_id": context_id,
            "n_sequences": len(sampled),
            "vocabulary_size": len(unique_symbols),
            "avg_length": float(np.mean(lengths)),
            "n_rules": 0,
            "entropy": 0.0,
            "top_transitions": [(f"{s}->{d}", c) for (s, d), c in top_transitions[:5]],
            "pcfg": None
        }


def compare_context_grammars(
    results: List[Dict]
) -> None:
    """Compare grammars across contexts."""
    print(f"\n{'=' * 70}")
    print("Cross-Context Grammar Comparison (112D)")
    print(f"{'=' * 70}")

    # Print comparison table
    print(f"\n{'Context':<15} {'Sequences':<12} {'Vocab':<8} {'Rules':<8} {'Entropy':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['context_id']:<15} {r['n_sequences']:<12} {r['vocabulary_size']:<8} {r['n_rules']:<8} {r['entropy']:<12.3f}")

    # Find most and least complex contexts
    valid_results = [r for r in results if r['entropy'] > 0]
    if valid_results:
        most_complex = max(valid_results, key=lambda x: x['entropy'])
        least_complex = min(valid_results, key=lambda x: x['entropy'])
        print(f"\nMost complex context: {most_complex['context_id']} (entropy={most_complex['entropy']:.3f})")
        print(f"Least complex context: {least_complex['context_id']} (entropy={least_complex['entropy']:.3f})")


def find_common_patterns(
    sequences: List[List[int]],
    min_length: int = 3,
    min_count: int = 50
) -> List[Tuple[Tuple[int, ...], int]]:
    """Find common sequential patterns (n-grams)."""
    ngrams = []

    for seq in sequences:
        for length in range(min_length, min(len(seq), min_length + 4)):
            for i in range(len(seq) - length + 1):
                ngram = tuple(seq[i:i + length])
                ngrams.append(ngram)

    # Count occurrences
    ngram_counts = Counter(ngrams)

    # Filter by minimum count
    common_patterns = [(ngram, count) for ngram, count in ngram_counts.items()
                      if count >= min_count]

    # Sort by frequency
    common_patterns.sort(key=lambda x: x[1], reverse=True)

    return common_patterns[:20]


def compare_with_baseline(
    results_112d: Dict,
    baseline_path: str = "/mnt/c/Users/sheel/Desktop/src/analysis/results/bat_pcfg_analysis.json"
) -> None:
    """Compare 112D results with ~30D baseline."""
    print(f"\n{'=' * 70}")
    print("112D vs ~30D Baseline Comparison")
    print(f"{'=' * 70}")

    try:
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)

        print(f"\nBaseline (~30D features):")
        print(f"  Contexts analyzed: {baseline['contexts_analyzed']}")
        baseline_rules = sum(r['n_rules'] for r in baseline['context_results'])
        print(f"  Total grammar rules: {baseline_rules}")

        print(f"\n112D Features:")
        print(f"  Contexts analyzed: {results_112d['contexts_analyzed']}")
        rules_112d = sum(r['n_rules'] for r in results_112d['context_results'])
        print(f"  Total grammar rules: {rules_112d}")

        print(f"\nGrammar rules increase: {rules_112d - baseline_rules:+d} ({(rules_112d/baseline_rules - 1)*100:+.1f}%)")

    except FileNotFoundError:
        print(f"  Baseline results not found at {baseline_path}")


def main():
    """Main analysis pipeline."""
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║     Egyptian Fruit Bat PCFG Syntax Analysis (112D Features)               ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")

    # Load data
    sequences_by_context = load_bat_sequences()

    # Analyze grammar for each context
    results = []

    # Analyze contexts with most data
    context_sizes = [(ctx, len(seq)) for ctx, seq in sequences_by_context.items()]
    context_sizes.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop 10 contexts by sequence count:")
    for ctx, count in context_sizes[:10]:
        print(f"  {ctx}: {count} sequences")

    # Analyze top contexts
    top_contexts = [ctx for ctx, _ in context_sizes[:10]]

    for ctx_id in top_contexts:
        sequences = sequences_by_context[ctx_id]
        result = analyze_context_grammar(ctx_id, sequences)
        results.append(result)

    # Compare grammars
    compare_context_grammars(results)

    # Find common patterns across all contexts
    print(f"\n{'=' * 70}")
    print("Common Sequential Patterns (All Contexts - 112D)")
    print(f"{'=' * 70}")

    all_sequences = []
    for sequences in sequences_by_context.values():
        all_sequences.extend(sequences[:5000])  # Limit for performance

    common_patterns = find_common_patterns(all_sequences, min_length=3, min_count=100)

    print(f"\nTop 10 most common 3+ vocalization patterns:")
    for i, (pattern, count) in enumerate(common_patterns[:10]):
        pattern_str = " -> ".join(str(p) for p in pattern)
        print(f"  {i+1}. {pattern_str} ({count} occurrences)")

    # Save results
    print(f"\n{'=' * 70}")
    print("Saving Analysis Results")
    print(f"{'=' * 70}")

    save_results = []
    for r in results:
        save_results.append({
            "context_id": r["context_id"],
            "n_sequences": r["n_sequences"],
            "vocabulary_size": r["vocabulary_size"],
            "avg_length": r["avg_length"],
            "n_rules": r["n_rules"],
            "entropy": r["entropy"],
            "top_transitions": r["top_transitions"]
        })

    results_data = {
        "analysis_type": "pcfg_syntax_induction_112d",
        "features": "112d_rosetta_features",
        "species": "egyptian_fruit_bat",
        "contexts_analyzed": len(results),
        "context_results": save_results,
        "common_patterns": [
            {"pattern": list(pattern), "count": count}
            for pattern, count in common_patterns[:20]
        ]
    }

    output_path = "/mnt/c/Users/sheel/Desktop/src/analysis/results/bat_pcfg_analysis_112d.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"  Results saved to: {output_path}")

    # Compare with baseline
    compare_with_baseline(results_data)

    print(f"\n{'=' * 70}")
    print("Analysis Complete!")
    print(f"{'=' * 70}")
    print(f"\nKey Findings (112D):")
    print(f"  - Analyzed {sum(r['n_sequences'] for r in results)} sequences across {len(results)} contexts")
    print(f"  - Total vocabulary: {sum(r['vocabulary_size'] for r in results)} unique symbols")
    print(f"  - Total grammar rules learned: {sum(r['n_rules'] for r in results)}")
    if common_patterns:
        print(f"  - Most common pattern: {' -> '.join(str(p) for p in common_patterns[0][0])} ({common_patterns[0][1]} occurrences)")


if __name__ == "__main__":
    main()
