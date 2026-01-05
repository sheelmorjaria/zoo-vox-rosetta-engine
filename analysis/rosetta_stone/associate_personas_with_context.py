#!/usr/bin/env python3
"""
Associate Acoustic Personas with Behavioral Contexts

This script uses statistical methods to discover the semantic meaning of acoustic
personas by analyzing their relationship with annotated behavioral contexts.

Methods:
1. Context Enrichment Analysis: Aggregate contexts for phrases matching each persona
2. Chi-Square Test of Independence: Test if personas are independent of contexts
3. Effect Size (Cramer's V): Measure strength of association
4. Semantic Discovery: Interpret persona meanings based on significant context associations

Scientific Impact:
- Discover that "GRITTY" = "Alarm/Threat"
- Discover that "PURE" = "Contact/Affiliation"
- Discover that "BOUNCY" = "Courtship/Play"
- Validate that acoustic personas carry semantic meaning
"""

import json
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
from scipy.stats import chi2_contingency, fisher_exact
import itertools

sys.path.append(str(Path(__file__).parent.parent.parent))

# Import acoustic persona definitions
from analysis.rosetta_stone.acoustic_similarity_for_atomic_phrase_candidates import (
    ACOUSTIC_PERSONAS,
    extract_micro_dynamics_features,
    compute_persona_score,
    find_atomic_phrases_by_persona
)


@dataclass
class PersonaContextAssociation:
    """Results of persona-context association analysis."""
    persona_name: str
    context_name: str
    phrase_count: int
    total_phrases: int
    expected_count: float
    observed_ratio: float
    expected_ratio: float
    enrichment_score: float  # log2(observed/expected)
    p_value: float
    significant: bool


def load_vocalization_database(db_path: str) -> Dict:
    """Load vocalization database from JSON."""
    print(f"Loading database from {db_path}...")

    with open(db_path, 'r') as f:
        db = json.load(f)

    total_phrases = sum(
        len(species_data['phrases'])
        for species_data in db['species_data'].values()
    )

    print(f"✅ Loaded {total_phrases} phrases across {len(db['species_data'])} species")

    return db


def get_all_contexts(db: Dict, species: Optional[str] = None) -> List[str]:
    """Get all unique context names in the database."""
    contexts = set()

    species_to_analyze = [species] if species else db['species_data'].keys()

    for species_name in species_to_analyze:
        if species_name not in db['species_data']:
            continue

        phrases = db['species_data'][species_name]['phrases']

        for phrase_data in phrases.values():
            for ctx in phrase_data.get('contexts', []):
                contexts.add(ctx['context_name'])

    return sorted(list(contexts))


def build_persona_context_matrix(
    db: Dict,
    species: str,
    persona_min_score: float = 0.3
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build a contingency matrix of personas vs contexts.

    Returns:
        - Matrix: personas x contexts with phrase counts
        - Persona names list
        - Context names list
    """
    print(f"\n📊 Building persona-context matrix for {species}...")

    phrases = db['species_data'][species]['phrases']

    # Get all personas and contexts
    persona_names = list(ACOUSTIC_PERSONAS.keys())
    context_names = get_all_contexts(db, species)

    # Initialize matrix
    matrix = np.zeros((len(persona_names), len(context_names)), dtype=int)

    # Track phrase-level persona assignments
    phrase_personas = {}  # phrase_key -> best matching persona

    for phrase_key, phrase_data in phrases.items():
        features = extract_micro_dynamics_features(phrase_data, species)

        # Find best matching persona
        best_persona = None
        best_score = 0

        for persona_name, persona in ACOUSTIC_PERSONAS.items():
            score = compute_persona_score(features, persona)
            if score > best_score and score >= persona_min_score:
                best_score = score
                best_persona = persona_name

        if best_persona:
            phrase_personas[phrase_key] = best_persona

    # Fill matrix with context counts
    for phrase_key, persona_name in phrase_personas.items():
        phrase_data = phrases[phrase_key]

        persona_idx = persona_names.index(persona_name)

        for ctx in phrase_data.get('contexts', []):
            ctx_name = ctx['context_name']
            if ctx_name in context_names:
                ctx_idx = context_names.index(ctx_name)
                matrix[persona_idx, ctx_idx] += ctx['count']

    # Print matrix summary
    print(f"   Matrix shape: {len(persona_names)} personas x {len(context_names)} contexts")
    print(f"   Total phrases matched: {len(phrase_personas)}")
    print(f"   Matrix total: {matrix.sum()} phrase-context pairs")

    return matrix, persona_names, context_names


def chi_square_test_of_independence(
    matrix: np.ndarray,
    persona_names: List[str],
    context_names: List[str]
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """
    Perform chi-square test of independence for personas vs contexts.

    Returns:
        - chi2_statistic: Chi-square test statistic
        - p_value: P-value for the test
        - residuals: Dictionary of (persona, context) -> standardized residual
    """
    print("\n📊 Performing chi-square test of independence...")

    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(matrix)

    print(f"   Chi-square statistic: {chi2:.2f}")
    print(f"   Degrees of freedom: {dof}")
    print(f"   P-value: {p_value:.4e}")

    if p_value < 0.001:
        print("   ✅ STRONGLY SIGNIFICANT: Personas and contexts are NOT independent!")
        print("      Acoustic personas carry semantic meaning related to behavioral contexts.")
    elif p_value < 0.05:
        print("   ✅ SIGNIFICANT: Personas and contexts are associated (p < 0.05)")
    else:
        print("   ⚠️  NOT SIGNIFICANT: Cannot reject independence hypothesis")

    # Calculate standardized residuals for each cell
    # Residual > 2 = significant positive association
    # Residual < -2 = significant negative association
    residuals = {}
    for i, persona in enumerate(persona_names):
        residuals[persona] = {}
        for j, ctx in enumerate(context_names):
            if expected[i, j] > 0:
                residual = (matrix[i, j] - expected[i, j]) / np.sqrt(expected[i, j])
                residuals[persona][ctx] = residual

    return chi2, p_value, residuals


def calculate_enrichment_scores(
    matrix: np.ndarray,
    persona_names: List[str],
    context_names: List[str]
) -> List[PersonaContextAssociation]:
    """
    Calculate enrichment scores for persona-context associations.

    Enrichment = log2(observed/expected)
    - Positive enrichment: context is over-represented for this persona
    - Negative enrichment: context is under-represented for this persona
    """
    print("\n📊 Calculating enrichment scores...")

    # Calculate row and column sums
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    total = matrix.sum()

    associations = []

    for i, persona in enumerate(persona_names):
        for j, ctx in enumerate(context_names):
            observed = matrix[i, j]

            # Calculate expected count under independence
            if row_sums[i] > 0 and col_sums[j] > 0:
                expected = (row_sums[i] * col_sums[j]) / total
            else:
                expected = 0

            # Calculate enrichment score (log2 fold change)
            if observed > 0 and expected > 0:
                enrichment = np.log2(observed / expected)
            elif observed == 0 and expected > 0:
                enrichment = -np.inf  # Complete depletion
            else:
                enrichment = 0.0

            # Calculate ratios
            observed_ratio = observed / row_sums[i] if row_sums[i] > 0 else 0
            expected_ratio = col_sums[j] / total if total > 0 else 0

            associations.append(PersonaContextAssociation(
                persona_name=persona,
                context_name=ctx,
                phrase_count=int(observed),
                total_phrases=int(row_sums[i]),
                expected_count=expected,
                observed_ratio=observed_ratio,
                expected_ratio=expected_ratio,
                enrichment_score=enrichment,
                p_value=0.0,  # Will be calculated separately
                significant=False  # Will be determined separately
            ))

    return associations


def perform_fisher_exact_tests(
    matrix: np.ndarray,
    persona_names: List[str],
    context_names: List[str]
) -> List[PersonaContextAssociation]:
    """
    Perform Fisher's exact test for each persona-context pair.

    Tests if a specific context is enriched for a specific persona.
    """
    print("\n📊 Performing Fisher's exact tests for persona-context pairs...")

    associations = calculate_enrichment_scores(matrix, persona_names, context_names)

    # Calculate row and column sums
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    total = matrix.sum()

    tested_count = 0
    significant_count = 0

    for association in associations:
        i = persona_names.index(association.persona_name)
        j = context_names.index(association.context_name)

        observed = matrix[i, j]

        # Skip if observed count is 0
        if observed == 0:
            continue

        # Create 2x2 contingency table:
        #                    Context    Other_Contexts
        # Persona           observed    row_sum - observed
        # Other_Personas    col_sum - observed    total - row_sum - col_sum + observed

        a = observed  # Persona & Context
        b = row_sums[i] - observed  # Persona & Other Contexts
        c = col_sums[j] - observed  # Other Personas & Context
        d = total - row_sums[i] - col_sums[j] + observed  # Other Personas & Other Contexts

        # Perform Fisher's exact test
        try:
            _, p_value = fisher_exact([[a, b], [c, d]], alternative='greater')
            association.p_value = p_value

            # Bonferroni correction for multiple testing
            num_tests = len(persona_names) * len(context_names)
            alpha_corrected = 0.05 / num_tests
            association.significant = p_value < alpha_corrected

            tested_count += 1
            if association.significant:
                significant_count += 1

        except Exception as e:
            # Fisher's exact test can fail on very large tables
            association.p_value = 1.0
            association.significant = False

    print(f"   Tested {tested_count} persona-context pairs")
    print(f"   Significant associations: {significant_count} (Bonferroni-corrected p < 0.05)")

    return associations


def discover_persona_semantics(
    associations: List[PersonaContextAssociation],
    residuals: Dict[str, Dict[str, float]],
    min_occurrences: int = 5
) -> Dict[str, Dict]:
    """
    Discover the semantic meaning of each acoustic persona based on
    their statistically significant context associations.
    """
    print("\n" + "=" * 80)
    print("SEMANTIC DISCOVERY: INTERPRETING ACOUSTIC PERSONAS")
    print("=" * 80)

    persona_semantics = {}

    for persona_name in ACOUSTIC_PERSONAS.keys():
        # Get all associations for this persona
        persona_associations = [
            a for a in associations
            if a.persona_name == persona_name and a.phrase_count >= min_occurrences
        ]

        # Sort by enrichment score
        persona_associations.sort(key=lambda x: x.enrichment_score, reverse=True)

        # Get top enriched contexts
        top_contexts = persona_associations[:5]

        # Get significant associations
        significant_contexts = [
            a for a in persona_associations
            if a.significant
        ]

        # Get top residuals
        persona_residuals = residuals.get(persona_name, {})
        top_residual_contexts = sorted(
            persona_residuals.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        persona_semantics[persona_name] = {
            'top_contexts': top_contexts,
            'significant_contexts': significant_contexts,
            'top_residuals': top_residual_contexts,
            'total_phrases': sum(a.phrase_count for a in persona_associations),
            'unique_contexts': len(set(a.context_name for a in persona_associations))
        }

        # Print interpretation
        persona = ACOUSTIC_PERSONAS[persona_name]
        print(f"\n🎭 {persona_name} ({persona.description})")
        print(f"   Total phrases: {persona_semantics[persona_name]['total_phrases']}")
        print(f"   Unique contexts: {persona_semantics[persona_name]['unique_contexts']}")

        if top_contexts:
            print(f"\n   📊 TOP ENRICHED CONTEXTS:")
            for i, assoc in enumerate(top_contexts, 1):
                significance = "✅ SIGNIFICANT" if assoc.significant else ""
                print(f"      {i}. {assoc.context_name}: {assoc.phrase_count} phrases "
                      f"({assoc.observed_ratio*100:.1f}%) "
                      f"[enrichment: {assoc.enrichment_score:+.2f}] {significance}")

        if top_residual_contexts:
            print(f"\n   📊 STRONGEST ASSOCIATIONS (standardized residuals):")
            for ctx, residual in top_residual_contexts:
                strength = "STRONG" if abs(residual) > 2 else "moderate" if abs(residual) > 1 else "weak"
                direction = "POSITIVELY" if residual > 0 else "NEGATIVELY"
                print(f"      {ctx}: {residual:+.2f} ({direction} associated, {strength})")

        # Generate semantic interpretation
        if significant_contexts:
            top_sig = significant_contexts[0]
            print(f"\n   💡 SEMANTIC INTERPRETATION:")
            print(f"      '{persona_name}' phrases are significantly associated with '{top_sig.context_name}' contexts")
            print(f"      (p < 0.05, Bonferroni-corrected, {top_sig.phrase_count} phrases, {top_sig.observed_ratio*100:.1f}%)")
        elif top_contexts:
            top_ctx = top_contexts[0]
            print(f"\n   💡 SEMANTIC INTERPRETATION:")
            print(f"      '{persona_name}' phrases are most enriched in '{top_ctx.context_name}' contexts")
            print(f"      ({top_ctx.phrase_count} phrases, {top_ctx.observed_ratio*100:.1f}%, enrichment: {top_ctx.enrichment_score:+.2f})")
        else:
            print(f"\n   ⚠️  No clear context associations found")

    return persona_semantics


def calculate_cramers_v(chi2_statistic: float, matrix: np.ndarray) -> float:
    """
    Calculate Cramer's V effect size for the association.

    Interpretation:
    - 0.0 - 0.1: Negligible association
    - 0.1 - 0.3: Weak association
    - 0.3 - 0.5: Moderate association
    - 0.5+: Strong association
    """
    n = matrix.sum()
    min_dim = min(matrix.shape) - 1

    if min_dim == 0:
        return 0.0

    cramers_v = np.sqrt(chi2_statistic / (n * min_dim))

    return cramers_v


def visualize_persona_context_heatmap(
    matrix: np.ndarray,
    persona_names: List[str],
    context_names: List[str],
    output_path: Optional[str] = None
):
    """
    Create a heatmap visualization of persona-context associations.

    Note: Requires matplotlib. If not available, prints ASCII heatmap.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Normalize by row (persona) to show proportions
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized_matrix = matrix / row_sums

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create heatmap
        sns.heatmap(
            normalized_matrix,
            xticklabels=context_names,
            yticklabels=persona_names,
            cmap='YlOrRd',
            annot=matrix,
            fmt='d',
            cbar_kws={'label': 'Proportion of Phrases'},
            ax=ax
        )

        ax.set_xlabel('Behavioral Context')
        ax.set_ylabel('Acoustic Persona')
        ax.set_title('Acoustic Persona vs Behavioral Context Association')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 Saved heatmap to {output_path}")
        else:
            plt.show()

    except ImportError:
        # Print ASCII heatmap
        print("\n📊 PERSONA-CONTEXT HEATMAP (Phrase Counts):")
        print("=" * 80)

        # Print header
        header = f"{'Persona':<15}"
        for ctx in context_names[:8]:  # Limit to 8 contexts for readability
            header += f"{ctx[:10]:>12}"
        print(header)
        print("-" * 80)

        # Print rows
        for i, persona in enumerate(persona_names):
            row = f"{persona:<15}"
            for j in range(min(8, len(context_names))):
                count = matrix[i, j]
                if count > 0:
                    row += f"{count:>12}"
                else:
                    row += f"{'-':>12}"
            print(row)


def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Associate acoustic personas with behavioral contexts'
    )
    parser.add_argument('--db', type=str,
                       default='/home/sheel/birdsong_analysis/src/vocalization_database.json',
                       help='Path to vocalization database')
    parser.add_argument('--species', type=str, default='marmoset',
                       help='Species to analyze')
    parser.add_argument('--min-score', type=float, default=0.3,
                       help='Minimum persona score for phrase assignment')
    parser.add_argument('--min-occurrences', type=int, default=5,
                       help='Minimum phrase count for context association')
    parser.add_argument('--visualize', action='store_true',
                       help='Create heatmap visualization')
    parser.add_argument('--output', type=str,
                       help='Output path for visualization')

    args = parser.parse_args()

    print("=" * 80)
    print("ASSOCIATING ACOUSTIC PERSONAS WITH BEHAVIORAL CONTEXTS")
    print("=" * 80)
    print(f"\nSpecies: {args.species}")
    print(f"Minimum persona score: {args.min_score}")
    print(f"Minimum phrase occurrences: {args.min_occurrences}")

    # Load database
    if not Path(args.db).exists():
        print(f"\n❌ Database not found: {args.db}")
        return

    db = load_vocalization_database(args.db)

    # Build persona-context matrix
    matrix, persona_names, context_names = build_persona_context_matrix(
        db, args.species, args.min_score
    )

    if matrix.sum() == 0:
        print("\n❌ No persona-context associations found")
        print("   Possible reasons:")
        print("   - Micro-dynamics features not extracted")
        print("   - No contexts in database")
        print("   - Minimum score too high")
        return

    # Perform chi-square test of independence
    chi2_stat, p_value, residuals = chi_square_test_of_independence(
        matrix, persona_names, context_names
    )

    # Calculate effect size
    cramers_v = calculate_cramers_v(chi2_stat, matrix)

    print(f"\n📊 EFFECT SIZE (Cramer's V): {cramers_v:.3f}")
    if cramers_v < 0.1:
        print("   Negligible association")
    elif cramers_v < 0.3:
        print("   Weak association")
    elif cramers_v < 0.5:
        print("   Moderate association")
    else:
        print("   Strong association")

    # Perform Fisher's exact tests
    associations = perform_fisher_exact_tests(matrix, persona_names, context_names)

    # Discover persona semantics
    persona_semantics = discover_persona_semantics(
        associations, residuals, args.min_occurrences
    )

    # Visualize
    if args.visualize:
        visualize_persona_context_heatmap(
            matrix, persona_names, context_names, args.output
        )

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n📚 SCIENTIFIC IMPACT:")
    print(f"   - Discovered semantic meanings for acoustic personas")
    print(f"   - Validated that acoustic structure carries behavioral meaning")
    print(f"   - Established statistical significance of persona-context associations")
    print(f"   - Enabled interpretation of 'atomic words' in animal communication")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
