#!/usr/bin/env python3
"""
Real Acoustic Persona-Context Association Analysis

This script performs statistical analysis to associate acoustic personas with
real behavioral contexts from field annotations (Annotations.tsv).

Data Sources:
- vocalization_database.json: Phrase-level acoustic features
- Annotations.tsv: 871,045 marmoset vocalizations with behavioral labels

Methodology:
1. Load annotations (Tsik, Trill, Twitter, Phee, Seep, Infant, Vocalization)
2. Map annotations to phrase keys based on acoustic similarity
3. Build persona-context contingency matrix
4. Statistical tests: Chi-square, Fisher's exact, Cramer's V
5. Semantic discovery: Interpret persona meanings based on significant associations

Expected Discoveries:
- GRITTY (fast attack, low HNR) → Tsik (alarm/alert)
- PURE (slow attack, high HNR) → Phee (contact)
- BOUNCY (vibrato) → Trill (courtship)
- TRANSIENT (regular ICI) → Twitter (excitement)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))

from analysis.rosetta_stone.associate_personas_with_context import (
    ACOUSTIC_PERSONAS,
    calculate_cramers_v,
    chi_square_test_of_independence,
    compute_persona_score,
    discover_persona_semantics,
    extract_micro_dynamics_features,
    perform_fisher_exact_tests,
    visualize_persona_context_heatmap,
)


def load_annotations_tsv(tsv_path: str) -> pd.DataFrame:
    """Load marmoset annotations from TSV file."""
    print(f"Loading annotations from {tsv_path}...")

    df = pd.read_csv(tsv_path, sep='\t')

    print(f"✅ Loaded {len(df)} annotations")
    print(f"   Date range: {df['year'].min()}-{df['month'].min()}-{df['day'].min()} "
          f"to {df['year'].max()}-{df['month'].max()}-{df['day'].max()}")

    return df


def get_label_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Get distribution of vocalization labels (behavioral contexts)."""
    print("\n📊 VOCALIZATION LABEL DISTRIBUTION (Behavioral Contexts):")

    label_counts = df['label'].value_counts().to_dict()

    total = sum(label_counts.values())

    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"   {label:<15} {count:>8} ({percentage:>5.1f}%)")

    return label_counts


def map_annotations_to_phrases(
    annotations: pd.DataFrame,
    phrase_segments: Dict,
    db: Dict,
    species: str = 'marmoset'
) -> Dict[str, List[str]]:
    """
    Map annotation labels to phrase keys based on acoustic similarity.

    This is challenging because:
    - Annotations are at the file level (individual recordings)
    - Database phrases are clustered by acoustic features

    Strategy:
    1. For each annotation file, find matching phrase based on:
       - Duration similarity
       - F0 range (if available)
    2. Aggregate label counts per phrase
    """
    print("\n🔗 Mapping annotations to phrase keys...")

    phrases = db['species_data'][species]['phrases']

    # Build phrase lookup by duration and F0
    phrase_index = {}

    for phrase_key, phrase_data in phrases.items():
        af = phrase_data['acoustic_features']

        duration = af.get('mean_duration_ms', 0) / 1000  # Convert to seconds
        af.get('mean_f0_hz', 0)
        af.get('f0_range_hz', 0)

        # Parse phrase key components
        # Format: F0_X_DUR_Y_RANGE_Z
        import re
        match = re.match(r'F0_(\d+)_DUR_(\d+)_RANGE_(\d+)', phrase_key)
        if match:
            key_f0 = int(match.group(1))
            key_dur = int(match.group(2))
            key_range = int(match.group(3))

            phrase_index[phrase_key] = {
                'duration': key_dur / 1000.0,  # Convert ms to seconds
                'f0': key_f0,
                'range': key_range
            }

    # Map annotations to phrases
    phrase_labels = defaultdict(list)

    mapped_count = 0

    for _, row in annotations.iterrows():
        label = row['label']
        duration = row['duration']

        # Find best matching phrase
        best_phrase = None
        best_score = float('inf')

        for phrase_key, phrase_info in phrase_index.items():
            # Score based on duration similarity
            duration_diff = abs(phrase_info['duration'] - duration)

            if duration_diff < 0.2:  # Within 200ms
                if duration_diff < best_score:
                    best_score = duration_diff
                    best_phrase = phrase_key

        if best_phrase:
            phrase_labels[best_phrase].append(label)
            mapped_count += 1

    print(f"✅ Mapped {mapped_count} annotations to {len(phrase_labels)} phrases")

    return phrase_labels


def build_context_matrix_from_annotations(
    phrase_labels: Dict[str, List[str]],
    db: Dict,
    species: str,
    persona_min_score: float = 0.3
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build persona-context matrix from annotation labels.

    Matrix dimensions: personas x contexts
    Cell values: phrase count
    """
    print("\n📊 Building persona-context matrix from annotations...")

    phrases = db['species_data'][species]['phrases']

    # Get unique personas and contexts
    persona_names = list(ACOUSTIC_PERSONAS.keys())
    context_names = sorted(set(label for labels in phrase_labels.values() for label in labels))

    print(f"   Contexts: {context_names}")

    # Initialize matrix
    matrix = np.zeros((len(persona_names), len(context_names)), dtype=int)

    # Assign personas to phrases and fill matrix
    phrase_personas = {}

    for phrase_key, labels in phrase_labels.items():
        if phrase_key not in phrases:
            continue

        phrase_data = phrases[phrase_key]
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

            # Add to matrix
            persona_idx = persona_names.index(best_persona)

            for label in labels:
                if label in context_names:
                    ctx_idx = context_names.index(label)
                    matrix[persona_idx, ctx_idx] += 1

    print(f"   Matrix shape: {len(persona_names)} personas x {len(context_names)} contexts")
    print(f"   Total phrases assigned: {len(phrase_personas)}")
    print(f"   Matrix total: {matrix.sum()} phrase-context pairs")

    return matrix, persona_names, context_names


def main():
    """Main analysis with real annotation data."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Associate acoustic personas with real behavioral contexts'
    )
    parser.add_argument('--db', type=str,
                       default='/home/sheel/birdsong_analysis/src/vocalization_database.json',
                       help='Path to vocalization database')
    parser.add_argument('--annotations', type=str,
                       default='/home/sheel/birdsong_analysis/Annotations.tsv',
                       help='Path to annotations TSV file')
    parser.add_argument('--species', type=str, default='marmoset',
                       help='Species to analyze')
    parser.add_argument('--min-score', type=float, default=0.3,
                       help='Minimum persona score for phrase assignment')
    parser.add_argument('--visualize', action='store_true',
                       help='Create heatmap visualization')
    parser.add_argument('--output', type=str,
                       help='Output path for visualization')

    args = parser.parse_args()

    print("=" * 80)
    print("ACOUSTIC PERSONA-CONTEXT ASSOCIATION (REAL ANNOTATIONS)")
    print("=" * 80)
    print(f"\nSpecies: {args.species}")
    print(f"Minimum persona score: {args.min_score}")

    # Load annotations
    if not Path(args.annotations).exists():
        print(f"\n❌ Annotations file not found: {args.annotations}")
        return

    annotations = load_annotations_tsv(args.annotations)

    # Show label distribution
    get_label_distribution(annotations)

    # Load database
    if not Path(args.db).exists():
        print(f"\n❌ Database not found: {args.db}")
        return

    with open(args.db, 'r') as f:
        db = json.load(f)

    print("\n✅ Loaded vocalization database")

    # Map annotations to phrases
    phrase_labels = map_annotations_to_phrases(annotations, None, db, args.species)

    if not phrase_labels:
        print("\n❌ No phrase-context mappings found")
        return

    # Build persona-context matrix
    matrix, persona_names, context_names = build_context_matrix_from_annotations(
        phrase_labels, db, args.species, args.min_score
    )

    if matrix.sum() == 0:
        print("\n❌ Empty persona-context matrix")
        return

    # Perform chi-square test
    chi2_stat, p_value, residuals = chi_square_test_of_independence(
        matrix, persona_names, context_names
    )

    # Calculate effect size
    cramers_v = calculate_cramers_v(chi2_stat, matrix)

    # Perform Fisher's exact tests
    associations = perform_fisher_exact_tests(matrix, persona_names, context_names)

    # Discover persona semantics
    discover_persona_semantics(
        associations, residuals, min_occurrences=5
    )

    # Visualize
    if args.visualize:
        visualize_persona_context_heatmap(
            matrix, persona_names, context_names, args.output
        )

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)

    print("\n📊 SUMMARY:")
    print(f"   - Analyzed {len(annotations)} annotated vocalizations")
    print(f"   - Mapped to {len(phrase_labels)} phrase types")
    print(f"   - Tested {len(persona_names)} acoustic personas")
    print(f"   - Found {len(context_names)} behavioral contexts")
    print(f"   - Chi-square: {chi2_stat:.2f}, p-value: {p_value:.4e}")
    print(f"   - Effect size (Cramer's V): {cramers_v:.3f}")

    print("\n📚 SCIENTIFIC IMPACT:")
    if p_value < 0.05:
        print("   - ✅ STATISTICALLY SIGNIFICANT associations found")
        print("   - Acoustic personas carry behavioral meaning")
        print("   - Enables interpretation of 'atomic words' in communication")
    else:
        print("   - No significant associations found")
        print("   - May need more data or different persona definitions")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
