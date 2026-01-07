#!/usr/bin/env python3
"""
Acoustic Similarity for Atomic Phrase Candidates

This script implements "Acoustic Personas" for discovering the smallest units
of meaning (atomic words) in animal vocalizations using micro-dynamics features.

Feature Categories:
1. Grit Factors: Harmonic-to-Noise Ratio (HNR), Spectral Flatness
2. Motion Factors: Attack Time, Decay Time, Vibrato, Jitter
3. Fingerprint Factors: MFCCs (1-4), Spectral Contrast
4. Rhythm Factors: Inter-Click Interval, Onset Rate

Acoustic Personas:
- GRITTY: Low HNR + High Flatness + Fast Attack (Aggressive, Alert)
- PURE: High HNR + Low Flatness + Slow Attack (Contact, Affiliative)
- BOUNCY: High Vibrato + Pulsed Amplitude (Courtship, Play)
- SHARP: Fast Attack + Fast Decay + High Spectral Contrast (Alarm, Startle)
- SUSTAINED: Slow Attack + Slow Decay + High Sustain (Territory, Long-range)
- TRANSIENT: High Onset Rate + Low ICI CV (Rhythmic, Mechanical)
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).parent.parent.parent))


@dataclass
class AcousticPersona:
    """Definition of an acoustic persona (semantic category)."""
    name: str
    description: str
    feature_weights: Dict[str, float]  # Feature -> target direction (+1 or -1)
    feature_ranges: Dict[str, Tuple[float, float]]  # Feature -> (min, max)


# Define acoustic personas based on feature combinations
ACOUSTIC_PERSONAS = {
    'gritty': AcousticPersona(
        name='GRITTY',
        description='Aggressive, alert calls with noisy texture and sharp onset',
        feature_weights={
            'harmonic_to_noise_ratio': -1.0,  # Low HNR = noisy
            'spectral_flatness': 1.0,         # High flatness = noise-like
            'attack_time_ms': -1.0,           # Fast attack
            'spectral_contrast': -1.0,         # Low contrast = diffuse
        },
        feature_ranges={
            'harmonic_to_noise_ratio': (0.0, 5.0),
            'spectral_flatness': (0.3, 1.0),
            'attack_time_ms': (0.0, 20.0),
            'spectral_contrast': (0.0, 5.0),
        }
    ),

    'pure': AcousticPersona(
        name='PURE',
        description='Clean tonal calls for contact and affiliation',
        feature_weights={
            'harmonic_to_noise_ratio': 1.0,   # High HNR = tonal
            'spectral_flatness': -1.0,        # Low flatness = tonal
            'attack_time_ms': 1.0,            # Slow attack = gentle
            'decay_time_ms': 1.0,             # Slow decay = smooth
        },
        feature_ranges={
            'harmonic_to_noise_ratio': (10.0, 100.0),
            'spectral_flatness': (0.0, 0.2),
            'attack_time_ms': (20.0, 100.0),
            'decay_time_ms': (50.0, 200.0),
        }
    ),

    'bouncy': AcousticPersona(
        name='BOUNCY',
        description='Playful, courtship calls with vibrato and amplitude modulation',
        feature_weights={
            'vibrato_rate_hz': 1.0,           # Strong vibrato
            'vibrato_depth': 1.0,             # Deep modulation
            'jitter': -1.0,                   # Low jitter = periodic
            'attack_time_ms': -1.0,           # Fast onset
        },
        feature_ranges={
            'vibrato_rate_hz': (5.0, 15.0),
            'vibrato_depth': (0.5, 2.0),
            'jitter': (0.0, 0.1),
            'attack_time_ms': (5.0, 30.0),
        }
    ),

    'sharp': AcousticPersona(
        name='SHARP',
        description='Alarm and startle calls with sharp onset and offset',
        feature_weights={
            'attack_time_ms': -1.0,           # Very fast attack
            'decay_time_ms': -1.0,            # Fast decay
            'spectral_contrast': 1.0,         # High contrast = formant structure
            'spectral_flatness': -1.0,        # Tonal
        },
        feature_ranges={
            'attack_time_ms': (0.0, 10.0),
            'decay_time_ms': (0.0, 30.0),
            'spectral_contrast': (10.0, 30.0),
            'spectral_flatness': (0.0, 0.3),
        }
    ),

    'sustained': AcousticPersona(
        name='SUSTAINED',
        description='Long-range territorial or contact calls with steady amplitude',
        feature_weights={
            'attack_time_ms': 1.0,            # Slow attack
            'decay_time_ms': 1.0,             # Slow decay
            'sustain_level': 1.0,             # High sustain
            'jitter': -1.0,                   # Stable pitch
        },
        feature_ranges={
            'attack_time_ms': (50.0, 200.0),
            'decay_time_ms': (100.0, 500.0),
            'sustain_level': (0.5, 1.0),
            'jitter': (0.0, 0.05),
        }
    ),

    'transient': AcousticPersona(
        name='TRANSIENT',
        description='Rhythmic mechanical sounds (clicks, pops) with regular timing',
        feature_weights={
            'onset_rate_hz': 1.0,             # High click rate
            'ici_coefficient_of_variation': -1.0,  # Low CV = regular
            'median_ici_ms': -1.0,            # Short intervals
            'harmonic_to_noise_ratio': -1.0,  # Low HNR = aperiodic
        },
        feature_ranges={
            'onset_rate_hz': (10.0, 100.0),
            'ici_coefficient_of_variation': (0.0, 0.3),
            'median_ici_ms': (5.0, 50.0),
            'harmonic_to_noise_ratio': (0.0, 3.0),
        }
    ),
}


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


def extract_micro_dynamics_features(
    phrase_data: Dict,
    species: str
) -> Dict[str, float]:
    """Extract micro-dynamics features from phrase data."""
    af = phrase_data['acoustic_features']

    # Extract all available micro-dynamics features
    features = {
        # Grit factors
        'harmonic_to_noise_ratio': af.get('harmonic_to_noise_ratio', 0.0),
        'spectral_flatness': af.get('spectral_flatness', 0.0),

        # Motion factors
        'attack_time_ms': af.get('attack_time_ms', 0.0),
        'decay_time_ms': af.get('decay_time_ms', 0.0),
        'sustain_level': af.get('sustain_level', 0.0),
        'vibrato_rate_hz': af.get('vibrato_rate_hz', 0.0),
        'vibrato_depth': af.get('vibrato_depth', 0.0),
        'jitter': af.get('jitter', 0.0),

        # Fingerprint factors
        'mfcc_1': af.get('mfcc_1', 0.0),
        'mfcc_2': af.get('mfcc_2', 0.0),
        'mfcc_3': af.get('mfcc_3', 0.0),
        'mfcc_4': af.get('mfcc_4', 0.0),
        'mfcc_delta_mean': af.get('mfcc_delta_mean', 0.0),
        'spectral_contrast': af.get('spectral_contrast', 0.0),

        # Rhythm factors
        'median_ici_ms': af.get('median_ici_ms', 0.0),
        'onset_rate_hz': af.get('onset_rate_hz', 0.0),
        'ici_coefficient_of_variation': af.get('ici_coefficient_of_variation', 0.0),
    }

    return features


def compute_persona_score(
    features: Dict[str, float],
    persona: AcousticPersona
) -> float:
    """
    Compute how well a phrase matches a persona.

    Returns a score between 0 (no match) and 1 (perfect match).
    """
    total_weight = 0.0
    score = 0.0

    for feature_name, direction in persona.feature_weights.items():
        if feature_name not in features:
            continue

        value = features[feature_name]

        # Skip if feature is zero (not extracted)
        if value == 0.0:
            continue

        # Check if value is in range
        min_val, max_val = persona.feature_ranges.get(feature_name, (0.0, 1.0))

        if min_val <= value <= max_val:
            # Value is in target range - compute normalized position
            if direction > 0:
                # Higher is better - normalize from min to max
                normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 1.0
            else:
                # Lower is better - normalize from max to min
                normalized = (max_val - value) / (max_val - min_val) if max_val > min_val else 1.0

            score += normalized
            total_weight += 1.0

    if total_weight == 0:
        return 0.0

    return score / total_weight


def find_atomic_phrases_by_persona(
    db: Dict,
    persona_name: str,
    species: Optional[str] = None,
    top_n: int = 20,
    min_score: float = 0.3
) -> List[Tuple[str, Dict[str, float], float]]:
    """
    Find phrases that match an acoustic persona.

    Returns:
        List of (phrase_key, features, score) tuples sorted by score
    """
    if persona_name not in ACOUSTIC_PERSONAS:
        print(f"❌ Unknown persona: {persona_name}")
        return []

    persona = ACOUSTIC_PERSONAS[persona_name]

    print(f"\n🔍 Searching for {persona.name} phrases...")
    print(f"   Description: {persona.description}")

    candidates = []

    # Search through species
    species_to_search = [species] if species else db['species_data'].keys()

    for species_name in species_to_search:
        if species_name not in db['species_data']:
            continue

        phrases = db['species_data'][species_name]['phrases']

        for phrase_key, phrase_data in phrases.items():
            # Extract micro-dynamics features
            features = extract_micro_dynamics_features(phrase_data, species_name)

            # Compute persona score
            score = compute_persona_score(features, persona)

            if score >= min_score:
                candidates.append((phrase_key, features, score))

    # Sort by score (descending)
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Return top N
    return candidates[:top_n]


def compute_multi_feature_similarity(
    features1: Dict[str, float],
    features2: Dict[str, float],
    feature_weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute similarity between two phrases in multi-dimensional feature space.

    Uses normalized Euclidean distance with optional feature weighting.
    """
    # Get common features (non-zero in both)
    common_features = []
    for f in features1.keys():
        if features1[f] > 0 and features2.get(f, 0) > 0:
            common_features.append(f)

    if not common_features:
        return 0.0

    # Normalize features
    # Define typical ranges for normalization
    feature_ranges = {
        'harmonic_to_noise_ratio': 100.0,
        'spectral_flatness': 1.0,
        'attack_time_ms': 200.0,
        'decay_time_ms': 500.0,
        'sustain_level': 1.0,
        'vibrato_rate_hz': 20.0,
        'vibrato_depth': 3.0,
        'jitter': 0.5,
        'mfcc_1': 100.0,
        'mfcc_2': 50.0,
        'mfcc_3': 30.0,
        'mfcc_4': 20.0,
        'mfcc_delta_mean': 10.0,
        'spectral_contrast': 50.0,
        'median_ici_ms': 200.0,
        'onset_rate_hz': 100.0,
        'ici_coefficient_of_variation': 1.0,
    }

    # Compute normalized distance
    diff_sum = 0.0
    weight_sum = 0.0

    for f in common_features:
        val1 = features1[f]
        val2 = features2[f]

        # Normalize difference
        range_val = feature_ranges.get(f, 1.0)
        diff = abs(val1 - val2) / range_val

        # Apply weight if provided
        weight = feature_weights.get(f, 1.0) if feature_weights else 1.0

        diff_sum += diff * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.0

    # Convert distance to similarity (1 - normalized_distance)
    avg_diff = diff_sum / weight_sum
    similarity = max(0.0, 1.0 - avg_diff)

    return similarity


def find_similar_phrases_multi_dimensional(
    db: Dict,
    query_phrase_key: str,
    species: str,
    top_n: int = 10,
    feature_weights: Optional[Dict[str, float]] = None
) -> List[Tuple[str, float]]:
    """
    Find phrases similar to a query phrase using multi-dimensional features.

    Goes beyond F0 matching to find "acoustic siblings" - phrases that sound
    similar even if they have different F0 values.
    """
    # Get query phrase
    if species not in db['species_data']:
        print(f"❌ Species not found: {species}")
        return []

    phrases = db['species_data'][species]['phrases']

    if query_phrase_key not in phrases:
        print(f"❌ Phrase not found: {query_phrase_key}")
        return []

    query_data = phrases[query_phrase_key]
    query_features = extract_micro_dynamics_features(query_data, species)

    print(f"\n🔍 Finding phrases similar to {query_phrase_key}...")
    print("   Query features:")
    for f, v in query_features.items():
        if v > 0:
            print(f"      {f}: {v:.2f}")

    # Compute similarities
    similarities = []

    for other_key, other_data in phrases.items():
        if other_key == query_phrase_key:
            continue

        other_features = extract_micro_dynamics_features(other_data, species)

        similarity = compute_multi_feature_similarity(
            query_features,
            other_features,
            feature_weights
        )

        if similarity > 0:
            similarities.append((other_key, similarity))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]


def analyze_persona_distribution(db: Dict, species: Optional[str] = None):
    """Analyze distribution of phrases across acoustic personas."""
    print("\n" + "=" * 80)
    print("ACOUSTIC PERSONA DISTRIBUTION")
    print("=" * 80)

    species_to_analyze = [species] if species else db['species_data'].keys()

    for species_name in species_to_analyze:
        if species_name not in db['species_data']:
            continue

        phrases = db['species_data'][species_name]['phrases']

        print(f"\n📊 {species_name.upper()}:")
        print(f"   Total phrases: {len(phrases)}")

        for persona_name, persona in ACOUSTIC_PERSONAS.items():
            matches = 0
            total_score = 0.0

            for phrase_key, phrase_data in phrases.items():
                features = extract_micro_dynamics_features(phrase_data, species_name)
                score = compute_persona_score(features, persona)

                if score > 0.3:
                    matches += 1
                    total_score += score

            if matches > 0:
                avg_score = total_score / matches
                print(f"   {persona.name}: {matches} phrases (avg score: {avg_score:.2f})")


def analyze_feature_coverage(db: Dict):
    """Analyze which micro-dynamics features are available in the database."""
    print("\n" + "=" * 80)
    print("MICRO-DYNAMICS FEATURE COVERAGE")
    print("=" * 80)

    all_features = [
        'harmonic_to_noise_ratio', 'spectral_flatness',
        'attack_time_ms', 'decay_time_ms', 'sustain_level',
        'vibrato_rate_hz', 'vibrato_depth', 'jitter',
        'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4',
        'mfcc_delta_mean', 'spectral_contrast',
        'median_ici_ms', 'onset_rate_hz', 'ici_coefficient_of_variation'
    ]

    for species_name, species_data in db['species_data'].items():
        phrases = species_data['phrases']

        print(f"\n📊 {species_name.upper()} ({len(phrases)} phrases):")

        feature_counts = {f: 0 for f in all_features}

        for phrase_data in phrases.values():
            af = phrase_data['acoustic_features']
            for feature in all_features:
                if af.get(feature, 0.0) > 0:
                    feature_counts[feature] += 1

        for feature in all_features:
            count = feature_counts[feature]
            percentage = (count / len(phrases)) * 100 if len(phrases) > 0 else 0
            if count > 0:
                print(f"   {feature:<30} {count:>5} ({percentage:>5.1f}%)")


def demo_persona_search(db: Dict):
    """Demonstrate persona-based search for atomic phrase discovery."""
    print("\n" + "=" * 80)
    print("ATOMIC PHRASE DISCOVERY: ACOUSTIC PERSONA SEARCH")
    print("=" * 80)

    # Demo 1: Find GRITTY phrases (aggressive alerts)
    print("\n" + "-" * 80)
    print("DEMO 1: Finding 'GRITTY' phrases (aggressive alerts)")
    print("-" * 80)

    gritty_candidates = find_atomic_phrases_by_persona(
        db, 'gritty', species='marmoset', top_n=10, min_score=0.4
    )

    if gritty_candidates:
        print(f"\n✅ Found {len(gritty_candidates)} 'GRITTY' phrases:")
        for i, (phrase_key, features, score) in enumerate(gritty_candidates[:5], 1):
            print(f"\n   {i}. {phrase_key} (score: {score:.3f})")
            print(f"      HNR: {features['harmonic_to_noise_ratio']:.2f}, "
                  f"Flatness: {features['spectral_flatness']:.3f}, "
                  f"Attack: {features['attack_time_ms']:.1f}ms")
    else:
        print("\n⚠️  No 'GRITTY' phrases found (database may not have micro-dynamics features)")

    # Demo 2: Find PURE phrases (contact calls)
    print("\n" + "-" * 80)
    print("DEMO 2: Finding 'PURE' phrases (contact calls)")
    print("-" * 80)

    pure_candidates = find_atomic_phrases_by_persona(
        db, 'pure', species='marmoset', top_n=10, min_score=0.4
    )

    if pure_candidates:
        print(f"\n✅ Found {len(pure_candidates)} 'PURE' phrases:")
        for i, (phrase_key, features, score) in enumerate(pure_candidates[:5], 1):
            print(f"\n   {i}. {phrase_key} (score: {score:.3f})")
            print(f"      HNR: {features['harmonic_to_noise_ratio']:.2f}, "
                  f"Flatness: {features['spectral_flatness']:.3f}, "
                  f"Attack: {features['attack_time_ms']:.1f}ms")

    # Demo 3: Multi-dimensional similarity search
    print("\n" + "-" * 80)
    print("DEMO 3: Multi-dimensional similarity search")
    print("-" * 80)

    # Pick a query phrase (first available)
    if 'marmoset' in db['species_data']:
        marmoset_phrases = db['species_data']['marmoset']['phrases']
        if marmoset_phrases:
            query_key = list(marmoset_phrases.keys())[0]
            similar = find_similar_phrases_multi_dimensional(
                db, query_key, 'marmoset', top_n=5
            )

            if similar:
                print(f"\n✅ Found {len(similar)} phrases similar to {query_key}:")
                for i, (other_key, similarity) in enumerate(similar, 1):
                    print(f"   {i}. {other_key} (similarity: {similarity:.3f})")


def print_usage_examples():
    """Print usage examples for the script."""
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)

    print("""
# Find 'GRITTY' phrases (aggressive alerts)
python acoustic_similarity_for_atomic_phrase_candidates.py --persona gritty --species marmoset

# Find 'PURE' phrases (contact calls)
python acoustic_similarity_for_atomic_phrase_candidates.py --persona pure --species marmoset

# Find phrases similar to a specific phrase
python acoustic_similarity_for_atomic_phrase_candidates.py --query F0_7000_DUR_5_RANGE_100 --species marmoset

# Analyze persona distribution
python acoustic_similarity_for_atomic_phrase_candidates.py --analyze-distribution

# Check feature coverage
python acoustic_similarity_for_atomic_phrase_candidates.py --analyze-coverage
    """)


def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Acoustic similarity for atomic phrase discovery'
    )
    parser.add_argument('--db', type=str,
                       default='/home/sheel/birdsong_analysis/src/vocalization_database.json',
                       help='Path to vocalization database')
    parser.add_argument('--persona', type=str,
                       choices=list(ACOUSTIC_PERSONAS.keys()),
                       help='Acoustic persona to search for')
    parser.add_argument('--species', type=str,
                       help='Species to filter by')
    parser.add_argument('--query', type=str,
                       help='Query phrase key for similarity search')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of results to return')
    parser.add_argument('--min-score', type=float, default=0.3,
                       help='Minimum persona score')
    parser.add_argument('--analyze-distribution', action='store_true',
                       help='Analyze persona distribution')
    parser.add_argument('--analyze-coverage', action='store_true',
                       help='Analyze feature coverage')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo analysis')

    args = parser.parse_args()

    # Load database
    if not Path(args.db).exists():
        print(f"❌ Database not found: {args.db}")
        return

    db = load_vocalization_database(args.db)

    # Run requested analysis
    if args.analyze_coverage:
        analyze_feature_coverage(db)

    if args.analyze_distribution:
        analyze_persona_distribution(db, args.species)

    if args.persona:
        candidates = find_atomic_phrases_by_persona(
            db, args.persona, args.species, args.top_n, args.min_score
        )

        print(f"\n✅ Found {len(candidates)} phrases matching '{args.persona}' persona:")
        for i, (phrase_key, features, score) in enumerate(candidates, 1):
            print(f"\n{i}. {phrase_key} (score: {score:.3f})")
            print("   Features:")
            for f, v in features.items():
                if v > 0:
                    print(f"      {f}: {v:.3f}")

    if args.query and args.species:
        similar = find_similar_phrases_multi_dimensional(
            db, args.query, args.species, args.top_n
        )

        print(f"\n✅ Found {len(similar)} phrases similar to {args.query}:")
        for i, (other_key, similarity) in enumerate(similar, 1):
            print(f"   {i}. {other_key} (similarity: {similarity:.3f})")

    if args.demo:
        demo_persona_search(db)

    # If no specific action, show usage
    if not (args.persona or args.query or args.analyze_distribution or
            args.analyze_coverage or args.demo):
        print_usage_examples()
        analyze_feature_coverage(db)


if __name__ == "__main__":
    main()
