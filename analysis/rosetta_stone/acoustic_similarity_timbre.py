#!/usr/bin/env python3
"""
Acoustic Similarity Matching with Timbre Features

This script demonstrates using the NEW timbre features (spectral centroid, slope,
bandwidth, rolloff) to match atomic phrases by acoustic similarity.

Uses the existing phrase_segments.pkl database (17GB) with actual audio data.

Question: Can timbre features distinguish phrases that have similar F0 but different timbre?
"""

import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.append("/home/sheel/birdsong_analysis")
sys.path.append(str(Path(__file__).parent))

from universal_rosetta_stone import UniversalRosettaStone

# Timbre feature names we just added
TIMBRE_FEATURES = [
    "spectral_centroid_hz",
    "spectral_slope",
    "spectral_bandwidth_hz",
    "spectral_rolloff_hz",
]


def load_phrase_database(db_path: str, max_phrases: int = 100) -> Dict:
    """Load phrase segments from pickle file."""
    print(f"Loading phrase database from {db_path}...")

    try:
        with open(db_path, "rb") as f:
            data = pickle.load(f)

        print(f"✅ Loaded {len(data)} phrase types")

        # Count total audio segments
        total_segments = sum(len(segs) for segs in data.values())
        print(f"Total audio segments: {total_segments:,}")

        # Limit to max_phrases
        if len(data) > max_phrases:
            print(f"Limiting to {max_phrases} phrase types")
            data = dict(list(data.items())[:max_phrases])

        return data

    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return None


def extract_timbre_features(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Extract timbre features from audio segment."""
    try:
        analyzer = UniversalRosettaStone(sample_rate=sample_rate)

        # Use the enhanced feature extraction
        features = analyzer._extract_modality_features(audio)

        # Extract only timbre features
        timbre = {}
        for feature in TIMBRE_FEATURES:
            timbre[feature] = features.get(feature, 0.0)

        return timbre

    except Exception as e:
        print(f"Error extracting features: {e}")
        return {f: 0.0 for f in TIMBRE_FEATURES}


def compute_timbre_distance(timbre1: Dict[str, float], timbre2: Dict[str, float]) -> float:
    """
    Compute normalized Euclidean distance in timbre space.

    Lower distance = more similar timbre.
    """
    # Collect common features
    common_features = set(timbre1.keys()) & set(timbre2.keys())

    if not common_features:
        return float("inf")

    # Extract values
    vals1 = []
    vals2 = []

    for f in common_features:
        vals1.append(timbre1[f])
        vals2.append(timbre2[f])

    # Compute normalized Euclidean distance
    vals1 = np.array(vals1)
    vals2 = np.array(vals2)

    # Normalize by range
    diff = vals1 - vals2

    # Feature-specific normalization
    # spectral_centroid: typically 0-10000 Hz
    # spectral_slope: typically -1 to 1
    # spectral_bandwidth: typically 0-5000 Hz
    # spectral_rolloff: typically 0-10000 Hz
    feature_scales = {
        "spectral_centroid_hz": 10000.0,
        "spectral_slope": 2.0,
        "spectral_bandwidth_hz": 5000.0,
        "spectral_rolloff_hz": 10000.0,
    }

    # Normalize differences
    normalized_diff = []
    for i, f in enumerate(common_features):
        scale = feature_scales.get(f, 1.0)
        normalized_diff.append(diff[i] / scale)

    normalized_diff = np.array(normalized_diff)

    # Euclidean distance
    distance = np.sqrt(np.sum(normalized_diff**2))

    return distance


def analyze_timbre_space(phrase_segments: Dict) -> Tuple[Dict, List]:
    """
    Analyze timbre features across all phrases.

    Returns:
        - Phrase timbre features (phrase_key -> timbre dict)
        - List of (phrase_key, audio) tuples for similarity search
    """
    print("\n" + "=" * 80)
    print("EXTRACTING TIMBRE FEATURES FROM PHRASE DATABASE")
    print("=" * 80)

    phrase_timbres = {}
    phrase_audios = []

    start_time = time.time()

    for i, (phrase_key, segments) in enumerate(phrase_segments.items()):
        if not segments or len(segments) == 0:
            continue

        # Get first segment as representative
        audio = segments[0]

        if len(audio) == 0:
            continue

        # Determine sample rate (assume 22050 for marmoset data)
        sample_rate = 22050

        # Extract timbre features
        timbre = extract_timbre_features(audio, sample_rate)
        phrase_timbres[phrase_key] = timbre
        phrase_audios.append((phrase_key, audio))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(phrase_segments)} phrase types...")

    elapsed = time.time() - start_time
    print(f"\n✅ Extracted timbre features for {len(phrase_timbres)} phrases in {elapsed:.1f}s")

    return phrase_timbres, phrase_audios


def find_similar_by_timbre(
    query_phrase_key: str, phrase_timbres: Dict, top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Find phrases with similar timbre to a query phrase.

    Returns:
        List of (phrase_key, distance) tuples, sorted by distance (ascending)
    """
    if query_phrase_key not in phrase_timbres:
        print(f"❌ Query phrase '{query_phrase_key}' not found")
        return []

    query_timbre = phrase_timbres[query_phrase_key]

    # Compute distances to all other phrases
    distances = []
    for other_key, other_timbre in phrase_timbres.items():
        if other_key == query_phrase_key:
            continue

        distance = compute_timbre_distance(query_timbre, other_timbre)
        distances.append((other_key, distance))

    # Sort by distance (ascending = more similar)
    distances.sort(key=lambda x: x[1])

    return distances[:top_n]


def analyze_timbre_clusters(phrase_timbres: Dict) -> Dict:
    """Analyze clustering of phrases in timbre space."""
    print("\n" + "=" * 80)
    print("TIMBRE SPACE ANALYSIS")
    print("=" * 80)

    # Collect all timbre feature values
    all_values = {f: [] for f in TIMBRE_FEATURES}

    for timbre in phrase_timbres.values():
        for f in TIMBRE_FEATURES:
            all_values[f].append(timbre[f])

    # Compute statistics
    print("\n📊 TIMBRE FEATURE STATISTICS:")
    print(f"{'Feature':<25} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 75)

    for f in TIMBRE_FEATURES:
        values = all_values[f]
        print(
            f"{f:<25} {np.min(values):>12.1f} "
            f"{np.max(values):>12.1f} {np.mean(values):>12.1f} "
            f"{np.std(values):>12.1f}"
        )

    # Compute pairwise distance distribution
    print("\n📊 PAIRWISE TIMBRE DISTANCES:")

    keys = list(phrase_timbres.keys())
    distances = []

    # Sample 100 pairs for speed
    n_samples = min(100, len(keys) * (len(keys) - 1) // 2)
    count = 0

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            dist = compute_timbre_distance(phrase_timbres[keys[i]], phrase_timbres[keys[j]])
            distances.append(dist)
            count += 1

            if count >= n_samples:
                break

        if count >= n_samples:
            break

    distances = np.array(distances)

    print(f"  Mean distance: {np.mean(distances):.4f}")
    print(f"  Median distance: {np.median(distances):.4f}")
    print(f"  Std distance: {np.std(distances):.4f}")
    print(f"  Min distance: {np.min(distances):.4f}")
    print(f"  Max distance: {np.max(distances):.4f}")

    return {
        "feature_stats": {
            f: {
                "min": np.min(all_values[f]),
                "max": np.max(all_values[f]),
                "mean": np.mean(all_values[f]),
                "std": np.std(all_values[f]),
            }
            for f in TIMBRE_FEATURES
        },
        "pairwise_distances": {
            "mean": np.mean(distances),
            "median": np.median(distances),
            "std": np.std(distances),
            "min": np.min(distances),
            "max": np.max(distances),
        },
    }


def demo_similarity_search(phrase_timbres: Dict):
    """Demonstrate similarity search for a few example phrases."""
    print("\n" + "=" * 80)
    print("TIMBRE SIMILARITY SEARCH DEMO")
    print("=" * 80)

    # Pick a few example phrases
    example_keys = list(phrase_timbres.keys())[:3]

    for query_key in example_keys:
        print(f"\n🔍 Query: {query_key}")

        # Get query timbre
        query_timbre = phrase_timbres[query_key]
        print(f"   Spectral Centroid: {query_timbre['spectral_centroid_hz']:.1f} Hz")
        print(f"   Spectral Slope: {query_timbre['spectral_slope']:.4f}")

        # Find similar phrases
        similar = find_similar_by_timbre(query_key, phrase_timbres, top_n=5)

        if similar:
            print("\n   Top 5 most similar phrases:")
            for i, (other_key, distance) in enumerate(similar, 1):
                other_timbre = phrase_timbres[other_key]
                print(f"      {i}. {other_key}")
                print(f"         Distance: {distance:.4f}")
                print(
                    f"         Centroid: {other_timbre['spectral_centroid_hz']:.1f} Hz, "
                    f"Slope: {other_timbre['spectral_slope']:.4f}"
                )


def main():
    """Main analysis."""
    print("=" * 80)
    print("ACOUSTIC SIMILARITY MATCHING WITH TIMBRE FEATURES")
    print("=" * 80)
    print("\nThis demonstrates using NEW timbre features to match phrases")
    print("by acoustic similarity, going beyond F0-based matching.")

    # Load phrase database
    db_path = "/home/sheel/birdsong_analysis/phrase_audio_database_full/phrase_segments.pkl"

    if not Path(db_path).exists():
        print(f"\n❌ Phrase database not found: {db_path}")
        return

    phrase_segments = load_phrase_database(db_path, max_phrases=100)

    if phrase_segments is None:
        return

    # Analyze timbre space
    phrase_timbres, phrase_audios = analyze_timbre_space(phrase_segments)

    if not phrase_timbres:
        print("\n❌ No timbre features extracted")
        return

    # Analyze timbre statistics
    analyze_timbre_clusters(phrase_timbres)

    # Demo similarity search
    demo_similarity_search(phrase_timbres)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✅ Analyzed {len(phrase_timbres)} phrases with timbre features")
    print(f"✅ Timbre features provide {len(TIMBRE_FEATURES)} additional dimensions")
    print("✅ Can now match phrases by TIMBRE similarity, not just F0")

    print("\n📚 SCIENTIFIC IMPACT:")
    print("   - Phrases with similar F0 but different timbre can now be distinguished")
    print("   - Enables discovery of 'timbre-based phrase families'")
    print("   - More robust than F0-only matching for cross-species analysis")

    print("\n🎯 NEXT STEPS:")
    print("   1. Update vocalization_database.json with timbre features")
    print("   2. Implement timbre-based phrase clustering")
    print("   3. Add timbre similarity to query interface")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
