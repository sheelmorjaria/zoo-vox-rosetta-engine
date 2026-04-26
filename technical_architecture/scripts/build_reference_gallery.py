#!/usr/bin/env python3
"""
Build Reference Gallery from Cache Data
========================================

This script builds a reference gallery for zero-shot classification by:
1. Loading NBD segment cache files
2. Extracting 112D features and species/context labels
3. Grouping by species and computing prototype embeddings
4. Saving as a JSON gallery for the Rust zero-shot router

Usage:
    python3 build_reference_gallery.py --output reference_gallery.json

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_cache_data(cache_dirs: list[str]) -> list[dict]:
    """Load all cache data from multiple directories."""
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
            except Exception as e:
                print(f"Warning: Could not load {cache_file}: {e}")

    return all_data


def quantize_features(features: list[float], k: int = 1020) -> int:
    """
    Compute cluster ID from 112D feature vector using hashing quantization.
    This must match the Rust implementation.
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


def siamese_embed(
    features: list[float], weights: list[list[float]], bias: list[float]
) -> list[float]:
    """
    Generate 64D embedding from 112D features.
    Simplified Python implementation of the Rust Siamese network.
    """
    latent_dim = len(bias)
    feature_dim = len(features)

    embedding = []
    for i in range(latent_dim):
        val = bias[i]
        for j in range(min(feature_dim, len(weights[i]))):
            val += weights[i][j] * features[j]
        # ReLU
        embedding.append(max(0.0, val))

    # L2 normalize
    norm = np.sqrt(sum(x * x for x in embedding))
    if norm > 1e-8:
        embedding = [x / norm for x in embedding]

    return embedding


def create_random_weights(
    feature_dim: int = 105, latent_dim: int = 64, seed: int = 42
) -> tuple[list[list[float]], list[float]]:
    """Create random weights for Siamese network (Xavier initialization)."""
    np.random.seed(seed)

    # Xavier initialization
    scale = np.sqrt(2.0 / feature_dim)
    weights = (np.random.randn(latent_dim, feature_dim) * scale).tolist()
    bias = (np.random.randn(latent_dim) * 0.1).tolist()

    return weights, bias


def get_taxon_from_context(context: int) -> str:
    """Map context ID to taxonomic group."""
    context_map = {
        0: "Unknown",
        1: "Food-related",
        2: "Social",
        3: "Territorial",
        4: "Aggression",
        5: "Mating",
        6: "Distress",
        7: "Exploration",
        8: "Sleep",
        9: "Grooming",
        10: "Mother-Infant",
        11: "Territorial",
        12: "Social",
    }
    return context_map.get(context, "Unknown")


def get_taxon_from_features(features: list[float]) -> str:
    """Infer taxonomic group from acoustic features (heuristic)."""
    if len(features) < 10:
        return "Unknown"

    f0 = features[0]
    duration = features[1]

    if f0 > 20000:
        return "Cetacean"
    elif f0 > 8000 and duration < 50:
        return "Mammal"  # Bat-like
    elif f0 > 2000 and f0 < 8000:
        return "Songbird"
    elif duration > 200:
        return "Mysticete"
    else:
        return "Unknown"


def build_reference_gallery(
    cache_data: list[dict],
    min_samples_per_species: int = 5,
    max_samples_per_species: int = 50,
) -> dict[str, Any]:
    """
    Build reference gallery from cache data.

    Groups segments by species/context and creates prototype embeddings.
    """
    # Create random weights for embedding
    weights, bias = create_random_weights(seed=42)

    # Group by source file to get file-level context/emitter
    file_data: dict[str, list[dict]] = defaultdict(list)
    for entry in cache_data:
        src = entry.get("source_file", "")
        if src:
            file_data[src].append(entry)

    # Collect samples by species label
    species_samples: dict[str, list[dict]] = defaultdict(list)

    for src, segments in file_data.items():
        if not segments:
            continue

        # Get file-level metadata
        emitter = segments[0].get("emitter", 0)
        context = segments[0].get("context", 0)

        # Create species label from context and emitter
        # For bat data, we use emitter ID as species proxy
        if emitter != 0:
            species_label = f"bat_emitter_{abs(emitter)}"
        else:
            context_name = get_taxon_from_context(context)
            species_label = f"context_{context_name}"

        # Aggregate features for this file
        for seg in segments:
            features = seg.get("features", [])
            if features and len(features) >= 112:
                species_samples[species_label].append(
                    {
                        "source_file": src,
                        "segment_idx": seg.get("segment_idx", 0),
                        "features": features[:112],  # Ensure 112D
                        "context": context,
                        "emitter": emitter,
                    }
                )

    # Create prototype embeddings for each species
    samples = []
    species_stats = {}

    for species_label, segs in species_samples.items():
        n_samples = len(segs)

        if n_samples < min_samples_per_species:
            continue

        # Limit samples
        segs_to_use = segs[:max_samples_per_species]

        # Compute prototype embedding (average of all samples)
        all_features = [s["features"] for s in segs_to_use]
        prototype_features = np.mean(all_features, axis=0).tolist()

        # Generate embedding
        embedding = siamese_embed(prototype_features, weights, bias)

        # Determine taxon
        taxon = get_taxon_from_features(prototype_features)
        if any(segs_to_use[0]["emitter"] != 0 for _ in [1]):
            taxon = "Mammal"  # Bats

        samples.append(
            {
                "species": species_label,
                "taxon": taxon,
                "embedding": embedding,
                "original_features": prototype_features,
            }
        )

        species_stats[species_label] = {
            "n_samples": n_samples,
            "n_used": len(segs_to_use),
            "taxon": taxon,
        }

    # Build gallery structure
    gallery = {
        "samples": samples,
        "taxon_index": {},
        "embedding_matrix": [],
        "species_labels": [],
        "taxon_labels": [],
        "metadata": {
            "total_samples": len(cache_data),
            "unique_species": len(samples),
            "species_stats": species_stats,
            "embedding_dim": 64,
            "feature_dim": 112,
        },
    }

    # Build indices
    for idx, sample in enumerate(samples):
        taxon = sample["taxon"]
        if taxon not in gallery["taxon_index"]:
            gallery["taxon_index"][taxon] = []
        gallery["taxon_index"][taxon].append(idx)

        gallery["embedding_matrix"].append(sample["embedding"])
        gallery["species_labels"].append(sample["species"])
        gallery["taxon_labels"].append(taxon)

    return gallery, weights, bias


def main():
    parser = argparse.ArgumentParser(description="Build reference gallery from cache data")
    parser.add_argument(
        "--output", "-o", default="reference_gallery.json", help="Output gallery JSON file"
    )
    parser.add_argument(
        "--weights-output", "-w", default="siamese_weights.json", help="Output weights file"
    )
    parser.add_argument("--min-samples", type=int, default=5, help="Minimum samples per species")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum samples per species")
    args = parser.parse_args()

    print("=" * 80)
    print("BUILD REFERENCE GALLERY FROM CACHE DATA")
    print("=" * 80)

    # Cache directories
    cache_dirs = [
        "bat_nbd_cache_parallel",
        "bat_fm_cache",
        "bat_nbd_cache_full",
    ]

    print("\n[1] Loading cache data...")
    cache_data = load_cache_data(cache_dirs)
    print(f"   Total entries loaded: {len(cache_data):,}")

    if not cache_data:
        print("ERROR: No cache data found")
        return

    print("\n[2] Building reference gallery...")
    gallery, weights, bias = build_reference_gallery(
        cache_data,
        min_samples_per_species=args.min_samples,
        max_samples_per_species=args.max_samples,
    )

    print(f"   Species prototypes: {len(gallery['samples'])}")
    print(f"   Embedding dimension: {gallery['metadata']['embedding_dim']}")

    # Show species distribution
    print("\n[SPECIES DISTRIBUTION]")
    stats = gallery["metadata"]["species_stats"]
    sorted_species = sorted(stats.items(), key=lambda x: x[1]["n_samples"], reverse=True)

    for species, info in sorted_species[:20]:
        print(f"   {species}: {info['n_samples']} samples, {info['n_used']} used ({info['taxon']})")

    # Show taxon distribution
    print("\n[TAXON DISTRIBUTION]")
    for taxon, indices in gallery["taxon_index"].items():
        print(f"   {taxon}: {len(indices)} species")

    # Save gallery
    print(f"\n[3] Saving gallery to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(gallery, f, indent=2)

    # Save weights
    weights_data = {
        "weights": weights,
        "bias": bias,
        "normalize": True,
    }
    print(f"[4] Saving weights to: {args.weights_output}")
    with open(args.weights_output, "w") as f:
        json.dump(weights_data, f, indent=2)

    print("\n" + "=" * 80)
    print("REFERENCE GALLERY BUILD COMPLETE")
    print("=" * 80)

    print(f"\nGallery file: {args.output}")
    print(f"Weights file: {args.weights_output}")
    print(f"\nTo use with zero-shot router:")
    print(f"  cargo run --release --bin eval_zero_shot -- manifest.json --gallery {args.output}")


if __name__ == "__main__":
    main()
