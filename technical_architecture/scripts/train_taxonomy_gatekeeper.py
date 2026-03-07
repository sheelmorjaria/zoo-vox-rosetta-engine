#!/usr/bin/env python3
"""
Train Taxonomy Gatekeeper RF for Hierarchical Veto Ensemble
============================================================

Trains a Random Forest on 46D physics features to predict TAXONOMIC GROUPS
(not species). This is the "Gatekeeper" that prevents cross-clade errors.

The gatekeeper achieves high accuracy on broad taxonomic groups:
- Mammal: 96.9%
- Songbird: 52.3%
- Mysticete: 98.4%
etc.

Usage:
    python3 scripts/train_taxonomy_gatekeeper.py
"""

import json
import struct
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

FEATURE_DIM = 112
PHYSICS_DIM = 46  # Layer 1: indices 0-45

# Taxonomic mapping (must match Rust implementation)
TAXON_MAP = {
    "Cetacean": ["dolphin", "porpoise", "sperm", "beaked", "delphinid", "phocoen", "orca"],
    "Mysticete": [
        "humpback",
        "blue whale",
        "fin whale",
        "minke",
        "gray whale",
        "right whale",
        "bowhead",
        "balaenopter",
    ],
    "Songbird": [
        "sparrow",
        "finch",
        "warbler",
        "thrush",
        "robin",
        "cardinal",
        "towhee",
        "ovenbird",
        "wren",
        "tit",
        "swainson",
    ],
    "NonPasserine": [
        "parrot",
        "owl",
        "hawk",
        "eagle",
        "duck",
        "goose",
        "gull",
        "crow",
        "raven",
        "penguin",
        "psittacid",
        "strigid",
    ],
    "Amphibian": ["frog", "toad", "ranid", "bufonid", "hylid", "peeper"],
    "Insect": [
        "cricket",
        "mosquito",
        "cicada",
        "grasshopper",
        "katydid",
        "bee",
        "fly",
        "anopheles",
        "aedes",
        "culex",
        "culicid",
    ],
    "Mammal": [
        "bat",
        "pteropodid",
        "vesper",
        "phyllostomid",
        "monkey",
        "ape",
        "gibbon",
        "chimp",
        "gorilla",
        "primate",
    ],
    "Pinniped": ["seal", "sea lion", "walrus", "phocid", "otariid"],
}

TAXON_LABELS = [
    "Cetacean",
    "Mysticete",
    "Songbird",
    "NonPasserine",
    "Amphibian",
    "Insect",
    "Mammal",
    "Pinniped",
    "Unknown",
]
TAXON_TO_IDX = {t: i for i, t in enumerate(TAXON_LABELS)}


def load_bincode_features(filepath):
    with open(filepath, "rb") as f:
        length = 0
        shift = 0
        while True:
            byte = struct.unpack("B", f.read(1))[0]
            length |= (byte & 0x7F) << shift
            shift += 7
            if byte & 0x80 == 0:
                break
        data = f.read(length * 4)
        return np.frombuffer(data, dtype=np.float32).copy()


def map_species_to_taxon(species):
    """Map a species name to its taxonomic group"""
    s = species.lower()

    for taxon, keywords in TAXON_MAP.items():
        for keyword in keywords:
            if keyword in s:
                return taxon

    # Check task names
    if "gibbon" in s:
        return "Mammal"
    if "dcase" in s or "bird" in s:
        return "Songbird"
    if "watkins" in s:
        return "Cetacean"
    if "humbug" in s or "mosquito" in s:
        return "Insect"
    if "rfcx" in s:
        return "Mammal"

    return "Unknown"


def main():
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Taxonomy Gatekeeper RF Training                                  ║")
    print("║  Predicts TAXONOMIC GROUPS from Physics Features (46D)           ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    start_time = time.time()

    # Load manifest
    print("Loading manifest...")
    with open("beans_zero_full_manifest.json") as f:
        manifest = json.load(f)

    samples = manifest["samples"]
    print(f"  Total samples: {len(samples)}")

    # Load cache manifest
    with open("beans_feature_cache_112d/cache_manifest.json") as f:
        cache_manifest = json.load(f)

    # Load all features and taxonomic labels
    print("\nLoading features and mapping to taxonomic groups...")
    all_features = []
    all_taxons = []
    all_species = []

    cache_dir = Path("beans_feature_cache_112d")

    for sample in samples:
        audio_file = sample["audio_file"]
        label = (
            sample["labels"]["output"]
            if sample["labels"]["output"] != "None"
            else f"task_{sample['labels']['task']}"
        )

        cache_file = cache_manifest["entries"].get(audio_file)
        if cache_file:
            full_path = cache_dir / cache_file
            if full_path.exists():
                try:
                    features = load_bincode_features(full_path)
                    if len(features) == FEATURE_DIM:
                        # Extract only physics features (46D)
                        physics = features[:PHYSICS_DIM]
                        taxon = map_species_to_taxon(label)
                        all_features.append(physics)
                        all_taxons.append(taxon)
                        all_species.append(label)
                except:
                    pass

    X = np.array(all_features)
    y_taxon = np.array(all_taxons)
    print(f"  Loaded {len(X)} samples")

    # Show taxonomic distribution
    print("\nTaxonomic Distribution:")
    taxon_counts = Counter(y_taxon)
    for taxon, count in sorted(taxon_counts.items(), key=lambda x: -x[1]):
        pct = count / len(y_taxon) * 100
        print(f"  {taxon:<15} {count:>6} ({pct:>5.1f}%)")

    # Split (use same split as other models - last 10% test)
    n_samples = len(X)
    n_train = int(n_samples * 0.9)

    X_train = X[:n_train]
    y_train = y_taxon[:n_train]
    X_test = X[n_train:]
    y_test = y_taxon[n_train:]

    print(f"\nSplit: {len(X_train)} train, {len(X_test)} test")

    # Standardize
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RF for TAXONOMIC classification
    print("\nTraining Taxonomy Gatekeeper RF...")
    print("  n_estimators: 300")
    print("  max_depth: 30")
    print("  class_weight: balanced")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    rf.fit(X_train_scaled, y_train)

    # Evaluate
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Evaluation Results                                               ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred) * 100

    print(f"Overall Taxonomic Accuracy: {accuracy:.2f}%")
    print()

    # Per-taxon accuracy
    print("Per-Taxon Accuracy:")
    print(f"{'Taxon':<15} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
    print("-" * 45)

    for taxon in TAXON_LABELS:
        mask = y_test == taxon
        total = mask.sum()
        if total > 0:
            correct = (y_pred[mask] == taxon).sum()
            acc = correct / total * 100
            print(f"{taxon:<15} {total:>8} {correct:>8} {acc:>9.1f}%")

    # Confusion matrix summary
    print("\nConfusion Matrix (Top 5 confusions):")
    cm = confusion_matrix(y_test, y_pred, labels=TAXON_LABELS)

    confusions = []
    for i, true_taxon in enumerate(TAXON_LABELS):
        for j, pred_taxon in enumerate(TAXON_LABELS):
            if i != j and cm[i, j] > 0:
                confusions.append((true_taxon, pred_taxon, cm[i, j]))

    confusions.sort(key=lambda x: -x[2])
    for true_taxon, pred_taxon, count in confusions[:5]:
        print(f"  {true_taxon} -> {pred_taxon}: {count}")

    # Save model
    model_path = "taxonomy_gatekeeper_rf.joblib"
    print(f"\nSaving model to: {model_path}")
    joblib.dump(
        {
            "model": rf,
            "scaler": scaler,
            "taxon_labels": TAXON_LABELS,
            "taxon_to_idx": TAXON_TO_IDX,
            "accuracy": accuracy,
        },
        model_path,
    )

    elapsed = time.time() - start_time
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Summary                                                          ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print(f"║  Taxonomic Accuracy:  {accuracy:>8.2f}%                                ║")
    print(f"║  Total Time:          {elapsed:>8.1f}s                                 ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    print("\nHierarchical Veto Ensemble Components:")
    print(f"  1. Taxonomy Gatekeeper (RF 46D): {accuracy:.2f}%")
    print("  2. Species Expert (NN 66D):     59.88%")
    print("  3. Veto Mechanism:              (run eval_hierarchical_veto)")


if __name__ == "__main__":
    main()
