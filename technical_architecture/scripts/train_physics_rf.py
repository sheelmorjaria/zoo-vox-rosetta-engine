#!/usr/bin/env python3
"""
Train Random Forest on Physics Features (46D) for Hybrid Expert Ensemble
========================================================================

Extracts 46D physics features (Layer 1) from the 112D cache and trains
a Random Forest classifier for the Hybrid Expert ensemble.

Usage:
    python3 scripts/train_physics_rf.py
"""

import json
import numpy as np
import struct
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

# Feature dimensions
FEATURE_DIM = 112
PHYSICS_DIM = 46  # Layer 1: indices 0-45

def load_bincode_features(filepath):
    """Load features stored in Rust bincode format (Vec<f32>)"""
    with open(filepath, 'rb') as f:
        # Read length as varint (bincode uses varint encoding)
        length = 0
        shift = 0
        while True:
            byte = struct.unpack('B', f.read(1))[0]
            length |= (byte & 0x7F) << shift
            shift += 7
            if byte & 0x80 == 0:
                break
        # Read features
        data = f.read(length * 4)  # 4 bytes per f32
        features = np.frombuffer(data, dtype=np.float32)
        return features.copy()  # Copy to make writable

def main():
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Random Forest Training - Physics Features (46D)                 ║")
    print("║  For Hybrid Expert Ensemble                                       ║")
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

    print(f"  Cached features: {len(cache_manifest['entries'])}")

    # Load all features and labels
    print("\nLoading features from cache...")
    all_features = []
    all_labels = []

    cache_dir = Path("beans_feature_cache_112d")

    for sample in samples:
        audio_file = sample["audio_file"]
        label = sample["labels"]["output"] if sample["labels"]["output"] != "None" else f"task_{sample['labels']['task']}"

        cache_file = cache_manifest["entries"].get(audio_file)
        if cache_file:
            full_path = cache_dir / cache_file
            if full_path.exists():
                try:
                    features = load_bincode_features(full_path)
                    if len(features) == FEATURE_DIM:
                        # Extract only physics features (first 46D)
                        physics = features[:PHYSICS_DIM]
                        all_features.append(physics)
                        all_labels.append(label)
                except Exception as e:
                    pass  # Skip problematic files

    print(f"  Loaded {len(all_features)} samples")

    if len(all_features) == 0:
        raise ValueError("No features loaded!")

    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"  Feature shape: {X.shape}")
    print(f"  Unique labels: {len(np.unique(y))}")

    # Split into train/test (90/10)
    n_samples = len(X)
    n_train = int(n_samples * 0.9)

    # Shuffle
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\nSplit: {len(X_train)} train, {len(X_test)} test")

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest with balanced class weights
    print("\nTraining Random Forest...")
    print("  n_estimators: 200")
    print("  max_depth: 30")
    print("  min_samples_split: 5")
    print("  class_weight: balanced")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    rf.fit(X_train_scaled, y_train)

    # Evaluate
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Evaluation Results                                               ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    y_pred = rf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred) * 100

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Correct: {np.sum(y_pred == y_test)} / {len(y_test)}")

    # Top 20 classes
    print("\nTop 20 Classes by Sample Count:")
    print(f"{'Class':<50} {'Total':>8} {'Correct':>8} {'Accuracy':>8}")
    print("-" * 76)

    unique_labels, counts = np.unique(y_test, return_counts=True)
    label_stats = list(zip(unique_labels, counts))
    label_stats.sort(key=lambda x: -x[1])

    for label, total in label_stats[:20]:
        mask = y_test == label
        correct = np.sum(y_pred[mask] == label)
        acc = correct / total * 100 if total > 0 else 0
        print(f"{label:<50} {total:>8} {correct:>8} {acc:>7.1f}%")

    # Save model
    model_path = "physics_rf_model.joblib"
    print(f"\nSaving model to: {model_path}")
    joblib.dump({
        'model': rf,
        'scaler': scaler,
        'feature_dim': PHYSICS_DIM,
        'accuracy': accuracy,
    }, model_path)

    # Summary
    elapsed = time.time() - start_time
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Summary                                                          ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print(f"║  Architecture:       Random Forest (Physics 46D)                 ║")
    print(f"║  Test Accuracy:      {accuracy:>8.2f}%                                   ║")
    print(f"║  Total Time:         {elapsed:>8.1f}s                                    ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    print("\nHybrid Expert Ensemble Summary:")
    print(f"  - Texture NN (66D):  59.88%")
    print(f"  - Physics RF (46D):  {accuracy:.2f}%")
    print("  - Combined ensemble: (run ensemble evaluation)")

    return accuracy

if __name__ == "__main__":
    main()
