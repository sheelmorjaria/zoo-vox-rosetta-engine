#!/usr/bin/env python3
"""
Train Random Forest on Full 112D Features for Ensemble
=======================================================

Trains RF on all 112D features to complement the Texture NN in the ensemble.

Usage:
    python3 scripts/train_full_rf.py
"""

import json
import numpy as np
import struct
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import time

FEATURE_DIM = 112

def load_bincode_features(filepath):
    """Load features stored in Rust bincode format"""
    with open(filepath, 'rb') as f:
        length = 0
        shift = 0
        while True:
            byte = struct.unpack('B', f.read(1))[0]
            length |= (byte & 0x7F) << shift
            shift += 7
            if byte & 0x80 == 0:
                break
        data = f.read(length * 4)
        return np.frombuffer(data, dtype=np.float32).copy()

def main():
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Random Forest Training - Full 112D Features                     ║")
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

    # Load all features
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
                        all_features.append(features)
                        all_labels.append(label)
                except:
                    pass

    X = np.array(all_features)
    y = np.array(all_labels)
    print(f"  Loaded {len(X)} samples, {len(np.unique(y))} classes")

    # Use same split as NN (last 10% as test)
    n_samples = len(X)
    n_train = int(n_samples * 0.9)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    
    print(f"\nSplit: {len(X_train)} train, {len(X_test)} test")

    # Standardize
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train RF
    print("\nTraining Random Forest on 112D...")
    print("  n_estimators: 300")
    print("  max_depth: 40")
    print("  class_weight: balanced")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=40,
        min_samples_split=3,
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

    # Save model
    model_path = "full_rf_model.joblib"
    print(f"\nSaving model to: {model_path}")
    joblib.dump({
        'model': rf,
        'scaler': scaler,
        'accuracy': accuracy,
    }, model_path)

    elapsed = time.time() - start_time
    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Summary                                                          ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print(f"║  Test Accuracy:      {accuracy:>8.2f}%                                   ║")
    print(f"║  Total Time:         {elapsed:>8.1f}s                                    ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    print("\nEnsemble Components:")
    print(f"  - Texture NN (66D + masking): 59.88%")
    print(f"  - Full RF (112D):              {accuracy:.2f}%")

if __name__ == "__main__":
    main()
