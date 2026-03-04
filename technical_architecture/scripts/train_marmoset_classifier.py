#!/usr/bin/env python3
"""
Train Rosetta-Net for Marmoset Calls using Transfer Learning

This demonstrates the species-specific training strategy:
1. Load pre-trained features (105D)
2. Train classifier on marmoset call types
3. Analyze learned representations

Usage:
    python train_marmoset_classifier.py
"""

import json
import struct
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def load_features(manifest_path: Path, cache_dir: Path):
    """Load features from cache"""
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load feature cache
    cache_path = cache_dir / "feature_cache_eval/all_features.bin"
    with open(cache_path, "rb") as f:
        _magic = struct.unpack("<I", f.read(4))[0]  # noqa: F841
        n_samples = struct.unpack("<I", f.read(4))[0]
        feature_dim = struct.unpack("<I", f.read(4))[0]

        features = []
        for _ in range(n_samples):
            vec = struct.unpack(f"<{feature_dim}f", f.read(feature_dim * 4))
            features.append(vec)
        features = np.array(features)

    labels = [s["labels"]["label_id"] for s in manifest["samples"]]
    call_types = [s["labels"]["call_type"] for s in manifest["samples"]]

    return features, np.array(labels), call_types, manifest


def main():
    print("=" * 70)
    print("ROSETTA-NET SPECIES-SPECIFIC TRAINING: MARMOSET CALLS")
    print("=" * 70)

    # Load data
    print("\nLoading features...")
    train_X, train_y, train_types, train_manifest = load_features(
        Path("marmoset_train_cache/marmoset_train_manifest.json"), Path("marmoset_train_cache")
    )
    val_X, val_y, val_types, val_manifest = load_features(
        Path("marmoset_val_cache/marmoset_val_manifest.json"), Path("marmoset_val_cache")
    )

    print(f"  Train: {len(train_X)} samples")
    print(f"  Val:   {len(val_X)} samples")
    print(f"  Features: {train_X.shape[1]}D")

    # Get call type names
    label_map = train_manifest["label_map"]
    idx_to_type = {v: k for k, v in label_map.items()}

    print(f"\nCall types: {list(label_map.keys())}")

    # Normalize features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)

    # Replace NaN/Inf with 0
    train_X_scaled = np.nan_to_num(train_X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    val_X_scaled = np.nan_to_num(val_X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Train classifier
    print("\n" + "-" * 70)
    print("TRAINING CLASSIFIER")
    print("-" * 70)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.neural_network import MLPClassifier

    # MLP (similar to Rosetta-Net architecture)
    print("\n1. MLP Classifier (Rosetta-Net style):")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=64,
        learning_rate="adaptive",
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False,
        random_state=42,
    )

    mlp.fit(train_X_scaled, train_y)

    train_acc = mlp.score(train_X_scaled, train_y)
    val_acc = mlp.score(val_X_scaled, val_y)

    print(f"  Train accuracy: {train_acc:.1%}")
    print(f"  Val accuracy:   {val_acc:.1%}")

    # Predictions
    val_pred = mlp.predict(val_X_scaled)

    print("\n  Classification Report:")
    print(
        classification_report(
            val_y, val_pred, target_names=[idx_to_type[i] for i in range(len(idx_to_type))]
        )
    )

    # Confusion matrix
    print("\n  Confusion Matrix:")
    cm = confusion_matrix(val_y, val_pred)
    print("     ", "  ".join(f"{idx_to_type[i][:6]:>6}" for i in range(len(idx_to_type))))
    for i, row in enumerate(cm):
        print(f"{idx_to_type[i][:6]:>6}", " ".join(f"{v:>6}" for v in row))

    # Random Forest for comparison
    print("\n2. Random Forest Classifier:")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1
    )
    rf.fit(train_X_scaled, train_y)

    rf_train_acc = rf.score(train_X_scaled, train_y)
    rf_val_acc = rf.score(val_X_scaled, val_y)

    print(f"  Train accuracy: {rf_train_acc:.1%}")
    print(f"  Val accuracy:   {rf_val_acc:.1%}")

    # Feature importance
    print("\n  Top 15 Important Features:")
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    for i, idx in enumerate(top_idx):
        print(f"    Feature {idx:3d}: {importances[idx]:.4f}")

    # Latent space analysis
    print("\n" + "=" * 70)
    print("LATENT SPACE ANALYSIS")
    print("=" * 70)

    import matplotlib
    from sklearn.decomposition import PCA

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Get MLP embeddings (last hidden layer)
    # For sklearn MLP, we can get intermediate activations
    def get_hidden_features(mlp, X):
        """Get activations from last hidden layer"""
        activations = X
        for i, (coef, intercept) in enumerate(zip(mlp.coefs_[:-1], mlp.intercepts_[:-1])):
            activations = np.maximum(0, activations @ coef + intercept)  # ReLU
        return activations

    hidden_features = get_hidden_features(mlp, val_X_scaled)
    print(f"\nHidden features shape: {hidden_features.shape}")

    # PCA visualization
    pca = PCA(n_components=2)
    hidden_2d = pca.fit_transform(hidden_features)

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"  PC1: {pca.explained_variance_ratio_[0] * 100:.1f}%")
    print(f"  PC2: {pca.explained_variance_ratio_[1] * 100:.1f}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: PCA of hidden features
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(label_map)))

    for i, call_type in enumerate(sorted(label_map.keys())):
        mask = np.array(val_types) == call_type
        ax1.scatter(
            hidden_2d[mask, 0], hidden_2d[mask, 1], c=[colors[i]], label=call_type, alpha=0.6, s=30
        )

    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax1.set_title("Marmoset Call Types in Learned Latent Space")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Confusion matrix
    ax2 = axes[1]
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks(range(len(label_map)))
    ax2.set_yticks(range(len(label_map)))
    ax2.set_xticklabels(
        [idx_to_type[i][:6] for i in range(len(label_map))], rotation=45, ha="right"
    )
    ax2.set_yticklabels([idx_to_type[i][:6] for i in range(len(label_map))])
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Confusion Matrix")

    # Add numbers to confusion matrix
    for i in range(len(label_map)):
        for j in range(len(label_map)):
            ax2.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=8,
            )

    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.savefig("marmoset_latent_space.png", dpi=150, bbox_inches="tight")
    print("\nSaved visualization to marmoset_latent_space.png")

    # Analysis
    print("\n" + "=" * 70)
    print("LINGUISTIC INSIGHTS")
    print("=" * 70)

    # Per-class accuracy
    print("\nPer-Call-Type Analysis:")
    print(f"{'Call Type':<15} {'Samples':<10} {'Accuracy':<10} {'Precision':<12} {'Nature'}")
    print("-" * 70)

    for i, call_type in enumerate(sorted(label_map.keys())):
        mask = val_y == i
        class_acc = (val_pred[mask] == i).mean() if mask.sum() > 0 else 0

        # Precision
        pred_mask = val_pred == i
        precision = (val_y[pred_mask] == i).mean() if pred_mask.sum() > 0 else 0

        # Determine nature
        if precision > 0.8:
            nature = "Fixed Signal"
        elif precision > 0.6:
            nature = "Semi-graded"
        else:
            nature = "Graded"

        print(
            f"{call_type:<15} {mask.sum():<10} {class_acc:.1%}      {precision:.1%}        {nature}"
        )

    # Cluster analysis
    print("\nCluster Separation Analysis:")
    from sklearn.metrics import silhouette_score

    silhouette = silhouette_score(hidden_features, val_y)
    print(f"  Silhouette score: {silhouette:.3f}")
    print("    > 0.5 = Well separated clusters")
    print("    0.25-0.5 = Moderate separation")
    print("    < 0.25 = Poor separation (graded/continuous)")

    if silhouette > 0.5:
        print("\n  ✓ Marmoset vocabulary appears DISCRETE (separate clusters)")
    elif silhouette > 0.25:
        print("\n  ~ Marmoset vocabulary appears STRATIFIED (mixed)")
    else:
        print("\n  ✗ Marmoset vocabulary appears GRADED (continuous manifold)")

    # Save model
    import joblib

    joblib.dump(
        {"mlp": mlp, "rf": rf, "scaler": scaler, "label_map": label_map},
        "marmoset_classifier.joblib",
    )
    print("\nSaved model to marmoset_classifier.joblib")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  Training Data:    {len(train_X)} samples
  Validation Data:  {len(val_X)} samples
  Call Types:       {len(label_map)} classes

  MLP Accuracy:     {val_acc:.1%}
  RF Accuracy:      {rf_val_acc:.1%}

  Silhouette Score: {silhouette:.3f}

  The classifier successfully learned to distinguish
  Marmoset call types using the 105D acoustic features.
""")


if __name__ == "__main__":
    main()
