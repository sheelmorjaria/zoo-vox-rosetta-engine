#!/usr/bin/env python3
"""
Convert sklearn RF models to JSON format for Rust evaluation.

This script converts trained Random Forest models from sklearn's joblib format
to a JSON format that can be loaded by the Rust rf_stacking_ensemble module.

Usage:
    python3 scripts/convert_rf_to_json.py

Input files (joblib):
    - physics_rf_model.joblib (46D)
    - full_rf_model.joblib (112D)

Output files (JSON):
    - physics_rf_model.json
    - full_rf_model.json
"""

import json
import struct
from pathlib import Path

import joblib
import numpy as np


def serialize_tree(tree, n_classes: int, feature_dim: int) -> dict:
    """Serialize a sklearn tree to Rust format."""
    nodes = []

    for i in range(tree.node_count):
        if tree.children_left[i] == -1:  # Leaf node
            # Get class prediction from value array
            value = tree.value[i][0]  # Shape: (n_classes,)
            prediction = int(np.argmax(value))
            nodes.append(
                {
                    "feature_idx": None,
                    "threshold": None,
                    "left": None,
                    "right": None,
                    "prediction": prediction,
                    "n_samples": int(tree.n_node_samples[i]),
                }
            )
        else:
            nodes.append(
                {
                    "feature_idx": int(tree.feature[i]),
                    "threshold": float(tree.threshold[i]),
                    "left": int(tree.children_left[i]),
                    "right": int(tree.children_right[i]),
                    "prediction": None,
                    "n_samples": int(tree.n_node_samples[i]),
                }
            )

    return {
        "nodes": nodes,
        "n_classes": n_classes,
        "feature_dim": feature_dim,
    }


def serialize_rf_model(model_data: dict, feature_dim: int, model_name: str) -> dict:
    """Serialize sklearn RF to Rust format."""
    rf = model_data["model"]
    scaler = model_data["scaler"]

    print(f"\n  Serializing {model_name}:")
    print(f"    Trees: {rf.n_estimators}")
    print(f"    Classes: {rf.n_classes_}")
    print(f"    Feature dim: {feature_dim}")

    trees = []
    for i, tree in enumerate(rf.estimators_):
        tree_data = serialize_tree(tree.tree_, rf.n_classes_, feature_dim)
        trees.append(tree_data)

        if (i + 1) % 50 == 0:
            print(f"    Serialized {i + 1}/{rf.n_estimators} trees...")

    # Get class labels
    class_labels = [str(c) for c in rf.classes_]

    result = {
        "trees": trees,
        "n_estimators": rf.n_estimators,
        "max_depth": rf.max_depth if rf.max_depth is not None else 0,
        "n_classes": rf.n_classes_,
        "feature_dim": feature_dim,
        "feature_means": scaler.mean_.tolist(),
        "feature_stds": scaler.scale_.tolist(),
        "class_labels": class_labels,
        "train_accuracy": 0.0,  # Not stored in model
        "val_accuracy": model_data.get("accuracy", 0.0),
    }

    print(f"    Done: {len(trees)} trees, {len(class_labels)} classes")
    return result


def main():
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║     Convert sklearn RF Models to Rust JSON Format                 ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    base_path = Path(__file__).parent.parent

    # Convert Physics RF (46D)
    physics_path = base_path / "physics_rf_model.joblib"
    if physics_path.exists():
        print(f"Loading Physics RF from: {physics_path}")
        physics_data = joblib.load(physics_path)
        physics_json = serialize_rf_model(physics_data, 46, "Physics RF (46D)")

        output_path = base_path / "physics_rf_model.json"
        with open(output_path, "w") as f:
            json.dump(physics_json, f)
        print(f"  Saved to: {output_path}")

        # Print file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
    else:
        print(f"WARNING: Physics RF not found at {physics_path}")

    # Convert Full RF (112D)
    full_path = base_path / "full_rf_model.joblib"
    if full_path.exists():
        print(f"\nLoading Full RF from: {full_path}")
        full_data = joblib.load(full_path)
        full_json = serialize_rf_model(full_data, 112, "Full RF (112D)")

        output_path = base_path / "full_rf_model.json"
        with open(output_path, "w") as f:
            json.dump(full_json, f)
        print(f"  Saved to: {output_path}")

        # Print file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
    else:
        print(f"WARNING: Full RF not found at {full_path}")

    # Convert Taxonomy Gatekeeper RF (76D)
    gatekeeper_path = base_path / "taxonomy_gatekeeper_rf.joblib"
    if gatekeeper_path.exists():
        print(f"\nLoading Taxonomy Gatekeeper RF from: {gatekeeper_path}")
        gatekeeper_data = joblib.load(gatekeeper_path)
        gatekeeper_json = serialize_rf_model(gatekeeper_data, 76, "Taxonomy Gatekeeper RF (76D)")

        output_path = base_path / "taxonomy_gatekeeper_rf.json"
        with open(output_path, "w") as f:
            json.dump(gatekeeper_json, f)
        print(f"  Saved to: {output_path}")

        # Print file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
    else:
        print(f"INFO: Taxonomy Gatekeeper RF not found at {gatekeeper_path}")

    print("\n╔═══════════════════════════════════════════════════════════════════╗")
    print("║  Conversion Complete                                              ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
