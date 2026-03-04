#!/usr/bin/env python3
"""
Train Gatekeeper RF (76D) with Consolidated Taxonomic Classes
=============================================================

This script is called by train_gatekeeper_rf_76d.rs to train the Random Forest
using sklearn, then serializes the model back to Rust using numpy format.
"""

import numpy as np
import json
import sys
from sklearn.ensemble import RandomForestClassifier

def main():
    # Load training data from Rust (saved in numpy format)
    train_x = np.load('gatekeeper_76d_train_x.npy')
    train_y = np.load('gatekeeper_76d_train_y.npy')
    val_x = np.load('gatekeeper_76d_val_x.npy')
    val_y = np.load('gatekeeper_76d_val_y.npy')

    # Load metadata
    with open('gatekeeper_76d_meta.json', 'r') as f:
        meta = json.load(f)

    print(f"Training shape: {train_x.shape}")
    print(f"Validation shape: {val_x.shape}")

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=3,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    rf.fit(train_x, train_y)

    # Evaluate
    train_acc = rf.score(train_x, train_y)
    val_acc = rf.score(val_x, val_y)

    print(f"\nTraining accuracy: {train_acc * 100:.2f}%")
    print(f"Validation accuracy: {val_acc * 100:.2f}%")

    # Serialize to Rust format (JSON for simplicity)
    rf_data = serialize_rf_to_rust(rf, meta)
    with open('gatekeeper_rf_76d.json', 'w') as f:
        json.dump(rf_data, f, indent=2)

    print("Model serialized to gatekeeper_rf_76d.json")
    return 0

def serialize_rf_to_rust(rf, meta):
    """Serialize sklearn RF to Rust format"""
    trees = []
    for tree in rf.estimators_:
        tree_data = serialize_tree_to_rust(tree.tree_)
        trees.append(tree_data)

    return {
        'trees': trees,
        'n_estimators': rf.n_estimators,
        'max_depth': rf.max_depth,
        'min_samples_split': rf.min_samples_split,
        'n_classes': int(rf.n_classes_),
        'feature_means': meta['feature_means'],
        'feature_stds': meta['feature_stds'],
        'class_labels': meta['class_labels'],
        'train_accuracy': train_acc * 100.0 if 'train_acc' in dir() else 0.0,
        'val_accuracy': val_acc * 100.0 if 'val_acc' in dir() else 0.0,
    }

def serialize_tree_to_rust(tree):
    """Serialize sklearn tree to Rust format"""
    nodes = []
    for i in range(tree.node_count):
        if tree.children_left[i] == -1:  # Leaf node
            nodes.append({
                'feature_idx': None,
                'threshold': None,
                'left': None,
                'right': None,
                'prediction': int(np.argmax(tree.value[i])),
                'n_samples': int(tree.n_node_samples[i]),
            })
        else:
            nodes.append({
                'feature_idx': int(tree.feature[i]),
                'threshold': float(tree.threshold[i]),
                'left': int(tree.children_left[i]),
                'right': int(tree.children_right[i]),
                'prediction': None,
                'n_samples': int(tree.n_node_samples[i]),
            })

    return {
        'nodes': nodes,
        'n_classes': int(tree.n_classes[0]) if len(tree.n_classes) > 0 else 1,
        'feature_dim': int(tree.n_features),
    }

if __name__ == '__main__':
    sys.exit(main())
