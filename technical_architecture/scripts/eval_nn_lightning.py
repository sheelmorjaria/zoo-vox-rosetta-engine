#!/usr/bin/env python3
"""
Evaluate Neural Network (112D GPU) using PyTorch Lightning
============================================================

This script evaluates the improved GPU-trained model on the BEANS-Zero test set (10% holdout).
It Features:
    - Balanced class weights
    - Label smoothing
    - Learning rate scheduling
    - Early stopping
    - GPU acceleration (CUDA)

Usage:
    python3 scripts/eval_nn_lightning.py
"""

import json
import numpy as np
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Configuration
BATCH_SIZE = 256
MAX_EPOCHS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05
DROPOUT = 0.3
LABEL_SMOOTHING = 0.1

# Feature dimension
FEATURE_DIM = 112


class RosettaNetPL(nn.Module):
    def __init__(self, n_classes: int, class_weights: torch.Tensor):
        super().__init__()
        self.n_classes = n_classes

        # Architecture: 112 -> 1024 -> 512 -> 256 -> 128 -> output
        self.fc1 = nn.Linear(FEATURE_DIM, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, n_classes)

        self.dropout = nn.Dropout(DROPOUT)
        self.class_weights = class_weights

        self.save_hyperparameters(
            {
                "n_classes": n_classes,
                "class_weights_shape": class_weights.shape,
            }
        )

    def forward(self, x):
        # Block 1: Linear -> BN -> GELU -> Dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Block 2: Linear -> BN -> GELU -> Dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Block 3: Linear -> BN -> GELU -> Dropout
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Block 4: Linear -> BN -> GELU -> Dropout
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Output layer
        return self.out(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # Weighted cross-entropy loss with label smoothing
        loss = F.cross_entropy(
            logits, y,
            weight=self.class_weights[y],
            label_smoothing=LABEL_SMOOTHING,
        )

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # Weighted cross-entropy loss with label smoothing
        loss = F.cross_entropy(
            logits, y,
            weight=self.class_weights[y],
            label_smoothing=LABEL_SMOOTHING,
        )

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        return optimizer


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


def load_data():
    """Load features and labels from cache"""
    print("Loading data from cache...")

    # Load manifest
    with open("beans_zero_full_manifest.json") as f:
        manifest = json.load(f)

    samples = manifest["samples"]
    print(f"  Total samples: {len(samples)}")

    # Load cache manifest
    with open("beans_feature_cache_112d/cache_manifest.json") as f:
        cache_manifest = json.load(f)

    print(f"  Cached features available: {len(cache_manifest['entries'])}")

    # Load all features and    all_features = []
    all_labels = []

    for sample in samples:
        audio_file = sample["audio_file"]
        label = sample["labels"]["output"] if sample["labels"]["output"] != "None" else f"task_{sample['labels']['task']}"

        cache_file = cache_manifest["entries"].get(audio_file)
        if cache_file:
            full_path = Path("beans_feature_cache_112d") / cache_file
            if full_path.exists():
                features = load_bincode_features(full_path)
                if features.shape[0] == FEATURE_DIM:
                    all_features.append(features)
                    all_labels.append(label)

    print(f"  Loaded {len(all_features)} samples")

    # Build label mapping
    unique_labels = sorted(set(all_labels))
    n_classes = len(unique_labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"  Classes: {n_classes}")

    # Compute class weights (balanced)
    class_counts = {}
    for label in all_labels:
        class_counts[label] = class_counts.get(label, 0) + 1

    total_samples = len(all_labels)
    class_weights = {}
    for label, count in class_counts.items():
        if count == 0:
            class_weights[label] = 1.0
        else:
            # Balanced: total_samples / (n_classes * count)
            weight = min(total_samples / (n_classes * count), 100.0)
            class_weights[label] = weight

    # Convert to tensor
    weights_tensor = torch.tensor(
        [class_weights[label_to_idx[label]] for label in unique_labels],
        dtype=torch.float32,
    )

    print(f"  Class weights: min={min(class_weights.values()):.2f}, max={max(class_weights.values()):.2f}")

    # Split into train/validation (90/10)
    n_train = int(len(all_features) * 0.9)
    print(f"  Train samples: {n_train}")
    print(f"  Val samples: {len(all_features) - n_train}")

    # Shuffle and split
    indices = np.random.permutation(len(all_features))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Compute normalization params from training set
    all_features_array = np.array(all_features)
    train_features = all_features_array[train_indices]
    feature_means = train_features.mean(axis=0)
    feature_stds = train_features.std(axis=0)

    # Normalize all features
    normalized_features = (all_features_array - feature_means) / feature_stds

    # Convert labels to indices
    label_indices = np.array([label_to_idx[label] for label in all_labels])

    # Split into train/val
    train_features = normalized_features[train_indices]
    train_labels = label_indices[train_indices]
    val_features = normalized_features[val_indices]
    val_labels = label_indices[val_indices]

    # Create tensors
    train_x = torch.tensor(train_features, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.long)
    val_x = torch.tensor(val_features, dtype=torch.float32)
    val_y = torch.tensor(val_labels, dtype=torch.long)

    print(f"  Train tensor shape: {train_x.shape}")
    print(f"  Val tensor shape: {val_x.shape}")

    # Create datasets
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    return (
        train_loader,
        val_loader,
        n_classes,
        weights_tensor,
        label_to_idx,
        feature_means,
        feature_stds,
    )


def evaluate_model():
    """Load and evaluate the trained PyTorch Lightning model"""
    print("=" * 80)
    print("GPU Neural Network Evaluation (112D Features)")
    print("=" * 80)
    print()

    # Load the saved model
    checkpoint = torch.load("rosetta_net_112d_gpu.pt")

    model_state_dict = checkpoint["model_state_dict"]
    label_to_idx = checkpoint["label_to_idx"]
    feature_means = checkpoint["feature_means"]
    feature_stds = checkpoint["feature_stds"]

    # Load manifest
    with open("beans_zero_full_manifest.json") as f:
        manifest = json.load(f)

    samples = manifest["samples"]

    # Load cache manifest
    with open("beans_feature_cache_112d/cache_manifest.json") as f:
        cache_manifest = json.load(f)

    # Build reverse mapping
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Create model
    model = RosettaNetPL(len(label_to_idx), torch.ones(len(label_to_idx)))

    model.load_state_dict(model_state_dict)
    model.eval()

    # Load features from last 10% for evaluation
    n_eval = int(len(samples) * 0.1)
    eval_start = len(samples) - n_eval
    print(f"  Evaluating on {n_eval} samples...")

    # Track predictions
    correct_species = 0
    correct_taxon = 0
    total_evaluated = 0

    for i in range(eval_start, len(samples)):
        sample = samples[i]
        audio_file = sample["audio_file"]
        true_label = sample["labels"]["output"] if sample["labels"]["output"] != "None" else f"task_{sample['labels']['task']}"

        cache_file = cache_manifest["entries"].get(audio_file)
        if cache_file:
            full_path = Path("beans_feature_cache_112d") / cache_file
            if full_path.exists():
                features = load_bincode_features(full_path)
                if features.shape[0] == FEATURE_DIM:
                    # Normalize
                    normalized = (features - feature_means) / feature_stds
                    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)

                    # Predict
                    with torch.no_grad():
                        logits = model(tensor)
                        pred_idx = torch.argmax(logits).item()
                        pred_label = idx_to_label[pred_idx]

                        if pred_label.lower() == true_label.lower():
                            correct_species += 1
                        total_evaluated += 1

    species_accuracy = correct_species / total_evaluated * 100.0
    print(f"\n  Species Accuracy: {species_accuracy:.2f}%")
    print(f"  Evaluated: {total_evaluated} samples")


def main():
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: CUDA not available, falling back to CPU!")
    print()

    print("=" * 80)
    print(" 1. Training Phase")
    print("=" * 80)
    print()

    # Load data
    (
        train_loader,
        val_loader,
        n_classes,
        weights_tensor,
        label_to_idx,
        feature_means,
        feature_stds,
    ) = load_data()

    # Create model
    model = RosettaNetPL(n_classes, weights_tensor)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_acc",
                patience=PATIENCE,
                mode="max",
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=".",
                filename="rosetta_net_pl-{epoch:02d}-{val_acc:.2f}.ckpt",
                save_top_k=1,
                monitor="val_acc",
                mode="max",
            ),
        ],
        logger=pl.loggers.TensorBoardLogger(log_graph_freq=100),
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 80)
    print("  2. Evaluation Phase")
    print("=" * 80)
    print()

    # Load best model and evaluate
    evaluate_model()
