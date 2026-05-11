#!/usr/bin/env python3
"""
Affective VAE Training - Module 1 (Dual-Stream)

Training script for the β-VAE that learns disentangled 16D affect representations
from 54D affective features extracted from 112D RosettaFeatures.

Architecture:
    Input: 54D affective features (AffectiveFeatureExtractor)
    Encoder: 54 → 128 → 128 (ReLU)
    Latent: 16D (μ, logσ) with β=2.0 for disentanglement
    Decoder: 16 → 128 → 128 → 54 (ReLU)

Training Targets:
    - Reconstruction loss < 0.1
    - KL divergence stable (not exploding)
    - β = 2.0 for disentangled representation

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .affective_encoder import AffectiveFeatureExtractor
from .affective_vae import BetaVAE, AffectVAECheckpoint

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AffectiveVAETrainingConfig:
    """Configuration for affective VAE training."""

    # Model architecture
    input_dim: int = 54
    latent_dim: int = 16
    hidden_dim: int = 128
    beta: float = 2.0  # β-VAE parameter for disentanglement

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    weight_decay: float = 1e-5

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Targets
    target_reconstruction_loss: float = 0.1

    # Paths
    checkpoint_dir: str = "models/dual_stream"
    log_interval: int = 10


# =============================================================================
# Dataset
# =============================================================================


class CachedFeaturesDataset(Dataset):
    """
    Dataset for training affective VAE on cached 112D features.

    Extracts 54D affective features from cached 112D RosettaFeatures
    and uses them for VAE training.
    """

    def __init__(self, features_npy: str, normalize: bool = True):
        """
        Initialize dataset from cached features.

        Args:
            features_npy: Path to .npy file with cached 112D features
            normalize: Whether to normalize features
        """
        self.features_112d = np.load(features_npy)
        logger.info(f"Loaded {len(self.features_112d)} feature vectors from {features_npy}")

        # Extract affective features
        self.extractor = AffectiveFeatureExtractor()
        self.features_affective = self.extractor.extract_affective_features_batch(
            self.features_112d
        )

        # Compute normalization stats
        if normalize:
            self.mean = self.features_affective.mean(axis=0)
            self.std = self.features_affective.std(axis=0) + 1e-8
            self.features_affective = (self.features_affective - self.mean) / self.std
        else:
            self.mean = np.zeros(54)
            self.std = np.ones(54)

        logger.info(f"Extracted {self.features_affective.shape[1]}D affective features")
        logger.info(f"Feature range: [{self.features_affective.min():.3f}, {self.features_affective.max():.3f}]")

    def __len__(self) -> int:
        return len(self.features_affective)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.features_affective[idx]).float()


@dataclass
class VocalizationSegment:
    """A single training example with 112D features."""
    features_112d: np.ndarray
    species: Optional[str] = None
    phrase_id: Optional[str] = None


class SegmentsDataset(Dataset):
    """
    Dataset for training from JSON segments file.

    Expects JSON format with "segments" list containing "features_112d" arrays.
    """

    def __init__(self, segments_json: str, normalize: bool = True):
        """
        Initialize dataset from JSON segments file.

        Args:
            segments_json: Path to JSON file with segments
            normalize: Whether to normalize features
        """
        self.segments: List[VocalizationSegment] = []
        self.extractor = AffectiveFeatureExtractor()

        with open(segments_json, "r") as f:
            data = json.load(f)

        for item in data.get("segments", []):
            if "features_112d" in item:
                features = np.array(item["features_112d"], dtype=np.float32)
                if len(features) == 112:
                    self.segments.append(VocalizationSegment(
                        features_112d=features,
                        species=item.get("species"),
                        phrase_id=item.get("phrase_id"),
                    ))

        logger.info(f"Loaded {len(self.segments)} segments from {segments_json}")

        # Extract and normalize affective features
        affective_features = []
        for seg in self.segments:
            affective = self.extractor.extract_affective_features(seg.features_112d)
            affective_features.append(affective)

        self.features_affective = np.array(affective_features, dtype=np.float32)

        if normalize:
            self.mean = self.features_affective.mean(axis=0)
            self.std = self.features_affective.std(axis=0) + 1e-8
            self.features_affective = (self.features_affective - self.mean) / self.std
        else:
            self.mean = np.zeros(54)
            self.std = np.ones(54)

        logger.info(f"Extracted {self.features_affective.shape[1]}D affective features")

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.features_affective[idx]).float()


# =============================================================================
# Training
# =============================================================================


class AffectiveVAETrainer:
    """Trainer for affective VAE."""

    def __init__(self, config: AffectiveVAETrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model
        self.model = BetaVAE(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            beta=config.beta,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Training state
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "val_loss": [],
        }

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for batch_idx, x in enumerate(train_loader):
            x = x.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            recon_x, mu, logvar = self.model(x)

            # Compute loss
            loss, recon_loss, kl_loss = self.model.loss_function(recon_x, x, mu, logvar)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            if batch_idx % self.config.log_interval == 0:
                logger.debug(
                    f"Batch {batch_idx}/{len(train_loader)}: "
                    f"Loss={loss.item():.4f}, Recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f}"
                )

        n_batches = len(train_loader)
        return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for x in val_loader:
            x = x.to(self.device)
            recon_x, mu, logvar = self.model(x)
            loss, recon_loss, kl_loss = self.model.loss_function(recon_x, x, mu, logvar)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        n_batches = len(val_loader)
        return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader

        Returns:
            Training history
        """
        logger.info(f"Training β-VAE with β={self.config.beta} on {self.device}")
        logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.epochs):
            # Train epoch
            train_loss, train_recon, train_kl = self.train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_loss, val_recon, val_kl = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                scheduler_metric = val_loss
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs}: "
                    f"Train={train_loss:.4f} (R={train_recon:.4f}, KL={train_kl:.4f}), "
                    f"Val={val_loss:.4f} (R={val_recon:.4f}, KL={val_kl:.4f})"
                )
            else:
                val_loss, val_recon, val_kl = train_loss, train_recon, train_kl
                scheduler_metric = train_loss
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs}: "
                    f"Loss={train_loss:.4f} (R={train_recon:.4f}, KL={train_kl:.4f})"
                )

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["recon_loss"].append(train_recon)
            self.history["kl_loss"].append(train_kl)

            # Learning rate scheduling
            self.scheduler.step(scheduler_metric)

            # Early stopping check
            if val_loss < self.best_loss - self.config.min_delta:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint("best_checkpoint.pt")
                logger.info(f"  New best model! Saving checkpoint.")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # Save periodic checkpoint
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

        # Save final model
        self.save_checkpoint("final_checkpoint.pt")
        logger.info("Training complete!")

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = Path(self.config.checkpoint_dir) / filename

        AffectVAECheckpoint.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=len(self.history["train_loss"]),
            loss=self.history["val_loss"][-1] if self.history["val_loss"] else 0.0,
            path=path,
        )
        logger.debug(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = AffectVAECheckpoint.load_checkpoint(path, device=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train affective VAE for dual-stream architecture"
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to cached features (.npy) or segments JSON",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["npy", "json"],
        default="npy",
        help="Data format (default: npy)",
    )

    # Model
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimension (default: 16)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension (default: 128)")
    parser.add_argument("--beta", type=float, default=2.0, help="β-VAE parameter (default: 2.0)")

    # Training
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")

    # Output
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/dual_stream",
        help="Checkpoint directory (default: models/dual_stream)",
    )
    parser.add_argument("--output-name", type=str, default="affective_vae.pt", help="Output model name")

    # Validation split
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    return parser.parse_args()


def main():
    """Main training entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create config
    config = AffectiveVAETrainingConfig(
        input_dim=54,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        beta=args.beta,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Load dataset
    if args.data_type == "npy":
        dataset = CachedFeaturesDataset(args.data, normalize=True)
    else:
        dataset = SegmentsDataset(args.data, normalize=True)

    logger.info(f"Dataset size: {len(dataset)} samples")

    # Split train/validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    logger.info(f"Train: {train_size} samples, Val: {val_size} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create trainer
    trainer = AffectiveVAETrainer(config)

    # Train
    history = trainer.train(train_loader, val_loader)

    # Save final model with explicit name
    final_path = Path(args.checkpoint_dir) / args.output_name
    trainer.model.save_model_only(final_path)
    logger.info(f"Saved final model to {final_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("Training Summary:")
    logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"  Final recon loss: {history['recon_loss'][-1]:.4f}")
    logger.info(f"  Final KL loss: {history['kl_loss'][-1]:.4f}")
    logger.info(f"  Target recon loss: {config.target_reconstruction_loss}")
    logger.info(f"  Target met: {history['recon_loss'][-1] < config.target_reconstruction_loss}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
