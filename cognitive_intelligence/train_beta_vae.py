#!/usr/bin/env python3
"""
β-VAE Training Pipeline (Sprint 1-2) - Stream 1

Training pipeline for β-VAE with β=2.0 on 30D affective features.
Target: KL loss stable, reconstruction loss <0.1, disentangled 16D latent space.

Key Features:
- β=2.0 for disentangled representation
- KL annealing for stable training
- Early stopping based on reconstruction loss
- Checkpointing and logging

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from cognitive_intelligence.affective_feature_extractor import AffectiveFeatureExtractor
from cognitive_intelligence.affective_vae import BetaVAE

logger = logging.getLogger(__name__)


@dataclass
class BetaVAETrainingConfig:
    """Configuration for β-VAE training."""

    # Model architecture
    input_dim: int = 30  # Affective features
    latent_dim: int = 16
    hidden_dim: int = 64
    beta: float = 2.0  # For disentanglement

    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 200
    weight_decay: float = 1e-5

    # KL annealing
    kl_annealing: bool = True
    kl_anneal_cycles: int = 5  # Number of annealing cycles
    kl_anneal_ratio: float = 0.5  # Ratio of epoch for annealing

    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = "models/checkpoints/beta_vae"
    save_every: int = 10

    # Targets
    target_recon_loss: float = 0.1
    target_kl_stable_epochs: int = 10  # KL stable for this many epochs

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BetaVAETrainer:
    """
    Trainer for β-VAE on affective features.

    Monitors:
    - Reconstruction loss (target: <0.1)
    - KL divergence (target: stable, not exploding)
    - Disentanglement (qualitative)
    """

    def __init__(
        self,
        model: BetaVAE,
        config: Optional[BetaVAETrainingConfig] = None,
    ):
        self.model = model.to(config.device if config else "cpu")
        self.config = config or BetaVAETrainingConfig()
        self.device = torch.device(self.config.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

        # Loss history
        self.train_losses: List[float] = []
        self.recon_losses: List[float] = []
        self.kl_losses: List[float] = []

        # KL stability tracking
        self.kl_history: List[float] = []

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"β-VAE Trainer initialized: "
            f"β={self.config.beta}, device={self.device}"
        )

    def compute_kl_weight(self, epoch: int) -> float:
        """
        Compute KL annealing weight.

        Gradually increases β from 0 to target over training.
        """
        if not self.config.kl_annealing:
            return 1.0

        # Cyclical annealing
        cycle_length = self.config.num_epochs / self.config.kl_anneal_cycles
        cycle_position = epoch % cycle_length
        anneal_position = cycle_position / (cycle_length * self.config.kl_anneal_ratio)

        return min(1.0, max(0.0, anneal_position))

    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Tuple[float, float, float]:
        """
        Train for one epoch.

        Returns:
            (avg_loss, avg_recon_loss, avg_kl_loss)
        """
        self.model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        kl_weight = self.compute_kl_weight(self.current_epoch)

        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(self.device)

            # Forward pass
            recon_batch, mu, logvar = self.model(data)

            # Compute loss
            loss_dict = self.model.loss_function(recon_batch, data, mu, logvar)

            # Apply KL annealing
            kl_loss = loss_dict["kl_loss"] * kl_weight
            loss = loss_dict["reconstruction_loss"] + self.config.beta * kl_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            total_recon += loss_dict["reconstruction_loss"].item()
            total_kl += kl_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches

        return avg_loss, avg_recon, avg_kl

    def validate(
        self,
        val_loader: Optional[DataLoader] = None,
    ) -> Tuple[float, float, float]:
        """
        Validate the model.

        Returns:
            (avg_loss, avg_recon_loss, avg_kl_loss)
        """
        self.model.eval()

        if val_loader is None:
            # Use training loader for validation if none provided
            val_loader = train_loader

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        with torch.no_grad():
            for (data,) in val_loader:
                data = data.to(self.device)

                # Forward pass
                recon_batch, mu, logvar = self.model(data)

                # Compute loss
                loss_dict = self.model.loss_function(recon_batch, data, mu, logvar)
                loss = loss_dict["reconstruction_loss"] + self.config.beta * loss_dict["kl_loss"]

                total_loss += loss.item()
                total_recon += loss_dict["reconstruction_loss"].item()
                total_kl += loss_dict["kl_loss"].item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches

        return avg_loss, avg_recon, avg_kl

    def check_kl_stability(self) -> bool:
        """
        Check if KL divergence has been stable.

        Stability means KL hasn't varied more than 10% over recent epochs.
        """
        if len(self.kl_history) < self.config.target_kl_stable_epochs:
            return False

        recent_kl = self.kl_history[-self.config.target_kl_stable_epochs:]
        kl_mean = np.mean(recent_kl)
        kl_std = np.std(recent_kl)

        # Stable if std is less than 10% of mean
        return (kl_std / (kl_mean + 1e-8)) < 0.1

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "recon_losses": self.recon_losses,
            "kl_losses": self.kl_losses,
            "best_loss": self.best_loss,
            "config": self.config,
        }

        path = Path(self.config.checkpoint_dir) / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.recon_losses = checkpoint.get("recon_losses", [])
        self.kl_losses = checkpoint.get("kl_losses", [])
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        logger.info(f"Loaded checkpoint from {path}, epoch {self.current_epoch}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict:
        """
        Full training loop.

        Returns:
            Training history dictionary
        """
        logger.info(
            f"Starting β-VAE training for {self.config.num_epochs} epochs"
        )

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_recon, train_kl = self.train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_loss, val_recon, val_kl = self.validate(val_loader)
            else:
                val_loss, val_recon, val_kl = train_loss, train_recon, train_kl

            # Track losses
            self.train_losses.append(train_loss)
            self.recon_losses.append(val_recon)
            self.kl_losses.append(val_kl)
            self.kl_history.append(train_kl)

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs}: "
                f"loss={train_loss:.4f}, recon={train_recon:.4f}, kl={train_kl:.4f}"
            )

            # Check targets
            recon_target_met = val_recon < self.config.target_recon_loss
            kl_stable = self.check_kl_stability()

            if recon_target_met and kl_stable:
                logger.info(
                    f"✓ Training targets met: recon={val_recon:.4f}, KL stable"
                )

            # Early stopping
            if self.config.early_stopping:
                if val_loss < self.best_loss - self.config.min_delta:
                    self.best_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0

                    # Save best model
                    self.save_checkpoint("best_model.pt")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch+1}, "
                        f"best loss {self.best_loss:.4f} at epoch {self.best_epoch+1}"
                    )
                    break

            # Periodic checkpointing
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

        # Final checkpoint
        self.save_checkpoint("final_model.pt")

        return {
            "train_losses": self.train_losses,
            "recon_losses": self.recon_losses,
            "kl_losses": self.kl_losses,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
        }


def create_training_data(
    features_112d: np.ndarray,
    feature_extractor: AffectiveFeatureExtractor,
) -> np.ndarray:
    """
    Create training data by extracting affective features.

    Args:
        features_112d: Array of shape (N, 112)
        feature_extractor: AffectiveFeatureExtractor instance

    Returns:
        Affective features of shape (N, 30)
    """
    logger.info(f"Extracting affective features from {features_112d.shape[0]} samples")

    # Compute normalization stats
    feature_extractor.compute_normalization_stats([features_112d[i] for i in range(len(features_112d))])

    # Extract features
    affective_features = []
    for i in range(len(features_112d)):
        extracted = feature_extractor.extract(features_112d[i])
        affective_features.append(extracted)

    return np.array(affective_features)


def train_beta_vae(
    features_112d: np.ndarray,
    config: Optional[BetaVAETrainingConfig] = None,
) -> Tuple[BetaVAE, BetaVAETrainer]:
    """
    Convenience function to train β-VAE on 112D features.

    Args:
        features_112d: Array of shape (N, 112)
        config: Optional training configuration

    Returns:
        (trained_model, trainer)
    """
    config = config or BetaVAETrainingConfig()

    # Create feature extractor
    feature_extractor = AffectiveFeatureExtractor()

    # Extract affective features
    affective_features = create_training_data(features_112d, feature_extractor)

    # Create model
    model = BetaVAE(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        beta=config.beta,
    )

    # Create trainer
    trainer = BetaVAETrainer(model, config)

    # Create data loaders
    dataset = TensorDataset(
        torch.from_numpy(affective_features).float()
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    # Train
    history = trainer.train(train_loader)

    # Save feature extractor stats
    feature_extractor.save_normalization_stats(
        os.path.join(config.checkpoint_dir, "normalization_stats.npz")
    )

    logger.info("β-VAE training complete")

    return model, trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create dummy training data
    np.random.seed(42)
    dummy_features_112d = np.random.randn(1000, 112).astype(np.float32)

    # Train
    model, trainer = train_beta_vae(dummy_features_112d)

    print(f"Training complete. Best loss: {trainer.best_loss:.4f}")
