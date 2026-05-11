#!/usr/bin/env python3
"""
VQ-VAE Training Pipeline (Sprint 1-2) - Stream 2

Training pipeline for VQ-VAE with EMA on 44D syntactic features.
Target: Codebook utilization >80%, commitment loss <0.05, discrete tokens.

Key Features:
- EMA codebook updates to prevent collapse
- Codebook revival for dead tokens
- Utilization tracking and monitoring
- Early stopping based on commitment loss

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
from torch.utils.data import DataLoader, TensorDataset

from cognitive_intelligence.syntactic_feature_extractor import SyntacticFeatureExtractor
from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE

logger = logging.getLogger(__name__)


@dataclass
class VQVAETrainingConfig:
    """Configuration for VQ-VAE training."""

    # Model architecture
    input_dim: int = 44  # Syntactic features
    codebook_size: int = 64
    codebook_dim: int = 32
    hidden_dim: int = 128

    # Training hyperparameters
    batch_size: int = 256
    learning_rate: float = 1e-3
    num_epochs: int = 200
    weight_decay: float = 1e-5

    # VQ-specific
    commitment_cost: float = 0.25
    decay: float = 0.99  # EMA decay
    revival_threshold: float = 0.01  # Dead token threshold

    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = "models/checkpoints/vqvae"
    save_every: int = 10

    # Targets
    target_commitment_loss: float = 0.05
    target_utilization: float = 80.0  # Percentage

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class VQVAETrainer:
    """
    Trainer for VQ-VAE on syntactic features.

    Monitors:
    - Reconstruction loss
    - Commitment loss (target: <0.05)
    - Codebook utilization (target: >80%)
    - Perplexity (diversity metric)
    """

    def __init__(
        self,
        model: SyntacticVQVAE,
        config: Optional[VQVAETrainingConfig] = None,
    ):
        self.model = model.to(config.device if config else "cpu")
        self.config = config or VQVAETrainingConfig()
        self.device = torch.device(self.config.device)

        # Optimizer (only encoder/decoder, not VQ layer which uses EMA)
        self.optimizer = torch.optim.Adam(
            list(self.model.encoder.parameters()) +
            list(self.model.decoder.parameters()),
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
        self.commit_losses: List[float] = []

        # Utilization history
        self.utilization_history: List[float] = []
        self.perplexity_history: List[float] = []

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"VQ-VAE Trainer initialized: "
            f"codebook_size={self.config.codebook_size}, device={self.device}"
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Tuple[float, float, float, float, float]:
        """
        Train for one epoch.

        Returns:
            (avg_loss, avg_recon_loss, avg_commit_loss, utilization, perplexity)
        """
        self.model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_commit = 0.0
        num_batches = 0

        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(self.device)

            # Forward pass
            x_recon, z, z_q, token_ids, perplexity = self.model(data)

            # Compute loss
            loss_dict = self.model.loss_function(x_recon, data, z, z_q)
            loss = loss_dict["total_loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            total_recon += loss_dict["recon_loss"].item()
            total_commit += loss_dict["commitment_loss"].item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_commit = total_commit / num_batches

        # Get utilization stats
        utilization = self.model.codebook_utilization()
        perplexity_val = perplexity.item()

        return avg_loss, avg_recon, avg_commit, utilization, perplexity_val

    def validate(
        self,
        val_loader: Optional[DataLoader] = None,
    ) -> Tuple[float, float, float, float, float]:
        """
        Validate the model.

        Returns:
            (avg_loss, avg_recon_loss, avg_commit_loss, utilization, perplexity)
        """
        self.model.eval()

        if val_loader is None:
            return self.train_epoch(train_loader)

        total_loss = 0.0
        total_recon = 0.0
        total_commit = 0.0
        num_batches = 0

        perplexity_sum = 0.0

        with torch.no_grad():
            for (data,) in val_loader:
                data = data.to(self.device)

                # Forward pass
                x_recon, z, z_q, token_ids, perplexity = self.model(data)

                # Compute loss
                loss_dict = self.model.loss_function(x_recon, data, z, z_q)
                loss = loss_dict["total_loss"]

                total_loss += loss.item()
                total_recon += loss_dict["recon_loss"].item()
                total_commit += loss_dict["commitment_loss"].item()
                perplexity_sum += perplexity.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_commit = total_commit / num_batches
        perplexity_val = perplexity_sum / num_batches

        utilization = self.model.codebook_utilization()

        return avg_loss, avg_recon, avg_commit, utilization, perplexity_val

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "recon_losses": self.recon_losses,
            "commit_losses": self.commit_losses,
            "utilization_history": self.utilization_history,
            "perplexity_history": self.perplexity_history,
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
        self.commit_losses = checkpoint.get("commit_losses", [])
        self.utilization_history = checkpoint.get("utilization_history", [])
        self.perplexity_history = checkpoint.get("perplexity_history", [])
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
            f"Starting VQ-VAE training for {self.config.num_epochs} epochs"
        )

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_recon, train_commit, util, perp = self.train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_loss, val_recon, val_commit, val_util, val_perp = self.validate(val_loader)
            else:
                val_loss, val_recon, val_commit, val_util, val_perp = (
                    train_loss, train_recon, train_commit, util, perp
                )

            # Track history
            self.train_losses.append(train_loss)
            self.recon_losses.append(val_recon)
            self.commit_losses.append(val_commit)
            self.utilization_history.append(val_util)
            self.perplexity_history.append(val_perp)

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs}: "
                f"loss={train_loss:.4f}, recon={train_recon:.4f}, "
                f"commit={train_commit:.4f}, util={val_util:.1f}%, "
                f"perp={val_perp:.2f}"
            )

            # Check targets
            commit_target_met = val_commit < self.config.target_commitment_loss
            util_target_met = val_util > self.config.target_utilization

            if commit_target_met and util_target_met:
                logger.info(
                    f"✓ Training targets met: "
                    f"commit={val_commit:.4f}, util={val_util:.1f}%"
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
            "commit_losses": self.commit_losses,
            "utilization_history": self.utilization_history,
            "perplexity_history": self.perplexity_history,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
        }


def create_training_data(
    features_112d: np.ndarray,
    feature_extractor: SyntacticFeatureExtractor,
) -> np.ndarray:
    """
    Create training data by extracting syntactic features.

    Args:
        features_112d: Array of shape (N, 112)
        feature_extractor: SyntacticFeatureExtractor instance

    Returns:
        Syntactic features of shape (N, 44)
    """
    logger.info(f"Extracting syntactic features from {features_112d.shape[0]} samples")

    # Compute normalization stats
    feature_extractor.compute_normalization_stats([features_112d[i] for i in range(len(features_112d))])

    # Extract features
    syntactic_features = []
    for i in range(len(features_112d)):
        extracted = feature_extractor.extract(features_112d[i])
        syntactic_features.append(extracted)

    return np.array(syntactic_features)


def train_vqvae(
    features_112d: np.ndarray,
    config: Optional[VQVAETrainingConfig] = None,
) -> Tuple[SyntacticVQVAE, VQVAETrainer]:
    """
    Convenience function to train VQ-VAE on 112D features.

    Args:
        features_112d: Array of shape (N, 112)
        config: Optional training configuration

    Returns:
        (trained_model, trainer)
    """
    config = config or VQVAETrainingConfig()

    # Create feature extractor
    feature_extractor = SyntacticFeatureExtractor()

    # Extract syntactic features
    syntactic_features = create_training_data(features_112d, feature_extractor)

    # Create model
    model = SyntacticVQVAE(
        input_dim=config.input_dim,
        codebook_size=config.codebook_size,
        codebook_dim=config.codebook_dim,
        hidden_dim=config.hidden_dim,
        commitment_cost=config.commitment_cost,
        decay=config.decay,
        revival_threshold=config.revival_threshold,
    )

    # Create trainer
    trainer = VQVAETrainer(model, config)

    # Create data loaders
    dataset = TensorDataset(
        torch.from_numpy(syntactic_features).float()
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

    logger.info("VQ-VAE training complete")

    return model, trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create dummy training data
    np.random.seed(42)
    dummy_features_112d = np.random.randn(2000, 112).astype(np.float32)

    # Train
    model, trainer = train_vqvae(dummy_features_112d)

    print(f"Training complete. Best loss: {trainer.best_loss:.4f}")
    print(f"Final utilization: {trainer.utilization_history[-1]:.1f}%")
