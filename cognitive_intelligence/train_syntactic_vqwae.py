#!/usr/bin/env python3
"""
Syntactic VQ-VAE Training - Module 2 (Dual-Stream)

Training script for the VQ-VAE that learns discrete token representations
from 44D syntactic features extracted from 112D RosettaFeatures.

Architecture:
    Input: 44D syntactic features (SyntacticFeatureExtractor)
    Encoder: 44 → 128 → 32D latent
    Vector Quantization: EMA codebook with 64 tokens
    Decoder: 32D → 128 → 44

Training Targets:
    - Codebook utilization > 80%
    - Commitment loss < 0.05
    - Perplexity > 0.5 (active tokens)

Key Features:
    - EMA codebook updates (decay=0.99) to prevent collapse
    - Codebook revival for dead tokens
    - Laplace smoothing for syntax graph

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import argparse
import json
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .syntactic_encoder import SyntacticFeatureExtractor
from .syntactic_vqvae import SyntacticVQVAE, EMAVectorQuantizer, VQVAECheckpoint

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SyntacticVQVAETrainingConfig:
    """Configuration for syntactic VQ-VAE training."""

    # Model architecture
    input_dim: int = 44
    codebook_size: int = 64
    codebook_dim: int = 32
    hidden_dim: int = 128
    ema_decay: float = 0.99
    ema_epsilon: float = 1e-5

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    weight_decay: float = 1e-5

    # Loss weights
    commitment_cost: float = 0.25

    # Codebook revival
    revival_threshold: int = 5  # Revive tokens unused for N epochs
    revival_interval: int = 10  # Check every N epochs

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Targets
    target_codebook_utilization: float = 0.8
    target_commitment_loss: float = 0.05

    # Paths
    checkpoint_dir: str = "models/dual_stream"
    log_interval: int = 10


# =============================================================================
# Dataset
# =============================================================================


class CachedSyntacticFeaturesDataset(Dataset):
    """
    Dataset for training syntactic VQ-VAE on cached 112D features.

    Extracts 44D syntactic features from cached 112D RosettaFeatures
    and uses them for VQ-VAE training.
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

        # Extract syntactic features
        self.extractor = SyntacticFeatureExtractor()
        self.features_syntactic = self.extractor.extract_syntactic_features_batch(
            self.features_112d
        )

        # Compute normalization stats
        if normalize:
            self.mean = self.features_syntactic.mean(axis=0)
            self.std = self.features_syntactic.std(axis=0) + 1e-8
            self.features_syntactic = (self.features_syntactic - self.mean) / self.std
        else:
            self.mean = np.zeros(44)
            self.std = np.ones(44)

        logger.info(f"Extracted {self.features_syntactic.shape[1]}D syntactic features")
        logger.info(
            f"Feature range: [{self.features_syntactic.min():.3f}, {self.features_syntactic.max():.3f}]"
        )

    def __len__(self) -> int:
        return len(self.features_syntactic)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.features_syntactic[idx]).float()


@dataclass
class VocalizationSegment:
    """A single training example with 112D features."""
    features_112d: np.ndarray
    species: Optional[str] = None
    phrase_id: Optional[str] = None


class SyntacticSegmentsDataset(Dataset):
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
        self.extractor = SyntacticFeatureExtractor()

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

        # Extract and normalize syntactic features
        syntactic_features = []
        for seg in self.segments:
            syntactic = self.extractor.extract_syntactic_features(seg.features_112d)
            syntactic_features.append(syntactic)

        self.features_syntactic = np.array(syntactic_features, dtype=np.float32)

        if normalize:
            self.mean = self.features_syntactic.mean(axis=0)
            self.std = self.features_syntactic.std(axis=0) + 1e-8
            self.features_syntactic = (self.features_syntactic - self.mean) / self.std
        else:
            self.mean = np.zeros(44)
            self.std = np.ones(44)

        logger.info(f"Extracted {self.features_syntactic.shape[1]}D syntactic features")

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.features_syntactic[idx]).float()


# =============================================================================
# Training
# =============================================================================


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    total_loss: float
    reconstruction_loss: float
    commitment_loss: float
    codebook_utilization: float
    perplexity: float
    dead_tokens: int


class SyntacticVQVAETrainer:
    """Trainer for syntactic VQ-VAE."""

    def __init__(self, config: SyntacticVQVAETrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model
        self.model = SyntacticVQVAE(
            input_dim=config.input_dim,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            hidden_dim=config.hidden_dim,
            decay=config.ema_decay,
            commitment_cost=config.commitment_cost,
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
            "commitment_loss": [],
            "codebook_utilization": [],
            "perplexity": [],
            "dead_tokens": [],
            "val_loss": [],
        }

        # Codebook usage tracking for revival
        self.token_usage: Counter = Counter()
        self.epochs_since_last_use = [0] * config.codebook_size

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> TrainingMetrics:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_commit = 0.0

        # Track token usage
        epoch_token_ids = []

        for batch_idx, x in enumerate(train_loader):
            x = x.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Encode (keep this for commitment loss)
            z_e = self.model.encoder(x)

            # Quantize using the VQ layer directly (no EMA update during training)
            # The straight-through estimator handles gradient flow to encoder
            z_q, token_ids, perplexity = self.model.vq(z_e)

            # Decode
            x_recon = self.model.decoder(z_q)

            # Compute losses
            # Reconstruction loss
            recon_loss = F.mse_loss(x_recon, x)

            # Commitment loss (encoder commitment to codebook)
            commit_loss = F.mse_loss(z_q.detach(), z_e) * self.config.commitment_cost

            # Total loss
            loss = recon_loss + commit_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_commit += commit_loss.item()

            # Track token usage
            epoch_token_ids.extend(token_ids.cpu().tolist())

            if batch_idx % self.config.log_interval == 0:
                logger.debug(
                    f"Batch {batch_idx}/{len(train_loader)}: "
                    f"Loss={loss.item():.4f}, Recon={recon_loss.item():.4f}, "
                    f"Commit={commit_loss.item():.4f}"
                )

        n_batches = len(train_loader)

        # Compute codebook metrics
        unique_tokens = len(set(epoch_token_ids))
        codebook_util = unique_tokens / self.config.codebook_size
        dead_tokens = self.config.codebook_size - unique_tokens

        # Update token usage tracking
        self._update_token_usage(epoch_token_ids)

        # Safe perplexity computation (avoid NaN from empty distributions)
        if epoch_token_ids:
            from collections import Counter
            token_counts = Counter(epoch_token_ids)
            total = len(epoch_token_ids)
            avg_probs = torch.tensor([token_counts.get(i, 0) / total for i in range(self.config.codebook_size)], dtype=torch.float32)
            perplexity_val = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()
        else:
            perplexity_val = 1.0  # No tokens processed

        return TrainingMetrics(
            total_loss=total_loss / n_batches,
            reconstruction_loss=total_recon / n_batches,
            commitment_loss=total_commit / n_batches,
            codebook_utilization=codebook_util,
            perplexity=perplexity_val,
            dead_tokens=dead_tokens,
        )

    def _update_token_usage(self, token_ids: List[int]):
        """Update token usage tracking for codebook revival."""
        # Count current epoch usage
        current_usage = Counter(token_ids)

        # Update epochs since last use
        for token_id in range(self.config.codebook_size):
            if token_id in current_usage:
                self.epochs_since_last_use[token_id] = 0
            else:
                self.epochs_since_last_use[token_id] += 1

    def revive_dead_codes(self):
        """Revive dead codebook entries by copying active ones."""
        dead_tokens = [
            i for i, count in enumerate(self.epochs_since_last_use)
            if count >= self.config.revival_threshold
        ]

        if not dead_tokens:
            return

        # Get active tokens
        active_tokens = [
            i for i, count in enumerate(self.epochs_since_last_use)
            if count == 0
        ]

        if not active_tokens:
            logger.warning("No active tokens to copy for revival")
            return

        logger.info(f"Reviving {len(dead_tokens)} dead codebook entries")

        # Revive by copying random active token with small noise
        with torch.no_grad():
            codebook = self.model.vq.codebook_ema

            for dead_token in dead_tokens:
                # Copy random active token
                source_token = np.random.choice(active_tokens)
                codebook[dead_token] = codebook[source_token] + torch.randn_like(
                    codebook[dead_token]
                ) * 0.01

                # Reset counter
                self.epochs_since_last_use[dead_token] = 0

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> TrainingMetrics:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_commit = 0.0

        epoch_token_ids = []

        for x in val_loader:
            x = x.to(self.device)

            # Encode
            z_e = self.model.encoder(x)

            # Quantize
            z_q, token_ids, perplexity = self.model.vq(z_e)

            # Decode
            x_recon = self.model.decoder(z_q)

            # Compute losses
            recon_loss = F.mse_loss(x_recon, x)
            commit_loss = F.mse_loss(z_q.detach(), z_e) * self.config.commitment_cost
            loss = recon_loss + commit_loss

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_commit += commit_loss.item()
            epoch_token_ids.extend(token_ids.cpu().tolist())

        n_batches = len(val_loader)

        unique_tokens = len(set(epoch_token_ids))
        codebook_util = unique_tokens / self.config.codebook_size
        dead_tokens = self.config.codebook_size - unique_tokens

        # Safe perplexity computation
        if epoch_token_ids:
            from collections import Counter
            token_counts = Counter(epoch_token_ids)
            total = len(epoch_token_ids)
            avg_probs = torch.tensor([token_counts.get(i, 0) / total for i in range(self.config.codebook_size)], dtype=torch.float32)
            perplexity_val = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()
        else:
            perplexity_val = 1.0

        return TrainingMetrics(
            total_loss=total_loss / n_batches,
            reconstruction_loss=total_recon / n_batches,
            commitment_loss=total_commit / n_batches,
            codebook_utilization=codebook_util,
            perplexity=perplexity_val,
            dead_tokens=dead_tokens,
        )

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
        logger.info(
            f"Training VQ-VAE (codebook={self.config.codebook_size}, "
            f"dim={self.config.codebook_dim}) on {self.device}"
        )
        logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.epochs):
            # Train epoch
            metrics = self.train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_loss = val_metrics.total_loss
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs}: "
                    f"Train={metrics.total_loss:.4f} (R={metrics.reconstruction_loss:.4f}, "
                    f"C={metrics.commitment_loss:.4f}), Val={val_loss:.4f}, "
                    f"Util={metrics.codebook_utilization:.2%}, Dead={metrics.dead_tokens}"
                )
            else:
                val_metrics = metrics
                val_loss = metrics.total_loss
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs}: "
                    f"Loss={metrics.total_loss:.4f} (R={metrics.reconstruction_loss:.4f}, "
                    f"C={metrics.commitment_loss:.4f}), "
                    f"Util={metrics.codebook_utilization:.2%}, Dead={metrics.dead_tokens}"
                )

            # Update history
            self.history["train_loss"].append(metrics.total_loss)
            self.history["recon_loss"].append(metrics.reconstruction_loss)
            self.history["commitment_loss"].append(metrics.commitment_loss)
            self.history["codebook_utilization"].append(metrics.codebook_utilization)
            self.history["perplexity"].append(metrics.perplexity)
            self.history["dead_tokens"].append(metrics.dead_tokens)
            if val_loader is not None:
                self.history["val_loss"].append(val_loss)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Codebook revival
            if (epoch + 1) % self.config.revival_interval == 0:
                self.revive_dead_codes()

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

        VQVAECheckpoint.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=len(self.history["train_loss"]),
            loss=self.history["val_loss"][-1] if self.history["val_loss"] else 0.0,
            path=path,
        )
        logger.debug(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train syntactic VQ-VAE for dual-stream architecture"
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
    parser.add_argument("--codebook-size", type=int, default=64, help="Codebook size (default: 64)")
    parser.add_argument("--codebook-dim", type=int, default=32, help="Codebook dimension (default: 32)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension (default: 128)")
    parser.add_argument("--ema-decay", type=float, default=0.99, help="EMA decay (default: 0.99)")

    # Training
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--commitment-cost", type=float, default=0.25, help="Commitment cost (default: 0.25)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")

    # Codebook revival
    parser.add_argument(
        "--revival-threshold",
        type=int,
        default=5,
        help="Revive tokens unused for N epochs (default: 5)",
    )
    parser.add_argument(
        "--revival-interval",
        type=int,
        default=10,
        help="Check for revival every N epochs (default: 10)",
    )

    # Output
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/dual_stream",
        help="Checkpoint directory (default: models/dual_stream)",
    )
    parser.add_argument("--output-name", type=str, default="syntactic_vqvae.pt", help="Output model name")

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
    config = SyntacticVQVAETrainingConfig(
        input_dim=44,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        hidden_dim=args.hidden_dim,
        ema_decay=args.ema_decay,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        commitment_cost=args.commitment_cost,
        patience=args.patience,
        revival_threshold=args.revival_threshold,
        revival_interval=args.revival_interval,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Load dataset
    if args.data_type == "npy":
        dataset = CachedSyntacticFeaturesDataset(args.data, normalize=True)
    else:
        dataset = SyntacticSegmentsDataset(args.data, normalize=True)

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
    trainer = SyntacticVQVAETrainer(config)

    # Train
    history = trainer.train(train_loader, val_loader)

    # Save final model with explicit name
    final_path = Path(args.checkpoint_dir) / args.output_name
    trainer.model.save_model_only(final_path)
    logger.info(f"Saved final model to {final_path}")

    # Print summary
    final_util = history["codebook_utilization"][-1]
    final_commit = history["commitment_loss"][-1]

    logger.info("=" * 60)
    logger.info("Training Summary:")
    logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"  Final recon loss: {history['recon_loss'][-1]:.4f}")
    logger.info(f"  Final commitment loss: {final_commit:.4f}")
    logger.info(f"  Final codebook utilization: {final_util:.2%}")
    logger.info(f"  Final perplexity: {history['perplexity'][-1]:.4f}")
    logger.info(f"  Final dead tokens: {history['dead_tokens'][-1]}")
    logger.info("")
    logger.info(f"  Target utilization: {config.target_codebook_utilization:.2%}")
    logger.info(f"  Target commitment loss: {config.target_commitment_loss:.4f}")
    logger.info(f"  Utilization target met: {final_util >= config.target_codebook_utilization}")
    logger.info(
        f"  Commitment target met: {final_commit <= config.target_commitment_loss}"
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
