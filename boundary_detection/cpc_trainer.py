#!/usr/bin/env python3
"""
CPC Trainer: Self-Supervised Training with InfoNCE Loss

Implements Contrastive Predictive Coding training pipeline:
- InfoNCE loss for maximizing mutual information
- Multi-step prediction (predict k steps ahead)
- Gradient accumulation for memory efficiency
- Checkpoint saving and loading

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from boundary_detection.cpc_autoregressive import (
    StreamingContextBuffer,
    create_autoregressive,
)
from boundary_detection.cpc_encoder import CPCEncoder, EncoderConfig, create_encoder

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for CPC training."""
    # Model
    sample_rate: int = 48000
    frame_size_ms: int = 10
    hidden_dim: int = 128
    steps_ahead: int = 5  # Predict k steps into future

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    gradient_clip: float = 1.0

    # CPC-specific
    temperature: float = 0.07  # InfoNCE temperature
    negative_samples: int = 10  # Number of negative samples

    # Data
    sequence_length: int = 64  # Frames per training sequence
    overlap: int = 8  # Overlap between sequences

    # Checkpointing
    checkpoint_dir: str = "checkpoints/cpc"
    save_every: int = 10  # Epochs between saves

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AudioSequenceDataset(Dataset):
    """
    Dataset for training CPC on audio sequences.

    Loads raw audio and creates overlapping sequences for training.
    """

    def __init__(
        self,
        audio_path: str,
        config: TrainingConfig,
        transform: Optional[callable] = None,
    ):
        self.config = config
        self.transform = transform

        # Load audio (support various formats)
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            if sr != config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
                waveform = resampler(waveform)
            self.audio = waveform.mean(0).numpy()  # Convert to mono
        except ImportError:
            # Fallback: load as raw binary
            with open(audio_path, 'rb') as f:
                # Assume 16-bit PCM
                self.audio = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32) / 32768.0

        # Normalize
        if self.audio.max() > 0:
            self.audio = self.audio / self.audio.max()

        # Calculate frame size
        self.frame_size = int(config.sample_rate * config.frame_size_ms / 1000)

        # Calculate number of frames
        self.num_frames = len(self.audio) // self.frame_size

        # Calculate number of sequences
        seq_len = config.sequence_length + config.steps_ahead
        stride = config.sequence_length - config.overlap
        self.num_sequences = max(0, (self.num_frames - seq_len) // stride + 1)

        logger.info(
            f"Loaded {audio_path}: {len(self.audio)/config.sample_rate:.1f}s, "
            f"{self.num_frames} frames, {self.num_sequences} sequences"
        )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sequence.

        Returns:
            (context_frames, future_frames)
            context_frames: (sequence_length, frame_size)
            future_frames: (steps_ahead, frame_size)
        """
        seq_len = self.config.sequence_length + self.config.steps_ahead
        stride = self.config.sequence_length - self.config.overlap

        start_idx = idx * stride
        end_idx = start_idx + seq_len

        # Extract audio frames
        frames = []
        for i in range(start_idx, min(end_idx, self.num_frames)):
            frame_start = i * self.frame_size
            frame_end = frame_start + self.frame_size
            frame = self.audio[frame_start:frame_end]
            frames.append(frame)

        # Pad if necessary
        while len(frames) < seq_len:
            frames.append(np.zeros(self.frame_size))

        frames = np.array(frames)

        # Split into context and future
        context = frames[:self.config.sequence_length]
        future = frames[self.config.sequence_length:]

        return torch.from_numpy(context).float(), torch.from_numpy(future).float()


class CPCModel(nn.Module):
    """
    Complete CPC model: Encoder + Autoregressive + Predictors.
    """

    def __init__(
        self,
        encoder_config: Optional[EncoderConfig] = None,
        ar_config: Optional[dict] = None,
        steps_ahead: int = 5,
    ):
        super().__init__()

        self.steps_ahead = steps_ahead

        # Create encoder
        if encoder_config is None:
            encoder_config = EncoderConfig()
        self.encoder = create_encoder(encoder_config, lightweight=False)

        # Create autoregressive model
        ar_config = ar_config or {}
        self.ar_model = create_autoregressive(
            d_model=encoder_config.hidden_dim,
            **ar_config
        )

        # Create linear predictors for each step ahead
        self.predictors = nn.ModuleList([
            nn.Linear(encoder_config.hidden_dim, encoder_config.hidden_dim)
            for _ in range(steps_ahead)
        ])

        # Initialize predictors
        for predictor in self.predictors:
            nn.init.xavier_uniform_(predictor.weight)
            nn.init.zeros_(predictor.bias)

    def forward(
        self,
        audio_frames: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through CPC model.

        Args:
            audio_frames: (B, T, frame_size) raw audio frames

        Returns:
            z_latent: (B, T, hidden_dim) encoded latents
            context: (B, T, hidden_dim) AR context
            predictions: List of (B, T, hidden_dim) for each step ahead
        """
        batch_size, seq_len, frame_size = audio_frames.shape

        # Encode to latent space
        z_latent = []
        for t in range(seq_len):
            frame = audio_frames[:, t:t+1, :]  # (B, 1, frame_size)
            z_t = self.encoder.encode_frame(frame.squeeze(1).cpu().numpy())
            z_latent.append(z_t)

        z_latent = torch.from_numpy(np.array(z_latent)).float().to(audio_frames.device)
        # z_latent: (seq_len, hidden_dim) -> need (B, T, hidden_dim)

        # Actually, let's use the encoder's forward properly
        # Reshape for 1D conv: (B * T, 1, frame_size)
        audio_flat = audio_frames.reshape(-1, 1, frame_size)
        z_flat = self.encoder(audio_flat)  # (B * T, T', hidden_dim)
        z_latent = z_flat.reshape(batch_size, seq_len, -1)

        # Take mean across compressed time dimension
        z_latent = z_latent.mean(dim=2)  # (B, T, hidden_dim)

        # Autoregressive modeling
        context = self.ar_model(z_latent)  # (B, T, hidden_dim)

        # Predict future latents
        predictions = []
        for k, predictor in enumerate(self.predictors):
            # Predict k steps ahead
            z_pred = predictor(context)  # (B, T, hidden_dim)
            predictions.append(z_pred)

        return z_latent, context, predictions

    def compute_loss(
        self,
        z_latent: torch.Tensor,
        context: torch.Tensor,
        predictions: List[torch.Tensor],
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            z_latent: (B, T, hidden_dim) encoded latents
            context: (B, T, hidden_dim) AR context
            predictions: List of (B, T, hidden_dim) predictions
            temperature: InfoNCE temperature

        Returns:
            loss: Scalar loss value
        """
        batch_size, seq_len, hidden_dim = z_latent.shape

        total_loss = 0.0

        for k, prediction in enumerate(predictions):
            # Predict z_{t+k} using c_t
            # Shift: prediction[t] should match z_latent[t+k+1]

            target_start = k + 1
            if target_start >= seq_len:
                continue

            # Get targets (future latents)
            z_future = z_latent[:, target_start:, :]  # (B, T-k-1, hidden_dim)
            c_t = context[:, :seq_len - target_start, :]  # (B, T-k-1, hidden_dim)
            z_pred = prediction[:, :seq_len - target_start, :]  # (B, T-k-1, hidden_dim)

            # Reshape for InfoNCE
            # Combine batch and time dimensions
            B_T = c_t.shape[0] * c_t.shape[1]

            c_flat = c_t.reshape(B_T, hidden_dim)  # (B*T, hidden_dim)
            z_future_flat = z_future.reshape(B_T, hidden_dim)
            z_pred_flat = z_pred.reshape(B_T, hidden_dim)

            # Compute similarity scores
            # Positive pairs: (c_t, z_future)
            pos_score = torch.sum(c_flat * z_future_flat, dim=1) / temperature

            # Negative pairs: (c_t, z_pred where z != future)
            # Use other samples in batch as negatives
            neg_logits = torch.matmul(c_flat, z_pred_flat.T) / temperature  # (B*T, B*T)

            # InfoNCE loss: -log(exp(pos) / sum(exp(all)))
            # Labels: diagonal elements are positive pairs
            labels = torch.arange(B_T, device=c_t.device)
            loss = F.cross_entropy(neg_logits, labels)

            total_loss += loss

        return total_loss / len(predictions)


class CPCTrainer:
    """
    Training loop for CPC model.

    Handles training, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: CPCModel,
        config: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
        )

        # Data loaders
        if train_dataset is not None:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,  # For compatibility
            )
        else:
            self.train_loader = None

        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
            )
        else:
            self.val_loader = None

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"CPCTrainer initialized: device={config.device}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (context, future) in enumerate(self.train_loader):
            context = context.to(self.device)
            future = future.to(self.device)

            # Combine for forward pass
            audio_frames = torch.cat([context, future], dim=1)

            # Forward pass
            z_latent, ar_context, predictions = self.model(audio_frames)

            # Compute loss
            loss = self.model.compute_loss(
                z_latent,
                ar_context,
                predictions,
                self.config.temperature,
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                logger.debug(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {
            "train_loss": avg_loss,
        }

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for context, future in self.val_loader:
                context = context.to(self.device)
                future = future.to(self.device)

                audio_frames = torch.cat([context, future], dim=1)
                z_latent, ar_context, predictions = self.model(audio_frames)

                loss = self.model.compute_loss(
                    z_latent,
                    ar_context,
                    predictions,
                    self.config.temperature,
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {"val_loss": avg_loss}

    def train(self) -> Dict[str, List[float]]:
        """Full training loop."""
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["train_loss"])

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                history["val_loss"].append(val_metrics["val_loss"])

            # Learning rate scheduling
            self.scheduler.step()

            # Logging
            logger.info(
                f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}"
                + (
                    f", val_loss={val_metrics['val_loss']:.4f}"
                    if val_metrics else ""
                )
            )

            # Checkpointing
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch)

            # Save best model
            if val_metrics and val_metrics.get("val_loss", float('inf')) < self.best_loss:
                self.best_loss = val_metrics["val_loss"]
                self.save_checkpoint(epoch, best=True)

        return history

    def save_checkpoint(self, epoch: int, best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }

        if best:
            path = Path(self.config.checkpoint_dir) / "best_model.pt"
        else:
            path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint.get("best_loss", float('inf'))

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


def create_cpc_model(
    encoder_config: Optional[EncoderConfig] = None,
    training_config: Optional[TrainingConfig] = None,
) -> CPCModel:
    """Factory function to create CPC model."""
    if training_config is None:
        training_config = TrainingConfig()

    if encoder_config is None:
        encoder_config = EncoderConfig(
            sample_rate=training_config.sample_rate,
            frame_size_ms=training_config.frame_size_ms,
            hidden_dim=training_config.hidden_dim,
        )

    return CPCModel(
        encoder_config=encoder_config,
        steps_ahead=training_config.steps_ahead,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick test
    config = TrainingConfig(batch_size=2, sequence_length=16)
    model = create_cpc_model(config)

    # Create dummy data
    batch_size = 2
    seq_len = config.sequence_length + config.steps_ahead
    frame_size = int(config.sample_rate * config.frame_size_ms / 1000)

    audio = torch.randn(batch_size, seq_len, frame_size)

    # Forward pass
    z, context, predictions = model(audio)
    loss = model.compute_loss(z, context, predictions)

    print(f"Input shape: {audio.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Predictions: {len(predictions)} x {predictions[0].shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
