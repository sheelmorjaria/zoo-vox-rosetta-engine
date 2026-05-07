#!/usr/bin/env python3
"""
DDSP Training Pipeline - Module 2 (v1.6.0)

Training pipeline for the DDSP Decoder that learns to map 112D RosettaFeatures
to DDSP control parameters for audio synthesis.

The training uses cached audio segments with extracted 112D features as input
and the original audio as the target. The decoder is trained end-to-end using
multi-scale spectral loss.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .ddsp_decoder import DDSPDecoder, DDSPDecoderConfig
from .multiscale_spectral_loss import CombinedLoss

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================


@dataclass
class VocalizationSegment:
    """A single training example."""

    features_112d: np.ndarray  # Shape: (112,)
    audio: np.ndarray  # Shape: (samples,)
    sample_rate: int
    f0_hz: Optional[np.ndarray] = None  # Optional F0 contour
    duration_ms: float = 0.0
    species: Optional[str] = None
    cluster_id: Optional[int] = None

    def __post_init__(self):
        if self.duration_ms == 0.0 and self.sample_rate > 0:
            self.duration_ms = len(self.audio) / self.sample_rate * 1000.0


class VocalizationDataset(Dataset):
    """
    Dataset for training DDSP decoder.

    Loads cached segments with 112D features and corresponding audio.
    Each item returns (features_112d, audio) for training.
    """

    def __init__(
        self,
        segments_json: str,
        audio_dir: Optional[str] = None,
        sample_rate: int = 48000,
        max_duration_ms: float = 500.0,
        augment: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            segments_json: Path to JSON file with cached segments
            audio_dir: Directory containing audio files (if using external audio)
            sample_rate: Target sample rate
            max_duration_ms: Maximum duration for segments (longer ones are cropped)
            augment: Apply data augmentation
        """
        self.sample_rate = sample_rate
        self.max_duration_ms = max_duration_ms
        self.augment = augment
        self.segments: List[VocalizationSegment] = []

        # Load segments from JSON
        self._load_segments(segments_json, audio_dir)

        logger.info(f"Loaded {len(self.segments)} segments for training")

    def _load_segments(self, segments_json: str, audio_dir: Optional[str]):
        """Load segments from JSON file."""
        with open(segments_json, "r") as f:
            data = json.load(f)

        for item in data.get("segments", []):
            # Extract features
            features = np.array(item["features_112d"], dtype=np.float32)
            if len(features) != 112:
                logger.warning(f"Skipping segment with {len(features)} features (expected 112)")
                continue

            # Load or generate audio
            if "audio_path" in item and audio_dir:
                audio_path = os.path.join(audio_dir, item["audio_path"])
                if os.path.exists(audio_path):
                    audio = np.load(audio_path)
                else:
                    # Generate placeholder audio for training setup
                    max_samples = int(self.max_duration_ms / 1000 * self.sample_rate)
                    audio = np.random.randn(max_samples).astype(np.float32) * 0.1
            else:
                # Use provided audio data or generate placeholder
                if "audio" in item:
                    audio = np.array(item["audio"], dtype=np.float32)
                else:
                    max_samples = int(self.max_duration_ms / 1000 * self.sample_rate)
                    audio = np.random.randn(max_samples).astype(np.float32) * 0.1

            # Resample if needed
            if item.get("sample_rate", self.sample_rate) != self.sample_rate:
                # Simple resampling (placeholder - use librosa/presets for real)
                ratio = self.sample_rate / item.get("sample_rate", self.sample_rate)
                audio = np.interp(
                    np.linspace(0, len(audio), int(len(audio) * ratio)),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)

            segment = VocalizationSegment(
                features_112d=features,
                audio=audio,
                sample_rate=self.sample_rate,
                f0_hz=np.array(item.get("f0_hz", []), dtype=np.float32)
                if "f0_hz" in item
                else None,
                species=item.get("species"),
                cluster_id=item.get("cluster_id"),
            )

            self.segments.append(segment)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training example.

        Returns:
            features_112d: (112,) feature vector
            audio: (samples,) audio waveform
        """
        segment = self.segments[idx]

        features = torch.from_numpy(segment.features_112d).float()
        audio = torch.from_numpy(segment.audio).float()

        # Crop audio if too long
        max_samples = int(self.max_duration_ms / 1000 * self.sample_rate)
        if len(audio) > max_samples:
            start = np.random.randint(0, len(audio) - max_samples + 1)
            audio = audio[start : start + max_samples]

        # Apply augmentation if enabled
        if self.augment:
            audio = self._augment_audio(audio)

        return features, audio

    def _augment_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to audio."""
        # Random gain
        gain = 10 ** (np.random.uniform(-1, 1) * 0.05)  # +/- 0.5 dB
        audio = audio * gain

        # Random phase shift (circular shift)
        if np.random.random() > 0.5:
            shift = np.random.randint(0, len(audio))
            audio = torch.roll(audio, shift, dims=0)

        return audio


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing/training without real data.

    Generates random 112D features and corresponding synthetic audio
    for initial testing of the training pipeline.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        sample_rate: int = 48000,
        duration_ms: float = 200.0,
    ):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms
        self.audio_length = int(sample_rate * duration_ms / 1000)

        logger.info(f"Created synthetic dataset with {num_samples} samples")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random features
        features = torch.randn(112).float()

        # Generate synthetic audio (harmonic series)
        t = torch.linspace(0, self.duration_ms / 1000, self.audio_length)
        audio = torch.zeros(self.audio_length)

        # Use first few features to determine synthesis parameters
        f0 = 4000 + features[0] * 2000  # Base frequency from feature 0
        num_harmonics = int(10 + features[1] * 20)  # Number of harmonics

        for h in range(1, num_harmonics + 1):
            amplitude = 1.0 / h  # 1/f amplitude decay
            phase = 2 * math.pi * f0 * h * t
            audio += amplitude * torch.sin(phase)

        # Normalize
        audio = audio / (audio.abs().max() + 1e-8) * 0.5

        return features, audio


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for DDSP decoder training."""

    # Data
    segments_json: str = ""
    audio_dir: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 4

    # Model
    hidden_dim: int = 256
    num_harmonics: int = 60
    num_noise_bands: int = 5
    dropout: float = 0.1

    # Training
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    # Loss
    spectral_loss_weight: float = 1.0
    time_loss_weight: float = 0.1

    # Validation
    val_split: float = 0.1
    early_stopping_patience: int = 10

    # Checkpointing
    checkpoint_dir: str = "checkpoints/ddsp_decoder"
    save_every: int = 10

    # Logging
    log_every: int = 100

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Synthetic data (for testing)
    use_synthetic_data: bool = False
    synthetic_samples: int = 1000


# =============================================================================
# Trainer
# =============================================================================


class DDSPDecoderTrainer:
    """
    Trainer for DDSP Decoder.

    Handles the complete training loop including:
    - Data loading and preprocessing
    - Forward/backward pass
    - Validation
    - Checkpointing
    - Early stopping
    """

    def __init__(
        self,
        model: DDSPDecoder,
        config: TrainingConfig,
    ):
        """
        Initialize trainer.

        Args:
            model: DDSPDecoder model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # Move model to device
        self.model.to(self.device)

        # Setup loss function
        self.loss_fn = CombinedLoss(
            spectral_weight=config.spectral_loss_weight,
            time_weight=config.time_loss_weight,
        )

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            patience=5,
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        logger.info(f"Trainer initialized on device: {self.device}")

    def setup_data(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Setup train and validation dataloaders.

        Returns:
            train_loader: Training data loader
            val_loader: Validation data loader (None if val_split=0)
        """
        # Create dataset
        if self.config.use_synthetic_data:
            dataset = SyntheticDataset(
                num_samples=self.config.synthetic_samples,
                duration_ms=200.0,
            )
        else:
            dataset = VocalizationDataset(
                segments_json=self.config.segments_json,
                audio_dir=self.config.audio_dir,
                augment=True,
            )

        # Split into train/val
        val_size = int(len(dataset) * self.config.val_split)
        train_size = len(dataset) - val_size

        if val_size > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset,
                [train_size, val_size],
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
        else:
            train_dataset = dataset
            val_loader = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        logger.info(f"Data loaders: train={train_size}, val={val_size if val_size else 0}")

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_spectral = 0.0
        total_time = 0.0
        num_batches = 0

        for batch_idx, (features, audio) in enumerate(train_loader):
            # Move to device
            features = features.to(self.device)
            audio = audio.to(self.device)

            # Forward pass through decoder
            harmonic_amps, noise_mags = self.model(features)

            # For now, use placeholder synthesis (would connect to DDSPSynthesizer)
            # The actual synthesis will be implemented in Module 3
            # For training, we use a simple reconstruction from features  # noqa: E501
            # In Module 3, this will be: pred_audio = self.synthesizer(...)  # noqa: E501

            # Placeholder: use features to generate pseudo-audio with gradients
            batch_size = features.shape[0]
            sample_length = audio.shape[1]

            # Create pseudo-audio from decoder outputs (with gradients)
            # Use harmonic amplitudes to shape a simple signal
            pred_audio = torch.zeros(batch_size, sample_length, device=self.device)
            for i in range(batch_size):
                # Create a simple harmonic signal from the amplitude distribution
                t = torch.linspace(0, 1, sample_length, device=self.device)
                signal = torch.zeros(sample_length, device=self.device)
                for h in range(min(10, self.model.num_harmonics)):  # Use first 10 harmonics
                    amp = harmonic_amps[i, h]
                    signal = signal + amp * torch.sin(2 * math.pi * (h + 1) * t)
                pred_audio[i] = signal

            # Normalize
            pred_audio = pred_audio / (pred_audio.abs().max(dim=1, keepdim=True)[0] + 1e-8)

            # Add batch dim to audio if needed
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)  # (B, 1, T)
            pred_audio = pred_audio.unsqueeze(1)  # (B, 1, T)

            # Compute loss
            loss, loss_dict = self.loss_fn(pred_audio, audio)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss_dict["total"]
            total_spectral += loss_dict["spectral"]
            total_time += loss_dict["time"]
            num_batches += 1

            # Log progress
            if batch_idx % self.config.log_every == 0:
                logger.info(
                    f"Epoch {self.current_epoch} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss_dict['total']:.6f} "
                    f"(spectral: {loss_dict['spectral']:.6f}, "
                    f"time: {loss_dict['time']:.6f})"
                )

        return {
            "loss": total_loss / num_batches,
            "spectral": total_spectral / num_batches,
            "time": total_time / num_batches,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for features, audio in val_loader:
            features = features.to(self.device)
            audio = audio.to(self.device)

            # Forward pass
            harmonic_amps, noise_mags = self.model(features)

            # Placeholder synthesis (Module 3)
            batch_size = features.shape[0]
            sample_length = audio.shape[1]

            # Create pseudo-audio from decoder outputs
            pred_audio = torch.zeros(batch_size, sample_length, device=self.device)
            for i in range(batch_size):
                t = torch.linspace(0, 1, sample_length, device=self.device)
                signal = torch.zeros(sample_length, device=self.device)
                for h in range(min(10, self.model.num_harmonics)):
                    amp = harmonic_amps[i, h]
                    signal = signal + amp * torch.sin(2 * math.pi * (h + 1) * t)
                pred_audio[i] = signal

            pred_audio = pred_audio / (pred_audio.abs().max(dim=1, keepdim=True)[0] + 1e-8)

            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
            pred_audio = pred_audio.unsqueeze(1)

            # Compute loss
            loss, _ = self.loss_fn(pred_audio, audio)

            total_loss += loss.item()
            num_batches += 1

        return {"loss": total_loss / num_batches}

    def train(self):
        """Run the complete training loop."""
        train_loader, val_loader = self.setup_data()

        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                logger.info(
                    f"Epoch {epoch} | "
                    f"Train Loss: {train_metrics['loss']:.6f} | "
                    f"Val Loss: {val_metrics['loss']:.6f}"
                )

                # Learning rate scheduling
                self.scheduler.step(val_metrics["loss"])

                # Early stopping check
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.patience_counter = 0
                    self.save_checkpoint("best.pt")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
            else:
                logger.info(f"Epoch {epoch} | Train Loss: {train_metrics['loss']:.6f}")

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")

        logger.info("Training complete!")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
        }

        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str) -> "DDSPDecoderTrainer":
        """Load trainer from checkpoint."""
        checkpoint = torch.load(checkpoint_path)

        # Recreate config
        config = TrainingConfig(**checkpoint["config"])

        # Recreate model
        model_config = DDSPDecoderConfig(
            hidden_dim=config.hidden_dim,
            num_harmonics=config.num_harmonics,
            num_noise_bands=config.num_noise_bands,
            dropout=config.dropout,
        )
        model = DDSPDecoder(config=model_config)

        # Create trainer
        trainer = cls(model, config)

        # Load state
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.current_epoch = checkpoint["epoch"]
        trainer.best_val_loss = checkpoint["best_val_loss"]

        logger.info(f"Loaded checkpoint from epoch {trainer.current_epoch}")
        return trainer


# =============================================================================
# Convenience Functions
# =============================================================================


def train_decoder(
    segments_json: str = "",
    audio_dir: Optional[str] = None,
    use_synthetic_data: bool = True,
    **kwargs,
) -> DDSPDecoder:
    """
    Train a DDSP decoder with default settings.

    Args:
        segments_json: Path to cached segments JSON
        audio_dir: Directory containing audio files
        use_synthetic_data: Use synthetic data for testing
        **kwargs: Additional training config overrides

    Returns:
        Trained DDSPDecoder model
    """
    # Create config
    config = TrainingConfig(
        segments_json=segments_json,
        audio_dir=audio_dir,
        use_synthetic_data=use_synthetic_data,
        **kwargs,
    )

    # Create model
    model_config = DDSPDecoderConfig(
        hidden_dim=config.hidden_dim,
        num_harmonics=config.num_harmonics,
        num_noise_bands=config.num_noise_bands,
        dropout=config.dropout,
    )
    model = DDSPDecoder(config=model_config)

    # Create trainer
    trainer = DDSPDecoderTrainer(model, config)

    # Train
    trainer.train()

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Train with synthetic data (for testing)
    print("\n=== Training DDSP Decoder with Synthetic Data ===\n")
    model = train_decoder(
        use_synthetic_data=True,
        synthetic_samples=100,
        num_epochs=5,
        batch_size=8,
        checkpoint_dir="checkpoints/ddsp_decoder_test",
    )

    print("\n=== Training Complete ===")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
