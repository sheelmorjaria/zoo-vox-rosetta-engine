#!/usr/bin/env python3
"""
Neural Post-Filter Training Pipeline - Module 4 (v1.6.0)

Training pipeline for the NeuralPostFilter that refines DDSP output to match
real bat vocalizations.

The training uses:
1. Cached audio segments with extracted 112D features as input
2. Pre-trained DDSP decoder + synthesizer to generate baseline audio
3. Post-filter trained to map DDSP output → real audio
4. Multi-scale spectral loss + perceptual loss for training

This is Option B from the neural vocoder analysis: retain DDSP's differentiability
and acoustic structure, add a lightweight refinement network for realistic textures.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Import NeuralPostFilter from ddsp_agent
from realtime.ddsp_agent import NeuralPostFilter

from .ddsp_decoder import DDSPDecoder
from .ddsp_synthesis import DDSPSynthesizer
from .multiscale_spectral_loss import MultiScaleSpectralLoss

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================


@dataclass
class PostFilterTrainingExample:
    """A single training example for post-filter training."""

    features_112d: np.ndarray  # Shape: (112,)
    ddsp_audio: np.ndarray  # Shape: (samples,) - DDSP generated audio
    target_audio: np.ndarray  # Shape: (samples,) - Real vocalization
    harmonic_amps: np.ndarray  # Shape: (60,) - DDSP harmonic amplitudes
    noise_mags: np.ndarray  # Shape: (5,) - DDSP noise magnitudes
    f0_hz: float  # Fundamental frequency used for synthesis


class PostFilterDataset(Dataset):
    """
    Dataset for training neural post-filter.

    Each item contains:
    - features_112d: Input features for DDSP synthesis
    - ddsp_audio: DDSP-generated audio (input to post-filter)
    - target_audio: Real bat vocalization (target for post-filter)
    - harmonic_amps, noise_mags: DDSP parameters (conditioning for post-filter)
    """

    def __init__(
        self,
        segments_json: str,
        audio_dir: Optional[str] = None,
        ddsp_decoder: Optional[DDSPDecoder] = None,
        ddsp_synthesizer: Optional[DDSPSynthesizer] = None,
        sample_rate: int = 48000,
        duration_ms: float = 200.0,
        device: str = "cpu",
        regenerate_ddsp: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            segments_json: Path to JSON file with cached segments
            audio_dir: Directory containing audio files
            ddsp_decoder: Pre-trained DDSP decoder (for generating DDSP audio)
            ddsp_synthesizer: DDSP synthesizer (for generating DDSP audio)
            sample_rate: Target sample rate
            duration_ms: Target duration for all segments
            device: Device for DDSP synthesis
            regenerate_ddsp: Regenerate DDSP audio even if cached
        """
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms
        self.target_samples = int(sample_rate * duration_ms / 1000)
        self.device = torch.device(device)
        self.regenerate_ddsp = regenerate_ddsp

        # Store DDSP models for synthesis
        self.ddsp_decoder = ddsp_decoder
        self.ddsp_synthesizer = ddsp_synthesizer

        # Load or generate examples
        self.examples: List[PostFilterTrainingExample] = []
        self._load_or_generate_examples(segments_json, audio_dir)

        logger.info(f"Loaded {len(self.examples)} post-filter training examples")

    def _load_or_generate_examples(self, segments_json: str, audio_dir: Optional[str]):
        """Load examples from JSON or generate from DDSP models."""
        with open(segments_json, "r") as f:
            data = json.load(f)

        # Check for cached post-filter data
        cache_file = segments_json.replace(".json", "_post_filter_cache.npy")

        if not self.regenerate_ddsp and os.path.exists(cache_file):
            logger.info(f"Loading cached post-filter examples from {cache_file}")
            self.examples = np.load(cache_file, allow_pickle=True).tolist()
            return

        # Generate examples from segments
        logger.info("Generating post-filter training examples...")

        for item in data.get("segments", []):
            # Extract features
            features = np.array(item["features_112d"], dtype=np.float32)
            if len(features) != 112:
                continue

            # Load or generate target audio
            if "audio_path" in item and audio_dir:
                audio_path = os.path.join(audio_dir, item["audio_path"])
                if os.path.exists(audio_path):
                    target_audio = np.load(audio_path)
                else:
                    continue
            elif "audio" in item:
                target_audio = np.array(item["audio"], dtype=np.float32)
            else:
                continue

            # Resample/trim to target length
            if len(target_audio) > self.target_samples:
                # Crop to center
                start = (len(target_audio) - self.target_samples) // 2
                target_audio = target_audio[start : start + self.target_samples]
            elif len(target_audio) < self.target_samples:
                # Pad with zeros
                padded = np.zeros(self.target_samples, dtype=np.float32)
                start = (self.target_samples - len(target_audio)) // 2
                padded[start : start + len(target_audio)] = target_audio
                target_audio = padded

            # Normalize
            target_audio = target_audio / (np.abs(target_audio).max() + 1e-8)

            # Generate DDSP audio if models provided
            if self.ddsp_decoder is not None and self.ddsp_synthesizer is not None:
                ddsp_audio, harmonic_amps, noise_mags, f0 = self._generate_ddsp_audio(features)
            else:
                # Use cached DDSP data if available
                ddsp_audio = np.array(
                    item.get("ddsp_audio", np.random.randn(self.target_samples) * 0.1)
                )
                harmonic_amps = np.array(item.get("harmonic_amps", np.random.rand(60)))
                noise_mags = np.array(item.get("noise_mags", np.random.rand(5)))
                f0 = item.get("f0_hz", 6000.0)

            # Normalize DDSP audio
            ddsp_audio = ddsp_audio / (np.abs(ddsp_audio).max() + 1e-8)

            example = PostFilterTrainingExample(
                features_112d=features,
                ddsp_audio=ddsp_audio.astype(np.float32),
                target_audio=target_audio.astype(np.float32),
                harmonic_amps=harmonic_amps.astype(np.float32),
                noise_mags=noise_mags.astype(np.float32),
                f0_hz=f0,
            )

            self.examples.append(example)

        # Cache examples for faster loading next time
        if self.examples:
            logger.info(f"Caching {len(self.examples)} examples to {cache_file}")
            np.save(cache_file, self.examples, allow_pickle=True)

    def _generate_ddsp_audio(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Generate DDSP audio from features."""
        with torch.no_grad():
            # Convert to tensor
            features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)

            # Run decoder
            harmonic_amps, noise_mags = self.ddsp_decoder(features_tensor)

            # Derive F0 from features
            f0 = 6000 + features[0] * 2000  # Base frequency from feature 0
            f0 = np.clip(f0, 3000, 15000)

            # Create F0 trajectory
            n_frames = int(self.duration_ms / 10)  # 10ms frames
            f0_tensor = torch.ones(1, n_frames, device=self.device) * f0

            # Expand parameters to time dimension
            harmonic_amps = harmonic_amps.unsqueeze(1).expand(1, n_frames, -1)
            noise_mags = noise_mags.unsqueeze(1).expand(1, n_frames, -1)

            # Run synthesizer
            audio, _ = self.ddsp_synthesizer(f0_tensor, harmonic_amps, noise_mags)

            # Extract to numpy
            ddsp_audio = audio.squeeze(0).cpu().numpy()
            harmonic_amps_np = harmonic_amps.mean(dim=1).squeeze(0).cpu().numpy()
            noise_mags_np = noise_mags.mean(dim=1).squeeze(0).cpu().numpy()

            # Trim/pad to target length
            if len(ddsp_audio) > self.target_samples:
                ddsp_audio = ddsp_audio[: self.target_samples]
            elif len(ddsp_audio) < self.target_samples:
                padded = np.zeros(self.target_samples, dtype=np.float32)
                padded[: len(ddsp_audio)] = ddsp_audio
                ddsp_audio = padded

            return ddsp_audio, harmonic_amps_np, noise_mags_np, float(f0)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training example.

        Returns:
            ddsp_audio: (samples,) DDSP-generated audio
            target_audio: (samples,) Real vocalization
            harmonic_amps: (60,) Harmonic amplitudes (conditioning)
            noise_mags: (5,) Noise magnitudes (conditioning)
        """
        example = self.examples[idx]

        ddsp_audio = torch.from_numpy(example.ddsp_audio).float()
        target_audio = torch.from_numpy(example.target_audio).float()
        harmonic_amps = torch.from_numpy(example.harmonic_amps).float()
        noise_mags = torch.from_numpy(example.noise_mags).float()

        return ddsp_audio, target_audio, harmonic_amps, noise_mags


class SyntheticPostFilterDataset(Dataset):
    """
    Synthetic dataset for testing post-filter training.

    Generates synthetic DDSP-like audio and target audio with subtle differences
    to test the training pipeline.
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

        logger.info(f"Created synthetic post-filter dataset with {num_samples} samples")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Generate random parameters
        f0 = 4000 + np.random.randn() * 2000
        harmonic_amps = F.softmax(torch.randn(60), dim=0)
        noise_mags = torch.rand(5)

        # Generate DDSP-like audio (clean harmonics)
        t = torch.linspace(0, self.duration_ms / 1000, self.audio_length)
        ddsp_audio = torch.zeros(self.audio_length)

        for h in range(1, 31):  # 30 harmonics
            amp = harmonic_amps[h - 1].item() if h <= 60 else 0.1 / h
            phase = 2 * np.pi * f0 * h * t.numpy()
            ddsp_audio += amp * torch.from_numpy(np.sin(phase))

        # Generate target audio (DDSP + realistic texture)
        # Add subtle modulation and noise
        target_audio = ddsp_audio.clone()

        # Add frequency modulation
        modulation = 0.02 * torch.sin(2 * np.pi * 10 * t)  # 10Hz vibrato
        target_audio = target_audio * (1 + modulation)

        # Add filtered noise
        noise = torch.randn(self.audio_length) * 0.05
        # Simple lowpass filter
        kernel_size = 51
        kernel = torch.ones(kernel_size) / kernel_size
        noise_filtered = F.conv1d(
            noise.view(1, 1, -1),
            kernel.view(1, 1, -1),
            padding=kernel_size // 2,
        ).squeeze()
        target_audio += noise_filtered[: self.audio_length] * 0.1

        # Normalize
        ddsp_audio = ddsp_audio / (ddsp_audio.abs().max() + 1e-8) * 0.8
        target_audio = target_audio / (target_audio.abs().max() + 1e-8) * 0.8

        return ddsp_audio, target_audio, harmonic_amps, noise_mags


# =============================================================================
# Perceptual Loss
# =============================================================================


class PerceptualLoss(nn.Module):
    """
    Perceptual loss for post-filter training.

    Uses a pretrained feature extractor (or simple spectral features) to compute
    perceptual similarity between predicted and target audio.
    """

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate

        # Use multi-scale spectral loss as perceptual loss
        # (In production, could use a pretrained model like VGGish or Enclap)
        self.spectral_loss = MultiScaleSpectralLoss(
            frame_lengths=[512, 1024, 2048, 4096],
            l1_weight=1.0,
            l2_weight=0.5,
        )

    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        # Ensure 3D input
        if pred_audio.dim() == 2:
            pred_audio = pred_audio.unsqueeze(1)
        if target_audio.dim() == 2:
            target_audio = target_audio.unsqueeze(1)

        return self.spectral_loss(pred_audio, target_audio)


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class PostFilterTrainingConfig:
    """Configuration for post-filter training."""

    # Data
    segments_json: str = ""
    audio_dir: Optional[str] = None
    batch_size: int = 16
    num_workers: int = 4

    # Model
    num_harmonics: int = 60
    num_noise_bands: int = 5

    # Training
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    # Loss weights
    spectral_loss_weight: float = 1.0
    perceptual_loss_weight: float = 0.5
    time_loss_weight: float = 0.1

    # Validation
    val_split: float = 0.1
    early_stopping_patience: int = 10

    # Checkpointing
    checkpoint_dir: str = "checkpoints/post_filter"
    save_every: int = 5

    # Logging
    log_every: int = 50

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # DDSP models (for generating training data)
    ddsp_decoder_path: Optional[str] = None
    ddsp_synthesizer_path: Optional[str] = None

    # Synthetic data (for testing)
    use_synthetic_data: bool = False
    synthetic_samples: int = 1000

    # Data generation
    duration_ms: float = 200.0
    regenerate_ddsp: bool = False


# =============================================================================
# Trainer
# =============================================================================


class PostFilterTrainer:
    """
    Trainer for neural post-filter.

    Training flow:
    1. Load cached segments with 112D features and real audio
    2. Generate DDSP audio using decoder + synthesizer (or load cached)
    3. Train post-filter: DDSP audio + params → refined audio
    4. Loss: multi-scale spectral (DDSP vs real)
    """

    def __init__(
        self,
        model: NeuralPostFilter,
        config: PostFilterTrainingConfig,
    ):
        """
        Initialize trainer.

        Args:
            model: NeuralPostFilter model to train
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # Move model to device
        self.model.to(self.device)

        # Setup loss functions
        self.spectral_loss = MultiScaleSpectralLoss(
            l1_weight=config.spectral_loss_weight,
            l2_weight=0.5,
        )
        self.perceptual_loss = PerceptualLoss(sample_rate=48000)

        # Combined loss
        def combined_loss(pred, target):
            spectral = self.spectral_loss(pred, target)
            perceptual = self.perceptual_loss(pred, target)
            time_loss = F.l1_loss(pred, target)

            return (
                config.spectral_loss_weight * spectral
                + config.perceptual_loss_weight * perceptual
                + config.time_loss_weight * time_loss
            )

        self.loss_fn = combined_loss

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

        logger.info(f"PostFilterTrainer initialized on device: {self.device}")

    def setup_data(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Setup train and validation dataloaders."""
        # Load DDSP models if provided
        ddsp_decoder = None
        ddsp_synthesizer = None

        if self.config.ddsp_decoder_path and os.path.exists(self.config.ddsp_decoder_path):
            logger.info(f"Loading DDSP decoder from {self.config.ddsp_decoder_path}")
            checkpoint = torch.load(self.config.ddsp_decoder_path, map_location=self.device)
            ddsp_decoder = DDSPDecoder()
            ddsp_decoder.load_state_dict(checkpoint["model_state_dict"])
            ddsp_decoder.to(self.device)
            ddsp_decoder.eval()

        if self.config.ddsp_synthesizer_path:
            logger.info(f"Loading DDSP synthesizer from {self.config.ddsp_synthesizer_path}")
            ddsp_synthesizer = DDSPSynthesizer.load(self.config.ddsp_synthesizer_path)
            ddsp_synthesizer.to(self.device)
            ddsp_synthesizer.eval()
        else:
            # Create default synthesizer
            ddsp_synthesizer = DDSPSynthesizer(
                sample_rate=48000,
                num_harmonics=self.config.num_harmonics,
                num_noise_bands=self.config.num_noise_bands,
            ).to(self.device)
            ddsp_synthesizer.eval()

        # Create dataset
        if self.config.use_synthetic_data:
            dataset = SyntheticPostFilterDataset(
                num_samples=self.config.synthetic_samples,
                duration_ms=self.config.duration_ms,
            )
        else:
            dataset = PostFilterDataset(
                segments_json=self.config.segments_json,
                audio_dir=self.config.audio_dir,
                ddsp_decoder=ddsp_decoder,
                ddsp_synthesizer=ddsp_synthesizer,
                duration_ms=self.config.duration_ms,
                device=self.config.device,
                regenerate_ddsp=self.config.regenerate_ddsp,
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
        num_batches = 0

        for batch_idx, (ddsp_audio, target_audio, harmonic_amps, noise_mags) in enumerate(
            train_loader
        ):
            # Move to device
            ddsp_audio = ddsp_audio.to(self.device)
            target_audio = target_audio.to(self.device)
            harmonic_amps = harmonic_amps.to(self.device)
            noise_mags = noise_mags.to(self.device)

            # Forward pass through post-filter
            refined_audio = self.model(ddsp_audio, harmonic_amps, noise_mags)

            # Ensure shapes match
            min_length = min(refined_audio.shape[-1], target_audio.shape[-1])
            refined_audio = refined_audio[..., :min_length]
            target_audio = target_audio[..., :min_length]

            # Compute loss
            loss = self.loss_fn(refined_audio, target_audio)

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
            total_loss += loss.item()
            num_batches += 1

            # Log progress
            if batch_idx % self.config.log_every == 0:
                logger.info(
                    f"Epoch {self.current_epoch} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Loss: {loss.item():.6f}"
                )

        return {"loss": total_loss / num_batches}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for ddsp_audio, target_audio, harmonic_amps, noise_mags in val_loader:
            ddsp_audio = ddsp_audio.to(self.device)
            target_audio = target_audio.to(self.device)
            harmonic_amps = harmonic_amps.to(self.device)
            noise_mags = noise_mags.to(self.device)

            # Forward pass
            refined_audio = self.model(ddsp_audio, harmonic_amps, noise_mags)

            min_length = min(refined_audio.shape[-1], target_audio.shape[-1])
            refined_audio = refined_audio[..., :min_length]
            target_audio = target_audio[..., :min_length]

            # Compute loss
            loss = self.loss_fn(refined_audio, target_audio)

            total_loss += loss.item()
            num_batches += 1

        return {"loss": total_loss / num_batches}

    def train(self):
        """Run the complete training loop."""
        train_loader, val_loader = self.setup_data()

        logger.info(f"Starting post-filter training for {self.config.num_epochs} epochs")

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
                    logger.info(f"New best validation loss: {self.best_val_loss:.6f}")
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
    def load_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "PostFilterTrainer":
        """Load trainer from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Recreate config
        config = PostFilterTrainingConfig(**checkpoint["config"])

        # Recreate model
        model = NeuralPostFilter(
            num_harmonics=config.num_harmonics,
            num_noise_bands=config.num_noise_bands,
        )

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


def train_post_filter(
    segments_json: str = "",
    audio_dir: Optional[str] = None,
    use_synthetic_data: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
) -> NeuralPostFilter:
    """
    Train a neural post-filter with default settings.

    Args:
        segments_json: Path to cached segments JSON
        audio_dir: Directory containing audio files
        use_synthetic_data: Use synthetic data for testing
        device: Device to train on
        **kwargs: Additional training config overrides

    Returns:
        Trained NeuralPostFilter model
    """
    # Create config
    config = PostFilterTrainingConfig(
        segments_json=segments_json,
        audio_dir=audio_dir,
        use_synthetic_data=use_synthetic_data,
        device=device,
        **kwargs,
    )

    # Create model
    model = NeuralPostFilter(
        num_harmonics=config.num_harmonics,
        num_noise_bands=config.num_noise_bands,
    )

    # Create trainer
    trainer = PostFilterTrainer(model, config)

    # Train
    trainer.train()

    return model


def export_post_filter_for_jetson(
    model: NeuralPostFilter,
    output_path: str = "exports/post_filter/post_filter.onnx",
    device: str = "cuda",
):
    """
    Export trained post-filter to ONNX for Jetson deployment.

    Args:
        model: Trained NeuralPostFilter model
        output_path: Path to save ONNX model
        device: Device used for export
    """
    model.eval()
    model.to(device)

    # Create dummy inputs
    batch_size = 1
    audio_length = 4800  # 100ms at 48kHz
    dummy_audio = torch.randn(batch_size, audio_length)
    dummy_harmonic = torch.randn(batch_size, 60)
    dummy_noise = torch.rand(batch_size, 5)

    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_audio, dummy_harmonic, dummy_noise),
        output_path,
        input_names=["ddsp_audio", "harmonic_amps", "noise_mags"],
        output_names=["refined_audio"],
        dynamic_axes={
            "ddsp_audio": {0: "batch_size", 1: "audio_length"},
            "refined_audio": {0: "batch_size", 1: "audio_length"},
        },
        opset_version=14,
    )

    logger.info(f"Exported post-filter to {output_path}")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("\n=== Neural Post-Filter Training Pipeline ===\n")
    print("Option B: Lightweight refinement on top of DDSP synthesis")
    print("  - Retains DDSP differentiability")
    print("  - Adds realistic textures via lightweight CNN")
    print("  - Target: <3ms latency on Jetson Orin\n")

    # Train with synthetic data (for testing)
    model = train_post_filter(
        use_synthetic_data=True,
        synthetic_samples=200,
        num_epochs=10,
        batch_size=8,
        checkpoint_dir="checkpoints/post_filter_test",
        device="cpu",  # Use CPU for testing
    )

    print("\n=== Training Complete ===")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Export for Jetson
    export_dir = "exports/post_filter"
    os.makedirs(export_dir, exist_ok=True)

    export_post_filter_for_jetson(
        model,
        output_path=f"{export_dir}/post_filter.onnx",
        device="cpu",
    )

    print(f"\nExported post-filter to {export_dir}/post_filter.onnx")
