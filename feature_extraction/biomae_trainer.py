#!/usr/bin/env python3
"""
BioMAE Trainer: Self-Supervised Training with Masked Autoencoding

Training loop for learning bioacoustic features from unlabelled spectrograms.
Uses 75% masking ratio following Audio MAE best practices.

Key features:
- 75% masking ratio (validated by MAE-AST research)
- Data augmentation: time stretch, pitch shift, noise injection
- Spectrogram reconstruction loss on masked patches only
- Checkpointing and logging support

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from feature_extraction.biomae import BioMAEModel, EncoderConfig, DecoderConfig
from feature_extraction.bio_spectrogram import (
    UltrasonicSpectrogram,
    SpectrogramConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for BioMAE training."""
    # Data
    sample_rate: int = 96000
    n_fft: int = 1024
    hop_length: int = 240
    img_size: tuple = (128, 128)

    # Model architecture
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 4
    output_dim: int = 112

    # Training
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10

    # MAE specific
    mask_ratio: float = 0.75  # Validated by Audio MAE research

    # Augmentation
    time_stretch_range: tuple = (0.8, 1.2)
    pitch_shift_range: tuple = -2, 2  # Semitones
    noise_level: float = 0.01

    # Paths
    checkpoint_dir: str = "checkpoints/biomae"
    log_interval: int = 10
    save_interval: int = 5  # Save every N epochs

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True  # FP16 training


class BioacousticAugmentation:
    """
    Data augmentation for bioacoustic spectrograms.

    Augmentations:
    1. Time stretching: Simulates vocalization rate variation
    2. Frequency shifting: Simulates pitch/size variation
    3. Noise injection: Improves robustness to recording noise
    4. Time masking: Simulates brief dropouts
    5. Frequency masking: Simulates spectral notches
    """

    def __init__(
        self,
        time_stretch_range: tuple = (0.8, 1.2),
        freq_shift_range: tuple = (-4, 4),  # STFT bins
        noise_level: float = 0.01,
        time_mask_param: int = 10,
        freq_mask_param: int = 8,
    ):
        self.time_stretch_range = time_stretch_range
        self.freq_shift_range = freq_shift_range
        self.noise_level = noise_level
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to spectrogram.

        Args:
            spec: Spectrogram (Freq, Time)

        Returns:
            Augmented spectrogram
        """
        aug_spec = spec.clone()

        # Time stretch (via interpolation)
        if torch.rand(1).item() < 0.5:
            factor = torch.empty(1).uniform_(*self.time_stretch_range).item()
            aug_spec = F.interpolate(
                aug_spec.unsqueeze(0).unsqueeze(0),
                scale_factor=(1.0, factor),
                mode='bilinear',
                align_corners=False,
            ).squeeze()

        # Frequency shift (roll along frequency axis)
        if torch.rand(1).item() < 0.5:
            shift = torch.empty(1).uniform_(*self.freq_shift_range).int().item()
            aug_spec = torch.roll(aug_spec, shifts=shift, dims=0)

        # Noise injection
        if self.noise_level > 0:
            noise = torch.randn_like(aug_spec) * self.noise_level
            aug_spec = aug_spec + noise

        # SpecAugment-style masking
        if torch.rand(1).item() < 0.5:
            # Time masking
            t = torch.randint(0, self.time_mask_param, (1,)).item()
            t0 = torch.randint(0, aug_spec.shape[1] - t, (1,)).item()
            aug_spec[:, t0:t0+t] = 0

        if torch.rand(1).item() < 0.5:
            # Frequency masking
            f = torch.randint(0, self.freq_mask_param, (1,)).item()
            f0 = torch.randint(0, aug_spec.shape[0] - f, (1,)).item()
            aug_spec[f0:f0+f, :] = 0

        return aug_spec


class AudioSequenceDataset(Dataset):
    """
    Dataset for loading audio sequences and computing spectrograms.

    Expects unlabelled audio files - self-supervised training requires
    no annotations.
    """

    def __init__(
        self,
        audio_files: List[Path],
        spec_config: SpectrogramConfig,
        img_size: tuple = (128, 128),
        augmentation: Optional[BioacousticAugmentation] = None,
        segment_duration: float = 1.0,  # Seconds
    ):
        self.audio_files = audio_files
        self.spec_config = spec_config
        self.img_size = img_size
        self.augmentation = augmentation
        self.segment_samples = int(segment_duration * spec_config.sample_rate)

        # Spectrogram computer
        self.spectrogram = UltrasonicSpectrogram(spec_config)

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load audio, compute spectrogram, apply augmentation.

        Returns:
            Spectrogram tensor (Freq, Time) resized to img_size
        """
        # Load audio (simplified - in practice use torchaudio.load)
        # For now, generate synthetic audio for testing
        audio = torch.randn(self.segment_samples)

        # Compute spectrogram
        spec = self.spectrogram(audio)  # (1, Freq, Time)
        spec = spec.squeeze(0)  # (Freq, Time)

        # Normalize
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)

        # Resize to target size
        spec = F.interpolate(
            spec.unsqueeze(0).unsqueeze(0),
            size=self.img_size,
            mode='bilinear',
            align_corners=False,
        ).squeeze()

        # Apply augmentation
        if self.augmentation is not None:
            spec = self.augmentation(spec)

        return spec


class BioMAETrainer:
    """
    Training loop for BioMAE self-supervised learning.

    Implements MAE-style training:
    1. Randomly mask 75% of spectrogram patches
    2. Encoder processes visible patches only
    3. Decoder reconstructs full spectrogram
    4. Loss: MSE on masked patches only

    Args:
        model: BioMAE model instance
        config: TrainingConfig with hyperparameters
    """

    def __init__(
        self,
        model: BioMAEModel,
        config: TrainingConfig,
    ):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model.to(self.device)

        # Optimizer (AdamW with weight decay)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6,
        )

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        # Augmentation
        self.augmentation = BioacousticAugmentation(
            time_stretch_range=config.time_stretch_range,
            pitch_shift_range=config.pitch_shift_range,
            noise_level=config.noise_level,
        )

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

    def train_step(
        self,
        batch: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step with MAE masking.

        Args:
            batch: Spectrogram batch (Batch, Freq, Time)

        Returns:
            Dictionary with loss metrics
        """
        self.model.train()

        # Move to device
        batch = batch.to(self.device)
        batch = batch.unsqueeze(1)  # Add channel dim

        # Generate random mask (75% masking)
        mask = self.model.generate_random_mask(
            batch.size(0),
            self.device,
            mask_ratio=self.config.mask_ratio,
        )

        # Forward pass with mixed precision
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                reconstructed, _ = self.model(batch, mask=mask)
                loss = self.compute_loss(batch, reconstructed, mask)
        else:
            reconstructed, _ = self.model(batch, mask=mask)
            loss = self.compute_loss(batch, reconstructed, mask)

        # Backward pass
        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return {
            'loss': loss.item(),
            'masked_ratio': mask.float().mean().item(),
        }

    def compute_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MAE reconstruction loss on masked patches only.

        Args:
            original: Original spectrogram (B, C, H, W)
            reconstructed: Reconstructed spectrogram (B, C, H, W)
            mask: Boolean mask (B, num_patches) where True=masked

        Returns:
            MSE loss on masked patches
        """
        # Normalize to [0, 1] for loss computation
        original_norm = (original - original.min()) / (original.max() - original.min() + 1e-8)
        recon_norm = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-8)

        # Compute loss only on masked patches
        # Reshape to patches
        B, C, H, W = original.shape
        pH, pW = self.model.encoder.config.patch_size

        # Reshape to (B, num_patches, patch_pixels)
        original_patches = original_norm.unfold(2, pH, pH).unfold(3, pW, pW)
        original_patches = original_patches.contiguous().view(B, C, -1, pH * pW)
        original_patches = original_patches.permute(0, 2, 3, 1).contiguous()  # (B, num_patches, patch_pixels, C)
        original_patches = original_patches.view(B, -1, pH * pW * C)

        recon_patches = recon_norm.unfold(2, pH, pH).unfold(3, pW, pW)
        recon_patches = recon_patches.contiguous().view(B, C, -1, pH * pW)
        recon_patches = recon_patches.permute(0, 2, 3, 1).contiguous()
        recon_patches = recon_patches.view(B, -1, pH * pW * C)

        # Select masked patches
        loss = 0.0
        for b in range(B):
            masked_indices = mask[b]
            if masked_indices.any():
                loss += F.mse_loss(
                    recon_patches[b, masked_indices],
                    original_patches[b, masked_indices],
                )

        return loss / B

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            batch = batch.to(self.device).unsqueeze(1)

            # Generate mask
            mask = self.model.generate_random_mask(
                batch.size(0),
                self.device,
                mask_ratio=self.config.mask_ratio,
            )

            # Forward pass
            reconstructed, _ = self.model(batch, mask=mask)
            loss = self.compute_loss(batch, reconstructed, mask)

            total_loss += loss.item()
            num_batches += 1

        return {'val_loss': total_loss / num_batches}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader

        Returns:
            Training history dictionary
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
        }

        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_losses = []

            # Training loop
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])

                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'masked': f"{metrics['masked_ratio']:.2%}",
                })

                # Logging
                if batch_idx % self.config.log_interval == 0:
                    logger.info(
                        f"Epoch {epoch+1} Batch {batch_idx}: "
                        f"loss={metrics['loss']:.4f} "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
                    )

            # Epoch metrics
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            history['train_loss'].append(avg_train_loss)
            history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['val_loss'])
                logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_metrics['val_loss']:.4f}")

                # Save best model
                if val_metrics['val_loss'] < self.best_loss:
                    self.best_loss = val_metrics['val_loss']
                    self.save_checkpoint('best.pt')
            else:
                logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")

            # Learning rate scheduler
            self.scheduler.step()

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pt')

        # Save final model
        self.save_checkpoint('final.pt')
        logger.info("Training completed!")

        return history

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def create_trainer(
    model: Optional[BioMAEModel] = None,
    config: Optional[TrainingConfig] = None,
) -> BioMAETrainer:
    """Factory function to create trainer."""
    if config is None:
        config = TrainingConfig()

    if model is None:
        encoder_config = EncoderConfig(
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            output_dim=config.output_dim,
        )
        decoder_config = DecoderConfig(
            embed_dim=config.embed_dim,
            depth=max(2, config.depth // 2),  # Decoder is shallower
        )
        model = BioMAEModel(encoder_config, decoder_config)

    return BioMAETrainer(model, config)


# Training script entry point

def main():
    """Example training script."""
    logging.basicConfig(level=logging.INFO)

    # Create model
    encoder_config = EncoderConfig(
        img_size=(128, 128),
        embed_dim=256,
        depth=4,
        num_heads=4,
        output_dim=112,
    )
    decoder_config = DecoderConfig(
        embed_dim=256,
        decoder_embed_dim=128,
        depth=2,
        num_heads=4,
        img_size=(128, 128),
    )
    model = BioMAEModel(encoder_config, decoder_config)

    # Create trainer
    config = TrainingConfig(
        batch_size=16,
        num_epochs=50,
        learning_rate=1e-4,
        mask_ratio=0.75,
    )
    trainer = create_trainer(model, config)

    # Note: In practice, load real audio data
    # For now, this is a template for the training setup
    logger.info("BioMAE trainer initialized. Load audio data to begin training.")


if __name__ == '__main__':
    main()
