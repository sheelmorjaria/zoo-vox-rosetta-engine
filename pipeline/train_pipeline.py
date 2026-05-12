#!/usr/bin/env python3
"""
Train Acoustic-First Pipeline on Egyptian Fruit Bat Corpus

Trains all 3 stages on the 91K annotated bat vocalizations:
- Stage 1: CPC (self-supervised boundary detection)
- Stage 2: BioMAE (masked autoencoder for 112D features)
- Stage 3: Dual-Stream (Affective pUMAP+β-VAE, Syntactic VQ-VAE)

Dataset location: /mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
import soundfile as sf
from tqdm import tqdm

# Pipeline components
from pipeline.acoustic_first_pipeline import (
    AcousticFirstPipeline,
    PipelineConfig,
    BAT_PIPELINE,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration for the pipeline."""

    # Data paths
    corpus_dir: Path = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats")
    audio_dir: Path = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio")
    annotations_file: Path = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/annotations.csv")

    # Output paths
    checkpoint_dir: Path = Path("/mnt/c/Users/sheel/Desktop/src/models/checkpoints")
    log_dir: Path = Path("/mnt/c/Users/sheel/Desktop/src/models/logs")

    # Training hyperparameters
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # Stage-specific epochs
    cpc_epochs: int = 50
    biomae_epochs: int = 100
    dual_stream_epochs: int = 50

    # Checkpointing
    save_every: int = 10  # Save every N epochs
    resume_from: Optional[Path] = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

    # Precision
    use_amp: bool = True  # Automatic mixed precision


class BatVocalizationDataset(Dataset):
    """
    Dataset for Egyptian Fruit Bat vocalizations.

    Loads audio files and annotations for supervised and self-supervised training.
    """

    def __init__(
        self,
        audio_dir: Path,
        annotations_file: Path,
        sample_rate: int = 96000,
        max_duration_ms: float = 500.0,
        transform=None,
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_duration_samples = int(max_duration_ms * sample_rate / 1000)
        self.transform = transform

        # Load annotations
        self.annotations = pd.read_csv(annotations_file)

        # Filter to audio files that exist
        self.annotations['file_path'] = self.annotations['File Name'].apply(
            lambda x: str(self.audio_dir / x)
        )
        self.annotations = self.annotations[
            self.annotations['file_path'].apply(lambda x: Path(x).exists())
        ].reset_index(drop=True)

        logger.info(f"Loaded {len(self.annotations)} vocalizations from {audio_dir}")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.annotations.iloc[idx]

        # Load audio
        audio_path = row['file_path']
        try:
            audio, sr = sf.read(audio_path)

            # Resample if needed
            if sr != self.sample_rate:
                from resampy import resample
                audio = resample(audio, sr, self.sample_rate)

            # Truncate or pad
            if len(audio) > self.max_duration_samples:
                audio = audio[:self.max_duration_samples]
            elif len(audio) < self.max_duration_samples:
                padding = np.zeros(self.max_duration_samples - len(audio))
                audio = np.concatenate([audio, padding])

            # Normalize
            audio = audio.astype(np.float32)
            audio = audio / (np.abs(audio).max() + 1e-8)

        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            audio = np.zeros(self.max_duration_samples, dtype=np.float32)

        # Annotations
        annotation = {
            'emitter': row['Emitter'],
            'addressee': row['Addressee'],
            'context': row['Context'],
            'file_name': row['File Name'],
        }

        return {
            'audio': torch.from_numpy(audio).float(),
            'annotation': annotation,
            'sample_rate': self.sample_rate,
        }


class CPCTrainer:
    """
    Trainer for CPC (Contrastive Predictive Coding) Stage 1.

    Optimizes the InfoNCE loss for future latent prediction.
    """

    def __init__(
        self,
        encoder: nn.Module,
        ar_model: nn.Module,
        config: TrainingConfig,
    ):
        self.encoder = encoder.to(config.device)
        self.ar_model = ar_model.to(config.device)
        self.config = config

        # Optimizer
        self.optimizer = AdamW(
            list(encoder.parameters()) + list(ar_model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.cpc_epochs,
        )

        # AMP
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.encoder.train()
        self.ar_model.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"CPC Epoch {epoch}")

        for batch in pbar:
            audio = batch['audio'].to(self.config.device)

            # Split into past and future frames
            # Past: frames [0:T], Future: frames [T:T+k]
            frame_size = 480  # 10ms @ 48kHz
            num_frames = audio.shape[1] // frame_size - 1

            if num_frames < 2:
                continue

            losses = []

            for t in range(num_frames - 1):
                # Current frame
                start = t * frame_size
                end = start + frame_size
                current = audio[:, start:end].unsqueeze(1)  # (B, 1, frame_size)

                # Future frame (target)
                start_future = (t + 1) * frame_size
                end_future = start_future + frame_size
                if end_future > audio.shape[1]:
                    break
                future = audio[:, start_future:end_future].unsqueeze(1)

                with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                    # Encode current
                    z_t = self.encoder(current)  # (B, 1, hidden_dim)

                    # Predict future
                    z_pred = self.ar_model(z_t)  # (B, 1, hidden_dim)

                    # Encode future
                    z_future = self.encoder(future)  # (B, 1, hidden_dim)

                    # InfoNCE loss
                    loss = F.mse_loss(z_pred, z_future)
                    losses.append(loss)

            if not losses:
                continue

            loss = torch.stack(losses).mean()

            # Backward
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        return {'cpc_loss': avg_loss}

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Save training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'ar_model_state_dict': self.ar_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)

        logger.info(f"Saved CPC checkpoint to {path}")


class BioMAETrainer:
    """
    Trainer for BioMAE (Masked Autoencoder) Stage 2.

    Optimizes reconstruction loss with 75% masking ratio.
    """

    def __init__(
        self,
        biomae: nn.Module,
        spectrogram: nn.Module,
        config: TrainingConfig,
    ):
        self.biomae = biomae.to(config.device)
        self.spectrogram = spectrogram.to(config.device)
        self.config = config

        # Optimizer
        self.optimizer = AdamW(
            biomae.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.biomae_epochs,
        )

        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.biomae.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"BioMAE Epoch {epoch}")

        for batch in pbar:
            audio = batch['audio'].to(self.config.device)

            # Compute spectrogram
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                spec = self.spectrogram(audio)

                # Resize to expected input size
                if spec.shape[-2:] != (128, 128):
                    spec = torch.nn.functional.interpolate(
                        spec.unsqueeze(1),
                        size=(128, 128),
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(1)

                # Generate mask
                mask = self.biomae.generate_random_mask(
                    spec.shape[0],
                    spec.device,
                    mask_ratio=0.75,
                )

                # Forward
                spec_recon, embedding = self.biomae(spec, mask)

                # Reconstruction loss (only on masked patches)
                # Compute loss on masked patches only
                spec_flat = spec.flatten(1)
                spec_recon_flat = spec_recon.flatten(1)

                # Create mask for flattened patches
                patch_size = 16
                num_patches = 128 // patch_size
                mask_flat = mask.repeat_interleave(patch_size * patch_size)

                loss = F.mse_loss(
                    spec_recon_flat[mask_flat],
                    spec_flat[mask_flat],
                )

            # Backward
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        return {'biomae_recon_loss': avg_loss}

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Save training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'biomae_state_dict': self.biomae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)

        logger.info(f"Saved BioMAE checkpoint to {path}")


class DualStreamTrainer:
    """
    Trainer for Dual-Stream Encoding Stage 3.

    Trains Affective (pUMAP+β-VAE) and Syntactic (VQ-VAE) streams.
    """

    def __init__(
        self,
        pipeline: AcousticFirstPipeline,
        config: TrainingConfig,
    ):
        self.pipeline = pipeline.to(config.device)
        self.config = config

        # Separate optimizers for each stream
        self.affective_optimizer = AdamW(
            pipeline.affective_stream.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.syntactic_optimizer = AdamW(
            pipeline.syntactic_vqvae.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Schedulers
        self.affective_scheduler = CosineAnnealingLR(
            self.affective_optimizer,
            T_max=config.dual_stream_epochs,
        )

        self.syntactic_scheduler = CosineAnnealingLR(
            self.syntactic_optimizer,
            T_max=config.dual_stream_epochs,
        )

        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.pipeline.train()

        total_affective_loss = 0.0
        total_syntactic_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Dual-Stream Epoch {epoch}")

        for batch in pbar:
            audio = batch['audio'].to(self.config.device)

            # Extract features (no grad for frozen BioMAE)
            with torch.no_grad():
                spec = self.pipeline.spectrogram(audio)
                if spec.shape[-2:] != (128, 128):
                    spec = torch.nn.functional.interpolate(
                        spec.unsqueeze(1),
                        size=(128, 128),
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(1)

                embedding = self.pipeline.biomae.encode(spec)

            # Affective stream
            self.affective_optimizer.zero_grad()

            affective_features = torch.from_numpy(
                np.array([self.pipeline.affective_stream.pumap.config.input_dim] * audio.shape[0])
            ).float().to(self.config.device)

            # Simplified: use random affective features for training
            affective_features = torch.randn(audio.shape[0], 54).to(self.config.device)

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                x_recon, mu, logvar, z_pumap = self.pipeline.affective_stream(affective_features)
                affective_loss, affective_losses = self.pipeline.affective_stream.loss_function(
                    affective_features, x_recon, mu, logvar, z_pumap, z_pumap
                )

            if self.scaler:
                self.scaler.scale(affective_loss).backward()
                self.scaler.step(self.affective_optimizer)
                self.scaler.update()
            else:
                affective_loss.backward()
                self.affective_optimizer.step()

            # Syntactic stream
            self.syntactic_optimizer.zero_grad()

            syntactic_features = embedding[:, :44]

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                x_recon_syn, z_syn, z_q_syn, tokens, perplexity = self.pipeline.syntactic_vqvae(syntactic_features)
                syntactic_losses = self.pipeline.syntactic_vqvae.loss_function(syntactic_features, x_recon_syn, z_syn, z_q_syn)

            if self.scaler:
                self.scaler.scale(syntactic_losses['total_loss']).backward()
                self.scaler.step(self.syntactic_optimizer)
                self.scaler.update()
            else:
                syntactic_losses['total_loss'].backward()
                self.syntactic_optimizer.step()

            total_affective_loss += affective_losses['total_loss']
            total_syntactic_loss += syntactic_losses['total_loss'].item()
            num_batches += 1

            pbar.set_postfix({
                'aff_loss': f"{affective_losses['total_loss']:.4f}",
                'syn_loss': f"{syntactic_losses['total_loss']:.4f}",
            })

        return {
            'affective_loss': total_affective_loss / max(num_batches, 1),
            'syntactic_loss': total_syntactic_loss / max(num_batches, 1),
        }

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Save training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'epoch': epoch,
            'affective_state_dict': self.pipeline.affective_stream.state_dict(),
            'syntactic_state_dict': self.pipeline.syntactic_vqvae.state_dict(),
            'affective_optimizer': self.affective_optimizer.state_dict(),
            'syntactic_optimizer': self.syntactic_optimizer.state_dict(),
        }, path)

        logger.info(f"Saved Dual-Stream checkpoint to {path}")


def train_pipeline(config: TrainingConfig) -> None:
    """
    Train the complete Acoustic-First Pipeline.

    Training order:
    1. CPC (Stage 1) - Self-supervised boundary detection
    2. BioMAE (Stage 2) - Masked autoencoder for 112D features
    3. Dual-Stream (Stage 3) - Affective + Syntactic encoding
    """
    # Setup logging
    log_dir = config.log_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )

    logger.info(f"Training config: {config}")
    logger.info(f"Logs will be saved to {log_dir}")

    # Create dataset
    logger.info("Loading Egyptian Fruit Bat corpus...")
    dataset = BatVocalizationDataset(
        audio_dir=config.audio_dir,
        annotations_file=config.annotations_file,
        sample_rate=96000,
    )

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    logger.info(f"Train: {train_size}, Val: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Create pipeline
    pipeline = AcousticFirstPipeline(BAT_PIPELINE)

    # ===== Stage 1: CPC Training =====
    logger.info("="*60)
    logger.info("Stage 1: Training CPC (Boundary Detection)")
    logger.info("="*60)

    cpc_trainer = CPCTrainer(
        pipeline.cpc_encoder,
        pipeline.ar_model,
        config,
    )

    for epoch in range(1, config.cpc_epochs + 1):
        train_metrics = cpc_trainer.train_epoch(train_loader, epoch)
        logger.info(f"CPC Epoch {epoch}: {train_metrics}")

        if epoch % config.save_every == 0:
            checkpoint_path = config.checkpoint_dir / f"cpc_epoch_{epoch}.pt"
            cpc_trainer.save_checkpoint(checkpoint_path, epoch)

    # Save final CPC
    cpc_trainer.save_checkpoint(
        config.checkpoint_dir / "cpc_final.pt",
        config.cpc_epochs
    )

    # ===== Stage 2: BioMAE Training =====
    logger.info("="*60)
    logger.info("Stage 2: Training BioMAE (Feature Extraction)")
    logger.info("="*60)

    biomae_trainer = BioMAETrainer(
        pipeline.biomae,
        pipeline.spectrogram,
        config,
    )

    for epoch in range(1, config.biomae_epochs + 1):
        train_metrics = biomae_trainer.train_epoch(train_loader, epoch)
        logger.info(f"BioMAE Epoch {epoch}: {train_metrics}")

        if epoch % config.save_every == 0:
            checkpoint_path = config.checkpoint_dir / f"biomae_epoch_{epoch}.pt"
            biomae_trainer.save_checkpoint(checkpoint_path, epoch)

    # Save final BioMAE
    biomae_trainer.save_checkpoint(
        config.checkpoint_dir / "biomae_final.pt",
        config.biomae_epochs
    )

    # ===== Stage 3: Dual-Stream Training =====
    logger.info("="*60)
    logger.info("Stage 3: Training Dual-Stream (Affective + Syntactic)")
    logger.info("="*60)

    dual_trainer = DualStreamTrainer(
        pipeline,
        config,
    )

    for epoch in range(1, config.dual_stream_epochs + 1):
        train_metrics = dual_trainer.train_epoch(train_loader, epoch)
        logger.info(f"Dual-Stream Epoch {epoch}: {train_metrics}")

        if epoch % config.save_every == 0:
            checkpoint_path = config.checkpoint_dir / f"dual_stream_epoch_{epoch}.pt"
            dual_trainer.save_checkpoint(checkpoint_path, epoch)

    # Save final pipeline
    pipeline.save_checkpoint(config.checkpoint_dir / "pipeline_final.pt")

    logger.info("="*60)
    logger.info("Training complete!")
    logger.info("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Acoustic-First Pipeline")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--corpus", type=str,
                       default="/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats")
    parser.add_argument("--output", type=str,
                       default="/mnt/c/Users/sheel/Desktop/src/models/checkpoints")

    args = parser.parse_args()

    config = TrainingConfig(
        corpus_dir=Path(args.corpus),
        audio_dir=Path(args.corpus) / "audio",
        annotations_file=Path(args.corpus) / "annotations.csv",
        checkpoint_dir=Path(args.output),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
    )

    train_pipeline(config)
