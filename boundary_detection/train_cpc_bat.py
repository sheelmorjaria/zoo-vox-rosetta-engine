#!/usr/bin/env python3
"""
Train CPC Model on Egyptian Fruit Bat Corpus

Trains the Contrastive Predictive Coding encoder on 91K bat vocalizations
to enable species-specific boundary detection via prediction errors.

After training, the PredictiveBoundaryDetector will detect phrase boundaries
at points where the trained model makes high prediction errors (surprising transitions).

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import pickle

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from boundary_detection.cpc_encoder import CPCEncoder, EncoderConfig
from boundary_detection.cpc_autoregressive import AutoregressiveMamba, TCNAutoregressive
from boundary_detection.cpc_trainer import TrainingConfig, CPCModel

logger = logging.getLogger(__name__)


@dataclass
class BatCPCConfig(TrainingConfig):
    """Configuration for CPC training on bat corpus."""

    # Bat-specific audio settings
    sample_rate: int = 250000  # Bat recordings at 250kHz
    frame_size_ms: int = 10

    # Model architecture
    hidden_dim: int = 128
    channels: tuple = (64, 128, 256)
    kernels: tuple = (5, 5, 3)
    strides: tuple = (2, 2, 1)

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-3
    num_epochs: int = 50
    steps_ahead: int = 12  # Predict 120ms ahead (12 frames × 10ms)

    # Data
    sequence_length: int = 64  # 640ms context
    overlap: int = 16

    # Paths
    corpus_dir: str = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio"
    checkpoint_dir: str = "/mnt/c/Users/sheel/Desktop/src/checkpoints/bat_cpc"
    max_files: int = 10000  # Limit for faster training (use all for production)
    index_cache: str = "/tmp/bat_cpc_dataset_index.pkl"


class LazyBatCorpusDataset(Dataset):
    """
    Dataset for training CPC on bat vocalizations with LAZY LOADING.

    Files are loaded on-demand during training, not all at once.
    This is essential for handling the full 91K file corpus without OOM errors.
    """

    def __init__(
        self,
        corpus_dir: str,
        config: BatCPCConfig,
    ):
        self.config = config
        self.corpus_dir = Path(corpus_dir)
        self.frame_size = int(config.sample_rate * config.frame_size_ms / 1000)
        self.overlap = config.overlap

        # Get list of wav files
        audio_files = sorted(self.corpus_dir.glob("*.wav"))
        if config.max_files:
            audio_files = audio_files[:config.max_files]

        self.audio_files = audio_files
        logger.info(f"Found {len(self.audio_files)} audio files")

        # Try to load cached index, otherwise build it
        self.sequence_index = []
        self.num_sequences = 0

        cache_path = Path(config.index_cache)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    # Verify cache is valid
                    if cached.get('num_files') == len(self.audio_files):
                        self.sequence_index = cached['index']
                        self.num_sequences = len(self.sequence_index)
                        logger.info(f"Loaded cached index: {self.num_sequences} sequences")
                        return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Build index from scratch
        self._build_sequence_index(cache_path)

    def _build_sequence_index(self, cache_path: Optional[Path] = None):
        """
        Build an index mapping sequence indices to (file_idx, frame_offset).
        Uses soundfile.info() to get duration without loading audio data.
        """
        self.sequence_index = []
        total_duration = 0.0

        seq_len = self.config.sequence_length + self.config.steps_ahead
        stride = self.config.sequence_length - self.overlap

        logger.info("Building sequence index (this may take a few minutes)...")

        for file_idx, audio_path in enumerate(tqdm(self.audio_files, desc="Indexing")):
            try:
                # Get duration WITHOUT loading audio
                info = sf.info(str(audio_path))
                file_samples = int(info.duration * info.samplerate)
                file_frames = file_samples // self.frame_size

                # Count sequences this file can provide
                num_file_seqs = max(0, (file_frames - seq_len) // stride + 1)

                # Add entries to index
                for seq_idx in range(num_file_seqs):
                    frame_offset = seq_idx * stride
                    self.sequence_index.append((file_idx, frame_offset))

                total_duration += info.duration

            except Exception as e:
                logger.warning(f"Failed to index {audio_path.name}: {e}")
                continue

        self.num_sequences = len(self.sequence_index)
        logger.info(
            f"Built index: {self.num_sequences} sequences from "
            f"{len(self.audio_files)} files ({total_duration/3600:.1f} hours)"
        )

        # Cache the index for faster restarts
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'num_files': len(self.audio_files),
                        'index': self.sequence_index,
                    }, f)
                logger.info(f"Cached index to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache index: {e}")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a training sequence. Loads audio file on-demand.
        """
        seq_len = self.config.sequence_length + self.config.steps_ahead

        # Get file and offset from index
        file_idx, frame_offset = self.sequence_index[idx]
        audio_path = self.audio_files[file_idx]

        # Load this specific file
        try:
            audio, sr = sf.read(str(audio_path))

            # Convert to mono if needed
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Resample if needed (bat files should already be 250kHz)
            if sr != self.config.sample_rate:
                import torchaudio
                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                audio = resampler(torch.from_numpy(audio).float()).numpy()

        except Exception as e:
            logger.warning(f"Failed to load {audio_path.name}: {e}")
            # Return zeros as fallback
            audio = np.zeros(self.frame_size * seq_len)

        # Normalize per-file
        if audio.max() > 0:
            audio = audio / audio.max()

        # Extract frames starting from frame_offset
        frames = []
        start_sample = frame_offset * self.frame_size

        for i in range(seq_len):
            frame_start = start_sample + i * self.frame_size
            frame_end = frame_start + self.frame_size

            if frame_end <= len(audio):
                frame = audio[frame_start:frame_end]
            else:
                # Pad with zeros if we run out of audio
                frame = np.zeros(self.frame_size)

            frames.append(frame)

        frames = np.array(frames)

        # Split into context and future
        context = frames[:self.config.sequence_length]
        future = frames[self.config.sequence_length:]

        return torch.from_numpy(context).float(), torch.from_numpy(future).float()


def infonce_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE loss for CPC training.

    Args:
        predictions: Predicted embeddings (B, D)
        targets: Target embeddings (B, D)
        negatives: Negative samples (B, K, D) where K is num negatives
        temperature: Softmax temperature

    Returns:
        Loss scalar
    """
    # Compute positive similarities
    pos_sim = torch.sum(predictions * targets, dim=-1) / temperature  # (B,)

    # Compute negative similarities
    neg_sim = torch.einsum('bd,bkd->bk', predictions, negatives) / temperature  # (B, K)

    # Concatenate and compute log_softmax
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+K)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    loss = F.cross_entropy(logits, labels)
    return loss


def train_epoch(
    model: CPCModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: BatCPCConfig,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for context, future in pbar:
        context = context.to(config.device)
        future = future.to(config.device)

        # Concatenate context and future for encoding
        audio_frames = torch.cat([context, future], dim=1)  # (B, T_context + T_future, frame_size)

        # Forward pass
        z_latent, context_vectors, predictions = model(audio_frames)

        # Compute loss: predictions should match actual future latents
        loss = 0.0
        batch_size, seq_len, hidden_dim = z_latent.shape
        context_len = context.shape[1]

        for k, pred in enumerate(predictions):
            # Actual future latent (shift by k+1 from context end)
            if context_len + k + 1 < seq_len:
                target = z_latent[:, context_len + k + 1, :]  # (B, D)
                # Use corresponding prediction (shifted)
                pred_t = pred[:, context_len - (k + 1), :]  # (B, D)

                # Generate negatives (from other samples in batch)
                negative_indices = torch.randperm(batch_size, device=config.device)
                negatives = target[negative_indices].unsqueeze(1)  # (B, 1, D)
                negatives = negatives.repeat(1, config.negative_samples, 1)  # (B, K, D)

                step_loss = infonce_loss(pred_t, target, negatives, config.temperature)
                loss += step_loss

        loss = loss / max(1, len(predictions))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return {"loss": total_loss / num_batches}


def save_checkpoint(
    model: CPCModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path,
):
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)

    logger.info(f"Saved checkpoint to {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train CPC on bat corpus")
    parser.add_argument("--corpus-dir", default="/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio")
    parser.add_argument("--checkpoint-dir", default="/mnt/c/Users/sheel/Desktop/src/checkpoints/bat_cpc")
    parser.add_argument("--max-files", type=int, default=91080)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create config
    config = BatCPCConfig(
        corpus_dir=args.corpus_dir,
        checkpoint_dir=args.checkpoint_dir,
        max_files=args.max_files,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    logger.info("CPC Training Configuration:")
    logger.info(f"  Corpus: {config.corpus_dir}")
    logger.info(f"  Sample rate: {config.sample_rate}Hz")
    logger.info(f"  Max files: {config.max_files}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Device: {config.device}")

    # Create dataset (lazy loading - fast init)
    logger.info("Creating dataset...")
    dataset = LazyBatCorpusDataset(config.corpus_dir, config)

    if len(dataset) == 0:
        logger.error("Dataset is empty!")
        return

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Create model
    logger.info("Creating CPC model...")

    encoder_config = EncoderConfig(
        sample_rate=config.sample_rate,
        frame_size_ms=config.frame_size_ms,
        hidden_dim=config.hidden_dim,
        num_channels=config.channels,
        kernel_sizes=config.kernels,
        strides=config.strides,
    )

    model = CPCModel(
        encoder_config=encoder_config,
        ar_config={},  # d_model comes from encoder_config.hidden_dim
        steps_ahead=config.steps_ahead,
    ).to(config.device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Training loop
    logger.info("Starting training...")

    best_loss = float("inf")

    for epoch in range(1, config.num_epochs + 1):
        stats = train_epoch(model, dataloader, optimizer, config, epoch)

        logger.info(f"Epoch {epoch}: loss = {stats['loss']:.4f}")

        # Save checkpoint
        if epoch % config.save_every == 0:
            checkpoint_path = Path(config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, stats['loss'], checkpoint_path)

        # Save best model
        if stats['loss'] < best_loss:
            best_loss = stats['loss']
            best_path = Path(config.checkpoint_dir) / "best_model.pt"
            save_checkpoint(model, optimizer, epoch, stats['loss'], best_path)

    logger.info("Training complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Model saved to {config.checkpoint_dir}")


if __name__ == "__main__":
    main()
