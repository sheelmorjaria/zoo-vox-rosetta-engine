#!/usr/bin/env python3
"""
Asteroid Training Script - Base Template
========================================

Base template for training species-specific Conv-TasNet models using Asteroid.
This can be extended for specific animal species with appropriate frequency ranges.

Usage:
    Subclass this template and set species-specific parameters.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# Try to import Asteroid
try:
    from asteroid import ConvTasNet
    from asteroid.losses import PITLossWrapper
    from asteroid.metrics import MetricsTracker
    import asteroid
    print(f"Asteroid version: {asteroid.__version__}")
    ASTEROID_AVAILABLE = True
except ImportError as e:
    print(f"Asteroid not available: {e}")
    print("Install with: pip install asteroid")
    ASTEROID_AVAILABLE = False
    sys.exit(1)


class SpeciesSpecificConfig:
    """Configuration for species-specific model training"""

    def __init__(self, species_name, f0_min_hz, f0_max_hz, sample_rate=44100):
        """
        Initialize species-specific configuration

        Args:
            species_name: Name of the species (e.g., "marmoset", "bat")
            f0_min_hz: Minimum fundamental frequency in Hz
            f0_max_hz: Maximum fundamental frequency in Hz
            sample_rate: Audio sample rate
        """
        self.species_name = species_name
        self.f0_min_hz = f0_min_hz
        self.f0_max_hz = f0_max_hz
        self.sample_rate = sample_rate

        # Calculate bandpass filter range with 30% margin
        self.filter_min_hz = max(100, int(f0_min_hz * 0.7))
        self.filter_max_hz = int(f0_max_hz * 1.3)

        # Background filter range (everything below target)
        self.background_max_hz = int(f0_min_hz * 0.6)

        # Conv-TasNet configuration
        self.num_sources = 2  # Target animal + background
        self.n_fft = 512
        self.n_freq = 257
        self.n_blocks = 8
        self.n_repeats = 3
        self.mask_act = 'relu'
        self.lr = 1e-3
        self.loss_func = 'si_snr'  # Scale-invariant signal-to-noise ratio
        self.batch_size = 4
        self.epochs = 50

        # Paths
        self.checkpoint_dir = Path(f"models/checkpoints/{species_name}")
        self.data_dir = Path(f"data/train/{species_name}")

    def print_config(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print(f"SPECIES-SPECIFIC CONFIGURATION: {self.species_name.upper()}")
        print("="*60)
        print(f"Species: {self.species_name}")
        print(f"F0 Range: {self.f0_min_hz} - {self.f0_max_hz} Hz")
        print(f"Target Filter: {self.filter_min_hz} - {self.filter_max_hz} Hz")
        print(f"Background Filter: < {self.background_max_hz} Hz")
        print(f"Sample Rate: {self.sample_rate} Hz")
        print("="*60)


class AnimalVocalizationDataset(Dataset):
    """
    Dataset for animal vocalization source separation.

    Expected data format:
    - mixture: Mixed audio with target animal + background
    - sources: [target_animal, background_noise]
    """

    def __init__(self, config: SpeciesSpecificConfig, segment=4.0):
        """
        Initialize dataset

        Args:
            config: SpeciesSpecificConfig instance
            segment: Segment length in seconds
        """
        self.config = config
        self.segment_len = int(config.sample_rate * segment)

        # Load file paths
        mixtures_dir = config.data_dir / "mixtures"
        sources_dir = config.data_dir / "sources"

        self.mixtures = list(mixtures_dir.glob("*.wav")) if mixtures_dir.exists() else []
        self.sources = list(sources_dir.glob("*.wav")) if sources_dir.exists() else []

        print(f"Found {len(self.mixtures)} mixture files")
        print(f"Found {len(self.sources)} source files")

    def __len__(self):
        return len(self.mixtures) if self.mixtures else 100  # Default for synthetic data

    def __getitem__(self, idx):
        """
        Get a training sample

        Returns:
            mixture: Tensor of shape (1, segment_len)
            sources: Tensor of shape (num_sources, segment_len)
        """
        import soundfile as sf
        from scipy import signal

        # If we have real data, load it
        if self.mixtures:
            mixture_path = self.mixtures[idx % len(self.mixtures)]
            mixture, sr = sf.read(mixture_path)

            # Convert to mono if stereo
            if len(mixture.shape) > 1:
                mixture = np.mean(mixture, axis=1)

            # Resample if needed
            if sr != self.config.sample_rate:
                num_samples = int(len(mixture) * self.config.sample_rate / sr)
                mixture = signal.resample(mixture, num_samples)

            # Normalize
            mixture = mixture / (np.abs(mixture).max() + 1e-8)
        else:
            # Generate synthetic mixture
            t = np.linspace(0, self.segment_len / self.config.sample_rate, self.segment_len)
            # Target: tone in species frequency range
            target_freq = (self.config.f0_min_hz + self.config.f0_max_hz) / 2
            mixture = 0.5 * np.sin(2 * np.pi * target_freq * t)
            # Add background noise
            mixture += 0.3 * np.random.randn(len(t))

        # Take segment
        if len(mixture) > self.segment_len:
            start = np.random.randint(0, len(mixture) - self.segment_len)
            mixture = mixture[start:start + self.segment_len]
        elif len(mixture) < self.segment_len:
            mixture = np.pad(mixture, (0, self.segment_len - len(mixture)))

        # Simulate sources using bandpass filters
        target_animal = self._filter_target_source(mixture)
        background = self._filter_background_source(mixture)
        sources = np.stack([target_animal, background])

        return torch.FloatTensor(mixture).unsqueeze(0), torch.FloatTensor(sources)

    def _filter_target_source(self, mixture):
        """Filter to extract target animal vocalization range"""
        from scipy import signal
        nyquist = self.config.sample_rate / 2
        low = self.config.filter_min_hz / nyquist
        high = self.config.filter_max_hz / nyquist
        b, a = signal.butter(4, [low, high], btype='bandpass')
        target = signal.filtfilt(b, a, mixture)
        return target * 0.7

    def _filter_background_source(self, mixture):
        """Filter to extract background noise"""
        from scipy import signal
        nyquist = self.config.sample_rate / 2
        high = min(self.config.background_max_hz / nyquist, 0.99)
        b, a = signal.butter(4, high, btype='lowpass')
        background = signal.filtfilt(b, a, mixture)
        return background * 0.5


class AsteroidTrainer:
    """Trainer for Conv-TasNet using Asteroid"""

    def __init__(self, config: SpeciesSpecificConfig):
        self.config = config
        self._setup_logging()

    def _setup_logging(self):
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def create_model(self):
        """Create Conv-TasNet model with Asteroid"""
        model = ConvTasNet(
            n_src=self.config.num_sources,
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_freq=self.config.n_freq,
            n_blocks=self.config.n_blocks,
            n_repeats=self.config.n_repeats,
            mask_act=self.config.mask_act,
        )
        return model

    def train(self, train_loader, val_loader=None):
        """Train the model"""
        self.logger.info(f"Starting training for {self.config.epochs} epochs")

        # Create model
        model = self.create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

        # Loss function (Permutation Invariant Training)
        loss_func = PITLossWrapper(
            loss_func=self.config.loss_func,
            pit_from="pw_pt"
        )

        # Training loop
        model.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for batch_idx, (mixtures, sources) in enumerate(train_loader):
                optimizer.zero_grad()

                # Forward pass
                estimates = model(mixtures)

                # Compute loss
                loss = loss_func(estimates, sources)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if batch_idx % 10 == 0:
                    self.logger.info(
                        f"Epoch {epoch}/{self.config.epochs}, "
                        f"Batch {batch_idx}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_loss = epoch_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch} complete. Average loss: {avg_loss:.4f}")

            # Validation
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for mixtures, sources in val_loader:
                        estimates = model(mixtures)
                        loss = loss_func(estimates, sources)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                self.logger.info(f"Validation loss: {avg_val_loss:.4f}")
                model.train()

        return model

    def export_to_onnx(self, model, output_path):
        """Export trained model to ONNX format"""
        self.logger.info(f"Exporting model to {output_path}")

        # Create dummy input
        dummy_input = torch.randn(1, 1, self.config.sample_rate)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['mixture'],
            output_names=['separated_sources'],
            dynamic_axes={
                'mixture': {0: 'batch_size', 2: 'time'},
                'separated_sources': {0: 'batch_size', 1: 'num_sources', 2: 'time'}
            }
        )

        self.logger.info(f"Model exported successfully to {output_path}")

        # Verify ONNX model
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        self.logger.info("ONNX model verified successfully")

        return output_path

    def save_checkpoint(self, model, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'species': self.config.species_name,
                'f0_min': self.config.f0_min_hz,
                'f0_max': self.config.f0_max_hz,
                'filter_min': self.config.filter_min_hz,
                'filter_max': self.config.filter_max_hz,
            }
        }, path)
        self.logger.info(f"Checkpoint saved to: {path}")
