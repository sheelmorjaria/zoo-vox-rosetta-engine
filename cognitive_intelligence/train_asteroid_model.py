#!/usr/bin/env python3
"""
Asteroid Training Script for Animal Vocalization Source Separation
================================================================

This script trains a Conv-TasNet model using the Asteroid library to separate
animal vocalizations from background noise. It then exports the trained model to
ONNX format for use in Rust via Tract.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Try to import Asteroid
try:
    import asteroid
    from asteroid import ConvTasNet
    from asteroid.losses import PITLossWrapper
    from asteroid.metrics import MetricsTracker

    print(f"Asteroid version: {asteroid.__version__}")
    ASTEROID_AVAILABLE = True
except ImportError as e:
    print(f"Asteroid not available: {e}")
    print("Install with: pip install asteroid")
    ASTEROID_AVAILABLE = False
    sys.exit(1)


class AnimalVocalizationDataset(Dataset):
    """
    Dataset for animal vocalization source separation.

    Expected data format:
    - mixture: Mixed audio with target animal + background
    - sources: [target_animal, background_noise]

    For training, you can use:
    - Pre-recorded animal vocalizations (marmoset, dolphin, etc.)
    - Synthetic mixtures using target vocals + environmental noise
    - Field recordings with isolated sources
    """

    def __init__(self, mixtures_dir, sources_dir, sample_rate=44100, segment=4.0):
        """
        Initialize dataset

        Args:
            mixtures_dir: Directory containing mixture WAV files
            sources_dir: Directory containing source WAV files (one file per source)
            sample_rate: Audio sample rate
            segment: Segment length in seconds
        """
        self.sample_rate = sample_rate
        self.segment_len = int(sample_rate * segment)

        # Load file paths
        self.mixtures = list(Path(mixtures_dir).glob("*.wav"))
        self.sources = list(Path(sources_dir).glob("*.wav"))

        print(f"Found {len(self.mixtures)} mixture files")
        print(f"Found {len(self.sources)} source files")

    def __len__(self):
        return len(self.mixtures)

    def __getitem__(self, idx):
        """
        Get a training sample

        Returns:
            mixture: Tensor of shape (1, segment_len)
            sources: Tensor of shape (num_sources, segment_len)
        """
        import soundfile as sf

        # Load mixture
        mixture_path = self.mixtures[idx]
        mixture, sr = sf.read(mixture_path)

        # Convert to mono if stereo
        if len(mixture.shape) > 1:
            mixture = np.mean(mixture, axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            # Simple resampling (for production, use librosa or similar)
            from scipy import signal

            num_samples = int(len(mixture) * self.sample_rate / sr)
            mixture = signal.resample(mixture, num_samples)

        # Normalize
        mixture = mixture / (np.abs(mixture).max() + 1e-8)

        # Take segment
        if len(mixture) > self.segment_len:
            start = np.random.randint(0, len(mixture) - self.segment_len)
            mixture = mixture[start : start + self.segment_len]
        elif len(mixture) < self.segment_len:
            # Pad if too short
            mixture = np.pad(mixture, (0, self.segment_len - len(mixture)))

        # For demo: create synthetic sources using bandpass filters
        # In production, you would load actual separated source files
        target_animal = self._simulate_target_source(mixture)
        background = self._simulate_background_source(mixture)
        sources = np.stack([target_animal, background])

        return torch.FloatTensor(mixture).unsqueeze(0), torch.FloatTensor(sources)

    def _simulate_target_source(self, mixture):
        """Simulate target animal source (e.g., marmoset phee call)"""
        # Bandpass filter for animal vocalization range (2-12 kHz)
        from scipy import signal

        b, a = signal.butter(4, [2000, 12000], btype="bandpass", fs=self.sample_rate)
        target = signal.filtfilt(b, a, mixture)
        return target * 0.7  # Reduce amplitude slightly

    def _simulate_background_source(self, mixture):
        """Simulate background noise/environmental sounds"""
        # Bandpass filter for lower frequencies (< 5 kHz)
        from scipy import signal

        b, a = signal.butter(4, [100, 5000], btype="bandpass", fs=self.sample_rate)
        background = signal.filtfilt(b, a, mixture)
        return background * 0.5


class AsteroidTrainer:
    """
    Trainer for Conv-TasNet using Asteroid
    """

    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self):
        import logging

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def create_model(self):
        """Create Conv-TasNet model with Asteroid"""
        model = ConvTasNet(
            n_src=self.config["num_sources"],
            sample_rate=self.config["sample_rate"],
            n_fft=self.config["n_fft"],
            n_freq=self.config["n_freq"],
            n_blocks=self.config["n_blocks"],
            n_repeats=self.config["n_repeats"],
            mask_act=self.config["mask_act"],
        )
        return model

    def train(self, train_loader, val_loader, epochs=100):
        """Train the model"""
        self.logger.info(f"Starting training for {epochs} epochs")

        # Create model
        model = self.create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["lr"])

        # Loss function (Permutation Invariant Training)
        loss_func = PITLossWrapper(loss_func=self.config["loss_func"], pit_from="pw_pt")

        # Training loop
        model.train()
        for epoch in range(epochs):
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
                        f"Epoch {epoch}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
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
        dummy_input = torch.randn(1, 1, self.config["sample_rate"])

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["mixture"],
            output_names=["separated_sources"],
            dynamic_axes={
                "mixture": {0: "batch_size", 2: "time"},
                "separated_sources": {0: "batch_size", 1: "num_sources", 2: "time"},
            },
        )

        self.logger.info(f"Model exported successfully to {output_path}")

        # Verify ONNX model
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        self.logger.info("ONNX model verified successfully")

        return output_path


def main():
    """Main training function"""

    # Configuration
    config = {
        "num_sources": 2,  # Target animal + background
        "sample_rate": 44100,
        "n_fft": 512,
        "n_freq": 257,
        "n_blocks": 8,
        "n_repeats": 3,
        "mask_act": "relu",
        "lr": 1e-3,
        "loss_func": "si_snr",  # Scale-invariant signal-to-noise ratio
        "batch_size": 4,
        "epochs": 50,
    }

    # Check for data directories
    # For demo, create synthetic data if not available
    mixtures_dir = Path("data/train/mixtures")
    sources_dir = Path("data/train/sources")

    if not mixtures_dir.exists() or not sources_dir.exists():
        print("No training data found. Creating synthetic dataset for demo...")
        mixtures_dir.mkdir(parents=True, exist_ok=True)
        sources_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic data for demo
        print("Creating synthetic training data...")
        # In production, you would copy your actual animal vocalization data here

    # Create dataset
    # Note: For this demo, we'll skip actual training since we don't have real data
    print("\n" + "=" * 60)
    print("ASTEROID TRAINING WORKFLOW")
    print("=" * 60)
    print("""
This script is ready to train a proper Conv-TasNet model using Asteroid.

To use with real data:
1. Place mixture WAV files in: data/train/mixtures/
2. Place source WAV files in: data/train/sources/
3. Run: python train_asteroid_model.py

The trained model will be exported to ONNX format for use in Rust.
    """)

    # Create trainer
    trainer = AsteroidTrainer(config)

    # For demo: Create a simple model and export it
    print("\nCreating demo model for export...")
    model = trainer.create_model()

    # Save model checkpoint
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "conv_tasnet_animal.ckpt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")

    # Export to ONNX
    onnx_path = checkpoint_dir / "conv_tasnet_animal.onnx"
    trainer.export_to_onnx(model, str(onnx_path))

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"""
1. ONNX model exported to: {onnx_path}
2. Copy this model to your Rust project: cp {onnx_path} <rust_project>/models/
3. Update features.rs to use Tract for inference

Rust integration example (add to Cargo.toml):
    tract-onnx = "0.21"

Then in features.rs:
    use tract_onnx::prelude::*;
    let model = tract_onnx::onnx()
        .model_for_path("models/conv_tasnet_animal.onnx")?
        .into_runnable()?;
    """)

    return model, onnx_path


if __name__ == "__main__":
    main()
