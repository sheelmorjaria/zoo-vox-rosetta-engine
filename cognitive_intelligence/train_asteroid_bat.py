#!/usr/bin/env python3
"""
Egyptian Fruit Bat-Specific Asteroid Training Script
=====================================================

Trains a Conv-TasNet model optimized for Egyptian fruit bat (Rousettus aegyptiacus)
vocalizations including FM sweeps and echolocation calls.

Species Characteristics:
- F0 Range: 0 - 17000 Hz (wide range including FM sweeps)
- Vocalization Type: FM sweeps, echolocation, social calls
- Filter Range: 100 - 22100 Hz (wide bandwidth)

Note: Bats require HIGHER sample rates for optimal performance (96kHz+).
This script uses 44.1kHz for compatibility but consider 96kHz for production.

Usage:
    python train_asteroid_bat.py

To train with real data:
1. Place mixture WAV files in: data/train/egyptian_bat/mixtures/
2. Place source WAV files in: data/train/egyptian_bat/sources/
3. Run this script

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_asteroid_base import AnimalVocalizationDataset, AsteroidTrainer, SpeciesSpecificConfig


def main():
    """Train bat-specific source separation model"""

    # Bat-specific configuration
    # Note: Using 44.1kHz for compatibility. For production, use 96kHz+.
    config = SpeciesSpecificConfig(
        species_name="egyptian_bat",
        f0_min_hz=100,  # Bats have very wide frequency range
        f0_max_hz=17000,
        sample_rate=44100,  # Consider 96000 for production
    )

    config.print_config()

    # Create checkpoint directory
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    print("\nCreating dataset...")
    dataset = AnimalVocalizationDataset(config, segment=4.0)

    # Create data loader
    from torch.utils.data import DataLoader

    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    # Create trainer
    trainer = AsteroidTrainer(config)

    # Check if we have real data
    has_real_data = len(dataset.mixtures) > 0

    if has_real_data:
        print("\n" + "=" * 60)
        print("TRAINING WITH REAL DATA")
        print("=" * 60)
        print(f"Found {len(dataset.mixtures)} mixture files")
        print(f"Training for {config.epochs} epochs...")

        # Train model
        model = trainer.train(train_loader)

    else:
        print("\n" + "=" * 60)
        print("NO TRAINING DATA FOUND - CREATING DEMO MODEL")
        print("=" * 60)
        print("\nTo train with real data:")
        print(f"1. Place mixture WAV files in: {config.data_dir / 'mixtures'}")
        print(f"2. Place source WAV files in: {config.data_dir / 'sources'}")
        print("3. Run this script again")
        print("\nCreating demo model for export...")

        # Create untrained model
        model = trainer.create_model()

    # Save checkpoint
    checkpoint_path = config.checkpoint_dir / "conv_tasnet_egyptian_bat.ckpt"
    trainer.save_checkpoint(model, checkpoint_path)

    # Export to ONNX
    onnx_path = config.checkpoint_dir / "conv_tasnet_egyptian_bat.onnx"
    trainer.export_to_onnx(model, str(onnx_path))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel files saved to: {config.checkpoint_dir}/")
    print(f"  - PyTorch checkpoint: {checkpoint_path}")
    print(f"  - ONNX model: {onnx_path}")

    print("\n" + "=" * 60)
    print("IMPORTANT NOTES FOR BAT VOCALIZATIONS")
    print("=" * 60)
    print(f"""
⚠️  SAMPLE RATE CONSIDERATION:
   Current: {config.sample_rate / 1000:.1f} kHz
   Recommended: 96 kHz+ for full bat frequency range

   Bats produce ultrasonic vocalizations up to 100+ kHz.
   For optimal performance, retrain with 96kHz or 192kHz audio.

🦇 BAT-SPECIFIC CHARACTERISTICS:
   - Species: Egyptian Fruit Bat (Rousettus aegyptiacus)
   - F0 Range: {config.f0_min_hz} - {config.f0_max_hz} Hz
   - Filter Range: {config.filter_min_hz} - {config.filter_max_hz} Hz
   - Vocalization Types: FM sweeps, echolocation, social calls

📝 RETRAINING WITH HIGHER SAMPLE RATE:
   1. Record/resample audio at 96kHz or 192kHz
   2. Update config: sample_rate=96000
   3. Retrain model
""")

    print("\n" + "=" * 60)
    print("RUST INTEGRATION")
    print("=" * 60)
    print(f"""
To use this model in Rust:

1. Copy ONNX model to your Rust project:
   cp {onnx_path} <rust_project>/models/

2. Update technical_architecture/src/source_separation.rs:
   model_path: "models/checkpoints/conv_tasnet_egyptian_bat.onnx"

3. Rebuild Rust library:
   cd technical_architecture && cargo build --release
""")

    return model, onnx_path


if __name__ == "__main__":
    main()
