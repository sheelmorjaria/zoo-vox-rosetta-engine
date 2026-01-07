#!/usr/bin/env python3
"""
Chimpanzee-Specific Asteroid Training Script
============================================

Trains a Conv-TasNet model optimized for chimpanzee (Pan troglodytes)
vocalizations including pants, hoots, and screams.

Species Characteristics:
- F0 Range: 0 - 1900 Hz
- Vocalization Type: Pants, hoots, screams, grunts
- Filter Range: 100 - 2470 Hz (low-frequency primate)

Chimpanzees have lower frequency vocalizations compared to birds/bats.
Standard 44.1kHz sample rate is sufficient.

Usage:
    python train_asteroid_chimpanzee.py

To train with real data:
1. Place mixture WAV files in: data/train/chimpanzee/mixtures/
2. Place source WAV files in: data/train/chimpanzee/sources/
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
    """Train chimpanzee-specific source separation model"""

    # Chimpanzee-specific configuration
    config = SpeciesSpecificConfig(
        species_name="chimpanzee",
        f0_min_hz=100,  # Low-frequency primate
        f0_max_hz=1900,
        sample_rate=44100
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
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Create trainer
    trainer = AsteroidTrainer(config)

    # Check if we have real data
    has_real_data = len(dataset.mixtures) > 0

    if has_real_data:
        print("\n" + "="*60)
        print("TRAINING WITH REAL DATA")
        print("="*60)
        print(f"Found {len(dataset.mixtures)} mixture files")
        print(f"Training for {config.epochs} epochs...")

        # Train model
        model = trainer.train(train_loader)

    else:
        print("\n" + "="*60)
        print("NO TRAINING DATA FOUND - CREATING DEMO MODEL")
        print("="*60)
        print("\nTo train with real data:")
        print(f"1. Place mixture WAV files in: {config.data_dir / 'mixtures'}")
        print(f"2. Place source WAV files in: {config.data_dir / 'sources'}")
        print("3. Run this script again")
        print("\nCreating demo model for export...")

        # Create untrained model
        model = trainer.create_model()

    # Save checkpoint
    checkpoint_path = config.checkpoint_dir / "conv_tasnet_chimpanzee.ckpt"
    trainer.save_checkpoint(model, checkpoint_path)

    # Export to ONNX
    onnx_path = config.checkpoint_dir / "conv_tasnet_chimpanzee.onnx"
    trainer.export_to_onnx(model, str(onnx_path))

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModel files saved to: {config.checkpoint_dir}/")
    print(f"  - PyTorch checkpoint: {checkpoint_path}")
    print(f"  - ONNX model: {onnx_path}")

    print("\n" + "="*60)
    print("CHIMPANZEE-SPECIFIC CHARACTERISTICS")
    print("="*60)
    print(f"""
🦧 CHIMPANZEE VOCALIZATIONS:
   - Species: Chimpanzee (Pan troglodytes)
   - F0 Range: {config.f0_min_hz} - {config.f0_max_hz} Hz (low-frequency)
   - Filter Range: {config.filter_min_hz} - {config.filter_max_hz} Hz
   - Vocalization Types:
     * Pant hoots (long-distance communication)
     * Screams (aggression/fear)
     * Grunts (food/affiliation)
     * Barks (alarm/warning)

📝 LOW-FREQUENCY CONSIDERATIONS:
   - Chimpanzees have some of the lowest F0 among primates
   - Similar frequency range to human speech
   - Wind noise and environmental noise overlap this range
   - Consider additional low-pass filtering for clean separation

🔬 RESEARCH APPLICATIONS:
   - Social communication analysis
   - Emotional state detection
   - Individual identification
   - Cultural transmission studies
""")

    print("\n" + "="*60)
    print("RUST INTEGRATION")
    print("="*60)
    print(f"""
To use this model in Rust:

1. Copy ONNX model to your Rust project:
   cp {onnx_path} <rust_project>/models/

2. Update technical_architecture/src/source_separation.rs:
   model_path: "models/checkpoints/conv_tasnet_chimpanzee.onnx"

3. Rebuild Rust library:
   cd technical_architecture && cargo build --release
""")

    return model, onnx_path


if __name__ == "__main__":
    main()
