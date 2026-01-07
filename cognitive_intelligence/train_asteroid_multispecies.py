#!/usr/bin/env python3
"""
Multi-Species Asteroid Training Script
======================================

Trains a single Conv-TasNet model for multiple species or trains separate
models for each species in one run.

This script can:
1. Train a general model that works for all species
2. Train separate species-specific models
3. Train models for a subset of species

Usage:
    # Train models for all species
    python train_asteroid_multispecies.py --all

    # Train specific species
    python train_asteroid_multispecies.py --species marmoset bat

    # Train general model (wide frequency range)
    python train_asteroid_multispecies.py --general

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_asteroid_base import AnimalVocalizationDataset, AsteroidTrainer, SpeciesSpecificConfig

# Species configurations
SPECIES_CONFIGS = {
    "marmoset": {
        "f0_min": 4000,
        "f0_max": 8000,
        "sample_rate": 44100,
        "description": "Mid-frequency primate (phee calls, trills)",
    },
    "egyptian_bat": {
        "f0_min": 100,
        "f0_max": 17000,
        "sample_rate": 44100,
        "description": "High-frequency (FM sweeps, echolocation)",
    },
    "dolphin": {
        "f0_min": 500,
        "f0_max": 16000,
        "sample_rate": 44100,
        "description": "Marine mammal (whistles, clicks)",
    },
    "chimpanzee": {
        "f0_min": 100,
        "f0_max": 1900,
        "sample_rate": 44100,
        "description": "Low-frequency primate (hoots, screams)",
    },
}


def train_species_model(species_name):
    """Train model for a single species"""
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL FOR: {species_name.upper()}")
    print("=" * 80)

    if species_name not in SPECIES_CONFIGS:
        print(f"⚠️  Unknown species: {species_name}")
        return None

    species_config = SPECIES_CONFIGS[species_name]

    config = SpeciesSpecificConfig(
        species_name=species_name,
        f0_min_hz=species_config["f0_min"],
        f0_max_hz=species_config["f0_max"],
        sample_rate=species_config["sample_rate"],
    )

    config.print_config()
    print(f"Description: {species_config['description']}")

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
        print(f"\nFound {len(dataset.mixtures)} mixture files")
        print(f"Training for {config.epochs} epochs...")
        model = trainer.train(train_loader)
    else:
        print("\nNo training data found - Creating demo model...")
        model = trainer.create_model()

    # Save checkpoint
    checkpoint_path = config.checkpoint_dir / f"conv_tasnet_{species_name}.ckpt"
    trainer.save_checkpoint(model, checkpoint_path)

    # Export to ONNX
    onnx_path = config.checkpoint_dir / f"conv_tasnet_{species_name}.onnx"
    trainer.export_to_onnx(model, str(onnx_path))

    print(f"✅ Model saved to: {onnx_path}")

    return {
        "species": species_name,
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "f0_range": (config.f0_min_hz, config.f0_max_hz),
        "filter_range": (config.filter_min_hz, config.filter_max_hz),
    }


def train_general_model():
    """Train a general model for all species"""
    print("\n" + "=" * 80)
    print("TRAINING GENERAL MULTI-SPECIES MODEL")
    print("=" * 80)

    # General model: wide frequency range covering all species
    config = SpeciesSpecificConfig(
        species_name="multispecies",
        f0_min_hz=100,  # Chimpanzee minimum
        f0_max_hz=17000,  # Bat maximum
        sample_rate=44100,
    )

    config.print_config()
    print("\nDescription: General model covering all species")
    print("  - Optimized for: Mixed-species environments")
    print("  - Frequency range: 100 - 17000 Hz")
    print("  - Best for: Field deployments with unknown species")

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
        print(f"\nFound {len(dataset.mixtures)} mixture files")
        print(f"Training for {config.epochs} epochs...")
        model = trainer.train(train_loader)
    else:
        print("\nNo training data found - Creating demo model...")
        model = trainer.create_model()

    # Save checkpoint
    checkpoint_path = config.checkpoint_dir / "conv_tasnet_multispecies.ckpt"
    trainer.save_checkpoint(model, checkpoint_path)

    # Export to ONNX
    onnx_path = config.checkpoint_dir / "conv_tasnet_multispecies.onnx"
    trainer.export_to_onnx(model, str(onnx_path))

    print(f"✅ Model saved to: {onnx_path}")

    return {
        "species": "multispecies",
        "checkpoint": str(checkpoint_path),
        "onnx": str(onnx_path),
        "f0_range": (config.f0_min_hz, config.f0_max_hz),
        "filter_range": (config.filter_min_hz, config.filter_max_hz),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Asteroid models for animal vocalization source separation"
    )
    parser.add_argument("--all", action="store_true", help="Train models for all species")
    parser.add_argument(
        "--general", action="store_true", help="Train a general multi-species model"
    )
    parser.add_argument(
        "--species",
        nargs="+",
        choices=list(SPECIES_CONFIGS.keys()),
        help="Specific species to train",
    )

    args = parser.parse_args()

    # Default: train general model if no arguments
    if not args.all and not args.general and not args.species:
        print("No arguments provided. Training general model...")
        args.general = True

    results = []

    if args.general:
        result = train_general_model()
        if result:
            results.append(result)

    if args.all:
        for species_name in SPECIES_CONFIGS.keys():
            result = train_species_model(species_name)
            if result:
                results.append(result)

    if args.species:
        for species_name in args.species:
            result = train_species_model(species_name)
            if result:
                results.append(result)

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\nTrained {len(results)} models:\n")

    for i, result in enumerate(results, 1):
        species = result["species"]
        f0_min, f0_max = result["f0_range"]
        filter_min, filter_max = result["filter_range"]
        print(f"{i}. {species.upper()}")
        print(f"   F0 Range: {f0_min} - {f0_max} Hz")
        print(f"   Filter Range: {filter_min} - {filter_max} Hz")
        print(f"   ONNX: {result['onnx']}")
        print()

    print("=" * 80)
    print("MODEL SELECTION GUIDE")
    print("=" * 80)
    print("""
1. SPECIES-SPECIFIC MODELS (best performance):
   - Use when you know the target species
   - Optimized filter ranges for that species
   - Better separation quality

2. GENERAL MULTI-SPECIES MODEL (flexible):
   - Use when species is unknown or varies
   - Wide frequency range (100 - 17000 Hz)
   - Good for field deployments

3. SAMPLE RATE CONSIDERATIONS:
   - Current models use 44.1kHz
   - For bats/dolphins with ultrasound, consider 96kHz+
   - Retrain with higher sample rate if needed
""")

    return results


if __name__ == "__main__":
    main()
