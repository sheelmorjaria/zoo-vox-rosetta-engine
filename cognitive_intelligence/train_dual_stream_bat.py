#!/usr/bin/env python3
"""
Train Dual-Stream Models on Egyptian Fruit Bat Dataset

Uses the pre-extracted 112D features from:
/data/egyptian_fruit_bats/extraction_112d/extraction_112d_labeled.json

Trains:
1. β-VAE (16D affect encoder)
2. VQ-VAE (64-token syntactic encoder)

Dataset: 8.9M segments from 91K annotated vocalizations

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from cognitive_intelligence.train_beta_vae import (
    BetaVAETrainingConfig,
    BetaVAETrainer,
    train_beta_vae,
)
from cognitive_intelligence.train_vqvae import (
    VQVAETrainingConfig,
    VQVAETrainer,
    train_vqvae,
)
from cognitive_intelligence.affective_vae import BetaVAE
from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for the Egyptian Fruit Bat dataset."""

    # Data paths
    data_root: str = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats"
    features_file: str = "extraction_112d/extraction_112d_labeled.json"
    annotations_file: str = "annotations.csv"

    # Sampling (8.9M is too much for quick training)
    max_samples: int = 100000  # Use 100k samples for training
    val_split: int = 10000     # 10k for validation
    test_split: int = 10000    # 10k for testing
    random_seed: int = 42


class BatDatasetLoader:
    """Load and sample from the Egyptian Fruit Bat dataset."""

    # Affective feature indices (subset of 112D)
    AFFECTIVE_INDICES = [
        0, 1, 2, 3, 4, 5,      # F0 stats
        6, 7,                   # RMS stats
        12, 13, 14, 15,         # HNR stats
        40, 41,                 # Jitter, Shimmer
        59, 60, 61, 62, 63,     # GLCM texture
        *range(76, 112),         # Micro texture (36D)
    ]

    # Syntactic feature indices
    SYNTACTIC_INDICES = [
        0, 1, 2, 3, 4,          # F0 stats
        6, 7, 8, 9, 10, 11,     # RMS, duration, voiced ratio
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,  # MFCCs
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55,  # Harmonic features
        *range(56, 76),         # Pitch geometry (20D)
    ]

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        self.features_path = Path(config.data_root) / config.features_file

    def load_features(self) -> np.ndarray:
        """
        Load and sample 112D features from the JSON file.

        Returns:
            (N, 112) array of features
        """
        logger.info(f"Loading features from {self.features_path}")

        # Stream the JSON file (it's 20GB!)
        features_list = []

        with open(self.features_path, 'r') as f:
            data = json.load(f)

        num_segments = data['num_segments']
        logger.info(f"Total segments in dataset: {num_segments:,}")

        # Sample segments
        if num_segments > self.config.max_samples:
            # For large datasets, use random sampling
            indices = self.rng.choice(
                num_segments,
                self.config.max_samples + self.config.val_split + self.config.test_split,
                replace=False,
            )
        else:
            indices = np.arange(num_segments)
            self.rng.shuffle(indices)

        # Extract features for sampled indices
        # Note: The JSON has nested structure - we need to stream it efficiently
        logger.info("Extracting features from sampled segments...")

        # For now, let's extract a subset by reading the JSON
        # In production, you'd want to use ijson for streaming
        sampled_indices = set(indices[:self.config.max_samples])

        count = 0
        for segment in data['segments']:
            if count in sampled_indices:
                features_list.append(segment['features_112d'])

            count += 1

            if count >= self.config.max_samples:
                break

        features_array = np.array(features_list, dtype=np.float32)
        logger.info(f"Loaded {len(features_array):,} feature vectors")

        return features_array

    def load_annotations(self):
        """Load the annotations CSV."""
        import pandas as pd

        annotations_path = Path(self.config.data_root) / self.config.annotations_file
        logger.info(f"Loading annotations from {annotations_path}")

        df = pd.read_csv(annotations_path)
        logger.info(f"Loaded {len(df):,} annotations")

        # Show unique contexts
        logger.info(f"Unique contexts: {df['Context'].unique().tolist()}")

        return df


def train_dual_stream_models(
    dataset_config: Optional[DatasetConfig] = None,
    vae_config: Optional[BetaVAETrainingConfig] = None,
    vqvae_config: Optional[VQVAETrainingConfig] = None,
):
    """
    Train both β-VAE and VQ-VAE on the bat dataset.

    Args:
        dataset_config: Dataset configuration
        vae_config: β-VAE training configuration
        vqvae_config: VQ-VAE training configuration
    """
    dataset_config = dataset_config or DatasetConfig()

    # Load features
    loader = BatDatasetLoader(dataset_config)
    features_112d = loader.load_features()

    # Load annotations (for reference)
    annotations = loader.load_annotations()

    print("\n" + "=" * 60)
    print("DUAL-STREAM MODEL TRAINING")
    print("=" * 60)
    print(f"Dataset: Egyptian Fruit Bats")
    print(f"Samples: {len(features_112d):,}")
    print(f"Annotations: {len(annotations):,}")
    print(f"Unique contexts: {len(annotations['Context'].unique())}")
    print("=" * 60)

    results = {}

    # Train β-VAE (Stream 1: Affect)
    print("\n[1/2] Training β-VAE (16D Affect Encoder)...")
    print("-" * 60)

    vae_config = vae_config or BetaVAETrainingConfig(
        input_dim=68,  # AffectiveFeatureExtractor output dimension
        latent_dim=16,
        beta=2.0,
        batch_size=256,
        num_epochs=100,  # Reduced for demo
        checkpoint_dir="models/beta_vae_egyptian_bat",
    )

    try:
        vae_model, vae_trainer = train_beta_vae(features_112d, vae_config)
        results['vae'] = {
            'best_loss': vae_trainer.best_loss,
            'final_recon': vae_trainer.recon_losses[-1],
            'final_kl': vae_trainer.kl_losses[-1],
        }
        print(f"✓ β-VAE trained! Best loss: {vae_trainer.best_loss:.4f}")
    except Exception as e:
        logger.error(f"β-VAE training failed: {e}")
        results['vae'] = {'error': str(e)}

    # Train VQ-VAE (Stream 2: Syntax)
    print("\n[2/2] Training VQ-VAE (64-Token Syntactic Encoder)...")
    print("-" * 60)

    vqvae_config = vqvae_config or VQVAETrainingConfig(
        input_dim=45,  # SyntacticFeatureExtractor output dimension
        codebook_size=64,
        codebook_dim=32,
        batch_size=256,
        num_epochs=100,  # Reduced for demo
        checkpoint_dir="models/vqvae_egyptian_bat",
    )

    try:
        vqvae_model, vqvae_trainer = train_vqvae(features_112d, vqvae_config)
        results['vqvae'] = {
            'best_loss': vqvae_trainer.best_loss,
            'final_commit': vqvae_trainer.commit_losses[-1],
            'final_util': vqvae_trainer.utilization_history[-1],
        }
        print(f"✓ VQ-VAE trained! Best loss: {vqvae_trainer.best_loss:.4f}")
        print(f"  Final utilization: {vqvae_trainer.utilization_history[-1]:.1f}%")
    except Exception as e:
        logger.error(f"VQ-VAE training failed: {e}")
        results['vqvae'] = {'error': str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    if 'vae' in results and 'error' not in results['vae']:
        print(f"β-VAE: Loss={results['vae']['best_loss']:.4f}, "
              f"Recon={results['vae']['final_recon']:.4f}, "
              f"KL={results['vae']['final_kl']:.4f}")

    if 'vqvae' in results and 'error' not in results['vqvae']:
        print(f"VQ-VAE: Loss={results['vqvae']['best_loss']:.4f}, "
              f"Commit={results['vqvae']['final_commit']:.4f}, "
              f"Util={results['vqvae']['final_util']:.1f}%")

    print("\nModel checkpoints saved to:")
    print(f"  - {vae_config.checkpoint_dir}")
    print(f"  - {vqvae_config.checkpoint_dir}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Check if dataset exists
    data_path = Path("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats")
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        print("Please ensure the dataset is in the correct location.")
        exit(1)

    # Quick demo with smaller sample
    print("\n🦇 Egyptian Fruit Bat Dual-Stream Training")
    print("=" * 60)
    print("\nQuick demo configuration:")
    print("  - Samples: 100,000 (from 8.9M available)")
    print("  - Epochs: 100 (reduce for quick test)")
    print("  - Batch size: 256")
    print("\nFor full training, modify DatasetConfig in this script.")
    print("\nStarting training...\n")

    results = train_dual_stream_models()
