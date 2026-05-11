#!/usr/bin/env python3
"""
Tests for Affective VAE Training Script

Tests the training pipeline for the affective VAE including:
- Dataset loading
- Trainer initialization
- Training loop execution
- Checkpoint saving/loading

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cognitive_intelligence.train_affective_vae import (
    AffectiveVAETrainingConfig,
    AffectiveVAETrainer,
    CachedFeaturesDataset,
    SegmentsDataset,
    VocalizationSegment,
)


class TestAffectiveVAETrainingConfig(unittest.TestCase):
    """Test training configuration."""

    def test_default_config(self):
        """Should create default config."""
        config = AffectiveVAETrainingConfig()

        self.assertEqual(config.input_dim, 54)
        self.assertEqual(config.latent_dim, 16)
        self.assertEqual(config.beta, 2.0)
        self.assertEqual(config.batch_size, 64)

    def test_custom_config(self):
        """Should accept custom parameters."""
        config = AffectiveVAETrainingConfig(
            latent_dim=32,
            beta=4.0,
            batch_size=128,
        )

        self.assertEqual(config.latent_dim, 32)
        self.assertEqual(config.beta, 4.0)
        self.assertEqual(config.batch_size, 128)


class TestCachedFeaturesDataset(unittest.TestCase):
    """Test dataset loading from cached features."""

    def setUp(self):
        """Create temporary test data."""
        # Create mock 112D features
        self.temp_features = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npy", delete=False
        )
        features = np.random.randn(100, 112).astype(np.float32)
        np.save(self.temp_features.name, features)
        self.temp_features.close()

    def tearDown(self):
        """Clean up temporary files."""
        Path(self.temp_features.name).unlink(missing_ok=True)

    def test_load_from_npy(self):
        """Should load and extract affective features."""
        dataset = CachedFeaturesDataset(self.temp_features.name, normalize=True)

        self.assertEqual(len(dataset), 100)
        self.assertEqual(dataset.features_affective.shape[1], 54)

    def test_normalization(self):
        """Should normalize features."""
        dataset = CachedFeaturesDataset(self.temp_features.name, normalize=True)

        # Check mean is approximately 0
        self.assertAlmostEqual(dataset.features_affective.mean(), 0.0, places=5)

    def test_getitem(self):
        """Should return tensors."""
        dataset = CachedFeaturesDataset(self.temp_features.name, normalize=True)

        item = dataset[0]

        self.assertIsInstance(item, torch.Tensor)
        self.assertEqual(item.shape, (54,))


class TestSegmentsDataset(unittest.TestCase):
    """Test dataset loading from JSON segments."""

    def setUp(self):
        """Create temporary test data."""
        self.temp_json = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )

        # Create mock segments
        segments = {
            "segments": [
                {
                    "features_112d": np.random.randn(112).tolist(),
                    "species": "marmoset",
                    "phrase_id": "test_001",
                }
                for _ in range(50)
            ]
        }

        json.dump(segments, self.temp_json)
        self.temp_json.close()

    def tearDown(self):
        """Clean up temporary files."""
        Path(self.temp_json.name).unlink(missing_ok=True)

    def test_load_from_json(self):
        """Should load and parse segments."""
        dataset = SegmentsDataset(self.temp_json.name, normalize=True)

        self.assertEqual(len(dataset), 50)
        self.assertEqual(dataset.features_affective.shape[1], 54)


class TestAffectiveVAETrainer(unittest.TestCase):
    """Test VAE trainer."""

    def setUp(self):
        """Create trainer config."""
        self.config = AffectiveVAETrainingConfig(
            batch_size=4,
            epochs=2,
            patience=10,
            checkpoint_dir=tempfile.mkdtemp(),
        )

    def test_trainer_initialization(self):
        """Should initialize trainer and model."""
        trainer = AffectiveVAETrainer(self.config)

        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.scheduler)

    def test_model_architecture(self):
        """Should create model with correct architecture."""
        trainer = AffectiveVAETrainer(self.config)

        # Check model parameters
        params = sum(p.numel() for p in trainer.model.parameters())
        self.assertGreater(params, 0)

    def test_save_load_checkpoint(self):
        """Should save and load checkpoint."""
        trainer = AffectiveVAETrainer(self.config)

        # Save checkpoint
        trainer.save_checkpoint("test_checkpoint.pt")

        # Check file exists
        checkpoint_path = Path(self.config.checkpoint_dir) / "test_checkpoint.pt"
        self.assertTrue(checkpoint_path.exists())


class TestTrainingLoop(unittest.TestCase):
    """Test training loop execution."""

    def setUp(self):
        """Create test dataset and trainer."""
        # Create mock dataset
        self.temp_features = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npy", delete=False
        )
        features = np.random.randn(50, 112).astype(np.float32)
        np.save(self.temp_features.name, features)
        self.temp_features.close()

        self.dataset = CachedFeaturesDataset(self.temp_features.name, normalize=True)

        self.config = AffectiveVAETrainingConfig(
            batch_size=4,
            epochs=1,
            patience=10,
            checkpoint_dir=tempfile.mkdtemp(),
        )
        self.trainer = AffectiveVAETrainer(self.config)

    def tearDown(self):
        """Clean up temporary files."""
        Path(self.temp_features.name).unlink(missing_ok=True)

    def test_train_epoch(self):
        """Should train for one epoch."""
        from torch.utils.data import DataLoader

        loader = DataLoader(self.dataset, batch_size=4, shuffle=True)

        train_loss, recon_loss, kl_loss = self.trainer.train_epoch(loader)

        self.assertIsInstance(train_loss, float)
        self.assertGreater(train_loss, 0)

    def test_validate(self):
        """Should validate model."""
        from torch.utils.data import DataLoader

        loader = DataLoader(self.dataset, batch_size=4, shuffle=False)

        val_loss, recon_loss, kl_loss = self.trainer.validate(loader)

        self.assertIsInstance(val_loss, float)
        self.assertGreater(val_loss, 0)


if __name__ == "__main__":
    unittest.main()
