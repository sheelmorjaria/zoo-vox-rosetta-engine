#!/usr/bin/env python3
"""
Tests for Syntactic VQ-VAE Training Script

Tests the training pipeline for the syntactic VQ-VAE including:
- Dataset loading
- Trainer initialization
- Codebook tracking
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

from cognitive_intelligence.train_syntactic_vqwae import (
    SyntacticVQVAETrainingConfig,
    SyntacticVQVAETrainer,
    TrainingMetrics,
    CachedSyntacticFeaturesDataset,
    SyntacticSegmentsDataset,
)


class TestSyntacticVQVAETrainingConfig(unittest.TestCase):
    """Test training configuration."""

    def test_default_config(self):
        """Should create default config."""
        config = SyntacticVQVAETrainingConfig()

        self.assertEqual(config.input_dim, 44)
        self.assertEqual(config.codebook_size, 64)
        self.assertEqual(config.codebook_dim, 32)
        self.assertEqual(config.batch_size, 64)

    def test_custom_config(self):
        """Should accept custom parameters."""
        config = SyntacticVQVAETrainingConfig(
            codebook_size=128,
            codebook_dim=64,
            batch_size=128,
        )

        self.assertEqual(config.codebook_size, 128)
        self.assertEqual(config.codebook_dim, 64)
        self.assertEqual(config.batch_size, 128)


class TestCachedSyntacticFeaturesDataset(unittest.TestCase):
    """Test dataset loading from cached features."""

    def setUp(self):
        """Create temporary test data."""
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
        """Should load and extract syntactic features."""
        dataset = CachedSyntacticFeaturesDataset(self.temp_features.name, normalize=True)

        self.assertEqual(len(dataset), 100)
        self.assertEqual(dataset.features_syntactic.shape[1], 44)

    def test_normalization(self):
        """Should normalize features."""
        dataset = CachedSyntacticFeaturesDataset(self.temp_features.name, normalize=True)

        # Check mean is approximately 0
        self.assertAlmostEqual(dataset.features_syntactic.mean(), 0.0, places=5)

    def test_getitem(self):
        """Should return tensors."""
        dataset = CachedSyntacticFeaturesDataset(self.temp_features.name, normalize=True)

        item = dataset[0]

        self.assertIsInstance(item, torch.Tensor)
        self.assertEqual(item.shape, (44,))


class TestSyntacticSegmentsDataset(unittest.TestCase):
    """Test dataset loading from JSON segments."""

    def setUp(self):
        """Create temporary test data."""
        self.temp_json = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )

        segments = {
            "segments": [
                {
                    "features_112d": np.random.randn(112).tolist(),
                    "species": "bat",
                    "phrase_id": f"test_{i:03d}",
                }
                for i in range(50)
            ]
        }

        json.dump(segments, self.temp_json)
        self.temp_json.close()

    def tearDown(self):
        """Clean up temporary files."""
        Path(self.temp_json.name).unlink(missing_ok=True)

    def test_load_from_json(self):
        """Should load and parse segments."""
        dataset = SyntacticSegmentsDataset(self.temp_json.name, normalize=True)

        self.assertEqual(len(dataset), 50)
        self.assertEqual(dataset.features_syntactic.shape[1], 44)


class TestSyntacticVQVAETrainer(unittest.TestCase):
    """Test VQ-VAE trainer."""

    def setUp(self):
        """Create trainer config."""
        self.config = SyntacticVQVAETrainingConfig(
            batch_size=4,
            epochs=2,
            patience=10,
            checkpoint_dir=tempfile.mkdtemp(),
        )

    def test_trainer_initialization(self):
        """Should initialize trainer and model."""
        trainer = SyntacticVQVAETrainer(self.config)

        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.scheduler)

    def test_model_architecture(self):
        """Should create model with correct architecture."""
        trainer = SyntacticVQVAETrainer(self.config)

        # Check model parameters
        params = sum(p.numel() for p in trainer.model.parameters())
        self.assertGreater(params, 0)

    def test_save_load_checkpoint(self):
        """Should save and load checkpoint."""
        trainer = SyntacticVQVAETrainer(self.config)

        # Save checkpoint
        trainer.save_checkpoint("test_checkpoint.pt")

        # Check file exists
        checkpoint_path = Path(self.config.checkpoint_dir) / "test_checkpoint.pt"
        self.assertTrue(checkpoint_path.exists())

    def test_codebook_initialization(self):
        """Should initialize codebook with correct size."""
        trainer = SyntacticVQVAETrainer(self.config)

        codebook_size = trainer.model.vq.codebook_size
        codebook_dim = trainer.model.vq.codebook_dim

        self.assertEqual(codebook_size, 64)
        self.assertEqual(codebook_dim, 32)


class TestTrainingLoop(unittest.TestCase):
    """Test training loop execution."""

    def setUp(self):
        """Create test dataset and trainer."""
        self.temp_features = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npy", delete=False
        )
        features = np.random.randn(50, 112).astype(np.float32)
        np.save(self.temp_features.name, features)
        self.temp_features.close()

        self.dataset = CachedSyntacticFeaturesDataset(self.temp_features.name, normalize=True)

        self.config = SyntacticVQVAETrainingConfig(
            batch_size=4,
            epochs=1,
            patience=10,
            checkpoint_dir=tempfile.mkdtemp(),
        )
        self.trainer = SyntacticVQVAETrainer(self.config)

    def tearDown(self):
        """Clean up temporary files."""
        Path(self.temp_features.name).unlink(missing_ok=True)

    def test_train_epoch(self):
        """Should train for one epoch."""
        from torch.utils.data import DataLoader

        loader = DataLoader(self.dataset, batch_size=4, shuffle=True)

        metrics = self.trainer.train_epoch(loader)

        self.assertIsInstance(metrics, TrainingMetrics)
        self.assertGreater(metrics.total_loss, 0)
        self.assertGreater(metrics.codebook_utilization, 0)
        self.assertLessEqual(metrics.codebook_utilization, 1.0)

    def test_validate(self):
        """Should validate model."""
        from torch.utils.data import DataLoader

        loader = DataLoader(self.dataset, batch_size=4, shuffle=False)

        metrics = self.trainer.validate(loader)

        self.assertIsInstance(metrics, TrainingMetrics)
        self.assertGreater(metrics.total_loss, 0)

    def test_codebook_utilization_tracking(self):
        """Should track codebook utilization."""
        from torch.utils.data import DataLoader

        loader = DataLoader(self.dataset, batch_size=4, shuffle=True)

        metrics = self.trainer.train_epoch(loader)

        # Check utilization is reasonable
        self.assertGreater(metrics.codebook_utilization, 0.0)
        self.assertLessEqual(metrics.codebook_utilization, 1.0)

        # Check dead tokens tracking
        self.assertGreaterEqual(metrics.dead_tokens, 0)
        self.assertLessEqual(metrics.dead_tokens, self.config.codebook_size)


class TestCodebookRevival(unittest.TestCase):
    """Test codebook revival mechanism."""

    def setUp(self):
        """Create trainer with small dataset."""
        self.temp_features = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npy", delete=False
        )
        # Small dataset to trigger dead tokens
        features = np.random.randn(10, 112).astype(np.float32)
        np.save(self.temp_features.name, features)
        self.temp_features.close()

        self.dataset = CachedSyntacticFeaturesDataset(self.temp_features.name, normalize=True)

        self.config = SyntacticVQVAETrainingConfig(
            batch_size=4,
            epochs=1,
            revival_threshold=1,  # Trigger revival quickly
            revival_interval=1,
            checkpoint_dir=tempfile.mkdtemp(),
        )
        self.trainer = SyntacticVQVAETrainer(self.config)

    def tearDown(self):
        """Clean up temporary files."""
        Path(self.temp_features.name).unlink(missing_ok=True)

    def test_revive_dead_codes(self):
        """Should revive dead codebook entries."""
        # Simulate mixed tokens: some active (0), some dead (>=threshold)
        # First 10 tokens are active, rest are dead
        self.trainer.epochs_since_last_use = [0] * 10 + [5] * 54

        # Call revival
        self.trainer.revive_dead_codes()

        # Check that some counters were reset (dead tokens should be revived)
        reset_count = sum(1 for c in self.trainer.epochs_since_last_use if c == 0)
        self.assertGreater(reset_count, 10)  # At least the original 10 active tokens


if __name__ == "__main__":
    unittest.main()
