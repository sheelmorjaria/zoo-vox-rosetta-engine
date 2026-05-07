#!/usr/bin/env python3
"""
Tests for Neural Post-Filter Training Pipeline.

Module 4 (v1.6.0): Tests for training the neural post-filter that refines
DDSP output to match real bat vocalizations.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for PyTorch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from cognitive_intelligence.train_post_filter import (
        PostFilterDataset,
        PostFilterTrainer,
        PostFilterTrainingConfig,
        SyntheticPostFilterDataset,
        export_post_filter_for_jetson,
        train_post_filter,
    )
    from realtime.ddsp_agent import NeuralPostFilter


# =============================================================================
# Synthetic Dataset Tests
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestSyntheticPostFilterDataset:
    """Test synthetic dataset for post-filter training."""

    def test_synthetic_dataset_length(self):
        """Synthetic dataset should have correct length."""
        dataset = SyntheticPostFilterDataset(num_samples=100)

        assert len(dataset) == 100

    def test_synthetic_dataset_output_shapes(self):
        """Synthetic dataset should produce correct output shapes."""
        dataset = SyntheticPostFilterDataset(num_samples=10, duration_ms=200.0)

        ddsp_audio, target_audio, harmonic_amps, noise_mags = dataset[0]

        assert ddsp_audio.shape[0] == 9600  # 200ms at 48kHz
        assert target_audio.shape[0] == 9600
        assert harmonic_amps.shape[0] == 60
        assert noise_mags.shape[0] == 5

    def test_synthetic_dataset_normalization(self):
        """Synthetic audio should be normalized."""
        dataset = SyntheticPostFilterDataset(num_samples=10)

        for i in range(len(dataset)):
            ddsp_audio, target_audio, _, _ = dataset[i]

            # Check that audio is normalized (peak <= 1.0)
            assert ddsp_audio.abs().max() <= 1.0
            assert target_audio.abs().max() <= 1.0


# =============================================================================
# Post-Filter Model Tests
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestNeuralPostFilterModel:
    """Test NeuralPostFilter model."""

    def test_model_forward_pass(self):
        """Model should process audio and parameters correctly."""
        model = NeuralPostFilter(num_harmonics=60, num_noise_bands=5)

        batch_size = 2
        audio_length = 4800

        ddsp_audio = torch.randn(batch_size, audio_length)
        harmonic_amps = torch.randn(batch_size, 60)
        noise_mags = torch.rand(batch_size, 5)

        output = model(ddsp_audio, harmonic_amps, noise_mags)

        assert output.shape == (batch_size, audio_length)

    def test_model_is_differentiable(self):
        """Model should be differentiable for training."""
        model = NeuralPostFilter(num_harmonics=60, num_noise_bands=5)

        ddsp_audio = torch.randn(1, 4800, requires_grad=True)
        harmonic_amps = torch.randn(1, 60)
        noise_mags = torch.rand(1, 5)

        output = model(ddsp_audio, harmonic_amps, noise_mags)
        loss = output.mean()
        loss.backward()

        # Gradients should be computed
        assert ddsp_audio.grad is not None

    def test_model_parameter_count(self):
        """Model should be lightweight (<100K parameters)."""
        model = NeuralPostFilter(num_harmonics=60, num_noise_bands=5)

        param_count = sum(p.numel() for p in model.parameters())

        assert param_count < 100000, f"Model has {param_count} parameters, expected <100K"

    def test_model_residual_connection(self):
        """Model should use residual connection (refinement added to input)."""
        model = NeuralPostFilter(num_harmonics=60, num_noise_bands=5)
        model.eval()

        ddsp_audio = torch.randn(1, 4800)
        harmonic_amps = torch.randn(1, 60)
        noise_mags = torch.rand(1, 5)

        with torch.no_grad():
            output = model(ddsp_audio, harmonic_amps, noise_mags)

        # Output should not be identical to input (refinement applied)
        assert not torch.allclose(output, ddsp_audio)


# =============================================================================
# Training Configuration Tests
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPostFilterTrainingConfig:
    """Test training configuration."""

    def test_default_config_values(self):
        """Default configuration should have sensible values."""
        config = PostFilterTrainingConfig()

        assert config.num_epochs > 0
        assert config.batch_size > 0
        assert config.learning_rate > 0
        assert config.num_harmonics == 60
        assert config.num_noise_bands == 5

    def test_config_override(self):
        """Configuration values should be overridable."""
        config = PostFilterTrainingConfig(
            num_epochs=5,
            batch_size=8,
            learning_rate=1e-4,
        )

        assert config.num_epochs == 5
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4


# =============================================================================
# Trainer Tests
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPostFilterTrainer:
    """Test post-filter trainer."""

    def test_trainer_initialization(self):
        """Trainer should initialize correctly."""
        model = NeuralPostFilter(num_harmonics=60, num_noise_bands=5)
        config = PostFilterTrainingConfig(
            use_synthetic_data=True,
            synthetic_samples=50,
            num_epochs=2,
            batch_size=4,
        )

        trainer = PostFilterTrainer(model, config)

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_trainer_setup_synthetic_data(self):
        """Trainer should setup synthetic dataloaders."""
        model = NeuralPostFilter(num_harmonics=60, num_noise_bands=5)
        config = PostFilterTrainingConfig(
            use_synthetic_data=True,
            synthetic_samples=50,
            num_epochs=2,
            batch_size=4,
            val_split=0.2,
        )

        trainer = PostFilterTrainer(model, config)
        train_loader, val_loader = trainer.setup_data()

        assert train_loader is not None
        assert val_loader is not None

    def test_trainer_training_step(self):
        """Trainer should complete a training step."""
        model = NeuralPostFilter(num_harmonics=60, num_noise_bands=5)
        config = PostFilterTrainingConfig(
            use_synthetic_data=True,
            synthetic_samples=20,
            num_epochs=1,
            batch_size=4,
            log_every=1,
        )

        trainer = PostFilterTrainer(model, config)
        train_loader, _ = trainer.setup_data()

        # Train one epoch
        metrics = trainer.train_epoch(train_loader)

        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_trainer_validation_step(self):
        """Trainer should complete a validation step."""
        model = NeuralPostFilter(num_harmonics=60, num_noise_bands=5)
        config = PostFilterTrainingConfig(
            use_synthetic_data=True,
            synthetic_samples=20,
            num_epochs=1,
            batch_size=4,
            val_split=0.5,
        )

        trainer = PostFilterTrainer(model, config)
        _, val_loader = trainer.setup_data()

        if val_loader is not None:
            metrics = trainer.validate(val_loader)

            assert "loss" in metrics
            assert metrics["loss"] > 0

    def test_trainer_checkpoint_save(self, tmp_path):
        """Trainer should save checkpoints."""
        model = NeuralPostFilter(num_harmonics=60, num_noise_bands=5)
        config = PostFilterTrainingConfig(
            checkpoint_dir=str(tmp_path),
            use_synthetic_data=True,
            synthetic_samples=20,
            num_epochs=1,
            batch_size=4,
        )

        trainer = PostFilterTrainer(model, config)

        trainer.save_checkpoint("test.pt")

        checkpoint_path = tmp_path / "test.pt"
        assert checkpoint_path.exists()

    def test_trainer_checkpoint_load(self, tmp_path):
        """Trainer should load checkpoints."""
        model = NeuralPostFilter(num_harmonics=60, num_noise_bands=5)
        config = PostFilterTrainingConfig(
            checkpoint_dir=str(tmp_path),
            use_synthetic_data=True,
            synthetic_samples=20,
            num_epochs=1,
            batch_size=4,
        )

        trainer = PostFilterTrainer(model, config)
        trainer.save_checkpoint("test.pt")

        # Load checkpoint
        loaded_trainer = PostFilterTrainer.load_checkpoint(str(tmp_path / "test.pt"), device="cpu")

        assert loaded_trainer.current_epoch == trainer.current_epoch
        assert loaded_trainer.best_val_loss == trainer.best_val_loss


# =============================================================================
# Full Training Tests
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.slow
class TestFullTraining:
    """Test full training pipeline (slow tests)."""

    def test_train_post_filter_synthetic(self, tmp_path):
        """Training should complete with synthetic data."""
        model = train_post_filter(
            use_synthetic_data=True,
            synthetic_samples=50,
            num_epochs=2,
            batch_size=4,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            device="cpu",
        )

        assert model is not None

        # Check checkpoint was created
        assert (tmp_path / "checkpoints" / "best.pt").exists()

    def test_export_post_filter_onnx(self, tmp_path):
        """Export should create ONNX file."""
        model = NeuralPostFilter(num_harmonics=60, num_noise_bands=5)

        output_path = str(tmp_path / "post_filter.onnx")
        export_post_filter_for_jetson(model, output_path=output_path, device="cpu")

        assert os.path.exists(output_path)

        # Verify it's a valid ONNX file
        try:
            import onnx

            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
        except ImportError:
            pass  # Skip verification if onnx not available


# =============================================================================
# Dataset Tests
# =============================================================================


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPostFilterDataset:
    """Test PostFilterDataset with cached segments."""

    def test_dataset_from_segments(self, tmp_path):
        """Dataset should load from segments JSON."""
        # Create mock segments file
        segments = {
            "segments": [
                {
                    "features_112d": np.random.randn(112).tolist(),
                    "audio": (np.random.randn(9600) * 0.1).tolist(),
                    "f0_hz": 6000.0,
                    "sample_rate": 48000,
                }
                for _ in range(10)
            ]
        }

        segments_path = tmp_path / "segments.json"
        with open(segments_path, "w") as f:
            json.dump(segments, f)

        # Create dataset
        dataset = PostFilterDataset(
            segments_json=str(segments_path),
            duration_ms=200.0,
            device="cpu",
        )

        assert len(dataset) == 10

    def test_dataset_output_shapes(self, tmp_path):
        """Dataset should produce correct output shapes."""
        # Create mock segments
        segments = {
            "segments": [
                {
                    "features_112d": np.random.randn(112).tolist(),
                    "audio": (np.random.randn(9600) * 0.1).tolist(),
                }
            ]
        }

        segments_path = tmp_path / "segments.json"
        with open(segments_path, "w") as f:
            json.dump(segments, f)

        dataset = PostFilterDataset(
            segments_json=str(segments_path),
            duration_ms=200.0,
            device="cpu",
        )

        ddsp_audio, target_audio, harmonic_amps, noise_mags = dataset[0]

        assert ddsp_audio.shape[0] == 9600
        assert target_audio.shape[0] == 9600
        assert harmonic_amps.shape[0] == 60
        assert noise_mags.shape[0] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
