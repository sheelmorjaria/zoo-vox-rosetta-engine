#!/usr/bin/env python3
"""
Module 2 TDD Tests: DDSP Decoder and Training Pipeline

This test suite verifies the DDSP decoder architecture, loss functions,
and training pipeline for the 112D → 65 DDSP parameter mapping.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
Module 2 (v1.6.0): DDSP Decoder Training Pipeline
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for PyTorch availability
torch = pytest.importorskip("torch")

from cognitive_intelligence.ddsp_decoder import (
    DDSPDecoder,
    DDSPDecoderConfig,
    DDSPDecoderLarge,
    DDSPDecoderLight,
    count_parameters,
    create_decoder,
    get_model_size_mb,
)
from cognitive_intelligence.ddsp_training import (
    DDSPDecoderTrainer,
    SyntheticDataset,
    TrainingConfig,
    train_decoder,
)
from cognitive_intelligence.multiscale_spectral_loss import (
    CombinedLoss,
    MultiScaleSpectralLoss,
    SpectralConvergenceLoss,
    TimeDomainLoss,
    create_loss_fn,
)

# =============================================================================
# TEST SUITE 1: DDSPDecoder Architecture
# =============================================================================


class TestDDSPDecoder:
    """Verify DDSPDecoder model architecture."""

    def test_decoder_initialization(self):
        """DDSPDecoder should initialize with correct dimensions."""
        decoder = DDSPDecoder()

        assert decoder.input_dim == 112
        assert decoder.output_dim == 65  # 60 harmonics + 5 noise bands
        assert decoder.num_harmonics == 60
        assert decoder.num_noise_bands == 5

    def test_decoder_forward_single_sample(self):
        """Forward pass with single sample should work."""
        decoder = DDSPDecoder()
        features = torch.randn(112)

        harmonic_amps, noise_mags = decoder(features)

        # Note: batch dimension may be added, handle both cases
        if harmonic_amps.dim() == 2:
            harmonic_amps = harmonic_amps.squeeze(0)
            noise_mags = noise_mags.squeeze(0)

        assert harmonic_amps.shape == (60,)
        assert noise_mags.shape == (5,)

        # Check output constraints
        assert pytest.approx(harmonic_amps.sum().item(), abs=0.01) == 1.0  # Sum to 1
        assert (noise_mags >= 0).all().item()  # Non-negative

    def test_decoder_forward_batch(self):
        """Forward pass with batch should work."""
        decoder = DDSPDecoder()
        batch_size = 8
        features = torch.randn(batch_size, 112)

        harmonic_amps, noise_mags = decoder(features)

        assert harmonic_amps.shape == (batch_size, 60)
        assert noise_mags.shape == (batch_size, 5)

        # Each row should sum to 1 for harmonics
        for i in range(batch_size):
            assert pytest.approx(harmonic_amps[i].sum().item(), abs=0.01) == 1.0

    def test_decoder_parameter_count(self):
        """Decoder should have reasonable parameter count."""
        decoder = DDSPDecoder()
        params = count_parameters(decoder)

        # Base model with 256 hidden dim
        # 112*256 + 256 + 256*256 + 256 + 256*65 + 65 = ~92K parameters
        assert params > 50000
        assert params < 200000

    def test_decoder_light_variant(self):
        """Light variant should have fewer parameters."""
        base_decoder = DDSPDecoder()
        light_decoder = DDSPDecoderLight()

        base_params = count_parameters(base_decoder)
        light_params = count_parameters(light_decoder)

        assert light_params < base_params

    def test_decoder_large_variant(self):
        """Large variant should have more parameters."""
        base_decoder = DDSPDecoder()
        large_decoder = DDSPDecoderLarge()

        base_params = count_parameters(base_decoder)
        large_params = count_parameters(large_decoder)

        assert large_params > base_params

    def test_decoder_inference_mode(self):
        """Inference mode should return structured output."""
        decoder = DDSPDecoder()
        features = torch.randn(112)

        result = decoder.inference(features)

        assert "harmonic_amps" in result
        assert "noise_mags" in result
        assert "confidence" in result

        # Handle both single sample and batch output
        harmonic = result["harmonic_amps"]
        noise = result["noise_mags"]

        if harmonic.dim() == 2:
            harmonic = harmonic.squeeze(0)
            noise = noise.squeeze(0)

        assert harmonic.shape == (60,)
        assert noise.shape == (5,)
        assert 0 <= result["confidence"].item() <= 1


# =============================================================================
# TEST SUITE 2: MultiScale Spectral Loss
# =============================================================================


class TestMultiScaleSpectralLoss:
    """Verify multi-scale spectral loss implementation."""

    def test_loss_initialization(self):
        """Loss function should initialize with default config."""
        loss_fn = MultiScaleSpectralLoss()

        assert len(loss_fn.frame_lengths) == 4
        assert 512 in loss_fn.frame_lengths
        assert 4096 in loss_fn.frame_lengths

    def test_loss_computation(self):
        """Loss computation should return scalar."""
        loss_fn = MultiScaleSpectralLoss()

        # Create sample audio
        pred = torch.randn(1, 16000)
        target = torch.randn(1, 16000)

        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0

    def test_loss_batch_computation(self):
        """Loss should work with batches."""
        loss_fn = MultiScaleSpectralLoss()

        batch_size = 4
        pred = torch.randn(batch_size, 16000)
        target = torch.randn(batch_size, 16000)

        loss = loss_fn(pred, target)

        assert loss.item() >= 0

    def test_loss_identical_audio(self):
        """Loss for identical audio should be zero."""
        loss_fn = MultiScaleSpectralLoss()

        audio = torch.randn(1, 16000)
        loss = loss_fn(audio, audio)

        # Due to floating point, should be very small
        assert loss.item() < 0.01

    def test_spectral_convergence_loss(self):
        """Spectral convergence loss should compute correctly."""
        loss_fn = SpectralConvergenceLoss()

        audio1 = torch.randn(1, 16000)
        audio2 = torch.randn(1, 16000)

        loss = loss_fn(audio1, audio2)

        assert 0 <= loss.item() <= 2  # Cosine distance: [0, 2]

    def test_time_domain_loss(self):
        """Time-domain loss should compute correctly."""
        loss_fn = TimeDomainLoss()

        pred = torch.randn(1, 16000)
        target = torch.randn(1, 16000)

        loss = loss_fn(pred, target)

        assert loss.item() >= 0

    def test_combined_loss(self):
        """Combined loss should return scalar and components."""
        loss_fn = CombinedLoss()

        pred = torch.randn(1, 16000)
        target = torch.randn(1, 16000)

        total, losses = loss_fn(pred, target)

        assert isinstance(total, torch.Tensor)
        assert isinstance(losses, dict)
        assert "total" in losses
        assert "spectral" in losses
        assert "time" in losses


# =============================================================================
# TEST SUITE 3: Training Pipeline
# =============================================================================


class TestTrainingPipeline:
    """Verify training pipeline components."""

    def test_training_config_defaults(self):
        """TrainingConfig should have sensible defaults."""
        config = TrainingConfig()

        assert config.batch_size == 32
        assert config.num_epochs == 100
        assert config.learning_rate == 1e-3
        assert config.device == "cuda" if torch.cuda.is_available() else "cpu"

    def test_synthetic_dataset(self):
        """SyntheticDataset should generate valid data."""
        dataset = SyntheticDataset(num_samples=10, duration_ms=100.0)

        assert len(dataset) == 10

        features, audio = dataset[0]

        assert features.shape == (112,)
        assert audio.shape == (4800,)  # 100ms at 48kHz

    def test_synthetic_dataset_batch(self):
        """DataLoader should batch synthetic data correctly."""
        dataset = SyntheticDataset(num_samples=16, duration_ms=100.0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        features, audio = next(iter(loader))

        assert features.shape == (4, 112)
        assert audio.shape == (4, 4800)

    def test_trainer_initialization(self):
        """Trainer should initialize correctly."""
        config = TrainingConfig(
            num_epochs=1,
            batch_size=2,
            use_synthetic_data=True,
            synthetic_samples=10,
            num_workers=0,  # Avoid multiprocessing issues in tests
        )

        model_config = DDSPDecoderConfig()
        model = DDSPDecoder(config=model_config)
        trainer = DDSPDecoderTrainer(model, config)

        assert trainer.current_epoch == 0
        assert trainer.best_val_loss == float("inf")

    def test_trainer_setup_data(self):
        """Trainer should create dataloaders correctly."""
        config = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            use_synthetic_data=True,
            synthetic_samples=20,
            val_split=0.2,
            num_workers=0,
        )

        model_config = DDSPDecoderConfig()
        model = DDSPDecoder(config=model_config)
        trainer = DDSPDecoderTrainer(model, config)

        train_loader, val_loader = trainer.setup_data()

        assert len(train_loader) > 0
        assert val_loader is not None

    @pytest.mark.slow
    def test_trainer_train_epoch(self):
        """Trainer should complete one training epoch."""
        config = TrainingConfig(
            num_epochs=1,
            batch_size=4,
            use_synthetic_data=True,
            synthetic_samples=20,
            log_every=1,
            num_workers=0,
        )

        model_config = DDSPDecoderConfig()
        model = DDSPDecoder(config=model_config)
        trainer = DDSPDecoderTrainer(model, config)

        train_loader, _ = trainer.setup_data()
        metrics = trainer.train_epoch(train_loader)

        assert "loss" in metrics
        assert "spectral" in metrics
        assert "time" in metrics
        assert metrics["loss"] > 0


# =============================================================================
# TEST SUITE 4: Integration Tests
# =============================================================================


class TestDDSPIntegration:
    """Integration tests for complete DDSP pipeline."""

    def test_full_forward_pass(self):
        """Complete forward pass through decoder."""
        decoder = DDSPDecoder()

        # Batch of features
        features = torch.randn(8, 112)

        # Forward pass
        harmonic_amps, noise_mags = decoder(features)

        # Verify shapes
        assert harmonic_amps.shape == (8, 60)
        assert noise_mags.shape == (8, 5)

        # Verify constraints
        assert (harmonic_amps >= 0).all().item()
        assert (noise_mags >= 0).all().item()

        # Harmonic amplitudes should sum to ~1 per sample
        sums = harmonic_amps.sum(dim=1)
        for s in sums:
            assert pytest.approx(s.item(), abs=0.01) == 1.0

    def test_train_decoder_convenience_function(self):
        """Convenience function should work with synthetic data."""
        # This is a minimal training run
        with tempfile.TemporaryDirectory() as tmpdir:
            model = train_decoder(
                use_synthetic_data=True,
                synthetic_samples=20,
                num_epochs=1,
                batch_size=4,
                checkpoint_dir=tmpdir,
                num_workers=0,
            )

            assert isinstance(model, DDSPDecoder)

            # Check checkpoint was created
            checkpoint_files = os.listdir(tmpdir)
            assert len(checkpoint_files) > 0

    def test_factory_function(self):
        """Factory function should create correct variants."""
        base_decoder = create_decoder("base")
        light_decoder = create_decoder("light")
        large_decoder = create_decoder("large")

        base_params = count_parameters(base_decoder)
        light_params = count_parameters(light_decoder)
        large_params = count_parameters(large_decoder)

        assert light_params < base_params < large_params

    def test_loss_factory_function(self):
        """Loss factory should create correct loss types."""
        spectral_loss = create_loss_fn("multiscale_spectral")
        convergence_loss = create_loss_fn("spectral_convergence")
        time_loss = create_loss_fn("time_domain")
        combined_loss = create_loss_fn("combined")

        assert isinstance(spectral_loss, MultiScaleSpectralLoss)
        assert isinstance(convergence_loss, SpectralConvergenceLoss)
        assert isinstance(time_loss, TimeDomainLoss)
        assert isinstance(combined_loss, CombinedLoss)


# =============================================================================
# TEST SUITE 5: Edge Cases
# =============================================================================


class TestDDSPEdgeCases:
    """Test edge cases and error handling."""

    def test_decoder_empty_batch(self):
        """Decoder should handle empty batch gracefully."""
        decoder = DDSPDecoder()
        features = torch.randn(0, 112)

        # Should produce empty output
        harmonic_amps, noise_mags = decoder(features)

        assert harmonic_amps.shape == (0, 60)
        assert noise_mags.shape == (0, 5)

    def test_decoder_wrong_input_dimensions(self):
        """Decoder should handle wrong input dimensions."""
        decoder = DDSPDecoder()

        # Wrong dimension (100 instead of 112)
        features_wrong = torch.randn(100)

        # Should fail when passing through linear layer
        with pytest.raises(RuntimeError):
            harmonic_amps, noise_mags = decoder(features_wrong)

    def test_loss_mismatched_lengths(self):
        """Loss should handle different length audio."""
        loss_fn = MultiScaleSpectralLoss()

        pred = torch.randn(1, 16000)
        target = torch.randn(1, 8000)  # Half length

        # Should still compute loss (crop or pad internally)
        loss = loss_fn(pred, target)

        assert loss.item() >= 0

    def test_model_size_calculation(self):
        """Model size calculation should be accurate."""
        decoder = DDSPDecoder()

        size_mb = get_model_size_mb(decoder)

        assert size_mb > 0
        assert size_mb < 10  # Should be less than 10MB


# =============================================================================
# PyTorch Availability Check
# =============================================================================

if __name__ == "__main__":
    # Check if PyTorch is available
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        pytest.main([__file__, "-v"])
    except ImportError:
        print("PyTorch not available. Skipping tests.")
        print("Install with: pip install torch")
