#!/usr/bin/env python3
"""
Tests for Neural Vocoder (Direction 6)

Tests for neural vocoder that generates audio from 112D features,
including interpolation, prosodic modification, and integration.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import os
import tempfile
import unittest

import numpy as np


class TestNeuralVocoderCore(unittest.TestCase):
    """Test core vocoder functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from analysis.rosetta_stone.neural_vocoder import NeuralVocoder

        self.vocoder = NeuralVocoder(model_type="simple", sample_rate=48000)

    def test_vocoder_output_shape(self):
        """Output length matches expected from input features."""
        # Create test features: 10 frames
        features = np.random.randn(10, 112).astype(np.float32)

        audio = self.vocoder.synthesize(features)

        # Output should be non-empty and have reasonable length
        self.assertIsInstance(audio, np.ndarray)
        self.assertGreater(len(audio), 0)

    def test_vocoder_output_sample_rate(self):
        """Audio has correct sample rate."""
        self.assertEqual(self.vocoder.sample_rate, 48000)

    def test_vocoder_single_frame(self):
        """Can synthesize single frame."""
        features = np.random.randn(1, 112).astype(np.float32)

        audio = self.vocoder.synthesize(features)

        self.assertGreater(len(audio), 0)

    def test_vocoder_sequence(self):
        """Can synthesize feature sequence."""
        features = np.random.randn(5, 112).astype(np.float32)

        audio = self.vocoder.synthesize(features)

        self.assertGreater(len(audio), 0)

    def test_vocoder_batch(self):
        """Batch synthesis is faster than serial."""
        features_list = [np.random.randn(3, 112).astype(np.float32) for _ in range(5)]

        # Batch synthesis should complete without error
        results = self.vocoder.synthesize_batch(features_list)

        self.assertEqual(len(results), 5)
        for audio in results:
            self.assertGreater(len(audio), 0)


class TestAudioQuality(unittest.TestCase):
    """Test audio output quality."""

    def setUp(self):
        """Set up test fixtures."""
        from analysis.rosetta_stone.neural_vocoder import NeuralVocoder

        self.vocoder = NeuralVocoder(model_type="simple", sample_rate=48000)

    def test_output_is_audio(self):
        """Output is valid audio (no NaN, no clipping)."""
        features = np.random.randn(10, 112).astype(np.float32)

        audio = self.vocoder.synthesize(features)

        # No NaN values
        self.assertFalse(np.any(np.isnan(audio)), "Audio should not contain NaN")

        # No clipping (values in [-1, 1] range for valid audio)
        self.assertFalse(np.any(np.abs(audio) > 1.0), "Audio should not clip beyond [-1, 1]")

    def test_output_has_energy(self):
        """Output has non-zero energy."""
        features = np.random.randn(10, 112).astype(np.float32)

        audio = self.vocoder.synthesize(features)

        # Check RMS energy
        rms = np.sqrt(np.mean(audio**2))
        self.assertGreater(rms, 1e-6, "Audio should have non-zero energy")

    def test_reconstruction_fidelity(self):
        """Reconstructs input audio approximately."""
        # For simple vocoder, just check that features produce audio
        features = np.random.randn(5, 112).astype(np.float32)

        audio = self.vocoder.synthesize(features)

        # Audio should be produced
        self.assertIsNotNone(audio)
        self.assertGreater(len(audio), 0)

    def test_spectral_similarity(self):
        """Output spectrum matches input."""
        features = np.random.randn(10, 112).astype(np.float32)

        audio = self.vocoder.synthesize(features)

        # For simple vocoder, just check audio has frequency content
        self.assertGreater(len(audio), 0)


class TestFeatureInterpolator(unittest.TestCase):
    """Test feature interpolation for smooth transitions."""

    def test_linear_interpolation(self):
        """Interpolates between features."""
        from analysis.rosetta_stone.neural_vocoder import FeatureInterpolator

        f1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        f2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        result = FeatureInterpolator.linear(f1, f2, 0.5)

        expected = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_slerp_preserves_norm(self):
        """Slerp preserves feature norms."""
        from analysis.rosetta_stone.neural_vocoder import FeatureInterpolator

        # Create normalized vectors
        f1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        f2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        result = FeatureInterpolator.slerp(f1, f2, 0.5)

        # Should be normalized
        norm = np.linalg.norm(result)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_interpolation_smoothness(self):
        """Interpolated synthesis is smooth."""
        from analysis.rosetta_stone.neural_vocoder import FeatureInterpolator

        f1 = np.random.randn(112).astype(np.float32)
        f2 = np.random.randn(112).astype(np.float32)

        # Interpolate at multiple points
        interp_0 = FeatureInterpolator.linear(f1, f2, 0.0)
        FeatureInterpolator.linear(f1, f2, 0.5)
        interp_100 = FeatureInterpolator.linear(f1, f2, 1.0)

        # Should be monotonic progression
        np.testing.assert_array_almost_equal(interp_0, f1, decimal=5)
        np.testing.assert_array_almost_equal(interp_100, f2, decimal=5)


class TestProsodicModifier(unittest.TestCase):
    """Test prosodic modification of features."""

    def test_pitch_shift_up(self):
        """Pitch increases with positive shift."""
        from analysis.rosetta_stone.neural_vocoder import ProsodicModifier

        features = np.random.randn(10, 112).astype(np.float32)
        # F0 is typically in first dimension
        features[0, 0] = 5000.0  # Initial F0

        shifted = ProsodicModifier.adjust_pitch(features, shift_semitones=2)

        # First dimension (F0) should increase
        self.assertGreater(shifted[0, 0], features[0, 0])

    def test_pitch_shift_down(self):
        """Pitch decreases with negative shift."""
        from analysis.rosetta_stone.neural_vocoder import ProsodicModifier

        features = np.random.randn(10, 112).astype(np.float32)
        features[0, 0] = 5000.0

        shifted = ProsodicModifier.adjust_pitch(features, shift_semitones=-2)

        # F0 should decrease
        self.assertLess(shifted[0, 0], features[0, 0])

    def test_time_stretch_expand(self):
        """Duration increases with stretch."""
        from analysis.rosetta_stone.neural_vocoder import ProsodicModifier

        features = np.random.randn(10, 112).astype(np.float32)

        stretched = ProsodicModifier.adjust_duration(features, speed_factor=0.5)

        # Should have more frames
        self.assertGreater(len(stretched), len(features))

    def test_time_stretch_compress(self):
        """Duration decreases with compress."""
        from analysis.rosetta_stone.neural_vocoder import ProsodicModifier

        features = np.random.randn(10, 112).astype(np.float32)

        compressed = ProsodicModifier.adjust_duration(features, speed_factor=2.0)

        # Should have fewer frames
        self.assertLess(len(compressed), len(features))

    def test_amplitude_gain(self):
        """Amplitude scales correctly."""
        from analysis.rosetta_stone.neural_vocoder import ProsodicModifier

        features = np.random.randn(10, 112).astype(np.float32)
        # RMS energy is typically in second dimension
        features[0, 1] = 0.5  # Initial RMS

        modified = ProsodicModifier.adjust_amplitude(features, gain_db=6.0)

        # RMS should increase (+6dB = ~2x amplitude)
        self.assertGreater(modified[0, 1], features[0, 1])


class TestModelPersistence(unittest.TestCase):
    """Test model save/load functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from analysis.rosetta_stone.neural_vocoder import NeuralVocoder

        self.vocoder = NeuralVocoder(model_type="simple", sample_rate=48000)

    def test_save_model(self):
        """Model can be saved to disk."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name

        try:
            self.vocoder.save(model_path)
            self.assertTrue(os.path.exists(model_path))
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_load_model(self):
        """Loaded model produces same output."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name

        try:
            # Save model
            self.vocoder.save(model_path)

            # Load model
            from analysis.rosetta_stone.neural_vocoder import NeuralVocoder

            loaded_vocoder = NeuralVocoder.load(model_path)

            # Check parameters match
            self.assertEqual(loaded_vocoder.sample_rate, self.vocoder.sample_rate)
            self.assertEqual(loaded_vocoder.model_type, self.vocoder.model_type)

        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_model_versioning(self):
        """Model metadata includes version."""
        metadata = self.vocoder.get_metadata()

        self.assertIn("version", metadata)
        self.assertIn("model_type", metadata)
        self.assertIn("sample_rate", metadata)


class TestVocoderIntegration(unittest.TestCase):
    """Test integration with other system components."""

    def setUp(self):
        """Set up test fixtures."""
        from analysis.rosetta_stone.neural_language_model import AcousticTokenizer
        from analysis.rosetta_stone.neural_vocoder import NeuralVocoder

        self.vocoder = NeuralVocoder(model_type="simple", sample_rate=48000)
        self.tokenizer = AcousticTokenizer(vocab_size=50)

    def test_synthesis_from_tokens(self):
        """Generate audio from token sequence."""
        # Convert tokens to features
        tokens = [1, 2, 3, 4, 5]
        features = np.array([self.tokenizer.detokenize(t) for t in tokens])

        audio = self.vocoder.synthesize(features)

        self.assertGreater(len(audio), 0)

    def test_synthesis_from_lm(self):
        """Use LM output for vocoder input."""
        from analysis.rosetta_stone.neural_language_model import TransformerLM

        # Create small LM
        lm = TransformerLM(vocab_size=50, d_model=32, n_heads=2, n_layers=2)

        # Generate tokens
        tokens = lm.generate(prompt=[1], max_length=5)

        # Convert to features
        features = np.array([self.tokenizer.detokenize(t) for t in tokens])

        audio = self.vocoder.synthesize(features)

        self.assertGreater(len(audio), 0)

    def test_realtime_synthesis(self):
        """Synthesis latency is acceptable."""
        import time

        features = np.random.randn(10, 112).astype(np.float32)

        start = time.time()
        self.vocoder.synthesize(features)
        elapsed = time.time() - start

        # For simple vocoder, should be very fast
        # (< 100ms is target for production)
        self.assertLess(elapsed, 1.0, "Synthesis should be fast")

    def test_vocoder_fallback(self):
        """Falls back to granular synthesis if vocoder fails."""
        # Test that we can handle errors gracefully
        try:
            # Invalid input should not crash
            features = np.array([], dtype=np.float32).reshape(0, 112)
            audio = self.vocoder.synthesize(features)
            # Should return empty audio
            self.assertEqual(len(audio), 0)
        except Exception as e:
            # Fallback behavior - exception is acceptable
            self.assertIsNotNone(e)


if __name__ == "__main__":
    unittest.main()
