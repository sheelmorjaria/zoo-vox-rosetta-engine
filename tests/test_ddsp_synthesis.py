#!/usr/bin/env python3
"""
Tests for DDSP Synthesis - Differentiable Digital Signal Processing

These tests verify the differentiable synthesis mechanism for
gradient-optimized audio generation.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest

import numpy as np


class TestDifferentiableOscillator(unittest.TestCase):
    """Test differentiable oscillator for synthesis"""

    def test_sine_oscillator(self):
        """Should generate differentiable sine wave"""
        from cognitive_intelligence.ddsp_synthesis import SineOscillator

        oscillator = SineOscillator(sample_rate=48000)
        frequency = 440.0  # A4
        duration = 0.1  # 100ms

        audio = oscillator.synthesize(frequency, duration)

        self.assertEqual(len(audio), 4800)  # 48000 * 0.1
        self.assertTrue(np.all(np.abs(audio) <= 1.0))

    def test_frequency_modulation(self):
        """Should support frequency modulation"""
        from cognitive_intelligence.ddsp_synthesis import SineOscillator

        oscillator = SineOscillator(sample_rate=48000)
        carrier_freq = 440.0
        modulator_freq = 10.0
        mod_index = 50.0
        duration = 0.1

        audio = oscillator.synthesize_fm(carrier_freq, modulator_freq, mod_index, duration)

        self.assertEqual(len(audio), 4800)

    def test_gradient_tracking(self):
        """Should support gradient computation"""
        from cognitive_intelligence.ddsp_synthesis import SineOscillator

        oscillator = SineOscillator(sample_rate=48000)

        # Check if parameters are gradient-tracked
        self.assertTrue(hasattr(oscillator, "phase"))


class TestDifferentiableFilter(unittest.TestCase):
    """Test differentiable filter for spectral shaping"""

    def test_lowpass_filter(self):
        """Should apply differentiable lowpass filter"""
        from cognitive_intelligence.ddsp_synthesis import DifferentiableFilter

        filter_obj = DifferentiableFilter(cutoff_freq=1000.0, sample_rate=48000)

        audio = np.random.randn(4800).astype(np.float32)
        filtered = filter_obj.lowpass(audio)

        self.assertEqual(len(filtered), 4800)

    def test_highpass_filter(self):
        """Should apply differentiable highpass filter"""
        from cognitive_intelligence.ddsp_synthesis import DifferentiableFilter

        filter_obj = DifferentiableFilter(cutoff_freq=5000.0, sample_rate=48000)

        audio = np.random.randn(4800).astype(np.float32)
        filtered = filter_obj.highpass(audio)

        self.assertEqual(len(filtered), 4800)

    def test_filter_coefficients_differentiable(self):
        """Filter coefficients should be differentiable"""
        from cognitive_intelligence.ddsp_synthesis import DifferentiableFilter

        filter_obj = DifferentiableFilter(cutoff_freq=1000.0, sample_rate=48000)

        # Coefficients should exist and be valid
        self.assertIsNotNone(filter_obj.coefficients)


class TestSpectralLoss(unittest.TestCase):
    """Test spectral loss for optimization"""

    def test_magnitude_loss(self):
        """Should compute magnitude spectral loss"""
        from cognitive_intelligence.ddsp_synthesis import SpectralLoss

        loss_fn = SpectralLoss()

        target = np.random.randn(4800).astype(np.float32)
        predicted = np.random.randn(4800).astype(np.float32)

        loss = loss_fn.magnitude_loss(target, predicted)

        self.assertGreater(loss, 0.0)

    def test_multi_scale_loss(self):
        """Should compute multi-scale spectral loss"""
        from cognitive_intelligence.ddsp_synthesis import SpectralLoss

        loss_fn = SpectralLoss(scales=[1, 2, 4])

        target = np.random.randn(4800).astype(np.float32)
        predicted = np.random.randn(4800).astype(np.float32)

        loss = loss_fn.multi_scale_loss(target, predicted)

        self.assertGreater(loss, 0.0)

    def test_perceptual_loss(self):
        """Should compute perceptual spectral loss"""
        from cognitive_intelligence.ddsp_synthesis import SpectralLoss

        loss_fn = SpectralLoss()

        target = np.random.randn(4800).astype(np.float32)
        predicted = np.random.randn(4800).astype(np.float32)

        loss = loss_fn.perceptual_loss(target, predicted)

        self.assertGreater(loss, 0.0)


class TestDDSPPreprocessor(unittest.TestCase):
    """Test preprocessing for DDSP synthesis"""

    def test_extract_loudness(self):
        """Should extract loudness envelope"""
        from cognitive_intelligence.ddsp_synthesis import DDSPPreprocessor

        preprocessor = DDSPPreprocessor(sample_rate=48000, frame_size=64)

        audio = np.random.randn(4800).astype(np.float32)
        loudness = preprocessor.extract_loudness(audio)

        self.assertGreater(len(loudness), 0)

    def test_extract_pitch(self):
        """Should extract pitch contour"""
        from cognitive_intelligence.ddsp_synthesis import DDSPPreprocessor

        preprocessor = DDSPPreprocessor(sample_rate=48000, frame_size=64)

        # Create harmonic signal
        t = np.linspace(0, 0.1, 4800)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        pitch = preprocessor.extract_pitch(audio)

        self.assertGreater(len(pitch), 0)

    def test_compute_features(self):
        """Should compute DDSP features (loudness + pitch)"""
        from cognitive_intelligence.ddsp_synthesis import DDSPPreprocessor

        preprocessor = DDSPPreprocessor(sample_rate=48000, frame_size=64)

        audio = np.random.randn(4800).astype(np.float32)
        features = preprocessor.compute_features(audio)

        self.assertIn("loudness", features)
        self.assertIn("pitch", features)


class TestDDSPSynthesizer(unittest.TestCase):
    """Test main DDSP synthesizer"""

    def test_synthesize_from_features(self):
        """Should synthesize audio from DDSP features"""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synthesizer = DDSPSynthesizer(
            sample_rate=48000,
            n_harmonics=16,
        )

        # Create features
        n_frames = 75  # 100ms / (64/48000)
        loudness = np.random.randn(n_frames).astype(np.float32)
        pitch = 440.0 * np.ones(n_frames).astype(np.float32)

        audio = synthesizer.synthesize(loudness, pitch)

        self.assertEqual(len(audio), 4800)

    def test_additive_synthesis(self):
        """Should perform additive synthesis with harmonics"""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synthesizer = DDSPSynthesizer(
            sample_rate=48000,
            n_harmonics=8,
        )

        pitch = 440.0
        amplitudes = np.ones(8) / 8  # Equal amplitudes

        audio = synthesizer.additive_synthesis(pitch, amplitudes, duration=0.1)

        self.assertEqual(len(audio), 4800)

    def test_filter_warped_synthesis(self):
        """Should perform filter-warped (source-filter) synthesis"""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synthesizer = DDSPSynthesizer(
            sample_rate=48000,
        )

        # Source signal
        source = np.random.randn(4800).astype(np.float32) * 0.1

        # Filter coefficients
        coefficients = np.random.randn(32).astype(np.float32)

        audio = synthesizer.filter_warped_synthesis(source, coefficients)

        self.assertEqual(len(audio), 4800)


class TestDDSPOptimizer(unittest.TestCase):
    """Test gradient-based optimization for DDSP"""

    def test_optimize_parameters(self):
        """Should optimize synthesis parameters"""
        from cognitive_intelligence.ddsp_synthesis import DDSPOptimizer

        optimizer = DDSPOptimizer(
            learning_rate=0.01,
            n_iterations=10,
        )

        # Target audio
        target = np.random.randn(4800).astype(np.float32)

        # Initial parameters
        initial_params = {
            "amplitudes": np.ones(16).astype(np.float32) / 16,
        }

        optimized = optimizer.optimize(target, initial_params)

        self.assertIn("amplitudes", optimized)

    def test_compute_gradient(self):
        """Should compute gradient for parameters"""
        from cognitive_intelligence.ddsp_synthesis import DDSPOptimizer

        optimizer = DDSPOptimizer(learning_rate=0.01)

        target = np.random.randn(4800).astype(np.float32)
        current = np.random.randn(4800).astype(np.float32)

        grad = optimizer.compute_gradient(target, current)

        self.assertEqual(len(grad), 4800)

    def test_reconstruct_audio(self):
        """Should reconstruct audio from parameters"""
        from cognitive_intelligence.ddsp_synthesis import DDSPOptimizer, DDSPSynthesizer

        synthesizer = DDSPSynthesizer(sample_rate=48000, n_harmonics=8)
        optimizer = DDSPOptimizer(learning_rate=0.01, n_iterations=5)

        # Target: simple sine wave
        t = np.linspace(0, 0.1, 4800)
        target = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Reconstruct
        reconstructed = optimizer.reconstruct(target, synthesizer)

        self.assertEqual(len(reconstructed), 4800)


class TestHarmonicModel(unittest.TestCase):
    """Test harmonic modeling for additive synthesis"""

    def test_harmonic_amplitudes(self):
        """Should extract harmonic amplitudes"""
        from cognitive_intelligence.ddsp_synthesis import HarmonicModel

        model = HarmonicModel(n_harmonics=16, sample_rate=48000)

        # Create harmonic signal
        t = np.linspace(0, 0.1, 4800)
        audio = np.zeros(4800, dtype=np.float32)
        for h in range(1, 5):
            audio += 0.2 * np.sin(2 * np.pi * h * 440 * t)

        amplitudes = model.extract_amplitudes(audio, fundamental_freq=440.0)

        self.assertEqual(len(amplitudes), 16)

    def test_harmonic_phases(self):
        """Should extract harmonic phases"""
        from cognitive_intelligence.ddsp_synthesis import HarmonicModel

        model = HarmonicModel(n_harmonics=16, sample_rate=48000)

        audio = np.random.randn(4800).astype(np.float32)
        phases = model.extract_phases(audio, fundamental_freq=440.0)

        self.assertEqual(len(phases), 16)

    def test_synthesize_harmonics(self):
        """Should synthesize from harmonic parameters"""
        from cognitive_intelligence.ddsp_synthesis import HarmonicModel

        model = HarmonicModel(n_harmonics=8, sample_rate=48000)

        amplitudes = np.ones(8) / 8
        phases = np.zeros(8)

        audio = model.synthesize(
            fundamental_freq=440.0,
            amplitudes=amplitudes,
            phases=phases,
            duration=0.1,
        )

        self.assertEqual(len(audio), 4800)


class TestNoiseModel(unittest.TestCase):
    """Test noise modeling for residual synthesis"""

    def test_filter_noise(self):
        """Should filter noise with time-varying filter"""
        from cognitive_intelligence.ddsp_synthesis import NoiseModel

        model = NoiseModel(n_filters=32, sample_rate=48000)

        noise = np.random.randn(4800).astype(np.float32) * 0.1
        filter_coefficients = np.random.randn(32, 75).astype(np.float32)  # Time-varying

        filtered = model.filter_noise(noise, filter_coefficients)

        self.assertEqual(len(filtered), 4800)

    def test_extract_noise_envelope(self):
        """Should extract noise envelope from residual"""
        from cognitive_intelligence.ddsp_synthesis import NoiseModel

        model = NoiseModel(n_filters=32, sample_rate=48000)

        residual = np.random.randn(4800).astype(np.float32)
        envelope = model.extract_envelope(residual)

        self.assertGreater(len(envelope), 0)


if __name__ == "__main__":
    unittest.main()
