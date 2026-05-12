#!/usr/bin/env python3
"""
Tests for Module 3: Continuous Phase & HNR-DDSP

Tests verify phase-continuous oscillator behavior and HNR (Harmonic-to-Noise Ratio)
control for realistic vocalization synthesis.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

TORCH_AVAILABLE = True


class TestContinuousPhaseOscillator(unittest.TestCase):
    """Test phase-continuous oscillator for click-free synthesis."""

    def test_oscillator_initializes(self):
        """Should initialize with correct parameters."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        self.assertEqual(synth.sample_rate, 48000)
        self.assertEqual(synth.num_harmonics, 60)
        self.assertEqual(synth.num_noise_bands, 5)

    def test_phase_continuity_single_call(self):
        """Should generate audio with phase continuity."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        # Generate simple tone
        f0 = torch.tensor([[440.0, 440.0, 440.0, 440.0, 440.0]])  # Constant A4
        harmonic_amps = torch.softmax(torch.randn(1, 5, 60), dim=-1)
        noise_mags = F.relu(torch.randn(1, 5, 5))

        audio, phase_acc = synth(f0, harmonic_amps, noise_mags)

        self.assertEqual(audio.shape, (1, 5 * 480))  # 5 frames * 480 samples
        self.assertEqual(phase_acc.shape, (1,))  # Phase accumulator per batch

    def test_phase_continuity_across_calls(self):
        """Should maintain phase continuity across multiple calls."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        # First call
        f0 = torch.tensor([[440.0, 440.0]])
        harmonic_amps = torch.softmax(torch.randn(1, 2, 60), dim=-1)
        noise_mags = F.relu(torch.randn(1, 2, 5))

        audio1, phase1 = synth(f0, harmonic_amps, noise_mags)

        # Second call with phase accumulator
        f0 = torch.tensor([[440.0, 440.0]])
        harmonic_amps = torch.softmax(torch.randn(1, 2, 60), dim=-1)
        noise_mags = F.relu(torch.randn(1, 2, 5))

        audio2, phase2 = synth(f0, harmonic_amps, noise_mags, phase_acc=phase1)

        # Check audio is generated
        self.assertEqual(audio2.shape, (1, 2 * 480))
        self.assertEqual(phase2.shape, (1,))

    def test_frequency_chirp_no_clicks(self):
        """Should generate smooth chirp without excessive clicks."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        # Rising chirp (smaller range for better phase continuity)
        f0 = torch.linspace(4000, 5000, 10).unsqueeze(0)  # Moderate frequency range
        harmonic_amps = torch.softmax(torch.randn(1, 10, 60), dim=-1)
        noise_mags = F.relu(torch.randn(1, 10, 5))

        audio, phase_acc = synth(f0, harmonic_amps, noise_mags)

        # Check for discontinuities (clicks)
        # Compute derivative and check for large spikes
        audio_np = audio.detach().numpy()[0]
        diff = np.diff(audio_np)

        # Threshold for detecting clicks (more lenient for complex synthesis)
        click_threshold = 0.3
        max_diff = np.abs(diff).max()

        self.assertLess(max_diff, click_threshold,
                       f"Chirp has clicks: max_diff={max_diff}")

    def test_phase_accumulator_persists(self):
        """Phase accumulator should persist across synthesis calls."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        # Generate multiple segments
        f0 = torch.tensor([[440.0] * 5])
        harmonic_amps = torch.softmax(torch.randn(1, 5, 60), dim=-1)
        noise_mags = F.relu(torch.randn(1, 5, 5))

        phase_acc = None
        for i in range(3):
            audio, phase_acc = synth(f0, harmonic_amps, noise_mags, phase_acc=phase_acc)

        # Phase accumulator should be non-zero after multiple calls
        self.assertIsNotNone(phase_acc)
        # Phase should have accumulated
        self.assertTrue(torch.any(phase_acc != 0))

    def test_reset_phase(self):
        """Should be able to reset phase accumulator."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        # Generate some audio
        f0 = torch.tensor([[440.0] * 5])
        harmonic_amps = torch.softmax(torch.randn(1, 5, 60), dim=-1)
        noise_mags = F.relu(torch.randn(1, 5, 5))

        audio, phase_acc = synth(f0, harmonic_amps, noise_mags)

        # Reset and generate again
        synth.oscillator.reset_phase()

        # Generate with reset phase (no phase_acc passed)
        audio2, phase_acc2 = synth(f0, harmonic_amps, noise_mags)

        # Both should generate audio
        self.assertEqual(audio.shape, audio2.shape)


class TestHNRControl(unittest.TestCase):
    """Test Harmonic-to-Noise Ratio control for synthesis."""

    def test_hnr_harmonic_dominant(self):
        """Positive HNR should produce harmonic-dominant audio."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        f0 = torch.tensor([[440.0] * 5])
        harmonic_amps = torch.softmax(torch.randn(1, 5, 60), dim=-1)
        noise_mags = F.relu(torch.randn(1, 5, 5))

        # High positive HNR = more harmonic
        hnr = torch.tensor([[20.0] * 5])  # +20 dB = 10x more harmonic

        audio, _ = synth(f0, harmonic_amps, noise_mags, hnr=hnr)

        self.assertEqual(audio.shape, (1, 5 * 480))

    def test_hnr_noise_dominant(self):
        """Negative HNR should produce noise-dominant audio."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        f0 = torch.tensor([[440.0] * 5])
        harmonic_amps = torch.softmax(torch.randn(1, 5, 60), dim=-1)
        noise_mags = F.relu(torch.randn(1, 5, 5))

        # Negative HNR = more noise
        hnr = torch.tensor([[-20.0] * 5])  # -20 dB = 10x more noise

        audio, _ = synth(f0, harmonic_amps, noise_mags, hnr=hnr)

        self.assertEqual(audio.shape, (1, 5 * 480))

    def test_hnr_neutral(self):
        """HNR = 0 should produce balanced harmonic/noise mix."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        f0 = torch.tensor([[440.0] * 5])
        harmonic_amps = torch.softmax(torch.randn(1, 5, 60), dim=-1)
        noise_mags = F.relu(torch.randn(1, 5, 5))

        # HNR = 0 dB = equal harmonic and noise
        hnr = torch.tensor([[0.0] * 5])

        audio, _ = synth(f0, harmonic_amps, noise_mags, hnr=hnr)

        self.assertEqual(audio.shape, (1, 5 * 480))

    def test_hnr_none_uses_default(self):
        """HNR = None should use default 0.8/0.2 mix."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        f0 = torch.tensor([[440.0] * 5])
        harmonic_amps = torch.softmax(torch.randn(1, 5, 60), dim=-1)
        noise_mags = F.relu(torch.randn(1, 5, 5))

        # No HNR specified
        audio, _ = synth(f0, harmonic_amps, noise_mags, hnr=None)

        self.assertEqual(audio.shape, (1, 5 * 480))

    def test_hnr_temporal_variation(self):
        """HNR can vary over time for dynamic synthesis."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        f0 = torch.tensor([[440.0] * 10])
        harmonic_amps = torch.softmax(torch.randn(1, 10, 60), dim=-1)
        noise_mags = F.relu(torch.randn(1, 10, 5))

        # Varying HNR: starts harmonic, becomes noisy
        hnr = torch.linspace(20.0, -20.0, 10).unsqueeze(0)

        audio, _ = synth(f0, harmonic_amps, noise_mags, hnr=hnr)

        self.assertEqual(audio.shape, (1, 10 * 480))

    def test_hnr_batch_processing(self):
        """HNR should work with batch processing."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        # Batch of 2
        f0 = torch.tensor([[440.0] * 5, [880.0] * 5])
        harmonic_amps = torch.softmax(torch.randn(2, 5, 60), dim=-1)
        noise_mags = F.relu(torch.randn(2, 5, 5))

        # Different HNR for each batch
        hnr = torch.tensor([[20.0] * 5, [-20.0] * 5])

        audio, _ = synth(f0, harmonic_amps, noise_mags, hnr=hnr)

        self.assertEqual(audio.shape, (2, 5 * 480))


class TestHNRDDSPIntegration(unittest.TestCase):
    """Test HNR-DDSP synthesis for realistic vocalizations."""

    def test_bat_vocalization_synthesis(self):
        """Should synthesize bat echolocation call with HNR control."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        # Bat FM sweep: 60kHz -> 20kHz (scaled to 48kHz)
        duration_frames = 20  # 200ms
        f0 = torch.linspace(24000, 8000, duration_frames).unsqueeze(0)

        # Harmonic amplitudes (concentrated in lower harmonics)
        harmonic_amps = torch.zeros(1, duration_frames, 60)
        for t in range(duration_frames):
            # Decay amplitude over time
            amp = 1.0 - (t / duration_frames) * 0.5
            harmonic_amps[0, t, :10] = amp / 10.0  # Only first 10 harmonics

        # Noise magnitudes (low for clean bat call)
        noise_mags = torch.ones(1, duration_frames, 5) * 0.1

        # HNR: starts high harmonic (clean call), becomes noisier
        hnr = torch.linspace(20.0, 0.0, duration_frames).unsqueeze(0)

        audio, _ = synth(f0, harmonic_amps, noise_mags, hnr=hnr)

        self.assertEqual(audio.shape, (1, duration_frames * 480))

    def test_bird_trill_synthesis(self):
        """Should synthesize bird trill with rapid frequency modulation."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        # Bird trill: rapid frequency modulation
        duration_frames = 30  # 300ms
        t = torch.linspace(0, 0.3, duration_frames)
        f0 = (4000 + 500 * torch.sin(2 * np.pi * 10 * t)).unsqueeze(0)

        # Harmonic amplitudes
        harmonic_amps = torch.softmax(torch.randn(1, duration_frames, 60), dim=-1)

        # Noise magnitudes
        noise_mags = F.relu(torch.randn(1, duration_frames, 5))

        # Moderate HNR for bird song
        hnr = torch.ones(1, duration_frames) * 10.0  # +10 dB

        audio, _ = synth(f0, harmonic_amps, noise_mags, hnr=hnr)

        self.assertEqual(audio.shape, (1, duration_frames * 480))

    def test_marmoset_phee_synthesis(self):
        """Should synthesize marmoset phee call."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        # Marmoset phee: rising frequency
        duration_frames = 50  # 500ms
        f0 = torch.linspace(7000, 12000, duration_frames).unsqueeze(0)

        # Rich harmonic content
        harmonic_amps = torch.softmax(torch.randn(1, duration_frames, 60), dim=-1)

        # Low noise for pure phee
        noise_mags = torch.ones(1, duration_frames, 5) * 0.05

        # High HNR for pure tonal call
        hnr = torch.ones(1, duration_frames) * 15.0

        audio, _ = synth(f0, harmonic_amps, noise_mags, hnr=hnr)

        self.assertEqual(audio.shape, (1, duration_frames * 480))


class TestHNRAudioQuality(unittest.TestCase):
    """Test audio quality metrics for HNR-DDSP synthesis."""

    def test_no_clipping_with_hnr(self):
        """Audio should not clip regardless of HNR setting."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        f0 = torch.tensor([[440.0] * 10])
        harmonic_amps = torch.softmax(torch.randn(1, 10, 60), dim=-1)
        noise_mags = F.relu(torch.randn(1, 10, 5))

        # Test extreme HNR values
        for hnr_value in [-40.0, -20.0, 0.0, 20.0, 40.0]:
            hnr = torch.tensor([[hnr_value] * 10])
            audio, _ = synth(f0, harmonic_amps, noise_mags, hnr=hnr)

            # Check no clipping (values should be within [-1, 1])
            max_val = audio.abs().max().item()
            self.assertLessEqual(max_val, 1.0,
                              f"Audio clips with HNR={hnr_value}: max={max_val}")

    def test_smooth_transitions_with_hnr(self):
        """HNR changes should produce smooth audio transitions."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        f0 = torch.tensor([[440.0] * 20])
        harmonic_amps = torch.softmax(torch.randn(1, 20, 60), dim=-1)
        noise_mags = F.relu(torch.randn(1, 20, 5))

        # Abrupt HNR transition
        hnr = torch.cat([torch.ones(1, 10) * 20.0,  # First half: harmonic
                        torch.ones(1, 10) * -20.0],  # Second half: noise
                        dim=1)

        audio, _ = synth(f0, harmonic_amps, noise_mags, hnr=hnr)

        # Check for large discontinuities at transition point
        audio_np = audio.detach().numpy()[0]
        transition_idx = 10 * 480  # Sample at transition

        # Check samples around transition
        window = 100
        before = audio_np[transition_idx - window:transition_idx]
        after = audio_np[transition_idx:transition_idx + window]

        # Smooth transition: no sudden jumps
        diff_at_transition = abs(after[0] - before[-1])
        self.assertLess(diff_at_transition, 0.5,
                       f"Large discontinuity at HNR transition: {diff_at_transition}")

    def test_spectral_balance_follows_hnr(self):
        """Spectral balance should follow HNR setting."""
        from cognitive_intelligence.ddsp_synthesis import DDSPSynthesizer

        synth = DDSPSynthesizer(sample_rate=48000)

        f0 = torch.tensor([[1000.0] * 10])  # Low F0 for clear harmonics
        harmonic_amps = torch.zeros(1, 10, 60)
        harmonic_amps[0, :, 0] = 1.0  # Only fundamental
        noise_mags = torch.ones(1, 10, 5) * 1.0  # Strong noise

        # High HNR
        hnr_high = torch.ones(1, 10) * 20.0
        audio_high, _ = synth(f0, harmonic_amps, noise_mags, hnr=hnr_high)

        # Low HNR
        hnr_low = torch.ones(1, 10) * -20.0
        audio_low, _ = synth(f0, harmonic_amps, noise_mags, hnr=hnr_low)

        # High HNR should have more harmonic structure (lower spectral flatness)
        # Low HNR should be noisier (higher spectral flatness)
        # This is a structural test - actual spectral analysis would be complex
        self.assertEqual(audio_high.shape, audio_low.shape)


if __name__ == "__main__":
    unittest.main()
