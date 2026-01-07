"""
TDD Tests for Micro-Dynamics Features (Shimmer, Spectral Flux, Harmonicity)

This test suite validates advanced acoustic features that complete the
"Acoustic Algebra" vector space:

1. **Shimmer** - Amplitude instability (companion to Jitter)
2. **Spectral Flux** - Rate of spectral change over time
3. **Harmonicity** - Degree of periodicity vs noise

These features enable:
- "Nervousness" as a 2D vector: [Jitter, Shimmer]
- "Texture" discrimination (Trills vs Flat tones)
- "Modality" slider (Tonal vs Noisy)

Architecture: Python Feature Extraction → 20D Vector Space

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest

import numpy as np

# =============================================================================
# Test 1: Shimmer (Amplitude Instability)
# =============================================================================


class TestShimmerExtraction(unittest.TestCase):
    """Test 1: Shimmer measures amplitude wobble"""

    def test_shimmer_calculates_amplitude_variation(self):
        """
        RED TEST: Shimmer should calculate peak-to-peak amplitude variation

        Scenario:
        - Steady tone: Low shimmer (<0.01)
        - Wobbly tone: High shimmer (>0.05)
        Expected: Shimmer returns coefficient of variation of peak amplitudes
        """
        # Arrange - Create test audio
        sr = 48000

        # Steady tone (low shimmer)
        steady_tone = self._create_steady_tone(sr, duration=0.1, freq=1000, amp_variation=0.005)

        # Wobbly tone (high shimmer) - amplitude LFO
        wobbly_tone = self._create_wobbly_tone(
            sr, duration=0.1, freq=1000, lfo_freq=15, lfo_depth=0.3
        )

        # Act
        from realtime.micro_dynamics_features import calculate_shimmer

        steady_shimmer = calculate_shimmer(steady_tone, sr)
        wobbly_shimmer = calculate_shimmer(wobbly_tone, sr)

        # Assert
        # Steady tone should have low shimmer
        self.assertLess(
            steady_shimmer, 0.02, f"Steady tone shimmer {steady_shimmer:.4f} should be <0.02"
        )

        # Wobbly tone should have high shimmer
        self.assertGreater(
            wobbly_shimmer, 0.10, f"Wobbly tone shimmer {wobbly_shimmer:.4f} should be >0.10"
        )

        # Wobbly should be significantly higher than steady
        self.assertGreater(
            wobbly_shimmer / steady_shimmer, 5.0, "Wobbly shimmer should be >5x steady shimmer"
        )

        print("✓ Shimmer test passed")
        print(f"  Steady tone shimmer: {steady_shimmer:.4f}")
        print(f"  Wobbly tone shimmer: {wobbly_shimmer:.4f}")
        print(f"  Ratio: {wobbly_shimmer / steady_shimmer:.1f}x")

    def test_shimmer_with_real_phee(self):
        """
        RED TEST: Shimmer should work with real marmoset phee structure

        Scenario: Real phee has natural vibrato (pitch + amplitude wobble)
        Expected: Shimmer captures the amplitude modulation component
        """
        # Arrange - Create realistic phee with vibrato
        sr = 48000
        phee = self._create_phee_with_vibrato(
            sr, duration=0.2, base_freq=7000, vibrato_rate=8, vibrato_depth=0.05
        )

        # Act
        from realtime.micro_dynamics_features import calculate_shimmer

        shimmer = calculate_shimmer(phee, sr)

        # Assert
        # Phee with vibrato should have moderate shimmer (0.02 - 0.10)
        self.assertGreater(shimmer, 0.01, "Phee shimmer should be >0.01")
        self.assertLess(shimmer, 0.20, "Phee shimmer should be <0.20")

        print(f"✓ Phee shimmer test passed: {shimmer:.4f}")

    def _create_steady_tone(self, sr, duration, freq, amp_variation):
        """Create a steady tone with minimal amplitude variation"""
        t = np.linspace(0, duration, int(sr * duration))
        # Add tiny random amplitude variation
        noise = np.random.normal(0, amp_variation, len(t))
        return 0.5 * np.sin(2 * np.pi * freq * t) + noise

    def _create_wobbly_tone(self, sr, duration, freq, lfo_freq, lfo_depth):
        """Create a tone with amplitude LFO (high shimmer)"""
        t = np.linspace(0, duration, int(sr * duration))
        # Amplitude modulation
        lfo = 1.0 + lfo_depth * np.sin(2 * np.pi * lfo_freq * t)
        return 0.5 * lfo * np.sin(2 * np.pi * freq * t)

    def _create_phee_with_vibrato(self, sr, duration, base_freq, vibrato_rate, vibrato_depth):
        """Create realistic phee with frequency AND amplitude vibrato"""
        t = np.linspace(0, duration, int(sr * duration))
        # FM vibrato
        freq_mod = base_freq * (1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t))
        # AM vibrato (shimmer) - typically 50% of FM depth
        amp_mod = 1.0 + (vibrato_depth * 0.5) * np.sin(2 * np.pi * vibrato_rate * t)

        # Generate with instantaneous frequency
        phase = 2 * np.pi * np.cumsum(freq_mod) / sr
        return 0.5 * amp_mod * np.sin(phase)


# =============================================================================
# Test 2: Spectral Flux (Texture Change)
# =============================================================================


class TestSpectralFluxExtraction(unittest.TestCase):
    """Test 2: Spectral Flux measures rate of spectral change"""

    def test_spectral_flux_discriminates_trill_vs_steady(self):
        """
        RED TEST: Spectral flux should distinguish trills from steady tones

        Scenario:
        - Steady tone: Low flux (static spectrum)
        - Trill: High flux (rapidly changing spectrum)
        Expected: Flux calculates L2-norm of spectral frame differences
        """
        # Arrange
        sr = 48000

        # Steady tone (low flux)
        steady = self._create_steady_tone(sr, duration=0.2, freq=5000)

        # Trill (high flux) - rapid frequency modulation
        trill = self._create_trill(sr, duration=0.2, freq1=4000, freq2=6000, rate=20)

        # Act
        from realtime.micro_dynamics_features import calculate_spectral_flux

        steady_flux = calculate_spectral_flux(steady, sr)
        trill_flux = calculate_spectral_flux(trill, sr)

        # Assert
        # Steady should have low flux
        self.assertLess(steady_flux, 5.0, f"Steady flux {steady_flux:.4f} should be <5.0")

        # Trill should have higher flux
        self.assertGreater(trill_flux, 10.0, f"Trill flux {trill_flux:.4f} should be >10.0")

        # Trill should be significantly higher
        self.assertGreater(trill_flux / steady_flux, 2.0, "Trill flux should be >2x steady flux")

        print("✓ Spectral flux test passed")
        print(f"  Steady tone flux: {steady_flux:.4f}")
        print(f"  Trill flux: {trill_flux:.4f}")
        print(f"  Ratio: {trill_flux / steady_flux:.1f}x")

    def test_spectral_flux_with_marmoset_twitter(self):
        """
        RED TEST: Spectral flux should capture marmoset "twitter" texture

        Scenario: Twitter has rapid frequency modulation (high texture)
        Expected: Flux returns high value indicating granular texture
        """
        # Arrange
        sr = 48000
        # Marmoset twitter: rapid, short notes
        twitter = self._create_marmoset_twitter(sr, duration=0.3)

        # Act
        from realtime.micro_dynamics_features import calculate_spectral_flux

        flux = calculate_spectral_flux(twitter, sr)

        # Assert
        # Twitter should have moderate-high flux
        self.assertGreater(flux, 5.0, "Twitter flux should be >5.0")
        self.assertLess(flux, 50.0, "Twitter flux should be <50.0")

        print(f"✓ Twitter flux test passed: {flux:.4f}")

    def _create_steady_tone(self, sr, duration, freq):
        """Create a steady pure tone"""
        t = np.linspace(0, duration, int(sr * duration))
        return 0.5 * np.sin(2 * np.pi * freq * t)

    def _create_trill(self, sr, duration, freq1, freq2, rate):
        """Create a trill (rapid alternation between two frequencies)"""
        t = np.linspace(0, duration, int(sr * duration))
        # Square wave modulation between frequencies
        mod = np.sign(np.sin(2 * np.pi * rate * t))
        freq = freq1 + (freq2 - freq1) * (mod + 1) / 2
        phase = 2 * np.pi * np.cumsum(freq) / sr
        return 0.5 * np.sin(phase)

    def _create_marmoset_twitter(self, sr, duration):
        """Create marmoset-like twitter (rapid short notes)"""
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.zeros_like(t)

        # Add 10 short notes with varying frequencies
        note_duration = duration / 15
        for i in range(10):
            start = int(i * note_duration * sr * 1.5)
            end = int((i * note_duration + note_duration * 0.7) * sr)
            if end < len(t):
                freq = 7000 + np.random.randint(-500, 500)
                note_t = np.arange(end - start) / sr
                audio[start:end] = 0.3 * np.sin(2 * np.pi * freq * note_t)

        return audio


# =============================================================================
# Test 3: Harmonicity (Tonal vs Noisy)
# =============================================================================


class TestHarmonicityExtraction(unittest.TestCase):
    """Test 3: Harmonicity measures degree of periodicity"""

    def test_harmonicity_discriminates_tone_vs_noise(self):
        """
        RED TEST: Harmonicity should distinguish pure tones from noise

        Scenario:
        - Pure tone: High harmonicity (>0.9)
        - White noise: Low harmonicity (<0.3)
        Expected: Harmonicity returns auto-correlation coefficient
        """
        # Arrange
        sr = 48000

        # Pure tone (high harmonicity)
        pure_tone = self._create_pure_tone(sr, duration=0.1, freq=5000)

        # White noise (low harmonicity)
        white_noise = np.random.normal(0, 0.5, int(sr * 0.1))

        # Mixed signal (medium harmonicity)
        mixed = 0.7 * pure_tone + 0.3 * white_noise

        # Act
        from realtime.micro_dynamics_features import calculate_harmonicity

        pure_harmonic = calculate_harmonicity(pure_tone, sr)
        noise_harmonic = calculate_harmonicity(white_noise, sr)
        mixed_harmonic = calculate_harmonicity(mixed, sr)

        # Assert
        # Pure tone should have high harmonicity
        self.assertGreater(
            pure_harmonic, 0.85, f"Pure tone harmonicity {pure_harmonic:.4f} should be >0.85"
        )

        # Noise should have low harmonicity
        self.assertLess(
            noise_harmonic, 0.3, f"Noise harmonicity {noise_harmonic:.4f} should be <0.3"
        )

        # Mixed should be in between
        self.assertGreater(mixed_harmonic, noise_harmonic, "Mixed harmonicity should be > noise")
        self.assertLess(mixed_harmonic, pure_harmonic, "Mixed harmonicity should be < pure")

        print("✓ Harmonicity test passed")
        print(f"  Pure tone: {pure_harmonic:.4f}")
        print(f"  Mixed: {mixed_harmonic:.4f}")
        print(f"  Noise: {noise_harmonic:.4f}")

    def test_harmonicity_with_fm_sweep(self):
        """
        RED TEST: Harmonicity should handle FM sweeps (bats)

        Scenario: FM sweeps are tonal but changing frequency
        Expected: Medium-high harmonicity (tonal but not pure)
        """
        # Arrange
        sr = 48000
        # Bat-like FM sweep
        fm_sweep = self._create_fm_sweep(sr, duration=0.05, start_freq=60000, end_freq=40000)

        # Act
        from realtime.micro_dynamics_features import calculate_harmonicity

        harmonic = calculate_harmonicity(fm_sweep, sr)

        # Assert
        # FM sweep should have low-medium harmonicity (tonal but rapidly changing)
        self.assertGreater(harmonic, 0.05, "FM sweep harmonicity should be >0.05")
        self.assertLess(harmonic, 0.5, "FM sweep harmonicity should be <0.5")

        print(f"✓ FM sweep harmonicity test passed: {harmonic:.4f}")

    def _create_pure_tone(self, sr, duration, freq):
        """Create a pure sine wave"""
        t = np.linspace(0, duration, int(sr * duration))
        return 0.5 * np.sin(2 * np.pi * freq * t)

    def _create_fm_sweep(self, sr, duration, start_freq, end_freq):
        """Create FM sweep (like bat echolocation)"""
        t = np.linspace(0, duration, int(sr * duration))
        # Linear frequency sweep
        freq = start_freq + (end_freq - start_freq) * t / duration
        phase = 2 * np.pi * np.cumsum(freq) / sr
        return 0.5 * np.sin(phase)


# =============================================================================
# Test 4: 20D Vector Integration
# =============================================================================


class Test20DVectorIntegration(unittest.TestCase):
    """Test 4: New features integrate into 20D vector"""

    def test_20d_vector_contains_all_features(self):
        """
        RED TEST: 20D vector should contain all features including new ones

        Expected:
        - shimmer (Motion Factor)
        - spectral_flux (Spectral Dynamics)
        - harmonicity (Grit Factor)
        """
        # Arrange
        from realtime.micro_dynamics_features import Vector20D

        # Create a 20D vector with all features
        vector = Vector20D(
            # Fundamental (3)
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            # Grit Factors (3) - Added harmonicity
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.3,
            harmonicity=0.85,
            # Motion Factors (7) - Added shimmer
            attack_time_ms=5.0,
            decay_time_ms=20.0,
            sustain_level=0.7,
            vibrato_rate_hz=7.0,
            vibrato_depth=0.02,
            jitter=0.01,
            shimmer=0.015,
            # Fingerprint Factors (5)
            mfcc_1=-10.0,
            mfcc_2=-5.0,
            mfcc_3=-2.0,
            mfcc_4=-1.0,
            spectral_contrast=20.0,
            # Spectral Dynamics (1) - New
            spectral_flux=1.5,
            # Rhythm Factors (3)
            median_ici_ms=15.0,
            onset_rate_hz=50.0,
            ici_coefficient_of_variation=0.3,
        )

        # Assert
        # Verify all 20 dimensions are present
        self.assertEqual(vector.mean_f0_hz, 7000.0)
        self.assertEqual(vector.shimmer, 0.015)
        self.assertEqual(vector.spectral_flux, 1.5)
        self.assertEqual(vector.harmonicity, 0.85)

        # Count fields (should be 22)
        import dataclasses

        field_count = len(dataclasses.fields(vector))
        self.assertEqual(field_count, 22, f"Vector should have 22 fields, got {field_count}")

        print(f"✓ 22D vector integration test passed ({field_count} dimensions)")

    def test_nervousness_2d_vector(self):
        """
        RED TEST: [Jitter, Shimmer] defines "Nervousness" state

        Expected:
        - Steady: Low Jitter, Low Shimmer
        - Breathy: Low Jitter, High Shimmer
        - Tight: High Jitter, Low Shimmer
        - Tremulous: High Jitter, High Shimmer
        """
        from realtime.micro_dynamics_features import Vector20D

        # Steady state
        steady = Vector20D(
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=25.0,
            spectral_flatness=0.2,
            harmonicity=0.95,
            attack_time_ms=5.0,
            decay_time_ms=20.0,
            sustain_level=0.7,
            vibrato_rate_hz=7.0,
            vibrato_depth=0.01,
            jitter=0.005,
            shimmer=0.005,
            mfcc_1=-10.0,
            mfcc_2=-5.0,
            mfcc_3=-2.0,
            mfcc_4=-1.0,
            spectral_contrast=20.0,
            spectral_flux=0.5,
            median_ici_ms=15.0,
            onset_rate_hz=50.0,
            ici_coefficient_of_variation=0.2,
        )

        # Tremulous state (high fear)
        tremulous = Vector20D(
            mean_f0_hz=7500.0,
            duration_ms=45.0,
            f0_range_hz=600.0,
            harmonic_to_noise_ratio=15.0,
            spectral_flatness=0.5,
            harmonicity=0.70,
            attack_time_ms=3.0,
            decay_time_ms=15.0,
            sustain_level=0.8,
            vibrato_rate_hz=12.0,
            vibrato_depth=0.08,
            jitter=0.05,
            shimmer=0.08,
            mfcc_1=-8.0,
            mfcc_2=-3.0,
            mfcc_3=-1.0,
            mfcc_4=-0.5,
            spectral_contrast=30.0,
            spectral_flux=3.0,
            median_ici_ms=12.0,
            onset_rate_hz=70.0,
            ici_coefficient_of_variation=0.5,
        )

        # Assert - Steady should be low jitter/shimmer
        self.assertLess(steady.jitter, 0.01, "Steady jitter should be low")
        self.assertLess(steady.shimmer, 0.01, "Steady shimmer should be low")

        # Assert - Tremulous should be high jitter/shimmer
        self.assertGreater(tremulous.jitter, 0.03, "Tremulous jitter should be high")
        self.assertGreater(tremulous.shimmer, 0.05, "Tremulous shimmer should be high")

        print("✓ Nervousness 2D vector test passed")
        print(f"  Steady: Jitter={steady.jitter:.3f}, Shimmer={steady.shimmer:.3f}")
        print(f"  Tremulous: Jitter={tremulous.jitter:.3f}, Shimmer={tremulous.shimmer:.3f}")


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
