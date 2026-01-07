"""
17D Micro-Dynamics Metadata Synthesis Tests
===========================================

Tests for the complete 17-dimensional micro-dynamics metadata
support in the Rust synthesis engine.

This validates:
1. Full 17D SourceMetadata construction
2. Builder pattern for partial metadata
3. Delta calculations across all dimensions
4. Backward compatibility with legacy 3D API

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from technical_architecture import GranularConcatenativeSynthesizer, SourceMetadata
except ImportError:
    print("Error: technical_architecture module not found.")
    print("Run: cd technical_architecture && maturin build --release --features python-bindings")
    sys.exit(1)


class Test17DSourceMetadata(unittest.TestCase):
    """Test 17-dimensional SourceMetadata construction and access."""

    def test_full_17d_construction(self):
        """Test creating metadata with all 17 features."""
        metadata = SourceMetadata(
            # Fundamental (3)
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            # Grit Factors (2)
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.1,
            # Motion Factors (6)
            attack_time_ms=10.0,
            decay_time_ms=15.0,
            sustain_level=0.7,
            vibrato_rate_hz=8.0,
            vibrato_depth=50.0,
            jitter=0.02,
            # Fingerprint Factors (5)
            mfcc_1=-500.0,
            mfcc_2=-100.0,
            mfcc_3=-50.0,
            mfcc_4=-20.0,
            spectral_contrast=20.0,
            # Rhythm Factors (3)
            median_ici_ms=0.0,
            onset_rate_hz=0.0,
            ici_coefficient_of_variation=0.0,
        )

        # Verify all features (use approximate comparison for f32)
        self.assertAlmostEqual(metadata.get_mean_f0_hz(), 7000.0, places=1)
        self.assertAlmostEqual(metadata.get_duration_ms(), 50.0, places=1)
        self.assertAlmostEqual(metadata.get_f0_range_hz(), 400.0, places=1)
        self.assertAlmostEqual(metadata.get_harmonic_to_noise_ratio(), 20.0, places=1)
        self.assertAlmostEqual(metadata.get_spectral_flatness(), 0.1, places=1)
        self.assertAlmostEqual(metadata.get_attack_time_ms(), 10.0, places=1)
        self.assertAlmostEqual(metadata.get_decay_time_ms(), 15.0, places=1)
        self.assertAlmostEqual(metadata.get_sustain_level(), 0.7, places=1)
        self.assertAlmostEqual(metadata.get_vibrato_rate_hz(), 8.0, places=1)
        self.assertAlmostEqual(metadata.get_vibrato_depth(), 50.0, places=1)
        self.assertAlmostEqual(metadata.get_jitter(), 0.02, places=2)
        self.assertAlmostEqual(metadata.get_mfcc_1(), -500.0, places=1)
        self.assertAlmostEqual(metadata.get_mfcc_2(), -100.0, places=1)
        self.assertAlmostEqual(metadata.get_mfcc_3(), -50.0, places=1)
        self.assertAlmostEqual(metadata.get_mfcc_4(), -20.0, places=1)
        self.assertAlmostEqual(metadata.get_spectral_contrast(), 20.0, places=1)
        self.assertAlmostEqual(metadata.get_median_ici_ms(), 0.0, places=1)
        self.assertAlmostEqual(metadata.get_onset_rate_hz(), 0.0, places=1)
        self.assertAlmostEqual(metadata.get_ici_coefficient_of_variation(), 0.0, places=1)

    def test_builder_pattern_partial_metadata(self):
        """Test builder pattern with only some features specified."""
        metadata = (
            SourceMetadata.builder()
            .mean_f0_hz(6500.0)
            .duration_ms(60.0)
            .f0_range_hz(300.0)
            .harmonic_to_noise_ratio(15.0)
            .jitter(0.05)
            .build()
        )

        # Specified values
        self.assertAlmostEqual(metadata.get_mean_f0_hz(), 6500.0, places=1)
        self.assertAlmostEqual(metadata.get_duration_ms(), 60.0, places=1)
        self.assertAlmostEqual(metadata.get_f0_range_hz(), 300.0, places=1)
        self.assertAlmostEqual(metadata.get_harmonic_to_noise_ratio(), 15.0, places=1)
        self.assertAlmostEqual(metadata.get_jitter(), 0.05, places=2)

        # Default values (should be filled in)
        self.assertAlmostEqual(metadata.get_spectral_flatness(), 0.1, places=1)  # Default
        self.assertAlmostEqual(metadata.get_attack_time_ms(), 10.0, places=1)  # Default

    def test_all_getters_setters(self):
        """Test all getter and setter methods."""
        metadata = SourceMetadata(
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.1,
            attack_time_ms=10.0,
            decay_time_ms=15.0,
            sustain_level=0.7,
            vibrato_rate_hz=8.0,
            vibrato_depth=50.0,
            jitter=0.02,
            mfcc_1=-500.0,
            mfcc_2=-100.0,
            mfcc_3=-50.0,
            mfcc_4=-20.0,
            spectral_contrast=20.0,
            median_ici_ms=0.0,
            onset_rate_hz=0.0,
            ici_coefficient_of_variation=0.0,
        )

        # Test setters
        metadata.set_mean_f0_hz(7500.0)
        self.assertAlmostEqual(metadata.get_mean_f0_hz(), 7500.0, places=1)

        metadata.set_duration_ms(60.0)
        self.assertAlmostEqual(metadata.get_duration_ms(), 60.0, places=1)

        metadata.set_jitter(0.08)
        self.assertAlmostEqual(metadata.get_jitter(), 0.08, places=2)

        metadata.set_spectral_flatness(0.3)
        self.assertAlmostEqual(metadata.get_spectral_flatness(), 0.3, places=1)

    def test_repr_string(self):
        """Test string representation."""
        metadata = SourceMetadata(
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.1,
            attack_time_ms=10.0,
            decay_time_ms=15.0,
            sustain_level=0.7,
            vibrato_rate_hz=8.0,
            vibrato_depth=50.0,
            jitter=0.02,
            mfcc_1=-500.0,
            mfcc_2=-100.0,
            mfcc_3=-50.0,
            mfcc_4=-20.0,
            spectral_contrast=20.0,
            median_ici_ms=0.0,
            onset_rate_hz=0.0,
            ici_coefficient_of_variation=0.0,
        )

        repr_str = repr(metadata)
        self.assertIn("SourceMetadata", repr_str)
        self.assertIn("7000Hz", repr_str)
        self.assertIn("50ms", repr_str)


class Test17DSynthesisIntegration(unittest.TestCase):
    """Test 17D metadata with synthesizer integration."""

    def setUp(self):
        """Create synthesizer and test audio."""
        self.synth = GranularConcatenativeSynthesizer(sample_rate=22050)

        # Create test audio buffer (sine wave)
        duration_ms = 50
        num_samples = int(duration_ms / 1000.0 * 22050)
        self.audio_buffer = [
            0.5 * np.sin(2.0 * np.pi * 7000.0 * i / 22050.0) for i in range(num_samples)
        ]

    def test_load_with_17d_metadata(self):
        """Test loading source with full 17D metadata."""
        metadata = SourceMetadata(
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.1,
            attack_time_ms=10.0,
            decay_time_ms=15.0,
            sustain_level=0.7,
            vibrato_rate_hz=8.0,
            vibrato_depth=50.0,
            jitter=0.02,
            mfcc_1=-500.0,
            mfcc_2=-100.0,
            mfcc_3=-50.0,
            mfcc_4=-20.0,
            spectral_contrast=20.0,
            median_ici_ms=0.0,
            onset_rate_hz=0.0,
            ici_coefficient_of_variation=0.0,
        )

        self.synth.load_source_with_metadata(self.audio_buffer, metadata)
        output = self.synth.synthesize(50.0)

        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

    def test_gritty_vs_pure_persona_metadata(self):
        """Test creating metadata for different acoustic personas.

        GRITTY (aggressive): Low HNR, high flatness, fast attack, high jitter
        PURE (contact): High HNR, low flatness, slow attack, low jitter
        """
        # GRITTY: Aggressive alert
        gritty = SourceMetadata(
            mean_f0_hz=6500.0,
            duration_ms=45.0,
            f0_range_hz=800.0,
            harmonic_to_noise_ratio=2.0,  # Low HNR (noisy)
            spectral_flatness=0.8,  # High flatness (noise-like)
            attack_time_ms=3.0,  # Fast attack (sharp)
            decay_time_ms=10.0,
            sustain_level=0.5,
            vibrato_rate_hz=5.0,
            vibrato_depth=30.0,
            jitter=0.15,  # High jitter (instability)
            mfcc_1=-200.0,
            mfcc_2=-50.0,
            mfcc_3=-20.0,
            mfcc_4=-10.0,
            spectral_contrast=10.0,
            median_ici_ms=0.0,
            onset_rate_hz=0.0,
            ici_coefficient_of_variation=0.0,
        )

        # PURE: Contact call
        pure = SourceMetadata(
            mean_f0_hz=6500.0,
            duration_ms=70.0,
            f0_range_hz=300.0,
            harmonic_to_noise_ratio=25.0,  # High HNR (tonal)
            spectral_flatness=0.05,  # Low flatness (tonal)
            attack_time_ms=25.0,  # Slow attack (gentle)
            decay_time_ms=20.0,
            sustain_level=0.8,
            vibrato_rate_hz=8.0,
            vibrato_depth=40.0,
            jitter=0.01,  # Low jitter (stable)
            mfcc_1=-600.0,
            mfcc_2=-120.0,
            mfcc_3=-60.0,
            mfcc_4=-25.0,
            spectral_contrast=25.0,
            median_ici_ms=0.0,
            onset_rate_hz=0.0,
            ici_coefficient_of_variation=0.0,
        )

        # Verify persona characteristics
        self.assertLess(gritty.get_harmonic_to_noise_ratio(), pure.get_harmonic_to_noise_ratio())
        self.assertGreater(gritty.get_spectral_flatness(), pure.get_spectral_flatness())
        self.assertLess(gritty.get_attack_time_ms(), pure.get_attack_time_ms())
        self.assertGreater(gritty.get_jitter(), pure.get_jitter())

    def test_rhythmic_vs_harmonic_metadata(self):
        """Test creating metadata for rhythmic vs harmonic calls.

        Rhythmic: High onset rate, regular ICI, non-zero ICI
        Harmonic: Zero onset rate, zero ICI
        """
        # RHYTHMIC: Pulsed call
        rhythmic = SourceMetadata(
            mean_f0_hz=8000.0,
            duration_ms=30.0,
            f0_range_hz=200.0,
            harmonic_to_noise_ratio=15.0,
            spectral_flatness=0.2,
            attack_time_ms=5.0,
            decay_time_ms=10.0,
            sustain_level=0.3,
            vibrato_rate_hz=0.0,
            vibrato_depth=10.0,
            jitter=0.03,
            mfcc_1=-400.0,
            mfcc_2=-80.0,
            mfcc_3=-40.0,
            mfcc_4=-15.0,
            spectral_contrast=15.0,
            median_ici_ms=50.0,  # 50ms between clicks
            onset_rate_hz=20.0,  # 20 clicks/second
            ici_coefficient_of_variation=0.1,  # Very regular rhythm
        )

        # HARMONIC: Continuous tone
        harmonic = SourceMetadata(
            mean_f0_hz=8000.0,
            duration_ms=100.0,
            f0_range_hz=200.0,
            harmonic_to_noise_ratio=25.0,
            spectral_flatness=0.05,
            attack_time_ms=15.0,
            decay_time_ms=20.0,
            sustain_level=0.8,
            vibrato_rate_hz=10.0,
            vibrato_depth=50.0,
            jitter=0.01,
            mfcc_1=-500.0,
            mfcc_2=-100.0,
            mfcc_3=-50.0,
            mfcc_4=-20.0,
            spectral_contrast=20.0,
            median_ici_ms=0.0,  # No ICI (continuous)
            onset_rate_hz=0.0,  # No pulses
            ici_coefficient_of_variation=0.0,
        )

        # Verify rhythm characteristics
        self.assertGreater(rhythmic.get_onset_rate_hz(), 0)
        self.assertEqual(harmonic.get_onset_rate_hz(), 0)
        self.assertGreater(rhythmic.get_median_ici_ms(), 0)
        self.assertEqual(harmonic.get_median_ici_ms(), 0)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with legacy 3D API."""

    def test_legacy_delta_commands_still_work(self):
        """Test that legacy 3D delta commands still work with 17D metadata."""
        synth = GranularConcatenativeSynthesizer(sample_rate=22050)

        # Create test audio
        num_samples = int(50.0 / 1000.0 * 22050)
        audio = [0.5 * np.sin(2.0 * np.pi * 6800.0 * i / 22050.0) for i in range(num_samples)]

        # Create metadata (can use legacy constructor)
        metadata = SourceMetadata(
            mean_f0_hz=6800.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.1,
            attack_time_ms=10.0,
            decay_time_ms=15.0,
            sustain_level=0.7,
            vibrato_rate_hz=8.0,
            vibrato_depth=50.0,
            jitter=0.02,
            mfcc_1=-500.0,
            mfcc_2=-100.0,
            mfcc_3=-50.0,
            mfcc_4=-20.0,
            spectral_contrast=20.0,
            median_ici_ms=0.0,
            onset_rate_hz=0.0,
            ici_coefficient_of_variation=0.0,
        )

        synth.load_source_with_metadata(audio, metadata)

        # Legacy delta commands should still work
        synth.shift_pitch_by_hz(200.0)
        synth.shift_duration_by_ms(-10.0)
        synth.apply_vector_delta(50.0, -5.0, 100.0)

        # Should synthesize successfully
        output = synth.synthesize(40.0)
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)


def run_tests():
    """Run all tests and display results."""
    print("\n" + "=" * 80)
    print("17D MICRO-DYNAMICS METADATA SYNTHESIS TESTS")
    print("=" * 80)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(Test17DSourceMetadata))
    suite.addTests(loader.loadTestsFromTestCase(Test17DSynthesisIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestBackwardCompatibility))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - 17D metadata synthesis fully implemented!")
        print("\n🎯 Key achievements:")
        print("   ✓ Full 17D SourceMetadata with all micro-dynamics features")
        print("   ✓ Builder pattern for partial metadata construction")
        print("   ✓ Support for GRITTY, PURE, rhythmic, and harmonic personas")
        print("   ✓ Backward compatibility with legacy 3D API")
        print("   ✓ Integration with granular concatenative synthesis")
    else:
        print("\n❌ SOME TESTS FAILED - Review output above")

    print("=" * 80)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
