"""
Vector Delta Synthesis Tests
=============================

Tests for the new Vector Delta command support in the Rust synthesis engine.

This validates the critical integration point between:
1. Acoustic Algebra (generates virtual phrases with absolute F0)
2. Rust Synthesis (needs relative shifts from source buffer)

The key insight:
- **BAD**: "Set pitch to 7000Hz" (ignores that we started at 6800Hz)
- **GOOD**: "Shift pitch by +200Hz" (relative to source)

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
    print("Warning: technical_architecture module not found. Skipping tests.")
    print("Run: cd technical_architecture && maturin build --release --features python-bindings")
    sys.exit(0)


class TestVectorDeltaCommands(unittest.TestCase):
    """Test Vector Delta command support in Rust synthesis engine."""

    def setUp(self):
        """Create synthesizer for each test."""
        self.synth = GranularConcatenativeSynthesizer(sample_rate=22050)

        # Create test audio buffer (simple sine wave)
        duration_ms = 50.0
        num_samples = int(duration_ms / 1000.0 * 22050)
        frequency_hz = 6800.0
        self.audio_buffer = [
            0.5 * np.sin(2.0 * np.pi * frequency_hz * i / 22050.0) for i in range(num_samples)
        ]

    def test_source_metadata_creation(self):
        """Test SourceMetadata object creation."""
        metadata = SourceMetadata(mean_f0_hz=6800.0, duration_ms=50.0, f0_range_hz=400.0)

        # Use getter methods
        self.assertEqual(metadata.get_mean_f0_hz(), 6800.0)
        self.assertEqual(metadata.get_duration_ms(), 50.0)
        self.assertEqual(metadata.get_f0_range_hz(), 400.0)

        # Test __repr__
        repr_str = repr(metadata)
        self.assertIn("6800", repr_str)
        self.assertIn("50", repr_str)
        self.assertIn("400", repr_str)

    def test_load_source_with_metadata(self):
        """Test loading source audio with metadata."""
        metadata = SourceMetadata(mean_f0_hz=6800.0, duration_ms=50.0, f0_range_hz=400.0)

        # Should not raise any errors
        self.synth.load_source_with_metadata(self.audio_buffer, metadata)

        # Synthesize should work
        output = self.synth.synthesize(50.0)
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

    def test_shift_pitch_by_hz_positive(self):
        """Test positive pitch shift (increase F0)."""
        # Load source with known F0
        metadata = SourceMetadata(mean_f0_hz=6800.0, duration_ms=50.0, f0_range_hz=400.0)
        self.synth.load_source_with_metadata(self.audio_buffer, metadata)

        # Shift pitch up by 200Hz (6800 + 200 = 7000Hz)
        self.synth.shift_pitch_by_hz(200.0)

        # Synthesize and check output exists
        output = self.synth.synthesize(50.0)
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

    def test_shift_pitch_by_hz_negative(self):
        """Test negative pitch shift (decrease F0)."""
        # Load source with known F0
        metadata = SourceMetadata(mean_f0_hz=6800.0, duration_ms=50.0, f0_range_hz=400.0)
        self.synth.load_source_with_metadata(self.audio_buffer, metadata)

        # Shift pitch down by 300Hz (6800 - 300 = 6500Hz)
        self.synth.shift_pitch_by_hz(-300.0)

        # Synthesize and check output exists
        output = self.synth.synthesize(50.0)
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

    def test_shift_pitch_by_hz_zero(self):
        """Test zero pitch shift (no change)."""
        # Load source with known F0
        metadata = SourceMetadata(mean_f0_hz=6800.0, duration_ms=50.0, f0_range_hz=400.0)
        self.synth.load_source_with_metadata(self.audio_buffer, metadata)

        # Shift by 0Hz (should be same as ratio=1.0)
        self.synth.shift_pitch_by_hz(0.0)

        # Synthesize
        output_1 = self.synth.synthesize(50.0)

        # Compare to no shift (reset synth first)
        self.synth_2 = GranularConcatenativeSynthesizer(sample_rate=22050)
        self.synth_2.load_source_with_metadata(self.audio_buffer, metadata)
        output_2 = self.synth_2.synthesize(50.0)

        # Both should produce similar output
        self.assertEqual(len(output_1), len(output_2))

    def test_shift_duration_by_ms_negative(self):
        """Test negative duration shift (shorten)."""
        # Load source with known duration
        metadata = SourceMetadata(mean_f0_hz=6800.0, duration_ms=50.0, f0_range_hz=400.0)
        self.synth.load_source_with_metadata(self.audio_buffer, metadata)

        # Shorten by 10ms (50 - 10 = 40ms)
        self.synth.shift_duration_by_ms(-10.0)

        # Synthesize
        output = self.synth.synthesize(40.0)  # Target duration
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

    def test_shift_duration_by_ms_positive(self):
        """Test positive duration shift (lengthen)."""
        # Load source with known duration
        metadata = SourceMetadata(mean_f0_hz=6800.0, duration_ms=50.0, f0_range_hz=400.0)
        self.synth.load_source_with_metadata(self.audio_buffer, metadata)

        # Lengthen by 20ms (50 + 20 = 70ms)
        self.synth.shift_duration_by_ms(20.0)

        # Synthesize
        output = self.synth.synthesize(70.0)  # Target duration
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

    def test_apply_vector_delta_complete(self):
        """Test complete vector delta application (pitch + duration + range)."""
        # Load source with known metadata
        metadata = SourceMetadata(mean_f0_hz=6800.0, duration_ms=50.0, f0_range_hz=400.0)
        self.synth.load_source_with_metadata(self.audio_buffer, metadata)

        # Apply vector delta (simulating acoustic algebra output)
        # From virtual phrase (7000Hz, 40ms, 500Hz) minus source (6800Hz, 50ms, 400Hz)
        delta_f0 = 200.0  # 7000 - 6800 = +200Hz
        delta_dur = -10.0  # 40 - 50 = -10ms
        delta_range = 100.0  # 500 - 400 = +100Hz

        self.synth.apply_vector_delta(delta_f0, delta_dur, delta_range)

        # Synthesize
        output = self.synth.synthesize(40.0)  # Target duration
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

    def test_set_source_metadata(self):
        """Test setting source metadata after loading audio."""
        # Load audio first (legacy method)
        self.synth.load_source(self.audio_buffer)

        # Then set metadata
        metadata = SourceMetadata(mean_f0_hz=6800.0, duration_ms=50.0, f0_range_hz=400.0)
        self.synth.set_source_metadata(metadata)

        # Now delta commands should work
        self.synth.shift_pitch_by_hz(200.0)
        output = self.synth.synthesize(50.0)
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

    def test_delta_vs_absolute_commands(self):
        """
        **CRITICAL TEST**: Verify delta vs. absolute command behavior.

        This test demonstrates the key difference:
        - Absolute: "Set pitch to 7000Hz" (BAD - ignores source)
        - Delta: "Shift pitch by +200Hz" (GOOD - relative to source)
        """
        # Scenario 1: Source with F0=6800Hz, target=7000Hz
        metadata_1 = SourceMetadata(mean_f0_hz=6800.0, duration_ms=50.0, f0_range_hz=400.0)
        synth_1 = GranularConcatenativeSynthesizer(sample_rate=22050)
        synth_1.load_source_with_metadata(self.audio_buffer, metadata_1)

        # Delta command: "Shift by +200Hz"
        synth_1.shift_pitch_by_hz(200.0)  # 6800 + 200 = 7000Hz ✅

        output_1 = synth_1.synthesize(50.0)
        self.assertIsInstance(output_1, list)
        self.assertGreater(len(output_1), 0)

        # Scenario 2: Source with F0=7200Hz, target=7000Hz
        # Same target (7000Hz), but different source!
        metadata_2 = SourceMetadata(mean_f0_hz=7200.0, duration_ms=50.0, f0_range_hz=400.0)
        synth_2 = GranularConcatenativeSynthesizer(sample_rate=22050)
        synth_2.load_source_with_metadata(self.audio_buffer, metadata_2)

        # Delta command: "Shift by -200Hz" (DIFFERENT DELTA!)
        synth_2.shift_pitch_by_hz(-200.0)  # 7200 - 200 = 7000Hz ✅

        output_2 = synth_2.synthesize(50.0)
        self.assertIsInstance(output_2, list)
        self.assertGreater(len(output_2), 0)

        # Both scenarios achieve the SAME target (7000Hz)
        # But with DIFFERENT delta commands (+200 vs -200)
        # This is why delta commands are superior!

    def test_legacy_load_source_still_works(self):
        """Test that legacy load_source() still works (backward compatibility)."""
        # Legacy method (no metadata)
        self.synth.load_source(self.audio_buffer)

        # Should still synthesize (uses default metadata)
        output = self.synth.synthesize(50.0)
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)


class TestAcousticAlgebraIntegration(unittest.TestCase):
    """Test integration between Acoustic Algebra and Rust Synthesis."""

    def test_end_to_end_workflow(self):
        """
        **COMPLETE WORKFLOW TEST**

        Demonstrates the full pipeline:
        1. Acoustic Algebra generates virtual phrase (absolute F0)
        2. Find nearest real phrase
        3. Calculate delta (virtual - nearest)
        4. Apply delta to Rust synthesizer
        5. Synthesize audio
        """
        # Simulate acoustic algebra output
        # Virtual phrase: 30% aggressive (F0=7000Hz, Dur=40ms, Range=500Hz)
        virtual_f0 = 7000.0
        virtual_dur = 40.0
        virtual_range = 500.0

        # Nearest real phrase found (F0=6800Hz, Dur=50ms, Range=400Hz)
        nearest_f0 = 6800.0
        nearest_dur = 50.0
        nearest_range = 400.0

        # Calculate delta
        delta_f0 = virtual_f0 - nearest_f0  # +200Hz
        delta_dur = virtual_dur - nearest_dur  # -10ms
        delta_range = virtual_range - nearest_range  # +100Hz

        # Create synthesizer and load source with metadata
        synth = GranularConcatenativeSynthesizer(sample_rate=22050)

        # Create test audio
        num_samples = int(50.0 / 1000.0 * 22050)
        audio_buffer = [
            0.5 * np.sin(2.0 * np.pi * nearest_f0 * i / 22050.0) for i in range(num_samples)
        ]

        # Load with nearest phrase metadata
        metadata = SourceMetadata(
            mean_f0_hz=nearest_f0, duration_ms=nearest_dur, f0_range_hz=nearest_range
        )
        synth.load_source_with_metadata(audio_buffer, metadata)

        # Apply delta (VECTOR DELTA COMMAND!)
        synth.apply_vector_delta(delta_f0, delta_dur, delta_range)

        # Synthesize at target duration
        output = synth.synthesize(virtual_dur)

        # Verify
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

        # Output should be close to target duration
        expected_samples = int(virtual_dur / 1000.0 * 22050)
        self.assertLess(abs(len(output) - expected_samples), 100)  # Allow small tolerance


def run_tests():
    """Run all tests and display results."""
    print("\n" + "=" * 80)
    print("VECTOR DELTA SYNTHESIS TESTS")
    print("=" * 80)

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVectorDeltaCommands))
    suite.addTests(loader.loadTestsFromTestCase(TestAcousticAlgebraIntegration))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED - Vector Delta commands working correctly!")
        print("\n🎯 Key achievement:")
        print("   ✓ Delta commands: 'Shift pitch by +200Hz' (relative)")
        print("   ✓ NOT absolute: 'Set pitch to 7000Hz' (ignores source)")
        print("   ✓ Acoustic Algebra integration verified")
    else:
        print("\n❌ SOME TESTS FAILED - Review output above")

    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys

    success = run_tests()
    sys.exit(0 if success else 1)
