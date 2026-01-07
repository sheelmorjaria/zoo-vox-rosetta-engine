"""
TDD Tests for 13 MFCC Expansion (Phase 1: The "Extractor")

This test suite validates the expansion from 4 to 13 MFCC coefficients
for improved formant/timbre discrimination:

1. **Test 1.1: Full MFCC Extraction** - Extract 13 coefficients using librosa
2. **Test 1.2: Vector Consistency** - Ensure shape consistency across audio clips

This expands the vector space from 20D to 29D (adding mfcc_5 through mfcc_13).

Architecture: Python Feature Extraction → 13D MFCC Vector Space

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest

import numpy as np

# =============================================================================
# Test 1.1: Full MFCC Extraction
# =============================================================================


class TestMFCCExtraction(unittest.TestCase):
    """Test 1.1: Extract full 13 MFCC coefficients"""

    def test_extract_full_13_mfcc(self):
        """
        RED TEST: Extract 13 MFCC coefficients using librosa

        Scenario:
        - Create synthetic audio with known spectral characteristics
        - Extract MFCCs with n_mfcc=13
        Expected:
        - Output vector length is exactly 13
        - Mean of MFCCs is calculated (time-averaged, not frame-based)
        """
        # Arrange - Create synthetic audio (sine wave at 5kHz)
        sr = 48000
        duration = 0.2  # 200ms
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 5000 * t)

        # Act - Extract MFCCs using the implementation
        from realtime.extract_real_micro_dynamics import extract_13_mfcc

        mfccs = extract_13_mfcc(audio, sr)

        # Assert - Vector length should be exactly 13
        self.assertIsInstance(mfccs, np.ndarray, "MFCC output should be numpy array")
        self.assertEqual(
            mfccs.shape[0], 13, f"MFCC vector should have 13 coefficients, got {mfccs.shape[0]}"
        )
        self.assertEqual(
            len(mfccs.shape),
            1,
            "MFCC output should be 1D array (time-averaged), not 2D (frame-based)",
        )

        # Assert - All values should be finite (no NaN/Inf)
        self.assertTrue(
            np.all(np.isfinite(mfccs)), "All MFCC coefficients should be finite (no NaN/Inf)"
        )

        # Assert - MFCC_0 (log-energy coefficient) should be finite
        # Note: MFCC[0] is log-energy, which is negative for amplitudes < 1.0
        self.assertTrue(
            np.isfinite(mfccs[0]), "First MFCC coefficient (log-energy) should be finite"
        )
        self.assertLess(
            mfccs[0],
            100,  # Upper bound check
            "First MFCC coefficient should be reasonable (< 100)",
        )

        print("✓ Full 13 MFCC extraction test passed")
        print(f"  Vector shape: {mfccs.shape}")
        print(f"  MFCCs: {mfccs}")

    def test_mfcc_vector_shape(self):
        """
        RED TEST: Ensure shape consistency across different audio clips

        Scenario:
        - Extract features from two different audio clips (different frequencies)
        Expected:
        - Both result in np.array of shape (13,)
        - Ensures downstream code (KNN/Algebra) doesn't crash on dimension mismatches
        """
        # Arrange - Create two different audio clips
        sr = 48000
        duration = 0.2
        t = np.linspace(0, duration, int(sr * duration))

        # Clip 1: Low frequency (2kHz)
        audio_low = 0.5 * np.sin(2 * np.pi * 2000 * t)

        # Clip 2: High frequency (8kHz)
        audio_high = 0.5 * np.sin(2 * np.pi * 8000 * t)

        # Act - Extract MFCCs from both
        from realtime.extract_real_micro_dynamics import extract_13_mfcc

        mfcc_low = extract_13_mfcc(audio_low, sr)
        mfcc_high = extract_13_mfcc(audio_high, sr)

        # Assert - Both should have shape (13,)
        self.assertEqual(
            mfcc_low.shape, (13,), f"Low freq MFCCs should have shape (13,), got {mfcc_low.shape}"
        )
        self.assertEqual(
            mfcc_high.shape,
            (13,),
            f"High freq MFCCs should have shape (13,), got {mfcc_high.shape}",
        )

        # Assert - Shapes should match each other
        self.assertEqual(
            mfcc_low.shape, mfcc_high.shape, "Both MFCC vectors should have identical shape"
        )

        # Assert - Different frequencies should produce different MFCCs
        # (especially in higher coefficients which capture spectral envelope)
        max_diff = np.max(np.abs(mfcc_low - mfcc_high))
        self.assertGreater(
            max_diff,
            0.1,
            f"Different frequencies should produce different MFCCs (max diff: {max_diff:.4f})",
        )

        print("✓ MFCC vector consistency test passed")
        print(f"  Low freq MFCCs: {mfcc_low}")
        print(f"  High freq MFCCs: {mfcc_high}")
        print(f"  Max difference: {max_diff:.4f}")


# =============================================================================
# Test 1.3: Real-World Audio Validation
# =============================================================================


class TestMFCCRealWorld(unittest.TestCase):
    """Test 1.3: Validate with realistic marmoset-like vocalizations"""

    def test_mfcc_with_realistic_phee(self):
        """
        RED TEST: Extract MFCCs from realistic phee-like structure

        Scenario:
        - Create phee with frequency modulation (7kHz base, vibrato)
        Expected:
        - 13 coefficients extracted successfully
        - Lower MFCCs (1-4) capture spectral envelope
        - Higher MFCCs (5-13) capture fine spectral structure
        """
        # Arrange - Create realistic phee
        sr = 48000
        duration = 0.3
        t = np.linspace(0, duration, int(sr * duration))

        # Phee with frequency vibrato
        base_freq = 7000
        vibrato_rate = 8  # Hz
        vibrato_depth = 0.05
        freq_mod = base_freq * (1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t))
        phase = 2 * np.pi * np.cumsum(freq_mod) / sr
        audio = 0.3 * np.sin(phase)

        # Act
        from realtime.extract_real_micro_dynamics import extract_13_mfcc

        mfccs = extract_13_mfcc(audio, sr)

        # Assert
        self.assertEqual(
            mfccs.shape[0], 13, f"Real phee should produce 13 MFCCs, got {mfccs.shape[0]}"
        )

        # Assert - Coefficients should show realistic distribution
        # (MFCC_0 is typically largest, decreasing thereafter)
        self.assertGreater(
            abs(mfccs[0]), abs(mfccs[1]), "MFCC_0 should typically be larger than MFCC_1"
        )

        # Assert - Higher coefficients should be non-zero (capture fine structure)
        higher_mfccs = mfccs[5:]  # mfcc_6 through mfcc_13
        self.assertGreater(
            np.mean(np.abs(higher_mfccs)),
            0.001,
            "Higher MFCCs (6-13) should have non-zero values capturing fine spectral structure",
        )

        print("✓ Real phee MFCC test passed")
        print(f"  Lower MFCCs (1-4): {mfccs[1:5]}")
        print(f"  Higher MFCCs (5-13): {mfccs[5:]}")
        print(f"  Mean abs higher MFCCs: {np.mean(np.abs(higher_mfccs)):.6f}")

    def test_mfcc_discriminates_vowel_quality(self):
        """
        RED TEST: MFCCs should distinguish different formant structures

        Scenario:
        - Create two signals with different spectral peaks (simulating vowels)
        Expected:
        - 13 MFCCs capture formant differences
        - Lower MFCCs (1-4) show significant differences
        """
        # Arrange - Create two different spectral shapes
        sr = 48000
        duration = 0.2
        t = np.linspace(0, duration, int(sr * duration))

        # Signal 1: Emphasize lower harmonics (like "ah" vowel)
        audio_vowel1 = (
            0.5 * np.sin(2 * np.pi * 500 * t)
            + 0.3 * np.sin(2 * np.pi * 1000 * t)
            + 0.2 * np.sin(2 * np.pi * 1500 * t)
            + 0.1 * np.sin(2 * np.pi * 2000 * t)
        )

        # Signal 2: Emphasize higher harmonics (like "ee" vowel)
        audio_vowel2 = (
            0.1 * np.sin(2 * np.pi * 500 * t)
            + 0.2 * np.sin(2 * np.pi * 1000 * t)
            + 0.3 * np.sin(2 * np.pi * 2000 * t)
            + 0.5 * np.sin(2 * np.pi * 3000 * t)
        )

        # Act
        from realtime.extract_real_micro_dynamics import extract_13_mfcc

        mfcc1 = extract_13_mfcc(audio_vowel1, sr)
        mfcc2 = extract_13_mfcc(audio_vowel2, sr)

        # Assert - Lower MFCCs (1-4) should differ significantly
        # (these capture broad spectral shape/formants)
        lower_mfcc1 = mfcc1[1:5]  # mfcc_1 through mfcc_4
        lower_mfcc2 = mfcc2[1:5]
        lower_diff = np.max(np.abs(lower_mfcc1 - lower_mfcc2))

        self.assertGreater(
            lower_diff,
            0.5,
            f"Lower MFCCs (1-4) should differ significantly "
            f"for different formant structures (max diff: {lower_diff:.4f})",
        )

        # Assert - Full 13D distance should be measurable
        full_diff = np.linalg.norm(mfcc1 - mfcc2)
        self.assertGreater(
            full_diff,
            1.0,
            f"Full 13D distance should be significant (Euclidean distance: {full_diff:.4f})",
        )

        print("✓ Vowel quality discrimination test passed")
        print(f"  Vowel 1 MFCCs (1-4): {lower_mfcc1}")
        print(f"  Vowel 2 MFCCs (1-4): {lower_mfcc2}")
        print(f"  Lower MFCC diff: {lower_diff:.4f}")
        print(f"  Full 13D distance: {full_diff:.4f}")


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
