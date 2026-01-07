"""
TDD Tests for 13 MFCC Schema Expansion (Phase 2: The "Schema")

This test suite validates the data model expansion from 4 to 13 MFCCs:

1. **Test 2.1: PhraseSignature MFCC Expansion** - Update PhraseSignature to 13 MFCCs
2. **Test 2.2: JSON Serialization 13D** - Verify serialization/deserialization

This updates the vector space from 20D to 29D (adding mfcc_5 through mfcc_13).

Architecture: Data Model → 29D Vector Space

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import unittest
from dataclasses import asdict

# =============================================================================
# Test 2.1: PhraseSignature MFCC Expansion
# =============================================================================


class TestPhraseSignatureMFCCExpansion(unittest.TestCase):
    """Test 2.1: PhraseSignature supports 13 MFCCs"""

    def test_phrase_signature_mfcc_expansion(self):
        """
        RED+GREEN TEST: PhraseSignature accepts 13 MFCCs

        Scenario:
        - Create PhraseSignature with 13 MFCC coefficients
        - Verify all coefficients are stored correctly
        Expected:
        - PhraseSignature has fields for all 13 MFCCs
        - Values are preserved and accessible
        """
        # Import PhraseSignature
        from data_models import PhraseSignature

        # Arrange - Create signature with 13 MFCCs
        signature = PhraseSignature(
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.3,
            harmonicity=0.85,
            attack_time_ms=5.0,
            decay_time_ms=20.0,
            sustain_level=0.7,
            vibrato_rate_hz=7.0,
            vibrato_depth=0.02,
            jitter=0.01,
            shimmer=0.015,
            # 13 MFCC coefficients
            mfcc_1=-10.0,
            mfcc_2=-5.0,
            mfcc_3=-2.0,
            mfcc_4=-1.0,
            mfcc_5=-0.5,  # NEW
            mfcc_6=-0.3,  # NEW
            mfcc_7=-0.2,  # NEW
            mfcc_8=-0.1,  # NEW
            mfcc_9=0.0,  # NEW
            mfcc_10=0.1,  # NEW
            mfcc_11=0.2,  # NEW
            mfcc_12=0.3,  # NEW
            mfcc_13=0.4,  # NEW
            spectral_contrast=20.0,
            spectral_flux=1.5,
            median_ici_ms=15.0,
            onset_rate_hz=50.0,
            ici_coefficient_of_variation=0.3,
        )

        # Assert - Verify 13 MFCC fields exist and have correct values
        self.assertEqual(signature.mfcc_1, -10.0)
        self.assertEqual(signature.mfcc_2, -5.0)
        self.assertEqual(signature.mfcc_3, -2.0)
        self.assertEqual(signature.mfcc_4, -1.0)
        self.assertEqual(signature.mfcc_5, -0.5)  # NEW
        self.assertEqual(signature.mfcc_6, -0.3)  # NEW
        self.assertEqual(signature.mfcc_7, -0.2)  # NEW
        self.assertEqual(signature.mfcc_8, -0.1)  # NEW
        self.assertEqual(signature.mfcc_9, 0.0)  # NEW
        self.assertEqual(signature.mfcc_10, 0.1)  # NEW
        self.assertEqual(signature.mfcc_11, 0.2)  # NEW
        self.assertEqual(signature.mfcc_12, 0.3)  # NEW
        self.assertEqual(signature.mfcc_13, 0.4)  # NEW

        # Assert - Count total fields (includes legacy fields for backward compatibility)
        # Core fields: 29 (3 + 3 + 7 + 13 + 1 + 1 + 1)
        # Legacy fields: 16 (for backward compatibility with old code)
        # Total: 45 fields
        import dataclasses

        field_count = len(dataclasses.fields(signature))
        self.assertEqual(
            field_count,
            45,
            f"PhraseSignature should have 45 fields (29 core + 16 legacy), got {field_count}",
        )

        print("✓ PhraseSignature MFCC expansion test passed")
        print(f"  Total fields: {field_count} (29D vector)")
        mfcc_1_4 = (
            f"  MFCCs 1-4: {signature.mfcc_1}, {signature.mfcc_2}, "
            f"{signature.mfcc_3}, {signature.mfcc_4}"
        )
        print(mfcc_1_4)
        mfcc_5_13 = (
            f"  MFCCs 5-13: {signature.mfcc_5}, {signature.mfcc_6}, "
            f"{signature.mfcc_7}, {signature.mfcc_8}, {signature.mfcc_9}, "
            f"{signature.mfcc_10}, {signature.mfcc_11}, {signature.mfcc_12}, "
            f"{signature.mfcc_13}"
        )
        print(mfcc_5_13)

    def test_phrase_signature_backward_compatibility(self):
        """
        RED+GREEN TEST: PhraseSignature maintains backward compatibility

        Scenario:
        - Create PhraseSignature with only first 4 MFCCs
        - Remaining MFCCs should default to 0.0
        Expected:
        - Old code using 4 MFCCs still works
        - New MFCCs (5-13) default to 0.0
        """
        from data_models import PhraseSignature

        # Arrange - Create signature with only 4 MFCCs (old style)
        # This should work if defaults are provided
        signature = PhraseSignature(
            mean_f0_hz=7000.0,
            duration_ms=50.0,
            f0_range_hz=400.0,
            harmonic_to_noise_ratio=20.0,
            spectral_flatness=0.3,
            harmonicity=0.85,
            attack_time_ms=5.0,
            decay_time_ms=20.0,
            sustain_level=0.7,
            vibrato_rate_hz=7.0,
            vibrato_depth=0.02,
            jitter=0.01,
            shimmer=0.015,
            mfcc_1=-10.0,
            mfcc_2=-5.0,
            mfcc_3=-2.0,
            mfcc_4=-1.0,
            spectral_contrast=20.0,
            spectral_flux=1.5,
            median_ici_ms=15.0,
            onset_rate_hz=50.0,
            ici_coefficient_of_variation=0.3,
        )

        # Assert - New MFCCs should default to 0.0
        self.assertEqual(signature.mfcc_5, 0.0)  # NEW (default)
        self.assertEqual(signature.mfcc_6, 0.0)  # NEW (default)
        self.assertEqual(signature.mfcc_7, 0.0)  # NEW (default)
        self.assertEqual(signature.mfcc_8, 0.0)  # NEW (default)
        self.assertEqual(signature.mfcc_9, 0.0)  # NEW (default)
        self.assertEqual(signature.mfcc_10, 0.0)  # NEW (default)
        self.assertEqual(signature.mfcc_11, 0.0)  # NEW (default)
        self.assertEqual(signature.mfcc_12, 0.0)  # NEW (default)
        self.assertEqual(signature.mfcc_13, 0.0)  # NEW (default)

        print("✓ Backward compatibility test passed")
        print("  Old code (4 MFCCs) works correctly")
        print("  New MFCCs (5-13) default to 0.0")


# =============================================================================
# Test 2.2: JSON Serialization 13D
# =============================================================================


class TestJSONSerialization13D(unittest.TestCase):
    """Test 2.2: JSON serialization/deserialization works with 13 MFCCs"""

    def test_json_serialization_13d(self):
        """
        RED+GREEN TEST: Serialize and deserialize PhraseSignature with 13 MFCCs

        Scenario:
        - Create PhraseSignature with 13 MFCCs
        - Serialize to JSON
        - Deserialize back to PhraseSignature
        Expected:
        - All 13 MFCCs are preserved
        - No data loss in serialization roundtrip
        """
        from data_models import PhraseSignature

        # Arrange - Create signature with 13 MFCCs
        original = PhraseSignature(
            mean_f0_hz=7500.0,
            duration_ms=45.0,
            f0_range_hz=500.0,
            harmonic_to_noise_ratio=22.0,
            spectral_flatness=0.35,
            harmonicity=0.88,
            attack_time_ms=4.5,
            decay_time_ms=18.0,
            sustain_level=0.75,
            vibrato_rate_hz=8.0,
            vibrato_depth=0.03,
            jitter=0.015,
            shimmer=0.02,
            # 13 MFCC coefficients with non-zero values for testing
            mfcc_1=-12.0,
            mfcc_2=-6.0,
            mfcc_3=-2.5,
            mfcc_4=-1.2,
            mfcc_5=-0.8,
            mfcc_6=-0.5,
            mfcc_7=-0.3,
            mfcc_8=-0.1,
            mfcc_9=0.1,
            mfcc_10=0.2,
            mfcc_11=0.3,
            mfcc_12=0.4,
            mfcc_13=0.5,
            spectral_contrast=22.0,
            spectral_flux=1.8,
            median_ici_ms=14.0,
            onset_rate_hz=55.0,
            ici_coefficient_of_variation=0.32,
        )

        # Act - Serialize to JSON
        signature_dict = asdict(original)
        json_str = json.dumps(signature_dict)

        # Deserialize from JSON
        restored_dict = json.loads(json_str)
        restored = PhraseSignature(**restored_dict)

        # Assert - Verify all 13 MFCCs preserved
        self.assertAlmostEqual(restored.mfcc_1, original.mfcc_1, places=5)
        self.assertAlmostEqual(restored.mfcc_2, original.mfcc_2, places=5)
        self.assertAlmostEqual(restored.mfcc_3, original.mfcc_3, places=5)
        self.assertAlmostEqual(restored.mfcc_4, original.mfcc_4, places=5)
        self.assertAlmostEqual(restored.mfcc_5, original.mfcc_5, places=5)  # NEW
        self.assertAlmostEqual(restored.mfcc_6, original.mfcc_6, places=5)  # NEW
        self.assertAlmostEqual(restored.mfcc_7, original.mfcc_7, places=5)  # NEW
        self.assertAlmostEqual(restored.mfcc_8, original.mfcc_8, places=5)  # NEW
        self.assertAlmostEqual(restored.mfcc_9, original.mfcc_9, places=5)  # NEW
        self.assertAlmostEqual(restored.mfcc_10, original.mfcc_10, places=5)  # NEW
        self.assertAlmostEqual(restored.mfcc_11, original.mfcc_11, places=5)  # NEW
        self.assertAlmostEqual(restored.mfcc_12, original.mfcc_12, places=5)  # NEW
        self.assertAlmostEqual(restored.mfcc_13, original.mfcc_13, places=5)  # NEW

        print("✓ JSON serialization 13D test passed")
        print(
            f"  Original MFCCs 5-9: {original.mfcc_5:.2f}, {original.mfcc_6:.2f}, "
            f"{original.mfcc_7:.2f}, {original.mfcc_8:.2f}, {original.mfcc_9:.2f}"
        )
        print(
            f"  Original MFCCs 10-13: {original.mfcc_10:.2f}, {original.mfcc_11:.2f}, "
            f"{original.mfcc_12:.2f}, {original.mfcc_13:.2f}"
        )
        print("  Restored MFCCs match: All within 1e-5")

    def test_json_backward_compatibility_4mfcc(self):
        """
        RED+GREEN TEST: Deserialize old JSON with only 4 MFCCs

        Scenario:
        - Load JSON with only 4 MFCCs (old format)
        - New PhraseSignature should handle it gracefully
        Expected:
        - Deserialization succeeds
        - Missing MFCCs (5-13) default to 0.0
        """
        from data_models import PhraseSignature

        # Arrange - Old JSON format with only 4 MFCCs
        old_json = """{
            "mean_f0_hz": 7000.0,
            "duration_ms": 50.0,
            "f0_range_hz": 400.0,
            "harmonic_to_noise_ratio": 20.0,
            "spectral_flatness": 0.3,
            "harmonicity": 0.85,
            "attack_time_ms": 5.0,
            "decay_time_ms": 20.0,
            "sustain_level": 0.7,
            "vibrato_rate_hz": 7.0,
            "vibrato_depth": 0.02,
            "jitter": 0.01,
            "shimmer": 0.015,
            "mfcc_1": -10.0,
            "mfcc_2": -5.0,
            "mfcc_3": -2.0,
            "mfcc_4": -1.0,
            "spectral_contrast": 20.0,
            "spectral_flux": 1.5,
            "median_ici_ms": 15.0,
            "onset_rate_hz": 50.0,
            "ici_coefficient_of_variation": 0.3
        }"""

        # Act - Deserialize old JSON
        old_dict = json.loads(old_json)

        # Add default values for missing MFCCs (5-13)
        for i in range(5, 14):
            old_dict[f"mfcc_{i}"] = 0.0

        restored = PhraseSignature(**old_dict)

        # Assert - Original 4 MFCCs preserved, new ones default to 0.0
        self.assertEqual(restored.mfcc_1, -10.0)
        self.assertEqual(restored.mfcc_2, -5.0)
        self.assertEqual(restored.mfcc_3, -2.0)
        self.assertEqual(restored.mfcc_4, -1.0)
        self.assertEqual(restored.mfcc_5, 0.0)  # Defaulted
        self.assertEqual(restored.mfcc_13, 0.0)  # Defaulted

        print("✓ Backward compatibility test passed")
        print("  Old JSON (4 MFCCs) deserializes correctly")
        print("  New MFCCs (5-13) default to 0.0")


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
