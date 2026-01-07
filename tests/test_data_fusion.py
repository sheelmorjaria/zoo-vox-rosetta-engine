#!/usr/bin/env python3
"""
Test Suite for Data Fusion Implementation
Testing Visual + Vocalization fusion with attention boosting
"""

import sys
import unittest

# Import data fusion module
sys.path.append("src")
import cognitive_intelligence.data_fusion as data_fusion


class TestDataFusion(unittest.TestCase):
    """Test Suite for Data Fusion Implementation"""

    def setUp(self):
        """Set up test fixtures for data fusion tests"""
        self.sample_audio_features = {
            "rms": 0.1,
            "f0": 6000.0,
            "spectral_centroid": 3000.0,
            "bandwidth": 2000.0,
            "context": "contact_call",
            "response_probability": 0.6,
        }
        # Ensure we're not using mock classes by importing directly
        from cognitive_intelligence.data_fusion import FusionConfig

        self.test_config = FusionConfig()

    def test_data_fusion_system_creation(self):
        """Test that Data Fusion System can be created"""
        from cognitive_intelligence.data_fusion import DataFusionSystem, FusionConfig

        # 1. Create configuration
        config = FusionConfig(
            attention_boost_factor=0.2,
            species_weights={"test": {"visual_weight": 0.5, "audio_weight": 0.5}},
        )

        # 2. Create DataFusionSystem instance
        fusion_system = DataFusionSystem(config)

        # 3. Verify configuration is applied
        self.assertEqual(fusion_system.config.attention_boost_factor, 0.2)
        self.assertTrue(fusion_system.config.enable_cross_modal_fusion)
        self.assertIsNotNone(fusion_system.cross_modal_fusion)
        self.assertIsNotNone(fusion_system.response_boost_logic)

    def test_visual_attention_calculation(self):
        """Test that visual attention scores are calculated correctly"""
        from cognitive_intelligence.data_fusion import (
            FusionConfig,
            VisualAttentionCalculator,
            VisualAttentionLevel,
            VisualFeatures,
        )

        # 1. Create calculator
        config = FusionConfig(attention_boost_factor=0.2)
        calculator = VisualAttentionCalculator(config)

        # 2. Test different attention levels
        test_cases = [
            # (visual_features, expected_score_range)
            (
                VisualFeatures(
                    attention_level=VisualAttentionLevel.VERY_HIGH,
                    gaze_direction="towards_camera",
                    movement_intensity=0.9,
                    confidence=0.95,
                ),
                (0.8, 1.0),  # Very high should be in upper range
            ),
            (
                VisualFeatures(
                    attention_level=VisualAttentionLevel.LOW,
                    gaze_direction="away",
                    movement_intensity=0.1,
                    confidence=0.5,
                ),
                (0.0, 0.4),  # Low should be in lower range
            ),
            (
                VisualFeatures(
                    attention_level=VisualAttentionLevel.HIGH,
                    gaze_direction="towards_camera",
                    movement_intensity=0.7,
                    confidence=0.9,
                ),
                (0.6, 0.9),  # High should be in mid-high range
            ),
        ]

        for features, (min_score, max_score) in test_cases:
            score = calculator.calculate_attention_score(features)
            self.assertGreaterEqual(
                score, min_score, f"Score should be >= {min_score} for {features.attention_level}"
            )
            self.assertLessEqual(
                score, max_score, f"Score should be <= {max_score} for {features.attention_level}"
            )

    def test_attention_boost_logic(self):
        """Test that Visual_Attention + Vocalization boosts Response_Probability by 20%"""
        from cognitive_intelligence.data_fusion import (
            AudioFeatures,
            FusionConfig,
            ResponseBoostLogic,
            VisualAttentionLevel,
            VisualFeatures,
        )

        # 1. Create boost logic with explicit config
        config = FusionConfig(
            attention_boost_factor=0.2,
            species_weights={"default": {"visual_weight": 0.5, "audio_weight": 0.5}},
        )
        boost_logic = ResponseBoostLogic(config)

        # 2. Test scenarios for attention boost
        test_cases = [
            # (visual_attention, vocalization_context, expected_boost)
            (VisualAttentionLevel.HIGH, "contact_call", 0.2),  # Full boost
            (VisualAttentionLevel.VERY_HIGH, "contact_call", 0.3),  # 1.5x boost for very high
            (VisualAttentionLevel.MODERATE, "contact_call", 0.1),  # Half boost
            (VisualAttentionLevel.LOW, "contact_call", 0.0),  # No boost for low
            (VisualAttentionLevel.HIGH, "alarm_call", 0.0),  # No boost for alarm
            (VisualAttentionLevel.HIGH, "unknown_context", 0.0),  # No boost for unknown
        ]

        for visual_attention, vocalization_context, expected_boost in test_cases:
            visual_features = VisualFeatures(attention_level=visual_attention)
            audio_features = AudioFeatures(
                rms=0.1,
                f0=6000.0,
                spectral_centroid=3000.0,
                bandwidth=2000.0,
                context=vocalization_context,
                response_probability=0.5,
            )

            # Calculate boost
            boost_amount = boost_logic.calculate_attention_boost(visual_features, audio_features)

            # Verify boost is correct
            context_str = f"{visual_attention} + {vocalization_context}"
            msg = f"Boost should be {expected_boost} for {context_str}"
            self.assertAlmostEqual(
                boost_amount,
                expected_boost,
                places=2,
                msg=msg,
            )

            # Test applying boost to features
            if expected_boost > 0:
                boosted_features = boost_logic.apply_response_boost(audio_features, visual_features)
                expected_probability = 0.5 + expected_boost
                self.assertAlmostEqual(
                    boosted_features.response_probability, expected_probability, places=2
                )
                self.assertEqual(
                    boosted_features.response_probability,
                    audio_features.response_probability + expected_boost,
                )

    def test_cross_modal_weighting(self):
        """Test that cross-modal weighting can be adjusted per species"""
        from cognitive_intelligence.data_fusion import CrossModalFusion, FusionConfig

        # 1. Create fusion with species-specific weights including default
        config = FusionConfig(
            species_weights={
                "marmoset": {"visual_weight": 0.3, "audio_weight": 0.7},
                "dolphin": {"visual_weight": 0.1, "audio_weight": 0.9},
                "human": {"visual_weight": 0.6, "audio_weight": 0.4},
            }
        )
        fusion = CrossModalFusion(config)

        # 2. Test same inputs with different species
        visual_score = 0.8
        auditory_score = 0.6

        test_results = {}
        for species in ["marmoset", "dolphin", "human"]:
            combined = fusion.attention_ensemble.combine_attention_signals(
                visual_score, auditory_score, species
            )
            test_results[species] = combined

        # Verify species-specific behavior
        # Marmoset (more audio weight) should be closer to auditory score
        self.assertLess(test_results["marmoset"], test_results["human"])
        # Dolphin (most audio weight) should be lowest
        self.assertLess(test_results["dolphin"], test_results["marmoset"])

    def test_attention_ensemble(self):
        """Test that attention signals are combined using ensemble method"""
        from cognitive_intelligence.data_fusion import AttentionEnsemble, FusionConfig

        # 1. Create attention ensemble
        config = FusionConfig()
        ensemble = AttentionEnsemble(config)

        # 2. Test attention combinations with default weights
        test_cases = [
            # (visual_attention, auditory_attention, expected_combined)
            (0.8, 0.6, 0.7),  # Weighted average with default 50/50
            (0.9, 0.1, 0.5),  # Visual主导
            (0.2, 0.9, 0.55),  # Auditory主导
        ]

        for visual, auditory, expected in test_cases:
            combined = ensemble.combine_attention_signals(visual, auditory)
            self.assertAlmostEqual(
                combined,
                expected,
                places=2,
                msg=f"Combined attention should be {expected} for {visual} + {auditory}",
            )

    def test_visual_audio_fusion(self):
        """Test that visual and audio features are fused correctly"""
        from cognitive_intelligence.data_fusion import (
            DataFusionSystem,
            FusionConfig,
            VisualAttentionLevel,
            VisualFeatures,
        )

        # 1. Create fusion system
        config = FusionConfig()
        fusion_system = DataFusionSystem(config)

        # 2. Create test features
        visual_features = VisualFeatures(
            attention_level=VisualAttentionLevel.HIGH,
            gaze_direction="towards_camera",
            movement_intensity=0.7,
            confidence=0.9,
        )

        # 3. Test fusion
        fusion_result = fusion_system.integrate_with_audio(
            self.sample_audio_features, visual_features, "marmoset"
        )

        # 4. Verify fusion result structure
        self.assertIsInstance(fusion_result, dict)
        self.assertIn("visual_attention_score", fusion_result)
        self.assertIn("auditory_attention_score", fusion_result)
        self.assertIn("combined_attention", fusion_result)
        self.assertIn("boosted_response_probability", fusion_result)
        self.assertIn("boost_applied", fusion_result)
        self.assertIn("visual_context", fusion_result)

        # Verify specific values
        self.assertGreater(fusion_result["visual_attention_score"], 0.5)  # High attention
        self.assertEqual(fusion_result["visual_context"]["attention_level"], "High")

    def test_species_specific_fusion_logic(self):
        """Test that species-specific fusion logic is applied"""
        from cognitive_intelligence.data_fusion import (
            DataFusionSystem,
            FusionConfig,
            VisualAttentionLevel,
            VisualFeatures,
        )

        # 1. Create fusion system
        config = FusionConfig()
        fusion_system = DataFusionSystem(config)

        # 2. Test with marmoset-specific features
        visual_features = VisualFeatures(attention_level=VisualAttentionLevel.HIGH)
        audio_features = {
            "rms": 0.1,
            "f0": 6000.0,
            "spectral_centroid": 3000.0,
            "bandwidth": 2000.0,
            "context": "contact_call",
            "response_probability": 0.6,
        }

        fusion_result = fusion_system.integrate_with_audio(
            audio_features, visual_features, "marmoset"
        )

        # 3. Verify marmoset-specific logic applied
        self.assertIn("enhanced_social_bonding", fusion_result)
        self.assertTrue(fusion_result["enhanced_social_bonding"])  # Marmoset contact call feature

    def test_contact_call_attention_boost(self):
        """Test that contact calls with high attention get 20% boost"""
        from cognitive_intelligence.data_fusion import (
            DataFusionSystem,
            FusionConfig,
            VisualAttentionLevel,
            VisualFeatures,
        )

        # 1. Create fusion system
        config = FusionConfig(attention_boost_factor=0.2)
        fusion_system = DataFusionSystem(config)

        # 2. Test scenario: High visual attention + contact call → 20% boost
        visual_features = VisualFeatures(
            attention_level=VisualAttentionLevel.HIGH, gaze_direction="towards_camera"
        )
        audio_features = {
            "rms": 0.1,
            "f0": 6000.0,
            "spectral_centroid": 3000.0,
            "bandwidth": 2000.0,
            "context": "contact_call",  # Must be contact call for boost
            "response_probability": 0.5,  # Base probability
        }

        fusion_result = fusion_system.integrate_with_audio(
            audio_features, visual_features, "default"
        )

        # 3. Verify 20% boost is applied
        self.assertTrue(fusion_result["boost_applied"])
        self.assertAlmostEqual(fusion_result["boost_amount"], 0.2, places=2)
        self.assertAlmostEqual(fusion_result["boosted_response_probability"], 0.7, places=2)

    def test_no_boost_for_alarm_calls(self):
        """Test that alarm calls never get attention boost regardless of attention"""
        from cognitive_intelligence.data_fusion import (
            DataFusionSystem,
            FusionConfig,
            VisualAttentionLevel,
            VisualFeatures,
        )

        # 1. Create fusion system
        config = FusionConfig()
        fusion_system = DataFusionSystem(config)

        # 2. Test scenario: High visual attention + alarm call → No boost
        visual_features = VisualFeatures(
            attention_level=VisualAttentionLevel.VERY_HIGH, gaze_direction="towards_camera"
        )
        audio_features = {
            "rms": 0.1,
            "f0": 6000.0,
            "spectral_centroid": 3000.0,
            "bandwidth": 2000.0,
            "context": "alarm_call",  # Alarm calls never get boost
            "response_probability": 0.6,
        }

        fusion_result = fusion_system.integrate_with_audio(
            audio_features, visual_features, "default"
        )

        # 3. Verify no boost is applied
        self.assertFalse(fusion_result["boost_applied"])
        self.assertEqual(fusion_result["boost_amount"], 0.0)
        self.assertEqual(fusion_result["boosted_response_probability"], 0.6)

    def test_performance_monitoring(self):
        """Test that performance statistics are tracked correctly"""
        from cognitive_intelligence.data_fusion import DataFusionSystem, FusionConfig

        # 1. Create fusion system
        config = FusionConfig()
        fusion_system = DataFusionSystem(config)

        # 2. Get initial stats
        initial_stats = fusion_system.get_performance_stats()

        # 3. Verify stats structure
        self.assertIsInstance(initial_stats, dict)
        self.assertIn("fusion_count", initial_stats)
        self.assertIn("boost_count", initial_stats)
        self.assertIn("boost_rate", initial_stats)
        self.assertIn("config", initial_stats)

        # 4. Perform some fusions
        visual_features = data_fusion.VisualFeatures(
            attention_level=data_fusion.VisualAttentionLevel.HIGH
        )

        # Fusion with boost
        fusion_system.integrate_with_audio(self.sample_audio_features, visual_features, "marmoset")

        # Fusion without boost
        audio_features_no_boost = self.sample_audio_features.copy()
        audio_features_no_boost["context"] = "alarm_call"
        fusion_system.integrate_with_audio(audio_features_no_boost, visual_features, "marmoset")

        # 5. Verify stats updated
        final_stats = fusion_system.get_performance_stats()
        self.assertEqual(final_stats["fusion_count"], 2)
        self.assertEqual(final_stats["boost_count"], 1)
        self.assertAlmostEqual(final_stats["boost_rate"], 0.5, places=2)

    def test_error_handling(self):
        """Test that errors are handled gracefully"""
        from cognitive_intelligence.data_fusion import DataFusionSystem, FusionConfig

        # 1. Create fusion system
        config = FusionConfig()
        fusion_system = DataFusionSystem(config)

        # 2. Test with invalid inputs
        invalid_audio = {"invalid": "data"}
        visual_features = data_fusion.VisualFeatures()

        # This should not raise an exception
        result = fusion_system.integrate_with_audio(invalid_audio, visual_features)
        self.assertIsInstance(result, dict)

    def test_context_aware_modulation(self):
        """Test that context-aware modulation enhances combined attention"""
        from cognitive_intelligence.data_fusion import AttentionEnsemble, FusionConfig

        # 1. Create ensemble with context awareness
        config = FusionConfig(context_aware_boosting=True)
        ensemble = AttentionEnsemble(config)

        # 2. Test high attention both modalities gets additional boost
        combined = ensemble.combine_attention_signals(0.8, 0.8, "default")
        # Should be slightly higher than 0.8 due to context boost
        self.assertGreater(combined, 0.8)

        # 3. Test lower attention doesn't get context boost
        combined = ensemble.combine_attention_signals(0.6, 0.6, "default")
        # Should be standard weighted average
        self.assertAlmostEqual(combined, 0.6, places=2)


if __name__ == "__main__":
    # Create test suite with all test cases
    suite = unittest.TestSuite()

    # Add all test methods
    test_methods = [
        "test_data_fusion_system_creation",
        "test_visual_attention_calculation",
        "test_attention_boost_logic",
        "test_cross_modal_weighting",
        "test_attention_ensemble",
        "test_visual_audio_fusion",
        "test_species_specific_fusion_logic",
        "test_contact_call_attention_boost",
        "test_no_boost_for_alarm_calls",
        "test_performance_monitoring",
        "test_error_handling",
        "test_context_aware_modulation",
    ]

    for method in test_methods:
        suite.addTest(TestDataFusion(method))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'=' * 50}")
    print("Data Fusion Test Results:")
    print(f"{'=' * 50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    success_rate = (
        (result.testsRun - len(result.failures) - len(result.errors))
        / result.testsRun * 100
    )
    print(f"Success rate: {success_rate:.1f}%")

    if result.failures:
        print(f"\n{'=' * 50}")
        print("FAILURES:")
        print(f"{'=' * 50}")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print(f"\n{'=' * 50}")
        print("ERRORS:")
        print(f"{'=' * 50}")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
