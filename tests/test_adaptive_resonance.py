#!/usr/bin/env python3
"""
Test suite for Adaptive Resonance enhancement
TDD implementation for Phase IV feature
"""

import sys
import unittest

sys.path.append("src")

from realtime.adaptive_resonance import (
    AdaptiveFilter,
    AdaptiveResonance,
    ContextualModulation,
    DynamicAdaptation,
    FastLearning,
    FeatureDetector,
    NoiseRobustness,
    ResonanceMatcher,
    ResonanceNetwork,
    ResonanceOptimizer,
    ResonanceValidator,
    StabilityMonitor,
)


class TestAdaptiveResonance(unittest.TestCase):
    """Test cases for Adaptive Resonance system"""

    def setUp(self):
        """Set up test environment"""
        self.ar = AdaptiveResonance()
        self.test_signal = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.test_features = {"frequency": 1000, "amplitude": 0.5, "duration": 0.1}
        self.test_context = {"social": True, "alert_level": 0.7}

    def test_adaptive_resonance_creation(self):
        """Test that adaptive resonance system can be created"""
        self.assertIsNotNone(self.ar)
        self.assertIsNotNone(self.ar.resonance_network)
        self.assertIsNotNone(self.ar.feature_detector)

    def test_resonance_network(self):
        """Test resonance network functionality"""
        network = ResonanceNetwork()

        # Test resonance computation
        resonance = network.compute_resonance(
            input_signal=self.test_signal, prototype_signal=self.test_signal
        )
        self.assertIsInstance(resonance, float)
        self.assertGreaterEqual(resonance, 0.0)
        self.assertLessEqual(resonance, 1.0)

        # Test prototype learning
        network.learn_prototype(signal=self.test_signal, context=self.test_context)
        self.assertGreater(len(network.get_prototypes()), 0)

    def test_adaptive_filter(self):
        """Test adaptive filter operations"""
        filter_obj = AdaptiveFilter()

        # Test filtering
        filtered_signal = filter_obj.apply_filter(
            input_signal=self.test_signal, filter_type="lowpass"
        )
        self.assertIsInstance(filtered_signal, list)
        self.assertEqual(len(filtered_signal), len(self.test_signal))

        # Test filter adaptation
        filter_obj.adapt_to_signal(self.test_signal)
        adaptation_gain = filter_obj.get_adaptation_gain()
        self.assertIsInstance(adaptation_gain, float)

    def test_feature_detector(self):
        """Test feature detection capabilities"""
        detector = FeatureDetector()

        # Test feature extraction
        features = detector.extract_features(self.test_signal)
        self.assertIsInstance(features, dict)
        self.assertIn("mean", features)
        self.assertIn("std", features)
        self.assertIn("energy", features)

        # Test feature matching
        match_score = detector.match_features(
            features1=self.test_features, features2=self.test_features
        )
        self.assertIsInstance(match_score, float)
        self.assertGreaterEqual(match_score, 0.0)
        self.assertLessEqual(match_score, 1.0)

    def test_resonance_matcher(self):
        """Test resonance matching system"""
        matcher = ResonanceMatcher()

        # Test resonance matching
        match = matcher.find_best_match(
            input_signal=self.test_signal,
            prototype_signals=[self.test_signal, [0.2, 0.3, 0.4, 0.5, 0.6]],
        )
        self.assertIsInstance(match, dict)
        self.assertIn("index", match)
        self.assertIn("score", match)

        # Test threshold matching
        threshold_match = matcher.match_with_threshold(input_signal=self.test_signal, threshold=0.8)
        self.assertIsInstance(threshold_match, bool)

    def test_stability_monitor(self):
        """Test stability monitoring"""
        monitor = StabilityMonitor()

        # Test stability tracking
        monitor.add_resonance_reading(0.8)
        monitor.add_resonance_reading(0.7)
        monitor.add_resonance_reading(0.9)

        stability = monitor.get_stability()
        self.assertIsInstance(stability, float)
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)

        # Test stability alert
        is_stable = monitor.is_stable(threshold=0.7)
        self.assertIsInstance(is_stable, bool)

    def test_fast_learning(self):
        """Test fast learning capabilities"""
        fast_learner = FastLearning()

        # Test rapid learning
        learning_rate = fast_learner.get_adaptive_learning_rate(base_rate=0.1, iteration=5)
        self.assertIsInstance(learning_rate, float)
        self.assertGreater(learning_rate, 0.0)

        # Test critical period detection
        is_critical = fast_learner.is_critical_period(signal_variance=0.5, context_similarity=0.8)
        self.assertIsInstance(is_critical, bool)

    def test_noise_robustness(self):
        """Test noise robustness features"""
        noise_robust = NoiseRobustness()

        # Test noise filtering
        clean_signal = noise_robust.denoise_signal(
            noisy_signal=[x + 0.1 for x in self.test_signal], noise_level=0.2
        )
        self.assertIsInstance(clean_signal, list)

        # Test noise adaptation
        noise_robust.adapt_to_noise_level(noise_level=0.3)
        robustness = noise_robust.get_robustness_level()
        self.assertIsInstance(robustness, float)

    def test_contextual_modulation(self):
        """Test contextual modulation"""
        context_mod = ContextualModulation()

        # Test context influence
        modulation = context_mod.apply_context_modulation(
            base_resonance=0.8, context=self.test_context
        )
        self.assertIsInstance(modulation, float)

        # Test context learning
        context_mod.learn_context_influence(context=self.test_context, resonance_strength=0.7)

    def test_dynamic_adaptation(self):
        """Test dynamic adaptation capabilities"""
        dynamic = DynamicAdaptation()

        # Test adaptation to changing conditions
        adaptation = dynamic.adapt_to_conditions(
            signal_statistics={"mean": 0.3, "variance": 0.1}, performance_metrics={"accuracy": 0.8}
        )
        self.assertIsInstance(adaptation, dict)

        # Test adaptation rate
        rate = dynamic.get_adaptation_rate()
        self.assertIsInstance(rate, float)

    def test_resonance_optimizer(self):
        """Test resonance optimization"""
        optimizer = ResonanceOptimizer()

        # Test parameter optimization
        params = optimizer.optimize_parameters(
            current_params={"learning_rate": 0.1}, performance_data=[0.7, 0.8, 0.9]
        )
        self.assertIsInstance(params, dict)

        # Test optimization history
        history = optimizer.get_optimization_history()
        self.assertIsInstance(history, list)

    def test_resonance_validator(self):
        """Test resonance validation"""
        validator = ResonanceValidator()

        # Test resonance validation
        is_valid = validator.validate_resonance(resonance_value=0.8, threshold=0.7)
        self.assertIsInstance(is_valid, bool)

        # Test statistical validation
        stats = validator.compute_resonance_statistics(resonance_values=[0.7, 0.8, 0.9, 0.6])
        self.assertIsInstance(stats, dict)
        self.assertIn("mean", stats)
        self.assertIn("std", stats)

    def test_integration_with_main_system(self):
        """Test integration with main analysis system"""
        # Test that AR can be integrated with main pipeline
        self.ar.initialize_resonance_system()

        # Test processing pipeline
        input_data = {
            "signal": self.test_signal,
            "features": self.test_features,
            "context": self.test_context,
        }

        result = self.ar.process_adaptive_resonance(input_data)
        self.assertIsInstance(result, dict)
        self.assertIn("resonance_score", result)
        self.assertIn("adapted_features", result)

    def test_real_time_adaptation(self):
        """Test real-time adaptation capabilities"""
        self.ar.enable_real_time_mode()

        # Simulate real-time processing
        for i in range(10):
            signal = [x * (1 + 0.1 * i) for x in self.test_signal]
            context = {"time": i, "alert_level": 0.5 + 0.05 * i}

            result = self.ar.real_time_process(signal, context)
            self.assertIsInstance(result, dict)

        # Check adaptation progress
        progress = self.ar.get_adaptation_progress()
        self.assertIsInstance(progress, dict)
        self.assertIn("iterations_completed", progress)

    def test_stability_under_change(self):
        """Test stability under changing conditions"""
        # Test with gradually changing signals
        for i in range(20):
            signal = [x * (1 + 0.05 * i) for x in self.test_signal]
            self.ar.process_adaptive_resonance(
                {"signal": signal, "features": self.test_features, "context": self.test_context}
            )

        # Check system stability
        stability = self.ar.get_system_stability()
        self.assertIsInstance(stability, float)
        self.assertGreater(stability, 0.5)  # Should remain reasonably stable

    def test_feature_preservation(self):
        """Test that important features are preserved"""
        original_features = self.test_features.copy()

        # Process through adaptive resonance
        result = self.ar.process_adaptive_resonance(
            {
                "signal": self.test_signal,
                "features": original_features,
                "context": self.test_context,
            }
        )

        # Check that key features are preserved
        adapted_features = result["adapted_features"]
        self.assertIsInstance(adapted_features, dict)

        # Important features should not change dramatically
        for key in ["mean", "energy"]:
            if key in original_features and key in adapted_features:
                relative_change = abs(adapted_features[key] - original_features[key]) / (
                    original_features[key] + 1e-8
                )
                self.assertLess(relative_change, 0.2)  # Less than 20% change


if __name__ == "__main__":
    unittest.main()
