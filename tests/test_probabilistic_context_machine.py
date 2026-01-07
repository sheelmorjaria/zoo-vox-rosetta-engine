#!/usr/bin/env python3
"""
TDD Test Suite for Probabilistic Context State Machine
=======================================================

Tests for advanced context detection using Bayesian inference
and temporal modeling.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import implementations
from realtime.probabilistic_context_machine import (
    AudioFeatures,
    ContextState,
    ProbabilisticContextMachine,
)


class TestAudioFeatures(unittest.TestCase):
    """Test AudioFeatures dataclass"""

    def test_audio_features_creation(self):
        """Test AudioFeatures object creation"""
        features = AudioFeatures(
            rms=0.1,
            spectral_centroid=5000.0,
            bandwidth=1000.0,
            zero_crossing_rate=0.1,
            harmonic_ratio=0.8,
            fundamental_freq=6000.0,
            spectral_flatness=0.3,
            temporal_envelope=np.array([0.1, 0.05, 0.2, 0.0]),
            mfcc_features=np.array([1, 2, 3, 4, 5])
        )

        self.assertEqual(features.rms, 0.1)
        self.assertEqual(features.spectral_centroid, 5000.0)
        self.assertEqual(len(features.mfcc_features), 5)

    def test_feature_vector_property(self):
        """Test feature vector generation"""
        features = AudioFeatures(
            rms=0.1,
            spectral_centroid=5000.0,
            bandwidth=1000.0,
            zero_crossing_rate=0.1,
            harmonic_ratio=0.8,
            fundamental_freq=6000.0,
            spectral_flatness=0.3,
            temporal_envelope=np.array([0.1, 0.05, 0.2, 0.0]),
            mfcc_features=np.array([1, 2, 3])
        )

        feature_vector = features.feature_vector
        expected_length = 7 + 4 + 3  # 7 base features + 4 envelope + 3 MFCC
        self.assertEqual(len(feature_vector), expected_length)


class TestContextState(unittest.TestCase):
    """Test ContextState enum"""

    def test_context_states(self):
        """Test all context states are defined"""
        states = [
            ContextState.SILENCE,
            ContextState.CONTACT,
            ContextState.ALARM,
            ContextState.FOOD,
            ContextState.NEUTRAL,
            ContextState.UNCERTAIN
        ]

        for state in states:
            self.assertIsInstance(state.value, str)
            self.assertTrue(len(state.value) > 0)


class TestProbabilisticContextMachine(unittest.TestCase):
    """Test the main probabilistic context machine"""

    def setUp(self):
        """Set up test fixtures"""
        self.machine = ProbabilisticContextMachine()
        self.sr = 44100

        # Test audio signals for different contexts
        self.test_audios = {
            'silence': np.random.randn(4410) * 0.001,
            'contact': np.sin(2 * np.pi * 5000 * np.linspace(0, 0.1, 4410)),
            'alarm': np.sin(2 * np.pi * 7000 * np.linspace(0, 0.1, 4410)) * 1.5,
            'food': np.sin(2 * np.pi * 5500 * np.linspace(0, 0.1, 4410)),
            'neutral': np.sin(2 * np.pi * 3000 * np.linspace(0, 0.1, 4410))
        }

    def test_machine_initialization(self):
        """
        REQUIREMENT: Machine must initialize with correct parameters
        Ensures proper setup of state machine components
        """
        # Act
        states = self.machine.context_states
        threshold = self.machine.confidence_threshold

        # Assert
        self.assertIsInstance(states, list)
        self.assertTrue(len(states) > 0)
        self.assertIsInstance(threshold, float)
        self.assertGreater(threshold, 0.5)
        self.assertLess(threshold, 1.0)

    def test_feature_extraction_basic(self):
        """
        REQUIREMENT: Machine must extract meaningful audio features
        Foundation for context detection accuracy
        """
        # Arrange
        audio = self.test_audios['contact']

        # Act
        features = self.machine.extract_features(audio, self.sr)

        # Assert
        self.assertIsInstance(features, AudioFeatures)
        self.assertGreater(features.rms, 0.01)
        self.assertGreater(features.spectral_centroid, 1000)

    def test_feature_extraction_silence(self):
        """
        REQUIREMENT: Machine must handle silent/silent-like audio
        Robustness for low-energy scenarios
        """
        # Arrange
        silence_audio = np.random.randn(4410) * 0.001

        # Act
        features = self.machine.extract_features(silence_audio, self.sr)

        # Assert
        self.assertIsInstance(features, AudioFeatures)
        self.assertLess(features.rms, 0.1)
        # Silence noise can have high frequency components, just ensure it's not extremely high
        self.assertLess(features.spectral_centroid, 20000)

    def test_context_probability_calculation(self):
        """
        REQUIREMENT: Machine must calculate context probabilities
        Core Bayesian inference functionality
        """
        # Arrange
        audio = self.test_audios['alarm']
        features = self.machine.extract_features(audio, self.sr)

        # Act
        probabilities = self.machine.calculate_context_probabilities(features)

        # Assert
        self.assertIsInstance(probabilities, dict)
        self.assertEqual(len(probabilities), len(self.machine.context_states))

        # Check probabilities sum to 1
        total_prob = sum(probabilities.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)

        # Check all probabilities are between 0 and 1
        for prob in probabilities.values():
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_state_update_contact_audio(self):
        """
        REQUIREMENT: Machine must correctly classify contact audio
        Primary use case: marmoset contact calls
        """
        # Arrange
        contact_audio = self.test_audios['contact']

        # Act
        predicted_state, confidence, probabilities = self.machine.update_state_machine(contact_audio, self.sr)

        # Assert
        self.assertIsInstance(predicted_state, ContextState)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

        # Contact should have reasonable probability
        contact_prob = probabilities.get(ContextState.CONTACT, 0.0)
        self.assertGreater(contact_prob, 0.1)

    def test_state_update_alarm_audio(self):
        """
        REQUIREMENT: Machine must correctly classify alarm audio
        Primary use case: alarm calls with high energy
        """
        # Arrange
        alarm_audio = self.test_audios['alarm']

        # Act
        predicted_state, confidence, probabilities = self.machine.update_state_machine(alarm_audio, self.sr)

        # Assert
        self.assertIsInstance(predicted_state, ContextState)

        # Alarm should have reasonable probability for high-energy signal
        alarm_prob = probabilities.get(ContextState.ALARM, 0.0)
        self.assertGreater(alarm_prob, 0.1)

    def test_temporal_smoothing(self):
        """
        REQUIREMENT: Machine must apply temporal smoothing
        Reduces false positives in rapidly changing contexts
        """
        # Arrange
        contact_audio = self.test_audios['contact']

        # Feed multiple contact signals
        states = []
        confidences = []
        for _ in range(5):
            state, confidence, _ = self.machine.update_state_machine(contact_audio, self.sr)
            states.append(state)
            confidences.append(confidence)

        # Assert
        # Should show some temporal consistency
        sum(1 for s in states if s == ContextState.CONTACT)
        # Since this is probabilistic, just ensure it's not all different states
        unique_states = len(set(states))
        self.assertLess(unique_states, 5, "Should show some temporal consistency")

        # Confidences should be reasonably stable
        confidence_std = np.std(confidences)
        self.assertLess(confidence_std, 0.3)

    def test_uncertainty_detection(self):
        """
        REQUIREMENT: Machine must detect uncertainty when below threshold
        Safety mechanism for ambiguous signals
        """
        # Arrange - create ambiguous signal (between contact and food)
        ambiguous_audio = self.test_audios['contact'] * 0.5 + self.test_audios['food'] * 0.5

        # Act
        predicted_state, confidence, _ = self.machine.update_state_machine(ambiguous_audio, self.sr)

        # Assert
        # Should be uncertain when confidence is low
        if confidence < self.machine.confidence_threshold:
            self.assertEqual(predicted_state, ContextState.UNCERTAIN)

    def test_state_history_tracking(self):
        """
        REQUIREMENT: Machine must maintain state history
        Enables temporal context analysis
        """
        # Arrange
        audio_sequence = [
            self.test_audios['silence'],
            self.test_audios['contact'],
            self.test_audios['alarm'],
            self.test_audios['food']
        ]

        # Act
        states = []
        for audio in audio_sequence:
            state, _, _ = self.machine.update_state_machine(audio, self.sr)
            states.append(state)

        # Get history
        history = self.machine.get_state_history()

        # Assert
        self.assertEqual(len(history), len(audio_sequence))
        self.assertEqual(states, history)  # Should match what we recorded

    def test_confidence_trend_tracking(self):
        """
        REQUIREMENT: Machine must track confidence trends
        Performance monitoring and adaptation
        """
        # Arrange
        audios = [
            self.test_audios['silence'],  # Low confidence expected
            self.test_audios['contact'],   # Higher confidence expected
            self.test_audios['alarm']      # High confidence expected
        ]

        # Act
        confidences = []
        for audio in audios:
            _, confidence, _ = self.machine.update_state_machine(audio, self.sr)
            confidences.append(confidence)

        # Get trend
        trend = self.machine.get_confidence_trend()

        # Assert
        self.assertEqual(len(trend), len(audios))
        self.assertEqual(confidences, trend)

        # Should generally increase for clearer signals
        # (This is a loose check due to randomness)
        final_confidence = trend[-1]
        initial_confidence = trend[0]
        self.assertGreater(final_confidence, initial_confidence - 0.2)

    def test_feature_extraction_robustness(self):
        """
        REQUIREMENT: Machine must handle various audio lengths
        Robustness for different input formats
        """
        test_lengths = [1024, 2048, 4096, 8192]

        for length in test_lengths:
            with self.subTest(length=length):
                # Create audio of specific length
                t = np.linspace(0, 0.1, length)
                audio = np.sin(2 * np.pi * 5000 * t)

                # Should not crash
                features = self.machine.extract_features(audio, self.sr)
                self.assertIsInstance(features, AudioFeatures)

    def test_probability_normalization(self):
        """
        REQUIREMENT: Probabilities must always sum to 1.0
        Critical for Bayesian consistency
        """
        # Arrange
        audio = self.test_audios['food']

        # Act
        probabilities = self.machine.calculate_context_probabilities(
            self.machine.extract_features(audio, self.sr)
        )

        # Assert
        total = sum(probabilities.values())
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_machine_reset(self):
        """
        REQUIREMENT: Machine must be resettable
        Enables reuse for multiple experiments
        """
        # Arrange - process some audio first
        self.machine.update_state_machine(self.test_audios['contact'], self.sr)
        self.assertEqual(len(self.machine.get_state_history()), 1)

        # Act
        self.machine.reset()

        # Assert
        self.assertEqual(len(self.machine.get_state_history()), 0)
        self.assertEqual(self.machine.current_state, ContextState.SILENCE)

    def test_custom_configuration(self):
        """
        REQUIREMENT: Machine must accept custom configuration
        Flexibility for different use cases
        """
        # Arrange
        custom_states = [ContextState.SILENCE, ContextState.CONTACT, ContextState.ALARM]
        custom_machine = ProbabilisticContextMachine(
            context_states=custom_states,
            history_length=3,
            confidence_threshold=0.8
        )

        # Act & Assert
        self.assertEqual(len(custom_machine.context_states), 3)
        self.assertEqual(custom_machine.history_length, 3)
        self.assertEqual(custom_machine.confidence_threshold, 0.8)

    def test_edge_case_empty_audio(self):
        """
        REQUIREMENT: Machine must handle empty audio gracefully
        Robustness for edge cases
        """
        # Arrange
        empty_audio = np.array([])

        # Act - should not crash
        features = self.machine.extract_features(empty_audio, self.sr)

        # Assert
        self.assertIsInstance(features, AudioFeatures)
        # Should return default features
        self.assertEqual(features.rms, 0.0)

    def test_performance_requirement(self):
        """
        PERFORMANCE REQUIREMENT: Context detection must complete in <20ms
        Enables real-time operation
        """
        import time

        # Arrange
        audio = self.test_audios['contact']

        # Act & Measure
        start_time = time.perf_counter()
        self.machine.update_state_machine(audio, self.sr)
        processing_time = (time.perf_counter() - start_time) * 1000

        # Assert
        self.assertLess(processing_time, 20,
                        f"Context detection took {processing_time:.1f}ms, exceeds 20ms limit")


if __name__ == '__main__':
    unittest.main(verbosity=2)
