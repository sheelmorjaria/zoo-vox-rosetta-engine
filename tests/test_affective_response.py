#!/usr/bin/env python3
"""
Tests for Affective Response Logic (Module 1)

Tests the biologically-inspired affective response logic with:
- De-escalation for high arousal (>0.8) to avoid panic cascades
- Matching for low arousal to maintain social contact
- Proper arousal/valence interpretation from 16D latent space

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest

import numpy as np
import torch

from cognitive_intelligence.affective_response import (
    AffectiveResponseConfig,
    AffectiveResponsePolicy,
    compute_affective_response,
    create_affective_response_policy,
)


class TestAffectiveExtraction(unittest.TestCase):
    """Test extraction of arousal and valence from affect vectors."""

    def setUp(self):
        """Create a policy with default config."""
        self.policy = create_affective_response_policy()

    def test_extract_arousal_from_numpy(self):
        """Should extract arousal (dim 0) from numpy array."""
        affect = np.array([0.7, 0.2, 0.5, 0.3, 0.1, 0.0, 0.2, 0.1,
                           0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        arousal = self.policy.extract_arousal(affect)
        self.assertEqual(arousal, 0.7)

    def test_extract_arousal_from_torch(self):
        """Should extract arousal (dim 0) from torch tensor."""
        affect = torch.tensor([0.7, 0.2, 0.5, 0.3, 0.1, 0.0, 0.2, 0.1,
                               0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        arousal = self.policy.extract_arousal(affect)
        self.assertAlmostEqual(arousal, 0.7, places=5)

    def test_extract_valence_from_numpy(self):
        """Should extract valence (dim 1) from numpy array."""
        affect = np.array([0.7, -0.5, 0.5, 0.3, 0.1, 0.0, 0.2, 0.1,
                           0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        valence = self.policy.extract_valence(affect)
        self.assertEqual(valence, -0.5)

    def test_extract_valence_clamps_to_range(self):
        """Valence should be clamped to [-1, 1]."""
        affect = np.array([0.7, 2.5, 0.5, 0.3, 0.1, 0.0, 0.2, 0.1,
                           0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        valence = self.policy.extract_valence(affect)
        self.assertEqual(valence, 1.0)

    def test_arousal_clamps_to_range(self):
        """Arousal should be clamped to [0, 1]."""
        affect = np.array([1.5, 0.2, 0.5, 0.3, 0.1, 0.0, 0.2, 0.1,
                           0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        arousal = self.policy.extract_arousal(affect)
        self.assertEqual(arousal, 1.0)


class TestDeescalationLogic(unittest.TestCase):
    """Test de-escalation logic for high arousal states."""

    def setUp(self):
        """Create a policy with default config."""
        self.policy = create_affective_response_policy()

    def test_high_arousal_triggers_deescalation(self):
        """High arousal (>0.8) should trigger de-escalation."""
        # Create high arousal state
        affect = np.zeros(16)
        affect[0] = 0.9  # High arousal

        target = self.policy.compute_target_affect(affect)

        # Arousal should be reduced
        self.assertLess(target[0], 0.9)
        # Target should be close to de-escalation target
        self.assertGreater(target[0], 0.5)

    def test_deescalation_preserves_other_dimensions(self):
        """De-escalation should preserve other affect dimensions."""
        affect = np.zeros(16)
        affect[0] = 0.9  # High arousal
        affect[1] = -0.5  # Negative valence
        affect[2] = 0.7  # Pitch variation

        target = self.policy.compute_target_affect(affect)

        # Other dimensions should be similar (within tolerance)
        # Valence may be slightly reduced for high arousal states
        self.assertAlmostEqual(target[2], 0.7, delta=0.1)

    def test_extreme_arousal_deescalation(self):
        """Extreme arousal (>0.95) should be de-escalated more aggressively."""
        affect = np.zeros(16)
        affect[0] = 0.99  # Extreme arousal

        target = self.policy.compute_target_affect(affect)

        # Should be reduced (decay_rate=0.3 moves toward 0.6: 0.99*0.7 + 0.6*0.3 = 0.873)
        # Still less than original 0.99
        self.assertLess(target[0], 0.99)
        self.assertGreater(target[0], 0.5)

    def test_panic_state_detection(self):
        """Should correctly detect panic states."""
        # Normal state
        affect_normal = np.zeros(16)
        affect_normal[0] = 0.5
        self.assertFalse(self.policy.is_panic_state(affect_normal))

        # Panic state
        affect_panic = np.zeros(16)
        affect_panic[0] = 0.95
        self.assertTrue(self.policy.is_panic_state(affect_panic))

    def test_should_deescalate_check(self):
        """should_deescalate should return True for high arousal."""
        # Below threshold
        affect_low = np.zeros(16)
        affect_low[0] = 0.7
        self.assertFalse(self.policy.should_deescalate(affect_low))

        # Above threshold
        affect_high = np.zeros(16)
        affect_high[0] = 0.85
        self.assertTrue(self.policy.should_deescalate(affect_high))


class TestEscalationLogic(unittest.TestCase):
    """Test escalation logic for low arousal states."""

    def setUp(self):
        """Create a policy with default config."""
        self.policy = create_affective_response_policy()

    def test_low_arousal_triggers_escalation(self):
        """Low arousal (<0.3) should trigger escalation."""
        affect = np.zeros(16)
        affect[0] = 0.1  # Low arousal

        target = self.policy.compute_target_affect(affect)

        # Arousal should be increased
        self.assertGreater(target[0], 0.1)

    def test_escalation_has_upper_limit(self):
        """Escalation should not push arousal into high range."""
        affect = np.zeros(16)
        affect[0] = 0.25  # Low arousal but not extreme

        target = self.policy.compute_target_affect(affect)

        # Should not exceed high threshold (0.8) * 0.9
        self.assertLess(target[0], 0.75)

    def test_should_escalate_check(self):
        """should_escalate should return True for low arousal."""
        # Above threshold
        affect_ok = np.zeros(16)
        affect_ok[0] = 0.4
        self.assertFalse(self.policy.should_escalate(affect_ok))

        # Below threshold
        affect_low = np.zeros(16)
        affect_low[0] = 0.2
        self.assertTrue(self.policy.should_escalate(affect_low))


class TestMatchingLogic(unittest.TestCase):
    """Test matching logic for medium arousal states."""

    def setUp(self):
        """Create a policy with default config."""
        self.policy = create_affective_response_policy()

    def test_medium_arousal_triggers_matching(self):
        """Medium arousal (0.3-0.8) should trigger matching."""
        affect = np.zeros(16)
        affect[0] = 0.5  # Medium arousal

        target = self.policy.compute_target_affect(affect)

        # Arousal should remain similar (within tolerance)
        self.assertAlmostEqual(target[0], 0.5, delta=0.1)

    def test_matching_adds_small_noise(self):
        """Matching should add small random noise for natural variation."""
        affect = np.zeros(16)
        affect[0] = 0.5

        target1 = self.policy.compute_target_affect(affect.copy())
        target2 = self.policy.compute_target_affect(affect.copy())

        # Two calls should produce slightly different results
        # due to random noise in matching
        self.assertNotEqual(target1[0], target2[0])


class TestIntegration(unittest.TestCase):
    """Integration tests for affective response logic."""

    def test_full_response_pipeline(self):
        """Test complete affective response pipeline."""
        policy = create_affective_response_policy()

        # Simulate sequence of affect states
        states = [
            np.zeros(16) + 0.2,  # Low arousal → escalate
            np.zeros(16) + 0.5,  # Medium → match
            np.zeros(16) + 0.9,  # High → de-escalate
        ]

        responses = []
        for state in states:
            state[0] = state[0, 0] if state.ndim > 1 else state[0]
            response = policy.compute_target_affect(state)
            responses.append(response)

        # First response: escalated
        self.assertGreater(responses[0][0], 0.2)

        # Second response: matched (similar)
        self.assertAlmostEqual(responses[1][0], 0.5, delta=0.1)

        # Third response: de-escalated
        self.assertLess(responses[2][0], 0.9)

    def test_convenience_function(self):
        """Test the convenience function compute_affective_response."""
        affect = np.zeros(16)
        affect[0] = 0.9  # High arousal

        target = compute_affective_response(affect)

        # Should de-escalate
        self.assertLess(target[0], 0.9)

    def test_configurable_thresholds(self):
        """Test that thresholds can be customized."""
        config = AffectiveResponseConfig(
            HIGH_AROUSAL_THRESHOLD=0.7,
            LOW_AROUSAL_THRESHOLD=0.4,
        )
        policy = AffectiveResponsePolicy(config)

        # Arousal of 0.75 should now trigger de-escalation (above threshold)
        affect = np.zeros(16)
        affect[0] = 0.75

        target = policy.compute_target_affect(affect)

        # Should de-escalate (arousal reduced)
        self.assertLess(target[0], 0.75)


if __name__ == "__main__":
    unittest.main()
