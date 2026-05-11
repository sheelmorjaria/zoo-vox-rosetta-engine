#!/usr/bin/env python3
"""
Tests for Dual-Stream Interaction Agent (Module 3)

These tests verify the dual-stream cognitive agent that processes
both continuous affect (Stream 1) and discrete syntax (Stream 2).

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from realtime.action_publisher import DualStreamAction, DualStreamState
from realtime.interaction_agent import DualStreamAgentConfig, DualStreamInteractionAgent


class TestDualStreamAgentConfig(unittest.TestCase):
    """Test dual-stream agent configuration."""

    def test_default_config(self):
        """Should initialize with default values."""
        config = DualStreamAgentConfig()

        self.assertEqual(config.response_cooldown_ms, 100.0)
        self.assertEqual(config.default_temporal_offset_ms, 150.0)
        self.assertEqual(config.high_arousal_threshold, 0.8)
        self.assertEqual(config.low_arousal_threshold, 0.3)
        self.assertEqual(config.arousal_deescalation_factor, 0.75)
        self.assertEqual(config.arousal_escalation_factor, 1.2)

    def test_custom_config(self):
        """Should accept custom configuration values."""
        config = DualStreamAgentConfig(
            response_cooldown_ms=200.0,
            high_arousal_threshold=0.7,
            top_k_valid_tokens=10,
        )

        self.assertEqual(config.response_cooldown_ms, 200.0)
        self.assertEqual(config.high_arousal_threshold, 0.7)
        self.assertEqual(config.top_k_valid_tokens, 10)


class TestDualStreamAgent(unittest.TestCase):
    """Test dual-stream interaction agent."""

    def test_agent_initialization(self):
        """Should initialize agent without models."""
        config = DualStreamAgentConfig()
        agent = DualStreamInteractionAgent(config)

        self.assertIsNotNone(agent.action_publisher)
        self.assertIsNone(agent.affective_vae)
        self.assertIsNone(agent.syntactic_vqvae)
        self.assertIsNone(agent.syntax_graph)
        self.assertFalse(agent.is_running())

    def test_agent_start_stop(self):
        """Should start and stop successfully."""
        config = DualStreamAgentConfig(
            action_endpoint="ipc:///tmp/test_dual_stream_actions.ipc",
        )
        agent = DualStreamInteractionAgent(config)

        agent.start()
        self.assertTrue(agent.is_running())

        agent.stop()
        self.assertFalse(agent.is_running())

    def test_handle_dual_stream_state(self):
        """Should process dual-stream state and generate action."""
        config = DualStreamAgentConfig()
        agent = DualStreamInteractionAgent(config)

        # Create dual-stream state
        state = DualStreamState(
            syntactic_token=5,
            affect_vector=np.array([0.5, 0.2, 0.1] + [0.0] * 13, dtype=np.float32),
            raw_features=np.random.randn(112).astype(np.float32),
            confidence=0.85,
            sequence=0,
        )

        # Without syntax graph, should echo token
        action = agent.handle_dual_stream_state(state)

        self.assertIsNotNone(action)
        self.assertEqual(action.syntactic_token, 5)  # Echo fallback
        self.assertEqual(action.affect_vector.shape, (16,))

    def test_confidence_threshold_filtering(self):
        """Should filter out low-confidence states."""
        config = DualStreamAgentConfig(confidence_threshold=0.7)
        agent = DualStreamInteractionAgent(config)

        # Low confidence state
        state = DualStreamState(
            syntactic_token=5,
            affect_vector=np.random.randn(16).astype(np.float32),
            confidence=0.5,  # Below threshold
            sequence=0,
        )

        action = agent.handle_dual_stream_state(state)
        self.assertIsNone(action)

        # High confidence state
        state.confidence = 0.85
        action = agent.handle_dual_stream_state(state)
        self.assertIsNotNone(action)

    def test_affective_deescalation(self):
        """Should de-escalate high arousal (>0.8)."""
        config = DualStreamAgentConfig(
            high_arousal_threshold=0.8,
            arousal_deescalation_factor=0.75,
        )
        agent = DualStreamInteractionAgent(config)

        # High arousal state
        high_arousal = np.zeros(16, dtype=np.float32)
        high_arousal[0] = 0.9  # Above threshold

        state = DualStreamState(
            syntactic_token=5,
            affect_vector=high_arousal,
            confidence=0.85,
            sequence=0,
        )

        action = agent.handle_dual_stream_state(state)

        self.assertIsNotNone(action)
        # Arousal should be scaled down
        self.assertLess(action.affect_vector[0], high_arousal[0])
        self.assertLess(action.affect_vector[0], 0.8)

    def test_affective_escalation(self):
        """Should escalate low arousal (<0.3)."""
        config = DualStreamAgentConfig(
            low_arousal_threshold=0.3,
            arousal_escalation_factor=1.2,
        )
        agent = DualStreamInteractionAgent(config)

        # Low arousal state
        low_arousal = np.zeros(16, dtype=np.float32)
        low_arousal[0] = 0.2  # Below threshold

        state = DualStreamState(
            syntactic_token=5,
            affect_vector=low_arousal,
            confidence=0.85,
            sequence=0,
        )

        action = agent.handle_dual_stream_state(state)

        self.assertIsNotNone(action)
        # Arousal should be scaled up
        self.assertGreater(action.affect_vector[0], low_arousal[0])

    def test_affective_matching(self):
        """Should match medium arousal (0.3-0.8)."""
        config = DualStreamAgentConfig()
        agent = DualStreamInteractionAgent(config)

        # Medium arousal state
        medium_arousal = np.zeros(16, dtype=np.float32)
        medium_arousal[0] = 0.5  # Within medium range

        state = DualStreamState(
            syntactic_token=5,
            affect_vector=medium_arousal,
            confidence=0.85,
            sequence=0,
        )

        action = agent.handle_dual_stream_state(state)

        self.assertIsNotNone(action)
        # Arousal should remain unchanged (matching behavior)
        self.assertEqual(action.affect_vector[0], medium_arousal[0])

    def test_rate_limiting(self):
        """Should enforce response cooldown."""
        config = DualStreamAgentConfig(
            response_cooldown_ms=200.0,
            confidence_threshold=0.5,
        )
        agent = DualStreamInteractionAgent(config)

        state = DualStreamState(
            syntactic_token=5,
            affect_vector=np.random.randn(16).astype(np.float32),
            confidence=0.85,
            sequence=0,
        )

        # First response should succeed
        action1 = agent.handle_dual_stream_state(state)
        self.assertIsNotNone(action1)

        # Immediate second response should be rate-limited
        action2 = agent.handle_dual_stream_state(state)
        self.assertIsNone(action2)

    def test_get_stats(self):
        """Should return accurate statistics."""
        config = DualStreamAgentConfig()
        agent = DualStreamInteractionAgent(config)

        stats = agent.get_stats()

        self.assertIn("running", stats)
        self.assertIn("states_processed", stats)
        self.assertIn("responses_sent", stats)
        self.assertIn("models_loaded", stats)

        # Initially should be false/zero
        self.assertFalse(stats["running"])
        self.assertEqual(stats["states_processed"], 0)


class TestDualStreamState(unittest.TestCase):
    """Test DualStreamState data structure."""

    def test_state_creation(self):
        """Should create state with all fields."""
        state = DualStreamState(
            syntactic_token=5,
            affect_vector=np.random.randn(16).astype(np.float32),
            raw_features=np.random.randn(112).astype(np.float32),
            confidence=0.85,
            sequence=0,
        )

        self.assertEqual(state.syntactic_token, 5)
        self.assertEqual(state.affect_vector.shape, (16,))
        self.assertEqual(state.raw_features.shape, (112,))
        self.assertEqual(state.confidence, 0.85)
        self.assertEqual(state.sequence, 0)

    def test_state_to_dict(self):
        """Should serialize to dictionary."""
        state = DualStreamState(
            syntactic_token=5,
            affect_vector=np.array([0.5] * 16, dtype=np.float32),
            raw_features=np.array([0.3] * 112, dtype=np.float32),
            confidence=0.85,
            sequence=0,
        )

        data = state.to_dict()

        self.assertEqual(data["syntactic_token"], 5)
        self.assertEqual(len(data["affect_vector"]), 16)
        self.assertEqual(len(data["raw_features"]), 112)
        self.assertEqual(data["confidence"], 0.85)

    def test_state_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "syntactic_token": 5,
            "affect_vector": [0.5] * 16,
            "raw_features": [0.3] * 112,
            "confidence": 0.85,
            "sequence": 0,
        }

        state = DualStreamState.from_dict(data)

        self.assertEqual(state.syntactic_token, 5)
        self.assertEqual(state.affect_vector.shape, (16,))
        self.assertEqual(state.raw_features.shape, (112,))

    def test_state_json_roundtrip(self):
        """Should survive JSON serialization roundtrip."""
        original = DualStreamState(
            syntactic_token=5,
            affect_vector=np.array([0.5] * 16, dtype=np.float32),
            raw_features=np.array([0.3] * 112, dtype=np.float32),
            confidence=0.85,
            sequence=0,
        )

        # Serialize
        json_str = original.to_json()

        # Deserialize
        restored = DualStreamState.from_json(json_str)

        # Verify
        self.assertEqual(restored.syntactic_token, original.syntactic_token)
        np.testing.assert_array_almost_equal(
            restored.affect_vector, original.affect_vector
        )
        np.testing.assert_array_almost_equal(
            restored.raw_features, original.raw_features
        )
        self.assertEqual(restored.confidence, original.confidence)


class TestDualStreamAction(unittest.TestCase):
    """Test DualStreamAction data structure."""

    def test_action_creation(self):
        """Should create action with all fields."""
        action = DualStreamAction(
            syntactic_token=10,
            affect_vector=np.random.randn(16).astype(np.float32),
            temporal_offset_ms=150.0,
            priority="high",
            sequence=1,
        )

        self.assertEqual(action.syntactic_token, 10)
        self.assertEqual(action.affect_vector.shape, (16,))
        self.assertEqual(action.temporal_offset_ms, 150.0)
        self.assertEqual(action.priority, "high")
        self.assertEqual(action.sequence, 1)

    def test_action_to_dict(self):
        """Should serialize to dictionary."""
        action = DualStreamAction(
            syntactic_token=10,
            affect_vector=np.array([0.5] * 16, dtype=np.float32),
            temporal_offset_ms=150.0,
            priority="normal",
            sequence=0,
        )

        data = action.to_dict()

        self.assertEqual(data["syntactic_token"], 10)
        self.assertEqual(len(data["affect_vector"]), 16)
        self.assertEqual(data["temporal_offset_ms"], 150.0)

    def test_action_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "syntactic_token": 10,
            "affect_vector": [0.5] * 16,
            "temporal_offset_ms": 150.0,
            "priority": "normal",
            "sequence": 0,
        }

        action = DualStreamAction.from_dict(data)

        self.assertEqual(action.syntactic_token, 10)
        self.assertEqual(action.affect_vector.shape, (16,))
        self.assertEqual(action.temporal_offset_ms, 150.0)

    def test_action_json_roundtrip(self):
        """Should survive JSON serialization roundtrip."""
        original = DualStreamAction(
            syntactic_token=10,
            affect_vector=np.array([0.5] * 16, dtype=np.float32),
            temporal_offset_ms=150.0,
            priority="normal",
            sequence=0,
        )

        # Serialize
        json_str = original.to_json()

        # Deserialize
        restored = DualStreamAction.from_json(json_str)

        # Verify
        self.assertEqual(restored.syntactic_token, original.syntactic_token)
        np.testing.assert_array_almost_equal(
            restored.affect_vector, original.affect_vector
        )
        self.assertEqual(restored.temporal_offset_ms, original.temporal_offset_ms)


class TestIntegration(unittest.TestCase):
    """Integration tests for dual-stream architecture."""

    def test_dual_stream_action_serialization(self):
        """Should serialize DualStreamAction to bytes."""
        action = DualStreamAction(
            syntactic_token=10,
            affect_vector=np.array([0.5] * 16, dtype=np.float32),
            temporal_offset_ms=150.0,
            priority="normal",
            sequence=0,
        )

        # Convert to bytes
        bytes_data = action.to_bytes()
        self.assertIsInstance(bytes_data, bytes)
        self.assertGreater(len(bytes_data), 0)

    def test_dual_stream_state_serialization(self):
        """Should serialize DualStreamState to bytes."""
        state = DualStreamState(
            syntactic_token=5,
            affect_vector=np.array([0.5] * 16, dtype=np.float32),
            raw_features=np.array([0.3] * 112, dtype=np.float32),
            confidence=0.85,
            sequence=0,
        )

        # Convert to bytes
        bytes_data = state.to_bytes()
        self.assertIsInstance(bytes_data, bytes)
        self.assertGreater(len(bytes_data), 0)


if __name__ == "__main__":
    unittest.main()
