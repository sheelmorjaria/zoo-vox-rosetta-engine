#!/usr/bin/env python3
"""
Tests for Interaction Agent

These tests verify the closed-loop cognitive agent that bridges
Rust Execution Layer and Python Logic Layer.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
from unittest.mock import Mock

import numpy as np


class TestInteractionAgentConfig(unittest.TestCase):
    """Test interaction agent configuration"""

    def test_default_config(self):
        """Default config should have correct endpoints"""
        from realtime.interaction_agent import InteractionAgentConfig

        config = InteractionAgentConfig()

        self.assertEqual(config.feature_endpoint, "ipc:///tmp/cognitive_features.ipc")
        self.assertEqual(config.action_endpoint, "ipc:///tmp/cognitive_actions.ipc")
        self.assertEqual(config.response_cooldown_ms, 100.0)
        self.assertEqual(config.max_responses_per_second, 5.0)
        self.assertFalse(config.verbose_logging)

    def test_custom_config(self):
        """Should accept custom configuration"""
        from realtime.interaction_agent import InteractionAgentConfig

        config = InteractionAgentConfig(
            feature_endpoint="tcp://localhost:5555",
            action_endpoint="tcp://localhost:5556",
            response_cooldown_ms=200.0,
            verbose_logging=True,
        )

        self.assertEqual(config.feature_endpoint, "tcp://localhost:5555")
        self.assertEqual(config.action_endpoint, "tcp://localhost:5556")
        self.assertEqual(config.response_cooldown_ms, 200.0)
        self.assertTrue(config.verbose_logging)


class TestAgentState(unittest.TestCase):
    """Test agent state enum"""

    def test_agent_state_values(self):
        """Agent state should have expected values"""
        from realtime.interaction_agent import AgentState

        self.assertEqual(AgentState.IDLE.value, "idle")
        self.assertEqual(AgentState.LISTENING.value, "listening")
        self.assertEqual(AgentState.RESPONDING.value, "responding")


class TestInteractionAgentCreation(unittest.TestCase):
    """Test interaction agent creation"""

    def test_agent_creation_default(self):
        """Should create agent with defaults"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        self.assertIsNotNone(agent.feature_subscriber)
        self.assertIsNotNone(agent.action_publisher)
        self.assertEqual(agent._events_processed, 0)
        self.assertEqual(agent._responses_sent, 0)

    def test_agent_creation_with_callbacks(self):
        """Should accept callbacks in constructor"""
        from realtime.interaction_agent import InteractionAgent

        on_feature = Mock()
        on_context = Mock()

        agent = InteractionAgent(
            on_feature_event=on_feature,
            on_context_change=on_context,
        )

        self.assertEqual(agent.on_feature_event, on_feature)
        self.assertEqual(agent.on_context_change, on_context)


class TestContextInference(unittest.TestCase):
    """Test context inference from 112D features"""

    def test_infer_alarm_context(self):
        """High F0 + high RMS should be alarm"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        # Create features: high F0, high RMS
        features = np.zeros(112, dtype=np.float32)
        features[0] = 9000.0  # High F0
        features[1] = 0.8  # High RMS

        context = agent._infer_context(features)

        self.assertEqual(context, "alarm")

    def test_infer_territorial_context(self):
        """High F0 (but not extreme) should be territorial"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        features = np.zeros(112, dtype=np.float32)
        features[0] = 7000.0  # Medium-high F0

        context = agent._infer_context(features)

        self.assertEqual(context, "territorial")

    def test_infer_social_context(self):
        """Low F0 should be social"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        features = np.zeros(112, dtype=np.float32)
        features[0] = 3000.0  # Low F0

        context = agent._infer_context(features)

        self.assertEqual(context, "social")

    def test_infer_contact_context(self):
        """Medium F0 should be contact"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        features = np.zeros(112, dtype=np.float32)
        features[0] = 5000.0  # Medium F0

        context = agent._infer_context(features)

        self.assertEqual(context, "contact")


class TestResponseTimeline(unittest.TestCase):
    """Test response timeline generation"""

    def test_alarm_timeline(self):
        """Alarm context should produce short, loud timeline"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        timeline = agent._create_response_timeline(42, "alarm")

        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].cluster_id, 42)
        self.assertEqual(timeline[0].duration_ms, 100.0)
        self.assertEqual(timeline[0].amplitude, 0.9)

    def test_territorial_timeline(self):
        """Territorial context should produce longer timeline"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        timeline = agent._create_response_timeline(99, "territorial")

        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].duration_ms, 200.0)
        self.assertEqual(timeline[0].amplitude, 0.85)

    def test_contact_timeline(self):
        """Contact context should produce medium timeline"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        timeline = agent._create_response_timeline(1, "contact")

        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].duration_ms, 150.0)
        self.assertEqual(timeline[0].amplitude, 0.75)


class TestMicroDynamicsDeltas(unittest.TestCase):
    """Test micro-dynamics delta generation"""

    def test_alarm_deltas(self):
        """Alarm context should raise F0 and energy"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        deltas = agent._create_deltas("alarm")

        self.assertIsNotNone(deltas)
        self.assertEqual(deltas.delta_mean_f0_hz, 500.0)
        self.assertEqual(deltas.delta_rms_energy, 0.2)

    def test_territorial_deltas(self):
        """Territorial context should raise F0 and duration"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        deltas = agent._create_deltas("territorial")

        self.assertIsNotNone(deltas)
        self.assertEqual(deltas.delta_mean_f0_hz, 200.0)
        self.assertEqual(deltas.delta_duration_ms, 20.0)

    def test_social_deltas(self):
        """Social context should lower F0"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        deltas = agent._create_deltas("social")

        self.assertIsNotNone(deltas)
        self.assertEqual(deltas.delta_mean_f0_hz, -100.0)

    def test_contact_deltas(self):
        """Contact context should have no deltas"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()

        deltas = agent._create_deltas("contact")

        self.assertIsNone(deltas)


class TestResponseDecision(unittest.TestCase):
    """Test response decision logic"""

    def test_should_respond_alarm(self):
        """Should respond to alarm context"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()
        agent._last_response_time = 0.0  # Allow response

        result = {
            "context_state": "alarm",
            "confidence": 0.8,
        }

        should = agent._should_respond(result)
        self.assertTrue(should)

    def test_should_respond_low_confidence(self):
        """Should not respond with low confidence"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()
        agent._last_response_time = 0.0

        result = {
            "context_state": "alarm",
            "confidence": 0.3,  # Low confidence
        }

        should = agent._should_respond(result)
        self.assertFalse(should)

    def test_should_respond_rate_limited(self):
        """Should not respond if rate limited"""
        import time

        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()
        agent._last_response_time = time.time()  # Just responded

        result = {
            "context_state": "alarm",
            "confidence": 0.8,
        }

        should = agent._should_respond(result)
        self.assertFalse(should)

    def test_should_not_respond_unknown_context(self):
        """Should not respond to unknown context"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()
        agent._last_response_time = 0.0

        result = {
            "context_state": "unknown",
            "confidence": 0.9,
        }

        should = agent._should_respond(result)
        self.assertFalse(should)


class TestFeatureEventProcessing(unittest.TestCase):
    """Test feature event processing"""

    def test_process_features_returns_context(self):
        """Processing should return context"""
        from realtime.interaction_agent import (
            FeatureEvent,
            InteractionAgent,
        )

        agent = InteractionAgent()

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.random.randn(112).astype(np.float32),
            timestamp=1000.0,
            sequence=1,
        )

        result = agent._process_features(event)

        self.assertIn("context_state", result)
        self.assertIn("confidence", result)
        self.assertIn("cluster_id", result)
        self.assertEqual(result["cluster_id"], 42)

    def test_process_features_extracts_key_values(self):
        """Processing should extract key feature values"""
        from realtime.interaction_agent import (
            FeatureEvent,
            InteractionAgent,
        )

        agent = InteractionAgent()

        features = np.zeros(112, dtype=np.float32)
        features[0] = 6000.0  # F0
        features[1] = 0.5  # RMS

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=1,
            features_112d=features,
            timestamp=0.0,
            sequence=1,
        )

        result = agent._process_features(event)

        self.assertEqual(result["cluster_id"], 1)
        self.assertEqual(result["sequence"], 1)


class TestStatistics(unittest.TestCase):
    """Test agent statistics"""

    def test_initial_stats(self):
        """Initial stats should be zero"""
        from realtime.interaction_agent import InteractionAgent

        agent = InteractionAgent()
        stats = agent.get_stats()

        self.assertEqual(stats["events_processed"], 0)
        self.assertEqual(stats["responses_sent"], 0)
        self.assertEqual(stats["state"], "idle")
        self.assertIsNone(stats["current_context"])

    def test_stats_after_processing(self):
        """Stats should update after processing"""
        from realtime.interaction_agent import (
            FeatureEvent,
            InteractionAgent,
        )

        agent = InteractionAgent()

        # Simulate processing an event
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=0.0,
            sequence=1,
        )

        agent._handle_feature_event(event)

        stats = agent.get_stats()
        self.assertEqual(stats["events_processed"], 1)


if __name__ == "__main__":
    unittest.main()
