#!/usr/bin/env python3
"""
End-to-End Integration Tests for Closed-Loop Interaction Agent

These tests verify the complete flow from Rust feature publishing to
Python processing to action publishing.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import time
import unittest

import numpy as np

from realtime.action_publisher import (
    MicroDynamicsDelta,
    SynthesisAction,
    TimelineEvent,
)

# Import all components
from realtime.feature_subscriber import (
    FeatureEvent,
)
from realtime.interaction_agent import (
    InteractionAgent,
    InteractionAgentConfig,
)


class TestFeatureEventRoundtrip(unittest.TestCase):
    """Test feature event serialization roundtrip"""

    def test_feature_event_json_roundtrip(self):
        """Should roundtrip through JSON"""
        original = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.random.randn(112).astype(np.float32),
            timestamp=1000.0,
            sequence=1,
        )

        # Serialize
        json_dict = original.to_json_dict()
        json_str = json.dumps(json_dict)

        # Deserialize
        decoded = FeatureEvent.from_bytes(json_str.encode("utf-8"))

        self.assertEqual(decoded.cluster_id, original.cluster_id)
        self.assertEqual(decoded.sequence, original.sequence)
        self.assertEqual(decoded.timestamp, original.timestamp)
        np.testing.assert_array_almost_equal(
            decoded.features_112d, original.features_112d, decimal=5
        )

    def test_feature_event_matches_rust_format(self):
        """Python format should match Rust format"""
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=1234567890.0,
            sequence=12345,
        )

        # Expected fields that Rust expects
        expected_fields = {
            "event_type",
            "cluster_id",
            "features_112d",
            "timestamp",
            "sequence",
        }

        json_dict = event.to_json_dict()
        self.assertEqual(set(json_dict.keys()), expected_fields)


class TestSynthesisActionRoundtrip(unittest.TestCase):
    """Test synthesis action serialization roundtrip"""

    def test_synthesis_action_json_roundtrip(self):
        """Should roundtrip through JSON"""
        original = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=[
                TimelineEvent(cluster_id=42, start_time_ms=0.0, duration_ms=150.0, amplitude=0.8),
                TimelineEvent(cluster_id=99, start_time_ms=160.0, duration_ms=200.0, amplitude=0.6),
            ],
            deltas=MicroDynamicsDelta(
                delta_mean_f0_hz=100.0,
                delta_duration_ms=20.0,
                delta_rms_energy=0.1,
            ),
            priority="high",
        )

        # Serialize
        json_str = original.to_json()
        json_bytes = json_str.encode("utf-8")

        # Deserialize
        decoded = SynthesisAction.from_json(json_bytes.decode("utf-8"))

        self.assertEqual(decoded.action_type, original.action_type)
        self.assertEqual(len(decoded.timeline), len(original.timeline))
        self.assertEqual(decoded.timeline[0].cluster_id, original.timeline[0].cluster_id)
        self.assertEqual(decoded.priority, original.priority)
        self.assertEqual(decoded.deltas.delta_mean_f0_hz, original.deltas.delta_mean_f0_hz)

    def test_synthesis_action_matches_rust_format(self):
        """Python format should match Rust format"""
        action = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=[
                TimelineEvent(cluster_id=42, start_time_ms=0.0, duration_ms=150.0),
            ],
            deltas=MicroDynamicsDelta(delta_mean_f0_hz=100.0),
            priority="normal",
        )

        # Parse as JSON to verify structure
        data = json.loads(action.to_json())

        # Expected fields that Rust expects
        self.assertIn("action_type", data)
        self.assertIn("timeline", data)
        self.assertIn("deltas", data)
        self.assertIn("priority", data)

        # Timeline event fields
        event = data["timeline"][0]
        self.assertIn("cluster_id", event)
        self.assertIn("start_time_ms", event)
        self.assertIn("duration_ms", event)
        self.assertIn("amplitude", event)


class TestClosedLoopFlow(unittest.TestCase):
    """Test the complete closed-loop flow"""

    def test_feature_to_action_flow(self):
        """Should process features and generate action"""
        # Create agent
        config = InteractionAgentConfig(
            feature_endpoint="ipc:///tmp/test_e2e_features.ipc",
            action_endpoint="ipc:///tmp/test_e2e_actions.ipc",
            response_cooldown_ms=0.0,  # Disable cooldown for testing
        )

        agent = InteractionAgent(config=config)

        # Create feature event simulating alarm call
        features = np.zeros(112, dtype=np.float32)
        features[0] = 9000.0  # High F0 -> alarm
        features[1] = 0.8  # High RMS

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=features,
            timestamp=time.time(),
            sequence=1,
        )

        # Process
        result = agent._process_features(event)

        # Verify context detection
        self.assertEqual(result["context_state"], "alarm")
        self.assertGreater(result["confidence"], 0.0)

        # Verify should respond
        should_respond = agent._should_respond(result)
        self.assertTrue(should_respond)

        # Generate timeline
        timeline = agent._create_response_timeline(42, "alarm")

        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0].cluster_id, 42)
        self.assertEqual(timeline[0].duration_ms, 100.0)
        self.assertEqual(timeline[0].amplitude, 0.9)

        # Generate deltas
        deltas = agent._create_deltas("alarm")

        self.assertIsNotNone(deltas)
        self.assertEqual(deltas.delta_mean_f0_hz, 500.0)

    def test_context_state_transitions(self):
        """Should track context state transitions"""
        agent = InteractionAgent()

        # Initial state
        self.assertIsNone(agent._current_context)

        # Process alarm
        features = np.zeros(112, dtype=np.float32)
        features[0] = 9000.0
        features[1] = 0.8

        event1 = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=1,
            features_112d=features,
            timestamp=0.0,
            sequence=1,
        )

        agent._handle_feature_event(event1)
        self.assertEqual(agent._current_context, "alarm")

        # Process contact call
        features[0] = 5000.0
        features[1] = 0.4

        event2 = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=2,
            features_112d=features,
            timestamp=1.0,
            sequence=2,
        )

        agent._handle_feature_event(event2)
        self.assertEqual(agent._current_context, "contact")


class TestRateLimiting(unittest.TestCase):
    """Test response rate limiting"""

    def test_rate_limiting_prevents_spam(self):
        """Should prevent excessive responses"""
        config = InteractionAgentConfig(
            response_cooldown_ms=1000.0,  # 1 second cooldown
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = time.time()

        # Should not respond immediately after last response
        result = {
            "context_state": "alarm",
            "confidence": 0.9,
        }

        should_respond = agent._should_respond(result)
        self.assertFalse(should_respond)

    def test_rate_limiting_allows_after_cooldown(self):
        """Should allow response after cooldown"""
        config = InteractionAgentConfig(
            response_cooldown_ms=0.01,  # 10ms cooldown
        )

        agent = InteractionAgent(config=config)
        agent._last_response_time = time.time() - 1.0  # 1 second ago

        result = {
            "context_state": "alarm",
            "confidence": 0.9,
        }

        should_respond = agent._should_respond(result)
        self.assertTrue(should_respond)


class TestConfidenceThreshold(unittest.TestCase):
    """Test confidence threshold filtering"""

    def test_low_confidence_no_response(self):
        """Should not respond to low confidence detection"""
        agent = InteractionAgent()
        agent._last_response_time = 0.0

        result = {
            "context_state": "alarm",
            "confidence": 0.3,  # Low confidence
        }

        should_respond = agent._should_respond(result)
        self.assertFalse(should_respond)

    def test_high_confidence_allows_response(self):
        """Should respond to high confidence detection"""
        agent = InteractionAgent()
        agent._last_response_time = 0.0

        result = {
            "context_state": "alarm",
            "confidence": 0.9,  # High confidence
        }

        should_respond = agent._should_respond(result)
        self.assertTrue(should_respond)


class TestStatisticsTracking(unittest.TestCase):
    """Test statistics tracking"""

    def test_event_counting(self):
        """Should count processed events"""
        agent = InteractionAgent()

        # Process multiple events
        for i in range(5):
            features = np.zeros(112, dtype=np.float32)
            features[0] = 5000.0

            event = FeatureEvent(
                event_type="feature_extraction",
                cluster_id=i,
                features_112d=features,
                timestamp=float(i),
                sequence=i,
            )

            agent._handle_feature_event(event)

        stats = agent.get_stats()
        self.assertEqual(stats["events_processed"], 5)

    def test_response_counting(self):
        """Should count sent responses"""
        from unittest.mock import MagicMock

        config = InteractionAgentConfig(
            response_cooldown_ms=0.0,
        )

        agent = InteractionAgent(config=config)

        # Mock the action publisher to always succeed
        agent.action_publisher.publish_timeline = MagicMock(return_value=True)

        # Process multiple alarm events (should trigger responses)
        for i in range(3):
            features = np.zeros(112, dtype=np.float32)
            features[0] = 9000.0  # High F0 -> alarm
            features[1] = 0.8

            event = FeatureEvent(
                event_type="feature_extraction",
                cluster_id=i,
                features_112d=features,
                timestamp=float(i),
                sequence=i,
            )

            agent._handle_feature_event(event)

        stats = agent.get_stats()
        self.assertGreaterEqual(stats["responses_sent"], 1)


class TestTimelineGeneration(unittest.TestCase):
    """Test synthesis timeline generation"""

    def test_alarm_timeline_is_urgent(self):
        """Alarm response should be short and loud"""
        agent = InteractionAgent()

        timeline = agent._create_response_timeline(1, "alarm")

        self.assertEqual(len(timeline), 1)
        self.assertLess(timeline[0].duration_ms, 150.0)
        self.assertGreater(timeline[0].amplitude, 0.8)

    def test_territorial_timeline_is_assertive(self):
        """Territorial response should be medium and strong"""
        agent = InteractionAgent()

        timeline = agent._create_response_timeline(1, "territorial")

        self.assertEqual(len(timeline), 1)
        self.assertGreater(timeline[0].duration_ms, 150.0)
        self.assertGreater(timeline[0].amplitude, 0.8)

    def test_social_timeline_is_conversational(self):
        """Social response should be longer"""
        agent = InteractionAgent()

        timeline = agent._create_response_timeline(1, "social")

        # Social might have multiple events
        self.assertGreaterEqual(len(timeline), 1)


class TestDeltaGeneration(unittest.TestCase):
    """Test micro-dynamics delta generation"""

    def test_alarm_deltas_raise_pitch(self):
        """Alarm deltas should raise pitch"""
        agent = InteractionAgent()

        deltas = agent._create_deltas("alarm")

        self.assertIsNotNone(deltas)
        self.assertGreater(deltas.delta_mean_f0_hz, 0)

    def test_social_deltas_lower_pitch(self):
        """Social deltas should lower pitch"""
        agent = InteractionAgent()

        deltas = agent._create_deltas("social")

        self.assertIsNotNone(deltas)
        self.assertLess(deltas.delta_mean_f0_hz, 0)

    def test_contact_deltas_minimal(self):
        """Contact deltas should be minimal"""
        agent = InteractionAgent()

        deltas = agent._create_deltas("contact")

        # Contact might have no deltas or minimal
        if deltas is not None:
            self.assertEqual(deltas.delta_mean_f0_hz, 0)


if __name__ == "__main__":
    unittest.main()
