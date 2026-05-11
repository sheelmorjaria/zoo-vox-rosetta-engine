#!/usr/bin/env python3
"""
Tests for Level 2.5 Interaction Agent

Tests spatial-social interaction logic, broadcast vs unicast handling,
and response policies.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
from unittest.mock import MagicMock, Mock

import numpy as np

# Try importing required modules
try:
    from fusion_intelligence.receiver_inference import (
        CallDirectionality,
        Level25Context,
        ReceiverInferenceEngine,
    )
    from realtime.action_publisher import DualStreamAction
    from realtime.level25_interaction_agent import (
        Level25Action,
        Level25InteractionAgent,
        Level25Orchestrator,
        ResponsePolicy,
        ResponseStrategy,
    )
    from spatial_intelligence.spatial_ingestor import SpatialFrame, SimulatedIngestor, SpatialObservation
    from spatial_intelligence.topology_engine import TopologyEngine
except ImportError as e:
    raise unittest.SkipTest(f"Required module not found: {e}")


class TestLevel25Action(unittest.TestCase):
    """Test the Level25Action dataclass."""

    def test_create_action(self):
        """Test creating a Level 2.5 action."""
        action = Level25Action(
            syntactic_token=5,
            affect_vector=np.random.randn(16).astype(np.float32),
            call_directionality=CallDirectionality.BROADCAST,
        )

        self.assertEqual(action.syntactic_token, 5)
        self.assertEqual(action.call_directionality, CallDirectionality.BROADCAST)
        self.assertTrue(action.broadcast_flag)

    def test_to_dual_stream_action(self):
        """Test conversion to DualStreamAction."""
        level25_action = Level25Action(
            syntactic_token=5,
            affect_vector=np.zeros(16),
            call_directionality=CallDirectionality.UNICAST,
            target_spatial_id="agent_002",
        )

        ds_action = level25_action.to_dual_stream_action()

        self.assertEqual(ds_action.syntactic_token, 5)
        self.assertEqual(ds_action.temporal_offset_ms, 150.0)


class TestResponsePolicy(unittest.TestCase):
    """Test the ResponsePolicy class."""

    def test_default_policy(self):
        """Test default response policy."""
        policy = ResponsePolicy()

        self.assertTrue(policy.respond_to_alarm)
        self.assertTrue(policy.respond_to_mating)
        self.assertFalse(policy.respond_to_territorial)
        self.assertTrue(policy.respond_to_contact)

    def test_high_arousal_broadcast_triggers_chorus(self):
        """Test that high arousal broadcast triggers chorus response."""
        policy = ResponsePolicy(respond_to_alarm=True)

        affect = np.array([0.9] + [0.0] * 15, dtype=np.float32)
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=affect,
            call_directionality=CallDirectionality.BROADCAST,
            receiver_probabilities={},
            timestamp_ns=0,
        )

        strategy = policy.should_respond(context)

        self.assertEqual(strategy, ResponseStrategy.BROADCAST_CHORUS)

    def test_low_arousal_directed_triggers_match(self):
        """Test that low arousal directed call triggers match."""
        policy = ResponsePolicy(respond_to_contact=True)

        affect = np.array([0.2] + [0.0] * 15, dtype=np.float32)
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=affect,
            call_directionality=CallDirectionality.UNICAST,
            receiver_probabilities={"agent_002": 0.9},
            timestamp_ns=0,
            nearby_count=1,
        )

        strategy = policy.should_respond(context)

        self.assertEqual(strategy, ResponseStrategy.UNICAST_MATCH)

    def test_ignored_when_no_response_policy(self):
        """Test that calls are ignored when policy says no."""
        policy = ResponsePolicy(respond_to_alarm=False, respond_to_contact=False)

        affect = np.array([0.9] + [0.0] * 15, dtype=np.float32)
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=affect,
            call_directionality=CallDirectionality.BROADCAST,
            receiver_probabilities={},
            timestamp_ns=0,
        )

        strategy = policy.should_respond(context)

        self.assertEqual(strategy, ResponseStrategy.BROADCAST_IGNORE)


class TestLevel25InteractionAgent(unittest.TestCase):
    """Test the Level25InteractionAgent class."""

    def setUp(self):
        """Set up test fixtures."""
        self.inference_engine = ReceiverInferenceEngine()
        self.policy = ResponsePolicy(respond_to_alarm=True, respond_to_contact=True)
        self.agent = Level25InteractionAgent(
            receiver_inference_engine=self.inference_engine,
            response_policy=self.policy,
            agent_id="test_agent",
        )

    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertIsNotNone(self.agent)
        self.assertEqual(self.agent.agent_id, "test_agent")

    def test_handle_broadcast_chorus(self):
        """Test handling broadcast chorus (alarm) call."""
        affect = np.array([0.9] + [0.0] * 15, dtype=np.float32)
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=5,
            affect_vector=affect,
            call_directionality=CallDirectionality.BROADCAST,
            receiver_probabilities={},
            timestamp_ns=0,
        )

        action = self.agent.handle_level_25_context(context)

        self.assertIsNotNone(action)
        self.assertEqual(action.call_directionality, CallDirectionality.BROADCAST)
        self.assertTrue(action.broadcast_flag)

    def test_handle_unicast_match(self):
        """Test handling unicast (directed) call."""
        affect = np.array([0.2] + [0.0] * 15, dtype=np.float32)
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=5,
            affect_vector=affect,
            call_directionality=CallDirectionality.UNICAST,
            receiver_probabilities={"agent_002": 0.9},
            timestamp_ns=0,
            nearby_count=1,
        )

        action = self.agent.handle_level_25_context(context)

        self.assertIsNotNone(action)
        self.assertEqual(action.call_directionality, CallDirectionality.UNICAST)
        self.assertEqual(action.target_spatial_id, "agent_001")

    def test_affect_de_escalation(self):
        """Test that high arousal is de-escalated."""
        high_arousal = np.array([0.9] + [0.0] * 15, dtype=np.float32)
        response_affect = self.agent._compute_affective_response(high_arousal)

        # Should be reduced by factor of 0.75
        self.assertLess(response_affect[0], high_arousal[0])
        self.assertAlmostEqual(response_affect[0], 0.675, places=3)

    def test_affect_escalation(self):
        """Test that low arousal is escalated for engagement."""
        low_arousal = np.array([0.2] + [0.0] * 15, dtype=np.float32)
        response_affect = self.agent._compute_affective_response(low_arousal)

        # Should be increased by factor of 1.2
        self.assertGreater(response_affect[0], low_arousal[0])
        self.assertAlmostEqual(response_affect[0], 0.24, places=2)

    def test_affect_matching(self):
        """Test that medium arousal is matched."""
        medium_arousal = np.array([0.5] + [0.0] * 15, dtype=np.float32)
        response_affect = self.agent._compute_affective_response(medium_arousal)

        # Should be approximately matched
        self.assertAlmostEqual(response_affect[0], medium_arousal[0], places=5)

    def test_ignored_call_tracking(self):
        """Test that ignored calls are tracked."""
        policy = ResponsePolicy(respond_to_alarm=False, respond_to_contact=False)
        agent = Level25InteractionAgent(
            receiver_inference_engine=self.inference_engine,
            response_policy=policy,
        )

        affect = np.array([0.9] + [0.0] * 15, dtype=np.float32)
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=affect,
            call_directionality=CallDirectionality.BROADCAST,
            receiver_probabilities={},
            timestamp_ns=0,
        )

        action = agent.handle_level_25_context(context)

        self.assertIsNone(action)
        self.assertEqual(agent.ignored_calls, 1)

    def test_get_statistics(self):
        """Test getting agent statistics."""
        stats = self.agent.get_statistics()

        self.assertIn("broadcast_responses", stats)
        self.assertIn("unicast_responses", stats)
        self.assertIn("ignored_calls", stats)
        self.assertIn("total_processed", stats)


class TestLevel25Orchestrator(unittest.TestCase):
    """Test the Level25Orchestrator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.inference_engine = ReceiverInferenceEngine()
        self.policy = ResponsePolicy(respond_to_alarm=True)
        self.agent = Level25InteractionAgent(
            receiver_inference_engine=self.inference_engine,
            response_policy=self.policy,
        )
        self.orchestrator = Level25Orchestrator(self.inference_engine, self.agent)

        # Set up spatial topology
        self.ingestor = SimulatedIngestor(num_agents=5, area_size=10.0)
        self.topology = TopologyEngine(max_agents=5, proximity_radius=5.0)

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        self.assertIsNotNone(self.orchestrator)

    def test_process_acoustic_event(self):
        """Test full pipeline processing."""
        # Generate spatial frame
        frame = self.ingestor.generate_frame(timestamp_ns=0)
        self.topology.update_topology(frame)

        if frame.observations:
            emitter = frame.observations[0]

            # High arousal broadcast
            affect = np.array([0.9] + [0.0] * 15, dtype=np.float32)

            action = self.orchestrator.process_acoustic_event(
                emitter_id=emitter.agent_id,
                syntactic_token=5,
                affect_vector=affect,
                topology=self.topology,
                timestamp_ns=0,
            )

            self.assertIsNotNone(action)
            self.assertGreater(self.orchestrator.events_processed, 0)

    def test_pipeline_statistics(self):
        """Test getting pipeline statistics."""
        stats = self.orchestrator.get_pipeline_statistics()

        self.assertIn("events_processed", stats)
        self.assertIn("responses_generated", stats)
        self.assertIn("response_rate", stats)
        self.assertIn("agent_stats", stats)

    def test_multiple_events(self):
        """Test processing multiple events."""
        frame = self.ingestor.generate_frame(timestamp_ns=0)
        self.topology.update_topology(frame)

        # Process several events
        for i in range(3):
            if frame.observations:
                emitter = frame.observations[0]
                affect = np.array([0.8] + [0.0] * 15, dtype=np.float32)

                self.orchestrator.process_acoustic_event(
                    emitter_id=emitter.agent_id,
                    syntactic_token=5,
                    affect_vector=affect,
                    topology=self.topology,
                    timestamp_ns=i * 10_000_000,
                )

        stats = self.orchestrator.get_pipeline_statistics()
        self.assertEqual(stats["events_processed"], 3)


class TestSpatialRouting(unittest.TestCase):
    """Test spatial routing logic for directed responses."""

    def test_broadcast_response_no_target(self):
        """Test that broadcast responses have no specific target."""
        affect = np.array([0.9] + [0.0] * 15, dtype=np.float32)
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=5,
            affect_vector=affect,
            call_directionality=CallDirectionality.BROADCAST,
            receiver_probabilities={},
            timestamp_ns=0,
        )

        agent = Level25InteractionAgent(
            receiver_inference_engine=ReceiverInferenceEngine(),
            response_policy=ResponsePolicy(respond_to_alarm=True),
        )

        action = agent.handle_level_25_context(context)

        self.assertIsNotNone(action)
        self.assertTrue(action.broadcast_flag)
        self.assertIsNone(action.target_spatial_id)

    def test_unicast_response_has_target(self):
        """Test that unicast responses target the emitter."""
        affect = np.array([0.3] + [0.0] * 15, dtype=np.float32)
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=5,
            affect_vector=affect,
            call_directionality=CallDirectionality.UNICAST,
            receiver_probabilities={"agent_002": 0.9},
            timestamp_ns=0,
            nearby_count=1,
        )

        agent = Level25InteractionAgent(
            receiver_inference_engine=ReceiverInferenceEngine(),
            response_policy=ResponsePolicy(respond_to_contact=True),
        )

        action = agent.handle_level_25_context(context)

        self.assertIsNotNone(action)
        self.assertFalse(action.broadcast_flag)
        self.assertEqual(action.target_spatial_id, "agent_001")


if __name__ == "__main__":
    unittest.main()
