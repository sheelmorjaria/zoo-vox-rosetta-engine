#!/usr/bin/env python3
"""
Tests for Fusion Intelligence Module (Level 2.5)

Tests receiver inference, multimodal fusion, and broadcast detection.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
from unittest.mock import MagicMock, Mock

import numpy as np

# Try importing required modules
try:
    from fusion_intelligence.receiver_inference import (
        BroadcastDetector,
        CallDirectionality,
        InferenceWeights,
        Level25Context,
        MultiModalFusionBuffer,
        ReceiverInferenceEngine,
    )
    from spatial_intelligence.spatial_ingestor import SimulatedIngestor, SpatialFrame, SpatialObservation
    from spatial_intelligence.topology_engine import TopologyEngine
except ImportError as e:
    raise unittest.SkipTest(f"Required module not found: {e}")


class TestInferenceWeights(unittest.TestCase):
    """Test the InferenceWeights dataclass."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = InferenceWeights()

        self.assertAlmostEqual(weights.proximity_weight, 0.6, places=5)
        self.assertAlmostEqual(weights.los_weight, 0.3, places=5)
        self.assertAlmostEqual(weights.social_weight, 0.1, places=5)

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1.0."""
        weights = InferenceWeights(
            proximity_weight=10.0,
            los_weight=5.0,
            social_weight=5.0,
        )

        total = weights.proximity_weight + weights.los_weight + weights.social_weight

        self.assertAlmostEqual(total, 1.0, places=5)


class TestLevel25Context(unittest.TestCase):
    """Test the Level25Context dataclass."""

    def test_create_context(self):
        """Test creating a Level 2.5 context."""
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=5,
            affect_vector=np.random.randn(16).astype(np.float32),
            call_directionality=CallDirectionality.UNICAST,
            receiver_probabilities={"agent_002": 0.8, "agent_003": 0.2},
            timestamp_ns=0,
        )

        self.assertEqual(context.emitter_id, "agent_001")
        self.assertEqual(context.syntactic_token, 5)
        self.assertEqual(context.call_directionality, CallDirectionality.UNICAST)

    def test_get_top_receivers(self):
        """Test getting top receivers."""
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=np.zeros(16),
            call_directionality=CallDirectionality.UNICAST,
            receiver_probabilities={
                "agent_002": 0.5,
                "agent_003": 0.3,
                "agent_004": 0.2,
            },
            timestamp_ns=0,
        )

        top_3 = context.get_top_receivers(3)

        self.assertEqual(len(top_3), 3)
        self.assertEqual(top_3[0][0], "agent_002")  # Highest probability
        self.assertEqual(top_3[0][1], 0.5)

    def test_has_targets(self):
        """Test checking if context has target receivers."""
        # With targets
        context_with = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=np.zeros(16),
            call_directionality=CallDirectionality.UNICAST,
            receiver_probabilities={"agent_002": 0.8},
            timestamp_ns=0,
        )
        self.assertTrue(context_with.has_targets())

        # Without targets
        context_without = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=np.zeros(16),
            call_directionality=CallDirectionality.BROADCAST,
            receiver_probabilities={},
            timestamp_ns=0,
        )
        self.assertFalse(context_without.has_targets())

    def test_is_broadcast(self):
        """Test broadcast check."""
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=np.zeros(16),
            call_directionality=CallDirectionality.BROADCAST,
            receiver_probabilities={},
            timestamp_ns=0,
        )

        self.assertTrue(context.is_broadcast())
        self.assertFalse(context.is_unicast())

    def test_is_unicast(self):
        """Test unicast check."""
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=np.zeros(16),
            call_directionality=CallDirectionality.UNICAST,
            receiver_probabilities={"agent_002": 0.9},
            timestamp_ns=0,
        )

        self.assertTrue(context.is_unicast())
        self.assertFalse(context.is_broadcast())


class TestReceiverInferenceEngine(unittest.TestCase):
    """Test the ReceiverInferenceEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = ReceiverInferenceEngine()
        self.topology = TopologyEngine(max_agents=10, proximity_radius=5.0)

    def test_create_engine(self):
        """Test creating a receiver inference engine."""
        engine = ReceiverInferenceEngine()

        self.assertIsNotNone(engine)
        self.assertAlmostEqual(engine.weights.proximity_weight, 0.6)

    def test_infer_with_no_nearby_agents(self):
        """Test inference when no agents are nearby."""
        # Empty topology
        context = self.engine.infer_receiver(
            emitter_id="agent_001",
            topology=self.topology,
            syntactic_token=0,
            affect_vector=np.zeros(16),
            timestamp_ns=0,
        )

        self.assertEqual(context.emitter_id, "agent_001")
        self.assertEqual(context.call_directionality, CallDirectionality.BROADCAST)
        self.assertEqual(len(context.receiver_probabilities), 0)
        self.assertEqual(context.nearby_count, 0)

    def test_infer_with_nearby_agents(self):
        """Test inference with nearby agents."""
        # Create a frame with 3 agents
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="agent_001", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_002", x=1.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)  # Close, ahead
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_003", x=4.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)  # Further away
        )

        self.topology.update_topology(frame)

        context = self.engine.infer_receiver(
            emitter_id="agent_001",
            topology=self.topology,
            syntactic_token=0,
            timestamp_ns=0,
        )

        # Should have detected nearby agents
        self.assertGreater(context.nearby_count, 0)
        self.assertIn("agent_002", context.receiver_probabilities)
        self.assertIn("agent_003", context.receiver_probabilities)

        # agent_002 should have higher probability (closer)
        self.assertGreater(
            context.receiver_probabilities["agent_002"],
            context.receiver_probabilities["agent_003"]
        )

    def test_classify_unicast(self):
        """Test classification as unicast."""
        # High concentration probability
        probs = {"agent_002": 0.8, "agent_003": 0.1, "agent_004": 0.1}

        directionality = self.engine._classify_directionality(probs)

        self.assertEqual(directionality, CallDirectionality.UNICAST)

    def test_classify_broadcast(self):
        """Test classification as broadcast."""
        # Evenly distributed probabilities
        probs = {"agent_002": 0.4, "agent_003": 0.35, "agent_004": 0.25}

        directionality = self.engine._classify_directionality(probs)

        self.assertEqual(directionality, CallDirectionality.BROADCAST)

    def test_line_of_sight_affects_probability(self):
        """Test that line-of-sight affects receiver probability."""
        # Create frame where agent_002 is ahead (in FoV) and agent_003 is behind
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="agent_001", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)  # Facing +X
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_002", x=1.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)  # Ahead
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_003", x=-1.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)  # Behind
        )

        self.topology.update_topology(frame)

        # Create engine with high LoS weight
        engine = ReceiverInferenceEngine(
            weights=InferenceWeights(proximity_weight=0.1, los_weight=0.8, social_weight=0.1)
        )

        context = engine.infer_receiver(
            emitter_id="agent_001",
            topology=self.topology,
            timestamp_ns=0,
        )

        # agent_002 should have much higher probability due to LoS
        self.assertGreater(
            context.receiver_probabilities["agent_002"],
            context.receiver_probabilities["agent_003"]
        )

    def test_update_social_affinity(self):
        """Test updating social affinity."""
        self.engine.update_social_affinity("agent_001", "agent_002", 0.8)

        affinity = self.engine._get_social_affinity("agent_001", "agent_002")

        self.assertAlmostEqual(affinity, 0.8)

    def test_social_affinity_clipping(self):
        """Test that social affinity is clipped to [0, 1]."""
        # Try to set invalid values
        self.engine.update_social_affinity("agent_001", "agent_002", -0.5)
        self.engine.update_social_affinity("agent_001", "agent_003", 1.5)

        affinity1 = self.engine._get_social_affinity("agent_001", "agent_002")
        affinity2 = self.engine._get_social_affinity("agent_001", "agent_003")

        # Should be clipped to valid range
        self.assertEqual(affinity1, 0.0)
        self.assertEqual(affinity2, 1.0)


class TestMultiModalFusionBuffer(unittest.TestCase):
    """Test the MultiModalFusionBuffer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.buffer = MultiModalFusionBuffer()

    def test_add_acoustic_event(self):
        """Test adding an acoustic event."""
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=np.zeros(16),
            call_directionality=CallDirectionality.BROADCAST,
            receiver_probabilities={},
            timestamp_ns=0,
        )

        self.buffer.add_acoustic_event(context)

        self.assertEqual(len(self.buffer.acoustic_events), 1)

    def test_update_spatial_topology(self):
        """Test updating spatial topology."""
        topology = TopologyEngine(max_agents=5)
        self.buffer.update_spatial_topology(topology, timestamp_ns=0)

        self.assertIsNotNone(self.buffer.spatial_topology)
        self.assertEqual(self.buffer.last_spatial_timestamp_ns, 0)

    def test_prune_old_events(self):
        """Test that old acoustic events are pruned."""
        # Add event at time 0
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=np.zeros(16),
            call_directionality=CallDirectionality.BROADCAST,
            receiver_probabilities={},
            timestamp_ns=0,
        )
        self.buffer.add_acoustic_event(context)

        # Update topology with much later time
        topology = TopologyEngine(max_agents=5)
        self.buffer.update_spatial_topology(topology, timestamp_ns=200_000_000)  # 200ms

        # Old event should be pruned (max_age_ms = 100)
        self.assertEqual(len(self.buffer.acoustic_events), 0)


class TestBroadcastDetector(unittest.TestCase):
    """Test the BroadcastDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = BroadcastDetector()

    def test_detect_by_arousal(self):
        """Test broadcast detection by high arousal."""
        # High arousal affect vector
        affect = np.array([0.9] + [0.0] * 15, dtype=np.float32)

        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=affect,
            call_directionality=CallDirectionality.BROADCAST,
            receiver_probabilities={"agent_002": 0.5},
            timestamp_ns=0,
        )

        is_broadcast = self.detector.is_broadcast_call(context)

        self.assertTrue(is_broadcast)

    def test_detect_by_entropy(self):
        """Test broadcast detection by high entropy."""
        # Evenly distributed probabilities = high entropy
        probs = {f"agent_{i:03d}": 1.0/10 for i in range(10)}

        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=np.zeros(16),  # Low arousal
            call_directionality=CallDirectionality.BROADCAST,
            receiver_probabilities=probs,
            timestamp_ns=0,
        )

        is_broadcast = self.detector.is_broadcast_call(context)

        # High entropy should trigger broadcast detection
        self.assertTrue(is_broadcast)

    def test_not_broadcast_when_concentrated(self):
        """Test that concentrated probability is not broadcast."""
        # Concentrated probability = low entropy
        affect = np.array([0.3] + [0.0] * 15, dtype=np.float32)  # Low arousal
        probs = {"agent_002": 0.9, "agent_003": 0.1}  # Low entropy

        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=affect,
            call_directionality=CallDirectionality.UNICAST,
            receiver_probabilities=probs,
            nearby_count=2,  # Has nearby agents
            timestamp_ns=0,
        )

        is_broadcast = self.detector.is_broadcast_call(context)

        self.assertFalse(is_broadcast)

    def test_compute_entropy(self):
        """Test entropy computation."""
        # Uniform distribution has maximum entropy
        uniform_probs = {str(i): 0.25 for i in range(4)}
        entropy_uniform = self.detector._compute_entropy(uniform_probs)

        # Concentrated distribution has low entropy
        concentrated_probs = {"0": 0.9, "1": 0.1}
        entropy_concentrated = self.detector._compute_entropy(concentrated_probs)

        self.assertGreater(entropy_uniform, entropy_concentrated)

    def test_detect_no_nearby_agents(self):
        """Test broadcast detection when no agents are nearby."""
        context = Level25Context(
            emitter_id="agent_001",
            syntactic_token=0,
            affect_vector=np.zeros(16),
            call_directionality=CallDirectionality.BROADCAST,
            receiver_probabilities={},
            timestamp_ns=0,
            nearby_count=0,
        )

        is_broadcast = self.detector.is_broadcast_call(context)

        self.assertTrue(is_broadcast)


if __name__ == "__main__":
    unittest.main()
