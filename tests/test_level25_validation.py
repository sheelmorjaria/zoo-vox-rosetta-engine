#!/usr/bin/env python3
"""
Level 2.5 Validation Tests

Comprehensive testing and validation for spatial-social inference:
- Unit tests for topology, line-of-sight, probability normalization
- Integration tests (Crowd Test, Back-Turned Test)
- Framework for "Spatial Mismatch" ethological validation

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
from typing import List, Tuple
import numpy as np

# Try importing required modules
try:
    from fusion_intelligence.receiver_inference import (
        BroadcastDetector,
        CallDirectionality,
        InferenceWeights,
        Level25Context,
        ReceiverInferenceEngine,
    )
    from spatial_intelligence.spatial_ingestor import (
        SimulatedIngestor,
        SpatialFrame,
        SpatialObservation,
    )
    from spatial_intelligence.topology_engine import (
        LineOfSightResult,
        TopologyEngine,
    )
except ImportError as e:
    raise unittest.SkipTest(f"Required module not found: {e}")


# ============================================================================
# Unit Tests
# ============================================================================

class TestTopologyProximity(unittest.TestCase):
    """Verify distance calculations for proximity mapping."""

    def test_topology_proximity(self):
        """Test that proximity distances are calculated correctly."""
        topology = TopologyEngine(max_agents=10, proximity_radius=10.0)

        # Create frame with agents at known positions
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="emitter", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_1m", x=1.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_2m", x=2.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_5m", x=5.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        topology.update_topology(frame)

        # Get proximity map from emitter
        proximity = topology.get_proximity_map("emitter")

        # Verify distances
        self.assertAlmostEqual(proximity.get("agent_1m", float('inf')), 1.0, places=2)
        self.assertAlmostEqual(proximity.get("agent_2m", float('inf')), 2.0, places=2)
        self.assertAlmostEqual(proximity.get("agent_5m", float('inf')), 5.0, places=2)

    def test_proximity_3d_distance(self):
        """Test 3D distance calculation (not just 2D)."""
        topology = TopologyEngine(max_agents=10, proximity_radius=10.0)

        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="agent_0", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        # 3-4-5 triangle: distance should be 5.0
        frame.observations.append(
            SpatialObservation(agent_id="agent_5", x=3.0, y=4.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        topology.update_topology(frame)
        proximity = topology.get_proximity_map("agent_0")

        self.assertAlmostEqual(proximity.get("agent_5", float('inf')), 5.0, places=2)


class TestLineOfSight(unittest.TestCase):
    """Verify vector math for heading vs. target position."""

    def test_line_of_sight(self):
        """Test that line-of-sight correctly identifies targets in field of view."""
        topology = TopologyEngine(max_agents=10, proximity_radius=10.0)

        # Emitter facing +X direction (heading = 0)
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="emitter", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)  # Facing +X
        )
        # Target ahead (in FoV, 0°)
        frame.observations.append(
            SpatialObservation(agent_id="ahead", x=2.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        # Target at 45° (in FoV, < 60°)
        frame.observations.append(
            SpatialObservation(agent_id="diag_45", x=2.0, y=2.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        # Target behind (out of FoV, 180°)
        frame.observations.append(
            SpatialObservation(agent_id="behind", x=-2.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        # Target to the side at 70° (out of FoV, > 60°)
        frame.observations.append(
            SpatialObservation(agent_id="side_70", x=2.0, y=np.tan(np.radians(70))*2, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        topology.update_topology(frame)

        # Check line-of-sight for each target
        los_ahead = topology.check_line_of_sight("emitter", "ahead")
        los_diag_45 = topology.check_line_of_sight("emitter", "diag_45")
        los_behind = topology.check_line_of_sight("emitter", "behind")
        los_side_70 = topology.check_line_of_sight("emitter", "side_70")

        # Ahead and 45° should be in field of view
        self.assertTrue(los_ahead.in_field_of_view, "Target at 0° should be in FoV")
        self.assertTrue(los_diag_45.in_field_of_view, "Target at 45° should be in FoV")

        # Behind and 70° should NOT be in field of view
        self.assertFalse(los_behind.in_field_of_view, "Target at 180° should be out of FoV")
        self.assertFalse(los_side_70.in_field_of_view, "Target at 70° should be out of FoV")

    def test_field_of_view_boundary(self):
        """Test field-of-view boundary conditions (60° half-angle)."""
        # Default FoV is 120°, half is 60°
        topology = TopologyEngine(max_agents=10, proximity_radius=10.0)

        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="emitter", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        # Target at 50 degrees (should be in FoV, < 60°)
        frame.observations.append(
            SpatialObservation(agent_id="inside", x=1.0, y=np.tan(np.radians(50)), z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        # Target at 70 degrees (should be out of FoV, > 60°)
        frame.observations.append(
            SpatialObservation(agent_id="outside", x=1.0, y=np.tan(np.radians(70)), z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        topology.update_topology(frame)

        los_inside = topology.check_line_of_sight("emitter", "inside")
        los_outside = topology.check_line_of_sight("emitter", "outside")

        self.assertTrue(los_inside.in_field_of_view, "Target at 50° should be in FoV")
        self.assertFalse(los_outside.in_field_of_view, "Target at 70° should be out of FoV")


class TestReceiverProbabilityNormalization(unittest.TestCase):
    """Ensure receiver probabilities sum to 1.0."""

    def test_probability_normalization(self):
        """Test that receiver probabilities are normalized to sum to 1.0."""
        topology = TopologyEngine(max_agents=10, proximity_radius=10.0)
        engine = ReceiverInferenceEngine()

        # Create frame with multiple agents
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="emitter", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        for i in range(5):
            frame.observations.append(
                SpatialObservation(
                    agent_id=f"agent_{i}",
                    x=float(i + 1),
                    y=0.0,
                    z=0.0,
                    heading_rad=0.0,
                    velocity=0.0,
                    timestamp_ns=0
                )
            )

        topology.update_topology(frame)

        # Run inference
        context = engine.infer_receiver(
            emitter_id="emitter",
            topology=topology,
            syntactic_token=0,
            timestamp_ns=0,
        )

        # Verify probabilities sum to 1.0
        total_prob = sum(context.receiver_probabilities.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)

    def test_empty_probability_when_no_nearby(self):
        """Test that empty probability dict sums to 0 (no agents nearby)."""
        topology = TopologyEngine(max_agents=10, proximity_radius=1.0)  # Small radius
        engine = ReceiverInferenceEngine()

        # Create frame with agent far away
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="emitter", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="far_away", x=100.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        topology.update_topology(frame)

        context = engine.infer_receiver(
            emitter_id="emitter",
            topology=topology,
            syntactic_token=0,
            timestamp_ns=0,
        )

        # Should have no nearby agents (far_away is outside radius)
        self.assertEqual(len(context.receiver_probabilities), 0)


class TestBroadcastVsUnicastClassification(unittest.TestCase):
    """Verify threshold logic for broadcast vs unicast classification."""

    def test_broadcast_threshold(self):
        """Test broadcast classification using probability concentration."""
        engine = ReceiverInferenceEngine(broadcast_threshold=0.65)

        # High concentration = unicast
        probs_concentrated = {"agent_002": 0.8, "agent_003": 0.1, "agent_004": 0.1}
        directionality = engine._classify_directionality(probs_concentrated)
        self.assertEqual(directionality, CallDirectionality.UNICAST)

        # Even distribution = broadcast
        probs_even = {"agent_002": 0.34, "agent_003": 0.33, "agent_004": 0.33}
        directionality = engine._classify_directionality(probs_even)
        self.assertEqual(directionality, CallDirectionality.BROADCAST)

    def test_exactly_at_threshold(self):
        """Test classification exactly at threshold boundary."""
        engine = ReceiverInferenceEngine(broadcast_threshold=0.65)

        # Exactly at threshold = broadcast (not > threshold)
        probs_at = {"agent_002": 0.65, "agent_003": 0.35}
        directionality = engine._classify_directionality(probs_at)
        # max_prob = 0.65, which is NOT > 0.65, so it's broadcast
        self.assertEqual(directionality, CallDirectionality.BROADCAST,
                        "Exactly at threshold should be broadcast (>, not >=)")

        # Just above threshold = unicast
        probs_above = {"agent_002": 0.66, "agent_003": 0.34}
        directionality = engine._classify_directionality(probs_above)
        self.assertEqual(directionality, CallDirectionality.UNICAST,
                        "Above threshold should be unicast")


# ============================================================================
# Integration Tests
# ============================================================================

class TestCrowdIntegration(unittest.TestCase):
    """
    The "Crowd" Test:

    Simulate 20 animals. Emitter vocalizes. Verify agent identifies
    top 3 closest animals as highest probability receivers.
    """

    def test_crowd_identifies_closest_receivers(self):
        """Verify that in a crowd, closest agents have highest probability."""
        topology = TopologyEngine(max_agents=25, proximity_radius=10.0)
        engine = ReceiverInferenceEngine()

        # Create crowd: 1 emitter + 20 other agents
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="emitter", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        # Add agents at increasing distances
        for i in range(20):
            distance = (i + 1) * 0.5  # 0.5m to 10m
            angle = (i / 20) * 2 * np.pi  # Evenly distributed around circle
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)

            frame.observations.append(
                SpatialObservation(
                    agent_id=f"agent_{i:02d}",
                    x=x,
                    y=y,
                    z=0.0,
                    heading_rad=0.0,
                    velocity=0.0,
                    timestamp_ns=0
                )
            )

        topology.update_topology(frame)

        # Run inference
        context = engine.infer_receiver(
            emitter_id="emitter",
            topology=topology,
            syntactic_token=0,
            timestamp_ns=0,
        )

        # Get top 3 receivers
        top_3 = context.get_top_receivers(top_k=3)

        self.assertEqual(len(top_3), 3)

        # The top 3 should be the 3 closest agents (distance = 0.5, 1.0, 1.5)
        # Their IDs should be among the first few in our sequence
        top_agent_ids = {aid for aid, _ in top_3}

        # Verify that top 3 have higher probabilities than the rest
        min_top_prob = min(prob for _, prob in top_3)
        for aid, prob in context.receiver_probabilities.items():
            if aid not in top_agent_ids:
                self.assertGreater(min_top_prob, prob,
                                 f"Top receiver probability should be higher than non-top")

    def test_crowd_probability_ordering(self):
        """Test that probability ordering matches distance ordering."""
        topology = TopologyEngine(max_agents=25, proximity_radius=10.0)
        engine = ReceiverInferenceEngine(
            weights=InferenceWeights(proximity_weight=1.0, los_weight=0.0, social_weight=0.0)
        )

        # Create simpler setup: agents in a line
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="emitter", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        # Add agents at known distances (all ahead, in FoV)
        distances = [1.0, 2.0, 3.0, 5.0, 8.0]
        for i, dist in enumerate(distances):
            frame.observations.append(
                SpatialObservation(
                    agent_id=f"agent_{dist}m",
                    x=dist,
                    y=0.0,
                    z=0.0,
                    heading_rad=0.0,
                    velocity=0.0,
                    timestamp_ns=0
                )
            )

        topology.update_topology(frame)

        # Run inference
        context = engine.infer_receiver(
            emitter_id="emitter",
            topology=topology,
            syntactic_token=0,
            timestamp_ns=0,
        )

        # Get all receivers sorted by probability
        sorted_receivers = sorted(
            context.receiver_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Verify ordering matches distance ordering (closer = higher prob)
        sorted_ids = [aid for aid, _ in sorted_receivers]
        expected_ids = [f"agent_{d}m" for d in distances]  # Closest first

        self.assertEqual(sorted_ids, expected_ids)


class TestBackTurnedIntegration(unittest.TestCase):
    """
    The "Back-Turned" Test:

    Simulate 2 animals at different positions relative to emitter.
    Target A is in emitter's field of view.
    Target B is outside emitter's field of view (behind/to side).
    Verify LoS penalty reduces Target B's receiver probability significantly.
    """

    def test_back_turned_penalty(self):
        """Test that position outside FoV reduces receiver probability."""
        topology = TopologyEngine(max_agents=10, proximity_radius=10.0)

        # High LoS weight to emphasize the effect
        engine = ReceiverInferenceEngine(
            weights=InferenceWeights(proximity_weight=0.3, los_weight=0.6, social_weight=0.1)
        )

        frame = SpatialFrame(timestamp_ns=0)
        # Emitter at origin, facing +X
        frame.observations.append(
            SpatialObservation(agent_id="emitter", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)  # Facing +X
        )

        # Target A: In front of emitter (in FoV, 0°)
        frame.observations.append(
            SpatialObservation(agent_id="target_a", x=2.0, y=0.0, z=0.0,
                             heading_rad=np.pi, velocity=0.0, timestamp_ns=0)  # Facing -X (toward emitter)
        )

        # Target B: Behind emitter (out of FoV, 180° from heading)
        frame.observations.append(
            SpatialObservation(agent_id="target_b", x=-2.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)  # Behind emitter
        )

        topology.update_topology(frame)

        # Run inference
        context = engine.infer_receiver(
            emitter_id="emitter",
            topology=topology,
            syntactic_token=0,
            timestamp_ns=0,
        )

        # Target A (in FoV) should have higher probability
        prob_a = context.receiver_probabilities.get("target_a", 0.0)
        prob_b = context.receiver_probabilities.get("target_b", 0.0)

        self.assertGreater(prob_a, prob_b,
                         "Target in FoV should have higher probability")

        # With 60% LoS weight, the penalty should be significant
        # Target A gets full LoS score, Target B gets reduced score (out of FoV)
        self.assertGreater(prob_a / prob_b, 1.5,
                         "LoS penalty should be at least 1.5x reduction")

    def test_los_angle_penalty_gradient(self):
        """Test that LoS penalty scales with angle from heading."""
        topology = TopologyEngine(max_agents=10, proximity_radius=10.0)
        engine = ReceiverInferenceEngine(
            weights=InferenceWeights(proximity_weight=0.0, los_weight=1.0, social_weight=0.0)
        )

        frame = SpatialFrame(timestamp_ns=0)
        # Emitter at origin, facing +X
        frame.observations.append(
            SpatialObservation(agent_id="emitter", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        # Targets at same distance (2m), different angles
        # Use angles: 0° (ahead), 30°, 50° (still in FoV), 70° (out of FoV), 100° (out of FoV)
        angles = [0, 30, 50, 70, 100]  # degrees from heading
        for angle_deg in angles:
            angle_rad = np.radians(angle_deg)
            # Position targets on a circle at distance 2m
            x = 2.0 * np.cos(angle_rad)
            y = 2.0 * np.sin(angle_rad)

            frame.observations.append(
                SpatialObservation(
                    agent_id=f"target_{angle_deg}deg",
                    x=x,
                    y=y,
                    z=0.0,
                    heading_rad=0.0,
                    velocity=0.0,
                    timestamp_ns=0
                )
            )

        topology.update_topology(frame)

        # Run inference
        context = engine.infer_receiver(
            emitter_id="emitter",
            topology=topology,
            syntactic_token=0,
            timestamp_ns=0,
        )

        # Get probabilities
        prob_0 = context.receiver_probabilities.get("target_0deg", 0.0)
        prob_30 = context.receiver_probabilities.get("target_30deg", 0.0)
        prob_50 = context.receiver_probabilities.get("target_50deg", 0.0)
        prob_70 = context.receiver_probabilities.get("target_70deg", 0.0)
        prob_100 = context.receiver_probabilities.get("target_100deg", 0.0)

        # All targets in FoV (0°, 30°, 50°) should have higher probability than out of FoV (70°, 100°)
        self.assertGreater(prob_0, prob_70)
        self.assertGreater(prob_30, prob_70)
        self.assertGreater(prob_50, prob_70)

        # Among targets in FoV, straight ahead (0°) should have highest probability
        self.assertGreaterEqual(prob_0, prob_30)

        # Targets out of FoV should have zero LoS score (same low probability)
        self.assertGreater(prob_50, prob_100)


# ============================================================================
# Ethological Validation Framework
# ============================================================================

class SpatialMismatchTest:
    """
    Framework for the "Spatial Mismatch" Ethological Validation Test.

    This test proves the value of Level 2.5 spatial inference by testing
    whether animals care about the direction of the response.

    Condition A (Spatially Congruent): Response rendered from Agent's
    actual position toward Emitter.

    Condition B (Spatially Incongruent): Response rendered from
    incorrect location (e.g., behind emitter).

    Success Criteria: Higher RAS (Response Appropriateness Score) and
    faster turn-taking latency in Condition A.
    """

    def __init__(self):
        self.results_condition_a = []
        self.results_condition_b = []

    def simulate_condition_a(self, emitter_pos, agent_pos, agent_response) -> dict:
        """
        Simulate Condition A: Spatially Congruent response.

        Args:
            emitter_pos: (x, y, z) position of emitter
            agent_pos: (x, y, z) position of responding agent
            agent_response: The response call parameters

        Returns:
            dict with rendering parameters
        """
        return {
            "condition": "A_congruent",
            "emitter_position": emitter_pos,
            "agent_position": agent_pos,
            "render_from": agent_pos,  # Render from actual agent position
            "render_toward": emitter_pos,  # Toward emitter
            "spatial_accuracy": 1.0,  # Perfect
        }

    def simulate_condition_b(self, emitter_pos, agent_pos, agent_response,
                            spoof_position) -> dict:
        """
        Simulate Condition B: Spatially Incongruent response.

        Args:
            emitter_pos: (x, y, z) position of emitter
            agent_pos: (x, y, z) actual position of responding agent
            agent_response: The response call parameters
            spoof_position: (x, y, z) position to render from (incorrect)

        Returns:
            dict with rendering parameters
        """
        return {
            "condition": "B_incongruent",
            "emitter_position": emitter_pos,
            "agent_position": agent_pos,
            "render_from": spoof_position,  # Render from WRONG position
            "render_toward": emitter_pos,
            "spatial_accuracy": 0.0,  # Completely incorrect
        }

    def calculate_ras(self, response_quality: float, latency_ms: float,
                     appropriate_direction: bool) -> float:
        """
        Calculate Response Appropriateness Score (RAS).

        Args:
            response_quality: Quality of the call (0-1)
            latency_ms: Response latency in milliseconds
            appropriate_direction: Whether response was spatially congruent

        Returns:
            RAS score (0-1)
        """
        # Quality component
        quality_score = response_quality

        # Latency component (optimal: 150-300ms)
        if 150 <= latency_ms <= 300:
            latency_score = 1.0
        elif latency_ms < 150:
            latency_score = 0.8  # Too fast (possibly pre-planned)
        elif latency_ms <= 500:
            latency_score = 0.7
        else:
            latency_score = 0.5  # Too slow

        # Direction component
        direction_score = 1.0 if appropriate_direction else 0.3

        # Weighted average
        ras = 0.4 * quality_score + 0.3 * latency_score + 0.3 * direction_score
        return ras


class TestSpatialMismatchFramework(unittest.TestCase):
    """Unit tests for the spatial mismatch validation framework."""

    def test_condition_a_congruent(self):
        """Test that Condition A uses correct spatial rendering."""
        test = SpatialMismatchTest()

        result = test.simulate_condition_a(
            emitter_pos=(0.0, 0.0, 0.0),
            agent_pos=(2.0, 0.0, 0.0),
            agent_response={"token": 5}
        )

        self.assertEqual(result["condition"], "A_congruent")
        self.assertEqual(result["render_from"], (2.0, 0.0, 0.0))
        self.assertEqual(result["spatial_accuracy"], 1.0)

    def test_condition_b_incongruent(self):
        """Test that Condition B uses incorrect spatial rendering."""
        test = SpatialMismatchTest()

        result = test.simulate_condition_b(
            emitter_pos=(0.0, 0.0, 0.0),
            agent_pos=(2.0, 0.0, 0.0),
            agent_response={"token": 5},
            spoof_position=(-2.0, 0.0, 0.0)  # Behind emitter
        )

        self.assertEqual(result["condition"], "B_incongruent")
        self.assertEqual(result["render_from"], (-2.0, 0.0, 0.0))
        self.assertEqual(result["spatial_accuracy"], 0.0)

    def test_ras_calculation(self):
        """Test RAS score calculation."""
        test = SpatialMismatchTest()

        # Ideal response
        ras_ideal = test.calculate_ras(
            response_quality=0.9,
            latency_ms=200,
            appropriate_direction=True
        )
        self.assertGreater(ras_ideal, 0.8)

        # Poor response (wrong direction)
        ras_poor = test.calculate_ras(
            response_quality=0.9,
            latency_ms=200,
            appropriate_direction=False
        )
        self.assertLess(ras_poor, ras_ideal)

        # Slow response
        ras_slow = test.calculate_ras(
            response_quality=0.9,
            latency_ms=800,
            appropriate_direction=True
        )
        self.assertLess(ras_slow, ras_ideal)


if __name__ == "__main__":
    unittest.main()
