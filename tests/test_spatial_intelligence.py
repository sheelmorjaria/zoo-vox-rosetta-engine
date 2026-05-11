#!/usr/bin/env python3
"""
Tests for Spatial Intelligence Module (Level 2.5)

Tests spatial ingestion, topology engine, and colony analysis.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Try importing required modules
try:
    from spatial_intelligence.spatial_ingestor import (
        DeepLabCutIngestor,
        SimulatedIngestor,
        SpatialFrame,
        SpatialIngestor,
        SpatialObservation,
        TrackingSource,
    )
    from spatial_intelligence.topology_engine import (
        AgentState,
        ColonyTopology,
        LineOfSightResult,
        ProximityResult,
        TopologyEngine,
    )
except ImportError as e:
    raise unittest.SkipTest(f"Required module not found: {e}")


class TestSpatialObservation(unittest.TestCase):
    """Test the SpatialObservation dataclass."""

    def test_create_observation(self):
        """Test creating a spatial observation."""
        obs = SpatialObservation(
            agent_id="agent_001",
            x=1.0,
            y=2.0,
            z=0.0,
            heading_rad=0.0,
            velocity=0.5,
            timestamp_ns=0,
        )

        self.assertEqual(obs.agent_id, "agent_001")
        self.assertEqual(obs.x, 1.0)
        self.assertEqual(obs.y, 2.0)

    def test_to_array(self):
        """Test conversion to numpy array."""
        obs = SpatialObservation(
            agent_id="agent_001",
            x=1.0,
            y=2.0,
            z=3.0,
            heading_rad=0.0,
            velocity=0.5,
            timestamp_ns=0,
        )

        arr = obs.to_array()

        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))

    def test_distance_to(self):
        """Test distance calculation between observations."""
        obs1 = SpatialObservation(
            agent_id="agent_001", x=0.0, y=0.0, z=0.0,
            heading_rad=0.0, velocity=0.0, timestamp_ns=0
        )
        obs2 = SpatialObservation(
            agent_id="agent_002", x=3.0, y=4.0, z=0.0,
            heading_rad=0.0, velocity=0.0, timestamp_ns=0
        )

        distance = obs1.distance_to(obs2)

        # Should be 5.0 (3-4-5 triangle)
        self.assertAlmostEqual(distance, 5.0, places=5)

    def test_angle_to_same_position(self):
        """Test angle calculation when positions are the same."""
        obs1 = SpatialObservation(
            agent_id="agent_001", x=0.0, y=0.0, z=0.0,
            heading_rad=0.0, velocity=0.0, timestamp_ns=0
        )
        obs2 = SpatialObservation(
            agent_id="agent_002", x=0.0, y=0.0, z=0.0,
            heading_rad=0.0, velocity=0.0, timestamp_ns=0
        )

        angle = obs1.angle_to(obs2)

        self.assertAlmostEqual(angle, 0.0, places=5)

    def test_angle_to_ahead(self):
        """Test angle when target is directly ahead."""
        obs1 = SpatialObservation(
            agent_id="agent_001", x=0.0, y=0.0, z=0.0,
            heading_rad=0.0, velocity=0.0, timestamp_ns=0  # Facing +X
        )
        obs2 = SpatialObservation(
            agent_id="agent_002", x=1.0, y=0.0, z=0.0,
            heading_rad=0.0, velocity=0.0, timestamp_ns=0
        )

        angle = obs1.angle_to(obs2)

        # Should be 0 (directly ahead)
        self.assertAlmostEqual(angle, 0.0, places=5)

    def test_angle_to_behind(self):
        """Test angle when target is directly behind."""
        obs1 = SpatialObservation(
            agent_id="agent_001", x=0.0, y=0.0, z=0.0,
            heading_rad=0.0, velocity=0.0, timestamp_ns=0  # Facing +X
        )
        obs2 = SpatialObservation(
            agent_id="agent_002", x=-1.0, y=0.0, z=0.0,
            heading_rad=0.0, velocity=0.0, timestamp_ns=0
        )

        angle = obs1.angle_to(obs2)

        # Should be pi (directly behind)
        self.assertAlmostEqual(angle, np.pi, places=5)


class TestSpatialFrame(unittest.TestCase):
    """Test the SpatialFrame container."""

    def test_create_frame(self):
        """Test creating a spatial frame."""
        frame = SpatialFrame(timestamp_ns=12345)

        self.assertEqual(frame.timestamp_ns, 12345)
        self.assertEqual(len(frame.observations), 0)

    def test_add_observations(self):
        """Test adding observations to a frame."""
        frame = SpatialFrame(timestamp_ns=0)

        obs1 = SpatialObservation(
            agent_id="agent_001", x=0.0, y=0.0, z=0.0,
            heading_rad=0.0, velocity=0.0, timestamp_ns=0
        )
        obs2 = SpatialObservation(
            agent_id="agent_002", x=1.0, y=0.0, z=0.0,
            heading_rad=0.0, velocity=0.0, timestamp_ns=0
        )

        frame.observations.extend([obs1, obs2])

        self.assertEqual(len(frame.observations), 2)

    def test_get_observation(self):
        """Test getting observation by agent ID."""
        frame = SpatialFrame(timestamp_ns=0)

        obs = SpatialObservation(
            agent_id="agent_001", x=1.0, y=2.0, z=0.0,
            heading_rad=0.0, velocity=0.0, timestamp_ns=0
        )
        frame.observations.append(obs)

        retrieved = frame.get_observation("agent_001")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.agent_id, "agent_001")
        self.assertEqual(retrieved.x, 1.0)

    def test_get_observation_not_found(self):
        """Test getting non-existent observation."""
        frame = SpatialFrame(timestamp_ns=0)

        retrieved = frame.get_observation("agent_999")

        self.assertIsNone(retrieved)

    def test_agent_ids(self):
        """Test getting all agent IDs."""
        frame = SpatialFrame(timestamp_ns=0)

        frame.observations.append(
            SpatialObservation(agent_id="agent_001", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_002", x=1.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        ids = frame.agent_ids()

        self.assertEqual(len(ids), 2)
        self.assertIn("agent_001", ids)
        self.assertIn("agent_002", ids)


class TestSpatialIngestor(unittest.TestCase):
    """Test the SpatialIngestor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.ingestor = SpatialIngestor()

    def test_process_empty_frame(self):
        """Test processing an empty frame."""
        raw_frame = {
            "timestamp_ns": 0,
            "source": "manual",
            "agents": [],
        }

        frame = self.ingestor.process_frame(raw_frame)

        self.assertEqual(frame.timestamp_ns, 0)
        self.assertEqual(len(frame.observations), 0)

    def test_process_frame_with_agents(self):
        """Test processing a frame with agent data."""
        raw_frame = {
            "timestamp_ns": 12345,
            "source": "manual",
            "agents": [
                {
                    "agent_id": "agent_001",
                    "x": 1.0,
                    "y": 2.0,
                    "z": 0.0,
                    "heading": 0.0,
                    "velocity": 0.5,
                    "confidence": 1.0,
                },
            ],
        }

        frame = self.ingestor.process_frame(raw_frame)

        self.assertEqual(len(frame.observations), 1)
        self.assertEqual(frame.observations[0].agent_id, "agent_001")
        self.assertEqual(frame.observations[0].x, 1.0)

    def test_coordinate_scaling(self):
        """Test coordinate scaling."""
        ingestor = SpatialIngestor(coordinate_scale=2.0)

        raw_frame = {
            "timestamp_ns": 0,
            "source": "manual",
            "agents": [
                {"agent_id": "agent_001", "x": 1.0, "y": 1.0, "z": 0.0,
                 "heading": 0.0, "velocity": 0.0},
            ],
        }

        frame = ingestor.process_frame(raw_frame)

        # Coordinates should be scaled by 2.0
        self.assertEqual(frame.observations[0].x, 2.0)
        self.assertEqual(frame.observations[0].y, 2.0)

    def test_get_latest_observation(self):
        """Test getting latest observation for an agent."""
        raw_frame = {
            "timestamp_ns": 0,
            "source": "manual",
            "agents": [
                {"agent_id": "agent_001", "x": 1.0, "y": 0.0, "z": 0.0,
                 "heading": 0.0, "velocity": 0.0},
            ],
        }

        self.ingestor.process_frame(raw_frame)

        obs = self.ingestor.get_latest_observation("agent_001")

        self.assertIsNotNone(obs)
        self.assertEqual(obs.agent_id, "agent_001")

    def test_prune_old_observations(self):
        """Test pruning old observations."""
        # Create ingestor with short max age for testing
        ingestor = SpatialIngestor(max_age_ms=50.0)

        # Add an observation
        raw_frame = {
            "timestamp_ns": 0,
            "source": "manual",
            "agents": [
                {"agent_id": "agent_001", "x": 0.0, "y": 0.0, "z": 0.0,
                 "heading": 0.0, "velocity": 0.0},
            ],
        }
        ingestor.process_frame(raw_frame)

        # Prune with cutoff time after the observation's max age (100ms > 50ms max_age)
        removed = ingestor.prune_old_observations(
            current_timestamp_ns=100_000_000,  # 100ms later
        )

        # Observation should be removed (older than 50ms max_age)
        self.assertEqual(removed, 1)
        self.assertIsNone(ingestor.get_latest_observation("agent_001"))


class TestSimulatedIngestor(unittest.TestCase):
    """Test the SimulatedIngestor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.ingestor = SimulatedIngestor(num_agents=5, area_size=10.0)

    def test_generate_frame(self):
        """Test generating a simulated frame."""
        frame = self.ingestor.generate_frame(timestamp_ns=0)

        self.assertEqual(len(frame.observations), 5)
        self.assertEqual(frame.timestamp_ns, 0)

    def test_agents_have_valid_positions(self):
        """Test that all agents have valid positions."""
        frame = self.ingestor.generate_frame(timestamp_ns=0)

        for obs in frame.observations:
            self.assertTrue(np.isfinite(obs.x))
            self.assertTrue(np.isfinite(obs.y))
            self.assertTrue(np.isfinite(obs.z))
            self.assertGreaterEqual(obs.velocity, 0.0)

    def test_agents_stay_in_bounds(self):
        """Test that agents stay within the defined area."""
        area_size = 10.0
        ingestor = SimulatedIngestor(num_agents=5, area_size=area_size)
        half_size = area_size / 2.0

        # Generate many frames
        for i in range(100):
            frame = ingestor.generate_frame(timestamp_ns=i * 33_000_000)

            for obs in frame.observations:
                # Agents should stay within bounds (with wrapping)
                self.assertGreaterEqual(obs.x, -half_size)
                self.assertLess(obs.x, half_size)
                self.assertGreaterEqual(obs.y, -half_size)
                self.assertLess(obs.y, half_size)

    def test_agent_ids_are_consistent(self):
        """Test that agent IDs are consistent across frames."""
        frame1 = self.ingestor.generate_frame(timestamp_ns=0)
        frame2 = self.ingestor.generate_frame(timestamp_ns=33_000_000)

        ids1 = set(frame1.agent_ids())
        ids2 = set(frame2.agent_ids())

        # Same agents should be in both frames
        self.assertEqual(ids1, ids2)


class TestTopologyEngine(unittest.TestCase):
    """Test the TopologyEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.topology = TopologyEngine(max_agents=10, proximity_radius=5.0)
        self.ingestor = SimulatedIngestor(num_agents=5, area_size=10.0)

    def test_update_topology(self):
        """Test updating topology from a frame."""
        frame = self.ingestor.generate_frame(timestamp_ns=0)

        updated = self.topology.update_topology(frame)

        self.assertEqual(updated, 5)
        self.assertEqual(len(self.topology.get_all_agent_ids()), 5)

    def test_get_proximity_map(self):
        """Test getting proximity map for an agent."""
        # Create frame with known positions
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="agent_001", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_002", x=1.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_003", x=10.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        self.topology.update_topology(frame)

        # agent_002 should be nearby (1m away)
        # agent_003 should not be nearby (10m away)
        nearby = self.topology.get_proximity_map("agent_001", max_radius=5.0)

        self.assertIn("agent_002", nearby)
        self.assertNotIn("agent_003", nearby)

    def test_check_line_of_sight(self):
        """Test line-of-sight checking."""
        # Create frame with agent looking at another
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="agent_001", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)  # Facing +X
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_002", x=1.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)  # Directly ahead
        )

        self.topology.update_topology(frame)

        los = self.topology.check_line_of_sight("agent_001", "agent_002")

        self.assertTrue(los.in_field_of_view)
        self.assertTrue(los.has_los)

    def test_line_of_sight_behind(self):
        """Test line-of-sight when target is behind."""
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="agent_001", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)  # Facing +X
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_002", x=-1.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)  # Behind
        )

        self.topology.update_topology(frame)

        los = self.topology.check_line_of_sight("agent_001", "agent_002")

        self.assertFalse(los.in_field_of_view)
        self.assertFalse(los.has_los)

    def test_get_proximity_result(self):
        """Test getting detailed proximity result."""
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="agent_001", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_002", x=1.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_003", x=2.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        self.topology.update_topology(frame)

        result = self.topology.get_proximity_result("agent_001")

        self.assertIsNotNone(result)
        self.assertEqual(result.agent_id, "agent_001")
        self.assertEqual(result.nearest_agent, "agent_002")
        self.assertAlmostEqual(result.nearest_distance, 1.0)
        self.assertEqual(len(result.nearby_agents), 2)

    def test_get_colony_center(self):
        """Test calculating colony center."""
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="agent_001", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_002", x=2.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        self.topology.update_topology(frame)

        center = self.topology.get_colony_center()

        # Center should be at (1, 0, 0)
        np.testing.assert_array_almost_equal(center, np.array([1.0, 0.0, 0.0]))

    def test_get_colony_spread(self):
        """Test calculating colony spread."""
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="agent_001", x=-2.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_002", x=2.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_003", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        self.topology.update_topology(frame)

        spread = self.topology.get_colony_spread()

        # Center is (0, 0, 0). Distances are [2.0, 2.0, 0.0]
        # Std of [2.0, 2.0, 0.0] ≈ 0.943
        self.assertAlmostEqual(spread, 0.943, places=2)

    def test_remove_stale_agents(self):
        """Test removing stale agents."""
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="agent_001", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        self.topology.update_topology(frame)

        # Remove agents older than 100ms
        removed = self.topology.remove_stale_agents(
            current_timestamp_ns=200_000_000,
            max_age_ms=100
        )

        self.assertEqual(removed, 1)
        self.assertEqual(len(self.topology.get_all_agent_ids()), 0)

    def test_get_topology_summary(self):
        """Test getting topology summary."""
        frame = self.ingestor.generate_frame(timestamp_ns=0)
        self.topology.update_topology(frame)

        summary = self.topology.get_topology_summary()

        self.assertIn("num_agents", summary)
        self.assertIn("colony_center", summary)
        self.assertIn("colony_spread", summary)
        self.assertEqual(summary["num_agents"], 5)


class TestColonyTopology(unittest.TestCase):
    """Test the ColonyTopology helper class."""

    def setUp(self):
        """Set up test fixtures."""
        self.topology = TopologyEngine(max_agents=10, proximity_radius=5.0)
        self.colony = ColonyTopology(self.topology)

    def test_find_clusters(self):
        """Test finding spatial clusters."""
        # Create two clusters of agents
        frame = SpatialFrame(timestamp_ns=0)

        # Cluster 1: agents near (0, 0)
        for i in range(3):
            frame.observations.append(
                SpatialObservation(
                    agent_id=f"cluster1_{i}",
                    x=np.random.uniform(-0.5, 0.5),
                    y=np.random.uniform(-0.5, 0.5),
                    z=0.0,
                    heading_rad=0.0,
                    velocity=0.0,
                    timestamp_ns=0
                )
            )

        # Cluster 2: agents near (10, 10)
        for i in range(3):
            frame.observations.append(
                SpatialObservation(
                    agent_id=f"cluster2_{i}",
                    x=10.0 + np.random.uniform(-0.5, 0.5),
                    y=10.0 + np.random.uniform(-0.5, 0.5),
                    z=0.0,
                    heading_rad=0.0,
                    velocity=0.0,
                    timestamp_ns=0
                )
            )

        self.topology.update_topology(frame)

        clusters = self.colony.find_clusters(cluster_distance=2.0, min_cluster_size=2)

        self.assertEqual(len(clusters), 2)

    def test_find_isolated_agents(self):
        """Test finding isolated agents."""
        frame = SpatialFrame(timestamp_ns=0)

        # Clustered agent
        frame.observations.append(
            SpatialObservation(agent_id="agent_001", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_002", x=1.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        # Isolated agent (10m away)
        frame.observations.append(
            SpatialObservation(agent_id="agent_isolated", x=10.0, y=10.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        self.topology.update_topology(frame)

        isolated = self.colony.find_isolated_agents(max_distance=3.0)

        self.assertIn("agent_isolated", isolated)
        self.assertNotIn("agent_001", isolated)
        self.assertNotIn("agent_002", isolated)

    def test_get_social_graph(self):
        """Test building social graph from proximity."""
        frame = SpatialFrame(timestamp_ns=0)
        frame.observations.append(
            SpatialObservation(agent_id="agent_001", x=0.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_002", x=1.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )
        frame.observations.append(
            SpatialObservation(agent_id="agent_003", x=2.0, y=0.0, z=0.0,
                             heading_rad=0.0, velocity=0.0, timestamp_ns=0)
        )

        self.topology.update_topology(frame)

        graph = self.colony.get_social_graph()

        self.assertIn("agent_001", graph)
        self.assertIn("agent_002", graph)
        # agent_002 should be connected to both 001 and 003
        self.assertIn("agent_001", graph["agent_002"])
        self.assertIn("agent_003", graph["agent_002"])


if __name__ == "__main__":
    unittest.main()
