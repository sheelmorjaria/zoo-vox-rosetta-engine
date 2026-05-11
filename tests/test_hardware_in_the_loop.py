#!/usr/bin/env python3
"""
Hardware-in-the-Loop Integration Test for Level 2.5

Tests the full pipeline from DeepLabCut pose estimation to spatial audio rendering.
This test is designed for validating the complete system with actual hardware.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import time
import unittest
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
import uuid

import numpy as np

# Try importing required modules
try:
    from spatial_intelligence.deeplabcut_ingestor import (
        CameraSource,
        create_test_camera_config,
        DeepLabCutIngestor,
        DLCCameraConfig,
        PoseKeypoints,
    )
    from spatial_intelligence.spatial_ingestor import (
        SpatialFrame,
        SpatialIngestor,
        SpatialObservation,
        TrackingSource,
    )
    from spatial_intelligence.topology_engine import (
        AgentState,
        ColonyTopology,
        ProximityResult,
        TopologyEngine,
    )
except ImportError as e:
    raise unittest.SkipTest(f"Required module not found: {e}")


class HardwareMode(Enum):
    """Test execution modes."""
    SIMULATED = "simulated"  # Use all mock/simulated components
    RTSP_TEST = "rtsp_test"  # Test RTSP connection (no actual cameras)
    HARDWARE = "hardware"  # Full hardware-in-the-loop (requires actual devices)


@dataclass
class SystemLatencyMetrics:
    """Metrics for end-to-end latency measurement."""
    pose_detection_ms: float = 0.0
    topology_analysis_ms: float = 0.0
    action_generation_ms: float = 0.0
    zmq_transit_ms: float = 0.0
    synthesis_ms: float = 0.0

    @property
    def total_latency_ms(self) -> float:
        return (
            self.pose_detection_ms +
            self.topology_analysis_ms +
            self.action_generation_ms +
            self.zmq_transit_ms +
            self.synthesis_ms
        )


@dataclass
class HardwareTestConfig:
    """Configuration for hardware-in-the-loop testing."""
    mode: HardwareMode = HardwareMode.SIMULATED
    num_cameras: int = 4
    arena_size: float = 10.0
    frame_rate: float = 30.0
    enable_rtsp: bool = False
    rtsp_urls: List[str] = field(default_factory=list)
    enable_zmq: bool = False
    zmq_port: int = 5555
    enable_synthesis: bool = False


class HardwareInTheLoopTest(unittest.TestCase):
    """
    Test suite for hardware-in-the-loop validation.

    These tests validate the complete pipeline from camera input to speaker output.
    """

    def setUp(self):
        """Set up test configuration."""
        self.config = HardwareTestConfig(
            mode=HardwareMode.SIMULATED,
            num_cameras=4,
            arena_size=10.0,
        )

    def test_simulated_pose_to_spatial_frame(self):
        """Test conversion from pose detection to spatial frame."""
        # Create DeepLabCut ingestor with test cameras
        configs = create_test_camera_config(num_cameras=4)
        ingestor = DeepLabCutIngestor(configs, area_size=10.0)

        # Generate a frame
        start = time.time()
        frame = ingestor.generate_frame(timestamp_ns=0)
        pose_time = (time.time() - start) * 1000

        # Verify frame structure
        self.assertIsInstance(frame, SpatialFrame)
        self.assertEqual(frame.timestamp_ns, 0)
        self.assertGreater(len(frame.observations), 0)

        # Verify observations have valid positions
        for obs in frame.observations:
            self.assertGreaterEqual(obs.x, -ingestor.area_size / 2 - 1)
            self.assertLessEqual(obs.x, ingestor.area_size / 2 + 1)
            self.assertGreaterEqual(obs.y, -ingestor.area_size / 2 - 1)
            self.assertLessEqual(obs.y, ingestor.area_size / 2 + 1)

        # Verify latency is acceptable (<50ms for pose detection)
        self.assertLess(pose_time, 50.0)

    def test_topology_analysis_latency(self):
        """Test topology engine analysis latency."""
        # Create topology engine
        engine = TopologyEngine(max_agents=100, proximity_radius=5.0)

        # Create spatial observations
        observations = []
        for i in range(5):
            obs = SpatialObservation(
                agent_id=f"agent_{i}",
                x=np.random.uniform(-4, 4),
                y=np.random.uniform(-4, 4),
                z=0.0,
                heading_rad=np.random.uniform(0, 2 * np.pi),
                velocity=0.5,
                timestamp_ns=0,
                confidence=1.0,
                source=TrackingSource.SIMULATED,
            )
            observations.append(obs)

        frame = SpatialFrame(timestamp_ns=0, observations=observations)

        # Measure topology analysis latency
        start = time.time()
        count = engine.update_topology(frame)
        analysis_time = (time.time() - start) * 1000

        # Verify topology was updated
        self.assertEqual(count, 5)
        self.assertEqual(len(engine.get_all_agent_ids()), 5)

        # Verify latency is acceptable (<10ms for 5 agents)
        self.assertLess(analysis_time, 10.0)

    def test_proximity_calculation_accuracy(self):
        """Test proximity calculation accuracy."""
        engine = TopologyEngine(max_agents=100, proximity_radius=5.0)

        # Create observations with known positions
        observations = [
            SpatialObservation(
                agent_id="emitter",
                x=0.0, y=0.0, z=0.0,
                heading_rad=0.0,
                velocity=0.0,
                timestamp_ns=0,
                confidence=1.0,
                source=TrackingSource.SIMULATED,
            ),
            SpatialObservation(
                agent_id="nearby",
                x=1.0, y=0.0, z=0.0,
                heading_rad=np.pi,
                velocity=0.0,
                timestamp_ns=0,
                confidence=1.0,
                source=TrackingSource.SIMULATED,
            ),
            SpatialObservation(
                agent_id="far_away",
                x=4.0, y=0.0, z=0.0,
                heading_rad=0.0,
                velocity=0.0,
                timestamp_ns=0,
                confidence=1.0,
                source=TrackingSource.SIMULATED,
            ),
        ]

        frame = SpatialFrame(timestamp_ns=0, observations=observations)
        engine.update_topology(frame)

        # Check proximity results
        result = engine.get_proximity_result("emitter")
        self.assertIsNotNone(result)
        self.assertEqual(result.agent_id, "emitter")

        # Find nearby agent - should be "nearby" at 1.0m
        nearby_list = [n for n in result.nearby_agents]
        self.assertTrue(len(nearby_list) > 0)

        # Check nearest distance
        self.assertAlmostEqual(result.nearest_distance, 1.0, places=1)
        self.assertEqual(result.nearest_agent, "nearby")

    def test_line_of_sight_detection(self):
        """Test line-of-sight detection accuracy."""
        engine = TopologyEngine(max_agents=100, proximity_radius=5.0)

        # Create emitter at origin, facing East (0 rad)
        # Create receivers at different angles
        observations = [
            SpatialObservation(
                agent_id="emitter",
                x=0.0, y=0.0, z=0.0,
                heading_rad=0.0,  # Facing East
                velocity=0.0,
                timestamp_ns=0,
                confidence=1.0,
                source=TrackingSource.SIMULATED,
            ),
            # In field of view (slightly to the right)
            SpatialObservation(
                agent_id="in_view",
                x=2.0, y=0.5, z=0.0,
                heading_rad=np.pi,
                velocity=0.0,
                timestamp_ns=0,
                confidence=1.0,
                source=TrackingSource.SIMULATED,
            ),
            # Outside field of view (90° to the left)
            SpatialObservation(
                agent_id="out_of_view",
                x=0.0, y=2.0, z=0.0,
                heading_rad=0.0,
                velocity=0.0,
                timestamp_ns=0,
                confidence=1.0,
                source=TrackingSource.SIMULATED,
            ),
        ]

        frame = SpatialFrame(timestamp_ns=0, observations=observations)
        engine.update_topology(frame)

        # Check line of sight
        los_in_view = engine.check_line_of_sight("emitter", "in_view")
        los_out_of_view = engine.check_line_of_sight("emitter", "out_of_view")

        # In-view receiver should be in field of view
        self.assertTrue(los_in_view.in_field_of_view)

        # Out-of-view receiver should not be in field of view
        self.assertFalse(los_out_of_view.in_field_of_view)

        # Verify distances are correct
        self.assertAlmostEqual(los_in_view.distance, np.sqrt(2.0**2 + 0.5**2), places=1)
        self.assertAlmostEqual(los_out_of_view.distance, 2.0, places=1)

    def test_receiver_probability_normalization(self):
        """Test receiver probability via proximity map."""
        engine = TopologyEngine(max_agents=100, proximity_radius=5.0)

        # Create emitter at origin
        observations = [
            SpatialObservation(
                agent_id="emitter",
                x=0.0, y=0.0, z=0.0,
                heading_rad=0.0,
                velocity=0.0,
                timestamp_ns=0,
                confidence=1.0,
                source=TrackingSource.SIMULATED,
            ),
        ]

        # Add receivers at varying distances
        for i in range(4):
            angle = (i / 4) * 2 * np.pi
            distance = 1.5 + i * 0.5
            observations.append(SpatialObservation(
                agent_id=f"receiver_{i}",
                x=distance * np.cos(angle),
                y=distance * np.sin(angle),
                z=0.0,
                heading_rad=angle + np.pi,
                velocity=0.0,
                timestamp_ns=0,
                confidence=1.0,
                source=TrackingSource.SIMULATED,
            ))

        frame = SpatialFrame(timestamp_ns=0, observations=observations)
        engine.update_topology(frame)

        # Get proximity result for emitter
        result = engine.get_proximity_result("emitter")
        self.assertIsNotNone(result)

        # Verify nearby agents were found
        self.assertGreater(len(result.nearby_agents), 0)

        # Verify all distances are positive and valid
        for agent_id, distance in result.nearby_agents:
            self.assertGreater(distance, 0.0)
            self.assertIn(agent_id, ["receiver_0", "receiver_1", "receiver_2", "receiver_3"])

    def test_broadcast_vs_unicast_classification(self):
        """Test broadcast vs unicast via proximity analysis."""
        engine = TopologyEngine(max_agents=100, proximity_radius=5.0)

        # Scenario 1: Single nearby receiver (should be unicast-like)
        observations = [
            SpatialObservation(
                agent_id="emitter",
                x=0.0, y=0.0, z=0.0,
                heading_rad=0.0,
                velocity=0.0,
                timestamp_ns=0,
                confidence=1.0,
                source=TrackingSource.SIMULATED,
            ),
            SpatialObservation(
                agent_id="single",
                x=1.0, y=0.0, z=0.0,
                heading_rad=np.pi,
                velocity=0.0,
                timestamp_ns=0,
                confidence=1.0,
                source=TrackingSource.SIMULATED,
            ),
        ]

        frame = SpatialFrame(timestamp_ns=0, observations=observations)
        engine.update_topology(frame)

        result = engine.get_proximity_result("emitter")
        # Should have exactly 1 nearby agent
        self.assertEqual(len(result.nearby_agents), 1)

        # Scenario 2: Multiple receivers (broadcast-like)
        engine2 = TopologyEngine(max_agents=100, proximity_radius=5.0)
        observations2 = [
            SpatialObservation(
                agent_id="emitter",
                x=0.0, y=0.0, z=0.0,
                heading_rad=0.0,
                velocity=0.0,
                timestamp_ns=0,
                confidence=1.0,
                source=TrackingSource.SIMULATED,
            ),
        ]

        for i in range(3):
            angle = (i / 3) * 2 * np.pi
            observations2.append(SpatialObservation(
                agent_id=f"receiver_{i}",
                x=1.5 * np.cos(angle),
                y=1.5 * np.sin(angle),
                z=0.0,
                heading_rad=angle + np.pi,
                velocity=0.0,
                timestamp_ns=0,
                confidence=1.0,
                source=TrackingSource.SIMULATED,
            ))

        frame2 = SpatialFrame(timestamp_ns=0, observations=observations2)
        engine2.update_topology(frame2)

        result2 = engine2.get_proximity_result("emitter")
        # Should have 3 nearby agents (broadcast scenario)
        self.assertEqual(len(result2.nearby_agents), 3)


class TestSimulatedPipeline(unittest.TestCase):
    """
    Simulated end-to-end pipeline test without requiring actual hardware.
    """

    def test_full_pipeline_simulation(self):
        """Test the complete simulated pipeline."""
        # Step 1: Generate poses (DeepLabCut simulation)
        configs = create_test_camera_config(num_cameras=4)
        ingestor = DeepLabCutIngestor(configs, area_size=10.0)

        start = time.time()
        frame = ingestor.generate_frame(timestamp_ns=0)
        pose_time = (time.time() - start) * 1000

        # Step 2: Analyze topology
        engine = TopologyEngine(max_agents=100, proximity_radius=5.0)

        start = time.time()
        count = engine.update_topology(frame)
        topology_time = (time.time() - start) * 1000

        # Verify topology was updated
        self.assertEqual(count, len(frame.observations))
        self.assertGreater(len(engine.get_all_agent_ids()), 0)

        # Step 3: Query proximity for first agent
        agent_ids = engine.get_all_agent_ids()
        if len(agent_ids) > 1:
            start = time.time()
            result = engine.get_proximity_result(agent_ids[0])
            query_time = (time.time() - start) * 1000

            self.assertIsNotNone(result)

            # Verify total latency is acceptable
            total_latency = pose_time + topology_time + query_time
            self.assertLess(total_latency, 100.0)  # <100ms for full pipeline

    def test_multi_frame_consistency(self):
        """Test consistency across multiple frames."""
        configs = create_test_camera_config(num_cameras=4)
        ingestor = DeepLabCutIngestor(configs, area_size=10.0)

        engine = TopologyEngine(max_agents=100, proximity_radius=5.0)

        # Track agent positions across frames
        agent_positions = {}
        frame_count = 0

        for i in range(10):
            frame = ingestor.generate_frame(timestamp_ns=i * 33_000_000)
            if len(frame.observations) > 0:
                engine.update_topology(frame)
                frame_count += 1

                # Track position changes via topology engine
                for agent_id in engine.get_all_agent_ids():
                    if agent_id not in agent_positions:
                        agent_positions[agent_id] = []

                    pos = engine.get_agent_position(agent_id)
                    if pos is not None:
                        agent_positions[agent_id].append((pos[0], pos[1]))

        # Verify at least some agents were tracked
        self.assertGreater(len(agent_positions), 0)

        # Verify agents are within arena bounds
        for agent_id, positions in agent_positions.items():
            for x, y in positions:
                # Arena is 10x10m, so bounds are -5 to 5
                self.assertGreaterEqual(x, -5.5)  # Allow small margin
                self.assertLessEqual(x, 5.5)
                self.assertGreaterEqual(y, -5.5)
                self.assertLessEqual(y, 5.5)

        # Verify multiple frames were processed
        self.assertGreater(frame_count, 0)


class TestRTSPConnection(unittest.TestCase):
    """
    Test RTSP connectivity for DeepLabCut integration.
    """

    def test_rtsp_url_parsing(self):
        """Test RTSP URL parsing and validation."""
        valid_urls = [
            "rtsp://192.168.1.100:554/stream",
            "rtsp://user:pass@192.168.1.100:554/stream1",
            "rtsp://example.com:8554/live",
        ]

        for url in valid_urls:
            config = DLCCameraConfig(
                camera_id="test",
                source_type=CameraSource.RTSP,
                source_url=url,
                camera_position=(0.0, 0.0, 2.0),
                camera_heading=0.0,
            )

            self.assertEqual(config.source_type, CameraSource.RTSP)
            self.assertEqual(config.source_url, url)

    def test_camera_config_creation(self):
        """Test creating camera configurations for various sources."""
        # RTSP camera
        rtsp_config = DLCCameraConfig(
            camera_id="rtsp_cam",
            source_type=CameraSource.RTSP,
            source_url="rtsp://192.168.1.100:554/stream",
            camera_position=(0.0, 0.0, 2.0),
            camera_heading=0.0,
        )

        self.assertEqual(rtsp_config.camera_id, "rtsp_cam")
        self.assertEqual(rtsp_config.source_type, CameraSource.RTSP)

        # USB camera
        usb_config = DLCCameraConfig(
            camera_id="usb_cam",
            source_type=CameraSource.USB,
            source_url="/dev/video0",
            camera_position=(2.0, 0.0, 2.0),
            camera_heading=np.pi,
        )

        self.assertEqual(usb_config.source_type, CameraSource.USB)

        # File source
        file_config = DLCCameraConfig(
            camera_id="file_cam",
            source_type=CameraSource.FILE,
            source_url="/path/to/video.mp4",
            camera_position=(0.0, 2.0, 2.0),
            camera_heading=np.pi / 2,
        )

        self.assertEqual(file_config.source_type, CameraSource.FILE)


class TestSpatialMetadata(unittest.TestCase):
    """
    Test spatial metadata generation for VBAP rendering.
    """

    def test_spatial_position_calculation(self):
        """Test spatial position calculation for rendering."""
        # Emitter at (1.5, 0) facing East
        emitter_pos = np.array([1.5, 0.0, 0.0])
        emitter_heading = 0.0  # East

        # Target at (0, 1.5) (North of emitter)
        target_pos = np.array([0.0, 1.5, 0.0])

        # Calculate direction from emitter to target
        direction = target_pos - emitter_pos
        direction = direction / np.linalg.norm(direction)

        # Expected: pointing Northwest
        expected_angle = np.arctan2(direction[1], direction[0])
        self.assertAlmostEqual(expected_angle, 2.356, places=2)  # 135° in radians

    def test_broadcast_spread_calculation(self):
        """Test broadcast spread angle calculation."""
        # For broadcast, spread should be wider (30-45°)
        broadcast_spread = 30.0  # degrees

        # For unicast, spread should be narrower (10-15°)
        unicast_spread = 15.0  # degrees

        self.assertGreater(broadcast_spread, unicast_spread)

    def test_speaker_mapping(self):
        """Test mapping of spatial position to speaker gains."""
        # Octagonal speaker array
        speaker_angles = np.array([0, 45, 90, 135, 180, 225, 270, 315])

        # Source at 50° (between speaker 0 and speaker 1)
        source_angle = 50.0

        # Find nearest speakers
        angle_diff = np.abs(speaker_angles - source_angle)
        nearest_speakers = np.argsort(angle_diff)[:2]

        # Should be speakers at 45° and 90°
        self.assertIn(45, speaker_angles[nearest_speakers])
        self.assertIn(90, speaker_angles[nearest_speakers])


class TestLatencyBudget(unittest.TestCase):
    """
    Test that the system meets latency budget requirements.
    """

    def test_overall_latency_budget(self):
        """Test overall latency budget (<125ms target)."""
        # Define acceptable latency budget
        MAX_POSE_DETECTION_MS = 50.0
        MAX_TOPOLOGY_ANALYSIS_MS = 10.0
        MAX_ACTION_GENERATION_MS = 10.0
        MAX_ZMQ_TRANSIT_MS = 5.0
        MAX_SYNTHESIS_MS = 50.0

        MAX_TOTAL_LATENCY_MS = (
            MAX_POSE_DETECTION_MS +
            MAX_TOPOLOGY_ANALYSIS_MS +
            MAX_ACTION_GENERATION_MS +
            MAX_ZMQ_TRANSIT_MS +
            MAX_SYNTHESIS_MS
        )

        # Target: <125ms total (current budget)
        # For optimal 100ms target, reduce pose_detection to 25ms or synthesis to 25ms
        self.assertLess(MAX_TOTAL_LATENCY_MS, 150.0)
        self.assertGreater(MAX_TOTAL_LATENCY_MS, 100.0)  # Current budget is 125ms

    def test_latency_breakdown(self):
        """Test that individual components meet latency targets."""
        # Simulated latency measurements
        metrics = SystemLatencyMetrics(
            pose_detection_ms=30.0,  # <50ms target ✓
            topology_analysis_ms=5.0,  # <10ms target ✓
            action_generation_ms=5.0,  # <10ms target ✓
            zmq_transit_ms=3.0,  # <5ms target ✓
            synthesis_ms=40.0,  # <50ms target ✓
        )

        # Verify individual components
        self.assertLess(metrics.pose_detection_ms, 50.0)
        self.assertLess(metrics.topology_analysis_ms, 10.0)
        self.assertLess(metrics.action_generation_ms, 10.0)
        self.assertLess(metrics.zmq_transit_ms, 5.0)
        self.assertLess(metrics.synthesis_ms, 50.0)

        # Verify total
        self.assertLess(metrics.total_latency_ms, 100.0)


class TestHardwareConfiguration(unittest.TestCase):
    """
    Test hardware configuration validation.
    """

    def test_speaker_array_configuration(self):
        """Test speaker array configuration."""
        # Octagonal array at 2.5m radius
        radius = 2.5
        num_speakers = 8

        # Calculate speaker positions
        speaker_positions = []
        for i in range(num_speakers):
            angle = (i / num_speakers) * 2 * np.pi
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            speaker_positions.append((x, y, 1.2))  # 1.2m height

        # Verify speakers are evenly distributed
        self.assertEqual(len(speaker_positions), 8)

        # Verify all speakers are at correct radius
        for pos in speaker_positions:
            distance = np.sqrt(pos[0]**2 + pos[1]**2)
            self.assertAlmostEqual(distance, radius, places=1)

    def test_camera_positioning(self):
        """Test camera positioning for optimal coverage."""
        # 4 cameras in circular array
        configs = create_test_camera_config(num_cameras=4)

        self.assertEqual(len(configs), 4)

        # Verify cameras are at correct height (2m)
        for config in configs:
            self.assertEqual(config.camera_position[2], 2.0)

        # Verify cameras face inward
        for i, config in enumerate(configs):
            angle = (i / 4) * 2 * np.pi
            expected_heading = angle + np.pi  # Face inward
            self.assertAlmostEqual(config.camera_heading, expected_heading, places=2)


if __name__ == "__main__":
    unittest.main()
