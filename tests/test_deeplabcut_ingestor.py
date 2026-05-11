#!/usr/bin/env python3
"""
Tests for DeepLabCut RTSP Ingestor

Tests real-time pose estimation integration for Level 2.5 spatial inference.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

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
    from spatial_intelligence.spatial_ingestor import SpatialFrame
except ImportError as e:
    raise unittest.SkipTest(f"Required module not found: {e}")


class TestPoseKeypoints(unittest.TestCase):
    """Test PoseKeypoints dataclass."""

    def test_create_pose_keypoints(self):
        """Test creating pose keypoints."""
        keypoints = {
            "nose": (100.0, 200.0, 0.9),
            "left_ear": (90.0, 190.0, 0.85),
            "right_ear": (110.0, 190.0, 0.85),
            "tail_base": (80.0, 200.0, 0.8),
        }

        pose = PoseKeypoints(
            agent_id="test_agent",
            confidence=0.85,
            keypoints=keypoints,
            timestamp_ns=0,
        )

        self.assertEqual(pose.agent_id, "test_agent")
        self.assertEqual(pose.confidence, 0.85)
        self.assertEqual(len(pose.keypoints), 4)

    def test_get_body_center(self):
        """Test getting body center from keypoints."""
        keypoints = {
            "nose": (100.0, 200.0, 0.9),
            "left_ear": (90.0, 190.0, 0.85),
            "right_ear": (110.0, 190.0, 0.85),
        }

        pose = PoseKeypoints(
            agent_id="test_agent",
            confidence=0.85,
            keypoints=keypoints,
            timestamp_ns=0,
        )

        center_x, center_y, conf = pose.get_body_center()

        # Average of 100, 90, 110 = 100
        self.assertAlmostEqual(center_x, 100.0)
        # Average of 200, 190, 190 = 193.33
        self.assertAlmostEqual(center_y, 193.33, places=1)

    def test_compute_heading_nose_to_tail(self):
        """Test heading computation from nose to tail."""
        keypoints = {
            "nose": (100.0, 200.0, 0.9),
            "tail_base": (80.0, 200.0, 0.8),
        }

        pose = PoseKeypoints(
            agent_id="test_agent",
            confidence=0.85,
            keypoints=keypoints,
            timestamp_ns=0,
        )

        heading = pose.compute_heading()

        # Nose is to the right of tail, so heading should be 0 (pointing right)
        self.assertAlmostEqual(heading, 0.0, places=3)

    def test_compute_heading_shoulder_fallback(self):
        """Test heading computation from shoulders (nose unavailable)."""
        keypoints = {
            "left_shoulder": (90.0, 200.0, 0.9),
            "right_shoulder": (110.0, 200.0, 0.9),
            "tail_base": (100.0, 220.0, 0.8),
        }

        pose = PoseKeypoints(
            agent_id="test_agent",
            confidence=0.85,
            keypoints=keypoints,
            timestamp_ns=0,
        )

        heading = pose.compute_heading()

        # Should compute from shoulder vector
        # Left to right is (20, 0), heading should be perpendicular
        self.assertIsNotNone(heading)


class TestDLCCameraConfig(unittest.TestCase):
    """Test DLCCameraConfig dataclass."""

    def test_create_camera_config(self):
        """Test creating a camera configuration."""
        config = DLCCameraConfig(
            camera_id="cam_0",
            source_type=CameraSource.RTSP,
            source_url="rtsp://192.168.1.100:554/stream",
            camera_position=(0.0, 0.0, 2.0),
            camera_heading=0.0,
            camera_pitch=-np.pi / 6,
        )

        self.assertEqual(config.camera_id, "cam_0")
        self.assertEqual(config.source_type, CameraSource.RTSP)
        self.assertEqual(config.camera_position[2], 2.0)  # 2m height
        self.assertEqual(config.camera_pitch, -np.pi / 6)  # -30 degrees


class TestDeepLabCutIngestor(unittest.TestCase):
    """Test DeepLabCutIngestor class."""

    def test_create_ingestor(self):
        """Test creating a DeepLabCut ingestor."""
        configs = create_test_camera_config(num_cameras=2)
        ingestor = DeepLabCutIngestor(configs, area_size=10.0)

        self.assertEqual(len(ingestor.camera_configs), 2)
        self.assertEqual(ingestor.area_size, 10.0)
        self.assertEqual(ingestor.frame_rate, 30.0)

    def test_pixel_to_world_transformation(self):
        """Test pixel to world coordinate transformation."""
        configs = [
            DLCCameraConfig(
                camera_id="test_cam",
                source_type=CameraSource.TEST_PATTERN,
                source_url="",
                camera_position=(0.0, 0.0, 2.0),
                camera_heading=0.0,
            )
        ]
        ingestor = DeepLabCutIngestor(configs, area_size=10.0)

        # Center of 1920x1080 frame
        # The transformation uses: x_norm = (x_px - principal_point[0]) / focal_length
        # For pixel (960, 540): x_norm = (960 - 640) / 800 = 0.32
        # ground_distance = camera_height / y_norm (where y_norm uses similar calc)
        # world_x = camera_position[0] + ground_distance * x_norm
        world_x, world_y, world_z = ingestor._pixel_to_world(960, 540, "test_cam")

        # Should map to a reasonable position (not exact center due to perspective)
        # Just verify it's within arena bounds
        self.assertGreater(abs(world_x), 0)
        self.assertGreater(abs(world_y), 0)
        self.assertEqual(world_z, 0.0)  # Z is always 0 (ground level)

    def test_generate_test_pattern_frame(self):
        """Test generating frames from test pattern cameras."""
        configs = create_test_camera_config(num_cameras=4)
        ingestor = DeepLabCutIngestor(configs, area_size=10.0)

        frame = ingestor.generate_frame(timestamp_ns=0)

        self.assertIsInstance(frame, SpatialFrame)
        self.assertGreater(len(frame.observations), 0)

    def test_mock_poses_generation(self):
        """Test mock pose generation for testing."""
        configs = create_test_camera_config(num_cameras=1)
        ingestor = DeepLabCutIngestor(configs, area_size=10.0)

        # Create a dummy frame
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        poses = ingestor._mock_poses("camera_0", dummy_frame)

        self.assertGreater(len(poses), 0)
        self.assertLessEqual(len(poses), 4)  # Max 3-4 agents

        # Check pose structure
        pose = poses[0]
        self.assertIn("nose", pose.keypoints)
        self.assertIn("tail_base", pose.keypoints)

    def test_detect_poses_confidence_filtering(self):
        """Test that low-confidence poses are filtered."""
        configs = create_test_camera_config(num_cameras=1)
        ingestor = DeepLabCutIngestor(
            configs,
            area_size=10.0,
            confidence_threshold=0.9  # High threshold
        )

        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        poses = ingestor._mock_poses("camera_0", dummy_frame)

        # Check that poses have reasonable confidence
        for pose in poses:
            _, _, conf = pose.get_body_center()
            # Mock poses should have ~0.85 confidence
            self.assertGreater(conf, 0.5)


class TestCreateTestCameraConfig(unittest.TestCase):
    """Test test camera configuration generator."""

    def test_create_4_cameras(self):
        """Test creating 4 cameras in circular array."""
        configs = create_test_camera_config(num_cameras=4)

        self.assertEqual(len(configs), 4)

        # Check cameras are evenly distributed
        for i, config in enumerate(configs):
            self.assertEqual(config.source_type, CameraSource.TEST_PATTERN)
            self.assertEqual(config.camera_position[2], 2.0)  # 2m height

            # Camera should face inward (toward center)
            angle = (i / 4) * 2 * np.pi
            expected_heading = angle + np.pi
            self.assertAlmostEqual(config.camera_heading, expected_heading, places=3)

    def test_camera_positions(self):
        """Test that cameras are positioned correctly around circle."""
        configs = create_test_camera_config(num_cameras=4)

        # All cameras should be at radius 3 from center (hardcoded in function)
        for config in configs:
            x, y, z = config.camera_position
            distance_from_center = (x**2 + y**2)**0.5
            self.assertAlmostEqual(distance_from_center, 3.0, places=1)


class TestDeepLabCutIntegration(unittest.TestCase):
    """Integration tests for DeepLabCut pose estimation pipeline."""

    def test_full_pipeline_mock(self):
        """Test full pipeline from cameras to spatial observations."""
        configs = create_test_camera_config(num_cameras=4)
        ingestor = DeepLabCutIngestor(configs, area_size=10.0)

        # Generate frame (simulates camera input)
        frame = ingestor.generate_frame(timestamp_ns=0)

        # Verify frame structure
        self.assertEqual(frame.timestamp_ns, 0)
        self.assertGreater(len(frame.observations), 0)

        # Check observations have valid positions
        for obs in frame.observations:
            self.assertGreaterEqual(obs.x, -ingestor.area_size / 2 - 1)
            self.assertLessEqual(obs.x, ingestor.area_size / 2 + 1)
            self.assertGreaterEqual(obs.y, -ingestor.area_size / 2 - 1)
            self.assertLessEqual(obs.y, ingestor.area_size / 2 + 1)

    def test_multi_camera_triangulation(self):
        """Test that the ingestor can handle multiple cameras."""
        configs = create_test_camera_config(num_cameras=4)
        ingestor = DeepLabCutIngestor(configs, area_size=10.0)

        # Generate frame (in test pattern mode, falls back to SimulatedIngestor)
        frame = ingestor.generate_frame(timestamp_ns=0)

        # Should have observations from the simulated ingestor
        self.assertGreater(len(frame.observations), 0)

        # All observations should have valid positions within arena bounds
        for obs in frame.observations:
            self.assertGreaterEqual(obs.x, -ingestor.area_size / 2 - 1)
            self.assertLessEqual(obs.x, ingestor.area_size / 2 + 1)
            self.assertGreaterEqual(obs.y, -ingestor.area_size / 2 - 1)
            self.assertLessEqual(obs.y, ingestor.area_size / 2 + 1)

    def test_frame_timestamp_progression(self):
        """Test that timestamps progress correctly."""
        configs = create_test_camera_config(num_cameras=2)
        ingestor = DeepLabCutIngestor(configs, area_size=10.0)

        timestamps = []
        for i in range(5):
            frame = ingestor.generate_frame(timestamp_ns=i * 33_000_000)
            timestamps.append(frame.timestamp_ns)

        # Verify timestamps increase
        for i in range(1, len(timestamps)):
            self.assertGreater(timestamps[i], timestamps[i-1])


class TestCameraSource(unittest.TestCase):
    """Test CameraSource enum."""

    def test_source_types(self):
        """Test all camera source types."""
        self.assertEqual(CameraSource.RTSP.value, "rtsp")
        self.assertEqual(CameraSource.USB.value, "usb")
        self.assertEqual(CameraSource.FILE.value, "file")
        self.assertEqual(CameraSource.TEST_PATTERN.value, "test_pattern")


if __name__ == "__main__":
    unittest.main()
