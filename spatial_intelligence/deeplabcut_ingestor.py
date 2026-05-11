#!/usr/bin/env python3
"""
DeepLabCut RTSP Ingestor for Real-Time Pose Estimation

Integrates with DeepLabCut for real-time markerless pose estimation
from RTSP video streams. Converts pose keypoints to SpatialObservations.

This module enables Level 2.5 spatial inference with live video input
from field cameras tracking animal colonies.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from spatial_intelligence.spatial_ingestor import (
    SimulatedIngestor,
    SpatialFrame,
    SpatialIngestor,
    SpatialObservation,
)

logger = logging.getLogger(__name__)


class CameraSource(Enum):
    """Type of camera source."""
    RTSP = "rtsp"
    USB = "usb"
    FILE = "file"
    TEST_PATTERN = "test_pattern"


@dataclass
class DLCCameraConfig:
    """Configuration for a single camera in the tracking array."""
    camera_id: str
    source_type: CameraSource
    source_url: str  # RTSP URL, device index, or file path

    # Camera position and orientation (for 3D triangulation)
    camera_position: Tuple[float, float, float]  # x, y, z in meters
    camera_heading: float  # Heading in radians
    camera_pitch: float = 0.0  # Pitch angle in radians
    camera_fov_deg: float = 90.0  # Field of view in degrees

    # Calibration parameters
    focal_length: float = 800.0  # Pixels
    principal_point: Tuple[float, float] = (640.0, 360.0)  # cx, cy

    # DeepLabCut model
    dlc_model_path: Optional[str] = None
    dlc_config_path: Optional[str] = None


@dataclass
class PoseKeypoints:
    """
    Detected pose keypoints from DeepLabCut.

    For marmoset tracking, typical keypoints include:
    - nose, left_ear, right_ear, left_shoulder, right_shoulder
    - tail_base, tail_tip
    """
    agent_id: str
    confidence: float  # Overall detection confidence (0-1)
    keypoints: Dict[str, Tuple[float, float, float]]  # name -> (x, y, conf)
    timestamp_ns: int

    def get_body_center(self) -> Tuple[float, float, float]:
        """Get the estimated center of the body (average of visible keypoints)."""
        visible_points = [(x, y) for name, (x, y, c) in self.keypoints.items() if c > 0.5]

        if not visible_points:
            return (0.0, 0.0, 0.0)

        avg_x = sum(p[0] for p in visible_points) / len(visible_points)
        avg_y = sum(p[1] for p in visible_points) / len(visible_points)

        # Average confidence
        avg_conf = sum(c for _, _, c in self.keypoints.values()) / len(self.keypoints)

        return (avg_x, avg_y, avg_conf)

    def compute_heading(self) -> float:
        """
        Compute heading direction from keypoints.

        Uses vector from tail_base to nose (or shoulders if nose unavailable).
        Returns heading in radians.
        """
        # Try nose to tail_base
        if "nose" in self.keypoints and "tail_base" in self.keypoints:
            nose_x, nose_y, nose_c = self.keypoints["nose"]
            tail_x, tail_y, tail_c = self.keypoints["tail_base"]

            if nose_c > 0.3 and tail_c > 0.3:
                dx = nose_x - tail_x
                dy = nose_y - tail_y
                return np.arctan2(dy, dx)

        # Fallback: shoulder vector
        if "left_shoulder" in self.keypoints and "right_shoulder" in self.keypoints:
            l_x, l_y, l_c = self.keypoints["left_shoulder"]
            r_x, r_y, r_c = self.keypoints["right_shoulder"]

            if l_c > 0.3 and r_c > 0.3:
                dx = r_x - l_x
                dy = r_y - l_y
                # Perpendicular to shoulder line (forward direction)
                return np.arctan2(dy, dx) + np.pi / 2

        return 0.0


class DeepLabCutIngestor(SpatialIngestor):
    """
    Real-time pose estimation ingestor using DeepLabCut.

    Processes RTSP video streams and converts detected poses to
    SpatialObservations for Level 2.5 spatial inference.
    """

    def __init__(
        self,
        camera_configs: List[DLCCameraConfig],
        area_size: float = 10.0,  # Arena size in meters
        frame_rate: float = 30.0,  # Target processing FPS
        confidence_threshold: float = 0.5,
    ):
        self.camera_configs = {c.camera_id: c for c in camera_configs}
        self.area_size = area_size
        self.frame_rate = frame_rate
        self.confidence_threshold = confidence_threshold

        # Video capture threads
        self.captures: Dict[str, cv2.VideoCapture] = {}
        self.capture_threads: Dict[str, threading.Thread] = {}
        self.running = False

        # Latest frames
        self.latest_frames: Dict[str, np.ndarray] = {}
        self.frame_lock = threading.Lock()

        # DeepLabCut models (lazy loaded)
        self.dlc_models: Dict[str, any] = None  # Lazy loaded per camera

        # Coordinate transformation parameters
        # Maps pixel coordinates to world coordinates
        self.world_scale = area_size / 1920.0  # Assuming 1920px width
        self.world_offset_x = -area_size / 2
        self.world_offset_y = -area_size / 2

        logger.info(f"DeepLabCutIngestor initialized with {len(camera_configs)} cameras")

    def _pixel_to_world(self, x_px: float, y_px: float, camera_id: str) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to world coordinates.

        Args:
            x_px: X pixel coordinate
            y_px: Y pixel coordinate
            camera_id: Camera identifier for calibration

        Returns:
            (x, y, z) world position in meters
        """
        config = self.camera_configs.get(camera_id)
        if config is None:
            # Default transformation
            x = x_px * self.world_scale + self.world_offset_x
            y = y_px * self.world_scale + self.world_offset_y
            return (x, y, 0.0)

        # Camera-specific transformation with perspective correction
        # Simple homography approximation
        x_norm = (x_px - config.principal_point[0]) / config.focal_length
        y_norm = (y_px - config.principal_point[1]) / config.focal_length

        # Assume camera is at height looking down
        camera_height = config.camera_position[2]
        ground_distance = camera_height / y_norm if y_norm > 0.01 else 5.0

        x = config.camera_position[0] + ground_distance * x_norm
        y = config.camera_position[1] + ground_distance * y_norm  # Forward direction

        return (x, y, 0.0)

    def start_capture(self):
        """Start video capture threads for all cameras."""
        self.running = True

        for camera_id, config in self.camera_configs.items():
            if config.source_type == CameraSource.TEST_PATTERN:
                # Don't open capture for test pattern
                continue

            cap = cv2.VideoCapture(config.source_url)
            if not cap.isOpened():
                logger.warning(f"Failed to open camera {camera_id} at {config.source_url}")
                continue

            # Configure capture
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, int(self.frame_rate))

            self.captures[camera_id] = cap

            # Start capture thread
            thread = threading.Thread(
                target=self._capture_loop,
                args=(camera_id, cap),
                daemon=True
            )
            self.capture_threads[camera_id] = thread
            thread.start()

        logger.info(f"Started {len(self.captures)} camera capture threads")

    def stop_capture(self):
        """Stop video capture threads."""
        self.running = False

        for camera_id, thread in self.capture_threads.items():
            thread.join(timeout=1.0)

        for camera_id, cap in self.captures.items():
            cap.release()

        self.captures.clear()
        self.capture_threads.clear()

        logger.info("Stopped all camera capture threads")

    def _capture_loop(self, camera_id: str, cap: cv2.VideoCapture):
        """Background thread for continuous frame capture."""
        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frames[camera_id] = frame
            else:
                logger.warning(f"Failed to read frame from {camera_id}")
                time.sleep(0.1)

    def get_latest_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Get the latest frame from a camera."""
        with self.frame_lock:
            return self.latest_frames.get(camera_id)

    def detect_poses(
        self,
        camera_id: str,
        frame: Optional[np.ndarray] = None
    ) -> List[PoseKeypoints]:
        """
        Detect poses from a camera frame using DeepLabCut.

        Args:
            camera_id: Camera identifier
            frame: Optional frame (uses latest if None)

        Returns:
            List of detected poses
        """
        if frame is None:
            frame = self.get_latest_frame(camera_id)

        if frame is None:
            logger.warning(f"No frame available for camera {camera_id}")
            return []

        config = self.camera_configs.get(camera_id)
        if config is None:
            return []

        # Try to load DeepLabCut model
        if self.dlc_models is None:
            try:
                import deeplabcut
                self.dlc_models = {}
            except ImportError:
                logger.warning("DeepLabCut not installed, using mock detections")
                return self._mock_poses(camera_id, frame)

        # TODO: Run actual DeepLabCut inference
        # For now, return mock detections
        return self._mock_poses(camera_id, frame)

    def _mock_poses(self, camera_id: str, frame: np.ndarray) -> List[PoseKeypoints]:
        """
        Generate mock pose detections for testing.

        In production, this would be replaced with actual DeepLabCut inference.
        """
        height, width = frame.shape[:2]
        timestamp_ns = time.time_ns()

        # Generate 2-3 mock agents per frame
        mock_poses = []
        num_agents = np.random.randint(2, 4)

        for i in range(num_agents):
            # Random position in frame
            x_px = np.random.uniform(100, width - 100)
            y_px = np.random.uniform(100, height - 100)

            # Create keypoints around this center
            keypoints = {
                "nose": (x_px + 20, y_px, 0.9),
                "left_ear": (x_px + 10, y_px - 10, 0.85),
                "right_ear": (x_px + 30, y_px - 10, 0.85),
                "tail_base": (x_px - 20, y_px, 0.8),
                "tail_tip": (x_px - 40, y_px + 5, 0.7),
            }

            pose = PoseKeypoints(
                agent_id=f"{camera_id}_agent_{i}",
                confidence=0.85,
                keypoints=keypoints,
                timestamp_ns=timestamp_ns
            )
            mock_poses.append(pose)

        return mock_poses

    def generate_frame(self, timestamp_ns: int) -> SpatialFrame:
        """
        Generate a SpatialFrame from current pose detections.

        Args:
            timestamp_ns: Timestamp for the frame

        Returns:
            SpatialFrame with observations from all cameras
        """
        frame = SpatialFrame(timestamp_ns=timestamp_ns)
        timestamp = time.time_ns()

        # Process each camera
        for camera_id, config in self.camera_configs.items():
            # Get current frame
            current_frame = self.get_latest_frame(camera_id)

            if current_frame is None and config.source_type != CameraSource.TEST_PATTERN:
                continue

            # Detect poses
            poses = self.detect_poses(camera_id, current_frame)

            # Convert poses to spatial observations
            for pose in poses:
                # Get body center
                center_x, center_y, conf = pose.get_body_center()

                if conf < self.confidence_threshold:
                    continue

                # Convert to world coordinates
                world_x, world_y, world_z = self._pixel_to_world(
                    center_x, center_y, camera_id
                )

                # Compute heading
                heading = pose.compute_heading()

                # Create observation
                obs = SpatialObservation(
                    agent_id=pose.agent_id,
                    x=world_x,
                    y=world_y,
                    z=world_z,
                    heading_rad=heading,
                    velocity=0.0,  # Could compute from previous frame
                    timestamp_ns=timestamp
                )

                frame.observations.append(obs)

        # If test pattern mode and no observations, generate some
        if not frame.observations and any(
            c.source_type == CameraSource.TEST_PATTERN for c in self.camera_configs.values()
        ):
            # Use simulated ingestor for test pattern
            sim = SimulatedIngestor(num_agents=3, area_size=self.area_size)
            return sim.generate_frame(timestamp_ns)

        return frame

    def process_frame(self, camera_id: str, frame: np.ndarray) -> List[SpatialObservation]:
        """
        Process a single frame from a camera.

        Useful for offline processing or single-camera setups.
        """
        poses = self.detect_poses(camera_id, frame)
        observations = []

        for pose in poses:
            center_x, center_y, conf = pose.get_body_center()

            if conf < self.confidence_threshold:
                continue

            world_x, world_y, world_z = self._pixel_to_world(center_x, center_y, camera_id)
            heading = pose.compute_heading()

            obs = SpatialObservation(
                agent_id=pose.agent_id,
                x=world_x,
                y=world_y,
                z=world_z,
                heading_rad=heading,
                velocity=0.0,
                timestamp_ns=time.time_ns()
            )

            observations.append(obs)

        return observations


def create_test_camera_config(num_cameras: int = 4) -> List[DLCCameraConfig]:
    """
    Create test camera configurations in a circular array.

    Args:
        num_cameras: Number of cameras to create

    Returns:
        List of camera configurations
    """
    configs = []
    radius = 3.0  # 3 meters from center
    area_size = 10.0

    for i in range(num_cameras):
        angle = (i / num_cameras) * 2 * np.pi
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        # Camera faces inward
        heading = angle + np.pi

        config = DLCCameraConfig(
            camera_id=f"camera_{i}",
            source_type=CameraSource.TEST_PATTERN,
            source_url="",
            camera_position=(x, y, 2.0),  # 2m height
            camera_heading=heading,
            camera_pitch=-np.pi / 6,  # 30° downward pitch
        )
        configs.append(config)

    return configs


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Test mode: create test cameras and generate frames
    configs = create_test_camera_config(num_cameras=4)
    ingestor = DeepLabCutIngestor(configs, area_size=10.0)

    # Generate a few frames
    for i in range(5):
        frame = ingestor.generate_frame(timestamp_ns=i * 33_000_000)  # 30 FPS
        print(f"\nFrame {i}: {len(frame.observations)} observations")

        for obs in frame.observations:
            print(f"  {obs.agent_id}: x={obs.x:.2f}, y={obs.y:.2f}, heading={obs.heading_rad:.2f}")

    print("\nDeepLabCut ingestor test complete!")
