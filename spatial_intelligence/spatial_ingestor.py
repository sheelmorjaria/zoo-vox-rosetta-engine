#!/usr/bin/env python3
"""
Spatial Ingestion Module (Level 2.5)

Ingests and normalizes spatial tracking data from multiple sources:
- DeepLabCut via RTSP (camera-based pose estimation)
- RFID triangulation
- Acoustic array triangulation
- Manual annotation

All data is normalized into a unified coordinate space and timestamp format.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TrackingSource(Enum):
    """Types of spatial tracking sources."""
    DEEPLABCUT = "deeplabcut"  # Camera-based pose estimation
    RFID = "rfid"              # RFID triangulation
    ACOUSTIC_ARRAY = "acoustic_array"  # Microphone array triangulation
    MANUAL = "manual"          # Human annotation
    SIMULATED = "simulated"    # Mock data for testing


@dataclass
class SpatialObservation:
    """
    Unified spatial observation for a single agent at a point in time.

    All coordinates are in meters, relative to a defined origin (0,0,0).
    The coordinate system is right-handed: X=forward, Y=left, Z=up.
    """
    agent_id: str                      # Matches Emitter ID from Level 2
    x: float                           # Position X (meters)
    y: float                           # Position Y (meters)
    z: float                           # Position Z (meters, height)
    heading_rad: float                 # Direction agent is facing (radians)
    velocity: float                    # Speed of movement (m/s)
    timestamp_ns: int                  # Nanosecond precision for sync
    confidence: float = 1.0            # Tracking confidence (0.0 to 1.0)
    source: TrackingSource = TrackingSource.MANUAL
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        """Convert position to numpy array for distance calculations."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def distance_to(self, other: 'SpatialObservation') -> float:
        """Calculate Euclidean distance to another observation."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def angle_to(self, other: 'SpatialObservation') -> float:
        """
        Calculate the angle from this agent's heading to another agent.

        Returns:
            Angle in radians. 0 means directly ahead, pi means directly behind.
        """
        # Vector from self to other
        diff = other.to_array() - self.to_array()
        diff = diff[:2]  # Only X-Y plane for heading

        # Normalize
        if np.linalg.norm(diff) < 1e-6:
            return 0.0

        diff = diff / np.linalg.norm(diff)

        # Heading vector
        heading = np.array([np.cos(self.heading_rad), np.sin(self.heading_rad)])

        # Angle between heading and direction to other
        dot = np.clip(np.dot(heading, diff), -1.0, 1.0)
        return float(np.arccos(dot))


@dataclass
class SpatialFrame:
    """
    A snapshot of all agent positions at a specific timestamp.
    """
    timestamp_ns: int
    observations: List[SpatialObservation] = field(default_factory=list)

    def get_observation(self, agent_id: str) -> Optional[SpatialObservation]:
        """Get observation for a specific agent."""
        for obs in self.observations:
            if obs.agent_id == agent_id:
                return obs
        return None

    def agent_ids(self) -> List[str]:
        """Get all agent IDs in this frame."""
        return [obs.agent_id for obs in self.observations]


class SpatialIngestor:
    """
    Ingests tracking data from various sources and normalizes it.

    Handles coordinate transformation, timestamp synchronization,
    and data validation.
    """

    def __init__(
        self,
        coordinate_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        coordinate_scale: float = 1.0,  # Multiplier for input coordinates
        max_age_ms: float = 500.0,  # Maximum age of observations to keep
    ):
        self.origin = coordinate_origin
        self.scale = coordinate_scale
        self.max_age_ms = max_age_ms

        # State storage
        self.latest_observations: Dict[str, SpatialObservation] = {}

        logger.info(
            f"SpatialIngestor initialized: origin={coordinate_origin}, "
            f"scale={coordinate_scale}"
        )

    def process_frame(self, raw_frame: Dict[str, Any]) -> SpatialFrame:
        """
        Process a raw frame from a tracking source.

        Args:
            raw_frame: Dictionary with tracking data. Expected format:
                {
                    "timestamp_ns": int,
                    "source": str,
                    "agents": [
                        {
                            "agent_id": str,
                            "x": float, "y": float, "z": float,
                            "heading": float,
                            "velocity": float,
                            "confidence": float
                        },
                        ...
                    ]
                }

        Returns:
            Normalized SpatialFrame
        """
        timestamp_ns = raw_frame.get("timestamp_ns", 0)
        source_str = raw_frame.get("source", "manual")

        try:
            source = TrackingSource(source_str)
        except ValueError:
            logger.warning(f"Unknown tracking source: {source_str}, using MANUAL")
            source = TrackingSource.MANUAL

        observations = []

        for agent_data in raw_frame.get("agents", []):
            obs = self._create_observation(agent_data, timestamp_ns, source)
            if obs is not None:
                observations.append(obs)
                self.latest_observations[obs.agent_id] = obs

        frame = SpatialFrame(timestamp_ns=timestamp_ns, observations=observations)

        logger.debug(f"Processed frame with {len(observations)} observations")

        return frame

    def _create_observation(
        self,
        agent_data: Dict[str, Any],
        timestamp_ns: int,
        source: TrackingSource,
    ) -> Optional[SpatialObservation]:
        """Create a SpatialObservation from raw agent data."""
        try:
            # Apply coordinate transformation
            x = (agent_data["x"] * self.scale) + self.origin[0]
            y = (agent_data["y"] * self.scale) + self.origin[1]
            z = (agent_data.get("z", 0.0) * self.scale) + self.origin[2]

            # Validate coordinates
            if not all(np.isfinite([x, y, z])):
                logger.warning(f"Invalid coordinates for agent {agent_data.get('agent_id')}")
                return None

            obs = SpatialObservation(
                agent_id=str(agent_data["agent_id"]),
                x=float(x),
                y=float(y),
                z=float(z),
                heading_rad=float(agent_data.get("heading", 0.0)),
                velocity=float(agent_data.get("velocity", 0.0)),
                timestamp_ns=timestamp_ns,
                confidence=float(agent_data.get("confidence", 1.0)),
                source=source,
                raw_data=agent_data,
            )

            return obs

        except KeyError as e:
            logger.warning(f"Missing required field in agent data: {e}")
            return None
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid data type in agent data: {e}")
            return None

    def get_latest_observation(self, agent_id: str) -> Optional[SpatialObservation]:
        """Get the most recent observation for an agent."""
        return self.latest_observations.get(agent_id)

    def get_all_positions(self) -> Dict[str, np.ndarray]:
        """Get current positions of all agents as dict of agent_id -> position array."""
        return {
            agent_id: obs.to_array()
            for agent_id, obs in self.latest_observations.items()
        }

    def prune_old_observations(self, current_timestamp_ns: int) -> int:
        """
        Remove observations older than max_age_ms.

        Returns:
            Number of observations removed.
        """
        cutoff_time = current_timestamp_ns - (int(self.max_age_ms * 1_000_000))
        to_remove = []

        for agent_id, obs in self.latest_observations.items():
            if obs.timestamp_ns < cutoff_time:
                to_remove.append(agent_id)

        for agent_id in to_remove:
            del self.latest_observations[agent_id]

        if to_remove:
            logger.debug(f"Pruned {len(to_remove)} old observations")

        return len(to_remove)


class SimulatedIngestor(SpatialIngestor):
    """
    Generate simulated spatial data for testing.

    Creates a virtual colony with agents moving in realistic patterns.
    """

    def __init__(
        self,
        num_agents: int = 10,
        area_size: float = 10.0,  # 10x10 meter area
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_agents = num_agents
        self.area_size = area_size

        # Initialize agent states
        self.agent_states: Dict[str, Dict[str, float]] = {}

        for i in range(num_agents):
            agent_id = f"agent_{i:03d}"
            self.agent_states[agent_id] = {
                "x": np.random.uniform(-area_size/2, area_size/2),
                "y": np.random.uniform(-area_size/2, area_size/2),
                "z": 0.0,
                "heading": np.random.uniform(0, 2 * np.pi),
                "velocity": 0.0,
            }

        logger.info(f"SimulatedIngestor initialized with {num_agents} agents")

    def generate_frame(self, timestamp_ns: int, dt_ms: float = 33.0) -> SpatialFrame:
        """
        Generate a simulated frame.

        Args:
            timestamp_ns: Frame timestamp
            dt_ms: Time delta from previous frame (default 33ms = ~30 FPS)
        """
        agents = []
        dt = dt_ms / 1000.0  # Convert to seconds

        for agent_id, state in self.agent_states.items():
            # Random walk with momentum
            state["heading"] += np.random.uniform(-0.2, 0.2)  # Slight heading change

            # Move forward
            speed = 0.5  # m/s average walking speed
            state["x"] += np.cos(state["heading"]) * speed * dt
            state["y"] += np.sin(state["heading"]) * speed * dt

            # Wrap around boundaries
            half_size = self.area_size / 2
            state["x"] = (state["x"] + half_size) % self.area_size - half_size
            state["y"] = (state["y"] + half_size) % self.area_size - half_size

            state["velocity"] = speed

            agents.append({
                "agent_id": agent_id,
                "x": state["x"],
                "y": state["y"],
                "z": state["z"],
                "heading": state["heading"],
                "velocity": state["velocity"],
                "confidence": 1.0,
            })

        raw_frame = {
            "timestamp_ns": timestamp_ns,
            "source": "simulated",
            "agents": agents,
        }

        return self.process_frame(raw_frame)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test simulated ingestor
    ingestor = SimulatedIngestor(num_agents=5, area_size=10.0)

    # Generate a few frames
    for i in range(5):
        timestamp_ns = i * 33_000_000  # 33ms increments
        frame = ingestor.generate_frame(timestamp_ns)

        print(f"\nFrame {i} (t={timestamp_ns}ns):")
        for obs in frame.observations:
            print(f"  {obs.agent_id}: ({obs.x:.2f}, {obs.y:.2f}) heading={obs.heading_rad:.2f}")

        # Show distances
        if len(frame.observations) >= 2:
            obs1, obs2 = frame.observations[0], frame.observations[1]
            dist = obs1.distance_to(obs2)
            angle = obs1.angle_to(obs2)
            print(f"  Distance {obs1.agent_id} <-> {obs2.agent_id}: {dist:.2f}m")
            print(f"  Angle from {obs1.agent_id} heading: {np.degrees(angle):.1f}°")
