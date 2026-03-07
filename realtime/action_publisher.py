#!/usr/bin/env python3
"""
Action Publisher - Python Layer
==============================

Publishes synthesis actions from Python Logic Layer to Rust Execution Layer.

This module implements the Python side of the Closed-Loop Interaction Agent,
sending synthesis timelines and micro-dynamics deltas to the Rust synthesizer.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Configuration defaults
ACTIONS_ENDPOINT = os.environ.get("RUST_ACTIONS_ENDPOINT", "ipc:///tmp/cognitive_actions.ipc")


class ActionPriority(Enum):
    """Priority levels for synthesis actions"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TimelineEvent:
    """Single event in synthesis timeline"""

    cluster_id: int
    start_time_ms: float
    duration_ms: float
    amplitude: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TimelineEvent":
        return cls(
            cluster_id=data["cluster_id"],
            start_time_ms=data["start_time_ms"],
            duration_ms=data["duration_ms"],
            amplitude=data.get("amplitude", 1.0),
        )


@dataclass
class MicroDynamicsDelta:
    """Delta transformations for synthesis"""

    delta_mean_f0_hz: float = 0.0
    delta_duration_ms: float = 0.0
    delta_f0_range_hz: float = 0.0
    delta_harmonic_to_noise_ratio: float = 0.0
    delta_attack_time_ms: float = 0.0
    delta_sustain_level: float = 0.0
    delta_rms_energy: float = 0.0

    def to_dict(self) -> dict:
        # Only include non-zero deltas for efficiency
        result = {}
        d = asdict(self)
        for k, v in d.items():
            if v != 0.0:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "MicroDynamicsDelta":
        return cls(
            delta_mean_f0_hz=data.get("delta_mean_f0_hz", 0.0),
            delta_duration_ms=data.get("delta_duration_ms", 0.0),
            delta_f0_range_hz=data.get("delta_f0_range_hz", 0.0),
            delta_harmonic_to_noise_ratio=data.get("delta_harmonic_to_noise_ratio", 0.0),
            delta_attack_time_ms=data.get("delta_attack_time_ms", 0.0),
            delta_sustain_level=data.get("delta_sustain_level", 0.0),
            delta_rms_energy=data.get("delta_rms_energy", 0.0),
        )


@dataclass
class SynthesisAction:
    """Action command from Python to Rust"""

    action_type: str
    timeline: List[TimelineEvent]
    deltas: Optional[MicroDynamicsDelta] = None
    priority: str = "normal"

    def to_json(self) -> str:
        return json.dumps(
            {
                "action_type": self.action_type,
                "timeline": [e.to_dict() for e in self.timeline],
                "deltas": self.deltas.to_dict() if self.deltas else None,
                "priority": self.priority,
            }
        )

    def to_bytes(self) -> bytes:
        return self.to_json().encode("utf-8")

    @classmethod
    def from_json(cls, json_str: str) -> "SynthesisAction":
        data = json.loads(json_str)
        return cls(
            action_type=data["action_type"],
            timeline=[TimelineEvent.from_dict(e) for e in data["timeline"]],
            deltas=MicroDynamicsDelta.from_dict(data["deltas"]) if data.get("deltas") else None,
            priority=data.get("priority", "normal"),
        )


@dataclass
class ActionPublisherConfig:
    """Configuration for action publisher"""

    action_endpoint: str = ACTIONS_ENDPOINT
    send_high_water_mark: int = 10


class ActionPublisher:
    """
    ZeroMQ publisher for synthesis actions to Rust.

    Sends synthesis timelines and delta transformations to the
    Rust Execution Layer for audio output.

    Usage:
        publisher = ActionPublisher()
        publisher.connect()

        timeline = [TimelineEvent(cluster_id=42, start_time_ms=0.0, duration_ms=150.0)]
        deltas = MicroDynamicsDelta(delta_mean_f0_hz=100.0)
        publisher.publish_timeline(timeline, deltas=deltas, priority="high")

        publisher.disconnect()
    """

    def __init__(self, config: Optional[ActionPublisherConfig] = None):
        """
        Initialize action publisher.

        Args:
            config: Publisher configuration (uses defaults if None)
        """
        self.config = config or ActionPublisherConfig()

        self._context: Optional[Any] = None
        self._socket: Optional[Any] = None

        # Statistics
        self._actions_sent = 0

        logger.info(f"ActionPublisher initialized for {self.config.action_endpoint}")

    def connect(self) -> None:
        """Connect to the Rust action subscriber"""
        try:
            import zmq
        except ImportError:
            logger.error("ZeroMQ not installed. Install with: pip install pyzmq")
            raise

        logger.info(f"Connecting to Rust Action Subscriber: {self.config.action_endpoint}")

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)

        # Set socket options
        self._socket.setsockopt(zmq.LINGER, 1000)
        self._socket.setsockopt(zmq.SNDHWM, self.config.send_high_water_mark)

        # Connect (Rust binds)
        self._socket.connect(self.config.action_endpoint)

        logger.info("✓ Connected to Rust Action Subscriber")

    def disconnect(self) -> None:
        """Disconnect from the Rust action subscriber"""
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

        logger.info("✓ Disconnected from Rust Action Subscriber")

    def publish_action(self, action: SynthesisAction) -> bool:
        """
        Publish a synthesis action to Rust.

        Args:
            action: The synthesis action to publish

        Returns:
            True if sent successfully
        """
        if not self._socket:
            logger.error("Not connected to Rust Action Subscriber")
            return False

        try:
            import zmq

            bytes_data = action.to_bytes()
            self._socket.send(bytes_data, zmq.DONTWAIT)
            self._actions_sent += 1

            if self._actions_sent % 100 == 0:
                logger.debug(f"Sent {self._actions_sent} actions")

            return True

        except Exception as e:
            logger.error(f"Failed to send action: {e}")
            return False

    def publish_timeline(
        self,
        timeline: List[TimelineEvent],
        deltas: Optional[MicroDynamicsDelta] = None,
        priority: str = "normal",
    ) -> bool:
        """
        Convenience method to publish a synthesis timeline.

        Args:
            timeline: List of timeline events
            deltas: Optional micro-dynamics deltas
            priority: Action priority

        Returns:
            True if sent successfully
        """
        action = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=timeline,
            deltas=deltas,
            priority=priority,
        )
        return self.publish_action(action)

    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics"""
        return {
            "actions_sent": self._actions_sent,
            "endpoint": self.config.action_endpoint,
        }


# Convenience function for creating a single-event timeline
def create_single_event_timeline(
    cluster_id: int,
    duration_ms: float = 150.0,
    amplitude: float = 1.0,
) -> List[TimelineEvent]:
    """
    Create a timeline with a single event.

    Args:
        cluster_id: Cluster ID to synthesize
        duration_ms: Duration in milliseconds
        amplitude: Amplitude (0.0 to 1.0)

    Returns:
        List with single TimelineEvent
    """
    return [
        TimelineEvent(
            cluster_id=cluster_id,
            start_time_ms=0.0,
            duration_ms=duration_ms,
            amplitude=amplitude,
        )
    ]


if __name__ == "__main__":
    # Demo/test mode
    logging.basicConfig(level=logging.INFO)

    publisher = ActionPublisher()

    print("Connecting publisher...")
    publisher.connect()

    # Send a test action
    timeline = create_single_event_timeline(cluster_id=42, duration_ms=200.0)
    deltas = MicroDynamicsDelta(delta_mean_f0_hz=100.0)

    print(f"Sending timeline: {timeline}")
    success = publisher.publish_timeline(timeline, deltas=deltas, priority="normal")

    print(f"Send successful: {success}")
    print(f"Stats: {publisher.get_stats()}")

    publisher.disconnect()
