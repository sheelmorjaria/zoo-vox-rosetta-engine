#!/usr/bin/env python3
"""
Tests for Action Publisher (Python → Rust)

These tests verify that Python can serialize synthesis actions
to be sent to the Rust Execution Layer.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import unittest
from dataclasses import asdict, dataclass
from typing import List, Optional

# ============================================================================
# Data Types for Testing
# ============================================================================


@dataclass
class TimelineEvent:
    """Single event in synthesis timeline"""

    cluster_id: int
    start_time_ms: float
    duration_ms: float
    amplitude: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


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


@dataclass
class SynthesisAction:
    """Action command from Python to Rust"""

    action_type: str
    timeline: List[TimelineEvent]
    deltas: Optional[MicroDynamicsDelta] = None
    priority: str = "normal"  # "low", "normal", "high", "critical"

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
            timeline=[TimelineEvent(**e) for e in data["timeline"]],
            deltas=MicroDynamicsDelta(**data["deltas"]) if data.get("deltas") else None,
            priority=data.get("priority", "normal"),
        )


# ============================================================================
# Tests
# ============================================================================


class TestTimelineEvent(unittest.TestCase):
    """Test timeline event serialization"""

    def test_timeline_event_creation(self):
        """Should create timeline event with all fields"""
        event = TimelineEvent(
            cluster_id=42,
            start_time_ms=0.0,
            duration_ms=150.0,
            amplitude=0.8,
        )

        self.assertEqual(event.cluster_id, 42)
        self.assertEqual(event.start_time_ms, 0.0)
        self.assertEqual(event.duration_ms, 150.0)
        self.assertEqual(event.amplitude, 0.8)

    def test_timeline_event_default_amplitude(self):
        """Amplitude should default to 1.0"""
        event = TimelineEvent(
            cluster_id=42,
            start_time_ms=0.0,
            duration_ms=150.0,
        )

        self.assertEqual(event.amplitude, 1.0)

    def test_timeline_event_to_dict(self):
        """Should serialize to dictionary"""
        event = TimelineEvent(
            cluster_id=42,
            start_time_ms=0.0,
            duration_ms=150.0,
            amplitude=0.8,
        )

        data = event.to_dict()

        self.assertEqual(data["cluster_id"], 42)
        self.assertEqual(data["start_time_ms"], 0.0)
        self.assertEqual(data["duration_ms"], 150.0)
        self.assertEqual(data["amplitude"], 0.8)


class TestMicroDynamicsDelta(unittest.TestCase):
    """Test micro-dynamics delta serialization"""

    def test_delta_creation(self):
        """Should create delta with all fields"""
        delta = MicroDynamicsDelta(
            delta_mean_f0_hz=100.0,
            delta_duration_ms=20.0,
            delta_f0_range_hz=50.0,
        )

        self.assertEqual(delta.delta_mean_f0_hz, 100.0)
        self.assertEqual(delta.delta_duration_ms, 20.0)
        self.assertEqual(delta.delta_f0_range_hz, 50.0)

    def test_delta_defaults(self):
        """All deltas should default to 0.0"""
        delta = MicroDynamicsDelta()

        self.assertEqual(delta.delta_mean_f0_hz, 0.0)
        self.assertEqual(delta.delta_duration_ms, 0.0)
        self.assertEqual(delta.delta_f0_range_hz, 0.0)

    def test_delta_to_dict_excludes_zeros(self):
        """to_dict should exclude zero deltas for efficiency"""
        delta = MicroDynamicsDelta(
            delta_mean_f0_hz=100.0,
            delta_duration_ms=0.0,  # Zero - should be excluded
            delta_f0_range_hz=50.0,
        )

        data = delta.to_dict()

        self.assertEqual(data["delta_mean_f0_hz"], 100.0)
        self.assertNotIn("delta_duration_ms", data)  # Should be excluded
        self.assertEqual(data["delta_f0_range_hz"], 50.0)

    def test_delta_to_dict_all_zeros(self):
        """Empty dict when all deltas are zero"""
        delta = MicroDynamicsDelta()
        data = delta.to_dict()
        self.assertEqual(data, {})


class TestSynthesisAction(unittest.TestCase):
    """Test synthesis action serialization"""

    def test_synthesis_action_creation(self):
        """Should create synthesis action"""
        action = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=[
                TimelineEvent(cluster_id=42, start_time_ms=0.0, duration_ms=150.0),
            ],
        )

        self.assertEqual(action.action_type, "synthesize_timeline")
        self.assertEqual(len(action.timeline), 1)
        self.assertIsNone(action.deltas)
        self.assertEqual(action.priority, "normal")

    def test_synthesis_action_with_deltas(self):
        """Should create action with deltas"""
        action = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=[
                TimelineEvent(cluster_id=42, start_time_ms=0.0, duration_ms=150.0),
            ],
            deltas=MicroDynamicsDelta(delta_mean_f0_hz=100.0),
            priority="high",
        )

        self.assertIsNotNone(action.deltas)
        self.assertEqual(action.deltas.delta_mean_f0_hz, 100.0)
        self.assertEqual(action.priority, "high")

    def test_synthesis_action_to_json(self):
        """Should serialize to JSON"""
        action = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=[
                TimelineEvent(cluster_id=42, start_time_ms=0.0, duration_ms=150.0),
            ],
            priority="high",
        )

        json_str = action.to_json()
        data = json.loads(json_str)

        self.assertEqual(data["action_type"], "synthesize_timeline")
        self.assertEqual(len(data["timeline"]), 1)
        self.assertEqual(data["timeline"][0]["cluster_id"], 42)
        self.assertEqual(data["priority"], "high")

    def test_synthesis_action_to_bytes(self):
        """Should serialize to bytes for ZeroMQ"""
        action = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=[
                TimelineEvent(cluster_id=1, start_time_ms=0.0, duration_ms=100.0),
            ],
        )

        bytes_data = action.to_bytes()

        self.assertIsInstance(bytes_data, bytes)

        # Verify round-trip
        decoded = json.loads(bytes_data.decode("utf-8"))
        self.assertEqual(decoded["action_type"], "synthesize_timeline")

    def test_synthesis_action_from_json(self):
        """Should deserialize from JSON"""
        json_str = json.dumps(
            {
                "action_type": "synthesize_timeline",
                "timeline": [
                    {"cluster_id": 42, "start_time_ms": 0.0, "duration_ms": 150.0, "amplitude": 1.0}
                ],
                "deltas": {"delta_mean_f0_hz": 100.0},
                "priority": "high",
            }
        )

        action = SynthesisAction.from_json(json_str)

        self.assertEqual(action.action_type, "synthesize_timeline")
        self.assertEqual(len(action.timeline), 1)
        self.assertEqual(action.timeline[0].cluster_id, 42)
        self.assertEqual(action.deltas.delta_mean_f0_hz, 100.0)
        self.assertEqual(action.priority, "high")

    def test_synthesis_action_roundtrip(self):
        """Should roundtrip through JSON serialization"""
        original = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=[
                TimelineEvent(cluster_id=42, start_time_ms=0.0, duration_ms=150.0, amplitude=0.8),
                TimelineEvent(cluster_id=99, start_time_ms=150.0, duration_ms=200.0, amplitude=1.0),
            ],
            deltas=MicroDynamicsDelta(
                delta_mean_f0_hz=100.0,
                delta_duration_ms=20.0,
            ),
            priority="critical",
        )

        # Roundtrip
        json_str = original.to_json()
        decoded = SynthesisAction.from_json(json_str)

        self.assertEqual(decoded.action_type, original.action_type)
        self.assertEqual(len(decoded.timeline), len(original.timeline))
        self.assertEqual(decoded.timeline[0].cluster_id, original.timeline[0].cluster_id)
        self.assertEqual(decoded.timeline[1].cluster_id, original.timeline[1].cluster_id)
        self.assertEqual(decoded.priority, original.priority)


class TestActionPublisherConfig(unittest.TestCase):
    """Test action publisher configuration"""

    def test_default_config(self):
        """Default config should have correct endpoint"""
        from realtime.action_publisher import ActionPublisherConfig

        config = ActionPublisherConfig()

        self.assertEqual(config.action_endpoint, "ipc:///tmp/cognitive_actions.ipc")
        self.assertEqual(config.send_high_water_mark, 10)

    def test_custom_config(self):
        """Should accept custom configuration"""
        from realtime.action_publisher import ActionPublisherConfig

        config = ActionPublisherConfig(
            action_endpoint="tcp://localhost:5556",
            send_high_water_mark=20,
        )

        self.assertEqual(config.action_endpoint, "tcp://localhost:5556")
        self.assertEqual(config.send_high_water_mark, 20)


class TestActionPublisherStats(unittest.TestCase):
    """Test action publisher statistics"""

    def test_initial_stats(self):
        """Initial stats should be zero"""
        from realtime.action_publisher import ActionPublisher

        publisher = ActionPublisher()
        stats = publisher.get_stats()

        self.assertEqual(stats["actions_sent"], 0)


if __name__ == "__main__":
    unittest.main()
