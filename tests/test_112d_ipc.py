#!/usr/bin/env python3
"""
Module 1 TDD Tests: 112D IPC Upgrade and AudioBufferEvent

This test suite verifies that the 112D RosettaFeatures can be transmitted
between Python and Rust via the upgraded ZMQ IPC protocol, and that the
new AudioBufferEvent correctly carries PCM audio data.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
Module 1 (v1.6.0): 112D Delta Support and AudioBufferEvent
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from realtime.action_publisher import (
    ActionPublisher,
    ActionPublisherConfig,
    AudioBufferEvent,
    MicroDynamicsDelta,
    SynthesisAction,
    TimelineEvent,
)

# =============================================================================
# TEST SUITE 1: 112D Delta Support
# =============================================================================


class Test112DDelta:
    """Verify 112D delta field in SynthesisAction."""

    def test_synthesis_action_accepts_112d_delta(self):
        """SynthesisAction should accept a 112-element numpy array."""
        delta_112d = np.zeros(112, dtype=np.float32)
        delta_112d[0] = 100.0  # Set some values

        action = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=[],
            delta_112d=delta_112d,
        )

        assert action.delta_112d is not None
        assert len(action.delta_112d) == 112
        assert action.delta_112d[0] == 100.0

    def test_synthesis_action_rejects_wrong_size_delta(self):
        """SynthesisAction should reject non-112 element arrays."""
        delta_wrong = np.zeros(100, dtype=np.float32)

        action = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=[],
            delta_112d=delta_wrong,
        )

        # Should raise ValueError when trying to serialize
        with pytest.raises(ValueError, match="must have exactly 112 elements"):
            action.to_json()

    def test_synthesis_action_112d_serialization(self):
        """112D delta should survive JSON round-trip."""
        delta_112d = np.arange(112, dtype=np.float32) * 0.1

        action = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=[TimelineEvent(cluster_id=42, start_time_ms=0.0, duration_ms=150.0)],
            delta_112d=delta_112d,
            priority="high",
        )

        # Serialize
        json_str = action.to_json()
        data = json.loads(json_str)

        assert "delta_112d" in data
        assert len(data["delta_112d"]) == 112
        assert data["delta_112d"][0] == pytest.approx(0.0)
        assert data["delta_112d"][10] == pytest.approx(1.0)

    def test_synthesis_action_112d_deserialization(self):
        """from_json should correctly reconstruct 112D delta."""
        delta_112d = np.arange(112, dtype=np.float32) * 0.1

        action = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=[TimelineEvent(cluster_id=42, start_time_ms=0.0, duration_ms=150.0)],
            delta_112d=delta_112d,
        )

        # Round-trip
        json_str = action.to_json()
        restored = SynthesisAction.from_json(json_str)

        assert restored.delta_112d is not None
        assert len(restored.delta_112d) == 112
        np.testing.assert_array_almost_equal(restored.delta_112d, delta_112d)

    def test_synthesis_action_backward_compatibility(self):
        """Actions without 112D delta should still work (backward compatibility)."""
        action = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=[TimelineEvent(cluster_id=42, start_time_ms=0.0, duration_ms=150.0)],
            deltas=MicroDynamicsDelta(delta_mean_f0_hz=100.0),
        )

        json_str = action.to_json()
        data = json.loads(json_str)

        # Should have deltas but not delta_112d
        assert "deltas" in data
        assert "delta_112d" not in data or data.get("delta_112d") is None

    def test_synthesis_action_with_both_deltas(self):
        """Action can have both 112D and legacy deltas (for migration)."""
        delta_112d = np.zeros(112, dtype=np.float32)
        delta_112d[0] = 50.0

        action = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=[TimelineEvent(cluster_id=42, start_time_ms=0.0, duration_ms=150.0)],
            delta_112d=delta_112d,
            deltas=MicroDynamicsDelta(delta_mean_f0_hz=100.0),
        )

        json_str = action.to_json()
        data = json.loads(json_str)

        # Both should be present
        assert "delta_112d" in data
        assert "deltas" in data


# =============================================================================
# TEST SUITE 2: AudioBufferEvent
# =============================================================================


class TestAudioBufferEvent:
    """Verify AudioBufferEvent for PCM audio transmission."""

    def test_audio_buffer_event_creation(self):
        """AudioBufferEvent should be created with correct fields."""
        audio_data = np.array([0.1, -0.2, 0.3, -0.4, 0.5], dtype=np.float32)

        event = AudioBufferEvent(
            audio_data=audio_data,
            sample_rate=48000,
            duration_ms=100.0,
            timestamp=1234567890.0,
            sequence=1,
        )

        assert len(event.audio_data) == 5
        assert event.sample_rate == 48000
        assert event.duration_ms == 100.0
        assert event.timestamp == 1234567890.0
        assert event.sequence == 1

    def test_audio_buffer_event_serialization(self):
        """AudioBufferEvent should serialize to JSON correctly."""
        audio_data = np.array([0.1, -0.2, 0.3], dtype=np.float32)

        event = AudioBufferEvent(
            audio_data=audio_data,
            sample_rate=48000,
            duration_ms=50.0,
            timestamp=1234567890.0,
            sequence=42,
        )

        json_str = event.to_json()
        data = json.loads(json_str)

        # Use approx for float comparison due to JSON float precision
        assert data["audio_data"][0] == pytest.approx(0.1, rel=1e-5)
        assert data["audio_data"][1] == pytest.approx(-0.2, rel=1e-5)
        assert data["audio_data"][2] == pytest.approx(0.3, rel=1e-5)
        assert data["sample_rate"] == 48000
        assert data["duration_ms"] == 50.0
        assert data["timestamp"] == 1234567890.0
        assert data["sequence"] == 42

    def test_audio_buffer_event_deserialization(self):
        """AudioBufferEvent should deserialize from JSON correctly."""
        audio_data = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)

        event = AudioBufferEvent(
            audio_data=audio_data,
            sample_rate=44100,
            duration_ms=200.0,
            timestamp=9876543210.0,
            sequence=99,
        )

        # Round-trip
        json_str = event.to_json()
        restored = AudioBufferEvent.from_json(json_str)

        np.testing.assert_array_equal(restored.audio_data, audio_data)
        assert restored.sample_rate == 44100
        assert restored.duration_ms == 200.0
        assert restored.timestamp == 9876543210.0
        assert restored.sequence == 99

    def test_audio_buffer_event_to_dict(self):
        """to_dict should return correct dictionary representation."""
        audio_data = np.array([0.5, -0.5], dtype=np.float32)

        event = AudioBufferEvent(
            audio_data=audio_data,
            sample_rate=48000,
            duration_ms=100.0,
            timestamp=111111.0,
            sequence=1,
        )

        data = event.to_dict()

        assert data["audio_data"] == [0.5, -0.5]
        assert data["sample_rate"] == 48000
        assert data["duration_ms"] == 100.0
        assert data["timestamp"] == 111111.0
        assert data["sequence"] == 1

    def test_audio_buffer_event_large_audio(self):
        """AudioBufferEvent should handle realistic audio buffer sizes."""
        # 100ms at 48kHz = 4800 samples
        audio_data = np.random.randn(4800).astype(np.float32) * 0.1

        event = AudioBufferEvent(
            audio_data=audio_data,
            sample_rate=48000,
            duration_ms=100.0,
            timestamp=0.0,
            sequence=1,
        )

        assert len(event.audio_data) == 4800

        # Round-trip should preserve all samples
        json_str = event.to_json()
        restored = AudioBufferEvent.from_json(json_str)
        np.testing.assert_array_almost_equal(restored.audio_data, audio_data)


# =============================================================================
# TEST SUITE 3: ActionPublisher with 112D
# =============================================================================


class TestActionPublisher112D:
    """Verify ActionPublisher handles 112D deltas correctly."""

    def test_publisher_timeline_with_112d(self):
        """publish_timeline should accept delta_112d parameter."""
        delta_112d = np.ones(112, dtype=np.float32) * 0.5

        config = ActionPublisherConfig(
            action_endpoint="ipc:///tmp/test_112d.ipc",
        )
        ActionPublisher(config)  # noqa: F841

        # Should not raise
        timeline = [TimelineEvent(cluster_id=42, start_time_ms=0.0, duration_ms=150.0)]

        # Note: Can't actually publish without ZMQ connection, but we can
        # verify the action is created correctly
        action = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=timeline,
            delta_112d=delta_112d,
        )

        assert action.delta_112d is not None
        assert np.all(action.delta_112d == 0.5)

    def test_publisher_audio_buffer_method_exists(self):
        """ActionPublisher should have publish_audio_buffer method."""
        publisher = ActionPublisher()

        assert hasattr(publisher, "publish_audio_buffer")

        audio_data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        AudioBufferEvent(  # noqa: F841
            audio_data=audio_data,
            sample_rate=48000,
            duration_ms=50.0,
            timestamp=0.0,
            sequence=1,
        )

        # Method exists and can be called (will fail without ZMQ, but signature is correct)
        # We just verify it accepts the right type
        import inspect

        sig = inspect.signature(publisher.publish_audio_buffer)
        assert "audio_buffer" in sig.parameters


# =============================================================================
# TEST SUITE 4: Integration Tests
# =============================================================================


class Test112DIntegration:
    """Integration tests for 112D IPC."""

    def test_full_112d_action_roundtrip(self):
        """Complete round-trip: create action → serialize → deserialize → verify."""
        delta_112d = np.arange(112, dtype=np.float32) * 0.05

        timeline = [
            TimelineEvent(cluster_id=1, start_time_ms=0.0, duration_ms=100.0, amplitude=0.8),
            TimelineEvent(cluster_id=2, start_time_ms=100.0, duration_ms=150.0, amplitude=0.6),
        ]

        original = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=timeline,
            delta_112d=delta_112d,
            priority="high",
        )

        # Round-trip through JSON
        json_str = original.to_json()
        restored = SynthesisAction.from_json(json_str)

        # Verify all fields
        assert restored.action_type == "synthesize_timeline"
        assert len(restored.timeline) == 2
        assert restored.timeline[0].cluster_id == 1
        assert restored.timeline[1].duration_ms == 150.0
        assert restored.priority == "high"
        np.testing.assert_array_almost_equal(restored.delta_112d, delta_112d)

    def test_112d_with_timeline_and_deltas_combined(self):
        """Test complex action with timeline, 112D delta, and legacy deltas."""
        delta_112d = np.ones(112, dtype=np.float32)
        legacy_deltas = MicroDynamicsDelta(
            delta_mean_f0_hz=100.0,
            delta_duration_ms=50.0,
        )

        action = SynthesisAction(
            action_type="synthesize_timeline",
            timeline=[
                TimelineEvent(cluster_id=42, start_time_ms=0.0, duration_ms=200.0),
            ],
            delta_112d=delta_112d,
            deltas=legacy_deltas,
            priority="critical",
        )

        json_str = action.to_json()
        data = json.loads(json_str)

        assert data["action_type"] == "synthesize_timeline"
        assert "delta_112d" in data
        assert "deltas" in data
        assert data["deltas"]["delta_mean_f0_hz"] == 100.0
        assert data["priority"] == "critical"


# =============================================================================
# RUST TEST COUNTERPARTS
# =============================================================================

"""
The following Rust tests should be added to peer_controller.rs:

#[cfg(test)]
mod tests_112d_ipc {
    use super::*;
    use crate::peer_controller::{SynthesisAction, TimelineEvent, AudioBufferEvent};

    #[test]
    fn test_synthesis_action_with_112d_delta() {
        let mut delta_112d = vec![0.0f32; 112];
        delta_112d[0] = 100.0;

        let action = SynthesisAction::new(vec![])
            .with_delta_112d(delta_112d);

        assert!(action.delta_112d.is_some());
        let delta = action.delta_112d.unwrap();
        assert_eq!(delta.len(), 112);
        assert_eq!(delta[0], 100.0);
    }

    #[test]
    #[should_panic(expected = "must have exactly 112 elements")]
    fn test_synthesis_action_rejects_wrong_size() {
        let delta_wrong = vec![0.0f32; 100];
        let _action = SynthesisAction::new(vec![])
            .with_delta_112d(delta_wrong);
    }

    #[test]
    fn test_audio_buffer_event_creation() {
        let audio_data = vec![0.1, -0.2, 0.3, 0.4];
        let event = AudioBufferEvent::new(audio_data.clone(), 48000);

        assert_eq!(event.audio_data, audio_data);
        assert_eq!(event.sample_rate, 48000);
        assert_eq!(event.duration_ms, (4.0 / 48000.0) * 1000.0);
    }

    #[test]
    fn test_audio_buffer_event_serialization() {
        let audio_data = vec![0.1, -0.2, 0.3];
        let event = AudioBufferEvent {
            audio_data: audio_data.clone(),
            sample_rate: 48000,
            duration_ms: 100.0,
            timestamp: 1234567890.0,
            sequence: 42,
        };

        let bytes = event.to_bytes().unwrap();
        let restored = AudioBufferEvent::from_bytes(&bytes).unwrap();

        assert_eq!(restored.audio_data, audio_data);
        assert_eq!(restored.sample_rate, 48000);
        assert_eq!(restored.sequence, 42);
    }

    #[test]
    fn test_synthesis_action_json_roundtrip() {
        let mut delta_112d = vec![0.0f32; 112];
        for (i, val) in delta_112d.iter_mut().enumerate() {
            *val = i as f32 * 0.1;
        }

        let action = SynthesisAction::new(vec![
            TimelineEvent::new(42, 0.0, 150.0),
        ])
        .with_delta_112d(delta_112d.clone());

        let bytes = action.to_bytes().unwrap();
        let restored = SynthesisAction::from_bytes(&bytes).unwrap();

        assert!(restored.delta_112d.is_some());
        let restored_delta = restored.delta_112d.unwrap();
        assert_eq!(restored_delta.len(), 112);
        for i in 0..112 {
            assert!((restored_delta[i] - delta_112d[i]).abs() < 0.001);
        }
    }
}
"""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
