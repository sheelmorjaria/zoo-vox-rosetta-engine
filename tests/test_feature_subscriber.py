#!/usr/bin/env python3
"""
Tests for Feature Event Subscriber (Python Layer)

These tests verify that Python can receive and deserialize feature events
from the Rust Execution Layer via ZeroMQ.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import unittest
from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureEvent:
    """Feature extraction event from Rust"""

    event_type: str
    cluster_id: int
    features_112d: np.ndarray
    timestamp: float
    sequence: int

    @classmethod
    def from_json(cls, data: dict) -> "FeatureEvent":
        """Deserialize from JSON dictionary"""
        return cls(
            event_type=data["event_type"],
            cluster_id=data["cluster_id"],
            features_112d=np.array(data["features_112d"], dtype=np.float32),
            timestamp=data["timestamp"],
            sequence=data["sequence"],
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "FeatureEvent":
        """Deserialize from JSON bytes"""
        return cls.from_json(json.loads(data.decode("utf-8")))

    def to_json_dict(self) -> dict:
        """Serialize to JSON-compatible dictionary"""
        return {
            "event_type": self.event_type,
            "cluster_id": self.cluster_id,
            "features_112d": self.features_112d.tolist(),
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }


class TestFeatureEventDeserialization(unittest.TestCase):
    """Test feature event deserialization from Rust"""

    def test_feature_event_from_json(self):
        """Should deserialize feature event from JSON"""
        json_data = {
            "event_type": "feature_extraction",
            "cluster_id": 42,
            "features_112d": [0.0] * 112,
            "timestamp": 1699345823.456,
            "sequence": 12345,
        }

        event = FeatureEvent.from_json(json_data)

        self.assertEqual(event.event_type, "feature_extraction")
        self.assertEqual(event.cluster_id, 42)
        self.assertEqual(len(event.features_112d), 112)
        self.assertEqual(event.timestamp, 1699345823.456)
        self.assertEqual(event.sequence, 12345)

    def test_feature_event_112d_array_shape(self):
        """Features array should have 112 dimensions"""
        json_data = {
            "event_type": "feature_extraction",
            "cluster_id": 1,
            "features_112d": list(range(112)),
            "timestamp": 0.0,
            "sequence": 1,
        }

        event = FeatureEvent.from_json(json_data)

        self.assertEqual(event.features_112d.shape, (112,))
        self.assertEqual(event.features_112d[0], 0.0)
        self.assertEqual(event.features_112d[111], 111.0)

    def test_feature_event_from_bytes(self):
        """Should deserialize from JSON bytes"""
        json_bytes = json.dumps(
            {
                "event_type": "feature_extraction",
                "cluster_id": 99,
                "features_112d": [0.5] * 112,
                "timestamp": 12345.0,
                "sequence": 1,
            }
        ).encode("utf-8")

        event = FeatureEvent.from_bytes(json_bytes)

        self.assertEqual(event.cluster_id, 99)
        self.assertEqual(event.features_112d.shape, (112,))
        self.assertAlmostEqual(event.features_112d[0], 0.5, places=5)

    def test_feature_event_to_json_dict(self):
        """Should serialize to JSON-compatible dict"""
        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.zeros(112, dtype=np.float32),
            timestamp=1000.0,
            sequence=1,
        )

        data = event.to_json_dict()

        self.assertEqual(data["event_type"], "feature_extraction")
        self.assertEqual(data["cluster_id"], 42)
        self.assertEqual(len(data["features_112d"]), 112)
        self.assertEqual(data["timestamp"], 1000.0)
        self.assertEqual(data["sequence"], 1)

    def test_feature_event_roundtrip(self):
        """Should roundtrip through JSON serialization"""
        original = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.random.randn(112).astype(np.float32),
            timestamp=12345.0,
            sequence=1,
        )

        # Serialize
        json_dict = original.to_json_dict()
        json_bytes = json.dumps(json_dict).encode("utf-8")

        # Deserialize
        decoded = FeatureEvent.from_bytes(json_bytes)

        self.assertEqual(decoded.cluster_id, original.cluster_id)
        self.assertEqual(decoded.sequence, original.sequence)
        self.assertEqual(decoded.timestamp, original.timestamp)
        np.testing.assert_array_almost_equal(
            decoded.features_112d, original.features_112d, decimal=5
        )

    def test_feature_event_float32_dtype(self):
        """Features should be float32"""
        json_data = {
            "event_type": "feature_extraction",
            "cluster_id": 1,
            "features_112d": [1.0] * 112,
            "timestamp": 0.0,
            "sequence": 1,
        }

        event = FeatureEvent.from_json(json_data)

        self.assertEqual(event.features_112d.dtype, np.float32)


class TestFeatureSubscriberConfig(unittest.TestCase):
    """Test feature subscriber configuration"""

    def test_default_config(self):
        """Default config should have correct endpoint"""
        from realtime.feature_subscriber import FeatureSubscriberConfig

        config = FeatureSubscriberConfig()

        self.assertEqual(config.event_endpoint, "ipc:///tmp/cognitive_features.ipc")
        self.assertEqual(config.receive_timeout_ms, 100)
        self.assertEqual(config.receive_high_water_mark, 100)

    def test_custom_config(self):
        """Should accept custom configuration"""
        from realtime.feature_subscriber import FeatureSubscriberConfig

        config = FeatureSubscriberConfig(
            event_endpoint="tcp://localhost:5555",
            receive_timeout_ms=200,
            receive_high_water_mark=50,
        )

        self.assertEqual(config.event_endpoint, "tcp://localhost:5555")
        self.assertEqual(config.receive_timeout_ms, 200)
        self.assertEqual(config.receive_high_water_mark, 50)


class TestFeatureSubscriberStats(unittest.TestCase):
    """Test feature subscriber statistics"""

    def test_initial_stats(self):
        """Initial stats should be zero"""
        from realtime.feature_subscriber import FeatureSubscriber

        subscriber = FeatureSubscriber()
        stats = subscriber.get_stats()

        self.assertEqual(stats["events_received"], 0)
        self.assertEqual(stats["last_sequence"], 0)
        self.assertIsNone(stats["last_timestamp"])


class TestFeatureEventUncertainty(unittest.TestCase):
    """Test uncertainty field on FeatureEvent from the module"""

    def test_feature_event_optional_uncertainty(self):
        """FeatureEvent accepts uncertainty field"""
        import numpy as np

        from realtime.feature_subscriber import FeatureEvent

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.random.randn(112).astype(np.float32),
            timestamp=1000.0,
            sequence=1,
            uncertainty=0.3,
        )

        self.assertEqual(event.uncertainty, 0.3)

    def test_feature_event_uncertainty_none_by_default(self):
        """Uncertainty defaults to None"""
        import numpy as np

        from realtime.feature_subscriber import FeatureEvent

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.random.randn(112).astype(np.float32),
            timestamp=1000.0,
            sequence=1,
        )

        self.assertIsNone(event.uncertainty)

    def test_feature_event_uncertainty_propagation(self):
        """Uncertainty propagates through deserialization"""

        from realtime.feature_subscriber import FeatureEvent

        json_data = {
            "event_type": "feature_extraction",
            "cluster_id": 42,
            "features_112d": [0.0] * 112,
            "timestamp": 1000.0,
            "sequence": 1,
            "uncertainty": 0.7,
        }

        event = FeatureEvent.from_json(json_data)

        self.assertEqual(event.uncertainty, 0.7)

    def test_feature_event_serialization_with_uncertainty(self):
        """JSON roundtrip preserves uncertainty"""
        import numpy as np

        from realtime.feature_subscriber import FeatureEvent

        original = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.random.randn(112).astype(np.float32),
            timestamp=1000.0,
            sequence=1,
            uncertainty=0.45,
        )

        # Serialize
        json_dict = original.to_json_dict()
        json_bytes = json.dumps(json_dict).encode("utf-8")

        # Deserialize
        decoded = FeatureEvent.from_bytes(json_bytes)

        self.assertEqual(decoded.uncertainty, 0.45)
        self.assertAlmostEqual(decoded.uncertainty, original.uncertainty, places=5)

    def test_feature_event_serialization_without_uncertainty(self):
        """Uncertainty is omitted from JSON when None"""
        import numpy as np

        from realtime.feature_subscriber import FeatureEvent

        event = FeatureEvent(
            event_type="feature_extraction",
            cluster_id=42,
            features_112d=np.random.randn(112).astype(np.float32),
            timestamp=1000.0,
            sequence=1,
            uncertainty=None,
        )

        json_dict = event.to_json_dict()

        # Uncertainty should not be in the dict when None
        self.assertNotIn("uncertainty", json_dict)


if __name__ == "__main__":
    unittest.main()
