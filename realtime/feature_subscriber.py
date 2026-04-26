#!/usr/bin/env python3
"""
Feature Event Subscriber - Python Layer
========================================

Subscribes to feature extraction events from Rust Execution Layer
and dispatches them to the cognitive processing pipeline.

This module implements the Python side of the Closed-Loop Interaction Agent,
receiving 112D feature vectors and cluster IDs from the Rust NBD/feature
extraction pipeline.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Configuration defaults
FEATURES_ENDPOINT = os.environ.get("RUST_FEATURES_ENDPOINT", "ipc:///tmp/cognitive_features.ipc")


@dataclass
class FeatureEvent:
    """Feature extraction event from Rust"""

    event_type: str
    cluster_id: int
    features_112d: np.ndarray
    timestamp: float
    sequence: int
    emitter_id: Optional[int] = None

    @classmethod
    def from_json(cls, data: dict) -> "FeatureEvent":
        """Deserialize from JSON dictionary"""
        return cls(
            event_type=data["event_type"],
            cluster_id=data["cluster_id"],
            features_112d=np.array(data["features_112d"], dtype=np.float32),
            timestamp=data["timestamp"],
            sequence=data["sequence"],
            emitter_id=data.get("emitter_id"),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "FeatureEvent":
        """Deserialize from JSON bytes"""
        return cls.from_json(json.loads(data.decode("utf-8")))

    def to_json_dict(self) -> dict:
        """Serialize to JSON-compatible dictionary"""
        result = {
            "event_type": self.event_type,
            "cluster_id": self.cluster_id,
            "features_112d": self.features_112d.tolist(),
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }
        if self.emitter_id is not None:
            result["emitter_id"] = self.emitter_id
        return result

    def __repr__(self) -> str:
        return (
            f"FeatureEvent(cluster={self.cluster_id}, "
            f"seq={self.sequence}, time={self.timestamp:.3f})"
        )


@dataclass
class FeatureSubscriberConfig:
    """Configuration for feature subscriber"""

    event_endpoint: str = FEATURES_ENDPOINT
    receive_timeout_ms: int = 100
    receive_high_water_mark: int = 100
    verbose_logging: bool = False


class FeatureSubscriber:
    """
    ZeroMQ subscriber for feature events from Rust.

    Connects to the Rust Execution Layer and receives feature extraction
    events in real-time.

    Usage:
        def on_event(event: FeatureEvent):
            print(f"Received cluster {event.cluster_id}")

        subscriber = FeatureSubscriber(on_event=on_event)
        subscriber.connect()
        subscriber.start()

        # ... later ...
        subscriber.stop()
    """

    def __init__(
        self,
        config: Optional[FeatureSubscriberConfig] = None,
        on_event: Optional[Callable[[FeatureEvent], None]] = None,
    ):
        """
        Initialize feature subscriber.

        Args:
            config: Subscriber configuration (uses defaults if None)
            on_event: Callback for received events
        """
        self.config = config or FeatureSubscriberConfig()
        self.on_event = on_event

        self._context: Optional[Any] = None
        self._socket: Optional[Any] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Statistics
        self._events_received = 0
        self._last_sequence = 0
        self._last_timestamp: Optional[float] = None

        logger.info(f"FeatureSubscriber initialized for {self.config.event_endpoint}")

    def connect(self) -> None:
        """Connect to the Rust feature publisher"""
        try:
            import zmq
        except ImportError:
            logger.error("ZeroMQ not installed. Install with: pip install pyzmq")
            raise

        logger.info(f"Connecting to Rust Feature Publisher: {self.config.event_endpoint}")

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)

        # Set socket options
        self._socket.setsockopt(zmq.LINGER, 1000)
        self._socket.setsockopt(zmq.RCVHWM, self.config.receive_high_water_mark)
        self._socket.setsockopt(zmq.RCVTIMEO, self.config.receive_timeout_ms)

        # Connect and subscribe
        self._socket.connect(self.config.event_endpoint)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")

        logger.info("✓ Connected to Rust Feature Publisher")

    def disconnect(self) -> None:
        """Disconnect from the Rust feature publisher"""
        self._running = False

        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

        logger.info("✓ Disconnected from Rust Feature Publisher")

    def start(self) -> None:
        """Start the subscriber thread"""
        if not self._socket:
            self.connect()

        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

        logger.info("Feature subscriber thread started")

    def stop(self) -> None:
        """Stop the subscriber thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.disconnect()

    def _receive_loop(self) -> None:
        """Main receive loop"""
        try:
            import zmq
        except ImportError:
            logger.error("ZeroMQ not available in receive loop")
            return

        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)

        while self._running:
            try:
                # Poll with timeout
                if poller.poll(self.config.receive_timeout_ms):
                    message = self._socket.recv(zmq.DONTWAIT)
                    event = FeatureEvent.from_bytes(message)

                    # Update statistics
                    self._events_received += 1
                    self._last_sequence = event.sequence
                    self._last_timestamp = event.timestamp

                    # Dispatch to callback
                    if self.on_event:
                        self.on_event(event)

                    if self.config.verbose_logging and self._events_received % 100 == 0:
                        logger.debug(f"Received {self._events_received} events")

            except zmq.ZMQError as e:
                if self._running:
                    logger.error(f"ZeroMQ error in receive loop: {e}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON message: {e}")
            except KeyError as e:
                logger.error(f"Missing required field in event: {e}")
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get subscriber statistics"""
        return {
            "events_received": self._events_received,
            "last_sequence": self._last_sequence,
            "last_timestamp": self._last_timestamp,
            "endpoint": self.config.event_endpoint,
            "running": self._running,
        }

    def is_running(self) -> bool:
        """Check if subscriber is running"""
        return self._running

    @property
    def events_received(self) -> int:
        """Number of events received"""
        return self._events_received


# Convenience function for creating a test subscriber
def create_test_subscriber(
    on_event: Callable[[FeatureEvent], None],
    endpoint: str = "ipc:///tmp/test_features.ipc",
) -> FeatureSubscriber:
    """
    Create a test subscriber with custom endpoint.

    Args:
        on_event: Callback for received events
        endpoint: ZeroMQ endpoint to connect to

    Returns:
        Configured FeatureSubscriber
    """
    config = FeatureSubscriberConfig(
        event_endpoint=endpoint,
        verbose_logging=True,
    )
    return FeatureSubscriber(config=config, on_event=on_event)


if __name__ == "__main__":
    # Demo/test mode
    logging.basicConfig(level=logging.INFO)

    def print_event(event: FeatureEvent):
        print(f"Event: {event}")

    subscriber = FeatureSubscriber(
        config=FeatureSubscriberConfig(verbose_logging=True),
        on_event=print_event,
    )

    print("Starting subscriber (Ctrl+C to stop)...")
    try:
        subscriber.connect()
        subscriber.start()

        # Run until interrupted
        while True:
            time.sleep(1.0)
            stats = subscriber.get_stats()
            print(f"Stats: {stats}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        subscriber.stop()
