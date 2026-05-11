#!/usr/bin/env python3
"""
Real-Time Dual-Stream Pipeline (Sprint 4)

Updates for real-time dual-stream integration:
- ZMQ IPC structures for DualStreamState
- Feature split extraction (30D affective + 44D syntactic)
- Integration with existing InteractionAgent

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import dataclasses
import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import zmq

from cognitive_intelligence.affective_response import AffectiveResponsePolicy
from cognitive_intelligence.syntax_graph import SyntaxGraph

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DualStreamAction:
    """
    Dual-stream synthesis action.

    Sent from Python Cognitive Agent to Rust Synthesis Engine.
    """
    # Discrete syntactic token from VQ-VAE
    syntactic_token: int

    # 16D continuous affect vector from β-VAE
    affect_vector: np.ndarray

    # Timing parameters
    temporal_offset_ms: float = 0.0
    duration_ms: float = 200.0

    # Metadata
    priority: str = "normal"  # low, normal, high, urgent
    sequence: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "syntactic_token": self.syntactic_token,
            "affect_vector": self.affect_vector.tolist(),
            "temporal_offset_ms": self.temporal_offset_ms,
            "duration_ms": self.duration_ms,
            "priority": self.priority,
            "sequence": self.sequence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DualStreamAction":
        """Create from dictionary."""
        return cls(
            syntactic_token=data["syntactic_token"],
            affect_vector=np.array(data["affect_vector"], dtype=np.float32),
            temporal_offset_ms=data.get("temporal_offset_ms", 0.0),
            duration_ms=data.get("duration_ms", 200.0),
            priority=data.get("priority", "normal"),
            sequence=data.get("sequence", 0),
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "DualStreamAction":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclasses.dataclass
class DualStreamState:
    """
    Dual-stream state received from Rust.

    Contains both continuous affect and discrete syntax from
    the 112D Rosetta feature extraction.
    """
    # 16D continuous affect vector from β-VAE
    affect_vector: np.ndarray

    # Discrete syntactic token from VQ-VAE (0-63)
    syntactic_token: int

    # Raw 112D features for fallback/validation
    raw_features: np.ndarray

    # Confidence scores
    affect_confidence: float = 1.0
    syntactic_confidence: float = 1.0

    # Metadata
    timestamp_ms: float = 0.0
    sequence: int = 0
    source_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "affect_vector": self.affect_vector.tolist(),
            "syntactic_token": self.syntactic_token,
            "raw_features": self.raw_features.tolist(),
            "affect_confidence": self.affect_confidence,
            "syntactic_confidence": self.syntactic_confidence,
            "timestamp_ms": self.timestamp_ms,
            "sequence": self.sequence,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DualStreamState":
        """Create from dictionary."""
        return cls(
            affect_vector=np.array(data["affect_vector"], dtype=np.float32),
            syntactic_token=data["syntactic_token"],
            raw_features=np.array(data["raw_features"], dtype=np.float32),
            affect_confidence=data.get("affect_confidence", 1.0),
            syntactic_confidence=data.get("syntactic_confidence", 1.0),
            timestamp_ms=data.get("timestamp_ms", 0.0),
            sequence=data.get("sequence", 0),
            source_id=data.get("source_id", ""),
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "DualStreamState":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class DualStreamStateSubscriber:
    """
    Subscribe to DualStreamState from Rust via ZMQ.

    Receives dual-stream encoded state from the Rust execution layer.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        topic: str = b"dual_stream_state",
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, topic)

        self.topic = topic
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        logger.info(f"DualStreamStateSubscriber connected to {host}:{port}")

    def receive(
        self,
        timeout_ms: int = 100,
    ) -> Optional[DualStreamState]:
        """
        Receive DualStreamState with timeout.

        Returns:
            DualStreamState if received, None if timeout
        """
        socks = dict(self.poller.poll(timeout_ms))

        if self.socket in socks and socks[self.socket] == zmq.POLLIN:
            topic, data = self.socket.recv_multipart()

            if topic == self.topic:
                try:
                    state = DualStreamState.from_json(data.decode())
                    return state
                except Exception as e:
                    logger.error(f"Failed to parse DualStreamState: {e}")

        return None

    def close(self) -> None:
        """Close the subscriber."""
        self.poller.unregister(self.socket)
        self.socket.close()
        self.context.term()


class DualStreamActionPublisher:
    """
    Publish DualStreamAction to Rust via ZMQ.

    Sends dual-stream synthesis actions to the Rust execution layer.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://{host}:{port}")

        logger.info(f"DualStreamActionPublisher bound to {host}:{port}")

    def publish(self, action: DualStreamAction) -> None:
        """Publish a DualStreamAction."""
        self.socket.send_string(action.to_json())

    def close(self) -> None:
        """Close the publisher."""
        self.socket.close()
        self.context.term()


class DualStreamInteractionAgent:
    """
    InteractionAgent v2.0 with dual-stream processing.

    Combines continuous affect and discrete syntax to generate
    biologically-appropriate responses.
    """

    def __init__(
        self,
        syntax_graph: Optional[SyntaxGraph] = None,
        affective_policy: Optional[AffectiveResponsePolicy] = None,
    ):
        self.syntax_graph = syntax_graph or SyntaxGraph()
        self.affective_policy = affective_policy or AffectiveResponsePolicy()

        # Track conversation state
        self.last_token: Optional[int] = None
        self.conversation_sequence: int = 0

        logger.info("DualStreamInteractionAgent initialized")

    def handle_dual_stream_state(
        self,
        state: DualStreamState,
    ) -> DualStreamAction:
        """
        Process dual-stream state and generate response action.

        Pipeline:
        1. Validate syntax (get valid next tokens)
        2. Compute affective response
        3. Select syntactic token
        4. Generate action
        """
        # Stream 2: Get valid next syntactic tokens
        valid_next = self.syntax_graph.get_valid_next_tokens(
            state.syntactic_token,
            top_k=5,
        )

        if not valid_next:
            # Fallback: any token is valid
            valid_next = [(i, 1.0/64) for i in range(64)]

        # Stream 1: Compute affective response
        target_affect = self.affective_policy.compute_target_affect(
            state.affect_vector
        )

        # Select syntactic token (highest probability)
        response_token = valid_next[0][0]

        # Update state
        self.last_token = response_token
        self.conversation_sequence += 1

        # Create action
        action = DualStreamAction(
            syntactic_token=response_token,
            affect_vector=target_affect,
            temporal_offset_ms=150.0,  # Biological response latency
            sequence=self.conversation_sequence,
        )

        logger.debug(
            f"Generated action: token={response_token}, "
            f"affect_arousal={target_affect[0]:.2f}"
        )

        return action

    def set_syntax_graph(self, syntax_graph: SyntaxGraph) -> None:
        """Update the syntax graph."""
        self.syntax_graph = syntax_graph

    def set_affective_policy(
        self,
        policy: AffectiveResponsePolicy,
    ) -> None:
        """Update the affective response policy."""
        self.affective_policy = policy


# =============================================================================
# Convenience functions
# =============================================================================

def create_dual_stream_pipeline(
    syntax_graph: Optional[SyntaxGraph] = None,
    state_host: str = "localhost",
    state_port: int = 5555,
    action_host: str = "localhost",
    action_port: int = 5556,
) -> Tuple[DualStreamStateSubscriber, DualStreamActionPublisher, DualStreamInteractionAgent]:
    """
    Create a complete dual-stream pipeline.

    Returns:
        (subscriber, publisher, agent)
    """
    subscriber = DualStreamStateSubscriber(state_host, state_port)
    publisher = DualStreamActionPublisher(action_host, action_port)
    agent = DualStreamInteractionAgent(syntax_graph)

    return subscriber, publisher, agent


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test the data structures
    affect = np.random.randn(16).astype(np.float32)
    raw = np.random.randn(112).astype(np.float32)

    state = DualStreamState(
        affect_vector=affect,
        syntactic_token=5,
        raw_features=raw,
        sequence=1,
    )

    # Test serialization
    json_str = state.to_json()
    state2 = DualStreamState.from_json(json_str)

    print(f"Original token: {state.syntactic_token}")
    print(f"Deserialized token: {state2.syntactic_token}")
    print(f"Match: {np.allclose(state.affect_vector, state2.affect_vector)}")

    # Test agent
    agent = DualStreamInteractionAgent()
    action = agent.handle_dual_stream_state(state)

    print(f"Action token: {action.syntactic_token}")
    print(f"Action affect shape: {action.affect_vector.shape}")
