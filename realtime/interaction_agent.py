#!/usr/bin/env python3
"""
Interaction Agent - Closed-Loop Cognitive Agent
================================================

This module implements the main Interaction Agent that orchestrates the
closed-loop communication between the Rust Execution Layer and Python
Logic Layer.

The agent:
1. Subscribes to 112D feature events from Rust
2. Processes them through the CognitiveLayer for context detection
3. Generates synthesis timelines based on behavioral intent
4. Publishes synthesis actions back to Rust for audio output

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from realtime.action_publisher import (
    ActionPublisher,
    ActionPublisherConfig,
    MicroDynamicsDelta,
    TimelineEvent,
)

# Import components
from realtime.feature_subscriber import (
    FeatureEvent,
    FeatureSubscriber,
    FeatureSubscriberConfig,
)

logger = logging.getLogger(__name__)

# Default endpoints
FEATURES_ENDPOINT = os.environ.get("RUST_FEATURES_ENDPOINT", "ipc:///tmp/cognitive_features.ipc")
ACTIONS_ENDPOINT = os.environ.get("RUST_ACTIONS_ENDPOINT", "ipc:///tmp/cognitive_actions.ipc")


class AgentState(Enum):
    """State of the Interaction Agent"""

    IDLE = "idle"  # Not processing
    LISTENING = "listening"  # Receiving features, analyzing
    RESPONDING = "responding"  # Generating and sending synthesis


@dataclass
class InteractionAgentConfig:
    """Configuration for Interaction Agent"""

    feature_endpoint: str = FEATURES_ENDPOINT
    action_endpoint: str = ACTIONS_ENDPOINT
    response_cooldown_ms: float = 100.0
    max_responses_per_second: float = 5.0
    verbose_logging: bool = False


class InteractionAgent:
    """
    Main Interaction Agent for closed-loop communication.

    This agent bridges the Rust Execution Layer (feature extraction, synthesis)
    with the Python Logic Layer (cognitive processing, decision making).

    Usage:
        agent = InteractionAgent()
        agent.start()

        # Agent will automatically:
        # 1. Receive feature events from Rust
        # 2. Process through cognitive layer
        # 3. Send synthesis actions back to Rust

        # ... later ...
        agent.stop()
    """

    def __init__(
        self,
        config: Optional[InteractionAgentConfig] = None,
        on_feature_event: Optional[Callable[[FeatureEvent], None]] = None,
        on_context_change: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Initialize the Interaction Agent.

        Args:
            config: Agent configuration
            on_feature_event: Optional callback for feature events
            on_context_change: Optional callback for context changes
        """
        self.config = config or InteractionAgentConfig()
        self.on_feature_event = on_feature_event
        self.on_context_change = on_context_change

        # Initialize components
        self.feature_subscriber = FeatureSubscriber(
            config=FeatureSubscriberConfig(
                event_endpoint=self.config.feature_endpoint,
                verbose_logging=self.config.verbose_logging,
            ),
            on_event=self._handle_feature_event,
        )

        self.action_publisher = ActionPublisher(
            config=ActionPublisherConfig(
                action_endpoint=self.config.action_endpoint,
            ),
        )

        # State management
        self.state = AgentState.IDLE
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Context tracking
        self._current_context: Optional[str] = None
        self._context_confidence: float = 0.0
        self._last_response_time: float = 0.0

        # Statistics
        self._events_processed = 0
        self._responses_sent = 0
        self._start_time: Optional[float] = None

        logger.info("InteractionAgent initialized")
        logger.info(f"  Feature endpoint: {self.config.feature_endpoint}")
        logger.info(f"  Action endpoint: {self.config.action_endpoint}")

    def start(self) -> None:
        """Start the interaction agent"""
        if self._running:
            logger.warning("Agent already running")
            return

        logger.info("Starting Interaction Agent...")

        # Connect components
        self.feature_subscriber.connect()
        self.action_publisher.connect()

        # Start subscriber
        self.feature_subscriber.start()

        self._running = True
        self._start_time = time.time()
        self.state = AgentState.LISTENING

        logger.info("✓ Interaction Agent started")
        logger.info(f"  State: {self.state.value}")

    def stop(self) -> None:
        """Stop the interaction agent"""
        if not self._running:
            return

        logger.info("Stopping Interaction Agent...")

        self._running = False
        self.state = AgentState.IDLE

        # Stop components
        self.feature_subscriber.stop()
        self.action_publisher.disconnect()

        logger.info("✓ Interaction Agent stopped")

    def _handle_feature_event(self, event: FeatureEvent) -> None:
        """
        Handle a feature event from Rust.

        This is the main event handler that processes incoming features
        and generates synthesis responses.

        Args:
            event: FeatureEvent from Rust
        """
        self._events_processed += 1

        # Update state
        self.state = AgentState.LISTENING

        # Call optional callback
        if self.on_feature_event:
            try:
                self.on_feature_event(event)
            except Exception as e:
                logger.error(f"Error in feature event callback: {e}")

        # Process through cognitive layer
        result = self._process_features(event)

        # Check if we should respond
        if self._should_respond(result):
            self._send_response(result, event)

        # Check for context change
        if result.get("context_state") != self._current_context:
            old_context = self._current_context
            self._current_context = str(result.get("context_state", "unknown"))
            self._context_confidence = result.get("confidence", 0.0)

            if self.on_context_change:
                try:
                    self.on_context_change(self._current_context, self._context_confidence)
                except Exception as e:
                    logger.error(f"Error in context change callback: {e}")

            logger.debug(f"Context changed: {old_context} -> {self._current_context}")

    def _process_features(self, event: FeatureEvent) -> Dict[str, Any]:
        """
        Process features through cognitive layer.

        This method would integrate with the actual CognitiveLayer.
        For now, it provides a simplified implementation.

        Args:
            event: Feature event to process

        Returns:
            Processing result with context and synthesis parameters
        """
        # Simplified context inference from 112D features
        context = self._infer_context(event.features_112d)

        # Calculate confidence
        confidence = self._calculate_confidence(event.features_112d, context)

        return {
            "context_state": context,
            "confidence": confidence,
            "cluster_id": event.cluster_id,
            "sequence": event.sequence,
            "timestamp": event.timestamp,
            "features_112d": event.features_112d,
        }

    def _infer_context(self, features_112d: np.ndarray) -> str:
        """
        Infer behavioral context from 112D features.

        This is a simplified version. The full implementation would
        use the ProbabilisticContextMachine.

        Args:
            features_112d: 112D feature vector

        Returns:
            Context string
        """
        # Extract key features
        f0 = float(features_112d[0]) if features_112d[0] > 0 else 5000.0
        rms = float(features_112d[1]) if len(features_112d) > 1 else 0.5

        # Simple rule-based context inference
        if f0 > 8000 and rms > 0.6:
            return "alarm"
        elif f0 > 6000:
            return "territorial"
        elif f0 < 4000:
            return "social"
        else:
            return "contact"

    def _calculate_confidence(self, features_112d: np.ndarray, context: str) -> float:
        """Calculate confidence in context detection."""
        # Simple confidence based on feature variance
        variance = np.var(features_112d)
        return min(0.95, max(0.3, 0.5 + variance * 0.1))

    def _should_respond(self, result: Dict[str, Any]) -> bool:
        """
        Determine if the agent should generate a response.

        Args:
            result: Processing result

        Returns:
            True if should respond
        """
        # Check rate limiting
        current_time = time.time()
        time_since_last = (current_time - self._last_response_time) * 1000

        if time_since_last < self.config.response_cooldown_ms:
            return False

        # Check confidence threshold
        if result.get("confidence", 0.0) < 0.5:
            return False

        # Check context - some contexts require response
        context = result.get("context_state", "")
        response_contexts = {"contact", "alarm", "territorial"}

        return context in response_contexts

    def _send_response(self, result: Dict[str, Any], event: FeatureEvent) -> None:
        """
        Send synthesis response to Rust.

        Args:
            result: Processing result
            event: Original feature event
        """
        self.state = AgentState.RESPONDING

        # Create timeline
        cluster_id = result.get("cluster_id", event.cluster_id)
        context = result.get("context_state", "contact")

        timeline = self._create_response_timeline(cluster_id, context)
        deltas = self._create_deltas(context)

        # Send action
        success = self.action_publisher.publish_timeline(
            timeline=timeline,
            deltas=deltas,
            priority="normal",
        )

        if success:
            self._responses_sent += 1
            self._last_response_time = time.time()
            logger.debug(f"Sent response: cluster={cluster_id}, context={context}")

        # Return to listening
        self.state = AgentState.LISTENING

    def _create_response_timeline(
        self,
        cluster_id: int,
        context: str,
    ) -> List[TimelineEvent]:
        """Create synthesis timeline for response."""
        # Adjust timing based on context
        if context == "alarm":
            duration = 100.0
            amplitude = 0.9
        elif context == "territorial":
            duration = 200.0
            amplitude = 0.85
        else:  # contact
            duration = 150.0
            amplitude = 0.75

        return [
            TimelineEvent(
                cluster_id=cluster_id,
                start_time_ms=0.0,
                duration_ms=duration,
                amplitude=amplitude,
            )
        ]

    def _create_deltas(self, context: str) -> Optional[MicroDynamicsDelta]:
        """Create micro-dynamics deltas based on context."""
        if context == "alarm":
            return MicroDynamicsDelta(
                delta_mean_f0_hz=500.0,
                delta_rms_energy=0.2,
            )
        elif context == "territorial":
            return MicroDynamicsDelta(
                delta_mean_f0_hz=200.0,
                delta_duration_ms=20.0,
            )
        elif context == "social":
            return MicroDynamicsDelta(
                delta_mean_f0_hz=-100.0,
                delta_sustain_level=0.1,
            )
        else:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0.0

        return {
            "state": self.state.value,
            "running": self._running,
            "uptime_seconds": uptime,
            "events_processed": self._events_processed,
            "responses_sent": self._responses_sent,
            "current_context": self._current_context,
            "context_confidence": self._context_confidence,
            "events_per_second": self._events_processed / max(uptime, 1.0),
            "responses_per_second": self._responses_sent / max(uptime, 1.0),
            "feature_subscriber": self.feature_subscriber.get_stats(),
            "action_publisher": self.action_publisher.get_stats(),
        }

    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._running

    @property
    def current_state(self) -> AgentState:
        """Get current agent state."""
        return self.state

    @property
    def current_context(self) -> Optional[str]:
        """Get current detected context."""
        return self._current_context


def create_test_agent(
    feature_endpoint: str = "ipc:///tmp/test_features.ipc",
    action_endpoint: str = "ipc:///tmp/test_actions.ipc",
) -> InteractionAgent:
    """
    Create a test agent with custom endpoints.

    Args:
        feature_endpoint: ZeroMQ endpoint for feature events
        action_endpoint: ZeroMQ endpoint for action commands

    Returns:
        Configured InteractionAgent
    """
    config = InteractionAgentConfig(
        feature_endpoint=feature_endpoint,
        action_endpoint=action_endpoint,
        verbose_logging=True,
    )
    return InteractionAgent(config=config)


if __name__ == "__main__":
    # Demo/test mode
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("Interaction Agent - Closed-Loop Cognitive Agent")
    print("=" * 60)

    def on_feature(event: FeatureEvent):
        print(f"[EVENT] Cluster {event.cluster_id}, Seq {event.sequence}")

    def on_context(context: str, confidence: float):
        print(f"[CONTEXT] {context} ({confidence:.2f})")

    agent = InteractionAgent(
        on_feature_event=on_feature,
        on_context_change=on_context,
    )

    print("\nStarting agent (Ctrl+C to stop)...")
    try:
        agent.start()

        # Print stats every 5 seconds
        while True:
            time.sleep(5.0)
            stats = agent.get_stats()
            print("\n--- Stats ---")
            print(f"State: {stats['state']}")
            print(f"Events: {stats['events_processed']}")
            print(f"Responses: {stats['responses_sent']}")
            print(f"Context: {stats['current_context']} ({stats['context_confidence']:.2f})")

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        agent.stop()
        print("\nFinal Stats:")
        for k, v in agent.get_stats().items():
            print(f"  {k}: {v}")
