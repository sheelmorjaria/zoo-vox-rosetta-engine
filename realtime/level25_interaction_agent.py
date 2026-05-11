#!/usr/bin/env python3
"""
Level 2.5 Interaction Agent

Extends the InteractionAgent to consume Level25Context with spatial-social
information. Handles broadcast vs unicast decision logic and spatial routing.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fusion_intelligence.receiver_inference import (
    CallDirectionality,
    Level25Context,
    ReceiverInferenceEngine,
)

from realtime.action_publisher import (
    ActionPublisher,
    DualStreamAction,
)

logger = logging.getLogger(__name__)


class ResponseStrategy(Enum):
    """Strategy for generating responses."""
    BROADCAST_CHORUS = "broadcast_chorus"  # Join the chorus (e.g., alarm call)
    BROADCAST_IGNORE = "broadcast_ignore"  # Ignore (not relevant to agent)
    UNICAST_MATCH = "unicast_match"  # Direct response to specific emitter
    UNICAST_DEFER = "unicast_defer"  # Defer to higher probability receiver
    UNICAST_DECLINE = "unicast_decline"  # Decline interaction (low affinity)


@dataclass
class Level25Action:
    """
    Level 2.5 synthesis action with spatial routing metadata.

    Extends DualStreamAction with spatial targeting for directional rendering.
    """
    syntactic_token: int
    affect_vector: np.ndarray  # 16D affect
    temporal_offset_ms: float = 150.0
    priority: str = "normal"
    sequence: int = 0

    # Level 2.5 spatial metadata
    call_directionality: CallDirectionality = CallDirectionality.BROADCAST
    target_spatial_id: Optional[str] = None  # For unicast routing
    broadcast_flag: bool = True  # For spatial rendering
    emitter_position: Optional[np.ndarray] = None  # [x, y, z]
    target_position: Optional[np.ndarray] = None  # [x, y, z]

    def to_dual_stream_action(self) -> DualStreamAction:
        """Convert to DualStreamAction for backward compatibility."""
        return DualStreamAction(
            syntactic_token=self.syntactic_token,
            affect_vector=self.affect_vector,
            temporal_offset_ms=self.temporal_offset_ms,
            priority=self.priority,
            sequence=self.sequence,
        )


@dataclass
class ResponsePolicy:
    """
    Response policy for different call types and social contexts.

    Defines how the agent should respond to various situations:
    - Alarm calls (high arousal broadcast)
    - Mating calls (directed unicast)
    - Territorial calls (directed with aggression)
    - Contact calls (low arousal broadcast)
    """
    respond_to_alarm: bool = True  # Join alarm chorus
    respond_to_mating: bool = True  # Respond to mating calls
    respond_to_territorial: bool = False  # Avoid aggressive interactions
    respond_to_contact: bool = True  # Acknowledge contact calls

    # Arousal thresholds for classification
    alarm_arousal_threshold: float = 0.7
    contact_arousal_threshold: float = 0.3

    def should_respond(self, context: Level25Context) -> ResponseStrategy:
        """
        Determine if and how to respond to a given context.

        Returns:
            ResponseStrategy indicating how to respond
        """
        arousal = context.affect_vector[0] if len(context.affect_vector) > 0 else 0.0

        # High arousal = alarm or distress
        if arousal > self.alarm_arousal_threshold:
            if self.respond_to_alarm and context.call_directionality == CallDirectionality.BROADCAST:
                return ResponseStrategy.BROADCAST_CHORUS

        # Low arousal = contact call
        elif arousal < self.contact_arousal_threshold:
            if self.respond_to_contact:
                if context.call_directionality == CallDirectionality.BROADCAST:
                    return ResponseStrategy.BROADCAST_CHORUS
                elif context.call_directionality == CallDirectionality.UNICAST:
                    # Check if agent is a likely receiver
                    top_receivers = context.get_top_receivers(top_k=3)
                    # (In real deployment, would check if our agent_id is in top receivers)
                    # For now, respond to directed low-arousal calls
                    return ResponseStrategy.UNICAST_MATCH

        # Directed call with high probability target
        if context.call_directionality == CallDirectionality.UNICAST:
            if context.receiver_probabilities:
                top_prob = max(context.receiver_probabilities.values())
                if top_prob > 0.8:
                    # Strongly directed - check if we should respond
                    if self.respond_to_mating:
                        return ResponseStrategy.UNICAST_MATCH
                    elif self.respond_to_territorial:
                        return ResponseStrategy.UNICAST_DECLINE

        # Default: ignore
        return ResponseStrategy.BROADCAST_IGNORE


class Level25InteractionAgent:
    """
    Level 2.5 Interaction Agent with spatial-social awareness.

    This agent extends the base InteractionAgent to:
    1. Consume Level25Context with receiver inference
    2. Apply response policies based on call directionality
    3. Generate spatially-routed synthesis actions
    4. Handle broadcast vs unicast decision logic
    """

    def __init__(
        self,
        receiver_inference_engine: ReceiverInferenceEngine,
        response_policy: Optional[ResponsePolicy] = None,
        action_publisher: Optional[ActionPublisher] = None,
        agent_id: str = "cognitive_agent",
    ):
        self.inference_engine = receiver_inference_engine
        self.policy = response_policy or ResponsePolicy()
        self.action_publisher = action_publisher
        self.agent_id = agent_id

        # Social affinity tracking
        self.social_affinity: Dict[str, float] = {}

        # Response statistics
        self.broadcast_responses = 0
        self.unicast_responses = 0
        self.ignored_calls = 0

        logger.info(f"Level25InteractionAgent initialized (agent_id={agent_id})")

    def handle_level_25_context(self, context: Level25Context) -> Optional[Level25Action]:
        """
        Process Level 2.5 context and generate appropriate response.

        Args:
            context: Level25Context with acoustic and spatial data

        Returns:
            Level25Action if responding, None if ignoring
        """
        # Determine response strategy
        strategy = self.policy.should_respond(context)

        if strategy == ResponseStrategy.BROADCAST_IGNORE:
            self.ignored_calls += 1
            logger.debug(f"Ignoring broadcast call from {context.emitter_id}")
            return None

        elif strategy == ResponseStrategy.BROADCAST_CHORUS:
            self.broadcast_responses += 1
            return self._generate_broadcast_response(context)

        elif strategy == ResponseStrategy.UNICAST_MATCH:
            self.unicast_responses += 1
            return self._generate_unicast_response(context)

        elif strategy == ResponseStrategy.UNICAST_DECLINE:
            self.ignored_calls += 1
            logger.debug(f"Declining directed call from {context.emitter_id}")
            return None

        else:
            return None

    def _generate_broadcast_response(self, context: Level25Context) -> Level25Action:
        """
        Generate a broadcast response (e.g., joining an alarm chorus).

        For broadcast calls, the agent responds from its own position
        rather than targeting a specific receiver.
        """
        # Match arousal but slightly de-escalate to prevent panic cascade
        target_affect = self._compute_affective_response(context.affect_vector)

        # Select syntactic token (would use syntax_graph in full implementation)
        response_token = context.syntactic_token  # Echo or use valid next

        return Level25Action(
            syntactic_token=response_token,
            affect_vector=target_affect,
            temporal_offset_ms=150.0,
            call_directionality=CallDirectionality.BROADCAST,
            broadcast_flag=True,
        )

    def _generate_unicast_response(self, context: Level25Context) -> Level25Action:
        """
        Generate a directed response to a specific emitter.

        For unicast calls, the response targets the specific emitter
        who initiated the interaction.
        """
        # Match or de-escalate affect based on social context
        target_affect = self._compute_affective_response(context.affect_vector)

        # Select syntactic response token
        response_token = context.syntactic_token  # Would use syntax_graph

        # Get top receiver for spatial targeting
        top_receivers = context.get_top_receivers(top_k=1)
        target_id = top_receivers[0][0] if top_receivers else context.emitter_id

        return Level25Action(
            syntactic_token=response_token,
            affect_vector=target_affect,
            temporal_offset_ms=150.0,
            call_directionality=CallDirectionality.UNICAST,
            target_spatial_id=context.emitter_id,  # Respond to emitter
            broadcast_flag=False,
        )

    def _compute_affective_response(self, incoming_affect: np.ndarray) -> np.ndarray:
        """
        Compute target affect based on incoming affect.

        De-escalate high arousal to prevent panic cascade.
        Match low arousal for social bonding.
        """
        if len(incoming_affect) == 0:
            return np.zeros(16, dtype=np.float32)

        arousal = incoming_affect[0]

        # De-escalate high arousal
        if arousal > 0.8:
            return incoming_affect * 0.75
        # Escalate slightly for engagement
        elif arousal < 0.3:
            return incoming_affect * 1.2
        # Match for social bonding
        else:
            return incoming_affect.copy()

    def publish_action(self, action: Level25Action) -> bool:
        """
        Publish a Level25Action to the synthesis layer.

        Returns:
            True if published successfully
        """
        if self.action_publisher is None:
            logger.warning("No action publisher configured")
            return False

        # Convert to DualStreamAction for compatibility
        ds_action = action.to_dual_stream_action()

        # Add spatial metadata to the publish call
        # (In full implementation, would extend the publish interface)
        return self.action_publisher.publish_dual_stream(ds_action)

    def get_statistics(self) -> Dict[str, Any]:
        """Get response statistics."""
        total = self.broadcast_responses + self.unicast_responses + self.ignored_calls
        return {
            "broadcast_responses": self.broadcast_responses,
            "unicast_responses": self.unicast_responses,
            "ignored_calls": self.ignored_calls,
            "total_processed": total,
            "response_rate": (self.broadcast_responses + self.unicast_responses) / max(total, 1),
        }


class Level25Orchestrator:
    """
    Orchestrates the full Level 2.5 pipeline.

    Connects spatial topology, receiver inference, and interaction agent
    for complete spatial-social cognitive processing.
    """

    def __init__(
        self,
        inference_engine: ReceiverInferenceEngine,
        interaction_agent: Level25InteractionAgent,
    ):
        self.inference_engine = inference_engine
        self.agent = interaction_agent

        # Pipeline statistics
        self.events_processed = 0
        self.responses_generated = 0

        logger.info("Level25Orchestrator initialized")

    def process_acoustic_event(
        self,
        emitter_id: str,
        syntactic_token: int,
        affect_vector: np.ndarray,
        topology,  # TopologyEngine
        timestamp_ns: int,
    ) -> Optional[Level25Action]:
        """
        Full pipeline: acoustic event -> receiver inference -> response generation.

        Args:
            emitter_id: ID of vocalizing agent
            syntactic_token: Syntactic token from VQ-VAE
            affect_vector: 16D affect vector from VAE
            topology: Current spatial topology
            timestamp_ns: Event timestamp

        Returns:
            Level25Action if responding, None if ignoring
        """
        self.events_processed += 1

        # Step 1: Infer receivers
        context = self.inference_engine.infer_receiver(
            emitter_id=emitter_id,
            topology=topology,
            syntactic_token=syntactic_token,
            affect_vector=affect_vector,
            timestamp_ns=timestamp_ns,
        )

        # Step 2: Generate response based on policy
        action = self.agent.handle_level_25_context(context)

        if action is not None:
            self.responses_generated += 1
            logger.info(
                f"Generated {action.call_directionality.value} response "
                f"to {context.emitter_id} "
                f"(arousal={context.affect_vector[0]:.2f})"
            )

        return action

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "events_processed": self.events_processed,
            "responses_generated": self.responses_generated,
            "response_rate": self.responses_generated / max(self.events_processed, 1),
            "agent_stats": self.agent.get_statistics(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from spatial_intelligence.spatial_ingestor import SimulatedIngestor
    from spatial_intelligence.topology_engine import TopologyEngine
    from fusion_intelligence.receiver_inference import ReceiverInferenceEngine

    # Create test setup
    ingestor = SimulatedIngestor(num_agents=5, area_size=10.0)
    topology = TopologyEngine(max_agents=5, proximity_radius=5.0)
    inference_engine = ReceiverInferenceEngine()

    # Create interaction agent
    policy = ResponsePolicy(
        respond_to_alarm=True,
        respond_to_contact=True,
        respond_to_mating=False,
    )
    agent = Level25InteractionAgent(
        receiver_inference_engine=inference_engine,
        response_policy=policy,
    )

    orchestrator = Level25Orchestrator(inference_engine, agent)

    # Simulate a vocalization
    frame = ingestor.generate_frame(timestamp_ns=0)
    topology.update_topology(frame)

    if frame.observations:
        emitter = frame.observations[0]

        # High arousal broadcast (alarm call)
        affect_alarm = np.array([0.9] + [0.0] * 15, dtype=np.float32)

        action = orchestrator.process_acoustic_event(
            emitter_id=emitter.agent_id,
            syntactic_token=5,
            affect_vector=affect_alarm,
            topology=topology,
            timestamp_ns=0,
        )

        if action:
            print(f"\nGenerated action: {action.call_directionality.value}")
            print(f"Target affect arousal: {action.affect_vector[0]:.2f}")

        print(f"\nPipeline stats: {orchestrator.get_pipeline_statistics()}")
