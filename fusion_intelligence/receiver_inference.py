#!/usr/bin/env python3
"""
Receiver Inference Engine (Level 2.5)

Fuses Level 2 Acoustic data with Spatial Topology to infer receivers.
Distinguishes between broadcast (general) and unicast (directed) calls.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import dataclasses
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cognitive_intelligence.syntax_graph import SyntaxGraph
from spatial_intelligence.spatial_ingestor import SpatialObservation
from spatial_intelligence.topology_engine import TopologyEngine

logger = logging.getLogger(__name__)


class CallDirectionality(Enum):
    """Type of call based on receiver inference."""
    BROADCAST = "broadcast"  # General call to any/nearby conspecifics
    UNICAST = "unicast"      # Directed call to specific individual


@dataclass
class Level25Context:
    """
    Level 2.5 Spatial-Social context.

    Extends Level 2 (Emitter ID + acoustic features) with spatial topology
    to infer receivers and call directionality.
    """
    emitter_id: str
    syntactic_token: int
    affect_vector: np.ndarray  # 16D affect vector
    call_directionality: CallDirectionality
    receiver_probabilities: Dict[str, float]  # agent_id -> probability
    timestamp_ns: int
    confidence: float = 1.0

    # Additional metadata
    raw_features: Optional[np.ndarray] = None  # 112D for fallback
    nearby_count: int = 0
    line_of_sight_count: int = 0

    def get_top_receivers(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k most likely receivers."""
        sorted_receivers = sorted(
            self.receiver_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_receivers[:top_k]

    def has_targets(self) -> bool:
        """Check if there are any probable receivers."""
        return len(self.receiver_probabilities) > 0

    def is_broadcast(self) -> bool:
        """Check if this is a broadcast call."""
        return self.call_directionality == CallDirectionality.BROADCAST

    def is_unicast(self) -> bool:
        """Check if this is a directed call."""
        return self.call_directionality == CallDirectionality.UNICAST


@dataclass
class InferenceWeights:
    """Weights for different inference factors."""
    proximity_weight: float = 0.6    # Closer = higher probability
    los_weight: float = 0.3         # Line-of-sight = higher probability
    social_weight: float = 0.1      # Social affinity = higher probability

    def __post_init__(self):
        """Normalize weights to sum to 1.0."""
        total = self.proximity_weight + self.los_weight + self.social_weight
        if total > 0:
            self.proximity_weight /= total
            self.los_weight /= total
            self.social_weight /= total


class ReceiverInferenceEngine:
    """
    Fuses L2 Acoustic data with Spatial Topology to infer receivers.

    Uses a weighted combination of:
    1. Proximity (closer agents more likely to be receivers)
    2. Line-of-sight (agents in field of view more likely)
    3. Social affinity (historical interaction patterns)
    """

    def __init__(
        self,
        weights: Optional[InferenceWeights] = None,
        broadcast_threshold: float = 0.65,
        syntax_graph: Optional[SyntaxGraph] = None,
    ):
        self.weights = weights or InferenceWeights()
        self.broadcast_threshold = broadcast_threshold
        self.syntax_graph = syntax_graph

        # Social affinity cache (emitter_id -> {target_id: affinity})
        self.social_affinity: Dict[str, Dict[str, float]] = {}

        logger.info(
            f"ReceiverInferenceEngine initialized: "
            f"weights=({self.weights.proximity_weight:.2f}, "
            f"{self.weights.los_weight:.2f}, {self.weights.social_weight:.2f}), "
            f"broadcast_threshold={broadcast_threshold}"
        )

    def infer_receiver(
        self,
        emitter_id: str,
        topology: TopologyEngine,
        syntactic_token: int = 0,
        affect_vector: Optional[np.ndarray] = None,
        timestamp_ns: int = 0,
    ) -> Level25Context:
        """
        Infer receivers for a vocalization event.

        Args:
            emitter_id: ID of the vocalizing agent
            topology: Current spatial topology
            syntactic_token: Syntactic token from VQ-VAE
            affect_vector: 16D affect vector from VAE
            timestamp_ns: Event timestamp

        Returns:
            Level25Context with inferred receivers and directionality
        """
        # Get nearby agents
        nearby_agents = topology.get_proximity_map(emitter_id)

        if not nearby_agents:
            # No nearby agents - pure broadcast
            affect = affect_vector if affect_vector is not None else np.zeros(16)
            return Level25Context(
                emitter_id=emitter_id,
                syntactic_token=syntactic_token,
                affect_vector=affect,
                call_directionality=CallDirectionality.BROADCAST,
                receiver_probabilities={},
                timestamp_ns=timestamp_ns,
                nearby_count=0,
                line_of_sight_count=0,
            )

        # Calculate receiver probabilities
        receiver_probs = {}

        for target_id, distance in nearby_agents.items():
            # Proximity score (closer = higher)
            prox_score = 1.0 / (1.0 + distance)

            # Line-of-sight score
            los_result = topology.check_line_of_sight(emitter_id, target_id)
            los_score = 1.0 if los_result.in_field_of_view else 0.2

            # Social affinity score
            social_score = self._get_social_affinity(emitter_id, target_id)

            # Weighted combination
            total_score = (
                self.weights.proximity_weight * prox_score +
                self.weights.los_weight * los_score +
                self.weights.social_weight * social_score
            )

            receiver_probs[target_id] = total_score

        # Normalize probabilities
        total = sum(receiver_probs.values())
        if total > 0:
            receiver_probs = {k: v / total for k, v in receiver_probs.items()}

        # Determine directionality
        directionality = self._classify_directionality(receiver_probs)

        # Count nearby agents and those with line-of-sight
        los_count = sum(
            1 for aid in receiver_probs.keys()
            if topology.check_line_of_sight(emitter_id, aid).in_field_of_view
        )

        affect = affect_vector if affect_vector is not None else np.zeros(16)
        return Level25Context(
            emitter_id=emitter_id,
            syntactic_token=syntactic_token,
            affect_vector=affect,
            call_directionality=directionality,
            receiver_probabilities=receiver_probs,
            timestamp_ns=timestamp_ns,
            nearby_count=len(receiver_probs),
            line_of_sight_count=los_count,
        )

    def _classify_directionality(self, probs: Dict[str, float]) -> CallDirectionality:
        """
        Classify call as broadcast or unicast based on probability distribution.

        If the top candidate has highly concentrated probability (> threshold),
        it's a directed (unicast) call. Otherwise, it's broadcast.
        """
        if not probs:
            return CallDirectionality.BROADCAST

        max_prob = max(probs.values())

        # High concentration = directed call
        if max_prob > self.broadcast_threshold:
            return CallDirectionality.UNICAST
        else:
            return CallDirectionality.BROADCAST

    def _get_social_affinity(self, emitter_id: str, target_id: str) -> float:
        """
        Get social affinity between two agents.

        Higher affinity = more likely to interact.
        Uses syntax graph if available, otherwise defaults to neutral.
        """
        if self.syntax_graph is not None:
            # Could use token transition probabilities as affinity proxy
            # For now, use default
            pass

        # Check cache
        if emitter_id in self.social_affinity:
            if target_id in self.social_affinity[emitter_id]:
                return self.social_affinity[emitter_id][target_id]

        # Default neutral affinity
        return 0.5

    def update_social_affinity(
        self,
        emitter_id: str,
        target_id: str,
        affinity: float,
    ):
        """
        Update social affinity between two agents.

        Affinity should be in [0, 1], where:
        - 0.0 = never interact
        - 0.5 = neutral
        - 1.0 = frequent interaction
        """
        affinity = np.clip(affinity, 0.0, 1.0)

        if emitter_id not in self.social_affinity:
            self.social_affinity[emitter_id] = {}

        self.social_affinity[emitter_id][target_id] = affinity

        logger.debug(
            f"Updated social affinity: {emitter_id} -> {target_id} = {affinity:.2f}"
        )

    def learn_from_interaction(
        self,
        emitter_id: str,
        actual_receiver_id: Optional[str],
        predicted_context: Level25Context,
    ):
        """
        Learn from an actual interaction to improve future inferences.

        If the actual receiver matches a high-probability prediction,
        reinforce those weights. If not, adjust.

        Args:
            emitter_id: Agent that vocalized
            actual_receiver_id: Agent that actually responded (None for broadcast)
            predicted_context: The context that was predicted
        """
        if actual_receiver_id is None:
            # Broadcast interaction - reinforce broadcast pattern
            # (No specific update needed for now)
            return

        # Check if prediction was correct
        top_receivers = predicted_context.get_top_receivers(top_k=3)
        top_ids = [aid for aid, _ in top_receivers]

        if actual_receiver_id in top_ids:
            # Correct prediction - reinforce social affinity
            current_affinity = self._get_social_affinity(emitter_id, actual_receiver_id)
            # Increase affinity slightly
            new_affinity = min(1.0, current_affinity + 0.05)
            self.update_social_affinity(emitter_id, actual_receiver_id, new_affinity)
        else:
            # Incorrect prediction - may need to adjust
            # For now, just log
            logger.info(
                f"Prediction miss: {emitter_id} -> {actual_receiver_id} "
                f"(predicted: {top_ids})"
            )


class MultiModalFusionBuffer:
    """
    Temporal buffer for synchronizing high-frequency acoustic events
    with lower-frequency spatial frames.

    Handles the case where an acoustic event occurs between spatial frames.
    """

    def __init__(
        self,
        max_age_ms: float = 100.0,
        spatial_frame_rate_ms: float = 33.0,  # ~30 FPS
    ):
        self.max_age_ms = max_age_ms
        self.spatial_frame_rate_ms = spatial_frame_rate_ms

        # Buffers
        self.acoustic_events: List[Level25Context] = []
        self.spatial_topology: Optional[TopologyEngine] = None
        self.last_spatial_timestamp_ns: int = 0

    def add_acoustic_event(self, context: Level25Context):
        """Add an acoustic event to the buffer."""
        self.acoustic_events.append(context)

    def update_spatial_topology(self, topology: TopologyEngine, timestamp_ns: int):
        """Update the spatial topology snapshot."""
        self.spatial_topology = topology
        self.last_spatial_timestamp_ns = timestamp_ns

        # Prune old acoustic events
        cutoff_time = timestamp_ns - int(self.max_age_ms * 1_000_000)
        self.acoustic_events = [
            e for e in self.acoustic_events
            if e.timestamp_ns >= cutoff_time
        ]

    def get_fused_context(
        self,
        acoustic_event: Level25Context,
    ) -> Optional[Level25Context]:
        """
        Get fused context for an acoustic event.

        If the event has no spatial data yet, use the latest spatial topology.
        """
        if acoustic_event.receiver_probabilities:
            # Already has spatial inference
            return acoustic_event

        if self.spatial_topology is None:
            # No spatial data available yet
            return acoustic_event

        # Run receiver inference with latest topology
        # (This would need the InferenceEngine - simplified here)
        return acoustic_event


class BroadcastDetector:
    """
    Detects broadcast calls based on acoustic + spatial patterns.

    Broadcast calls typically have:
    - High amplitude (affect vector: high arousal)
    - Repetitive structure (syntactic: alarm/mating patterns)
    - No specific receiver focus (spatial: uniform probability distribution)
    """

    def __init__(
        self,
        arousal_threshold: float = 0.7,
        probability_entropy_threshold: float = 1.5,
    ):
        self.arousal_threshold = arousal_threshold
        self.entropy_threshold = probability_entropy_threshold

    def is_broadcast_call(self, context: Level25Context) -> bool:
        """
        Determine if a call is a broadcast based on multiple signals.

        Args:
            context: Level25Context with acoustic and spatial data

        Returns:
            True if likely a broadcast call
        """
        # Check 1: High arousal in affect vector
        if len(context.affect_vector) > 0:
            arousal = context.affect_vector[0]  # Dim 0 = arousal
            if arousal > self.arousal_threshold:
                return True

        # Check 2: High entropy in receiver probabilities
        # (Evenly distributed = broadcast)
        if context.receiver_probabilities:
            entropy = self._compute_entropy(context.receiver_probabilities)
            if entropy > self.entropy_threshold:
                return True

        # Check 3: No nearby agents
        if context.nearby_count == 0:
            return True

        return False

    def _compute_entropy(self, probs: Dict[str, float]) -> float:
        """Compute Shannon entropy of probability distribution."""
        if not probs:
            return 0.0

        entropy = 0.0
        for p in probs.values():
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from spatial_intelligence.spatial_ingestor import SimulatedIngestor

    # Create test setup
    ingestor = SimulatedIngestor(num_agents=5, area_size=10.0)
    topology = TopologyEngine(max_agents=5, proximity_radius=5.0)
    inference_engine = ReceiverInferenceEngine()

    # Generate spatial frame
    frame = ingestor.generate_frame(timestamp_ns=0)
    topology.update_topology(frame)

    # Simulate a vocalization from first agent
    if frame.observations:
        emitter = frame.observations[0]

        context = inference_engine.infer_receiver(
            emitter_id=emitter.agent_id,
            topology=topology,
            syntactic_token=5,
            affect_vector=np.random.randn(16).astype(np.float32),
            timestamp_ns=0,
        )

        print(f"\n=== Receiver Inference Results ===")
        print(f"Emitter: {context.emitter_id}")
        print(f"Directionality: {context.call_directionality.value}")
        print(f"Nearby agents: {context.nearby_count}")
        print(f"Line-of-sight agents: {context.line_of_sight_count}")

        if context.receiver_probabilities:
            print(f"\nTop 3 receivers:")
            for aid, prob in context.get_top_receivers(3):
                print(f"  {aid}: {prob:.3f}")
