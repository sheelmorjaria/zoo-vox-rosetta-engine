#!/usr/bin/env python3
"""
Receiver Inference Engine for Level 2.5 Spatial Awareness

Predicts the intended receiver of a vocalization based on
spatial proximity, social ties, and line-of-sight analysis.

Enables the system to distinguish between:
- Broadcast calls (addressed to colony/flock)
- Unicast calls (addressed to specific individual)
- Self-directed calls (no clear receiver)

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .topology_engine import TopologyEngine, Edge

logger = logging.getLogger(__name__)


class AddressType(Enum):
    """Type of vocalization addressing."""
    BROADCAST = "broadcast"  # Addressed to colony/flock
    UNICAST = "unicast"    # Addressed to specific individual
    SELF = "self"          # Self-directed or ambiguous
    AI = "ai"              # Addressed to AI system


@dataclass
class ReceiverPrediction:
    """
    Result of receiver inference.

    Attributes:
        address_type: Type of addressing
        receiver_id: Predicted receiver bat ID (-1 if none/AI)
        confidence: Confidence of prediction (0-1)
        scores: Raw scores for all potential receivers
        reasoning: Human-readable explanation
    """
    address_type: AddressType
    receiver_id: int
    confidence: float
    scores: Dict[int, float]
    reasoning: str


class ReceiverInferenceEngine:
    """
    Predicts the intended receiver of a vocalization.

    Uses weighted combination of:
    1. Proximity (closer = more likely receiver)
    2. Social ties (stronger ties = more likely)
    3. Line-of-sight (visible = more likely)
    4. Body orientation (facing = more likely, if available)

    The final score combines these factors to predict
    whether a call is broadcast, unicast, or self-directed.
    """

    def __init__(
        self,
        w_proximity: float = 0.4,
        w_social: float = 0.4,
        w_los: float = 0.2,
        broadcast_threshold: float = 0.4,
        unicast_threshold: float = 0.6,
    ):
        """
        Initialize receiver inference engine.

        Args:
            w_proximity: Weight for proximity factor
            w_social: Weight for social tie factor
            w_los: Weight for line-of-sight factor
            broadcast_threshold: Min score for unicast (below = broadcast)
            unicast_threshold: Min score for confident unicast
        """
        self.w_proximity = w_proximity
        self.w_social = w_social
        self.w_los = w_los

        self.broadcast_threshold = broadcast_threshold
        self.unicast_threshold = unicast_threshold

        # Validate weights sum to ~1
        total = w_proximity + w_social + w_los
        if not np.isclose(total, 1.0, atol=0.1):
            logger.warning(f"Weights sum to {total}, normalizing")
            self.w_proximity = w_proximity / total
            self.w_social = w_social / total
            self.w_los = w_los / total

        logger.info(
            f"ReceiverInferenceEngine initialized: "
            f"proximity={self.w_proximity:.2f}, "
            f"social={self.w_social:.2f}, "
            f"los={self.w_los:.2f}"
        )

    def infer_receiver(
        self,
        caller_id: int,
        topology: TopologyEngine,
        orientation: Optional[np.ndarray] = None,
        arousal: float = 0.5,
    ) -> ReceiverPrediction:
        """
        Predict the intended receiver of a vocalization.

        Args:
            caller_id: ID of calling bat
            topology: Current colony topology
            orientation: Optional facing direction (3D vector)
            arousal: Arousal level (0-1) from VAE

        Returns:
            ReceiverPrediction with prediction details
        """
        if caller_id not in topology.nodes:
            return ReceiverPrediction(
                address_type=AddressType.SELF,
                receiver_id=-1,
                confidence=0.0,
                scores={},
                reasoning="Caller not in topology",
            )

        caller_node = topology.nodes[caller_id]
        scores = {}
        reasoning_parts = []

        # Score each potential receiver
        for other_id, other_node in topology.nodes.items():
            if other_id == caller_id:
                continue

            # Get edge data
            edge = topology.get_edge(caller_id, other_id)
            if edge is None:
                continue

            # Compute individual scores
            proximity_score = self._compute_proximity_score(edge.distance)
            social_score = edge.social_tie_strength
            los_score = 1.0 if edge.line_of_sight else 0.1

            # Orientation bonus (if available)
            orientation_bonus = 0.0
            if orientation is not None:
                orientation_bonus = self._compute_orientation_score(
                    caller_node.position,
                    other_node.position,
                    orientation
                )

            # Combined score
            combined = (
                self.w_proximity * proximity_score +
                self.w_social * social_score +
                self.w_los * los_score +
                0.1 * orientation_bonus  # Small bonus for facing
            )

            scores[other_id] = combined

        # Determine addressing type
        if not scores:
            # No potential receivers found
            return ReceiverPrediction(
                address_type=AddressType.BROADCAST,
                receiver_id=-1,
                confidence=0.5,
                scores={},
                reasoning="No nearby bats, likely broadcast",
            )

        # Find best receiver
        best_receiver = max(scores, key=scores.get)
        best_score = scores[best_receiver]

        # Determine if this is AI-directed
        # (High arousal + facing AI emitter position = AI-directed)
        if arousal > 0.7:
            # Check if facing known AI emitter location
            # For now, assume high arousal could be AI-directed
            if best_score < self.broadcast_threshold:
                return ReceiverPrediction(
                    address_type=AddressType.AI,
                    receiver_id=-1,  # AI is not a bat ID
                    confidence=0.7,
                    scores=scores,
                    reasoning=f"High arousal ({arousal:.2f}) suggests AI-directed call",
                )

        # Determine address type
        if best_score >= self.unicast_threshold:
            address_type = AddressType.UNICAST
            confidence = best_score
            reasoning = (
                f"Unicast to bat {best_receiver} "
                f"(score={best_score:.2f})"
            )
        elif best_score >= self.broadcast_threshold:
            address_type = AddressType.UNICAST
            confidence = best_score
            reasoning = (
                f"Weak unicast to bat {best_receiver} "
                f"(score={best_score:.2f}, may be broadcast)"
            )
        else:
            # All scores low -> broadcast
            address_type = AddressType.BROADCAST
            # For broadcast, receiver_id = -1 (no specific receiver)
            best_receiver = -1
            confidence = 1.0 - best_score  # Higher confidence in broadcast
            reasoning = (
                f"Broadcast (no clear target, max score={best_score:.2f})"
            )

        return ReceiverPrediction(
            address_type=address_type,
            receiver_id=best_receiver,
            confidence=confidence,
            scores=scores,
            reasoning=reasoning,
        )

    def _compute_proximity_score(self, distance: float) -> float:
        """
        Compute proximity score (closer = higher).

        Uses exponential decay with characteristic distance of 1m.
        """
        return np.exp(-distance / 1.0)

    def _compute_orientation_score(
        self,
        from_pos: np.ndarray,
        to_pos: np.ndarray,
        orientation: np.ndarray,
    ) -> float:
        """
        Compute orientation bonus (facing target = higher).

        Args:
            from_pos: Caller position
            to_pos: Potential receiver position
            orientation: Caller's facing direction (normalized)

        Returns:
            Bonus score (0-1)
        """
        # Direction to target
        to_target = to_pos - from_pos
        to_target = to_target / (np.linalg.norm(to_target) + 1e-8)

        # Cosine similarity
        cos_angle = np.dot(orientation, to_target)

        # Convert to [0, 1] range
        # cos_angle = 1 means facing directly toward
        # cos_angle = 0 means perpendicular
        # cos_angle = -1 means facing away
        if cos_angle < 0:
            return 0.0  # Facing away, no bonus

        return cos_angle  # Facing toward, proportional bonus


class InteractionLogger:
    """
    Logs interaction patterns for social tie learning.

    Tracks who vocalizes to whom over time to build
    social tie strengths used by ReceiverInferenceEngine.
    """

    def __init__(
        self,
        topology: TopologyEngine,
        learning_rate: float = 0.1,
    ):
        """
        Initialize interaction logger.

        Args:
            topology: Topology engine to update
            learning_rate: How fast to update tie strengths
        """
        self.topology = topology
        self.learning_rate = learning_rate
        self.interaction_counts: Dict[Tuple[int, int], int] = {}

    def log_interaction(
        self,
        caller_id: int,
        receiver_id: int,
        interaction_type: str = "vocalization",
    ) -> None:
        """
        Log an interaction between two bats.

        Args:
            caller_id: Calling bat ID
            receiver_id: Receiving bat ID (-1 for broadcast)
            interaction_type: Type of interaction
        """
        if receiver_id < 0:
            # Broadcast interaction
            # Could update tie strengths with all nearby bats
            return

        # Increment count
        key = (caller_id, receiver_id)
        self.interaction_counts[key] = \
            self.interaction_counts.get(key, 0) + 1

        # Update social tie in topology
        # More interactions = stronger tie
        count = self.interaction_counts[key]

        # Normalize to [0, 1] using sigmoid
        # 10 interactions -> ~0.73 strength
        # 50 interactions -> ~0.99 strength
        normalized = 1.0 / (1.0 + np.exp(-0.1 * (count - 25)))

        self.topology.update_social_tie(
            caller_id,
            receiver_id,
            normalized,
        )

    def get_interaction_matrix(self) -> np.ndarray:
        """
        Get interaction matrix as numpy array.

        Returns:
            N x N matrix where N is number of bats
        """
        bat_ids = sorted(self.topology.nodes.keys())
        n = len(bat_ids)
        matrix = np.zeros((n, n))

        for i, from_id in enumerate(bat_ids):
            for j, to_id in enumerate(bat_ids):
                if from_id == to_id:
                    continue

                edge = self.topology.get_edge(from_id, to_id)
                if edge is not None:
                    matrix[i, j] = edge.social_tie_strength

        return matrix


# Preset configurations

# Default receiver inference engine
DEFAULT_RECEIVER_INFERENCE = ReceiverInferenceEngine(
    w_proximity=0.4,
    w_social=0.4,
    w_los=0.2,
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from .topology_engine import TopologyEngine

    print("Receiver Inference Demo")
    print("=" * 50)

    # Create topology
    topology = TopologyEngine()

    # Add bats in a cluster
    positions = {
        1: np.array([0.0, 0.0, 1.0]),
        2: np.array([0.5, 0.0, 1.0]),
        3: np.array([2.0, 0.0, 1.0]),
        4: np.array([1.0, 1.0, 1.5]),
    }

    for bat_id, pos in positions.items():
        topology.update_node(bat_id, pos)

    # Add some social ties
    topology.update_social_tie(1, 2, 0.9)  # Strong tie
    topology.update_social_tie(1, 3, 0.3)  # Weak tie

    # Create inference engine
    engine = DEFAULT_RECEIVER_INFERENCE

    # Test: Bat 1 vocalizes
    prediction = engine.infer_receiver(1, topology)

    print(f"Address Type: {prediction.address_type.value}")
    print(f"Receiver ID: {prediction.receiver_id}")
    print(f"Confidence: {prediction.confidence:.2f}")
    print(f"Reasoning: {prediction.reasoning}")
    print(f"Scores: {prediction.scores}")
