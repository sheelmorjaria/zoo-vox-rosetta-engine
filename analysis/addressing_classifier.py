#!/usr/bin/env python3
"""
Broadcast vs. Unicast Addressing Classifier

Analyzes vocalizations to determine if they are:
- Broadcast: Addressed to colony/flock (alarm, coordination)
- Unicast: Addressed to specific individual (aggression, mating)

Uses spatial awareness (Level 2.5) and syntactic analysis
to classify addressing patterns across the colony.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class AddressMode(Enum):
    """Addressing mode of a vocalization."""
    BROADCAST = "broadcast"      # To colony/flock
    UNICAST = "unicast"        # To specific individual
    AMBIGUOUS = "ambiguous"    # Cannot determine


@dataclass
class AddressingClassification:
    """
    Result of addressing classification.

    Attributes:
        mode: Predicted addressing mode
        confidence: Confidence of prediction
        target_bat_id: Predicted receiver (-1 if broadcast)
        spatial_score: Score from spatial analysis
        syntactic_score: Score from syntactic analysis
        affect_score: Score from affect analysis
        reasoning: Explanation of classification
    """
    mode: AddressMode
    confidence: float
    target_bat_id: int
    spatial_score: float
    syntactic_score: float
    affect_score: float
    reasoning: str


@dataclass
class AddressingPattern:
    """
    Pattern of addressing for a specific bat or dyad.
    """
    bat_id: int
    broadcast_count: int
    unicast_count: int
    unicast_targets: Dict[int, int]  # target -> count
    preferred_mode: AddressMode
    social_network_centrality: float


class AddressingClassifier:
    """
    Classifies vocalizations as broadcast or unicast.

    Uses multi-modal evidence:
    1. Spatial: Proximity and line-of-sight to specific bat
    2. Syntactic: Specific syntactic constructions for unicast
    3. Affective: Arousal and valence patterns
    """

    def __init__(
        self,
        spatial_threshold: float = 0.6,
        syntactic_broadcast_tokens: set = None,
        syntactic_unicast_tokens: set = None,
    ):
        """
        Initialize addressing classifier.

        Args:
            spatial_threshold: Min spatial score for unicast
            syntactic_broadcast_tokens: Tokens indicating broadcast
            syntactic_unicast_tokens: Tokens indicating unicast
        """
        self.spatial_threshold = spatial_threshold

        # Default: alarm tokens are broadcast
        self.syntactic_broadcast_tokens = syntactic_broadcast_tokens or {5, 15, 25}
        self.syntactic_unicast_tokens = syntactic_unicast_tokens or {1, 2, 3, 10}

        # Learned patterns
        self.patterns: Dict[int, AddressingPattern] = {}

        logger.info("AddressingClassifier initialized")

    def classify(
        self,
        caller_id: int,
        syntactic_token: int,
        affect_vector: np.ndarray,
        spatial_prediction: Optional[Tuple[int, float]] = None,
        colony_size: int = 50,
    ) -> AddressingClassification:
        """
        Classify a vocalization as broadcast or unicast.

        Args:
            caller_id: Calling bat ID
            syntactic_token: VQ-VAE token
            affect_vector: 16D affect vector
            spatial_prediction: (target_id, confidence) from spatial inference
            colony_size: Estimated colony size

        Returns:
            AddressingClassification with prediction
        """
        scores = {"spatial": 0.5, "syntactic": 0.5, "affect": 0.5}
        reasoning_parts = []

        # Spatial score
        if spatial_prediction is not None:
            target_id, spatial_conf = spatial_prediction
            if target_id >= 0 and spatial_conf > self.spatial_threshold:
                scores["spatial"] = 0.9  # Likely unicast
                reasoning_parts.append(f"Spatial: Targeted at bat {target_id}")
            elif target_id == -1:
                scores["spatial"] = 0.1  # Likely broadcast
                reasoning_parts.append("Spatial: No specific target")
            else:
                scores["spatial"] = 0.5  # Ambiguous
        else:
            # No spatial info
            reasoning_parts.append("Spatial: No data")

        # Syntactic score
        if syntactic_token in self.syntactic_broadcast_tokens:
            scores["syntactic"] = 0.1  # Broadcast indicator
            reasoning_parts.append(f"Syntactic: Broadcast token {syntactic_token}")
        elif syntactic_token in self.syntactic_unicast_tokens:
            scores["syntactic"] = 0.9  # Unicast indicator
            reasoning_parts.append(f"Syntactic: Unicast token {syntactic_token}")
        else:
            # Neutral token
            scores["syntactic"] = 0.5

        # Affective score
        # High arousal + negative valence often = alarm (broadcast)
        arousal = affect_vector[0]
        valence = affect_vector[1]

        if arousal > 0.7 and valence < -0.3:
            scores["affect"] = 0.2  # Suggests broadcast (alarm)
            reasoning_parts.append("Affect: Alarm-like (high arousal, negative)")
        elif 0.3 < arousal < 0.7 and valence > 0:
            scores["affect"] = 0.8  # Suggests unicast (social)
            reasoning_parts.append("Affect: Social interaction")
        else:
            scores["affect"] = 0.5

        # Combine scores
        combined = (
            0.4 * scores["spatial"] +
            0.4 * scores["syntactic"] +
            0.2 * scores["affect"]
        )

        # Determine mode
        if combined > 0.65:
            mode = AddressMode.UNICAST
            target_id = spatial_prediction[0] if spatial_prediction else -1
            confidence = combined
        elif combined < 0.35:
            mode = AddressMode.BROADCAST
            target_id = -1
            confidence = 1.0 - combined
        else:
            mode = AddressMode.AMBIGUOUS
            target_id = -1
            confidence = 0.5

        return AddressingClassification(
            mode=mode,
            confidence=confidence,
            target_bat_id=target_id,
            spatial_score=scores["spatial"],
            syntactic_score=scores["syntactic"],
            affect_score=scores["affect"],
            reasoning="; ".join(reasoning_parts),
        )

    def analyze_addressing_patterns(
        self,
        classifications: List[AddressingClassification],
    ) -> Dict[int, AddressingPattern]:
        """
        Analyze addressing patterns for individual bats.

        Args:
            classifications: List of classifications with caller IDs

        Returns:
            Dictionary mapping bat_id to AddressingPattern
        """
        # This would need caller_id in classifications
        # For now, return empty dict
        return {}

    def compare_broadcast_unicast_features(
        self,
        broadcast_calls: List[np.ndarray],
        unicast_calls: List[np.ndarray],
    ) -> Dict:
        """
        Compare acoustic features of broadcast vs unicast calls.

        Tests hypotheses about syntactic/affective differences.

        Args:
            broadcast_calls: List of affect vectors from broadcast calls
            unicast_calls: List of affect vectors from unicast calls

        Returns:
            Dictionary with statistical comparisons
        """
        # Compute mean affect for each type
        broadcast_mean = np.mean(broadcast_calls, axis=0)
        unicast_mean = np.mean(unicast_calls, axis=0)

        # Compute variance
        broadcast_var = np.var(broadcast_calls, axis=0)
        unicast_var = np.var(unicast_calls, axis=0)

        # Hypothesis: Broadcast has lower variance (more stereotyped)
        mean_broadcast_var = np.mean(broadcast_var)
        mean_unicast_var = np.mean(unicast_var)

        return {
            "broadcast_mean_affect": broadcast_mean,
            "unicast_mean_affect": unicast_mean,
            "broadcast_variance": broadcast_var,
            "unicast_variance": unicast_var,
            "mean_broadcast_variance": mean_broadcast_var,
            "mean_unicast_variance": mean_unicast_var,
            "broadcast_more_stereotyped": mean_broadcast_var < mean_unicast_var,
        }


# Preset configurations

# Default addressing classifier
DEFAULT_ADDRESSING_CLASSIFIER = AddressingClassifier()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Addressing Classifier Demo")
    print("=" * 50)

    classifier = DEFAULT_ADDRESSING_CLASSIFIER

    # Test broadcast call
    broadcast_affect = np.array([
        0.9,   # High arousal
        -0.8,  # Negative valence
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])

    result = classifier.classify(
        caller_id=1,
        syntactic_token=5,  # Alarm token
        affect_vector=broadcast_affect,
        spatial_prediction=(-1, 0.9),  # No target
    )

    print(f"\nBroadcast Call Classification:")
    print(f"  Mode: {result.mode.value}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Reasoning: {result.reasoning}")

    # Test unicast call
    unicast_affect = np.array([
        0.5,   # Medium arousal
        0.3,   # Positive valence
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])

    result = classifier.classify(
        caller_id=1,
        syntactic_token=2,  # Social token
        affect_vector=unicast_affect,
        spatial_prediction=(5, 0.8),  # Target bat 5
    )

    print(f"\nUnicast Call Classification:")
    print(f"  Mode: {result.mode.value}")
    print(f"  Target: {result.target_bat_id}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Reasoning: {result.reasoning}")
