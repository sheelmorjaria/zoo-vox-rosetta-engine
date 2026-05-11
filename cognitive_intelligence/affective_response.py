#!/usr/bin/env python3
"""
Affective Response Logic (Module 1 Deep Dive)

Implements biologically-inspired affective response logic with:
- De-escalation for high arousal (>0.8) to avoid panic cascades
- Matching for low arousal to maintain social contact
- Proper arousal/valence interpretation from 16D latent space

Key Behaviors:
- High arousal (>0.8) → De-escalate (target arousal = 0.6)
- Low arousal (<0.3) → Escalate slightly for engagement
- Medium arousal → Match for social bonding

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AffectiveResponseConfig:
    """Configuration for affective response logic."""

    # Dimension indices for 16D affect vector
    AROUSAL_DIM: int = 0
    VALENCE_DIM: int = 1
    PITCH_VARIATION_DIM: int = 2

    # Thresholds
    HIGH_AROUSAL_THRESHOLD: float = 0.8
    LOW_AROUSAL_THRESHOLD: float = 0.3

    # Response parameters
    DEESCALATION_TARGET_AROUSAL: float = 0.6
    ESCALATION_FACTOR: float = 1.2
    MATCH_TOLERANCE: float = 0.05

    # Safety limits
    MAX_AROUSAL: float = 0.95
    MIN_AROUSAL: float = 0.05


# =============================================================================
# Affective Response Logic
# =============================================================================


class AffectiveResponsePolicy:
    """
    Biologically-inspired affective response logic.

    Implements de-escalation for high arousal states to prevent
    panic cascades in colonies, and matching for social bonding.
    """

    def __init__(self, config: Optional[AffectiveResponseConfig] = None):
        self.config = config or AffectiveResponseConfig()

    def extract_arousal(self, affect_vector: np.ndarray | torch.Tensor) -> float:
        """
        Extract arousal level from affect vector.

        Args:
            affect_vector: 16D affect vector

        Returns:
            Arousal level (0-1)
        """
        if isinstance(affect_vector, torch.Tensor):
            arousal = affect_vector[..., self.config.AROUSAL_DIM].item()
        else:
            arousal = affect_vector[..., self.config.AROUSAL_DIM].item()

        # Clamp to valid range
        return max(0.0, min(1.0, arousal))

    def extract_valence(self, affect_vector: np.ndarray | torch.Tensor) -> float:
        """
        Extract valence from affect vector.

        Args:
            affect_vector: 16D affect vector

        Returns:
            Valence (-1 to 1)
        """
        if isinstance(affect_vector, torch.Tensor):
            valence = affect_vector[..., self.config.VALENCE_DIM].item()
        else:
            valence = affect_vector[..., self.config.VALENCE_DIM].item()

        # Clamp to valid range
        return max(-1.0, min(1.0, valence))

    def compute_target_affect(
        self,
        incoming_affect: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """
        Compute target affect based on incoming affect state.

        Response Logic:
        - High arousal (>0.8): De-escalate to 0.6 to prevent panic cascade
        - Low arousal (<0.3): Escalate slightly (×1.2) for social contact
        - Medium arousal: Match within tolerance for social bonding

        Args:
            incoming_affect: 16D affect vector

        Returns:
            target_affect: 16D target affect vector for response
        """
        # Convert to numpy for processing
        if isinstance(incoming_affect, torch.Tensor):
            affect = incoming_affect.detach().cpu().numpy()
        else:
            affect = incoming_affect.copy()

        # Ensure 1D
        if affect.ndim > 1:
            affect = affect.flatten()

        # Extract arousal
        arousal = self.extract_arousal(affect)

        # Compute response based on arousal level
        if arousal > self.config.HIGH_AROUSAL_THRESHOLD:
            # High arousal: De-escalate to avoid panic cascade
            target_affect = self._deescalate_affect(affect, arousal)
            response_type = "de-escalation"

        elif arousal < self.config.LOW_AROUSAL_THRESHOLD:
            # Low arousal: Escalate slightly for engagement
            target_affect = self._escalate_affect(affect, arousal)
            response_type = "escalation"

        else:
            # Medium arousal: Match for social bonding
            target_affect = self._match_affect(affect, arousal)
            response_type = "matching"

        logger.debug(
            f"Arousal {arousal:.3f} → {response_type}: "
            f"target arousal {target_affect[self.config.AROUSAL_DIM]:.3f}"
        )

        return target_affect

    def _deescalate_affect(
        self,
        affect: np.ndarray,
        current_arousal: float,
    ) -> np.ndarray:
        """
        De-escalate high arousal to prevent panic cascade.

        Maps arousal toward the de-escalation target (0.6) while
        preserving other affect dimensions.
        """
        target = affect.copy()

        # Smoothly interpolate toward de-escalation target
        # Use exponential decay for natural transition
        decay_rate = 0.3  # How quickly to move toward target
        new_arousal = (
            current_arousal * (1 - decay_rate) +
            self.config.DEESCALATION_TARGET_AROUSAL * decay_rate
        )

        target[self.config.AROUSAL_DIM] = new_arousal

        # Also slightly reduce intensity (harshness) for high arousal states
        if len(affect) > self.config.VALENCE_DIM:
            # If valence is negative (harsh), move toward neutral
            valence = affect[self.config.VALENCE_DIM]
            if valence < 0:
                target[self.config.VALENCE_DIM] = valence * 0.8  # Reduce harshness

        return target

    def _escalate_affect(
        self,
        affect: np.ndarray,
        current_arousal: float,
    ) -> np.ndarray:
        """
        Escalate low arousal for social engagement.

        Increases arousal slightly to maintain social contact.
        """
        target = affect.copy()

        # Escalate with multiplier, but cap at maximum
        new_arousal = min(
            current_arousal * self.config.ESCALATION_FACTOR,
            self.config.HIGH_AROUSAL_THRESHOLD * 0.9  # Don't escalate to high
        )

        target[self.config.AROUSAL_DIM] = new_arousal

        return target

    def _match_affect(
        self,
        affect: np.ndarray,
        current_arousal: float,
    ) -> np.ndarray:
        """
        Match affect for social bonding.

        Within tolerance range, match the incoming affect
        to promote social cohesion.
        """
        target = affect.copy()

        # Small adjustment toward match (within tolerance)
        # This promotes social bonding while allowing natural variation
        noise = np.random.uniform(-self.config.MATCH_TOLERANCE, self.config.MATCH_TOLERANCE)
        target[self.config.AROUSAL_DIM] = np.clip(
            current_arousal + noise,
            self.config.MIN_AROUSAL,
            self.config.MAX_AROUSAL,
        )

        return target

    def is_panic_state(self, affect_vector: np.ndarray | torch.Tensor) -> bool:
        """
        Detect panic state (extremely high arousal).

        Args:
            affect_vector: 16D affect vector

        Returns:
            True if in panic state (>0.9 arousal)
        """
        arousal = self.extract_arousal(affect_vector)
        return arousal > 0.9

    def should_deescalate(self, affect_vector: np.ndarray | torch.Tensor) -> bool:
        """Check if de-escalation response is appropriate."""
        arousal = self.extract_arousal(affect_vector)
        return arousal > self.config.HIGH_AROUSAL_THRESHOLD

    def should_escalate(self, affect_vector: np.ndarray | torch.Tensor) -> bool:
        """Check if escalation response is appropriate."""
        arousal = self.extract_arousal(affect_vector)
        return arousal < self.config.LOW_AROUSAL_THRESHOLD


# =============================================================================
# Utility Functions
# =============================================================================


def create_affective_response_policy(
    config: Optional[AffectiveResponseConfig] = None,
) -> AffectiveResponsePolicy:
    """Factory function to create affective response policy."""
    return AffectiveResponsePolicy(config)


def compute_affective_response(
    incoming_affect: np.ndarray | torch.Tensor,
    policy: Optional[AffectiveResponsePolicy] = None,
) -> np.ndarray:
    """
    Convenience function to compute affective response.

    Args:
        incoming_affect: 16D affect vector from β-VAE
        policy: Optional pre-configured policy

    Returns:
        target_affect: 16D target affect for response generation
    """
    if policy is None:
        policy = create_affective_response_policy()

    return policy.compute_target_affect(incoming_affect)
