#!/usr/bin/env python3
"""
Interaction Agent v3.0: Probabilistic Closed-Loop Agent

Statistical upgrade to Stage 4 Closed-Loop Agent addressing two critical flaws:
1. OOD Detection Failure: L2 distance fails in high-dimensional spaces
2. Syntax Inflexibility: Rigid bigram automaton prevents novel syntax

v3.0 Upgrades:
- Mahalanobis Distance: Accounts for covariance structure (D² = (x-μ)ᵀΣ⁻¹(x-μ))
- Chi-squared Distribution: Statistically sound OOD thresholds
- Autoregressive Transformer: Probabilistic syntax generation
- Temperature/Top-p Sampling: Novel but statistically plausible sequences

This replaces the rigid bigram automaton with a flexible statistical approach
that enables novel combinatorial syntax while maintaining biological plausibility.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Callable

import numpy as np
import torch

from .mahalanobis_ood import (
    MahalanobisOOD,
    OODCalibrator,
    OODCalibrationConfig,
    OODStatistics,
)
from .syntax_transformer import (
    SyntaxTransformer,
    SyntaxTransformerTrainer,
    TransformerConfig,
)
from .syntax_sampler import (
    SyntaxSampler,
    SamplingConfig,
    SamplingMode,
    SamplingResult,
)

logger = logging.getLogger(__name__)


class ResponseMode(IntEnum):
    """Cognitive response modes based on OOD state."""
    CONGRUENT = 0  # In-distribution: match affect + probabilistic syntax
    CAUTIOUS = 1  # Marginal OOD: conservative response
    SUPPRESS = 2  # OOD: suppress response (passive monitoring)


@dataclass
class CognitiveState:
    """Internal cognitive state tracking."""
    token_history: List[int]
    affect_history: List[np.ndarray]
    ood_count: int = 0
    total_processed: int = 0
    current_mode: ResponseMode = ResponseMode.CONGRUENT


@dataclass
class AgentConfig:
    """Configuration for InteractionAgentV3."""
    # OOD Detection
    ood_threshold: float = 0.99  # Chi-squared confidence
    ood_suppression_count: int = 3  # Consecutive OODs before suppression

    # Affective Matching
    arousal_deescalation_threshold: float = 0.8
    arousal_escalation_threshold: float = 0.3
    arousal_match_factor: float = 1.0
    affect_smoothing: float = 0.3  # EMA smoothing for affect

    # Sampling Strategy
    sampling_mode: SamplingMode = SamplingMode.COMBINED
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: Optional[int] = None

    # Response Timing
    default_response_delay_ms: float = 150.0
    ood_response_delay_ms: float = 300.0  # Longer delay for OOD

    # History
    max_token_history: int = 10
    max_affect_history: int = 5


class InteractionAgentV3:
    """
    Probabilistic Closed-Loop Agent v3.0.

    Cognitive Cycle:
    1. Receive DualStreamState (syntactic token + affect vector)
    2. OOD Check: Mahalanobis distance vs chi-squared threshold
    3. Update History: Token sequence and affect history
    4. Generate Response:
       - If OOD: Conservative response or suppress
       - If in-distribution: Probabilistic token generation + affect matching
    5. Publish DualStreamAction to Rust synthesizer

    Key Improvements over v2.0:
    - Statistical OOD detection (Mahalanobis vs L2)
    - Probabilistic syntax (Transformer vs bigram matrix)
    - Temperature-controlled creativity
    - Novel but statistically plausible sequences
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        ood_detector: Optional[MahalanobisOOD] = None,
        syntax_transformer: Optional[SyntaxTransformer] = None,
        syntax_sampler: Optional[SyntaxSampler] = None,
    ):
        """
        Initialize InteractionAgentV3.

        Args:
            config: Agent configuration
            ood_detector: Pre-trained Mahalanobis OOD detector
            syntax_transformer: Pre-trained Syntax Transformer
            syntax_sampler: Configured syntax sampler
        """
        self.config = config or AgentConfig()

        # Core components
        self.ood_detector = ood_detector
        self.syntax_transformer = syntax_transformer
        self.syntax_sampler = syntax_sampler or SyntaxSampler(
            SamplingConfig(
                mode=self.config.sampling_mode,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        )

        # Cognitive state
        self.cognitive_state = CognitiveState(
            token_history=[],
            affect_history=[],
        )

        # Response statistics
        self.stats = {
            "processed": 0,
            "ood_detected": 0,
            "responses_generated": 0,
            "suppressed": 0,
        }

        logger.info("InteractionAgentV3 initialized")

    def handle_dual_stream_state(
        self,
        state: "DualStreamState",
    ) -> Optional["DualStreamAction"]:
        """
        Process incoming dual-stream state and generate response.

        Args:
            state: DualStreamState from Rust (syntactic_token + affect_vector)

        Returns:
            DualStreamAction for synthesis, or None if suppressed
        """
        self.stats["processed"] += 1
        self.cognitive_state.total_processed += 1

        # Step 1: OOD Detection (Mahalanobis distance)
        is_ood, md_squared, reason = self._check_ood(state)

        if is_ood:
            self.stats["ood_detected"] += 1
            self.cognitive_state.ood_count += 1
            logger.debug(f"OOD detected: {reason}")

            # Check if should suppress
            if self.cognitive_state.ood_count >= self.config.ood_suppression_count:
                self.cognitive_state.current_mode = ResponseMode.SUPPRESS
                self.stats["suppressed"] += 1
                logger.info("Suppressing response due to consecutive OOD detections")
                return None
        else:
            # Reset OOD counter on in-distribution input
            self.cognitive_state.ood_count = 0
            self.cognitive_state.current_mode = ResponseMode.CONGRUENT

        # Step 2: Update History
        self._update_history(state)

        # Step 3: Generate Response
        if self.cognitive_state.current_mode == ResponseMode.SUPPRESS:
            return None

        response_token, response_affect = self._generate_response(state, is_ood)

        if response_token is None:
            return None

        # Step 4: Create Action
        action = self._create_action(
            response_token,
            response_affect,
            state.sequence,
            is_ood,
        )

        self.stats["responses_generated"] += 1
        logger.debug(
            f"Generated response: token={response_token}, "
            f"mode={self.cognitive_state.current_mode.name}"
        )

        return action

    def _check_ood(
        self,
        state: "DualStreamState",
    ) -> Tuple[bool, float, str]:
        """
        Check OOD using Mahalanobis distance.

        Args:
            state: Input dual-stream state

        Returns:
            (is_ood, md_squared, reason)
        """
        if self.ood_detector is None:
            # No OOD detector configured, assume in-distribution
            return False, 0.0, "No OOD detector configured"

        # Check Mahalanobis distance for affect vector
        is_ood, md_squared, reason = self.ood_detector.is_ood(
            state.affect_vector,
            state.syntactic_token,
        )

        return is_ood, md_squared, reason

    def _update_history(self, state: "DualStreamState") -> None:
        """Update cognitive history with new state."""
        # Update token history
        self.cognitive_state.token_history.append(state.syntactic_token)
        if len(self.cognitive_state.token_history) > self.config.max_token_history:
            self.cognitive_state.token_history = \
                self.cognitive_state.token_history[-self.config.max_token_history:]

        # Update affect history
        self.cognitive_state.affect_history.append(state.affect_vector.copy())
        if len(self.cognitive_state.affect_history) > self.config.max_affect_history:
            self.cognitive_state.affect_history = \
                self.cognitive_state.affect_history[-self.config.max_affect_history:]

        # Update sampler history
        self.syntax_sampler.update_history(state.syntactic_token)

    def _generate_response(
        self,
        state: "DualStreamState",
        is_ood: bool,
    ) -> Tuple[Optional[int], np.ndarray]:
        """
        Generate response token and affect vector.

        Args:
            state: Input state
            is_ood: Whether input was OOD

        Returns:
            (response_token, response_affect) or (None, None) if suppressed
        """
        # Determine response mode
        if is_ood:
            mode = ResponseMode.CAUTIOUS
        else:
            mode = ResponseMode.CONGRUENT

        # Generate syntactic response
        response_token = self._generate_syntactic_response(mode, state)

        if response_token is None:
            return None, np.zeros(16)

        # Generate affective response
        response_affect = self._generate_affective_response(state.affect_vector, mode)

        return response_token, response_affect

    def _generate_syntactic_response(
        self,
        mode: ResponseMode,
        state: "DualStreamState",
    ) -> Optional[int]:
        """
        Generate syntactic response token.

        Uses probabilistic Transformer for CONGRUENT mode,
        falls back to greedy/conservative for CAUTIOUS mode.

        Args:
            mode: Response mode
            state: Input state

        Returns:
            Response token ID, or None if should suppress
        """
        if mode == ResponseMode.SUPPRESS:
            return None

        if self.syntax_transformer is None:
            # No transformer, use simple bigram-like behavior
            # Return the same token (echo response)
            return state.syntactic_token

        # Prepare input prefix
        prefix = torch.tensor([self.cognitive_state.token_history])

        try:
            # Get logits from transformer
            with torch.no_grad():
                logits = self.syntax_transformer(prefix)[:, -1, :]

            # Sample based on mode
            if mode == ResponseMode.CAUTIOUS:
                # Use greedy for cautious mode
                old_mode = self.syntax_sampler.config.mode
                self.syntax_sampler.config.mode = SamplingMode.GREEDY

            result = self.syntax_sampler.sample_next_token(
                logits,
                forbidden_tokens=None,  # Could add special tokens here
            )

            if mode == ResponseMode.CAUTIOUS:
                # Restore original mode
                self.syntax_sampler.config.mode = old_mode

            return result.token_id

        except Exception as e:
            logger.warning(f"Transformer sampling failed: {e}, falling back to echo")
            return state.syntactic_token

    def _generate_affective_response(
        self,
        incoming_affect: np.ndarray,
        mode: ResponseMode,
    ) -> np.ndarray:
        """
        Generate affective response based on incoming affect.

        Implements biologically-inspired affective matching:
        - High arousal (>0.8): De-escalate to avoid panic cascade
        - Low arousal (<0.3): Escalate slightly for engagement
        - Medium arousal: Match for social bonding

        Args:
            incoming_affect: 16D affect vector
            mode: Response mode

        Returns:
            16D response affect vector
        """
        # Assume first dimension is arousal
        arousal = incoming_affect[0]

        if mode == ResponseMode.CAUTIOUS:
            # Cautious: dampen all affect
            return incoming_affect * 0.5

        # Apply affective matching logic
        if arousal > self.config.arousal_deescalation_threshold:
            # De-escalate high arousal
            target_affect = incoming_affect * self.config.arousal_match_factor * 0.75
        elif arousal < self.config.arousal_escalation_threshold:
            # Escalate low arousal for engagement
            target_affect = incoming_affect * 1.2
            # Clamp to reasonable range
            target_affect = np.clip(target_affect, -1.0, 1.0)
        else:
            # Match for social bonding
            target_affect = incoming_affect * self.config.arousal_match_factor

        # Apply smoothing with history
        if self.cognitive_state.affect_history:
            last_affect = self.cognitive_state.affect_history[-1]
            smoothed = (
                self.config.affect_smoothing * last_affect +
                (1 - self.config.affect_smoothing) * target_affect
            )
            return smoothed

        return target_affect

    def _create_action(
        self,
        response_token: int,
        response_affect: np.ndarray,
        sequence: int,
        is_ood: bool,
    ) -> "DualStreamAction":
        """
        Create DualStreamAction from response components.

        Args:
            response_token: Syntactic response token
            response_affect: 16D affect vector
            sequence: Sequence number
            is_ood: Whether this was an OOD response

        Returns:
            DualStreamAction for publishing
        """
        from realtime.action_publisher import DualStreamAction

        # Determine delay (longer for OOD responses)
        delay = (
            self.config.ood_response_delay_ms if is_ood
            else self.config.default_response_delay_ms
        )

        # Determine priority
        priority = "high" if is_ood else "normal"

        return DualStreamAction(
            syntactic_token=response_token,
            affect_vector=response_affect.astype(np.float32),
            temporal_offset_ms=delay,
            priority=priority,
            sequence=sequence + 1,  # Increment sequence
        )

    def reset_cognitive_state(self) -> None:
        """Reset cognitive state (e.g., after session timeout)."""
        self.cognitive_state = CognitiveState(
            token_history=[],
            affect_history=[],
        )
        self.syntax_sampler.history = []
        logger.info("Cognitive state reset")

    def get_stats(self) -> Dict[str, any]:
        """Get agent statistics."""
        return {
            **self.stats,
            "ood_rate": (
                self.stats["ood_detected"] / self.stats["processed"]
                if self.stats["processed"] > 0 else 0
            ),
            "response_rate": (
                self.stats["responses_generated"] / self.stats["processed"]
                if self.stats["processed"] > 0 else 0
            ),
            "current_mode": self.cognitive_state.current_mode.name,
            "token_history_length": len(self.cognitive_state.token_history),
            "affect_history_length": len(self.cognitive_state.affect_history),
        }

    def compute_confidence_score(self, state: "DualStreamState") -> float:
        """
        Compute overall confidence score for input state.

        Combines:
        - Mahalanobis distance confidence (via OOD detector)
        - Syntax probability (via Transformer)
        - Affect coherence (via history consistency)

        Args:
            state: Input dual-stream state

        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Default

        # OOD confidence
        if self.ood_detector:
            ood_confidence = self.ood_detector.compute_confidence(
                state.affect_vector,
                state.syntactic_token,
            )
            confidence += ood_confidence * 0.4

        # Syntax probability (if transformer available)
        if self.syntax_transformer and self.cognitive_state.token_history:
            try:
                prefix = torch.tensor([self.cognitive_state.token_history])
                with torch.no_grad():
                    logits = self.syntax_transformer(prefix)[:, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                    token_prob = probs[0, state.syntactic_token].item()
                    confidence += token_prob * 0.3
            except Exception:
                pass

        # Affect coherence
        if self.cognitive_state.affect_history:
            last_affect = self.cognitive_state.affect_history[-1]
            coherence = 1.0 - np.mean(np.abs(state.affect_vector - last_affect))
            confidence += coherence * 0.3

        return min(1.0, max(0.0, confidence))


# =============================================================================
# Factory Functions and Utilities
# =============================================================================

def create_agent_v3(
    ood_statistics_path: Optional[str] = None,
    transformer_checkpoint_path: Optional[str] = None,
    config: Optional[AgentConfig] = None,
) -> InteractionAgentV3:
    """
    Factory function to create InteractionAgentV3.

    Args:
        ood_statistics_path: Path to OOD statistics JSON
        transformer_checkpoint_path: Path to Transformer checkpoint
        config: Agent configuration

    Returns:
        Configured InteractionAgentV3
    """
    # Load OOD detector
    ood_detector = None
    if ood_statistics_path:
        ood_detector = MahalanobisOOD.load(ood_statistics_path)
        logger.info(f"Loaded OOD detector from {ood_statistics_path}")

    # Load Syntax Transformer
    syntax_transformer = None
    if transformer_checkpoint_path:
        trainer = SyntaxTransformerTrainer.load_checkpoint(transformer_checkpoint_path)
        syntax_transformer = trainer.model
        logger.info(f"Loaded Syntax Transformer from {transformer_checkpoint_path}")

    # Create sampler
    sampler_config = SamplingConfig(
        mode=config.sampling_mode if config else SamplingMode.COMBINED,
        temperature=config.temperature if config else 0.8,
        top_p=config.top_p if config else 0.9,
    )
    syntax_sampler = SyntaxSampler(sampler_config)

    # Create agent
    agent = InteractionAgentV3(
        config=config,
        ood_detector=ood_detector,
        syntax_transformer=syntax_transformer,
        syntax_sampler=syntax_sampler,
    )

    return agent


# Preset configurations

CONSERVATIVE_AGENT_CONFIG = AgentConfig(
    ood_threshold=0.999,  # 99.9% confidence (strict)
    ood_suppression_count=2,  # Faster suppression
    sampling_mode=SamplingMode.COMBINED,
    temperature=0.6,  # Lower temperature
    top_p=0.8,
    arousal_deescalation_threshold=0.7,
)


BALANCED_AGENT_CONFIG = AgentConfig(
    ood_threshold=0.99,  # 99% confidence
    ood_suppression_count=3,
    sampling_mode=SamplingMode.COMBINED,
    temperature=0.8,
    top_p=0.9,
    arousal_deescalation_threshold=0.8,
)


CREATIVE_AGENT_CONFIG = AgentConfig(
    ood_threshold=0.95,  # 95% confidence (permissive)
    ood_suppression_count=5,  # Slower suppression
    sampling_mode=SamplingMode.COMBINED,
    temperature=1.2,  # Higher temperature
    top_p=0.95,
    arousal_deescalation_threshold=0.9,
)


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)

    # Create agent with default config
    agent = create_agent_v3(
        ood_statistics_path=None,  # Demo mode: no OOD detector
        transformer_checkpoint_path=None,  # Demo mode: no transformer
        config=BALANCED_AGENT_CONFIG,
    )

    # Simulate incoming states
    from realtime.action_publisher import DualStreamState

    states = [
        DualStreamState(
            syntactic_token=5,
            affect_vector=np.random.randn(16).astype(np.float32) * 0.1,
            sequence=i,
        )
        for i in range(10)
    ]

    for state in states:
        action = agent.handle_dual_stream_state(state)
        if action:
            print(f"Response: token={action.syntactic_token}, "
                  f"delay={action.temporal_offset_ms}ms")
        else:
            print("Response suppressed")

    # Print stats
    stats = agent.get_stats()
    print(f"\nStats: {stats}")


if __name__ == '__main__':
    main()
