#!/usr/bin/env python3
"""
Syntax Sampler: Probabilistic Token Sampling

Handles probabilistic sampling from the Syntax Transformer output using:
- Temperature scaling (controls randomness)
- Top-k filtering (restrict to k most likely tokens)
- Top-p (nucleus) sampling (restrict to minimal set covering p% probability)

This replaces the rigid bigram automaton's binary valid/invalid check
with a probabilistic approach that enables novel syntax.

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Tuple, Dict, Callable

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SamplingMode(IntEnum):
    """Sampling strategy modes."""
    GREEDY = 0  # Always pick highest probability
    TEMPERATURE = 1  # Temperature sampling
    TOP_K = 2  # Top-k sampling
    TOP_P = 3  # Nucleus (top-p) sampling
    COMBINED = 4  # Temperature + top-p (recommended)


@dataclass
class SamplingConfig:
    """Configuration for probabilistic sampling."""
    # Sampling mode
    mode: SamplingMode = SamplingMode.COMBINED

    # Temperature (lower = more conservative, higher = more random)
    temperature: float = 0.8

    # Top-k sampling parameters
    top_k: Optional[int] = 10  # If set, only consider top-k tokens

    # Top-p (nucleus) sampling parameters
    top_p: float = 0.9  # Nucleus threshold (0.9 = 90% probability mass)

    # Filtering
    min_probability: float = 1e-8  # Minimum probability to consider
    forbid_repetition: bool = True  # Prevent immediate token repetition
    max_repetition_penalty: float = 0.5  # Penalty for repeated tokens


@dataclass
class SamplingResult:
    """Result of a sampling operation."""
    token_id: int
    probability: float
    entropy: float  # Information entropy of the distribution
    was_forced: bool  # True if fallback (e.g., all tokens filtered)
    num_candidates: int  # Number of tokens considered


class SyntaxSampler:
    """
    Probabilistic sampler for VQ-VAE token sequences.

    Replaces the rigid bigram automaton with a flexible sampling
    mechanism that enables novel but statistically plausible syntax.
    """

    def __init__(
        self,
        config: Optional[SamplingConfig] = None,
    ):
        if config is None:
            config = SamplingConfig()

        self.config = config
        self.history: List[int] = []  # Track recent tokens for repetition penalty

    def sample_next_token(
        self,
        logits: torch.Tensor,
        forbidden_tokens: Optional[set[int]] = None,
    ) -> SamplingResult:
        """
        Sample the next token from the model's output logits.

        Args:
            logits: (num_tokens,) or (1, num_tokens) output logits
            forbidden_tokens: Set of tokens to forbid (e.g., special tokens)

        Returns:
            SamplingResult with selected token and metadata
        """
        # Ensure logits are 1D
        if logits.dim() == 2:
            logits = logits.squeeze(0)

        num_tokens = logits.shape[0]

        # Apply repetition penalty if configured
        if self.config.forbid_repetition and self.history:
            logits = self._apply_repetition_penalty(logits)

        # Apply temperature
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature

        # Apply sampling strategy
        if self.config.mode == SamplingMode.GREEDY:
            return self._greedy_sample(logits, forbidden_tokens)
        elif self.config.mode == SamplingMode.TOP_K:
            return self._top_k_sample(logits, self.config.top_k, forbidden_tokens)
        elif self.config.mode == SamplingMode.TOP_P:
            return self._top_p_sample(logits, self.config.top_p, forbidden_tokens)
        elif self.config.mode == SamplingMode.COMBINED:
            # Temperature + top-p (recommended)
            return self._top_p_sample(logits, self.config.top_p, forbidden_tokens)
        else:
            # Default to temperature sampling
            return self._temperature_sample(logits, forbidden_tokens)

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Apply penalty to recently generated tokens."""
        penalized_logits = logits.clone()

        for token_id in self.history[-3:]:  # Penalize last 3 tokens
            penalized_logits[token_id] -= self.config.max_repetition_penalty

        return penalized_logits

    def _greedy_sample(
        self,
        logits: torch.Tensor,
        forbidden_tokens: Optional[set[int]] = None,
    ) -> SamplingResult:
        """Always pick the highest probability token."""
        # Mask forbidden tokens
        if forbidden_tokens:
            logits = logits.clone()
            for token_id in forbidden_tokens:
                logits[token_id] = -float('Inf')

        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Get max
        token_id = int(torch.argmax(probs).item())
        probability = float(probs[token_id])

        # Calculate entropy
        entropy = float(self._compute_entropy(probs))

        return SamplingResult(
            token_id=token_id,
            probability=probability,
            entropy=entropy,
            was_forced=False,
            num_candidates=1,
        )

    def _temperature_sample(
        self,
        logits: torch.Tensor,
        forbidden_tokens: Optional[set[int]] = None,
    ) -> SamplingResult:
        """Sample from temperature-scaled distribution."""
        # Mask forbidden tokens
        if forbidden_tokens:
            logits = logits.clone()
            for token_id in forbidden_tokens:
                logits[token_id] = -float('Inf')

        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample
        token_id = int(torch.multinomial(probs, 1).item())
        probability = float(probs[token_id])

        # Calculate entropy
        entropy = float(self._compute_entropy(probs))

        return SamplingResult(
            token_id=token_id,
            probability=probability,
            entropy=entropy,
            was_forced=False,
            num_candidates=int((probs > self.config.min_probability).sum()),
        )

    def _top_k_sample(
        self,
        logits: torch.Tensor,
        k: int,
        forbidden_tokens: Optional[set[int]] = None,
    ) -> SamplingResult:
        """Sample from top-k most likely tokens."""
        num_tokens = logits.shape[0]
        k = min(k, num_tokens)

        # Get top-k
        top_k_logits, top_k_indices = torch.topk(logits, k)

        # Mask forbidden tokens
        if forbidden_tokens:
            mask = torch.ones(k, dtype=torch.bool)
            for i, token_id in enumerate(top_k_indices):
                if int(token_id) in forbidden_tokens:
                    mask[i] = False

            if not mask.any():
                # All top-k tokens forbidden, fall back to greedy
                return self._greedy_sample(logits, forbidden_tokens)

            top_k_logits = top_k_logits[mask]
            top_k_indices = top_k_indices[mask]

        # Sample from top-k
        probs = F.softmax(top_k_logits, dim=-1)
        idx = torch.multinomial(probs, 1)

        token_id = int(top_k_indices[idx].item())
        probability = float(probs[idx])

        # Calculate entropy
        entropy = float(self._compute_entropy(probs))

        return SamplingResult(
            token_id=token_id,
            probability=probability,
            entropy=entropy,
            was_forced=False,
            num_candidates=len(top_k_indices),
        )

    def _top_p_sample(
        self,
        logits: torch.Tensor,
        p: float,
        forbidden_tokens: Optional[set[int]] = None,
    ) -> SamplingResult:
        """
        Nucleus (top-p) sampling.

        Select the smallest set of tokens whose cumulative probability
        exceeds p, then sample from that set.
        """
        # Sort by logit descending
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        # Compute cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find the cutoff point
        sorted_indices_to_remove = cumulative_probs > p

        # Shift to keep at least the first token
        sorted_indices_to_remove[0] = False

        # Check if we have valid tokens after removing
        if sorted_indices_to_remove.all():
            # All tokens would be removed, fall back to greedy
            return self._greedy_sample(logits, forbidden_tokens)

        # Remove tokens above threshold
        filtered_logits = sorted_logits.clone()
        filtered_logits[sorted_indices_to_remove] = -float('Inf')

        # Mask forbidden tokens
        if forbidden_tokens:
            for i, token_id in enumerate(sorted_indices):
                if int(token_id) in forbidden_tokens:
                    filtered_logits[i] = -float('Inf')

        # Check if any valid tokens remain
        if torch.isinf(filtered_logits).all():
            # All tokens filtered, pick the highest probability (may be forbidden)
            return self._greedy_sample(logits, forbidden_tokens)

        # Sample from filtered distribution
        probs = F.softmax(filtered_logits, dim=-1)
        idx = torch.multinomial(probs, 1)

        selected_idx = int(idx.item())
        token_id = int(sorted_indices[selected_idx].item())
        probability = float(probs[idx])

        # Calculate entropy of filtered distribution
        entropy = float(self._compute_entropy(probs))

        # Count number of candidates
        num_candidates = int(~torch.isinf(filtered_logits).sum())

        return SamplingResult(
            token_id=token_id,
            probability=probability,
            entropy=entropy,
            was_forced=False,
            num_candidates=num_candidates,
        )

    def _compute_entropy(
        self,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Shannon entropy of a probability distribution."""
        # Filter out near-zero probabilities
        valid_mask = probs > self.config.min_probability
        valid_probs = probs[valid_mask]

        if valid_probs.numel() == 0:
            return torch.tensor(0.0)

        entropy = -torch.sum(valid_probs * torch.log(valid_probs + 1e-10))
        return entropy

    def update_history(self, token_id: int) -> None:
        """Update history with newly generated token."""
        self.history.append(token_id)
        # Keep only recent history
        if len(self.history) > 10:
            self.history = self.history[-10:]

    def compute_sequence_probability(
        self,
        model: Callable,
        token_sequence: List[int],
    ) -> float:
        """
        Compute the probability of a complete token sequence.

        Useful for validation and debugging.

        Args:
            model: SyntaxTransformer that returns logits
            token_sequence: List of token IDs

        Returns:
            Joint probability P(token_0, ..., token_T)
        """
        model.eval()
        log_prob = 0.0

        with torch.no_grad():
            for i in range(len(token_sequence) - 1):
                # Get logits for current prefix
                prefix = torch.tensor([token_sequence[:i+1]])
                logits = model(prefix)[:, -1, :]  # Get last token logits

                # Get probability of next token
                probs = F.softmax(logits, dim=-1)
                next_token = token_sequence[i + 1]
                token_prob = probs[0, next_token].item()

                log_prob += np.log(token_prob + 1e-10)

        return np.exp(log_prob)

    def get_top_k_tokens(
        self,
        logits: torch.Tensor,
        k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Get the top-k most likely tokens with their probabilities.

        Useful for diagnostics and debugging.

        Args:
            logits: Output logits
            k: Number of top tokens to return

        Returns:
            List of (token_id, probability) tuples
        """
        if logits.dim() == 2:
            logits = logits.squeeze(0)

        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k)

        return [
            (int(token_id), float(prob))
            for token_id, prob in zip(top_k_indices, top_k_probs)
        ]


# Preset configurations

CONSERVATIVE_SAMPLING = SamplingConfig(
    mode=SamplingMode.COMBINED,
    temperature=0.6,  # Lower temperature = more conservative
    top_p=0.8,  # Smaller nucleus = more focused
    forbid_repetition=True,
)

BALANCED_SAMPLING = SamplingConfig(
    mode=SamplingMode.COMBINED,
    temperature=0.8,
    top_p=0.9,
    forbid_repetition=True,
)

CREATIVE_SAMPLING = SamplingConfig(
    mode=SamplingMode.COMBINED,
    temperature=1.2,  # Higher temperature = more creative
    top_p=0.95,  # Larger nucleus = more diverse
    forbid_repetition=False,
)


def create_syntax_sampler(
    config: Optional[SamplingConfig] = None,
) -> SyntaxSampler:
    """
    Factory function to create syntax sampler.

    Args:
        config: Sampling configuration (uses BALANCED_SAMPLING if None)

    Returns:
        Configured SyntaxSampler
    """
    return SyntaxSampler(config)


def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)

    # Create dummy logits (64 tokens)
    np.random.seed(42)
    logits = torch.randn(64)

    # Test different sampling modes
    configs = [
        ("Greedy", SamplingConfig(mode=SamplingMode.GREEDY)),
        ("Conservative", CONSERVATIVE_SAMPLING),
        ("Balanced", BALANCED_SAMPLING),
        ("Creative", CREATIVE_SAMPLING),
    ]

    for name, config in configs:
        sampler = create_syntax_sampler(config)
        result = sampler.sample_next_token(logits)

        print(f"{name}: token={result.token_id}, p={result.probability:.3f}, "
              f"entropy={result.entropy:.3f}, candidates={result.num_candidates}")


if __name__ == '__main__':
    main()
