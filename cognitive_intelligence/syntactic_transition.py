#!/usr/bin/env python3
"""
Laplace-Smoothed Syntactic Transition Matrix

Models bigram transitions between syntactic tokens with Laplace smoothing (α=0.01)
to handle unobserved bigrams and enable grammatically valid generative variations.

The transition matrix captures:
- Observed bigram probabilities from training data
- Smoothed probabilities for unseen transitions
- Backoff to unigram for cold-start

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class TransitionConfig:
    """Configuration for transition matrix learning."""
    vocab_size: int = 64  # Number of syntactic tokens
    alpha: float = 0.01  # Laplace smoothing parameter
    eos_token: int = 0  # End-of-sequence token
    bos_token: int = 1  # Beginning-of-sequence token
    min_count: int = 1  # Minimum count for a transition to be considered
    temperature: float = 1.0  # Sampling temperature


class SyntacticTransitionMatrix(nn.Module):
    """
    Bigram transition matrix with Laplace smoothing.

    The transition matrix P(j|i) represents the probability of token j
    following token i. Laplace smoothing ensures that:

    P(j|i) = (count(i→j) + α) / (sum_j count(i→j) + α * V)

    where V is the vocabulary size. This prevents zero probability for
    unseen transitions and enables novel but grammatically valid sequences.

    Example:
        >>> transitions = SyntacticTransitionMatrix(vocab_size=64, alpha=0.01)
        >>> sequences = [[1, 5, 10, 0], [1, 8, 15, 0]]  # Training sequences
        >>> transitions.update_counts(sequences)
        >>> transitions.finalize()  # Build probability matrix
        >>> # Sample next token
        >>> next_token = transitions.sample_next(current_token=5)
    """

    def __init__(
        self,
        vocab_size: int = 64,
        alpha: float = 0.01,
        eos_token: int = 0,
        bos_token: int = 1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.alpha = alpha
        self.eos_token = eos_token
        self.bos_token = bos_token

        # Raw count matrix (for incremental updates)
        self.register_buffer(
            'count_matrix',
            torch.zeros(vocab_size, vocab_size, dtype=torch.float32)
        )

        # Smoothed probability matrix (computed after finalize())
        self.register_buffer(
            'prob_matrix',
            torch.zeros(vocab_size, vocab_size, dtype=torch.float32)
        )

        # Log probability matrix (for numerical stability)
        self.register_buffer(
            'log_prob_matrix',
            torch.zeros(vocab_size, vocab_size, dtype=torch.float32)
        )

        # Unigram counts (for backoff)
        self.register_buffer(
            'unigram_counts',
            torch.zeros(vocab_size, dtype=torch.float32)
        )

        self._finalized = False

        logger.info(
            f"SyntacticTransitionMatrix initialized: "
            f"vocab_size={vocab_size}, alpha={alpha}"
        )

    def update_counts(self, sequences: Iterable[List[int]]) -> None:
        """
        Update bigram counts from sequences.

        Args:
            sequences: Iterable of token sequences (lists of integers)
        """
        for seq in sequences:
            # Add BOS at start, EOS at end
            full_seq = [self.bos_token] + list(seq) + [self.eos_token]

            for i in range(len(full_seq) - 1):
                src = full_seq[i]
                tgt = full_seq[i + 1]

                if 0 <= src < self.vocab_size and 0 <= tgt < self.vocab_size:
                    self.count_matrix[src, tgt] += 1
                    self.unigram_counts[tgt] += 1

        self._finalized = False

    def update_from_batch(self, tokens: torch.Tensor) -> None:
        """
        Update counts from a batch of token sequences.

        Args:
            tokens: Tensor of shape (batch, seq_len) with token IDs
        """
        batch_size, seq_len = tokens.shape

        for b in range(batch_size):
            for i in range(seq_len - 1):
                src = tokens[b, i].item()
                tgt = tokens[b, i + 1].item()

                if 0 <= src < self.vocab_size and 0 <= tgt < self.vocab_size:
                    self.count_matrix[src, tgt] += 1
                    self.unigram_counts[tgt] += 1

        self._finalized = False

    def finalize(self) -> None:
        """
        Build smoothed probability matrix from counts.

        Applies Laplace smoothing:
        P(j|i) = (count(i→j) + α) / (sum_k count(i→k) + α * V)
        """
        # Add alpha smoothing
        smoothed_counts = self.count_matrix + self.alpha

        # Compute row sums
        row_sums = smoothed_counts.sum(dim=1, keepdim=True)

        # Normalize
        self.prob_matrix = smoothed_counts / (row_sums + 1e-10)

        # Compute log probabilities
        self.log_prob_matrix = torch.log(self.prob_matrix + 1e-10)

        self._finalized = True

        # Log statistics
        num_nonzero = (self.count_matrix > 0).sum().item()
        total_pairs = self.vocab_size * self.vocab_size
        sparsity = 1.0 - num_nonzero / total_pairs

        logger.info(
            f"Transition matrix finalized:\n"
            f"  Non-zero transitions: {num_nonzero}/{total_pairs} ({100-sparsity:.1f}%)\n"
            f"  Sparsity: {sparsity:.2%}\n"
            f"  Alpha: {self.alpha}"
        )

    def forward(self, src_tokens: torch.Tensor) -> torch.Tensor:
        """
        Get transition probabilities for source tokens.

        Args:
            src_tokens: Source token IDs (Batch,) or (Batch, 1)

        Returns:
            Probability distribution over next tokens (Batch, vocab_size)
        """
        if not self._finalized:
            self.finalize()

        if src_tokens.dim() == 1:
            src_tokens = src_tokens.unsqueeze(1)

        batch_size = src_tokens.shape[0]

        # Gather probability rows for each source token
        probs = self.prob_matrix[src_tokens.squeeze(1)]  # (Batch, vocab_size)

        return probs

    def log_prob(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of target tokens given source tokens.

        Args:
            src_tokens: Source token IDs (Batch,)
            tgt_tokens: Target token IDs (Batch,)

        Returns:
            Log probabilities (Batch,)
        """
        if not self._finalized:
            self.finalize()

        batch_size = src_tokens.shape[0]

        # Gather log probabilities
        log_probs = self.log_prob_matrix[src_tokens, tgt_tokens]

        return log_probs

    def sample_next(
        self,
        src_token: int,
        temperature: float = 1.0,
        forbidden_tokens: Optional[List[int]] = None,
    ) -> int:
        """
        Sample next token given current token.

        Args:
            src_token: Current token ID
            temperature: Sampling temperature (higher = more random)
            forbidden_tokens: Tokens to exclude from sampling

        Returns:
            Sampled next token ID
        """
        if not self._finalized:
            self.finalize()

        # Get probability distribution
        probs = self.prob_matrix[src_token].clone()

        # Apply temperature
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)
            probs = probs / probs.sum()

        # Zero out forbidden tokens
        if forbidden_tokens:
            for t in forbidden_tokens:
                probs[t] = 0.0

        # Renormalize
        probs = probs / probs.sum()

        # Sample
        next_token = torch.multinomial(probs, num_samples=1).item()

        return next_token

    def generate_sequence(
        self,
        max_length: int = 20,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> List[int]:
        """
        Generate a token sequence using the transition matrix.

        Args:
            max_length: Maximum sequence length
            temperature: Sampling temperature
            seed: Random seed for reproducibility

        Returns:
            Generated token sequence (excluding BOS, including EOS)
        """
        if seed is not None:
            torch.manual_seed(seed)

        sequence = [self.bos_token]
        current_token = self.bos_token

        for _ in range(max_length - 1):
            next_token = self.sample_next(current_token, temperature=temperature)

            if next_token == self.eos_token:
                sequence.append(next_token)
                break

            sequence.append(next_token)
            current_token = next_token

        return sequence[1:]  # Remove BOS

    def sequence_log_prob(self, sequence: List[int]) -> float:
        """
        Compute total log probability of a sequence.

        Args:
            sequence: Token sequence (should include EOS)

        Returns:
            Total log probability
        """
        if not self._finalized:
            self.finalize()

        # Add BOS if not present
        full_seq = [self.bos_token] + list(sequence)

        total_log_prob = 0.0
        for i in range(len(full_seq) - 1):
            src = full_seq[i]
            tgt = full_seq[i + 1]
            total_log_prob += self.log_prob_matrix[src, tgt].item()

        return total_log_prob

    def perplexity(self, sequences: Iterable[List[int]]) -> float:
        """
        Compute perplexity on a set of sequences.

        Lower perplexity indicates better fit.

        Args:
            sequences: Test sequences

        Returns:
            Perplexity score
        """
        if not self._finalized:
            self.finalize()

        total_log_prob = 0.0
        total_tokens = 0

        for seq in sequences:
            full_seq = [self.bos_token] + list(seq)

            for i in range(len(full_seq) - 1):
                src = full_seq[i]
                tgt = full_seq[i + 1]
                total_log_prob += self.log_prob_matrix[src, tgt].item()
                total_tokens += 1

        if total_tokens == 0:
            return float('inf')

        # Perplexity = exp(-1/N * sum(log P))
        avg_log_prob = total_log_prob / total_tokens
        perplexity = torch.exp(torch.tensor(-avg_log_prob)).item()

        return perplexity

    def get_entropy(self, src_token: int) -> float:
        """
        Compute entropy of transition distribution for a token.

        Higher entropy = more uncertainty = more diverse continuations.

        Args:
            src_token: Source token ID

        Returns:
            Entropy in nats
        """
        if not self._finalized:
            self.finalize()

        probs = self.prob_matrix[src_token]
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

        return entropy

    def save(self, path: Path) -> None:
        """Save transition matrix to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'vocab_size': self.vocab_size,
            'alpha': self.alpha,
            'eos_token': self.eos_token,
            'bos_token': self.bos_token,
            'count_matrix': self.count_matrix,
            'prob_matrix': self.prob_matrix,
            'log_prob_matrix': self.log_prob_matrix,
            'unigram_counts': self.unigram_counts,
            'finalized': self._finalized,
        }, path)

        logger.info(f"Saved transition matrix to {path}")

    def load(self, path: Path) -> None:
        """Load transition matrix from file."""
        path = Path(path)
        checkpoint = torch.load(path, map_location='cpu')

        self.vocab_size = checkpoint['vocab_size']
        self.alpha = checkpoint['alpha']
        self.eos_token = checkpoint['eos_token']
        self.bos_token = checkpoint['bos_token']

        self.count_matrix = checkpoint['count_matrix']
        self.prob_matrix = checkpoint['prob_matrix']
        self.log_prob_matrix = checkpoint['log_prob_matrix']
        self.unigram_counts = checkpoint['unigram_counts']
        self._finalized = checkpoint['finalized']

        logger.info(f"Loaded transition matrix from {path}")


def create_transition_matrix(
    vocab_size: int = 64,
    alpha: float = 0.01,
    sequences: Optional[Iterable[List[int]]] = None,
) -> SyntacticTransitionMatrix:
    """
    Factory function to create and optionally train a transition matrix.

    Args:
        vocab_size: Size of the vocabulary
        alpha: Laplace smoothing parameter
        sequences: Optional training sequences

    Returns:
        Trained transition matrix
    """
    matrix = SyntacticTransitionMatrix(vocab_size=vocab_size, alpha=alpha)

    if sequences is not None:
        matrix.update_counts(sequences)
        matrix.finalize()

    return matrix


# Integration with VQ-VAE

class VQVAEWithTransitions(nn.Module):
    """
    VQ-VAE with transition matrix for syntactic modeling.

    Combines discrete tokenization with bigram language modeling
    for grammatically aware generation.
    """

    def __init__(
        self,
        vqvae: nn.Module,
        transition_matrix: SyntacticTransitionMatrix,
        transition_weight: float = 0.1,
    ):
        super().__init__()

        self.vqvae = vqvae
        self.transitions = transition_matrix
        self.transition_weight = transition_weight

    def forward(
        self,
        x: torch.Tensor,
        prev_tokens: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Forward pass with transition-aware loss.

        Args:
            x: Input features
            prev_tokens: Previous token IDs for transition loss

        Returns:
            (reconstruction, vq_loss, transition_loss, tokens)
        """
        # VQ-VAE forward
        x_recon, z, z_q, tokens, perplexity = self.vqvae(x)

        # VQ loss
        vq_losses = self.vqvae.loss_function(x, x_recon, z, z_q)
        total_loss = vq_losses['total_loss']

        # Transition loss (if previous tokens provided)
        transition_loss = torch.tensor(0.0)
        if prev_tokens is not None:
            # Compute log prob of current tokens given previous
            log_probs = self.transitions.log_prob(prev_tokens, tokens.flatten())
            transition_loss = -log_probs.mean()
            total_loss = total_loss + self.transition_weight * transition_loss

        return (
            x_recon,
            total_loss,
            transition_loss.item() if isinstance(transition_loss, torch.Tensor) else transition_loss,
            tokens,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test transition matrix
    transitions = create_transition_matrix(vocab_size=64, alpha=0.01)

    # Create some fake training sequences
    sequences = [
        [1, 5, 10, 15, 0],
        [1, 8, 12, 0],
        [1, 5, 20, 25, 30, 0],
        [1, 8, 15, 0],
    ]

    print("Training transition matrix...")
    transitions.update_counts(sequences)
    transitions.finalize()

    # Sample sequences
    print("\nGenerated sequences:")
    for i in range(5):
        seq = transitions.generate_sequence(max_length=10, temperature=0.8)
        log_p = transitions.sequence_log_prob(seq)
        print(f"  {seq} (logP={log_p:.2f})")

    # Compute perplexity
    test_sequences = [
        [1, 5, 10, 0],
        [1, 8, 12, 0],
    ]
    ppl = transitions.perplexity(test_sequences)
    print(f"\nPerplexity: {ppl:.2f}")
