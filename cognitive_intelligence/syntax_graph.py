#!/usr/bin/env python3
"""
Syntax Graph with Laplace Smoothing (Module 2 Deep Dive)

Implements probabilistic transition matrix for discrete syntactic tokens
with Laplace smoothing to prevent zero-probability bigrams from corpus sparsity.

Formula:
    P(t_i | t_{i-1}) = (Count(t_{i-1}, t_i) + α) / (Count(t_{i-1}) + α·N)

where α = 0.01 (smoothing parameter) and N = vocabulary size.

Key Benefits:
- No zero-probability bigrams (all transitions possible)
- Prevents agent from getting stuck in grammar dead-ends
- Handles unseen but biologically valid sequences

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SyntaxGraphConfig:
    """Configuration for syntax graph."""

    # Vocabulary size (number of discrete tokens)
    num_tokens: int = 64

    # Laplace smoothing parameter (α)
    # Prevents zero-probability transitions
    alpha: float = 0.01

    # Default filename for saving/loading
    default_filename: str = "syntax_graph.json"


# =============================================================================
# Syntax Graph Implementation
# =============================================================================


class SyntaxGraph:
    """
    Probabilistic transition matrix with Laplace smoothing.

    Models the probability of transitioning from one syntactic token
    to another, enabling biologically-appropriate call sequencing.

    Example:
        >>> graph = SyntaxGraph(num_tokens=64, alpha=0.01)
        >>> graph.update_from_corpus([[0, 5, 12], [5, 12, 8]])
        >>> valid_next = graph.get_valid_next_tokens(current_token=5)
    """

    def __init__(
        self,
        num_tokens: int = 64,
        alpha: float = 0.01,
    ):
        """
        Initialize syntax graph.

        Args:
            num_tokens: Size of the token vocabulary
            alpha: Laplace smoothing parameter (prevents zero probabilities)
        """
        self.num_tokens = num_tokens
        self.alpha = alpha

        # Initialize with uniform smoothing (all transitions equally likely)
        # This ensures no zero probabilities even before training
        self.transitions = np.full(
            (num_tokens, num_tokens),
            alpha / (num_tokens * alpha),  # Uniform distribution
            dtype=np.float64,
        )

        # Raw counts for updating
        self.counts = np.zeros((num_tokens, num_tokens), dtype=np.float64)

        # Token labels (optional, for interpretability)
        self.token_labels: List[str] = [f"token_{i}" for i in range(num_tokens)]

        logger.info(
            f"SyntaxGraph initialized: {num_tokens} tokens, "
            f"α={alpha}"
        )

    def update_from_corpus(
        self,
        token_sequences: List[List[int] | np.ndarray],
    ) -> None:
        """
        Build smoothed transition matrix from corpus of token sequences.

        Applies Laplace smoothing:
            P(t_i | t_{i-1}) = (Count + α) / (Total + α·N)

        Args:
            token_sequences: List of token sequences (each sequence is list of ints)
        """
        # Reset counts
        self.counts = np.zeros((self.num_tokens, self.num_tokens), dtype=np.float64)

        # Count bigrams in corpus
        for sequence in token_sequences:
            seq_array = np.asarray(sequence)
            if len(seq_array) < 2:
                continue

            # Count transitions
            for i in range(len(seq_array) - 1):
                t_prev = int(seq_array[i])
                t_next = int(seq_array[i + 1])

                # Validate token indices
                if 0 <= t_prev < self.num_tokens and 0 <= t_next < self.num_tokens:
                    self.counts[t_prev, t_next] += 1

        # Apply Laplace smoothing to compute transition probabilities
        self._apply_laplace_smoothing()

        logger.info(
            f"Updated syntax graph from {len(token_sequences)} sequences, "
            f"total_bigrams={int(self.counts.sum())}"
        )

    def _apply_laplace_smoothing(self) -> None:
        """
        Apply Laplace smoothing to raw counts.

        Formula:
            P(t_i | t_{i-1}) = (Count(t_{i-1}, t_i) + α) / (Count(t_{i-1}) + α·N)

        This ensures no zero probabilities, preventing the agent from
        getting stuck in dead-ends where the next token has 0 probability.
        """
        for i in range(self.num_tokens):
            row_sum = self.counts[i].sum()

            # Compute smoothed probabilities for row i
            for j in range(self.num_tokens):
                numerator = self.counts[i, j] + self.alpha
                denominator = row_sum + self.alpha * self.num_tokens

                self.transitions[i, j] = numerator / denominator

        # Verify normalization (each row should sum to ~1.0)
        row_sums = self.transitions.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            # Renormalize to handle floating point errors
            self.transitions = self.transitions / self.transitions.sum(axis=1, keepdims=True)

    def get_transition_probability(
        self,
        from_token: int,
        to_token: int,
    ) -> float:
        """
        Get transition probability P(to_token | from_token).

        Args:
            from_token: Source token index
            to_token: Destination token index

        Returns:
            Probability of transitioning from from_token to to_token
        """
        if not (0 <= from_token < self.num_tokens and 0 <= to_token < self.num_tokens):
            return 0.0

        return float(self.transitions[from_token, to_token])

    def get_valid_next_tokens(
        self,
        current_token: int,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Return top-k valid next tokens by probability.

        Args:
            current_token: Current token index
            top_k: Number of top tokens to return

        Returns:
            List of (token_id, probability) tuples, sorted by probability descending
        """
        if not (0 <= current_token < self.num_tokens):
            return []

        # Get probabilities for all possible next tokens
        probs = self.transitions[current_token]

        # Get top-k indices
        top_indices = np.argsort(probs)[-top_k:][::-1]

        # Return as list of (token, probability) tuples
        return [
            (int(idx), float(probs[idx]))
            for idx in top_indices
        ]

    def sample_next_token(
        self,
        current_token: int,
        temperature: float = 1.0,
    ) -> int:
        """
        Sample next token from transition distribution.

        Args:
            current_token: Current token index
            temperature: Sampling temperature (>1 = more random, <1 = more deterministic)

        Returns:
            Sampled next token index
        """
        if not (0 <= current_token < self.num_tokens):
            return np.random.randint(0, self.num_tokens)

        # Get probabilities and apply temperature
        probs = self.transitions[current_token]

        if temperature != 1.0:
            # Apply temperature scaling
            log_probs = np.log(probs + 1e-10) / temperature
            exp_probs = np.exp(log_probs - log_probs.max())
            probs = exp_probs / exp_probs.sum()

        # Sample from distribution
        return int(np.random.choice(self.num_tokens, p=probs))

    def has_valid_transition(
        self,
        from_token: int,
        to_token: int,
    ) -> bool:
        """
        Check if transition has non-zero probability.

        With Laplace smoothing, this should always return True
        (unless there's a bug in the implementation).

        Args:
            from_token: Source token
            to_token: Destination token

        Returns:
            True if transition probability > 0
        """
        prob = self.get_transition_probability(from_token, to_token)
        return prob > 0

    def get_entropy(self, token: int) -> float:
        """
        Compute entropy of transition distribution for a token.

        Higher entropy = more uncertainty/randomness in next token.
        Lower entropy = more deterministic transitions.

        Args:
            token: Token index

        Returns:
            Entropy in nats
        """
        if not (0 <= token < self.num_tokens):
            return 0.0

        probs = self.transitions[token]
        # Avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)

        return float(-np.sum(probs * np.log(probs)))

    def save(self, path: Optional[Path] = None) -> None:
        """
        Save syntax graph to JSON file.

        Args:
            path: Path to save file (default: syntax_graph.json)
        """
        if path is None:
            path = Path(self.token_labels[0].split('_')[0] + '_' + self.__class__.__name__.lower() + '.json')

        path = Path(path)

        data = {
            "num_tokens": self.num_tokens,
            "alpha": self.alpha,
            "transitions": self.transitions.tolist(),
            "counts": self.counts.tolist(),
            "token_labels": self.token_labels,
        }

        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved syntax graph to {path}")

    @classmethod
    def load(cls, path: Path) -> "SyntaxGraph":
        """
        Load syntax graph from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Loaded SyntaxGraph instance
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Syntax graph file not found: {path}")

        data = json.loads(path.read_text())

        # Create instance
        graph = cls(
            num_tokens=data["num_tokens"],
            alpha=data["alpha"],
        )

        # Load data
        graph.transitions = np.array(data["transitions"], dtype=np.float64)
        graph.counts = np.array(data["counts"], dtype=np.float64)
        graph.token_labels = data.get("token_labels", [f"token_{i}" for i in range(graph.num_tokens)])

        logger.info(f"Loaded syntax graph from {path}")
        return graph


# =============================================================================
# Utility Functions
# =============================================================================


def create_syntax_graph(
    num_tokens: int = 64,
    alpha: float = 0.01,
) -> SyntaxGraph:
    """Factory function to create syntax graph."""
    return SyntaxGraph(num_tokens=num_tokens, alpha=alpha)


def build_syntax_graph_from_corpus(
    token_sequences: List[List[int]],
    num_tokens: int = 64,
    alpha: float = 0.01,
) -> SyntaxGraph:
    """
    Build syntax graph from corpus.

    Args:
        token_sequences: List of token sequences
        num_tokens: Vocabulary size
        alpha: Laplace smoothing parameter

    Returns:
        Trained SyntaxGraph
    """
    graph = create_syntax_graph(num_tokens=num_tokens, alpha=alpha)
    graph.update_from_corpus(token_sequences)
    return graph
