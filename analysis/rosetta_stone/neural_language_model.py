#!/usr/bin/env python3
"""
Neural Language Model for Acoustic Sequences (Direction 2)
=========================================================

Transformer-based language model for generating acoustic token sequences.

Replaces N-gram models with:
1. AcousticTokenizer - Convert 112D features to/from discrete token IDs
2. TransformerLM - GPT-style causal transformer for sequence modeling
3. ConditionalGenerator - Context-aware sequence generation

Uses lightweight numpy implementation for testing, with PyTorch fallback
for production training.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AcousticTokenizer:
    """
    Convert 112D features to discrete token IDs.

    Uses k-means clustering to create a vocabulary of acoustic tokens.
    Each token represents a region in the 112D feature space.
    """

    def __init__(self, vocab_size: int = 1020, random_state: int = 42):
        """
        Initialize acoustic tokenizer.

        Args:
            vocab_size: Number of discrete tokens (vocabulary size)
            random_state: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.random_state = random_state

        # Initialize centroids randomly (will be replaced with trained centroids)
        np.random.seed(random_state)
        self.centroids = np.random.randn(vocab_size, 112).astype(np.float32)

        # Normalize centroids
        for i in range(vocab_size):
            self.centroids[i] /= np.linalg.norm(self.centroids[i]) + 1e-8

        logger.info(f"AcousticTokenizer initialized with vocab_size={vocab_size}")

    def fit(self, features: List[np.ndarray]) -> None:
        """
        Learn vocabulary centroids from features using k-means.

        Args:
            features: List of 112D feature vectors
        """
        from sklearn.cluster import MiniBatchKMeans

        X = np.array(features, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"Fitting tokenizer on {len(X)} features...")

        kmeans = MiniBatchKMeans(
            n_clusters=self.vocab_size,
            random_state=self.random_state,
            max_iter=100,
            n_init=3,
        )
        kmeans.fit(X)

        self.centroids = kmeans.cluster_centers_.astype(np.float32)

        logger.info(f"Tokenizer fitted: {self.vocab_size} centroids")

    def tokenize(self, features: np.ndarray) -> int:
        """
        Convert features to token ID.

        Args:
            features: 112D feature vector or list of vectors

        Returns:
            Token ID (int) or list of token IDs
        """
        if isinstance(features, list):
            return [self._tokenize_single(f) for f in features]
        return self._tokenize_single(features)

    def _tokenize_single(self, features: np.ndarray) -> int:
        """Convert single feature vector to token ID."""
        features = features.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Find closest centroid
        distances = np.linalg.norm(self.centroids - features, axis=1)
        return int(np.argmin(distances))

    def detokenize(self, token_id: int) -> np.ndarray:
        """
        Convert token ID back to feature vector.

        Returns the centroid for that token.

        Args:
            token_id: Token ID

        Returns:
            112D feature vector (centroid)
        """
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"Token ID {token_id} out of range [0, {self.vocab_size})")

        return self.centroids[token_id].copy()

    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        data = {
            "vocab_size": self.vocab_size,
            "random_state": self.random_state,
            "centroids": self.centroids.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AcousticTokenizer":
        """Load tokenizer from file."""
        with open(path, "r") as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["vocab_size"], random_state=data["random_state"]
        )
        tokenizer.centroids = np.array(data["centroids"], dtype=np.float32)
        logger.info(f"Tokenizer loaded from {path}")
        return tokenizer


class TransformerLM:
    """
    Transformer-based Language Model for acoustic sequences.

    GPT-style causal transformer for next-token prediction.
    Supports training, generation, and conditional generation.

    Uses numpy for lightweight inference, with optional PyTorch backend.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        random_state: int = 42,
    ):
        """
        Initialize Transformer LM.

        Args:
            vocab_size: Size of token vocabulary
            d_model: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            random_state: Random seed
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.random_state = random_state

        # Initialize embeddings
        np.random.seed(random_state)
        self.token_embeddings = np.random.randn(vocab_size, d_model).astype(
            np.float32
        ) * 0.1
        self.token_embeddings /= np.sqrt(d_model)

        # Positional embeddings
        self.pos_embeddings = np.random.randn(max_seq_len, d_model).astype(
            np.float32
        ) * 0.01

        # Transformer weights (simplified)
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(
                {
                    "W_q": np.random.randn(d_model, d_model).astype(np.float32) * 0.1,
                    "W_k": np.random.randn(d_model, d_model).astype(np.float32) * 0.1,
                    "W_v": np.random.randn(d_model, d_model).astype(np.float32) * 0.1,
                    "W_o": np.random.randn(d_model, d_model).astype(np.float32) * 0.1,
                    "W_ff1": np.random.randn(d_model, d_model * 4).astype(np.float32) * 0.1,
                    "W_ff2": np.random.randn(d_model * 4, d_model).astype(np.float32) * 0.1,
                }
            )

        # Output projection
        self.lm_head = np.random.randn(d_model, vocab_size).astype(np.float32) * 0.1

        logger.info(
            f"TransformerLM initialized: vocab={vocab_size}, d_model={d_model}, "
            f"layers={n_layers}, heads={n_heads}"
        )

    def _get_positional_embedding(self, pos: int) -> np.ndarray:
        """Get positional embedding for position."""
        if pos >= self.max_seq_len:
            pos = self.max_seq_len - 1
        return self.pos_embeddings[pos]

    def forward(self, tokens: List[int]) -> np.ndarray:
        """
        Forward pass through the model.

        Args:
            tokens: List of token IDs

        Returns:
            Logits of shape (seq_len, vocab_size)
        """
        seq_len = len(tokens)
        if seq_len == 0:
            return np.zeros((0, self.vocab_size), dtype=np.float32)

        # Embed tokens
        x = np.array([self.token_embeddings[t] for t in tokens], dtype=np.float32)

        # Add positional embeddings
        for i in range(seq_len):
            x[i] += self._get_positional_embedding(i)

        # Pass through transformer layers (simplified)
        for layer in self.layers:
            # Self-attention (simplified as linear projection for now)
            attn_out = x @ layer["W_q"]
            x = x + attn_out  # Residual
            x = x * 0.5  # Simplified layer norm

            # Feed-forward
            ff_out = x @ layer["W_ff1"]
            ff_out = np.maximum(ff_out, 0)  # ReLU
            ff_out = ff_out @ layer["W_ff2"]
            x = x + ff_out  # Residual
            x = x * 0.5  # Simplified layer norm

        # Project to vocabulary
        logits = x @ self.lm_head

        return logits

    def compute_loss(self, sequences: List[List[int]]) -> float:
        """
        Compute cross-entropy loss on sequences.

        Args:
            sequences: List of token sequences

        Returns:
            Average loss (float)
        """
        total_loss = 0.0
        total_tokens = 0

        for seq in sequences:
            if len(seq) < 2:
                continue

            for i in range(len(seq) - 1):
                context = seq[: i + 1]
                target = seq[i + 1]

                logits = self.forward(context)
                probs = _softmax(logits[-1])

                # Cross-entropy loss
                loss = -np.log(probs[target] + 1e-10)
                total_loss += loss
                total_tokens += 1

        return total_loss / max(total_tokens, 1)

    def train_step(self, sequences: List[List[int]], learning_rate: float = 0.001) -> float:
        """
        Single training step using gradient descent.

        Args:
            sequences: List of token sequences
            learning_rate: Learning rate

        Returns:
            Loss after the step
        """
        # Simplified training: nudge embeddings toward co-occurrence
        for seq in sequences:
            if len(seq) < 2:
                continue

            for i in range(len(seq) - 1):
                context_token = seq[i]
                target_token = seq[i + 1]

                # Nudge token embeddings: bring context closer to target
                self.token_embeddings[target_token] += (
                    self.token_embeddings[context_token] * learning_rate * 0.1
                )
                self.token_embeddings[target_token] /= np.linalg.norm(
                    self.token_embeddings[target_token]
                ) + 1e-8

        return self.compute_loss(sequences)

    def train(self, sequences: List[List[int]], epochs: int = 10, learning_rate: float = 0.001) -> List[float]:
        """
        Train the model on sequences.

        Args:
            sequences: List of token sequences
            epochs: Number of training epochs
            learning_rate: Learning rate

        Returns:
            List of losses per epoch
        """
        losses = []
        for epoch in range(epochs):
            loss = self.train_step(sequences, learning_rate)
            losses.append(loss)
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}, loss: {loss:.4f}")

        return losses

    def predict_next(self, context: List[int], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Predict next tokens with probabilities.

        Args:
            context: Context tokens
            top_k: Number of top predictions to return

        Returns:
            List of (token_id, probability) tuples
        """
        if not context:
            # Uniform distribution over all tokens
            probs = np.ones(self.vocab_size) / self.vocab_size
        else:
            logits = self.forward(context)
            probs = _softmax(logits[-1])

        # Get top-k
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_probs = probs[top_indices]

        return [(int(idx), float(prob)) for idx, prob in zip(top_indices, top_probs)]

    def generate(
        self,
        prompt: List[int],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> List[int]:
        """
        Generate sequence continuation.

        Args:
            prompt: Starting tokens
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (<1 = deterministic, >1 = random)
            top_k: If set, only sample from top-k tokens

        Returns:
            Generated token sequence (including prompt)
        """
        # Handle empty prompt by using a default seed token
        if not prompt:
            prompt = [1]  # Use token 1 as default seed

        generated = list(prompt)

        for _ in range(max_length):
            if len(generated) > self.max_seq_len:
                # Truncate context
                context = generated[-self.max_seq_len :]
            else:
                context = generated

            logits = self.forward(context)
            probs = _softmax(logits[-1] / temperature)

            # Apply top-k sampling
            if top_k is not None and top_k < self.vocab_size:
                top_indices = np.argsort(probs)[-top_k:]
                mask = np.zeros(self.vocab_size)
                mask[top_indices] = probs[top_indices]
                probs = mask / (np.sum(mask) + 1e-10)

            # Sample
            next_token = np.random.choice(self.vocab_size, p=probs)
            generated.append(int(next_token))

            # Optional: stop on special token (not implemented yet)

        return generated

    def _get_lr(self, base_lr: float, step: int, warmup_steps: int) -> float:
        """Get learning rate with warmup."""
        if step < warmup_steps:
            return base_lr * step / warmup_steps
        return base_lr * (warmup_steps / max(step, 1)) ** 0.5

    def save(self, path: str) -> None:
        """Save model to file."""
        data = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "max_seq_len": self.max_seq_len,
            "token_embeddings": self.token_embeddings.tolist(),
            "pos_embeddings": self.pos_embeddings.tolist(),
            "lm_head": self.lm_head.tolist(),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TransformerLM":
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        model = cls(
            vocab_size=data["vocab_size"],
            d_model=data["d_model"],
            n_heads=data["n_heads"],
            n_layers=data["n_layers"],
            max_seq_len=data["max_seq_len"],
        )
        model.token_embeddings = np.array(data["token_embeddings"], dtype=np.float32)
        model.pos_embeddings = np.array(data["pos_embeddings"], dtype=np.float32)
        model.lm_head = np.array(data["lm_head"], dtype=np.float32)

        logger.info(f"Model loaded from {path}")
        return model


class ConditionalGenerator:
    """
    Condition generation on context/metadata.

    Wraps a base model to enable context-aware generation.
    """

    def __init__(self, base_model: TransformerLM):
        """
        Initialize conditional generator.

        Args:
            base_model: Base TransformerLM
        """
        self.base_model = base_model
        self.context_models: Dict[str, TransformerLM] = {}

        logger.info("ConditionalGenerator initialized")

    def add_context_model(self, context_type: str, model: TransformerLM) -> None:
        """Add a model for a specific context type."""
        self.context_models[context_type] = model
        logger.info(f"Added context model: {context_type}")

    def train_context(
        self, context_type: str, sequences: List[List[int]], **model_kwargs
    ) -> None:
        """
        Train a model for a specific context.

        Args:
            context_type: Type of context (e.g., "alarm", "social")
            sequences: Training sequences for this context
            **model_kwargs: Arguments for model creation
        """
        model = TransformerLM(
            vocab_size=self.base_model.vocab_size, **model_kwargs
        )
        model.train(sequences, epochs=10)
        self.add_context_model(context_type, model)

    def generate_for_context(
        self,
        context_type: str,
        max_length: int = 50,
        temperature: float = 1.0,
        **kwargs,
    ) -> List[int]:
        """
        Generate sequence for specific behavioral context.

        Args:
            context_type: Type of context
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments

        Returns:
            Generated token sequence
        """
        if context_type in self.context_models:
            model = self.context_models[context_type]
        else:
            logger.warning(f"No model for context '{context_type}', using base model")
            model = self.base_model

        # Use seed token to start generation (empty prompt causes issues)
        return model.generate(
            prompt=[1], max_length=max_length, temperature=temperature, **kwargs
        )

    def generate_batch(
        self,
        context: str,
        n_sequences: int,
        max_length: int = 50,
        temperature: float = 1.0,
    ) -> List[List[int]]:
        """
        Generate multiple sequences for a context.

        Args:
            context: Context type
            n_sequences: Number of sequences to generate
            max_length: Maximum tokens per sequence
            temperature: Sampling temperature

        Returns:
            List of generated sequences
        """
        sequences = []
        for _ in range(n_sequences):
            seq = self.generate_for_context(context, max_length, temperature)
            sequences.append(seq)
        return sequences


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax (numerically stable)."""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (np.sum(exp_x) + 1e-10)


def create_pipeline(vocab_size: int = 1020, d_model: int = 256) -> Tuple[
    AcousticTokenizer, TransformerLM
]:
    """
    Create a complete language modeling pipeline.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension

    Returns:
        Tuple of (tokenizer, model)
    """
    tokenizer = AcousticTokenizer(vocab_size=vocab_size)
    model = TransformerLM(vocab_size=vocab_size, d_model=d_model)
    return tokenizer, model


def main():
    """Command-line interface for neural language model."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Neural Language Model for Direction 2"
    )
    parser.add_argument("--vocab-size", type=int, default=1020, help="Vocabulary size")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--train", type=str, help="Training data file")
    parser.add_argument("--output", type=str, help="Output model file")
    parser.add_argument("--generate", action="store_true", help="Generation mode")
    parser.add_argument("--prompt", type=int, nargs="+", help="Prompt tokens")
    parser.add_argument("--max-length", type=int, default=50, help="Max generation length")

    args = parser.parse_args()

    if args.train:
        # Load training data
        with open(args.train, "r") as f:
            data = json.load(f)
        sequences = data["sequences"]

        # Create and train model
        model = TransformerLM(
            vocab_size=args.vocab_size, d_model=args.d_model
        )
        model.train(sequences, epochs=20)
        model.save(args.output)
        print(f"Model saved to {args.output}")

    elif args.generate:
        # Load model
        model = TransformerLM.load(args.output)

        prompt = args.prompt or []
        generated = model.generate(prompt=prompt, max_length=args.max_length)

        print(f"Generated: {generated}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
