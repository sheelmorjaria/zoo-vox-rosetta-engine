#!/usr/bin/env python3
"""
Transformer Attention Module - Neural Architecture Modernization
================================================================

Multi-head attention and Transformer architectures for capturing
long-range dependencies in animal vocalization sequences.

This module implements:
- Multi-head self-attention for sequence modeling
- Transformer encoder layers
- Vocalization-specific transformer models
- Efficient attention mechanisms for edge deployment
- Attention visualization for interpretability

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MultiHeadAttention:
    """
    Multi-head self-attention mechanism.

    Splits the input into multiple heads, applies scaled dot-product
    attention to each, and concatenates the results.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of parallel attention heads
            dropout: Dropout probability
            causal: Whether to apply causal masking (autoregressive)
        """
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal

        # Initialize projection matrices
        scale = 1.0 / math.sqrt(self.head_dim)
        self.w_q = np.random.randn(embed_dim, embed_dim) * scale
        self.w_k = np.random.randn(embed_dim, embed_dim) * scale
        self.w_v = np.random.randn(embed_dim, embed_dim) * scale
        self.w_o = np.random.randn(embed_dim, embed_dim) * scale

        # Training mode (default to eval mode)
        self.training = False

        logger.debug(
            f"MultiHeadAttention: embed_dim={embed_dim}, num_heads={num_heads}, "
            f"head_dim={self.head_dim}"
        )

    def forward(
        self, x: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask of shape (batch, seq_len, seq_len)

        Returns:
            Tuple of (output, attention_weights)
            - output: Shape (batch, seq_len, embed_dim)
            - attention_weights: Shape (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = x @ self.w_q  # (batch, seq_len, embed_dim)
        k = x @ self.w_k
        v = x @ self.w_v

        # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        # Apply causal mask if needed
        if self.causal:
            causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            scores = scores + (causal_mask * -1e9)

        # Apply provided mask
        if mask is not None:
            scores = scores + mask * -1e9

        # Softmax to get attention weights
        attention_weights = self._softmax(scores, axis=-1)

        # Apply dropout
        if self.dropout > 0 and self.training:
            dropout_mask = np.random.binomial(1, 1 - self.dropout, attention_weights.shape)
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout)

        # Apply attention to values
        output = attention_weights @ v  # (batch, num_heads, seq_len, head_dim)

        # Transpose back: (batch, seq_len, num_heads, head_dim)
        output = output.transpose(0, 2, 1, 3)

        # Concatenate heads: (batch, seq_len, embed_dim)
        output = output.reshape(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = output @ self.w_o

        return output, attention_weights

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def train(self) -> None:
        """Set to training mode."""
        self.training = True

    def eval(self) -> None:
        """Set to evaluation mode."""
        self.training = False


class TransformerEncoder:
    """
    Transformer encoder layer with self-attention and feed-forward network.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        """
        Initialize transformer encoder layer.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward network hidden dimension
            dropout: Dropout probability
        """
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)

        # Feed-forward network parameters
        scale = 1.0 / math.sqrt(embed_dim)
        self.w1 = np.random.randn(embed_dim, ff_dim) * scale
        self.b1 = np.zeros(ff_dim)
        self.w2 = np.random.randn(ff_dim, embed_dim) * scale
        self.b2 = np.zeros(embed_dim)

        # Layer norm parameters
        self.gamma1 = np.ones(embed_dim)
        self.beta1 = np.zeros(embed_dim)
        self.gamma2 = np.ones(embed_dim)
        self.beta2 = np.zeros(embed_dim)

        self.dropout = dropout

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of transformer encoder.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.attention.forward(x)
        x = self._layer_norm(x + attn_output, self.gamma1, self.beta1)

        # Feed-forward network with residual connection and layer norm
        ff_output = self._feed_forward(x)
        x = self._layer_norm(x + ff_output, self.gamma2, self.beta2)

        return x

    def _feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Apply feed-forward network."""
        hidden = x @ self.w1 + self.b1
        hidden = np.maximum(0, hidden)  # ReLU
        output = hidden @ self.w2 + self.b2
        return output

    def _layer_norm(
        self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + eps)
        return gamma * normalized + beta


class VocalizationTransformer:
    """
    Transformer model for vocalization sequence processing.

    Handles:
    - Sequence classification (context detection)
    - Long-range dependency modeling
    - Attention visualization for interpretability
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 256,
        num_classes: Optional[int] = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize vocalization transformer.

        Args:
            vocab_size: Size of the vocabulary (cluster IDs)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward network hidden dimension
            num_classes: Number of output classes (None for sequence-to-sequence)
            max_seq_len: Maximum sequence length for positional encoding
            dropout: Dropout probability
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len

        # Token embedding
        scale = 1.0 / math.sqrt(embed_dim)
        self.token_embedding = np.random.randn(vocab_size, embed_dim) * scale

        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_seq_len, embed_dim)

        # Transformer encoder layers
        self.layers = [
            TransformerEncoder(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ]

        # Output projection for classification
        if num_classes is not None:
            self.output_proj = np.random.randn(embed_dim, num_classes) * scale
            self.output_bias = np.zeros(num_classes)

        # Store attention weights for visualization
        self._attention_cache: List[np.ndarray] = []

        logger.info(
            f"VocalizationTransformer: vocab={vocab_size}, embed_dim={embed_dim}, "
            f"layers={num_layers}, heads={num_heads}"
        )

    def _create_positional_encoding(self, max_len: int, dim: int) -> np.ndarray:
        """Create sinusoidal positional encoding."""
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim, 2) * -(math.log(10000.0) / dim))

        pe = np.zeros((max_len, dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def get_positional_encoding(self, seq_len: int) -> np.ndarray:
        """Get positional encoding for a sequence length."""
        return self.positional_encoding[:seq_len]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of vocalization transformer.

        Args:
            x: Input token IDs of shape (batch, seq_len)

        Returns:
            Output logits of shape (batch, num_classes) if num_classes is set,
            otherwise (batch, seq_len, embed_dim)
        """
        batch_size, seq_len = x.shape
        self._attention_cache.clear()

        # Token embedding
        tokens = self.token_embedding[x]  # (batch, seq_len, embed_dim)

        # Add positional encoding
        positions = self.positional_encoding[:seq_len]
        x = tokens + positions[np.newaxis, :, :]

        # Pass through transformer layers
        for layer in self.layers:
            x = layer.forward(x)

        # Pool across sequence (mean pooling)
        pooled = np.mean(x, axis=1)  # (batch, embed_dim)

        # Output projection
        if self.num_classes is not None:
            logits = pooled @ self.output_proj + self.output_bias
            return logits

        return pooled

    def get_attention_maps(self, x: np.ndarray) -> List[np.ndarray]:
        """
        Get attention weights for visualization.

        Args:
            x: Input token IDs of shape (batch, seq_len)

        Returns:
            List of attention maps, one per layer
        """
        # Forward pass with attention caching
        batch_size, seq_len = x.shape
        attention_maps = []

        # Token embedding
        tokens = self.token_embedding[x]
        positions = self.positional_encoding[:seq_len]
        h = tokens + positions[np.newaxis, :, :]

        # Pass through layers and collect attention
        for layer in self.layers:
            h, attn_weights = layer.attention.forward(h)
            attention_maps.append(attn_weights)
            h = layer._layer_norm(h + layer._feed_forward(h), layer.gamma2, layer.beta2)

        return attention_maps

    def predict(self, x: np.ndarray) -> Tuple[int, float]:
        """
        Make a prediction with confidence score.

        Args:
            x: Input token IDs of shape (batch, seq_len)

        Returns:
            Tuple of (predicted_class, confidence)
        """
        logits = self.forward(x)
        probs = self._softmax(logits, axis=-1)
        pred_class = int(np.argmax(probs[0]))
        confidence = float(probs[0, pred_class])
        return pred_class, confidence

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class EfficientAttention:
    """
    Memory-efficient attention mechanisms for edge deployment.

    Supports:
    - Flash attention (when available)
    - Chunked attention for long sequences
    - Sparse attention patterns
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        chunk_size: Optional[int] = None,
        use_flash_attention: bool = False,
    ):
        """
        Initialize efficient attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            chunk_size: Chunk size for chunked attention (None for full attention)
            use_flash_attention: Whether to use flash attention (requires CUDA)
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.chunk_size = chunk_size
        self.use_flash_attention = use_flash_attention

        # Initialize projections
        scale = 1.0 / math.sqrt(embed_dim)
        self.w_qkv = np.random.randn(embed_dim, 3 * embed_dim) * scale
        self.w_o = np.random.randn(embed_dim, embed_dim) * scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with efficient attention.

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        if self.chunk_size is not None and seq_len > self.chunk_size:
            return self._chunked_attention(x)

        # Standard attention
        qkv = x @ self.w_qkv
        q, k, v = np.split(qkv, 3, axis=-1)

        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        attn_weights = self._softmax(scores, axis=-1)
        output = attn_weights @ v

        # Combine heads
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        output = output @ self.w_o

        return output

    def _chunked_attention(self, x: np.ndarray) -> np.ndarray:
        """Process attention in chunks for memory efficiency."""
        batch_size, seq_len, embed_dim = x.shape
        outputs = []

        for i in range(0, seq_len, self.chunk_size):
            chunk_end = min(i + self.chunk_size, seq_len)
            chunk = x[:, i:chunk_end, :]
            output_chunk = self._standard_attention(chunk)
            outputs.append(output_chunk)

        return np.concatenate(outputs, axis=1)

    def _standard_attention(self, x: np.ndarray) -> np.ndarray:
        """Standard attention computation."""
        batch_size, seq_len, _ = x.shape

        qkv = x @ self.w_qkv
        q, k, v = np.split(qkv, 3, axis=-1)

        head_dim = self.embed_dim // self.num_heads
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        attn_weights = self._softmax(scores, axis=-1)
        output = attn_weights @ v

        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        output = output @ self.w_o

        return output

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Transformer Attention Module")
    print("=" * 50)

    # Test multi-head attention
    mha = MultiHeadAttention(embed_dim=112, num_heads=4)
    x = np.random.randn(2, 16, 112).astype(np.float32)
    output, weights = mha.forward(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # Test vocalization transformer
    model = VocalizationTransformer(
        vocab_size=100,
        embed_dim=64,
        num_heads=2,
        num_layers=2,
        num_classes=4,
    )

    seq = np.array([[1, 5, 10, 15, 20], [2, 6, 11, 16, 21]], dtype=np.int32)
    logits = model.forward(seq)

    print(f"\nSequence shape: {seq.shape}")
    print(f"Logits shape: {logits.shape}")

    # Get attention maps
    attention_maps = model.get_attention_maps(seq)
    print(f"Number of attention layers: {len(attention_maps)}")
