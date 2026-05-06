#!/usr/bin/env python3
"""
State Space Model (Mamba-style) - Neural Architecture Modernization
====================================================================

Selective State Space Models for efficient long-range modeling
in neural boundary detection and sequence processing.

This module implements:
- Selective SSM (Mamba-style) for linear-complexity sequence modeling
- Mamba blocks with 1D convolution and selective scan
- MambaBoundaryDetector for neural boundary detection
- Efficient alternatives to Transformers for long sequences

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SSMConfig:
    """Configuration for State Space Model"""

    d_model: int = 64  # Model dimension
    d_state: int = 16  # SSM state dimension
    d_conv: int = 4  # Convolution kernel size
    expand: int = 2  # Expansion factor for inner dimension
    dt_rank: Optional[int] = None  # Rank for delta projection
    dt_min: float = 0.001  # Minimum timestep
    dt_max: float = 0.1  # Maximum timestep
    dt_init: str = "random"  # Delta initialization


class SelectiveSSM:
    """
    Selective State Space Model (Mamba-style core).

    Uses data-dependent selection for input-invariant dynamics,
    enabling effective modeling of sequences with complex patterns.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
    ):
        """
        Initialize selective SSM.

        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            dt_rank: Rank for delta (default: d_model)
        """
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank or d_model

        d_inner = d_model * expand

        # Input projections
        scale = 1.0 / math.sqrt(d_model)
        self.in_proj = np.random.randn(d_model, 2 * d_inner) * scale

        # Convolution
        self.conv1d_weight = np.random.randn(d_conv, d_inner) * scale
        self.conv1d_bias = np.zeros(d_inner)

        # x projection (for selective mechanism)
        self.x_proj = np.random.randn(d_inner, self.dt_rank + 2 * d_state) * scale

        # dt projection
        self.dt_proj = np.random.randn(self.dt_rank, d_inner) * scale

        # SSM parameters (A, B, C)
        # A: (d_inner, d_state) - state transition matrix
        self.A_log = np.random.randn(d_inner, d_state) * scale
        self.A = np.exp(self.A_log)  # Ensure A > 0

        # D: skip connection
        self.D = np.zeros(d_model)

        # Output projection
        self.out_proj = np.random.randn(d_inner, d_model) * scale

        logger.debug(
            f"SelectiveSSM: d_model={d_model}, d_state={d_state}, "
            f"d_inner={d_inner}"
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of selective SSM.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        xz = x @ self.in_proj  # (batch, seq_len, 2 * d_inner)
        x, z = np.split(xz, 2, axis=-1)  # Each (batch, seq_len, d_inner)

        # 1D Convolution
        x = self._conv1d(x)  # (batch, seq_len, d_inner)

        # Apply activation
        x = self._silu(x)

        # Selective SSM
        y = self._selective_scan(x)  # (batch, seq_len, d_inner)

        # Gating
        y = y * self._silu(z)

        # Output projection
        output = y @ self.out_proj  # (batch, seq_len, d_model)

        return output

    def _conv1d(self, x: np.ndarray) -> np.ndarray:
        """Apply 1D convolution along sequence dimension."""
        batch_size, seq_len, d_inner = x.shape
        kernel_size = self.conv1d_weight.shape[0]

        # Simple convolution implementation
        output = np.zeros((batch_size, seq_len, d_inner), dtype=x.dtype)

        # Apply bias
        x = x + self.conv1d_bias

        for b in range(batch_size):
            for i in range(seq_len):
                # Gather kernel window
                for k in range(kernel_size):
                    src_idx = i - k
                    if 0 <= src_idx < seq_len:
                        output[b, i] += x[b, src_idx] * self.conv1d_weight[k]

        return output

    def _selective_scan(self, u: np.ndarray) -> np.ndarray:
        """
        Selective scan operation (simplified SSM).

        This is a simplified implementation that captures the essential
        behavior of selective state space models while avoiding complex
        tensor operations.

        Args:
            u: Input tensor of shape (batch, seq_len, d_inner)

        Returns:
            Output tensor of shape (batch, seq_len, d_inner)
        """
        batch_size, seq_len, d_inner = u.shape

        # Project to get delta, B, C
        x_proj = u @ self.x_proj  # (batch, seq_len, dt_rank + 2 * d_state)

        delta = x_proj[:, :, : self.dt_rank]
        delta = delta @ self.dt_proj  # (batch, seq_len, d_inner)
        delta = self._softplus(delta)  # Ensure positive

        # Simplified selective scan - accumulate with decay
        y = np.zeros((batch_size, seq_len, d_inner), dtype=u.dtype)
        h = np.zeros((batch_size, d_inner), dtype=u.dtype)

        for t in range(seq_len):
            # Decay previous state
            decay_factor = 1.0 - np.clip(delta[:, t, 0], 0, 0.5)  # (batch,)
            h = h * np.expand_dims(decay_factor, -1)  # (batch, d_inner)

            # Add contribution from current input
            h = h + u[:, t, :]  # (batch, d_inner)

            # Output is current state
            y[:, t, :] = h

        return y

    def _silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU activation function."""
        return x / (1.0 + np.exp(-x))

    def _softplus(self, x: np.ndarray) -> np.ndarray:
        """Softplus activation."""
        return np.log(1.0 + np.exp(x))


class MambaBlock:
    """
    Mamba block combining 1D conv and selective SSM.

    This is the core building block of Mamba architecture.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        """
        Initialize Mamba block.

        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
        """
        self.d_model = d_model
        self.norm = LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of Mamba block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-norm
        normalized = self.norm.forward(x)

        # SSM processing
        ssm_out = self.ssm.forward(normalized)

        # Residual connection
        return x + ssm_out


class LayerNorm:
    """Layer normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initialize layer norm.

        Args:
            d_model: Feature dimension
            eps: Small constant for numerical stability
        """
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Normalized tensor
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta


class MambaBoundaryDetector:
    """
    Mamba-based boundary detector for neural boundary detection.

    Uses Mamba architecture to efficiently process long audio
    sequences and detect phrase boundaries with linear complexity.
    """

    def __init__(
        self,
        input_dim: int = 112,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 2,
        n_classes: int = 2,
    ):
        """
        Initialize Mamba boundary detector.

        Args:
            input_dim: Input feature dimension (112D RosettaFeatures)
            d_model: Model dimension
            d_state: SSM state dimension
            n_layers: Number of Mamba layers
            n_classes: Number of output classes (boundary/no-boundary)
        """
        self.input_dim = input_dim
        self.d_model = d_model
        self.d_state = d_state
        self.n_classes = n_classes

        # Input projection
        scale = 1.0 / math.sqrt(input_dim)
        self.input_proj = np.random.randn(input_dim, d_model) * scale

        # Mamba layers
        self.layers = [
            MambaBlock(d_model, d_state, d_conv=4, expand=2) for _ in range(n_layers)
        ]

        # Output projection for classification
        self.output_proj = np.random.randn(d_model, n_classes) * scale
        self.output_bias = np.zeros(n_classes)

        logger.info(
            f"MambaBoundaryDetector: input_dim={input_dim}, d_model={d_model}, "
            f"layers={n_layers}"
        )

    def detect_boundaries(self, x: np.ndarray) -> np.ndarray:
        """
        Detect boundaries in input sequence.

        Args:
            x: Input features of shape (batch, seq_len, input_dim)

        Returns:
            Boundary probabilities of shape (batch, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Project to model dimension
        h = x @ self.input_proj  # (batch, seq_len, d_model)

        # Pass through Mamba layers
        for layer in self.layers:
            h = layer.forward(h)

        # Project to logits
        logits = h @ self.output_proj + self.output_bias  # (batch, seq_len, n_classes)

        # Get boundary probabilities (class 1)
        probs = self._softmax(logits, axis=-1)
        boundary_probs = probs[:, :, 1]  # (batch, seq_len)

        return boundary_probs

    def detect_with_confidence(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect boundaries with confidence scores.

        Args:
            x: Input features of shape (batch, seq_len, input_dim)

        Returns:
            Tuple of (boundary_probs, max_confidence)
        """
        boundary_probs = self.detect_boundaries(x)

        # Confidence is the max probability at each position
        confidence = np.maximum(boundary_probs, 1.0 - boundary_probs)

        return boundary_probs, confidence

    def find_boundary_indices(
        self, x: np.ndarray, threshold: float = 0.5
    ) -> list:
        """
        Find boundary indices above threshold.

        Args:
            x: Input features of shape (batch, seq_len, input_dim)
            threshold: Confidence threshold for boundary detection

        Returns:
            List of boundary indices
        """
        boundary_probs = self.detect_boundaries(x)

        # For single batch, find indices where prob > threshold
        indices = np.where(boundary_probs[0] > threshold)[0].tolist()

        return indices

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class MambaEncoder:
    """
    Mamba encoder for sequence encoding.

    Efficiently encodes long sequences into compact representations.
    """

    def __init__(
        self,
        input_dim: int = 112,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 4,
    ):
        """
        Initialize Mamba encoder.

        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            d_state: SSM state dimension
            n_layers: Number of Mamba layers
        """
        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        scale = 1.0 / math.sqrt(input_dim)
        self.input_proj = np.random.randn(input_dim, d_model) * scale

        # Mamba layers
        self.layers = [
            MambaBlock(d_model, d_state, d_conv=4, expand=2) for _ in range(n_layers)
        ]

        # Output norm
        self.norm = LayerNorm(d_model)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input sequence.

        Args:
            x: Input features of shape (batch, seq_len, input_dim)

        Returns:
            Encoded representation of shape (batch, seq_len, d_model)
        """
        # Project to model dimension
        h = x @ self.input_proj

        # Pass through Mamba layers
        for layer in self.layers:
            h = layer.forward(h)

        # Apply final norm
        h = self.norm.forward(h)

        return h

    def pool(self, x: np.ndarray, method: str = "mean") -> np.ndarray:
        """
        Pool sequence to single representation.

        Args:
            x: Input features of shape (batch, seq_len, input_dim)
            method: Pooling method ("mean", "max", "last")

        Returns:
            Pooled representation of shape (batch, d_model)
        """
        h = self.encode(x)

        if method == "mean":
            return np.mean(h, axis=1)
        elif method == "max":
            return np.max(h, axis=1)
        elif method == "last":
            return h[:, -1]
        else:
            raise ValueError(f"Unknown pooling method: {method}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("State Space Model (Mamba-style)")
    print("=" * 50)

    # Test selective SSM
    ssm = SelectiveSSM(d_model=64, d_state=16, d_conv=4, expand=2)
    x = np.random.randn(2, 100, 64).astype(np.float32)
    output = ssm.forward(x)

    print(f"SSM input shape: {x.shape}")
    print(f"SSM output shape: {output.shape}")

    # Test Mamba boundary detector
    detector = MambaBoundaryDetector(input_dim=112, d_model=64, d_state=16, n_layers=2)
    features = np.random.randn(1, 200, 112).astype(np.float32)
    boundaries = detector.detect_boundaries(features)

    print(f"\nBoundary detector input shape: {features.shape}")
    print(f"Boundary probabilities shape: {boundaries.shape}")

    # Find boundary indices
    indices = detector.find_boundary_indices(features, threshold=0.5)
    print(f"Detected boundaries at: {indices}")
