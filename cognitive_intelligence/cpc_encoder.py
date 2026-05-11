#!/usr/bin/env python3
"""
CPC (Contrastive Predictive Coding) Encoder for Predictive NBD

Architecture:
- 1D Convolutional encoder for audio feature extraction
- Projection head for contrastive learning
- Outputs 128D latent representations

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

import logging
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CPCEncoder(nn.Module):
    """
    CPC Encoder for audio feature extraction.

    Architecture:
    1. 1D Conv layers with downsampling
    2. Projection head to latent space
    3. Output: (batch, seq_len, hidden_dim)

    Args:
        sample_rate: Input audio sample rate (default 48kHz)
        frame_size: Frame size in samples (default 480 = 10ms @ 48kHz)
        hidden_dim: Latent dimension (default 128)
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        frame_size: int = 480,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hidden_dim = hidden_dim

        # 1D Convolutional encoder
        # Input: (batch, 1, frame_size)
        self.encoder = nn.Sequential(
            # Layer 1: Large kernel for temporal context
            nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=5),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            # Layer 2: Medium kernel
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            # Layer 3: Smaller kernel
            nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            # Layer 4: Final conv
            nn.Conv1d(64, hidden_dim, kernel_size=3, stride=1, padding=1),
        )

        # Projection head (for contrastive learning)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode audio frame to latent representation.

        Args:
            x: (batch, frame_size) or (batch, 1, frame_size)

        Returns:
            z: (batch, 1, hidden_dim) - latent representation
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        # Encode
        z = self.encoder(x)  # (batch, hidden_dim, seq_len)

        # Global average pooling over sequence
        z = z.mean(dim=2)  # (batch, hidden_dim)

        # Project
        z = self.projection(z)  # (batch, hidden_dim)

        # Add sequence dimension for compatibility
        z = z.unsqueeze(1)  # (batch, 1, hidden_dim)

        return z

    def encode_single(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode a single audio frame (no batch dimension).

        Args:
            audio: (frame_size,)

        Returns:
            z: (hidden_dim,)
        """
        with torch.no_grad():
            x = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, frame_size)
            z = self.forward(x)  # (1, 1, hidden_dim)
            return z.squeeze(0).squeeze(0)  # (hidden_dim,)


class CPCModel(nn.Module):
    """
    Full CPC model with encoder and autoregressive prediction.

    For training, this combines the encoder with the AR model.
    """

    def __init__(
        self,
        encoder: CPCEncoder,
        ar_model: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.ar_model = ar_model

    def forward(self, x: torch.Tensor, steps_ahead: int = 5) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with encoding and prediction.

        Args:
            x: (batch, frame_size) audio
            steps_ahead: Number of steps to predict

        Returns:
            z: Encoded latent
            predictions: List of predictions for each step ahead
        """
        z = self.encoder(x)
        predictions = self.ar_model(z, steps_ahead)
        return z, predictions


def create_encoder(
    hidden_dim: int = 128,
    sample_rate: int = 48000,
    frame_size_ms: float = 10.0,
) -> CPCEncoder:
    """
    Create a CPC encoder with specified parameters.

    Args:
        hidden_dim: Latent dimension
        sample_rate: Audio sample rate
        frame_size_ms: Frame size in milliseconds

    Returns:
        CPCEncoder instance
    """
    frame_size = int(sample_rate * frame_size_ms / 1000)
    return CPCEncoder(
        sample_rate=sample_rate,
        frame_size=frame_size,
        hidden_dim=hidden_dim,
    )


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_encoder():
    """Test encoder functionality."""
    print("=" * 60)
    print("Testing CPC Encoder")
    print("=" * 60)

    # Create encoder
    encoder = create_encoder(hidden_dim=128)

    # Test parameters
    num_params = count_parameters(encoder)
    print(f"Encoder parameters: {num_params:,}")

    # Test forward pass
    batch_size = 4
    frame_size = 480  # 10ms @ 48kHz
    x = torch.randn(batch_size, frame_size)

    z = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {z.shape}")

    assert z.shape == (batch_size, 1, 128), f"Unexpected shape: {z.shape}"

    # Test single frame encoding
    audio = torch.randn(frame_size)
    z_single = encoder.encode_single(audio)
    print(f"Single frame output shape: {z_single.shape}")

    assert z_single.shape == (128,), f"Unexpected shape: {z_single.shape}"

    print("\n✓ Encoder tests passed")


if __name__ == "__main__":
    test_encoder()
