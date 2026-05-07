#!/usr/bin/env python3
"""
DDSP Decoder - Module 2 (v1.6.0)

PyTorch MLP that maps 112D RosettaFeatures to 65 DDSP control parameters:
- 60 harmonic amplitudes (for additive synthesis)
- 5 noise band magnitudes (for filtered noise synthesis)

This is the core neural component for DDSP-based animal vocalization synthesis.

Architecture:
    112D Input → Hidden(256) → ReLU → Dropout → Hidden(256) → ReLU → Dropout → 65D Output
                                                         ↓
                                            ┌─────────────────────────┐
                                            │ 60 Harmonic Amps (Softmax) │
                                            │ 5 Noise Mags (ReLU)       │
                                            └─────────────────────────┘

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DDSPDecoderConfig:
    """Configuration for DDSPDecoder."""

    # Input dimension (112D RosettaFeatures)
    input_dim: int = 112

    # Hidden layer dimension
    hidden_dim: int = 256

    # Number of harmonic amplitudes (for additive synthesis)
    num_harmonics: int = 60

    # Number of noise bands (for filtered noise synthesis)
    num_noise_bands: int = 5

    # Dropout rate
    dropout: float = 0.1

    # Layer initialization
    init_gain: float = 1.0

    @property
    def output_dim(self) -> int:
        """Total output dimension (harmonics + noise)."""
        return self.num_harmonics + self.num_noise_bands


# =============================================================================
# DDSP Decoder Model
# =============================================================================


class DDSPDecoder(nn.Module):
    """
    MLP: 112D RosettaFeatures → 65 DDSP parameters.

    The decoder takes the 112-dimensional RosettaFeatures vector and produces
    the control parameters needed for DDSP synthesis:
    - 60 harmonic amplitudes (softmax normalized, sum to 1.0)
    - 5 noise band magnitudes (relu activated, non-negative)

    This enables gradient-based synthesis where the entire pipeline is
    differentiable from features to audio output.

    Example:
        >>> decoder = DDSPDecoder()
        >>> features_112d = torch.randn(1, 112)  # Batch of 1
        >>> harmonic_amps, noise_mags = decoder(features_112d)
        >>> print(harmonic_amps.shape)  # torch.Size([1, 60])
        >>> print(noise_mags.shape)  # torch.Size([1, 5])
    """

    def __init__(
        self,
        config: Optional[DDSPDecoderConfig] = None,
        hidden_dim: int = 256,
        num_harmonics: int = 60,
        num_noise_bands: int = 5,
        dropout: float = 0.1,
    ):
        """
        Initialize DDSPDecoder.

        Args:
            config: Full configuration object (overrides other args if provided)
            hidden_dim: Hidden layer dimension
            num_harmonics: Number of harmonic amplitude outputs
            num_noise_bands: Number of noise band magnitude outputs
            dropout: Dropout rate
        """
        super().__init__()

        # Use config if provided, otherwise use individual args
        if config is not None:
            self.input_dim = config.input_dim
            hidden_dim = config.hidden_dim
            num_harmonics = config.num_harmonics
            num_noise_bands = config.num_noise_bands
            dropout = config.dropout
        else:
            self.input_dim = 112

        self.hidden_dim = hidden_dim
        self.num_harmonics = num_harmonics
        self.num_noise_bands = num_noise_bands
        self.output_dim = num_harmonics + num_noise_bands

        # Build MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

        # Initialize weights
        self._initialize_weights()

        logger.info(
            f"DDSPDecoder initialized: {self.input_dim}D → {self.output_dim}D "
            f"({num_harmonics} harmonics + {num_noise_bands} noise bands)"
        )

    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        features_112d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: 112D features → harmonic amps + noise mags.

        Args:
            features_112d: Input tensor of shape (B, 112) or (112,)
                - B: Batch size
                - 112: RosettaFeatures dimension

        Returns:
            harmonic_amps: Harmonic amplitudes, shape (B, 60)
                - Softmax normalized (sums to 1.0 per sample)
                - Controls relative strength of each harmonic
            noise_mags: Noise band magnitudes, shape (B, 5)
                - ReLU activated (non-negative)
                - Controls strength of each noise frequency band
        """
        # Ensure input is 2D (add batch dim if needed)
        if features_112d.dim() == 1:
            features_112d = features_112d.unsqueeze(0)

        # Pass through MLP
        x = self.mlp(features_112d)  # (B, 65)

        # Split into harmonics and noise
        harmonic_amps = x[:, : self.num_harmonics]  # (B, 60)
        noise_mags = x[:, self.num_harmonics :]  # (B, 5)

        # Apply output activations
        harmonic_amps = F.softmax(harmonic_amps, dim=-1)  # Normalize to sum=1
        noise_mags = F.relu(noise_mags)  # Ensure non-negative

        return harmonic_amps, noise_mags

    def inference(
        self,
        features_112d: torch.Tensor,
    ) -> dict:
        """
        Inference mode with additional metadata.

        Args:
            features_112d: Input tensor of shape (B, 112) or (112,)

        Returns:
            Dictionary containing:
                - harmonic_amps: (B, 60) harmonic amplitudes
                - noise_mags: (B, 5) noise magnitudes
                - f0_hz: Placeholder for fundamental frequency (should come from input)
                - confidence: Output confidence estimate
        """
        with torch.no_grad():
            harmonic_amps, noise_mags = self.forward(features_112d)

            # Simple confidence estimate based on output distribution
            # Higher confidence when energy is concentrated in few harmonics
            energy_entropy = -(harmonic_amps * torch.log(harmonic_amps + 1e-8)).sum(dim=-1)
            max_entropy = math.log(self.num_harmonics)
            confidence = 1.0 - (energy_entropy / max_entropy)

            return {
                "harmonic_amps": harmonic_amps,
                "noise_mags": noise_mags,
                "confidence": confidence,
            }


# =============================================================================
# Specialized Decoder Variants
# =============================================================================


class DDSPDecoderLight(DDSPDecoder):
    """
    Lightweight variant with smaller hidden dimension.

    Useful for edge deployment (Jetson) where memory and compute are limited.
    """

    def __init__(
        self,
        num_harmonics: int = 60,
        num_noise_bands: int = 5,
        dropout: float = 0.0,  # No dropout for inference
    ):
        super().__init__(
            hidden_dim=128,  # Smaller hidden layer
            num_harmonics=num_harmonics,
            num_noise_bands=num_noise_bands,
            dropout=dropout,
        )


class DDSPDecoderLarge(DDSPDecoder):
    """
    Large variant with deeper architecture and more parameters.

    Useful for maximum quality when compute is available.
    """

    def __init__(
        self,
        num_harmonics: int = 60,
        num_noise_bands: int = 5,
        dropout: float = 0.15,
    ):
        # Use custom config for deeper network
        config = DDSPDecoderConfig(
            hidden_dim=512,
            num_harmonics=num_harmonics,
            num_noise_bands=num_noise_bands,
            dropout=dropout,
        )
        super().__init__(config=config)


# =============================================================================
# Utility Functions
# =============================================================================


def count_parameters(model: DDSPDecoder) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: DDSPDecoder) -> float:
    """Get the model size in megabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def create_decoder(
    variant: str = "base",
    num_harmonics: int = 60,
    num_noise_bands: int = 5,
    **kwargs,
) -> DDSPDecoder:
    """
    Factory function to create DDSPDecoder variants.

    Args:
        variant: One of "base", "light", "large"
        num_harmonics: Number of harmonic amplitude outputs
        num_noise_bands: Number of noise band outputs
        **kwargs: Additional arguments passed to decoder constructor

    Returns:
        DDSPDecoder instance
    """
    if variant == "base":
        return DDSPDecoder(
            num_harmonics=num_harmonics,
            num_noise_bands=num_noise_bands,
            **kwargs,
        )
    elif variant == "light":
        return DDSPDecoderLight(
            num_harmonics=num_harmonics,
            num_noise_bands=num_noise_bands,
            **kwargs,
        )
    elif variant == "large":
        return DDSPDecoderLarge(
            num_harmonics=num_harmonics,
            num_noise_bands=num_noise_bands,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose from: base, light, large")


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create decoder
    decoder = DDSPDecoder()

    # Print model info
    print("\n=== DDSPDecoder Model Info ===")
    print(f"Parameters: {count_parameters(decoder):,}")
    print(f"Model size: {get_model_size_mb(decoder):.2f} MB")

    # Test forward pass
    batch_size = 4
    features_112d = torch.randn(batch_size, 112)

    harmonic_amps, noise_mags = decoder(features_112d)

    print("\n=== Forward Pass Test ===")
    print(f"Input shape: {features_112d.shape}")
    print(f"Harmonic amps shape: {harmonic_amps.shape}")
    print(f"Noise mags shape: {noise_mags.shape}")
    print(f"Harmonic amps sum (should be ~1.0): {harmonic_amps[0].sum().item():.6f}")
    print(f"Noise mags min (should be >= 0): {noise_mags[0].min().item():.6f}")

    # Test inference mode
    print("\n=== Inference Mode Test ===")
    result = decoder.inference(features_112d[0])
    print(f"Confidence: {result['confidence'][0].item():.3f}")

    # Test light variant
    print("\n=== Light Variant ===")
    light_decoder = DDSPDecoderLight()
    print(f"Parameters: {count_parameters(light_decoder):,}")
    print(f"Model size: {get_model_size_mb(light_decoder):.2f} MB")
