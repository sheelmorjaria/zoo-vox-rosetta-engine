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
# Module 4: Dual-Stream with FiLM (Feature-wise Linear Modulation)
# =============================================================================


@dataclass
class FiLMConfig:
    """Configuration for FiLM-based affect modulation."""

    # Affect vector dimension (16D from β-VAE)
    affect_dim: int = 16

    # Number of FiLM layers (which hidden layers to modulate)
    num_film_layers: int = 2

    # FiLM hidden dimension (for generating γ, β parameters)
    film_hidden_dim: int = 64

    # Whether to freeze base MLP weights (preserve pre-trained 112D model)
    freeze_base_mlp: bool = True


class FiLMGenerator(nn.Module):
    """
    Generate FiLM parameters (γ, β) from affect vector.

    FiLM (Feature-wise Linear Modulation) enables the affect vector to
    modulate the intermediate activations of the DDSP decoder:

        y = γ * x + β

    where γ (scale) and β (shift) are generated from the affect vector.

    This preserves pre-trained weights while enabling affective control.
    """

    def __init__(
        self,
        affect_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 2,
        film_hidden_dim: int = 64,
    ):
        """
        Initialize FiLM generator.

        Args:
            affect_dim: Dimension of affect vector (16D from β-VAE)
            hidden_dim: Hidden dimension of layers to modulate
            num_layers: Number of FiLM modulation layers
            film_hidden_dim: Hidden dimension for FiML MLP
        """
        super().__init__()

        self.affect_dim = affect_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create FiML MLP for each layer to modulate
        # Each layer generates (γ, β) = 2 * hidden_dim parameters
        self.film_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(affect_dim, film_hidden_dim),
                nn.ReLU(),
                nn.Linear(film_hidden_dim, hidden_dim * 2),  # γ and β
            )
            for _ in range(num_layers)
        ])

        logger.info(f"FiLMGenerator initialized: {num_layers} layers, affect_dim={affect_dim}")

    def forward(self, affect_vector: torch.Tensor) -> list:
        """
        Generate FiLM parameters from affect vector.

        Args:
            affect_vector: Affect vector of shape (B, affect_dim)

        Returns:
            List of (γ, β) tuples, one for each FiLM layer
                Each γ, β has shape (B, hidden_dim)
        """
        films = []
        for layer in self.film_layers:
            params = layer(affect_vector)  # (B, hidden_dim * 2)
            gamma, beta = torch.chunk(params, 2, dim=-1)  # Each (B, hidden_dim)
            films.append((gamma, beta))
        return films


class DualStreamDDSPDecoder(nn.Module):
    """
    Dual-Stream DDSP Decoder with FiLM modulation.

    This decoder combines:
    - Stream 1: Continuous affect vector (16D) → FiLM modulation parameters
    - Stream 2: Discrete syntactic token (via 112D features) → Base DDSP parameters

    The FiLM layers preserve pre-trained 112D DDSP weights while enabling
    affective modulation through learned γ (scale) and β (shift) parameters.

    Architecture:
        112D Features → Base MLP (pre-trained, optionally frozen)
        16D Affect → FiLM Generator → (γ, β) parameters
        Combined: FiML-modulated activations → 65D DDSP output

    Training Strategy:
        Phase 1: Freeze base MLP, train only FiLM generator (preserve weights)
        Phase 2: Fine-tune entire network end-to-end

    Example:
        >>> decoder = DualStreamDDSPDecoder()
        >>> features_112d = torch.randn(1, 112)
        >>> affect_vector = torch.randn(1, 16)
        >>> harmonic_amps, noise_mags = decoder(features_112d, affect_vector)
    """

    def __init__(
        self,
        base_decoder: Optional[DDSPDecoder] = None,
        affect_dim: int = 16,
        num_film_layers: int = 2,
        film_hidden_dim: int = 64,
        freeze_base_mlp: bool = True,
    ):
        """
        Initialize dual-stream decoder.

        Args:
            base_decoder: Pre-trained DDSPDecoder (creates new if None)
            affect_dim: Dimension of affect vector (16D from β-VAE)
            num_film_layers: Number of hidden layers to apply FiLM
            film_hidden_dim: Hidden dimension for FiML MLP
            freeze_base_mlp: Whether to freeze base MLP weights
        """
        super().__init__()

        # Create or use provided base decoder
        if base_decoder is None:
            self.base_decoder = DDSPDecoder()
        else:
            self.base_decoder = base_decoder

        # Store dimensions
        self.affect_dim = affect_dim
        self.hidden_dim = self.base_decoder.hidden_dim
        self.num_harmonics = self.base_decoder.num_harmonics
        self.num_noise_bands = self.base_decoder.num_noise_bands

        # Create FiLM generator for affect modulation
        self.film_gen = FiLMGenerator(
            affect_dim=affect_dim,
            hidden_dim=self.hidden_dim,
            num_layers=num_film_layers,
            film_hidden_dim=film_hidden_dim,
        )

        # Optionally freeze base MLP to preserve pre-trained weights
        if freeze_base_mlp:
            self._freeze_base_mlp()

        # Get base MLP layers for FiLM modulation
        # The base MLP has structure: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear
        # We'll apply FiLM after the first two Linear layers
        self.base_layers = list(self.base_decoder.mlp.children())

        logger.info(
            f"DualStreamDDSPDecoder initialized: affect_dim={affect_dim}, "
            f"film_layers={num_film_layers}, frozen={freeze_base_mlp}"
        )

    def _freeze_base_mlp(self):
        """Freeze base MLP parameters to preserve pre-trained weights."""
        for param in self.base_decoder.parameters():
            param.requires_grad = False
        logger.info("Base MLP weights frozen (pre-trained weights preserved)")

    def unfreeze_base_mlp(self):
        """Unfreeze base MLP for end-to-end fine-tuning."""
        for param in self.base_decoder.parameters():
            param.requires_grad = True
        logger.info("Base MLP weights unfrozen (end-to-end fine-tuning enabled)")

    def forward(
        self,
        features_112d: torch.Tensor,
        affect_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dual-stream inputs.

        Args:
            features_112d: Acoustic features of shape (B, 112)
            affect_vector: Continuous affect vector of shape (B, 16)

        Returns:
            harmonic_amps: Harmonic amplitudes, shape (B, 60)
            noise_mags: Noise band magnitudes, shape (B, 5)
        """
        batch_size = features_112d.shape[0]

        # Ensure input is 2D
        if features_112d.dim() == 1:
            features_112d = features_112d.unsqueeze(0)
        if affect_vector.dim() == 1:
            affect_vector = affect_vector.unsqueeze(0)

        # Generate FiLM parameters from affect vector
        films = self.film_gen(affect_vector)

        # Pass through base MLP with FiLM modulation
        x = features_112d
        film_idx = 0

        for i, layer in enumerate(self.base_layers):
            x = layer(x)

            # Apply FiLM after ReLU activations (not after Dropout or final Linear)
            if isinstance(layer, nn.ReLU) and film_idx < len(films):
                gamma, beta = films[film_idx]
                x = gamma * x + beta  # FiLM modulation
                film_idx += 1

        # Split into harmonics and noise
        harmonic_amps = x[:, : self.num_harmonics]
        noise_mags = x[:, self.num_harmonics :]

        # Apply output activations
        harmonic_amps = F.softmax(harmonic_amps, dim=-1)
        noise_mags = F.relu(noise_mags)

        return harmonic_amps, noise_mags

    def inference(
        self,
        features_112d: torch.Tensor,
        affect_vector: torch.Tensor,
    ) -> dict:
        """
        Inference mode with additional metadata.

        Args:
            features_112d: Input tensor of shape (B, 112) or (112,)
            affect_vector: Affect vector of shape (B, 16) or (16,)

        Returns:
            Dictionary containing harmonic amps, noise mags, and confidence
        """
        with torch.no_grad():
            harmonic_amps, noise_mags = self.forward(features_112d, affect_vector)

            # Confidence estimate
            energy_entropy = -(harmonic_amps * torch.log(harmonic_amps + 1e-8)).sum(dim=-1)
            max_entropy = math.log(self.num_harmonics)
            confidence = 1.0 - (energy_entropy / max_entropy)

            return {
                "harmonic_amps": harmonic_amps,
                "noise_mags": noise_mags,
                "confidence": confidence,
            }

    def apply_affect_arousal(
        self,
        features_112d: torch.Tensor,
        arousal_level: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply arousal-based affect modulation using simplified affect vector.

        This is a convenience method for testing arousal modulation without
        a full 16D affect vector. Creates a minimal affect vector with
        the specified arousal level in dimension 0.

        Args:
            features_112d: Acoustic features of shape (B, 112)
            arousal_level: Arousal level (0-1, where 1 = high arousal)

        Returns:
            harmonic_amps: Harmonic amplitudes with arousal modulation
            noise_mags: Noise magnitudes with arousal modulation
        """
        batch_size = features_112d.shape[0]

        # Create minimal affect vector (arousal in dim 0, zeros elsewhere)
        affect_vector = torch.zeros(batch_size, self.affect_dim)
        affect_vector[:, 0] = arousal_level

        return self.forward(features_112d, affect_vector)


def create_dual_stream_decoder(
    pretrained_path: Optional[str] = None,
    affect_dim: int = 16,
    freeze_base: bool = True,
) -> DualStreamDDSPDecoder:
    """
    Factory function to create a dual-stream decoder.

    Args:
        pretrained_path: Path to pre-trained base decoder weights
        affect_dim: Dimension of affect vector
        freeze_base: Whether to freeze base MLP weights

    Returns:
        Initialized DualStreamDDSPDecoder
    """
    base_decoder = None

    if pretrained_path:
        try:
            base_decoder = DDSPDecoder()
            state_dict = torch.load(pretrained_path, map_location="cpu")
            base_decoder.load_state_dict(state_dict)
            logger.info(f"Loaded pre-trained decoder from {pretrained_path}")
        except Exception as e:
            logger.warning(f"Failed to load pre-trained decoder: {e}")
            base_decoder = None

    return DualStreamDDSPDecoder(
        base_decoder=base_decoder,
        affect_dim=affect_dim,
        freeze_base_mlp=freeze_base,
    )


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
