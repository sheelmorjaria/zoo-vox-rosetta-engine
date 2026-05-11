#!/usr/bin/env python3
"""
Dual-Stream DDSP Decoder with FiLM Layers (Sprint 3)

Implements FiLM (Feature-wise Linear Modulation) to preserve pre-trained
112D DDSP weights while accepting 16D affect modulation.

Key Benefits:
- Pre-trained DDSP weights preserved
- Only FiLM γ/β parameters trained initially
- Affect vector generates per-layer modulation

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class FiLMDecoderConfig:
    """Configuration for FiLM-based DDSP decoder."""

    # Affect dimensions
    affect_dim: int = 16

    # DDSP base network
    input_dim: int = 112
    hidden_dim: int = 256
    output_dim: int = 65  # 60 harmonic amplitudes + 5 noise magnitudes

    # FiLM layers
    num_film_layers: int = 2  # Number of layers to apply FiLM
    film_hidden_dim: int = 256

    # Training
    freeze_base: bool = True  # Freeze base MLP during FiLM training


class FiLMGenerator(nn.Module):
    """
    Generate FiLM parameters (γ, β) from affect vector.

    For each FiLM layer, generates:
    - γ (gamma): scaling parameter
    - β (beta): shifting parameter

    Formula: FiLM(x) = γ * x + β
    """

    def __init__(
        self,
        affect_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.affect_dim = affect_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Create FiML layers for each target layer
        self.film_layers = nn.ModuleList([
            nn.Linear(affect_dim, hidden_dim * 2)  # 2 for γ and β
            for _ in range(num_layers)
        ])

        logger.debug(
            f"FiLMGenerator: affect_dim={affect_dim}, "
            f"hidden_dim={hidden_dim}, num_layers={num_layers}"
        )

    def forward(self, affect: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate FiLM parameters from affect vector.

        Args:
            affect: Affect vector of shape (batch, affect_dim)

        Returns:
            List of (gamma, beta) tuples, one per FiLM layer
        """
        films = []

        for layer in self.film_layers:
            params = layer(affect)  # (batch, hidden_dim * 2)

            # Split into gamma and beta
            gamma, beta = torch.chunk(params, 2, dim=-1)  # Each (batch, hidden_dim)

            films.append((gamma, beta))

        return films


class BaseDDSPDecoder(nn.Module):
    """
    Base DDSP decoder MLP (pre-trained).

    This is the network that would have been pre-trained on
    112D → 65D harmonic+noise parameter mapping.
    """

    def __init__(
        self,
        input_dim: int = 112,
        hidden_dim: int = 256,
        output_dim: int = 65,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Base MLP (pre-trained)
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base MLP."""
        for layer in self.layers:
            x = layer(x)
        return x

    def get_layer_outputs(
        self,
        x: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Get outputs at each layer for FiLM application.

        Returns intermediate activations before each ReLU.
        """
        outputs = []

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                outputs.append(x)
            else:
                x = layer(x)

        return outputs


class DualStreamDDSPDecoder(nn.Module):
    """
    DDSP Decoder with FiLM modulation for dual-stream control.

    Architecture:
        112D Features → Base MLP (frozen) → FiLM modulation → 65D DDSP params
                         ↑
                    16D Affect → FiLM Generator → γ, β

    Training Strategy:
        1. Freeze base MLP (pre-trained weights)
        2. Train only FiLM γ/β generators
        3. Fine-tune entire network end-to-end
    """

    def __init__(
        self,
        pretrained_model: Optional[BaseDDSPDecoder] = None,
        config: Optional[FiLMDecoderConfig] = None,
    ):
        super().__init__()
        self.config = config or FiLMDecoderConfig()

        # Base DDSP decoder (pre-trained)
        if pretrained_model is not None:
            self.base_decoder = pretrained_model
        else:
            self.base_decoder = BaseDDSPDecoder(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
            )

        # FiLM generator for affect modulation
        self.film_gen = FiLMGenerator(
            affect_dim=self.config.affect_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_film_layers,
        )

        # Freeze base decoder if specified
        if self.config.freeze_base:
            for param in self.base_decoder.parameters():
                param.requires_grad = False
            logger.info("Base decoder parameters frozen")

        logger.info(
            f"DualStreamDDSPDecoder initialized with "
            f"FiLM layers: {self.config.num_film_layers}"
        )

    def unfreeze_base(self) -> None:
        """Unfreeze base decoder for fine-tuning."""
        for param in self.base_decoder.parameters():
            param.requires_grad = True
        logger.info("Base decoder parameters unfrozen for fine-tuning")

    def forward(
        self,
        features_112d: torch.Tensor,
        affect_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with dual-stream input.

        Args:
            features_112d: Acoustic features (batch, 112)
            affect_vector: Affect vector (batch, 16)

        Returns:
            harmonic_amps: (batch, 60) - harmonic amplitudes
            noise_mags: (batch, 5) - noise magnitudes
            full_output: (batch, 65) - combined DDSP parameters
        """
        # Generate FiLM parameters
        films = self.film_gen(affect_vector)

        # Get layer-by-layer outputs from base decoder
        x = features_112d
        layer_idx = 0
        film_idx = 0

        for i, layer in enumerate(self.base_decoder.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)

                # Apply FiLM modulation at specified layers
                if film_idx < len(films) and layer_idx < self.config.num_film_layers * 2:
                    gamma, beta = films[film_idx]
                    x = gamma * x + beta  # FiLM modulation
                    film_idx += 1

                layer_idx += 1
            else:
                x = layer(x)

        # Split into harmonic and noise
        full_output = x
        harmonic_amps = F.softmax(x[:, :60], dim=-1)
        noise_mags = F.relu(x[:, 60:])

        return harmonic_amps, noise_mags, full_output

    def loss_function(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss for decoder training.

        Args:
            predicted: Predicted DDSP parameters (batch, 65)
            target: Target DDSP parameters (batch, 65)

        Returns:
            Loss value
        """
        # Split into harmonic and noise
        pred_harmonic = predicted[:, :60]
        pred_noise = predicted[:, 60:]
        target_harmonic = target[:, :60]
        target_noise = target[:, 60:]

        # Cross-entropy for harmonic distribution
        harmonic_loss = F.cross_entropy(pred_harmonic, target_harmonic)

        # MSE for noise magnitudes
        noise_loss = F.mse_loss(pred_noise, target_noise)

        return harmonic_loss + noise_loss


class DualStreamDDSPTrainer:
    """
    Trainer for dual-stream DDSP decoder with FiLM.
    """

    def __init__(
        self,
        model: DualStreamDDSPDecoder,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = torch.device(device)

        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

        logger.info(
            f"DualStreamDDSPTrainer initialized: "
            f"{sum(p.numel() for p in trainable_params)} trainable parameters"
        )

    def train_step(
        self,
        features_112d: torch.Tensor,
        affect_vector: torch.Tensor,
        target_ddsp: torch.Tensor,
    ) -> float:
        """Single training step."""
        self.model.train()

        # Forward pass
        _, _, predicted = self.model(features_112d, affect_vector)

        # Compute loss
        loss = self.model.loss_function(predicted, target_ddsp)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(
        self,
        features_112d: torch.Tensor,
        affect_vector: torch.Tensor,
        target_ddsp: torch.Tensor,
    ) -> float:
        """Validation step."""
        self.model.eval()

        with torch.no_grad():
            _, _, predicted = self.model(features_112d, affect_vector)
            loss = self.model.loss_function(predicted, target_ddsp)

        return loss.item()


def create_dual_stream_decoder(
    pretrained_model: Optional[BaseDDSPDecoder] = None,
    config: Optional[FiLMDecoderConfig] = None,
) -> DualStreamDDSPDecoder:
    """Factory function to create dual-stream decoder."""
    return DualStreamDDSPDecoder(pretrained_model, config)


if __name__ == "__main__":
    # Test the decoder
    config = FiLMDecoderConfig()

    # Create models
    base_decoder = BaseDDSPDecoder()
    decoder = create_dual_stream_decoder(base_decoder, config)

    # Create dummy inputs
    batch_size = 4
    features_112d = torch.randn(batch_size, 112)
    affect_vector = torch.randn(batch_size, 16)

    # Forward pass
    harmonic, noise, full = decoder(features_112d, affect_vector)

    print(f"Harmonic amplitudes shape: {harmonic.shape}")
    print(f"Noise magnitudes shape: {noise.shape}")
    print(f"Full output shape: {full.shape}")

    # Test FiLM effect
    affect_high_arousal = torch.zeros(batch_size, 16)
    affect_high_arousal[:, 0] = 1.0  # High arousal

    affect_low_arousal = torch.zeros(batch_size, 16)
    affect_low_arousal[:, 0] = 0.0  # Low arousal

    _, _, output_high = decoder(features_112d, affect_high_arousal)
    _, _, output_low = decoder(features_112d, affect_low_arousal)

    # Outputs should differ due to FiLM modulation
    diff = (output_high - output_low).abs().mean().item()
    print(f"Output difference (high vs low arousal): {diff:.4f}")
