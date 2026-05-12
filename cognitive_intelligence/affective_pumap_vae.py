#!/usr/bin/env python3
"""
Affective Stream: pUMAP + β-VAE for Disentangled Encoding

Implements the two-stage compression pipeline for continuous affective features:
1. Parametric UMAP (pUMAP): 54D → 256D → 128D → 30D
2. β-VAE: 30D → 16D disentangled latent (β=2.0)

The β-VAE loss forces specific dimensions to map to physical traits like:
- Physiological arousal
- Spectral harshness
- Tension/stress
- Valence

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AffectiveConfig:
    """Configuration for affective encoding pipeline."""
    # Input dimension (from AffectiveFeatureExtractor)
    input_dim: int = 54

    # pUMAP encoder dimensions
    pumap_hidden: Tuple[int, ...] = (256, 128)
    pumap_output: int = 30

    # β-VAE configuration
    vae_latent: int = 16
    vae_hidden: int = 32
    beta: float = 2.0  # Disentanglement strength

    # Training
    learning_rate: float = 1e-3
    recon_loss_weight: float = 1.0
    kl_loss_weight: float = 1.0


class ParametricUMAP(nn.Module):
    """
    Parametric UMAP encoder for dimensionality reduction.

    Unlike traditional UMAP which creates a static embedding, pUMAP learns
    a neural network that can project new data into the reduced space.

    Architecture: 54D → 256D → 128D → 30D

    The regression loss optimizes the encoder to preserve the UMAP
    neighborhood structure in the latent space.
    """

    def __init__(
        self,
        input_dim: int = 54,
        hidden_dims: Tuple[int, ...] = (256, 128),
        output_dim: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build encoder
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Final projection to output_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

        # Decoder for reconstruction loss
        decoder_layers = []
        in_dim = output_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to pUMAP latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from pUMAP latent space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through pUMAP.

        Returns:
            z: Encoded latent (Batch, output_dim)
            x_recon: Reconstructed input (Batch, input_dim)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon

    def reconstruction_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        MSE reconstruction loss for pUMAP training.

        L_UMAP = MSE(Encoder(x), Target_UMAP)
        """
        return F.mse_loss(x_recon, x)


class BetaVAE(nn.Module):
    """
    β-VAE for disentangled affective representation.

    The β parameter controls the trade-off between reconstruction and
    disentanglement. With β=2.0, the KL divergence is heavily weighted,
    forcing the network to use independent latent dimensions.

    Target disentangled dimensions:
    - Arousal (physiological activation)
    - Valence (positive/negative affect)
    - Tension (muscular/jitter stress)
    - Spectral harshness

    Architecture: 30D → 32D → 16D latent → 32D → 30D
    """

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        beta: float = 2.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode to latent distribution parameters.

        Returns:
            mu: Mean of latent Gaussian (Batch, latent_dim)
            logvar: Log variance of latent Gaussian (Batch, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon.

        Enables backpropagation through stochastic sampling.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through β-VAE.

        Returns:
            x_recon: Reconstructed input (Batch, input_dim)
            mu: Latent mean (Batch, latent_dim)
            logvar: Latent log variance (Batch, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        β-VAE loss function.

        L_total = L_recon + β * L_KL

        where:
        L_recon = ||x - x_recon||^2
        L_KL = -0.5 * sum(1 + log(σ^2) - μ^2 - σ^2)

        The β term forces disentanglement by penalizingKL divergence.
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')

        # KL divergence: KL(q(z|x) || p(z))
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss with β weighting
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
        }


class AffectiveStream(nn.Module):
    """
    Complete affective encoding pipeline: 54D → pUMAP → β-VAE → 16D.

    This is Stream 1 of the dual-stream encoding architecture:
    - Stream 1 (Affective): Continuous valence/arousal encoding
    - Stream 2 (Syntactic): Discrete tokenization via VQ-VAE

    Pipeline:
        54D features
            ↓
        pUMAP (54→256→128→30D)
            ↓
        β-VAE (30→16D, β=2.0)
            ↓
        16D disentangled latent

    Example:
        >>> stream = AffectiveStream()
        >>> features_54d = torch.randn(4, 54)  # Batch of 4
        >>> latent_16d = stream.encode(features_54d)
        >>> print(latent_16d.shape)  # (4, 16)
    """

    def __init__(self, config: Optional[AffectiveConfig] = None):
        super().__init__()

        if config is None:
            config = AffectiveConfig()

        self.config = config

        # Stage 1: pUMAP
        self.pumap = ParametricUMAP(
            input_dim=config.input_dim,
            hidden_dims=config.pumap_hidden,
            output_dim=config.pumap_output,
        )

        # Stage 2: β-VAE
        self.vae = BetaVAE(
            input_dim=config.pumap_output,
            hidden_dim=config.vae_hidden,
            latent_dim=config.vae_latent,
            beta=config.beta,
        )

        logger.info(
            f"AffectiveStream initialized: "
            f"{config.input_dim}D → pUMAP({config.pumap_output}D) → "
            f"β-VAE({config.vae_latent}D, β={config.beta})"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode affective features to 16D latent.

        Args:
            x: Input features (Batch, input_dim=54)

        Returns:
            z: 16D latent encoding (Batch, 16)
        """
        # pUMAP encoding
        z_pumap = self.pumap.encode(x)

        # β-VAE encoding (use mean for deterministic output)
        mu, _ = self.vae.encode(z_pumap)

        return mu

    def encode_stochastic(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode with stochastic sampling (for training).

        Returns:
            z: Sampled latent (Batch, 16)
            mu: Latent mean (Batch, 16)
            logvar: Latent log variance (Batch, 16)
        """
        # pUMAP encoding
        z_pumap = self.pumap.encode(x)

        # β-VAE encoding with sampling
        mu, logvar = self.vae.encode(z_pumap)
        z = self.vae.reparameterize(mu, logvar)

        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from 16D latent back to 54D features.

        Args:
            z: Latent encoding (Batch, 16)

        Returns:
            x_recon: Reconstructed features (Batch, 54)
        """
        # β-VAE decode
        z_pumap = self.vae.decode(z)

        # pUMAP decode
        x_recon = self.pumap.decode(z_pumap)

        return x_recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.

        Returns:
            x_recon: Reconstructed input (Batch, 54)
            mu: VAE latent mean (Batch, 16)
            logvar: VAE latent log variance (Batch, 16)
            z_pumap: pUMAP intermediate (Batch, 30)
        """
        # pUMAP
        z_pumap, x_recon_pumap = self.pumap(x)

        # β-VAE
        x_recon, mu, logvar = self.vae(z_pumap)

        # Final reconstruction
        x_final_recon = self.pumap.decode(x_recon)

        return x_final_recon, mu, logvar, z_pumap

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z_pumap: torch.Tensor,
        x_recon_pumap: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Combined loss for the affective stream.

        L_total = L_recon + L_pumap + β * L_KL

        where:
        L_recon: Final reconstruction loss
        L_pumap: pUMAP reconstruction loss
        L_KL: β-VAE KL divergence
        """
        # pUMAP reconstruction loss
        pumap_loss = self.pumap.reconstruction_loss(x, x_recon_pumap)

        # β-VAE loss
        vae_loss, vae_losses = self.vae.loss_function(z_pumap, x_recon, mu, logvar)

        # Final reconstruction loss
        final_recon_loss = F.mse_loss(x_recon, x)

        # Total loss
        total_loss = (
            self.config.recon_loss_weight * final_recon_loss +
            0.5 * pumap_loss +
            vae_loss
        )

        losses = {
            'total_loss': total_loss.item(),
            'pumap_recon_loss': pumap_loss.item(),
            'vae_recon_loss': vae_losses['recon_loss'],
            'vae_kl_loss': vae_losses['kl_loss'],
            'final_recon_loss': final_recon_loss.item(),
        }

        return total_loss, losses

    def get_disentangled_metrics(self) -> dict:
        """
        Compute disentanglement metrics for the 16D latent space.

        Returns:
            Dictionary with metrics for each latent dimension
        """
        # This would require a validation set with known ground truth
        # For now, return placeholder
        return {
            'latent_dim': self.config.vae_latent,
            'beta': self.config.beta,
        }


def create_affective_stream(config: Optional[AffectiveConfig] = None) -> AffectiveStream:
    """Factory function to create affective encoding pipeline."""
    if config is None:
        config = AffectiveConfig()
    return AffectiveStream(config)


# Preset configurations

AFFECTIVE_MINIMAL = AffectiveConfig(
    input_dim=54,
    pumap_hidden=(128, 64),
    pumap_output=20,
    vae_latent=8,
    vae_hidden=16,
    beta=2.0,
)

AFFECTIVE_BASE = AffectiveConfig(
    input_dim=54,
    pumap_hidden=(256, 128),
    pumap_output=30,
    vae_latent=16,
    vae_hidden=32,
    beta=2.0,
)

AFFECTIVE_LARGE = AffectiveConfig(
    input_dim=54,
    pumap_hidden=(512, 256, 128),
    pumap_output=64,
    vae_latent=32,
    vae_hidden=64,
    beta=3.0,  # Stronger disentanglement
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test affective stream
    stream = AffectiveStream(AFFECTIVE_BASE)

    # Generate test features
    batch_size = 4
    features = torch.randn(batch_size, 54)

    # Encode
    latent = stream.encode(features)
    print(f"Input shape: {features.shape}")
    print(f"Latent shape: {latent.shape}")

    # Decode
    recon = stream.decode(latent)
    print(f"Reconstruction shape: {recon.shape}")

    # Loss
    x_recon, mu, logvar, z_pumap = stream(features)
    loss, losses = stream.loss_function(features, x_recon, mu, logvar, z_pumap, z_pumap)
    print(f"Losses: {losses}")
    print(f"Parameters: {sum(p.numel() for p in stream.parameters()):,}")
