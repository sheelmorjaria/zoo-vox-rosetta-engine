#!/usr/bin/env python3
"""
β-VAE for Affective Encoding (Stream 1)

Implements a β-VAE (β=2.0) for disentangled affective representation learning.
Encodes 54D affective features to 16D continuous latent space.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AffectVAEConfig:
    """Configuration for β-VAE encoder."""
    input_dim: int = 54
    latent_dim: int = 16
    hidden_dim: int = 64
    beta: float = 2.0


class BetaVAE(nn.Module):
    """β-VAE: ~54D affective features → 16D disentangled latent space."""

    def __init__(
        self,
        input_dim: int = 54,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        beta: float = 2.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to mu and logvar."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode -> reparameterize -> decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """β-VAE loss with weighted KL divergence."""
        BCE = F.mse_loss(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = BCE + self.beta * KLD
        return {
            "total_loss": total_loss,
            "reconstruction_loss": BCE,
            "kl_loss": KLD,
        }

    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without sampling (for inference)."""
        mu, _ = self.encode(x)
        return mu


def create_affect_vae(config: Optional[AffectVAEConfig] = None) -> BetaVAE:
    """Create a β-VAE with default or custom config."""
    if config is None:
        config = AffectVAEConfig()
    return BetaVAE(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        beta=config.beta,
    )
