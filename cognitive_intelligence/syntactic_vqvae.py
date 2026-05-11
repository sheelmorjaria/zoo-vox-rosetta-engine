#!/usr/bin/env python3
"""
VQ-VAE for Syntactic Encoding (Stream 2)

Implements a VQ-VAE with EMA codebook updates for discrete syntactic tokenization.
Encodes 44D syntactic features to discrete tokens (0-63).

Key Benefits:
- EMA codebook updates prevent codebook collapse
- >80% codebook utilization target
- Codebook revival for dead tokens

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
class VQVAEConfig:
    """Configuration for VQ-VAE."""
    input_dim: int = 44
    codebook_size: int = 64
    codebook_dim: int = 32
    hidden_dim: int = 128
    commitment_cost: float = 0.25
    decay: float = 0.99
    epsilon: float = 1e-5
    # Revival threshold: tokens used less than this % of batches are "dead"
    revival_threshold: float = 0.01


class EMAVectorQuantizer(nn.Module):
    """
    Vector quantizer with EMA codebook updates and codebook revival.

    Prevents codebook collapse via:
    1. Exponential moving average of encoder outputs
    2. Codebook revival for unused tokens
    """

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        revival_threshold: float = 0.01,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.epsilon = epsilon
        self.revival_threshold = revival_threshold

        # EMA codebook
        self.register_buffer(
            "codebook_ema",
            torch.randn(codebook_size, codebook_dim),
        )
        self.register_buffer(
            "cluster_size_ema",
            torch.zeros(codebook_size),
        )

        # Track per-token usage for revival
        self.register_buffer(
            "token_usage_count",
            torch.zeros(codebook_size),
        )
        self.register_buffer(
            "total_batches",
            torch.tensor(0.0),
        )

    def update_ema(self, z_flat: torch.Tensor, token_ids: torch.Tensor) -> None:
        """
        Update EMA codebook and cluster sizes.

        This is called during training to move the codebook vectors
        toward the mean of encoder outputs assigned to each code.
        """
        # Get one-hot encoding of token assignments
        encodings = F.one_hot(token_ids, self.codebook_size).float()

        # Sum over batch to get counts per code
        cluster_sizes = encodings.sum(dim=0)

        # Weighted sum of encoder outputs for each code
        # z_flat: (batch * seq, codebook_dim)
        # encodings.T: (codebook_size, batch * seq)
        # encodings.T @ z_flat: (codebook_size, codebook_dim)
        encoded_sum = torch.matmul(encodings.T, z_flat)

        # Update EMA cluster sizes
        self.cluster_size_ema.mul_(self.decay).add_(cluster_sizes, alpha=1 - self.decay)

        # Update EMA codebook
        # Add small epsilon to avoid division by zero
        cluster_size_ema_expanded = self.cluster_size_ema.unsqueeze(1)
        codebook_update = encoded_sum / (cluster_size_ema_expanded + self.epsilon)
        self.codebook_ema.mul_(self.decay).add_(codebook_update, alpha=1 - self.decay)

        # Track usage for revival
        self.token_usage_count.add_((cluster_sizes > 0).float())
        self.total_batches.add_(1.0)

    def revive_dead_codes(self, z_flat: torch.Tensor) -> None:
        """
        Revive dead codes by replacing them with random encoder outputs.

        A code is considered "dead" if it's used less than revival_threshold
        of the time across all batches.
        """
        if self.total_batches < 10:
            # Don't revive in the first few batches
            return

        # Calculate usage percentage for each code
        usage_rate = self.token_usage_count / self.total_batches

        # Find dead codes (usage rate below threshold)
        dead_codes = usage_rate < self.revival_threshold

        if dead_codes.any():
            num_dead = dead_codes.sum().item()
            if num_dead > 0:
                # Sample random encoder outputs to replace dead codes
                num_samples = min(num_dead, z_flat.shape[0])
                indices = torch.randperm(z_flat.shape[0], device=z_flat.device)[:num_samples]

                # Replace dead codes with random encoder outputs
                dead_indices = torch.where(dead_codes)[0]
                for i, dead_idx in enumerate(dead_indices[:num_samples]):
                    self.codebook_ema[dead_idx] = z_flat[indices[i]].detach()
                    self.cluster_size_ema[dead_idx] = 0.0
                    self.token_usage_count[dead_idx] = 0.0

                logger.debug(f"Revived {num_dead} dead codes")

    def forward(self, z: torch.Tensor, training: bool = True) -> tuple:
        """
        Quantize z using EMA codebook.

        Args:
            z: Encoder output of shape (batch, ..., codebook_dim)
            training: Whether in training mode (updates EMA, revives dead codes)

        Returns:
            z_q: Quantized vectors
            token_ids: Discrete token indices
            perplexity: Codebook utilization metric
        """
        # Flatten for distance computation
        z_flat = z.reshape(-1, self.codebook_dim)

        # Compute distances to codebook
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z^T * e
        z_norm = (z_flat ** 2).sum(dim=1, keepdim=True)
        codebook_norm = (self.codebook_ema ** 2).sum(dim=1)
        distances = z_norm + codebook_norm - 2 * torch.matmul(z_flat, self.codebook_ema.t())

        # Find nearest codebook entry
        token_ids = torch.argmin(distances, dim=1)

        # Reshape to original shape
        token_ids = token_ids.view(z.shape[:-1])

        # Quantize
        z_flat_q = self.codebook_ema[token_ids.flatten()]
        z_q = z_flat_q.view_as(z)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        # Update EMA and revive dead codes during training
        if training:
            self.update_ema(z_flat, token_ids.flatten())
            self.revive_dead_codes(z_flat)

        # Perplexity for monitoring
        encodings = F.one_hot(token_ids.flatten(), self.codebook_size).float()
        perplexity = torch.exp(
            -torch.sum(
                encodings.mean(0) * torch.log(encodings.mean(0) + self.epsilon),
            )
        )

        return z_q, token_ids, perplexity

    def get_utilization_stats(self) -> dict:
        """Get codebook utilization statistics."""
        if self.total_batches < 1:
            # Return zeros before training
            return {
                "utilization_percent": 0.0,
                "active_codes": 0,
                "total_codes": self.codebook_size,
                "dead_codes": self.codebook_size,
                "per_code_usage": [0.0] * self.codebook_size,
            }

        usage_rate = self.token_usage_count / self.total_batches
        active_codes = (usage_rate > self.revival_threshold).sum().item()

        return {
            "utilization_percent": active_codes / self.codebook_size * 100,
            "active_codes": active_codes,
            "total_codes": self.codebook_size,
            "dead_codes": self.codebook_size - active_codes,
            "per_code_usage": usage_rate.cpu().numpy().tolist(),
        }


class SyntacticVQVAE(nn.Module):
    """VQ-VAE: ~44D syntactic features → N discrete tokens."""

    def __init__(
        self,
        input_dim: int = 44,
        codebook_size: int = 64,
        codebook_dim: int = 32,
        hidden_dim: int = 128,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        revival_threshold: float = 0.01,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, codebook_dim),
        )

        # VQ layer with EMA and revival
        self.vq = EMAVectorQuantizer(
            codebook_size,
            codebook_dim,
            decay,
            revival_threshold=revival_threshold,
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(codebook_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space."""
        return self.encoder(x)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode from quantized latent."""
        return self.decoder(z_q)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple:
        """Forward pass: encode → quantize → decode."""
        z = self.encode(x)
        z_q, token_ids, perplexity = self.vq(z, training=self.training)
        x_recon = self.decode(z_q)
        return x_recon, z, z_q, token_ids, perplexity

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
        z_q: torch.Tensor,
    ) -> dict:
        """VQ-VAE loss with commitment term."""
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        # Commitment loss (encoder commitment to codebook)
        commitment_loss = F.mse_loss(z_q.detach(), z)

        # Note: codebook loss is handled via EMA updates, not backprop
        total_loss = recon_loss + commitment_loss * self.commitment_cost

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "commitment_loss": commitment_loss,
        }

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize input to discrete tokens (inference)."""
        with torch.no_grad():
            z = self.encode(x)
            _, token_ids, _ = self.vq(z, training=False)
        return token_ids

    def codebook_utilization(self) -> float:
        """Calculate codebook utilization percentage."""
        stats = self.vq.get_utilization_stats()
        return stats["utilization_percent"]

    def get_utilization_stats(self) -> dict:
        """Get detailed codebook utilization statistics."""
        return self.vq.get_utilization_stats()


def create_syntactic_vqvae(config: Optional[VQVAEConfig] = None) -> SyntacticVQVAE:
    """Create a VQ-VAE with default or custom config."""
    if config is None:
        config = VQVAEConfig()

    return SyntacticVQVAE(
        input_dim=config.input_dim,
        codebook_size=config.codebook_size,
        codebook_dim=config.codebook_dim,
        hidden_dim=config.hidden_dim,
        commitment_cost=config.commitment_cost,
        decay=config.decay,
        revival_threshold=config.revival_threshold,
    )
