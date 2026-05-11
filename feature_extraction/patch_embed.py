#!/usr/bin/env python3
"""
Patch Embedding for Spectrogram Tokenization

ViT-style patch extraction from spectrograms. Converts a 2D spectrogram
into a sequence of patch embeddings for Transformer processing.

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class PatchEmbedConfig:
    """Configuration for patch embedding."""
    # Spectrogram dimensions (frequency bins, time frames)
    img_size: tuple = (128, 128)  # (freq_bins, time_frames)

    # Patch dimensions
    patch_size: tuple = (16, 16)  # Non-overlapping patches

    # Embedding dimensions
    in_channels: int = 1  # Single channel for mono spectrogram
    embed_dim: int = 256  # Transformer embedding dimension

    # Positional encoding
    drop_patch: float = 0.0  # Patch dropout rate
    drop_pos: float = 0.0  # Positional embedding dropout

    @property
    def num_patches(self) -> int:
        """Number of patches after splitting spectrogram."""
        return (
            self.img_size[0] // self.patch_size[0]
        ) * (
            self.img_size[1] // self.patch_size[1]
        )

    @property
    def num_patches_freq(self) -> int:
        """Number of patches along frequency axis."""
        return self.img_size[0] // self.patch_size[0]

    @property
    def num_patches_time(self) -> int:
        """Number of patches along time axis."""
        return self.img_size[1] // self.patch_size[1]


def create_patch_embedding(
    img_size: tuple = (128, 128),
    patch_size: tuple = (16, 16),
    in_channels: int = 1,
    embed_dim: int = 256,
) -> PatchEmbedding:
    """Factory function to create patch embedding layer."""
    config = PatchEmbedConfig(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )
    return PatchEmbedding(config)


class PatchEmbedding(nn.Module):
    """
    Splits spectrogram into non-overlapping patches and projects them.

    Similar to Vision Transformer (ViT) patch embedding, but adapted for
    spectrograms where the two dimensions represent frequency and time.

    Process:
    1. Conv2D with kernel=patch_size, stride=patch_size extracts patches
    2. Flatten spatial dimensions: (B, C, H_f, W_t) -> (B, H_f*W_t, D)
    3. Add CLS token (learnable classification token)
    4. Add positional embeddings (learnable position information)

    Args:
        config: PatchEmbedConfig with parameters

    Example:
        >>> config = PatchEmbedConfig(img_size=(128, 256), patch_size=(16, 16))
        >>> embed = PatchEmbedding(config)
        >>> spec = torch.randn(2, 1, 128, 256)  # Batch of 2 spectrograms
        >>> patches = embed(spec)
        >>> print(patches.shape)  # (2, 129, 256) = (B, num_patches+CLS, embed_dim)
    """

    def __init__(self, config: PatchEmbedConfig):
        super().__init__()
        self.config = config

        # Project patches using Conv2d
        # Kernel size = patch size, stride = patch size (non-overlapping)
        self.proj = nn.Conv2d(
            config.in_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        # CLS token (classification token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))

        # Positional embeddings (including CLS token position)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.num_patches + 1, config.embed_dim)
        )

        # Dropout
        self.drop_patch = nn.Dropout(config.drop_patch)
        self.drop_pos = nn.Dropout(config.drop_pos)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using trunc normal."""
        # CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Projection layer
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert spectrogram to patch embeddings.

        Args:
            x: Input spectrogram (Batch, Channels, Freq, Time)
            mask: Optional binary mask (Batch, num_patches) where 1=keep, 0=mask
                  Used during MAE training to indicate which patches are visible

        Returns:
            Patch embeddings (Batch, num_patches+1, embed_dim)
            - First token is CLS token
            - Remaining tokens are patch embeddings with positional encoding
        """
        B = x.shape[0]

        # Project patches: (B, C, H, W) -> (B, D, H//Ph, W//Pw)
        x = self.proj(x)

        # Flatten: (B, D, H//Ph, W//Pw) -> (B, D, num_patches)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)

        # Apply dropout to patches
        x = self.drop_patch(x)

        # Concatenate CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, D)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.drop_pos(x)

        # Apply mask if provided (for MAE training)
        if mask is not None:
            # Keep only unmasked patches (excluding CLS)
            # mask: (B, num_patches) where 1=visible, 0=masked
            batch_indices = torch.arange(B, device=x.device).unsqueeze(1)
            visible_indices = mask.nonzero(as_tuple=False)  # (N_visible, 2)

            # Keep CLS token + visible patches
            cls = x[:, 0:1, :]  # (B, 1, D)
            patches = x[:, 1:, :]  # (B, num_patches, D)

            # For each batch, select visible patches
            visible_patches = []
            for b in range(B):
                batch_mask = mask[b]  # (num_patches,)
                visible_patches.append(patches[b:b+1, batch_mask, :])

            # Recombine (this is a simplification; efficient indexing may differ)
            # For now, return full sequence - masking handled in trainer
            pass

        return x

    def get_patch_positions(self) -> torch.Tensor:
        """
        Get the 2D positions of each patch in the original spectrogram.

        Returns:
            Tensor of shape (num_patches, 2) with (freq_idx, time_idx) for each patch
        """
        freq_positions = torch.arange(self.config.num_patches_freq)
        time_positions = torch.arange(self.config.num_patches_time)

        # Create grid of positions
        freq_grid = freq_positions.repeat_interleave(self.config.num_patches_time)
        time_grid = time_positions.repeat(self.config.num_patches_freq)

        return torch.stack([freq_grid, time_grid], dim=1)


class AdaptivePatchEmbedding(nn.Module):
    """
    Adaptive patch embedding that handles variable spectrogram sizes.

    Unlike standard PatchEmbedding which expects fixed img_size, this
    version adapts to any input size by computing positions dynamically.
    """

    def __init__(
        self,
        patch_size: tuple = (16, 16),
        in_channels: int = 1,
        embed_dim: int = 256,
        max_patches: int = 256,  # Maximum number of patches for positional embeds
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.max_patches = max_patches

        # Projection layer
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Learnable positional embeddings for up to max_patches
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_patches + 1, embed_dim)
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert variable-size spectrogram to patch embeddings.

        Args:
            x: Input spectrogram (Batch, Channels, Freq, Time)

        Returns:
            Patch embeddings (Batch, num_patches+1, embed_dim)
        """
        B = x.shape[0]

        # Project patches
        x = self.proj(x)
        num_patches = x.shape[2] * x.shape[3]

        # Flatten
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, D)

        # Add positional embeddings (slice to actual number of patches)
        x = x + self.pos_embed[:, :num_patches + 1, :]

        return x


# Preset configurations

MAE_BASE_PATCHES = PatchEmbedConfig(
    img_size=(128, 128),
    patch_size=(16, 16),
    embed_dim=768,
)

MAE_LARGE_PATCHES = PatchEmbedConfig(
    img_size=(224, 224),
    patch_size=(16, 16),
    embed_dim=1024,
)

BIOMAE_PATCHES = PatchEmbedConfig(
    img_size=(128, 256),  # Wider for time dimension
    patch_size=(16, 16),
    embed_dim=256,
)
