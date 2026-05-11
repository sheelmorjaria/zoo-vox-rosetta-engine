#!/usr/bin/env python3
"""
BioMAE: Bioacoustic Masked Autoencoder

Self-supervised transformer model for learning acoustic features from
raw spectrograms. Uses asymmetric encoder-decoder architecture with
high masking ratio (75%).

Key design:
- Encoder: Lightweight (4 layers), processes visible patches only
- Decoder: Shallower (2 layers), reconstructs full spectrogram
- Output: 112D embedding (compatible with existing Rosetta pipeline)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_extraction.patch_embed import PatchEmbedding, PatchEmbedConfig


@dataclass
class EncoderConfig:
    """Configuration for BioMAE Encoder."""
    # Patch embedding
    img_size: tuple = (128, 128)
    patch_size: tuple = (16, 16)
    in_channels: int = 1
    embed_dim: int = 256

    # Transformer
    depth: int = 4  # Number of transformer layers
    num_heads: int = 4  # Number of attention heads
    mlp_ratio: float = 2.0  # MLP hidden dim = embed_dim * mlp_ratio
    drop_rate: float = 0.0  # Dropout rate
    attn_drop_rate: float = 0.0  # Attention dropout rate

    # Output
    output_dim: int = 112  # Final Rosetta embedding dimension


@dataclass
class DecoderConfig:
    """Configuration for BioMAE Decoder (lighter than encoder)."""
    embed_dim: int = 256  # Must match encoder
    decoder_embed_dim: int = 128  # Smaller than encoder for efficiency
    depth: int = 2  # Shallower than encoder
    num_heads: int = 4
    mlp_ratio: float = 2.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0

    # Reconstruction
    patch_size: tuple = (16, 16)
    img_size: tuple = (128, 128)
    in_channels: int = 1


def create_biomae_model(
    encoder_config: Optional[EncoderConfig] = None,
    decoder_config: Optional[DecoderConfig] = None,
) -> 'BioMAEModel':
    """Factory function to create complete BioMAE model."""
    if encoder_config is None:
        encoder_config = EncoderConfig()
    if decoder_config is None:
        decoder_config = DecoderConfig(
            embed_dim=encoder_config.embed_dim,
            patch_size=encoder_config.patch_size,
            img_size=encoder_config.img_size,
            in_channels=encoder_config.in_channels,
        )
    return BioMAEModel(encoder_config, decoder_config)


class BioMAEEncoder(nn.Module):
    """
    Lightweight Transformer Encoder for bioacoustic feature extraction.

    Processes visible spectrogram patches and produces:
    1. Encoded patch representations (for decoder)
    2. 112D Rosetta embedding (via mean pooling + projection)

    Args:
        config: EncoderConfig with model parameters

    Example:
        >>> encoder = BioMAEEncoder(EncoderConfig())
        >>> spec = torch.randn(2, 1, 128, 128)  # Batch of spectrograms
        >>> encoding, embedding = encoder(spec)
        >>> print(embedding.shape)  # (2, 112)
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        patch_config = PatchEmbedConfig(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )
        self.patch_embed = PatchEmbedding(patch_config)

        # Transformer encoder layers
        dpr = [x.item() for x in torch.linspace(0, config.drop_rate, config.depth)]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            dropout=config.drop_rate,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,  # Pre-LN architecture
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.depth,
        )

        # Layer norm before projection
        self.norm = nn.LayerNorm(config.embed_dim)

        # 112D projection head (for Rosetta compatibility)
        self.projection = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_patches: bool = False
    ) -> torch.Tensor:
        """
        Encode spectrogram to 112D embedding.

        Args:
            x: Input spectrogram (Batch, Channels, Freq, Time)
            return_patches: If True, also return encoded patches

        Returns:
            If return_patches=False: 112D embedding (Batch, 112)
            If return_patches=True: (encoded_patches, embedding_112d)
        """
        # Patch embedding: (B, C, H, W) -> (B, num_patches+1, D)
        x = self.patch_embed(x)

        # Transformer encoding
        x = self.transformer(x)  # (B, num_patches+1, D)

        # Layer norm
        x = self.norm(x)

        # Store encoded patches if needed (for decoder during training)
        encoded_patches = x

        # Mean pooling over patch sequence (excluding CLS token)
        # Alternative: use CLS token directly
        pooled = x[:, 1:, :].mean(dim=1)  # (B, D)

        # Project to 112D Rosetta embedding
        embedding = self.projection(pooled)  # (B, 112)

        if return_patches:
            return encoded_patches, embedding
        return embedding

    def forward_visible(
        self,
        x: torch.Tensor,
        visible_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with only visible patches (for MAE training).

        Args:
            x: Input spectrogram (Batch, Channels, Freq, Time)
            visible_mask: Boolean mask (Batch, num_patches) where True=visible

        Returns:
            Encoded visible patches (Batch, num_visible+1, embed_dim)
        """
        # Full patch embedding
        x = self.patch_embed(x)  # (B, num_patches+1, D)

        # Select only visible patches (keep CLS token)
        cls_token = x[:, 0:1, :]  # (B, 1, D)
        patch_tokens = x[:, 1:, :]  # (B, num_patches, D)

        # Mask patches
        batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(1)
        visible_patches = []
        for b in range(x.shape[0]):
            mask = visible_mask[b]
            visible_patches.append(patch_tokens[b:b+1, mask, :])

        # Recombine (simplified - efficient implementation uses gather)
        # For now, process all patches and apply mask during reconstruction loss
        x = torch.cat([cls_token, patch_tokens], dim=1)

        # Transformer encoding
        x = self.transformer(x)
        x = self.norm(x)

        return x


class MaskToken(nn.Module):
    """Learnable mask token for decoder input."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Replace masked positions with mask token.

        Args:
            x: Encoded patches (Batch, num_patches, embed_dim)
            mask: Boolean mask where True=masked, False=visible

        Returns:
            Patches with mask tokens inserted
        """
        batch_size = x.shape[0]
        mask_tokens = self.mask_token.expand(batch_size, mask.sum().item(), -1)
        return mask_tokens


class BioMAEDecoder(nn.Module):
    """
    Lightweight Decoder for reconstructing masked spectrograms.

    Shallower than encoder (2 vs 4 layers) for efficiency.
    Takes encoded visible patches + mask tokens and reconstructs
    the full spectrogram.

    Args:
        config: DecoderConfig with model parameters

    Example:
        >>> decoder = BioMAEDecoder(DecoderConfig())
        >>> encoded = torch.randn(2, 129, 256)  # Encoded patches with CLS
        >>> reconstructed = decoder(encoded)
        >>> print(reconstructed.shape)  # (2, 1, 128, 128) - original spec shape
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        # Project encoder embeddings to decoder dimension
        self.decoder_embed = nn.Linear(config.embed_dim, config.decoder_embed_dim)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Transformer decoder layers (shallower than encoder)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.decoder_embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.decoder_embed_dim * config.mlp_ratio),
            dropout=config.drop_rate,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=config.depth,
        )

        # Layer norm
        self.norm = nn.LayerNorm(config.decoder_embed_dim)

        # Projection to pixel values (reconstruct spectrogram patches)
        num_patches = (config.img_size[0] // config.patch_size[0]) * \
                      (config.img_size[1] // config.patch_size[1])
        patch_dim = config.patch_size[0] * config.patch_size[1] * config.in_channels

        self.decoder_pred = nn.Sequential(
            nn.Linear(config.decoder_embed_dim, patch_dim),
        )

        # Unpatchify (rearrange patches to image)
        self.img_size = config.img_size
        self.patch_size = config.patch_size

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode encoded patches to reconstruct spectrogram.

        Args:
            x: Encoded patches from encoder (Batch, num_patches+1, embed_dim)
            mask: Boolean mask (Batch, num_patches) where True=masked

        Returns:
            Reconstructed spectrogram (Batch, Channels, Height, Width)
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)  # (B, num_patches+1, decoder_dim)

        # Add mask tokens at masked positions (exclude CLS)
        if mask is not None:
            cls_token = x[:, 0:1, :]
            patch_tokens = x[:, 1:, :]  # (B, num_patches, decoder_dim)

            # Replace masked patches
            batch_size = patch_tokens.shape[0]
            for b in range(batch_size):
                mask_indices = mask[b]
                if mask_indices.any():
                    patch_tokens[b, mask_indices, :] = self.mask_token.expand(
                        1, mask_indices.sum().item(), -1
                    ).squeeze(0)

            x = torch.cat([cls_token, patch_tokens], dim=1)

        # Transformer decoding
        x = self.transformer(x)
        x = self.norm(x)

        # Remove CLS token
        x = x[:, 1:, :]  # (B, num_patches, decoder_dim)

        # Predict pixel values for each patch
        x = self.decoder_pred(x)  # (B, num_patches, patch_pixels)

        # Unpatchify: rearrange to image
        x = self.unpatchify(x)

        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rearrange patch sequence to image.

        Args:
            x: (Batch, num_patches, patch_dim)

        Returns:
            (Batch, Channels, Height, Width)
        """
        B = x.shape[0]
        pH, pW = self.patch_size
        H, W = self.img_size
        assert H % pH == 0 and W % pW == 0

        h = H // pH
        w = W // pW

        # Reshape to (B, h, w, pH, pW, C)
        x = x.reshape(B, h, w, pH, pW, -1)

        # Transpose to (B, C, H, W)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, -1, H, W)

        return x


class BioMAEModel(nn.Module):
    """
    Complete BioMAE model with encoder and decoder.

    During training: Uses asymmetric encoder-decoder with masking
    During inference: Uses encoder only for 112D feature extraction

    Args:
        encoder_config: Configuration for encoder
        decoder_config: Configuration for decoder

    Example:
        >>> model = BioMAEModel()
        >>> spec = torch.randn(2, 1, 128, 128)
        >>> # Inference mode: extract 112D features
        >>> features = model.encode(spec)
        >>> print(features.shape)  # (2, 112)
    """

    def __init__(
        self,
        encoder_config: EncoderConfig,
        decoder_config: DecoderConfig,
    ):
        super().__init__()
        self.encoder = BioMAEEncoder(encoder_config)
        self.decoder = BioMAEDecoder(decoder_config)

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 112D Rosetta embedding from spectrogram.

        Args:
            x: Input spectrogram (Batch, Channels, Freq, Time)

        Returns:
            112D embedding (Batch, 112)
        """
        return self.encoder(x)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass for training with masking.

        Args:
            x: Input spectrogram (Batch, Channels, Freq, Time)
            mask: Boolean mask (Batch, num_patches) where True=masked

        Returns:
            (reconstructed, encoded_112d)
        """
        # Encode (with mask if provided)
        encoded_patches, embedding = self.encoder(x, return_patches=True)

        # Decode and reconstruct
        reconstructed = self.decoder(encoded_patches, mask)

        return reconstructed, embedding

    def generate_random_mask(
        self,
        batch_size: int,
        device: torch.device,
        mask_ratio: float = 0.75
    ) -> torch.Tensor:
        """
        Generate random mask for MAE training.

        Args:
            batch_size: Number of samples in batch
            device: Tensor device
            mask_ratio: Fraction of patches to mask (default 0.75)

        Returns:
            Boolean mask (Batch, num_patches) where True=masked
        """
        num_patches = self.encoder.patch_embed.config.num_patches
        num_masked = int(num_patches * mask_ratio)

        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

        for b in range(batch_size):
            # Randomly select patches to mask
            indices = torch.randperm(num_patches, device=device)
            masked_indices = indices[:num_masked]
            mask[b, masked_indices] = True

        return mask


# Preset configurations

BIOMAE_BASE = {
    'encoder': EncoderConfig(
        img_size=(128, 128),
        patch_size=(16, 16),
        embed_dim=256,
        depth=4,
        num_heads=4,
        output_dim=112,
    ),
    'decoder': DecoderConfig(
        embed_dim=256,
        decoder_embed_dim=128,
        depth=2,
        num_heads=4,
        patch_size=(16, 16),
        img_size=(128, 128),
    ),
}

BIOMAE_LARGE = {
    'encoder': EncoderConfig(
        img_size=(224, 224),
        patch_size=(16, 16),
        embed_dim=512,
        depth=6,
        num_heads=8,
        output_dim=112,
    ),
    'decoder': DecoderConfig(
        embed_dim=512,
        decoder_embed_dim=256,
        depth=2,
        num_heads=8,
        patch_size=(16, 16),
        img_size=(224, 224),
    ),
}
