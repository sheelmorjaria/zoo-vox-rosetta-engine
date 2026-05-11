#!/usr/bin/env python3
"""
BioMAE: Bioacoustic Masked Autoencoder for Learned Feature Extraction

Replaces hand-crafted 112D Rosetta Features with self-supervised learned embeddings.
Key advantages:
- Log-linear spectrograms preserve ultrasonic physics (no Mel-scale bias)
- Single forward pass via TensorRT (<5ms) vs 112 sequential algorithms
- Self-supervised learning captures acoustic structure natively

Components:
- UltrasonicSpectrogram: Log-linear spectrogram computation
- PatchEmbedding: ViT-style spectrogram tokenization
- BioMAEEncoder: Lightweight transformer encoder (4 layers, 256 dim)
- BioMAEDecoder: Lightweight decoder for MAE reconstruction
- BioMAETrainer: Self-supervised training with 75% masking

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from feature_extraction.bio_spectrogram import (
    UltrasonicSpectrogram,
    SpectrogramConfig,
    create_spectrogram,
)
from feature_extraction.patch_embed import (
    PatchEmbedding,
    PatchEmbedConfig,
    create_patch_embedding,
)
from feature_extraction.biomae import (
    BioMAEEncoder,
    BioMAEDecoder,
    BioMAEModel,
    EncoderConfig,
    DecoderConfig,
    create_biomae_model,
)

__all__ = [
    # Spectrogram
    "UltrasonicSpectrogram",
    "SpectrogramConfig",
    "create_spectrogram",
    # Patch Embedding
    "PatchEmbedding",
    "PatchEmbedConfig",
    "create_patch_embedding",
    # BioMAE
    "BioMAEEncoder",
    "BioMAEDecoder",
    "BioMAEModel",
    "EncoderConfig",
    "DecoderConfig",
    "create_biomae_model",
]
