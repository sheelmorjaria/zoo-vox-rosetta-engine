#!/usr/bin/env python3
"""
Self-Supervised Predictive Boundary Detection

Replaces heuristic-based Neural Boundary Detector (NBD) with a Contrastive
Predictive Coding (CPC) approach. Semantic boundaries are detected where
prediction errors spike, indicating acoustic state transitions.

Components:
- CPCEncoder: 1D Conv encoder for audio → latent representation
- AutoregressiveMamba: Temporal context modeling with O(1) streaming
- CPCTrainer: Self-supervised training with InfoNCE loss
- PredictiveBoundaryDetector: Adaptive boundary extraction

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from boundary_detection.cpc_encoder import CPCEncoder, EncoderConfig, create_encoder
from boundary_detection.cpc_autoregressive import (
    AutoregressiveMamba,
    TCNAutoregressive,
    StreamingContextBuffer,
    create_autoregressive,
)
from boundary_detection.cpc_trainer import (
    CPCTrainer,
    CPCModel,
    AudioSequenceDataset,
    TrainingConfig,
    create_cpc_model,
)
from boundary_detection.predictive_boundary import (
    BoundaryType,
    PredictiveBoundaryDetector,
    PredictionResult,
    BoundaryDetectorConfig,
    AdaptiveDebounceStrategy,
    create_boundary_detector,
)

__all__ = [
    # Encoder
    "CPCEncoder",
    "EncoderConfig",
    "create_encoder",
    # Autoregressive models
    "AutoregressiveMamba",
    "TCNAutoregressive",
    "StreamingContextBuffer",
    "create_autoregressive",
    # Training
    "CPCTrainer",
    "CPCModel",
    "AudioSequenceDataset",
    "TrainingConfig",
    "create_cpc_model",
    # Boundary detection
    "PredictiveBoundaryDetector",
    "BoundaryType",
    "PredictionResult",
    "BoundaryDetectorConfig",
    "AdaptiveDebounceStrategy",
    "create_boundary_detector",
]
