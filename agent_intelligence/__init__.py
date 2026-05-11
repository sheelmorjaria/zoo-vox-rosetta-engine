#!/usr/bin/env python3
"""
Agent Intelligence: Probabilistic Closed-Loop Components

This package implements the statistical upgrades to Stage 4 Closed-Loop Agent:
- Mahalanobis OOD detection (replaces L2 distance)
- Lightweight Transformer syntax engine (replaces rigid bigram automaton)
- Probabilistic sampling with top-p (nucleus) sampling

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

__version__ = "3.0.0"

__all__ = [
    # Mahalanobis OOD Detection
    "OODCalibrator",
    "OODCalibrationConfig",
    "OODStatistics",
    "MahalanobisOOD",
    # Syntax Transformer
    "SyntaxTransformer",
    "SyntaxTransformerTrainer",
    "TransformerConfig",
    # Syntax Sampling
    "SyntaxSampler",
    "SamplingConfig",
    "SamplingMode",
    "SamplingResult",
    # Interaction Agent v3.0
    "InteractionAgentV3",
    "ResponseMode",
    "AgentConfig",
    "CognitiveState",
    "create_agent_v3",
    # Preset Configurations
    "CONSERVATIVE_AGENT_CONFIG",
    "BALANCED_AGENT_CONFIG",
    "CREATIVE_AGENT_CONFIG",
    "CONSERVATIVE_SAMPLING",
    "BALANCED_SAMPLING",
    "CREATIVE_SAMPLING",
]

# Import for availability
from .mahalanobis_ood import (
    OODCalibrator,
    OODCalibrationConfig,
    OODStatistics,
    MahalanobisOOD,
    STANDARD_OOD_CONFIG,
    STRICT_OOD_CONFIG,
)
from .syntax_transformer import (
    SyntaxTransformer,
    SyntaxTransformerTrainer,
    TransformerConfig,
    MINIMAL_TRANSFORMER_CONFIG,
    STANDARD_TRANSFORMER_CONFIG,
)
from .syntax_sampler import (
    SyntaxSampler,
    SamplingConfig,
    SamplingMode,
    SamplingResult,
    CONSERVATIVE_SAMPLING,
    BALANCED_SAMPLING,
    CREATIVE_SAMPLING,
)
from .interaction_agent_v3 import (
    InteractionAgentV3,
    ResponseMode,
    AgentConfig,
    CognitiveState,
    create_agent_v3,
    CONSERVATIVE_AGENT_CONFIG,
    BALANCED_AGENT_CONFIG,
    CREATIVE_AGENT_CONFIG,
)
