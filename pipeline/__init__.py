#!/usr/bin/env python3
"""
Acoustic-First Pipeline Package

Implements the foundational paradigms:
- Acoustic-First Paradigm: Raw acoustic physics as primary substrate
- Intra-Call Paradigm: Micro-modulations within vocalization boundaries

Pipeline Stages:
1. CPC: Self-Supervised Predictive Boundary Detection
2. BioMAE: Feature Extraction via Masked Autoencoder
3. Dual-Stream: Affective (pUMAP+β-VAE) + Syntactic (VQ-VAE)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

from .acoustic_first_pipeline import (
    AcousticFirstPipeline,
    PipelineConfig,
    PipelineOutput,
    create_pipeline,
    BAT_PIPELINE,
    BIRD_PIPELINE,
    MINIMAL_PIPELINE,
)

__all__ = [
    'AcousticFirstPipeline',
    'PipelineConfig',
    'PipelineOutput',
    'create_pipeline',
    'BAT_PIPELINE',
    'BIRD_PIPELINE',
    'MINIMAL_PIPELINE',
]
