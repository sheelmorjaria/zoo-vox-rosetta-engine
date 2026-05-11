#!/usr/bin/env python3
"""
Corpus Analysis: Continuous Manifold Mining Pipeline

This package implements Stage 3 of the Animal Language Processing pipeline,
replacing PCA+BGMM with UMAP+VAE for continuous vocal manifold modeling.

Modules:
- parametric_umap: Non-linear dimensionality reduction (112D → 30D)
- vocal_vae: Continuous latent space modeling (30D → 16D)
- medoid_extractor: Quality-weighted medoid extraction with HDBSCAN
- manifest_builder: Generate continuous_manifold_manifest.json

Author: Zoo Vox Research Team
License: CC BY-ND 4.0 International
"""

from __future__ import annotations

__version__ = "1.1.0"

__all__ = [
    "ParametricUMAPTrainer",
    "VocalManifoldReducer",
    "VocalVAE",
    "VocalVAETrainer",
    "MedoidExtractor",
    "ManifestBuilder",
]

# Lazy imports to avoid heavy dependencies
def _lazy_import():
    from corpus_analysis.parametric_umap import ParametricUMAPTrainer, VocalManifoldReducer
    from corpus_analysis.vocal_vae import VocalVAE, VocalVAETrainer
    from corpus_analysis.medoid_extractor import MedoidExtractor
    from corpus_analysis.manifest_builder import ManifestBuilder
    return {
        "ParametricUMAPTrainer": ParametricUMAPTrainer,
        "VocalManifoldReducer": VocalManifoldReducer,
        "VocalVAE": VocalVAE,
        "VocalVAETrainer": VocalVAETrainer,
        "MedoidExtractor": MedoidExtractor,
        "ManifestBuilder": ManifestBuilder,
    }
