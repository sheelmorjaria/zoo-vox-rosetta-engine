# Dual-Stream Model Files

This directory contains the trained models for the dual-stream acoustic-syntactic architecture.

## Files

### affective_vae.pt
β-VAE for continuous affect encoding (Stream 1)
- Input: 54D affective features extracted from 112D RosettaFeatures
- Output: 16D disentangled latent space (β = 2.0)
- Status: Placeholder (requires training)

### syntactic_vqvae.pt
VQ-VAE with EMA for discrete syntactic tokenization (Stream 2)
- Input: 44D syntactic features extracted from 112D RosettaFeatures
- Output: 64 discrete tokens with 32D codebook
- Status: Placeholder (requires training)

### syntax_graph.json
Laplace-smoothed transition matrix for syntax validation
- Vocabulary: 64 tokens
- Smoothing parameter: α = 0.01
- Status: Placeholder (requires corpus tokenization)

## Training

See `cognitive_intelligence/train_affective_vae.py` and `cognitive_intelligence/train_syntactic_vqwae.py` for training scripts.

## Building Syntax Graph

After training the VQ-VAE, build the syntax graph:

```bash
python -m cognitive_intelligence.build_syntax_graph \
    --vqvae models/dual_stream/syntactic_vqvae.pt \
    --data data/cached_features.npy \
    --data-type npy \
    --output models/dual_stream/syntax_graph.json
```

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
