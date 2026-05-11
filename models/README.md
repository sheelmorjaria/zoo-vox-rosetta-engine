# Models Directory - Dual-Stream Architecture

This directory contains the trained model files for the dual-stream acoustic-syntactic architecture.

## Directory Structure

```
models/
├── README.md                           # This file
├── synthesis_manifest.json             # Model metadata and paths
├── dual_stream/                        # Dual-stream model checkpoints
│   ├── affective_vae.pt               # β-VAE for Stream 1 (16D latent)
│   ├── syntactic_vqvae.pt             # VQ-VAE for Stream 2 (64 tokens)
│   └── syntax_graph.json              # Laplace-smoothed transition matrix
├── onnx/                               # ONNX exports for Rust inference
│   ├── affective_encoder.onnx          # 54D → 16D encoder
│   ├── syntactic_encoder.onnx          # 44D → token encoder
│   └── synthesis_decoder.onnx          # DDSP decoder with FiLM
└── legacy/                             # Pre-dual-stream models (archived)
    ├── ddsp_decoder.pt                 # Original 112D → 65D MLP
    └── bgmm_centroids.npy              # 45-cluster BGMM centroids
```

## Model Files

### Affective VAE (Stream 1)
- **File**: `dual_stream/affective_vae.pt`
- **Architecture**: 54D input → 128D hidden → 16D latent (β=2.0)
- **Purpose**: Continuous affect/prosodic encoding
- **Training**: Disentangled latent space with β-VAE loss

### Syntactic VQ-VAE (Stream 2)
- **File**: `dual_stream/syntactic_vqvae.pt`
- **Architecture**: 44D input → 32D codebook → 64 discrete tokens
- **Purpose**: Discrete syntactic tokenization
- **Training**: EMA codebook updates (decay=0.99) to prevent collapse

### Syntax Graph
- **File**: `dual_stream/syntax_graph.json`
- **Contents**: 64×64 transition matrix with Laplace smoothing (α=0.01)
- **Purpose**: Validate syntactic sequences and predict next tokens

### ONNX Exports
- **Affective Encoder**: `onnx/affective_encoder.onnx` - Rust-side 16D encoding
- **Syntactic Encoder**: `onnx/syntactic_encoder.onnx` - Rust-side token encoding
- **Synthesis Decoder**: `onnx/synthesis_decoder.onnx` - FiLM-based DDSP synthesis

## Synthesis Manifest

The `synthesis_manifest.json` file contains:
- Vocabulary size (64 tokens)
- Affect latent dimension (16)
- Model paths for all components
- Feature dimensions for each stream

## Training Status

Current models are **untrained placeholders**. To train:

1. **Affective VAE**:
   ```bash
   python cognitive_intelligence/train_affective_vae.py --data data/cached_features.npy
   ```

2. **Syntactic VQ-VAE**:
   ```bash
   python cognitive_intelligence/train_syntactic_vqvae.py --data data/cached_features.npy
   ```

3. **Syntax Graph**:
   ```bash
   python cognitive_intelligence/build_syntax_graph.py --corpus data/tokenized_corpus.json
   ```

4. **Export to ONNX**:
   ```bash
   python cognitive_intelligence/export_to_onnx.py
   ```

## Model Metrics (After Training)

### Target Metrics
- **β-VAE**: Reconstruction loss < 0.1, KL divergence stable
- **VQ-VAE**: Codebook utilization > 80%, commitment loss < 0.05
- **Syntax Graph**: Covers >95% of biological bigrams in corpus

## Loading Models

```python
# Load Affective VAE
from cognitive_intelligence.affective_vae import AffectVAECheckpoint

vae = AffectVAECheckpoint.load_model_only("models/dual_stream/affective_vae.pt")

# Load Syntactic VQ-VAE
from cognitive_intelligence.syntactic_vqvae import VQVAECheckpoint

vqvae = VQVAECheckpoint.load_model_only("models/dual_stream/syntactic_vqvae.pt")

# Load Syntax Graph
from cognitive_intelligence.syntax_graph import SyntaxGraph

graph = SyntaxGraph.load_json("models/dual_stream/syntax_graph.json")

# Create Dual-Stream Agent
from realtime.interaction_agent import DualStreamInteractionAgent, DualStreamAgentConfig

config = DualStreamAgentConfig(
    affective_vae_path="models/dual_stream/affective_vae.pt",
    syntactic_vqvae_path="models/dual_stream/syntactic_vqvae.pt",
    syntax_graph_path="models/dual_stream/syntax_graph.json",
)
agent = DualStreamInteractionAgent(config)
```

## Version History

- **v2.0.0** (2026-05-09): Dual-stream architecture with FiLM synthesis
- **v1.6.0** (2025-11-15): Original single-stream DDSP decoder
