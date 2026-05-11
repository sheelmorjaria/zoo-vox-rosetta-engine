# Model Training Guide: β-VAE and VQ-VAE

This guide explains how to train the dual-stream neural network models for the Egyptian Fruit Bat vocalization analysis system.

## Overview

The dual-stream architecture requires two trained models:

1. **β-VAE (Stream 1)**: Encodes ~30D continuous affect features → 16D disentangled latent space
2. **VQ-VAE (Stream 2)**: Encodes ~50D syntactic features → 64 discrete tokens

## Prerequisites

```bash
pip install torch numpy scipy scikit-learn
```

For GPU acceleration (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Training Pipeline

### Step 1: Prepare Training Data

First, extract 112D RosettaFeatures from your audio dataset:

```python
from analysis.feature_pipeline import DEFAULT_ROSETTA_EXTRACTOR
import numpy as np
from pathlib import Path

# Configuration
AUDIO_DIR = Path("data/bat_audio/")
FEATURES_OUTPUT = Path("data/bat_features_112d.npy")

# Extract features from all audio files
extractor = DEFAULT_ROSETTA_EXTRACTOR
all_features = []

for audio_file in AUDIO_DIR.glob("*.wav"):
    from scipy.io import wavfile
    sample_rate, audio = wavfile.read(audio_file)
    
    # Convert to float and normalize
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Extract 112D features
    features_112d = extractor.extract(audio, sample_rate)
    all_features.append(features_112d)

# Save features
np.save(FEATURES_OUTPUT, np.array(all_features))
print(f"Extracted {len(all_features)} feature vectors")
```

### Step 2: Train β-VAE (16D Affect Encoder)

```python
from cognitive_intelligence.train_beta_vae import train_beta_vae, BetaVAETrainingConfig
import numpy as np

# Load 112D features
features_112d = np.load("data/bat_features_112d.npy")

# Configure training
config = BetaVAETrainingConfig(
    input_dim=30,              # Affective features subset
    latent_dim=16,             # 16D disentangled latent space
    hidden_dim=128,
    beta=2.0,                  # For disentanglement
    batch_size=256,
    learning_rate=1e-3,
    num_epochs=200,
    kl_annealing=True,         # Stable KL training
    target_recon_loss=0.1,     # Target reconstruction loss
    checkpoint_dir="models/beta_vae_bat",
)

# Train
model, trainer = train_beta_vae(features_112d, config)

# Results
print(f"Best loss: {trainer.best_loss:.4f}")
print(f"Reconstruction loss: {trainer.recon_losses[-1]:.4f}")
print(f"KL loss: {trainer.kl_losses[-1]:.4f}")
```

**Target Metrics:**
- Reconstruction loss < 0.1
- KL divergence stable (not exploding)
- Disentangled dimensions (inspect latent space)

### Step 3: Train VQ-VAE (64-Token Syntactic Encoder)

```python
from cognitive_intelligence.train_vqvae import train_vqvae, VQVAETrainingConfig
import numpy as np

# Load 112D features (same data)
features_112d = np.load("data/bat_features_112d.npy")

# Configure training
config = VQVAETrainingConfig(
    input_dim=50,              # Syntactic features subset
    codebook_size=64,          # 64 discrete tokens
    codebook_dim=32,           # Codebook vector dimension
    hidden_dim=128,
    commitment_cost=0.25,       # VQ commitment weight
    decay=0.99,                # EMA decay
    batch_size=256,
    learning_rate=1e-3,
    num_epochs=200,
    target_commitment_loss=0.05,
    target_utilization=80.0,   # >80% codebook usage
    checkpoint_dir="models/vqvae_bat",
)

# Train
model, trainer = train_vqvae(features_112d, config)

# Results
print(f"Best loss: {trainer.best_loss:.4f}")
print(f"Commitment loss: {trainer.commit_losses[-1]:.4f}")
print(f"Codebook utilization: {trainer.utilization_history[-1]:.1f}%")
```

**Target Metrics:**
- Commitment loss < 0.05
- Codebook utilization > 80%
- Perplexity (diversity) > 10

### Step 4: Export to ONNX (Optional)

For deployment in Rust via TensorRT:

```python
import torch
import torch.onnx

# Load trained models
vae_model = torch.load("models/beta_vae_bat/best_model.pt")
vqvae_model = torch.load("models/vqvae_bat/best_model.pt")

# Export β-VAE encoder
dummy_input = torch.randn(1, 30)  # Batch size 1, 30D input
torch.onnx.export(
    vae_model.encoder,
    dummy_input,
    "models/affect_encoder.onnx",
    input_names=["affective_features"],
    output_names=["affect_vector"],
    dynamic_axes={
        "affective_features": {0: "batch_size"},
        "affect_vector": {0: "batch_size"},
    },
)

# Export VQ-VAE encoder
dummy_input = torch.randn(1, 50)  # Batch size 1, 50D input
torch.onnx.export(
    vqvae_model.encoder,
    dummy_input,
    "models/syntactic_encoder.onnx",
    input_names=["syntactic_features"],
    output_names=["token_logits"],
    dynamic_axes={
        "syntactic_features": {0: "batch_size"},
        "token_logits": {0: "batch_size"},
    },
)
```

## Monitoring Training

### TensorBoard Logging (Optional)

Add to your training script:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/beta_vae_experiment")

# During training
writer.add_scalar("Loss/train", train_loss, epoch)
writer.add_scalar("Loss/reconstruction", train_recon, epoch)
writer.add_scalar("Loss/KL", train_kl, epoch)
writer.add_scalar("Metrics/Latent/std", latent_std, epoch)
```

### Key Metrics to Monitor

**β-VAE:**
| Metric | Target | Description |
|--------|--------|-------------|
| Reconstruction Loss | < 0.1 | MSE between input and output |
| KL Divergence | Stable | Should not explode |
| Latent Std | ~1.0 per dim | Disentanglement indicator |

**VQ-VAE:**
| Metric | Target | Description |
|--------|--------|-------------|
| Commitment Loss | < 0.05 | Encoder-codebook alignment |
| Codebook Utilization | > 80% | % of tokens used |
| Perplexity | > 10 | Diversity of token usage |

## Troubleshooting

### Issue: KL Divergence Explodes

**Solution:** Enable KL annealing (already in default config):

```python
config = BetaVAETrainingConfig(
    kl_annealing=True,
    kl_anneal_cycles=5,
)
```

### Issue: Codebook Collapse (<50% utilization)

**Solution:** Lower commitment cost or increase revival threshold:

```python
config = VQVAETrainingConfig(
    commitment_cost=0.1,   # Lower from 0.25
    revival_threshold=0.05,  # More aggressive revival
)
```

### Issue: Reconstruction Loss Plateaus

**Solution:** Increase model capacity or learning rate:

```python
config = BetaVAETrainingConfig(
    hidden_dim=256,  # Increase from 128
    learning_rate=3e-3,  # Increase from 1e-3
)
```

## Using Trained Models

After training, use the models with the analysis frameworks:

```python
from analysis import (
    FeaturePipeline,
    GradedContinuumAnalyzer,
    AddressingClassifier,
)
import torch

# Load trained models
vae_model = torch.load("models/beta_vae_bat/best_model.pt")
vqvae_model = torch.load("models/vqvae_bat/best_model.pt")

# Create pipeline with trained models
pipeline = FeaturePipeline(
    vae_encoder=TrainedAffectiveVAEEncoder(vae_model),
    vqvae_encoder=TrainedSyntacticVQVAEEncoder(vqvae_model),
)

# Extract features from new audio
features = pipeline.process_audio_file("new_bat_call.wav", "seg_001")

# Use with analysis frameworks
analyzer = GradedContinuumAnalyzer()
dispute = analyzer.analyze_dispute(
    affect_trajectory=features.affect_vector_16d[np.newaxis, :],
    timestamps_ms=np.array([0]),
    participants=[1, 2],
    dispute_id="new_dispute",
)
```

## Dataset Size Recommendations

| Dataset Size | Training Time | Expected Quality |
|--------------|---------------|------------------|
| 1,000 samples | ~30 min (GPU) | Baseline |
| 5,000 samples | ~2 hours (GPU) | Good |
| 10,000+ samples | ~4 hours (GPU) | Excellent |

The Egyptian Fruit Bat dataset has 516 phrases. Consider:
- Data augmentation (time stretching, pitch shifting)
- Combining with related species data
- Pre-training on larger dataset, fine-tuning on bats

## Next Steps

After training both models:

1. **Validate** reconstruction quality on test set
2. **Export** to ONNX for Rust deployment
3. **Integrate** with analysis frameworks
4. **Deploy** to field system

See `docs/implementation_roadmap.md` for deployment phases.
