# Technical Design Document (TDD): Continuous Manifold Mining & Medoid Extraction

**Document Version:** 1.1 (Refined)
**Component:** Animal Language Processing (ALP) Pipeline - Stage 3 (Corpus Analysis & Syntactic Mining)
**Integration:** Dual-Stream Architecture (feeds into Stream 1: Continuous Affective/Prosodic)
**Author:** Zoo Vox Research Team
**Status:** Refined Architecture - Ready for Implementation

---

## Change Log (v1.0 → v1.1)

| Issue | v1.0 | v1.1 (Refined) |
|-------|------|----------------|
| Parametric UMAP API | Incorrect import shown | Uses `cuml.UMAP` (RAPIDS) or PyTorch encoder |
| VAE logvar handling | `log(1e-6 + fc_var(h))` | Direct `fc_var` outputs logvar (standard) |
| Integration context | Standalone document | Explicitly maps to Dual-Stream architecture |
| BioMAE compatibility | Unspecified | Works with BioMAE 112D output (v1.7.0+) |
| ONNX exportability | Not addressed | All components exportable to TensorRT |

---

## 1. Introduction & Motivation

### 1.1 The Problem: Linear Reductions, Discretization, and Averaging

The current Stage 3 methodology processes the 112D Rosetta features through:

```
Subsampling → PCA → Bayesian Gaussian Mixture Models (BGMM) → Pruning (<1%) → Centroid Extraction
```

This pipeline suffers from three critical flaws:

1. **PCA Destroys Non-Linear Gradients:** Animal vocal manifolds are highly non-linear. PCA assumes linear variance; reducing 112D to 30D via PCA mathematically flattens the subtle intra-call graded continua the system was built to capture.

2. **The Long-Tail Fallacy:** Pruning clusters representing <1% of the dataset is statistically dangerous. In ethology, rare calls (e.g., specific predator alarms) carry the highest semantic urgency. Deleting them biases the system toward mundane vocalizations.

3. **The Centroid Fallacy:** Picking the mathematical center of a cluster as the "archetype" often results in an acoustically blurry or biologically impossible "average" sound, leading to poor synthesis quality.

### 1.2 The Solution: Continuous Manifolds and Pristine Medoids

We propose replacing the rigid, discrete pipeline with a continuous topology:

1. **Replace PCA with Parametric UMAP:** Preserve non-linear local gradients crucial for graded vocalizations.

2. **Replace BGMM with a Variational Autoencoder (VAE):** Create a continuous latent space. Instead of forcing calls into discrete clusters, the VAE allows the generative engine to smoothly interpolate between archetypes, preserving graded continuity.

3. **Replace Centroids with Quality-Weighted Medoids:** Select real audio segments that minimize distance to neighboring points *and* maximize Signal-to-Noise Ratio (SNR), ensuring the exemplar is biologically pristine.

### 1.3 Integration with Dual-Stream Architecture

This TDD implements the **offline "Teacher" pipeline** for **Stream 1 (Continuous Affective/Prosodic)** of the Dual-Stream Architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Dual-Stream Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [112D BioMAE Embeddings]                                            │
│           │                                                          │
│           ├─────────────────────────────────────────────────────┐  │
│           │                                                     │  │
│           ▼                                                     │  │
│  ┌─────────────────────────┐                                    │  │
│  │  THIS TDD:              │                                    │  │
│  │  Continuous Manifold    │                                    │  │
│  │  Mining (Offline)       │                                    │  │
│  │  - UMAP 112D→30D        │                                    │  │
│  │  - VAE 30D→16D          │                                    │  │
│  │  - Medoid Extraction    │                                    │  │
│  └────────────┬────────────┘                                    │  │
│               │                                                 │  │
│               ▼                                                 │  │
│       [continuous_manifold_manifest.json]                         │  │
│               │                                                 │  │
│               ▼                                                 │  │
│  ┌─────────────────────────┐                                    │  │
│  │  DualStreamAgent        │                                    │  │
│  │  (Runtime Navigation)   │                                    │  │
│  └────────────┬────────────┘                                    │  │
│               │                                                 │  │
│               ▼                                                 │  │
│       [DDSP Synthesis Engine]                                    │  │
│       (FiLM-modulated by 16D manifold coords)                     │  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. System Architecture Overview

The new Stage 3 operates as an offline "Teacher" pipeline that generates a continuous map of the species' vocal repertoire.

```text
[112D BioMAE Embeddings (from Stage 2)]
                |
                v
    [1. Parametric UMAP Reduction] (112D -> 30D, Non-linear)
                |
                v
    [2. VAE Latent Space Modeling] (30D -> 16D Continuous Manifold)
                |
                v
    [3. HDBSCAN Zoning] (Identify dense regions without forcing discrete clusters)
                |
                v
    [4. Quality-Weighted Medoid Extraction] (Find pristine real audio exemplars)
                |
                v
    [continuous_manifold_manifest.json]
    (Maps Latent Coordinates -> Real Audio -> SNR/Medoid Scores)
```

---

## 3. Module 1: Non-Linear Dimensionality Reduction (Parametric UMAP)

### 3.1 Objective

Reduce the 112D BioMAE embeddings to a 30D space while preserving the non-linear local gradients (the "graded continuum") of the vocalizations.

### 3.2 Implementation Options

**Option A: RAPIDS cuml.UMAP (GPU-accelerated, recommended for Jetson)**
```python
# corpus_analysis/parametric_umap.py
import cuml
from cuml.manifold import UMAP
import torch
import torch.nn as nn

class VocalManifoldReducer(nn.Module):
    """
    Parametric UMAP: 112D BioMAE Embeddings -> 30D Non-linear space.
    Preserves the graded continuum that PCA destroys.

    Uses cuml.UMAP (RAPIDS) for GPU acceleration on Jetson devices.
    """
    def __init__(self, input_dim=112, output_dim=30):
        super().__init__()
        # Learnable encoder for parametric inference
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)


class ParametricUMAPTrainer:
    """
    Trains a parametric UMAP model using cuml for initialization.
    """
    def __init__(
        self,
        input_dim=112,
        output_dim=30,
        n_neighbors=15,
        min_dist=0.1,
        metric='euclidean'
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric

        # Initialize encoder
        self.encoder = VocalManifoldReducer(input_dim, output_dim)

        # Standard UMAP for target initialization
        self.umap_init = cuml.UMAP(
            n_components=output_dim,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric
        )

    def train(self, data_112d, epochs=100, lr=1e-3, batch_size=256):
        """
        Train parametric UMAP using reconstruction loss.

        Args:
            data_112d: (N, 112) BioMAE embeddings
            epochs: Training epochs
            lr: Learning rate
            batch_size: Batch size

        Returns:
            embedding_30d: (N, 30) reduced embeddings
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        import torch.optim as optim

        # Step 1: Initialize with standard UMAP
        print("Initializing UMAP embedding...")
        target_30d = self.umap_init.fit_transform(data_112d)

        # Step 2: Create PyTorch dataset
        dataset = TensorDataset(
            torch.FloatTensor(data_112d),
            torch.FloatTensor(target_30d)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Step 3: Train encoder to match UMAP targets
        optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.encoder.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                pred = self.encoder(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.6f}")

        # Step 4: Generate final embeddings
        with torch.no_grad():
            embedding_30d = self.encoder(torch.FloatTensor(data_112d)).cpu().numpy()

        return embedding_30d
```

**Option B: Pure PyTorch (CPU-compatible, no RAPIDS dependency)**
```python
class VocalManifoldReducer(nn.Module):
    """
    CPU-compatible parametric UMAP using pure PyTorch.
    """
    def __init__(self, input_dim=112, output_dim=30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)
```

### 3.3 ONNX Export

```python
def export_umap_to_onnx(encoder, output_path="models/umap_encoder.onnx"):
    """
    Export trained UMAP encoder to ONNX for TensorRT deployment.
    """
    import torch.onnx

    dummy_input = torch.randn(1, 112)
    torch.onnx.export(
        encoder,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        input_names=['bio_mae_embedding'],
        output_names=['umap_coords'],
        dynamic_axes={
            'bio_mae_embedding': {0: 'batch_size'},
            'umap_coords': {0: 'batch_size'}
        }
    )
    print(f"Exported UMAP encoder to {output_path}")
```

---

## 4. Module 2: Continuous Latent Space Modeling (VAE)

### 4.1 Objective

Learn a probabilistic, continuous manifold. Replace discrete cluster IDs with continuous coordinates, allowing the downstream synthesizer to smoothly interpolate between calls.

### 4.2 Implementation (REFINED - Fixed logvar handling)

```python
# corpus_analysis/vocal_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VocalVAE(nn.Module):
    """
    VAE: 30D UMAP space -> 16D Continuous Latent Space.
    Replaces discrete BGMM clusters.

    Key refinement: fc_var directly outputs logvar (standard VAE formulation).
    """
    def __init__(
        self,
        input_dim=30,
        latent_dim=16,
        hidden_dim=128,
        beta=1.0  # For β-VAE: increase for disentanglement
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder: q(z|x)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Direct outputs for mu and logvar (standard VAE)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: p(x|z)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        """
        Encode input to latent distribution parameters.

        Returns:
            mu: (B, latent_dim) mean of latent Gaussian
            logvar: (B, latent_dim) log of variance (unconstrained)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where sigma = exp(0.5 * logvar)

        CHANGED FROM v1.0: No log(1e-6 + x) wrapper.
        The fc_logvar layer outputs logvar directly.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent sample to reconstruction."""
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass: reconstruction, mu, logvar.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(self, recon, x, mu, logvar):
        """
        VAE loss = Reconstruction Loss + KL Divergence

        Args:
            recon: Reconstructed input
            x: Original input
            mu: Latent mean
            logvar: Latent log variance

        Returns:
            total_loss, recon_loss, kl_loss
        """
        # Reconstruction loss (MSE for continuous data)
        recon_loss = F.mse_loss(recon, x, reduction='sum')

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # β-VAE: Weight KL by beta factor
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss


class VocalVAETrainer:
    """
    Training loop for Vocal VAE.
    """
    def __init__(
        self,
        input_dim=30,
        latent_dim=16,
        beta=1.0,
        learning_rate=1e-3,
        device='cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = VocalVAE(input_dim, latent_dim, beta=beta).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(
        self,
        data_30d,
        epochs=200,
        batch_size=128,
        early_stopping_patience=20
    ):
        """
        Train VAE on UMAP-reduced data.

        Returns:
            training_history: Dict with loss curves
        """
        from torch.utils.data import TensorDataset, DataLoader

        dataset = TensorDataset(torch.FloatTensor(data_30d))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_total = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0

            for (batch,) in dataloader:
                batch = batch.to(self.device)

                # Forward pass
                recon, mu, logvar = self.model(batch)
                total_loss, recon_loss, kl_loss = self.model.loss_function(
                    recon, batch, mu, logvar
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_total += total_loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()

            # Record history
            avg_total = epoch_total / len(data_30d)
            avg_recon = epoch_recon / len(data_30d)
            avg_kl = epoch_kl / len(data_30d)

            history['total_loss'].append(avg_total)
            history['recon_loss'].append(avg_recon)
            history['kl_loss'].append(avg_kl)

            # Logging
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: "
                      f"Total={avg_total:.4f}, "
                      f"Recon={avg_recon:.4f}, "
                      f"KL={avg_kl:.4f}")

            # Early stopping
            if avg_total < best_loss:
                best_loss = avg_total
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return history

    def encode(self, data_30d):
        """Encode data to 16D latent space."""
        self.model.eval()
        with torch.no_grad():
            mu, logvar = self.model.encode(torch.FloatTensor(data_30d).to(self.device))
            # Return mean (deterministic) for downstream use
            return mu.cpu().numpy()
```

### 4.3 ONNX Export

```python
def export_vae_to_onnx(
    vae_model,
    output_dir="models/vae"
):
    """
    Export VAE encoder and decoder to ONNX separately.

    The encoder is used at runtime to map 30D UMAP coords → 16D manifold.
    The decoder can be used for generation/manifold exploration.
    """
    import torch.onnx

    # Export encoder
    dummy_input_30d = torch.randn(1, 30)
    torch.onnx.export(
        vae_model.encoder,
        dummy_input_30d,
        f"{output_dir}/vae_encoder.onnx",
        export_params=True,
        opset_version=17,
        input_names=['umap_coords'],
        output_names=['latent_coords'],
        dynamic_axes={
            'umap_coords': {0: 'batch_size'},
            'latent_coords': {0: 'batch_size'}
        }
    )

    # Export decoder
    dummy_input_16d = torch.randn(1, 16)
    torch.onnx.export(
        vae_model.decoder,
        dummy_input_16d,
        f"{output_dir}/vae_decoder.onnx",
        export_params=True,
        opset_version=17,
        input_names=['latent_coords'],
        output_names=['umap_coords'],
        dynamic_axes={
            'latent_coords': {0: 'batch_size'},
            'umap_coords': {0: 'batch_size'}
        }
    )

    print(f"Exported VAE to {output_dir}/")
```

---

## 5. Module 3: Quality-Weighted Medoid Extraction

### 5.1 Objective

Identify dense regions in the VAE latent space, and select the *most biologically pristine* real audio segment as the exemplar, replacing the blurry mathematical centroid.

### 5.2 HDBSCAN Zoning (No Pruning)

Instead of BGMM (which requires specifying cluster count and leads to pruning rare clusters), use HDBSCAN. HDBSCAN finds dense regions organically and labels rare, isolated calls as "noise" rather than deleting them. We will explicitly *preserve* noise points as high-value "rare calls."

### 5.3 Medoid + SNR Selection Algorithm (unchanged from v1.0)

```python
# corpus_analysis/medoid_extractor.py
import numpy as np
from sklearn.metrics import pairwise_distances
import hdbscan

class MedoidExtractor:
    """
    Replaces Centroids with Quality-Weighted Medoids.
    Ensures exemplars are biologically possible and acoustically pristine.
    """
    def __init__(self, min_cluster_size=50):
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'  # Excess of Mass
        )

    def extract_exemplars(self, latent_coords_16d, original_audio_snrs):
        """
        Extract medoid-based exemplars from VAE latent space.

        Args:
            latent_coords_16d: (N, 16) VAE latent coordinates
            original_audio_snrs: (N,) Signal-to-Noise Ratio for each audio file

        Returns:
            exemplars: Dict mapping exemplar IDs to metadata
        """
        labels = self.clusterer.fit_predict(latent_coords_16d)

        exemplars = {}
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                # Handle Long-Tail Fallacy: Preserve rare calls!
                # For noise points (rare calls), we keep ALL of them as exemplars
                # because they cannot be averaged.
                rare_indices = np.where(labels == -1)[0]
                for idx in rare_indices:
                    exemplars[f"rare_{idx}"] = {
                        "latent_coord": latent_coords_16d[idx].tolist(),
                        "audio_path": get_audio_path(idx),
                        "snr": float(original_audio_snrs[idx]),
                        "type": "rare"
                    }
                continue

            # Handle Dense Clusters
            cluster_mask = (labels == label)
            cluster_points = latent_coords_16d[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]

            # 1. Calculate distance matrix within cluster
            dist_matrix = pairwise_distances(cluster_points, metric='euclidean')

            # 2. Find Medoid (point with minimum total distance to all others)
            medoid_local_idx = np.argmin(dist_matrix.sum(axis=1))
            medoid_global_idx = cluster_indices[medoid_local_idx]

            # 3. Quality Weighting: Check SNR
            best_idx = self._find_pristine_exemplar(
                cluster_points,
                cluster_indices,
                medoid_global_idx,
                original_audio_snrs,
                snr_threshold=20.0  # dB
            )

            exemplars[f"zone_{label}"] = {
                "latent_coord": latent_coords_16d[best_idx].tolist(),
                "audio_path": get_audio_path(best_idx),
                "snr": float(original_audio_snrs[best_idx]),
                "type": "dense_zone"
            }

        return exemplars

    def _find_pristine_exemplar(
        self,
        cluster_points,
        cluster_indices,
        medoid_idx,
        snrs,
        snr_threshold
    ):
        """Find the highest-SNR exemplar near the medoid."""
        # If the mathematical medoid has high SNR, use it
        if snrs[medoid_idx] >= snr_threshold:
            return medoid_idx

        # Otherwise, find the point in the top 10% closest to medoid
        # with the highest SNR
        dists_to_medoid = pairwise_distances(
            cluster_points,
            [cluster_points[np.where(cluster_indices == medoid_idx)[0][0]]]
        ).flatten()
        percentile_90 = np.percentile(dists_to_medoid, 10)
        close_indices = np.where(dists_to_medoid <= percentile_90)[0]

        best_local_idx = close_indices[
            np.argmax(snrs[cluster_indices[close_indices]])
        ]
        return cluster_indices[best_local_idx]


def get_audio_path(idx):
    """Placeholder: Get audio file path from index."""
    return f"audio/segment_{idx:06d}.wav"
```

---

## 6. Module 4: Manifest Generation

### 6.1 New Manifest Format

The `continuous_manifold_manifest.json` stores the model weights and exemplar bank.

```json
{
  "version": "1.1",
  "metadata": {
    "created_at": "2026-05-10T00:00:00Z",
    "species": "Rousettus aegyptiacus",
    "total_segments": 50000,
    "num_exemplars": {
      "dense_zones": 42,
      "rare_calls": 127
    }
  },
  "manifold_parameters": {
    "umap_input_dim": 112,
    "umap_output_dim": 30,
    "vae_latent_dim": 16
  },
  "model_paths": {
    "parametric_umap_onnx": "models/umap_encoder.onnx",
    "vae_encoder_onnx": "models/vae_encoder.onnx",
    "vae_decoder_onnx": "models/vae_decoder.onnx"
  },
  "exemplar_bank": {
    "zone_0": {
      "latent_coord_16d": [0.12, -0.45, 0.23, ...],
      "audio_path": "audio/bat_00123.wav",
      "snr": 42.1,
      "type": "dense_zone",
      "cluster_size": 1523
    },
    "rare_89": {
      "latent_coord_16d": [2.31, 1.98, -0.76, ...],
      "audio_path": "audio/bat_04456.wav",
      "snr": 35.2,
      "type": "rare",
      "description": "predator_alarm_variant"
    }
  },
  "manifold_statistics": {
    "latent_mean": [0.01, -0.02, ...],
    "latent_std": [1.02, 0.98, ...],
    "interpolation_validated": true
  }
}
```

---

## 7. Integration with Dual-Stream Architecture

### 7.1 Pipeline Flow

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    Offline Teacher Pipeline                         │
│                   (This TDD - Stage 3 Upgrade)                      │
└─────────────────────────────────────────────────────────────────────┘

Input: 112D BioMAE embeddings (v1.7.0+)
  │
  ├─→ Parametric UMAP (112D → 30D)
  │     └─→ Export: umap_encoder.onnx
  │
  ├─→ Vocal VAE (30D → 16D)
  │     ├─→ Export: vae_encoder.onnx (runtime)
  │     └─→ Export: vae_decoder.onnx (generation)
  │
  ├─→ HDBSCAN Zoning + Medoid Extraction
  │     └─→ Output: exemplar_bank
  │
  └─→ Output: continuous_manifold_manifest.json

┌─────────────────────────────────────────────────────────────────────┐
│                    Runtime Pipeline (Dual-Stream)                  │
└─────────────────────────────────────────────────────────────────────┘

Rust Execution Layer:
  [Raw Audio] → [BioMAE ONNX] → [112D Embedding]
                            │
                            ├─→ [UMAP ONNX] → 30D
                            │                 │
                            │                 └─→ [VAE Encoder ONNX] → 16D
                            │                                             │
                            │                                             ▼
                            │                                    ┌──────────────┐
                            │                                    │ Stream 1     │
                            │                                    │ 16D Manifold  │
                            │                                    │ (Continuous)  │
                            │                                    └──────────────┘
                            │                                             │
                            └─→ (Separate path)                           │
                                          │                              │
                                          ▼                              │
                                   ┌──────────────┐                       │
                                   │ Stream 2     │                       │
                                   │ VQ-VAE        │                       │
                                   │ (Discrete)    │                       │
                                   └──────────────┘                       │
                                          │                              │
                                          └──────────┬───────────────────┘
                                                     ▼
                                          [DualStreamAgent]
                                                     │
                                                     ▼
                                          [DDSP Synthesis Engine]
                                          (FiLM-modulated by Stream 1)
```

### 7.2 File Structure Integration

```
src/
├── corpus_analysis/              # NEW: Offline teacher pipeline
│   ├── __init__.py
│   ├── parametric_umap.py        # Module 1
│   ├── vocal_vae.py              # Module 2
│   ├── medoid_extractor.py       # Module 3
│   └── manifest_builder.py       # Module 4
│
├── cognitive_intelligence/        # EXISTING: Stream 1 affective VAE
│   ├── affective_vae.py          # (Separate from vocal VAE)
│   └── affective_export.py
│
├── technical_architecture/src/   # Rust integration
│   ├── manifold_encoder.rs       # NEW: UMAP+VAE inference
│   ├── synthesis.rs              # EXISTING: Updated for FiLM
│   └── ...
│
├── models/
│   ├── umap_encoder.onnx         # NEW: Exported from Module 1
│   ├── vae_encoder.onnx          # NEW: Exported from Module 2
│   ├── vae_decoder.onnx          # NEW: Exported from Module 2
│   └── continuous_manifold_manifest.json  # NEW: Output from Module 4
```

---

## 8. Implementation Order (4-Week Sprints)

### Week 1-2: Manifold Learning (Modules 1 & 2)
- [ ] Implement Parametric UMAP with cuml (GPU) and PyTorch fallback (CPU)
- [ ] Train on cached 112D BioMAE embeddings
- [ ] Implement Vocal VAE with standard logvar handling
- [ ] Train VAE on UMAP-reduced data
- [ ] Verify graded sequences form continuous trajectories
- [ ] Export both to ONNX for validation

### Week 3: Medoid Extraction (Module 3)
- [ ] Implement HDBSCAN zoning
- [ ] Implement Quality-Weighted Medoid extraction
- [ ] Run "Long-Tail Rescue" test (inject 0.5% rare calls)
- [ ] Verify rare calls preserved as exemplars

### Week 4: Manifest & Integration (Module 4)
- [ ] Implement manifest builder
- [ ] Create `continuous_manifold_manifest.json`
- [ ] Document integration with DualStreamAgent
- [ ] Run "Blurry Centroid" test (compare SNR/formant quality)
- [ ] Update DDSP FiLM layers for 16D manifold input

---

## 9. Testing & Validation

### 9.1 Unit Tests

```python
# tests/test_continuous_manifold.py

def test_umap_preserves_gradients():
    """
    Verify UMAP preserves graded call structure.
    Create synthetic 112D arcs; verify UMAP preserves them.
    """
    # Create synthetic graded sequence
    n_points = 100
    graded_arc = np.linspace(np.zeros(112), np.ones(112), n_points)
    add_small_noise(graded_arc, sigma=0.01)

    # Run UMAP
    reducer = VocalManifoldReducer()
    embedding_30d = reducer.encode(graded_arc)

    # Check: Points should remain ordered along the manifold
    # (First derivative should be positive throughout)
    assert np.all(np.diff(embedding_30d[:, 0]) > 0)


def test_vae_interpolation_smoothness():
    """
    Verify VAE latent space is smoothly interpolable.
    """
    vae = VocalVAE()
    z_a = torch.randn(1, 16)
    z_b = torch.randn(1, 16)

    # Interpolate at 10 steps
    alphas = np.linspace(0, 1, 10)
    for alpha in alphas:
        z_interp = (1 - alpha) * z_a + alpha * z_b
        x_recon = vae.decode(z_interp)

        # Check: Reconstruction should be valid (no NaN/Inf)
        assert torch.all(torch.isfinite(x_recon))


def test_medoid_is_real():
    """
    Verify medoid indices point to real data.
    """
    extractor = MedoidExtractor()
    latent_coords = np.random.randn(100, 16)
    snrs = np.random.uniform(10, 50, 100)

    exemplars = extractor.extract_exemplars(latent_coords, snrs)

    for exemplar_id, meta in exemplars.items():
        # Verify audio path format
        assert meta["audio_path"].startswith("audio/")
        # Verify SNR is in valid range
        assert 0 <= meta["snr"] <= 100
        # Verify latent coord dimension
        assert len(meta["latent_coord"]) == 16
```

### 9.2 Integration Tests

**The "Long-Tail Rescue" Test:**
```python
def test_long_tail_rescue():
    """
    Inject 50 rare alarm calls into 10,000 contact calls.
    Verify new system preserves them; old system deletes them.
    """
    # Create dataset: 10,000 contact calls + 50 alarm calls
    contact_calls = generate_contact_calls(n=10000)
    alarm_calls = generate_alarm_calls(n=50)  # 0.5%
    dataset = np.vstack([contact_calls, alarm_calls])

    # Run new pipeline
    extractor = MedoidExtractor()
    exemplars = extractor.extract_exemplars(dataset, snrs)

    # Count rare exemplars
    rare_count = sum(1 for e in exemplars.values() if e["type"] == "rare")

    # Verify: At least 40 of 50 rare calls preserved (allowing some clustering)
    assert rare_count >= 40, f"Only {rare_count}/50 rare calls preserved"
```

**The "Blurry Centroid" Test:**
```python
def test_medoid_vs_centroid_quality():
    """
    Compare SNR and formant clarity of medoid vs. centroid.
    """
    # Get cluster exemplars
    medoid_exemplar = get_medoid_exemplar(cluster_id=0)
    centroid_audio = generate_centroid_audio(cluster_id=0)

    # Compare SNR
    assert medoid_exemplar["snr"] > calculate_snr(centroid_audio)

    # Compare formant clarity (spectral centroid concentration)
    medoid_sc = spectral_centroid(medoid_exemplar["audio"])
    centroid_sc = spectral_centroid(centroid_audio)
    assert medoid_sc > centroid_sc  # Higher = clearer harmonics
```

### 9.3 Ethological Validation

**Continuous Turing Test:**
- Allow agent to navigate *between* VAE coordinates (interpolating)
- Animals should sustain longer interactions with smooth prosody modulation
- Compare interaction duration: discrete jumps vs. continuous interpolation

---

## 10. Dependencies

### Python Packages
```txt
# GPU path (recommended for Jetson)
cuml==23.0+  # RAPIDS UMAP
torch>=2.0
hdbscan>=0.8
scikit-learn>=1.0

# CPU path (fallback)
umap-learn>=0.5
torch>=2.0
hdbscan>=0.8
scikit-learn>=1.0
```

### Rust Dependencies
```toml
# Cargo.toml additions
[dependencies]
tract-onnx = "0.20"
ndarray = "0.15"
serde = { version = "1.0", features = ["derive"] }
```

---

**Author:** Zoo Vox Research Team
**License:** CC BY-ND 4.0 International
**Last Updated:** 2026-05-10
