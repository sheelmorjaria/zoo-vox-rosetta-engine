# Continuous Manifold Integration Plan

**Component:** Stage 3 Continuous Manifold → Dual-Stream Architecture
**Status:** Integration Planning
**Date:** 2026-05-10

---

## Overview

The Continuous Manifold Mining system is an **offline "Teacher" pipeline** that generates the foundational latent space for **Stream 1 (Continuous Affective/Prosodic)** of the Dual-Stream Architecture.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Integration Architecture                             │
└─────────────────────────────────────────────────────────────────────────┘

                    OFFLINE TEACHER PIPELINE
                 (Continuous Manifold TDD v1.1)
                            │
           ┌────────────────┴────────────────┐
           │                                 │
           ▼                                 ▼
    [Models]                        [Manifest]
    - umap_encoder.onnx              - Exemplar Bank
    - vae_encoder.onnx               - Latent Statistics
    - vae_decoder.onnx               - SNR Metadata
           │
           └────────────────┬────────────────┘
                            │
                    ┌───────▼───────┐
                    │   Runtime     │
                    │   Bridge      │
                    └───────┬───────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   [Rust]              [Python]            [ZMQ]
  Execution          Logic              IPC
   Layer             Layer

┌─────────────────────────────────────────────────────────────────────────┐
│                    EXISTING COMPONENTS (Modified)                      │
├─────────────────────────────────────────────────────────────────────────┤
│  Rust (technical_architecture/src/)                                   │
│  ├── biomae_extractor.rs       → Already outputs 112D                 │
│  ├── manifold_encoder.rs        → NEW: UMAP+VAE inference              │
│  ├── synthesis.rs               → MODIFY: FiLM input from 16D manifold │
│  └── micro_dynamics_extractor.rs → EXISTING: 112D extraction          │
│                                                                     │
│  Python (cognitive_intelligence/)                                    │
│  ├── affective_vae.py          → EXISTING: β-VAE for affect (16D)    │
│  ├── dual_stream_agent.py      → MODIFY: Navigate manifold space     │
│  └── interaction_agent.py      → MODIFY: Stream 1 uses manifold      │
│                                                                     │
│  ZMQ IPC                                                            │
│  ├── DualStreamState           → MODIFY: Add manifold_16d field      │
│  └── DualStreamAction          → MODIFY: Add manifold_target field   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    NEW COMPONENTS (Created)                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Python (corpus_analysis/)                                             │
│  ├── parametric_umap.py        → NEW: Parametric UMAP training       │
│  ├── vocal_vae.py              → NEW: Vocal VAE training             │
│  ├── medoid_extractor.py       → NEW: Quality-weighted medoids      │
│  └── manifest_builder.py       → NEW: Generate manifest JSON        │
│                                                                     │
│  Models/                                                            │
│  ├── umap_encoder.onnx         → NEW: Exported UMAP encoder         │
│  ├── vae_encoder.onnx          → NEW: Exported VAE encoder          │
│  ├── vae_decoder.onnx          → NEW: Exported VAE decoder          │
│  └── continuous_manifold_manifest.json → NEW: Manifest             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Modifications

### 1. Rust: New Manifold Encoder

**File:** `technical_architecture/src/manifold_encoder.rs` (NEW)

```rust
//! Continuous Manifold Encoder: UMAP + VAE inference via ONNX
//!
//! Pipeline: 112D BioMAE → UMAP (30D) → VAE (16D)
//!
//! This is the runtime component of the Continuous Manifold system.
//! The offline training happens in Python (corpus_analysis/).

use tract_onnx::prelude::*;

#[derive(Clone, Debug)]
pub struct ManifoldCoord {
    /// 16D coordinate on the continuous vocal manifold
    pub coords: [f32; 16],
    /// Confidence score (based on VAE KL divergence)
    pub confidence: f32,
}

pub struct ManifoldEncoder {
    /// UMAP encoder: 112D → 30D
    umap_model: RunnableArc<SimplePlan<f32, CachedFact>,
    /// VAE encoder: 30D → 16D
    vae_encoder_model: RunnableArc<SimplePlan<f32, CachedFact>>,
}

impl ManifoldEncoder {
    /// Load ONNX models exported from Python training
    pub fn new(
        umap_path: &str,
        vae_encoder_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Load UMAP encoder
        let umap_model = tract_onnx::onnx()
            .model_for_path(umap_path)?
            .into_declared_runnable()?
            .into_runnable()?;

        // Load VAE encoder
        let vae_encoder_model = tract_onnx::onnx()
            .model_for_path(vae_encoder_path)?
            .into_declared_runnable()?
            .into_runnable()?;

        Ok(Self {
            umap_model,
            vae_encoder_model,
        })
    }

    /// Encode 112D BioMAE embedding to 16D manifold coordinate
    pub fn encode(&self, bio_mae_112d: &[f32; 112]) -> Result<ManifoldCoord, Error> {
        // Step 1: UMAP reduction 112D → 30D
        let umap_input = Tensor::from(&[bio_mae_112d[..]]).into_shape(&[1, 112])?;
        let umap_output = self.umap_model.run(tvec!(umap_input))?;
        let umap_30d: Vec<f32> = umap_output[0].to_array_view()?.iter().copied().collect();

        // Step 2: VAE encoding 30D → 16D
        let vae_input = Tensor::from(&umap_30d[..]).into_shape(&[1, 30])?;
        let vae_output = self.vae_encoder_model.run(tvec!(vae_input))?;
        let vae_16d: Vec<f32> = vae_output[0].to_array_view()?.iter().copied().collect();

        // Convert to array
        let mut coords = [0.0; 16];
        coords.copy_from_slice(&vae_16d[..16]);

        Ok(ManifoldCoord {
            coords,
            confidence: 1.0, // TODO: Compute from VAE logvar
        })
    }

    /// Batch encode for efficiency
    pub fn encode_batch(&self, bio_mae_embeddings: &[Vec<f32>]) -> Result<Vec<ManifoldCoord>, Error> {
        bio_mae_embeddings
            .iter()
            .map(|emb| {
                let mut arr = [0.0; 112];
                arr.copy_from_slice(&emb[..112.min(emb.len())]);
                self.encode(&arr)
            })
            .collect()
    }
}
```

---

### 2. Python: Update DualStreamState

**File:** `realtime/action_publisher.py` (MODIFY)

```python
# ADD to DualStreamState
@dataclass
class DualStreamState:
    """Dual-stream state received from Rust."""
    # EXISTING fields
    syntactic_token: int
    affect_vector: np.ndarray  # 16D β-VAE affect (existing)
    raw_features: np.ndarray  # 112D for fallback
    confidence: float
    sequence: int

    # NEW: Continuous Manifold fields
    manifold_16d: Optional[np.ndarray] = None  # 16D manifold coordinate
    manifold_confidence: float = 1.0
    nearest_exemplar_id: Optional[str] = None  # "zone_0" or "rare_42"
```

---

### 3. Python: Update DualStreamAgent

**File:** `realtime/interaction_agent.py` (MODIFY)

```python
class DualStreamInteractionAgent:
    """
    InteractionAgent v2.1 with continuous manifold navigation.

    The agent now navigates a continuous 16D manifold space instead of
    discrete cluster IDs, enabling smooth prosodic interpolation.
    """
    def __init__(self, manifold_manifest_path: str):
        # Load continuous manifold manifest
        with open(manifold_manifest_path) as f:
            self.manifest = json.load(f)

        self.exemplar_bank = self.manifest["exemplar_bank"]
        self.latent_dim = self.manifest["manifold_parameters"]["vae_latent_dim"]

        # Pre-compute nearest neighbor index for fast lookup
        self._build_exemplar_index()

    def _build_exemplar_index(self):
        """Build KD-tree for fast nearest exemplar lookup."""
        from sklearn.neighbors import KDTree

        exemplar_coords = []
        exemplar_ids = []
        for exemplar_id, meta in self.exemplar_bank.items():
            exemplar_coords.append(meta["latent_coord_16d"])
            exemplar_ids.append(exemplar_id)

        self.exemplar_tree = KDTree(np.array(exemplar_coords))
        self.exemplar_ids = exemplar_ids

    def handle_manifold_state(self, manifold_16d: np.ndarray) -> DualStreamAction:
        """
        Generate response by navigating the manifold.

        Instead of selecting a discrete cluster, the agent:
        1. Finds nearest exemplar on manifold
        2. Computes target trajectory (e.g., de-escalate arousal)
        3. Returns interpolated manifold coordinate
        """
        # Find nearest exemplar
        dist, idx = self.exemplar_tree.query([manifold_16d], k=1)
        nearest_exemplar_id = self.exemplar_ids[idx[0][0]]
        nearest_exemplar = self.exemplar_bank[nearest_exemplar_id]

        # Compute target manifold coordinate
        # Example: Move toward "calm" region of manifold
        target_manifold = self._compute_affective_trajectory(
            current=manifold_16d,
            target_arousal=0.3  # De-escalate
        )

        return DualStreamAction(
            syntactic_token=self._get_valid_next_token(),
            manifold_target=target_manifold,  # NEW: 16D target coordinate
            temporal_offset_ms=150.0,
        )

    def _compute_affective_trajectory(
        self,
        current: np.ndarray,
        target_arousal: float
    ) -> np.ndarray:
        """
        Compute target manifold coordinate for affective modulation.

        Uses the VAE decoder to explore the manifold.
        """
        # Load VAE decoder (cached)
        if not hasattr(self, '_vae_decoder'):
            import onnx
            self._vae_decoder = onnx.load(
                self.manifest["model_paths"]["vae_decoder_onnx"]
            )

        # TODO: Implement gradient-based navigation on manifold
        # For now, simple linear interpolation toward low-arousal exemplar
        return current * 0.8  # Simplified: shrink toward origin
```

---

### 4. Rust: Update Synthesis Engine

**File:** `technical_architecture/src/synthesis.rs` (MODIFY)

```rust
/// ADD to existing DDSP synthesis
impl DualStreamSynthesizer {
    /// Apply manifold-based FiLM modulation to DDSP parameters
    pub fn apply_manifold_modulation(
        &self,
        base_params: &DDSPParams,
        manifold_16d: &[f32; 16],
    ) -> DDSPParams {
        /// Map 16D manifold coordinates to FiLM parameters
        /// Dimensions 0-3 map to arousal, valence, pitch, harshness
        let arousal = manifold_16d[0].clamp(0.0, 1.0);
        let valence = manifold_16d[1].clamp(-1.0, 1.0);
        let pitch_shift = manifold_16d[2].clamp(-2.0, 2.0);  // Semitones
        let harshness = manifold_16d[3].clamp(0.0, 1.0);

        DDSPParams {
            // Arousal → HNR scaling (high arousal = more noise)
            harmonic_amplitudes: base_params.harmonic_amplitudes
                .iter()
                .map(|&amp| amp * (1.0 - arousal * 0.3))
                .collect(),

            // Arousal → Noise magnitudes (inverse)
            noise_magnitudes: base_params.noise_magnitudes
                .iter()
                .map(|&mag| mag * (1.0 + arousal * 0.5))
                .collect(),

            // Pitch shift
            f0_hz: base_params.f0_hz * (2.0_f32).powf(pitch_shift / 12.0),

            // Harshness → Jitter/Shimmer
            jitter: harshness * 0.1,  // Up to 10% jitter
            shimmer: harshness * 0.05,  // Up to 5% shimmer

            ..*base_params
        }
    }
}
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE → RUNTIME FLOW                          │
└─────────────────────────────────────────────────────────────────────────┘

PHASE 1: OFFLINE TRAINING (Python, corpus_analysis/)

    [Raw Audio Corpus]
           │
           ├─→ Stage 2: BioMAE Extraction
           │    └─→ bio_mae_112d.npy (50,000 × 112)
           │
           ├─→ Module 1: Parametric UMAP
           │    ├─→ Train: VocalManifoldReducer
           │    ├─→ Fit: cuml.UMAP → target_30d
           │    └─→ Export: umap_encoder.onnx
           │
           ├─→ Module 2: Vocal VAE
           │    ├─→ Train: VocalVAE (30D → 16D)
           │    └─→ Export: vae_encoder.onnx, vae_decoder.onnx
           │
           ├─→ Module 3: Medoid Extraction
           │    ├─→ HDBSCAN zoning
           │    └─→ Quality-weighted medoids
           │
           └─→ Module 4: Manifest Builder
                └─→ continuous_manifold_manifest.json

PHASE 2: RUNTIME INFERENCE (Rust + Python)

    [Live Audio Input]
           │
           ├─→ Rust: BioMAEExtractor (biomae_extractor.rs)
           │    └─→ bio_mae_112d: [f32; 112]
           │
           ├─→ Rust: ManifoldEncoder (manifold_encoder.rs)
           │    ├─→ UMAP: 112D → 30D (umap_encoder.onnx)
           │    └─→ VAE: 30D → 16D (vae_encoder.onnx)
           │    └─→ manifold_16d: [f32; 16]
           │
           ├─→ ZMQ IPC: DualStreamState
           │    └─→ Python: DualStreamInteractionAgent
           │         ├─→ Query: nearest_exemplar_id
           │         ├─→ Compute: target_manifold_16d
           │         └─→ Select: syntactic_token (Stream 2)
           │
           └─→ ZMQ IPC: DualStreamAction
                └─→ Rust: DualStreamSynthesizer
                     ├─→ FiLM modulation: manifold_16d → DDSP params
                     └─→ Output: Synthesized audio
```

---

## File Creation Order

### Phase 1: Offline Training (Weeks 1-3)

| Order | File | Purpose | Dependencies |
|-------|------|---------|--------------|
| 1 | `corpus_analysis/__init__.py` | Package init | None |
| 2 | `corpus_analysis/parametric_umap.py` | UMAP training | BioMAE v1.7.0 |
| 3 | `corpus_analysis/vocal_vae.py` | VAE training | parametric_umap.py |
| 4 | `corpus_analysis/medoid_extractor.py` | Medoid extraction | vocal_vae.py |
| 5 | `corpus_analysis/manifest_builder.py` | Manifest generation | medoid_extractor.py |
| 6 | `tests/test_continuous_manifold.py` | Test suite | All above |

### Phase 2: Runtime Integration (Week 4)

| Order | File | Purpose | Dependencies |
|-------|------|---------|--------------|
| 7 | `technical_architecture/src/manifold_encoder.rs` | Rust inference | Phase 1 complete |
| 8 | `realtime/action_publisher.py` (modify) | Add manifold fields to ZMQ | manifold_encoder.rs |
| 9 | `realtime/interaction_agent.py` (modify) | Manifold navigation | action_publisher.py |
| 10 | `technical_architecture/src/synthesis.rs` (modify) | FiLM modulation | interaction_agent.py |
| 11 | `tests/test_manifold_integration.py` | End-to-end tests | All above |

---

## Validation Checklist

Before field deployment:

- [ ] **Offline Tests:**
  - [ ] `test_umap_preserves_gradients()` passes
  - [ ] `test_vae_interpolation_smoothness()` passes
  - [ ] `test_medoid_is_real()` passes
  - [ ] `test_long_tail_rescue()` passes (≥80% rare calls preserved)

- [ ] **ONNX Export:**
  - [ ] `umap_encoder.onnx` validates with onnx.checker
  - [ ] `vae_encoder.onnx` validates with onnx.checker
  - [ ] `vae_decoder.onnx` validates with onnx.checker

- [ ] **Rust Integration:**
  - [ ] `ManifoldEncoder::encode()` returns valid 16D coords
  - [ ] Batch encode processes 1000 embeddings < 100ms

- [ ] **End-to-End:**
  - [ ] Full pipeline: Audio → 112D → 30D → 16D → Exemplar lookup
  - [ ] ZIPC IPC: DualStreamState includes manifold_16d
  - [ ] Synthesis: FiLM modulation produces audible affect changes

---

**Author:** Zoo Vox Research Team
**License:** CC BY-ND 4.0 International
