# Dual-Stream Acoustic-Syntactic Architecture

## Overview

The dual-stream architecture addresses the **Discretization Paradox** in animal vocalization analysis by implementing separate processing pathways for continuous affective/prosodic features and discrete syntactic/semantic features. This design is inspired by the Hickok & Poeppel dual-stream model of human speech processing.

### The Problem: Discretization Paradox

The original single-stream pipeline forced 112D RosettaFeatures into a 45-cluster Bayesian GMM for classification. This created a fundamental paradox:

- **Continuous information loss**: Intra-call affect and prosody variations were discarded when forcing features into discrete clusters
- **Syntactic combinatorics undervalued**: Inter-call relationships and sequencing patterns were inadequately captured by a single clustering layer

### The Solution: Dual-Stream Architecture

```
[Raw Audio] → [NBD] → [112D Rosetta Features]
                                |
                +---------------+---------------+
                |                               |
    [Stream 1: Continuous]          [Stream 2: Discrete]
    (Affective/Prosodic)            (Syntactic/Semantic)
                |                               |
    [54D Affective Features]        [44D Syntactic Features]
                |                               |
    [β-VAE Encoder]                 [VQ-VAE Tokenizer]
    (16D latent space)              (64 discrete tokens)
                |                               |
    [Affect Vector]                 [Syntactic Token]
    [Arousal, Valence, etc.]        [Call Type, Syntax]
                |                               |
                +---------------+---------------+
                                |
                    [Dual-Stream Cognitive Agent]
                    (InteractionAgent v2.0)
                                |
                    [FiLM-Based DDSP Synthesis]
                    (Affect-modulated audio)
```

## Critical Risk Mitigations (The "Watch-Outs")

**Before beginning implementation, the following architectural risks must be acknowledged and mitigated:**

### Risk A: Python Inference Latency Will Break the Sub-50ms Budget

**The Problem**: The original plan implied the Python InteractionAgent would run the VAE and VQ-VAE encoders in real-time. Standard PyTorch inference on a 54D/44D tensor, even on GPU, introduces non-deterministic overhead (memory allocation, GIL contention). If Python bottlenecked the original 45-cluster GMM, it will absolutely bottleneck two neural networks.

**The Fix**: The VAE and VQ-VAE encoders must be treated the same way as the original DDSP decoder. They must be exported to ONNX/TensorRT and executed on the edge device.

**Action**: Move VAE/VQ-VAE encoder inference to the Rust layer (or a dedicated TensorRT Python microservice), passing only the resulting 16D Affect Vector and Discrete Token over ZMQ to the Python cognitive agent.

**Status**: ✅ Mitigated in architecture design
- VAE/VQ-VAE will be exported to ONNX format
- Rust execution layer (`affect_encoder.rs`, `syntactic_encoder.rs`) will handle inference
- ZMQ passes only lightweight DualStreamState (16D + int + metadata)

### Risk B: The DDSP Decoder Retraining Penalty

**The Problem**: Changing the DualStreamDDSPDecoder input from 112D to 128D (112 + 16 affect) means the existing, highly tuned DDSP model must be trained from scratch. The original model was trained on pure acoustic physics; injecting a latent 16D vector into the middle of it may cause catastrophic forgetting or unstable gradient flows.

**The Fix**: Instead of concatenating the affect vector at the input layer, use **Feature-wise Linear Modulation (FiLM)**. The 112D features pass through the MLP as usual, but the 16D affect vector is used to generate scaling (γ) and shifting (β) parameters that modulate the hidden layers.

**Action**: Implement FiLM layers in the DDSP decoder. This allows initialization with the pre-trained 112D weights and only trains the FiLM parameters, preserving acoustic fidelity.

**Status**: ✅ Mitigated in implementation
- `DualStreamDDSPDecoder` uses `FiLMGenerator` for affect modulation
- Base MLP weights can be frozen and preserved
- Only FiLM γ/β generators require training

### Risk C: VQ-VAE Codebook Collapse

**The Problem**: VQ-VAEs are notoriously prone to "codebook collapse," where only a fraction of the discrete tokens are ever used, and the rest remain dead. This would result in a syntactic vocabulary of 64 tokens where only 10 are actually utilized.

**The Fix**: Implement Exponential Moving Average (EMA) updates for the codebook, and employ codebook revival techniques (resetting dead codes to random encoder outputs).

**Status**: ✅ Mitigated in implementation
- `EMAVectorQuantizer` uses EMA for codebook updates (decay=0.99)
- `revive_dead_codes()` method copies active codes to dead ones with noise
- Perplexity tracking monitors utilization in real-time

---

## Technical Deep Dives

### Module 1 Deep Dive: Affective VAE

To ensure the 16D latent space actually maps to biological arousal/valence and not just acoustic noise, the VAE must be constrained.

#### Loss Function Update

Use a **β-VAE** approach (weighting the KL divergence penalty higher, β=2.0). This forces the latent space to be disentangled, increasing the likelihood that individual dimensions map to interpretable biological traits:

```
Loss = Reconstruction + β × KL_Divergence
       = MSE(recon, input) + 2.0 × KL(q(z|x) || p(z))
```

**Expected Interpretability**:
- Dim 0 ≈ Arousal (low → high energy)
- Dim 1 ≈ Harshness (smooth → noisy)
- Dim 2 ≈ Pitch variation (flat → vibrato)

#### Affective Response Logic

The simplistic "multiply by 1.2" approach is insufficient for biological interaction. Implement **Affective Matching vs. De-escalation**:

```python
def compute_target_affect(incoming_affect: np.ndarray) -> np.ndarray:
    """
    Biologically-inspired affective response policy.

    Rules:
    - Arousal > 0.8: De-escalate to avoid panic cascade
    - Arousal < 0.3: Escalate for social contact
    - Otherwise: Match for social bonding
    """
    arousal = incoming_affect[0]

    if arousal > 0.8:
        # De-escalate: prevent panic cascade
        target_arousal = 0.6
        return incoming_affect * (target_arousal / arousal)
    elif arousal < 0.3:
        # Escalate: maintain social contact
        target_arousal = 0.4
        return incoming_affect * (target_arousal / arousal)
    else:
        # Match: social bonding
        return incoming_affect
```

### Module 2 Deep Dive: Syntax Graph

#### Laplace Smoothing

The transition matrix must have Laplace smoothing. If a valid biological bigram was simply not observed in the training corpus, a hard 0 probability will prevent the agent from ever generating it.

**Formula**:
```
P(t_i | t_{i-1}) = (Count(t_{i-1}, t_i) + α) / (Count(t_{i-1}) + α·N)

where:
- α = 0.01 (smoothing parameter)
- N = 64 (vocabulary size)
```

**Implementation**:
```python
class SyntaxGraph:
    def __init__(self, num_tokens: int = 64, alpha: float = 0.01):
        self.num_tokens = num_tokens
        self.alpha = alpha
        # Initialize with uniform probability (smoothed)
        self.transitions = np.full((num_tokens, num_tokens), alpha / num_tokens)
```

### Module 4 Deep Dive: Rust Synthesis Modulation

The `apply_affect_modulation` function needs explicit mathematical mapping from the 16D latent vector to acoustic parameters.

#### Explicit Dimension Mapping

| Latent Dimension | Acoustic Parameter | Mathematical Mapping | Biological Effect |
|-----------------|-------------------|---------------------|-------------------|
| 0: Arousal | HNR Scaling | `hnr' = hnr × (1.0 - arousal × 0.5)` | High arousal → lower HNR (more noise/chaos) |
| 1: Valence | Jitter Factor | `jitter' = jitter × (1.0 - valence × 0.3)` | Negative valence → more jitter (roughness) |
| 2: Pitch Variation | Vibrato Depth | `vibrato' = vibrato_base × max(0, pitch_dim)` | Higher value → deeper vibrato (0-50 Hz) |
| 3-15 | Reserved | — | Future biological mappings |

**Rust Implementation**:
```rust
pub fn map_affect_to_acoustic(&self, affect_vector: &[f32; 16]) -> AffectModulation {
    AffectModulation {
        // Dimension 0: Arousal (0-1) → HNR scaling
        // Higher arousal = more noise = lower HNR
        arousal_hnr_scaling: 1.0 - (affect_vector[0] * 0.5),

        // Dimension 1: Valence (-1 to 1) → Jitter
        // Negative valence = more jitter
        valence_jitter_factor: 1.0 + (-affect_vector[1] * 0.3),

        // Dimension 2: Pitch variation → Vibrato depth
        pitch_vibrato_depth: affect_vector[2].max(0.0).min(1.0) * 50.0,

        reserved: [0.0; 13],
    }
}
```

---

## Refined Implementation Schedule (2-Week Sprints)

### Sprint 1 & 2: Offline Training & Model Architecture (Weeks 1-4)

**Focus**: Get the math right without real-time constraints.

1. Implement `AffectiveFeatureExtractor` and `SyntacticFeatureExtractor`
2. Train **β-VAE** (β=2.0) on cached 54D segments
   - Target: KL loss stable, reconstruction < 0.1
3. Train **VQ-VAE with EMA** on cached 44D segments
   - Target: Codebook utilization > 80%, commitment loss < 0.05
4. Extract **Syntax Graph** from VQ-VAE tokenized corpus
   - Apply Laplace smoothing (α = 0.01)

**Milestone**: Offline validation — Can we reconstruct 112D features by combining the VAE output and VQ-VAE output?

### Sprint 3: DDSP FiLM Retraining (Week 5)

**Focus**: Teach the synthesizer to hear the Affect.

1. Implement `DualStreamDDSPDecoder` using **FiLM layers** (not concatenation)
2. **Freeze** the base 112D MLP weights
3. Train only the FiLM γ/β generators using the 16D affect vectors
4. Fine-tune the entire network end-to-end
5. Export to ONNX/TensorRT

**Milestone**: Synthetic audio demonstrates perceptible affective variation when the 16D vector is perturbed, while maintaining syntactic structure.

### Sprint 4: Real-Time Pipeline Integration (Week 6)

**Focus**: Wire it together under strict latency constraints.

1. Update Rust `micro_dynamics_extractor` to output split 54D/44D arrays
2. Deploy **ONNX VAE/VQ-VAE encoders** to the edge device
3. Update ZMQ `FeatureEventPublisher` to stream `DualStreamState`
4. Update Python `InteractionAgent` to consume `DualStreamState` and query `SyntaxGraph`

**Milestone**: Sub-100ms end-to-end latency achieved in lab environment.

### Sprint 5: Ethological Validation (Week 7-8)

**Focus**: Prove the biology works.

1. Deploy "Semantic vs. Affective Mismatch" test
2. Run **Condition A** (Congruent), **Condition B** (Syntactic Mismatch), **Condition C** (Affective Mismatch)
3. Measure RAS (Response Appropriateness Score) and Acoustic Convergence

**Milestone**: Statistically significant proof that subjects react differently to Stream 1 (Affect) vs Stream 2 (Syntax) manipulations.

---

## Architecture Components

### Stream 1: Continuous Affective/Prosodic Stream

**Purpose**: Capture graded continuum of internal state, arousal, and affect.

**Input**: 54D continuous features extracted from 112D RosettaFeatures
**Output**: 16D disentangled latent space (β-VAE)

#### Feature Isolation

The 54D affective features are extracted from specific indices of the 112D vector:

| Source | Indices | Features |
|--------|---------|----------|
| Layer 1: Base Physics | 0, 7, 35-38 | F0, RMS, HNR, Jitter, Shimmer, Vibrato |
| Layer 2: Macro Texture | 59, 62, 67 | GLCM texture features |
| Layer 3: Micro Texture | 76-111 | Spectral derivatives, FM, dynamics, rhythm |

#### β-VAE Architecture

```python
β-VAE: 54D input → 16D latent (β = 2.0)
```

- **β = 2.0**: Higher KL divergence penalty forces disentanglement
- **Interpretable dimensions**: Dim 0 ≈ Arousal, Dim 1 ≈ Harshness, etc.
- **Smooth modulation**: Continuous interpolation in latent space

#### Affective Response Policy

The agent implements biologically-inspired affect matching:

```python
def compute_target_affect(incoming_affect):
    arousal = incoming_affect[0]

    if arousal > 0.8:
        # De-escalate to avoid panic cascade
        return incoming_affect * 0.75
    elif arousal < 0.3:
        # Escalate slightly for engagement
        return incoming_affect * 1.2
    else:
        # Match for social bonding
        return incoming_affect
```

### Stream 2: Discrete Syntactic Stream

**Purpose**: Capture discrete inter-call sequencing and call categories.

**Input**: 44D syntactic features extracted from 112D RosettaFeatures
**Output**: 64 discrete tokens with learned transition probabilities

#### Feature Isolation

The 44D syntactic features emphasize spectral shape and categorical information:

| Source | Indices | Features |
|--------|---------|----------|
| Layer 1: Base Physics | 1-6, 8-34 | MFCCs, spectral shape |
| Layer 2: Macro Texture | 46-58, 60-61, 63-75 | Harmonic structure, pitch geometry |

#### VQ-VAE with EMA Codebook

```python
VQ-VAE: 44D input → 64 discrete tokens (32D codebook)
```

**Key Features**:
- **EMA codebook updates**: Exponential moving average prevents collapse
- **Codebook revival**: Dead tokens are revived from active ones
- **Perplexity tracking**: Monitors codebook utilization (>80% target)

#### Syntax Graph with Laplace Smoothing

Probabilistic transition matrix with smoothing to prevent zero-probability bigrams:

```
P(t_i | t_{i-1}) = (Count + α) / (Total + α·N)

where α = 0.01 (smoothing parameter)
```

**Features**:
- Validates bigram legality
- Generates syntactically-valid sequences
- Provides top-k valid next tokens

### Stream Convergence: Dual-Stream Cognitive Agent

**Purpose**: Combine both streams to generate contextually-appropriate responses.

#### DualStreamState Structure

```python
@dataclass
class DualStreamState:
    syntactic_token: int      # Discrete token from VQ-VAE
    affect_vector: np.ndarray # 16D continuous from β-VAE
    raw_features: np.ndarray  # 112D for fallback
    confidence: float         # Combined confidence
    sequence: int             # Temporal ordering
```

#### DualStreamAction Structure

```python
@dataclass
class DualStreamAction:
    syntactic_token: int      # Response call type
    affect_vector: np.ndarray # Response affect modulation
    temporal_offset_ms: float # Response delay (default 150ms)
    priority: str
    sequence: int
```

#### Decision Logic

```python
def handle_dual_stream_state(state):
    # Stream 2: Validate syntactic response
    valid_next = syntax_graph.get_valid_next_tokens(
        state.syntactic_token, top_k=5
    )
    response_token = valid_next[0][0]

    # Stream 1: Compute affective response
    target_affect = compute_affective_response(state.affect_vector)

    return DualStreamAction(
        syntactic_token=response_token,
        affect_vector=target_affect,
        temporal_offset_ms=150.0
    )
```

### Dual-Stream Synthesis

**Purpose**: Generate affect-modulated audio using FiLM (Feature-wise Linear Modulation).

#### FiLM Architecture

FiLM layers allow affective control without retraining the entire DDSP network:

```python
γ, β = FiLMGenerator(affect_vector)  # 16D → layer-wise scale/shift

# Apply FiLM modulation
output = γ * base_network(features) + β
```

**Benefits**:
- Preserves pre-trained 112D DDSP weights
- Only FiLM parameters need training
- Enables smooth affective modulation

#### Affect Modulation Mapping

The 16D affect vector maps to acoustic parameters:

| Dimension | Acoustic Parameter | Effect |
|-----------|-------------------|--------|
| 0 (Arousal) | HNR scaling | High arousal → lower HNR (more noise) |
| 1 (Valence) | Jitter factor | Negative valence → more jitter |
| 2 (Pitch) | Vibrato depth | Higher value → deeper vibrato (0-50 Hz) |
| 3-15 | Reserved | Future mappings |

## File Structure

### Python Modules (Logic Layer)

```
cognitive_intelligence/
├── affective_vae.py           # β-VAE for Stream 1
├── affective_encoder.py       # Affective feature extraction
├── syntactic_vqvae.py         # VQ-VAE for Stream 2
├── syntactic_encoder.py       # Syntactic feature extraction
├── syntax_graph.py            # Laplace-smoothed transition matrix
├── ddsp_decoder.py            # Dual-stream DDSP with FiLM
├── train_affective_vae.py     # β-VAE training script
└── train_syntactic_vqwae.py   # VQ-VAE training script

realtime/
├── interaction_agent.py       # DualStreamInteractionAgent v2.0
└── action_publisher.py        # DualStreamAction/State + ZMQ

tests/
├── test_affective_vae.py      # Module 1 tests (20 tests)
├── test_syntactic_vqvae.py    # Module 2 tests (28 tests)
├── test_dual_stream_agent.py  # Module 3 tests (21 tests)
└── test_dual_stream_synthesis.py # Module 4 tests (22 tests)
```

### Rust Modules (Execution Layer)

```
technical_architecture/src/
├── synthesis.rs               # AffectModulation, DualStreamSynthesizer
├── affect_encoder.rs          # VAE ONNX inference (planned)
└── syntactic_encoder.rs       # VQ-VAE ONNX inference (planned)
```

### Data Files

```
models/dual_stream/
├── affective_vae.pt           # Trained β-VAE weights
├── syntactic_vqvae.pt         # Trained VQ-VAE weights
├── syntax_graph.json          # Laplace-smoothed transitions
└── synthesis_manifest.json    # Complete metadata

models/
└── synthesis_manifest.json    # Updated with dual-stream config
```

## Training Pipeline

### Phase 1: Feature Extraction & Model Training

1. **Extract stream-specific features** from cached 112D data
2. **Train β-VAE** on 54D affective features (β=2.0)
   - Target: Reconstruction loss < 0.1, KL stable
3. **Train VQ-VAE** on 44D syntactic features
   - Target: Codebook utilization >80%, commitment loss < 0.05

### Phase 2: Syntax Graph Construction

```bash
python -m cognitive_intelligence.build_syntax_graph \
    --vqvae models/dual_stream/syntactic_vqvae.pt \
    --data data/cached_features.npy \
    --output models/dual_stream/syntax_graph.json
```

### Phase 3: DDSP FiLM Training

1. Freeze pre-trained 112D DDSP base network
2. Train only FiLM γ/β generators
3. Fine-tune entire network end-to-end
4. Export to ONNX/TensorRT for Rust inference

## Performance Targets

### Offline Training Metrics
| Metric | Target |
|--------|--------|
| β-VAE reconstruction loss | < 0.1 |
| β-VAE KL divergence | Stable (not exploding) |
| VQ-VAE codebook utilization | > 80% |
| VQ-VAE commitment loss | < 0.05 |
| Syntax graph coverage | > 95% of biological bigrams |

### Real-Time Performance
| Component | Target Latency |
|-----------|----------------|
| End-to-end | < 100ms (99th percentile) |
| VAE encoder (TensorRT) | < 5ms |
| VQ-VAE encoder (TensorRT) | < 5ms |
| DDSP synthesis | < 50ms |

## Usage Examples

### Feature Extraction

```python
from cognitive_intelligence.affective_encoder import AffectiveFeatureExtractor
from cognitive_intelligence.syntactic_encoder import SyntacticFeatureExtractor

# Extract 112D Rosetta features
features_112d = extract_rosetta_features(audio)

# Split into streams
affective = AffectiveFeatureExtractor.extract(features_112d)  # 54D
syntactic = SyntacticFeatureExtractor.extract(features_112d)  # 44D
```

### Model Inference

```python
from cognitive_intelligence.affective_vae import BetaVAE
from cognitive_intelligence.syntactic_vqvae import SyntacticVQVAE

# Load models
affective_vae = BetaVAE.load_from_checkpoint("models/dual_stream/affective_vae.pt")
syntactic_vqvae = SyntacticVQVAE.load_from_checkpoint("models/dual_stream/syntactic_vqwae.pt")

# Encode
affect_vector = affective_vae.encode(affective)      # 16D
syntactic_token = syntactic_vqvae.tokenize(syntactic) # int
```

### Dual-Stream Agent

```python
from realtime.interaction_agent import DualStreamInteractionAgent
from realtime.action_publisher import DualStreamState, DualStreamAction

# Initialize agent
agent = DualStreamInteractionAgent()
agent.start()

# Create input state
state = DualStreamState(
    syntactic_token=42,
    affect_vector=affect_vector,
    raw_features=features_112d,
    confidence=0.85,
    sequence=0
)

# Handle and generate response
action = agent.handle_dual_stream_state(state)

# Response action contains:
# - syntactic_token: Discrete call type for response
# - affect_vector: Continuous modulation for synthesis
# - temporal_offset_ms: When to play response
```

### Synthesis with Affect Modulation

```python
from cognitive_intelligence.ddsp_decoder import DualStreamDDSPDecoder

# Create decoder
decoder = DualStreamDDSPDecoder.create_with_pretrained(
    pretrained_path="models/ddsp_pretrained.pt",
    affect_dim=16
)

# Synthesize with affect
harmonic_amps, noise_mags = decoder(
    features_112d=features_112d,
    affect_vector=action.affect_vector
)
```

## Verification Checklist (Go/No-Go for Field Deployment)

**Before deploying to a live colony (e.g., bats or marmosets), the following integration tests must pass:**

| Test | Description | Success Criteria |
|------|-------------|------------------|
| **Disentanglement Test** | Verify 16D latent space is biologically interpretable | Perturbing one dimension results in smooth, monotonic acoustic change (e.g., HNR increases) without altering macro-spectral shape |
| **Syntax Integrity Test** | Ensure syntactic validity of generated responses | Agent NEVER generates a zero-probability bigram when constrained by SyntaxGraph |
| **Latency Profile Test** | Real-time budget compliance | 99th percentile latency (Mic → NBD → VAE/VQ-VAE → ZMQ → Agent → ZMQ → Synthesis) is **< 80ms** |
| **OOD Resilience Test** | Out-of-distribution robustness | Gaussian noise injection causes VAE confidence to drop, triggering Confidence-Based Suppression (not hallucination) |
| **Codebook Utilization Test** | VQ-VAE vocabulary utilization | Codebook utilization **> 80%** (no collapse) |
| **FiLM Preservation Test** | Pre-trained weight protection | Pre-trained DDSP weights preserved within **5%** of original values |
| **Affective Matching Test** | Biological response correctness | High arousal (>0.8) triggers de-escalation (target arousal = 0.6) to avoid panic cascade |

### Ethological Validation Conditions

**"Semantic vs. Affective Mismatch" Test**:

| Condition | Stream 1 (Affect) | Stream 2 (Syntax) | Expected Subject Response |
|-----------|------------------|-------------------|---------------------------|
| **A: Congruent** | High arousal match | Valid syntactic sequence | High RAS (> 0.8), strong acoustic convergence |
| **B: Syntactic Mismatch** | High arousal match | Invalid syntactic sequence | Low RAS (< 0.5), confusion/rejection |
| **C: Affective Mismatch** | Low arousal response | Valid syntactic sequence | Medium RAS, reduced social engagement |

**Statistical Validation**: p < 0.05 for significant difference across conditions

## References

### Biological Inspiration
- Hickok, G., & Poeppel, D. (2007). The cortical organization of speech processing. *Nature Reviews Neuroscience*, 8(5), 393-402.

### Technical References
- β-VAE: Higgins, I., et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.
- VQ-VAE: Oord, A., et al. (2017). Neural Discrete Representation Learning.
- FiLM: Perez, E., et al. (2018). Film: Visual Reasoning with a General Conditioning Layer.

---

**Author**: Sheel Morjaria (sheelmorjaria@gmail.com)
**License**: CC BY-ND 4.0 International
**Version**: 1.0.0
**Date**: 2026-05-09
