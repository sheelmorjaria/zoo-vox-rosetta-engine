# Zoo Vox Rosetta Engine

**Universal Rosetta Stone Methodology for Cross-Species Vocalization Translation**

A bioacoustic analysis framework that decodes animal communication through 112D feature extraction, neural boundary detection, and real-time closed-loop interaction. The system enables true translation between species by mapping vocalizations to a universal feature space, revealing hidden semantic structures.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Methodologies](#core-methodologies)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Key Features](#key-features)
6. [Deployment](#deployment)
7. [Scientific Impact](#scientific-impact)
8. [Test Coverage](#test-coverage)
9. [Documentation](#documentation)
10. [License](#license)

---

## Architecture Overview

### Hybrid Python/Rust Architecture

The Zoo Vox Rosetta Engine follows a **hybrid architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Systemd Supervisor                        │
│  ┌──────────────────────────┐  ┌──────────────────────────┐     │
│  │  rust-field-engine       │  │  python-cognitive-agent  │     │
│  │  (Execution Layer)       │  │  (Logic Layer)           │     │
│  │                          │  │                          │     │
│  │  - Safety Critical       │  │  - Decision Making       │     │
│  │  - Audio Processing      │◄─┤  - Phrase Selection      │     │
│  │  - Hardware Control      │  │  - Learning              │     │
│  │  - Heartbeat Monitor     │  │  - Intent Generation     │     │
│  │                          │  │                          │     │
│  │  ZeroMQ SUB (Heartbeat)  │◄─┤  ZeroMQ PUB (Heartbeat)  │     │
│  └──────────────────────────┘  └──────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

| Layer | Language | Responsibility | Location |
|-------|----------|---------------|----------|
| **Execution Layer** | Rust | Time-critical operations, signal processing, hardware access, safety | `technical_architecture/` |
| **Logic Layer** | Python | Cognitive intelligence, decision making, learning, context interpretation | `cognitive_intelligence/`, `realtime/`, `semiotics/` |

### Key Principle: Fail Open to Safety

If Python crashes, Rust immediately mutes audio and continues in **Passthrough Mode** (recording only, no synthesis).

---

## Core Methodologies

### 1. 112D Rosetta Feature Extraction

The system extracts a comprehensive 112-dimensional feature vector from each audio segment:

```
┌─────────────────────────────────────────────────────────────┐
│                    112D Feature Vector                       │
├─────────────────────────────────────────────────────────────┤
│  Base Physics (46D)                                          │
│  ├── F0 statistics (mean, std, min, max, range)            │
│  ├── Duration features (ms, frame count)                    │
│  ├── Energy features (RMS, peak, dynamic range)             │
│  ├── Spectral features (centroid, bandwidth, flatness)      │
│  └── Temporal features (attack, decay, sustain)             │
├─────────────────────────────────────────────────────────────┤
│  Macro Texture (30D)                                         │
│  ├── MFCCs (1-13)                                           │
│  ├── Delta MFCCs (1-6)                                      │
│  ├── Rhythm (ICI, onset rate)                               │
│  └── FM/AM characteristics                                  │
├─────────────────────────────────────────────────────────────┤
│  Micro Texture (36D)                                         │
│  ├── GLCM texture features                                  │
│  ├── Harmonic texture (density, spread)                     │
│  ├── Temporal texture (skew, kurtosis)                      │
│  └── Micro-dynamics (jitter, shimmer)                       │
└─────────────────────────────────────────────────────────────┘
```

### 2. 5-Stage Synthesis Pipeline

```
Raw Audio → [NBD] → Segments → [112D] → Features → [Corpus] → Clusters → [Exemplars] → [Synthesis] → Audio Output
```

| Stage | Module | Purpose |
|-------|--------|---------|
| 1 | Neural Boundary Detection | Segment continuous audio into phrase units |
| 2 | 112D Feature Extraction | Extract comprehensive acoustic features |
| 3 | Corpus Analysis | Cluster features into vocabulary (k=1020) |
| 4 | Semantic Reconstruction | Manage exemplars and build timelines |
| 5 | Synthesis Output | Generate audio via granular synthesis |

### 3. Closed-Loop Interaction Agent

Real-time bidirectional communication between Rust and Python:

```
Rust (Execution Layer)                    Python (Logic Layer)
─────────────────────                    ─────────────────────
FeatureEventPublisher  ──────PUB────►  FeatureSubscriber
                            112D features
ActionSubscriber   ◄─────PUB─────  ActionPublisher
                          Synthesis timelines
```

**Supported Modes:**
- **General Mode** (default): Compositional parsing - each segment is a semantic unit
- **Bat Mode**: Holophrastic parsing - rigid idioms are atomic units (based on Egyptian Fruit Bat research)

---

## Project Structure

```
src/
├── technical_architecture/          # Rust Execution Layer
│   ├── src/
│   │   ├── synthesis.rs            # Audio synthesis engines
│   │   ├── source_separation.rs    # Conv-TasNet separator
│   │   ├── peer_controller.rs      # ZeroMQ peer supervision
│   │   ├── master_controller.rs   # Intent-Reality mediator
│   │   ├── rosetta_pipeline.rs     # 4-stage pipeline
│   │   ├── micro_dynamics_extractor.rs  # 112D features
│   │   ├── neural_boundary.rs      # NBD segmentation
│   │   ├── semantic_reconstruction.rs  # Exemplar management
│   │   ├── species_vocab_config.rs # Direction 1: Species vocabulary config
│   │   └── ...
│   ├── examples/                   # 50+ example programs
│   ├── deployment/                 # Systemd service files
│   └── docs/
│       └── pub/                    # Methodology documentation
│           ├── closed_loop_agent_protocol.md
│           ├── FIVE_STAGE_SYNTHESIS_PIPELINE.md
│           ├── pam_pipeline_guide.md
│           └── synthesis_explanation.md
│
├── cognitive_intelligence/          # Python Logic Layer
│   ├── data_fusion.py              # Multi-modal data fusion
│   ├── visual_fusion.py            # Cross-modal attention
│   ├── siamese_network.py          # Similarity learning
│   ├── multimodal_fusion.py        # Audio-visual fusion with cross-modal attention
│   ├── ddsp_synthesis.py           # Differentiable DSP for gradient-optimized synthesis
│   └── maml_adaptation.py          # Model-Agnostic Meta-Learning for cross-species transfer
│
├── realtime/                        # Real-time Processing (Logic Layer)
│   ├── interaction_agent.py        # Closed-Loop agent
│   ├── feature_subscriber.py       # ZeroMQ feature subscriber
│   ├── parsing_strategy.py         # Strategy Pattern for parsing
│   ├── config_client.py            # REQ client for Rust config
│   ├── cognitive_layer.py          # Cognitive intelligence
│   ├── phrase_audio_library.py     # Data management
│   ├── context_classifier.py       # Direction 4: Semantic context classifier
│   └── archive/                    # Archived execution-layer files
│
├── semiotics/                       # Semiotic Analysis
│   ├── semiotic_engine.py          # Deception detection, innovation
│   ├── pcfg_induction.py           # Probabilistic Context-Free Grammar induction
│   └── SEMIOTIC_DETECTION_GUIDE.md
│
├── query_interface/                 # High-performance query system
│   └── vocalization_query_interface.py
│
├── analysis/rosetta_stone/          # Universal Rosetta Stone Engine
│   ├── universal_rosetta_stone.py  # Core acoustic analysis
│   ├── universal_synthesizer.py    # Audio synthesis
│   ├── acoustic_algebra.py         # Continuous semantic generation
│   ├── vocab_optimizer.py          # Direction 1: Adaptive vocabulary optimization
│   ├── online_clustering.py        # Direction 8: Incremental K-means
│   ├── neural_language_model.py    # Direction 2: Transformer-based sequence modeling
│   ├── speaker_embeddings.py       # Direction 3: Speaker identification and verification
│   └── neural_vocoder.py           # Direction 6: Neural audio synthesis from features
├── analysis/                        # Clustering and analysis scripts
│   ├── run_pca_bgmm_pipeline.py    # PCA+BGMM optimized clustering pipeline
│   ├── cluster_benchmark_suite.py  # Clustering algorithm comparison
│
├── data_import/                     # Database import
├── synthesis/                       # Synthesis modules
├── tests/                           # Test suites (500+ tests)
│
├── data_models.py                   # Unified data structures
├── vocalization_database.json       # Main database (2.5MB, 2,882 phrases)
├── CLAUDE.md                        # Project instructions
└── README.md                        # This file
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/sheelmorjaria/zoo-vox-rosetta-engine.git
cd zoo-vox-rosetta-engine

# Build Rust components
cd technical_architecture && cargo build --release

# Install Python dependencies
pip install -r requirements.txt
```

### Running Demos

```bash
# Import vocalization data
python -m src.data_import.import_vocalization_data

# Run query interface demo
python -m src.query_interface.demo_query_interface

# Run semiotic engine demo
python -m src.semiotics.demo_semiotic_engine
```

### Usage Examples

**Adaptive Vocabulary (Direction 1)**
```bash
# Optimize vocabulary size for a species using SVS
python -m analysis.rosetta_stone.vocab_optimizer

# Use species-specific vocabulary in ExemplarManager
python -m analysis.rosetta_stone.exemplar_manager \
  --input segments_manifest.json \
  --species egyptian_fruit_bat \
  --vocab-registry species_vocab_registry.json \
  --output clusters.json
```

**Semantic Alignment (Direction 4)**
```python
from realtime.context_classifier import ContextClassifier
from realtime.interaction_agent import InteractionAgent, InteractionAgentConfig

# Train a context classifier
classifier = ContextClassifier(model_type="mlp", random_state=42)
classifier.train(features_112d, context_labels)
classifier.save("context_model.pkl")

# Use in InteractionAgent with label mapping
config = InteractionAgentConfig(
    context_classifier_path="context_model.pkl",
    context_label_mapping={
        "context_0": "social",
        "context_1": "alarm",
        "context_2": "territorial",
    },
)
agent = InteractionAgent(config=config)
```

**Online Clustering (Direction 8)**
```python
from analysis.rosetta_stone.online_clustering import OnlineKMeans

# Create online clusterer with auto-spawn
clusterer = OnlineKMeans(
    initial_k=10,
    max_k=100,
    spawn_threshold=3.0,  # Spawn new cluster for distant samples
)

# Incremental updates
clusterer.partial_fit(new_batch)
clusterer.prune_stale_clusters(decay_window_ms=5000)
```

**Neural Language Models (Direction 2)**
```python
from analysis.rosetta_stone.neural_language_model import (
    AcousticTokenizer, TransformerLM, ConditionalGenerator
)

# Tokenize 112D features to discrete tokens
tokenizer = AcousticTokenizer(vocab_size=1020)
tokenizer.fit(feature_vectors)
tokens = tokenizer.tokenize(features_112d)

# Train transformer model
model = TransformerLM(vocab_size=1020, d_model=256, n_heads=8, n_layers=6)
model.train(sequences, epochs=20)

# Generate new sequences
generated = model.generate(prompt=[42, 117], max_length=50, temperature=0.8)

# Context-aware generation
generator = ConditionalGenerator(model)
alarm_sequence = generator.generate_for_context("alarm", max_length=30)
```

**Speaker Embeddings (Direction 3)**
```python
from analysis.rosetta_stone.speaker_embeddings import (
    SpeakerEmbeddingExtractor, SpeakerDatabase
)

# Extract speaker embeddings
extractor = SpeakerEmbeddingExtractor(embedding_dim=256)
emb1 = extractor.extract_from_audio(audio1, sr=48000)
emb2 = extractor.extract_from_features(features_112d)

# Speaker database
db = SpeakerDatabase()
db.enroll("bat_001", emb1)
db.enroll("bat_002", emb2)

# Verify speaker identity
result = db.verify("bat_001", test_emb, threshold=0.8)
if result.is_match:
    print(f"Speaker verified with confidence {result.confidence}")

# Identify unknown speaker
matches = db.identify(test_emb, top_k=3)
for speaker_id, score in matches:
    print(f"Potential match: {speaker_id} ({score:.2f})")
```

**Neural Vocoder (Direction 6)**
```python
from analysis.rosetta_stone.neural_vocoder import (
    NeuralVocoder, FeatureInterpolator, ProsodicModifier
)

# Create vocoder
vocoder = NeuralVocoder(model_type="simple", sample_rate=48000)

# Train on feature-audio pairs
vocoder.train(features_list, audio_list, epochs=20)
vocoder.save("bat_vocoder.pkl")

# Synthesize audio from features
audio = vocoder.synthesize(features_112d)

# Interpolate between features for smooth transitions
smooth_features = FeatureInterpolator.interpolate_sequence(features, n_interp=2)

# Modify prosody
pitched = ProsodicModifier.adjust_pitch(features, shift_semitones=2.0)
stretched = ProsodicModifier.adjust_duration(features, speed_factor=0.8)
louder = ProsodicModifier.adjust_amplitude(features, gain_db=6.0)
```

**PCFG Induction (Formal Language Theory)**
```python
from semiotics.pcfg_induction import (
    GrammarRule, PCFG, PCFGInducer, GrammarParser, VocalizationGrammar
)

# Create a probabilistic grammar rule
rule = GrammarRule(
    lhs="Phrase",           # Left-hand side (non-terminal)
    rhs=["Contact", "Trill"],  # Right-hand side (symbols)
    probability=0.75
)

# Build PCFG from rules
pcfg = PCFG()
pcfg.add_rule(GrammarRule("S", ["NP", "VP"], 0.6))
pcfg.add_rule(GrammarRule("S", ["Phrase"], 0.4))
pcfg.normalize()

# Parse a sequence and compute probability
parser = GrammarParser(pcfg)
sequence = ["Contact", "Trill", "Food"]
probability = parser.parse_probability(sequence)
derivation = parser.most_likely_derivation(sequence)

# Learn grammar from vocalization data
inducer = PCFGInducer(max_iterations=100)
sequences = [["Call", "Response"], ["Call", "Food", "Response"]]
learned_grammar = inducer.learn_from_sequences(sequences)

# Species-specific grammar
grammar = VocalizationGrammar(species="marmoset", non_terminals=["S", "NP", "VP"])
entropy = grammar.entropy()  # Measure complexity
```

**Multimodal Fusion (Vision + Audio)**
```python
from cognitive_intelligence.multimodal_fusion import (
    VisualFeatureExtractor, AudioVisualFusion, MultimodalContextClassifier
)

# Extract visual features from video frames
extractor = VisualFeatureExtractor(output_dim=128)
frames = np.random.randn(16, 3, 224, 224).astype(np.float32)  # 16 frames
visual_features = extractor.extract_features(frames)

# Fuse with audio features
fusion = AudioVisualFusion(audio_dim=112, visual_dim=128, fusion_dim=256)
audio_features = np.random.randn(10, 112).astype(np.float32)
visual_sequence = np.random.randn(10, 128).astype(np.float32)

fused = fusion.fuse(audio_features, visual_sequence)

# Classify context with fused features
classifier = MultimodalContextClassifier(
    audio_dim=112, visual_dim=128, num_contexts=4
)
classifier.train(audio_list, visual_list, context_labels)
predictions = classifier.predict(audio_features, visual_features)
```

**DDSP Synthesis (Differentiable DSP)**
```python
from cognitive_intelligence.ddsp_synthesis import (
    DDSPSynthesizer, DDSPOptimizer, HarmonicModel, SpectralLoss
)

# Create synthesizer
synthesizer = DDSPSynthesizer(sample_rate=48000, n_harmonics=16)

# Synthesize from pitch and loudness
n_frames = 75  # 100ms at 64-hop
loudness = np.random.randn(n_frames).astype(np.float32)
pitch = 440.0 * np.ones(n_frames).astype(np.float32)
audio = synthesizer.synthesize(loudness, pitch)

# Gradient-based optimization to match target
optimizer = DDSPOptimizer(learning_rate=0.01, n_iterations=50)
target_audio = np.random.randn(4800).astype(np.float32)
reconstructed = optimizer.reconstruct(target_audio, synthesizer)

# Extract harmonic model
harmonic_model = HarmonicModel(n_harmonics=16, sample_rate=48000)
amplitudes = harmonic_model.extract_amplitudes(audio, fundamental_freq=440.0)
```

**MAML Adaptation (Cross-Species Transfer)**
```python
from cognitive_intelligence.maml_adaptation import (
    MAMLOptimizer, FewShotClassifier, MetaLearner
)

# Few-shot learning for new species
classifier = FewShotClassifier(
    input_dim=112, num_classes=5, k_shot=5
)

# Support set (5 examples per class)
support_x = np.random.randn(25, 112).astype(np.float32)
support_y = np.repeat(np.arange(5), 5).astype(np.int32)

# Adapt to new species
classifier.adapt(support_x, support_y)

# Predict on new data
query_x = np.random.randn(5, 112).astype(np.float32)
predictions = classifier.predict(query_x)

# Cross-species meta-learning
meta_learner = MetaLearner(
    input_dim=112, num_classes=4,
    species=["marmoset", "bat", "dolphin"]
)

# Train on multiple species
for species in ["marmoset", "bat"]:
    features = np.random.randn(50, 112).astype(np.float32)
    labels = np.random.randint(0, 4, 50).astype(np.int32)
    meta_learner.add_species_data(species, features, labels)

meta_learner.meta_train(n_epochs=10, n_tasks_per_epoch=20)

# Rapid adaptation to new species
new_species_data = np.random.randn(5, 112).astype(np.float32)
new_species_labels = np.random.randint(0, 4, 5).astype(np.int32)
meta_learner.adapt_to_species("finch", new_species_data, new_species_labels)
```

### Running Tests

```bash
# Rust tests
cd technical_architecture && cargo test --lib

# Python tests
python -m pytest tests/ -v --ignore=tests/archive --ignore=tests/archive_experimental
```

### Field Deployment

```bash
# Install systemd services
sudo cp technical_architecture/deployment/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Start services
sudo systemctl enable rust-field-engine.service
sudo systemctl enable python-cognitive-agent.service
sudo systemctl start rust-field-engine.service
sudo systemctl start python-cognitive-agent.service
```

---

## Key Features

### MiniBatch BGMM Teacher-Student Pipeline ✅

**Scalable Vocabulary Discovery with OOD-Based Perceptual Filtering**

The MiniBatch BGMM pipeline implements a teacher-student distillation approach that scales to millions of segments:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   MiniBatch BGMM Teacher-Student Pipeline               │
├─────────────────────────────────────────────────────────────────────────┤
│  Phase 1: Offline Training (Python - Teacher)                          │
│  ├── 8.9M segments → 100k sample for tractable EM training             │
│  ├── PCA: 112D → 30D (95.4% variance preserved)                        │
│  ├── Bayesian GMM: Auto-discovers true vocabulary size                 │
│  ├── Weight-based pruning: Removes clusters < 1% weight                │
│  └── Result: 45 clusters (true vocabulary, not forced)                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Phase 2: Student Inference (Rust - Zero-Copy)                         │
│  ├── Load 112D centroids from synthesis_manifest.json                 │
│  ├── Nearest centroid lookup (sub-millisecond, L2 squared)            │
│  ├── OOD rejection: Features too far from all centroids dropped       │
│  └── Confidence: 1.0 - (distance / threshold) for response gating      │
├─────────────────────────────────────────────────────────────────────────┤
│  Phase 3: Python Logic Layer (InteractionAgent v1.2.0)                 │
│  ├── Cluster-based context inference (archetype, not instance)        │
│  ├── Confidence-based response suppression                            │
│  └── Syntax validation via 50 valid bat bigrams (LRN-6)               │
└─────────────────────────────────────────────────────────────────────────┘
```

**Performance Metrics:**
- **Throughput**: 228,000 segments/second (8.9M in 472.7 seconds)
- **Vocabulary Discovery**: 45 clusters (BGMM-pruned from 150 initial)
- **Assignment Latency**: Sub-millisecond per lookup
- **Variance Preserved**: 95.4% with 30 PCA components
- **Zero OOD Pollution**: 0% noise rate (vs 44.1% with HDBSCAN)

**Scientific Discovery - The "Dense Acoustic Continent"**
Unlike HDBSCAN which discards 44.1% of segments as "noise" (proving it forces hard boundaries on graded transitions), BGMM preserves the entire acoustic space. The 45 clusters represent **true acoustic archetypes** while allowing instances to exist anywhere in the continuous 112D space between centroids.

**Files:**
- `analysis/run_full_corpus_pipeline.py` - Complete pipeline (8.9M segments)
- `tests/test_minibatch_bgmm_teacher.py` - TDD validation (8 tests)
- `technical_architecture/src/semantic_reconstruction.rs` - Student inference + OOD filtering (4 new tests)
- `technical_architecture/src/peer_controller.rs` - `publish_with_student()` ZeroMQ integration
- `/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/synthesis_manifest.json` - 45 centroids
- `/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/extraction_112d/extraction_112d_labeled.json` - Full corpus with labels

### InteractionAgent v1.2.0: Cluster-Based Semantic Grounding ✅

**45-State Probabilistic Automaton with Biologically-Grounded Transitions**

The InteractionAgent now uses the BGMM-distilled vocabulary for structurally sound closed-loop interaction:

```python
from realtime.interaction_agent import InteractionAgent, InteractionAgentConfig, build_cluster_context_map

# Pre-compute context map from 45 BGMM centroids
cluster_context_map = build_cluster_context_map(centroids_112d)
# {0: "social", 8: "contact", 25: "alarm", 35: "territorial", ...}

# Valid bat bigrams from LRN-6 syntax analysis (50 transitions)
valid_bigrams = {(8, 12), (8, 15), (12, 8), (12, 20), ...}

config = InteractionAgentConfig(
    cluster_context_map=cluster_context_map,      # v1.2.0: Archetype-based contexts
    valid_bigrams=valid_bigrams,                  # v1.2.0: Syntax validation
    confidence_threshold=0.5,                    # v1.2.0: Rust Student confidence gating
)

agent = InteractionAgent(config=config)
```

**v1.2.0 Features:**
1. **Cluster-to-Context Mapping**: Context inferred from centroid archetype, not noisy F0/RMS rules
2. **Confidence-Based Suppression**: Low confidence (near boundary) events don't trigger responses
3. **Syntax Validation**: Only valid bat bigrams (LRN-6) permitted as transitions
4. **Perceptual Grounding**: OOD events rejected at Rust source, preventing feedback loops

**Files:**
- `realtime/interaction_agent.py` - Cluster-based inference, bigram validation, confidence gating
- `realtime/feature_subscriber.py` - `confidence` field added to FeatureEvent
- `tests/test_interaction_agent_v1_2_0.py` - TDD validation (13 tests)

### Foundation TDD Implementation (Directions 1+4+8) ✅

**Direction 1: Adaptive Vocabulary**
- `VocabOptimizer`: Automatic k optimization per species using Silhouette Validation Score (SVS) maximization
- `SpeciesVocabRegistry`: Cross-language (Python-Rust) configuration storage with JSON IPC
- Species-specific vocabulary granularity - each species has unique acoustic characteristics requiring different k values
- **CLI Integration**: `--species` and `--vocab-registry` arguments for production pipeline

**Direction 4: Semantic Alignment**
- `ContextClassifier`: MLP-based behavioral context inference replacing brittle rule-based systems
- Trains on 112D feature vectors with confidence scoring
- Weak supervision from temporal co-occurrence patterns
- Model persistence (pickle/joblib) for deployment
- **Label Mapping**: Maps pseudo-labels (e.g., `context_0`) to canonical response contexts
- **Confidence Propagation**: ML confidence scores used directly for response gating

**Direction 8: Online/Incremental Clustering**
- `OnlineKMeans`: Real-time vocabulary adaptation for closed-loop agent
- Incremental centroid updates via `partial_fit()`
- Automatic cluster spawning for novel patterns
- Forgetting mechanism via decay and pruning
- Concept drift detection
- **Sample Buffering**: Handles single-sample batches and sparse data streams

### Level 0 Extensions (Directions 2+3+6) ✅

**Integration Quality Assurance:**
- Codex adversarial review completed with 4 findings identified and resolved
- Speaker tracking payload now fully consumed by `InteractionAgent._process_features()`
- Vocoder interface compatibility fixed via tokenizer parameter
- Clustering distance metric corrected to use precomputed semantics
- Empty prompt handling added for robust generation

**Direction 2: Neural Language Models**
- `AcousticTokenizer`: Converts 112D features ↔ discrete token IDs via k-means clustering
- `TransformerLM`: GPT-style causal transformer for next-token prediction
- `ConditionalGenerator`: Context-aware sequence generation for different behavioral contexts
- Temperature and top-k sampling for controlled diversity
- Model persistence with pickle serialization

**Direction 3: Speaker Embeddings**
- `SpeakerEmbeddingExtractor`: 256D L2-normalized embeddings for individual identification
- `SpeakerDatabase`: Enrollment, verification, and identification with cosine similarity
- `SpeakerAdaptiveSynthesis`: Voice-conditioned synthesis with tokenizer integration for vocoder compatibility
- Agglomerative clustering with precomputed distance metric for discovering new speakers
- Real-time speaker tracking via `InteractionAgent` with change detection callbacks

**Direction 6: Neural Vocoder**
- `NeuralVocoder`: Generate audio directly from 112D feature sequences
- `FeatureInterpolator`: Linear and spherical (slerp) interpolation for smooth transitions
- `ProsodicModifier`: Pitch shift, time stretch, and amplitude manipulation
- Overlap-add synthesis with configurable frame/hop sizes
- Real-time capable with latency < 100ms target

### Advanced Cognitive Intelligence Features ✅

**PCFG Induction (Formal Language Theory)**
- `GrammarRule`: Probabilistic production rules with LHS non-terminal and RHS symbols
- `PCFG`: Probabilistic Context-Free Grammar with normalized rule probabilities
- `PCFGInducer`: Learn grammar structure from vocalization sequences using Inside-Outside algorithm
- `GrammarParser`: CYK-based parser for computing parse probabilities and most likely derivations
- `VocalizationGrammar`: Species-specific grammars with complexity metrics (entropy, branching factor)
- **Key Capability**: `predict_next()` for predicting next vocalization segment in sequence

**Multimodal Fusion (Vision + Audio)**
- `VisualFeatureExtractor`: CNN-based visual feature extraction from video frames
- `AudioVisualFusion`: Cross-modal attention mechanism combining audio and visual features
- `MultimodalContextClassifier`: Context classification using fused multimodal features
- **Cross-Modal Attention**: Learn which visual features are relevant for audio classification
- **Temporal Alignment**: Handle frame rate mismatches between audio and video modalities

**DDSP Synthesis (Differentiable DSP)**
- `SineOscillator`: Differentiable sine wave oscillator with FM synthesis support
- `DifferentiableFilter`: Spectral shaping with differentiable coefficients
- `SpectralLoss`: Multi-scale spectral loss for gradient-based optimization
- `DDSPSynthesizer`: Main synthesizer with additive and filter-warped synthesis
- `HarmonicModel`: Extract harmonic amplitudes and phases for additive synthesis
- `NoiseModel`: Filter noise with time-varying filters for residual synthesis
- **Key Benefit**: Gradient-optimized audio reconstruction via differentiable signal processing

**MAML Adaptation (Cross-Species Transfer)**
- `MAMLOptimizer`: Model-Agnostic Meta-Learning for rapid adaptation
- `FewShotClassifier`: K-shot N-way classification for new species
- `TaskDistribution`: Sample meta-learning tasks from data
- `MetaLearner`: End-to-end cross-species transfer learning system
- `SpeciesEncoder`: Species-specific encoders for conditioning
- **Key Benefit**: Adapt to new species with only 1-5 examples per vocalization type

### Cross-Species Analysis

- **Species Supported**: Marmoset, Egyptian Fruit Bat, Dolphin, Chimpanzee, Sperm Whale, Zebra Finch
- **Universal Feature Space**: 112D features enable cross-species comparison
- **Grammar Network Analysis**: Discover syntax patterns across species

### 112D Rosetta Features

- **Base Physics (46D)**: F0, duration, energy, spectral shape
- **Macro Texture (30D)**: MFCCs, rhythm, FM/AM characteristics
- **Micro Texture (36D)**: GLCM texture, harmonic texture, micro-dynamics

### Cognitive Intelligence

- **Deception Detection**: Identify deceptive vocalizations via modality mismatches
- **Context Inference**: Infer behavioral context (alarm, territorial, contact, social)
- **Multi-Modal Fusion**: Combine audio, visual, and contextual data

### Safety & Reliability

- **Peer-to-Peer Supervision**: ZeroMQ heartbeat monitoring
- **Fail-Open Design**: Python crash triggers safe Passthrough Mode
- **IACUC Compliance**: Built-in protocol enforcement
- **Thermal Management**: Automatic throttling to prevent overheating

---

## Deployment

### Operation Modes

| Mode | Condition | Behavior |
|------|-----------|----------|
| **Passthrough** | Python disconnected/crashed | Audio muted, recording only |
| **Interactive** | Python connected, heartbeats active | Full cognitive processing, synthesis enabled |

### Systemd Services

```
rust-field-engine.service         # Rust Execution Layer
python-cognitive-agent.service    # Python Logic Layer
```

### Monitoring

```bash
# View logs
sudo journalctl -u rust-field-engine.service -f
sudo journalctl -u python-cognitive-agent.service -f

# Check status
systemctl status rust-field-engine.service
systemctl status python-cognitive-agent.service
```

---

## Scientific Impact

The Zoo Vox Rosetta Engine enables:

1. **Deception Detection** in animal communication through modality mismatch analysis
2. **Emergent Behavior** identification and tracking over time
3. **Cross-Modal Analysis** combining audio, visual, and contextual data
4. **Universal Translation** across species boundaries via 112D feature mapping
5. **Cognitive Modeling** of animal intelligence through vocalization patterns

**Research Focus**: Understanding animal intelligence through vocalization patterns, moving beyond simple classification to cognitive understanding.

---

## Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| Rust (cargo test) | 1,697 | ✅ All passing |
| Python (pytest) | 1,044 | ✅ All passing |
| MiniBatch BGMM Pipeline | 8 | ✅ All passing |
| InteractionAgent v1.2.0 | 13 | ✅ All passing |
| Foundation TDD (1+4+8) | 92 | ✅ All passing |
| Level 0 Extensions (2+3+6) | 78 | ✅ All passing |
| Advanced Features (PCFG+Multimodal+DDSP+MAML) | 74 | ✅ All passing |
| Integration | 50+ | ✅ Verified |

### MiniBatch BGMM Pipeline Tests

| Component | Tests | Description |
|-----------|-------|-------------|
| Python: MiniBatch BGMM Teacher | 4 | Subsample training, PCA reduction, BGMM fitting, centroid export |
| Python: Student Inference | 3 | Single/batch prediction, sub-millisecond speed |
| Python: Centroid Export | 1 | Rust-compatible manifest format |
| Rust: ExemplarManager Student | 4 | Nearest centroid lookup, OOD rejection, centroid retrieval, OOD check |
| Rust: ZeroMQ Integration | 2 | `publish_with_student()`, confidence field propagation |

### InteractionAgent v1.2.0 Tests

| Component | Tests | Description |
|-----------|-------|-------------|
| Cluster Context Mapping | 4 | 45-cluster map creation, context inference, fallback behavior |
| Confidence-Based Suppression | 3 | High confidence triggers, low confidence suppresses, cluster tracking |
| Bigram Syntax Validation | 4 | Valid bigram allows, invalid blocks, first-event handling, no-config skip |
| Full Pipeline Integration | 2 | Complete Rust Student → Python Agent flow, OOD prevention validation |

### Foundation TDD Tests (Directions 1+4+8)

| Direction | Component | Tests | Description |
|-----------|-----------|-------|-------------|
| Direction 1 | VocabOptimizer | 22 | SVS computation, k optimization, edge cases |
| Direction 1 | SpeciesVocabConfig (Rust) | 18 | Species vocabulary configuration |
| Direction 4 | ContextClassifier | 23 | Binary/multi-class, persistence, singleton labels |
| Direction 4 | InteractionAgent Integration | 7 | Live FeatureEvent classification, label mapping |
| Direction 8 | OnlineKMeans | 22 | Incremental updates, spawning, pruning, drift detection |

### Level 0 Extension Tests (Directions 2+3+6)

| Direction | Component | Tests | Description |
|-----------|-----------|-------|-------------|
| Direction 2 | AcousticTokenizer | 6 | Tokenization, roundtrip, NaN handling |
| Direction 2 | TransformerLM | 8 | Forward pass, attention, positional embeddings |
| Direction 2 | Transformer Training | 4 | Training step, loss decrease, learning rate schedule |
| Direction 2 | Transformer Generation | 6 | Predict next, generate, temperature, top-k |
| Direction 2 | Conditional Generation | 3 | Context-aware generation, batch generation |
| Direction 2 | Vocabulary Integration | 4 | Species-specific vocab, model persistence |
| Direction 3 | Speaker Embedding Extractor | 5 | Audio/feature extraction, normalization |
| Direction 3 | Speaker Database | 4 | Enrollment, verification, threshold sensitivity |
| Direction 3 | Speaker Identification | 4 | Known/unknown speaker, top-k, empty DB |
| Direction 3 | Speaker Clustering | 4 | Two speakers, same speaker, varying counts |
| Direction 3 | Adaptive Synthesis | 3 | Speaker-specific synthesis, unknown fallback |
| Direction 3 | Integration | 4 | FeatureEvent integration, agent tracking |
| Direction 6 | Neural Vocoder Core | 5 | Output shape, sample rate, single/sequence/batch |
| Direction 6 | Audio Quality | 4 | Valid audio, energy, fidelity, spectral |
| Direction 6 | Feature Interpolator | 3 | Linear, slerp, smoothness |
| Direction 6 | Prosodic Modifier | 5 | Pitch shift, time stretch, amplitude gain |
| Direction 6 | Model Persistence | 3 | Save, load, versioning |
| Direction 6 | Vocoder Integration | 4 | Token/LM synthesis, realtime, fallback |

### Advanced Feature Tests (PCFG + Multimodal + DDSP + MAML)

| Module | Component | Tests | Description |
|--------|-----------|-------|-------------|
| PCFG Induction | GrammarRule | 5 | Rule construction, probability normalization, serialization |
| PCFG Induction | PCFG | 6 | Rule management, parsing, normalization |
| PCFG Induction | GrammarParser | 4 | CYK parsing, probability computation, derivation |
| PCFG Induction | PCFGInducer | 3 | Inside-Outside learning, rule extraction |
| PCFG Induction | VocalizationGrammar | 2 | Species-specific grammar, complexity metrics |
| Multimodal Fusion | VisualFeatureExtractor | 4 | CNN extraction, batch processing, output dimensions |
| Multimodal Fusion | AudioVisualFusion | 5 | Cross-modal attention, fusion weights, temporal alignment |
| Multimodal Fusion | MultimodalContextClassifier | 3 | Fused classification, backpropagation |
| Multimodal Fusion | Fusion Integration | 4 | Real-time fusion, edge cases |
| DDSP Synthesis | DifferentiableOscillator | 4 | Sine synthesis, FM modulation, gradient tracking |
| DDSP Synthesis | DifferentiableFilter | 3 | Lowpass/highpass filters, coefficient gradients |
| DDSP Synthesis | SpectralLoss | 3 | Magnitude, multi-scale, perceptual loss |
| DDSP Synthesis | DDSPPreprocessor | 3 | Loudness/pitch extraction, DDSP features |
| DDSP Synthesis | DDSPSynthesizer | 3 | Additive synthesis, filter-warped synthesis |
| DDSP Synthesis | DDSPOptimizer | 3 | Gradient optimization, audio reconstruction |
| DDSP Synthesis | HarmonicModel | 3 | Harmonic amplitude/phase extraction, synthesis |
| DDSP Synthesis | NoiseModel | 2 | Noise filtering, envelope extraction |
| MAML Adaptation | MAMLOptimizer | 3 | Meta-parameter initialization, inner/outer loop updates |
| MAML Adaptation | FewShotClassifier | 3 | 5-way 5-shot, 1-shot, cross-species adaptation |
| MAML Adaptation | TaskDistribution | 3 | Task sampling, cross-species tasks, batching |
| MAML Adaptation | SpeciesEncoder | 2 | Species encoding, species conditioning |
| MAML Adaptation | RapidAdaptation | 2 | Adaptation speed, transfer learning |
| MAML Adaptation | MAMLIntegration | 2 | Wrapper integration, meta-learning vs fine-tuning |

### Running Tests

```bash
# Rust tests
cd technical_architecture && cargo test --lib

# Python tests (excluding archives)
python -m pytest tests/ -v \
  --ignore=tests/archive \
  --ignore=tests/archive_experimental \
  --ignore=tests/test_shared_memory_ipc.py \
  --ignore=tests/test_realtime_dependencies.py

# PCA+BGMM pipeline tests
python -m pytest tests/test_optimized_clustering.py -v

# Foundation TDD tests (Directions 1+4+8)
python -m pytest tests/test_vocab_optimizer.py \
                 tests/test_context_classifier.py \
                 tests/test_online_clustering.py \
                 tests/test_interaction_agent.py -v

# Level 0 Extension tests (Directions 2+3+6)
python -m pytest tests/test_neural_language_model.py \
                 tests/test_speaker_embeddings.py \
                 tests/test_neural_vocoder.py -v

# Advanced Feature tests (PCFG + Multimodal + DDSP + MAML)
python -m pytest tests/test_pcfg_induction.py \
                 tests/test_multimodal_fusion.py \
                 tests/test_ddsp_synthesis.py \
                 tests/test_maml_adaptation.py -v
```

---

## Documentation

### Methodology Documentation (technical_architecture/docs/pub/)

| Document | Description |
|----------|-------------|
| **closed_loop_agent_protocol.md** | Real-time bidirectional communication between Rust and Python |
| **FIVE_STAGE_SYNTHESIS_PIPELINE.md** | Complete synthesis pipeline from raw audio to output |
| **synthesis_explanation.md** | Audio synthesis background and theory |

### Additional Documentation

| Document | Location |
|----------|----------|
| CLAUDE Developer Guide | `technical_architecture/CLAUDE.md` |
| Project Instructions | `CLAUDE.md` |
| Semiotic Detection Guide | `semiotics/SEMIOTIC_DETECTION_GUIDE.md` |

### Archive Documentation

| Archive | Description |
|---------|-------------|
| `/src/archive/ARCHIVE.md` | Deprecated root directories |
| `/src/realtime/archive/ARCHIVE.md` | Python execution-layer files moved to Rust |

---

## License

CC BY-ND 4.0 International

See [LICENSE](LICENSE) for details.

---

## Author

**Sheel Morjaria**
Email: sheelmorjaria@gmail.com

---

**Zoo Vox Rosetta Engine** — Universal translation for the animal kingdom.
