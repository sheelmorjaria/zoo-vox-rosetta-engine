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
│   └── siamese_network.py          # Similarity learning
│
├── realtime/                        # Real-time Processing (Logic Layer)
│   ├── interaction_agent.py        # Closed-Loop agent
│   ├── feature_subscriber.py       # ZeroMQ feature subscriber
│   ├── parsing_strategy.py         # Strategy Pattern for parsing
│   ├── config_client.py            # REQ client for Rust config
│   ├── cognitive_layer.py          # Cognitive intelligence
│   ├── phrase_audio_library.py     # Data management
│   └── archive/                    # Archived execution-layer files
│
├── semiotics/                       # Semiotic Analysis
│   ├── semiotic_engine.py          # Deception detection, innovation
│   └── SEMIOTIC_DETECTION_GUIDE.md
│
├── query_interface/                 # High-performance query system
│   └── vocalization_query_interface.py
│
├── analysis/rosetta_stone/          # Universal Rosetta Stone Engine
│   ├── universal_rosetta_stone.py  # Core acoustic analysis
│   ├── universal_synthesizer.py    # Audio synthesis
│   └── acoustic_algebra.py         # Continuous semantic generation
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
| Rust (cargo test) | 1,652 | ✅ All passing |
| Python (pytest) | 729 | ✅ All passing |
| Integration | 50+ | ✅ Verified |

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
