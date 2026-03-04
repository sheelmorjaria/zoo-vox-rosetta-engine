Here is the revised `README.md`, updated to reflect the full scope of the Zoo Vox Rosetta Engine, including the Neural Boundary Detection segmentation strategy and the 112D feature architecture.

```markdown
# Zoo Vox Rosetta Engine

**A Cognitive Architecture for Cross-Species Identification, Linguistic Discovery, and Two-Way Synthesis.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Paper](https://img.shields.io/badge/Publication-Preprint-blue.svg)](#publication)

The Zoo Vox Rosetta Engine is a bioacoustic analysis framework designed to map the "Linguistic Topology" of animal communication. It moves beyond simple classification to distinguish between **Discrete Syntax** (songbirds), **Syntactic Prosody** (bats), and **True Graded Continua** (marmosets).

---

## Key Features

*   **112D Micro-Dynamics Stack:** A hierarchical feature vector capturing Physics, Texture, and Temporal Dynamics (ADSR, Jitter, Shimmer).
*   **Neural Boundary Detection (NBD):** Semantic segmentation that finds acoustic gestures within continuous graded streams.
*   **Linguistic Topology Mapping:** Automated discovery of atomic clusters and syntactic N-gram templates.
*   **Social Network Analysis:** Mapping of colony interaction dynamics (Hub-and-Spoke, Dyadic pairs).
*   **Rosetta Synthesis Library:** A granular synthesis engine for generating novel vocalizations using validated syntactic templates.

---

## Architecture

The engine operates in three distinct phases: **Perception**, **Cognition**, and **Action**.

### 1. Perception: The 112D Feature Stack

Standard features (e.g., MFCCs) fail to resolve the fine-grained details of graded vocalizations. Our 112D stack captures the full bio-acoustic spectrum:

| Layer | Dimensions | Components | Purpose |
| :--- | :--- | :--- | :--- |
| **Layer 1** | **46D** | F0, Duration, HNR, ICI, **ADSR Envelope** | Base Physics & Temporal Shape. |
| **Layer 2** | **30D** | Harmonic Ratios, Pitch Geometry, GLCM | Macro Texture (Timbre). |
| **Layer 3** | **36D** | AM/FM Spectra, Rhythm Histograms, **Jitter/Shimmer**, Spectral Flux | Micro Texture & Perturbations (Identity). |

### 2. Cognition: Segmentation & Linguistics

#### Neural Boundary Detection (NBD)
Standard energy-based detection fragments graded calls. NBD solves this by monitoring the derivative of the feature stream.
*   **Mechanism:** Detects semantic shifts in spectral texture and pitch geometry.
*   **Output:** Isolates "acoustic gestures" (e.g., the rise and fall of a single FM sweep) rather than just silences.

#### Linguistic Topology Mapping
*   **Atomic Clustering:** Uses HDBSCAN on the 112D space to determine if the species uses discrete "islands" of sound or a "dense cloud" (Graded Continuum).
*   **Syntactic Mining:** Discretizes the continuous manifold into acoustic states and mines N-grams to find reusable "sentence templates."

### 3. Action: Synthesis

The engine includes a **Rosetta Synthesis Library** (`rosetta_synthesis_library/`), which acts as a "Lego Set" for animal communication:
*   **Grains:** Representative audio segments for each acoustic state.
*   **Templates:** Validated N-gram patterns associated with behavioral contexts.
*   **Usage:** "Speak Bat" by assembling grains according to templates.

---

## Installation

### Prerequisites
*   Rust (1.70+)
*   Cargo
*   (Optional) Python 3.8+ for visualization scripts.

### Build

```bash
git clone https://github.com/sheelmorjaria/zoo-vox-rosetta-engine.git
cd zoo-vox-rosetta-engine
cargo build --release
```

---

## Usage

### 1. Feature Extraction
Extract the 112D feature vector from audio files.

```bash
cargo run --release --bin extract_features -- --input audio.wav --output features.json
```

### 2. Neural Boundary Detection (Segmentation)
Segment a long recording into semantic units using NBD.

```bash
cargo run --release --bin segment_audio -- --input recording.wav --output segments/
```

### 3. Syntax Mining
Mine for reusable syntactic patterns (N-grams) from a cache of segments.

```bash
cargo run --release --bin mine_syntax -- --cache segment_cache/ --output syntax_results.json
```

### 4. Social Network Analysis
Analyze interaction graphs from Emitter/Receiver metadata.

```bash
cargo run --release --bin analyze_social -- --manifest annotations.json
```

---

## Scientific Findings

This engine was used to analyze the Egyptian Fruit Bat (*Rousettus aegyptiacus*), leading to the discovery of **"Syntactic Prosody."**

| Species | Atomic Strategy | Syntactic Strategy | Topology |
| :--- | :--- | :--- | :--- |
| **Bengalese Finch** | Discrete | Discrete | Combinatorial Syntax |
| **Egyptian Fruit Bat** | **Graded** | **Discrete** | **Syntactic Prosody** |
| **Marmoset** | Graded | Graded | True Continuum |

**Key Discovery:** While individual bat calls are acoustically graded (improvised notes), they are assembled using rigid, reusable syntactic templates (e.g., `[391, 391, 391]` for territorial contexts). This refutes the "Minimal Signal" hypothesis for this species.

---

## Dataset Requirements

To replicate these findings, datasets must meet the **"Data Spectrum"** requirements:

*   **Level 1 (Minimal):** Sequential Audio + Context Labels.
*   **Level 2.5 (Functional):** Sequential Audio + Context + **Emitter ID**.
*   **Level 3 (Interaction-Ready):** Sequential Audio + Context + Emitter ID + **Receiver ID**.

---

## Publication

**Title:** *Syntactic Prosody in Graded Vocalizations: A 112-Dimensional Cognitive Architecture for Decoding Egyptian Fruit Bat Communication*

**Abstract:**
The study introduces the Zoo Vox Rosetta Engine, a cognitive architecture utilizing a novel 112-Dimensional Micro-Dynamics Feature Stack. Analysis of 1.57 million vocalizations revealed a Dense Graded Continuum at the atomic level but discovered a hidden layer of Discrete Syntactic Templates at the sequence level.

**Full Paper:** See `docs/publication.pdf` or the preprint link in the repository.

**Citation:**
```bibtex
@article{morjaria2024zoo,
  title={Syntactic Prosody in Graded Vocalizations},
  author={Morjaria, Sheel},
  journal={bioRxiv},
  year={2024},
  url={https://github.com/sheelmorjaria/zoo-vox-rosetta-engine}
}
```

---

## Project Structure

```text
zoo-vox-rosetta-engine/
├── src/
│   ├── bin/
│   │   ├── extract_features.rs    # 112D extraction
│   │   ├── segment_audio.rs      # NBD implementation
│   │   ├── mine_syntax.rs        # N-gram mining
│   │   └── build_library.rs      # Synthesis library builder
│   ├── features/
│   │   ├── mod.rs
│   │   ├── physics.rs            # Layer 1
│   │   ├── texture.rs            # Layer 2 & 3
│   │   └── perturbation.rs       # Jitter/Shimmer
│   ├── segmentation/
│   │   └── nbd.rs                # Neural Boundary Detection
│   └── lib.rs
├── rosetta_synthesis_library/    # Pre-built synthesis assets
├── docs/
│   └── publication.pdf
└── Cargo.toml
```

---

## Contributing

Contributions are welcome, particularly in:
*   Support for additional species datasets.
*   Optimization of the NBD segmentation algorithm.
*   Real-time synthesis plugins.

## License

This project is licensed under the Creative Commons Attribution-NoDerivatives 4.0 International License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

*   Dataset: [Egyptian Fruit Bat Dataset](https://github.com/earthspecies/library/tree/main/egyptian_fruit_bat) by Yovel et al.
*   Benchmark: [BEANS-Zero](https://github.com/earthspecies/beans-zero) by Ghani et al.
```