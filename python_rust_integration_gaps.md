# Python-Rust Integration Gap Analysis

Analysis of integration gaps between the Python Logic Layer (`realtime/`, `cognitive_intelligence/`, `semiotics/`, `src/`) and the Rust Execution Layer (`technical_architecture/`).

---

## Critical

### 1. Duplicated Acoustic Grammar Data

The bat grammar (valid bigrams, rigid idioms, opener/closer sets) is hardcoded independently in both layers:

- **Rust**: `technical_architecture/src/acoustic_profile.rs:193-255` — `BatProfile` stores 50 valid bigrams, position weights, transition strictness
- **Python**: `realtime/parsing_strategy.py:186-226` — `HolophrasticStrategy` stores the exact same 50 bigrams, `OPENERS` set, `CLOSERS` set, and `RIGID_IDIOMS`

**Problem**: Any linguistic discovery must be manually updated in two separate codebases. If Phase 2 research finds a new rigid idiom, it has to be added to both files independently. This is a maintenance burden and a source of drift.

**Fix**: Make Rust the single source of truth for acoustic profiles. Python should request profile data from Rust at startup via the IPC bridge, not maintain its own copy.

---

### 2. Missing Emitter ID Bridge

Research (`bat_phase5_emitter_analysis.py`) shows that context in bats is determined by emitter identity ("who is calling"), not just acoustic features. But the `FeatureEvent` sent from Rust to Python only carries:

- 112D feature vector
- Cluster ID

**Problem**: There is no `emitter_id` field in the IPC bridge. The Python context inference engine (`interaction_agent.py`) can only infer "alarm", "territorial", "social", "contact" from acoustic features alone — which research shows is insufficient for bats.

**Fix**: Add emitter ID metadata to the `FeatureEvent` IPC message, possibly from source separation or voice fingerprinting in Rust.

---

## High

### 3. Rust Making Cognitive Decisions

`BatProfile` in Rust (`acoustic_profile.rs:193-345`) implements:

- **Transition strictness** (0.98) — rejects 99.98% of possible bigrams
- **Position-weighted feature logic** — weights temporal vs harmonic features differently for openers/closers
- **Valid bigram enforcement** — determines which sequences are "grammatically" valid

These are linguistic/cognitive decisions. By hardcoding them in Rust, the Python layer cannot:

- Dynamically adapt to new dialects discovered during interaction
- Adjust transition strictness based on learning
- Override grammar rules for novel contexts

**Fix**: Rust should enforce only acoustic safety constraints (volume, frequency limits). Grammar validation and transition logic should be driven by Python, with Rust receiving a validated sequence to synthesize.

---

### 4. Execution-Layer Code Still in Python

Five Python files still contain synthesis/audio processing that belongs in Rust:

| File | Class | What it does |
|---|---|---|
| `realtime/phrase_audio_library.py:1333` | `VocalizationSynthesizer` | Audio synthesis methods |
| `realtime/persona_voice_synthesis_engine.py:257` | `GranularVoiceSynthesizer` | Granular synthesis |
| `realtime/system.py:234` | `NaturalVocalizationSynthesizer` | Natural voice synthesis |
| `realtime/realtime_phrase_integration.py` | `RealTimePhraseIntegrator` | Real-time audio integration |
| `realtime/online_phrase_discovery_agent.py:320` | `RustSynthesizerBridge` | Bridge to Rust synthesis (should be a thin wrapper, not implementing audio logic) |

**Fix**: Migrate to Rust synthesis modules. Python retains only high-level intent generation (which phrase, what context, what timing).

---

## Medium

### 5. No Reflexive/Autonomous Mode

The system has only two modes:

- **Passthrough**: Python disconnected → audio muted, recording only
- **Interactive**: Python connected → full cognitive processing

**Problem**: If Python is temporarily offline (crash, restart, high latency), Rust immediately mutes all output. There is no intermediate **Reflexive Mode** where Rust could use its built-in acoustic profile to perform basic, non-cognitive responses (e.g., echoing a detected call within the restrictive grammar).

**Fix**: Add a third mode where Rust can autonomously generate simple responses using its `BatProfile` grammar, without Python's cognitive layer. This would maintain interaction continuity during brief Python outages.

---

### 6. Vocabulary Optimization Gap

The `VocabOptimizer` discovers k=1020 optimal vocabulary size during Python Phase 2 analysis, but Rust's `FeatureEventPublisher` must already have a fixed k value to publish cluster IDs.

**Problem**: The vocabulary discovery happens in Python but the application is hardcoded in Rust's clustering pipeline. There is no IPC bridge to update Rust's vocabulary size dynamically based on Python's findings.

**Fix**: Add a configuration IPC channel where Python can push updated vocabulary parameters to Rust at runtime.

---

### 7. Missing GPU/Hardware Acceleration

Archived Python files with no Rust equivalents:

- `gpu_phase_vocoder.py` → GPU-accelerated phase vocoding
- `gpu_phrase_integration.py` → CUDA phrase integration
- `jetson_accelerated_core.py` → Jetson-specific optimizations
- `opencl_wrapper.py` → OpenCL cross-platform acceleration
- `fpga_jetson_acceleration.py` → FPGA/Jetson hardware acceleration

These were archived as "to be migrated to Rust" but the Rust implementations don't exist yet. For field deployment on edge hardware, these are important.

---

## Low

### 8. Synthesis Timeline Construction Split

Python generates synthesis timelines with micro-dynamics deltas (amplitude, delta F0, attack/decay), then sends them to Rust. But the low-level construction of these audio "recipes" is signal parameterization that Rust could handle more efficiently.

**Fix**: Python should send high-level intent ("alarm response, confidence 0.8"), and Rust should construct the actual synthesis timeline using its acoustic profile, with Python's intent as input.

---

## Priority Summary

| Priority | Gap | Impact |
|---|---|---|
| **Critical** | Duplicated grammar data (Python + Rust) | Drift, maintenance burden |
| **Critical** | Missing Emitter ID in IPC | Context inference blind spot |
| **High** | Rust making cognitive decisions | Limits adaptive learning |
| **High** | 5 Python files doing Rust's job | Performance, safety gaps |
| **Medium** | No Reflexive Mode | Interaction breaks on Python crash |
| **Medium** | Vocab optimization not bridged | Can't adapt vocabulary at runtime |
| **Medium** | No GPU/hardware acceleration | Field deployment limited |
| **Low** | Synthesis timeline split | Suboptimal responsibility boundary |
