# Animal Vocalization Analysis Framework

A comprehensive research framework for analyzing animal vocalizations using the Universal Rosetta Stone methodology. Features advanced cognitive intelligence capabilities, cross-species analysis, and a hybrid Python/Rust architecture optimized for field deployment.

## Architecture Overview

### Execution vs. Logic Split

This framework follows a **hybrid architecture** combining Python and Rust:

- **Rust (Execution Layer)**: Time-critical operations, signal processing, hardware access, safety
  - Location: `technical_architecture/`
  - Zero-copy operations, memory safety, deterministic performance
  - **Field Survival**: Environmental monitoring, power management, wildlife detection, offline queuing

- **Python (Logic Layer)**: Cognitive intelligence, decision making, learning, context interpretation
  - Location: `cognitive_intelligence/`, `realtime/`, `semiotics/`
  - Rapid development, scientific computing, ML frameworks

### Peer-to-Peer Supervision

```
┌─────────────────────────────────────────────────────────────────┐
│                        Systemd Supervisor                        │
│  ┌──────────────────────────┐  ┌──────────────────────────┐     │
│  │  rust-field-engine       │  │  python-cognitive-agent  │     │
│  │  (Technical Architect)   │  │  (Logic Layer)           │     │
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

**Key Principle**: Fail open to safety. If Python crashes, Rust immediately mutes audio and continues in Passthrough Mode.

---

## Project Structure

```
src/
├── __init__.py                        # Main package exports
├── data_models.py                     # Unified data structures
├── vocalization_database.json         # Main database (2.5MB, 2,882 phrases)
│
├── analysis/                          # ⭐ STEP 1: Acoustic-First Analysis
│   └── rosetta_stone/                 #     Universal Rosetta Stone Engine
│       ├── universal_rosetta_stone.py #     Core acoustic analysis
│       ├── universal_synthesizer.py   #     Audio synthesis
│       ├── complete_extraction_pipeline.py # [NEW] Complete extraction
│       └── demo_unknown_species.py    #     Demo for new species
│
├── technical_architecture/            # ✅ STEP 6: Rust Execution Layer (Active)
│   ├── src/
│   │   ├── synthesis.rs              # Audio synthesis engines
│   │   ├── source_separation.rs      # Conv-TasNet separator
│   │   ├── thermal.rs                # Thermal management
│   │   ├── safety.rs                 # Safety monitoring
│   │   ├── ptp.rs                    # IEEE 1588 PTP timing
│   │   ├── logging.rs                # Provenance logging
│   │   ├── master_controller.rs     # Intent-Reality mediator
│   │   ├── peer_controller.rs       # ZeroMQ peer controller
│   │   ├── environmental_monitor.rs # Field: Rain/temp/light sensing
│   │   ├── power_manager.rs          # Field: Battery/solar management
│   │   ├── wildlife_sentry.rs        # Field: Background species detection
│   │   ├── data_synchronizer.rs      # Field: Offline black box queue
│   │   └── acoustic_simulator.rs     # Field: TDD test fixture
│   ├── deployment/                   # Systemd deployment files
│   │   ├── rust-field-engine.service
│   │   ├── python-cognitive-agent.service
│   │   ├── python_heartbeat_client.py
│   │   └── README.md
│   └── Cargo.toml
│
├── cognitive_intelligence/            # ✅ Python Logic Layer (Active)
│   ├── data_fusion.py                # Multi-modal data fusion
│   ├── visual_fusion.py              # Cross-modal attention
│   ├── siamese_network.py            # Similarity learning
│   ├── train_asteroid_base.py        # Base training template
│   ├── train_asteroid_marmoset.py    # Marmoset-specific model (4-8kHz)
│   ├── train_asteroid_bat.py         # Bat-specific model (100-17000Hz)
│   ├── train_asteroid_dolphin.py     # Dolphin-specific model (500-16000Hz)
│   ├── train_asteroid_chimpanzee.py  # Chimpanzee-specific model (100-1900Hz)
│   ├── train_asteroid_multispecies.py # Multi-species training
│   └── ASTEROID_TRAINING_README.md   # Training guide
│
├── realtime/                          # ✅ Active (Logic Layer Only)
│   ├── cognitive_layer.py            # Cognitive intelligence
│   ├── adaptive_context_switcher.py # Context interpretation
│   ├── adaptive_resonance.py         # Adaptive resonance theory
│   ├── deep_reinforcement_learning.py # ML training
│   ├── context_aware_synthesis.py    # Phrase selection logic
│   ├── probabilistic_context_machine.py # Decision making
│   ├── phrase_audio_library.py       # Data management
│   ├── unified_database.py           # Data access
│   ├── task_management.py            # Orchestration
│   └── archive/                      # Archived execution-layer Python
│       └── ARCHIVE.md                # (35 files moved to Rust)
│
├── query_interface/                   # ✅ Active
│   ├── vocalization_query_interface.py
│   └── demo_query_interface.py
│
├── semiotics/                         # ✅ Active
│   ├── semiotic_engine.py
│   ├── demo_semiotic_engine.py
│   └── SEMIOTIC_DETECTION_GUIDE.md
│
├── synthesis/                         # ✅ Active
│   ├── advanced_harmonic_extensions.py
│   ├── advanced_phrase_synthesizer.py
│   └── __init__.py
│
├── analysis/                          # ✅ Active
│   └── rosetta_stone/
│       └── universal_rosetta_stone.py
│
├── data_import/                        # ✅ Active
│   ├── import_vocalization_data.py
│   └── __init__.py
│
├── scientific_validation/              # ✅ Active
│   ├── ab_testing_controller.py
│   └── provenance_tracer.py
│
├── tests/                              # ✅ Active (canonical versions only)
│   ├── test_*.py
│   └── (28 duplicate test files with _1.py suffix archived)
│
├── archive/                            # ✅ Archived Content
│   ├── jungle-monitoring-system/      # Deprecated duplicate
│   ├── audio_engine/                   # Unused Rust implementation
│   ├── cognition/                      # Superseded by cognitive_intelligence
│   ├── hybrid/                         # Unused neural bridge
│   ├── test_cache/                     # Temporary cache files
│   ├── duplicate_tests/                # 28 backup test files
│   └── ARCHIVE.md                      # Archive documentation
│
└── [other active directories...]
```

---

## Complete Research Workflow

The framework follows a **six-step workflow** from raw audio to cognitive intelligence:

### STEP 1: Acoustic-First Analysis (`analysis/rosetta_stone/`)

**The Starting Point** - Extract phrase, sentence, and grammar information from raw audio.

```python
from analysis.rosetta_stone import UniversalRosettaStone

# Initialize analyzer
analyzer = UniversalRosettaStone(sample_rate=48000)

# Process raw audio → phrases
phrases = analyzer.segment_phrases(audio_data)

# Build vocabulary (atomic units)
vocabulary = analyzer.build_vocabulary(phrases, f0_bin_size=200)

# Discover grammar rules
grammar = analyzer.discover_grammar(phrases)

# Detect sentences (phrase sequences)
sentences = analyzer.discover_sentences(phrases, gaps)
```

**Output Database Format:**
- **Phrase Keys**: `F0_6400_DUR_50_RANGE_0` (binned acoustic features)
- **Sentences**: Groups of phrases with timestamps
- **Grammar**: Transition patterns between phrase types

**Key Classes:**
- `PhraseSignature` - Acoustic phrase representation with modality detection
- `Sentence` - Individual vocalization containing phrases
- `UniversalRosettaStone` - Main analysis engine

---

### STEP 1.5: Atomic Word Discovery Using Micro-Dynamics

**Enhanced Method** - Discover smallest semantic units (atomic words) using multi-dimensional acoustic features beyond simple F0 binning.

#### Traditional Approach vs. Micro-Dynamics

| Approach | Features | Phrase Key Example | Limitation |
|----------|----------|-------------------|------------|
| **Traditional** | F0, Duration, Range | `F0_7400_DUR_50_RANGE_300` | Groups dissimilar sounds with same F0 |
| **Micro-Dynamics** | 17 acoustic features | Multi-dimensional persona matching | Requires feature extraction |

#### Micro-Dynamics Feature Categories

**1. Grit Factors** (Timbre texture)
- `harmonic_to_noise_ratio` - Harmonic purity vs noise
- `spectral_flatness` - Noise-like vs tonal

**2. Motion Factors** (Envelope dynamics)
- `attack_time_ms` - Onset speed (fast = sharp, slow = gentle)
- `decay_time_ms` - Release speed
- `sustain_level` - Steady-state amplitude
- `vibrato_rate_hz` - Pitch modulation frequency
- `vibrato_depth` - Pitch modulation depth
- `jitter` - Micro-perturbations (instability vs stability)

**3. Fingerprint Factors** (Spectral shape)
- `mfcc_1` through `mfcc_4` - Mel-frequency cepstral coefficients
- `spectral_contrast` - Formant structure strength

**4. Rhythm Factors** (Temporal patterns)
- `median_ici_ms` - Inter-click interval
- `onset_rate_hz` - Click/event rate
- `ici_coefficient_of_variation` - Rhythm regularity

#### Acoustic Personas for Semantic Discovery

The framework defines **6 acoustic personas** that map acoustic features to semantic meaning:

| Persona | Semantic Category | Key Features | Example Context |
|---------|-----------------|--------------|-----------------|
| **GRITTY** | Aggressive alerts | Low HNR, high flatness, fast attack | Threat, confrontation |
| **PURE** | Contact/affiliation | High HNR, low flatness, slow attack | Food sharing, bonding |
| **BOUNCY** | Courtship/play | High vibrato, low jitter, pulsed | Mating, social play |
| **SHARP** | Alarm/startle | Very fast attack/decay, high contrast | Predator detection |
| **SUSTAINED** | Territory/long-range | Slow attack/decay, high sustain | Territorial claims |
| **TRANSIENT** | Rhythmic/mechanical | High onset rate, regular ICI | Mechanical sounds |

#### Usage

```python
from analysis.rosetta_stone.acoustic_similarity_for_atomic_phrase_candidates import (
    find_atomic_phrases_by_persona,
    find_similar_phrases_multi_dimensional,
    ACOUSTIC_PERSONAS
)

# Find "GRITTY" phrases (aggressive alerts)
gritty_phrases = find_atomic_phrases_by_persona(
    db=vocalization_database,
    persona_name='gritty',
    species='marmoset',
    top_n=20,
    min_score=0.4
)

# Each result: (phrase_key, features_dict, score)
for phrase_key, features, score in gritty_phrases:
    print(f"{phrase_key}: HNR={features['harmonic_to_noise_ratio']:.2f}, "
          f"Attack={features['attack_time_ms']:.1f}ms")

# Find acoustically similar phrases (beyond F0)
similar_phrases = find_similar_phrases_multi_dimensional(
    db=vocalization_database,
    query_phrase_key='F0_7400_DUR_50_RANGE_300',
    species='marmoset',
    top_n=10
)
```

#### Why Micro-Dynamics Matter

**Example**: Two phrases with identical F0 (7400 Hz) but different meanings:

| Phrase | F0 | Attack | HNR | Vibrato | Persona | Meaning |
|--------|-----|--------|-----|---------|---------|---------|
| A | 7400 Hz | 5 ms | 2.0 | 0 Hz | GRITTY | Alarm |
| B | 7400 Hz | 50 ms | 25.0 | 8 Hz | PURE | Contact |

**Traditional approach**: Groups A and B together (`F0_7400`) ❌
**Micro-dynamics**: Separates A (GRITTY) from B (PURE) ✅

#### Scientific Validation

Persona-based discovery enables:
1. **Fine-grained semantic categories** - Distinguish subtle behavioral contexts
2. **Cross-F0 similarity search** - Find "acoustic siblings" with different pitch
3. **Context-aware synthesis** - Generate context-appropriate vocalizations
4. **Quantified semantic meaning** - Score-based matching instead of binary inclusion

#### Command-Line Interface

```bash
# Find GRITTY phrases
python analysis/rosetta_stone/acoustic_similarity_for_atomic_phrase_candidates.py \
    --persona gritty --species marmoset --top-n 20

# Find phrases similar to specific phrase
python analysis/rosetta_stone/acoustic_similarity_for_atomic_phrase_candidates.py \
    --query F0_7400_DUR_50_RANGE_300 --species marmoset

# Analyze persona distribution
python analysis/rosetta_stone/acoustic_similarity_for_atomic_phrase_candidates.py \
    --analyze-distribution

# Check feature coverage in database
python analysis/rosetta_stone/acoustic_similarity_for_atomic_phrase_candidates.py \
    --analyze-coverage
```

---

### STEP 2: Data Import (`data_import/`)

Import the analyzed data into the unified database structure.

```bash
python3 src/data_import/import_vocalization_data.py
```

Creates `vocalization_database.json` with 2,882 phrases from 4 species.

---

### STEP 3: Query Interface (`query_interface/`)

High-performance querying with pre-built indexes.

```python
from src import get_query_interface

interface = get_query_interface()

# Search by F0 range
results = interface.search_phrases_by_f0_range(5000, 10000)

# Search by duration
results = interface.search_phrases_by_duration(30, 100)

# Find similar phrases
results = interface.find_similar_phrases(phrase, n=10)
```

---

### STEP 4: Cognitive Intelligence (`cognitive_intelligence/`, `semiotics/`)

Cognitive analysis for deception detection, innovation tracking, and cross-modal fusion.

```python
from src import SemioticEngine, SemioticContext

engine = SemioticEngine()
context = SemioticContext(species=Species.MARMOSET, ...)
result = engine.analyze_semiotics(phrase, context)
```

---

### STEP 5: Python Logic Layer (`realtime/`)

Cognitive decision making and phrase selection logic.

```python
from realtime.cognitive_layer import CognitiveLayer
from realtime.context_aware_synthesis import ContextAwareSynthesizer

# Make cognitive decisions
layer = CognitiveLayer()
decision = layer.decide(context, state)

# Select phrases based on context
synthesizer = ContextAwareSynthesizer()
selected = synthesizer.select_phrases(context, library)
```

---

## PRODUCTION DEPLOYMENT PIPELINE

### End-to-End Workflow for Deployed Systems

This section provides the complete methodology for deploying animal vocalization analysis systems in the field, from raw audio to cognitive understanding with species-specific source separation.

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

Raw Audio
   │
   ├─► [STEP 1] Species-Specific Source Separation
   │       └─► Separate target species from background noise
   │       └─► Uses Conv-TasNet models optimized for species F0 range
   │
   ├─► [STEP 2] Phrase, Sentence & Grammar Extraction
   │       ├─► Segment phrases (acoustic units)
   │       ├─► Detect sentences (phrase sequences)
   │       ├─► Discover grammar (transition patterns)
   │       └─► Context association (environmental, temporal)
   │
   ├─► [STEP 3] Synthesis Method Selection
   │       ├─► Concatenative (perfect fidelity, low flexibility)
   │       └─► Granular Concatenative (near-perfect fidelity, high flexibility)
   │
   ├─► [STEP 4] Cognitive Intelligence
   │       ├─► Phrase selection based on context
   │       ├─► Semiotic analysis (deception, innovation)
   │       └─► Cross-modal data fusion
   │
   └─► [STEP 5] Response Generation
       ├─► Rust execution layer (safety-critical)
       ├─► Environmental monitoring
       └─► Real-time synthesis
```

---

### STEP 1: Species-Specific Source Separation

**Objective**: Isolate target species vocalizations from environmental noise and other species.

#### Train Species-Specific Models

```bash
# Train models for your target species
cd cognitive_intelligence

# Option A: Train all species
python train_asteroid_multispecies.py --all

# Option B: Train specific species
python train_asteroid_multispecies.py --species marmoset egyptian_bat

# Option C: Train individual species
python train_asteroid_marmoset.py       # F0: 4000-8000 Hz
python train_asteroid_bat.py            # F0: 100-17000 Hz
python train_asteroid_dolphin.py        # F0: 500-16000 Hz
python train_asteroid_chimpanzee.py     # F0: 100-1900 Hz
```

#### Species-Specific Configuration

| Species | F0 Range | Filter Range | Sample Rate | Use Case |
|---------|----------|--------------|-------------|----------|
| **Marmoset** | 4000-8000 Hz | 2800-10400 Hz | 44.1kHz | Mid-frequency primate |
| **Egyptian Fruit Bat** | 100-17000 Hz | 100-22100 Hz | 96kHz* | Wide range, FM sweeps |
| **Dolphin** | 500-16000 Hz | 350-20800 Hz | 96kHz* | Whistles, clicks |
| **Chimpanzee** | 100-1900 Hz | 100-2470 Hz | 44.1kHz | Low-frequency primate |

*Use 96kHz for ultrasonic vocalizations (bats, dolphins)

#### Deploy Source Separation in Rust

```rust
// Update technical_architecture/src/source_separation.rs
use technical_architecture::SourceSeparator;

let config = SeparatorConfig {
    model_path: "models/checkpoints/marmoset/conv_tasnet_marmoset.onnx",
    sample_rate: 44100,
    num_sources: 2,  // Target + background
    chunk_size: 4096,
};

let separator = SourceSeparator::new(config)?;
let separated = separator.separate(audio_buffer)?;
```

**Output**: Clean audio with target species isolated from background.

---

### STEP 2: Complete Extraction Pipeline (One Script)

**New Script**: `analysis/rosetta_stone/complete_extraction_pipeline.py`

Extracts phrases, sentences, grammar, and segmented audio in one pass with context association.

```bash
# Run complete extraction pipeline
python analysis/rosetta_stone/complete_extraction_pipeline.py \
    --input audio/field_recording.wav \
    --species marmoset \
    --output results/marmoset_session_001 \
    --separate-model models/checkpoints/marmoset/conv_tasnet_marmoset.onnx
```

#### What It Extracts

**1. Phrase Segmentation**
```python
# Acoustic phrase extraction
phrases = [
    Phrase(
        phrase_key="F0_7400_DUR_50_RANGE_300",
        audio_segment=audio[start:end],
        f0_mean_hz=7400,
        duration_ms=50,
        f0_range_hz=300,
        modality="harmonic",
        timestamp_ms=1234.5
    ),
    # ... more phrases
]
```

**2. Sentence Detection**
```python
# Phrase sequences with timing
sentences = [
    Sentence(
        sentence_id=1,
        phrase_sequence=["F0_7400", "F0_7800", "F0_7500"],
        start_time_ms=1234.5,
        end_time_ms=1456.8,
        gap_pattern="medium"  # Inter-phrase gap pattern
    ),
    # ... more sentences
]
```

**3. Grammar Discovery**
```python
# Transition probability matrix
grammar = {
    "F0_7400": {"F0_7800": 0.6, "F0_7500": 0.3, "F0_7200": 0.1},
    "F0_7800": {"F0_7500": 0.7, "F0_7400": 0.2, "F0_8000": 0.1},
    # ... more transitions
}

# Syntax rules discovered from patterns
syntax_rules = [
    "PHEE_CALL → TRILL → TWITTER",  # Common sequence
    "PHEE_CALL → PHEE_CALL (0.3-1.0s gap)",  # Repetition pattern
]
```

**4. Context Association**
```python
# Environmental and temporal context
context = {
    "time_of_day": "dawn",  # From timestamp
    "weather": "clear",     # From environmental sensors
    "location": "feeding_site",  # From GPS/cage ID
    "social_context": "group_present",  # From multi-animal detection
    "previous_interactions": [
        {"type": "response_to", "speaker": "marmoset_002", "latency_s": 0.5}
    ]
}
```

**5. Segmented Audio Export**
```
results/marmoset_session_001/
├── phrases/
│   ├── F0_7400_DUR_50_RANGE_300_phrase_001.wav
│   ├── F0_7800_DUR_50_RANGE_200_phrase_002.wav
│   └── ...
├── sentences/
│   ├── sentence_001_phee_call_sequence.wav
│   ├── sentence_002_trill_sequence.wav
│   └── ...
├── grammar.json
├── phrases.json
├── sentences.json
└── context.json
```

#### Python API

```python
from analysis.rosetta_stone import CompleteExtractionPipeline
from technical_architecture import SourceSeparator

# Initialize pipeline
pipeline = CompleteExtractionPipeline(
    species="marmoset",
    sample_rate=44100,
    source_separator=SourceSeparator(species_model_path="...")
)

# Process audio file
results = pipeline.process(
    audio_path="audio/field_recording.wav",
    extract_audio_segments=True,
    discover_grammar=True,
    associate_context=True
)

# Access results
print(f"Extracted {len(results['phrases'])} phrases")
print(f"Detected {len(results['sentences'])} sentences")
print(f"Grammar: {results['grammar']}")
print(f"Context: {results['context']}")
```

---

### STEP 3: Synthesis Method Selection

**Decision Tree**: Choose the right synthesis method for your use case.

```
                    Start
                      │
          Do you need parameter variation?
                      │
           ┌──────────┴──────────┐
           │                     │
          YES                   NO
           │                     │
    Need specific pitch?    Have exact segment?
           │                     │
    ┌──────┴──────┐        ┌────┴────┐
   YES           NO       YES       NO
    │             │         │         │
Granular     Granular   Concatenative  [Error: No audio]
(1 voice)   (Morpher)   (perfect)
```

#### Concatenative Synthesis

**Use when**: You have exact audio segments and need perfect fidelity.

```python
# Load segmented phrases from extraction pipeline
from realtime.phrase_audio_library import PhraseAudioLibrary

library = PhraseAudioLibrary.load("results/marmoset_session_001/")

# Select phrases by acoustic features
phrases = library.get_phrases(
    f0_min=7000,
    f0_max=8000,
    duration_min=40,
    duration_max=60
)

# Concatenate (perfect fidelity, no manipulation)
output = library.concatenate(phrases)
```

**Characteristics**:
- ✅ Perfect fidelity (t-SNE distance: 4.2)
- ✅ Preserves all natural characteristics
- ❌ No parameter flexibility
- ❌ Limited to available phrases

#### Granular Concatenative Synthesis

**Use when**: You need systematic parameter variation while preserving formants.

```python
from technical_architecture import GranularConcatenativeSynthesizer

# Load source phrase
source_phrase = library.get_phrase("F0_7400_DUR_50_RANGE_300")

# Create synthesizer
synth = GranularConcatenativeSynthesizer(sample_rate=22050)
synth.load_source(source_phrase.audio)

# Systematic parameter variation
for pitch_shift in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]:
    synth.set_pitch_shift(pitch_shift)
    output = synth.synthesize(duration_ms=50.0)

    # Generate pitch continuum (7400, 7030, 6660, 7400, 8140, 8880, 9620 Hz)
    # Even if these exact pitches don't exist in database!
```

**Characteristics**:
- ✅ Near-perfect fidelity (t-SNE distance: 6.452)
- ✅ Preserves formant structure
- ✅ Enables parameter variation
- ✅ 76.1% better than additive synthesis
- ⚠️ Requires real audio source

#### Comparison Table

| Feature | Concatenative | Granular | Additive |
|---------|---------------|----------|----------|
| **Fidelity (t-SNE)** | 4.208 | 6.452 | 27.052 |
| **Formant Preservation** | ✅ Perfect | ✅ Excellent | ❌ Poor |
| **Pitch Flexibility** | ❌ None | ✅ Excellent | ✅ Excellent |
| **Time Flexibility** | ❌ None | ✅ Yes | ✅ Yes |
| **Requires Real Audio** | ✅ Yes | ✅ Yes | ❌ No |
| **Use Case** | Natural playback | Systematic variation | Synthetic sounds |

---

### STEP 4: Species-Specific Source Separation (Production)

#### Model Selection Guide

**For Field Deployment**:

```python
# Deploy with species-specific models
DEPLOYMENT_CONFIG = {
    "location_jungle": {
        "primary_species": "marmoset",
        "separator_model": "models/checkpoints/marmoset/conv_tasnet_marmoset.onnx",
        "f0_range": (4000, 8000),
        "filter_range": (2800, 10400),
    },
    "location_cave": {
        "primary_species": "egyptian_bat",
        "separator_model": "models/checkpoints/egyptian_bat/conv_tasnet_egyptian_bat.onnx",
        "f0_range": (100, 17000),
        "filter_range": (100, 22100),
        "sample_rate": 96000,  # Higher for ultrasonic
    },
    "location_marine": {
        "primary_species": "dolphin",
        "separator_model": "models/checkpoints/dolphin/conv_tasnet_dolphin.onnx",
        "f0_range": (500, 16000),
        "filter_range": (350, 20800),
        "sample_rate": 96000,  # For ultrasonic clicks
    },
}
```

#### Multi-Species Environments

```python
# For environments with multiple species
from technical_architecture import SourceSeparator, MultiSpeciesSeparator

# Load multiple models
separator = MultiSpeciesSeparator()
separator.load_model("marmoset", "models/checkpoints/marmoset/...")
separator.load_model("egyptian_bat", "models/checkpoints/egyptian_bat/...")

# Auto-detect species by frequency content
audio_buffer = read_audio("field_recording.wav")
species = separator.detect_species(audio_buffer)  # Returns: "marmoset"

# Use appropriate model
separated_audio = separator.separate(audio_buffer, model=species)
```

#### Rust Integration

```rust
// technical_architecture/src/source_separation.rs

pub struct SpeciesSeparator {
    models: HashMap<String, TractModel>,
    active_model: Option<String>,
}

impl SpeciesSeparator {
    pub fn detect_species(&self, audio: &[f32]) -> Option<String> {
        // Analyze frequency content
        let spectrum = self.compute_spectrum(audio);

        // Check F0 ranges
        let dominant_f0 = self.find_dominant_f0(&spectrum);

        match dominant_f0 {
            100..=1900 => Some("chimpanzee"),
            2800..=10400 => Some("marmoset"),
            100..=22100 => Some("egyptian_bat"),
            350..=20800 => Some("dolphin"),
            _ => None,
        }
    }

    pub fn separate(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        // Auto-detect species
        let species = self.detect_species(audio)
            .ok_or_else(|| anyhow!("Cannot detect species"))?;

        // Load appropriate model
        self.load_model_if_needed(species)?;

        // Separate
        self.models.get_mut(species).unwrap().separate(audio)
    }
}
```

---

### STEP 5: Response Generation

#### Complete Response Pipeline

```python
# Full pipeline: extraction → analysis → synthesis
from analysis.rosetta_stone import CompleteExtractionPipeline
from realtime.cognitive_layer import CognitiveLayer
from technical_architecture import GranularConcatenativeSynthesizer

# Step 1: Extract (with species-specific source separation)
pipeline = CompleteExtractionPipeline(species="marmoset")
results = pipeline.process(audio_path, separate_species=True)

# Step 2: Analyze (cognitive intelligence)
cognitive = CognitiveLayer()
response_type = cognitive.decide(
    context=results['context'],
    grammar=results['grammar'],
    detected_phrases=results['phrases']
)

# Step 3: Synthesize (concatenative or granular)
if response_type.requires_parameter_variation:
    # Use granular for systematic variation
    synth = GranularConcatenativeSynthesizer(sample_rate=22050)
    synth.load_source(response_type.source_phrase.audio)
    synth.set_pitch_shift(response_type.pitch_shift)
    output = synth.synthesize(duration_ms=response_type.duration_ms)
else:
    # Use concatenative for perfect fidelity
    output = library.concatenate(response_type.target_phrases)

# Step 4: Safety checks and playback
from technical_architecture import SafetyMonitor
safety = SafetyMonitor()
if safety.check_audio_safety(output):
    play_audio(output)
```

---

## METHODOLOGY RECOMMENDATIONS

### Latest Scientific Findings

**1. Granular Synthesis Achieves Bio-Acoustic Fidelity**
- t-SNE distance: 6.452 (< 7.0 target) ✅
- 76.1% improvement over additive synthesis
- Preserves formant structure while enabling parameter variation
- **Implication**: Use granular synthesis for systematic experiments

**2. Additive Synthesis is Insufficient**
- t-SNE distance: 27.052 (failed)
- Cannot capture inharmonic partials and formant structure
- **Implication**: Avoid additive synthesis for bio-acoustic research

**3. Species-Specific Source Separation Improves Accuracy**
- General model: 70-80% separation accuracy
- Species-specific model: 85-95% separation accuracy
- **Implication**: Train species-specific models for deployment

### Decision Matrix

| Scenario | Extraction Method | Synthesis Method | Source Separation |
|----------|-------------------|------------------|-------------------|
| **Basic analysis** | Universal Rosetta Stone | N/A | Not needed |
| **Playback natural calls** | Segmented phrases | Concatenative | Not needed |
| **Pitch continuum testing** | Segmented phrases | Granular | Not needed |
| **Noisy field recordings** | + Source separation | Concatenative/Granular | Species-specific |
| **Multi-species environment** | + Species detection | Concatenative/Granular | Multi-model |
| **Real-time interaction** | Real-time extraction | Granular (fast) | Species-specific |

### Production Checklist

**Before Field Deployment**:
- [ ] Train species-specific source separation model
- [ ] Test extraction pipeline on field recordings
- [ ] Validate synthesis method (concatenative vs granular)
- [ ] Configure Rust safety limits
- [ ] Test environmental monitoring integration
- [ ] Verify power management configuration

**During Deployment**:
- [ ] Monitor separation quality (log SI-SDR metrics)
- [ ] Track synthesis performance (t-SNE validation)
- [ ] Record environmental context
- [ ] Check thermal and power status
- [ ] Validate safety limits

**Post-Deployment**:
- [ ] Analyze extracted phrases and sentences
- [ ] Update grammar rules
- [ ] Re-train models if performance degrades
- [ ] Archive audio segments for future research

---

### STEP 6: Rust Execution Layer (`technical_architecture/`)

Safety-critical audio processing, granular synthesis, and field deployment.

```rust
use technical_architecture::TechnicalArchitect;

let architect = TechnicalArchitect::new(config).await?;
let result = architect.process_audio_frame(audio, intent).await?;
```

**Granular Concatenative Synthesis** - High-fidelity vocalization synthesis:
```python
from technical_architecture import GranularConcatenativeSynthesizer

# Create synthesizer
synth = GranularConcatenativeSynthesizer(sample_rate=22050)

# Load source audio
synth.load_source(audio_buffer)

# Set pitch shift (0.9 = lower, 1.1 = higher)
synth.set_pitch_shift(0.9)

# Synthesize with preserved formant structure
output = synth.synthesize(duration_ms=100.0)
```

**Scientific Validation**: t-SNE distance = 6.452 (< 7.0 target) ✅
- 76.1% improvement over additive synthesis (distance 27.0)
- Preserves formant structure while enabling systematic parameter variation
- See `/realtime/GRANULAR_SYNTHESIS_FINDINGS.md` for details

**When to Use Granular vs Concatenative:**
- **Concatenative**: Use when you have exact audio segments (perfect fidelity, low flexibility)
- **Granular**: Use when you need systematic parameter variation (near-perfect fidelity, high flexibility)
  - Pitch continuum testing (7500Hz, 7600Hz, 7700Hz... even if not in database)
  - Controlling confounds (same phrase, different pitches, constant duration)
  - Acoustic feature boundary testing (JND measurements)
  - Creating novel stimuli (hybrid calls)

---

## Key Components

### 1. Acoustic-First Analysis Engine (`analysis/rosetta_stone/`)

**The Foundation** - Species-agnostic acoustic analysis for discovering phrase, sentence, and grammar structure.

**Core Classes:**
- `UniversalRosettaStone` - Main analysis engine
  - `detect_modality()` - Classifies: Harmonic, FM Sweep, Transient, Rhythmic
  - `segment_phrases()` - Segments audio into phrase units
  - `build_vocabulary()` - Clusters similar phrases into atomic units
  - `discover_grammar()` - Discovers grammatical transition rules
  - `discover_sentences()` - Groups phrases into sentences
  - `detect_superposition()` - Finds simultaneous phrase layers

- `PhraseSignature` - Acoustic phrase representation
  - Modality-specific feature extraction
  - Distance metrics for phrase similarity
  - Microharmonic similarity calculation

- `Sentence` - Individual vocalization containing phrases
  - `discover_atomic_units()` - Creates binned phrase keys
  - Validates phrase groupings

**Output:**
- Phrase keys: `F0_6400_DUR_50_RANGE_0` format
- Vocabulary: Grouped phrases by acoustic similarity
- Grammar: Transition matrices between phrase types
- Sentences: Phrase sequences with timing

**Usage:**
```python
from analysis.rosetta_stone import UniversalRosettaStone

analyzer = UniversalRosettaStone(sample_rate=48000)
phrases = analyzer.segment_phrases(audio_data)
vocabulary = analyzer.build_vocabulary(phrases)
grammar = analyzer.discover_grammar(phrases)
```

---

### 2. Rust Execution Layer (`technical_architecture/`)

**Core Modules:**
- **Synthesis** (`synthesis.rs`) - Granular, concatenative, superpositional synthesis
- **Source Separation** (`source_separation.rs`) - Conv-TasNet via ONNX/Tract
- **Thermal Management** (`thermal.rs`) - Temperature monitoring and throttling
- **Safety Monitoring** (`safety.rs`) - Watchdog timers, safety limits
- **PTP Clock** (`ptp.rs`) - IEEE 1588 precision timing (nanosecond)
- **Provenance Logging** (`logging.rs`) - Deterministic audit trails

**Production Deployment Modules:**
- **IACUC Compliance** (`iacuc_compliance.rs`) - 29 tests
  - Legal animal research protocol enforcement
  - Time window, volume, species, and daily limit enforcement
  - Compliance audit trails and report generation

- **Time-Series Archive** (`time_series_archive.rs`) - 24 tests
  - High-frequency time-series data storage
  - Query by time range with downsampled aggregation
  - Retention policies and storage quotas

- **Auto-Calibration** (`auto_calibration.rs`) - 17 tests
  - Pink noise calibration tone generation
  - Loopback gain analysis for drift detection
  - Health status reporting with automatic scheduling

- **Shadow Model Monitoring** (`shadow_model_monitor.rs`) - 26 tests
  - Parallel inference: active model vs frozen baseline
  - Concept drift detection and alerting
  - Automatic model rollback capability

- **Remote Web Dashboard** (`web_dashboard.rs`) - 25 tests
  - HTTPS/WebSocket server for remote monitoring
  - JWT token authentication with expiration
  - Emergency stop, manual override, parameter adjustment
  - Real-time spectrogram and gauge streaming
  - Command audit logging

- **Multi-Node Coordination** (`multi_node_coordination.rs`) - 21 tests
  - PTP grandmaster election (IEEE 1588 clock class/accuracy)
  - TDMA scheduling for acoustic interference avoidance
  - Data fusion with location triangulation
  - Cluster management with automatic failover

**Field Deployment Modules:**
- **Environmental Monitor** (`environmental_monitor.rs`) - 46 tests
  - Rain intensity classification (None → Storm)
  - Temperature classification (Freezing → Extreme)
  - Light level classification (Dark → Night)
  - Solar forecasting integration
  - Forces Passthrough Mode in adverse conditions

- **Power Manager** (`power_manager.rs`) - 54 tests
  - Battery state tracking with health estimation
  - Power modes: Normal (>80%), Medium (50-80%), Low (20-50%), Critical (<20%)
  - Solar prediction for task deferral decisions
  - Atomic flags for FPGA, source separation, synthesis throttling
  - Power budget calculation with runtime estimation

- **Wildlife Sentry** (`wildlife_sentry.rs`) - 24 tests
  - FFT-based vocalization detection
  - Species signatures: marmoset, dolphin, bat, finch
  - Wake trigger generation with urgency levels
  - Debounce mechanism for rapid successive calls

- **Data Synchronizer** (`data_synchronizer.rs`) - 20 tests
  - Priority-based sync (Critical > High > Normal > Low)
  - Bandwidth throttling
  - Multi-storage backend (SSD, USB, SD Card)
  - Compression support

- **Acoustic Simulator** (`acoustic_simulator.rs`) - 43 tests
  - Environmental noise generation (rain, wind, insects, birds)
  - SNR mixing for testing
  - Environment simulation (jungle, rainforest, open field)

**Master Controller:**
- **UnifiedMasterController** - Intent-Reality mediator
  - Translates Python intents into physical Rust actions
  - Enforces thermal, safety, and hardware constraints
  - Watchdog monitoring with crash isolation

- **PeerController** - ZeroMQ heartbeat monitoring
  - Non-blocking heartbeat polling (0ms timeout)
  - Automatic mode switching (Passthrough ↔ Interactive)
  - 100ms timeout (5 missed heartbeats = disconnect)

**Build:**
```bash
cd technical_architecture
cargo build --release
cargo test  # 408 tests passing
```

---

### 2. Python Logic Layer

**Cognitive Intelligence (`cognitive_intelligence/`):**
- `data_fusion.py` - Multi-modal data fusion
- `visual_fusion.py` - Cross-modal attention
- `siamese_network.py` - Similarity learning
- `train_asteroid_model.py` - Source separation training

**Real-time Processing (`realtime/` - Logic Layer Only):**
- `cognitive_layer.py` - Cognitive intelligence and decision making
- `adaptive_context_switcher.py` - Context interpretation
- `adaptive_resonance.py` - Adaptive resonance theory
- `deep_reinforcement_learning.py` - ML training
- `context_aware_synthesis.py` - Phrase selection logic
- `probabilistic_context_machine.py` - Decision making
- `phrase_audio_library.py` - Data management
- `unified_database.py` - Data access
- `task_management.py` - Orchestration

**Note:** 35 execution-layer Python files previously in `realtime/` have been archived. See `realtime/archive/ARCHIVE.md`.

---

### 3. Bio-Acoustic Turing Test (`realtime/`)

**Live Animal Validation Framework** - Determines if animals can distinguish between natural and granular-synthesized vocalizations.

```python
from realtime.bio_acoustic_turing_test import BioAcousticTuringTest

# Create Turing test instance
turing_test = BioAcousticTuringTest(
    subject_id='marmoset_001',
    species='marmoset',
    output_dir='./results'
)

# Phase 1: Concatenative baseline (natural recordings)
turing_test.set_phase('concatenative_baseline')
turing_test.add_stimulus('natural_phee', audio_data, 'concatenative')
result = turing_test.run_trial('natural_phee')

# Phase 2: Granular synthesis (pitch-shifted variants)
turing_test.set_phase('granular_synthesis')
turing_test.add_stimulus('granular_phee', granular_audio, 'granular')
result = turing_test.run_trial('granular_phee')

# Phase 3: Statistical analysis
hypothesis = turing_test.evaluate_hypothesis()

if hypothesis['passed']:
    print("✅ TURING TEST PASSED - Animals cannot distinguish!")
```

**Components:**
- `StimulusController` - Manages audio playback with counterbalanced sequences
- `ResponseRecorder` - Records animal responses and measures latency
- `ExperimentDesign` - Handles randomization and inter-trial intervals
- `StatisticalAnalyzer` - Chi-square tests, t-tests, Turing test evaluation
- `BioAcousticTuringTest` - Main orchestrator

**Demo:**
```bash
python3 realtime/demo_bio_acoustic_turing_test.py
```

**Tests:**
```bash
python3 -m pytest realtime/test_bio_acoustic_turing_test.py -v
```

---

### 4. Query Interface (`query_interface/`)

- High-performance query system with pre-built indexes
- Real-time search: F0 range, duration, similarity
- Grammar network analysis and cross-species comparisons
- Main entry: `VocalizationQueryInterface` and `get_query_interface()`

---

### 4. Semiotic Analysis (`semiotics/`)

- Advanced cognitive intelligence capabilities
- Deception detection and innovation tracking
- Directed communication analysis
- Cross-modal attention fusion

---

## Deployment

### Systemd Services

Two services managed by systemd:

**1. Rust Field Engine** (`technical_architecture/deployment/rust-field-engine.service`)
- Safety-critical execution layer
- Binds ZeroMQ SUB socket for heartbeats
- Starts in Passthrough Mode (safe default)

**2. Python Cognitive Agent** (`technical_architecture/deployment/python-cognitive-agent.service`)
- Logic layer with cognitive intelligence
- Connects to Rust and sends heartbeats (20ms interval)
- Automatically restarted on crash (Let it crash philosophy)

### Installation

```bash
# Copy systemd files
sudo cp technical_architecture/deployment/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable rust-field-engine.service
sudo systemctl enable python-cognitive-agent.service

# Start both services
sudo systemctl start rust-field-engine.service
sudo systemctl start python-cognitive-agent.service

# View logs
sudo journalctl -u rust-field-engine.service -f
sudo journalctl -u python-cognitive-agent.service -f
```

### Operation Modes

**Passthrough Mode** (Safe Default):
- Python disconnected or heartbeats stopped
- Audio muted
- Raw audio recording continues
- Passive monitoring

**Interactive Mode** (Active):
- Python connected and sending heartbeats
- Processing intents from Python
- Synthesizing responses
- Full cognitive interaction

---

## Quick Start

### STEP 1: Analyze Raw Audio (Acoustic-First)

**For new species or new audio data:**

```bash
# Run acoustic analysis on raw audio
python3 src/analysis/rosetta_stone/demo_unknown_species.py
```

This extracts:
- **Phrases**: Acoustic units with F0, duration, range features
- **Vocabulary**: Grouped similar phrases (atomic units)
- **Grammar**: Transition rules between phrase types
- **Sentences**: Phrase sequences with timing

### STEP 2: Import Database

```bash
# Import vocalization data (populates query interface)
python3 src/data_import/import_vocalization_data.py
```

Creates `vocalization_database.json` with 2,882 phrases.

### STEP 3: Run Demos

```bash
# Query interface demo
python3 src/query_interface/demo_query_interface.py

# Semiotic engine demo
python3 src/semiotics/demo_semiotic_engine.py
```

### Run Tests

```bash
# Python tests
python3 -m pytest tests/ -v

# Rust tests (in technical_architecture/)
cd technical_architecture && cargo test
```

### Python API Usage

```python
from src import (
    Species, VocalizationModality,
    Phrase, AcousticFeatures,
    get_query_interface,
    SemioticEngine
)

# Query interface example
interface = get_query_interface()
results = interface.search_phrases_by_f0_range(5000, 10000)

# Semiotic analysis example
engine = SemioticEngine()
context = SemioticContext(species=Species.MARMOSET, ...)
result = engine.analyze_semiotics(phrase, context)
```

### Rust API Usage

```rust
use technical_architecture::{
    TechnicalArchitect, PeerController,
    OperationMode, PeerControllerConfig,
    IacucComplianceEngine, MultiNodeCoordinator,
    WebDashboard, TimeSeriesArchiver,
};

// Create technical architect
let config = TechArchConfig::default();
let architect = TechnicalArchitect::new(config).await?;

// Create IACUC compliance engine
let iacuc = IacucComplianceEngine::new(protocol)?;
let check = iacuc.check_compliance(&intent)?;

// Create multi-node coordinator
let config = ClusterConfig::default();
let coordinator = MultiNodeCoordinator::new("node1".to_string(), config);
coordinator.elect_grandmaster(my_info).await?;

// Create web dashboard
let dashboard = WebDashboard::new(config)?;
dashboard.connect_client("client1", "127.0.0.1", &token)?;

// Create peer controller for heartbeat monitoring
let config = PeerControllerConfig::default();
let mut controller = PeerController::new(config)?;

// Main loop
loop {
    let mode = controller.tick()?;

    match mode {
        OperationMode::Passthrough => {
            // Safe mode - recording only
        }
        OperationMode::Interactive => {
            // Active mode - process Python intents
        }
    }
}
```

---

## Database Status

- **Total Phrases**: 2,882 from 4 species
- **Marmoset**: 1,351 phrases (harmonic communication)
- **Egyptian Fruit Bat**: 516 phrases (FM sweep communication)
- **Dolphin**: 387 phrases (whistle communication)
- **Chimpanzee**: 628 phrases (mixed communication)

---

## Key Features

### Cross-Species Analysis
- Universal Rosetta Stone methodology
- Species-specific acoustic analysis strategies
- Comparative semiotic patterns

### Cognitive Intelligence
- Deceptive communication detection
- Emergent behavior identification
- Directed communication analysis
- Cross-modal attention fusion

### High Performance
- Zero-copy Rust operations for audio processing
- Optimized data structures and indexing
- Real-time query capabilities
- Deterministic timing with PTP

### Safety & Reliability
- Peer-to-peer supervision with systemd
- Automatic crash recovery (Let it crash)
- Fail-open to safety design
- Thermal throttling and emergency mute

### Production Deployment Capabilities
- **IACUC Compliance**: Legal animal research protocol enforcement with audit trails
- **Time-Series Archiving**: High-frequency data storage with retention policies
- **Auto-Calibration**: Self-health checks with pink noise calibration and drift detection
- **Shadow Model Monitoring**: Concept drift detection with automatic model rollback
- **Remote Dashboard**: HTTPS/WebSocket monitoring with emergency stop capabilities
- **Multi-Node Coordination**: PTP grandmaster election and TDMA scheduling for arrays

### Field Deployment Capabilities
- **Environmental Monitoring**: Automatic session management based on rain, temperature, light
- **Power Management**: Solar-aware battery optimization with adaptive throttling
- **Wildlife Detection**: Low-power background sentry for target species vocalizations
- **Offline Resilience**: Black box data queuing with priority-based synchronization
- **TDD Infrastructure**: Acoustic simulation for comprehensive testing

---

## Scientific Impact

This framework transforms animal communication research by:
1. Moving beyond simple classification to cognitive understanding
2. Enabling deception detection in animal communication
3. Tracking emergent cultural behaviors
4. Recognizing intentional, targeted communication
5. Providing comparative analysis across species

---

## Test Coverage

The framework has comprehensive test coverage ensuring reliability and correctness:

```
Rust Execution Layer: 408 tests passing
├── Core Modules: 179 tests
│   ├── Peer Controller: 79 tests
│   ├── Master Controller: 17 tests
│   └── Other modules: 83 tests
│
├── Production Deployment: 142 tests (NEW)
│   ├── IACUC Compliance Engine: 29 tests
│   ├── Time-Series Archive: 24 tests
│   ├── Auto-Calibration: 17 tests
│   ├── Shadow Model Monitoring: 26 tests
│   ├── Remote Web Dashboard: 25 tests
│   └── Multi-Node Coordination: 21 tests
│
└── Field Deployment: 187 tests
    ├── Environmental Monitor: 46 tests
    ├── Power Manager: 54 tests
    ├── Wildlife Sentry: 24 tests
    ├── Data Synchronizer: 20 tests
    └── Acoustic Simulator: 43 tests

Python Logic Layer: 50+ test files
```

**Test Domains Covered:**
- **Production Compliance**: IACUC protocol enforcement, calibration scheduling, shadow model monitoring
- **Remote Operations**: Web dashboard with authentication, command audit logging
- **Multi-Node Coordination**: PTP grandmaster election, TDMA scheduling, data fusion
- Environmental condition classification and override logic
- Battery state management and solar prediction
- Wildlife detection with FFT analysis
- Offline queuing with priority handling
- Peer-to-peer heartbeat monitoring
- Intent-Reality mediation
- Thermal throttling and safety limits
- Source separation and synthesis

---

## Archival Information

**Documentation:**
- `technical_architecture/TDD_PLAN_FIELD_FEATURES.md` - Field deployment implementation plan (COMPLETE)
- `technical_architecture/CLAUDE.md` - Detailed developer guide with API examples

**Archived Directories:** See `/src/archive/ARCHIVE.md` for details

- `jungle-monitoring-system/` - Superseded by `cognitive_intelligence/`
- `audio_engine/` - Superseded by `technical_architecture/`
- `cognition/` - Superseded by `cognitive_intelligence/`
- `hybrid/` - Unused neural bridge implementation
- `test_cache/` - Temporary cache files
- `duplicate_tests/` - 28 backup test files with `_1.py` suffix

**Realtime Archive:** `/src/realtime/archive/ARCHIVE.md`

35 execution-layer Python files moved to Rust implementation.

---

## License

**CC BY-ND 4.0 International** - See main project license for details.

---

## Author

Sheel Morjaria (sheelmorjaria@gmail.com)
