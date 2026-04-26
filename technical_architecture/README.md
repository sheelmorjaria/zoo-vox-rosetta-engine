# Technical Architecture - Rust Execution Layer

High-performance Rust execution layer for the animal vocalization analysis framework. Provides safety-critical audio processing, environmental monitoring, and production deployment capabilities.

## Table of Contents

1. [Overview](#overview)
2. [Zoo Vox Rosetta Engine v2.0](#zoo-vox-rosetta-engine-v20)
3. [Four-Stage Pipeline](#four-stage-pipeline)
   - [45D Acoustic Features](#45d-acoustic-feature-system)
   - [112D Rosetta Features](#112d-rosetta-features)
   - [Dynamic Segmentation](#dynamic-segmentation-change-point-detection)
   - [Grading Score System](#grading-score-system)
   - [Cascaded Architecture](#cascaded-architecture-router--analyzer)
4. [Semantic Grounding](#semantic-grounding)
   - [Human-Guided Context Discovery](#human-guided-context-discovery)
   - [Rosetta Pipeline](#rosetta-pipeline-integrated-zoo-vox-rosetta-engine)
5. [Research Workflow](#research-workflow)
6. [Query Interface - Semantic Search](#query-interface---semantic-search)
7. [PAM Pipeline - Passive Acoustic Monitoring](#pam-pipeline---passive-acoustic-monitoring)
8. [Closed-Loop Interaction Agent](#closed-loop-interaction-agent)
9. [Granular Concatenative Synthesis](#granular-concatenative-synthesis)
10. [Build & Test](#build)
11. [Architecture](#architecture)
12. [Deployment](#deployment)
13. [Performance](#performance)
14. [Scientific Context](#scientific-context)

---

## Overview

This is the **Rust Execution Layer** implementing the Zoo Vox Rosetta Engine - a complete signal-to-semantic pipeline for animal vocalization analysis.

The system follows a **Pipeline-First** architecture where Rust handles the entire 4-stage processing flow:
- **Stage 1:** Dynamic Segmentation (Change Point Detection)
- **Stage 2:** 45D Feature Extraction
- **Stage 3:** Cascaded Classification (Router → Analyzer)
- **Stage 4:** Semantic Grounding (Human-Guided Dictionary)

The design follows **"Fail Open to Safety"** - if the Python cognitive layer crashes, Rust continues in safe Passthrough Mode with raw recording.

---

## Zoo Vox Rosetta Engine v2.0

Multi-modality species adaptation framework for cross-species vocalization analysis. The v2.0 update adds support for species with different encoding strategies beyond temporal phrase patterns.

### Compatibility Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    SPECIES COMPATIBILITY (Updated for Dynamic Segmentation)              │
├───────────────────┬─────────────────┬─────────────────┬─────────────────────────────────┤
│ Species           │ Compatibility   │ Phrases/File    │ Key Enabler                     │
├───────────────────┼─────────────────┼─────────────────┼─────────────────────────────────┤
│ Sperm Whale       │ ✓✓✓ EXCELLENT   │ 428.9 (highest) │ Temporal patterns (codas)       │
│ Meerkat           │ ✓✓✓ EXCELLENT   │ 11.42           │ Multi-call sequences            │
│ Zebra Finch       │ ✓✓✓ EXCELLENT   │ 23.2            │ Dynamic Seg + N-gram syntax     │
│ Marmoset          │ ✓✓✓ EXCELLENT   │ 0.39            │ Semantic Grounding (graded)     │
│ Egyptian Bat      │ ✓✓ GOOD         │ 4.72            │ FM sweep weights + Duration     │
│ Dolphin           │ ✓✓ GOOD         │ 0.30            │ Dynamic Seg (1.5s whistles)     │
│ Orcas             │ ✓✓ GOOD         │ 7.5             │ Spectral + Sequence weights     │
│ Bird Songs        │ ✓✓ GOOD         │ 10.3            │ Standard phrase analysis        │
│ Macaque           │ ✓✓ GOOD         │ 1.0 (lowest)    │ Spectral Shape weights          │
│ Giant Otter       │ ✓ MODERATE      │ 1.4             │ Formant + Spectral features     │
└───────────────────┴─────────────────┴─────────────────┴─────────────────────────────────┘
```

### New Modules (v2.0)

#### 1. SpectralModule (Dolphin FM Whistles)

Analyzes frequency-modulated signals for species that use continuous frequency contours rather than discrete phrases.

```rust
use technical_architecture::{SpectralModule, ContourConfig, FMType};

let config = ContourConfig {
    min_sweep_range: 1000.0,  // Minimum frequency sweep in Hz
    min_duration_ms: 100.0,   // Minimum contour duration
    frequency_bins: 8,        // Discretization resolution
    time_bins: 10,
};
let module = SpectralModule::new(config);

// Analyze dolphin whistle
let contours = module.analyze(&audio, 192000);

for contour in &contours {
    println!("FM Type: {:?}", contour.features.fm_type);
    println!("Frequency range: {} - {} Hz",
        contour.features.f_min, contour.features.f_max);
}
```

**FM Classifications:**
- `Rising` - Upsweep (increasing frequency)
- `Falling` - Downsweep (decreasing frequency)
- `UShaped` - Down then up
- `InvertedU` - Up then down
- `Complex` - Multiple inflection points
- `Flat` - Minimal modulation

#### 2. SequenceModule (Combinatorial Syntax)

N-gram analysis for species with combinatorial phrase sequences (zebra finch, orcas).

```rust
use technical_architecture::SequenceModule;

let module = SequenceModule::new(3);  // Max trigram analysis

let sequence = vec![0, 1, 2, 3, 0, 1, 2, 4, 0, 1, 2];
let analysis = module.analyze(&sequence);

// Motif detection
for motif in &analysis.motifs {
    println!("Pattern {:?}: {} occurrences",
        motif.pattern, motif.occurrences);
}

// N-gram statistics
println!("Unique bigrams: {}", analysis.ngram_stats.unique_bigrams);
println!("Perplexity: {:.2}", analysis.perplexity);
```

#### 3. SpeciesConfigFactory (Species Adaptation)

Creates species-specific configurations with appropriate analysis modules and parameters.

```rust
use technical_architecture::{SpeciesConfigFactory, AnalysisModule, AnalysisModality};

// Get configuration for a species
let config = SpeciesConfigFactory::create("zebra_finch");

// Check required modules
if config.requires_module(AnalysisModule::Sequence) {
    // Use SequenceModule for zebra finch
}

// Check modality
match config.modality() {
    AnalysisModality::Temporal => /* phrase analysis */,
    AnalysisModality::Spectral => /* FM contour analysis */,
    AnalysisModality::Hybrid => /* combined approach */,
}
```

### Species-Specific Configurations

| Species | Encoding Strategy | Modality | Required Modules |
|---------|------------------|----------|------------------|
| Sperm Whale | CodaType | Temporal | Temporal |
| Dolphin | FrequencyModulated | Spectral | Spectral |
| Zebra Finch | Combinatorial | Temporal | Temporal, Sequence |
| Orca | Combinatorial | Hybrid | Temporal, Sequence, Spectral |
| Meerkat | Quantitative | Temporal | Temporal, Count |
| Egyptian Bat | DurationMediated | Temporal | Temporal, Duration |
| Marmoset | PhraseType | Temporal | Temporal |
| Macaque | Minimal | Temporal | Temporal, Spectral |
| Giant Otter | Minimal | Temporal | Temporal, Spectral |

### Cross-Species Analysis Results

Based on within-call phrase discovery analysis across multiple datasets:

| Dataset | Files | Phrases | Types | Entropy |
|---------|-------|---------|-------|---------|
| Dominica Sperm Whale | 39 | 16,729 | 5 | 0.226 bits |
| Zebra Finch | 143 | 3,319 | 15 | 2.352 bits |
| Bird Songs | 42 | 434 | 8 | ~1.5 bits |
| Orcas | 25 | 187 | 6 | ~1.2 bits |
| Giant Otter | 331 | 463 | 2 | ~0.5 bits |
| Macaque | 999 | 999 | 1 | ~0.0 bits |
| Dolphin Whistles | 3,215 | 965 | 2 | 0.926 bits |

**Key Findings:**
- **Sperm whale codas** are highly stereotyped (5 types, 0.226 bits entropy) - ideal for phrase detection
- **Zebra finch songs** have combinatorial syntax (15 types, 2.352 bits) - requires sequence analysis
- **Dolphin whistles** use FM contours, not phrases - requires spectral module
- **Macaque calls** are nearly uniform - requires spectral fine structure for discrimination

### Phrase Data Preparation Pipeline (v2.0)

The phrase data preparation system provides complete **45D acoustic feature extraction** and species-specific phrase segmentation for all supported species. See the [45D Acoustic Feature System](#45d-acoustic-feature-system) section for detailed feature documentation.

#### Modules

| Module | Purpose |
|--------|---------|
| `zoo_vox_data_models` | Core data structures (45D features, PhrasePrototype, libraries) |
| `zoo_vox_features` | 45D acoustic feature extraction |
| `zoo_vox_extraction` | Species-specific phrase segmentation |
| `zoo_vox_library` | Phrase library building and management |
| `zoo_vox_within_call` | Within-call phrase discovery using acoustic similarity |

#### Usage Example

```rust
use technical_architecture::{
    ZooVoxFeatureExtractor, ZooVoxPhraseExtractor,
    ZooVoxExtractionConfig, ZooVoxLibraryBuilder,
};

// Extract features from audio
let mut extractor = ZooVoxFeatureExtractor::new(48000);
let features = extractor.extract(&audio)?;

println!("F0: {} Hz", features.mean_f0_hz);
println!("Duration: {} ms", features.duration_ms);
println!("HNR: {} dB", features.harmonic_to_noise_ratio);

// Extract phrases with species-specific segmentation
let config = ZooVoxExtractionConfig::for_species("zebra_finch", 48000);
let mut phrase_extractor = ZooVoxPhraseExtractor::new(config);
let phrases = phrase_extractor.extract_phrases(&audio, "zebra_finch", None)?;

// Build phrase library
let builder = ZooVoxLibraryBuilder::new().with_similarity_threshold(0.85);
let library = builder.build_library(phrases, "zebra_finch", None)?;

println!("Total phrases: {}", library.total_phrases);
println!("Type entropy: {:.3} bits", library.type_entropy);
```

#### Running the Demo

```bash
cargo run --release --example zoo_vox_rosetta_phrase_data_demo
cargo run --release --example zoo_vox_zebra_finch_extraction
```

### Within-Call Phrase Discovery (Acoustic Similarity Engine)

The `zoo_vox_within_call` module implements phrase discovery within single vocalizations using acoustic similarity rather than clustering. This approach recognizes that animal vocalizations exist on **continuous acoustic manifolds**, not discrete islands.

#### Key Insight

```
Animal vocalizations form continuous gradients:
  Phee ←───────→ Trill ←───────→ Twitter ←──────→ Tsik
     (continuous acoustic transitions, not separate clusters)

HDBSCAN expects ISLANDS. You have a CONTINENT.
```

#### Components

| Component | Description |
|-----------|-------------|
| `WithinCallAnalyzer` | Main analyzer using acoustic similarity for phrase typing |
| `WithinCallConfig` | Species-specific configuration |
| `DiscoveredPhraseType` | Phrase type with centroid features and instances |
| `PhraseMotif` | Recurring patterns within calls |
| `SimilarityBasedLibraryBuilder` | Alternative library builder using similarity grouping |

#### Species-Specific Thresholds

| Species | Encoding | Similarity Threshold | Rationale |
|---------|----------|---------------------|-----------|
| Sperm Whale | Coda-type | 0.90 | Strict for short clicks |
| Dolphin/Orca | FM whistle | 0.80 | Loose for continuous contours |
| Zebra Finch | Combinatorial | 0.85 | Balanced for discrete syllables |
| Marmoset | Harmonic | 0.85 | Balanced for harmonic calls |

#### Usage Example

```rust
use technical_architecture::{
    WithinCallAnalyzer, WithinCallConfig,
    SimilarityBasedLibraryBuilder,
};

// Create analyzer with species-specific config
let mut analyzer = WithinCallAnalyzer::for_species("zebra_finch");

// Discover phrase types within a vocalization
let result = analyzer.discover_phrases(phrases, "call_001", "zebra_finch");

// View discovered types
for pt in &result.phrase_types {
    println!("Type {}: {} occurrences, F0={:.0}Hz",
        pt.type_id, pt.occurrence_count, pt.centroid_features.mean_f0_hz);
}

// Find recurring motifs
let motifs = analyzer.find_motifs(&result, 2, 2);
for motif in &motifs {
    println!("Pattern {:?}: {} occurrences",
        motif.pattern, motif.occurrence_count);
}

// Build library using similarity-based grouping
let builder = SimilarityBasedLibraryBuilder::for_species("zebra_finch");
let library = builder.build_library(phrases, "zebra_finch")?;
```

#### Comparison: Standard vs Similarity-Based

| Aspect | Standard (Key-Based) | Similarity-Based |
|--------|---------------------|------------------|
| Grouping method | F0 + Duration bins | Full 45D acoustic similarity |
| Approach | Discrete binning | Continuous acoustic manifold |
| Thresholds | Fixed | Species-specific |
| Variability metrics | No | Intra-type variability computed |
| Best for | Quick typing | Detailed within-call analysis |

#### Running the Demo

```bash
cargo run --release --example zoo_vox_within_call_demo
```

#### Test Coverage

All 9 unit tests passing:

```bash
cargo test zoo_vox_within_call --lib

# Tests:
# - test_within_call_config_default
# - test_within_call_config_species
# - test_analyzer_creation
# - test_analyzer_for_species
# - test_discover_phrases_empty
# - test_discover_phrases_basic
# - test_transition_matrix
# - test_find_motifs
# - test_similarity_based_library_builder
```

---

## 45D Acoustic Feature System

### Overview

The 45-dimensional feature vector provides comprehensive bioacoustic characterization for cross-species analysis with expanded coverage of fine temporal structure and psychoacoustic features.

### Feature Groups (9 groups × 5 features = 45D)

| Group | Dimensions | Description | Species Relevance |
|-------|------------|-------------|-------------------|
| **Spectral** | D0-D4 | Centroid, spread, skewness, kurtosis, tilt | All species |
| **Harmonic** | D5-D9 | F0, harmonicity, harmonic_ratio, inharmonicity, noise_ratio | Songbirds, primates |
| **Temporal** | D10-D14 | RMS, ZCR, attack, decay, sustain | All species |
| **Modulation** | D15-D19 | AM rate/depth, FM rate/slope, spectral_flux | Dolphins, bats |
| **Cepstral** | D20-D24 | MFCC 1-5 | All species |
| **Formant** | D25-D29 | F1, F2, F3, B1, B2 | Primates, humans |

---

## 112D Rosetta Features

### Overview

The **112D Rosetta Feature Vector** is the complete universal representation for cross-species vocalization analysis. It builds upon the 45D acoustic foundation by adding macro-texture and micro-texture layers for fine-grained species discrimination.

### Architecture (112D = 46D + 30D + 36D)

```
┌─────────────────────────────────────────────────────────────────┐
│                    112D ROSETTA FEATURES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  LAYER 1 (46D)  │  │  LAYER 2 (30D)  │  │  LAYER 3 (36D)  │ │
│  │  Base Physics   │  │  Macro Texture  │  │  Micro Texture  │ │
│  │                 │  │                 │  │                 │ │
│  │  - F0 (3D)      │  │  - Harmonic     │  │  - Spectral     │ │
│  │  - Duration (1D)│  │    Texture      │  │    Derivatives  │ │
│  │  - Energy (3D)  │  │  - Pitch        │  │  - FM/AM Bins   │ │
│  │  - Harmonicity  │  │    Geometry     │  │  - ICI Bins     │ │
│  │    (3D)         │  │  - GLCM Texture │  │  - Rhythm Hist  │ │
│  │  - Envelope (4D)│  │                 │  │                 │ │
│  │  - MFCC (14D)   │  │  Universal      │  │  Species-       │ │
│  │  - Spectral     │  │  Taxonomy       │  │  Specific       │ │
│  │    Shape (12D)  │  │  Classification │  │  Identity       │ │
│  │  - Formants (6D)│  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│           │                    │                    │           │
│           └────────────────────┴────────────────────┘           │
│                           │                                    │
│                           ▼                                    │
│              Universal Taxonomic Classification                 │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1: Base Physics (46D)

Universal acoustic features that apply to all species:

| Feature Group | Dimensions | Description |
|---------------|------------|-------------|
| F0 | 3D | mean, std_dev, range |
| Duration | 1D | phrase duration in ms |
| Energy | 3D | rms, max, dynamic_range |
| Harmonicity | 3D | hnr, harmonic_to_noise, spectral_flatness |
| Envelope | 4D | attack_time, decay_time, sustain_level, vibrato_depth |
| MFCC | 14D | Spectral envelope coefficients |
| Spectral Shape | 12D | centroid, spread, skewness, kurtosis, slope, etc. |
| Formants | 6D | f1, f2, f3, b1, b2, b3 |

### Layer 2: Macro Texture (30D)

Species group discrimination through harmonic and pitch analysis:

| Feature Group | Dimensions | Description |
|---------------|------------|-------------|
| Harmonic Texture | 12D | Harmonicity multi-scale (mean, std, skew, kurt, range, iqr) × 3 time windows |
| Pitch Geometry | 9D | F0 contour statistics (mean, std, min, max, range, slope, etc.) |
| GLCM Texture | 9D | Gray-level co-occurrence matrix (contrast, dissimilarity, homogeneity) |

### Layer 3: Micro Texture (36D)

Fine species identity through temporal and spectral micro-structure:

| Feature Group | Dimensions | Description |
|---------------|------------|-------------|
| Spectral Derivatives | 13D | Delta MFCC coefficients |
| FM/AM Bins | 8D | Frequency and amplitude modulation distribution |
| ICI Bins | 10D | Inter-call interval histogram |
| Rhythm Histogram | 5D | Onset rate and rhythmic patterns |

### Extraction API

```rust
use technical_architecture::MicroDynamicsExtractor;

// Create extractor with sample rate
let extractor = MicroDynamicsExtractor::new(48000);

// Extract 112D features from audio buffer
let features_112d = extractor.extract_rosetta(&audio)?;

// Access individual layers
let base_physics = &features_112d.base_physics;  // 46D
let macro_texture = &features_112d.macro_texture; // 30D
let micro_texture = &features_112d.micro_texture; // 36D

// Convert to flat vector for ML
let flat_vector: Vec<f32> = features_112d.to_flat_vec(); // 112D
```

### Usage in PAM Pipeline

The 112D features are the primary input for species classification in the PAM Pipeline:

```rust
use technical_architecture::{MicroDynamicsExtractor, PAMRouter};

// Extract features
let extractor = MicroDynamicsExtractor::new(48000);
let features_112d = extractor.extract_rosetta(&phrase_audio)?;

// Route to acoustic specialist
let router = PAMRouter::with_acoustic_groups()?;
let group = router.route_to_group(&features_112d)?;

// Classify with species-specific model
let result = router.classify(&features_112d, group)?;
println!("Species: {}, Confidence: {:.2}", result.species, result.confidence);
```

---
| **Micro-dynamics** | D30-D34 | Onset rate, median ICI, ICI CV, burst rate, gap rate | Sperm whales, insects |
| **Psychoacoustic** | D35-D39 | Loudness, sharpness, roughness, fluctuation, tonality | All species |
| **TFS** | D40-D44 | ACF peak, ACF strength, SFM, periodicity, entropy | All species |

### Usage

```rust
use technical_architecture::ZooVoxFeatureExtractor;

let mut extractor = ZooVoxFeatureExtractor::new(sample_rate);
let features = extractor.extract_45d(&audio)?;

// Access individual features
println!("Spectral centroid: {}", features.spectral_centroid());
println!("FM slope: {}", features.fm_slope());
println!("Onset rate: {}", features.onset_rate());

// Get full 45D vector
let vector = features.to_vector();
```

### BEANS-Zero Benchmark Results

The 45D feature system was validated on the BEANS-Zero benchmark (92K samples across 13 datasets):

| Metric | Result |
|--------|--------|
| Overall F1 | 47.83% |
| Best Dataset | HICEAS (90.4%) - Marine mammals |
| Feature extraction | ~5 min for 20K samples |

---

## Dynamic Segmentation (Change Point Detection)

### Overview

The dynamic segmenter discovers atomic phrase units using change point detection rather than fixed thresholds. This adapts to species with different vocalization tempos automatically.

### Algorithm

1. **Hierarchical Segmentation**: Motif → Syllable → Note levels
2. **Change Point Detection**: Statistical tests on spectral/temporal features
3. **Species-Specific Thresholds**: Tempo factors scale for fast (bats) vs slow (whales) species

### Key Parameters

```rust
use technical_architecture::{
    DynamicSegmenter, DynamicSegmenterConfig,
    species::SpeciesConfigFactory,
};

// Get species-specific config
let factory = SpeciesConfigFactory::create("zebra_finch");
let thresholds = factory.hierarchical_thresholds();

// Create segmenter
let config = DynamicSegmenterConfig {
    motif_min_ms: thresholds.motif_min_ms,
    syllable_min_ms: thresholds.syllable_min_ms,
    note_min_ms: thresholds.note_min_ms,
    ..Default::default()
};

let segmenter = DynamicSegmenter::new(config);
let candidates = segmenter.segment(&audio, sample_rate)?;
```

### Species Tempo Factors

| Species | Tempo Factor | Motif Min | Notes |
|---------|--------------|-----------|-------|
| Zebra Finch | 1.0 (reference) | 50ms | Standard bird tempo |
| Egyptian Bat | 0.3 | 15ms | 3x faster (echolocation) |
| Dolphin | 2.0 | 100ms | Slow whistles |
| Sperm Whale | 2.5 | 125ms | Very slow codas |

---

## Grading Score System

### Overview

The grading score system distinguishes between **discrete** and **graded** vocalizations within a species' repertoire. This is critical for understanding communication complexity.

### Discrete vs Graded Vocalizations

- **Discrete**: Tight clusters, low intra-type variance (< 0.05 grading score)
- **Graded**: Continuous transitions, high intra-type variance (> 0.05 grading score)

### Implementation

```rust
use technical_architecture::{
    TypedPhraseCandidate, EmissionStrategy,
    dynamic_segmenter::GRADING_THRESHOLD,
};

// After phrase typing
for typed in &typed_candidates {
    if typed.is_graded {
        // Graded call - emit type ID + 45D vector
        // Use Continuous emission strategy
        println!("Type {} is GRADED (score: {:.3})",
            typed.phrase_type_id, typed.grading_score);
    } else {
        // Discrete call - emit type ID only
        // Use Discrete emission strategy
        println!("Type {} is DISCRETE (score: {:.3})",
            typed.phrase_type_id, typed.grading_score);
    }
}
```

### Marmoset Analysis Results

Analysis of 964,643 marmoset phrase candidates:

| Type | Occurrences | Grading Score | Classification |
|------|-------------|---------------|----------------|
| Type_1 | 312,451 | 0.021 | Discrete (stereotyped) |
| Type_2 | 287,234 | 0.018 | Discrete (stereotyped) |
| Type_3 | 156,892 | 0.067 | Graded (42% instances) |
| Overall | 23 types | - | **88.9% Discrete** |

**Finding**: Marmosets are primarily discrete callers with some graded elements, contrary to pure-continuous models.

### Emission Strategy

| Strategy | Output | Bandwidth | Use Case |
|----------|--------|-----------|----------|
| Discrete | Type ID only | ~2 bytes | Stereotyped calls |
| Continuous | Type ID + 45D vector | ~182 bytes | Graded calls needing nuance |

---

## Cascaded Architecture: Router → Analyzer

### Overview

The BEANS-Zero benchmark revealed that **species-specific weights cannot be used in global k-NN search**. Different weights per prototype make distances non-comparable.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ZOO VOX ROSETTA ENGINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PHASE 1: Global Discrimination         PHASE 2: Contextual Analysis   │
│   ┌─────────────────────────┐           ┌─────────────────────────────┐ │
│   │ Species Identification  │           │ Phrase/State Classification │ │
│   │                         │           │                             │ │
│   │ • Unified 45D Space     │ ───────►  │ • Species-Specific Weights  │ │
│   │ • k-NN Search           │  Dolphin  │ • Grading Score Analysis    │ │
│   │ • "Universal Ruler"     │           │ • Within-Type Variance      │ │
│   └─────────────────────────┘           └─────────────────────────────┘ │
│                                                                         │
│   Goal: "What species?"                 Goal: "What phrase type?"       │
│   Weights: Unified/Equal                Weights: Species-Specific       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Benchmark Results

| Approach | F1 Score | Change |
|----------|----------|--------|
| Unweighted (baseline) | 47.83% | - |
| Unified weights | 47.13% | -1.46% |
| Species-specific per prototype | 29.71% | **-37.88%** |

**Key Finding**: Applying different weights per prototype breaks distance comparability. Species-specific weights must only be applied in Phase 2 (within-species analysis).

### Implementation

```rust
// Phase 1: Global Species ID (unified weights)
let global_engine = AcousticSimilarityEngine::with_metric(45, SimilarityMetric::Cosine);
// Uses equal weights for all comparisons
let species = identify_species(&query, &prototypes, &global_engine);

// Phase 2: Within-species analysis (species-specific weights)
let analyzer = WithinCallAnalyzer::for_species(&species);
// Now uses species-specific weights for phrase typing
let result = analyzer.discover_phrases(phrases, call_id, &species);
```

### Species-Specific Weights

| Species | Key Weight | Rationale |
|---------|-----------|-----------|
| Dolphin | Modulation: 2.5x | FM slope critical for whistles |
| Sperm Whale | Temporal: 2.5x | Timing is everything for codas |
| Zebra Finch | Harmonic: 1.8x | Harmonic stack structure in song |
| Egyptian Bat | Micro-dynamics: 2.0x | Rapid changes in echolocation |

---

## Human-Guided Context Discovery

### Overview

The `AnnotationAligner` module bridges human annotations (Raven, Audacity, CSV) with dynamically discovered phrases. This enables **semi-supervised ground truthing** - anchoring acoustic types to biological meaning.

### Strategy: "Anchor and Propagate"

1. **Input**: Audio + Human Annotation File
2. **Discovery**: Dynamic segmentation finds precise boundaries
3. **Alignment**: Match discovered phrases to annotations (time overlap)
4. **Labeling**: Assign semantic labels to phrase types
5. **Propagation**: Use similarity engine to find labeled types in unlabeled data

### Supported Formats

| Format | Description |
|--------|-------------|
| Raven | Pro selection tables (tab-separated) |
| Audacity | Label tracks (tab-separated: start, end, label) |
| CSV | start_ms, end_ms, label, [context] |
| JSON | Array of annotation objects |

### Usage

```rust
use technical_architecture::{
    AnnotationAligner, HumanAnnotation, AnnotationFormat,
    SemanticPhraseDictionary,
};

let aligner = AnnotationAligner::new();

// Parse annotations from file
let annotations = aligner.parse_annotations(&content, AnnotationFormat::Audacity)?;

// Align with discovered phrases
let labeled = aligner.align(&candidates, &annotations);

// Build semantic dictionary
let dict = SemanticPhraseDictionary::build(&labeled, &clusters, &features);

// Query semantic meaning
for type_id in dict.type_centroids.keys() {
    println!("{}: {}", type_id, dict.describe_type(type_id));
    // Output: "Type_1: Alarm (90%) [Predator]"
}

// Save for field deployment
let json = dict.to_json()?;
std::fs::write("semantic_dictionary.json", &json)?;
```

### Fuzzy vs Precise Boundaries

Human annotations are often approximate. The dynamic segmenter provides precise boundaries:

```
Human Annotation:  [0.0s - 1.0s] "Song"
Dynamic Segmenter: [0.05s - 0.35s] Syllable A
                    [0.40s - 0.90s] Syllable B

Result: Both syllables tagged with context "Song"
```

This enables studying **internal syntax** of complex vocalizations.

### Recommended Workflow

1. **Select Gold Standard Files**: 50-100 representative files per species
2. **Annotate**: Label broad contexts (Feeding, Alarm, Social)
3. **Run Pipeline**: Align → Extract → Build dictionary
4. **Deploy**: Load dictionary into Field Engine
5. **Field Operation**: Match new phrases to labeled types

---

## Rosetta Pipeline (Integrated Zoo Vox Rosetta Engine)

The Rosetta Pipeline integrates all components into a unified system for field deployment:

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ROSETTA PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: Audio Stream + Environmental State                                   │
│                      │                                                       │
│                      ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 1: GLOBAL SPECIES IDENTIFICATION                                │   │
│  │ • Uses unified weights for cross-species comparison                  │   │
│  │ • 45D feature extraction                                             │   │
│  │ • Matches to all loaded species dictionaries                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                      │                                                       │
│                      ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 2a: SEMANTIC GROUNDING (Human-Guided)                          │   │
│  │ • Lookup acoustic type in pre-seeded dictionary                      │   │
│  │ • Maps "Type_53" → "Phee" with 95% confidence                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                      │                                                       │
│                      ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 2b: CONTEXTUAL ENRICHMENT                                      │   │
│  │ • Combine Semantic Label + Situational Context                       │   │
│  │ • "Phee" + "Windy" → "Long_Range_Contact"                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                      │                                                       │
│                      ▼                                                       │
│  OUTPUT: ContextEnrichedPhrase                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Concepts

#### Semantic Identity (The "What")
- **Source:** Human-Guided Dictionary (Offline)
- **Logic:** `AnnotationAligner` maps acoustic types to semantic labels
- **Output:** `semantic_label: "Phee_Call"`

#### Pragmatic Context (The "Why/When")
- **Source:** Real-time Sensors & Syntax Analysis (Online)
- **Logic:**
  - *Syntax:* 5th repetition → "High Urgency"
  - *Environment:* High wind → "Long-range Contact"
- **Output:** `inferred_intent: "Long_Range_Contact"`

### Usage

```rust
use technical_architecture::{
    RosettaPipeline, RosettaBundle, SemanticPhraseDictionary,
    FeatureWeights, EnvState,
};

// 1. Create semantic dictionary from human annotations
let dictionary = SemanticPhraseDictionary {
    species: "marmoset".to_string(),
    type_to_labels,  // From Human-Guided Discovery
    type_centroids,  // From feature extraction
    total_phrases: 5000,
    num_types: 3700,
};

// 2. Create deployable bundle
let bundle = RosettaBundle::new(
    "marmoset",
    FeatureWeights::marmoset(),
    dictionary,
    FeatureWeights::unified(),
);

// 3. Initialize pipeline and load bundles
let mut pipeline = RosettaPipeline::new()?;
pipeline.load_bundle(bundle);

// 4. Process audio with environmental context
let result = pipeline.process_stream(&audio, EnvState::Wind)?;

// 5. Access context-enriched phrases
for phrase in &result.phrases {
    println!("Label: {} ({}% confidence)",
        phrase.semantic_label,
        phrase.label_confidence * 100.0);
    println!("Intent: {}", phrase.inferred_intent);
}
```

### Output Structure: ContextEnrichedPhrase

```rust
pub struct ContextEnrichedPhrase {
    // Layer 1: Acoustic Identity
    pub phrase_type_id: String,      // "Type_53"
    pub grading_score: f32,          // 0.0 = discrete, 1.0 = graded

    // Layer 2: Semantic Identity (from Human Annotations)
    pub semantic_label: String,      // "Phee_Call"
    pub label_confidence: f32,       // 0.95

    // Layer 3: Pragmatic Context (from Environment/Syntax)
    pub syntax_role: SyntaxRole,     // Initiator, Reply, Solo
    pub environmental_state: EnvState, // Quiet, Wind, Rain, Storm
    pub inferred_intent: String,     // "Long_Range_Contact"
}
```

### Bundle Serialization

The `RosettaBundle` packages everything needed for field deployment:

```rust
// Save bundle for deployment
bundle.save("marmoset_rosetta.json")?;

// Or compressed binary
bundle.save_binary("marmoset_rosetta.rosetta")?;

// Load in field engine
let bundle = RosettaBundle::load_binary("marmoset_rosetta.rosetta")?;
pipeline.load_bundle(bundle);
```

### Intent Inference Rules

| Semantic Label | Environment | Inferred Intent |
|----------------|-------------|-----------------|
| Phee | Wind | Long_Range_Contact |
| Phee | Quiet | Social_Contact |
| Tsik | Storm | Emergency_Alert |
| Tsik | Quiet | Warning |
| Twitter | Any | Social_Bonding |
| Fighting | Any | Aggression |
| Mating | Any | Reproductive |

---

## Research Workflow

The research workflow is now centered on **building and validating Semantic Dictionaries** rather than simply populating a database.

### Phase 1: Human-Guided Dictionary Construction

**Step 1: Define Gold Standard Dataset**
Select a representative subset of audio files (50-2000) per species.
- Cover diverse contexts: Feeding, Alarm, Social, Mating
- Include variations in individual, time of day, environment

**Step 2: Expert Annotation**
Use tools like Raven or Audacity to label vocalizations.
- Input: Audio files + Timestamps
- Output: Annotation files (`.txt` or `.csv`) with context labels

**Step 3: Run Human-Guided Discovery**
Execute the discovery pipeline:
```bash
# Egyptian Fruit Bats
cargo run --release --example human_guided_context_discovery

# Marmosets
cargo run --release --example marmoset_context_discovery --features symphonia
```

The pipeline performs:
- **Dynamic Segmentation:** Finds precise phrase boundaries
- **45D Feature Extraction:** Comprehensive acoustic analysis
- **Alignment:** Maps boundaries to human labels
- **Output:** `SemanticPhraseDictionary` (Acoustic Type → Semantic Label)

### Phase 2: Bundle Creation

**Step 4: Create RosettaBundle**
Package the dictionary with species-specific weights:
```rust
use technical_architecture::{
    RosettaBundle, SemanticPhraseDictionary, FeatureWeights
};

let bundle = RosettaBundle::new(
    "marmoset",
    FeatureWeights::marmoset(),
    dictionary,  // From Step 3
    FeatureWeights::unified(),
);

// Save for deployment
bundle.save("marmoset_rosetta.json")?;
bundle.save_binary("marmoset_rosetta.rosetta")?;  // Compressed
```

### Phase 3: Field Validation

**Step 5: Live Context Inference**
Run the `RosettaPipeline` on raw field audio:
```rust
let mut pipeline = RosettaPipeline::new()?;
pipeline.load_bundle_from_file("marmoset_rosetta.rosetta")?;

let result = pipeline.process_stream(&audio, EnvState::Wind)?;
for phrase in &result.phrases {
    println!("Label: {} ({}% confidence)",
        phrase.semantic_label, phrase.label_confidence * 100.0);
    println!("Intent: {}", phrase.inferred_intent);
}
```

**Step 6: Validate Output**
- Monitor for `ContextEnrichedPhrase` events
- Validate that "Inferred Intent" matches observed behavior
- Flag "Novel" phrases for human review (Active Learning)

### Expected Results

| Species | Files | Phrase Types | Primary Labels |
|---------|-------|--------------|----------------|
| Egyptian Fruit Bats | 2,000 | 768 | Fighting, Grooming, Mating, Protest |
| Marmosets | 5,000 | 3,700 | Phee, Twitter, Tsik, Trill, Infant_cry |

| Persona | Acoustic Characteristics | Use Case |
|---------|------------------------|----------|
| **PURE** | HNR > 20dB, flatness < 0.1, slow attack (>20ms) | Validate tonal synthesis quality |
| **GRITTY** | HNR < 5dB, flatness > 0.6, fast attack (<5ms) | Validate texture/noise reproduction |
| **RHYTHMIC** | Onset rate > 15Hz, regular ICI (CV < 0.3) | Validate temporal pattern synthesis |
| **HARMONIC** | Zero onset rate, zero ICI | Validate continuous tone synthesis |

**Step 8: Bio-Acoustic Turing Test**
```bash
python3 ../realtime/demo_bio_acoustic_turing_test.py
```
- Live animal testing
- Determines if animals distinguish natural vs. synthesized
- Validates behavioral relevance

### Phase 5: Field Deployment

**Step 9: Production Deployment**
```bash
# Deploy Rust engine and Python agent
sudo cp deployment/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start rust-field-engine.service
sudo systemctl start python-cognitive-agent.service
```

**Step 10: Field Validation**
- Environmental monitoring (rain, temperature, light)
- Power management (battery, solar optimization)
- Wildlife sentry (background species detection)
- Offline data synchronization

---

## Features

### Core Modules
- **Synthesis** - Granular, concatenative, superpositional synthesis engines
  - **Granular Concatenative Synthesis**: t-SNE distance 6.452 (< 7.0 target) ✅
  - 76.1% improvement over additive synthesis (distance 27.0)
  - Preserves formant structure while enabling pitch/time manipulation
  - See `GRANULAR_SYNTHESIS_FINDINGS.md` in parent directory
- **Source Separation** - Conv-TasNet via ONNX/Tract
- **PTP Clock** - IEEE 1588 precision timing (nanosecond accuracy)
- **Safety Monitor** - Watchdog timers and safety limits
- **Thermal Management** - Temperature monitoring and throttling
- **Provenance Logging** - Deterministic audit trails

### Production Deployment (All with comprehensive tests)
| Module | Tests | Description |
|--------|-------|-------------|
| IACUC Compliance | 29 | Legal animal research protocol enforcement |
| Time-Series Archive | 24 | High-frequency data storage and querying |
| Auto-Calibration | 17 | Self-health checks with drift detection |
| Shadow Model Monitoring | 26 | AI concept drift detection and rollback |
| Remote Web Dashboard | 25 | HTTPS/WebSocket monitoring and control |
| Multi-Node Coordination | 21 | PTP election and TDMA scheduling |

### Field Deployment
| Module | Tests | Description |
|--------|-------|-------------|
| Environmental Monitor | 46 | Rain, temperature, light sensing |
| Power Manager | 54 | Battery/solar optimization |
| Wildlife Sentry | 24 | Background species detection |
| Data Synchronizer | 20 | Offline black box queuing |
| Acoustic Simulator | 43 | TDD test fixture |

---

## Build

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name

# Run benchmarks
cargo run --example benchmark_peer_controller --release
```

---

## Test Coverage

**1287 tests passing** - Comprehensive coverage of all functionality

```
Rust Execution Layer: 1287 tests passing
├── Rust Tests: 463+ tests
│   ├── Core Modules: 187 tests
│   ├── Production Deployment: 142 tests
│   ├── Field Deployment: 187 tests
│   ├── 17D Metadata: 6 tests
│   ├── Granular Synthesis: 5 tests
│   ├── Zoo Vox Rosetta 2.0: 27 tests
│   │   ├── zoo_vox_data_models: 6 tests
│   │   ├── zoo_vox_features: 4 tests
│   │   ├── zoo_vox_extraction: 5 tests
│   │   ├── zoo_vox_library: 3 tests
│   │   └── zoo_vox_within_call: 9 tests
│   ├── 45D Acoustic Features: 12 tests
│   ├── Dynamic Segmenter: 15 tests
│   ├── Acoustic Similarity Engine: 8 tests
│   ├── Species Config: 54 tests
│   ├── Annotation Aligner: 5 tests
│   └── Semantic Dictionary: 3 tests
└── Python Tests: 12 tests (Formant Barrier Validation)
    ├── test_harmonic_cannot_become_transient ✅
    ├── test_transient_cannot_become_harmonic ✅
    ├── test_harmonic_preserves_spectral_flatness ✅
    ├── test_transient_preserves_spectral_flatness ✅
    ├── test_nearest_neighbor_preserves_modality ✅
    ├── test_persona_router_logic ✅
    ├── test_persona_switching_for_modality ✅
    ├── test_small_grains_create_temporal_artifacts ✅
    └── ...and more
```

### Zoo Vox Rosetta 2.0 Tests

Run the phrase data preparation tests:

```bash
# All Zoo Vox tests
cargo test zoo_vox --lib

# Specific modules
cargo test zoo_vox_data_models --lib
cargo test zoo_vox_features --lib
cargo test zoo_vox_extraction --lib
cargo test zoo_vox_library --lib
cargo test zoo_vox_within_call --lib
```

---

## Python Integration

This library can be used as a Python module via PyO3 bindings:

```python
import technical_architecture

# Create technical architect
architect = technical_architecture.TechnicalArchitect()

# Get operation mode
mode = architect.get_operation_mode()
```

See `CLAUDE.md` for detailed API documentation.

---

## Query Interface - Semantic Search

The query interface now supports searching by **Semantic Meaning** rather than just acoustic parameters. This enables researchers to find vocalizations based on what they *mean*, not just how they sound.

### Semantic Queries

Search by semantic labels from the Human-Guided Dictionary:

```python
from query_interface import get_query_interface

qi = get_query_interface()

# Find all alarm calls across all species
alarm_calls = qi.search_by_semantic_label("Alarm")

# Find all contact calls with high confidence
contact_calls = qi.search_by_semantic_label("Contact", min_confidence=0.8)

# Find calls with high emotional intensity (Grading Score > 0.7)
intense_calls = qi.search_by_grading_score(min_score=0.7)

# Find context-specific interactions
long_range = qi.search_by_intent("Long_Range_Contact")
```

### Intent-Based Search

Search by inferred intent (combination of semantic label + environmental context):

```python
# Find all territorial defense calls
territorial = qi.search_by_intent("Territorial_Defense")

# Find all social bonding interactions
social = qi.search_by_intent("Social_Bonding")

# Find emergency alerts (high urgency in adverse conditions)
emergency = qi.search_by_intent("Emergency_Alert")
```

### Cross-Species Semantic Search

Find similar semantic contexts across species:

```python
# Find all "alarm" type calls across species
# Marmoset: Tsik, Bat: Protest, Bird: Alarm Call
cross_species_alarms = qi.search_semantic_across_species("Alarm")

for species, calls in cross_species_alarms.items():
    print(f"\n{species}:")
    for call in calls[:5]:
        print(f"  {call.semantic_label}: {call.label_confidence:.0%}")
```

### Legacy Acoustic Search

Acoustic search (by 45D vector similarity) is still available for discovering **novel phrases** not yet in the dictionary:

```python
# Find phrases acoustically similar to a target
similar = qi.find_similar_45d("marmoset_phee_001", k=10)

# Find phrases by F0 range (for acoustic analysis)
by_pitch = qi.search_by_f0_range(5000, 10000)

# Find phrases by duration
by_duration = qi.search_by_duration(50, 200)
```

### API Reference

**Semantic Search Methods:**
- `search_by_semantic_label(label, min_confidence=0.0)` - Search by semantic label
- `search_by_intent(intent)` - Search by inferred intent
- `search_by_grading_score(min_score, max_score=1.0)` - Search by grading score
- `search_semantic_across_species(label)` - Cross-species semantic search

**Legacy Acoustic Methods:**
- `find_similar_45d(phrase_key, k=5)` - Find k nearest neighbors in 45D space
- `search_by_f0_range(min_hz, max_hz)` - Search by fundamental frequency
- `search_by_duration(min_ms, max_ms)` - Search by duration

---

## PAM Pipeline - Passive Acoustic Monitoring

### Overview

The **PAM Pipeline** is a complete 4-phase Passive Acoustic Monitoring system designed for field deployment in wildlife monitoring scenarios. It processes audio streams in real-time to detect and classify animal vocalizations.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PAM Pipeline Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Phase 1    │───▶│   Phase 2    │───▶│   Phase 3    │───▶│  Phase 4  │ │
│  │  Ingestion   │    │   Routing    │    │  Filtering   │    │  Output   │ │
│  │  & Boundary  │    │  112D Feat   │    │  Threshold   │    │  & AL     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ Streaming    │    │ PAMRouter    │    │ Confidence   │    │ Detection │ │
│  │ Buffer       │    │ AcousticGrp  │    │ >= 1.5       │    │ Payload   │ │
│  │ NBD          │    │ Specialists  │    │              │    │ JSON      │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase Descriptions

| Phase | Component | Description |
|-------|-----------|-------------|
| **1** | Real-Time Ingestion & Boundary Detection | Streaming audio buffer with system timestamps and Neural Boundary Detection |
| **2** | Feature Extraction & Hierarchical Routing | 112D Rosetta features → Acoustic Group routing |
| **3** | Confidence Threshold Filtering | Confidence >= 1.5 required for detection |
| **4** | Active Learning & Output | Uncertainty flagging and JSON payload output |

### Acoustic Groups (13 Specialized Classifiers)

| Group | Species | Acoustic Characteristics |
|-------|---------|--------------------------|
| HARMONIC_SONG | Songbirds, Zebra Finch | Rich harmonic content, clear F0 |
| FREQUENCY_MODULATED | Dolphins, Bats | FM sweeps, whistles, clicks |
| BROADBAND_NOISY | Primates, Macaques | Noisy screams, grunts |
| PULED_STACCATO | Sperm Whales | Click trains, codas |
| HARMONIC_TRILLED | Marmosets, Warblers | Rapid pitch modulation |
| LOW_FREQUENCY | Elephants, Bison | < 500 Hz fundamental |
| HIGH_FREQUENCY | Bats, Shrews | > 20 kHz ultrasonic |
| COMPLEX_MULTIMODAL | Orca, Giant Otter | Multiple encoding strategies |
| GRADUAL_TRANSITION | Meerkat, Suricata | Smooth vocalization transitions |
| TRANSIENT_IMPULS | Seals, Sea Lions | Short impulsive sounds |
| ENVIRONMENTAL_SOUND | Wind, Rain, Insects | Non-biological ambient |
| UNKNOWN_CLASS | Unidentified | Insufficient training data |
| SILENCE_BACKGROUND | Quiet periods | Below energy threshold |

### Usage

```bash
# Real-time mode (read from stdin)
cargo run --release --bin pam_pipeline -- --real-time

# Process audio file (raw f32 samples)
cargo run --release --bin pam_pipeline -- --input audio.raw

# Custom confidence threshold
cargo run --release --bin pam_pipeline -- --threshold 1.5 --input audio.raw

# Verbose output
cargo run --release --bin pam_pipeline -- --verbose --format jsonl --input audio.raw
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | None | Input audio file (raw f32 samples) |
| `--real-time` | false | Read from stdin in real-time |
| `--threshold` | 1.5 | Confidence threshold for detection |
| `--sample-rate` | 44100 | Sample rate in Hz |
| `--hop-size` | 512 | Hop size in samples |
| `--min-phrase-duration` | 50.0 | Minimum phrase duration in ms |
| `--al-low` | 1.4 | Active learning lower margin |
| `--al-high` | 1.5 | Active learning upper margin |
| `--format` | jsonl | Output format (jsonl, json, text) |
| `--verbose` | false | Enable debug logging |

### Output Format

```json
{
  "timestamp": "2026-03-10T23:50:12Z",
  "phrase_start_ms": 1234,
  "phrase_duration_ms": 567,
  "species": "zebra_finch",
  "confidence": 1.87,
  "acoustic_group": "HARMONIC_SONG",
  "features_112d": [/* 112 float values */],
  "active_learning_flag": false
}
```

### Documentation

See [`docs/pam_pipeline_guide.md`](docs/pam_pipeline_guide.md) for comprehensive documentation including:
- Phase-by-phase architecture details
- Active learning integration
- Acoustic specialist methodology
- Example workflows

---

## Closed-Loop Interaction Agent

### Overview

The **Closed-Loop Interaction Agent** enables real-time bidirectional communication between the Rust Execution Layer and Python Logic Layer for cognitive intelligence and decision-making.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLOSED-LOOP AGENT SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     RUST EXECUTION LAYER                              │   │
│  │                                                                       │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐   │   │
│  │  │  Audio Input    │───►│       NBD       │───►│   112D Feature  │   │   │
│  │  │  (48kHz, mono)  │    │  (Boundaries)   │    │   Extraction    │   │   │
│  │  └─────────────────┘    └─────────────────┘    └────────┬────────┘   │   │
│  │                                                          │            │   │
│  │  ┌─────────────────┐                            ┌───────▼────────┐   │   │
│  │  │   Synthesis     │◄───────────────────────────│   FeatureEvent │   │   │
│  │  │   Pipeline      │                            │   Publisher    │   │   │
│  │  └────────┬────────┘                            └────────────────┘   │   │
│  │           │                                                     │   │   │
│  │  ┌────────▼────────┐                            ┌────────────────┐   │   │
│  │  │  Audio Output   │                            │   ActionSub-   │   │   │
│  │  │  (Speaker/DAC)  │                            │   scriber      │   │   │
│  │  └─────────────────┘                            └───────▲────────┘   │   │
│  │                                                         │            │   │
│  └─────────────────────────────────────────────────────────│────────────┘   │
│                                                            │                │
│                              ┌─────────────────────────────▼──────────────┐ │
│                              │         ZeroMQ IPC Transport               │ │
│                              │  ipc:///tmp/cognitive_features.ipc (PUB)   │ │
│                              │  ipc:///tmp/cognitive_actions.ipc (SUB)    │ │
│                              └─────────────────────────────┬──────────────┘ │
│                                                          │                 │
│  ┌───────────────────────────────────────────────────────▼──────────────┐  │
│  │                     PYTHON LOGIC LAYER                                │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  FeatureSubscriber ◄─────────────────────────────────────────┐ │  │  │
│  │  │  - Receives 112D features from Rust                          │ │  │  │
│  │  │  - Context inference (alarm, contact, social)               │ │  │  │
│  │  │  └──────────────────────────────────────────────────────────┘ │  │  │
│  │  │                                                               │  │  │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │  │  │
│  │  │  │  InteractionAgent (Cognitive Intelligence)              │  │  │  │
│  │  │  │  - Strategy Pattern parsing (Compositional/Holophrastic)│ │  │  │
│  │  │  │  - Confidence threshold gating                           │  │  │  │
│  │  │  │  - Response rate limiting                                │  │  │  │
│  │  │  └─────────────────────────────────────────────────────────┘  │  │  │
│  │  │                                                               │  │  │
│  │  │  ┌──────────────────────────────────────────────────────────┐ │  │  │
│  │  │  │  ActionPublisher ──────────────────────────────────────►│ │  │  │
│  │  │  │  - Sends synthesis timelines to Rust                    │ │  │  │
│  │  └────────┴──────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Language | Role |
|-----------|----------|------|
| `FeatureEventPublisher` | Rust | Streams 112D features over ZeroMQ PUB |
| `FeatureSubscriber` | Python | Receives feature events with numpy arrays |
| `InteractionAgent` | Python | Cognitive processing and decision making |
| `ParsingStrategyFactory` | Python | Creates domain-specific parsers |
| `ActionPublisher` | Python | Sends synthesis timelines to Rust |
| `ActionSubscriber` | Rust | Receives synthesis commands |

### Data Structures

**FeatureEvent** (Rust → Python):
```rust
pub struct FeatureEvent {
    pub cluster_id: u32,
    pub features_112d: Vec<f32>,
    pub timestamp: f64,
    pub sequence: u64,
    pub emitter_id: Option<i32>,  // NEW: Source separation identity
}
```

**SynthesisAction** (Python → Rust):
```python
@dataclass
class SynthesisAction:
    timeline: List[TimelineEvent]
    micro_deltas: List[MicroDynamicsDelta]
    priority: int
```

### Strategy Pattern (Domain-Specific Parsing)

The Interaction Agent supports different parsing strategies based on domain mode:

| Strategy | Mode | Use Case |
|----------|------|----------|
| `CompositionalStrategy` | "general" | Segments = words (Marmoset, Songbirds) |
| `HolophrasticStrategy` | "bat" | Rigid idioms = atomic units (Egyptian Fruit Bat) |

```python
from realtime.interaction_agent import InteractionAgent, InteractionAgentConfig

# Configure for bat-specific parsing
config = InteractionAgentConfig(
    domain_mode="bat",  # Use holophrastic strategy
    feature_endpoint="ipc:///tmp/cognitive_features.ipc",
    action_endpoint="ipc:///tmp/cognitive_actions.ipc",
)

agent = InteractionAgent(config)
agent.start()
```

### Config Server (Python-Rust Data Sync)

The `ConfigServer` provides REQ/REP endpoints for Python to load acoustic profile data from Rust:

```rust
// Rust side
let server = ConfigServer::with_default_endpoint()?;
let request = server.recv_request()?;
let response = ConfigResponse {
    request_id: request.request_id,
    success: true,
    data: Some(acoustic_profile_json),
    error: None,
};
server.send_response(&response)?;
```

```python
# Python side
from realtime.config_client import ConfigClient

client = ConfigClient()
profile = client.request_acoustic_profile("bat")
if profile:
    print(f"Loaded {len(profile.valid_bigrams)} bigrams from Rust")
```

### Documentation

See [`docs/closed_loop_agent_protocol.md`](docs/closed_loop_agent_protocol.md) for comprehensive documentation including:
- Communication protocol specifications
- Context inference engine
- Response generation pipeline
- Deployment configuration
- Strategy Pattern usage (Section 15)

---

### Overview

Granular Concatenative Synthesis enables high-fidelity animal vocalization synthesis by manipulating real audio samples through grain windows. This approach preserves formant structure (vocal tract resonances) while enabling systematic pitch and time manipulation.

### Scientific Validation

**t-SNE Distance Results:**
- Natural ↔ Granular: **6.452** ✅ (target < 7.0)
- Natural ↔ Concatenative: 4.208 (baseline)
- Natural ↔ Additive: 27.052 (failed)

**Key Finding**: 76.1% improvement over additive synthesis proves that bio-acoustic complexity lies in the spectral envelope (formants), which mathematical sine waves cannot replicate.

### Python API

```python
from technical_architecture import GranularConcatenativeSynthesizer

# Create synthesizer
synth = GranularConcatenativeSynthesizer(sample_rate=22050)

# Load source audio (real animal vocalization)
synth.load_source(audio_buffer)

# Configure parameters
synth.set_pitch_shift(0.9)      # Lower pitch (0.9 = 10% lower)
synth.set_grain_size_ms(20.0)   # Grain size in milliseconds

# Synthesize output
output_audio = synth.synthesize(duration_ms=100.0)
```

### When to Use Granular Synthesis

**Use Granular Synthesis when:**
1. **Pitch Continuum Testing**: Need pitches not in database (7500Hz, 7600Hz, 7700Hz...)
2. **Controlling Confounds**: Same phrase, different pitches, constant duration
3. **Acoustic Feature Boundaries**: Just-noticeable-difference (JND) measurements
4. **Novel Stimuli**: Creating hybrid calls that don't exist naturally

**Use Concatenative Synthesis when:**
- You have exact audio segments you need (perfect fidelity, low flexibility)
- No parameter variation required
- Playing back natural recordings

### Rust Implementation

The granular synthesis engine is implemented in `src/synthesis.rs`:

```rust
pub struct GrainWindow {
    samples: Vec<f32>,
}

pub struct GranularVoice {
    source_buffer: Vec<f32>,
    sample_rate: usize,
    grain_size_ms: f32,
    window: Vec<f32>,
    position: f32,
    pitch_shift_ratio: f32,
}

pub struct GranularConcatenativeSynthesizer {
    sample_rate: usize,
    voices: Vec<GranularVoice>,
}
```

**Algorithm:**
1. Read sample from source buffer at current position (linear interpolation)
2. Apply grain window envelope based on position within grain
3. Advance position based on pitch shift ratio
4. Wrap around source buffer (looping)

### Test-Driven Development

All components implemented using TDD methodology:

```bash
# Run granular synthesis tests
cargo test --lib -- test_granular test_grain_window -- --nocapture
```

**5 tests, all passing:**
- `test_grain_window_hanning` ✅
- `test_granular_voice_pitch_shift` ✅
- `test_granular_voice_time_stretch` ✅
- `test_granular_morpher_overlap` ✅
- `test_granular_concatenative_synthesizer` ✅

### Validation

Run the granular synthesis validation script:

```bash
python3 ../realtime/granular_synthesis_validation.py
```

This validates that granular synthesis achieves t-SNE distance < 7.0, confirming bio-acoustic validity.

### Bio-Acoustic Turing Test

With granular synthesis validated, the next step is live animal testing using the Bio-Acoustic Turing Test framework:

```bash
python3 ../realtime/demo_bio_acoustic_turing_test.py
```

This determines if live animals can distinguish between natural and granular-synthesized vocalizations.

### Documentation

- **Scientific Findings**: `../realtime/GRANULAR_SYNTHESIS_FINDINGS.md`
- **Validation Script**: `../realtime/granular_synthesis_validation.py`
- **Turing Test Framework**: `../realtime/bio_acoustic_turing_test.py`

---

## Granular Synthesis Limitations: The Formant Barrier

### Overview

**Granular Concatenative Synthesis is a "Warping" technology, not a "Creation" technology.**

It is mathematically bound to the **Spectral Envelope** (DNA) of the source audio you load. You can stretch, squash, and shift the pitch of a "Harmonic" sound, but it will never lose its "Harmonic" nature to become a "Transient."

### The "Formant" Barrier (Why Transmutation Fails)

Granular synthesis works by chopping audio into tiny windows ("grains") and playing them back. It preserves the **Resonance** (Formants) of the original vocal tract.

**The Constraint:**
- **Source:** "Harmonic Phee" (Sine-like, smooth)
- **Operation:** Granular Time Stretch / Pitch Shift
- **Result:** A **"Different" Harmonic Phee** (Deeper/Longer)

**What happens to the "Transient" traits?**
- **Spectral Flatness:** Remains low (Tone-like)
- **Harmonic-to-Noise Ratio (HNR):** Remains high (Pure)
- **Attack Slope:** Remains smooth (Gradual)

**Verdict:**
You **cannot** synthesize a "Click" (Transient) from a "Whistle" (Harmonic) using Granular Synthesis. The smoothness is "baked in" to the source audio buffer.

### What Works vs. What Doesn't

| Goal | Possible? | Technique |
| :--- | :--- | :--- |
| **Harmonic → Deeper Harmonic** | ✅ **Yes** | Granular Pitch Shift |
| **Harmonic → Longer Harmonic** | ✅ **Yes** | Granular Time Stretch |
| **FM Sweep → Harmonic** | ✅ **Yes** (Limited) | Pitch Freezing / Large Grains |
| **Harmonic → Transient Click** | ❌ **No** | Impossible (DNA Mismatch) |
| **Transient → Harmonic** | ❌ **No** | Impossible (DNA Mismatch) |
| **Harmonic → Rhythmic Pulse** | ⚠️ **Artificial** | Granular Cloud (Sounds synthetic) |

### The Exception: FM Sweep → Harmonic ("Freezing" Effect)

There is one case where you **can** alter modality: **Reducing Frequency Modulation.**

If you synthesize an FM Sweep using **Large Grains** or apply **Synchronous Pitch Shifting**, you can "freeze" the frequency movement, effectively turning an FM Sweep into a pseudo-Harmonic tone.

**How it works:**
1. **Source:** Bat "FM Sweep" (7kHz → 9kHz)
2. **Granular Config:** Large grain size (covers the whole sweep) OR aggressive pitch averaging
3. **Synthesis:** The engine smooths out the frequency curve
4. **Result:** A "Choral" or "Warbled" tone that is closer to Harmonic than FM

**Use Case:**
Generating a "Calming" variant of a "Highly Modulated" call.

### The "Rhythmic" Illusion (Granular Artifacts)

You *can* make a Harmonic sound "Rhythmic" by reducing the grain size and randomizing the trigger (Granular Cloud).

**The Result:**
A "Sparkly" or "Bubbly" texture
- **Is it Rhythmic?** Yes, it has a temporal beat
- **Is it Biological?** Usually not. It sounds artificial/synthetic

**Verdict:**
Useful for generating **"Machine"** or **"Anomalous"** stimuli (testing Semiotic Emergence), but not for mimicking natural biological calls.

### The Solution: "Persona Switching" (Source Selection)

Since Granular Synthesis is **Source-Dependent**, you handle Modality via your **Persona Router**, not the Synthesis Engine itself.

**The Logic:**
Different Modalities = Different Source Buffers.

| Target Modality | Granular Config | Source Buffer (Persona) | Result |
| :--- | :--- | :--- | :--- |
| **Harmonic** | 20ms Grain, Pitch Shift | `marmoset_phee.wav` | **Warped Phee** (Preserves Tone) |
| **FM Sweep** | 10ms Grain, Preserve Time | `bat_fm_sweep.wav` | **Warped Sweep** (Preserves Modulation) |
| **Transient** | 2ms Grain (Chopper) | `bat_click.wav` | **Warped Click** (Preserves Attack) |

**Implementation:**
Your `ContextualAgent` handles the "Switch."
```python
def select_modality_source(self, intent_modality):
    if intent_modality == "HARMONIC":
        synth.load_source(self.personas.get("MARMOSET_PHEE"))
    elif intent_modality == "TRANSIENT":
        synth.load_source(self.personas.get("BAT_CLICK"))
    # Granular Engine just applies the "Warping" (Pitch/Time) to the selected buffer
```

### Acoustic Algebra Limitation (High-Dim Vectors)

This impacts your **Acoustic Algebra**.

If you calculate a **Target Vector** that is "50% Harmonic and 50% Transient," your **`find_nearest_real_phrase()`** function will always pick the **closest Neighbor**.

**Scenario:** Target is 50% Transient
- **Neighbor A:** 0% Transient (Distance: 0.4)
- **Neighbor B:** 100% Transient (Distance: 0.6)
- **Engine Choice:** It picks **Neighbor A**
- **Result:** You synthesize a **"Gritty Harmonic"** sound (using Transient source is too risky/distant)

**Conclusion:**
Acoustic Algebra can **nuance** a modality (make a Harmonic sound slightly harsher), but it cannot **cross the bridge** (Harmonic → Transient) because the **Spectral Envelope** prevents it.

### TDD Validation

All formant barrier constraints are validated by TDD tests in `tests/test_granular_synthesis_limitations.py`:

```bash
python3 -m pytest tests/test_granular_synthesis_limitations.py -v

# Expected: 12/12 tests passing
# - test_harmonic_cannot_become_transient_via_granular_synthesis ✅
# - test_transient_cannot_become_harmonic_via_granular_synthesis ✅
# - test_harmonic_preserves_spectral_flatness_after_pitch_shift ✅
# - test_transient_preserves_spectral_flatness ✅
# - test_nearest_neighbor_preserves_modality ✅
# - test_persona_router_logic ✅
# - test_persona_switching_for_modality ✅
# - test_small_grains_create_temporal_artifacts ✅
# - ...and more
```

### Summary

**Key Principle:** Rely on your **Persona Layer** (Source Buffers) to handle Modality. Rely on **Granular Synthesis** to handle **Gradient Intensity** (Emotional nuance) within that Modality.

| Aspect | Responsibility |
| :--- | :--- |
| **Modality Selection** | Persona Router (Source Selection) |
| **Pitch Warping** | Granular Synthesis (Pitch Shift) |
| **Time Warping** | Granular Synthesis (Time Stretch) |
| **Emotional Nuance** | 17D Metadata Deltas (within modality) |
| **Cross-Modality Synthesis** | ❌ Impossible (Formant Barrier) |

---

## Architecture

The system follows a **Pipeline-First** architecture where the Rust Execution Layer handles the complete signal-to-semantic flow.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Systemd Supervisor                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  rust-field-engine (Rosetta Pipeline)                                │   │
│  │                                                                      │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────┐ │   │
│  │  │   Stage 1   │   │   Stage 2   │   │   Stage 3   │   │ Stage 4  │ │   │
│  │  │  Dynamic    │──▶│    45D      │──▶│  Cascaded   │──▶│ Semantic │ │   │
│  │  │ Segmentation│   │  Features   │   │ Classifier  │   │ Grounding│ │   │
│  │  └─────────────┘   └─────────────┘   └─────────────┘   └──────────┘ │   │
│  │         │                                     │                       │   │
│  │         ▼                                     ▼                       │   │
│  │  ┌─────────────────┐                ┌──────────────────────┐        │   │
│  │  │ Environmental   │                │    RosettaBundle     │        │   │
│  │  │ Monitor/Sentry  │───────────────▶│ (Dictionary + Weights)│       │   │
│  │  └─────────────────┘                └──────────────────────┘        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                   │                                          │
│                                   │ ZeroMQ: ContextEnrichedPhrase Event      │
│                                   ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  python-cognitive-agent                                              │   │
│  │  - Receives Semantic Labels ("Phee") + Intent ("Contact")            │   │
│  │  - Decision Making & Learning                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Role | Technology |
|-----------|------|------------|
| **Dynamic Segmenter** | Finds precise phrase boundaries | Change Point Detection |
| **45D Feature Extractor** | Comprehensive acoustic analysis | FFT + Spectral Analysis |
| **Cascaded Classifier** | Species ID → Phrase Type | Router → Analyzer pattern |
| **Semantic Grounding** | Maps acoustics to meaning | Human-Guided Dictionary |
| **RosettaBundle** | Deployable artifact | Dictionary + Weights |
| **Environmental Monitor** | Context enrichment | Rain, Wind, Temperature |

---

## Deployment

See `deployment/` directory for systemd service files:

```bash
# Copy systemd files
sudo cp deployment/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable rust-field-engine.service
sudo systemctl enable python-cognitive-agent.service

# Start services
sudo systemctl start rust-field-engine.service
sudo systemctl start python-cognitive-agent.service
```

---

## Performance

Key performance metrics from benchmarking:

- **Message Processing**: > 10M ops/sec
- **Heartbeat Latency**: < 1μs average
- **Mode Switching**: < 1μs (immediate flag update)
- **Timeout Detection**: < 1μs (near-instantaneous)
- **Concurrent Access**: Lock-free atomic operations

Run benchmarks:
```bash
cargo run --example benchmark_peer_controller --release
```

---

## Dependencies

See `Cargo.toml` for full dependencies. Key dependencies:

- `tokio` - Async runtime
- `tract-onnx` - ONNX ML inference
- `serde` - Serialization
- `zmq` - ZeroMQ for inter-process communication
- `chrono` - Time handling
- `ndarray` - Numerical computing

---

## Documentation

- `CLAUDE.md` - Comprehensive developer guide
- `TDD_PLAN_FIELD_FEATURES.md` - Field deployment implementation plan (COMPLETE)
- `TDD_PLAN_PRODUCTION_FEATURES.md` - Production features plan (COMPLETE)
- `deployment/README.md` - Deployment instructions

### New Documentation (2026)

- `docs/pam_pipeline_guide.md` - Passive Acoustic Monitoring pipeline documentation
- `docs/closed_loop_agent_protocol.md` - Closed-loop Interaction Agent methodology (v1.1.0)
- `docs/acoustic_specialist_rf_methodology.md` - Acoustic specialist training methodology
- `docs/detection_pipeline_methodology.md` - Detection pipeline methodology
- `docs/classification_tasks.md` - Species classification task definitions

---

## License

**CC BY-ND 4.0 International** - See main project license for details.

---

## Author

Sheel Morjaria (sheelmorjaria@gmail.com)

---

## Scientific Context

This framework transforms animal communication research by:
1. **Metadata-Driven Synthesis**: 17-dimensional acoustic feature control
2. **Vector Delta Commands**: Relative transformations for precise manipulation
3. **Bio-Acoustic Validation**: t-SNE distance < 7.0 confirms naturalness
4. **Acoustic Personas**: Scientific profiles for validation (PURE, GRITTY, RHYTHMIC, HARMONIC)
5. **Cross-Species Analysis**: Marmoset, Bat, Dolphin, Chimpanzee, Sperm Whale, Zebra Finch
6. **Deception Detection**: Identifying intentional communication
7. **Emergent Behavior Tracking**: Cultural change detection

The research impact focuses on understanding animal intelligence through vocalization patterns, enabled by high-fidelity synthesis that preserves bio-acoustic complexity.
