# PAM Pipeline Documentation

## Overview

The `pam_pipeline.rs` binary is a complete **Passive Acoustic Monitoring (PAM)** system that processes audio streams in real-time to detect and classify animal vocalizations. It follows a 4-phase architecture designed for field deployment in wildlife monitoring scenarios.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PAM Pipeline Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Phase 1    │───▶│   Phase 2    │───▶│   Phase 3    │───▶│  Phase 4  │ │
│  │  Ingestion   │    │   Routing    │    │  Filtering   │    │  Output   │ │
│  │  & Boundary  │    │  112D Features│   │  Threshold   │    │  & AL     │ │
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

---

## Table of Contents

1. [Usage](#usage)
2. [Phase 1: Real-Time Ingestion & Boundary Detection](#phase-1-real-time-ingestion--boundary-detection)
3. [Phase 2: Feature Extraction & Hierarchical Routing](#phase-2-feature-extraction--hierarchical-routing)
4. [Phase 3: Confidence Threshold Filtering](#phase-3-confidence-threshold-filtering)
5. [Phase 4: Active Learning & Output](#phase-4-active-learning--output)
6. [Dependencies](#dependencies)
7. [Configuration](#configuration)
8. [Output Formats](#output-formats)
9. [Example Workflows](#example-workflows)

---

## Usage

### Command Line Interface

```bash
# Real-time mode (read from stdin)
cargo run --bin pam_pipeline -- --real-time

# Process audio file (raw f32 samples)
cargo run --bin pam_pipeline -- --input audio.raw

# Custom confidence threshold
cargo run --bin pam_pipeline -- --threshold 1.5 --input audio.raw

# Verbose output with text format
cargo run --bin pam_pipeline -- --verbose --format text --real-time

# Full configuration
cargo run --bin pam_pipeline -- \
  --input audio.raw \
  --threshold 1.5 \
  --sample-rate 44100 \
  --hop-size 512 \
  --min-phrase-duration 50.0 \
  --al-low 1.4 \
  --al-high 1.5 \
  --format jsonl \
  --verbose
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | None | Input audio file (raw f32 samples) |
| `--real-time` | false | Read from stdin in real-time |
| `--threshold` | 1.5 | Confidence threshold for detection |
| `--sample-rate` | 44100 | Sample rate in Hz |
| `--hop-size` | 512 | Hop size in samples (~11.6ms at 44.1kHz) |
| `--min-phrase-duration` | 50.0 | Minimum phrase duration in ms |
| `--al-low` | 1.4 | Active learning lower margin |
| `--al-high` | 1.5 | Active learning upper margin |
| `--format` | jsonl | Output format (jsonl, json, text) |
| `--verbose` | false | Enable debug logging to stderr |

---

## Phase 1: Real-Time Ingestion & Boundary Detection

### Purpose
Ingest audio samples with real system timestamps and segment into phrase units based on spectral boundaries.

### Components

#### `StreamingBuffer`
Location: `src/streaming.rs`

A ring buffer that maintains a sliding window of audio samples with real-time timestamp tracking.

```rust
pub struct StreamingBuffer {
    config: StreamingConfig,
    buffer: VecDeque<f32>,      // Ring buffer
    total_samples: usize,        // Total samples ingested
    sample_rate: u32,
}

pub struct RealTimeTimestamp {
    pub system_time: SystemTime,  // Actual wall-clock time
    pub sample_offset: usize,      // Sample position
    pub duration_ms: f32,          // Duration in milliseconds
}
```

**Key Features:**
- Uses `SystemTime::now()` for real timestamps (not synthetic file offsets)
- Automatic ring buffer management (configurable duration)
- Sample range extraction for analysis

#### `NeuralBoundaryDetector`
Location: `src/neural_boundary.rs`

Detects phrase boundaries using spectral change profiles rather than simple energy thresholds.

```rust
pub struct NeuralBoundaryDetector {
    config: BoundaryDetectorConfig,
    temporal_weights: Array1<f32>,      // Learned weights
    energy_weight: f32,
    spectral_change_weight: f32,
    last_boundary_sample: usize,
}
```

**Key Features:**
- Detects low-energy FM sweeps (critical for bat ultrasonic detection)
- Spectral centroid tracking for timbral shift detection
- Debounce enforcement for minimum phrase duration
- Three boundary types: Hard, Soft, Transitional

### Data Flow

```
Audio Samples → StreamingBuffer.add_samples()
                        │
                        ▼
              RealTimeTimestamp (system clock)
                        │
                        ▼
              NeuralBoundaryDetector.detect_boundaries()
                        │
                        ▼
              Vec<PhraseBoundary> (time_ms, confidence, type)
```

---

## Phase 2: Feature Extraction & Hierarchical Routing

### Purpose
Extract 112-dimensional features and route to the appropriate acoustic specialist model.

### Components

#### `PAMRouter`
Location: `src/pam_router.rs`

Stateless hierarchical classifier that routes segments to specialist models.

```rust
pub struct PAMRouter {
    config: PAMRouterConfig,
    specialists: HashMap<AcousticGroup, RandomForestClassifier>,
}

pub struct PAMResult {
    pub species: String,
    pub confidence: f32,
    pub acoustic_group: AcousticGroup,
    pub features_112d: Vec<f32>,
    pub taxon: ConsolidatedTaxon,
    pub inference_time_us: u64,
    pub active_learning: bool,
}
```

#### `AcousticGroup` (13 Specialist Groups)
Location: `src/pam_router.rs`

```rust
pub enum AcousticGroup {
    // Mammals (3 groups)
    UltrasonicMammal,    // Bats: 20-100kHz FM sweeps
    SonicLongMammal,     // Baleen whales: 20-5000Hz, long duration
    SonicShortMammal,    // Primates: mid F0, variable

    // Birds (3 groups)
    BirdHighFreq,        // Songbirds: high F0, fast modulation
    BirdLowFreq,         // Doves, owls: low F0, long duration
    BirdMechanical,      // Hummingbirds: broadband, pulse-like

    // Marine Mammals (3 groups)
    MarineWhistle,       // Dolphins: FM sweeps, harmonic
    MarineClick,         // Porpoises, sperm whales: impulsive
    MarineMoan,          // Baleen whales: low F0, long duration

    // Insects (2 groups)
    InsectWingbeat,      // Mosquitoes, flies: steady F0, pure tones
    InsectStridulation,  // Crickets, cicadas: broadband, impulsive

    // Other
    Amphibian,           // Frogs, toads: pulse trains, trills
    Unknown,
}
```

#### Species-to-Acoustic-Group Mapping
Location: `src/pam_router.rs::map_species_to_acoustic()`

```rust
pub fn map_species_to_acoustic(species: &str) -> AcousticGroup {
    // Uses acoustic properties rather than taxonomy
    // Example mappings:
    // "Egyptian Fruit Bat" → UltrasonicMammal
    // "Tursiops truncatus" → MarineWhistle
    // "Humpback Whale" → MarineMoan
    // "Zebra Finch" → BirdHighFreq
    // "Common Marmoset" → SonicShortMammal
}
```

#### 112D Feature Stack
Location: `src/taxonomic_router.rs`

```
┌─────────────────────────────────────────────────────────────┐
│                    112D Feature Vector                       │
├─────────────────────────────────────────────────────────────┤
│  Physics Features (46D)                                     │
│  ├── F0 statistics (mean, std, min, max, range)            │
│  ├── Duration features (ms, frame count)                    │
│  ├── Energy features (RMS, peak, dynamic range)             │
│  ├── Spectral features (centroid, bandwidth, flatness)      │
│  └── Temporal features (attack, decay, sustain)             │
├─────────────────────────────────────────────────────────────┤
│  Macro Texture Features (30D)                               │
│  ├── FM sweep characteristics                               │
│  ├── Harmonic structure                                     │
│  ├── Pulse train patterns                                   │
│  └── Amplitude modulation                                   │
├─────────────────────────────────────────────────────────────┤
│  Micro Texture Features (36D)                               │
│  ├── Fine spectral details                                  │
│  ├── Jitter and shimmer                                     │
│  └── Micro-temporal patterns                                │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Phrase Samples → extract_features_placeholder()
                        │
                        ▼
              Vec<f32> (112D feature vector placeholder)
                        │
                        ▼
              infer_acoustic_group_from_features()
                        │
                        ▼
              AcousticGroup (routing decision)
                        │
                        ▼
              PAMRouter.classify(features, group)
                        │
                        ▼
              Option<PAMResult> (species, confidence)
```

**Note:** The current implementation uses placeholder feature extraction for demonstration. In production, this would be replaced with the full 112D feature extraction from `taxonomic_router.rs`.

---

## Phase 3: Confidence Threshold Filtering

### Purpose
Reject weak or uncertain detections below the confidence threshold.

### Implementation

```rust
impl PAMRouter {
    /// Classify with confidence threshold filtering
    pub fn classify(&self, features_112d: &[f32], group: AcousticGroup) -> Result<Option<PAMResult>> {
        let result = self.extract_and_route(features_112d, group)?;

        // Apply confidence threshold
        match result {
            Some(ref r) if r.confidence >= self.config.confidence_threshold => Ok(result),
            _ => Ok(None),  // Below threshold - rejected
        }
    }
}
```

### Threshold Configuration

| Acoustic Group | Default Threshold | Rationale |
|----------------|-------------------|-----------|
| UltrasonicMammal | 1.4 | Lower for distant bat calls |
| BirdMechanical | 1.4 | Mechanical sounds are distinctive |
| MarineClick | 1.4 | Clicks have clear signatures |
| InsectWingbeat | 1.3 | Steady tones are easy to detect |
| Standard Groups | 1.5 | Default conservative threshold |
| Unknown | 2.0 | Higher for unclassified sounds |

### Class Weight Balancing

Uses `ClassWeightMode::Balanced` from `src/classical_ml.rs` to handle class imbalance:

```rust
pub enum ClassWeightMode {
    None,                           // No weighting
    Balanced,                       // n_samples / (n_classes * n_samples_for_class)
    Custom(HashMap<usize, f32>),   // Manual weights
}
```

This ensures rare species are not ignored in favor of common ones.

---

## Phase 4: Active Learning & Output

### Purpose
Flag uncertain detections for expert labeling and generate JSON output.

### Components

#### `ActiveLearningConfig`
Location: `src/active_learning.rs`

```rust
pub struct ActiveLearningConfig {
    pub margin_low: f32,              // 1.4 - lower bound of uncertainty
    pub margin_high: f32,             // 1.5 - upper bound of uncertainty
    pub save_uncertain_samples: bool, // Whether to save to disk
    pub uncertain_samples_dir: PathBuf,
}

pub fn flag_for_active_learning(confidence: f32, config: &ActiveLearningConfig) -> bool {
    confidence >= config.margin_low && confidence < config.margin_high
}
```

#### `DetectionPayload`
Location: `src/active_learning.rs`

```rust
pub struct DetectionPayload {
    pub timestamp_ms: u64,              // Unix timestamp in milliseconds
    pub species: String,                 // Predicted species label
    pub confidence: f32,                 // Confidence score
    pub acoustic_group: String,          // Acoustic group used
    pub taxon: String,                   // Taxonomic group
    pub inference_time_us: u64,          // Inference time in microseconds
    pub active_learning: bool,           // Flagged for expert labeling?
    pub uncertain_sample_path: Option<String>, // Path to saved sample
}
```

### Active Learning Margin

```
Confidence Score
    │
2.0 ┼────────────────────────────────────────
    │
1.5 ┼────────────────────────────────────────
    │    ▲                                   │
    │    │ ACTIVE LEARNING ZONE              │  ← Flag for expert labeling
    │    │ (1.4 - 1.5)                       │
    │    ▼                                   │
1.4 ┼────────────────────────────────────────
    │
1.0 ┼────────────────────────────────────────
    │                                        │  ← Below threshold: rejected
0.0 ┼────────────────────────────────────────
```

### Sample Persistence

When a detection falls in the active learning zone, the audio sample is saved for expert review:

```rust
pub fn save_uncertain_sample(
    audio: &[f32],
    species: &str,
    timestamp_ms: u64,
    config: &ActiveLearningConfig,
) -> Result<PathBuf> {
    // Saves to: uncertain_samples/{species}_{timestamp}.bin
    // Format: raw f32 samples (little-endian)
}
```

---

## Dependencies

### Module Dependency Graph

```
pam_pipeline.rs
├── src/streaming.rs
│   ├── StreamingBuffer
│   ├── RealTimeTimestamp
│   ├── DebounceTimer
│   └── SpectralChangeProfile
│
├── src/neural_boundary.rs
│   ├── NeuralBoundaryDetector
│   ├── BoundaryDetectorConfig
│   ├── PhraseBoundary
│   ├── BoundaryType (Hard, Soft, Transitional)
│   └── segment_into_phrases()
│
├── src/pam_router.rs
│   ├── PAMRouter
│   ├── PAMRouterConfig
│   ├── PAMResult
│   ├── AcousticGroup (13 groups)
│   └── map_species_to_acoustic()
│
├── src/active_learning.rs
│   ├── ActiveLearningConfig
│   ├── DetectionPayload
│   ├── flag_for_active_learning()
│   ├── generate_sample_path()
│   └── save_uncertain_sample()
│
└── src/taxonomic_router.rs (indirect)
    ├── FEATURE_DIM = 112
    ├── ConsolidatedTaxon
    ├── map_species_to_taxon()
    └── consolidate_taxon()
```

### External Crate Dependencies

| Crate | Purpose |
|-------|---------|
| `anyhow` | Error handling with context |
| `clap` | Command-line argument parsing |
| `serde` | Serialization |
| `serde_json` | JSON output |
| `ndarray` | Feature vector operations |
| `chrono` | Timestamp handling |

### Internal Module Dependencies

```toml
[dependencies]
technical_architecture = { path = ".." }

# From technical_architecture:
# - streaming: RealTimeTimestamp, StreamingBuffer, DebounceTimer
# - neural_boundary: NeuralBoundaryDetector, BoundaryDetectorConfig
# - pam_router: PAMRouter, AcousticGroup, PAMResult
# - active_learning: ActiveLearningConfig, DetectionPayload
# - taxonomic_router: ConsolidatedTaxon, FEATURE_DIM
# - classical_ml: RandomForestClassifier, ClassWeightMode
```

---

## Configuration

### Default Configuration

```rust
// Phase 1: Streaming
StreamingConfig {
    hop_size: 512,                    // ~11.6ms at 44.1kHz
    sample_rate: 44100,
    buffer_duration_secs: 60.0,       // 1-minute ring buffer
    min_phrase_duration_ms: 50.0,     // Minimum phrase length
}

BoundaryDetectorConfig {
    hop_size: 512,
    sample_rate: 44100,
    min_phrase_duration_ms: 50.0,
    threshold: 0.3,                   // Lower for better detection
    smoothing_frames: 3,
}

// Phase 2: Routing
PAMRouterConfig {
    confidence_threshold: 1.5,
    active_learning_low: 1.4,
    active_learning_high: 1.5,
    models_dir: PathBuf::from("specialist_rf_models"),
}

// Phase 4: Active Learning
ActiveLearningConfig {
    margin_low: 1.4,
    margin_high: 1.5,
    save_uncertain_samples: true,
    uncertain_samples_dir: PathBuf::from("uncertain_samples"),
}
```

---

## Output Formats

### JSONL (Default)

One JSON object per line (ideal for streaming):

```json
{"timestamp_ms":1678886400000,"species":"Tursiops truncatus","confidence":1.85,"acoustic_group":"Marine Whistle","taxon":"Cetacea","inference_time_us":450,"active_learning":false,"uncertain_sample_path":null}
{"timestamp_ms":1678886401234,"species":"Delphinus delphis","confidence":1.42,"acoustic_group":"Marine Whistle","taxon":"Cetacea","inference_time_us":380,"active_learning":true,"uncertain_sample_path":"uncertain_samples/Delphinus_delphis_1678886401234.bin"}
```

### JSON

Compact single-line JSON:

```json
{"timestamp_ms":1678886400000,"species":"Tursiops truncatus","confidence":1.85,"acoustic_group":"Marine Whistle","taxon":"Cetacea","inference_time_us":450,"active_learning":false}
```

### Text (Human-Readable)

```
[1678886400.000] Tursiops truncatus (Cetacea) - confidence: 1.85, group: Marine Whistle, inference: 450us
[1678886401.234] Delphinus delphis (Cetacea) - confidence: 1.42, group: Marine Whistle, inference: 380us [ACTIVE_LEARNING]
```

---

## Example Workflows

### Real-Time Field Deployment

```bash
# Start continuous monitoring with verbose logging
cargo run --release --bin pam_pipeline -- \
  --real-time \
  --threshold 1.5 \
  --sample-rate 44100 \
  --verbose \
  --format jsonl \
  | tee detections.log
```

### Batch Processing

```bash
# Process multiple audio files
for file in recordings/*.raw; do
  cargo run --release --bin pam_pipeline -- \
    --input "$file" \
    --format jsonl \
    >> all_detections.jsonl
done
```

### Integration with Expert Review

```bash
# Run pipeline and collect uncertain samples
cargo run --release --bin pam_pipeline -- \
  --real-time \
  --al-low 1.4 \
  --al-high 1.5

# Expert reviews uncertain_samples/ directory
# Labels are added to training set
# Models are retrained and deployed
```

### Latency Monitoring

```bash
# Monitor inference time with text output
cargo run --release --bin pam_pipeline -- \
  --real-time \
  --format text \
  --verbose 2>&1 \
  | grep "inference:"
```

---

## Performance Characteristics

| Metric | Target | Notes |
|--------|--------|-------|
| Inference Time | < 1000µs | Per segment |
| Memory | O(buffer_duration) | Ring buffer size |
| Latency | < 200ms | End-to-end detection |
| Segment Independence | Yes | No state leakage |

---

## File Structure

```
technical_architecture/
├── src/
│   ├── bin/
│   │   └── pam_pipeline.rs      # Main pipeline binary
│   ├── streaming.rs             # Phase 1: Ingestion
│   ├── neural_boundary.rs       # Phase 1: Boundary detection
│   ├── pam_router.rs            # Phase 2: Routing
│   ├── active_learning.rs       # Phase 4: Active learning
│   ├── taxonomic_router.rs      # 112D features
│   └── classical_ml.rs          # RF classifier
├── specialist_rf_models/        # Trained specialist models
│   ├── specialist_rf_ultrasonic_mammal.json
│   ├── specialist_rf_marine_whistle.json
│   └── ...
├── uncertain_samples/           # Flagged samples for review
│   ├── Tursiops_truncatus_1678886400000.bin
│   └── ...
└── docs/
    └── pam_pipeline_guide.md    # This document
```

---

## Test Coverage

```
Total PAM Tests: 48

Phase 1 (streaming.rs):      11 tests
Phase 1 (neural_boundary.rs): 13 tests
Phase 2 (pam_router.rs):     11 tests
Phase 3 (pam_router.rs):      3 tests
Phase 4 (active_learning.rs): 9 tests
Integration (pam_pipeline):   5 tests

Run: cargo test --lib 2>&1 | grep -c "ok$"
Result: 1629 tests pass
```
