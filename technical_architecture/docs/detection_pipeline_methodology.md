# Bioacoustic Detection Pipeline Methodology

## Overview

This document describes the methodology used in `detection_pipeline.rs` for real-time bioacoustic detection with timestamps. The pipeline implements a **hierarchical routing** architecture that combines taxonomic classification with confidence-based detection thresholds.

## Purpose

Unlike classification tasks (which identify *what* species is present), **detection tasks** answer:
- **When** did a vocalization occur? (timestamp)
- **What** species produced it? (identification)
- **How confident** is the prediction? (probability)

Detection pipelines are essential for:
- Real-time wildlife monitoring
- Acoustic survey analysis
- Biodiversity assessment
- Behavioral ecology studies

## Architecture

### Hierarchical Router

The pipeline uses a two-stage hierarchical architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Detection Payload                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  features_112d: Vec<f32>                                 │   │
│  │  start_time_ms: f64                                      │   │
│  │  end_time_ms: f64                                        │   │
│  │  source_file: String                                     │   │
│  │  true_label: String                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: Taxonomic Routing                   │
│                                                                 │
│   map_species_to_taxon(true_label) → Taxon                      │
│                                                                 │
│   Routes to appropriate specialist based on taxonomic group     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 2: Specialist Prediction               │
│                                                                 │
│   specialist.predict(features_112d) → species_idx               │
│   specialist.predict_proba(features_112d) → probabilities       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 3: Confidence Filtering                │
│                                                                 │
│   if confidence >= threshold:                                   │
│       → Accept detection                                        │
│   else:                                                         │
│       → Reject (low confidence)                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Detection Output                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  start_s: f64         // Start time in seconds           │   │
│  │  end_s: f64           // End time in seconds             │   │
│  │  duration_s: f64      // Segment duration                │   │
│  │  species: String      // Predicted species (canonical)   │   │
│  │  confidence: f32      // Prediction probability          │   │
│  │  taxon: String        // Taxonomic group                 │   │
│  │  source_file: String  // Original audio file             │   │
│  │  true_label: String   // Ground truth (canonical)        │   │
│  │  correct: bool        // Prediction matches ground truth │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Structures

### DetectionPayload

Input segment with pre-computed features:

```rust
pub struct DetectionPayload {
    pub features_112d: Vec<f32>,    // 112-dimensional feature vector
    pub start_time_ms: f64,          // Segment start time (milliseconds)
    pub end_time_ms: f64,            // Segment end time (milliseconds)
    pub source_file: String,         // Source audio filename
    pub segment_idx: usize,          // Segment index in source
    pub true_label: String,          // Ground truth species label
}
```

### Detection

Output detection with timestamp and classification:

```rust
pub struct Detection {
    pub start_s: f64,         // Start time in seconds
    pub end_s: f64,           // End time in seconds
    pub duration_s: f64,      // Duration in seconds
    pub species: String,      // Predicted species (canonical form)
    pub confidence: f32,      // Classification confidence [0, 1]
    pub taxon: String,        // Taxonomic group (e.g., "Songbird")
    pub source_file: String,  // Source audio file
    pub true_label: String,   // Ground truth (canonical form)
    pub correct: bool,        // true if species == true_label
}
```

### DetectionStats

Aggregate statistics for the pipeline run:

```rust
pub struct DetectionStats {
    pub total_segments: usize,              // Total segments processed
    pub positive_detections: usize,         // Segments passing threshold
    pub rejected_by_threshold: usize,       // Segments rejected (low confidence)
    pub correct_detections: usize,          // Correctly classified detections
    pub accuracy: f64,                      // Accuracy of positive detections
    pub avg_inference_time_us: f64,         // Average inference time (microseconds)
    pub taxon_distribution: HashMap<String, usize>,  // Counts per taxon
}
```

## Taxonomic Routing

### Taxon Groups

The pipeline routes to 8 taxonomic specialists:

| Taxon | Examples | Model File |
|-------|----------|------------|
| `Songbird` | Sparrows, finches, warblers | `specialist_rf_songbird.bincode` |
| `Cetacean` | Dolphins, porpoises, orcas | `specialist_rf_cetacean.bincode` |
| `Mysticete` | Humpback, blue, fin whales | `specialist_rf_mysticete.bincode` |
| `NonPasserine` | Parrots, owls, ducks | `specialist_rf_non_passerine.bincode` |
| `Amphibian` | Frogs, toads | `specialist_rf_amphibian.bincode` |
| `Pinniped` | Seals, sea lions | `specialist_rf_pinniped.bincode` |
| `Insect` | Crickets, mosquitoes, cicadas | `specialist_rf_insect.bincode` |
| `Mammal` | Bats, primates | `specialist_rf_mammal.bincode` |

### Species-to-Taxon Mapping

Species labels are mapped to taxonomic groups using keyword matching:

```rust
pub fn map_species_to_taxon(species: &str) -> Taxon {
    let s = species.to_lowercase();

    // Cetaceans (toothed whales)
    if s.contains("dolphin") || s.contains("porpoise")
        || s.contains("orca") || s.contains("delphinid")
    {
        return Taxon::Cetacean;
    }

    // Mysticetes (baleen whales)
    if s.contains("humpback") || s.contains("blue whale")
        || s.contains("fin whale") || s.contains("balaenopter")
    {
        return Taxon::Mysticete;
    }

    // ... additional mappings ...

    Taxon::Unknown
}
```

## Confidence Thresholding

### Purpose

Low-confidence predictions are rejected to improve precision. This is critical for:
- **Reducing false positives**: Avoid incorrect detections
- **Quality control**: Only report high-confidence identifications
- **Resource efficiency**: Focus on reliable detections

### Threshold Parameter

Default threshold: `1.5` (configurable via `--threshold`)

```rust
fn detect(&self, payload: &DetectionPayload) -> Option<(Detection, u64)> {
    // ... prediction logic ...

    let confidence = *proba.get(pred_idx).unwrap_or(&0.0);

    if confidence < self.threshold {
        return None;  // Reject low-confidence detection
    }

    // ... create detection ...
}
```

### Threshold Selection

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| 0.5 | Permissive, high recall | Exploratory analysis |
| 1.0 | Balanced | General purpose |
| 1.5 | Strict, high precision | Scientific surveys |
| 2.0 | Very strict | Critical applications |

## Label Canonicalization

### Purpose

Species labels may appear in multiple forms:
- Common names: "Bottlenose Dolphin"
- Scientific names: "Tursiops truncatus"
- Variants: "Minke whale" vs "Minke Whale"

Canonicalization maps all variants to a single standard form.

### Canonical Map

```rust
fn build_label_canonical_map() -> HashMap<String, String> {
    let mappings = [
        ("Dall's Porpoise", "Phocoenoides dalli"),
        ("Harbor Porpoise", "Phocoena phocoena"),
        ("Bottlenose Dolphin", "Tursiops truncatus"),
        ("Humpback Whale", "Megaptera novaeangliae"),
        // ...
    ];
    // ...
}
```

### Canonicalization Logic

```rust
fn normalize_label(label: &str, canonical_map: &HashMap<String, String>) -> String {
    canonical_map
        .get(&label.to_lowercase())
        .cloned()
        .unwrap_or_else(|| label.to_string())
}
```

## Inference Pipeline

### Detection Flow

```rust
impl HierarchicalRouter {
    fn detect(&self, payload: &DetectionPayload) -> Option<(Detection, u64)> {
        let start = Instant::now();

        // 1. Route to taxonomic specialist
        let taxon = map_species_to_taxon(&payload.true_label);
        let model = self.specialists.get(&taxon)?;

        // 2. Extract features
        let features = ndarray::Array1::from_vec(payload.features_112d.clone());

        // 3. Predict species
        let pred_idx = model.predict(&features);
        let proba = model.predict_proba(&features);
        let confidence = *proba.get(pred_idx).unwrap_or(&0.0);

        // 4. Apply confidence threshold
        if confidence < self.threshold {
            return None;
        }

        // 5. Canonicalize labels
        let species_canonical = normalize_label(&species, &self.canonical_map);
        let true_canonical = normalize_label(&payload.true_label, &self.canonical_map);

        // 6. Create detection with timing
        let inference_time_us = start.elapsed().as_micros() as u64;

        Some((Detection {
            start_s: payload.start_time_ms / 1000.0,
            end_s: payload.end_time_ms / 1000.0,
            species: species_canonical,
            confidence,
            correct: species_canonical == true_canonical,
            // ...
        }, inference_time_us))
    }
}
```

## Feature Processing

### 112-Dimensional Feature Vector

Each segment is represented by pre-computed features stored in `beans_feature_cache_112d/`:

| Feature Block | Dimensions | Description |
|--------------|------------|-------------|
| Physics | 46 | F0, bandwidth, duration, energy |
| Macro Texture | 30 | Spectral shape, harmonic structure |
| Micro Texture | 36 | Modulation, FM/AM patterns |

### Feature Loading

```rust
let cache_dir = Path::new("beans_feature_cache_112d");
let reader = BufReader::new(file);
let features: Vec<f32> = bincode::deserialize_from(reader)?;

if features.len() == FEATURE_DIM {  // 112
    segments.push(DetectionPayload {
        features_112d: features,
        // ...
    });
}
```

## Timestamp Generation

### Segment Timing

For evaluation on the BEANS dataset, timestamps are generated synthetically:

```rust
let sample_rate = 48000.0;           // 48 kHz
let segment_duration_ms = 1000.0;    // 1 second segments

let start_time_ms = idx as f64 * segment_duration_ms;
let end_time_ms = start_time_ms + segment_duration_ms;
```

### Real-World Deployment

In production, timestamps would come from:
- Audio file metadata
- Real-time audio stream timestamps
- Segmentation algorithm outputs

## Command-Line Interface

### Usage

```bash
cargo run --release --bin detection_pipeline -- [OPTIONS]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | `detections.json` | Output file path |
| `-t, --threshold` | `1.5` | Confidence threshold |
| `--max-segments` | `0` (unlimited) | Maximum segments to process |
| `-v, --verbose` | `false` | Enable verbose logging |

### Example

```bash
# Run with strict threshold on first 1000 segments
cargo run --release --bin detection_pipeline -- \
    --threshold 2.0 \
    --max-segments 1000 \
    --output survey_detections.json \
    --verbose
```

## Output Format

### JSON Structure

```json
{
  "detections": [
    {
      "start_s": 0.0,
      "end_s": 1.0,
      "duration_s": 1.0,
      "species": "Tursiops truncatus",
      "confidence": 0.85,
      "taxon": "Cetacean",
      "source_file": "beans_audio_full_rust/sample_000042.wav",
      "true_label": "Tursiops truncatus",
      "correct": true
    }
  ],
  "stats": {
    "total_segments": 50000,
    "positive_detections": 15000,
    "rejected_by_threshold": 35000,
    "correct_detections": 12000,
    "accuracy": 0.80,
    "avg_inference_time_us": 150.5,
    "taxon_distribution": {
      "Songbird": 8000,
      "Cetacean": 4000,
      "Amphibian": 2000,
      "Insect": 1000
    }
  }
}
```

## Performance Metrics

### Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Detection Rate** | `positive_detections / total_segments` | Proportion of segments with detections |
| **Precision** | `correct_detections / positive_detections` | Accuracy of positive detections |
| **Rejection Rate** | `rejected_by_threshold / total_segments` | Proportion filtered by threshold |
| **Inference Time** | `avg_inference_time_us` | Latency per detection |

### Sample Output

```
╔═══════════════════════════════════════════════════════════════╗
║  Results                                                      ║
╠═══════════════════════════════════════════════════════════════╣
║  Segments:        50000                                       ║
║  Detections:      15000  (30.0%)                              ║
║  Correct:         12000  (acc: 80.0%)                         ║
║  Rejected:        35000                                       ║
║  Avg time:        150.5µs                                     ║
╚═══════════════════════════════════════════════════════════════╝
```

## Design Decisions

### Why Hierarchical Routing?

1. **Specialization**: Each taxonomic group has unique acoustic signatures
2. **Efficiency**: Only load relevant models for target taxa
3. **Accuracy**: Specialists outperform generalist classifiers
4. **Scalability**: Easy to add new taxonomic groups

### Why Confidence Thresholding?

1. **False positive reduction**: Avoid reporting uncertain detections
2. **Application-specific tuning**: Adjust threshold for precision/recall trade-off
3. **Quality assurance**: Only high-confidence detections are reported
4. **Resource optimization**: Focus attention on reliable identifications

### Why Label Canonicalization?

1. **Consistency**: Handle naming variations in datasets
2. **Evaluation accuracy**: Correctly match predictions to ground truth
3. **Integration**: Standardize labels across multiple data sources
4. **Reporting**: Clean, consistent species names in output

## Limitations

1. **Routing assumes known taxon**: Uses ground truth label for routing (for evaluation)
2. **Synthetic timestamps**: Real deployment needs actual audio timestamps
3. **No temporal smoothing**: Each segment classified independently
4. **Single model format**: Currently only supports JSON models (bincode pending)

## Future Improvements

1. **Learned routing**: Replace keyword-based routing with classifier
2. **Temporal modeling**: Add HMM/CRF for sequence modeling
3. **Multi-label detection**: Support multiple species per segment
4. **Online learning**: Update models with new detections
5. **Audio streaming**: Direct integration with real-time audio feeds
6. **Bincode support**: Load models from bincode format for efficiency

## References

- BEANS benchmark: https://github.com/earthspecies/beans
- Taxonomic routing: `src/taxonomic_router.rs`
- Random Forest implementation: `src/classical_ml.rs`
- Feature extraction: `src/train_curriculum_nn_112d.rs`
