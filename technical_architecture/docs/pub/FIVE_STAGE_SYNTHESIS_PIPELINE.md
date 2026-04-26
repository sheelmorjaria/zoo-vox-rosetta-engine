# 5-Stage Synthesis Pipeline

A comprehensive pipeline for converting raw animal vocalizations into synthetic audio using the112D feature extraction, neural boundary detection, and granular synthesis.

---

## Overview

The Zoo Vox Rosetta Engine implements a 5-stage synthesis pipeline that transforms continuous animal audio into discrete semantic units, clusters them into a vocabulary, and reconstructs novel vocalizations through granular synthesis.

```
Raw Audio → [NBD] → Segments → [112D] → Features → [Corpus] → Clusters → [Exemplars] → [Synthesis] → Audio Output
```

---

## Stage 1: Neural Boundary Detection (NBD)

**Module:** `technical_architecture/src/neural_boundary.rs`

### Purpose
Segment continuous audio streams into discrete phrase units by detecting semantic boundaries.

### Key Components

#### `NeuralBoundaryDetector`
```rust
pub struct NeuralBoundaryDetector {
    config: BoundaryDetectorConfig,
    sample_rate: u32,
    // Internal state for boundary detection
}

pub struct BoundaryDetectorConfig {
    pub window_size_ms: f32,        // Analysis window size
    pub hop_size_ms: f32,          // Hop between windows
    pub threshold: f32,             // Boundary detection threshold
    pub min_phrase_duration_ms: f32, // Minimum phrase length
    pub max_phrase_duration_ms: f32, // Maximum phrase length
}
```

#### `NbdPhraseBoundary`
```rust
pub struct NbdPhraseBoundary {
    pub time_ms: f64,           // Time of boundary in milliseconds
    pub confidence: f32,       // Detection confidence (0.0-1.0)
    pub boundary_type: NbdBoundaryType,
}

pub enum NbdBoundaryType {
    Hard,    // Clear silence/gap
    Soft,    // Graded transition
    Semantic, // Meaning-based boundary
}
```

### API

```rust
// Create detector with configuration
let detector = NeuralBoundaryDetector::new(hop_size, sample_rate);

// Detect boundaries in audio
let boundaries: Vec<NbdPhraseBoundary> = detector.detect_boundaries(&audio);

// Segment audio into phrases
let phrases: Vec<Vec<f32>> = segment_into_phrases(&audio, &boundaries, sample_rate);
```

### Output
- **Segments**: Isolated audio buffers containing individual phrase units
- **Boundaries**: List of `NbdPhraseBoundary` with timing and confidence

---

## Stage 2: 112D Feature Extraction

**Module:** `technical_architecture/src/micro_dynamics_extractor.rs`

### Purpose
Extract a comprehensive 112-dimensional feature vector from each audio segment, capturing both acoustic physics and prosodic texture.

### Feature Architecture

The 112D `RosettaFeatures` struct is organized into three layers:

#### Layer 1: Base Physics (46D)
Core acoustic measurements that define the signal's physical properties.

| Category | Features | Count |
|----------|----------|-------|
| **Fundamental** | mean_f0_hz, duration_ms, f0_range_hz | 3 |
| **Energy** | rms_energy, peak_amplitude, zero_crossing_rate | 3 |
| **Spectral Shape** | spectral_centroid, spectral_bandwidth, spectral_rolloff, spectral_flatness, spectral_contrast | 5 |
| **Harmonicity** | harmonic_to_noise_ratio, harmonicity, spectral_flux | 3 |
| **Temporal** | attack_time_ms, decay_time_ms, sustain_level, release_time_ms | 4 |
| **Modulation** | vibrato_rate_hz, vibrato_depth, tremolo_rate_hz, tremolo_depth | 4 |
| **Stability** | jitter, shimmer, f0_drift_hz | 3 |
| **Spectral Dynamics** | spectral_flux_mean, spectral_flux_variance, spectral_flux_skew | 3 |
| **Band Energy** | low_band_energy, mid_band_energy, high_band_energy, band_energy_ratio | 4 |
| **Zero Crossing** | zcr_mean, zcr_variance, zcr_skewness | 3 |
| **Amplitude Stats** | amplitude_mean, amplitude_variance, amplitude_skewness | 3 |
| **Duration** | duration_frames, duration_normalized | 2 |
| **Pitch Stats** | f0_min_hz, f0_max_hz, f0_std_hz | 3 |

**Layer 1 Subtotal: 46D**

#### Layer 2: Macro Texture (30D)
Prosodic and rhythmic characteristics that describe the overall "shape" of the vocalization.

| Category | Features | Count |
|----------|----------|-------|
| **MFCCs** | mfcc_1 through mfcc_13 | 13 |
| **Delta MFCCs** | delta_mfcc_1 through delta_mfcc_6 | 6 |
| **Rhythm** | median_ici_ms, onset_rate_hz, ici_coefficient_of_variation | 3 |
| **Contour** | f0_contour_slope, f0_contour_curvature, inflection_count | 3 |
| **Energy Envelope** | envelope_skew, envelope_kurtosis, envelope_slope | 3 |
| **FM Characteristics** | fm_sweep_rate, fm_depth, fm_type | 2 |

**Layer 2 Subtotal: 30D**

#### Layer 3: Micro Texture (36D)
Fine-grained textural features that capture the unique "fingerprint" of each vocalization.

| Category | Features | Count |
|----------|----------|-------|
| **GLCM Texture** | glcm_contrast, glcm_dissimilarity, glcm_homogeneity, glcm_energy, glcm_correlation, glcm_entropy, gray_level_nonuniformity, gray_level_uniformity | 8 |
| **Spectral Texture** | spectral_skewness, spectral_kurtosis, spectral_variance, spectral_range, spectral_iqr | 5 |
| **Harmonic Texture** | harmonic_density, harmonic_spread, harmonic_regularity, tristimulus_inharmonic_ratio | 4 |
| **Temporal Texture** | temporal_skew, temporal_kurtosis, temporal_range, temporal_iqr | 4 |
| **SFM Features** | sfm_mean, sfm_variance, sfm_skewness, sfm_kurtosis | 4 |
| **Perceptual** | roughness, breathiness, brightness | 3 |
| **Quality** | voicing_degree, pitch_accuracy | 2 |
| **Micro Dynamics** | micro_flutter_depth, micro_flutter_rate, micro_tremolo_depth, micro_tremolo_rate, shimmer_2, shimmer_3, shimmer_5, jitter_ddp | 6 |

**Layer 3 Subtotal: 36D**

### Key Components

#### `MicroDynamicsExtractor`
```rust
pub struct MicroDynamicsExtractor {
    sample_rate: u32,
    fft_size: usize,
    hop_size: usize,
}

pub struct RosettaFeatures {
    // Layer 1: Base Physics (46D)
    pub mean_f0_hz: f32,
    pub duration_ms: f32,
    // ... 44 more fields ...

    // Layer 2: Macro Texture (30D)
    pub mfcc_1: f32,
    // ... 29 more fields ...

    // Layer 3: Micro Texture (36D)
    pub glcm_contrast: f32,
    // ... 35 more fields ...
}
```

### API

```rust
// Create extractor
let extractor = MicroDynamicsExtractor::new(sample_rate);

// Extract features from audio segment
let features: RosettaFeatures = extractor.extract_rosetta(&audio_segment)?;

// Convert to array for ML
let array_112d: [f32; 112] = features.to_array();
```

### Output
- **RosettaFeatures**: Complete 112D feature structure
- **Feature Array**: `[f32; 112]` for ML/clustering use

---

## Stage 3: Corpus Analysis

**Python Module:** `analysis/rosetta_stone/exemplar_manager.py`
**Rust Bridge:** `technical_architecture/src/manifest_bridge.rs`

### Purpose
Cluster extracted features into a vocabulary of k symbols (default: k=1020) and select the best exemplar audio for each cluster.

### Key Components

#### `ExemplarManager` (Python)
```python
class ExemplarManager:
    def __init__(self, vocabulary_size: int = 1020):
        self.vocabulary_size = vocabulary_size
        self.segments: List[SegmentInfo] = []
        self.clusters: Dict[int, ClusterInfo] = {}

    def load_manifest(self, manifest_path: str) -> int:
        """Load segments from Rust-generated JSON manifest."""

    def cluster_features(self, k: Optional[int] = None) -> None:
        """Cluster features using MiniBatchKMeans."""

    def select_exemplars(self) -> Dict[int, ClusterInfo]:
        """Select best exemplar (closest to centroid) for each cluster."""

    def save_exemplars(self, output_path: str) -> None:
        """Save cluster info to JSON for Rust synthesis."""
```

#### `SegmentInfo` and `ClusterInfo` (Python)
```python
@dataclass
class SegmentInfo:
    file_path: str
    features_112d: List[float]
    duration_ms: float
    mean_f0_hz: float
    cluster_id: Optional[int] = None

@dataclass
class ClusterInfo:
    cluster_id: int
    centroid_112d: List[float]
    exemplar_audio: str           # Path to best audio for this cluster
    exemplar_features_112d: List[float]
    num_segments: int
    mean_distance_to_centroid: float
```

### Clustering Algorithm

Uses **MiniBatchKMeans** for efficient clustering of large datasets:

```python
self.kmeans = MiniBatchKMeans(
    n_clusters=k,           # Default: 1020
    batch_size=1000,
    random_state=42,
    max_iter=300,
    n_init=10
)
```

### Exemplar Selection

For each cluster, the segment with minimum distance to the cluster centroid is selected as the exemplar:

```python
# Find segment closest to centroid
for segment in segments:
    distance = np.linalg.norm(segment.features_112d - centroid)
    if distance < best_distance:
        best_distance = distance
        best_segment = segment
```

### API (Command Line)

```bash
python -m analysis.rosetta_stone.exemplar_manager \
    --input segments_manifest.json \
    --output clusters.json \
    --synthesis-manifest synthesis_manifest.json \
    --k 1020
```

### Output Files

1. **clusters.json**: Full cluster information with centroids and exemplars
2. **synthesis_manifest.json**: Optimized manifest for Rust synthesis

---

## Stage 4: Semantic Reconstruction

**Module:** `technical_architecture/src/semantic_reconstruction.rs`

### Purpose
Manage exemplars (best audio per cluster) and synthesize audio from semantic timelines.

### Key Components

#### `SourceMetadata112D`
```rust
/// Wraps RosettaFeatures with synthesis metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceMetadata112D {
    pub features: RosettaFeatures,    // Full 112D features
    pub cluster_id: Option<u32>,       // Assigned cluster from Stage 3
}

impl SourceMetadata112D {
    pub fn from_features(features: &RosettaFeatures) -> Self;
    pub fn from_features_with_cluster(features: &RosettaFeatures, cluster_id: u32) -> Self;
    pub fn to_array_112d(&self) -> [f32; 112];

    /// Quality score for exemplar selection
    pub fn quality_score(&self) -> f32 {
        // Based on RMS energy, HNR, jitter, shimmer
        // Higher quality = better exemplar candidate
    }
}
```

#### `ExemplarManager` (Rust)
```rust
/// Stores best audio per cluster ID
pub struct ExemplarManager {
    exemplars: HashMap<u32, ExemplarEntry>,
}

pub struct ExemplarEntry {
    pub cluster_id: u32,
    pub audio: Vec<f32>,
    pub metadata: SourceMetadata112D,
}

impl ExemplarManager {
    pub fn new() -> Self;

    /// Register exemplar, keeps highest quality per cluster
    pub fn register_exemplar(&mut self, cluster_id: u32, audio: Vec<f32>, features: RosettaFeatures);

    pub fn get_exemplar(&self, cluster_id: u32) -> Option<&ExemplarEntry>;
    pub fn len(&self) -> usize;
    pub fn clear(&mut self);
}
```

#### `SynthesisTimeline`
```rust
pub struct SynthesisTimeline {
    events: Vec<SemanticTimelineEvent>,
}

pub struct SemanticTimelineEvent {
    pub cluster_id: u32,        // Which exemplar to use
    pub start_time_ms: f64,     // When to start playback
    pub duration_ms: f64,       // How long to play
    pub amplitude: f32,         // Volume (0.0-1.0)
}

impl SynthesisTimeline {
    pub fn new() -> Self;
    pub fn add_event(&mut self, event: SemanticTimelineEvent);
    pub fn total_duration_ms(&self) -> f64;
    pub fn get_events_in_range(&self, start_ms: f64, end_ms: f64) -> Vec<&SemanticTimelineEvent>;
}
```

#### `CachedGranularSynthesizer`
```rust
pub struct CachedGranularSynthesizer {
    config: SynthesisConfig112D,
    sources: HashMap<u32, SourceEntry>,
}

pub struct SynthesisConfig112D {
    pub sample_rate: u32,       // Default: 48000
    pub crossfade_ms: f32,      // Default: 10.0
    pub max_grains: usize,      // Default: 32
}

impl CachedGranularSynthesizer {
    pub fn new(config: SynthesisConfig112D) -> Self;

    /// Register audio source with 112D metadata
    pub fn register_source(&mut self, cluster_id: u32, audio: Vec<f32>, metadata: SourceMetadata112D);

    /// Synthesize audio from timeline
    pub async fn synthesize_timeline(&self, timeline: &SynthesisTimeline) -> anyhow::Result<Vec<f32>>;

    pub fn source_count(&self) -> usize;
    pub fn clear_sources(&mut self);
}
```

### API

```rust
// Create synthesizer
let config = SynthesisConfig112D::default();  // 48kHz, 10ms crossfade
let mut synth = CachedGranularSynthesizer::new(config);

// Register exemplars from Stage 3
for (cluster_id, exemplar) in clusters {
    let audio = load_audio(&exemplar.audio_path)?;
    let metadata = SourceMetadata112D::from_features_with_cluster(&exemplar.features, cluster_id);
    synth.register_source(cluster_id, audio, metadata);
}

// Create timeline from N-gram
let mut timeline = SynthesisTimeline::new();
timeline.add_event(SemanticTimelineEvent { cluster_id: 1, start_time_ms: 0.0, duration_ms: 100.0, amplitude: 1.0 });
timeline.add_event(SemanticTimelineEvent { cluster_id: 2, start_time_ms: 100.0, duration_ms: 100.0, amplitude: 1.0 });
timeline.add_event(SemanticTimelineEvent { cluster_id: 3, start_time_ms: 200.0, duration_ms: 100.0, amplitude: 1.0 });

// Synthesize
let output_audio = synth.synthesize_timeline(&timeline).await?;
```

---

## Stage 5: Synthesis Output

**Module:** `technical_architecture/src/synthesis.rs`

### Purpose
Output synthesized audio as playable WAV files.

### Output Format

- **Sample Rate**: 48000 Hz (configurable)
- **Channels**: Mono
- **Bit Depth**: 32-bit float

### File Output

```rust
// Save synthesized audio to WAV
fn save_wav(path: &Path, audio: &[f32], sample_rate: u32) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in audio {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}
```

---

## Pipeline Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE CONTROLLER                                │
│                    (technical_architecture/src/manifest_bridge.rs)           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: NBD SEGMENTATION                                            │   │
│  │ Raw Audio (Continuous) → Neural Boundary Detection → Isolated Phrases│   │
│  │ Module: neural_boundary.rs                                           │   │
│  └────────────────────────────┬────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: 112D FEATURE EXTRACTION                                     │   │
│  │ Audio Segments → RosettaFeatures (112D) → Feature Vectors           │   │
│  │ Module: micro_dynamics_extractor.rs                                  │   │
│  └────────────────────────────┬────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: CORPUS ANALYSIS (Python)                                    │   │
│  │ ┌───────────────────┐      ┌───────────────────────────────────────┐ │   │
│  │ │ MiniBatchKMeans   │      │ ExemplarManager                       │ │   │
│  │ │ (k=1020 clusters) │      │ (Best audio per cluster ID)           │ │   │
│  │ └───────────────────┘      └───────────────────────────────────────┘ │   │
│  │ Module: exemplar_manager.py                                          │   │
│  │ Output: clusters.json, synthesis_manifest.json                       │   │
│  └────────────────────────────┬────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: SEMANTIC RECONSTRUCTION                                     │   │
│  │ ┌───────────────────┐      ┌───────────────────────────────────────┐ │   │
│  │ │ ExemplarManager   │      │ CachedGranularSynthesizer             │ │   │
│  │ │ (Rust side)       │      │ (register_source + synthesize_timeline)│ │   │
│  │ └───────────────────┘      └───────────────────────────────────────┘ │   │
│  │ Module: semantic_reconstruction.rs                                   │   │
│  │ Input: synthesis_manifest.json, audio files                          │   │
│  └────────────────────────────┬────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STAGE 5: SYNTHESIS OUTPUT                                            │   │
│  │ N-gram Templates → Granular Synthesis → WAV Audio                    │   │
│  │ Module: synthesis.rs                                                 │   │
│  │ Output: synthetic_vocalization.wav                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Rust/Python Bridge (JSON Manifests)

### `SegmentsManifest` (Stage 1-2 Output)

**File:** `segments_manifest.json`

```json
{
  "version": "1.0",
  "sample_rate": 44100,
  "source_file": "raw_audio.wav",
  "segments": [
    {
      "file_path": "segments/seg_001.wav",
      "features_112d": [0.5, 0.3, ...],  // 112 values
      "duration_ms": 150.5,
      "mean_f0_hz": 8500.0,
      "start_time_ms": 0.0,
      "end_time_ms": 150.5
    },
    // ... more segments
  ]
}
```

### `ClustersManifest` (Stage 3 Output)

**File:** `clusters.json`

```json
{
  "vocabulary_size": 1020,
  "num_clusters": 1020,
  "clusters": {
    "0": {
      "cluster_id": 0,
      "centroid_112d": [0.5, 0.3, ...],  // 112 values
      "exemplar_audio": "segments/seg_042.wav",
      "exemplar_features_112d": [0.52, 0.31, ...],
      "num_segments": 15,
      "mean_distance_to_centroid": 0.85
    },
    // ... more clusters
  }
}
```

### `SynthesisManifest` (Stage 4 Input)

**File:** `synthesis_manifest.json`

```json
{
  "vocabulary_size": 1020,
  "exemplars": [
    {
      "cluster_id": 0,
      "audio_path": "segments/seg_042.wav",
      "metadata": {
        "mean_f0_hz": 8500.0,
        "duration_ms": 150.5,
        "f0_range_hz": 500.0,
        "rms_energy": 0.75,
        "harmonic_to_noise_ratio": 25.0,
        "attack_time_ms": 15.0,
        "decay_time_ms": 50.0
      }
    },
    // ... more exemplars
  ]
}
```

---

## Test Coverage

### Rust Tests
- **semantic_reconstruction_tests.rs**: 23 tests
- **synthesis_pipeline_integration_tests.rs**: 27 tests
- **manifest_bridge.rs**: 6 inline tests
- **Total**: 1564+ tests passing

### Python Tests
- **test_exemplar_manager.py**: 14 tests

---

## Usage Example

### Full Pipeline Execution

```rust
use technical_architecture::{
    PipelineController, NeuralBoundaryDetector, MicroDynamicsExtractor,
    CachedGranularSynthesizer, SynthesisConfig112D, SynthesisTimeline,
    SemanticTimelineEvent, SourceMetadata112D,
};

// 1. Load raw audio
let audio = load_audio("input.wav")?;
let sample_rate = 48000;

// 2. Stage 1: Segment
let detector = NeuralBoundaryDetector::new(512, sample_rate);
let boundaries = detector.detect_boundaries(&audio);
let phrases = segment_into_phrases(&audio, &boundaries, sample_rate);

// 3. Stage 2: Extract features
let extractor = MicroDynamicsExtractor::new(sample_rate);
let mut manifest = SegmentsManifest::new(sample_rate);

for (i, phrase) in phrases.iter().enumerate() {
    let features = extractor.extract_rosetta(phrase)?;
    let path = format!("segments/seg_{:03}.wav", i);
    save_wav(&path, phrase, sample_rate)?;
    manifest.add_segment(&path, &features, Some(start_ms), Some(end_ms));
}
manifest.save(Path::new("segments_manifest.json"))?;

// 4. Stage 3: Run Python clustering (external)
// python -m analysis.rosetta_stone.exemplar_manager -i segments_manifest.json -o clusters.json

// 5. Stage 4: Load clusters and synthesize
let mut controller = PipelineController::new(sample_rate);
controller.load_clusters_manifest(Path::new("clusters.json"))?;

let mut synth = CachedGranularSynthesizer::new(SynthesisConfig112D::default());

// Load exemplars into synthesizer
for exemplar in controller.synthesis_manifest().unwrap().exemplars.iter() {
    let audio = load_audio(&exemplar.audio_path)?;
    let metadata = SourceMetadata112D::from_features(&features);
    synth.register_source(exemplar.cluster_id, audio, metadata);
}

// Create timeline from N-gram
let mut timeline = SynthesisTimeline::new();
timeline.add_event(SemanticTimelineEvent { cluster_id: 1, start_time_ms: 0.0, duration_ms: 100.0, amplitude: 1.0 });
timeline.add_event(SemanticTimelineEvent { cluster_id: 2, start_time_ms: 100.0, duration_ms: 100.0, amplitude: 1.0 });

// 6. Stage 5: Synthesize output
let output = synth.synthesize_timeline(&timeline).await?;
save_wav("output.wav", &output, sample_rate)?;
```

---

## Performance Characteristics

| Stage | Language | Purpose | Typical Processing Time |
|-------|----------|---------|------------------------|
| 1. NBD | Rust | Audio segmentation | ~10ms per second of audio |
| 2. 112D | Rust | Feature extraction | ~50ms per segment |
| 3. Corpus | Python | Clustering (k=1020) | ~5s for 10K segments |
| 4. Reconstruction | Rust | Timeline synthesis | Real-time capable |
| 5. Output | Rust | WAV encoding | ~1ms per second of audio |

---

## Author

**Sheel Morjaria**
Email: sheelmorjaria@gmail.com
License: CC BY-ND 4.0 International
