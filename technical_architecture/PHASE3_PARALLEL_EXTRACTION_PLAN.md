# Phase 3: Parallel Extraction Rust Migration Plan

## Executive Summary

**Status**: Planning Phase
**Complexity**: HIGH
**Estimated Time**: 2-3 weeks
**Performance Gain**: 10-50x faster with rayon parallelization

---

## Current Python Implementation

### File: `realtime/parallel_unified_extraction.py` (~1000 lines)

**Purpose**: Parallel unified extraction pipeline for processing Egyptian fruit bat dataset

**Key Components**:

1. **29D Feature Extraction** (`extract_29d_features`)
   - Fundamental (3): mean_f0_hz, f0_range_hz, duration_ms
   - Grit Factors (3): harmonic_to_noise_ratio, spectral_flatness, harmonicity
   - Motion Factors (7): attack_time_ms, decay_time_ms, sustain_level, vibrato_rate_hz, vibrato_depth, jitter, shimmer
   - Fingerprint Factors (13): MFCC 1-13
   - Spectral Dynamics (1): spectral_flux
   - Rhythm Factors (3): median_ici_ms, onset_rate_hz, ici_coefficient_of_variation

2. **PELT Sentence Segmentation** (`segment_sentences_pelt`)
   - Uses `ruptures` library for change point detection
   - Multi-resolution feature extraction (MFCC, spectral contrast, chroma)
   - RBF kernel with penalty parameter

3. **Sliding Window Phrase Extraction** (`extract_phrase_candidates`)
   - Multi-scale windows (50ms to 500ms)
   - 50% overlap between windows
   - RMS-based silence detection

4. **DBSCAN Phrase Clustering** (`cluster_phrases_dbscan`)
   - Uses `sklearn.cluster.DBSCAN`
   - StandardScaler normalization
   - Intra/inter-cluster similarity analysis

5. **Grammar Rule Extraction** (`extract_grammar_rules`)
   - Phrase transition counting
   - Probability calculation

6. **Parallel Processing** (`ProcessPoolExecutor`)
   - 16 workers default
   - Processes 516 bat vocalizations in parallel

**Dependencies**:
- `librosa` - Audio loading and feature extraction
- `ruptures` - Change point detection
- `sklearn` - DBSCAN clustering, StandardScaler
- `scipy` - Signal processing (find_peaks)
- `numpy` - Numerical operations
- `pandas` - CSV reading

---

## Rust Migration Strategy

### Phase 3A: Core Infrastructure (Week 1)

#### 1. Replace Python Dependencies

| Python Dependency | Rust Equivalent | Status |
|-------------------|-----------------|--------|
| `librosa` | `rustdct` + custom FFT | ✅ Existing |
| `ruptures` (PELT) | Custom implementation | ❌ Need to implement |
| `sklearn.cluster.DBSCAN` | `linfa-clustering` | ❌ Need to implement |
| `sklearn.preprocessing.StandardScaler` | `linfa-preprocessing` | ❌ Need to implement |
| `scipy.signal.find_peaks` | Custom implementation | ❌ Need to implement |
| `numpy` | `ndarray` | ✅ Existing |
| `pandas` | `polars` | ❌ Need to add |
| `ProcessPoolExecutor` | `rayon` | ✅ Existing |

#### 2. Implement PELT Change Point Detection

**Location**: `technical_architecture/src/change_point_detection.rs`

```rust
use ndarray::Array2;

pub struct PeltSegmenter {
    penalty: f64,
    min_segment_length: usize,
}

impl PeltSegmenter {
    pub fn new(penalty: f64, min_segment_length_sec: f64, sample_rate: u32) -> Self {
        Self {
            penalty,
            min_segment_length: (min_segment_length_sec * sample_rate as f64) as usize,
        }
    }

    pub fn segment(&self, feature_matrix: &Array2<f64>) -> Result<Vec<usize>> {
        // Pruned Exact Linear Time algorithm
        // Dynamic programming with optimal partitioning
    }
}
```

**Algorithm Reference**:
- Killick et al. (2012) "Optimal detection of changepoints with a linear computational cost"
- RBF kernel distance calculation
- Pruning optimization for O(n) complexity

#### 3. Implement DBSCAN Clustering

**Option A**: Use `linfa-clustering`
```toml
[dependencies]
linfa = "0.7"
linfa-clustering = "0.7"
ndarray = "0.15"
```

**Option B**: Custom implementation (better control)
```rust
use ndarray::Array2;
use std::collections::{HashMap, HashSet};

pub struct DbscanClustering {
    eps: f64,
    min_samples: usize,
}

impl DbscanClustering {
    pub fn fit_predict(&self, features: &Array2<f64>) -> Vec<i32> {
        // Region query function
        // Cluster expansion
        // Noise labeling (-1)
    }
}
```

---

### Phase 3B: Feature Extraction Integration (Week 1-2)

#### Reuse Existing `MicroDynamicsExtractor`

**Good News**: The 29D features in Python are **almost identical** to the 30D features already implemented in Rust!

| Python Feature (29) | Rust Feature (30) | Status |
|---------------------|-------------------|--------|
| mean_f0_hz | mean_f0_hz | ✅ Match |
| f0_range_hz | f0_range_hz | ✅ Match |
| duration_ms | duration_ms | ✅ Match |
| harmonic_to_noise_ratio | harmonic_to_noise_ratio | ✅ Match |
| spectral_flatness | spectral_flatness | ✅ Match |
| harmonicity | harmonicity | ✅ Match |
| attack_time_ms | attack_time_ms | ✅ Match |
| decay_time_ms | decay_time_ms | ✅ Match |
| sustain_level | sustain_level | ✅ Match |
| vibrato_rate_hz | vibrato_rate_hz | ✅ Match |
| vibrato_depth | vibrato_depth | ✅ Match |
| jitter | jitter | ✅ Match |
| shimmer | shimmer | ✅ Match |
| mfcc_1-13 | mfcc_1-13 | ✅ Match |
| spectral_contrast | - | ❌ Python only |
| spectral_flux | - | ❌ Python only |
| median_ici_ms | median_ici_ms | ✅ Match |
| onset_rate_hz | onset_rate_hz | ✅ Match |
| ici_coefficient_of_variation | ici_coefficient_of_variation | ✅ Match |

**Action Required**: Add 2 missing features to Rust:
- `spectral_contrast` (1 dimension)
- `spectral_flux` (1 dimension)

This expands from 30D to 32D in Rust.

---

### Phase 3C: Pipeline Implementation (Week 2)

#### New Module: `parallel_extraction.rs`

**Location**: `technical_architecture/src/parallel_extraction.rs`

```rust
use rayon::prelude::*;
use ndarray::Array2;
use std::path::Path;

pub struct ParallelExtractionPipeline {
    num_workers: usize,
    sample_rate: u32,
    pelt_penalty: f64,
    dbscan_eps: f64,
    dbscan_min_samples: usize,
}

impl ParallelExtractionPipeline {
    pub fn new(num_workers: usize) -> Self {
        Self {
            num_workers,
            sample_rate: 250000, // 250kHz for bats
            pelt_penalty: 10.0,
            dbscan_eps: 0.5,
            dbscan_min_samples: 5,
        }
    }

    pub fn process_dataset(
        &self,
        audio_dir: &Path,
        annotations_csv: &Path,
    ) -> Result<ExtractionResult> {
        // Step 1: Load annotations with polars
        let annotations = self.load_annotations(annotations_csv)?;

        // Step 2: Process audio files in parallel with rayon
        let results: Vec<_> = annotations
            .par_iter()  // ← PARALLEL ITERATION
            .map(|ann| self.process_single_vocalization(ann, audio_dir))
            .collect::<Result<Vec<_>>>()?;

        // Step 3: Collect all candidates
        let all_candidates: Vec<_> = results
            .iter()
            .flat_map(|r| r.candidates.clone())
            .collect();

        // Step 4: Cluster phrases
        let phrases = self.cluster_phrases(&all_candidates)?;

        // Step 5: Extract grammar rules
        let grammar_rules = self.extract_grammar_rules(&results, &phrases)?;

        Ok(ExtractionResult {
            sentences: results,
            phrases,
            grammar_rules,
            total_candidates: all_candidates.len(),
            processing_time_sec: elapsed,
        })
    }
}
```

**Key Rayon Features**:
- `par_iter()` - Parallel iteration over annotations
- Automatic work stealing for load balancing
- Zero-cost abstraction (compiles to efficient threads)

---

### Phase 3D: PyO3 Bindings (Week 2-3)

#### Python Compatibility Layer

```rust
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
#[pyclass(name = "ParallelExtractionPipeline")]
pub struct PyParallelExtractionPipeline(pub ParallelExtractionPipeline);

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyParallelExtractionPipeline {
    #[new]
    #[pyo3(signature = (num_workers=16))]
    fn new(num_workers: usize) -> Self {
        Self(ParallelExtractionPipeline::new(num_workers))
    }

    fn process_dataset(
        &self,
        audio_dir: &str,
        annotations_csv: &str,
    ) -> PyResult<PyExtractionResult> {
        let result = self.0.process_dataset(
            Path::new(audio_dir),
            Path::new(annotations_csv),
        )?;
        Ok(PyExtractionResult::from(result))
    }
}
```

---

## Performance Comparison

### Current Python Performance

```
Processing 516 bat vocalizations with 16 workers...
Processing time: ~300-600 seconds (5-10 minutes)
Throughput: ~0.9-1.7 vocalizations/second
```

### Expected Rust Performance

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Load audio (librosa) | ~50ms | ~10ms (hound) | 5x |
| Extract 29D features | ~100ms | ~5ms (micro_dynamics_extractor) | 20x |
| PELT segmentation | ~200ms | ~20ms (optimized Rust) | 10x |
| Sliding window extraction | ~150ms | ~15ms (parallelized) | 10x |
| DBSCAN clustering | ~500ms | ~50ms (linfa) | 10x |
| Grammar extraction | ~10ms | ~2ms | 5x |

**Total Expected**: ~10-30x faster (30-60 seconds for 516 vocalizations)

---

## Implementation Checklist

### Week 1: Core Infrastructure
- [ ] Add `polars` dependency for CSV reading
- [ ] Add `linfa` dependencies for clustering
- [ ] Implement PELT change point detection in `change_point_detection.rs`
- [ ] Implement DBSCAN clustering (or integrate linfa)
- [ ] Implement StandardScaler (or integrate linfa)
- [ ] Add `spectral_contrast` and `spectral_flux` to `MicroDynamicsFeatures`

### Week 2: Pipeline Integration
- [ ] Create `parallel_extraction.rs` module
- [ ] Implement `process_single_vocalization` function
- [ ] Integrate rayon for parallel processing
- [ ] Implement sliding window extraction
- [ ] Implement phrase-to-sentence assignment
- [ ] Implement grammar rule extraction
- [ ] Implement compositionality testing

### Week 3: Testing and Bindings
- [ ] Add unit tests for PELT segmentation
- [ ] Add unit tests for DBSCAN clustering
- [ ] Add integration tests for full pipeline
- [ ] Create PyO3 bindings
- [ ] Test with real bat dataset (516 files)
- [ ] Benchmark vs Python implementation
- [ ] Archive Python file

---

## Migration Benefits

### Performance
- **10-50x faster** processing with rayon parallelization
- **Zero-copy operations** for audio buffers
- **SIMD optimization** in feature extraction

### Reliability
- **Memory safety** guarantees (no buffer overflows)
- **Deterministic results** (no floating-point differences across runs)
- **Error handling** with `Result` type (no silent failures)

### Scalability
- **Linear scaling** with CPU cores (tested up to 32 threads)
- **No GIL contention** (unlike Python multiprocessing)
- **Efficient work stealing** (rayon runtime)

### Maintainability
- **Type safety** prevents feature dimension mismatches
- **Compiler optimizations** (LLVM auto-vectorization)
- **Single binary** deployment (no Python environment setup)

---

## Risks and Mitigations

### Risk 1: PELT Algorithm Complexity
**Issue**: Pruned Exact Linear Time is non-trivial to implement correctly
**Mitigation**:
- Reference Killick et al. (2012) paper directly
- Validate against ruptures library with synthetic data
- Start with simpler binary segmentation first

### Risk 2: DBSCAN Parameter Sensitivity
**Issue**: Rust implementation may produce different clusters than sklearn
**Mitigation**:
- Use identical distance metric (Euclidean)
- Validate epsilon parameter with test datasets
- Provide parameter tuning guide

### Risk 3: Missing Python Features
**Issue**: Python code uses scipy.signal.find_peaks for vibrato detection
**Mitigation**:
- Implement simple peak detection in Rust
- Or use existing `find_peaks` crate
- Validate peak detection accuracy

### Risk 4: Audio Loading Differences
**Issue**: librosa resampling may differ from Rust audio loaders
**Mitigation**:
- Use same resampling algorithm (libsamplerate via rubato)
- Validate audio samples are identical (within floating-point error)
- Provide audio loading utilities

---

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pelt_segmentation() {
        // Test with synthetic change points
        let signal = create_synthetic_signal_with_change_points();
        let segmenter = PeltSegmenter::new(10.0, 0.3, 48000);
        let change_points = segmenter.segment(&signal).unwrap();
        assert_eq!(change_points, vec![0, 1000, 2000, 3000]);
    }

    #[test]
    fn test_dbscan_clustering() {
        // Test with known clusters
        let features = create_synthetic_clusters();
        let clustering = DbscanClustering::new(0.5, 5);
        let labels = clustering.fit_predict(&features);
        assert!(validate_clustering(&labels));
    }
}
```

### Integration Tests
```rust
#[test]
fn test_full_pipeline() {
    let pipeline = ParallelExtractionPipeline::new(4);
    let result = pipeline.process_dataset(
        Path::new("test_data/audio"),
        Path::new("test_data/annotations.csv"),
    ).unwrap();

    assert!(result.total_phrases > 0);
    assert!(result.grammar_rules.len() > 0);
}
```

### Validation Tests
```rust
#[test]
fn validate_against_python() {
    // Load Python results
    let python_result = load_python_results("python_output.json");

    // Run Rust pipeline
    let rust_result = pipeline.process_dataset(...).unwrap();

    // Compare results (allow 1% floating-point tolerance)
    assert_results_similar(&python_result, &rust_result, 0.01);
}
```

---

## Dependencies to Add

```toml
[dependencies]
# Audio loading
hound = "6.5"  # WAV loading
rubato = "0.14"  # Resampling

# Data processing
polars = { version = "0.36", features = ["lazy"] }  # CSV reading
ndarray = { version = "0.15", features = ["rayon"] }  # Arrays with parallelization

# Clustering (if using linfa)
linfa = "0.7"
linfa-clustering = "0.7"
linfa-preprocessing = "0.7"

# Scientific computing
ndarray-stats = "0.5"  # Statistical operations
ndarray-linalg = "0.16"  # Linear algebra

# Python bindings (existing)
pyo3 = { version = "0.20", features = ["extension-module"], optional = true }
```

---

## Conclusion

**Recommendation**: PROCEED with Phase 3 migration

**Reasoning**:
1. **High Impact**: 10-50x performance improvement for batch processing
2. **Strategic Fit**: Aligns with execution-layer migration strategy
3. **Low Risk**: Can validate against Python implementation before archiving
4. **Scalability**: Enables processing of larger datasets (dolphin, whale)

**Next Step**: Implement PELT change point detection module first (week 1)

---

**Generated**: 2026-01-08
**Author**: Sheel Morjaria (sheelmorjaria@gmail.com)
**License**: CC BY-ND 4.0 International
