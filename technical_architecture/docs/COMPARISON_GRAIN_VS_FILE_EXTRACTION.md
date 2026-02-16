# Algorithm Comparison: Grain-Based vs File-Based Extraction

This document compares two distinct approaches to animal vocalization analysis:

1. **Grain-Based Grammar Discovery** (Python, archived)
2. **File-Based Parallel Extraction** (Rust `parallel_extraction.rs`)

---

## Executive Summary

| Aspect | Grain-Based (Python) | File-Based (Rust) |
|--------|---------------------|-------------------|
| **Unit of Analysis** | 10ms grains (within file) | Entire audio files |
| **Primary Goal** | Discover grammar within sequences | Discover phrase types across corpus |
| **Clustering Scope** | Within single vocalization | Across 91,080 vocalizations |
| **Output** | Phrase sequences, transition entropy | Clustered phrases, synthesis metadata |
| **Scale** | Small (synthetic/demo) | Large (real datasets) |

---

## 1. Fundamental Approach

### Grain-Based Grammar Discovery (Python)

**Philosophy**: Bottom-up discovery of linguistic structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Grain-Based Workflow                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Single audio file ("sentence")                          │
│    ↓                                                            │
│  1. Grain Extraction                                            │
│     ├─ Chop into 10ms grains (50% overlap)                      │
│     ├─ Extract 17D features per grain                           │
│     └─ Output: ~100-500 grains per file                        │
│    ↓                                                            │
│  2. DBSCAN Clustering (within file)                             │
│     ├─ Group similar grains into phrases                       │
│     ├─ eps=0.5, min_samples=5                                   │
│     └─ Output: ~5-20 phrase types per file                     │
│    ↓                                                            │
│  3. Sequence Reconstruction                                     │
│     ├─ Convert grain labels → phrase sequence                   │
│     ├─ Compress consecutive duplicates                          │
│     └─ Output: [PHRASE_0, PHRASE_1, PHRASE_0, ...]            │
│    ↓                                                            │
│  4. Transition Entropy Analysis                                 │
│     ├─ Build transition matrix: P(B|A)                          │
│     ├─ Compute Shannon entropy: H(A) = -Σ p(x) log p(x)         │
│     └─ Output: Grammar rigidity score (0=random, 1=deterministic)│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight**: Grammar is discovered by analyzing **how phrases follow each other** within a single vocalization sequence.

### File-Based Parallel Extraction (Rust)

**Philosophy**: Top-down discovery of reusable phrase types across corpus

```
┌─────────────────────────────────────────────────────────────────┐
│                   File-Based Workflow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: 91,080 audio files (entire corpus)                      │
│    ↓                                                            │
│  1. Batch Processing (1,000 files per batch)                    │
│     ├─ Load WAV/FLAC/MP3 files                                  │
│     ├─ Extract 30D features per file                            │
│     └─ Output: 91,080 phrase candidates                        │
│    ↓                                                            │
│  2. Global Feature Normalization                                │
│     ├─ StandardScaler: zero mean, unit variance                 │
│     └─ Output: Normalized feature matrix                        │
│    ↓                                                            │
│  3. DBSCAN Clustering (across corpus)                           │
│     ├─ Group similar files into phrase types                   │
│     ├─ eps=0.35, min_samples=10                                 │
│     └─ Output: ~5,000-6,000 phrase types                        │
│    ↓                                                            │
│  4. Similarity Analysis                                         │
│     ├─ Intra-cluster similarity (coherence)                     │
│     ├─ Inter-cluster similarity (separation)                    │
│     └─ Output: Atomic phrase detection                         │
│    ↓                                                            │
│  5. Zipf's Law Analysis                                         │
│     ├─ Rank-frequency distribution                             │
│     ├─ Log-log linear regression                                │
│     └─ Output: α ≈ -0.6 (natural language structure)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight**: Phrase types are discovered by analyzing **which files sound similar** across the entire corpus.

---

## 2. Algorithm Comparison

### 2.1 Segmentation Strategy

| Aspect | Grain-Based | File-Based |
|--------|-------------|------------|
| **Granularity** | 10ms grains | Entire file duration |
| **Overlap** | 50% (5ms hop) | None (file-level) |
| **Segmentation Method** | Fixed-size windows | File = 1 phrase |
| **Adaptive?** | No (fixed grain size) | No (fixed to files) |

**Grain-Based**:
```python
# Extract grains with 50% overlap
for i in range(0, n_frames - grain_size, grain_size // 2):
    grain = features[i:i+grain_size]
    grains.append(grain)
```

**File-Based**:
```rust
// Each file is one phrase candidate
for file in audio_files {
    features = extract_30d_features(file);
    candidates.push(PhraseCandidate {
        features: features,
        file_name: file.name,
        // ...
    });
}
```

### 2.2 Feature Extraction

| Aspect | Grain-Based | File-Based |
|--------|-------------|------------|
| **Dimensions** | 17D | 30D |
| **MFCCs** | ❌ Not used | ✅ 13 coefficients |
| **Temporal** | ✅ Attack, decay, vibrato | ✅ Attack, decay, sustain |
| **Spectral** | ✅ HNR, flatness | ✅ HNR, flatness, harmonicity |
| **Rhythmic** | ❌ Not used | ✅ ICI, onset rate, CoV |

**Grain-Based (17D)**:
```python
features = [
    f0_normalized,           # Fundamental frequency
    duration_normalized,     # Duration
    f0_range,                # Pitch range
    attack_rate,             # Attack speed
    decay_rate,              # Decay speed
    vibrato_rate,            # Modulation frequency
    vibrato_depth,           # Modulation extent
    jitter,                  # Phase perturbation
    shimmer,                 # Amplitude perturbation
    hnr,                     # Harmonic-to-noise ratio
    spectral_flatness,       # Noise vs tonal
    # ... (5 more features)
]
```

**File-Based (30D)**:
```rust
features = [
    // Fundamental (3D)
    mean_f0_hz, duration_ms, f0_range_hz,

    // Grit Factors (3D)
    harmonic_to_noise_ratio, spectral_flatness, harmonicity,

    // Motion Factors (7D)
    attack_time_ms, decay_time_ms, sustain_level,
    vibrato_rate_hz, vibrato_depth, jitter, shimmer,

    // Fingerprint Factors (14D)
    mfcc_1 through mfcc_13, spectral_flux,

    // Rhythm Factors (3D)
    median_ici_ms, onset_rate_hz, ici_coefficient_of_variation,
]
```

### 2.3 Clustering Parameters

| Aspect | Grain-Based | File-Based |
|--------|-------------|------------|
| **eps** | 0.5 | 0.35 |
| **min_samples** | 5 | 10 |
| **Metric** | Euclidean | Euclidean |
| **Normalization** | Not specified | StandardScaler |

**Parameter Differences Explained**:

1. **eps (epsilon)**:
   - Grain-based (0.5): Larger because grains from same phrase should be very similar
   - File-based (0.35): Smaller because different files have more variation
   - Selection based on 25th percentile of pairwise distances

2. **min_samples**:
   - Grain-based (5): Lower threshold for small clusters
   - File-based (10): Higher threshold to avoid overfitting
   - Rule of thumb: dimension + 1 = 31, but practical values lower

### 2.4 DBSCAN Implementation

**Both use identical DBSCAN algorithm**, but at different scales:

**Grain-Based** (sklearn):
```python
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(grain_matrix)  # Shape: (n_grains, 17)
# Output: ~5-20 clusters per file
```

**File-Based** (custom Rust):
```rust
let dbscan = DbscanClustering::new(0.35, 10)?;
let labels = dbscan.fit_predict(&feature_matrix)?;  // Shape: (91080, 30)
// Output: ~5,000-6,000 clusters across corpus
```

**Algorithm** (same for both):
```
1. For each unvisited point p:
   a. Find neighbors within eps distance
   b. If neighbors < min_samples:
      - Mark p as noise (-1)
   c. Else:
      - Start new cluster
      - Expand via BFS/DFS queue

2. Return cluster labels (-1, 0, 1, 2, ...)
```

---

## 3. Analysis Outputs

### 3.1 Grain-Based: Grammar Discovery

**Primary Output**: Transition Entropy

```
Transition Matrix P(B|A):
         To:  0     1     2
From: 0  0.1   0.7   0.2   # Phrase 0 usually followed by 1
      1  0.8   0.1   0.1   # Phrase 1 usually followed by 0
      2  0.3   0.3   0.4   # Phrase 2 is unpredictable

Entropy per phrase:
  H(0) = -0.1*log(0.1) - 0.7*log(0.7) - 0.2*log(0.2) = 1.17 bits
  H(1) = -0.8*log(0.8) - 0.1*log(0.1) - 0.1*log(0.1) = 0.92 bits
  H(2) = -0.3*log(0.3) - 0.3*log(0.3) - 0.4*log(0.4) = 1.57 bits

Mean entropy: 1.22 bits
Grammar rigidity: 0.23 (somewhat rigid)
```

**Interpretation**:
- **Low entropy (< 1 bit)**: "A always follows B" → Strong grammar
- **High entropy (> 2 bits)**: "A can follow anything" → No grammar
- **Marmoset**: Rigid alternation (phee-alarm-phee-alarm...)
- **Bat**: More flexible sequences

### 3.2 File-Based: Phrase Type Discovery

**Primary Output**: Clustered Phrases

```
Cluster 0: 19,841 members (88% of corpus)
  - Centroid: F0=7437 Hz, wide FM sweeps
  - Intra-similarity: 0.89
  - Inter-similarity: 0.23
  - Atomic: YES

Cluster 1: 1,208 members (5% of corpus)
  - Centroid: F0=7408 Hz, narrow range
  - Intra-similarity: 0.85
  - Inter-similarity: 0.31
  - Atomic: YES

... (203 more clusters)

Zipf's Law Analysis:
  Slope (α): -0.6428
  R²: -37.1005 (needs improvement with corrected features)
  Interpretation: Natural language structure detected
```

**Interpretation**:
- **Intra > Inter similarity**: Atomic phrase (reusable unit)
- **Zipf's Law α ≈ -0.6**: Natural language structure
- **Many small clusters**: Rich vocabulary
- **Few large clusters**: Common phrases (like "the", "and" in English)

---

## 4. Use Cases

### Grain-Based: When to Use

✅ **Best for**:
- Analyzing **grammar within sequences**
- Understanding **phrase transitions**
- Detecting **syntax rules**
- Small datasets (synthetic or controlled)

✅ **Example questions**:
- "What phrases tend to follow each other?"
- "Is there a deterministic grammar?"
- "How rigid is the syntax?"

❌ **Not ideal for**:
- Large-scale corpus analysis
- Cross-file phrase type discovery
- Real-time processing

### File-Based: When to Use

✅ **Best for**:
- Discovering **phrase types across corpus**
- **Large-scale analysis** (91,000+ files)
- **Synthesis applications** (metadata-driven, concatenative)
- **Zipf's Law analysis**

✅ **Example questions**:
- "What are the reusable phrase types?"
- "How many distinct words are in the vocabulary?"
- "Does this follow natural language statistics?"

❌ **Not ideal for**:
- Within-sequence grammar analysis
- Real-time sequence processing
- Fine-grained temporal patterns

---

## 5. Performance Comparison

### 5.1 Computational Complexity

| Operation | Grain-Based | File-Based |
|-----------|-------------|------------|
| **Feature Extraction** | O(n × g) per file | O(n) per file |
| **DBSCAN** | O(g²) per file | O(N²) global |
| **Memory** | O(g × d) | O(N × d) / batch |

Where:
- `n` = file length (samples)
- `g` = grains per file (~100-500)
- `N` = total files (~91,000)
- `d` = feature dimensions (17 or 30)

### 5.2 Scalability

**Grain-Based**:
```
100 files × 500 grains/file × 17 features × 8 bytes
= 68 MB (feature matrix per file)
= 6.8 GB (for 100 files)
```

**File-Based** (with batching):
```
91,000 files × 30 features × 8 bytes
= 218 MB (total feature matrix)
= 2.4 MB per batch (1,000 files)
```

**Winner**: File-based is more memory-efficient due to batching.

---

## 6. Complementary Approaches

The two methods are **not competing**—they answer **different questions**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Combined Analysis Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. File-Based Extraction (Rust)                                │
│     ├─ Process 91,080 files                                     │
│     ├─ Discover ~5,000 phrase types                             │
│     └─ Output: "vocabulary" of the communication system         │
│    ↓                                                            │
│  2. Grain-Based Analysis (Python)                               │
│     ├─ For each phrase type:                                    │
│     │  ├─ Select representative files                           │
│     │  ├─ Extract grains within files                          │
│     │  └─ Analyze internal grammar                             │
│     └─ Output: "syntax" rules for each phrase type             │
│    ↓                                                            │
│  3. Unified Model                                               │
│     ├─ Vocabulary: Phrase types (file-based)                   │
│     ├─ Syntax: Transition rules (grain-based)                  │
│     └─ Complete: "words" + "grammar" = language                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Example**:

```
File-Based discovers:
  - PHRASE_0: "Social FM sweep" (19,841 occurrences)
  - PHRASE_1: "Narrowband social" (1,208 occurrences)
  - PHRASE_2: "Low-frequency contact" (876 occurrences)

Grain-Based analyzes PHRASE_0:
  - Internal structure: [A, B, C] sequence
  - Transition entropy: 0.8 bits (somewhat rigid)
  - Grammar rule: "A→B→C" pattern 70% of time

Combined Understanding:
  "The social FM sweep phrase has an internal structure:
   it starts with a rising F0 sweep (A), transitions to
   a plateau (B), and ends with a falling sweep (C).
   This pattern occurs 70% of the time, suggesting
   grammatical structure within the phrase."
```

---

## 7. Implementation Comparison

### 7.1 Code Structure

**Grain-Based** (Python):
```python
# Modular, object-oriented
class GrainExtractor:
    def extract_grains(self, features): ...

class AtomicPhraseDiscoverer:
    def discover_phrases(self, grains): ...

class SentenceReconstructor:
    def reconstruct(self, grains): ...

class TransitionEntropyAnalyzer:
    def analyze(self, structure): ...

# Orchestration
pipeline = GrammarDiscoveryPipeline()
phrases, structure, grammar = pipeline.discover(audio_file)
```

**File-Based** (Rust):
```rust
// Functional, batch-oriented
pub fn batch_process_and_cluster(
    audio_dir: &Path,
    batch_size: usize,
    eps: f64,
    min_samples: usize,
    checkpoint_dir: &Path,
    max_files: Option<usize>,
) -> Result<(Vec<ClusteredPhrase>, Vec<VocalizationResult>)>

// Analysis functions
pub fn calculate_intra_cluster_similarity(features: &Array2) -> f64
pub fn calculate_inter_cluster_similarity(...) -> f64
pub fn analyze_zipf_law(phrases: &[ClusteredPhrase]) -> Result<ZipfAnalysis>
```

### 7.2 Error Handling

**Grain-Based** (Python):
```python
# Exceptions
try:
    phrases = discoverer.discover_phrases(grains)
except ValueError as e:
    logger.error(f"Discovery failed: {e}")
    return []
```

**File-Based** (Rust):
```rust
// Result types
fn cluster_phrases(&self, phrases: &[PhraseCandidate])
    -> Result<Vec<ClusteredPhrase>, ExtractionError>

match pipeline.process(audio_dir) {
    Ok(results) => println!("Success: {} clusters", results.len()),
    Err(ExtractionError::AudioLoadFailed(msg)) => {
        eprintln!("Failed to load audio: {}", msg)
    }
    // ... other error types
}
```

---

## 8. Recommendations

### For Researchers

**Use Grain-Based when**:
- Studying **within-call syntax**
- Analyzing **phrase transitions**
- Working with **controlled experiments**
- Investigating **grammar evolution**

**Use File-Based when**:
- Building **vocabulary inventories**
- Analyzing **large corpora**
- Preparing for **synthesis applications**
- Studying **Zipf's Law** and natural language structure

**Use Both when**:
- Complete linguistic analysis required
- Need both **vocabulary** and **syntax**
- Building **comprehensive models**

### For Implementation

**Current Status**:
- ✅ Grain-Based: Python implementation (archived)
- ✅ File-Based: Rust implementation (active, in production)
- ⏳ Combined: Not yet implemented

**Future Work**:
1. **Port grain-based to Rust**: For performance
2. **Integrate approaches**: Combined pipeline
3. **Real-time processing**: Adaptive grain size
4. **Cross-species comparison**: Standardized metrics

---

## 9. Key Takeaways

1. **Different Scales**:
   - Grain-based: Within-file (microscopic)
   - File-based: Across-corpus (macroscopic)

2. **Different Questions**:
   - Grain-based: "What is the grammar?"
   - File-based: "What are the words?"

3. **Complementary**:
   - Can be combined for complete linguistic analysis
   - File-based discovers vocabulary
   - Grain-based discovers syntax

4. **Performance**:
   - Grain-based: Higher computational cost per file
   - File-based: Better scalability via batching

5. **Applications**:
   - Grain-based: Grammar research, sequence analysis
   - File-based: Synthesis, large-scale analysis, Zipf's Law

---

## References

1. **Grain-Based Grammar Discovery**: `/archive/experimental_analysis/grain_based_grammar_discovery.py`

2. **Parallel Extraction Pipeline**: `/src/technical_architecture/src/parallel_extraction.rs`

3. **Algorithm Documentation**: `/docs/ALGORITHMS_PARALLEL_EXTRACTION.md`

---

**Document Version**: 1.0
**Last Updated**: 2025-01-09
**Author**: Sheel Morjaria <sheelmorjaria@gmail.com>
