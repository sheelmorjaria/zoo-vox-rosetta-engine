# Algorithm Documentation: `parallel_extraction.rs`

This document explains the algorithms and techniques used in the Rust parallel extraction pipeline for animal vocalization analysis.

---

## Table of Contents

1. [Audio Loading & Decoding](#1-audio-loading--decoding)
2. [Feature Extraction (30D Micro-Dynamics)](#2-feature-extraction-30d-micro-dynamics)
3. [Feature Normalization (StandardScaler)](#3-feature-normalization-standardscaler)
4. [Clustering (DBSCAN)](#4-clustering-dbscan)
5. [Similarity Metrics](#5-similarity-metrics)
6. [Batch Processing & Checkpointing](#6-batch-processing--checkpointing)
7. [Turn-Taking Analysis](#7-turn-taking-analysis)
8. [Zipf's Law Analysis](#8-zipfs-law-analysis)

---

## 1. Audio Loading & Decoding

### Hybrid Audio Loading Strategy

The pipeline uses a **hybrid approach** for maximum efficiency:

| Format | Decoder | Rationale |
|--------|---------|-----------|
| WAV | `hound` | Simpler, faster, lower overhead |
| FLAC/MP3/AAC/OGG | `symphonia` | Modern multi-format decoder |

**Implementation** (`load_audio_file`):
```rust
match extension.as_deref() {
    Some("wav") => load_wav_file(path),      // Fast path for WAV
    Some("flac") | Some("mp3") => load_symphonia_file(path),
    _ => load_wav_file(path).or_else(|_| load_symphonia_file(path))
}
```

**Why this matters:**
- **Performance**: Hound is ~2-3x faster for WAV than Symphonia
- **Flexibility**: Symphonia supports lossless (FLAC) and lossy (MP3, AAC) formats
- **Sample rate handling**: All audio converted to mono f32 at original sample rate

---

## 2. Feature Extraction (30D Micro-Dynamics)

### The 30D Feature Vector

Each audio segment is converted to a **30-dimensional feature vector** capturing temporal, spectral, and rhythmic characteristics:

```
┌─ Fundamental (3D) ──────────────────────────────┐
│ • mean_f0_hz          # Mean fundamental frequency │
│ • duration_ms         # Segment duration          │
│ • f0_range_hz        # Pitch range               │
└───────────────────────────────────────────────────┘

┌─ Grit Factors (3D) ─────────────────────────────┐
│ • harmonic_to_noise_ratio  # Timbre harshness    │
│ • spectral_flatness       # Noise vs tonal       │
│ • harmonicity             # Harmonic content      │
└───────────────────────────────────────────────────┘

┌─ Motion Factors (7D) ───────────────────────────┐
│ • attack_time_ms    # Attack speed               │
│ • decay_time_ms     # Decay speed                │
│ • sustain_level     # Steady-state amplitude     │
│ • vibrato_rate_hz   # Modulation frequency       │
│ • vibrato_depth     # Modulation extent          │
│ • jitter            # Phase perturbation         │
│ • shimmer           # Amplitude perturbation     │
└───────────────────────────────────────────────────┘

┌─ Fingerprint Factors (14D) ─────────────────────┐
│ • mfcc_1 through mfcc_13  # Spectral envelope    │
│ • spectral_flux           # Spectral change      │
└───────────────────────────────────────────────────┘

┌─ Rhythm Factors (3D) ───────────────────────────┐
│ • median_ici_ms                  # Timing pattern │
│ • onset_rate_hz                  # Event density  │
│ • ici_coefficient_of_variation   # Timing variability│
└───────────────────────────────────────────────────┘
```

### MFCC Extraction Pipeline

The **Mel-Frequency Cepstral Coefficients (MFCCs)** are computed as follows:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MFCC Pipeline                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Power Spectrum                                            │
│     ├─ Compute FFT magnitude                                   │
│     └─ Square to get power                                     │
│                                                                 │
│  2. Mel Filterbank (26 triangular filters)                     │
│     ├─ Convert Hz to Mel scale: 2595*log10(1 + f/700)         │
│     ├─ Space filters evenly on Mel scale                       │
│     └─ Apply triangular weighting to spectrum                  │
│                                                                 │
│  3. Log Mel-Energies                                           │
│     └─ log(mel_energy)  (with floor at -11.5 for zeros)       │
│                                                                 │
│  4. Discrete Cosine Transform (DCT-II)                         │
│     └─ mfcc[k] = sqrt(2/n) * Σ log_mel[i] * cos(πk(2i+1)/2n)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why MFCCs?**
- **Perceptual relevance**: Mel scale approximates human/bat hearing
- **Dimensionality reduction**: 26 Mel bands → 13 coefficients
- **Decorrelation**: DCT roughly decorrelates features
- **Standard in speech processing**: Proven effectiveness for vocalizations

### Temporal Features

**Attack/Decay Time**:
```rust
attack_time = time_to_reach_90%_of_peak
decay_time = time_to_fall_to_10%_of_peak
```

**Vibrato Detection**:
- Find peaks in amplitude envelope
- Calculate inter-peak intervals
- Vibrato rate = 1 / mean_interval
- Vibrato depth = amplitude_variation / mean_amplitude

### Perturbation Features

**Jitter** (phase perturbation):
```rust
jitter = std_dev(zero_crossing_intervals) / mean_interval
```

**Shimmer** (amplitude perturbation):
```rust
shimmer = std_dev(peak_amplitudes) / mean_amplitude
```

---

## 3. Feature Normalization (StandardScaler)

### Why Normalize?

Features have **different scales**:
- `mean_f0_hz`: ~10,000 Hz (bats use ultrasonic vocalizations)
- `harmonicity`: 0.0 to 1.0
- `attack_time_ms`: 0.0 to 100.0 ms

Without normalization, high-magnitude features (like F0) would dominate distance calculations.

### Z-Score Normalization

**Algorithm**:
```rust
mean[i] = (1/n) * Σ x[j][i]
std[i] = sqrt((1/n) * Σ (x[j][i] - mean[i])^2)
normalized[i] = (x[i] - mean[i]) / std[i]
```

**Properties**:
- Each feature has **mean = 0** and **std = 1**
- Preserves the **shape** of the distribution
- Sensitive to **outliers** (robust scalers could be used alternatively)

---

## 4. Clustering (DBSCAN)

### Density-Based Spatial Clustering

**DBSCAN** groups points based on **density**, not distance to centroids.

### Key Concepts

```
┌───────────────────────────────────────────────────────────────┐
│                     DBSCAN Concepts                           │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  Core Point:                                                  │
│    A point with ≥ min_samples within eps distance             │
│                                                               │
│  Border Point:                                                │
│    Within eps distance of a core point, but                  │
│    has < min_samples in its own neighborhood                 │
│                                                               │
│  Noise Point:                                                 │
│    Not within eps distance of any core point                 │
│                                                               │
│  Cluster:                                                     │
│    All core points connected via density-reachability,       │
│    plus their associated border points                       │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Algorithm Steps

```
1. For each unvisited point p:
   a. Find neighbors within eps distance
   b. If neighbors < min_samples:
      - Mark p as noise (-1)
   c. Else:
      - Create new cluster
      - Expand cluster by adding density-reachable points
      - Use BFS/DFS queue for expansion

2. Return cluster labels:
   - -1: noise
   - 0, 1, 2, ...: cluster IDs
```

### Distance Metric

**Euclidean Distance** (squared for efficiency):
```rust
dist²(a, b) = Σ (a[i] - b[i])²
```

**Why Euclidean?**
- Simple and interpretable
- Works well with normalized features
- Standard choice for DBSCAN

### Parameter Selection

**eps (epsilon)**:
- Maximum distance for neighborhood
- **Too small**: Many small clusters, lots of noise
- **Too large**: One giant cluster
- **Selection method**: K-distance graph (elbow method)

**min_samples**:
- Minimum points for core region
- **Too small**: Noisy clusters, overfitting
- **Too large**: Missed clusters, underfitting
- **Rule of thumb**: `dimension + 1` = 31 for 30D features

**Current Configuration**:
- `eps = 0.35` (25th percentile of pairwise distances)
- `min_samples = 10` (stricter than default 5)

### Advantages of DBSCAN

1. **No predefined cluster count**: Discovers natural clusters
2. **Arbitrary shapes**: Not limited to convex/ spherical clusters
3. **Noise handling**: Explicitly identifies outliers
4. **Single pass**: O(n log n) with spatial indexing (O(n²) without)

### Disadvantages

1. **Struggles with varying densities**: All clusters must have similar density
2. **Parameter sensitivity**: Performance heavily depends on eps/min_samples
3. **Curse of dimensionality**: Distance becomes less meaningful in high dimensions

---

## 5. Similarity Metrics

### Cosine Similarity

**Definition**:
```rust
cosine_sim(a, b) = (a · b) / (||a|| * ||b||)
                 = Σ(a[i] * b[i]) / (sqrt(Σa[i]²) * sqrt(Σb[i]²))
```

**Range**: -1 to 1
- **1.0**: Identical direction (same features, different magnitude)
- **0.0**: Orthogonal (completely different)
- **-1.0**: Opposite direction

**Why Cosine Similarity?**
- **Magnitude-invariant**: Focuses on feature ratios, not absolute values
- **Interpretable**: Direct measure of feature alignment
- **Text/audio standard**: Proven effective for high-dimensional feature vectors

### Intra-Cluster Similarity

Measures **cluster coherence**:

```rust
intra_sim = mean(cosine_sim(p[i], p[j])) for all i, j in cluster
```

**High intra-similarity** → Compact, coherent cluster

### Inter-Cluster Similarity

Measures **cluster separation**:

```rust
centroid = mean(cluster_features)
inter_sim = mean(cosine_sim(centroid, other_points))
```

**Low inter-similarity** → Well-separated from other clusters

### Atomic Phrase Detection

A phrase is **"atomic"** (reusable as a unit) if:

```rust
is_atomic = (intra_sim > 0.7) && (intra_sim > inter_sim)
```

**Rationale**:
- **Cohesive**: Members are similar to each other
- **Distinct**: Different from other clusters
- These correspond to reusable "words" in the communication system

---

## 6. Batch Processing & Checkpointing

### Memory Efficiency

Processing 91,080 audio files with 30D features would require:

```
91,080 files × 1,000 segments/file × 30 features × 8 bytes/feature
= ~21.9 GB just for feature matrices
```

**Solution**: Process in **batches of 1,000 files**

### Checkpointing Strategy

```
┌───────────────────────────────────────────────────────────────┐
│                    Checkpoint Pipeline                        │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Extract features from batch (1,000 files)                 │
│     └─ Save to checkpoint: candidates_checkpoint.json        │
│                                                               │
│  2. Accumulate candidates from all batches                    │
│                                                               │
│  3. After all batches:                                        │
│     ├─ Load all candidates from checkpoint                    │
│     ├─ Normalize features (global StandardScaler)             │
│     ├─ Cluster all candidates (global DBSCAN)                 │
│     └─ Assign cluster IDs                                     │
│                                                               │
│  4. Save final results                                        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

**Why this approach?**
- **Memory bounded**: Never load more than 1,000 files at once
- **Resumable**: Can restart from last checkpoint if interrupted
- **Global clustering**: All candidates clustered together (not per-batch)

---

## 7. Turn-Taking Analysis

### Conversation Detection

**Pattern**: A → B → A sequences

```rust
conversations = detect_ABA_sequences(annotations)
```

**ABA Detection Algorithm**:
1. Iterate through annotated vocalizations
2. Detect when emitter switches back to original speaker
3. Extract sequences as "conversations"

### Metrics Computed

**Turn-Switch Rate**:
```rust
turn_switch_rate = (n_switches / n_total - 1) × 100%
```

**Response Time Statistics**:
- Mean gap between vocalizations
- Median gap
- Immediate responses (< 100ms)

**Conversation Statistics**:
- Mean length (number of turns)
- Multi-turn conversations (> 2 turns)
- Long conversations (> 10 turns)

**Pattern Classification**:
- **ABA**: Speaker A → B → A (taking back turn)
- **Dyadic**: Exactly 2 speakers
- **Multi-party**: 3+ speakers

### Social Network Analysis

Constructs a **directed graph** of speaker interactions:

```rust
adjacency_matrix[emitter][addressee] += 1
```

**Metrics**:
- Out-degree: How often emitter initiates
- In-degree: How often addressee is targeted
- Reciprocity: A↔B mutual exchanges

---

## 8. Zipf's Law Analysis

### Zipf's Law in Natural Language

**Zipf's Law**: Frequency of a word is inversely proportional to its rank.

```
frequency(rank) = C / rank^α
```

Where:
- **C**: Constant (frequency of most common word)
- **α**: Zipf exponent (typically ~1.0 for natural language)
- **rank**: Word position when sorted by frequency (1 = most common)

### Linear Regression Method

**Log-log transformation**:
```
log(frequency) = log(C) - α * log(rank)
```

This is a **linear equation**: `y = b + mx`

Where:
- `y = log(frequency)`
- `x = log(rank)`
- `b = log(C)` (intercept)
- `m = -α` (slope)

**Least Squares Solution**:
```rust
α = - (n*Σ(xy) - Σx*Σy) / (n*Σ(x²) - (Σx)²)
R² = 1 - (SS_res / SS_tot)
```

### Interpretation

**Slope (α)**:
- **α ≈ -1.0**: Natural language (English, Mandarin, etc.)
- **α ≈ -0.6**: Animal vocalizations (bats, dolphins)
- **α ≈ 0**: Flat distribution (no structure)

**R² (Correlation)**:
- **R² > 0.8**: Strong Zipf's Law compliance
- **R² > 0.6**: Moderate compliance
- **R² < 0.6**: Weak/no natural language structure

### Biological Significance

**Why does this matter?**
- **Natural languages** follow Zipf's Law
- **Animal communication** that follows Zipf's Law suggests **linguistic structure**
- **Deviations** indicate simpler communication systems

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Audio Loading | O(n) | Linear in audio length |
| MFCC Extraction | O(n log n) | FFT dominates |
| DBSCAN (no index) | O(n²) | Pairwise distances |
| DBSCAN (with index) | O(n log n) | KD-tree or ball tree |
| StandardScaler | O(n × d) | n samples, d dimensions |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Audio buffer | O(n) | Per file |
| Feature matrix | O(n × d) | n samples, 30 dimensions |
| Distance matrix | O(n²) | For DBSCAN (avoided with batching) |
| Checkpoint | O(b × d) | b = batch size (1,000) |

### Optimization Techniques

1. **Parallelization**: Rayon for parallel feature extraction
2. **Batching**: Memory-bounded processing
3. **Checkpointing**: Resumable computation
4. **Distance squared**: Avoid sqrt() in comparisons
5. **Sparse operations**: Skip distance calculations for distant points

---

## References

1. **DBSCAN**: Ester, M., et al. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise." KDD.

2. **MFCC**: Davis, S., & Mermelstein, P. (1980). "Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences." IEEE TASLP.

3. **Zipf's Law**: Zipf, G. K. (1949). "Human Behavior and the Principle of Least Effort." Addison-Wesley.

4. **Turn-Taking**: Sacks, H., Schegloff, E. A., & Jefferson, G. (1974). "A simplest systematics for the organization of turn-taking for conversation." Language.

5. **Cosine Similarity**: Singhal, A. (2001). "Modern information retrieval: A brief overview." IEEE Data Eng. Bull.

---

## Appendix: Example Workflow

```
Input: 91,080 audio files (WAV, FLAC, MP3)

┌─────────────────────────────────────────────────────────────┐
│ 1. Batch Extraction                                         │
│    ├─ Batch 1: files 0-1,000     → candidates_1.json       │
│    ├─ Batch 2: files 1,000-2,000 → candidates_2.json       │
│    └─ ...                                                    │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Feature Extraction (30D)                                 │
│    ├─ MFCCs (13D)                                           │
│    ├─ Temporal (7D)                                         │
│    ├─ Spectral (6D)                                         │
│    └─ Rhythmic (3D)                                         │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Normalization                                            │
│    └─ StandardScaler: zero mean, unit variance              │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. DBSCAN Clustering                                        │
│    ├─ eps = 0.35                                            │
│    ├─ min_samples = 10                                      │
│    └─ Output: cluster labels (-1, 0, 1, 2, ...)            │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Analysis                                                 │
│    ├─ Phrase types: ~5,000 clusters                         │
│    ├─ Atomic phrases: Intra-sim > Inter-sim                │
│    ├─ Zipf's Law: α ≈ -0.6                                 │
│    └─ Turn-taking: 66.5% turn-switch rate                  │
└─────────────────────────────────────────────────────────────┘
```

---

**Document Version**: 1.0
**Last Updated**: 2025-01-09
**Author**: Sheel Morjaria <sheelmorjaria@gmail.com>
