# Atomic Phrase Detection Analysis

**Date**: 2026-01-08
**Source**: `realtime/parallel_extraction_optimized.py` and `realtime/parallel_unified_extraction.py`

---

## What is an Atomic Phrase?

An **atomic phrase** is a cluster of acoustically similar audio segments (phrase candidates) that:
1. **Has high internal coherence** (members are similar to each other)
2. **Is well-separated from other clusters** (distinct from other phrases)
3. **Represents a reusable vocal unit** in the animal's communication system

**Atomicity Criteria** (line 447 of `parallel_unified_extraction.py`):
```python
is_atomic = (intra_sim > 0.2) and (inter_sim < 0.6)
```

---

## How Atomic Phrases Are Found

### Step 1: Extract Phrase Candidates

**Location**: `parallel_extraction_optimized.py:209-271`

**Process**:
1. Load audio file (already segmented into vocalizations)
2. Pre-compute features for entire audio (MFCCs, spectral features, etc.)
3. Apply **sliding windows** of 3 sizes: 100ms, 200ms, 400ms
4. Use **75% overlap** between windows (hop = window_size // 4)
5. Filter by **RMS threshold** (> 0.001) to skip quiet segments
6. Extract **29D features** for each window

**Result**: Thousands of phrase candidates per audio file

**Data Structure**:
```python
@dataclass
class PhraseCandidate:
    start_sample: int
    end_sample: int
    features_29d: Dict[str, float]
    source_sentence_id: str
    window_id: int
    context: int
```

---

### Step 2: Normalize Features

**Location**: `parallel_unified_extraction.py:410-420`

**Process**:
1. Collect all phrase candidates across all audio files
2. Convert to feature matrix (n_candidates × 29_features)
3. Apply **StandardScaler** normalization:
   - Subtract mean for each feature
   - Divide by standard deviation
   - Result: zero-mean, unit-variance features

**Purpose**: Ensures all features contribute equally to clustering

---

### Step 3: DBSCAN Clustering

**Location**: `parallel_unified_extraction.py:421-474`

**Algorithm**: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**Parameters**:
- `eps = 0.5` (maximum distance between neighbors)
- `min_samples = 5` (minimum cluster size)

**Process**:
1. Apply DBSCAN to normalized features
2. Each cluster = one phrase (atomic or non-atomic)
3. Label `-1` = noise (ignored)

**Output**: Cluster IDs for each phrase candidate

---

### Step 4: Calculate Cluster Similarities

**Location**: `parallel_unified_extraction.py:437-447`

#### Intra-Cluster Similarity (`intra_sim`)

**Definition**: Average pairwise cosine similarity **within** a cluster

**Formula**:
```python
def _calculate_intra_cluster_similarity(cluster_features):
    if len(cluster_features) < 2:
        return 1.0

    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            # Cosine similarity
            dot = np.dot(cluster_features[i], cluster_features[j])
            norm_i = np.linalg.norm(cluster_features[i])
            norm_j = np.linalg.norm(cluster_features[j])
            if norm_i > 0 and norm_j > 0:
                sim = dot / (norm_i * norm_j)
                similarities.append(sim)

    return np.mean(similarities)
```

**Interpretation**:
- High `intra_sim` (> 0.2) = cluster members are similar
- Low `intra_sim` = cluster is scattered/diverse

---

#### Inter-Cluster Similarity (`inter_sim`)

**Definition**: Average similarity **from cluster centroid to nearest other cluster members**

**Formula**:
```python
def _calculate_inter_cluster_similarity(all_features, cluster_indices, labels, cluster_id):
    other_indices = np.where(labels != cluster_id)[0]

    if len(other_indices) == 0:
        return 0.0

    cluster_members = all_features[cluster_indices]
    other_members = all_features[other_indices]

    # Calculate centroid of this cluster
    centroid = np.mean(cluster_members, axis=0)

    # Calculate similarities to other cluster members
    similarities = []
    for other in other_members:
        dot = np.dot(centroid, other)
        norm_centroid = np.linalg.norm(centroid)
        norm_other = np.linalg.norm(other)
        if norm_centroid > 0 and norm_other > 0:
            sim = dot / (norm_centroid * norm_other)
            similarities.append(sim)

    return np.mean(similarities)
```

**Interpretation**:
- Low `inter_sim` (< 0.6) = cluster is distinct from others
- High `inter_sim` = cluster blends with neighbors (not distinct)

---

### Step 5: Determine Atomicity

**Location**: `parallel_unified_extraction.py:447`

**Atomicity Formula**:
```python
is_atomic = (intra_sim > 0.2) and (inter_sim < 0.6)
```

**Atomic Phrase Criteria**:
1. **Internal cohesion**: `intra_sim > 0.2`
   - Cluster members are similar to each other
   - Represents a consistent acoustic pattern

2. **External separation**: `inter_sim < 0.6`
   - Cluster is distinct from other clusters
   - Not easily confused with other phrases

**Result**:
- `is_atomic = True`: Phrase is a reusable vocal unit
- `is_atomic = False`: Phrase is not coherent enough or not distinct enough

---

### Step 6: Filter Atomic Phrases

**Location**: `parallel_extraction_optimized.py:664`

```python
phrases = cluster_phrases_dbscan(candidate_objs, eps=0.5, min_samples=5)
atomic_phrases = [p for p in phrases if p.is_atomic]
```

**Statistics Reported**:
```python
{
    "total_phrases": 150,           # All clusters
    "atomic_phrases": 85,            # Only atomic ones
    "atomic_ratio": 0.57             # 57% are atomic
}
```

---

## Data Structure: AtomicPhrase

**Location**: `parallel_unified_extraction.py:452-472`

```python
@dataclass
class AtomicPhrase:
    phrase_id: str                    # e.g., "phrase_42"
    cluster_id: int                   # DBSCAN cluster ID
    features_29d: Dict[str, float]    # Centroid features (29D)
    member_candidates: List[Dict]      # All phrase candidates in cluster
    intra_cluster_similarity: float    # Internal coherence (0-1)
    inter_cluster_similarity: float    # External separation (0-1)
    is_atomic: bool                   # True if atomic phrase
    contexts: List[int]                # Context labels of members
```

**Example**:
```python
AtomicPhrase(
    phrase_id="phrase_42",
    cluster_id=42,
    features_29d={
        "mean_f0_hz": 15000.0,
        "duration_ms": 150.0,
        "mfcc_1": -500.0,
        # ... 26 more features
    },
    member_candidates=[
        {"source_sentence_id": "bat_001", "window_id": 5, ...},
        {"source_sentence_id": "bat_007", "window_id": 12, ...},
        {"source_sentence_id": "bat_023", "window_id": 3, ...},
        # ... more members
    ],
    intra_cluster_similarity=0.75,    # High coherence
    inter_cluster_similarity=0.35,    # Well-separated
    is_atomic=True,                    # ✅ ATOMIC
    contexts=[1, 1, 2, 1, 3]           # Used in multiple contexts
)
```

---

## Compositionality Detection

**Location**: `parallel_unified_extraction.py:532-560`

**Definition**: Phrase reuse across sentences

**Formula**:
```python
compositionality_ratio = reusable_phrases / total_unique_phrases
```

**Reusable Phrase**: Used in > 1 sentence

**Purpose**: Measures how much phrases are reused (composed) vs. unique

**Example**:
```python
{
    "total_unique_phrases": 85,
    "reusable_phrases": 52,             # Used in multiple sentences
    "compositionality_ratio": 0.61,     # 61% compositionality
    "phrase_usage": {
        "phrase_42": {"sentence_count": 15, "contexts": {1, 2, 3}},
        "phrase_7": {"sentence_count": 8, "contexts": {1, 1}},
        # ...
    }
}
```

---

## Thresholds Analysis

### Atomicity Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| **intra_sim > 0.2** | 0.2 | Minimum internal coherence |
| **inter_sim < 0.6** | 0.6 | Maximum external similarity |

**Why These Thresholds?**

**Intra-cluster > 0.2**:
- Cosine similarity ranges from -1 to 1
- 0.2 = 20% similarity (moderate correlation)
- Ensures cluster members share acoustic characteristics
- Prevents over-fragmentation (too many small clusters)

**Inter-cluster < 0.6**:
- 0.6 = 60% similarity (high correlation)
- Ensures phrases are distinct from each other
- Prevents under-fragmentation (merging distinct phrases)

### DBSCAN Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **eps** | 0.5 | Maximum distance for neighbors |
| **min_samples** | 5 | Minimum cluster size |

**Why These Values?**

**eps = 0.5**:
- Normalized feature space (unit variance)
- 0.5 = moderate similarity threshold
- Balances over-clustering vs. under-clustering

**min_samples = 5**:
- Prevents spurious clusters from noise
- Ensures phrases are observed multiple times
- Provides statistical robustness

---

## Optimizations in `parallel_extraction_optimized.py`

### Key Differences from Original

1. **Skip PELT**: Each audio file is already one vocalization (sentence)
2. **Pre-compute features**: MFCCs computed once, then segmented
3. **Fewer window sizes**: 3 sizes (100ms, 200ms, 400ms) vs. 7
4. **Higher overlap**: 75% vs. 50% (more candidates)
5. **Faster audio loading**: `soundfile` vs. `librosa`
6. **Memory optimization**: `PhraseCandidate` doesn't store audio_segment

### Performance

**Expected Speedup**: 5-10x faster than original

**Throughput**: ~1-2 files/sec (vs. ~0.2 files/sec)

---

## Rust Implementation Implications

### Missing Components

The current Rust implementation (`parallel_extraction.rs`) is **missing**:

1. ❌ **Intra-cluster similarity calculation**
2. ❌ **Inter-cluster similarity calculation**
3. ❌ **Atomic phrase determination logic**
4. ❌ **Compositionality detection**
5. ❌ **Phrase reuse tracking**

### Required Additions

To match Python functionality, Rust needs:

```rust
// 1. Intra-cluster similarity
pub fn calculate_intra_cluster_similarity(cluster_features: &Array2<f64>) -> f64 {
    // Average pairwise cosine similarity within cluster
}

// 2. Inter-cluster similarity
pub fn calculate_inter_cluster_similarity(
    all_features: &Array2<f64>,
    cluster_indices: &[usize],
    labels: &[i32],
    cluster_id: i32,
) -> f64 {
    // Average similarity from centroid to other clusters
}

// 3. Atomic phrase check
pub fn is_atomic_phrase(intra_sim: f64, inter_sim: f64) -> bool {
    intra_sim > 0.2 && inter_sim < 0.6
}

// 4. ClusteredPhrase with atomicity
pub struct ClusteredPhrase {
    pub phrase: PhraseCandidate,
    pub cluster_id: i32,
    pub intra_sim: f64,
    pub inter_sim: f64,
    pub is_atomic: bool,
}
```

---

## Summary

### What is an Atomic Phrase?

An atomic phrase is a **coherent, distinct cluster of similar audio segments** that represents a reusable vocal unit in animal communication.

### How Are They Found?

1. Extract phrase candidates (sliding windows)
2. Normalize features (StandardScaler)
3. Cluster with DBSCAN (eps=0.5, min_samples=5)
4. Calculate similarities:
   - **Intra**: Average pairwise similarity within cluster
   - **Inter**: Average similarity to other clusters
5. Filter by atomicity: `intra > 0.2` AND `inter < 0.6`

### Why Atomic Phrases Matter?

- **Compositionality**: Reusable building blocks of communication
- **Grammar**: Combine to form complex vocalizations
- **Efficiency**: Reduce search space for synthesis
- **Interpretability**: Discrete units vs. continuous variation

---

**Generated**: 2026-01-08
**Author**: Sheel Morjaria (sheelmorjaria@gmail.com)
**License**: CC BY-ND 4.0 International
