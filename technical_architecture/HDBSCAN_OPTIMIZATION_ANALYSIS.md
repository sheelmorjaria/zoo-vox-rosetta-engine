# HDBSCAN Performance Optimization Analysis

## Current Status
- **Dataset**: 871,045 phrases × 15D features
- **Running time**: ~8+ hours (still in progress)
- **CPU usage**: 31 cores (3096% CPU)
- **Algorithm**: HDBSCAN with min_cluster_size=933, min_samples=699

---

## Critical Bottlenecks Identified

### 1. O(n²) Core Distance Computation (CRITICAL)
**Location**: `hdbscan.rs:233-266` - `compute_core_distances_parallel()`

```rust
for i in 0..n_samples {
    for j in 0..n_samples {
        if i != j {
            let dist = self.euclidean_distance(&feature_rows[i], &feature_rows[j]);
            distances.push(dist);
        }
    }
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    core_dist = distances[min_samples - 1];
}
```

**Problem**: For 871,045 samples:
- Distance computations: 871,045² ≈ **758 billion operations**
- Even with parallelization, this takes hours
- Sorting 870K distances per sample adds significant overhead

**Impact**: This is the primary bottleneck - approximately 70-80% of total runtime

---

### 2. Inefficient Edge Heap (HIGH)
**Location**: `hdbscan.rs:142-143` - `build_mst_online()`

```rust
// Sort edges by weight on EVERY iteration
edge_heap.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap());
```

**Problem**:
- Full O(h log h) sort on every MST iteration
- For n=871K, heap grows to hundreds of thousands of edges
- Should use binary heap with O(log h) push/pop

**Impact**: ~15-20% of MST construction time

---

### 3. Linear Search for Valid Edge (MEDIUM)
**Location**: `hdbscan.rs:146-153`

```rust
let mut min_idx = 0;
while min_idx < edge_heap.len() {
    let edge = &edge_heap[min_idx];
    if in_mst.contains(&edge.from) ^ in_mst.contains(&edge.to) {
        break;
    }
    min_idx += 1;
}
```

**Problem**: Linear scan through sorted edges to find one that connects MST to non-MST vertex

**Impact**: Worst case O(h) per iteration

---

### 4. O(n) Label Reassignment on Every Merge (MEDIUM)
**Location**: `hdbscan.rs:404-409`

```rust
for label in labels.iter_mut() {
    if *label == old_label {
        *label = new_label;
    }
}
```

**Problem**: Linear scan through all 871K labels for each of ~871K merge operations

**Impact**: O(n²) overall for cluster extraction

---

### 5. No Early Termination or Approximation
**Problem**: Full exact computation with no fallback to approximate methods for large datasets

---

## Optimization Strategies (Priority Order)

### P0: Use Approximate KNN for Core Distances (10-100x speedup)

**Strategy**: Replace exact KNN with approximate nearest neighbor search

**Options**:
1. **Use KD-Tree** (from `kdtree` crate) - O(n log n) for 15D
2. **Use HNSW** (from `hnsw_rs` crate) - O(n log n) with excellent empirical performance
3. **Use Ball Tree** - Better for higher dimensions

**Implementation** (HNSW example):
```rust
use hnsw_rs::Hnsw;

fn compute_core_distances_approx(&self, features: &Array2<f64>) -> Result<Vec<f64>> {
    let n_samples = features.nrows();
    let feature_rows: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| features.row(i).to_vec())
        .collect();

    // Build HNSW index (O(n log n))
    let mut hnsw = Hnsw::new(32, n_samples, 16, 200);
    for (i, row) in feature_rows.iter().enumerate() {
        hnsw.insert(i, row);
    }

    // Query KNN (O(log n) per query)
    let core_distances: Vec<f64> = feature_rows.par_iter()
        .map(|row| {
            let neighbors = hnsw.search(row, self.min_samples);
            neighbors.last().map(|d| d.distance).unwrap_or(0.0)
        })
        .collect();

    Ok(core_distances)
}
```

**Expected Speedup**: 10-100x on core distance computation

**Trade-off**: Slight accuracy loss (typically <1-2% for clustering)

---

### P1: Replace Vector with Binary Heap (5-10x speedup for MST)

**Strategy**: Use `std::collections::BinaryHeap` or `priority-queue` crate

```rust
use std::collections::BinaryHeap;
use std::cmp::Reverse;

#[derive(PartialEq, Eq)]
struct HeapEdge {
    weight: f64,
    from: usize,
    to: usize,
}

impl Ord for HeapEdge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Reverse(self.weight.partial_cmp(&other.weight).unwrap())
            .cmp(&Reverse(other.weight.partial_cmp(&self.weight).unwrap()))
    }
}

impl PartialOrd for HeapEdge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// In build_mst_online:
let mut edge_heap: BinaryHeap<HeapEdge> = BinaryHeap::new();

// Push: O(log h)
edge_heap.push(HeapEdge { from: new_vertex, to: j, weight: dist });

// Pop: O(log h)
while let Some(edge) = edge_heap.pop() {
    if in_mst.contains(&edge.from) ^ in_mst.contains(&edge.to) {
        // Valid edge found
        mst_edges.push(MstEdge { from: edge.from, to: edge.to, weight: edge.weight });
        break;
    }
}
```

**Expected Speedup**: 5-10x for MST construction phase

---

### P2: Use Union-Find for Cluster Extraction (2-5x speedup)

**Strategy**: Replace linear label reassignment with Disjoint Set Union (DSU)

```rust
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        let mut parent = Vec::with_capacity(n);
        let mut rank = vec![0; n];
        for i in 0..n {
            parent.push(i);
        }
        Self { parent, rank }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // Path compression
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let px = self.find(x);
        let py = self.find(y);
        if px != py {
            // Union by rank
            if self.rank[px] < self.rank[py] {
                self.parent[px] = py;
            } else if self.rank[px] > self.rank[py] {
                self.parent[py] = px;
            } else {
                self.parent[py] = px;
                self.rank[px] += 1;
            }
        }
    }
}
```

**Expected Speedup**: 2-5x for cluster extraction

---

### P3: Spatial Partitioning for MST (2-3x speedup)

**Strategy**: Use spatial data structure (KD-tree) to find nearest neighbor in MST more efficiently

**Approach**: Only compute distances to nearby vertices when adding new vertex to MST

---

### P4: Sampling-Based Approximation (Optional)

**Strategy**: For very large datasets (>100K), use hierarchical approach:
1. Cluster a random sample (e.g., 50K points)
2. Assign remaining points to nearest cluster center

**Trade-off**: Faster but less accurate

---

## Recommended Implementation Plan

### Phase 1: Quick Wins (can be done immediately)
1. Replace `Vec` with `BinaryHeap` for edge management (P1)
2. Implement Union-Find for cluster extraction (P2)

**Expected improvement**: 2-3x overall speedup

### Phase 2: Core Optimization (requires dependency addition)
3. Add `hnsw_rs` or `kdtree` dependency
4. Implement approximate KNN for core distances (P0)

**Expected improvement**: 10-50x overall speedup

### Phase 3: Advanced (optional)
5. Implement spatial partitioning for MST (P3)
6. Add sampling-based fallback for very large datasets (P4)

---

## Alternative: Use Existing Optimized HDBSCAN

Instead of re-implementing, consider using existing optimized implementations:

### Option A: Python's `hdbscan` library
```bash
pip install hdbscan
```
- Uses optimized Cython/numba
- Has approximate algorithms for large datasets
- Can be called from Rust via Python bridge

### Option B: `scikit-learn`'s HDBSCAN
- Well-optimized C++ backend
- Handles large datasets efficiently

### Option C: R's `dbscan` package
- High-performance implementation
- Has HDBSCAN with KD-tree/Ball-tree support

---

## Dependency Additions

To implement optimizations, add to `Cargo.toml`:

```toml
[dependencies]
# For approximate nearest neighbor (P0)
hnsw_rs = "0.3"  # or "kdtree" = "0.7"

# For better heap (optional, std::collections::BinaryHeap works)
priority-queue = "2.0"  # Optional upgrade
```

---

## Estimated Runtime Improvements

| Optimization | Current Phase | Expected Speedup | New Runtime |
|--------------|---------------|------------------|-------------|
| Baseline | Core distances (8hr+) | - | 8-12 hours |
| P0: Approx KNN | Core distances | 10-100x | 5-50 minutes |
| P1: Binary Heap | MST | 5-10x | 30-60 min → 3-6 min |
| P2: Union-Find | Extraction | 2-5x | 10-20 min → 2-4 min |
| **All (P0+P1+P2)** | **Full pipeline** | **50-200x** | **10-30 minutes** |

---

## Decision Matrix

| Strategy | Speedup | Complexity | Risk | Recommendation |
|----------|---------|------------|------|----------------|
| P0: Approx KNN | 10-100x | Medium | Low | **DO IT** - Critical bottleneck |
| P1: Binary Heap | 5-10x | Low | Low | **DO IT** - Easy win |
| P2: Union-Find | 2-5x | Low | Low | **DO IT** - Easy win |
| P3: Spatial MST | 2-3x | High | Medium | Maybe - More complex |
| P4: Sampling | 10x+ | Low | High | Maybe - Quality tradeoff |
| Use Python hdbscan | 50x+ | Low | Low | **CONSIDER** - Fastest solution |

---

## Immediate Action Items

1. **Stop current run** if estimated time > 24 hours
2. **Implement P1 + P2** (Binary Heap + Union-Find) - quick wins
3. **Implement P0** (Approx KNN) - game changer
4. **Or use Python's hdbscan** as fastest path to results

---

## Conclusion

The current implementation has an O(n²) bottleneck in core distance computation that dominates runtime. The single most impactful optimization is replacing exact KNN with approximate nearest neighbor search (HNSW or KD-tree), which can provide 10-100x speedup with minimal accuracy loss.

Combined with Binary Heap and Union-Find optimizations, total speedup of **50-200x** is achievable, reducing runtime from 8-12 hours to **10-30 minutes**.
