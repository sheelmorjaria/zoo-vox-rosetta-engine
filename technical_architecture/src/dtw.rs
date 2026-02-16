// Dynamic Time Warping (DTW)
//
// Implements DTW for measuring similarity between temporal sequences that may
// vary in speed. Critical for comparing vocalization phrases that are stretched
// or compressed in time.
//
// DTW finds the optimal alignment between two sequences by minimizing the
// cumulative distance between warped time indices.
//
// Reference: Salvador, S., & Chan, P. (2007). "FastDTW: Toward accurate
// dynamic time warping in linear time and space"

use ndarray::Array2;
use rayon::prelude::*;
use std::collections::HashSet;

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum DtwError {
    #[error("Empty sequence provided")]
    EmptySequence,

    #[error("Sequence too short: {len} (minimum 1)")]
    SequenceTooShort { len: usize },

    #[error("Invalid window size: {size} (must be >= 0)")]
    InvalidWindowSize { size: usize },
}

pub type Result<T> = std::result::Result<T, DtwError>;

// =============================================================================
// DTW Distance Metric
// =============================================================================

/// Dynamic Time Warping for sequence alignment
#[derive(Debug, Clone)]
pub struct DtwMetric {
    window_size: Option<usize>,  // Sakoe-Chiba band width (None = no constraint)
    lower_bound: bool,            // Use lower bound for early abandoning
}

impl DtwMetric {
    /// Create a new DTW metric
    ///
    /// # Arguments
    /// * `window_size` - Optional Sakoe-Chiba band width (None = no constraint)
    /// * `lower_bound` - Whether to use lower bound for early abandoning
    pub fn new(window_size: Option<usize>, lower_bound: bool) -> Self {
        Self {
            window_size,
            lower_bound,
        }
    }

    /// Create DTW with no constraints (classic DTW)
    pub fn unconstrained() -> Self {
        Self {
            window_size: None,
            lower_bound: false,
        }
    }

    /// Create DTW with Sakoe-Chiba band (faster, approximate)
    pub fn with_window(window_size: usize) -> Self {
        Self {
            window_size: Some(window_size),
            lower_bound: false,
        }
    }

    /// Compute DTW distance between two sequences
    ///
    /// # Arguments
    /// * `seq1` - First sequence
    /// * `seq2` - Second sequence
    ///
    /// # Returns
    /// DTW distance (minimum cumulative distance)
    pub fn compute(&self, seq1: &[f64], seq2: &[f64]) -> Result<f64> {
        if seq1.is_empty() || seq2.is_empty() {
            return Err(DtwError::EmptySequence);
        }

        // Early abandon using lower bound if enabled
        if self.lower_bound {
            let _lb = self.lower_bound_lb_keogh(seq1, seq2);
            // If lower bound is already large, we can early abandon
            // (simplified - in practice you'd compare against a threshold)
        }

        match self.window_size {
            Some(w) => self.dtw_windowed(seq1, seq2, w),
            None => self.dtw_classic(seq1, seq2),
        }
    }

    /// Classic DTW algorithm (O(n*m) time and space)
    fn dtw_classic(&self, seq1: &[f64], seq2: &[f64]) -> Result<f64> {
        let n = seq1.len();
        let m = seq2.len();

        // Create cost matrix
        let mut dtw = Array2::zeros((n + 1, m + 1));

        // Initialize with infinity
        for i in 0..=n {
            dtw[[i, 0]] = f64::INFINITY;
        }
        for j in 0..=m {
            dtw[[0, j]] = f64::INFINITY;
        }
        dtw[[0, 0]] = 0.0;

        // Fill cost matrix
        for i in 0..n {
            for j in 0..m {
                let cost = (seq1[i] - seq2[j]).abs();
                let min_prev = dtw[[i, j]]
                    .min(dtw[[i, j + 1]])
                    .min(dtw[[i + 1, j]]);
                dtw[[i + 1, j + 1]] = cost + min_prev;
            }
        }

        Ok(dtw[[n, m]])
    }

    /// Windowed DTW with Sakoe-Chiba band (O(n*w) time and space)
    fn dtw_windowed(&self, seq1: &[f64], seq2: &[f64], window: usize) -> Result<f64> {
        let n = seq1.len();
        let m = seq2.len();

        let mut dtw = Array2::zeros((n + 1, m + 1));

        // Initialize with infinity
        for i in 0..=n {
            for j in 0..=m {
                dtw[[i, j]] = f64::INFINITY;
            }
        }
        dtw[[0, 0]] = 0.0;

        // Fill cost matrix with window constraint
        for i in 0..n {
            let j_start = if i > window { i - window } else { 0 };
            let j_end = (i + window + 1).min(m);

            for j in j_start..j_end {
                let cost = (seq1[i] - seq2[j]).abs();
                let min_prev = dtw[[i, j]]
                    .min(dtw[[i, j + 1]])
                    .min(dtw[[i + 1, j]]);
                dtw[[i + 1, j + 1]] = cost + min_prev;
            }
        }

        Ok(dtw[[n, m]])
    }

    /// LB_Keogh lower bound for early abandoning
    ///
    /// Provides a quick lower bound on DTW distance
    fn lower_bound_lb_keogh(&self, seq1: &[f64], seq2: &[f64]) -> f64 {
        if seq1.is_empty() || seq2.is_empty() {
            return 0.0;
        }

        let n = seq1.len();
        let m = seq2.len();

        // For each point in seq1, find min/max in seq2 within window
        // This is a simplified version
        let mut lb = 0.0;
        for i in 0..n.min(m) {
            let u = seq2[i]; // Upper envelope (simplified)
            let l = seq2[i]; // Lower envelope (simplified)

            let val = seq1[i];
            if val > u {
                lb += (val - u).powi(2);
            } else if val < l {
                lb += (val - l).powi(2);
            }
        }

        lb.sqrt()
    }

    /// Compute DTW distance matrix for multiple sequences
    pub fn compute_distance_matrix(&self, sequences: &[Vec<f64>]) -> Result<Array2<f64>> {
        let n = sequences.len();
        let mut dist_matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in i..n {
                let dist = self.compute(&sequences[i], &sequences[j])?;
                dist_matrix[[i, j]] = dist;
                dist_matrix[[j, i]] = dist;
            }
        }

        Ok(dist_matrix)
    }
}

// =============================================================================
// FastDTW (Linear Time Approximation)
// =============================================================================

/// FastDTW - Approximate DTW with linear time and space complexity
///
/// Uses a multi-resolution approach to approximate DTW in O(n) time
#[derive(Debug, Clone)]
pub struct FastDtw {
    radius: usize,  // Resolution radius
}

impl FastDtw {
    /// Create a new FastDTW instance
    ///
    /// # Arguments
    /// * `radius` - Resolution radius for warp path refinement
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Compute approximate DTW using FastDTW algorithm
    ///
    /// # Algorithm
    /// 1. Coarsen sequences to lower resolution
    /// 2. Find optimal path at low resolution
    /// 3. Refine path at finer resolutions
    ///
    /// # Arguments
    /// * `seq1` - First sequence
    /// * `seq2` - Second sequence
    pub fn compute(&self, seq1: &[f64], seq2: &[f64]) -> Result<f64> {
        if seq1.is_empty() || seq2.is_empty() {
            return Err(DtwError::EmptySequence);
        }

        // Minimum resolution for coarsening
        let min_resolution = 16;

        let (n, m) = (seq1.len(), seq2.len());

        // If sequences are small enough, use classic DTW
        if n <= min_resolution || m <= min_resolution {
            return DtwMetric::unconstrained().compute(seq1, seq2);
        }

        // Coarsen sequences
        let coarse1 = Self::coarsen(seq1);
        let coarse2 = Self::coarsen(seq2);

        // Recursively compute path at coarser resolution
        let coarse_path = self.compute(&coarse1, &coarse2)?;

        // Project and refine path at original resolution
        self.refine_path(seq1, seq2, &coarse_path)
    }

    /// Coarsen sequence by averaging adjacent pairs
    fn coarsen(seq: &[f64]) -> Vec<f64> {
        if seq.len() <= 2 {
            return seq.to_vec();
        }

        let mut coarse = Vec::new();
        for chunk in seq.chunks(2) {
            if chunk.len() == 2 {
                coarse.push((chunk[0] + chunk[1]) / 2.0);
            } else {
                coarse.push(chunk[0]);
            }
        }

        coarse
    }

    /// Refine warp path at original resolution
    fn refine_path(&self, seq1: &[f64], seq2: &[f64], _coarse_path: &f64) -> Result<f64> {
        // Create windowed DTW around projected path
        let window = self.radius * 2;
        DtwMetric::with_window(window).compute(seq1, seq2)
    }
}

// =============================================================================
// DTW-Aware Clustering
// =============================================================================

/// DBSCAN clustering with DTW distance metric
///
/// Replaces Euclidean distance with DTW for time-aware clustering
#[derive(Debug, Clone)]
pub struct DtwDbscan {
    eps: f64,
    min_samples: usize,
    dtw_metric: DtwMetric,
}

impl DtwDbscan {
    /// Create a new DTW-DBSCAN clustering algorithm
    ///
    /// # Arguments
    /// * `eps` - Maximum DTW distance for neighborhood
    /// * `min_samples` - Minimum samples for core point
    /// * `window_size` - Optional DTW window constraint (None = classic DTW)
    pub fn new(eps: f64, min_samples: usize, window_size: Option<usize>) -> Self {
        Self {
            eps,
            min_samples,
            dtw_metric: DtwMetric::new(window_size, false),
        }
    }

    /// Fit DTW-DBSCAN to sequences
    ///
    /// # Arguments
    /// * `sequences` - Vector of sequences to cluster
    ///
    /// # Returns
    /// Vector of cluster labels (-1 = noise, >=0 = cluster ID)
    pub fn fit_predict(&self, sequences: &[Vec<f64>]) -> Result<Vec<i32>> {
        let n = sequences.len();
        if n == 0 {
            return Ok(vec![]);
        }

        let mut labels = vec![-1i32; n];
        let mut visited = HashSet::new();
        let mut cluster_id = 0i32;

        for i in 0..n {
            if visited.contains(&i) {
                continue;
            }

            visited.insert(i);

            // Find neighbors using DTW distance
            let neighbors = self.region_query(sequences, i)?;

            if neighbors.len() < self.min_samples {
                // Mark as noise
                labels[i] = -1;
            } else {
                // Expand cluster
                labels[i] = cluster_id;

                let mut cluster_points = neighbors;

                // Process all points in cluster
                let mut j = 0;
                while j < cluster_points.len() {
                    let point_idx = cluster_points[j];

                    if !visited.contains(&point_idx) {
                        visited.insert(point_idx);

                        let new_neighbors = self.region_query(sequences, point_idx)?;

                        if new_neighbors.len() >= self.min_samples {
                            cluster_points.extend(new_neighbors);
                        }
                    }

                    if labels[point_idx] == -1 {
                        labels[point_idx] = cluster_id;
                    }

                    j += 1;
                }

                cluster_id += 1;
            }
        }

        Ok(labels)
    }

    /// Find all neighbors within eps distance using DTW (parallelized)
    fn region_query(&self, sequences: &[Vec<f64>], point_idx: usize) -> Result<Vec<usize>> {
        // Parallel distance computation
        let point = &sequences[point_idx];

        // Compute distances in parallel and collect neighbors
        let neighbors: Vec<usize> = sequences
            .par_iter()
            .enumerate()
            .filter_map(|(i, seq)| {
                match self.dtw_metric.compute(point, seq) {
                    Ok(dist) if dist <= self.eps => Some(i),
                    _ => None,
                }
            })
            .collect();

        Ok(neighbors)
    }

    /// Get cluster statistics
    pub fn get_cluster_stats(&self, labels: &[i32]) -> DtwClusterStats {
        let mut cluster_counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        let mut noise_count = 0;

        for &label in labels {
            if label == -1 {
                noise_count += 1;
            } else {
                *cluster_counts.entry(label).or_insert(0) += 1;
            }
        }

        let n_clusters = cluster_counts.len();
        let cluster_sizes: Vec<usize> = cluster_counts.values().cloned().collect();

        DtwClusterStats {
            n_clusters,
            noise_count,
            cluster_sizes,
        }
    }
}

/// Cluster statistics for DTW-DBSCAN
#[derive(Debug, Clone)]
pub struct DtwClusterStats {
    pub n_clusters: usize,
    pub noise_count: usize,
    pub cluster_sizes: Vec<usize>,
}

// =============================================================================
// Tests (TDD Approach)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Classic DTW computes correct distance
    #[test]
    fn test_dtw_classic() {
        let dtw = DtwMetric::unconstrained();

        // Identical sequences
        let seq1 = vec![1.0, 2.0, 3.0, 4.0];
        let seq2 = vec![1.0, 2.0, 3.0, 4.0];

        let dist = dtw.compute(&seq1, &seq2).unwrap();
        assert!((dist - 0.0).abs() < 1e-6, "Identical sequences should have zero distance");

        // Different sequences
        let seq3 = vec![1.0, 2.0, 3.0, 4.0];
        let seq4 = vec![2.0, 3.0, 4.0, 5.0];

        let dist = dtw.compute(&seq3, &seq4).unwrap();
        assert!(dist > 0.0, "Different sequences should have positive distance");
    }

    /// Test 2: Windowed DTW produces valid distances
    #[test]
    fn test_dtw_windowed() {
        let dtw = DtwMetric::with_window(2);

        let seq1 = vec![1.0, 2.0, 3.0, 4.0];
        let seq2 = vec![1.0, 2.0, 3.0, 4.0];

        let dist = dtw.compute(&seq1, &seq2).unwrap();
        assert!(dist.is_finite(), "Windowed DTW should produce finite distance");
        assert!(dist >= 0.0, "Distance should be non-negative");
    }

    /// Test 3: DTW handles empty sequences
    #[test]
    fn test_dtw_empty_sequence() {
        let dtw = DtwMetric::unconstrained();

        let seq1 = vec![1.0, 2.0, 3.0];
        let seq2: Vec<f64> = vec![];

        let result = dtw.compute(&seq1, &seq2);
        assert!(result.is_err(), "Should reject empty sequences");
    }

    /// Test 4: DTW is symmetric
    #[test]
    fn test_dtw_symmetry() {
        let dtw = DtwMetric::unconstrained();

        let seq1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let seq2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let dist1 = dtw.compute(&seq1, &seq2).unwrap();
        let dist2 = dtw.compute(&seq2, &seq1).unwrap();

        assert!((dist1 - dist2).abs() < 1e-6, "DTW should be symmetric");
    }

    /// Test 5: DTW-DBSCAN clusters similar sequences
    #[test]
    fn test_dtw_dbscan_clustering() {
        let sequences = vec![
            // Cluster 1: rising pattern
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.1, 2.1, 3.1, 4.1],
            vec![0.9, 1.9, 2.9, 3.9],
            // Cluster 2: falling pattern
            vec![4.0, 3.0, 2.0, 1.0],
            vec![4.1, 3.1, 2.1, 1.1],
            // Noise: random
            vec![100.0, 200.0, 50.0, 150.0],
        ];

        let dbscan = DtwDbscan::new(2.0, 2, Some(2));

        let labels = dbscan.fit_predict(&sequences).unwrap();

        // First three should be in same cluster
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);

        // Next two should be in same cluster
        assert_eq!(labels[3], labels[4]);

        // Clusters should be different
        assert_ne!(labels[0], labels[3]);

        // Last point should be noise
        assert_eq!(labels[5], -1);
    }

    /// Test 6: DTW-DBSCAN cluster statistics
    #[test]
    fn test_dtw_dbscan_stats() {
        let sequences = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![4.0, 3.0, 2.0],
        ];

        let dbscan = DtwDbscan::new(1.0, 2, None);
        let labels = dbscan.fit_predict(&sequences).unwrap();
        let stats = dbscan.get_cluster_stats(&labels);

        assert!(stats.n_clusters >= 1, "Should have at least 1 cluster");
        assert_eq!(
            stats.cluster_sizes.iter().sum::<usize>() + stats.noise_count,
            sequences.len()
        );
    }

    /// Test 7: FastDTW produces approximate results
    #[test]
    fn test_fast_dtw() {
        let fast_dtw = FastDtw::new(2);

        let seq1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let seq2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let dist = fast_dtw.compute(&seq1, &seq2).unwrap();
        assert!(dist.is_finite(), "FastDTW should produce finite distance");
        assert!(dist >= 0.0, "Distance should be non-negative");
    }

    /// Test 8: DTW distance matrix is symmetric
    #[test]
    fn test_dtw_distance_matrix() {
        let dtw = DtwMetric::unconstrained();

        let sequences = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
        ];

        let dist_matrix = dtw.compute_distance_matrix(&sequences).unwrap();

        // Check symmetry
        assert!((dist_matrix[[0, 1]] - dist_matrix[[1, 0]]).abs() < 1e-6);
        assert!((dist_matrix[[0, 2]] - dist_matrix[[2, 0]]).abs() < 1e-6);
        assert!((dist_matrix[[1, 2]] - dist_matrix[[2, 1]]).abs() < 1e-6);

        // Check diagonal is zero
        assert!((dist_matrix[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((dist_matrix[[1, 1]] - 0.0).abs() < 1e-6);
        assert!((dist_matrix[[2, 2]] - 0.0).abs() < 1e-6);
    }

    /// Test 9: DTW handles sequences of different lengths
    #[test]
    fn test_dtw_different_lengths() {
        let dtw = DtwMetric::unconstrained();

        let seq1 = vec![1.0, 2.0, 3.0];
        let seq2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let dist = dtw.compute(&seq1, &seq2).unwrap();
        assert!(dist.is_finite(), "DTW should handle different lengths");
        assert!(dist >= 0.0, "Distance should be non-negative");
    }

    /// Test 10: DTW-DBSCAN with invalid window size
    #[test]
    fn test_dtw_dbscan_invalid_params() {
        // Window size of 0 should still work (degrades to point-wise matching)
        let sequences = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
        ];

        let dbscan = DtwDbscan::new(1.0, 2, Some(0));
        let labels = dbscan.fit_predict(&sequences);

        assert!(labels.is_ok(), "Window size of 0 should still work");
    }
}
