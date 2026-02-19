// Clustering Module
//
// Implements DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
// algorithm for clustering phrase candidates based on their 30D feature vectors.
//
// Reference: Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996).
// "A density-based algorithm for discovering clusters in large spatial databases with noise"

use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand::Rng;
use std::collections::VecDeque;

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum ClusteringError {
    #[error("Insufficient data: need at least {min} samples, got {actual}")]
    InsufficientData { min: usize, actual: usize },

    #[error("Invalid epsilon: {eps} (must be > 0)")]
    InvalidEpsilon { eps: f64 },

    #[error("Invalid min_samples: {min_samples} (must be > 0)")]
    InvalidMinSamples { min_samples: usize },

    #[error("Feature dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Feature extraction failed: {0}")]
    FeatureExtractionFailed(String),
}

pub type Result<T> = std::result::Result<T, ClusteringError>;

// =============================================================================
// DBSCAN Clustering
// =============================================================================

/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
///
/// Clusters points based on density. Points in low-density regions are marked as noise.
#[derive(Debug, Clone)]
pub struct DbscanClustering {
    eps: f64,
    min_samples: usize,
}

impl DbscanClustering {
    /// Create a new DBSCAN clustering algorithm
    ///
    /// # Arguments
    /// * `eps` - Maximum distance between two samples for one to be considered as in the neighborhood of the other
    /// * `min_samples` - Minimum number of samples in a neighborhood for a point to be considered as a core point
    pub fn new(eps: f64, min_samples: usize) -> Result<Self> {
        if eps <= 0.0 {
            return Err(ClusteringError::InvalidEpsilon { eps });
        }
        if min_samples == 0 {
            return Err(ClusteringError::InvalidMinSamples { min_samples });
        }

        Ok(Self { eps, min_samples })
    }

    /// Fit DBSCAN clustering to feature matrix
    ///
    /// # Arguments
    /// * `features` - Feature matrix (samples x dimensions)
    ///
    /// # Returns
    /// Vector of cluster labels (-1 = noise, >=0 = cluster ID)
    pub fn fit_predict(&self, features: &Array2<f64>) -> Result<Vec<i32>> {
        let n_samples = features.nrows();

        if n_samples < self.min_samples {
            return Err(ClusteringError::InsufficientData {
                min: self.min_samples,
                actual: n_samples,
            });
        }

        let mut labels = vec![-1i32; n_samples]; // -1 = unvisited/noise
        let mut cluster_id = 0i32;

        for i in 0..n_samples {
            if labels[i] != -1 {
                continue; // Already visited
            }

            // Find neighbors
            let neighbors = self.find_neighbors(features, i);

            if neighbors.len() < self.min_samples {
                labels[i] = -1; // Mark as noise
                continue;
            }

            // Start a new cluster
            labels[i] = cluster_id;
            let mut queue = VecDeque::new();

            // Add neighbors to queue (excluding current point)
            for &neighbor in &neighbors {
                if neighbor != i && labels[neighbor] == -1 {
                    queue.push_back(neighbor);
                    labels[neighbor] = cluster_id;
                }
            }

            // Expand cluster
            while let Some(current) = queue.pop_front() {
                let current_neighbors = self.find_neighbors(features, current);

                if current_neighbors.len() >= self.min_samples {
                    // Current is a core point, expand its neighbors
                    for &neighbor in &current_neighbors {
                        if labels[neighbor] == -1 {
                            // Unvisited point
                            queue.push_back(neighbor);
                            labels[neighbor] = cluster_id;
                        }
                    }
                }
            }

            cluster_id += 1;
        }

        Ok(labels)
    }

    /// Find neighbors of a point within eps distance
    fn find_neighbors(&self, features: &Array2<f64>, point_idx: usize) -> Vec<usize> {
        let point = features.row(point_idx);
        let mut neighbors = Vec::new();
        let eps_squared = self.eps * self.eps;

        for (i, row) in features.rows().into_iter().enumerate() {
            let dist_sq = self.euclidean_distance_squared(&point.to_vec(), &row.to_vec());
            if dist_sq <= eps_squared {
                neighbors.push(i);
            }
        }

        neighbors
    }

    /// Compute squared Euclidean distance between two points (as slices)
    #[inline]
    fn euclidean_distance_squared(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Get cluster statistics
    pub fn get_cluster_stats(&self, labels: &[i32]) -> ClusterStats {
        let mut cluster_counts: std::collections::HashMap<i32, usize> =
            std::collections::HashMap::new();
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

        ClusterStats {
            n_clusters,
            noise_count,
            cluster_sizes,
        }
    }
}

// =============================================================================
// Cluster Statistics
// =============================================================================

/// Statistics about clustering results
#[derive(Debug, Clone)]
pub struct ClusterStats {
    /// Number of clusters found
    pub n_clusters: usize,
    /// Number of noise points
    pub noise_count: usize,
    /// Size of each cluster
    pub cluster_sizes: Vec<usize>,
}

// =============================================================================
// Standard Scaler (for feature normalization)
// =============================================================================

/// StandardScaler for normalizing features to zero mean and unit variance
#[derive(Debug, Clone)]
pub struct StandardScaler {
    means: Option<Array1<f64>>,
    scales: Option<Array1<f64>>,
}

impl StandardScaler {
    /// Create a new StandardScaler
    pub fn new() -> Self {
        Self {
            means: None,
            scales: None,
        }
    }

    /// Fit the scaler to feature matrix
    pub fn fit(&mut self, features: &Array2<f64>) -> Result<()> {
        let n_features = features.ncols();

        // Compute mean for each feature
        let means = features.mean_axis(Axis(0)).ok_or_else(|| {
            ClusteringError::FeatureExtractionFailed("Failed to compute mean".to_string())
        })?;

        // Compute standard deviation for each feature
        let mut scales = Array1::zeros(n_features);
        for i in 0..n_features {
            let col = features.column(i);
            let mean = means[i];
            let variance =
                col.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / col.len() as f64;
            scales[i] = variance.sqrt();
        }

        // Avoid division by zero
        for i in 0..scales.len() {
            if scales[i] < 1e-10 {
                scales[i] = 1.0;
            }
        }

        self.means = Some(means);
        self.scales = Some(scales);

        Ok(())
    }

    /// Transform features using fitted scaler
    pub fn transform(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let means = self.means.as_ref().ok_or_else(|| {
            ClusteringError::FeatureExtractionFailed("Scaler not fitted".to_string())
        })?;
        let scales = self.scales.as_ref().ok_or_else(|| {
            ClusteringError::FeatureExtractionFailed("Scaler not fitted".to_string())
        })?;

        let mut normalized = features.clone();
        for mut row in normalized.rows_mut() {
            for i in 0..row.len() {
                row[i] = (row[i] - means[i]) / scales[i];
            }
        }

        Ok(normalized)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, features: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(features)?;
        self.transform(features)
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Custom Error for FeatureExtractionFailed
// =============================================================================

impl ClusteringError {
    pub fn feature_extraction_failed(msg: String) -> Self {
        ClusteringError::FeatureExtractionFailed(msg)
    }
}

// =============================================================================
// MiniBatch K-Means Clustering
// =============================================================================

/// MiniBatch K-Means clustering algorithm
///
/// Faster variant of K-Means that uses small random batches to update cluster centers.
/// Scales linearly with dataset size, making it suitable for large datasets.
///
/// Reference: Sculley, D. (2010). "Web-scale k-means clustering"
#[derive(Debug, Clone)]
pub struct MiniBatchKMeans {
    n_clusters: usize,
    batch_size: usize,
    max_iter: usize,
    tol: f64,
    random_state: Option<u64>,
}

impl MiniBatchKMeans {
    /// Create a new MiniBatch K-Means clustering algorithm
    ///
    /// # Arguments
    /// * `n_clusters` - Number of clusters to form
    /// * `batch_size` - Size of mini-batches (default: 100)
    /// * `max_iter` - Maximum number of iterations (default: 100)
    /// * `tol` - Relative tolerance for convergence (default: 1e-4)
    /// * `random_state` - Random seed for reproducibility (optional)
    pub fn new(
        n_clusters: usize,
        batch_size: usize,
        max_iter: usize,
        tol: f64,
        random_state: Option<u64>,
    ) -> Result<Self> {
        if n_clusters == 0 {
            return Err(ClusteringError::InvalidMinSamples { min_samples: 0 });
        }
        if batch_size == 0 {
            return Err(ClusteringError::InvalidMinSamples { min_samples: 0 });
        }

        Ok(Self {
            n_clusters,
            batch_size,
            max_iter,
            tol,
            random_state,
        })
    }

    /// Fit MiniBatch K-Means to feature matrix
    ///
    /// # Arguments
    /// * `features` - Feature matrix (samples x dimensions)
    ///
    /// # Returns
    /// Vector of cluster labels (0 to n_clusters-1)
    pub fn fit_predict(&self, features: &Array2<f64>) -> Result<Vec<i32>> {
        let n_samples = features.nrows();
        let n_dims = features.ncols();

        if n_samples < self.n_clusters {
            return Err(ClusteringError::InsufficientData {
                min: self.n_clusters,
                actual: n_samples,
            });
        }

        // Initialize cluster centers using k-means++ style
        let mut centers = self.initialize_centers(features);

        let mut inertia_prev = f64::INFINITY;

        // Mini-batch iterations
        for iter in 0..self.max_iter {
            // Create random batch indices
            let batch_indices = self.get_batch_indices(n_samples, iter);

            // Update centers using mini-batch
            let mut counts = vec![0usize; self.n_clusters];
            let mut updates = vec![vec![0.0f64; n_dims]; self.n_clusters];

            for &idx in &batch_indices {
                // Find nearest center
                let (nearest_cluster, _) = self.find_nearest_center(features.row(idx), &centers);

                // Accumulate updates
                for d in 0..n_dims {
                    updates[nearest_cluster][d] += features[[idx, d]];
                }
                counts[nearest_cluster] += 1;
            }

            // Apply updates with learning rate
            let learning_rate = 1.0 / (iter as f64 + 1.0);

            for k in 0..self.n_clusters {
                if counts[k] > 0 {
                    for d in 0..n_dims {
                        let update = updates[k][d] / counts[k] as f64;
                        centers[k][d] =
                            (1.0 - learning_rate) * centers[k][d] + learning_rate * update;
                    }
                }
            }

            // Check for convergence (every 10 iterations)
            if iter % 10 == 0 {
                let inertia = self.compute_inertia(features, &centers);
                let improvement = (inertia_prev - inertia) / inertia_prev.abs().max(1e-10);

                if improvement < self.tol {
                    break;
                }
                inertia_prev = inertia;
            }
        }

        // Assign final labels
        let mut labels = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let (cluster, _) = self.find_nearest_center(features.row(i), &centers);
            labels.push(cluster as i32);
        }

        Ok(labels)
    }

    /// Initialize cluster centers using random sampling
    fn initialize_centers(&self, features: &Array2<f64>) -> Vec<Vec<f64>> {
        let n_dims = features.ncols();
        let mut centers = Vec::with_capacity(self.n_clusters);

        // Use random seed if provided
        let mut rng = if let Some(seed) = self.random_state {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        // Randomly select initial centers from data
        let n_samples = features.nrows();
        let mut used_indices = std::collections::HashSet::new();

        for _ in 0..self.n_clusters {
            let mut idx;
            loop {
                idx = rng.gen_range(0..n_samples);
                if used_indices.insert(idx) {
                    break;
                }
            }

            let mut center = Vec::with_capacity(n_dims);
            for d in 0..n_dims {
                center.push(features[[idx, d]]);
            }
            centers.push(center);
        }

        centers
    }

    /// Get random batch indices
    fn get_batch_indices(&self, n_samples: usize, iter: usize) -> Vec<usize> {
        let mut rng = if let Some(seed) = self.random_state {
            rand::rngs::StdRng::seed_from_u64(seed + iter as u64)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let mut indices = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            indices.push(rng.gen_range(0..n_samples));
        }
        indices
    }

    /// Find nearest center for a sample
    fn find_nearest_center(
        &self,
        sample: ndarray::ArrayView1<f64>,
        centers: &[Vec<f64>],
    ) -> (usize, f64) {
        let mut nearest = 0;
        let mut min_dist = f64::MAX;

        for (k, center) in centers.iter().enumerate() {
            let mut dist = 0.0;
            for (_d, (&val, &center_val)) in sample.iter().zip(center.iter()).enumerate() {
                let diff = val - center_val;
                dist += diff * diff;
            }

            if dist < min_dist {
                min_dist = dist;
                nearest = k;
            }
        }

        (nearest, min_dist.sqrt())
    }

    /// Compute inertia (sum of squared distances to nearest center)
    fn compute_inertia(&self, features: &Array2<f64>, centers: &[Vec<f64>]) -> f64 {
        let mut inertia = 0.0;

        for i in 0..features.nrows() {
            let (_, dist) = self.find_nearest_center(features.row(i), centers);
            inertia += dist * dist;
        }

        inertia
    }

    /// Get cluster statistics
    pub fn get_cluster_stats(&self, labels: &[i32]) -> ClusterStats {
        let mut cluster_counts: std::collections::HashMap<i32, usize> =
            std::collections::HashMap::new();

        for &label in labels {
            *cluster_counts.entry(label).or_insert(0) += 1;
        }

        let n_clusters = cluster_counts.len();
        let cluster_sizes: Vec<usize> = cluster_counts.values().cloned().collect();

        // MiniBatch K-Means has no noise, so noise_count = 0
        ClusterStats {
            n_clusters,
            noise_count: 0,
            cluster_sizes,
        }
    }
}

// =============================================================================
// Tests (TDD Approach)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use std::collections::HashSet;

    /// Create well-separated clusters
    fn create_separated_clusters() -> Array2<f64> {
        // Create two well-separated 2D clusters
        let mut data = Vec::with_capacity(20);

        // Cluster 1: centered at (0, 0)
        for _ in 0..10 {
            let mut rng = rand::thread_rng();
            let x: f64 = rand::Rng::gen_range(&mut rng, -1.0..1.0);
            let y: f64 = rand::Rng::gen_range(&mut rng, -1.0..1.0);
            data.push([x, y]);
        }

        // Cluster 2: centered at (10, 10)
        for _ in 0..10 {
            let mut rng = rand::thread_rng();
            let x: f64 = 10.0 + rand::Rng::gen_range(&mut rng, -1.0..1.0);
            let y: f64 = 10.0 + rand::Rng::gen_range(&mut rng, -1.0..1.0);
            data.push([x, y]);
        }

        // Convert to Array2
        let mut array = Array2::zeros((20, 2));
        for (i, row) in data.iter().enumerate() {
            array[[i, 0]] = row[0];
            array[[i, 1]] = row[1];
        }
        array
    }

    /// Create single cluster
    fn create_single_cluster(n: usize) -> Array2<f64> {
        let mut array = Array2::zeros((n, 2));

        for i in 0..n {
            let mut rng = rand::thread_rng();
            let x: f64 = rand::Rng::gen_range(&mut rng, -1.0..1.0);
            let y: f64 = rand::Rng::gen_range(&mut rng, -1.0..1.0);
            array[[i, 0]] = x;
            array[[i, 1]] = y;
        }
        array
    }

    // ===== Test 1: Constructor validation =====
    #[test]
    fn test_dbscan_new_valid_parameters() {
        let dbscan = DbscanClustering::new(1.0, 5);
        assert!(dbscan.is_ok());
    }

    #[test]
    fn test_dbscan_new_invalid_epsilon_zero() {
        let dbscan = DbscanClustering::new(0.0, 5);
        assert!(dbscan.is_err());
    }

    #[test]
    fn test_dbscan_new_invalid_epsilon_negative() {
        let dbscan = DbscanClustering::new(-1.0, 5);
        assert!(dbscan.is_err());
    }

    #[test]
    fn test_dbscan_new_invalid_min_samples_zero() {
        let dbscan = DbscanClustering::new(1.0, 0);
        assert!(dbscan.is_err());
    }

    // ===== Test 2: Single cluster detection =====
    #[test]
    fn test_dbscan_single_cluster() {
        let dbscan = DbscanClustering::new(2.0, 3).unwrap();
        let features = create_single_cluster(20);

        let labels = dbscan.fit_predict(&features).unwrap();
        let stats = dbscan.get_cluster_stats(&labels);

        // Should find one cluster
        assert_eq!(stats.n_clusters, 1);
        // All points should be in the cluster (no noise)
        assert_eq!(stats.noise_count, 0);
        assert_eq!(stats.cluster_sizes[0], 20);
    }

    // ===== Test 3: Separated clusters =====
    #[test]
    fn test_dbscan_separated_clusters() {
        let dbscan = DbscanClustering::new(2.0, 3).unwrap();
        let features = create_separated_clusters();

        let labels = dbscan.fit_predict(&features).unwrap();
        let stats = dbscan.get_cluster_stats(&labels);

        // Should find two clusters
        assert_eq!(stats.n_clusters, 2);
        // Minimal noise
        assert!(stats.noise_count < 5);
    }

    // ===== Test 4: Insufficient data =====
    #[test]
    fn test_dbscan_insufficient_data() {
        let dbscan = DbscanClustering::new(1.0, 10).unwrap();
        let features = create_single_cluster(5);

        let result = dbscan.fit_predict(&features);
        assert!(result.is_err());
    }

    // ===== Test 5: Noise detection =====
    #[test]
    fn test_dbscan_noise_detection() {
        let dbscan = DbscanClustering::new(1.0, 5).unwrap();

        // Create scattered points
        let features = Array2::from_shape_fn((20, 2), |(i, j)| {
            if i < 5 {
                0.0 // Cluster 1
            } else if i < 10 {
                10.0 // Cluster 2
            } else {
                (i * j) as f64 // Scattered noise points
            }
        });

        let labels = dbscan.fit_predict(&features).unwrap();
        let stats = dbscan.get_cluster_stats(&labels);

        // Should detect some noise
        assert!(stats.noise_count > 0);
    }

    // ===== Test 6: Epsilon affects clustering =====
    #[test]
    fn test_epsilon_affects_clustering() {
        let features = create_separated_clusters();

        // Small epsilon - more clusters
        let dbscan_small = DbscanClustering::new(0.5, 3).unwrap();
        let labels_small = dbscan_small.fit_predict(&features).unwrap();
        let stats_small = dbscan_small.get_cluster_stats(&labels_small);

        // Large epsilon - fewer clusters
        let dbscan_large = DbscanClustering::new(20.0, 3).unwrap();
        let labels_large = dbscan_large.fit_predict(&features).unwrap();
        let stats_large = dbscan_large.get_cluster_stats(&labels_large);

        // Small epsilon should find more clusters
        assert!(stats_small.n_clusters >= stats_large.n_clusters);
    }

    // ===== Test 7: Euclidean distance =====
    #[test]
    fn test_euclidean_distance_squared() {
        let dbscan = DbscanClustering::new(1.0, 3).unwrap();

        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        let dist_sq = dbscan.euclidean_distance_squared(&a, &b);
        // Distance should be 5 (3-4-5 triangle)
        assert!((dist_sq.sqrt() - 5.0).abs() < 1e-10);
    }

    // ===== Test 8: Standard Scaler =====
    #[test]
    fn test_standard_scaler_fit_transform() {
        let mut scaler = StandardScaler::new();

        // Create features with different scales
        let features = arr2(&[
            [1.0, 100.0],
            [2.0, 200.0],
            [3.0, 300.0],
            [4.0, 400.0],
            [5.0, 500.0],
        ]);

        let normalized = scaler.fit_transform(&features).unwrap();

        // Check that normalized features have approximately zero mean
        for col in normalized.columns() {
            let mean = col.iter().sum::<f64>() / col.len() as f64;
            assert!(mean.abs() < 1e-10);
        }
    }

    // ===== Test 9: Cluster statistics =====
    #[test]
    fn test_cluster_stats() {
        let dbscan = DbscanClustering::new(1.0, 3).unwrap();
        let features = create_separated_clusters();

        let labels = dbscan.fit_predict(&features).unwrap();
        let stats = dbscan.get_cluster_stats(&labels);

        // Check that total points match
        let total: usize = stats.cluster_sizes.iter().copied().sum::<usize>() + stats.noise_count;
        assert_eq!(total, 20); // 10 + 10 points
    }

    // ===== Test 10: All noise =====
    #[test]
    fn test_all_noise() {
        let dbscan = DbscanClustering::new(0.1, 10).unwrap();

        // Very scattered points
        let features = Array2::from_shape_fn((20, 2), |(i, j)| (i * j) as f64);

        let labels = dbscan.fit_predict(&features).unwrap();
        let stats = dbscan.get_cluster_stats(&labels);

        // With small epsilon and high min_samples, most points should be noise
        assert!(stats.noise_count >= 15);
    }

    // ===== Test 11: Labels are contiguous =====
    #[test]
    fn test_labels_are_contiguous() {
        let dbscan = DbscanClustering::new(2.0, 3).unwrap();
        let features = create_separated_clusters();

        let labels = dbscan.fit_predict(&features).unwrap();

        // Find all unique labels (excluding noise)
        let unique_labels: HashSet<i32> = labels.iter().filter(|&&l| l >= 0).cloned().collect();

        // Check that labels are contiguous starting from 0
        for &label in &unique_labels {
            assert!(label >= 0);
            assert!(label < unique_labels.len() as i32);
        }
    }

    // ===== Test 12: Deterministic results =====
    #[test]
    fn test_deterministic_results() {
        let dbscan = DbscanClustering::new(2.0, 3).unwrap();
        let features = create_single_cluster(20);

        let labels1 = dbscan.fit_predict(&features).unwrap();
        let labels2 = dbscan.fit_predict(&features).unwrap();

        // Results should be identical
        assert_eq!(labels1, labels2);
    }

    // ===== MiniBatch K-Means Tests =====

    /// Test 13: MiniBatch K-Means basic functionality
    #[test]
    fn test_minibatch_kmeans_basic() {
        let kmeans = MiniBatchKMeans::new(2, 10, 50, 1e-4, Some(42)).unwrap();

        // Create two well-separated clusters
        let mut data = Vec::new();
        for i in 0..10 {
            data.push([i as f64, i as f64]);
        }
        for i in 0..10 {
            data.push([100.0 + i as f64, 100.0 + i as f64]);
        }

        let mut features = Array2::zeros((20, 2));
        for (i, row) in data.iter().enumerate() {
            features[[i, 0]] = row[0];
            features[[i, 1]] = row[1];
        }

        let labels = kmeans.fit_predict(&features).unwrap();

        // Should have 2 clusters
        let stats = kmeans.get_cluster_stats(&labels);
        assert_eq!(stats.n_clusters, 2);
    }

    /// Test 14: MiniBatch K-Means rejects invalid parameters
    #[test]
    fn test_minibatch_kmeans_invalid_params() {
        // n_clusters = 0 should fail
        let result = MiniBatchKMeans::new(0, 100, 100, 1e-4, None);
        assert!(result.is_err());

        // batch_size = 0 should fail
        let result = MiniBatchKMeans::new(5, 0, 100, 1e-4, None);
        assert!(result.is_err());
    }

    /// Test 15: MiniBatch K-Means deterministic with random seed
    #[test]
    fn test_minibatch_kmeans_deterministic() {
        let mut data = Vec::new();
        for i in 0..20 {
            data.push([i as f64, (i * 2) as f64]);
        }

        let mut features = Array2::zeros((20, 2));
        for (i, row) in data.iter().enumerate() {
            features[[i, 0]] = row[0];
            features[[i, 1]] = row[1];
        }

        let kmeans1 = MiniBatchKMeans::new(3, 10, 50, 1e-4, Some(42)).unwrap();
        let kmeans2 = MiniBatchKMeans::new(3, 10, 50, 1e-4, Some(42)).unwrap();

        let labels1 = kmeans1.fit_predict(&features).unwrap();
        let labels2 = kmeans2.fit_predict(&features).unwrap();

        assert_eq!(
            labels1, labels2,
            "Should produce identical results with same seed"
        );
    }

    /// Test 16: MiniBatch K-Means cluster statistics
    #[test]
    fn test_minibatch_kmeans_cluster_stats() {
        let kmeans = MiniBatchKMeans::new(3, 10, 50, 1e-4, Some(42)).unwrap();

        let mut features = Array2::zeros((30, 2));
        for i in 0..30 {
            features[[i, 0]] = i as f64;
            features[[i, 1]] = (i * 2) as f64;
        }

        let labels = kmeans.fit_predict(&features).unwrap();
        let stats = kmeans.get_cluster_stats(&labels);

        assert_eq!(stats.n_clusters, 3);
        assert_eq!(stats.cluster_sizes.iter().sum::<usize>(), 30);
    }

    /// Test 17: MiniBatch K-Means insufficient data
    #[test]
    fn test_minibatch_kmeans_insufficient_data() {
        let kmeans = MiniBatchKMeans::new(10, 10, 50, 1e-4, Some(42)).unwrap();

        let features = Array2::zeros((5, 2));

        let result = kmeans.fit_predict(&features);
        assert!(result.is_err());
    }

    /// Test 18: MiniBatch K-Means handles single cluster
    #[test]
    fn test_minibatch_kmeans_single_cluster() {
        let kmeans = MiniBatchKMeans::new(1, 10, 50, 1e-4, Some(42)).unwrap();

        let mut features = Array2::zeros((20, 2));
        for i in 0..20 {
            features[[i, 0]] = (i as f64) / 10.0;
            features[[i, 1]] = (i as f64) / 10.0;
        }

        let labels = kmeans.fit_predict(&features).unwrap();
        let stats = kmeans.get_cluster_stats(&labels);

        assert_eq!(stats.n_clusters, 1);
        assert_eq!(stats.cluster_sizes[0], 20);
    }
}
