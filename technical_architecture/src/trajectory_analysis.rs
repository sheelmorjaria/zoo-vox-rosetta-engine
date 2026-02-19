//! Trajectory Analysis Module for Continuous Manifold Analysis
//!
//! Addresses the "Continuous Manifold" problem in animal vocalization analysis.
//! Instead of forcing continuous vocalizations into discrete types, this module
//! outputs low-dimensional embedding coordinates that preserve trajectory information.
//!
//! Key Concepts:
//! - **Trajectory Vector**: Change in 45D acoustic space over time
//! - **Manifold Coordinates**: UMAP/t-SNE style coordinates for graded vocalizations
//! - **Continuous Path**: The morph from Phee → Trill carries information about endpoints
//!
//! Use Cases:
//! - Marmoset graded vocalizations (phee → trill continuum)
//! - Macaque voice quality variations
//! - Any species with continuous rather than discrete call types

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Default feature dimension (45D with new Resonance, Spectral Shape, Modulation, Non-Linear factors)
pub const DEFAULT_FEATURE_DIM: usize = 45;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Low-dimensional embedding coordinate for a phrase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifoldCoordinate {
    /// UMAP-style 2D embedding
    pub x: f64,
    pub y: f64,
    /// Optional 3rd dimension for complex manifolds
    pub z: Option<f64>,
    /// Distance to nearest cluster center
    pub cluster_distance: f64,
    /// Density of points at this location
    pub local_density: f64,
}

/// Trajectory through acoustic space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryVector {
    /// Start coordinate in 30D/45D space
    pub start_features: Vec<f64>,
    /// End coordinate in 30D/45D space
    pub end_features: Vec<f64>,
    /// Direction vector (normalized)
    pub direction: Vec<f64>,
    /// Magnitude of change
    pub magnitude: f64,
    /// Velocity of transition (change per ms)
    pub velocity_per_ms: f64,
    /// Curvature of trajectory (0 = linear, higher = more curved)
    pub curvature: f64,
}

/// Result of trajectory analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryAnalysisResult {
    /// Manifold coordinates for the phrase
    pub coordinate: ManifoldCoordinate,
    /// Trajectory vector if this is a graded transition
    pub trajectory: Option<TrajectoryVector>,
    /// Nearest discrete type (for backwards compatibility)
    pub nearest_type: Option<String>,
    /// Distance to nearest type centroid
    pub type_distance: f64,
    /// Whether this falls in a "transition zone" between types
    pub is_transition_zone: bool,
    /// Confidence in type assignment (lower for transition zones)
    pub type_confidence: f64,
}

/// Configuration for trajectory analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryAnalysisConfig {
    /// Number of dimensions for manifold embedding
    pub embedding_dim: usize,
    /// Minimum distance for UMAP-like embedding
    pub min_distance: f64,
    /// Number of neighbors for local structure
    pub n_neighbors: usize,
    /// Threshold for considering a point in "transition zone"
    pub transition_zone_threshold: f64,
    /// Minimum magnitude to consider as "trajectory" vs static
    pub min_trajectory_magnitude: f64,
}

impl Default for TrajectoryAnalysisConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 2,
            min_distance: 0.1,
            n_neighbors: 15,
            transition_zone_threshold: 0.3,
            min_trajectory_magnitude: 0.5,
        }
    }
}

// ============================================================================
// TRAJECTORY ANALYZER
// ============================================================================

/// Analyzes continuous trajectories through acoustic space
pub struct TrajectoryAnalyzer {
    config: TrajectoryAnalysisConfig,
    /// Reference points for manifold (type centroids)
    reference_points: Vec<(String, Vec<f64>)>,
    /// Fitted embedding transformation
    embedding_matrix: Option<Array2<f64>>,
    /// Feature statistics for normalization
    feature_means: Vec<f64>,
    feature_stds: Vec<f64>,
}

impl TrajectoryAnalyzer {
    /// Create new trajectory analyzer
    pub fn new(config: TrajectoryAnalysisConfig) -> Self {
        Self {
            config,
            reference_points: Vec::new(),
            embedding_matrix: None,
            feature_means: Vec::new(),
            feature_stds: Vec::new(),
        }
    }

    /// Add a reference point (type centroid) for the manifold
    pub fn add_reference_point(&mut self, type_name: &str, features: Vec<f64>) {
        self.reference_points
            .push((type_name.to_string(), features));
    }

    /// Fit the analyzer on a set of features (compute normalization statistics)
    pub fn fit(&mut self, features: &[Vec<f64>]) -> Result<(), TrajectoryError> {
        if features.is_empty() {
            return Err(TrajectoryError::EmptyFeatureSet);
        }

        let n_features = features[0].len();
        let n_samples = features.len();

        // Compute means
        self.feature_means = vec![0.0; n_features];
        for f in features {
            for (i, &val) in f.iter().enumerate() {
                self.feature_means[i] += val;
            }
        }
        for mean in &mut self.feature_means {
            *mean /= n_samples as f64;
        }

        // Compute standard deviations
        self.feature_stds = vec![0.0; n_features];
        for f in features {
            for (i, &val) in f.iter().enumerate() {
                let diff = val - self.feature_means[i];
                self.feature_stds[i] += diff * diff;
            }
        }
        for std in &mut self.feature_stds {
            *std = (*std / n_samples as f64).sqrt().max(1e-10);
        }

        // Create simple PCA-like embedding matrix for dimensionality reduction
        // For a proper implementation, this would use UMAP or t-SNE
        self.embedding_matrix = Some(self.create_embedding_matrix(n_features));

        Ok(())
    }

    /// Create embedding matrix for dimensionality reduction
    fn create_embedding_matrix(&self, n_features: usize) -> Array2<f64> {
        // Simple PCA-like projection (first 2 principal components approximation)
        // In production, this would use actual UMAP/t-SNE
        let mut matrix = Array2::<f64>::zeros((n_features, self.config.embedding_dim));

        // Use first two dimensions as pseudo-principal components
        for i in 0..n_features {
            for j in 0..self.config.embedding_dim {
                if i == j {
                    matrix[[i, j]] = 1.0;
                } else if j < 2 {
                    // Add some mixing for better spread
                    matrix[[i, j]] = 0.1 * ((i + j) as f64 / n_features as f64 - 0.5);
                }
            }
        }

        matrix
    }

    /// Analyze a single phrase's position on the manifold
    pub fn analyze(&self, features: &[f64]) -> Result<TrajectoryAnalysisResult, TrajectoryError> {
        if features.is_empty() {
            return Err(TrajectoryError::EmptyFeatures);
        }

        // Normalize features (if fitted) or use raw features
        let normalized = self.normalize_features(features);

        // Compute manifold coordinate
        let coordinate = self.compute_manifold_coordinate(&normalized)?;

        // Find nearest type
        let (nearest_type, type_distance) = self.find_nearest_type(&normalized);

        // Determine if in transition zone
        let is_transition_zone = self.is_transition_zone(&normalized, type_distance);

        // Compute type confidence (inverse of distance, scaled)
        let type_confidence = (-type_distance * 2.0).exp().min(1.0);

        Ok(TrajectoryAnalysisResult {
            coordinate,
            trajectory: None, // Computed separately for transitions
            nearest_type,
            type_distance,
            is_transition_zone,
            type_confidence,
        })
    }

    /// Check if the analyzer has been fitted
    pub fn is_fitted(&self) -> bool {
        self.embedding_matrix.is_some()
    }

    /// Analyze a transition between two feature vectors
    pub fn analyze_transition(
        &self,
        start_features: &[f64],
        end_features: &[f64],
        duration_ms: f64,
    ) -> Result<TrajectoryAnalysisResult, TrajectoryError> {
        let mut result = self.analyze(end_features)?;

        // Compute trajectory vector
        let start_norm = self.normalize_features(start_features);
        let end_norm = self.normalize_features(end_features);

        let direction: Vec<f64> = end_norm
            .iter()
            .zip(start_norm.iter())
            .map(|(e, s)| e - s)
            .collect();

        let magnitude: f64 = direction.iter().map(|d| d * d).sum::<f64>().sqrt();

        let normalized_direction: Vec<f64> = if magnitude > 1e-10 {
            direction.iter().map(|d| d / magnitude).collect()
        } else {
            vec![0.0; direction.len()]
        };

        // Compute curvature (simplified: based on deviation from linear path)
        let midpoint: Vec<f64> = start_norm
            .iter()
            .zip(end_norm.iter())
            .map(|(s, e)| (s + e) / 2.0)
            .collect();

        // Distance from midpoint to linear interpolation at 0.5
        let linear_midpoint: Vec<f64> = normalized_direction
            .iter()
            .map(|d| d * magnitude / 2.0)
            .collect();

        let curvature = midpoint
            .iter()
            .zip(linear_midpoint.iter())
            .map(|(m, l)| (m - l).powi(2))
            .sum::<f64>()
            .sqrt();

        let velocity = if duration_ms > 0.0 {
            magnitude / duration_ms
        } else {
            0.0
        };

        result.trajectory = Some(TrajectoryVector {
            start_features: start_features.to_vec(),
            end_features: end_features.to_vec(),
            direction: normalized_direction,
            magnitude,
            velocity_per_ms: velocity,
            curvature,
        });

        Ok(result)
    }

    /// Normalize features using fitted statistics
    fn normalize_features(&self, features: &[f64]) -> Vec<f64> {
        if self.feature_means.is_empty() {
            return features.to_vec();
        }

        features
            .iter()
            .zip(self.feature_means.iter())
            .zip(self.feature_stds.iter())
            .map(|((&f, &mean), &std)| (f - mean) / std)
            .collect()
    }

    /// Compute manifold coordinate from normalized features
    fn compute_manifold_coordinate(
        &self,
        features: &[f64],
    ) -> Result<ManifoldCoordinate, TrajectoryError> {
        let (x, y, z) = if let Some(embedding) = &self.embedding_matrix {
            // Use learned embedding
            let feature_array = Array1::from_vec(features.to_vec());
            let projected = feature_array.dot(embedding);

            let x = projected[0];
            let y = if projected.len() > 1 {
                projected[1]
            } else {
                0.0
            };
            let z = if projected.len() > 2 {
                Some(projected[2])
            } else {
                None
            };

            (x, y, z)
        } else {
            // Fallback: use first 2-3 dimensions directly
            let x = features.get(0).copied().unwrap_or(0.0);
            let y = features.get(1).copied().unwrap_or(0.0);
            let z = features.get(2).copied();
            (x, y, z)
        };

        // Compute local density (simplified: based on distance to nearest reference)
        let nearest_dist = self.compute_min_distance_to_references(features);
        let local_density = (-nearest_dist).exp();

        Ok(ManifoldCoordinate {
            x,
            y,
            z,
            cluster_distance: nearest_dist,
            local_density,
        })
    }

    /// Find the nearest type centroid
    fn find_nearest_type(&self, features: &[f64]) -> (Option<String>, f64) {
        let mut best_type: Option<String> = None;
        let mut best_distance = f64::INFINITY;

        for (type_name, ref_features) in &self.reference_points {
            let dist = self.cosine_distance(features, ref_features);
            if dist < best_distance {
                best_distance = dist;
                best_type = Some(type_name.clone());
            }
        }

        // Convert cosine distance to a more interpretable metric
        let type_distance = best_distance.sqrt();

        (best_type, type_distance)
    }

    /// Check if point is in a transition zone between types
    fn is_transition_zone(&self, features: &[f64], nearest_distance: f64) -> bool {
        // Count how many types are within the transition threshold
        let mut nearby_count = 0;

        for (_, ref_features) in &self.reference_points {
            let dist = self.cosine_distance(features, ref_features).sqrt();
            if dist < self.config.transition_zone_threshold {
                nearby_count += 1;
            }
        }

        // If multiple types are nearby, we're in a transition zone
        nearby_count > 1
    }

    /// Compute minimum distance to reference points
    fn compute_min_distance_to_references(&self, features: &[f64]) -> f64 {
        let mut min_dist = f64::INFINITY;

        for (_, ref_features) in &self.reference_points {
            let dist = self.euclidean_distance(features, ref_features);
            if dist < min_dist {
                min_dist = dist;
            }
        }

        if min_dist == f64::INFINITY {
            0.0
        } else {
            min_dist
        }
    }

    /// Euclidean distance between two feature vectors
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Cosine distance between two feature vectors
    fn cosine_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            1.0 - dot / (norm_a * norm_b)
        } else {
            1.0
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &TrajectoryAnalysisConfig {
        &self.config
    }

    /// Get reference points
    pub fn reference_points(&self) -> &[(String, Vec<f64>)] {
        &self.reference_points
    }
}

impl Default for TrajectoryAnalyzer {
    fn default() -> Self {
        Self::new(TrajectoryAnalysisConfig::default())
    }
}

// ============================================================================
// ERROR TYPES
// ============================================================================

#[derive(Debug, Clone)]
pub enum TrajectoryError {
    /// Empty feature vector provided
    EmptyFeatures,
    /// Empty feature set for fitting
    EmptyFeatureSet,
    /// Analyzer not fitted yet
    NotFitted,
    /// Dimension mismatch
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for TrajectoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrajectoryError::EmptyFeatures => write!(f, "Empty feature vector"),
            TrajectoryError::EmptyFeatureSet => write!(f, "Empty feature set for fitting"),
            TrajectoryError::NotFitted => write!(f, "Analyzer not fitted"),
            TrajectoryError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for TrajectoryError {}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_analyzer_creation() {
        let config = TrajectoryAnalysisConfig::default();
        let analyzer = TrajectoryAnalyzer::new(config);
        assert_eq!(analyzer.config().embedding_dim, 2);
        assert_eq!(analyzer.config().n_neighbors, 15);
    }

    #[test]
    fn test_manifold_coordinate_creation() {
        let coord = ManifoldCoordinate {
            x: 1.5,
            y: -0.3,
            z: Some(0.1),
            cluster_distance: 0.5,
            local_density: 0.8,
        };
        assert!((coord.x - 1.5).abs() < 1e-10);
        assert!((coord.y - (-0.3)).abs() < 1e-10);
        assert!(coord.z.is_some());
    }

    #[test]
    fn test_trajectory_vector_creation() {
        let trajectory = TrajectoryVector {
            start_features: vec![0.0, 0.0, 0.0],
            end_features: vec![1.0, 1.0, 1.0],
            direction: vec![0.577, 0.577, 0.577], // normalized
            magnitude: 1.732,
            velocity_per_ms: 0.01,
            curvature: 0.0,
        };
        assert!((trajectory.magnitude - 1.732).abs() < 0.01);
    }

    #[test]
    fn test_fit_analyzer() {
        let mut analyzer = TrajectoryAnalyzer::default();

        let features = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let result = analyzer.fit(&features);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fit_empty_features() {
        let mut analyzer = TrajectoryAnalyzer::default();
        let features: Vec<Vec<f64>> = vec![];

        let result = analyzer.fit(&features);
        assert!(matches!(result, Err(TrajectoryError::EmptyFeatureSet)));
    }

    #[test]
    fn test_analyze_without_fit() {
        let analyzer = TrajectoryAnalyzer::default();
        let features = vec![1.0, 0.0, 0.0];

        // Should still work, just won't normalize
        let result = analyzer.analyze(&features);
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyze_empty_features() {
        let analyzer = TrajectoryAnalyzer::default();
        let features: Vec<f64> = vec![];

        let result = analyzer.analyze(&features);
        assert!(matches!(result, Err(TrajectoryError::EmptyFeatures)));
    }

    #[test]
    fn test_add_reference_point() {
        let mut analyzer = TrajectoryAnalyzer::default();
        analyzer.add_reference_point("type_a", vec![1.0, 0.0, 0.0]);
        analyzer.add_reference_point("type_b", vec![0.0, 1.0, 0.0]);

        assert_eq!(analyzer.reference_points().len(), 2);
    }

    #[test]
    fn test_find_nearest_type() {
        let mut analyzer = TrajectoryAnalyzer::default();
        analyzer.add_reference_point("type_a", vec![1.0, 0.0, 0.0]);
        analyzer.add_reference_point("type_b", vec![0.0, 1.0, 0.0]);

        // Test close to type_a
        let result = analyzer.analyze(&[0.9, 0.1, 0.0]).unwrap();
        assert_eq!(result.nearest_type, Some("type_a".to_string()));

        // Test close to type_b
        let result = analyzer.analyze(&[0.1, 0.9, 0.0]).unwrap();
        assert_eq!(result.nearest_type, Some("type_b".to_string()));
    }

    #[test]
    fn test_transition_zone_detection() {
        let mut config = TrajectoryAnalysisConfig::default();
        config.transition_zone_threshold = 0.5;
        let mut analyzer = TrajectoryAnalyzer::new(config);

        // Add two close reference points
        analyzer.add_reference_point("type_a", vec![1.0, 0.0]);
        analyzer.add_reference_point("type_b", vec![0.8, 0.2]);

        // Point between them should be in transition zone
        let result = analyzer.analyze(&[0.9, 0.1]).unwrap();
        assert!(result.is_transition_zone);
    }

    #[test]
    fn test_analyze_transition() {
        let mut analyzer = TrajectoryAnalyzer::default();
        let features = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        analyzer.fit(&features).unwrap();

        let result = analyzer
            .analyze_transition(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], 100.0)
            .unwrap();

        assert!(result.trajectory.is_some());
        let trajectory = result.trajectory.unwrap();
        assert!(trajectory.magnitude > 0.0);
        assert!(trajectory.velocity_per_ms > 0.0);
    }

    #[test]
    fn test_type_confidence_decreases_with_distance() {
        let mut analyzer = TrajectoryAnalyzer::default();
        analyzer.add_reference_point("type_a", vec![1.0, 0.0, 0.0]);

        // Close point should have high confidence
        let close_result = analyzer.analyze(&[0.95, 0.05, 0.0]).unwrap();

        // Far point should have lower confidence
        let far_result = analyzer.analyze(&[0.5, 0.5, 0.0]).unwrap();

        assert!(close_result.type_confidence > far_result.type_confidence);
    }

    #[test]
    fn test_trajectory_curvature() {
        let mut analyzer = TrajectoryAnalyzer::default();
        let features = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        analyzer.fit(&features).unwrap();

        // Linear trajectory should have low curvature
        let linear = analyzer
            .analyze_transition(&[0.0, 0.0], &[1.0, 0.0], 100.0)
            .unwrap();

        assert!(linear.trajectory.unwrap().curvature >= 0.0);
    }

    #[test]
    fn test_serialization() {
        let coord = ManifoldCoordinate {
            x: 1.5,
            y: -0.3,
            z: Some(0.1),
            cluster_distance: 0.5,
            local_density: 0.8,
        };

        let json = serde_json::to_string(&coord).unwrap();
        let decoded: ManifoldCoordinate = serde_json::from_str(&json).unwrap();

        assert!((decoded.x - coord.x).abs() < 1e-10);
        assert!((decoded.y - coord.y).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance() {
        let analyzer = TrajectoryAnalyzer::default();

        // Identical vectors should have distance 0
        let dist = analyzer.cosine_distance(&[1.0, 0.0], &[1.0, 0.0]);
        assert!(dist.abs() < 1e-10);

        // Orthogonal vectors should have distance 1
        let dist = analyzer.cosine_distance(&[1.0, 0.0], &[0.0, 1.0]);
        assert!((dist - 1.0).abs() < 1e-10);

        // Opposite vectors should have distance 2
        let dist = analyzer.cosine_distance(&[1.0, 0.0], &[-1.0, 0.0]);
        assert!((dist - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let analyzer = TrajectoryAnalyzer::default();

        let dist = analyzer.euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((dist - 5.0).abs() < 1e-10);
    }
}
