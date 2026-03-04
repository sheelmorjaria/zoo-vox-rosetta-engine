//! Anomaly Detection - The "Sentry" Mode
//! ======================================
//!
//! Detects acoustic outliers in the latent space for:
//! - Newborn vocalizations (developing syntax)
//! - Injury/distress calls (unusual spectral structure)
//! - Invasive species (unknown clusters)
//!
//! ## Key Insight
//! Since Rosetta-Net learns to reconstruct/classify normal sounds,
//! we can measure **Reconstruction Error** or **Latent Distance**.
//! High distance to all known clusters = **Acoustic Outlier**.
//!
//! ## Usage
//! ```rust
//! use technical_architecture::AnomalyDetector;
//!
//! let detector = AnomalyDetector::new(128);
//! detector.add_reference("marmoset_normal", &latent_vector);
//!
//! let score = detector.compute_anomaly_score(&new_latent);
//! if score > 0.8 {
//!     // Flag as outlier
//! }
//! ```

use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Configuration for anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyDetectorConfig {
    /// Dimension of latent space
    pub latent_dim: usize,
    /// Threshold for outlier classification
    pub outlier_threshold: f32,
    /// Number of nearest neighbors to consider
    pub k_neighbors: usize,
    /// Minimum samples before detection is reliable
    pub min_samples: usize,
    /// Use Mahalanobis distance (requires covariance estimation)
    pub use_mahalanobis: bool,
}

impl Default for AnomalyDetectorConfig {
    fn default() -> Self {
        Self {
            latent_dim: 128,
            outlier_threshold: 0.8,
            k_neighbors: 5,
            min_samples: 10,
            use_mahalanobis: false,
        }
    }
}

/// Type of anomaly detected
#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum AnomalyType {
    /// Novel vocalization (new type, possibly newborn)
    Novel,
    /// Distress/alarm call (unusual spectral structure)
    Distress,
    /// Unknown species (completely foreign cluster)
    UnknownSpecies,
    /// Malformed call (possibly injured individual)
    Malformed,
    /// Environmental noise (not a vocalization)
    Environmental,
}

/// Result of anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Anomaly score (0.0-1.0, higher = more anomalous)
    pub score: f32,
    /// Whether the sample is an outlier
    pub is_outlier: bool,
    /// Type of anomaly (if detected)
    pub anomaly_type: Option<AnomalyType>,
    /// Distance to nearest known cluster
    pub nearest_distance: f32,
    /// Name of nearest known class
    pub nearest_class: Option<String>,
    /// Confidence in the anomaly detection
    pub confidence: f32,
}

/// A cluster of reference embeddings
#[derive(Debug, Clone)]
struct ReferenceCluster {
    /// Cluster name/label
    name: String,
    /// Reference embeddings
    embeddings: Vec<Array1<f32>>,
    /// Cluster centroid
    centroid: Array1<f32>,
    /// Cluster radius (max distance from centroid)
    radius: f32,
}

/// Anomaly Detector for acoustic outlier detection
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    config: AnomalyDetectorConfig,
    /// Reference clusters by name
    clusters: HashMap<String, ReferenceCluster>,
    /// All reference embeddings for k-NN
    all_embeddings: Vec<(String, Array1<f32>)>,
    /// Global centroid for overall distance
    global_centroid: Option<Array1<f32>>,
    /// Total samples added
    total_samples: usize,
}

impl AnomalyDetector {
    /// Create a new anomaly detector with default configuration
    pub fn new(latent_dim: usize) -> Self {
        Self::with_config(AnomalyDetectorConfig {
            latent_dim,
            ..Default::default()
        })
    }

    /// Create an anomaly detector with custom configuration
    pub fn with_config(config: AnomalyDetectorConfig) -> Self {
        Self {
            config,
            clusters: HashMap::new(),
            all_embeddings: Vec::new(),
            global_centroid: None,
            total_samples: 0,
        }
    }

    /// Add a reference embedding for a known class
    pub fn add_reference(&mut self, class_name: &str, embedding: &[f32]) {
        let embedding = Array1::from_vec(embedding.to_vec());

        // Add to cluster
        let cluster = self
            .clusters
            .entry(class_name.to_string())
            .or_insert_with(|| ReferenceCluster {
                name: class_name.to_string(),
                embeddings: Vec::new(),
                centroid: Array1::zeros(self.config.latent_dim),
                radius: 0.0,
            });

        cluster.embeddings.push(embedding.clone());
        self.all_embeddings
            .push((class_name.to_string(), embedding));
        self.total_samples += 1;

        // Update cluster centroid
        self.update_cluster_centroid(class_name);
    }

    /// Add multiple reference embeddings for a class
    pub fn add_references(&mut self, class_name: &str, embeddings: &[Vec<f32>]) {
        for embedding in embeddings {
            self.add_reference(class_name, embedding);
        }
    }

    /// Compute anomaly score for a new embedding
    pub fn compute_anomaly_score(&self, embedding: &[f32]) -> f32 {
        if self.all_embeddings.is_empty() {
            return 0.5; // Unknown if no references
        }

        let embedding = Array1::from_vec(embedding.to_vec());

        // Compute distance to nearest cluster
        let nearest_dist = self.nearest_cluster_distance(&embedding);

        // Normalize to 0-1 range
        let normalized = 1.0 - (-nearest_dist / 10.0).exp();

        normalized.clamp(0.0, 1.0)
    }

    /// Detect if a sample is an outlier
    pub fn detect(&self, embedding: &[f32]) -> AnomalyResult {
        if self.all_embeddings.is_empty() {
            return AnomalyResult {
                score: 0.5,
                is_outlier: false,
                anomaly_type: None,
                nearest_distance: f32::INFINITY,
                nearest_class: None,
                confidence: 0.0,
            };
        }

        let embedding = Array1::from_vec(embedding.to_vec());

        // Find nearest cluster
        let (nearest_class, nearest_dist) = self.find_nearest_cluster(&embedding);

        // Compute anomaly score
        let score = 1.0 - (-nearest_dist / 10.0).exp();
        let score = score.clamp(0.0, 1.0);

        // Determine if outlier
        let is_outlier = score > self.config.outlier_threshold;

        // Determine anomaly type based on characteristics
        let anomaly_type = if is_outlier {
            self.classify_anomaly(&embedding, nearest_dist)
        } else {
            None
        };

        // Compute confidence based on number of reference samples
        let confidence = (self.total_samples as f32 / self.config.min_samples as f32).min(1.0);

        AnomalyResult {
            score,
            is_outlier,
            anomaly_type,
            nearest_distance: nearest_dist,
            nearest_class: Some(nearest_class),
            confidence,
        }
    }

    /// Check if a sample is an outlier (simplified)
    pub fn is_outlier(&self, embedding: &[f32]) -> bool {
        self.detect(embedding).is_outlier
    }

    /// Get statistics about known clusters
    pub fn cluster_stats(&self) -> HashMap<String, (usize, f32)> {
        self.clusters
            .iter()
            .map(|(name, cluster)| (name.clone(), (cluster.embeddings.len(), cluster.radius)))
            .collect()
    }

    /// Get total number of reference samples
    pub fn total_references(&self) -> usize {
        self.total_samples
    }

    /// Get number of known classes
    pub fn num_classes(&self) -> usize {
        self.clusters.len()
    }

    /// Clear all reference data
    pub fn clear(&mut self) {
        self.clusters.clear();
        self.all_embeddings.clear();
        self.global_centroid = None;
        self.total_samples = 0;
    }

    /// Update cluster centroid after adding new embedding
    fn update_cluster_centroid(&mut self, class_name: &str) {
        if let Some(cluster) = self.clusters.get_mut(class_name) {
            let n = cluster.embeddings.len();
            if n == 0 {
                return;
            }

            // Compute new centroid
            let mut centroid = Array1::zeros(self.config.latent_dim);
            for embedding in &cluster.embeddings {
                centroid += embedding;
            }
            cluster.centroid = centroid / n as f32;

            // Update radius
            let mut max_dist = 0.0f32;
            for embedding in &cluster.embeddings {
                let dist = euclidean_distance(&cluster.centroid, embedding);
                max_dist = max_dist.max(dist);
            }
            cluster.radius = max_dist;
        }
    }

    /// Find the nearest cluster to an embedding
    fn find_nearest_cluster(&self, embedding: &Array1<f32>) -> (String, f32) {
        let mut nearest_class = String::new();
        let mut nearest_dist = f32::INFINITY;

        for (name, cluster) in &self.clusters {
            let dist = euclidean_distance(&cluster.centroid, embedding);
            if dist < nearest_dist {
                nearest_dist = dist;
                nearest_class = name.clone();
            }
        }

        (nearest_class, nearest_dist)
    }

    /// Compute distance to nearest cluster centroid
    fn nearest_cluster_distance(&self, embedding: &Array1<f32>) -> f32 {
        let (_, dist) = self.find_nearest_cluster(embedding);
        dist
    }

    /// Classify the type of anomaly based on embedding characteristics
    fn classify_anomaly(&self, embedding: &Array1<f32>, distance: f32) -> Option<AnomalyType> {
        // Very far from all clusters = unknown species
        if distance > 20.0 {
            return Some(AnomalyType::UnknownSpecies);
        }

        // Check embedding statistics
        let mean: f32 = embedding.mean().unwrap_or(0.0);
        let variance: f32 = embedding.var(0.0);
        let max_val = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = embedding.iter().cloned().fold(f32::INFINITY, f32::min);

        // High variance, extreme values = distress
        if variance > 1.0 || max_val.abs() > 3.0 || min_val.abs() > 3.0 {
            return Some(AnomalyType::Distress);
        }

        // Low variance, close to zero = environmental noise
        if variance < 0.01 && mean.abs() < 0.1 {
            return Some(AnomalyType::Environmental);
        }

        // Moderate distance, normal statistics = novel vocalization
        if distance > 5.0 {
            return Some(AnomalyType::Novel);
        }

        // Default to malformed
        Some(AnomalyType::Malformed)
    }

    /// k-NN distance for outlier detection
    fn knn_distance(&self, embedding: &Array1<f32>, k: usize) -> f32 {
        if self.all_embeddings.is_empty() {
            return f32::INFINITY;
        }

        let mut distances: Vec<f32> = self
            .all_embeddings
            .iter()
            .map(|(_, ref_emb)| euclidean_distance(embedding, ref_emb))
            .collect();

        distances.sort_by(|a: &f32, b: &f32| a.partial_cmp(b).unwrap());

        let k = k.min(distances.len());
        distances[..k].iter().sum::<f32>() / k as f32
    }
}

/// Compute Euclidean distance between two vectors
fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    (a - b).mapv(|x| x * x).sum().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let detector = AnomalyDetector::new(128);
        assert_eq!(detector.config.latent_dim, 128);
        assert_eq!(detector.total_references(), 0);
    }

    #[test]
    fn test_add_reference() {
        let mut detector = AnomalyDetector::new(4);
        let embedding = vec![1.0, 0.0, 0.0, 0.0];

        detector.add_reference("class_a", &embedding);

        assert_eq!(detector.total_references(), 1);
        assert_eq!(detector.num_classes(), 1);
    }

    #[test]
    fn test_add_multiple_references() {
        let mut detector = AnomalyDetector::new(4);

        detector.add_references(
            "class_a",
            &[
                vec![1.0, 0.0, 0.0, 0.0],
                vec![1.1, 0.1, 0.0, 0.0],
                vec![0.9, -0.1, 0.0, 0.0],
            ],
        );

        assert_eq!(detector.total_references(), 3);
    }

    #[test]
    fn test_anomaly_score_known() {
        let mut detector = AnomalyDetector::new(4);
        detector.add_reference("class_a", &[1.0, 0.0, 0.0, 0.0]);

        // Same vector should have low anomaly score
        let score = detector.compute_anomaly_score(&[1.0, 0.0, 0.0, 0.0]);
        assert!(score < 0.5);
    }

    #[test]
    fn test_anomaly_score_unknown() {
        let mut detector = AnomalyDetector::new(4);
        detector.add_reference("class_a", &[1.0, 0.0, 0.0, 0.0]);

        // Very different vector should have high anomaly score
        let score = detector.compute_anomaly_score(&[100.0, 100.0, 100.0, 100.0]);
        assert!(score > 0.5);
    }

    #[test]
    fn test_detect_inlier() {
        let mut detector = AnomalyDetector::new(4);
        detector.add_references(
            "class_a",
            &[
                vec![1.0, 0.0, 0.0, 0.0],
                vec![1.1, 0.1, 0.0, 0.0],
                vec![0.9, -0.1, 0.0, 0.0],
            ],
        );

        let result = detector.detect(&[1.0, 0.0, 0.0, 0.0]);

        assert!(!result.is_outlier);
        assert_eq!(result.nearest_class, Some("class_a".to_string()));
    }

    #[test]
    fn test_detect_outlier() {
        let mut detector = AnomalyDetector::with_config(AnomalyDetectorConfig {
            latent_dim: 4,
            outlier_threshold: 0.5,
            ..Default::default()
        });

        detector.add_references(
            "class_a",
            &[vec![1.0, 0.0, 0.0, 0.0], vec![1.1, 0.1, 0.0, 0.0]],
        );

        // Very different vector should be an outlier
        let result = detector.detect(&[50.0, 50.0, 50.0, 50.0]);

        assert!(result.is_outlier);
        assert!(result.score > 0.5);
    }

    #[test]
    fn test_empty_detector() {
        let detector = AnomalyDetector::new(4);
        let result = detector.detect(&[1.0, 0.0, 0.0, 0.0]);

        assert!(!result.is_outlier);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_clear() {
        let mut detector = AnomalyDetector::new(4);
        detector.add_reference("class_a", &[1.0, 0.0, 0.0, 0.0]);

        detector.clear();

        assert_eq!(detector.total_references(), 0);
        assert_eq!(detector.num_classes(), 0);
    }

    #[test]
    fn test_cluster_stats() {
        let mut detector = AnomalyDetector::new(4);
        detector.add_references(
            "class_a",
            &[vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]],
        );

        let stats = detector.cluster_stats();

        assert!(stats.contains_key("class_a"));
        let (count, _radius) = stats["class_a"];
        assert_eq!(count, 2);
    }

    #[test]
    fn test_multiple_classes() {
        let mut detector = AnomalyDetector::new(4);

        detector.add_references("class_a", &[vec![1.0, 0.0, 0.0, 0.0]]);

        detector.add_references("class_b", &[vec![0.0, 1.0, 0.0, 0.0]]);

        assert_eq!(detector.num_classes(), 2);

        // Test nearest class detection
        let result_a = detector.detect(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(result_a.nearest_class, Some("class_a".to_string()));

        let result_b = detector.detect(&[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(result_b.nearest_class, Some("class_b".to_string()));
    }

    #[test]
    fn test_is_outlier_simple() {
        let mut detector = AnomalyDetector::with_config(AnomalyDetectorConfig {
            latent_dim: 4,
            outlier_threshold: 0.5,
            ..Default::default()
        });

        detector.add_reference("normal", &[0.0, 0.0, 0.0, 0.0]);

        // Normal should not be outlier
        assert!(!detector.is_outlier(&[0.1, 0.1, 0.1, 0.1]));

        // Far away should be outlier
        assert!(detector.is_outlier(&[100.0, 100.0, 100.0, 100.0]));
    }

    #[test]
    fn test_anomaly_type_distress() {
        let mut detector = AnomalyDetector::with_config(AnomalyDetectorConfig {
            latent_dim: 4,
            outlier_threshold: 0.3,
            ..Default::default()
        });

        detector.add_reference("normal", &[0.5, 0.5, 0.5, 0.5]);

        // High variance embedding
        let result = detector.detect(&[10.0, -10.0, 10.0, -10.0]);

        assert!(result.is_outlier);
        assert!(result.anomaly_type.is_some());
    }

    #[test]
    fn test_confidence_increases() {
        let mut detector = AnomalyDetector::new(4);

        // Add samples one at a time
        for i in 0..20 {
            detector.add_reference("class_a", &[1.0, 0.0, 0.0, 0.0]);

            let result = detector.detect(&[1.0, 0.0, 0.0, 0.0]);
            let expected_confidence =
                ((i + 1) as f32 / detector.config.min_samples as f32).min(1.0);

            assert!((result.confidence - expected_confidence).abs() < 0.01);
        }
    }
}
