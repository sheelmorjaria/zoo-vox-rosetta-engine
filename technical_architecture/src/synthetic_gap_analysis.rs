//! Synthetic Gap Analysis Module for Inter-Type Discriminability
//!
//! Validates synthesis quality by checking if synthesized calls are faithful
//! to their specific call type, not just globally close to nature.
//!
//! Key Concept: If you synthesize a "Type A" call, it should be acoustically
//! closer to natural "Type A" calls than to natural "Type B" calls.
//!
//! Metric: inter_type_discriminability_index
//! - High value (> 0.8): Synthesis faithfully preserves type identity
//! - Low value (< 0.5): Synthesis "drifts" into wrong call types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Result of synthetic gap analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticGapResult {
    /// Name/ID of the synthesized sample
    pub sample_id: String,
    /// The intended type for synthesis
    pub intended_type: String,
    /// Distance to the centroid of the intended type
    pub distance_to_intended: f64,
    /// Distance to the nearest *different* type centroid
    pub distance_to_nearest_other: f64,
    /// Inter-type discriminability index (0-1)
    /// Higher = better discriminability
    pub discriminability_index: f64,
    /// Whether synthesis passed the discriminability threshold
    pub passed: bool,
    /// The type the synthesis is actually closest to (may differ from intended)
    pub actual_nearest_type: String,
    /// Warning if synthesis drifted to wrong type
    pub drift_warning: Option<String>,
}

/// Aggregated statistics for a synthesis run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticGapStatistics {
    /// Total samples analyzed
    pub total_samples: usize,
    /// Samples that passed discriminability check
    pub passed_samples: usize,
    /// Pass rate (0-1)
    pub pass_rate: f64,
    /// Mean discriminability index
    pub mean_discriminability: f64,
    /// Standard deviation of discriminability
    pub std_discriminability: f64,
    /// Per-type statistics
    pub per_type_stats: HashMap<String, TypeDiscriminabilityStats>,
    /// Global t-SNE distance (for comparison)
    pub global_tsne_distance: f64,
    /// Number of samples that drifted to wrong type
    pub drift_count: usize,
}

/// Statistics for a single type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDiscriminabilityStats {
    /// Type name
    pub type_name: String,
    /// Number of samples of this type
    pub sample_count: usize,
    /// Mean discriminability for this type
    pub mean_discriminability: f64,
    /// Pass rate for this type
    pub pass_rate: f64,
    /// Most common drift target (if any)
    pub common_drift_target: Option<String>,
}

/// Configuration for synthetic gap analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticGapConfig {
    /// Minimum discriminability index to pass
    pub min_discriminability: f64,
    /// Weight for distance to intended type
    pub intended_weight: f64,
    /// Weight for distance to other types
    pub other_weight: f64,
    /// Whether to issue warnings on drift
    pub warn_on_drift: bool,
    /// Maximum acceptable global t-SNE distance
    pub max_global_distance: f64,
}

impl Default for SyntheticGapConfig {
    fn default() -> Self {
        Self {
            min_discriminability: 0.7,
            intended_weight: 1.0,
            other_weight: 0.5,
            warn_on_drift: true,
            max_global_distance: 7.0,
        }
    }
}

// ============================================================================
// SYNTHETIC GAP ANALYZER
// ============================================================================

/// Analyzes synthesis quality using inter-type discriminability
pub struct SyntheticGapAnalyzer {
    config: SyntheticGapConfig,
    /// Type centroids (type_name -> feature vector)
    type_centroids: HashMap<String, Vec<f64>>,
    /// Individual natural samples by type (type_name -> list of features)
    natural_samples: HashMap<String, Vec<Vec<f64>>>,
    /// Analysis results
    results: Vec<SyntheticGapResult>,
}

impl SyntheticGapAnalyzer {
    /// Create new analyzer with default config
    pub fn new() -> Self {
        Self::with_config(SyntheticGapConfig::default())
    }

    /// Create new analyzer with custom config
    pub fn with_config(config: SyntheticGapConfig) -> Self {
        Self {
            config,
            type_centroids: HashMap::new(),
            natural_samples: HashMap::new(),
            results: Vec::new(),
        }
    }

    /// Add a natural sample for reference
    pub fn add_natural_sample(&mut self, type_name: &str, features: Vec<f64>) {
        self.natural_samples
            .entry(type_name.to_string())
            .or_default()
            .push(features);
    }

    /// Compute centroids from added natural samples
    pub fn compute_centroids(&mut self) {
        for (type_name, samples) in &self.natural_samples {
            if samples.is_empty() {
                continue;
            }

            let n_features = samples[0].len();
            let mut centroid = vec![0.0; n_features];

            for sample in samples {
                for (i, &val) in sample.iter().enumerate() {
                    centroid[i] += val;
                }
            }

            let n = samples.len() as f64;
            for val in &mut centroid {
                *val /= n;
            }

            self.type_centroids.insert(type_name.clone(), centroid);
        }
    }

    /// Analyze a synthesized sample
    pub fn analyze_synthesis(
        &mut self,
        sample_id: &str,
        intended_type: &str,
        features: &[f64],
    ) -> Result<SyntheticGapResult, SyntheticGapError> {
        if features.is_empty() {
            return Err(SyntheticGapError::EmptyFeatures);
        }

        // Get intended type centroid
        let intended_centroid = self
            .type_centroids
            .get(intended_type)
            .ok_or(SyntheticGapError::UnknownType(intended_type.to_string()))?;

        // Compute distance to intended type
        let distance_to_intended = self.cosine_distance(features, intended_centroid);

        // Find nearest other type
        let mut nearest_other_type = String::new();
        let mut distance_to_nearest_other = f64::INFINITY;

        for (type_name, centroid) in &self.type_centroids {
            if type_name != intended_type {
                let dist = self.cosine_distance(features, centroid);
                if dist < distance_to_nearest_other {
                    distance_to_nearest_other = dist;
                    nearest_other_type = type_name.clone();
                }
            }
        }

        // Find actual nearest type (including intended)
        let mut actual_nearest_type = intended_type.to_string();
        let mut min_distance = distance_to_intended;

        for (type_name, centroid) in &self.type_centroids {
            let dist = self.cosine_distance(features, centroid);
            if dist < min_distance {
                min_distance = dist;
                actual_nearest_type = type_name.clone();
            }
        }

        // Compute discriminability index
        // High when distance_to_intended is small AND distance_to_nearest_other is large
        let discriminability =
            self.compute_discriminability_index(distance_to_intended, distance_to_nearest_other);

        // Determine if passed
        let passed = discriminability >= self.config.min_discriminability;

        // Check for drift
        let drift_warning = if actual_nearest_type != intended_type {
            if self.config.warn_on_drift {
                Some(format!(
                    "Synthesis drifted to '{}' instead of intended '{}'",
                    actual_nearest_type, intended_type
                ))
            } else {
                None
            }
        } else {
            None
        };

        let result = SyntheticGapResult {
            sample_id: sample_id.to_string(),
            intended_type: intended_type.to_string(),
            distance_to_intended,
            distance_to_nearest_other,
            discriminability_index: discriminability,
            passed,
            actual_nearest_type,
            drift_warning,
        };

        self.results.push(result.clone());
        Ok(result)
    }

    /// Compute discriminability index
    fn compute_discriminability_index(
        &self,
        distance_to_intended: f64,
        distance_to_nearest_other: f64,
    ) -> f64 {
        // Discriminability = how much closer we are to intended vs other
        // Scale: 0 = equally close, 1 = much closer to intended
        if distance_to_nearest_other == 0.0 {
            return 0.0;
        }

        let ratio = distance_to_intended / distance_to_nearest_other;

        // Map ratio to 0-1 scale
        // ratio = 0.5 (intended is half as far) -> high discriminability
        // ratio = 1.0 (equal distance) -> neutral
        // ratio = 2.0 (intended is twice as far) -> low discriminability
        1.0 / (1.0 + ratio * 2.0)
    }

    /// Get aggregated statistics
    pub fn compute_statistics(&self) -> SyntheticGapStatistics {
        let total_samples = self.results.len();
        let passed_samples = self.results.iter().filter(|r| r.passed).count();
        let pass_rate = if total_samples > 0 {
            passed_samples as f64 / total_samples as f64
        } else {
            0.0
        };

        // Compute mean and std discriminability
        let mean_discriminability = if total_samples > 0 {
            self.results
                .iter()
                .map(|r| r.discriminability_index)
                .sum::<f64>()
                / total_samples as f64
        } else {
            0.0
        };

        let std_discriminability = if total_samples > 1 {
            let variance: f64 = self
                .results
                .iter()
                .map(|r| (r.discriminability_index - mean_discriminability).powi(2))
                .sum::<f64>()
                / (total_samples - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Compute per-type statistics
        let mut per_type_stats: HashMap<String, TypeDiscriminabilityStats> = HashMap::new();
        let mut drift_count = 0;

        for result in &self.results {
            let entry = per_type_stats
                .entry(result.intended_type.clone())
                .or_insert(TypeDiscriminabilityStats {
                    type_name: result.intended_type.clone(),
                    sample_count: 0,
                    mean_discriminability: 0.0,
                    pass_rate: 0.0,
                    common_drift_target: None,
                });
            entry.sample_count += 1;
            entry.mean_discriminability += result.discriminability_index;
            if result.passed {
                entry.pass_rate += 1.0;
            }

            if result.actual_nearest_type != result.intended_type {
                drift_count += 1;
            }
        }

        // Finalize per-type stats
        for stats in per_type_stats.values_mut() {
            if stats.sample_count > 0 {
                stats.mean_discriminability /= stats.sample_count as f64;
                stats.pass_rate /= stats.sample_count as f64;
            }
        }

        // Compute global t-SNE distance (average distance to all centroids)
        let global_tsne_distance = if total_samples > 0 {
            self.results
                .iter()
                .map(|r| r.distance_to_intended)
                .sum::<f64>()
                / total_samples as f64
        } else {
            0.0
        };

        SyntheticGapStatistics {
            total_samples,
            passed_samples,
            pass_rate,
            mean_discriminability,
            std_discriminability,
            per_type_stats,
            global_tsne_distance,
            drift_count,
        }
    }

    /// Get all results
    pub fn results(&self) -> &[SyntheticGapResult] {
        &self.results
    }

    /// Get the configuration
    pub fn config(&self) -> &SyntheticGapConfig {
        &self.config
    }

    /// Clear results
    pub fn clear_results(&mut self) {
        self.results.clear();
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
}

impl Default for SyntheticGapAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ERROR TYPES
// ============================================================================

#[derive(Debug, Clone)]
pub enum SyntheticGapError {
    /// Empty feature vector
    EmptyFeatures,
    /// Unknown type (not in reference set)
    UnknownType(String),
    /// No reference samples added
    NoReferenceSamples,
}

impl std::fmt::Display for SyntheticGapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SyntheticGapError::EmptyFeatures => write!(f, "Empty feature vector"),
            SyntheticGapError::UnknownType(t) => write!(f, "Unknown type: {}", t),
            SyntheticGapError::NoReferenceSamples => write!(f, "No reference samples added"),
        }
    }
}

impl std::error::Error for SyntheticGapError {}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_gap_analyzer_creation() {
        let analyzer = SyntheticGapAnalyzer::new();
        assert_eq!(analyzer.config().min_discriminability, 0.7);
    }

    #[test]
    fn test_add_natural_samples() {
        let mut analyzer = SyntheticGapAnalyzer::new();
        analyzer.add_natural_sample("type_a", vec![1.0, 0.0, 0.0]);
        analyzer.add_natural_sample("type_a", vec![0.9, 0.1, 0.0]);
        analyzer.add_natural_sample("type_b", vec![0.0, 1.0, 0.0]);

        assert_eq!(analyzer.natural_samples.get("type_a").unwrap().len(), 2);
        assert_eq!(analyzer.natural_samples.get("type_b").unwrap().len(), 1);
    }

    #[test]
    fn test_compute_centroids() {
        let mut analyzer = SyntheticGapAnalyzer::new();
        analyzer.add_natural_sample("type_a", vec![1.0, 0.0]);
        analyzer.add_natural_sample("type_a", vec![0.0, 1.0]);
        analyzer.compute_centroids();

        let centroid = analyzer.type_centroids.get("type_a").unwrap();
        assert!((centroid[0] - 0.5).abs() < 1e-10);
        assert!((centroid[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_analyze_synthesis_close_to_intended() {
        let mut analyzer = SyntheticGapAnalyzer::new();
        analyzer.add_natural_sample("type_a", vec![1.0, 0.0, 0.0]);
        analyzer.add_natural_sample("type_b", vec![0.0, 1.0, 0.0]);
        analyzer.compute_centroids();

        // Synthesis close to type_a
        let result = analyzer
            .analyze_synthesis("synth_001", "type_a", &[0.95, 0.05, 0.0])
            .unwrap();

        assert_eq!(result.intended_type, "type_a");
        assert!(result.discriminability_index > 0.5);
        assert!(result.passed);
    }

    #[test]
    fn test_analyze_synthesis_drifted() {
        let mut analyzer = SyntheticGapAnalyzer::new();
        analyzer.add_natural_sample("type_a", vec![1.0, 0.0, 0.0]);
        analyzer.add_natural_sample("type_b", vec![0.0, 1.0, 0.0]);
        analyzer.compute_centroids();

        // Synthesis drifted to type_b (intended type_a)
        let result = analyzer
            .analyze_synthesis("synth_002", "type_a", &[0.1, 0.9, 0.0])
            .unwrap();

        assert!(result.drift_warning.is_some());
        assert_eq!(result.actual_nearest_type, "type_b");
    }

    #[test]
    fn test_analyze_empty_features() {
        let mut analyzer = SyntheticGapAnalyzer::new();
        analyzer.add_natural_sample("type_a", vec![1.0, 0.0]);
        analyzer.compute_centroids();

        let result = analyzer.analyze_synthesis("synth_001", "type_a", &[]);
        assert!(matches!(result, Err(SyntheticGapError::EmptyFeatures)));
    }

    #[test]
    fn test_analyze_unknown_type() {
        let mut analyzer = SyntheticGapAnalyzer::new();
        analyzer.add_natural_sample("type_a", vec![1.0, 0.0]);
        analyzer.compute_centroids();

        let result = analyzer.analyze_synthesis("synth_001", "unknown_type", &[1.0, 0.0]);
        assert!(matches!(result, Err(SyntheticGapError::UnknownType(_))));
    }

    #[test]
    fn test_compute_statistics() {
        let mut analyzer = SyntheticGapAnalyzer::new();
        analyzer.add_natural_sample("type_a", vec![1.0, 0.0]);
        analyzer.add_natural_sample("type_b", vec![0.0, 1.0]);
        analyzer.compute_centroids();

        // Add multiple synthesis results
        analyzer
            .analyze_synthesis("s1", "type_a", &[0.9, 0.1])
            .unwrap();
        analyzer
            .analyze_synthesis("s2", "type_a", &[0.8, 0.2])
            .unwrap();
        analyzer
            .analyze_synthesis("s3", "type_b", &[0.1, 0.9])
            .unwrap();

        let stats = analyzer.compute_statistics();

        assert_eq!(stats.total_samples, 3);
        assert!(stats.mean_discriminability > 0.0);
        assert!(stats.pass_rate > 0.0);
    }

    #[test]
    fn test_discriminability_index_calculation() {
        let analyzer = SyntheticGapAnalyzer::new();

        // When intended is much closer than other
        let high = analyzer.compute_discriminability_index(0.1, 0.9);
        assert!(high > 0.7);

        // When distances are equal (ratio = 1.0, discriminability = 1/(1+2) = 0.33)
        let neutral = analyzer.compute_discriminability_index(0.5, 0.5);
        assert!((neutral - 0.33).abs() < 0.1);

        // When intended is farther (bad)
        let low = analyzer.compute_discriminability_index(0.9, 0.1);
        assert!(low < 0.3);

        // When intended is very close and other is far (ratio = 0.1/0.9 = 0.11)
        let very_high = analyzer.compute_discriminability_index(0.1, 0.9);
        assert!(very_high > 0.7);
    }

    #[test]
    fn test_serialization() {
        let result = SyntheticGapResult {
            sample_id: "test".to_string(),
            intended_type: "type_a".to_string(),
            distance_to_intended: 0.2,
            distance_to_nearest_other: 0.8,
            discriminability_index: 0.75,
            passed: true,
            actual_nearest_type: "type_a".to_string(),
            drift_warning: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let decoded: SyntheticGapResult = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.sample_id, "test");
        assert!((decoded.discriminability_index - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_drift_count() {
        let mut analyzer = SyntheticGapAnalyzer::new();
        analyzer.add_natural_sample("type_a", vec![1.0, 0.0]);
        analyzer.add_natural_sample("type_b", vec![0.0, 1.0]);
        analyzer.compute_centroids();

        analyzer
            .analyze_synthesis("s1", "type_a", &[0.9, 0.1])
            .unwrap(); // No drift
        analyzer
            .analyze_synthesis("s2", "type_a", &[0.1, 0.9])
            .unwrap(); // Drift

        let stats = analyzer.compute_statistics();
        assert_eq!(stats.drift_count, 1);
    }
}
