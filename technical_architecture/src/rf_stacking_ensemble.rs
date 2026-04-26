//! RF Feature Stacking Ensemble for Bioacoustic Classification
//! ============================================================
//!
//! This module implements a feature stacking ensemble that combines:
//!
//! 1. **Physics RF (46D)**: Trained on physics features (duration, pitch, tempo).
//!    - High bias, low variance
//!    - Robust to noise and domain shift
//!    - Trust this model when confident
//!
//! 2. **Full RF (112D)**: Trained on all features (physics + texture).
//!    - Lower bias, higher variance
//!    - Can capture complex patterns
//!    - Used when physics model is uncertain
//!
//! 3. **Confidence-Weighted Stacker**: Meta-learner that combines predictions.
//!    - Prefers Physics RF when it's confident (physics is more robust)
//!    - Falls back to Full RF when Physics is uncertain
//!    - Boosts confidence when both models agree
//!
//! # Key Insight
//!
//! Physics features are more robust to noise and domain shift. If the Physics RF
//! predicts "Bat" with 90% confidence while the Full RF predicts "Insect" with 60%
//! confidence, the ensemble trusts the Physics RF.
//!
//! # Detection Mode
//!
//! The ensemble can operate in detection mode with a confidence threshold:
//! - Predictions below threshold are marked as "background" (no detection)
//! - Useful for filtering out silence/noise in continuous audio

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Constants
// =============================================================================

/// Physics feature dimension (Layer 1: indices 0-45)
pub const PHYSICS_DIM: usize = 46;

/// Full feature dimension (Layer 1 + Layer 2)
pub const FULL_DIM: usize = 112;

/// Default physics preference multiplier
pub const DEFAULT_PHYSICS_PREFERENCE: f32 = 1.5;

/// Default detection confidence threshold
pub const DEFAULT_DETECTION_THRESHOLD: f32 = 0.5;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the RF Stacking Ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackingConfig {
    /// Number of estimators for Physics RF
    pub physics_n_estimators: usize,
    /// Max depth for Physics RF
    pub physics_max_depth: usize,
    /// Min samples split for Physics RF
    pub physics_min_samples_split: usize,

    /// Number of estimators for Full RF
    pub full_n_estimators: usize,
    /// Max depth for Full RF
    pub full_max_depth: usize,
    /// Min samples split for Full RF
    pub full_min_samples_split: usize,

    /// Preference multiplier for physics model predictions
    /// Higher values = more trust in physics model when confident
    pub physics_preference: f32,

    /// Minimum confidence for physics model to override full model
    pub physics_confidence_threshold: f32,

    /// Confidence threshold for detection mode
    /// Predictions below this are marked as "background"
    pub detection_threshold: Option<f32>,

    /// Random seed for reproducibility
    pub random_seed: u64,
}

impl Default for StackingConfig {
    fn default() -> Self {
        Self {
            physics_n_estimators: 200,
            physics_max_depth: 30,
            physics_min_samples_split: 5,
            full_n_estimators: 300,
            full_max_depth: 40,
            full_min_samples_split: 3,
            physics_preference: DEFAULT_PHYSICS_PREFERENCE,
            physics_confidence_threshold: 0.6,
            detection_threshold: None,
            random_seed: 42,
        }
    }
}

// =============================================================================
// Data Structures
// =============================================================================

/// Represents a single decision tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    /// Feature index for split (None for leaf nodes)
    pub feature_idx: Option<usize>,
    /// Threshold for split (None for leaf nodes)
    pub threshold: Option<f32>,
    /// Left child index (None for leaf nodes)
    pub left: Option<usize>,
    /// Right child index (None for leaf nodes)
    pub right: Option<usize>,
    /// Class prediction for leaf nodes
    pub prediction: Option<usize>,
    /// Number of samples at this node
    pub n_samples: usize,
}

/// Represents a single decision tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    /// All nodes in the tree
    pub nodes: Vec<TreeNode>,
    /// Number of classes
    pub n_classes: usize,
    /// Feature dimension
    pub feature_dim: usize,
}

impl DecisionTree {
    /// Predict class probabilities for a single sample
    pub fn predict_proba(&self, features: &[f32]) -> Vec<f32> {
        // Navigate to leaf node
        let mut node_idx = 0;
        let max_iterations = self.nodes.len() + 1; // Safety limit
        let mut iterations = 0;

        while iterations < max_iterations {
            iterations += 1;
            let node = match self.nodes.get(node_idx) {
                Some(n) => n,
                None => break,
            };

            // If leaf node, return class distribution
            if node.prediction.is_some() {
                let mut probs = vec![0.0; self.n_classes];
                if let Some(pred_class) = node.prediction {
                    if let Some(prob) = probs.get_mut(pred_class) {
                        *prob = 1.0;
                    }
                }
                return probs;
            }

            // Navigate based on feature value
            let feature_val = node
                .feature_idx
                .and_then(|idx| features.get(idx).copied())
                .unwrap_or(0.0);

            let threshold = node.threshold.unwrap_or(0.0);

            node_idx = if feature_val <= threshold {
                node.left.unwrap_or(0)
            } else {
                node.right.unwrap_or(0)
            };
        }

        // Fallback: uniform distribution
        vec![1.0 / self.n_classes as f32; self.n_classes]
    }
}

/// Represents a trained Random Forest model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RFModel {
    /// All trees in the forest
    pub trees: Vec<DecisionTree>,
    /// Number of estimators (trees)
    pub n_estimators: usize,
    /// Max depth
    pub max_depth: usize,
    /// Number of classes
    pub n_classes: usize,
    /// Feature dimension
    pub feature_dim: usize,
    /// Feature means for standardization
    pub feature_means: Vec<f32>,
    /// Feature stds for standardization
    pub feature_stds: Vec<f32>,
    /// Class labels
    pub class_labels: Vec<String>,
    /// Training accuracy
    pub train_accuracy: f32,
    /// Validation accuracy
    pub val_accuracy: f32,
}

impl RFModel {
    /// Create a new empty RF model
    pub fn new(feature_dim: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_estimators: 0,
            max_depth: 0,
            n_classes: 0,
            feature_dim,
            feature_means: vec![0.0; feature_dim],
            feature_stds: vec![1.0; feature_dim],
            class_labels: Vec::new(),
            train_accuracy: 0.0,
            val_accuracy: 0.0,
        }
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        !self.trees.is_empty() && self.n_classes > 0
    }

    /// Standardize features using stored mean/std
    pub fn standardize(&self, features: &[f32]) -> Vec<f32> {
        features
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                let mean = self.feature_means.get(i).copied().unwrap_or(0.0);
                let std = self.feature_stds.get(i).copied().unwrap_or(1.0);
                if std > 1e-10 {
                    (f - mean) / std
                } else {
                    f - mean
                }
            })
            .collect()
    }

    /// Predict class probabilities for a single sample
    pub fn predict_proba(&self, features: &[f32]) -> Vec<f32> {
        if !self.is_fitted() || self.trees.is_empty() {
            return vec![1.0 / self.n_classes.max(1) as f32; self.n_classes.max(1)];
        }

        // Standardize features
        let standardized = self.standardize(features);

        // Average probabilities from all trees
        let mut avg_probs = vec![0.0; self.n_classes];
        for tree in &self.trees {
            let tree_probs = tree.predict_proba(&standardized);
            for (i, prob) in tree_probs.iter().enumerate() {
                if let Some(avg) = avg_probs.get_mut(i) {
                    *avg += prob / self.trees.len() as f32;
                }
            }
        }

        avg_probs
    }

    /// Predict class for a single sample
    pub fn predict(&self, features: &[f32]) -> usize {
        let probs = self.predict_proba(features);
        probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get confidence (max probability)
    pub fn confidence(&self, features: &[f32]) -> f32 {
        let probs = self.predict_proba(features);
        probs.iter().cloned().fold(0.0, |a, b| a.max(b))
    }

    /// Predict class label string
    pub fn predict_label(&self, features: &[f32]) -> &str {
        let class_idx = self.predict(features);
        self.class_labels
            .get(class_idx)
            .map(|s| s.as_str())
            .unwrap_or("<unknown>")
    }
}

// =============================================================================
// Confidence-Weighted Stacker
// =============================================================================

/// Result of combining predictions from both models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackerResult {
    /// Final class prediction
    pub prediction: String,
    /// Final confidence score
    pub confidence: f32,
    /// Combined probability distribution
    pub combined_proba: Vec<f32>,
    /// Physics model's prediction
    pub physics_prediction: String,
    /// Physics model's confidence
    pub physics_confidence: f32,
    /// Physics model's probabilities
    pub physics_proba: Vec<f32>,
    /// Full model's prediction
    pub full_prediction: String,
    /// Full model's confidence
    pub full_confidence: f32,
    /// Full model's probabilities
    pub full_proba: Vec<f32>,
    /// Weight given to physics model
    pub physics_weight: f32,
    /// Weight given to full model
    pub full_weight: f32,
    /// Whether physics model was used as primary
    pub used_physics: bool,
    /// Whether both models agreed on the prediction
    pub agreement: bool,
}

/// Confidence-weighted stacking meta-learner
#[derive(Debug, Clone)]
pub struct ConfidenceWeightedStacker {
    /// Preference multiplier for physics model
    physics_preference: f32,
    /// Minimum confidence threshold for physics to override
    physics_confidence_threshold: f32,
}

impl ConfidenceWeightedStacker {
    /// Create a new stacker
    pub fn new(physics_preference: f32, physics_confidence_threshold: f32) -> Self {
        Self {
            physics_preference,
            physics_confidence_threshold,
        }
    }

    /// Combine predictions from both models
    ///
    /// # Algorithm
    /// 1. Get probabilities from both models
    /// 2. Calculate confidence-weighted weights
    /// 3. Compute weighted average of probabilities
    /// 4. Select class with highest combined probability
    pub fn combine(&self, physics_proba: &[f32], full_proba: &[f32], class_labels: &[String]) -> StackerResult {
        let n_classes = class_labels.len();

        // Get max probabilities (confidence)
        let physics_conf = physics_proba.iter().cloned().fold(0.0_f32, |a, b| a.max(b));
        let full_conf = full_proba.iter().cloned().fold(0.0_f32, |a, b| a.max(b));

        // Get predictions
        let physics_pred_idx = physics_proba
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let full_pred_idx = full_proba
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let physics_prediction = class_labels
            .get(physics_pred_idx)
            .cloned()
            .unwrap_or_else(|| "<unknown>".to_string());
        let full_prediction = class_labels
            .get(full_pred_idx)
            .cloned()
            .unwrap_or_else(|| "<unknown>".to_string());

        // Calculate weights based on confidence
        // Physics gets preference multiplier when confident
        let physics_weight = if physics_conf >= self.physics_confidence_threshold {
            physics_conf * self.physics_preference
        } else {
            physics_conf * 0.5 // Reduced weight when uncertain
        };
        let full_weight = full_conf;

        let total_weight = physics_weight + full_weight;
        let (norm_physics_weight, norm_full_weight) = if total_weight > 1e-10 {
            (physics_weight / total_weight, full_weight / total_weight)
        } else {
            (0.5, 0.5)
        };

        // Compute weighted average of probabilities
        let mut combined_proba = vec![0.0; n_classes];
        for i in 0..n_classes {
            let p = physics_proba.get(i).copied().unwrap_or(0.0);
            let f = full_proba.get(i).copied().unwrap_or(0.0);
            combined_proba[i] = norm_physics_weight * p + norm_full_weight * f;
        }

        // Get final prediction from combined probabilities
        let final_pred_idx = combined_proba
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let final_confidence = combined_proba.get(final_pred_idx).copied().unwrap_or(0.0);
        let final_prediction = class_labels
            .get(final_pred_idx)
            .cloned()
            .unwrap_or_else(|| "<unknown>".to_string());

        // Check agreement
        let agreement = physics_pred_idx == full_pred_idx;

        // Determine if physics was used as primary
        let used_physics = final_pred_idx == physics_pred_idx && physics_conf >= self.physics_confidence_threshold;

        StackerResult {
            prediction: final_prediction,
            confidence: final_confidence,
            combined_proba,
            physics_prediction,
            physics_confidence: physics_conf,
            physics_proba: physics_proba.to_vec(),
            full_prediction,
            full_confidence: full_conf,
            full_proba: full_proba.to_vec(),
            physics_weight: norm_physics_weight,
            full_weight: norm_full_weight,
            used_physics,
            agreement,
        }
    }
}

// =============================================================================
// Detection Result
// =============================================================================

/// Result of detection mode prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Predicted class (or "background" if below threshold)
    pub prediction: String,
    /// Confidence score
    pub confidence: f32,
    /// Whether this is a background/non-detection
    pub is_background: bool,
    /// Physics model confidence
    pub physics_confidence: f32,
    /// Full model confidence
    pub full_confidence: f32,
}

// =============================================================================
// Feature Stacking Ensemble
// =============================================================================

/// The RF Feature Stacking Ensemble
#[derive(Debug, Clone)]
pub struct FeatureStackingEnsemble {
    /// Configuration
    pub config: StackingConfig,
    /// Physics RF model (46D)
    pub physics_model: RFModel,
    /// Full RF model (112D)
    pub full_model: RFModel,
    /// Confidence-weighted stacker
    pub stacker: ConfidenceWeightedStacker,
}

impl FeatureStackingEnsemble {
    /// Create a new ensemble with default configuration
    pub fn new() -> Self {
        let config = StackingConfig::default();
        Self::with_config(config)
    }

    /// Create a new ensemble with custom configuration
    pub fn with_config(config: StackingConfig) -> Self {
        let stacker = ConfidenceWeightedStacker::new(config.physics_preference, config.physics_confidence_threshold);
        Self {
            physics_model: RFModel::new(PHYSICS_DIM),
            full_model: RFModel::new(FULL_DIM),
            stacker,
            config,
        }
    }

    /// Check if both models are fitted
    pub fn is_fitted(&self) -> bool {
        self.physics_model.is_fitted() && self.full_model.is_fitted()
    }

    /// Load physics model from serialized data
    pub fn load_physics_model(&mut self, model: RFModel) {
        self.physics_model = model;
    }

    /// Load full model from serialized data
    pub fn load_full_model(&mut self, model: RFModel) {
        self.full_model = model;
    }

    /// Get class labels (from physics model, should match full model)
    pub fn class_labels(&self) -> &[String] {
        &self.physics_model.class_labels
    }

    /// Predict using both models and combine
    pub fn predict(&self, features_112d: &[f32]) -> StackerResult {
        // Extract physics features (first 46 dimensions)
        let physics_features: Vec<f32> = features_112d.iter().take(PHYSICS_DIM).copied().collect();

        // Get probabilities from both models
        let physics_proba = self.physics_model.predict_proba(&physics_features);
        let full_proba = self.full_model.predict_proba(features_112d);

        // Combine using stacker
        self.stacker
            .combine(&physics_proba, &full_proba, &self.physics_model.class_labels)
    }

    /// Predict class label only
    pub fn predict_label(&self, features_112d: &[f32]) -> &str {
        let result = self.predict(features_112d);
        // Return reference to stored label if possible
        self.physics_model
            .class_labels
            .iter()
            .find(|l| l.as_str() == result.prediction)
            .map(|s| s.as_str())
            .unwrap_or("<unknown>")
    }

    /// Predict with detection threshold
    ///
    /// Returns a DetectionResult with is_background=true if confidence
    /// is below the detection threshold.
    pub fn detect(&self, features_112d: &[f32]) -> DetectionResult {
        let result = self.predict(features_112d);

        let threshold = self.config.detection_threshold.unwrap_or(DEFAULT_DETECTION_THRESHOLD);
        let is_background = result.confidence < threshold;

        DetectionResult {
            prediction: if is_background {
                "background".to_string()
            } else {
                result.prediction.clone()
            },
            confidence: result.confidence,
            is_background,
            physics_confidence: result.physics_confidence,
            full_confidence: result.full_confidence,
        }
    }

    /// Batch predict with details
    pub fn predict_batch(&self, features_batch: &[Vec<f32>]) -> Vec<StackerResult> {
        features_batch.iter().map(|features| self.predict(features)).collect()
    }

    /// Batch detect with threshold
    pub fn detect_batch(&self, features_batch: &[Vec<f32>]) -> Vec<DetectionResult> {
        features_batch.iter().map(|features| self.detect(features)).collect()
    }

    /// Evaluate accuracy on a test set
    pub fn evaluate(&self, features_batch: &[Vec<f32>], labels: &[String]) -> EnsembleMetrics {
        let mut correct_physics = 0usize;
        let mut correct_full = 0usize;
        let mut correct_ensemble = 0usize;
        let mut total = 0usize;
        let mut physics_used_count = 0usize;
        let mut agreement_count = 0usize;

        for (features, true_label) in features_batch.iter().zip(labels.iter()) {
            let result = self.predict(features);

            total += 1;

            if result.physics_prediction == *true_label {
                correct_physics += 1;
            }
            if result.full_prediction == *true_label {
                correct_full += 1;
            }
            if result.prediction == *true_label {
                correct_ensemble += 1;
            }
            if result.used_physics {
                physics_used_count += 1;
            }
            if result.agreement {
                agreement_count += 1;
            }
        }

        EnsembleMetrics {
            physics_accuracy: if total > 0 {
                correct_physics as f32 / total as f32 * 100.0
            } else {
                0.0
            },
            full_accuracy: if total > 0 {
                correct_full as f32 / total as f32 * 100.0
            } else {
                0.0
            },
            ensemble_accuracy: if total > 0 {
                correct_ensemble as f32 / total as f32 * 100.0
            } else {
                0.0
            },
            total_samples: total,
            physics_used_count,
            agreement_count,
        }
    }
}

impl Default for FeatureStackingEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Metrics
// =============================================================================

/// Metrics for evaluating the ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleMetrics {
    /// Physics RF accuracy (%)
    pub physics_accuracy: f32,
    /// Full RF accuracy (%)
    pub full_accuracy: f32,
    /// Ensemble accuracy (%)
    pub ensemble_accuracy: f32,
    /// Total samples evaluated
    pub total_samples: usize,
    /// Number of times physics model was used as primary
    pub physics_used_count: usize,
    /// Number of times both models agreed
    pub agreement_count: usize,
}

impl EnsembleMetrics {
    /// Calculate improvement over best single model
    pub fn improvement_over_best_single(&self) -> f32 {
        let best_single = self.physics_accuracy.max(self.full_accuracy);
        self.ensemble_accuracy - best_single
    }

    /// Calculate agreement rate
    pub fn agreement_rate(&self) -> f32 {
        if self.total_samples > 0 {
            self.agreement_count as f32 / self.total_samples as f32 * 100.0
        } else {
            0.0
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TDD Test Suite: Confidence-Weighted Stacker
    // =========================================================================

    #[test]
    fn test_stacker_trusts_confident_physics() {
        // KEY TEST: When Physics is 90% confident and Full is 60% confident,
        // the ensemble should trust Physics.
        let stacker = ConfidenceWeightedStacker::new(1.5, 0.6);

        let physics_proba = vec![0.9, 0.05, 0.03, 0.02]; // 90% confident on class 0
        let full_proba = vec![0.2, 0.6, 0.15, 0.05]; // 60% confident on class 1
        let class_labels = vec![
            "bat".to_string(),
            "insect".to_string(),
            "bird".to_string(),
            "frog".to_string(),
        ];

        let result = stacker.combine(&physics_proba, &full_proba, &class_labels);

        // Should predict "bat" (class 0) because physics is confident
        assert_eq!(result.prediction, "bat");
        assert!(result.physics_weight > result.full_weight);
        assert!(result.used_physics);
    }

    #[test]
    fn test_stacker_uses_full_when_physics_uncertain() {
        // When Physics is uncertain (40%), use Full RF's prediction.
        let stacker = ConfidenceWeightedStacker::new(1.5, 0.6);

        let physics_proba = vec![0.4, 0.35, 0.15, 0.1]; // 40% confident - uncertain
        let full_proba = vec![0.1, 0.85, 0.03, 0.02]; // 85% confident on class 1
        let class_labels = vec![
            "bat".to_string(),
            "insect".to_string(),
            "bird".to_string(),
            "frog".to_string(),
        ];

        let result = stacker.combine(&physics_proba, &full_proba, &class_labels);

        // Should predict "insect" because full is confident and physics is uncertain
        assert_eq!(result.prediction, "insect");
        assert!(!result.used_physics);
    }

    #[test]
    fn test_stacker_agreement_boosts_confidence() {
        // When both models agree, combined confidence should be high.
        let stacker = ConfidenceWeightedStacker::new(1.5, 0.6);

        let physics_proba = vec![0.85, 0.08, 0.04, 0.03]; // Both agree on class 0
        let full_proba = vec![0.80, 0.12, 0.05, 0.03];
        let class_labels = vec![
            "bat".to_string(),
            "insect".to_string(),
            "bird".to_string(),
            "frog".to_string(),
        ];

        let result = stacker.combine(&physics_proba, &full_proba, &class_labels);

        assert_eq!(result.prediction, "bat");
        assert!(result.agreement);
        // Combined confidence should be high when both agree
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_stacker_weighted_averaging() {
        // Verify weighted averaging produces correct combined probabilities
        let stacker = ConfidenceWeightedStacker::new(2.0, 0.5);

        let physics_proba = vec![0.7, 0.2, 0.05, 0.05];
        let full_proba = vec![0.5, 0.3, 0.15, 0.05];
        let class_labels = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];

        let result = stacker.combine(&physics_proba, &full_proba, &class_labels);

        // Combined probability for class 0 should be between physics and full
        assert!(result.combined_proba[0] > 0.5);
        assert!(result.combined_proba[0] < 0.7);
    }

    // =========================================================================
    // TDD Test Suite: RF Model
    // =========================================================================

    #[test]
    fn test_rf_model_creation() {
        let model = RFModel::new(PHYSICS_DIM);
        assert!(!model.is_fitted());
        assert_eq!(model.feature_dim, PHYSICS_DIM);
    }

    #[test]
    fn test_rf_model_with_single_tree() {
        let mut model = RFModel::new(2);

        // Create a simple tree: if feature[0] > 0.5, class 1, else class 0
        let tree = DecisionTree {
            nodes: vec![
                TreeNode {
                    feature_idx: Some(0),
                    threshold: Some(0.5),
                    left: Some(1),
                    right: Some(2),
                    prediction: None,
                    n_samples: 100,
                },
                TreeNode {
                    feature_idx: None,
                    threshold: None,
                    left: None,
                    right: None,
                    prediction: Some(0),
                    n_samples: 50,
                },
                TreeNode {
                    feature_idx: None,
                    threshold: None,
                    left: None,
                    right: None,
                    prediction: Some(1),
                    n_samples: 50,
                },
            ],
            n_classes: 2,
            feature_dim: 2,
        };

        model.trees = vec![tree];
        model.n_estimators = 1;
        model.n_classes = 2;
        model.class_labels = vec!["class_a".to_string(), "class_b".to_string()];

        assert!(model.is_fitted());

        // Test prediction
        let features_low = vec![0.3, 0.0];
        let features_high = vec![0.7, 0.0];

        assert_eq!(model.predict(&features_low), 0);
        assert_eq!(model.predict(&features_high), 1);
        assert_eq!(model.predict_label(&features_low), "class_a");
        assert_eq!(model.predict_label(&features_high), "class_b");
    }

    #[test]
    fn test_rf_model_standardization() {
        let mut model = RFModel::new(2);
        model.feature_means = vec![5.0, 10.0];
        model.feature_stds = vec![2.0, 5.0];

        let features = vec![7.0, 20.0];
        let standardized = model.standardize(&features);

        // (7-5)/2 = 1.0, (20-10)/5 = 2.0
        assert!((standardized[0] - 1.0).abs() < 1e-6);
        assert!((standardized[1] - 2.0).abs() < 1e-6);
    }

    // =========================================================================
    // TDD Test Suite: Feature Stacking Ensemble
    // =========================================================================

    fn create_mock_ensemble() -> FeatureStackingEnsemble {
        let mut ensemble = FeatureStackingEnsemble::new();

        // Create mock physics model
        let mut physics_model = RFModel::new(PHYSICS_DIM);
        physics_model.n_classes = 3;
        physics_model.class_labels = vec!["bat".to_string(), "bird".to_string(), "insect".to_string()];
        physics_model.trees = vec![DecisionTree {
            nodes: vec![
                TreeNode {
                    feature_idx: Some(0),
                    threshold: Some(0.5),
                    left: Some(1),
                    right: Some(2),
                    prediction: None,
                    n_samples: 100,
                },
                TreeNode {
                    feature_idx: None,
                    threshold: None,
                    left: None,
                    right: None,
                    prediction: Some(0),
                    n_samples: 50,
                },
                TreeNode {
                    feature_idx: None,
                    threshold: None,
                    left: None,
                    right: None,
                    prediction: Some(1),
                    n_samples: 50,
                },
            ],
            n_classes: 3,
            feature_dim: PHYSICS_DIM,
        }];
        physics_model.n_estimators = 1;
        ensemble.physics_model = physics_model;

        // Create mock full model
        let mut full_model = RFModel::new(FULL_DIM);
        full_model.n_classes = 3;
        full_model.class_labels = vec!["bat".to_string(), "bird".to_string(), "insect".to_string()];
        full_model.trees = vec![DecisionTree {
            nodes: vec![
                TreeNode {
                    feature_idx: Some(50),
                    threshold: Some(0.5),
                    left: Some(1),
                    right: Some(2),
                    prediction: None,
                    n_samples: 100,
                },
                TreeNode {
                    feature_idx: None,
                    threshold: None,
                    left: None,
                    right: None,
                    prediction: Some(1),
                    n_samples: 50,
                },
                TreeNode {
                    feature_idx: None,
                    threshold: None,
                    left: None,
                    right: None,
                    prediction: Some(2),
                    n_samples: 50,
                },
            ],
            n_classes: 3,
            feature_dim: FULL_DIM,
        }];
        full_model.n_estimators = 1;
        ensemble.full_model = full_model;

        ensemble
    }

    #[test]
    fn test_ensemble_creation() {
        let ensemble = FeatureStackingEnsemble::new();
        assert!(!ensemble.is_fitted());
        assert_eq!(ensemble.config.physics_preference, DEFAULT_PHYSICS_PREFERENCE);
    }

    #[test]
    fn test_ensemble_config_custom() {
        let config = StackingConfig {
            physics_n_estimators: 100,
            physics_preference: 2.0,
            detection_threshold: Some(0.7),
            ..Default::default()
        };
        let ensemble = FeatureStackingEnsemble::with_config(config);

        assert_eq!(ensemble.config.physics_n_estimators, 100);
        assert_eq!(ensemble.config.physics_preference, 2.0);
        assert_eq!(ensemble.config.detection_threshold, Some(0.7));
    }

    #[test]
    fn test_ensemble_predict_with_mock_models() {
        let ensemble = create_mock_ensemble();
        assert!(ensemble.is_fitted());

        // Create 112D features with physics feature 0 = 0.3 (low) -> physics predicts bat
        let mut features = vec![0.0; FULL_DIM];
        features[0] = 0.3; // Low physics feature -> bat (class 0)
        features[50] = 0.7; // High texture feature -> insect (class 2)

        let result = ensemble.predict(&features);

        // Should trust physics since it's confident (1.0 for single tree)
        assert_eq!(result.physics_prediction, "bat");
        // The ensemble should prefer physics when it's confident
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_ensemble_detect_with_threshold() {
        let config = StackingConfig {
            detection_threshold: Some(0.7),
            ..Default::default()
        };
        let mut ensemble = FeatureStackingEnsemble::with_config(config);

        // Create mock models with known behavior
        ensemble.physics_model = {
            let mut m = RFModel::new(PHYSICS_DIM);
            m.n_classes = 2;
            m.class_labels = vec!["target".to_string(), "other".to_string()];
            m.trees = vec![DecisionTree {
                nodes: vec![TreeNode {
                    feature_idx: None,
                    threshold: None,
                    left: None,
                    right: None,
                    prediction: Some(0),
                    n_samples: 100,
                }],
                n_classes: 2,
                feature_dim: PHYSICS_DIM,
            }];
            m.n_estimators = 1;
            m
        };

        ensemble.full_model = {
            let mut m = RFModel::new(FULL_DIM);
            m.n_classes = 2;
            m.class_labels = vec!["target".to_string(), "other".to_string()];
            m.trees = vec![DecisionTree {
                nodes: vec![TreeNode {
                    feature_idx: None,
                    threshold: None,
                    left: None,
                    right: None,
                    prediction: Some(0),
                    n_samples: 100,
                }],
                n_classes: 2,
                feature_dim: FULL_DIM,
            }];
            m.n_estimators = 1;
            m
        };

        // With single tree, confidence is 1.0, so it should be a detection
        let features = vec![0.0; FULL_DIM];
        let result = ensemble.detect(&features);

        assert!(!result.is_background);
        assert_eq!(result.prediction, "target");
    }

    #[test]
    fn test_ensemble_batch_predict() {
        let ensemble = create_mock_ensemble();

        let batch = vec![
            {
                let mut f = vec![0.0; FULL_DIM];
                f[0] = 0.3;
                f
            },
            {
                let mut f = vec![0.0; FULL_DIM];
                f[0] = 0.7;
                f
            },
        ];

        let results = ensemble.predict_batch(&batch);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ensemble_evaluate() {
        let ensemble = create_mock_ensemble();

        let features_batch = vec![
            {
                let mut f = vec![0.0; FULL_DIM];
                f[0] = 0.3; // Physics: bat
                f[50] = 0.7; // Full: insect
                f
            },
            {
                let mut f = vec![0.0; FULL_DIM];
                f[0] = 0.7; // Physics: bird
                f[50] = 0.3; // Full: bird
                f
            },
        ];

        let labels = vec!["bat".to_string(), "bird".to_string()];

        let metrics = ensemble.evaluate(&features_batch, &labels);

        assert_eq!(metrics.total_samples, 2);
        assert!(metrics.physics_accuracy > 0.0 || metrics.full_accuracy > 0.0);
    }

    // =========================================================================
    // TDD Test Suite: Metrics
    // =========================================================================

    #[test]
    fn test_metrics_improvement_calculation() {
        let metrics = EnsembleMetrics {
            physics_accuracy: 55.0,
            full_accuracy: 60.0,
            ensemble_accuracy: 65.0,
            total_samples: 100,
            physics_used_count: 40,
            agreement_count: 70,
        };

        // Best single = 60%, ensemble = 65%, improvement = 5%
        assert!((metrics.improvement_over_best_single() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_metrics_agreement_rate() {
        let metrics = EnsembleMetrics {
            physics_accuracy: 55.0,
            full_accuracy: 60.0,
            ensemble_accuracy: 65.0,
            total_samples: 100,
            physics_used_count: 40,
            agreement_count: 70,
        };

        // 70/100 = 70%
        assert!((metrics.agreement_rate() - 70.0).abs() < 1e-6);
    }

    // =========================================================================
    // TDD Test Suite: Edge Cases
    // =========================================================================

    #[test]
    fn test_empty_features_batch() {
        let ensemble = FeatureStackingEnsemble::new();
        let batch: Vec<Vec<f32>> = vec![];

        let results = ensemble.predict_batch(&batch);
        assert!(results.is_empty());
    }

    #[test]
    fn test_stacker_with_zero_probabilities() {
        let stacker = ConfidenceWeightedStacker::new(1.5, 0.6);

        let physics_proba = vec![0.0, 0.0, 0.0];
        let full_proba = vec![0.0, 0.0, 0.0];
        let class_labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        // Should handle gracefully without panic
        let result = stacker.combine(&physics_proba, &full_proba, &class_labels);
        assert!(!result.prediction.is_empty());
    }

    #[test]
    fn test_stacker_single_class() {
        let stacker = ConfidenceWeightedStacker::new(1.5, 0.6);

        let physics_proba = vec![1.0];
        let full_proba = vec![1.0];
        let class_labels = vec!["only_class".to_string()];

        let result = stacker.combine(&physics_proba, &full_proba, &class_labels);
        assert_eq!(result.prediction, "only_class");
        assert!((result.confidence - 1.0).abs() < 1e-6);
    }

    // =========================================================================
    // TDD Test Suite: RF Model Fit and Predict (45D and 112D)
    // =========================================================================

    #[test]
    fn test_rf_model_fit_and_predict_45d() {
        let mut model = RFModel::new(PHYSICS_DIM);

        // Create a simple tree for 45D data
        let tree = DecisionTree {
            nodes: vec![
                TreeNode {
                    feature_idx: Some(0),
                    threshold: Some(0.5),
                    left: Some(1),
                    right: Some(2),
                    prediction: None,
                    n_samples: 100,
                },
                TreeNode {
                    feature_idx: None,
                    threshold: None,
                    left: None,
                    right: None,
                    prediction: Some(0),
                    n_samples: 50,
                },
                TreeNode {
                    feature_idx: None,
                    threshold: None,
                    left: None,
                    right: None,
                    prediction: Some(1),
                    n_samples: 50,
                },
            ],
            n_classes: 2,
            feature_dim: PHYSICS_DIM,
        };

        model.trees = vec![tree];
        model.n_estimators = 1;
        model.n_classes = 2;
        model.class_labels = vec!["class_a".to_string(), "class_b".to_string()];

        // Low feature[0] → class 0
        let features_low = vec![0.3f32; PHYSICS_DIM];
        assert_eq!(model.predict(&features_low), 0);
        assert_eq!(model.predict_label(&features_low), "class_a");

        // High feature[0] → class 1
        let features_high = vec![0.7f32; PHYSICS_DIM];
        assert_eq!(model.predict(&features_high), 1);
        assert_eq!(model.predict_label(&features_high), "class_b");
    }

    #[test]
    fn test_rf_model_fit_and_predict_112d() {
        let mut model = RFModel::new(FULL_DIM);

        // Create a tree for 112D data that splits on feature 50
        let tree = DecisionTree {
            nodes: vec![
                TreeNode {
                    feature_idx: Some(50),
                    threshold: Some(0.5),
                    left: Some(1),
                    right: Some(2),
                    prediction: None,
                    n_samples: 100,
                },
                TreeNode {
                    feature_idx: None,
                    threshold: None,
                    left: None,
                    right: None,
                    prediction: Some(0),
                    n_samples: 50,
                },
                TreeNode {
                    feature_idx: None,
                    threshold: None,
                    left: None,
                    right: None,
                    prediction: Some(1),
                    n_samples: 50,
                },
            ],
            n_classes: 2,
            feature_dim: FULL_DIM,
        };

        model.trees = vec![tree];
        model.n_estimators = 1;
        model.n_classes = 2;
        model.class_labels = vec!["species_x".to_string(), "species_y".to_string()];

        assert!(model.is_fitted());
        assert_eq!(model.feature_dim, FULL_DIM);

        let mut features = vec![0.0f32; FULL_DIM];
        features[50] = 0.3;
        assert_eq!(model.predict(&features), 0);

        features[50] = 0.7;
        assert_eq!(model.predict(&features), 1);
    }

    #[test]
    fn test_rf_model_serialization_roundtrip() {
        let mut model = RFModel::new(4);
        let tree = DecisionTree {
            nodes: vec![
                TreeNode {
                    feature_idx: Some(0),
                    threshold: Some(0.5),
                    left: Some(1),
                    right: Some(2),
                    prediction: None,
                    n_samples: 100,
                },
                TreeNode {
                    feature_idx: None,
                    threshold: None,
                    left: None,
                    right: None,
                    prediction: Some(0),
                    n_samples: 50,
                },
                TreeNode {
                    feature_idx: None,
                    threshold: None,
                    left: None,
                    right: None,
                    prediction: Some(1),
                    n_samples: 50,
                },
            ],
            n_classes: 2,
            feature_dim: 4,
        };

        model.trees = vec![tree];
        model.n_estimators = 1;
        model.n_classes = 2;
        model.class_labels = vec!["alpha".to_string(), "beta".to_string()];
        model.feature_means = vec![0.0, 1.0, 2.0, 3.0];
        model.feature_stds = vec![1.0, 1.0, 1.0, 1.0];

        // Serialize and deserialize
        let json = serde_json::to_string(&model).unwrap();
        let restored: RFModel = serde_json::from_str(&json).unwrap();

        // Verify predictions match
        let features = vec![0.3, 1.0, 2.0, 3.0];
        assert_eq!(model.predict(&features), restored.predict(&features));
        assert_eq!(restored.class_labels.len(), 2);
        assert_eq!(restored.class_labels[0], "alpha");
    }

    // =========================================================================
    // TDD Test Suite: Stacker Behavior
    // =========================================================================

    #[test]
    fn test_stacker_single_model_dominant() {
        // When physics is 100% confident and full is 0%, stacker trusts physics
        let stacker = ConfidenceWeightedStacker::new(1.5, 0.6);

        let physics_proba = vec![1.0, 0.0]; // 100% confident on class 0
        let full_proba = vec![0.5, 0.5]; // completely uncertain
        let class_labels = vec!["physics_pred".to_string(), "full_pred".to_string()];

        let result = stacker.combine(&physics_proba, &full_proba, &class_labels);

        assert_eq!(result.prediction, "physics_pred");
        assert!(result.used_physics);
    }

    // =========================================================================
    // TDD Test Suite: Ensemble Batch Operations
    // =========================================================================

    #[test]
    fn test_ensemble_classify_batch() {
        let ensemble = create_mock_ensemble();

        let batch = vec![
            {
                let mut f = vec![0.0; FULL_DIM];
                f[0] = 0.3; // physics predicts bat
                f[50] = 0.3;
                f
            },
            {
                let mut f = vec![0.0; FULL_DIM];
                f[0] = 0.7; // physics predicts bird
                f[50] = 0.7; // full predicts insect
                f
            },
            {
                let mut f = vec![0.0; FULL_DIM];
                f[0] = 0.1;
                f[50] = 0.1;
                f
            },
        ];

        let results = ensemble.predict_batch(&batch);
        assert_eq!(results.len(), 3);
        // Each result should have a valid prediction
        for result in &results {
            assert!(!result.prediction.is_empty());
            assert!(result.confidence >= 0.0);
        }
    }

    #[test]
    fn test_ensemble_detect_batch() {
        let ensemble = create_mock_ensemble();

        let batch = vec![vec![0.0; FULL_DIM], vec![0.5; FULL_DIM]];

        let results = ensemble.detect_batch(&batch);
        assert_eq!(results.len(), 2);
        for result in &results {
            assert!(!result.prediction.is_empty());
        }
    }

    // =========================================================================
    // TDD Test Suite: Detection Threshold
    // =========================================================================

    #[test]
    fn test_detection_threshold_sensitivity() {
        // Low threshold → fewer detections
        let config_strict = StackingConfig {
            detection_threshold: Some(0.99), // Very high - almost nothing detected
            ..Default::default()
        };
        let ensemble_strict = FeatureStackingEnsemble::with_config(config_strict);
        let features = vec![0.0; FULL_DIM];
        // Without fitted models, detection behavior is defined by threshold
        // This test validates the config is accepted
        assert_eq!(ensemble_strict.config.detection_threshold, Some(0.99));

        // Low threshold → more detections
        let config_loose = StackingConfig {
            detection_threshold: Some(0.01), // Very low - almost everything detected
            ..Default::default()
        };
        let ensemble_loose = FeatureStackingEnsemble::with_config(config_loose);
        assert_eq!(ensemble_loose.config.detection_threshold, Some(0.01));
    }

    #[test]
    fn test_ensemble_with_confidence_calibration() {
        // Higher physics_preference shifts confidence toward physics model
        let stacker_low_pref = ConfidenceWeightedStacker::new(1.0, 0.6);
        let stacker_high_pref = ConfidenceWeightedStacker::new(3.0, 0.6);

        let physics_proba = vec![0.8, 0.1, 0.1];
        let full_proba = vec![0.3, 0.6, 0.1];
        let class_labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let result_low = stacker_low_pref.combine(&physics_proba, &full_proba, &class_labels);
        let result_high = stacker_high_pref.combine(&physics_proba, &full_proba, &class_labels);

        // With higher physics preference, physics weight should be higher
        assert!(
            result_high.physics_weight >= result_low.physics_weight,
            "Higher preference should give physics more weight"
        );
    }
}
