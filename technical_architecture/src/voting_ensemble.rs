//! Voting Ensemble for Species Classification
//! ===========================================
//!
//! Architecture:
//! ```text
//!   INPUT: 112D Feature Vector
//!       │
//! ┌─────┴─────┐
//! ▼           ▼
//! [NN 112D]   [RF 112D]
//! │           │
//! ▼           ▼
//! Top-5       Probability
//! Candidates  Distribution
//! │           │
//! └─────┬─────┘
//!       ▼
//! [Ensemble Voter]
//!       │
//!       ▼
//! FINAL PREDICTION
//! ```
//!
//! Features:
//! - Grid Search for optimal weight optimization
//! - Confidence-based dynamic weighting
//! - Top-K candidate shortlisting from NN

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the Voting Ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingEnsembleConfig {
    /// Base weight for NN (1.0 - weight = RF weight)
    pub nn_weight: f32,
    /// Number of top candidates to consider from NN
    pub top_k: usize,
    /// Enable confidence-based dynamic weighting
    pub dynamic_weighting: bool,
    /// High confidence threshold for NN (increase NN weight above this)
    pub nn_high_confidence_threshold: f32,
    /// Weight boost when NN is confident
    pub nn_confidence_boost: f32,
    /// Low confidence threshold (decrease NN weight below this)
    pub nn_low_confidence_threshold: f32,
    /// Weight penalty when NN is uncertain
    pub nn_confidence_penalty: f32,
}

impl Default for VotingEnsembleConfig {
    fn default() -> Self {
        Self {
            nn_weight: 0.5,
            top_k: 5,
            dynamic_weighting: false,
            nn_high_confidence_threshold: 0.9,
            nn_confidence_boost: 0.3,
            nn_low_confidence_threshold: 0.5,
            nn_confidence_penalty: 0.3,
        }
    }
}

impl VotingEnsembleConfig {
    /// Create config with static weight
    pub fn with_static_weight(nn_weight: f32) -> Self {
        Self {
            nn_weight,
            dynamic_weighting: false,
            ..Default::default()
        }
    }

    /// Create config with dynamic weighting enabled
    pub fn with_dynamic_weighting() -> Self {
        Self {
            dynamic_weighting: true,
            ..Default::default()
        }
    }
}

// =============================================================================
// Prediction Structures
// =============================================================================

/// Single prediction from a model
#[derive(Debug, Clone)]
pub struct ModelPrediction {
    /// Probability distribution over all classes
    pub probabilities: Vec<f32>,
    /// Ground truth label index (for evaluation)
    pub label: usize,
}

impl ModelPrediction {
    pub fn new(probabilities: Vec<f32>, label: usize) -> Self {
        Self { probabilities, label }
    }

    /// Get the top-K predicted class indices
    pub fn top_k_indices(&self, k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = self.probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.iter().take(k).map(|(i, _)| *i).collect()
    }

    /// Get the predicted class (argmax)
    pub fn predicted_class(&self) -> usize {
        self.probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get the confidence (max probability)
    pub fn confidence(&self) -> f32 {
        self.probabilities.iter().cloned().fold(0.0, f32::max)
    }
}

/// Combined prediction pair from NN and RF
#[derive(Debug, Clone)]
pub struct EnsembleInput {
    pub nn_probs: Vec<f32>,
    pub rf_probs: Vec<f32>,
    pub label: usize,
}

impl EnsembleInput {
    pub fn new(nn_probs: Vec<f32>, rf_probs: Vec<f32>, label: usize) -> Self {
        Self { nn_probs, rf_probs, label }
    }

    pub fn nn_prediction(&self) -> ModelPrediction {
        ModelPrediction::new(self.nn_probs.clone(), self.label)
    }

    pub fn rf_prediction(&self) -> ModelPrediction {
        ModelPrediction::new(self.rf_probs.clone(), self.label)
    }
}

// =============================================================================
// Grid Search Optimizer
// =============================================================================

/// Result of grid search optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSearchResult {
    pub optimal_weight: f32,
    pub optimal_accuracy: f64,
    pub weight_accuracy_curve: Vec<(f32, f64)>,
    pub nn_only_accuracy: f64,
    pub rf_only_accuracy: f64,
}

impl GridSearchResult {
    /// Calculate improvement over best single model
    pub fn improvement_over_best_single(&self) -> f64 {
        let best_single = self.nn_only_accuracy.max(self.rf_only_accuracy);
        self.optimal_accuracy - best_single
    }
}

/// Grid Search optimizer for finding optimal ensemble weights
pub struct GridSearchOptimizer {
    /// Step size for weight search
    pub step: f32,
    /// Minimum weight to try
    pub min_weight: f32,
    /// Maximum weight to try
    pub max_weight: f32,
}

impl Default for GridSearchOptimizer {
    fn default() -> Self {
        Self {
            step: 0.05,
            min_weight: 0.0,
            max_weight: 1.0,
        }
    }
}

impl GridSearchOptimizer {
    pub fn new(step: f32) -> Self {
        Self { step, ..Default::default() }
    }

    /// Find optimal NN weight using grid search
    pub fn optimize(&self, validation_data: &[EnsembleInput]) -> GridSearchResult {
        let mut weight_accuracy_curve: Vec<(f32, f64)> = Vec::new();
        let mut best_weight = 0.0;
        let mut best_accuracy = 0.0;

        // Calculate number of steps
        let n_steps = ((self.max_weight - self.min_weight) / self.step) as usize + 1;

        for i in 0..=n_steps {
            let w = (self.min_weight + (i as f32 * self.step)).min(self.max_weight);

            let mut correct = 0;
            for pred in validation_data {
                let predicted = self.predict_with_weight(&pred.nn_probs, &pred.rf_probs, w);
                if predicted == pred.label {
                    correct += 1;
                }
            }

            let accuracy = correct as f64 / validation_data.len() as f64;
            weight_accuracy_curve.push((w, accuracy));

            if accuracy > best_accuracy {
                best_accuracy = accuracy;
                best_weight = w;
            }
        }

        // Calculate baseline accuracies
        let nn_only_accuracy = self.evaluate_single_model(validation_data, 1.0);
        let rf_only_accuracy = self.evaluate_single_model(validation_data, 0.0);

        GridSearchResult {
            optimal_weight: best_weight,
            optimal_accuracy: best_accuracy,
            weight_accuracy_curve,
            nn_only_accuracy,
            rf_only_accuracy,
        }
    }

    /// Predict using a specific weight
    fn predict_with_weight(&self, nn_probs: &[f32], rf_probs: &[f32], nn_weight: f32) -> usize {
        let rf_weight = 1.0 - nn_weight;

        let ensemble_probs: Vec<f32> = nn_probs.iter()
            .zip(rf_probs.iter())
            .map(|(nn_p, rf_p)| (nn_p * nn_weight) + (rf_p * rf_weight))
            .collect();

        ensemble_probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Evaluate single model accuracy
    fn evaluate_single_model(&self, validation_data: &[EnsembleInput], nn_weight: f32) -> f64 {
        let mut correct = 0;
        for pred in validation_data {
            let predicted = self.predict_with_weight(&pred.nn_probs, &pred.rf_probs, nn_weight);
            if predicted == pred.label {
                correct += 1;
            }
        }
        correct as f64 / validation_data.len() as f64
    }

    /// Print grid search results
    pub fn print_results(&self, result: &GridSearchResult) {
        println!("╔═══════════════════════════════════════════════════════════════════╗");
        println!("║  Grid Search Results - Ensemble Weight Optimization              ║");
        println!("╚═══════════════════════════════════════════════════════════════════╝");
        println!();
        println!("Weight | Accuracy");
        println!("-------|---------");

        for (w, acc) in &result.weight_accuracy_curve {
            let marker = if (*w - result.optimal_weight).abs() < 0.01 { " ★" } else { "" };
            println!("{:.2}   | {:.2}%{}", w, acc * 100.0, marker);
        }

        println!();
        println!("╔═══════════════════════════════════════════════════════════════════╗");
        println!("║  Summary                                                         ║");
        println!("╠═══════════════════════════════════════════════════════════════════╣");
        println!("║  NN Only:      {:>6.2}%                                          ║", result.nn_only_accuracy * 100.0);
        println!("║  RF Only:      {:>6.2}%                                          ║", result.rf_only_accuracy * 100.0);
        println!("║  Optimal:      {:>6.2}% (weight={:.2})                           ║",
            result.optimal_accuracy * 100.0, result.optimal_weight);
        println!("║  Improvement:  {:>+6.2}%                                         ║",
            result.improvement_over_best_single() * 100.0);
        println!("╚═══════════════════════════════════════════════════════════════════╝");
    }
}

// =============================================================================
// Ensemble Voter
// =============================================================================

/// The main ensemble voter that combines NN and RF predictions
pub struct EnsembleVoter {
    config: VotingEnsembleConfig,
    optimizer: GridSearchOptimizer,
}

impl EnsembleVoter {
    pub fn new(config: VotingEnsembleConfig) -> Self {
        Self {
            config,
            optimizer: GridSearchOptimizer::default(),
        }
    }

    /// Create voter with default configuration
    pub fn with_defaults() -> Self {
        Self::new(VotingEnsembleConfig::default())
    }

    /// Get configuration
    pub fn config(&self) -> &VotingEnsembleConfig {
        &self.config
    }

    /// Optimize weights using grid search
    pub fn optimize_weights(&self, validation_data: &[EnsembleInput]) -> GridSearchResult {
        let result = self.optimizer.optimize(validation_data);
        self.optimizer.print_results(&result);
        result
    }

    /// Predict using ensemble
    pub fn predict(&self, nn_probs: &[f32], rf_probs: &[f32]) -> usize {
        let weight = if self.config.dynamic_weighting {
            self.calculate_dynamic_weight(nn_probs)
        } else {
            self.config.nn_weight
        };

        self.predict_with_weight(nn_probs, rf_probs, weight)
    }

    /// Predict with confidence score
    pub fn predict_with_confidence(&self, nn_probs: &[f32], rf_probs: &[f32]) -> (usize, f32) {
        let weight = if self.config.dynamic_weighting {
            self.calculate_dynamic_weight(nn_probs)
        } else {
            self.config.nn_weight
        };

        let ensemble_probs = self.fuse_probabilities(nn_probs, rf_probs, weight);

        let (pred_class, confidence) = ensemble_probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &p)| (i, p))
            .unwrap_or((0, 0.0));

        (pred_class, confidence)
    }

    /// Predict using Top-K shortlist from NN
    pub fn predict_with_topk(&self, nn_probs: &[f32], rf_probs: &[f32], k: usize) -> usize {
        let weight = if self.config.dynamic_weighting {
            self.calculate_dynamic_weight(nn_probs)
        } else {
            self.config.nn_weight
        };

        // Get top-K indices from NN
        let mut indexed: Vec<(usize, f32)> = nn_probs.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k_indices: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();

        // Find best among top-K using ensemble
        let rf_weight = 1.0 - weight;
        let mut best_idx = top_k_indices[0];
        let mut best_score = (nn_probs[best_idx] * weight) + (rf_probs[best_idx] * rf_weight);

        for &idx in &top_k_indices[1..] {
            let score = (nn_probs[idx] * weight) + (rf_probs[idx] * rf_weight);
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }

        best_idx
    }

    /// Calculate dynamic weight based on NN confidence
    fn calculate_dynamic_weight(&self, nn_probs: &[f32]) -> f32 {
        let nn_confidence = nn_probs.iter().cloned().fold(0.0f32, f32::max);

        let base_weight = self.config.nn_weight;

        if nn_confidence > self.config.nn_high_confidence_threshold {
            // NN is very confident - increase its weight
            (base_weight + self.config.nn_confidence_boost).min(1.0)
        } else if nn_confidence < self.config.nn_low_confidence_threshold {
            // NN is uncertain - decrease its weight (trust RF more)
            (base_weight - self.config.nn_confidence_penalty).max(0.0)
        } else {
            // Normal confidence - use base weight
            base_weight
        }
    }

    /// Fuse probabilities with given weight
    fn fuse_probabilities(&self, nn_probs: &[f32], rf_probs: &[f32], nn_weight: f32) -> Vec<f32> {
        let rf_weight = 1.0 - nn_weight;
        nn_probs.iter()
            .zip(rf_probs.iter())
            .map(|(nn_p, rf_p)| (nn_p * nn_weight) + (rf_p * rf_weight))
            .collect()
    }

    /// Predict with specific weight
    fn predict_with_weight(&self, nn_probs: &[f32], rf_probs: &[f32], nn_weight: f32) -> usize {
        let ensemble_probs = self.fuse_probabilities(nn_probs, rf_probs, nn_weight);
        ensemble_probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Evaluate ensemble on validation data
    pub fn evaluate(&self, validation_data: &[EnsembleInput]) -> f64 {
        let mut correct = 0;
        for pred in validation_data {
            let predicted = self.predict(&pred.nn_probs, &pred.rf_probs);
            if predicted == pred.label {
                correct += 1;
            }
        }
        correct as f64 / validation_data.len() as f64
    }

    /// Evaluate with Top-K shortlist
    pub fn evaluate_with_topk(&self, validation_data: &[EnsembleInput], k: usize) -> f64 {
        let mut correct = 0;
        for pred in validation_data {
            let predicted = self.predict_with_topk(&pred.nn_probs, &pred.rf_probs, k);
            if predicted == pred.label {
                correct += 1;
            }
        }
        correct as f64 / validation_data.len() as f64
    }
}

// =============================================================================
// Metrics
// =============================================================================

/// Evaluation metrics for ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleMetrics {
    pub accuracy: f64,
    pub top5_accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub nn_weight: f32,
    pub dynamic_weighting: bool,
}

impl EnsembleMetrics {
    pub fn compute(predictions: &[usize], labels: &[usize], num_classes: usize, nn_weight: f32, dynamic: bool) -> Self {
        let n = predictions.len();
        if n == 0 {
            return Self {
                accuracy: 0.0,
                top5_accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                nn_weight,
                dynamic_weighting: dynamic,
            };
        }

        // Accuracy
        let correct = predictions.iter().zip(labels.iter()).filter(|(p, l)| p == l).count();
        let accuracy = correct as f64 / n as f64;

        // Macro-averaged precision and recall
        let mut precision_sum = 0.0;
        let mut recall_sum = 0.0;
        let mut valid_classes = 0;

        for c in 0..num_classes {
            let tp = predictions.iter().zip(labels.iter()).filter(|(p, l)| **p == c && **l == c).count();
            let fp = predictions.iter().zip(labels.iter()).filter(|(p, l)| **p == c && **l != c).count();
            let fn_ = predictions.iter().zip(labels.iter()).filter(|(p, l)| **p != c && **l == c).count();

            let class_precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
            let class_recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };

            let class_count = labels.iter().filter(|&&l| l == c).count();
            if class_count > 0 {
                precision_sum += class_precision;
                recall_sum += class_recall;
                valid_classes += 1;
            }
        }

        let precision = if valid_classes > 0 { precision_sum / valid_classes as f64 } else { 0.0 };
        let recall = if valid_classes > 0 { recall_sum / valid_classes as f64 } else { 0.0 };
        let f1_score = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };

        Self {
            accuracy,
            top5_accuracy: accuracy, // Same for top-1
            precision,
            recall,
            f1_score,
            nn_weight,
            dynamic_weighting: dynamic,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_probabilities(n_classes: usize, correct_class: usize, confidence: f32) -> Vec<f32> {
        let mut probs = vec![0.01; n_classes];
        let remaining = 1.0 - (confidence + 0.01 * (n_classes - 1) as f32);
        let noise_per_class = remaining / (n_classes - 1) as f32;

        for i in 0..n_classes {
            if i == correct_class {
                probs[i] = confidence;
            } else {
                probs[i] = 0.01 + noise_per_class;
            }
        }
        probs
    }

    // ==========================================================================
    // ModelPrediction Tests
    // ==========================================================================

    #[test]
    fn test_model_prediction_top_k() {
        let probs = vec![0.1, 0.5, 0.2, 0.15, 0.05];
        let pred = ModelPrediction::new(probs, 1);

        let top3 = pred.top_k_indices(3);
        assert_eq!(top3, vec![1, 2, 3]); // 0.5, 0.2, 0.15
    }

    #[test]
    fn test_model_prediction_predicted_class() {
        let probs = vec![0.1, 0.7, 0.1, 0.05, 0.05];
        let pred = ModelPrediction::new(probs, 1);

        assert_eq!(pred.predicted_class(), 1);
    }

    #[test]
    fn test_model_prediction_confidence() {
        let probs = vec![0.1, 0.7, 0.1, 0.05, 0.05];
        let pred = ModelPrediction::new(probs, 1);

        assert!((pred.confidence() - 0.7).abs() < 0.001);
    }

    // ==========================================================================
    // GridSearchOptimizer Tests
    // ==========================================================================

    #[test]
    fn test_grid_search_finds_optimal_weight() {
        let optimizer = GridSearchOptimizer::new(0.1);

        // Create validation data where ensemble should beat single models
        let mut data = Vec::new();

        // Case 1: NN correct, RF wrong
        data.push(EnsembleInput::new(
            create_test_probabilities(10, 3, 0.8),  // NN predicts 3 correctly
            create_test_probabilities(10, 5, 0.6),  // RF predicts 5 wrongly
            3,
        ));

        // Case 2: RF correct, NN wrong
        data.push(EnsembleInput::new(
            create_test_probabilities(10, 7, 0.6),  // NN predicts 7 wrongly
            create_test_probabilities(10, 2, 0.8),  // RF predicts 2 correctly
            2,
        ));

        let result = optimizer.optimize(&data);

        // Should find some optimal weight
        assert!(result.optimal_weight >= 0.0 && result.optimal_weight <= 1.0);
        assert!(result.optimal_accuracy > 0.0);
    }

    #[test]
    fn test_grid_search_nn_only_accuracy() {
        let optimizer = GridSearchOptimizer::new(0.25);

        // Create data where NN is always correct
        let data = vec![
            EnsembleInput::new(
                create_test_probabilities(5, 0, 0.9),
                create_test_probabilities(5, 1, 0.8),
                0,
            ),
            EnsembleInput::new(
                create_test_probabilities(5, 2, 0.85),
                create_test_probabilities(5, 3, 0.7),
                2,
            ),
        ];

        let result = optimizer.optimize(&data);

        // NN only (weight=1.0) should be 100% accurate
        assert!((result.nn_only_accuracy - 1.0).abs() < 0.001);
        // RF only should be 0% accurate
        assert!((result.rf_only_accuracy - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_grid_search_rf_only_accuracy() {
        let optimizer = GridSearchOptimizer::new(0.25);

        // Create data where RF is always correct
        let data = vec![
            EnsembleInput::new(
                create_test_probabilities(5, 1, 0.6),  // NN wrong
                create_test_probabilities(5, 0, 0.9),  // RF correct
                0,
            ),
            EnsembleInput::new(
                create_test_probabilities(5, 3, 0.5),  // NN wrong
                create_test_probabilities(5, 2, 0.85), // RF correct
                2,
            ),
        ];

        let result = optimizer.optimize(&data);

        // RF only (weight=0.0) should be 100% accurate
        assert!((result.rf_only_accuracy - 1.0).abs() < 0.001);
        // NN only should be 0% accurate
        assert!((result.nn_only_accuracy - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_grid_search_improvement_calculation() {
        let mut result = GridSearchResult {
            optimal_weight: 0.5,
            optimal_accuracy: 0.75,
            weight_accuracy_curve: vec![],
            nn_only_accuracy: 0.6,
            rf_only_accuracy: 0.65,
        };

        // 0.75 - 0.65 = 0.10 improvement
        assert!((result.improvement_over_best_single() - 0.10).abs() < 0.001);

        // When ensemble is worse than both
        result.optimal_accuracy = 0.5;
        assert!(result.improvement_over_best_single() < 0.0);
    }

    // ==========================================================================
    // EnsembleVoter Tests
    // ==========================================================================

    #[test]
    fn test_ensemble_voter_static_weight() {
        let config = VotingEnsembleConfig::with_static_weight(0.5);
        let voter = EnsembleVoter::new(config);

        let nn_probs = vec![0.1, 0.6, 0.2, 0.1];
        let rf_probs = vec![0.2, 0.3, 0.4, 0.1];

        // Fused: [0.15, 0.45, 0.3, 0.1]
        // Max is index 1
        assert_eq!(voter.predict(&nn_probs, &rf_probs), 1);
    }

    #[test]
    fn test_ensemble_voter_weight_0_trusts_rf() {
        let config = VotingEnsembleConfig::with_static_weight(0.0);
        let voter = EnsembleVoter::new(config);

        let nn_probs = vec![0.9, 0.1];  // NN says 0
        let rf_probs = vec![0.1, 0.9];  // RF says 1

        // Weight 0 means 100% RF
        assert_eq!(voter.predict(&nn_probs, &rf_probs), 1);
    }

    #[test]
    fn test_ensemble_voter_weight_1_trusts_nn() {
        let config = VotingEnsembleConfig::with_static_weight(1.0);
        let voter = EnsembleVoter::new(config);

        let nn_probs = vec![0.9, 0.1];  // NN says 0
        let rf_probs = vec![0.1, 0.9];  // RF says 1

        // Weight 1 means 100% NN
        assert_eq!(voter.predict(&nn_probs, &rf_probs), 0);
    }

    #[test]
    fn test_ensemble_voter_dynamic_weighting_confident_nn() {
        let mut config = VotingEnsembleConfig::with_dynamic_weighting();
        config.nn_weight = 0.5;
        config.nn_high_confidence_threshold = 0.8;
        config.nn_confidence_boost = 0.3;

        let voter = EnsembleVoter::new(config);

        // NN is very confident (0.95 > 0.8)
        let nn_probs = vec![0.95, 0.05];
        let rf_probs = vec![0.4, 0.6];

        // Should trust NN more (weight becomes 0.5 + 0.3 = 0.8)
        // 0.95 * 0.8 + 0.4 * 0.2 = 0.76 + 0.08 = 0.84 for class 0
        // 0.05 * 0.8 + 0.6 * 0.2 = 0.04 + 0.12 = 0.16 for class 1
        assert_eq!(voter.predict(&nn_probs, &rf_probs), 0);
    }

    #[test]
    fn test_ensemble_voter_dynamic_weighting_uncertain_nn() {
        let mut config = VotingEnsembleConfig::with_dynamic_weighting();
        config.nn_weight = 0.5;
        config.nn_low_confidence_threshold = 0.5;
        config.nn_confidence_penalty = 0.3;

        let voter = EnsembleVoter::new(config);

        // NN is uncertain (0.4 < 0.5)
        let nn_probs = vec![0.4, 0.3, 0.3];
        let rf_probs = vec![0.2, 0.7, 0.1];

        // Should trust RF more (weight becomes 0.5 - 0.3 = 0.2)
        // 0.4 * 0.2 + 0.2 * 0.8 = 0.08 + 0.16 = 0.24 for class 0
        // 0.3 * 0.2 + 0.7 * 0.8 = 0.06 + 0.56 = 0.62 for class 1
        // 0.3 * 0.2 + 0.1 * 0.8 = 0.06 + 0.08 = 0.14 for class 2
        assert_eq!(voter.predict(&nn_probs, &rf_probs), 1);
    }

    #[test]
    fn test_ensemble_voter_predict_with_confidence() {
        let config = VotingEnsembleConfig::with_static_weight(0.5);
        let voter = EnsembleVoter::new(config);

        let nn_probs = vec![0.8, 0.1, 0.1];
        let rf_probs = vec![0.7, 0.2, 0.1];

        let (pred_class, confidence) = voter.predict_with_confidence(&nn_probs, &rf_probs);

        assert_eq!(pred_class, 0);
        // Fused: 0.5 * 0.8 + 0.5 * 0.7 = 0.75
        assert!((confidence - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_ensemble_voter_topk_prediction() {
        let config = VotingEnsembleConfig::with_static_weight(0.5);
        let voter = EnsembleVoter::new(config);

        // NN top-3: [4, 1, 3] with probs [0.3, 0.25, 0.2]
        let nn_probs = vec![0.05, 0.25, 0.1, 0.2, 0.3, 0.1];
        // RF favors class 1
        let rf_probs = vec![0.1, 0.6, 0.1, 0.1, 0.05, 0.05];

        // Top-3 from NN: 4, 1, 3
        // Ensemble scores:
        // Class 4: 0.3 * 0.5 + 0.05 * 0.5 = 0.175
        // Class 1: 0.25 * 0.5 + 0.6 * 0.5 = 0.425 <- max
        // Class 3: 0.2 * 0.5 + 0.1 * 0.5 = 0.15

        assert_eq!(voter.predict_with_topk(&nn_probs, &rf_probs, 3), 1);
    }

    #[test]
    fn test_ensemble_voter_evaluate() {
        let config = VotingEnsembleConfig::with_static_weight(0.5);
        let voter = EnsembleVoter::new(config);

        let data = vec![
            EnsembleInput::new(
                vec![0.9, 0.1],
                vec![0.8, 0.2],
                0, // Correct
            ),
            EnsembleInput::new(
                vec![0.2, 0.8],
                vec![0.3, 0.7],
                1, // Correct
            ),
            EnsembleInput::new(
                vec![0.4, 0.6],
                vec![0.9, 0.1], // RF says 0, NN says 1 -> ensemble may say 0
                1, // Wrong (ensemble will predict 0)
            ),
        ];

        let accuracy = voter.evaluate(&data);
        // First two correct, third wrong -> 66.7%
        assert!(accuracy > 0.5 && accuracy < 0.9);
    }

    // ==========================================================================
    // VotingEnsembleConfig Tests
    // ==========================================================================

    #[test]
    fn test_config_default() {
        let config = VotingEnsembleConfig::default();

        assert!((config.nn_weight - 0.5).abs() < 0.001);
        assert_eq!(config.top_k, 5);
        assert!(!config.dynamic_weighting);
    }

    #[test]
    fn test_config_static_weight() {
        let config = VotingEnsembleConfig::with_static_weight(0.3);

        assert!((config.nn_weight - 0.3).abs() < 0.001);
        assert!(!config.dynamic_weighting);
    }

    #[test]
    fn test_config_dynamic_weighting() {
        let config = VotingEnsembleConfig::with_dynamic_weighting();

        assert!(config.dynamic_weighting);
    }

    // ==========================================================================
    // EnsembleMetrics Tests
    // ==========================================================================

    #[test]
    fn test_metrics_perfect_predictions() {
        let predictions = vec![0, 1, 2, 0, 1];
        let labels = vec![0, 1, 2, 0, 1];

        let metrics = EnsembleMetrics::compute(&predictions, &labels, 3, 0.5, false);

        assert!((metrics.accuracy - 1.0).abs() < 0.001);
        assert!((metrics.precision - 1.0).abs() < 0.001);
        assert!((metrics.recall - 1.0).abs() < 0.001);
        assert!((metrics.f1_score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_metrics_half_correct() {
        let predictions = vec![0, 1, 2, 2, 1]; // Last 2 wrong
        let labels = vec![0, 1, 2, 0, 0];

        let metrics = EnsembleMetrics::compute(&predictions, &labels, 3, 0.5, false);

        assert!((metrics.accuracy - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_metrics_empty() {
        let predictions: Vec<usize> = vec![];
        let labels: Vec<usize> = vec![];

        let metrics = EnsembleMetrics::compute(&predictions, &labels, 3, 0.5, false);

        assert!((metrics.accuracy - 0.0).abs() < 0.001);
    }
}
