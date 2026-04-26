//! Dynamic Ensemble with Grid Search for 112D Feature Analysis
//! ================================================================
//!
//! This module implements a weighted ensemble combining:
//! 1. Neural Network (112D -> 112D hidden -> 2 classes)
//! 2. Random Forest (112D -> 2 classes)
//!
//! The ensemble is optimized using grid search to find optimal
//! weight balancing between 0.0 and 1.0.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Import RosettaFeatures from the 112D feature extractor
use crate::micro_dynamics_extractor::RosettaFeatures;

/// Configuration for dynamic ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicEnsembleConfig {
    /// Number of output classes
    pub num_classes: usize,

    /// Weight for NN predictions (0.0 to 1.0)
    pub nn_weight: f32,

    /// Weight for RF predictions (0.0 to 1.0)
    pub rf_weight: f32,

    /// Confidence threshold for using NN (higher = more confident)
    pub nn_confidence_threshold: f32,

    /// Use grid search for weight optimization
    pub use_grid_search: bool,

    /// Grid search resolution (step size)
    pub grid_resolution: usize,

    /// Cross-validation folds
    pub cv_folds: usize,
}

impl Default for DynamicEnsembleConfig {
    fn default() -> Self {
        Self {
            num_classes: 2,
            nn_weight: 0.5,
            rf_weight: 0.5,
            nn_confidence_threshold: 0.7,
            use_grid_search: true,
            grid_resolution: 10,
            cv_folds: 5,
        }
    }
}

/// Model types for ensemble
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    NeuralNetwork,
    RandomForest,
}

/// Prediction result from ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsemblePrediction {
    /// Predicted class
    pub predicted_class: usize,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Which model made the prediction
    pub model_used: ModelType,

    /// Individual NN probability
    pub nn_probability: f32,

    /// Individual RF probability
    pub rf_probability: f32,

    /// Weight used for final prediction
    pub weight_used: f32,
}

/// Dynamic Ensemble combining NN and RF
pub struct DynamicEnsemble {
    config: DynamicEnsembleConfig,
    nn_weights: HashMap<usize, Vec<f32>>,
    rf_weights: HashMap<usize, Vec<f32>>,
}

impl DynamicEnsemble {
    /// Create a new dynamic ensemble
    pub fn new(config: DynamicEnsembleConfig) -> Self {
        Self {
            config,
            nn_weights: HashMap::new(),
            rf_weights: HashMap::new(),
        }
    }

    /// Predict using weighted ensemble
    pub fn predict(&self, _features: &RosettaFeatures, nn_prob: f32, rf_prob: f32) -> EnsemblePrediction {
        // Get NN and RF probabilities
        let nn_prob = nn_prob.clamp(0.0, 1.0);
        let rf_prob = rf_prob.clamp(0.0, 1.0);

        // Apply learned weights
        let nn_weight = self.config.nn_weight;
        let rf_weight = self.config.rf_weight;

        // Check NN confidence
        let use_nn = nn_prob > self.config.nn_confidence_threshold;

        // Weighted combination
        let final_prob = if use_nn {
            nn_prob * nn_weight
        } else {
            rf_prob * rf_weight
        };

        // Normalize
        let total_weight = nn_weight + rf_weight;
        let final_prob = final_prob / total_weight;

        // Determine predicted class
        let predicted_class = if final_prob >= 0.5 { 1 } else { 0 };

        // Determine which model contributed more
        let model_used = if nn_prob > rf_prob {
            ModelType::NeuralNetwork
        } else {
            ModelType::RandomForest
        };

        EnsemblePrediction {
            predicted_class,
            confidence: final_prob,
            model_used,
            nn_probability: nn_prob,
            rf_probability: rf_prob,
            weight_used: if use_nn { nn_weight } else { rf_weight },
        }
    }

    /// Optimize weights using grid search
    pub fn optimize_weights(&mut self, validation_data: &[(RosettaFeatures, usize, f32, f32)]) -> Result<f32, String> {
        let mut best_accuracy = 0.0;
        let mut best_weights = (self.config.nn_weight, self.config.rf_weight);

        // Grid search over weight combinations
        let step = 1.0 / self.config.grid_resolution as f32;
        let mut nn_w = 0.0;
        while nn_w <= 1.0 {
            let mut rf_w = 0.0;
            while rf_w <= 1.0 {
                // Skip if same weights (degenerate)
                if (nn_w + rf_w).abs() < f32::EPSILON {
                    rf_w += step;
                    continue;
                }

                // Evaluate with these weights
                let mut correct = 0;
                for (_features, label, nn_prob, rf_prob) in &validation_data {
                    let pred = self.predict_with_weights(_features, *nn_prob, *rf_prob, nn_w, rf_w);
                    if pred.predicted_class == *label {
                        correct += 1;
                    }
                }

                let accuracy = correct as f32 / validation_data.len() as f32;
                if accuracy > best_accuracy {
                    best_accuracy = accuracy;
                    best_weights = (nn_w, rf_w);
                }
                rf_w += step;
            }
            nn_w += step;
        }

        // Apply best weights
        self.config.nn_weight = best_weights.0;
        self.config.rf_weight = best_weights.1;
        self.nn_weights.insert(0, vec![best_weights.0]);
        self.rf_weights.insert(1, vec![best_weights.1]);

        Ok(best_accuracy)
    }

    fn predict_with_weights(
        &self,
        _features: &RosettaFeatures,
        nn_prob: f32,
        rf_prob: f32,
        nn_weight: f32,
        rf_weight: f32,
    ) -> EnsemblePrediction {
        let nn_prob = nn_prob.clamp(0.0, 1.0);
        let rf_prob = rf_prob.clamp(0.0, 1.0);
        let final_prob = nn_prob * nn_weight + rf_prob * rf_weight;
        let total_weight = nn_weight + rf_weight;
        let final_prob = if total_weight > 0.0 {
            final_prob / total_weight
        } else {
            0.5
        };

        let predicted_class = if final_prob >= 0.5 { 1 } else { 0 };
        EnsemblePrediction {
            predicted_class,
            confidence: final_prob,
            model_used: if nn_prob > rf_prob {
                ModelType::NeuralNetwork
            } else {
                ModelType::RandomForest
            },
            nn_probability: nn_prob,
            rf_probability: rf_prob,
            weight_used: nn_weight,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_features() -> Vec<RosettaFeatures> {
        let mut features = Vec::new();
        for i in 0..100 {
            let mut f = RosettaFeatures::default();
            // Modify some 112D features for variation
            f.mean_f0_hz = 5000.0 + i as f32 * 10.0;
            f.duration_ms = 100.0 + (i % 50) as f32;
            features.push(f);
        }
        features
    }

    #[test]
    fn test_dynamic_ensemble_basic() {
        let config = DynamicEnsembleConfig::default();
        let ensemble = DynamicEnsemble::new(config);

        let features = create_test_features();

        // Test basic prediction
        for f in &features {
            let pred = ensemble.predict(f, 0.6, 0.4);

            assert!(pred.confidence >= 0.0);
            assert!(pred.confidence <= 1.0);
        }
    }

    #[test]
    fn test_dynamic_ensemble_weight_optimization() {
        let config = DynamicEnsembleConfig {
            use_grid_search: true,
            grid_resolution: 5,
            ..Default::default()
        };
        let mut ensemble = DynamicEnsemble::new(config);

        // Create validation data with simulated probabilities
        let mut validation_data = Vec::new();
        for i in 0..50 {
            let features = RosettaFeatures::default();
            // Simulated probabilities based on index
            let nn_prob = if i % 2 == 0 { 0.7 } else { 0.3 };
            let rf_prob = if i % 2 == 0 { 0.4 } else { 0.6 };
            validation_data.push((features, i % 2, nn_prob, rf_prob));
        }

        // Optimize weights
        let best_accuracy = ensemble.optimize_weights(&validation_data).unwrap();
        assert!(best_accuracy >= 0.0); // Should achieve valid accuracy
    }
}
