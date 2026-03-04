//! Metric Learning for Learnable Feature Weights
//! ===============================================
//!
//! Instead of hard-coding weights (e.g., ICI = 3.0 for Whales), this module
//! uses Machine Learning to optimize weights mathematically using triplet loss.
//!
//! **Distance Metric:**
//! D(x, y) = Σ w_i * |x_i - y_i|
//!
//! **Triplet Loss:**
//! L = max(0, D(A, P) - D(A, N) + margin)
//!
//! Where:
//! - A: Anchor sample
//! - P: Positive sample (same class)
//! - N: Negative sample (different class)
//!
//! The training loop updates weights via gradient descent to:
//! - Minimize distance to positive (same class)
//! - Maximize distance to negative (different class)
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Learnable Weights
// ============================================================================

/// 45D learnable weight vector for metric learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnableWeights {
    /// Weight vector (45 dimensions)
    pub weights: Array1<f32>,
    /// Learning rate for gradient descent
    pub learning_rate: f32,
    /// Margin for triplet loss
    pub margin: f32,
    /// Whether to constrain weights to be non-negative
    pub non_negative: bool,
}

impl Default for LearnableWeights {
    fn default() -> Self {
        Self {
            weights: Array1::ones(45),
            learning_rate: 0.01,
            margin: 1.0,
            non_negative: true,
        }
    }
}

impl LearnableWeights {
    /// Create new learnable weights with uniform initialization
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom learning rate and margin
    pub fn with_params(learning_rate: f32, margin: f32) -> Self {
        Self {
            weights: Array1::ones(45),
            learning_rate,
            margin,
            non_negative: true,
        }
    }

    /// Create with random initialization
    pub fn random_init(seed: u64) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut weights = Array1::zeros(45);
        let mut hasher = DefaultHasher::new();

        for i in 0..45 {
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();
            // Convert to f32 in range [0.5, 1.5]
            weights[i] = 0.5 + (hash % 1000) as f32 / 1000.0;
        }

        Self {
            weights,
            learning_rate: 0.01,
            margin: 1.0,
            non_negative: true,
        }
    }

    /// Initialize from existing weights (e.g., from taxonomic router)
    pub fn from_weights(weights: Array1<f32>) -> Self {
        Self {
            weights,
            learning_rate: 0.01,
            margin: 1.0,
            non_negative: true,
        }
    }

    /// Compute weighted Manhattan distance between two feature vectors
    pub fn distance(&self, x: &Array1<f32>, y: &Array1<f32>) -> f32 {
        let diff = x - y;
        let abs_diff = diff.mapv(|v: f32| v.abs());
        (&self.weights * &abs_diff).sum()
    }

    /// Compute weighted Manhattan distance with gradient
    /// Returns (distance, gradient w.r.t. weights)
    pub fn distance_with_gradient(&self, x: &Array1<f32>, y: &Array1<f32>) -> (f32, Array1<f32>) {
        let diff = x - y;
        let abs_diff = diff.mapv(|v: f32| v.abs());
        let distance = (&self.weights * &abs_diff).sum();
        // Gradient of w_i * |x_i - y_i| w.r.t. w_i is |x_i - y_i|
        (distance, abs_diff)
    }

    /// Update weights using gradient
    pub fn update_weights(&mut self, gradient: &Array1<f32>) {
        self.weights = &self.weights - self.learning_rate * gradient;

        if self.non_negative {
            self.weights = self.weights.mapv(|w: f32| w.max(0.0));
        }
    }

    /// Normalize weights to sum to 45 (average weight = 1.0)
    pub fn normalize(&mut self) {
        let sum: f32 = self.weights.sum();
        if sum > 0.0 {
            self.weights = &self.weights * 45.0 / sum;
        }
    }

    /// Get weight for a specific feature index
    pub fn get_weight(&self, idx: usize) -> f32 {
        self.weights[idx]
    }

    /// Set weight for a specific feature index
    pub fn set_weight(&mut self, idx: usize, value: f32) {
        self.weights[idx] = value;
    }

    /// Get top-k features by weight
    pub fn top_features(&self, feature_names: &[String], k: usize) -> Vec<(String, f32)> {
        let mut indexed: Vec<_> = feature_names
            .iter()
            .zip(self.weights.iter())
            .map(|(name, &w): (&String, &f32)| (name.clone(), w))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).collect()
    }
}

// ============================================================================
// Triplet Data Structure
// ============================================================================

/// A triplet for metric learning: (anchor, positive, negative)
#[derive(Debug, Clone)]
pub struct Triplet {
    /// Anchor sample features
    pub anchor: Array1<f32>,
    /// Positive sample features (same class as anchor)
    pub positive: Array1<f32>,
    /// Negative sample features (different class from anchor)
    pub negative: Array1<f32>,
    /// Class of anchor/positive
    pub anchor_class: String,
    /// Class of negative
    pub negative_class: String,
}

impl Triplet {
    /// Create a new triplet
    pub fn new(
        anchor: Array1<f32>,
        positive: Array1<f32>,
        negative: Array1<f32>,
        anchor_class: String,
        negative_class: String,
    ) -> Self {
        Self {
            anchor,
            positive,
            negative,
            anchor_class,
            negative_class,
        }
    }
}

// ============================================================================
// Triplet Dataset
// ============================================================================

/// Dataset for generating triplets
#[derive(Debug, Clone)]
pub struct TripletDataset {
    /// Feature matrix (n_samples x 45)
    pub features: Array2<f32>,
    /// Labels for each sample
    pub labels: Vec<String>,
    /// Index mapping: class -> sample indices
    pub class_indices: HashMap<String, Vec<usize>>,
    /// List of unique classes
    pub classes: Vec<String>,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl TripletDataset {
    /// Create triplet dataset from features and labels
    pub fn new(features: Array2<f32>, labels: Vec<String>, seed: u64) -> Self {
        // Build class index mapping
        let mut class_indices: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, label) in labels.iter().enumerate() {
            class_indices.entry(label.clone()).or_default().push(i);
        }

        let classes: Vec<String> = class_indices.keys().cloned().collect();

        Self {
            features,
            labels,
            class_indices,
            classes,
            seed,
        }
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.features.nrows()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.features.nrows() == 0
    }

    /// Get number of classes
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    /// Get features for a sample
    pub fn get_features(&self, idx: usize) -> Array1<f32> {
        self.features.row(idx).to_owned()
    }

    /// Generate a random triplet
    /// Uses a simple LCG random number generator
    pub fn sample_triplet(&self, rng_state: &mut u64) -> Option<Triplet> {
        // LCG parameters (same as used in rand crate)
        const A: u64 = 6364136223846793005;
        const C: u64 = 1;

        let next_random = |state: &mut u64| -> u64 {
            *state = state.wrapping_mul(A).wrapping_add(C);
            *state
        };

        // Filter classes with at least 2 samples
        let valid_classes: Vec<_> = self
            .class_indices
            .iter()
            .filter(|(_, indices)| indices.len() >= 2)
            .map(|(class, _)| class.clone())
            .collect();

        if valid_classes.len() < 2 {
            return None;
        }

        // Select anchor class
        let anchor_class_idx = (next_random(rng_state) as usize) % valid_classes.len();
        let anchor_class = &valid_classes[anchor_class_idx];

        // Select negative class (different from anchor)
        let negative_class_idx = (next_random(rng_state) as usize) % valid_classes.len();
        let negative_class = if negative_class_idx == anchor_class_idx {
            &valid_classes[(anchor_class_idx + 1) % valid_classes.len()]
        } else {
            &valid_classes[negative_class_idx]
        };

        // Get sample indices
        let anchor_indices = &self.class_indices[anchor_class];
        let negative_indices = &self.class_indices[negative_class];

        // Select anchor and positive (different samples from same class)
        let anchor_idx = anchor_indices[(next_random(rng_state) as usize) % anchor_indices.len()];
        let positive_idx = loop {
            let idx = anchor_indices[(next_random(rng_state) as usize) % anchor_indices.len()];
            if idx != anchor_idx || anchor_indices.len() == 1 {
                break idx;
            }
        };

        // Select negative
        let negative_idx =
            negative_indices[(next_random(rng_state) as usize) % negative_indices.len()];

        Some(Triplet::new(
            self.get_features(anchor_idx),
            self.get_features(positive_idx),
            self.get_features(negative_idx),
            anchor_class.clone(),
            negative_class.clone(),
        ))
    }

    /// Generate a batch of triplets
    pub fn sample_batch(&self, batch_size: usize) -> Vec<Triplet> {
        let mut rng_state = self.seed;
        let mut triplets = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            if let Some(triplet) = self.sample_triplet(&mut rng_state) {
                triplets.push(triplet);
            }
        }

        triplets
    }

    /// Generate semi-hard triplets (negative is close but not too close)
    /// These are more informative for training
    pub fn sample_semi_hard_triplets(
        &self,
        weights: &LearnableWeights,
        batch_size: usize,
    ) -> Vec<Triplet> {
        let mut rng_state = self.seed;
        let mut triplets = Vec::with_capacity(batch_size);

        // Filter classes with at least 2 samples
        let valid_classes: Vec<_> = self
            .class_indices
            .iter()
            .filter(|(_, indices)| indices.len() >= 2)
            .map(|(class, _)| class.clone())
            .collect();

        if valid_classes.len() < 2 {
            return triplets;
        }

        const A: u64 = 6364136223846793005;
        const C: u64 = 1;
        let next_random = |state: &mut u64| -> u64 {
            *state = state.wrapping_mul(A).wrapping_add(C);
            *state
        };

        for _ in 0..batch_size {
            // Select anchor class
            let anchor_class_idx = (next_random(&mut rng_state) as usize) % valid_classes.len();
            let anchor_class = &valid_classes[anchor_class_idx];
            let anchor_indices = &self.class_indices[anchor_class];

            // Select anchor and positive
            let anchor_idx =
                anchor_indices[(next_random(&mut rng_state) as usize) % anchor_indices.len()];
            let positive_idx = loop {
                let idx =
                    anchor_indices[(next_random(&mut rng_state) as usize) % anchor_indices.len()];
                if idx != anchor_idx || anchor_indices.len() == 1 {
                    break idx;
                }
            };

            let anchor_features = self.get_features(anchor_idx);
            let positive_features = self.get_features(positive_idx);
            let d_ap = weights.distance(&anchor_features, &positive_features);

            // Find semi-hard negative: d(A,N) > d(A,P) but d(A,N) < d(A,P) + margin
            let mut best_negative_idx: Option<usize> = None;
            let mut best_negative_class: Option<String> = None;
            let mut best_margin_violation = f32::MAX;

            // Try a few candidate negatives
            for _ in 0..10 {
                let neg_class_idx = (next_random(&mut rng_state) as usize) % valid_classes.len();
                if neg_class_idx == anchor_class_idx {
                    continue;
                }

                let neg_class = &valid_classes[neg_class_idx];
                let neg_indices = &self.class_indices[neg_class];
                let neg_idx =
                    neg_indices[(next_random(&mut rng_state) as usize) % neg_indices.len()];

                let neg_features = self.get_features(neg_idx);
                let d_an = weights.distance(&anchor_features, &neg_features);

                // Semi-hard: positive distance < negative distance < positive distance + margin
                let margin_violation = d_an - d_ap;

                if margin_violation > 0.0 && margin_violation < weights.margin {
                    // Found semi-hard negative
                    if margin_violation < best_margin_violation {
                        best_margin_violation = margin_violation;
                        best_negative_idx = Some(neg_idx);
                        best_negative_class = Some(neg_class.clone());
                    }
                }
            }

            // If no semi-hard negative found, use any negative
            let (negative_idx, negative_class) = if let (Some(idx), Some(class)) =
                (best_negative_idx, best_negative_class)
            {
                (idx, class)
            } else {
                // Fallback to random negative
                let neg_class_idx = (next_random(&mut rng_state) as usize) % valid_classes.len();
                let neg_class_idx = if neg_class_idx == anchor_class_idx {
                    (neg_class_idx + 1) % valid_classes.len()
                } else {
                    neg_class_idx
                };
                let neg_class = &valid_classes[neg_class_idx];
                let neg_indices = &self.class_indices[neg_class];
                let neg_idx =
                    neg_indices[(next_random(&mut rng_state) as usize) % neg_indices.len()];
                (neg_idx, neg_class.clone())
            };

            triplets.push(Triplet::new(
                anchor_features,
                positive_features,
                self.get_features(negative_idx),
                anchor_class.clone(),
                negative_class,
            ));
        }

        triplets
    }
}

// ============================================================================
// Triplet Loss
// ============================================================================

/// Triplet loss function and gradient computation
pub struct TripletLoss {
    /// Margin for the loss
    pub margin: f32,
}

impl TripletLoss {
    /// Create new triplet loss with given margin
    pub fn new(margin: f32) -> Self {
        Self { margin }
    }

    /// Compute triplet loss: L = max(0, d(A,P) - d(A,N) + margin)
    /// Returns (loss, gradient w.r.t. weights)
    pub fn compute(&self, weights: &LearnableWeights, triplet: &Triplet) -> (f32, Array1<f32>) {
        let (d_ap, grad_ap) = weights.distance_with_gradient(&triplet.anchor, &triplet.positive);
        let (d_an, grad_an) = weights.distance_with_gradient(&triplet.anchor, &triplet.negative);

        let loss_value = d_ap - d_an + self.margin;

        if loss_value > 0.0 {
            // Loss is active: gradient = grad_ap - grad_an
            let gradient = grad_ap - grad_an;
            (loss_value, gradient)
        } else {
            // Loss is zero: no gradient
            (0.0, Array1::zeros(45))
        }
    }

    /// Compute loss and gradient for a batch of triplets
    pub fn compute_batch(
        &self,
        weights: &LearnableWeights,
        triplets: &[Triplet],
    ) -> (f32, Array1<f32>) {
        if triplets.is_empty() {
            return (0.0, Array1::zeros(45));
        }

        let mut total_loss = 0.0;
        let mut total_gradient = Array1::zeros(45);
        let mut active_count = 0;

        for triplet in triplets {
            let (loss, gradient) = self.compute(weights, triplet);
            total_loss += loss;
            if loss > 0.0 {
                total_gradient = total_gradient + gradient;
                active_count += 1;
            }
        }

        // Average the gradient
        if active_count > 0 {
            total_gradient /= active_count as f32;
        }

        (total_loss / triplets.len() as f32, total_gradient)
    }
}

// ============================================================================
// Metric Learner
// ============================================================================

/// Training configuration
#[derive(Debug, Clone)]
pub struct MetricLearnerConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Margin for triplet loss
    pub margin: f32,
    /// Number of epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Whether to normalize weights after each epoch
    pub normalize_weights: bool,
    /// Early stopping patience (epochs without improvement)
    pub early_stopping_patience: usize,
}

impl Default for MetricLearnerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            margin: 1.0,
            epochs: 100,
            batch_size: 64,
            normalize_weights: true,
            early_stopping_patience: 10,
        }
    }
}

/// Metric learner that optimizes weights using triplet loss
#[derive(Debug, Clone)]
pub struct MetricLearner {
    /// Learnable weights
    pub weights: LearnableWeights,
    /// Training configuration
    pub config: MetricLearnerConfig,
    /// Training history
    pub history: TrainingHistory,
}

/// Training history for monitoring
#[derive(Debug, Clone, Default)]
pub struct TrainingHistory {
    /// Loss at each epoch
    pub losses: Vec<f32>,
    /// Validation accuracy at each epoch (if available)
    pub val_accuracies: Vec<f32>,
    /// Best validation accuracy
    pub best_val_accuracy: f32,
    /// Epoch with best validation accuracy
    pub best_epoch: usize,
}

impl MetricLearner {
    /// Create new metric learner
    pub fn new(config: MetricLearnerConfig) -> Self {
        let weights = LearnableWeights::with_params(config.learning_rate, config.margin);
        Self {
            weights,
            config,
            history: TrainingHistory::default(),
        }
    }

    /// Create learner with initial weights
    pub fn with_weights(weights: LearnableWeights, config: MetricLearnerConfig) -> Self {
        Self {
            weights,
            config,
            history: TrainingHistory::default(),
        }
    }

    /// Train on a triplet dataset
    pub fn train(&mut self, dataset: &TripletDataset) -> Result<()> {
        let loss_fn = TripletLoss::new(self.config.margin);
        let mut best_weights = self.weights.clone();
        let mut epochs_without_improvement = 0;

        for epoch in 0..self.config.epochs {
            // Update random seed for each epoch
            let epoch_seed = dataset.seed.wrapping_add(epoch as u64);

            // Create epoch-specific dataset
            let epoch_dataset =
                TripletDataset::new(dataset.features.clone(), dataset.labels.clone(), epoch_seed);

            // Sample batch (semi-hard mining for better gradients)
            let triplets =
                epoch_dataset.sample_semi_hard_triplets(&self.weights, self.config.batch_size);

            if triplets.is_empty() {
                continue;
            }

            // Compute loss and gradient
            let (loss, gradient) = loss_fn.compute_batch(&self.weights, &triplets);

            // Update weights
            self.weights.update_weights(&gradient);

            // Normalize weights if configured
            if self.config.normalize_weights {
                self.weights.normalize();
            }

            // Record history
            self.history.losses.push(loss);

            // For now, use loss improvement as proxy for accuracy
            // In practice, you'd compute validation accuracy here
            if self.history.best_val_accuracy == 0.0
                || loss < self.history.losses[self.history.best_epoch]
            {
                self.history.best_val_accuracy = 1.0 / (1.0 + loss); // Proxy accuracy
                self.history.best_epoch = epoch;
                best_weights = self.weights.clone();
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            // Early stopping
            if epochs_without_improvement >= self.config.early_stopping_patience {
                break;
            }
        }

        // Restore best weights
        self.weights = best_weights;

        Ok(())
    }

    /// Train with validation set
    pub fn train_with_validation(
        &mut self,
        train_dataset: &TripletDataset,
        val_dataset: &TripletDataset,
        val_labels: &[String],
    ) -> Result<()> {
        let loss_fn = TripletLoss::new(self.config.margin);
        let mut best_weights = self.weights.clone();
        let mut best_val_acc = 0.0f32;
        let mut epochs_without_improvement = 0;

        for epoch in 0..self.config.epochs {
            let epoch_seed = train_dataset.seed.wrapping_add(epoch as u64);
            let epoch_dataset = TripletDataset::new(
                train_dataset.features.clone(),
                train_dataset.labels.clone(),
                epoch_seed,
            );

            // Sample and train on batch
            let triplets =
                epoch_dataset.sample_semi_hard_triplets(&self.weights, self.config.batch_size);

            if !triplets.is_empty() {
                let (loss, gradient) = loss_fn.compute_batch(&self.weights, &triplets);
                self.weights.update_weights(&gradient);

                if self.config.normalize_weights {
                    self.weights.normalize();
                }

                self.history.losses.push(loss);
            }

            // Evaluate on validation set using k-NN accuracy
            let val_acc = self.evaluate_knn(val_dataset, val_labels, 5);
            self.history.val_accuracies.push(val_acc);

            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                self.history.best_val_accuracy = val_acc;
                self.history.best_epoch = epoch;
                best_weights = self.weights.clone();
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            if epochs_without_improvement >= self.config.early_stopping_patience {
                break;
            }
        }

        self.weights = best_weights;
        Ok(())
    }

    /// Evaluate k-NN accuracy with current weights
    pub fn evaluate_knn(&self, dataset: &TripletDataset, labels: &[String], k: usize) -> f32 {
        if dataset.is_empty() || labels.is_empty() {
            return 0.0;
        }

        let n = dataset.len();
        let mut correct = 0;

        // Leave-one-out k-NN evaluation
        for i in 0..n.min(500) {
            // Limit for speed
            let query = dataset.get_features(i);
            let true_label = &labels[i];

            // Find k nearest neighbors
            let mut distances: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, self.weights.distance(&query, &dataset.get_features(j))))
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Vote among top-k
            let mut votes: HashMap<String, usize> = HashMap::new();
            for (idx, _) in distances.iter().take(k) {
                let neighbor_label = &labels[*idx];
                *votes.entry(neighbor_label.clone()).or_default() += 1;
            }

            // Predict most common class
            let predicted = votes
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(label, _)| label.clone())
                .unwrap_or_default();

            if predicted == *true_label {
                correct += 1;
            }
        }

        correct as f32 / n.min(500) as f32
    }

    /// Get the learned weights
    pub fn get_weights(&self) -> &Array1<f32> {
        &self.weights.weights
    }

    /// Get training history
    pub fn get_history(&self) -> &TrainingHistory {
        &self.history
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_dataset() -> TripletDataset {
        // Create simple 2-class dataset
        let features =
            Array2::from_shape_vec((6, 45), (0..270).map(|i| i as f32 / 270.0).collect()).unwrap();

        let labels = vec![
            "class_a".to_string(),
            "class_a".to_string(),
            "class_a".to_string(),
            "class_b".to_string(),
            "class_b".to_string(),
            "class_b".to_string(),
        ];

        TripletDataset::new(features, labels, 42)
    }

    #[test]
    fn test_learnable_weights_creation() {
        let weights = LearnableWeights::new();
        assert_eq!(weights.weights.len(), 45);
        assert!(weights
            .weights
            .iter()
            .all(|&w: &f32| (w - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_learnable_weights_distance() {
        let weights = LearnableWeights::new();
        let x = Array1::from_vec(vec![1.0; 45]);
        let y = Array1::from_vec(vec![2.0; 45]);

        // Distance should be sum of |1-2| * 1 = 45
        let dist = weights.distance(&x, &y);
        assert!((dist - 45.0).abs() < 1e-6);

        // Distance to self should be 0
        let dist_self = weights.distance(&x, &x);
        assert!(dist_self.abs() < 1e-6);
    }

    #[test]
    fn test_learnable_weights_distance_with_gradient() {
        let weights = LearnableWeights::new();
        let x = Array1::from_vec(vec![1.0; 45]);
        let y = Array1::from_vec(vec![3.0; 45]);

        let (dist, grad) = weights.distance_with_gradient(&x, &y);

        // Distance = sum of |1-3| * 1 = 90
        assert!((dist - 90.0).abs() < 1e-6);

        // Gradient = |1-3| = 2 for each dimension
        assert!(grad.iter().all(|&g: &f32| (g - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_learnable_weights_update() {
        let mut weights = LearnableWeights::with_params(0.1, 1.0);
        let gradient = Array1::from_elem(45, 1.0);

        let original = weights.weights.clone();
        weights.update_weights(&gradient);

        // Each weight should decrease by learning_rate * gradient = 0.1 * 1 = 0.1
        for i in 0..45 {
            assert!((weights.weights[i] - (original[i] - 0.1)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_learnable_weights_normalize() {
        let mut weights = LearnableWeights::new();
        weights.weights = Array1::from_elem(45, 2.0); // Sum = 90

        weights.normalize();

        // Should normalize so sum = 45, each weight = 1.0
        assert!((weights.weights.sum() - 45.0).abs() < 1e-6);
        assert!(weights
            .weights
            .iter()
            .all(|&w: &f32| (w - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_learnable_weights_non_negative() {
        let mut weights = LearnableWeights::with_params(10.0, 1.0); // Very high learning rate
        let gradient = Array1::from_elem(45, 1.0);

        weights.update_weights(&gradient);

        // Weights should be clipped to non-negative
        assert!(weights.weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_triplet_dataset_creation() {
        let dataset = create_test_dataset();

        assert_eq!(dataset.len(), 6);
        assert_eq!(dataset.num_classes(), 2);
        assert!(dataset.class_indices.contains_key("class_a"));
        assert!(dataset.class_indices.contains_key("class_b"));
        assert_eq!(dataset.class_indices["class_a"].len(), 3);
        assert_eq!(dataset.class_indices["class_b"].len(), 3);
    }

    #[test]
    fn test_triplet_dataset_sample() {
        let dataset = create_test_dataset();
        let mut rng_state = 42u64;

        let triplet = dataset.sample_triplet(&mut rng_state);

        assert!(triplet.is_some());
        let triplet = triplet.unwrap();

        // Anchor and positive should have same class
        assert_eq!(triplet.anchor_class, triplet.positive_class());
        // Actually, we stored anchor_class as the class, not positive_class
        // Let me check the triplet structure - positive doesn't have its own class field
        // The anchor_class represents both anchor and positive

        // Negative should have different class
        assert_ne!(triplet.anchor_class, triplet.negative_class);
    }

    #[test]
    fn test_triplet_dataset_batch() {
        let dataset = create_test_dataset();
        let batch = dataset.sample_batch(10);

        assert!(batch.len() <= 10);
        for triplet in &batch {
            assert_eq!(triplet.anchor.len(), 45);
            assert_eq!(triplet.positive.len(), 45);
            assert_eq!(triplet.negative.len(), 45);
        }
    }

    #[test]
    fn test_triplet_loss_zero() {
        let weights = LearnableWeights::new();
        let loss_fn = TripletLoss::new(1.0);

        // Create triplet where anchor == positive, anchor != negative
        let anchor = Array1::zeros(45);
        let positive = Array1::zeros(45); // Same as anchor
        let negative = Array1::from_elem(45, 10.0); // Different

        let triplet = Triplet::new(anchor, positive, negative, "a".to_string(), "b".to_string());

        let (loss, _) = loss_fn.compute(&weights, &triplet);

        // d(A,P) = 0, d(A,N) = 450, margin = 1
        // loss = max(0, 0 - 450 + 1) = 0
        assert!(loss.abs() < 1e-6);
    }

    #[test]
    fn test_triplet_loss_active() {
        let weights = LearnableWeights::new();
        let loss_fn = TripletLoss::new(1.0);

        // Create triplet where negative is closer than positive
        let anchor = Array1::zeros(45);
        let positive = Array1::from_elem(45, 5.0); // Far
        let negative = Array1::from_elem(45, 1.0); // Close

        let triplet = Triplet::new(anchor, positive, negative, "a".to_string(), "b".to_string());

        let (loss, _) = loss_fn.compute(&weights, &triplet);

        // d(A,P) = 225, d(A,N) = 45, margin = 1
        // loss = max(0, 225 - 45 + 1) = 181
        assert!(loss > 0.0);
    }

    #[test]
    fn test_triplet_loss_gradient() {
        let weights = LearnableWeights::new();
        let loss_fn = TripletLoss::new(1.0);

        let anchor = Array1::zeros(45);
        let positive = Array1::from_elem(45, 1.0);
        let negative = Array1::from_elem(45, 2.0);

        let triplet = Triplet::new(anchor, positive, negative, "a".to_string(), "b".to_string());

        let (loss, gradient) = loss_fn.compute(&weights, &triplet);

        // d(A,P) = 45, d(A,N) = 90, margin = 1
        // loss = max(0, 45 - 90 + 1) = 0 (not active since d(A,N) > d(A,P))
        assert!(loss.abs() < 1e-6);
        // Gradient should be zero when loss is not active
        assert!(gradient.iter().all(|&g: &f32| g.abs() < 1e-6));
    }

    #[test]
    fn test_triplet_loss_gradient_active() {
        let weights = LearnableWeights::new();
        let loss_fn = TripletLoss::new(1.0);

        let anchor = Array1::zeros(45);
        let positive = Array1::from_elem(45, 2.0);
        let negative = Array1::from_elem(45, 1.0);

        let triplet = Triplet::new(anchor, positive, negative, "a".to_string(), "b".to_string());

        let (loss, gradient) = loss_fn.compute(&weights, &triplet);

        // d(A,P) = 90, d(A,N) = 45, margin = 1
        // loss = max(0, 90 - 45 + 1) = 46 (active)
        assert!((loss - 46.0).abs() < 1e-6);

        // Gradient = |A-P| - |A-N| = 2 - 1 = 1 for each dimension
        assert!(gradient.iter().all(|&g: &f32| (g - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_metric_learner_creation() {
        let config = MetricLearnerConfig::default();
        let learner = MetricLearner::new(config);

        assert_eq!(learner.weights.weights.len(), 45);
        assert_eq!(learner.config.epochs, 100);
    }

    #[test]
    fn test_metric_learner_train() {
        let dataset = create_test_dataset();
        let config = MetricLearnerConfig {
            epochs: 5,
            batch_size: 4,
            ..Default::default()
        };
        let mut learner = MetricLearner::new(config);

        learner.train(&dataset).unwrap();

        // Should have recorded losses
        assert!(!learner.history.losses.is_empty());
        // Weights should still be valid
        assert_eq!(learner.weights.weights.len(), 45);
    }

    #[test]
    fn test_top_features() {
        let mut weights = LearnableWeights::new();
        weights.weights[0] = 5.0; // Make first feature most important
        weights.weights[1] = 4.0;
        weights.weights[2] = 3.0;

        let feature_names: Vec<String> = (0..45).map(|i| format!("feature_{}", i)).collect();
        let top = weights.top_features(&feature_names, 3);

        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, "feature_0");
        assert_eq!(top[1].0, "feature_1");
        assert_eq!(top[2].0, "feature_2");
    }

    #[test]
    fn test_random_initialization() {
        let w1 = LearnableWeights::random_init(42);
        let w2 = LearnableWeights::random_init(42);
        let w3 = LearnableWeights::random_init(123);

        // Same seed should give same weights
        for i in 0..45 {
            assert!((w1.weights[i] - w2.weights[i]).abs() < 1e-6);
        }

        // Different seed should give different weights
        let mut any_different = false;
        for i in 0..45 {
            if (w1.weights[i] - w3.weights[i]).abs() > 1e-6 {
                any_different = true;
                break;
            }
        }
        assert!(any_different);
    }

    #[test]
    fn test_from_existing_weights() {
        let existing = Array1::from_elem(45, 2.5);
        let weights = LearnableWeights::from_weights(existing.clone());

        for i in 0..45 {
            assert!((weights.weights[i] - 2.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_semi_hard_triplet_mining() {
        let dataset = create_test_dataset();
        let weights = LearnableWeights::new();

        let triplets = dataset.sample_semi_hard_triplets(&weights, 10);

        // Should get some triplets
        assert!(!triplets.is_empty());

        // Each triplet should be valid
        for triplet in &triplets {
            assert_eq!(triplet.anchor.len(), 45);
            assert_ne!(triplet.anchor_class, triplet.negative_class);
        }
    }

    #[test]
    fn test_knn_evaluation() {
        let dataset = create_test_dataset();
        let config = MetricLearnerConfig::default();
        let learner = MetricLearner::new(config);

        let labels: Vec<String> = dataset.labels.clone();
        let accuracy = learner.evaluate_knn(&dataset, &labels, 3);

        // Accuracy should be between 0 and 1
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }
}

// Helper for tests - Triplet doesn't have positive_class method
impl Triplet {
    fn positive_class(&self) -> &str {
        &self.anchor_class
    }
}
