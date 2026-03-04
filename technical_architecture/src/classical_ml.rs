//! Classical Machine Learning for 45D Feature Classification
//! ==========================================================
//!
//! This module implements "Feature-First" classical ML using the 45D Rosetta
//! feature vectors. Instead of deep learning on raw audio, we use efficient
//! classical algorithms that learn non-linear combinations of features.
//!
//! **Key Insight:**
//! The 45D vector compresses 1 second of audio (44,100 samples) into just
//! 45 numbers. This makes ML extremely efficient - no need for millions of
//! training samples.
//!
//! **Algorithms:**
//! - Random Forest: Learns feature importance automatically
//! - Logistic Regression: Fast baseline with interpretable weights
//! - Decision Tree: Simple interpretable rules
//!
//! **Expected Improvement:**
//! - k-NN baseline: 38.56% accuracy
//! - Random Forest: ~55-65% accuracy (learns non-linear feature combinations)
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use anyhow::Result;
use ndarray::{s, Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Feature Dataset
// ============================================================================

/// Dataset of 45D feature vectors with labels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDataset {
    /// Feature matrix (n_samples x 45)
    pub features: Array2<f32>,
    /// Labels (n_samples)
    pub labels: Vec<String>,
    /// Label to index mapping
    pub label_to_idx: HashMap<String, usize>,
    /// Index to label mapping
    pub idx_to_label: HashMap<usize, String>,
    /// Feature names (45 features)
    pub feature_names: Vec<String>,
}

impl FeatureDataset {
    /// Create a new empty dataset
    pub fn new() -> Self {
        Self {
            features: Array2::zeros((0, 45)),
            labels: Vec::new(),
            label_to_idx: HashMap::new(),
            idx_to_label: HashMap::new(),
            feature_names: Self::default_feature_names(),
        }
    }

    /// Create dataset with capacity
    pub fn with_capacity(n_samples: usize) -> Self {
        Self {
            features: Array2::zeros((n_samples, 45)),
            labels: Vec::with_capacity(n_samples),
            label_to_idx: HashMap::new(),
            idx_to_label: HashMap::new(),
            feature_names: Self::default_feature_names(),
        }
    }

    /// Default 45D feature names
    pub fn default_feature_names() -> Vec<String> {
        vec![
            // Fundamental (3)
            "mean_f0_hz".to_string(),
            "duration_ms".to_string(),
            "f0_range_hz".to_string(),
            // Grit (3)
            "hnr".to_string(),
            "spectral_flatness".to_string(),
            "harmonicity".to_string(),
            // Motion (7)
            "attack_time_ms".to_string(),
            "decay_time_ms".to_string(),
            "sustain_level".to_string(),
            "vibrato_rate_hz".to_string(),
            "vibrato_depth".to_string(),
            "jitter".to_string(),
            "shimmer".to_string(),
            // Fingerprint/MFCC (14)
            "mfcc_01".to_string(),
            "mfcc_02".to_string(),
            "mfcc_03".to_string(),
            "mfcc_04".to_string(),
            "mfcc_05".to_string(),
            "mfcc_06".to_string(),
            "mfcc_07".to_string(),
            "mfcc_08".to_string(),
            "mfcc_09".to_string(),
            "mfcc_10".to_string(),
            "mfcc_11".to_string(),
            "mfcc_12".to_string(),
            "mfcc_13".to_string(),
            "mfcc_14".to_string(),
            // Rhythm (3)
            "tempo_bpm".to_string(),
            "pulse_clarity".to_string(),
            "rhythm_regularity".to_string(),
            // Resonance (6)
            "formant_1_hz".to_string(),
            "formant_2_hz".to_string(),
            "formant_3_hz".to_string(),
            "bandwidth_1".to_string(),
            "bandwidth_2".to_string(),
            "dispersion".to_string(),
            // Spectral Shape (4)
            "spectral_centroid".to_string(),
            "spectral_spread".to_string(),
            "spectral_skewness".to_string(),
            "spectral_kurtosis".to_string(),
            // Modulation (3)
            "spectral_tilt".to_string(),
            "fm_slope".to_string(),
            "am_depth".to_string(),
            // Non-Linear (2)
            "subharmonic_ratio".to_string(),
            "spectral_entropy".to_string(),
        ]
    }

    /// Add a sample to the dataset
    pub fn add_sample(&mut self, features: Array1<f32>, label: &str) {
        // Ensure features is 45D
        assert_eq!(features.len(), 45, "Features must be 45-dimensional");

        // Add label to mapping if new
        let label_idx = if !self.label_to_idx.contains_key(label) {
            let idx = self.label_to_idx.len();
            self.label_to_idx.insert(label.to_string(), idx);
            self.idx_to_label.insert(idx, label.to_string());
            idx
        } else {
            self.label_to_idx[label]
        };

        // Append features
        let n = self.features.nrows();
        let mut new_features = Array2::zeros((n + 1, 45));
        if n > 0 {
            new_features.slice_mut(s![0..n, ..]).assign(&self.features);
        }
        new_features.row_mut(n).assign(&features);
        self.features = new_features;

        self.labels.push(label.to_string());
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Get number of classes
    pub fn num_classes(&self) -> usize {
        self.label_to_idx.len()
    }

    /// Split into train and test sets
    pub fn train_test_split(&self, test_ratio: f32, seed: u64) -> (Self, Self) {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let n = self.len();
        let test_size = (n as f32 * test_ratio) as usize;

        // Create shuffled indices
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let test_indices: Vec<usize> = indices.iter().take(test_size).cloned().collect();
        let train_indices: Vec<usize> = indices.iter().skip(test_size).cloned().collect();

        let mut train = FeatureDataset::new();
        let mut test = FeatureDataset::new();

        // Copy mappings
        train.label_to_idx = self.label_to_idx.clone();
        train.idx_to_label = self.idx_to_label.clone();
        test.label_to_idx = self.label_to_idx.clone();
        test.idx_to_label = self.idx_to_label.clone();

        for &i in &train_indices {
            train.add_sample(self.features.row(i).to_owned(), &self.labels[i]);
        }

        for &i in &test_indices {
            test.add_sample(self.features.row(i).to_owned(), &self.labels[i]);
        }

        (train, test)
    }

    /// Normalize features (z-score normalization)
    pub fn normalize(&mut self) {
        let n = self.features.nrows();
        if n == 0 {
            return;
        }

        for j in 0..45 {
            let col = self.features.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let std = col.std(0.0);
            if std > 1e-10 {
                for i in 0..n {
                    self.features[[i, j]] = (self.features[[i, j]] - mean) / std;
                }
            }
        }
    }

    /// Balance classes by undersampling majority classes
    pub fn balance_classes(&self, seed: u64) -> Self {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Count samples per class
        let mut class_counts: HashMap<&str, usize> = HashMap::new();
        for label in &self.labels {
            *class_counts.entry(label.as_str()).or_default() += 1;
        }

        // Find minimum class count
        let min_count = class_counts.values().min().copied().unwrap_or(0);
        if min_count == 0 {
            return self.clone();
        }

        // Group indices by class
        let mut class_indices: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, label) in self.labels.iter().enumerate() {
            class_indices.entry(label.as_str()).or_default().push(i);
        }

        // Create balanced dataset
        let mut balanced = FeatureDataset::new();
        balanced.label_to_idx = self.label_to_idx.clone();
        balanced.idx_to_label = self.idx_to_label.clone();

        for (label, indices) in &class_indices {
            let mut indices = indices.clone();
            indices.shuffle(&mut rng);
            for &i in indices.iter().take(min_count) {
                balanced.add_sample(self.features.row(i).to_owned(), label);
            }
        }

        balanced
    }

    /// Get feature statistics
    pub fn feature_statistics(&self) -> FeatureStatistics {
        let n = self.features.nrows();
        if n == 0 {
            return FeatureStatistics::default();
        }

        let mut means = Array1::zeros(45);
        let mut stds = Array1::zeros(45);
        let mut mins = Array1::from_elem(45, f32::INFINITY);
        let mut maxs = Array1::from_elem(45, f32::NEG_INFINITY);

        for j in 0..45 {
            let col = self.features.column(j);
            means[j] = col.mean().unwrap_or(0.0);
            stds[j] = col.std(0.0);
            mins[j] = col.iter().cloned().fold(f32::INFINITY, f32::min);
            maxs[j] = col.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        }

        FeatureStatistics {
            means,
            stds,
            mins,
            maxs,
            n_samples: n,
            n_classes: self.num_classes(),
        }
    }
}

impl Default for FeatureDataset {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeatureStatistics {
    pub means: Array1<f32>,
    pub stds: Array1<f32>,
    pub mins: Array1<f32>,
    pub maxs: Array1<f32>,
    pub n_samples: usize,
    pub n_classes: usize,
}

// ============================================================================
// Decision Tree Classifier
// ============================================================================

/// Simple decision tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    /// Feature index for split (None = leaf)
    pub feature_idx: Option<usize>,
    /// Threshold for split
    pub threshold: Option<f32>,
    /// Left child index
    pub left: Option<usize>,
    /// Right child index
    pub right: Option<usize>,
    /// Class prediction (for leaf nodes)
    pub prediction: Option<usize>,
    /// Class distribution at this node
    pub class_counts: Vec<usize>,
}

impl TreeNode {
    fn leaf(class_counts: Vec<usize>) -> Self {
        let prediction = class_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i);
        Self {
            feature_idx: None,
            threshold: None,
            left: None,
            right: None,
            prediction,
            class_counts,
        }
    }

    fn split(feature_idx: usize, threshold: f32, class_counts: Vec<usize>) -> Self {
        Self {
            feature_idx: Some(feature_idx),
            threshold: Some(threshold),
            left: None,
            right: None,
            prediction: None,
            class_counts,
        }
    }

    fn is_leaf(&self) -> bool {
        self.feature_idx.is_none()
    }
}

/// Decision Tree Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeClassifier {
    /// Tree nodes
    nodes: Vec<TreeNode>,
    /// Maximum depth
    max_depth: usize,
    /// Minimum samples to split
    min_samples_split: usize,
    /// Number of classes
    n_classes: usize,
    /// Feature importances
    feature_importances: Array1<f32>,
}

impl DecisionTreeClassifier {
    /// Create a new decision tree
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            nodes: Vec::new(),
            max_depth,
            min_samples_split,
            n_classes: 0,
            feature_importances: Array1::zeros(45),
        }
    }

    /// Train the decision tree
    pub fn fit(&mut self, dataset: &FeatureDataset) -> Result<()> {
        if dataset.is_empty() {
            anyhow::bail!("Cannot train on empty dataset");
        }

        self.n_classes = dataset.num_classes();
        self.nodes.clear();
        self.feature_importances = Array1::zeros(45);

        // Convert to label indices
        let y: Vec<usize> = dataset
            .labels
            .iter()
            .map(|l| dataset.label_to_idx.get(l).copied().unwrap_or(0))
            .collect();

        // Build tree recursively
        let indices: Vec<usize> = (0..dataset.len()).collect();
        self.build_node(&dataset.features, &y, &indices, 0)?;

        // Normalize feature importances
        let total: f32 = self.feature_importances.sum();
        if total > 0.0 {
            self.feature_importances.mapv_inplace(|x| x / total);
        }

        Ok(())
    }

    fn build_node(
        &mut self,
        x: &Array2<f32>,
        y: &[usize],
        indices: &[usize],
        depth: usize,
    ) -> Result<usize> {
        // Check stopping conditions
        if indices.is_empty() {
            let node_idx = self.nodes.len();
            self.nodes.push(TreeNode::leaf(vec![0; self.n_classes]));
            return Ok(node_idx);
        }

        // Count classes
        let mut class_counts = vec![0usize; self.n_classes];
        for &i in indices {
            class_counts[y[i]] += 1;
        }

        // Check if pure node
        let n_classes_present = class_counts.iter().filter(|&&c| c > 0).count();
        if n_classes_present == 1
            || depth >= self.max_depth
            || indices.len() < self.min_samples_split
        {
            let node_idx = self.nodes.len();
            self.nodes.push(TreeNode::leaf(class_counts));
            return Ok(node_idx);
        }

        // Find best split
        let (best_feature, best_threshold, best_gain) =
            self.find_best_split(x, y, indices, &class_counts)?;

        if best_gain <= 0.0 {
            let node_idx = self.nodes.len();
            self.nodes.push(TreeNode::leaf(class_counts));
            return Ok(node_idx);
        }

        // Create split node
        let node_idx = self.nodes.len();
        self.nodes
            .push(TreeNode::split(best_feature, best_threshold, class_counts));

        // Split indices
        let (left_indices, right_indices): (Vec<_>, Vec<_>) = indices
            .iter()
            .partition(|&&i| x[[i, best_feature]] <= best_threshold);

        // Update feature importance
        self.feature_importances[best_feature] += best_gain * indices.len() as f32;

        // Build children
        let left_idx = self.build_node(x, y, &left_indices, depth + 1)?;
        let right_idx = self.build_node(x, y, &right_indices, depth + 1)?;

        // Update node with children
        self.nodes[node_idx].left = Some(left_idx);
        self.nodes[node_idx].right = Some(right_idx);

        Ok(node_idx)
    }

    fn find_best_split(
        &self,
        x: &Array2<f32>,
        y: &[usize],
        indices: &[usize],
        class_counts: &[usize],
    ) -> Result<(usize, f32, f32)> {
        let n = indices.len() as f32;
        let parent_gini = Self::gini(class_counts);

        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gain = 0.0;

        // Try each feature
        for feat in 0..45 {
            // Get unique thresholds
            let mut values: Vec<f32> = indices.iter().map(|&i| x[[i, feat]]).collect();
            // Use total_cmp for proper ordering (handles NaN)
            values.sort_by(|a: &f32, b: &f32| a.total_cmp(b));

            // Try thresholds between consecutive values
            for i in 1..values.len() {
                if (values[i] - values[i - 1]).abs() < 1e-10 {
                    continue;
                }
                let threshold = (values[i] + values[i - 1]) / 2.0;

                // Calculate gain
                let mut left_counts = vec![0usize; self.n_classes];
                let mut right_counts = vec![0usize; self.n_classes];

                for &idx in indices {
                    if x[[idx, feat]] <= threshold {
                        left_counts[y[idx]] += 1;
                    } else {
                        right_counts[y[idx]] += 1;
                    }
                }

                let left_n = left_counts.iter().sum::<usize>() as f32;
                let right_n = right_counts.iter().sum::<usize>() as f32;

                if left_n == 0.0 || right_n == 0.0 {
                    continue;
                }

                let left_gini = Self::gini(&left_counts);
                let right_gini = Self::gini(&right_counts);

                let weighted_gini = (left_n / n) * left_gini + (right_n / n) * right_gini;
                let gain = parent_gini - weighted_gini;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feat;
                    best_threshold = threshold;
                }
            }
        }

        Ok((best_feature, best_threshold, best_gain))
    }

    fn gini(class_counts: &[usize]) -> f32 {
        let n = class_counts.iter().sum::<usize>() as f32;
        if n == 0.0 {
            return 0.0;
        }

        let mut sum_sq = 0.0;
        for &c in class_counts {
            let p = c as f32 / n;
            sum_sq += p * p;
        }

        1.0 - sum_sq
    }

    /// Predict class for a single sample
    pub fn predict(&self, features: &Array1<f32>) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }

        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];

            if node.is_leaf() {
                return node.prediction.unwrap_or(0);
            }

            let feat = node.feature_idx.unwrap();
            let thresh = node.threshold.unwrap();

            if features[feat] <= thresh {
                node_idx = node.left.unwrap_or(0);
            } else {
                node_idx = node.right.unwrap_or(0);
            }
        }
    }

    /// Predict classes for multiple samples
    pub fn predict_batch(&self, features: &Array2<f32>) -> Vec<usize> {
        features
            .rows()
            .into_iter()
            .map(|row: ndarray::ArrayView1<f32>| self.predict(&row.to_owned()))
            .collect()
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> &Array1<f32> {
        &self.feature_importances
    }

    /// Get top N most important features
    pub fn top_features(&self, n: usize) -> Vec<(usize, f32)> {
        let mut importances: Vec<(usize, f32)> = self
            .feature_importances
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        importances.sort_by(|a, b| b.1.total_cmp(&a.1));
        importances.into_iter().take(n).collect()
    }
}

// ============================================================================
// Random Forest Classifier
// ============================================================================

/// Class weight mode for Random Forest
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ClassWeightMode {
    /// No class weighting (default)
    #[default]
    None,
    /// Automatic balanced weighting: n_samples / (n_classes * n_samples_for_class)
    Balanced,
    /// Custom weights per class index
    Custom(HashMap<usize, f32>),
}

/// Random Forest Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestClassifier {
    /// Decision trees
    trees: Vec<DecisionTreeClassifier>,
    /// Number of trees
    n_estimators: usize,
    /// Maximum depth per tree
    max_depth: usize,
    /// Minimum samples to split
    min_samples_split: usize,
    /// Number of features to consider at each split
    max_features: Option<usize>,
    /// Number of classes
    n_classes: usize,
    /// Feature importances (averaged across trees)
    feature_importances: Array1<f32>,
    /// Class weight mode
    class_weight: ClassWeightMode,
    /// Computed class weights
    computed_weights: HashMap<usize, f32>,
    /// Label to index mapping
    pub label_to_idx: HashMap<String, usize>,
    /// Index to label mapping
    pub idx_to_label: HashMap<usize, String>,
}

impl RandomForestClassifier {
    /// Create a new Random Forest
    pub fn new(n_estimators: usize, max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_estimators,
            max_depth,
            min_samples_split,
            max_features: None,
            n_classes: 0,
            feature_importances: Array1::zeros(45),
            class_weight: ClassWeightMode::None,
            computed_weights: HashMap::new(),
            label_to_idx: HashMap::new(),
            idx_to_label: HashMap::new(),
        }
    }

    /// Enable balanced class weighting
    pub fn with_balanced_weights(mut self) -> Self {
        self.class_weight = ClassWeightMode::Balanced;
        self
    }

    /// Set custom class weights
    pub fn with_class_weights(mut self, weights: HashMap<usize, f32>) -> Self {
        self.class_weight = ClassWeightMode::Custom(weights);
        self
    }

    /// Set max features per split
    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = Some(max_features);
        self
    }

    /// Compute class weights from dataset
    fn compute_class_weights(&mut self, dataset: &FeatureDataset) {
        self.computed_weights.clear();

        match &self.class_weight {
            ClassWeightMode::None => {
                // All weights = 1.0
                for &class_idx in dataset.label_to_idx.values() {
                    self.computed_weights.insert(class_idx, 1.0);
                }
            }
            ClassWeightMode::Balanced => {
                // Compute balanced weights: n_samples / (n_classes * n_samples_for_class)
                let n_samples = dataset.len();
                let n_classes = dataset.num_classes();

                // Count samples per class
                let mut class_counts: HashMap<usize, usize> = HashMap::new();
                for label in &dataset.labels {
                    let class_idx = dataset.label_to_idx.get(label).copied().unwrap_or(0);
                    *class_counts.entry(class_idx).or_default() += 1;
                }

                // Compute weights
                for (&class_idx, &count) in &class_counts {
                    let weight = (n_samples as f32) / (n_classes as f32 * count as f32);
                    self.computed_weights.insert(class_idx, weight);
                }
            }
            ClassWeightMode::Custom(weights) => {
                self.computed_weights = weights.clone();
            }
        }
    }

    /// Train the Random Forest
    pub fn fit(&mut self, dataset: &FeatureDataset) -> Result<()> {
        if dataset.is_empty() {
            anyhow::bail!("Cannot train on empty dataset");
        }

        self.n_classes = dataset.num_classes();
        self.trees.clear();
        self.feature_importances = Array1::zeros(45);

        // Store label mappings
        self.label_to_idx = dataset.label_to_idx.clone();
        self.idx_to_label = dataset.idx_to_label.clone();

        // Compute class weights
        self.compute_class_weights(dataset);

        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        for tree_idx in 0..self.n_estimators {
            let mut rng = rand::rngs::StdRng::seed_from_u64((tree_idx + 42) as u64);

            // Bootstrap sample (with class-weighted sampling if enabled)
            let n_samples = dataset.len();
            let sample_size = (n_samples as f32 * 0.8) as usize;

            let bootstrap_indices = match &self.class_weight {
                ClassWeightMode::None => {
                    // Standard bootstrap
                    let all_indices: Vec<usize> = (0..n_samples).collect();
                    let mut indices = Vec::with_capacity(sample_size);
                    for _ in 0..sample_size {
                        let idx = *all_indices.choose(&mut rng).unwrap();
                        indices.push(idx);
                    }
                    indices
                }
                _ => {
                    // Weighted bootstrap: oversample minority classes
                    let mut indices = Vec::with_capacity(sample_size);

                    // Group samples by class
                    let mut class_samples: HashMap<usize, Vec<usize>> = HashMap::new();
                    for (i, label) in dataset.labels.iter().enumerate() {
                        let class_idx = dataset.label_to_idx.get(label).copied().unwrap_or(0);
                        class_samples.entry(class_idx).or_default().push(i);
                    }

                    // Sample from each class proportionally to its weight
                    let total_weight: f32 = self.computed_weights.values().sum();
                    for (&class_idx, samples) in &class_samples {
                        let weight = self
                            .computed_weights
                            .get(&class_idx)
                            .copied()
                            .unwrap_or(1.0);
                        let n_to_sample =
                            ((weight / total_weight) * sample_size as f32 * self.n_classes as f32)
                                as usize;
                        let n_to_sample = n_to_sample.min(samples.len()).max(1);

                        for _ in 0..n_to_sample {
                            let idx = *samples.choose(&mut rng).unwrap();
                            indices.push(idx);
                        }
                    }

                    // Fill remaining with weighted random sampling
                    while indices.len() < sample_size {
                        // Select class by weight
                        let r: f32 = rng.gen_range(0.0..total_weight);
                        let mut cumsum = 0.0;
                        for (&class_idx, samples) in &class_samples {
                            cumsum += self
                                .computed_weights
                                .get(&class_idx)
                                .copied()
                                .unwrap_or(1.0);
                            if cumsum >= r && !samples.is_empty() {
                                let idx = *samples.choose(&mut rng).unwrap();
                                indices.push(idx);
                                break;
                            }
                        }
                    }

                    indices
                }
            };

            // Create bootstrap dataset
            let mut bootstrap_dataset = FeatureDataset::new();
            bootstrap_dataset.label_to_idx = dataset.label_to_idx.clone();
            bootstrap_dataset.idx_to_label = dataset.idx_to_label.clone();

            for &i in &bootstrap_indices {
                bootstrap_dataset
                    .add_sample(dataset.features.row(i).to_owned(), &dataset.labels[i]);
            }

            // Train tree
            let mut tree = DecisionTreeClassifier::new(self.max_depth, self.min_samples_split);
            tree.fit(&bootstrap_dataset)?;

            // Accumulate feature importances
            self.feature_importances = &self.feature_importances + tree.feature_importances();

            self.trees.push(tree);
        }

        // Average feature importances
        if self.n_estimators > 0 {
            self.feature_importances
                .mapv_inplace(|x| x / self.n_estimators as f32);
        }

        Ok(())
    }

    /// Predict class for a single sample (majority voting)
    pub fn predict(&self, features: &Array1<f32>) -> usize {
        if self.trees.is_empty() {
            return 0;
        }

        // Count votes
        let mut votes = vec![0usize; self.n_classes];
        for tree in &self.trees {
            let pred = tree.predict(features);
            votes[pred] += 1;
        }

        // Return class with most votes
        votes
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Predict classes for multiple samples
    pub fn predict_batch(&self, features: &Array2<f32>) -> Vec<usize> {
        features
            .rows()
            .into_iter()
            .map(|row: ndarray::ArrayView1<f32>| self.predict(&row.to_owned()))
            .collect()
    }

    /// Get prediction probabilities
    pub fn predict_proba(&self, features: &Array1<f32>) -> Array1<f32> {
        if self.trees.is_empty() {
            return Array1::zeros(self.n_classes);
        }

        // Count votes
        let mut votes = vec![0usize; self.n_classes];
        for tree in &self.trees {
            let pred = tree.predict(features);
            votes[pred] += 1;
        }

        // Convert to probabilities
        let total = self.trees.len() as f32;
        Array1::from_vec(votes.iter().map(|&c| c as f32 / total).collect())
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> &Array1<f32> {
        &self.feature_importances
    }

    /// Get top N most important features
    pub fn top_features(&self, n: usize) -> Vec<(usize, f32)> {
        let mut importances: Vec<(usize, f32)> = self
            .feature_importances
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        importances.sort_by(|a, b| b.1.total_cmp(&a.1));
        importances.into_iter().take(n).collect()
    }

    /// Get number of trees
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }
}

// ============================================================================
// Evaluation Metrics
// ============================================================================

/// Classification metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    /// Overall accuracy
    pub accuracy: f32,
    /// Macro F1 score
    pub macro_f1: f32,
    /// Weighted F1 score
    pub weighted_f1: f32,
    /// Per-class metrics
    pub per_class: HashMap<String, ClassMetrics>,
    /// Confusion matrix (predicted x actual counts)
    pub confusion_matrix: Vec<Vec<usize>>,
}

/// Per-class metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassMetrics {
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub support: usize,
}

/// Evaluate classifier predictions
pub fn evaluate_predictions(
    predictions: &[usize],
    labels: &[usize],
    idx_to_label: &HashMap<usize, String>,
) -> ClassificationMetrics {
    let n_classes = idx_to_label.len();
    let n = predictions.len();

    if n == 0 {
        return ClassificationMetrics::default();
    }

    // Calculate confusion matrix
    let mut confusion_matrix = vec![vec![0usize; n_classes]; n_classes];
    let mut correct = 0;

    for (&pred, &actual) in predictions.iter().zip(labels.iter()) {
        if pred < n_classes && actual < n_classes {
            confusion_matrix[pred][actual] += 1;
            if pred == actual {
                correct += 1;
            }
        }
    }

    // Calculate per-class metrics
    let mut per_class: HashMap<String, ClassMetrics> = HashMap::new();

    for (&class_idx, class_name) in idx_to_label {
        // True positives
        let tp = if class_idx < n_classes {
            confusion_matrix[class_idx][class_idx]
        } else {
            0
        };

        // False positives (predicted as class but actually other)
        let fp: usize = if class_idx < n_classes {
            confusion_matrix[class_idx]
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != class_idx)
                .map(|(_, &c)| c)
                .sum()
        } else {
            0
        };

        // False negatives (actually class but predicted as other)
        let fn_: usize = confusion_matrix
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != class_idx)
            .map(|(i, row)| {
                if class_idx < n_classes {
                    row[class_idx]
                } else {
                    0
                }
            })
            .sum();

        // Support (actual count)
        let support: usize = confusion_matrix
            .iter()
            .map(|row| {
                if class_idx < n_classes {
                    row[class_idx]
                } else {
                    0
                }
            })
            .sum();

        let precision = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f32 / (tp + fn_) as f32
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        per_class.insert(
            class_name.clone(),
            ClassMetrics {
                precision,
                recall,
                f1,
                support,
            },
        );
    }

    // Calculate macro F1
    let macro_f1 = per_class.values().map(|m| m.f1).sum::<f32>() / per_class.len() as f32;

    // Calculate weighted F1
    let total_support: usize = per_class.values().map(|m| m.support).sum();
    let weighted_f1 = if total_support > 0 {
        per_class
            .values()
            .map(|m| m.f1 * m.support as f32 / total_support as f32)
            .sum()
    } else {
        0.0
    };

    ClassificationMetrics {
        accuracy: correct as f32 / n as f32,
        macro_f1,
        weighted_f1,
        per_class,
        confusion_matrix,
    }
}

// ============================================================================
// Hierarchical Classifier
// ============================================================================

/// Taxonomic group for hierarchical classification
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaxonomicGroup {
    /// Cetaceans (whales, dolphins)
    Cetacean,
    /// Birds (songbirds, parrots, etc.)
    Bird,
    /// Insects (mosquitoes, etc.)
    Insect,
    /// Primates (marmosets, gibbons, etc.)
    Primate,
    /// Bats
    Bat,
    /// Other mammals
    Mammal,
    /// Amphibians (frogs, etc.)
    Amphibian,
    /// Unknown or other
    #[default]
    Unknown,
}

impl TaxonomicGroup {
    /// Detect taxonomic group from species name
    pub fn from_species_name(name: &str) -> Self {
        let name_lower = name.to_lowercase();

        // Cetaceans
        if name_lower.contains("whale")
            || name_lower.contains("dolphin")
            || name_lower.contains("orca")
            || name_lower.contains("porpoise")
            || name_lower.contains("sperm")
            || name_lower.contains("humpback")
        {
            return Self::Cetacean;
        }

        // Birds
        if name_lower.contains("bird")
            || name_lower.contains("finch")
            || name_lower.contains("sparrow")
            || name_lower.contains("thrush")
            || name_lower.contains("wren")
            || name_lower.contains("warbler")
            || name_lower.contains("crow")
            || name_lower.contains("raven")
            || name_lower.contains("parrot")
            || name_lower.contains("owl")
            || name_lower.contains("eagle")
            || name_lower.contains("hawk")
            || name_lower.contains("swainson")
            || name_lower.contains("ovenbird")
            || name_lower.contains("song")
            || name_lower.contains("call")
        {
            return Self::Bird;
        }

        // Insects
        if name_lower.contains("mosquito")
            || name_lower.contains("insect")
            || name_lower.contains("bee")
            || name_lower.contains("cricket")
            || name_lower.contains("an arabiensis")
            || name_lower.contains("an. ")
            || name_lower.contains("ae. ")
            || name_lower.contains("culex")
            || name_lower.contains("aedes")
        {
            return Self::Insect;
        }

        // Primates
        if name_lower.contains("marmoset")
            || name_lower.contains("monkey")
            || name_lower.contains("ape")
            || name_lower.contains("chimp")
            || name_lower.contains("gorilla")
            || name_lower.contains("gibbon")
            || name_lower.contains("lemur")
            || name_lower.contains("meerkat")
        {
            return Self::Primate;
        }

        // Bats
        if name_lower.contains("bat")
            || name_lower.contains("fruit bat")
            || name_lower.contains("microbat")
            || name_lower.contains("megabat")
        {
            return Self::Bat;
        }

        // Amphibians
        if name_lower.contains("frog")
            || name_lower.contains("toad")
            || name_lower.contains("salamander")
            || name_lower.contains("newt")
        {
            return Self::Amphibian;
        }

        // Other mammals
        if name_lower.contains("dog")
            || name_lower.contains("cat")
            || name_lower.contains("wolf")
            || name_lower.contains("bear")
            || name_lower.contains("deer")
            || name_lower.contains("elephant")
        {
            return Self::Mammal;
        }

        Self::Unknown
    }

    /// Get all variants
    pub fn all() -> Vec<Self> {
        vec![
            Self::Cetacean,
            Self::Bird,
            Self::Insect,
            Self::Primate,
            Self::Bat,
            Self::Mammal,
            Self::Amphibian,
            Self::Unknown,
        ]
    }
}

/// Hierarchical Classifier with two levels:
/// Level 1: Predict Taxonomic Group (Cetacean, Bird, etc.)
/// Level 2: Predict Species within Group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalClassifier {
    /// Level 1 classifier (taxonomic group)
    pub level1: RandomForestClassifier,
    /// Level 2 classifiers (one per taxonomic group)
    pub level2: HashMap<TaxonomicGroup, RandomForestClassifier>,
    /// Mapping from species to taxonomic group
    pub species_to_group: HashMap<String, TaxonomicGroup>,
    /// Group-specific label mappings
    pub group_label_maps: HashMap<TaxonomicGroup, (HashMap<String, usize>, HashMap<usize, String>)>,
}

impl HierarchicalClassifier {
    /// Create new hierarchical classifier
    pub fn new(n_estimators: usize, max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            level1: RandomForestClassifier::new(n_estimators, max_depth, min_samples_split),
            level2: HashMap::new(),
            species_to_group: HashMap::new(),
            group_label_maps: HashMap::new(),
        }
    }

    /// Train the hierarchical classifier
    pub fn fit(&mut self, dataset: &FeatureDataset) -> Result<()> {
        // Build species-to-group mapping
        self.species_to_group.clear();
        for label in &dataset.labels {
            let group = TaxonomicGroup::from_species_name(label);
            self.species_to_group.insert(label.clone(), group);
        }

        // Create Level 1 dataset (features -> taxonomic group)
        let mut level1_dataset = FeatureDataset::new();
        let mut group_datasets: HashMap<TaxonomicGroup, FeatureDataset> = HashMap::new();

        for group in TaxonomicGroup::all() {
            group_datasets.insert(group, FeatureDataset::new());
        }

        for i in 0..dataset.len() {
            let features = dataset.features.row(i).to_owned();
            let species_label = &dataset.labels[i];
            let group = self
                .species_to_group
                .get(species_label)
                .copied()
                .unwrap_or(TaxonomicGroup::Unknown);

            // Add to level 1 dataset
            let group_label = format!("{:?}", group);
            level1_dataset.add_sample(features.clone(), &group_label);

            // Add to group-specific dataset
            if let Some(group_dataset) = group_datasets.get_mut(&group) {
                group_dataset.add_sample(features, species_label);
            }
        }

        // Train Level 1 classifier
        println!("Training Level 1 (Taxonomic Group) classifier...");
        println!("  Groups: {} classes", level1_dataset.num_classes());
        self.level1.fit(&level1_dataset)?;

        // Train Level 2 classifiers (one per group)
        self.level2.clear();
        self.group_label_maps.clear();

        for (group, group_dataset) in group_datasets {
            if group_dataset.len() < 10 || group_dataset.num_classes() < 2 {
                continue; // Skip groups with too few samples
            }

            println!(
                "Training Level 2 for {:?}: {} samples, {} species",
                group,
                group_dataset.len(),
                group_dataset.num_classes()
            );

            let mut classifier = RandomForestClassifier::new(
                self.level1.n_estimators,
                self.level1.max_depth,
                self.level1.min_samples_split,
            )
            .with_balanced_weights();

            classifier.fit(&group_dataset)?;

            self.level2.insert(group, classifier);
            self.group_label_maps.insert(
                group,
                (
                    group_dataset.label_to_idx.clone(),
                    group_dataset.idx_to_label.clone(),
                ),
            );
        }

        Ok(())
    }

    /// Predict using hierarchical classification
    pub fn predict(&self, features: &Array1<f32>) -> (String, TaxonomicGroup, f32) {
        // Level 1: Predict taxonomic group
        let group_idx = self.level1.predict(features);

        // Get group label from level 1 classifier
        let group_label = self
            .level1
            .idx_to_label
            .get(&group_idx)
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string());

        // Parse group from string
        let group = match group_label.as_str() {
            "Cetacean" => TaxonomicGroup::Cetacean,
            "Bird" => TaxonomicGroup::Bird,
            "Insect" => TaxonomicGroup::Insect,
            "Primate" => TaxonomicGroup::Primate,
            "Bat" => TaxonomicGroup::Bat,
            "Mammal" => TaxonomicGroup::Mammal,
            "Amphibian" => TaxonomicGroup::Amphibian,
            _ => TaxonomicGroup::Unknown,
        };

        // Level 2: Predict species within group
        if let Some(level2_classifier) = self.level2.get(&group) {
            let species_idx = level2_classifier.predict(features);

            if let Some((_, idx_to_label)) = self.group_label_maps.get(&group) {
                if let Some(species) = idx_to_label.get(&species_idx) {
                    // Get confidence from level 2 classifier
                    let proba = level2_classifier.predict_proba(features);
                    let confidence = proba[species_idx];
                    return (species.clone(), group, confidence);
                }
            }
        }

        // Fallback: return group as species
        (group_label, group, 0.5)
    }

    /// Predict batch
    pub fn predict_batch(&self, features: &Array2<f32>) -> Vec<(String, TaxonomicGroup, f32)> {
        features
            .rows()
            .into_iter()
            .map(|row: ndarray::ArrayView1<f32>| self.predict(&row.to_owned()))
            .collect()
    }

    /// Evaluate hierarchical classifier
    pub fn evaluate(&self, dataset: &FeatureDataset) -> ClassificationMetrics {
        let predictions: Vec<usize> = dataset
            .labels
            .iter()
            .map(|label| dataset.label_to_idx.get(label).copied().unwrap_or(0))
            .collect();

        let mut correct = 0;
        let mut group_correct = 0;

        for i in 0..dataset.len() {
            let features = dataset.features.row(i).to_owned();
            let (predicted_species, predicted_group, _) = self.predict(&features);
            let true_label = &dataset.labels[i];

            // Check species accuracy
            if &predicted_species == true_label {
                correct += 1;
            }

            // Check group accuracy
            let true_group = self
                .species_to_group
                .get(true_label)
                .copied()
                .unwrap_or(TaxonomicGroup::Unknown);
            if predicted_group == true_group {
                group_correct += 1;
            }
        }

        let n = dataset.len();

        ClassificationMetrics {
            accuracy: correct as f32 / n as f32,
            macro_f1: 0.0, // Would need per-class calculation
            weighted_f1: 0.0,
            per_class: HashMap::new(),
            confusion_matrix: vec![],
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_feature_dataset_creation() {
        let dataset = FeatureDataset::new();
        assert!(dataset.is_empty());
        assert_eq!(dataset.len(), 0);
        assert_eq!(dataset.feature_names.len(), 45);
    }

    #[test]
    fn test_feature_dataset_add_sample() {
        let mut dataset = FeatureDataset::new();
        let features = Array1::zeros(45);

        dataset.add_sample(features.clone(), "class_a");
        dataset.add_sample(features, "class_b");

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.num_classes(), 2);
    }

    #[test]
    fn test_feature_dataset_label_mapping() {
        let mut dataset = FeatureDataset::new();
        let features = Array1::zeros(45);

        dataset.add_sample(features.clone(), "bird");
        dataset.add_sample(features.clone(), "whale");
        dataset.add_sample(features, "bird");

        assert_eq!(dataset.num_classes(), 2);
        assert!(dataset.label_to_idx.contains_key("bird"));
        assert!(dataset.label_to_idx.contains_key("whale"));
    }

    #[test]
    fn test_feature_dataset_normalize() {
        let mut dataset = FeatureDataset::new();

        // Add samples with different values
        let mut f1 = Array1::zeros(45);
        f1[0] = 0.0;
        f1[1] = 100.0;

        let mut f2 = Array1::zeros(45);
        f2[0] = 10.0;
        f2[1] = 200.0;

        dataset.add_sample(f1, "a");
        dataset.add_sample(f2, "b");

        dataset.normalize();

        // After normalization, means should be ~0
        let mean0 = dataset.features.column(0).mean().unwrap();
        let mean1 = dataset.features.column(1).mean().unwrap();

        assert!((mean0).abs() < 0.01);
        assert!((mean1).abs() < 0.01);
    }

    #[test]
    fn test_feature_dataset_train_test_split() {
        let mut dataset = FeatureDataset::new();
        let features = Array1::zeros(45);

        for i in 0..100 {
            dataset.add_sample(features.clone(), &format!("class_{}", i % 5));
        }

        let (train, test) = dataset.train_test_split(0.2, 42);

        assert!(train.len() > 70 && train.len() < 90);
        assert!(test.len() > 10 && test.len() < 30);
    }

    #[test]
    fn test_feature_dataset_balance_classes() {
        let mut dataset = FeatureDataset::new();
        let features = Array1::zeros(45);

        // Add imbalanced samples
        for _ in 0..100 {
            dataset.add_sample(features.clone(), "majority");
        }
        for _ in 0..10 {
            dataset.add_sample(features.clone(), "minority");
        }

        let balanced = dataset.balance_classes(42);

        // After balancing, both classes should have 10 samples
        assert_eq!(balanced.len(), 20);
    }

    #[test]
    fn test_feature_statistics() {
        let mut dataset = FeatureDataset::new();

        let mut f1 = Array1::zeros(45);
        f1[0] = 0.0;
        f1[1] = 10.0;

        let mut f2 = Array1::zeros(45);
        f2[0] = 10.0;
        f2[1] = 20.0;

        dataset.add_sample(f1, "a");
        dataset.add_sample(f2, "b");

        let stats = dataset.feature_statistics();

        assert_eq!(stats.n_samples, 2);
        assert_eq!(stats.n_classes, 2);
        assert!((stats.means[0] - 5.0).abs() < 0.01);
        assert!((stats.means[1] - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_decision_tree_creation() {
        let tree = DecisionTreeClassifier::new(10, 2);
        assert_eq!(tree.max_depth, 10);
        assert_eq!(tree.min_samples_split, 2);
    }

    #[test]
    fn test_decision_tree_fit_simple() {
        let mut dataset = FeatureDataset::new();

        // Create simple separable data
        let mut class_a = Array1::zeros(45);
        class_a[0] = 0.0;

        let mut class_b = Array1::zeros(45);
        class_b[0] = 10.0;

        for _ in 0..10 {
            dataset.add_sample(class_a.clone(), "a");
            dataset.add_sample(class_b.clone(), "b");
        }

        let mut tree = DecisionTreeClassifier::new(5, 2);
        tree.fit(&dataset).unwrap();

        assert!(!tree.nodes.is_empty());
    }

    #[test]
    fn test_decision_tree_predict() {
        let mut dataset = FeatureDataset::new();

        let mut class_a = Array1::zeros(45);
        class_a[0] = 0.0;

        let mut class_b = Array1::zeros(45);
        class_b[0] = 10.0;

        for _ in 0..10 {
            dataset.add_sample(class_a.clone(), "a");
            dataset.add_sample(class_b.clone(), "b");
        }

        let mut tree = DecisionTreeClassifier::new(5, 2);
        tree.fit(&dataset).unwrap();

        // Test predictions
        let pred_a = tree.predict(&class_a);
        let pred_b = tree.predict(&class_b);

        // Should predict different classes
        assert_ne!(pred_a, pred_b);
    }

    #[test]
    fn test_decision_tree_feature_importance() {
        let mut dataset = FeatureDataset::new();

        let mut class_a = Array1::zeros(45);
        class_a[0] = 0.0;

        let mut class_b = Array1::zeros(45);
        class_b[0] = 10.0;

        for _ in 0..10 {
            dataset.add_sample(class_a.clone(), "a");
            dataset.add_sample(class_b.clone(), "b");
        }

        let mut tree = DecisionTreeClassifier::new(5, 2);
        tree.fit(&dataset).unwrap();

        let importances = tree.feature_importances();

        // Feature 0 should have highest importance
        let top = tree.top_features(1);
        assert_eq!(top[0].0, 0);
    }

    #[test]
    fn test_random_forest_creation() {
        let rf = RandomForestClassifier::new(100, 10, 2);
        assert_eq!(rf.n_estimators, 100);
        assert_eq!(rf.max_depth, 10);
    }

    #[test]
    fn test_random_forest_fit() {
        let mut dataset = FeatureDataset::new();

        let mut class_a = Array1::zeros(45);
        class_a[0] = 0.0;
        class_a[1] = 5.0;

        let mut class_b = Array1::zeros(45);
        class_b[0] = 10.0;
        class_b[1] = 15.0;

        for _ in 0..20 {
            dataset.add_sample(class_a.clone(), "a");
            dataset.add_sample(class_b.clone(), "b");
        }

        let mut rf = RandomForestClassifier::new(10, 5, 2);
        rf.fit(&dataset).unwrap();

        assert_eq!(rf.n_trees(), 10);
    }

    #[test]
    fn test_random_forest_predict() {
        let mut dataset = FeatureDataset::new();

        let mut class_a = Array1::zeros(45);
        class_a[0] = 0.0;

        let mut class_b = Array1::zeros(45);
        class_b[0] = 10.0;

        for _ in 0..20 {
            dataset.add_sample(class_a.clone(), "a");
            dataset.add_sample(class_b.clone(), "b");
        }

        let mut rf = RandomForestClassifier::new(10, 5, 2);
        rf.fit(&dataset).unwrap();

        let pred_a = rf.predict(&class_a);
        let pred_b = rf.predict(&class_b);

        assert_ne!(pred_a, pred_b);
    }

    #[test]
    fn test_random_forest_predict_batch() {
        let mut dataset = FeatureDataset::new();

        let mut class_a = Array1::zeros(45);
        class_a[0] = 0.0;

        let mut class_b = Array1::zeros(45);
        class_b[0] = 10.0;

        for _ in 0..20 {
            dataset.add_sample(class_a.clone(), "a");
            dataset.add_sample(class_b.clone(), "b");
        }

        let mut rf = RandomForestClassifier::new(10, 5, 2);
        rf.fit(&dataset).unwrap();

        let batch = Array2::from_shape_fn(
            (4, 45),
            |(i, j)| {
                if i < 2 {
                    class_a[j]
                } else {
                    class_b[j]
                }
            },
        );

        let predictions = rf.predict_batch(&batch);

        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_random_forest_feature_importance() {
        let mut dataset = FeatureDataset::new();

        let mut class_a = Array1::zeros(45);
        class_a[0] = 0.0;

        let mut class_b = Array1::zeros(45);
        class_b[0] = 10.0;

        for _ in 0..20 {
            dataset.add_sample(class_a.clone(), "a");
            dataset.add_sample(class_b.clone(), "b");
        }

        let mut rf = RandomForestClassifier::new(10, 5, 2);
        rf.fit(&dataset).unwrap();

        let importances = rf.feature_importances();
        assert_eq!(importances.len(), 45);

        // Feature 0 should have highest importance
        let top = rf.top_features(1);
        assert_eq!(top[0].0, 0);
    }

    #[test]
    fn test_random_forest_predict_proba() {
        let mut dataset = FeatureDataset::new();

        let mut class_a = Array1::zeros(45);
        class_a[0] = 0.0;

        let mut class_b = Array1::zeros(45);
        class_b[0] = 10.0;

        for _ in 0..20 {
            dataset.add_sample(class_a.clone(), "a");
            dataset.add_sample(class_b.clone(), "b");
        }

        let mut rf = RandomForestClassifier::new(10, 5, 2);
        rf.fit(&dataset).unwrap();

        let proba = rf.predict_proba(&class_a);

        assert_eq!(proba.len(), 2);
        // Probabilities should sum to 1
        let sum: f32 = proba.sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_evaluate_predictions() {
        let mut idx_to_label = HashMap::new();
        idx_to_label.insert(0, "a".to_string());
        idx_to_label.insert(1, "b".to_string());

        let predictions = vec![0, 0, 1, 1];
        let labels = vec![0, 0, 1, 0]; // 3 correct, 1 wrong

        let metrics = evaluate_predictions(&predictions, &labels, &idx_to_label);

        assert!((metrics.accuracy - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_evaluate_predictions_perfect() {
        let mut idx_to_label = HashMap::new();
        idx_to_label.insert(0, "a".to_string());
        idx_to_label.insert(1, "b".to_string());

        let predictions = vec![0, 0, 1, 1];
        let labels = vec![0, 0, 1, 1];

        let metrics = evaluate_predictions(&predictions, &labels, &idx_to_label);

        assert!((metrics.accuracy - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gini_calculation() {
        let pure = vec![10, 0, 0];
        assert!((DecisionTreeClassifier::gini(&pure) - 0.0).abs() < 0.01);

        let mixed = vec![5, 5];
        assert!((DecisionTreeClassifier::gini(&mixed) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_classification_metrics_f1() {
        let mut idx_to_label = HashMap::new();
        idx_to_label.insert(0, "a".to_string());
        idx_to_label.insert(1, "b".to_string());

        let predictions = vec![0, 1, 1, 1];
        let labels = vec![0, 0, 1, 1];

        let metrics = evaluate_predictions(&predictions, &labels, &idx_to_label);

        assert!(metrics.macro_f1 > 0.0);
        assert!(metrics.weighted_f1 > 0.0);
    }

    #[test]
    fn test_feature_names_complete() {
        let names = FeatureDataset::default_feature_names();
        assert_eq!(names.len(), 45);

        // Check key features exist
        assert!(names.contains(&"mean_f0_hz".to_string()));
        assert!(names.contains(&"tempo_bpm".to_string()));
        assert!(names.contains(&"spectral_centroid".to_string()));
        assert!(names.contains(&"fm_slope".to_string()));
    }

    #[test]
    fn test_dataset_serialization() {
        let mut dataset = FeatureDataset::new();
        let features = Array1::zeros(45);

        dataset.add_sample(features.clone(), "class_a");
        dataset.add_sample(features, "class_b");

        let json = serde_json::to_string(&dataset).unwrap();
        let decoded: FeatureDataset = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded.num_classes(), 2);
    }

    #[test]
    fn test_random_forest_serialization() {
        let mut dataset = FeatureDataset::new();

        let mut class_a = Array1::zeros(45);
        class_a[0] = 0.0;

        let mut class_b = Array1::zeros(45);
        class_b[0] = 10.0;

        for _ in 0..20 {
            dataset.add_sample(class_a.clone(), "a");
            dataset.add_sample(class_b.clone(), "b");
        }

        let mut rf = RandomForestClassifier::new(5, 3, 2);
        rf.fit(&dataset).unwrap();

        let json = serde_json::to_string(&rf).unwrap();
        let decoded: RandomForestClassifier = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.n_trees(), 5);
    }

    #[test]
    fn test_full_training_pipeline() {
        // Create synthetic dataset
        let mut dataset = FeatureDataset::new();

        for i in 0..100 {
            let mut features = Array1::zeros(45);
            let class = if i < 50 { "class_a" } else { "class_b" };

            // Make features separable
            features[0] = if class == "class_a" { 1.0 } else { 10.0 };
            features[1] = if class == "class_a" { 2.0 } else { 20.0 };

            dataset.add_sample(features, class);
        }

        // Split
        let (train, test) = dataset.train_test_split(0.2, 42);

        // Train Random Forest
        let mut rf = RandomForestClassifier::new(50, 10, 2);
        rf.fit(&train).unwrap();

        // Evaluate on test set
        let test_labels: Vec<usize> = test
            .labels
            .iter()
            .map(|l| test.label_to_idx.get(l).copied().unwrap_or(0))
            .collect();

        let predictions = rf.predict_batch(&test.features);

        let metrics = evaluate_predictions(&predictions, &test_labels, &test.idx_to_label);

        // Should achieve high accuracy on this simple dataset
        assert!(
            metrics.accuracy > 0.9,
            "Accuracy should be > 90% on separable data"
        );
    }
}
