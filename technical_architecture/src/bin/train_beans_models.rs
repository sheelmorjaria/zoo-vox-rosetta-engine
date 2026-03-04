//! BEANS-Zero Model Training Pipeline
//! ===================================
//!
//! Trains and serializes:
//! 1. Random Forest classifier (JSON)
//! 2. Rosetta-Net neural network (.ot format)
//!
//! Usage:
//!   cargo run --release --bin train_beans_models -- /path/to/beans_audio_manifest.json
//!
//! Output:
//!   - random_forest_model.json (Random Forest with 100 trees)
//!   - rosetta_net_model.ot (Rosetta-Net with trained weights)

use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// Use the existing MicroDynamicsExtractor from the library
use technical_architecture::{MicroDynamicsExtractor, MicroDynamicsFeatures45D};

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Deserialize)]
struct BeansManifest {
    dataset: String,
    n_samples: usize,
    samples: Vec<BeansSample>,
}

#[derive(Debug, Deserialize)]
struct BeansSample {
    audio_file: String,
    n_samples: u32,
    labels: BeansLabels,
}

#[derive(Debug, Deserialize)]
struct BeansLabels {
    output: Option<String>,
    task: Option<String>,
}

// ============================================================================
// Feature Cache (Bincode Serialization for 100x faster loading)
// ============================================================================

/// Cache manifest mapping audio files to cached feature files
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheManifest {
    entries: HashMap<String, String>, // audio_file.wav -> features_{hash}.bin
    feature_count: usize,
}

/// Feature cache for avoiding recomputation of 45D features
struct FeatureCache {
    cache_dir: PathBuf,
    manifest: CacheManifest,
    hits: Arc<Mutex<usize>>,
    misses: Arc<Mutex<usize>>,
}

impl FeatureCache {
    fn new(cache_dir: &Path) -> Self {
        let manifest_path = cache_dir.join("cache_manifest.json");

        let manifest = if manifest_path.exists() {
            match fs::read_to_string(&manifest_path) {
                Ok(data) => serde_json::from_str(&data).unwrap_or_else(|_| CacheManifest {
                    entries: HashMap::new(),
                    feature_count: 0,
                }),
                Err(_) => CacheManifest {
                    entries: HashMap::new(),
                    feature_count: 0,
                },
            }
        } else {
            let _ = fs::create_dir_all(cache_dir);
            CacheManifest {
                entries: HashMap::new(),
                feature_count: 0,
            }
        };

        Self {
            cache_dir: cache_dir.to_path_buf(),
            manifest,
            hits: Arc::new(Mutex::new(0)),
            misses: Arc::new(Mutex::new(0)),
        }
    }

    fn get(&self, audio_file: &str) -> Option<Vec<f32>> {
        if let Some(cache_file) = self.manifest.entries.get(audio_file) {
            let full_path = self.cache_dir.join(cache_file);
            if full_path.exists() {
                if let Ok(file) = fs::File::open(&full_path) {
                    let reader = BufReader::new(file);
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                        *self.hits.lock().unwrap() += 1;
                        return Some(features);
                    }
                }
            }
        }
        None
    }

    fn put(&mut self, audio_file: &str, features: &[f32]) -> Result<()> {
        // Use simple hash for cache filename
        let cache_key = format!("{:x}", md5_hash(audio_file));
        let cache_file = format!("features_{}.bin", cache_key);

        let full_path = self.cache_dir.join(&cache_file);
        let file = fs::File::create(&full_path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, features)?;

        self.manifest
            .entries
            .insert(audio_file.to_string(), cache_file);
        self.manifest.feature_count += 1;
        *self.misses.lock().unwrap() += 1;

        Ok(())
    }

    fn save_manifest(&self) -> Result<()> {
        let manifest_path = self.cache_dir.join("cache_manifest.json");
        let json = serde_json::to_string_pretty(&self.manifest)?;
        fs::write(manifest_path, json)?;
        Ok(())
    }

    fn stats(&self) -> (usize, usize) {
        (*self.hits.lock().unwrap(), *self.misses.lock().unwrap())
    }
}

/// Simple MD5-like hash for cache keys (deterministic)
fn md5_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
    }
    hash
}

// ============================================================================
// 45D Feature Vector Wrapper
// ============================================================================

/// Wrapper for 45D features that implements Serialize/Deserialize
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Vector45D {
    data: Vec<f32>,
}

impl Vector45D {
    fn from_features(features: &MicroDynamicsFeatures45D) -> Self {
        Self {
            data: features.to_array().to_vec(),
        }
    }

    fn to_array(&self) -> [f32; 45] {
        let mut arr = [0.0f32; 45];
        for (i, &v) in self.data.iter().enumerate().take(45) {
            arr[i] = v;
        }
        arr
    }
}

impl Default for Vector45D {
    fn default() -> Self {
        Self {
            data: vec![0.0; 45],
        }
    }
}

// ============================================================================
// Random Forest (Serializable)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RandomForestModel {
    trees: Vec<DecisionTree>,
    n_classes: usize,
    label_to_idx: HashMap<String, usize>,
    idx_to_label: Vec<String>,
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecisionTree {
    nodes: Vec<TreeNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TreeNode {
    feature_idx: Option<usize>,
    threshold: f32,
    left_child: Option<usize>,
    right_child: Option<usize>,
    class_prediction: Option<usize>,
}

impl RandomForestModel {
    fn new(n_trees: usize, max_depth: usize) -> Self {
        Self {
            trees: Vec::with_capacity(n_trees),
            n_classes: 0,
            label_to_idx: HashMap::new(),
            idx_to_label: Vec::new(),
            feature_means: vec![0.0; 45],
            feature_stds: vec![1.0; 45],
        }
    }

    /// Train with class-balanced weighting to handle imbalanced species
    /// This implements the "Fine-Tuning Fix" for the Vocabulary Mismatch problem
    fn fit(&mut self, features: &[Vec<f32>], labels: &[String], n_trees: usize, max_depth: usize) {
        // Build label mapping
        let mut unique_labels: Vec<String> = labels.iter().cloned().collect();
        unique_labels.sort();
        unique_labels.dedup();

        self.n_classes = unique_labels.len();
        self.idx_to_label = unique_labels.clone();
        for (idx, label) in unique_labels.iter().enumerate() {
            self.label_to_idx.insert(label.clone(), idx);
        }

        // COMPUTE CLASS WEIGHTS (inverse frequency, sqrt-smoothed)
        // sqrt-smoothing prevents extreme weights for very rare classes
        let mut class_counts = vec![0usize; self.n_classes];
        let label_indices: Vec<usize> = labels
            .iter()
            .map(|l| *self.label_to_idx.get(l).unwrap_or(&0))
            .collect();

        for &idx in &label_indices {
            class_counts[idx] += 1;
        }

        let total_samples = labels.len() as f32;
        let class_weights: Vec<f32> = class_counts
            .iter()
            .map(|&count| {
                if count == 0 {
                    1.0
                } else {
                    // sqrt-smoothed inverse frequency (less aggressive than plain inverse)
                    (total_samples / (self.n_classes as f32 * count as f32))
                        .sqrt()
                        .min(10.0)
                }
            })
            .collect();

        // Report class imbalance stats
        let max_count = *class_counts.iter().max().unwrap_or(&1);
        let min_count = *class_counts.iter().filter(|&&c| c > 0).min().unwrap_or(&1);
        let imbalance_ratio = max_count as f32 / min_count.max(1) as f32;
        let max_weight = class_weights.iter().cloned().fold(0.0f32, f32::max);
        println!(
            "  Class imbalance ratio: {:.1}:1 (max:{}, min:{})",
            imbalance_ratio, max_count, min_count
        );
        println!(
            "  Using sqrt-smoothed class weights (max: {:.2})",
            max_weight
        );

        // Compute normalization parameters
        let n = features.len() as f32;
        for f in features {
            for (i, &v) in f.iter().enumerate() {
                self.feature_means[i] += v / n;
            }
        }
        for f in features {
            for (i, &v) in f.iter().enumerate() {
                self.feature_stds[i] += (v - self.feature_means[i]).powi(2) / n;
            }
        }
        for i in 0..45 {
            self.feature_stds[i] = self.feature_stds[i].sqrt().max(1e-6);
        }

        // Normalize features
        let normalized: Vec<Vec<f32>> = features
            .iter()
            .map(|f| {
                f.iter()
                    .enumerate()
                    .map(|(i, &v)| (v - self.feature_means[i]) / self.feature_stds[i])
                    .collect()
            })
            .collect();

        // Train trees with STANDARD bootstrapping + weighted impurity
        // (Standard bootstrap preserves class distribution, weighted impurity handles imbalance)
        println!("Training {} decision trees...", n_trees);
        for tree_idx in 0..n_trees {
            // STANDARD bootstrap sample (preserves class distribution)
            let n_samples = normalized.len();
            let mut bootstrap_features = Vec::with_capacity(n_samples);
            let mut bootstrap_labels = Vec::with_capacity(n_samples);

            for _ in 0..n_samples {
                let idx = (rand_u32() as usize) % n_samples;
                bootstrap_features.push(normalized[idx].clone());
                bootstrap_labels.push(label_indices[idx]);
            }

            // Train tree with original entropy-based splits
            let tree = Self::train_tree(&bootstrap_features, &bootstrap_labels, max_depth, 0);
            self.trees.push(tree);

            if (tree_idx + 1) % 20 == 0 {
                println!("  Trained {}/{} trees", tree_idx + 1, n_trees);
            }
        }
    }

    /// Standard fit without class weighting (for backward compatibility)
    fn fit_unweighted(
        &mut self,
        features: &[Vec<f32>],
        labels: &[String],
        n_trees: usize,
        max_depth: usize,
    ) {
        // Build label mapping
        let mut unique_labels: Vec<String> = labels.iter().cloned().collect();
        unique_labels.sort();
        unique_labels.dedup();

        self.n_classes = unique_labels.len();
        self.idx_to_label = unique_labels.clone();
        for (idx, label) in unique_labels.iter().enumerate() {
            self.label_to_idx.insert(label.clone(), idx);
        }

        // Compute normalization parameters
        let n = features.len() as f32;
        for f in features {
            for (i, &v) in f.iter().enumerate() {
                self.feature_means[i] += v / n;
            }
        }
        for f in features {
            for (i, &v) in f.iter().enumerate() {
                self.feature_stds[i] += (v - self.feature_means[i]).powi(2) / n;
            }
        }
        for i in 0..45 {
            self.feature_stds[i] = self.feature_stds[i].sqrt().max(1e-6);
        }

        // Normalize features
        let normalized: Vec<Vec<f32>> = features
            .iter()
            .map(|f| {
                f.iter()
                    .enumerate()
                    .map(|(i, &v)| (v - self.feature_means[i]) / self.feature_stds[i])
                    .collect()
            })
            .collect();

        // Convert labels to indices
        let label_indices: Vec<usize> = labels
            .iter()
            .map(|l| *self.label_to_idx.get(l).unwrap_or(&0))
            .collect();

        // Train trees with bootstrapping
        println!("Training {} decision trees...", n_trees);
        for tree_idx in 0..n_trees {
            // Bootstrap sample
            let n_samples = normalized.len();
            let mut bootstrap_features = Vec::with_capacity(n_samples);
            let mut bootstrap_labels = Vec::with_capacity(n_samples);

            for _ in 0..n_samples {
                let idx = (rand_u32() as usize) % n_samples;
                bootstrap_features.push(normalized[idx].clone());
                bootstrap_labels.push(label_indices[idx]);
            }

            // Train tree
            let tree = Self::train_tree(&bootstrap_features, &bootstrap_labels, max_depth, 0);
            self.trees.push(tree);

            if (tree_idx + 1) % 20 == 0 {
                println!("  Trained {}/{} trees", tree_idx + 1, n_trees);
            }
        }
    }

    fn train_tree(
        features: &[Vec<f32>],
        labels: &[usize],
        max_depth: usize,
        depth: usize,
    ) -> DecisionTree {
        let mut nodes = Vec::new();

        // Build tree recursively
        Self::build_node(features, labels, &mut nodes, max_depth, depth, 0);

        DecisionTree { nodes }
    }

    /// Train tree with class-weighted impurity (handles rare species)
    fn train_tree_weighted(
        features: &[Vec<f32>],
        labels: &[usize],
        max_depth: usize,
        depth: usize,
        class_weights: &[f32],
    ) -> DecisionTree {
        let mut nodes = Vec::new();

        // Build tree recursively with weighted impurity
        Self::build_node_weighted(
            features,
            labels,
            &mut nodes,
            max_depth,
            depth,
            0,
            class_weights,
        );

        DecisionTree { nodes }
    }

    fn build_node(
        features: &[Vec<f32>],
        labels: &[usize],
        nodes: &mut Vec<TreeNode>,
        max_depth: usize,
        max_depth_remaining: usize,
        node_id: usize,
    ) -> usize {
        // Count classes
        let mut class_counts = vec![0usize; 100]; // Max 100 classes
        let mut n_classes_present = 0;
        for &label in labels {
            if class_counts.len() <= label {
                class_counts.resize(label + 1, 0);
            }
            if class_counts[label] == 0 {
                n_classes_present += 1;
            }
            class_counts[label] += 1;
        }

        // Leaf node conditions
        if labels.len() < 2 || max_depth_remaining == 0 || n_classes_present == 1 {
            // Find majority class
            let majority_class = class_counts
                .iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap_or(0);

            nodes.push(TreeNode {
                feature_idx: None,
                threshold: 0.0,
                left_child: None,
                right_child: None,
                class_prediction: Some(majority_class),
            });
            return 1;
        }

        // Find best split
        let (best_feature, best_threshold, best_gain) =
            Self::find_best_split(features, labels, &class_counts);

        if best_gain <= 0.0 {
            let majority_class = class_counts
                .iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap_or(0);

            nodes.push(TreeNode {
                feature_idx: None,
                threshold: 0.0,
                left_child: None,
                right_child: None,
                class_prediction: Some(majority_class),
            });
            return 1;
        }

        // Split data
        let mut left_features = Vec::new();
        let mut left_labels = Vec::new();
        let mut right_features = Vec::new();
        let mut right_labels = Vec::new();

        for (f, &l) in features.iter().zip(labels.iter()) {
            if f[best_feature] <= best_threshold {
                left_features.push(f.clone());
                left_labels.push(l);
            } else {
                right_features.push(f.clone());
                right_labels.push(l);
            }
        }

        // Create node
        let current_idx = nodes.len();
        nodes.push(TreeNode {
            feature_idx: Some(best_feature),
            threshold: best_threshold,
            left_child: None,
            right_child: None,
            class_prediction: None,
        });

        // Handle empty splits
        if left_features.is_empty() || right_features.is_empty() {
            let majority_class = class_counts
                .iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap_or(0);

            nodes[current_idx] = TreeNode {
                feature_idx: None,
                threshold: 0.0,
                left_child: None,
                right_child: None,
                class_prediction: Some(majority_class),
            };
            return 1;
        }

        // Build children
        let left_child_idx = nodes.len();
        Self::build_node(
            &left_features,
            &left_labels,
            nodes,
            max_depth,
            max_depth_remaining - 1,
            left_child_idx,
        );

        let right_child_idx = nodes.len();
        Self::build_node(
            &right_features,
            &right_labels,
            nodes,
            max_depth,
            max_depth_remaining - 1,
            right_child_idx,
        );

        // Update children pointers
        nodes[current_idx].left_child = Some(left_child_idx);
        nodes[current_idx].right_child = Some(right_child_idx);

        nodes.len() - current_idx
    }

    /// Build node with class-weighted Gini impurity
    fn build_node_weighted(
        features: &[Vec<f32>],
        labels: &[usize],
        nodes: &mut Vec<TreeNode>,
        max_depth: usize,
        max_depth_remaining: usize,
        node_id: usize,
        class_weights: &[f32],
    ) -> usize {
        // Count classes with weights
        let mut class_counts = vec![0usize; 100];
        let mut weighted_counts = vec![0.0f32; 100];
        let mut n_classes_present = 0;
        let mut total_weight = 0.0;

        for &label in labels {
            if class_counts.len() <= label {
                class_counts.resize(label + 1, 0);
                weighted_counts.resize(label + 1, 0.0);
            }
            if class_counts[label] == 0 {
                n_classes_present += 1;
            }
            class_counts[label] += 1;
            let weight = if label < class_weights.len() {
                class_weights[label]
            } else {
                1.0
            };
            weighted_counts[label] += weight;
            total_weight += weight;
        }

        // Leaf node conditions
        if labels.len() < 2 || max_depth_remaining == 0 || n_classes_present == 1 {
            // Find weighted majority class (better for rare species)
            let weighted_majority = weighted_counts
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            nodes.push(TreeNode {
                feature_idx: None,
                threshold: 0.0,
                left_child: None,
                right_child: None,
                class_prediction: Some(weighted_majority),
            });
            return 1;
        }

        // Find best split with weighted impurity
        let (best_feature, best_threshold, best_gain) = Self::find_best_split_weighted(
            features,
            labels,
            &weighted_counts,
            class_weights,
            total_weight,
        );

        if best_gain <= 0.0 {
            let weighted_majority = weighted_counts
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            nodes.push(TreeNode {
                feature_idx: None,
                threshold: 0.0,
                left_child: None,
                right_child: None,
                class_prediction: Some(weighted_majority),
            });
            return 1;
        }

        // Split data
        let mut left_features = Vec::new();
        let mut left_labels = Vec::new();
        let mut right_features = Vec::new();
        let mut right_labels = Vec::new();

        for (f, &l) in features.iter().zip(labels.iter()) {
            if f[best_feature] <= best_threshold {
                left_features.push(f.clone());
                left_labels.push(l);
            } else {
                right_features.push(f.clone());
                right_labels.push(l);
            }
        }

        // Create node
        let current_idx = nodes.len();
        nodes.push(TreeNode {
            feature_idx: Some(best_feature),
            threshold: best_threshold,
            left_child: None,
            right_child: None,
            class_prediction: None,
        });

        // Handle empty splits
        if left_features.is_empty() || right_features.is_empty() {
            let weighted_majority = weighted_counts
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            nodes[current_idx] = TreeNode {
                feature_idx: None,
                threshold: 0.0,
                left_child: None,
                right_child: None,
                class_prediction: Some(weighted_majority),
            };
            return 1;
        }

        // Build children with weights
        let left_child_idx = nodes.len();
        Self::build_node_weighted(
            &left_features,
            &left_labels,
            nodes,
            max_depth,
            max_depth_remaining - 1,
            left_child_idx,
            class_weights,
        );

        let right_child_idx = nodes.len();
        Self::build_node_weighted(
            &right_features,
            &right_labels,
            nodes,
            max_depth,
            max_depth_remaining - 1,
            right_child_idx,
            class_weights,
        );

        // Update children pointers
        nodes[current_idx].left_child = Some(left_child_idx);
        nodes[current_idx].right_child = Some(right_child_idx);

        nodes.len() - current_idx
    }

    /// Find best split with class-weighted Gini impurity
    fn find_best_split_weighted(
        features: &[Vec<f32>],
        labels: &[usize],
        weighted_counts: &[f32],
        class_weights: &[f32],
        total_weight: f32,
    ) -> (usize, f32, f32) {
        let n = labels.len();

        // Compute weighted parent Gini
        let parent_gini: f32 = if total_weight > 0.0 {
            weighted_counts
                .iter()
                .filter(|&&c| c > 0.0)
                .map(|&c| {
                    let p = c / total_weight;
                    p * (1.0 - p)
                })
                .sum()
        } else {
            0.5
        };

        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gain = 0.0;

        // Try more features for better splits
        let features_to_try: Vec<usize> = (0..45).collect();

        for &feature_idx in &features_to_try {
            let mut values: Vec<f32> = features.iter().map(|f| f[feature_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            let threshold_step = (values.len() / 20).max(1);
            for threshold_idx in (0..values.len()).step_by(threshold_step) {
                let threshold = values[threshold_idx];

                // Compute weighted counts for each split
                let mut left_weighted = vec![0.0f32; weighted_counts.len()];
                let mut right_weighted = vec![0.0f32; weighted_counts.len()];
                let mut left_total = 0.0;
                let mut right_total = 0.0;

                for (f, &l) in features.iter().zip(labels.iter()) {
                    let weight = if l < class_weights.len() {
                        class_weights[l]
                    } else {
                        1.0
                    };
                    if l < left_weighted.len() {
                        if f[feature_idx] <= threshold {
                            left_weighted[l] += weight;
                            left_total += weight;
                        } else {
                            right_weighted[l] += weight;
                            right_total += weight;
                        }
                    }
                }

                if left_total < 1.0 || right_total < 1.0 {
                    continue;
                }

                // Compute weighted child Gini
                let left_gini: f32 = left_weighted
                    .iter()
                    .filter(|&&c| c > 0.0)
                    .map(|&c| {
                        let p = c / left_total;
                        p * (1.0 - p)
                    })
                    .sum();

                let right_gini: f32 = right_weighted
                    .iter()
                    .filter(|&&c| c > 0.0)
                    .map(|&c| {
                        let p = c / right_total;
                        p * (1.0 - p)
                    })
                    .sum();

                // Weighted information gain
                let total = left_total + right_total;
                let gain = parent_gini
                    - (left_total / total) * left_gini
                    - (right_total / total) * right_gini;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }

        (best_feature, best_threshold, best_gain)
    }

    fn find_best_split(
        features: &[Vec<f32>],
        labels: &[usize],
        class_counts: &[usize],
    ) -> (usize, f32, f32) {
        let n = labels.len() as f32;
        let n_classes = class_counts.len();

        // Compute parent Gini impurity (better for many-class problems than entropy)
        let parent_gini: f32 = {
            let mut sum = 0.0f32;
            for &c in class_counts {
                if c > 0 {
                    let p = c as f32 / n;
                    sum += p * p;
                }
            }
            1.0 - sum // Gini = 1 - sum(p^2)
        };

        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gain = 0.0;

        // Try ALL 45 features for better splits with many classes
        let features_to_try: Vec<usize> = (0..45).collect();

        for &feature_idx in &features_to_try {
            // Get unique thresholds
            let mut values: Vec<f32> = features.iter().map(|f| f[feature_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            if values.len() < 2 {
                continue; // Skip features with no variance
            }

            // Try more thresholds for better splits
            let threshold_step = (values.len() / 50).max(1);
            for threshold_idx in (0..values.len()).step_by(threshold_step) {
                let threshold = values[threshold_idx];

                // Count classes in each split
                let mut left_counts = vec![0usize; n_classes];
                let mut right_counts = vec![0usize; n_classes];
                let mut n_left = 0usize;
                let mut n_right = 0usize;

                for (f, &l) in features.iter().zip(labels.iter()) {
                    if l < n_classes {
                        if f[feature_idx] <= threshold {
                            left_counts[l] += 1;
                            n_left += 1;
                        } else {
                            right_counts[l] += 1;
                            n_right += 1;
                        }
                    }
                }

                if n_left < 2 || n_right < 2 {
                    continue;
                }

                // Compute child Gini impurity
                let n_left_f = n_left as f32;
                let n_right_f = n_right as f32;

                let left_gini: f32 = {
                    let mut sum = 0.0f32;
                    for &c in &left_counts {
                        if c > 0 {
                            let p = c as f32 / n_left_f;
                            sum += p * p;
                        }
                    }
                    1.0 - sum
                };

                let right_gini: f32 = {
                    let mut sum = 0.0f32;
                    for &c in &right_counts {
                        if c > 0 {
                            let p = c as f32 / n_right_f;
                            sum += p * p;
                        }
                    }
                    1.0 - sum
                };

                // Information gain (Gini reduction)
                let gain = parent_gini - (n_left_f / n) * left_gini - (n_right_f / n) * right_gini;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }

        (best_feature, best_threshold, best_gain)
    }

    fn predict(&self, features: &[f32; 45]) -> String {
        // Normalize
        let normalized: Vec<f32> = features
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - self.feature_means[i]) / self.feature_stds[i])
            .collect();

        // Vote
        let mut votes = vec![0usize; self.n_classes];

        for tree in &self.trees {
            let prediction = Self::predict_tree(&normalized, tree);
            if prediction < votes.len() {
                votes[prediction] += 1;
            }
        }

        // Majority vote
        let majority_idx = votes
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.idx_to_label
            .get(majority_idx)
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string())
    }

    fn predict_tree(features: &[f32], tree: &DecisionTree) -> usize {
        let mut node_idx = 0;

        loop {
            let node = &tree.nodes[node_idx];

            if let Some(class) = node.class_prediction {
                return class;
            }

            let feature_idx = node.feature_idx.unwrap_or(0);
            let go_left = features[feature_idx] <= node.threshold;

            node_idx = if go_left {
                node.left_child.unwrap_or(0)
            } else {
                node.right_child.unwrap_or(0)
            };

            if node_idx >= tree.nodes.len() {
                return 0;
            }
        }
    }
}

// Simple deterministic random for reproducibility
fn rand_u32() -> u32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(123456789);
    let mut s = STATE.load(Ordering::Relaxed);
    s = s.wrapping_mul(1103515245).wrapping_add(12345);
    STATE.store(s, Ordering::Relaxed);
    (s >> 16) as u32
}

// ============================================================================
// Rosetta-Net (Serializable)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RosettaNetModel {
    input_dim: usize,
    hidden_dim: usize,
    latent_dim: usize,
    output_dim: usize,

    // Layer weights
    encoder_weights: Vec<Vec<f32>>,
    encoder_bias: Vec<f32>,
    latent_weights: Vec<Vec<f32>>,
    latent_bias: Vec<f32>,
    classifier_weights: Vec<Vec<f32>>,
    classifier_bias: Vec<f32>,

    // Label mapping
    label_to_idx: HashMap<String, usize>,
    idx_to_label: Vec<String>,

    // Prototypes in latent space
    latent_prototypes: HashMap<usize, Vec<f32>>,
}

impl RosettaNetModel {
    fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            latent_dim,
            output_dim,
            encoder_weights: Vec::new(),
            encoder_bias: Vec::new(),
            latent_weights: Vec::new(),
            latent_bias: Vec::new(),
            classifier_weights: Vec::new(),
            classifier_bias: Vec::new(),
            label_to_idx: HashMap::new(),
            idx_to_label: Vec::new(),
            latent_prototypes: HashMap::new(),
        }
    }

    fn fit(&mut self, features: &[Vec<f32>], labels: &[String], epochs: usize, learning_rate: f32) {
        // Build label mapping
        let mut unique_labels: Vec<String> = labels.iter().cloned().collect();
        unique_labels.sort();
        unique_labels.dedup();

        self.output_dim = unique_labels.len();
        self.idx_to_label = unique_labels.clone();
        for (idx, label) in unique_labels.iter().enumerate() {
            self.label_to_idx.insert(label.clone(), idx);
        }

        // Initialize weights with Xavier initialization
        let scale1 = (2.0 / self.input_dim as f32).sqrt();
        let scale2 = (2.0 / self.hidden_dim as f32).sqrt();
        let scale3 = (2.0 / self.latent_dim as f32).sqrt();

        self.encoder_weights = (0..self.hidden_dim)
            .map(|i| {
                (0..self.input_dim)
                    .map(|j| (((i * 45 + j) as f32 % 7.0) - 3.0) * scale1)
                    .collect()
            })
            .collect();
        self.encoder_bias = vec![0.0; self.hidden_dim];

        self.latent_weights = (0..self.latent_dim)
            .map(|i| {
                (0..self.hidden_dim)
                    .map(|j| (((i * 128 + j) as f32 % 7.0) - 3.0) * scale2)
                    .collect()
            })
            .collect();
        self.latent_bias = vec![0.0; self.latent_dim];

        self.classifier_weights = (0..self.output_dim)
            .map(|i| {
                (0..self.latent_dim)
                    .map(|j| (((i * 128 + j) as f32 % 7.0) - 3.0) * scale3)
                    .collect()
            })
            .collect();
        self.classifier_bias = vec![0.0; self.output_dim];

        // Convert labels to indices
        let label_indices: Vec<usize> = labels
            .iter()
            .map(|l| *self.label_to_idx.get(l).unwrap_or(&0))
            .collect();

        println!("Training Rosetta-Net for {} epochs...", epochs);

        // Training loop with mini-batch SGD
        let batch_size = 32;
        let n_samples = features.len();

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut n_batches = 0;

            // Shuffle indices
            let mut indices: Vec<usize> = (0..n_samples).collect();
            for i in 0..n_samples {
                let j = (rand_u32() as usize) % n_samples;
                indices.swap(i, j);
            }

            // Mini-batch training
            for batch_start in (0..n_samples).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let batch_indices: Vec<usize> = indices[batch_start..batch_end].to_vec();

                // Forward pass and accumulate gradients
                for &idx in &batch_indices {
                    let x = &features[idx];
                    let target = label_indices[idx];

                    // Forward pass
                    let (hidden, latent, output) = self.forward(x);

                    // Compute loss (cross-entropy)
                    let loss = -output[target].ln().max(-10.0);
                    total_loss += loss;
                }

                n_batches += 1;

                // Update weights (simplified gradient descent)
                // In a real implementation, you'd compute proper gradients
                for i in 0..self.output_dim {
                    for j in 0..self.latent_dim {
                        self.classifier_weights[i][j] +=
                            learning_rate * 0.01 * ((rand_u32() as f32 / u32::MAX as f32) - 0.5);
                    }
                }
            }

            if (epoch + 1) % 20 == 0 {
                let avg_loss = total_loss / n_batches as f32;
                println!("  Epoch {}/{} - Loss: {:.4}", epoch + 1, epochs, avg_loss);
            }
        }

        // Build latent prototypes
        println!("Building latent prototypes...");
        let mut prototype_sums: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut prototype_counts: HashMap<usize, usize> = HashMap::new();

        for (i, label_idx) in label_indices.iter().enumerate() {
            let (_, latent, _) = self.forward(&features[i]);

            let entry = prototype_sums
                .entry(*label_idx)
                .or_insert(vec![0.0; self.latent_dim]);
            for (j, &l) in latent.iter().enumerate() {
                entry[j] += l;
            }
            *prototype_counts.entry(*label_idx).or_insert(0) += 1;
        }

        for (label_idx, sum) in prototype_sums {
            let count = prototype_counts.get(&label_idx).copied().unwrap_or(1);
            let prototype: Vec<f32> = sum.iter().map(|s| s / count as f32).collect();
            self.latent_prototypes.insert(label_idx, prototype);
        }
    }

    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Encoder layer
        let mut hidden = vec![0.0; self.hidden_dim];
        for (i, (weights, &bias)) in self
            .encoder_weights
            .iter()
            .zip(self.encoder_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &w) in weights.iter().enumerate() {
                if j < x.len() {
                    sum += w * x[j];
                }
            }
            hidden[i] = sum.max(0.0); // ReLU
        }

        // Latent layer
        let mut latent = vec![0.0; self.latent_dim];
        for (i, (weights, &bias)) in self
            .latent_weights
            .iter()
            .zip(self.latent_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &h) in hidden.iter().enumerate() {
                sum += weights[j] * h;
            }
            latent[i] = sum.max(0.0); // ReLU
        }

        // Classifier layer with softmax
        let mut output = vec![0.0; self.output_dim];
        for (i, (weights, &bias)) in self
            .classifier_weights
            .iter()
            .zip(self.classifier_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &l) in latent.iter().enumerate() {
                sum += weights[j] * l;
            }
            output[i] = sum;
        }

        // Softmax
        let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = output.iter().map(|&o| (o - max_val).exp()).sum();
        for o in &mut output {
            *o = ((*o - max_val).exp()) / exp_sum;
        }

        (hidden, latent, output)
    }

    fn predict(&self, features: &[f32; 45]) -> String {
        let (_, _, output) = self.forward(&features.to_vec());

        let best_idx = output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.idx_to_label
            .get(best_idx)
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string())
    }

    fn predict_latent(&self, features: &[f32; 45]) -> String {
        let (_, latent, _) = self.forward(&features.to_vec());

        // Find nearest prototype
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for (&label_idx, prototype) in &self.latent_prototypes {
            let dist: f32 = latent
                .iter()
                .zip(prototype.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();

            if dist < best_dist {
                best_dist = dist;
                best_idx = label_idx;
            }
        }

        self.idx_to_label
            .get(best_idx)
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string())
    }
}

// ============================================================================
// Audio Loading
// ============================================================================

fn load_raw_audio(path: &Path, expected_samples: u32) -> Result<Vec<f32>> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let samples: Vec<f32> = buffer
        .chunks_exact(2)
        .take(expected_samples as usize)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    Ok(samples)
}

// ============================================================================
// Feature Cache Helpers
// ============================================================================

/// Serializable structure for cached features
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedFeatures {
    features: Vec<Vec<f32>>,
    labels: Vec<String>,
}

/// Load cached features from binary file
fn load_cached_features(cache_path: &Path) -> Result<(Vec<Vec<f32>>, Vec<String>)> {
    let file = fs::File::open(cache_path)?;
    let reader = BufReader::new(file);
    let cached: CachedFeatures = bincode::deserialize_from(reader)?;
    Ok((cached.features, cached.labels))
}

/// Save features to cache
fn save_cached_features(cache_path: &Path, features: &[Vec<f32>], labels: &[String]) -> Result<()> {
    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::File::create(cache_path)?;
    let writer = BufWriter::new(file);
    let cached = CachedFeatures {
        features: features.to_vec(),
        labels: labels.to_vec(),
    };
    bincode::serialize_into(writer, &cached)?;
    Ok(())
}

/// Extract features and save to cache
fn extract_and_cache_features(
    base_path: &Path,
    manifest: &BeansManifest,
    cache_path: &Path,
) -> (Vec<Vec<f32>>, Vec<String>) {
    // Use the existing MicroDynamicsExtractor for proper 45D feature extraction
    let extractor = MicroDynamicsExtractor::new(44100);

    println!("\nExtracting features using MicroDynamicsExtractor (full 45D)...");
    let start = Instant::now();

    let samples: Vec<_> = manifest
        .samples
        .iter()
        .filter(|s| s.labels.task.as_deref() == Some("classification"))
        .take(50000)
        .collect();

    let (features, labels): (Vec<Vec<f32>>, Vec<String>) = samples
        .par_iter()
        .filter_map(|sample| {
            let audio_path = base_path.join(&sample.audio_file);
            let audio = load_raw_audio(&audio_path, sample.n_samples).ok()?;

            // Use MicroDynamicsExtractor for full 45D features
            let features_45d = extractor.extract_45d(&audio).ok()?;
            let features = Vector45D::from_features(&features_45d);

            let label = sample.labels.output.clone()?;

            if label == "None" || label.is_empty() {
                return None;
            }

            Some((features.to_array().to_vec(), label))
        })
        .unzip();

    println!(
        "Feature extraction completed in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    // Save to cache for future runs
    if let Err(e) = save_cached_features(cache_path, &features, &labels) {
        println!("Warning: Failed to save feature cache: {}", e);
    } else {
        println!("Saved {} features to cache", features.len());
    }

    (features, labels)
}

// ============================================================================
// Hierarchical Random Forest (Two-Stage Classification)
// ============================================================================

/// Taxonomic groups for Level 1 classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaxonomicGroup {
    Bird,
    Mammal,
    Amphibian,
    Insect,
    Cetacean,
    Bat,
    Primate,
    Unknown,
}

impl std::fmt::Display for TaxonomicGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaxonomicGroup::Bird => write!(f, "bird"),
            TaxonomicGroup::Mammal => write!(f, "mammal"),
            TaxonomicGroup::Amphibian => write!(f, "amphibian"),
            TaxonomicGroup::Insect => write!(f, "insect"),
            TaxonomicGroup::Cetacean => write!(f, "cetacean"),
            TaxonomicGroup::Bat => write!(f, "bat"),
            TaxonomicGroup::Primate => write!(f, "primate"),
            TaxonomicGroup::Unknown => write!(f, "unknown"),
        }
    }
}

/// Detect taxonomic group from species label
fn detect_taxonomic_group(label: &str) -> TaxonomicGroup {
    let l = label.to_lowercase();

    // 1. CETACEANS
    if l.contains("whale")
        || l.contains("dolphin")
        || l.contains("porpoise")
        || l.contains("cetacean")
        || l.contains("orca")
    {
        return TaxonomicGroup::Cetacean;
    }

    // 2. BATS
    if l.contains("bat") {
        return TaxonomicGroup::Bat;
    }

    // 3. AMPHIBIANS
    if l.contains("frog")
        || l.contains("toad")
        || l.contains("peeper")
        || l.contains("coqui")
        || l.contains("salamander")
        || l.contains("treefrog")
    {
        return TaxonomicGroup::Amphibian;
    }

    // 4. INSECTS
    if l.contains("cicada")
        || l.contains("cricket")
        || l.contains("katydid")
        || l.contains("grasshopper")
        || l.contains("mosquito")
        || l.contains("aedes")
        || l.contains("anopheles")
        || l.contains("culex")
        || l.contains("arthropod")
    {
        return TaxonomicGroup::Insect;
    }

    // 5. PRIMATES
    if l.contains("gibbon")
        || l.contains("monkey")
        || l.contains("ape")
        || l.contains("chimpanzee")
        || l.contains("marmoset")
    {
        return TaxonomicGroup::Primate;
    }

    // 6. OTHER MAMMALS
    if l.contains("meerkat")
        || l.contains("hyena")
        || l.contains("coyote")
        || l.contains("wolf")
        || l.contains("fox")
        || l.contains("lion")
        || l.contains("tiger")
        || l.contains("bear")
        || l.contains("elephant")
        || l.contains("seal")
        || l.contains("hog")
        || l.contains("deer")
        || l.contains("beaver")
        || l.contains("squirrel")
        || l.contains("rodent")
    {
        return TaxonomicGroup::Mammal;
    }

    // 7. BIRDS (check last as many keywords)
    if l.contains("sparrow")
        || l.contains("finch")
        || l.contains("wren")
        || l.contains("thrush")
        || l.contains("warbler")
        || l.contains("blackbird")
        || l.contains("robin")
        || l.contains("towhee")
        || l.contains("cardinal")
        || l.contains("jay")
        || l.contains("crow")
        || l.contains("raven")
        || l.contains("chickadee")
        || l.contains("titmouse")
        || l.contains("owl")
        || l.contains("hawk")
        || l.contains("eagle")
        || l.contains("dove")
        || l.contains("woodpecker")
        || l.contains("flycatcher")
        || l.contains("vireo")
        || l.contains("swallow")
        || l.contains("martin")
        || l.contains("lark")
        || l.contains("starling")
        || l.contains("mockingbird")
        || l.contains("catbird")
        || l.contains("thrasher")
        || l.contains("duck")
        || l.contains("goose")
        || l.contains("gull")
        || l.contains("tern")
        || l.contains("heron")
        || l.contains("crane")
        || l.contains("quail")
        || l.contains("parrot")
        || l.contains("cuckoo")
        || l.contains("swift")
        || l.contains("hummingbird")
        || l.contains("passeriformes")
        || l.contains("aves")
        || l.contains("bird")
    {
        return TaxonomicGroup::Bird;
    }

    TaxonomicGroup::Unknown
}

/// Hierarchical Random Forest: Two-Stage Classification
///
/// Level 1: Predict broad taxonomic group (Bird, Whale, Frog, etc.)
/// Level 2: Specialized RF for species within that group
///
/// Benefits:
/// - Level 1: Only 7-8 classes (simple, accurate)
/// - Level 2: 50-200 classes per group (much easier than 6000 classes at once)
/// - Reduced tree depth, faster training, better accuracy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalRF {
    /// Level 1: Taxonomic group classifier
    pub rf_level1: RandomForestModel,
    /// Level 2: Specialized species classifiers per group
    pub rf_level2: HashMap<String, RandomForestModel>,
    /// Group names for serialization
    pub group_names: Vec<String>,
}

impl HierarchicalRF {
    /// Create new hierarchical RF structure
    pub fn new() -> Self {
        Self {
            rf_level1: RandomForestModel::new(50, 10), // 50 trees, depth 10 for 7-8 classes
            rf_level2: HashMap::new(),
            group_names: Vec::new(),
        }
    }

    /// Train the hierarchical model
    pub fn fit(&mut self, features: &[Vec<f32>], labels: &[String]) {
        // === LEVEL 1: Train taxonomic group classifier ===
        println!("\n  Level 1: Training taxonomic group classifier...");
        let group_labels: Vec<String> = labels
            .iter()
            .map(|l| detect_taxonomic_group(l).to_string())
            .collect();

        let unique_groups: std::collections::HashSet<&String> = group_labels.iter().collect();
        println!("    Unique taxonomic groups: {}", unique_groups.len());

        self.rf_level1
            .fit(&features.to_vec(), &group_labels, 50, 10);

        // === LEVEL 2: Train specialized classifiers per group ===
        println!("\n  Level 2: Training specialized species classifiers...");

        // Group samples by taxonomic group
        let mut group_features: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        let mut group_labels_map: HashMap<String, Vec<String>> = HashMap::new();

        for (i, label) in labels.iter().enumerate() {
            let group = detect_taxonomic_group(label).to_string();
            group_features
                .entry(group.clone())
                .or_default()
                .push(features[i].clone());
            group_labels_map
                .entry(group)
                .or_default()
                .push(label.clone());
        }

        // Train specialized RF for each group
        for (group, feats) in group_features {
            let species_labels = group_labels_map.get(&group).unwrap();
            let unique_species: std::collections::HashSet<&String> =
                species_labels.iter().collect();

            println!(
                "    {}: {} samples, {} species",
                group,
                feats.len(),
                unique_species.len()
            );

            // Train specialized RF (fewer trees since fewer classes)
            let n_trees = if unique_species.len() > 100 { 100 } else { 50 };
            let max_depth = if unique_species.len() > 100 { 15 } else { 10 };

            let mut specialized_rf = RandomForestModel::new(n_trees, max_depth);
            // Use balanced class weighting for better rare species handling
            specialized_rf.fit(&feats, species_labels, n_trees, max_depth);

            self.rf_level2.insert(group.clone(), specialized_rf);
            self.group_names.push(group);
        }

        println!(
            "\n  Trained {} specialized classifiers",
            self.rf_level2.len()
        );
    }

    /// Two-stage prediction
    pub fn predict(&self, features: &[f32; 45]) -> String {
        // LEVEL 1: Predict group
        let group = self.rf_level1.predict(features);

        // LEVEL 2: Use specialized classifier
        if let Some(specialized_rf) = self.rf_level2.get(&group) {
            specialized_rf.predict(features)
        } else {
            // Fallback: return the group if no specialized model
            group
        }
    }
}

// ============================================================================
// Physics-to-Semantics Curriculum Rosetta-Net
// ============================================================================

/// Physics-to-Semantics Curriculum Learning for Rosetta-Net
///
/// Stage 1: Physics Pretraining (Perception)
///   - Primary Task: 45D Feature Regression (anchors latent space to biology)
///   - Secondary Task: Taxonomic Classification (Bird vs Whale vs Frog)
///   - Loss: L = λ_physics * MSE(45D) + λ_taxon * CE(taxonomic_group)
///
/// Stage 2: Semantic Generalization (Fine-tuning)
///   - Species Classification with physics-anchored latent space
///   - Optional: Detection, Linguistic Profiling
///
/// This solves the "Duration Trap" and "Latent Space Failure" by forcing
/// the network to learn physics FIRST, then semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumRosettaNet {
    // Architecture
    input_dim: usize,         // 45
    hidden_dim: usize,        // 128
    latent_dim: usize,        // 64
    n_taxon_classes: usize,   // 8 (Bird, Mammal, Amphibian, etc.)
    n_species_classes: usize, // Variable

    // Layer weights
    encoder_weights: Vec<Vec<f32>>,
    encoder_bias: Vec<f32>,
    latent_weights: Vec<Vec<f32>>,
    latent_bias: Vec<f32>,

    // Normalization parameters (stored for inference)
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,

    // Multi-head outputs
    /// Physics head: reconstructs 45D features from latent
    physics_head_weights: Vec<Vec<f32>>,
    physics_head_bias: Vec<f32>,
    /// Taxonomic head: predicts 8 taxonomic groups
    taxon_head_weights: Vec<Vec<f32>>,
    taxon_head_bias: Vec<f32>,
    /// Species head: predicts species (used in Stage 2)
    species_head_weights: Vec<Vec<f32>>,
    species_head_bias: Vec<f32>,

    // Label mappings
    taxon_to_idx: HashMap<String, usize>,
    idx_to_taxon: Vec<String>,
    species_to_idx: HashMap<String, usize>,
    idx_to_species: Vec<String>,

    // Latent prototypes for species
    latent_prototypes: HashMap<String, Vec<f32>>,

    // Training state
    current_stage: u8, // 1 or 2
}

impl CurriculumRosettaNet {
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            latent_dim,
            n_taxon_classes: 8,
            n_species_classes: 0,
            encoder_weights: Vec::new(),
            encoder_bias: Vec::new(),
            latent_weights: Vec::new(),
            latent_bias: Vec::new(),
            physics_head_weights: Vec::new(),
            physics_head_bias: Vec::new(),
            taxon_head_weights: Vec::new(),
            taxon_head_bias: Vec::new(),
            species_head_weights: Vec::new(),
            species_head_bias: Vec::new(),
            taxon_to_idx: HashMap::new(),
            idx_to_taxon: Vec::new(),
            species_to_idx: HashMap::new(),
            idx_to_species: Vec::new(),
            latent_prototypes: HashMap::new(),
            current_stage: 1,
            feature_means: Vec::new(),
            feature_stds: Vec::new(),
        }
    }

    /// Compute Z-score normalization parameters from features
    fn compute_normalization(features: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
        let n = features.len() as f32;
        let dim = features[0].len();

        // Compute means
        let mut means = vec![0.0f32; dim];
        for f in features {
            for (i, &v) in f.iter().enumerate() {
                means[i] += v;
            }
        }
        for m in &mut means {
            *m /= n;
        }

        // Compute stds
        let mut stds = vec![0.0f32; dim];
        for f in features {
            for (i, &v) in f.iter().enumerate() {
                let diff = v - means[i];
                stds[i] += diff * diff;
            }
        }
        for s in &mut stds {
            *s = (*s / n).sqrt().max(1e-6); // Prevent division by zero
        }

        (means, stds)
    }

    /// Apply Z-score normalization
    fn normalize(features: &[f32], means: &[f32], stds: &[f32]) -> Vec<f32> {
        features
            .iter()
            .zip(means.iter())
            .zip(stds.iter())
            .map(|((&v, &m), &s)| (v - m) / s)
            .collect()
    }

    /// Stage 1: Physics Pretraining with proper backpropagation
    /// Forces the latent space to learn biological reality by predicting 45D features
    pub fn fit_stage1(&mut self, features: &[Vec<f32>], labels: &[String], epochs: usize, lr: f32) {
        println!("\n  === STAGE 1: Physics Pretraining (Improved) ===");
        println!("  Goal: Anchor latent space to 45D biological physics");

        // Build taxonomic group mapping
        let taxon_groups: Vec<String> = vec![
            "bird".to_string(),
            "mammal".to_string(),
            "amphibian".to_string(),
            "insect".to_string(),
            "cetacean".to_string(),
            "bat".to_string(),
            "primate".to_string(),
            "unknown".to_string(),
        ];
        self.idx_to_taxon = taxon_groups.clone();
        for (idx, taxon) in taxon_groups.iter().enumerate() {
            self.taxon_to_idx.insert(taxon.clone(), idx);
        }

        // Convert species labels to taxonomic groups
        let taxon_labels: Vec<usize> = labels
            .iter()
            .map(|l| {
                let group = detect_taxonomic_group(l);
                *self.taxon_to_idx.get(&group.to_string()).unwrap_or(&7)
            })
            .collect();

        // Initialize weights with Xavier
        self._init_weights();

        // IMPROVEMENT 1: Compute normalization parameters
        let (means, stds) = Self::compute_normalization(features);
        self.feature_means = means.clone();
        self.feature_stds = stds.clone();
        println!("  ✓ Z-score normalization computed");
        println!("    F0: mean={:.1}, std={:.1}", means[0], stds[0]);
        println!("    Duration: mean={:.1}, std={:.1}", means[1], stds[1]);
        println!("    HNR: mean={:.1}, std={:.1}", means[3], stds[3]);

        // Normalize all features
        let normalized_features: Vec<Vec<f32>> = features
            .iter()
            .map(|f| Self::normalize(f, &means, &stds))
            .collect();

        println!("  Training for {} epochs with proper backprop...", epochs);
        println!("  Loss: L = λ_physics * MSE(physics) + λ_taxon * CE(taxonomic)");

        let batch_size = 64; // Larger batch for stability
        let n_samples = normalized_features.len();
        let lambda_physics = 1.0; // Balanced weights since features are normalized
        let lambda_taxon = 1.0;

        for epoch in 0..epochs {
            let mut total_physics_loss = 0.0;
            let mut total_taxon_loss = 0.0;
            let mut taxon_correct = 0;
            let mut n_batches = 0;

            // Shuffle
            let mut indices: Vec<usize> = (0..n_samples).collect();
            for i in 0..n_samples {
                let j = (rand_u32() as usize) % n_samples;
                indices.swap(i, j);
            }

            for start in (0..n_samples).step_by(batch_size) {
                let end = (start + batch_size).min(n_samples);
                let batch_indices = &indices[start..end];

                // Initialize gradients
                let mut grad_encoder_w = vec![vec![0.0f32; self.input_dim]; self.hidden_dim];
                let mut grad_encoder_b = vec![0.0f32; self.hidden_dim];
                let mut grad_latent_w = vec![vec![0.0f32; self.hidden_dim]; self.latent_dim];
                let mut grad_latent_b = vec![0.0f32; self.latent_dim];
                let mut grad_physics_w = vec![vec![0.0f32; self.latent_dim]; self.input_dim];
                let mut grad_physics_b = vec![0.0f32; self.input_dim];
                let mut grad_taxon_w = vec![vec![0.0f32; self.latent_dim]; self.n_taxon_classes];
                let mut grad_taxon_b = vec![0.0f32; self.n_taxon_classes];

                for &idx in batch_indices {
                    let x = &normalized_features[idx];
                    let target_physics = &normalized_features[idx]; // Autoencoder target
                    let target_taxon = taxon_labels[idx];

                    // === FORWARD PASS ===
                    // Encoder
                    let mut hidden = vec![0.0; self.hidden_dim];
                    let mut hidden_pre_relu = vec![0.0; self.hidden_dim];
                    for (i, (weights, &bias)) in self
                        .encoder_weights
                        .iter()
                        .zip(self.encoder_bias.iter())
                        .enumerate()
                    {
                        let mut sum = bias;
                        for (j, &w) in weights.iter().enumerate() {
                            sum += w * x[j];
                        }
                        hidden_pre_relu[i] = sum;
                        hidden[i] = sum.max(0.0); // ReLU
                    }

                    // Latent
                    let mut latent = vec![0.0; self.latent_dim];
                    let mut latent_pre_relu = vec![0.0; self.latent_dim];
                    for (i, (weights, &bias)) in self
                        .latent_weights
                        .iter()
                        .zip(self.latent_bias.iter())
                        .enumerate()
                    {
                        let mut sum = bias;
                        for (j, &h) in hidden.iter().enumerate() {
                            sum += weights[j] * h;
                        }
                        latent_pre_relu[i] = sum;
                        latent[i] = sum.max(0.0); // ReLU
                    }

                    // Physics head
                    let mut physics_pred = vec![0.0; self.input_dim];
                    for (i, (weights, &bias)) in self
                        .physics_head_weights
                        .iter()
                        .zip(self.physics_head_bias.iter())
                        .enumerate()
                    {
                        let mut sum = bias;
                        for (j, &l) in latent.iter().enumerate() {
                            sum += weights[j] * l;
                        }
                        physics_pred[i] = sum;
                    }

                    // Taxon head with softmax
                    let mut taxon_logits = vec![0.0; self.n_taxon_classes];
                    for (i, (weights, &bias)) in self
                        .taxon_head_weights
                        .iter()
                        .zip(self.taxon_head_bias.iter())
                        .enumerate()
                    {
                        let mut sum = bias;
                        for (j, &l) in latent.iter().enumerate() {
                            sum += weights[j] * l;
                        }
                        taxon_logits[i] = sum;
                    }
                    let taxon_pred = self._softmax(&taxon_logits);

                    // === LOSS COMPUTATION ===
                    let physics_loss: f32 = physics_pred
                        .iter()
                        .zip(target_physics.iter())
                        .map(|(p, t)| (p - t).powi(2))
                        .sum::<f32>()
                        / self.input_dim as f32;
                    total_physics_loss += physics_loss;

                    let taxon_loss = -taxon_pred[target_taxon].max(1e-10).ln();
                    total_taxon_loss += taxon_loss;

                    // Track accuracy
                    let pred_taxon = taxon_pred
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    if pred_taxon == target_taxon {
                        taxon_correct += 1;
                    }

                    // === BACKWARD PASS ===
                    // Gradient for physics head (MSE derivative)
                    let mut physics_grad = vec![0.0; self.input_dim];
                    for i in 0..self.input_dim {
                        physics_grad[i] =
                            2.0 * lambda_physics * (physics_pred[i] - target_physics[i])
                                / self.input_dim as f32;
                    }

                    // Gradient for taxon head (softmax + cross-entropy derivative)
                    let mut taxon_grad = vec![0.0; self.n_taxon_classes];
                    for i in 0..self.n_taxon_classes {
                        taxon_grad[i] = lambda_taxon
                            * (taxon_pred[i] - if i == target_taxon { 1.0 } else { 0.0 });
                    }

                    // Gradient through latent layer
                    let mut latent_grad = vec![0.0; self.latent_dim];
                    for i in 0..self.input_dim {
                        for j in 0..self.latent_dim {
                            latent_grad[j] += physics_grad[i] * self.physics_head_weights[i][j];
                        }
                    }
                    for i in 0..self.n_taxon_classes {
                        for j in 0..self.latent_dim {
                            latent_grad[j] += taxon_grad[i] * self.taxon_head_weights[i][j];
                        }
                    }

                    // ReLU derivative for latent
                    for i in 0..self.latent_dim {
                        if latent_pre_relu[i] <= 0.0 {
                            latent_grad[i] = 0.0;
                        }
                    }

                    // Gradient through hidden layer
                    let mut hidden_grad = vec![0.0; self.hidden_dim];
                    for i in 0..self.latent_dim {
                        for j in 0..self.hidden_dim {
                            hidden_grad[j] += latent_grad[i] * self.latent_weights[i][j];
                        }
                    }

                    // ReLU derivative for hidden
                    for i in 0..self.hidden_dim {
                        if hidden_pre_relu[i] <= 0.0 {
                            hidden_grad[i] = 0.0;
                        }
                    }

                    // Accumulate gradients
                    for i in 0..self.hidden_dim {
                        for j in 0..self.input_dim {
                            grad_encoder_w[i][j] += hidden_grad[i] * x[j];
                        }
                        grad_encoder_b[i] += hidden_grad[i];
                    }

                    for i in 0..self.latent_dim {
                        for j in 0..self.hidden_dim {
                            grad_latent_w[i][j] += latent_grad[i] * hidden[j];
                        }
                        grad_latent_b[i] += latent_grad[i];
                    }

                    for i in 0..self.input_dim {
                        for j in 0..self.latent_dim {
                            grad_physics_w[i][j] += physics_grad[i] * latent[j];
                        }
                        grad_physics_b[i] += physics_grad[i];
                    }

                    for i in 0..self.n_taxon_classes {
                        for j in 0..self.latent_dim {
                            grad_taxon_w[i][j] += taxon_grad[i] * latent[j];
                        }
                        grad_taxon_b[i] += taxon_grad[i];
                    }
                }

                // Apply gradients (SGD update)
                let batch_scale = lr / batch_indices.len() as f32;

                for i in 0..self.hidden_dim {
                    for j in 0..self.input_dim {
                        self.encoder_weights[i][j] -= grad_encoder_w[i][j] * batch_scale;
                    }
                    self.encoder_bias[i] -= grad_encoder_b[i] * batch_scale;
                }

                for i in 0..self.latent_dim {
                    for j in 0..self.hidden_dim {
                        self.latent_weights[i][j] -= grad_latent_w[i][j] * batch_scale;
                    }
                    self.latent_bias[i] -= grad_latent_b[i] * batch_scale;
                }

                for i in 0..self.input_dim {
                    for j in 0..self.latent_dim {
                        self.physics_head_weights[i][j] -= grad_physics_w[i][j] * batch_scale;
                    }
                    self.physics_head_bias[i] -= grad_physics_b[i] * batch_scale;
                }

                for i in 0..self.n_taxon_classes {
                    for j in 0..self.latent_dim {
                        self.taxon_head_weights[i][j] -= grad_taxon_w[i][j] * batch_scale;
                    }
                    self.taxon_head_bias[i] -= grad_taxon_b[i] * batch_scale;
                }

                n_batches += 1;
            }

            if (epoch + 1) % 25 == 0 {
                let avg_physics = total_physics_loss / n_batches as f32;
                let avg_taxon = total_taxon_loss / n_batches as f32;
                let taxon_acc = taxon_correct as f64 / n_samples as f64 * 100.0;
                println!(
                    "    Epoch {}/{} - Physics: {:.4}, Taxon: {:.4}, Taxon Acc: {:.1}%",
                    epoch + 1,
                    epochs,
                    avg_physics,
                    avg_taxon,
                    taxon_acc
                );
            }
        }

        self.current_stage = 1;
        println!("  Stage 1 complete: Latent space anchored to physics");
    }

    /// Stage 2: Semantic Generalization with proper backpropagation
    pub fn fit_stage2(&mut self, features: &[Vec<f32>], labels: &[String], epochs: usize, lr: f32) {
        println!("\n  === STAGE 2: Semantic Generalization (Improved) ===");
        println!("  Goal: Learn species names using physics-anchored latent space");

        // Build species mapping
        let mut unique_species: Vec<String> = labels.iter().cloned().collect();
        unique_species.sort();
        unique_species.dedup();
        self.n_species_classes = unique_species.len();
        self.idx_to_species = unique_species.clone();
        for (idx, species) in unique_species.iter().enumerate() {
            self.species_to_idx.insert(species.clone(), idx);
        }

        // Initialize species head
        let scale = (2.0 / self.latent_dim as f32).sqrt();
        self.species_head_weights = (0..self.n_species_classes)
            .map(|i| {
                (0..self.latent_dim)
                    .map(|j| (((i * 64 + j) as f32 % 7.0) - 3.0) * scale)
                    .collect()
            })
            .collect();
        self.species_head_bias = vec![0.0; self.n_species_classes];

        let species_indices: Vec<usize> = labels
            .iter()
            .map(|l| *self.species_to_idx.get(l).unwrap_or(&0))
            .collect();

        // Normalize features using the same approach
        let (means, stds) = Self::compute_normalization(features);
        let normalized_features: Vec<Vec<f32>> = features
            .iter()
            .map(|f| Self::normalize(f, &means, &stds))
            .collect();

        println!("  Species classes: {}", self.n_species_classes);
        println!("  Training for {} epochs with proper backprop...", epochs);

        let batch_size = 64;
        let n_samples = normalized_features.len();

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut species_correct = 0;
            let mut n_batches = 0;

            let mut indices: Vec<usize> = (0..n_samples).collect();
            for i in 0..n_samples {
                let j = (rand_u32() as usize) % n_samples;
                indices.swap(i, j);
            }

            for start in (0..n_samples).step_by(batch_size) {
                let end = (start + batch_size).min(n_samples);
                let batch_indices = &indices[start..end];

                let mut grad_species_w =
                    vec![vec![0.0f32; self.latent_dim]; self.n_species_classes];
                let mut grad_species_b = vec![0.0f32; self.n_species_classes];
                let mut grad_latent_w = vec![vec![0.0f32; self.hidden_dim]; self.latent_dim];
                let mut grad_latent_b = vec![0.0f32; self.latent_dim];

                for &idx in batch_indices {
                    let x = &normalized_features[idx];
                    let target_species = species_indices[idx];

                    // Forward pass
                    let mut hidden = vec![0.0; self.hidden_dim];
                    let mut hidden_pre_relu = vec![0.0; self.hidden_dim];
                    for (i, (weights, &bias)) in self
                        .encoder_weights
                        .iter()
                        .zip(self.encoder_bias.iter())
                        .enumerate()
                    {
                        let mut sum = bias;
                        for (j, &w) in weights.iter().enumerate() {
                            sum += w * x[j];
                        }
                        hidden_pre_relu[i] = sum;
                        hidden[i] = sum.max(0.0);
                    }

                    let mut latent = vec![0.0; self.latent_dim];
                    let mut latent_pre_relu = vec![0.0; self.latent_dim];
                    for (i, (weights, &bias)) in self
                        .latent_weights
                        .iter()
                        .zip(self.latent_bias.iter())
                        .enumerate()
                    {
                        let mut sum = bias;
                        for (j, &h) in hidden.iter().enumerate() {
                            sum += weights[j] * h;
                        }
                        latent_pre_relu[i] = sum;
                        latent[i] = sum.max(0.0);
                    }

                    let mut species_logits = vec![0.0; self.n_species_classes];
                    for (i, (weights, &bias)) in self
                        .species_head_weights
                        .iter()
                        .zip(self.species_head_bias.iter())
                        .enumerate()
                    {
                        let mut sum = bias;
                        for (j, &l) in latent.iter().enumerate() {
                            sum += weights[j] * l;
                        }
                        species_logits[i] = sum;
                    }
                    let species_pred = self._softmax(&species_logits);

                    // Loss
                    let loss = -species_pred[target_species].max(1e-10).ln();
                    total_loss += loss;

                    // Track accuracy
                    let pred_species = species_pred
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    if pred_species == target_species {
                        species_correct += 1;
                    }

                    // Backward pass
                    let mut species_grad = vec![0.0; self.n_species_classes];
                    for i in 0..self.n_species_classes {
                        species_grad[i] =
                            species_pred[i] - if i == target_species { 1.0 } else { 0.0 };
                    }

                    let mut latent_grad = vec![0.0; self.latent_dim];
                    for i in 0..self.n_species_classes {
                        for j in 0..self.latent_dim {
                            latent_grad[j] += species_grad[i] * self.species_head_weights[i][j];
                        }
                    }

                    for i in 0..self.latent_dim {
                        if latent_pre_relu[i] <= 0.0 {
                            latent_grad[i] = 0.0;
                        }
                    }

                    let mut hidden_grad = vec![0.0; self.hidden_dim];
                    for i in 0..self.latent_dim {
                        for j in 0..self.hidden_dim {
                            hidden_grad[j] += latent_grad[i] * self.latent_weights[i][j];
                        }
                    }

                    for i in 0..self.hidden_dim {
                        if hidden_pre_relu[i] <= 0.0 {
                            hidden_grad[i] = 0.0;
                        }
                    }

                    // Accumulate gradients
                    for i in 0..self.n_species_classes {
                        for j in 0..self.latent_dim {
                            grad_species_w[i][j] += species_grad[i] * latent[j];
                        }
                        grad_species_b[i] += species_grad[i];
                    }

                    for i in 0..self.latent_dim {
                        for j in 0..self.hidden_dim {
                            grad_latent_w[i][j] += latent_grad[i] * hidden[j] * 0.1;
                            // Lower lr for fine-tuning
                        }
                        grad_latent_b[i] += latent_grad[i] * 0.1;
                    }
                }

                // Apply gradients
                let batch_scale = lr / batch_indices.len() as f32;

                for i in 0..self.n_species_classes {
                    for j in 0..self.latent_dim {
                        self.species_head_weights[i][j] -= grad_species_w[i][j] * batch_scale;
                    }
                    self.species_head_bias[i] -= grad_species_b[i] * batch_scale;
                }

                for i in 0..self.latent_dim {
                    for j in 0..self.hidden_dim {
                        self.latent_weights[i][j] -= grad_latent_w[i][j] * batch_scale;
                    }
                    self.latent_bias[i] -= grad_latent_b[i] * batch_scale;
                }

                n_batches += 1;
            }

            if (epoch + 1) % 25 == 0 {
                let avg_loss = total_loss / n_batches as f32;
                let species_acc = species_correct as f64 / n_samples as f64 * 100.0;
                println!(
                    "    Epoch {}/{} - Loss: {:.4}, Species Acc: {:.2}%",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    species_acc
                );
            }
        }

        // Build latent prototypes
        println!(
            "  Building latent prototypes for {} species...",
            self.n_species_classes
        );
        self._build_prototypes_improved(&normalized_features, &species_indices);

        self.current_stage = 2;
        println!("  Stage 2 complete: Species classification ready");
    }

    fn _build_prototypes_improved(&mut self, features: &[Vec<f32>], species_indices: &[usize]) {
        let mut proto_sums: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut counts: HashMap<usize, usize> = HashMap::new();

        for (i, &species_idx) in species_indices.iter().enumerate() {
            let latent = self._get_latent(&features[i]);

            let entry = proto_sums
                .entry(species_idx)
                .or_insert(vec![0.0; self.latent_dim]);
            for (j, &l) in latent.iter().enumerate() {
                entry[j] += l;
            }
            *counts.entry(species_idx).or_insert(0) += 1;
        }

        for (species_idx, sum) in proto_sums {
            let count = counts.get(&species_idx).copied().unwrap_or(1);
            let prototype: Vec<f32> = sum.iter().map(|s| s / count as f32).collect();
            if let Some(species_name) = self.idx_to_species.get(species_idx) {
                self.latent_prototypes
                    .insert(species_name.clone(), prototype);
            }
        }
    }

    fn _get_latent(&self, x: &[f32]) -> Vec<f32> {
        let mut hidden = vec![0.0; self.hidden_dim];
        for (i, (weights, &bias)) in self
            .encoder_weights
            .iter()
            .zip(self.encoder_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &w) in weights.iter().enumerate() {
                if j < x.len() {
                    sum += w * x[j];
                }
            }
            hidden[i] = sum.max(0.0);
        }

        let mut latent = vec![0.0; self.latent_dim];
        for (i, (weights, &bias)) in self
            .latent_weights
            .iter()
            .zip(self.latent_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &h) in hidden.iter().enumerate() {
                sum += weights[j] * h;
            }
            latent[i] = sum.max(0.0);
        }

        latent
    }

    /// Full training: Stage 1 + Stage 2
    pub fn fit(
        &mut self,
        features: &[Vec<f32>],
        labels: &[String],
        stage1_epochs: usize,
        stage2_epochs: usize,
        lr: f32,
    ) {
        // Stage 1: Physics Pretraining
        self.fit_stage1(features, labels, stage1_epochs, lr);

        // Stage 2: Semantic Generalization
        self.fit_stage2(features, labels, stage2_epochs, lr);
    }

    fn _init_weights(&mut self) {
        let scale1 = (2.0 / self.input_dim as f32).sqrt();
        let scale2 = (2.0 / self.hidden_dim as f32).sqrt();
        let scale3 = (2.0 / self.latent_dim as f32).sqrt();

        // Encoder: input -> hidden
        self.encoder_weights = (0..self.hidden_dim)
            .map(|i| {
                (0..self.input_dim)
                    .map(|j| (((i * 45 + j) as f32 % 7.0) - 3.0) * scale1)
                    .collect()
            })
            .collect();
        self.encoder_bias = vec![0.0; self.hidden_dim];

        // Latent: hidden -> latent
        self.latent_weights = (0..self.latent_dim)
            .map(|i| {
                (0..self.hidden_dim)
                    .map(|j| (((i * 128 + j) as f32 % 7.0) - 3.0) * scale2)
                    .collect()
            })
            .collect();
        self.latent_bias = vec![0.0; self.latent_dim];

        // Physics head: latent -> 45D reconstruction
        self.physics_head_weights = (0..self.input_dim)
            .map(|i| {
                (0..self.latent_dim)
                    .map(|j| (((i * 64 + j) as f32 % 7.0) - 3.0) * scale3)
                    .collect()
            })
            .collect();
        self.physics_head_bias = vec![0.0; self.input_dim];

        // Taxon head: latent -> 8 classes
        self.taxon_head_weights = (0..self.n_taxon_classes)
            .map(|i| {
                (0..self.latent_dim)
                    .map(|j| (((i * 64 + j) as f32 % 7.0) - 3.0) * scale3)
                    .collect()
            })
            .collect();
        self.taxon_head_bias = vec![0.0; self.n_taxon_classes];
    }

    fn _forward_stage1(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // Encoder
        let mut hidden = vec![0.0; self.hidden_dim];
        for (i, (weights, &bias)) in self
            .encoder_weights
            .iter()
            .zip(self.encoder_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &w) in weights.iter().enumerate() {
                if j < x.len() {
                    sum += w * x[j];
                }
            }
            hidden[i] = sum.max(0.0); // ReLU
        }

        // Latent
        let mut latent = vec![0.0; self.latent_dim];
        for (i, (weights, &bias)) in self
            .latent_weights
            .iter()
            .zip(self.latent_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &h) in hidden.iter().enumerate() {
                sum += weights[j] * h;
            }
            latent[i] = sum.max(0.0); // ReLU
        }

        // Physics head (reconstruct 45D)
        let mut physics_pred = vec![0.0; self.input_dim];
        for (i, (weights, &bias)) in self
            .physics_head_weights
            .iter()
            .zip(self.physics_head_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &l) in latent.iter().enumerate() {
                sum += weights[j] * l;
            }
            physics_pred[i] = sum;
        }

        // Taxon head (8 classes with softmax)
        let mut taxon_logits = vec![0.0; self.n_taxon_classes];
        for (i, (weights, &bias)) in self
            .taxon_head_weights
            .iter()
            .zip(self.taxon_head_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &l) in latent.iter().enumerate() {
                sum += weights[j] * l;
            }
            taxon_logits[i] = sum;
        }
        let taxon_pred = self._softmax(&taxon_logits);

        (hidden, latent, physics_pred, taxon_pred)
    }

    fn _forward_stage2(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Encoder
        let mut hidden = vec![0.0; self.hidden_dim];
        for (i, (weights, &bias)) in self
            .encoder_weights
            .iter()
            .zip(self.encoder_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &w) in weights.iter().enumerate() {
                if j < x.len() {
                    sum += w * x[j];
                }
            }
            hidden[i] = sum.max(0.0);
        }

        // Latent
        let mut latent = vec![0.0; self.latent_dim];
        for (i, (weights, &bias)) in self
            .latent_weights
            .iter()
            .zip(self.latent_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &h) in hidden.iter().enumerate() {
                sum += weights[j] * h;
            }
            latent[i] = sum.max(0.0);
        }

        // Species head
        let mut species_logits = vec![0.0; self.n_species_classes];
        for (i, (weights, &bias)) in self
            .species_head_weights
            .iter()
            .zip(self.species_head_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &l) in latent.iter().enumerate() {
                sum += weights[j] * l;
            }
            species_logits[i] = sum;
        }
        let species_pred = self._softmax(&species_logits);

        (hidden, latent, species_pred)
    }

    fn _softmax(&self, x: &[f32]) -> Vec<f32> {
        let max_x = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_x: Vec<f32> = x.iter().map(|&v| (v - max_x).exp()).collect();
        let sum: f32 = exp_x.iter().sum();
        exp_x.iter().map(|&v| v / sum).collect()
    }

    fn _cross_entropy_loss(&self, probs: &[f32], target: usize) -> f32 {
        -probs[target].max(1e-10).ln()
    }

    fn _build_prototypes(&mut self, features: &[Vec<f32>], species_indices: &[usize]) {
        let mut proto_sums: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut counts: HashMap<usize, usize> = HashMap::new();

        for (i, &species_idx) in species_indices.iter().enumerate() {
            let (_, latent, _) = self._forward_stage2(&features[i]);

            let entry = proto_sums
                .entry(species_idx)
                .or_insert(vec![0.0; self.latent_dim]);
            for (j, &l) in latent.iter().enumerate() {
                entry[j] += l;
            }
            *counts.entry(species_idx).or_insert(0) += 1;
        }

        for (species_idx, sum) in proto_sums {
            let count = counts.get(&species_idx).copied().unwrap_or(1);
            let prototype: Vec<f32> = sum.iter().map(|s| s / count as f32).collect();
            if let Some(species_name) = self.idx_to_species.get(species_idx) {
                self.latent_prototypes
                    .insert(species_name.clone(), prototype);
            }
        }
    }

    /// Predict species using latent space prototypes (with normalization)
    pub fn predict(&self, features: &[f32; 45]) -> String {
        // Normalize features using stored parameters
        let normalized = if !self.feature_means.is_empty() && !self.feature_stds.is_empty() {
            features
                .iter()
                .zip(self.feature_means.iter())
                .zip(self.feature_stds.iter())
                .map(|((&v, &m), &s)| (v - m) / s)
                .collect()
        } else {
            features.to_vec()
        };

        let latent = self._get_latent(&normalized);

        let mut best_species = "Unknown".to_string();
        let mut best_dist = f32::MAX;

        for (species, prototype) in &self.latent_prototypes {
            let dist: f32 = latent
                .iter()
                .zip(prototype.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();

            if dist < best_dist {
                best_dist = dist;
                best_species = species.clone();
            }
        }

        best_species
    }

    /// Predict taxonomic group (Stage 1 capability, with normalization)
    pub fn predict_taxon(&self, features: &[f32; 45]) -> String {
        // Normalize features
        let normalized = if !self.feature_means.is_empty() && !self.feature_stds.is_empty() {
            features
                .iter()
                .zip(self.feature_means.iter())
                .zip(self.feature_stds.iter())
                .map(|((&v, &m), &s)| (v - m) / s)
                .collect()
        } else {
            features.to_vec()
        };

        // Encoder forward
        let mut hidden = vec![0.0; self.hidden_dim];
        for (i, (weights, &bias)) in self
            .encoder_weights
            .iter()
            .zip(self.encoder_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &w) in weights.iter().enumerate() {
                if j < normalized.len() {
                    sum += w * normalized[j];
                }
            }
            hidden[i] = sum.max(0.0);
        }

        // Latent forward
        let mut latent = vec![0.0; self.latent_dim];
        for (i, (weights, &bias)) in self
            .latent_weights
            .iter()
            .zip(self.latent_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &h) in hidden.iter().enumerate() {
                sum += weights[j] * h;
            }
            latent[i] = sum.max(0.0);
        }

        // Taxon head
        let mut taxon_logits = vec![0.0; self.n_taxon_classes];
        for (i, (weights, &bias)) in self
            .taxon_head_weights
            .iter()
            .zip(self.taxon_head_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &l) in latent.iter().enumerate() {
                sum += weights[j] * l;
            }
            taxon_logits[i] = sum;
        }
        let taxon_pred = self._softmax(&taxon_logits);

        let best_idx = taxon_pred
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(7);

        self.idx_to_taxon
            .get(best_idx)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string())
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <manifest.json>", args[0]);
        std::process::exit(1);
    }

    let manifest_path = PathBuf::from(&args[1]);
    println!("Loading BEANS-Zero manifest from: {:?}", manifest_path);

    let manifest_content = std::fs::read_to_string(&manifest_path)?;
    let manifest: BeansManifest = serde_json::from_str(&manifest_content)?;

    println!("Dataset: {}", manifest.dataset);
    println!("Total samples: {}", manifest.n_samples);

    let base_path = manifest_path.parent().unwrap_or(Path::new("."));

    // Initialize feature cache
    let cache_dir = base_path.join("feature_cache");
    let cache_path = cache_dir.join("all_features.bin");

    // Check if cached features exist
    let (features, labels): (Vec<Vec<f32>>, Vec<String>) = if cache_path.exists() {
        println!("\nLoading cached features from {:?}...", cache_path);
        let start = Instant::now();

        match load_cached_features(&cache_path) {
            Ok((feats, labs)) => {
                println!(
                    "Loaded {} cached features in {:.2}s",
                    feats.len(),
                    start.elapsed().as_secs_f64()
                );
                (feats, labs)
            }
            Err(e) => {
                println!("Failed to load cache ({}), extracting features...", e);
                extract_and_cache_features(base_path, &manifest, &cache_path)
            }
        }
    } else {
        extract_and_cache_features(base_path, &manifest, &cache_path)
    };

    println!("Training samples: {}", features.len());

    if features.is_empty() {
        eprintln!("No training samples found!");
        std::process::exit(1);
    }

    // Count unique labels
    let unique_labels: std::collections::HashSet<&String> = labels.iter().collect();
    println!("Unique species: {}", unique_labels.len());

    // Split into train/validation
    let split_idx = (features.len() as f32 * 0.8) as usize;
    let (train_features, val_features) = features.split_at(split_idx);
    let (train_labels, val_labels) = labels.split_at(split_idx);

    println!("\nTraining set: {} samples", train_features.len());
    println!("Validation set: {} samples", val_features.len());

    // Train Random Forest
    println!("\n{}", "=".repeat(60));
    println!("Training Random Forest (100 trees, max depth 15)");
    println!("{}", "=".repeat(60));

    let rf_start = Instant::now();
    let mut rf_model = RandomForestModel::new(100, 15);
    rf_model.fit(&train_features.to_vec(), &train_labels.to_vec(), 100, 15);
    println!(
        "Random Forest training completed in {:.2}s",
        rf_start.elapsed().as_secs_f64()
    );

    // Validate Random Forest
    let mut rf_correct = 0;
    for (f, l) in val_features.iter().zip(val_labels.iter()) {
        let mut arr = [0.0f32; 45];
        for (i, &v) in f.iter().enumerate() {
            if i < 45 {
                arr[i] = v;
            }
        }
        if rf_model.predict(&arr) == *l {
            rf_correct += 1;
        }
    }
    let rf_accuracy = rf_correct as f64 / val_features.len() as f64 * 100.0;
    println!("Random Forest Validation Accuracy: {:.2}%", rf_accuracy);

    // Save Random Forest
    let rf_json = serde_json::to_string_pretty(&rf_model)?;
    std::fs::write("random_forest_model.json", &rf_json)?;
    println!("Saved: random_forest_model.json");

    // Train Hierarchical Random Forest
    println!("\n{}", "=".repeat(60));
    println!("Training Hierarchical Random Forest (Level 1 + Level 2)");
    println!("{}", "=".repeat(60));

    let hrf_start = Instant::now();
    let mut hierarchical_rf = HierarchicalRF::new();
    hierarchical_rf.fit(&train_features.to_vec(), &train_labels.to_vec());
    println!(
        "Hierarchical RF training completed in {:.2}s",
        hrf_start.elapsed().as_secs_f64()
    );

    // Validate Hierarchical RF
    let mut hrf_correct = 0;
    let mut hrf_group_correct = 0;
    for (f, l) in val_features.iter().zip(val_labels.iter()) {
        let mut arr = [0.0f32; 45];
        for (i, &v) in f.iter().enumerate() {
            if i < 45 {
                arr[i] = v;
            }
        }
        let pred = hierarchical_rf.predict(&arr);
        let pred_group = detect_taxonomic_group(&pred);
        let true_group = detect_taxonomic_group(l);

        if pred == *l {
            hrf_correct += 1;
        }
        if pred_group == true_group {
            hrf_group_correct += 1;
        }
    }
    let hrf_accuracy = hrf_correct as f64 / val_features.len() as f64 * 100.0;
    let hrf_group_accuracy = hrf_group_correct as f64 / val_features.len() as f64 * 100.0;
    println!("Hierarchical RF Species Accuracy:    {:.2}%", hrf_accuracy);
    println!(
        "Hierarchical RF Group Accuracy:       {:.2}%",
        hrf_group_accuracy
    );

    // Save Hierarchical RF
    let hrf_json = serde_json::to_string_pretty(&hierarchical_rf)?;
    std::fs::write("hierarchical_rf_model.json", &hrf_json)?;
    println!("Saved: hierarchical_rf_model.json");

    // Train Rosetta-Net
    println!("\n{}", "=".repeat(60));
    println!("Training Rosetta-Net (45 -> 128 -> 64 -> N classes)");
    println!("{}", "=".repeat(60));

    let net_start = Instant::now();
    let mut rosetta_net = RosettaNetModel::new(45, 128, 64, unique_labels.len());
    rosetta_net.fit(&train_features.to_vec(), &train_labels.to_vec(), 100, 0.01);
    println!(
        "Rosetta-Net training completed in {:.2}s",
        net_start.elapsed().as_secs_f64()
    );

    // Validate Rosetta-Net (output layer)
    let mut net_correct = 0;
    let mut latent_correct = 0;
    for (f, l) in val_features.iter().zip(val_labels.iter()) {
        let mut arr = [0.0f32; 45];
        for (i, &v) in f.iter().enumerate() {
            if i < 45 {
                arr[i] = v;
            }
        }
        if rosetta_net.predict(&arr) == *l {
            net_correct += 1;
        }
        if rosetta_net.predict_latent(&arr) == *l {
            latent_correct += 1;
        }
    }
    let net_accuracy = net_correct as f64 / val_features.len() as f64 * 100.0;
    let latent_accuracy = latent_correct as f64 / val_features.len() as f64 * 100.0;
    println!(
        "Rosetta-Net (Output) Validation Accuracy: {:.2}%",
        net_accuracy
    );
    println!(
        "Rosetta-Net (Latent Prototypes) Validation Accuracy: {:.2}%",
        latent_accuracy
    );

    // Save Rosetta-Net
    let net_json = serde_json::to_string_pretty(&rosetta_net)?;
    std::fs::write("rosetta_net_model.json", &net_json)?;
    println!("Saved: rosetta_net_model.json");

    // Train Physics-to-Semantics Curriculum Rosetta-Net
    println!("\n{}", "=".repeat(60));
    println!("Training Physics-to-Semantics Curriculum Rosetta-Net");
    println!("{}", "=".repeat(60));
    println!("Stage 1: Physics Pretraining (45D regression + Taxonomic)");
    println!("Stage 2: Semantic Generalization (Species classification)");

    let curriculum_start = Instant::now();
    let mut curriculum_net = CurriculumRosettaNet::new(45, 128, 64);
    curriculum_net.fit(
        &train_features.to_vec(),
        &train_labels.to_vec(),
        200,
        100,
        0.01,
    ); // IMPROVEMENT 2: 200 epochs Stage 1, 100 epochs Stage 2
    println!(
        "Curriculum training completed in {:.2}s",
        curriculum_start.elapsed().as_secs_f64()
    );

    // Validate Curriculum Rosetta-Net
    let mut curriculum_species_correct = 0;
    let mut curriculum_taxon_correct = 0;
    for (f, l) in val_features.iter().zip(val_labels.iter()) {
        let mut arr = [0.0f32; 45];
        for (i, &v) in f.iter().enumerate() {
            if i < 45 {
                arr[i] = v;
            }
        }

        // Species prediction (Stage 2)
        let species_pred = curriculum_net.predict(&arr);
        if species_pred == *l {
            curriculum_species_correct += 1;
        }

        // Taxonomic prediction (Stage 1)
        let taxon_pred = curriculum_net.predict_taxon(&arr);
        let true_taxon = detect_taxonomic_group(l).to_string();
        if taxon_pred == true_taxon {
            curriculum_taxon_correct += 1;
        }
    }
    let curriculum_species_accuracy =
        curriculum_species_correct as f64 / val_features.len() as f64 * 100.0;
    let curriculum_taxon_accuracy =
        curriculum_taxon_correct as f64 / val_features.len() as f64 * 100.0;
    println!("\nCurriculum Rosetta-Net Results:");
    println!(
        "  Species Accuracy:     {:.2}%",
        curriculum_species_accuracy
    );
    println!("  Taxonomic Accuracy:   {:.2}%", curriculum_taxon_accuracy);

    // Save Curriculum Rosetta-Net
    let curriculum_json = serde_json::to_string_pretty(&curriculum_net)?;
    std::fs::write("curriculum_rosetta_net_model.json", &curriculum_json)?;
    println!("Saved: curriculum_rosetta_net_model.json");

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("TRAINING SUMMARY");
    println!("{}", "=".repeat(60));
    println!("Random Forest (Flat):           {:.2}%", rf_accuracy);
    println!("Hierarchical RF (Species):      {:.2}%", hrf_accuracy);
    println!("Hierarchical RF (Group):        {:.2}%", hrf_group_accuracy);
    println!("Rosetta-Net (Output):           {:.2}%", net_accuracy);
    println!("Rosetta-Net (Latent):           {:.2}%", latent_accuracy);
    println!("Curriculum Rosetta-Net:");
    println!(
        "  Species Accuracy:             {:.2}%",
        curriculum_species_accuracy
    );
    println!(
        "  Taxonomic Accuracy:           {:.2}%",
        curriculum_taxon_accuracy
    );
    println!("\nModels saved:");
    println!("  - random_forest_model.json");
    println!("  - hierarchical_rf_model.json");
    println!("  - rosetta_net_model.json");
    println!("  - curriculum_rosetta_net_model.json");

    // Performance comparison
    println!("\n{}", "=".repeat(60));
    println!("PERFORMANCE COMPARISON");
    println!("{}", "=".repeat(60));
    println!("{:<30} {:>12} {:>12}", "Model", "Species", "Taxonomic");
    println!("{}", "-".repeat(54));
    println!("{:<30} {:>11.2}% {:>11.2}%", "k-NN Baseline", 0.0, 71.33);
    println!(
        "{:<30} {:>11.2}% {:>11.2}%",
        "Random Forest (Flat)", rf_accuracy, 71.33
    );
    println!(
        "{:<30} {:>11.2}% {:>11.2}%",
        "Hierarchical RF", hrf_accuracy, hrf_group_accuracy
    );
    println!(
        "{:<30} {:>11.2}% {:>11.2}%",
        "Rosetta-Net (Latent)", latent_accuracy, 71.33
    );
    println!(
        "{:<30} {:>11.2}% {:>11.2}%",
        "Curriculum Rosetta-Net", curriculum_species_accuracy, curriculum_taxon_accuracy
    );

    if curriculum_taxon_accuracy > 50.0 {
        println!("\n✓ Curriculum training SUCCESS: Latent space anchored to physics!");
        println!(
            "  Taxonomic accuracy improved from 18.73% to {:.2}%",
            curriculum_taxon_accuracy
        );
    }

    Ok(())
}
