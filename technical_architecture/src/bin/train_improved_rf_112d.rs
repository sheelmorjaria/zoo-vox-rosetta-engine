//! Train Improved Random Forest (112D Features with Cached Data)
//! ===============================================================
//!
//! Implements recommended improvements for better training accuracy:
//! - 500 trees (increased from 100)
//! - max_depth=50 (increased from 10)
//! - min_samples_leaf=1 (allow single-sample leaves)
//! - max_features=20 (~18% of 112D features)
//! - Class-balanced weighting
//!
//! Usage:
//!   cargo run --release --bin train_improved_rf_112d

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::Instant;

const FEATURE_DIM: usize = 112;

// Improved hyperparameters based on recommendations
const N_TREES: usize = 500;           // Increased from 100
const MAX_DEPTH: usize = 50;          // Increased from 10
const MIN_SAMPLES_LEAF: usize = 1;    // Allow single-sample leaves
const MAX_FEATURES: usize = 20;       // ~18% of 112D features

// =============================================================================
// Model Structures
// =============================================================================

#[derive(Debug, Serialize)]
struct RandomForestModel {
    trees: Vec<DecisionTree>,
    n_classes: usize,
    label_to_idx: HashMap<String, usize>,
    idx_to_label: Vec<String>,
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
    #[serde(skip)]
    max_features: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct DecisionTree {
    nodes: Vec<TreeNode>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TreeNode {
    feature_idx: Option<usize>,
    threshold: f32,
    left_child: Option<usize>,
    right_child: Option<usize>,
    class_prediction: Option<usize>,
}

// =============================================================================
// Manifest Structures
// =============================================================================

#[derive(Debug, Deserialize)]
struct BeansManifest {
    #[allow(dead_code)]
    dataset: String,
    #[allow(dead_code)]
    n_samples: usize,
    samples: Vec<BeansSample>,
}

#[derive(Debug, Deserialize, Clone)]
struct BeansSample {
    audio_file: String,
    #[allow(dead_code)]
    n_samples: u32,
    labels: BeansLabels,
}

#[derive(Debug, Deserialize, Clone)]
struct BeansLabels {
    output: String,
    task: String,
}

#[derive(Debug, Deserialize)]
struct CacheManifest {
    entries: HashMap<String, String>,
    #[allow(dead_code)]
    feature_count: usize,
}

// =============================================================================
// Random Forest Implementation
// =============================================================================

impl RandomForestModel {
    fn new() -> Self {
        Self {
            trees: Vec::with_capacity(N_TREES),
            n_classes: 0,
            label_to_idx: HashMap::new(),
            idx_to_label: Vec::new(),
            feature_means: vec![0.0; FEATURE_DIM],
            feature_stds: vec![1.0; FEATURE_DIM],
            max_features: MAX_FEATURES,
        }
    }

    fn fit(&mut self, features: &[Vec<f32>], labels: &[String]) {
        // Build label mapping
        let mut unique_labels: Vec<String> = labels.iter().cloned().collect();
        unique_labels.sort();
        unique_labels.dedup();

        self.n_classes = unique_labels.len();
        self.idx_to_label = unique_labels.clone();
        for (idx, label) in unique_labels.iter().enumerate() {
            self.label_to_idx.insert(label.clone(), idx);
        }

        // Convert labels to indices
        let label_indices: Vec<usize> = labels
            .iter()
            .map(|l| *self.label_to_idx.get(l).unwrap_or(&0))
            .collect();

        // Compute class weights (sqrt-smoothed inverse frequency)
        let mut class_counts = vec![0usize; self.n_classes];
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
                    (total_samples / (self.n_classes as f32 * count as f32))
                        .sqrt()
                        .min(10.0)
                }
            })
            .collect();

        // Report stats
        let max_count = *class_counts.iter().max().unwrap_or(&1);
        let min_count = *class_counts.iter().filter(|&&c| c > 0).min().unwrap_or(&1);
        let imbalance_ratio = max_count as f32 / min_count.max(1) as f32;
        println!("  Classes: {}", self.n_classes);
        println!("  Class imbalance ratio: {:.1}:1", imbalance_ratio);

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
        for i in 0..FEATURE_DIM {
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

        // Train trees with bootstrap sampling
        println!("  Training {} trees (max_depth={}, max_features={})...",
            N_TREES, MAX_DEPTH, MAX_FEATURES);

        for tree_idx in 0..N_TREES {
            // Bootstrap sample
            let n_samples = normalized.len();
            let mut bootstrap_features = Vec::with_capacity(n_samples);
            let mut bootstrap_labels = Vec::with_capacity(n_samples);
            let mut bootstrap_weights = Vec::with_capacity(n_samples);

            for _ in 0..n_samples {
                let idx = (rand_u32() as usize) % n_samples;
                bootstrap_features.push(normalized[idx].clone());
                bootstrap_labels.push(label_indices[idx]);
                bootstrap_weights.push(class_weights[label_indices[idx]]);
            }

            // Train tree
            let tree = self.train_tree(
                &bootstrap_features,
                &bootstrap_labels,
                &bootstrap_weights,
                MAX_DEPTH,
            );
            self.trees.push(tree);

            if (tree_idx + 1) % 50 == 0 {
                println!("    Trained {}/{} trees", tree_idx + 1, N_TREES);
            }
        }
    }

    fn train_tree(
        &self,
        features: &[Vec<f32>],
        labels: &[usize],
        weights: &[f32],
        max_depth: usize,
    ) -> DecisionTree {
        let mut nodes = Vec::new();
        self.build_node(features, labels, weights, &mut nodes, max_depth, 0);
        DecisionTree { nodes }
    }

    fn build_node(
        &self,
        features: &[Vec<f32>],
        labels: &[usize],
        weights: &[f32],
        nodes: &mut Vec<TreeNode>,
        max_depth: usize,
        depth: usize,
    ) {
        // Count classes with weights
        let mut class_counts = Vec::new();
        for &label in labels {
            if class_counts.len() <= label {
                class_counts.resize(label + 1, 0.0);
            }
        }

        let mut weighted_counts = vec![0.0f32; class_counts.len()];
        for (&label, &w) in labels.iter().zip(weights.iter()) {
            weighted_counts[label] += w;
        }

        let n_classes_present = weighted_counts.iter().filter(|&&c| c > 0.0).count();

        // Leaf conditions (allow single-sample leaves with MIN_SAMPLES_LEAF)
        let total_weight: f32 = weights.iter().sum();
        if labels.len() <= MIN_SAMPLES_LEAF || depth >= max_depth || n_classes_present <= 1 {
            let majority_class = weighted_counts
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            nodes.push(TreeNode {
                feature_idx: None,
                threshold: 0.0,
                left_child: None,
                right_child: None,
                class_prediction: Some(majority_class),
            });
            return;
        }

        // Find best split using weighted Gini impurity
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gain = f32::NEG_INFINITY;

        // Select random subset of features (max_features)
        let mut feature_indices: Vec<usize> = (0..FEATURE_DIM).collect();
        shuffle_slice(&mut feature_indices);
        let selected_features: Vec<usize> = feature_indices
            .into_iter()
            .take(MAX_FEATURES)
            .collect();

        let parent_gini = self.weighted_gini(&weighted_counts, total_weight);

        for feature_idx in selected_features {
            let mut values: Vec<f32> = features.iter().map(|f| f[feature_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            // Sample thresholds
            let step = (values.len() / 20).max(1);
            for threshold_idx in (0..values.len()).step_by(step) {
                let threshold = values[threshold_idx];

                let mut left_counts = vec![0.0f32; weighted_counts.len()];
                let mut right_counts = weighted_counts.clone();
                let mut left_weight = 0.0f32;
                let mut right_weight = total_weight;

                for ((f, &l), &w) in features.iter().zip(labels.iter()).zip(weights.iter()) {
                    if f[feature_idx] <= threshold {
                        left_counts[l] += w;
                        left_weight += w;
                        right_counts[l] -= w;
                        right_weight -= w;
                    }
                }

                if left_weight < 0.001 || right_weight < 0.001 {
                    continue;
                }

                let left_gini = self.weighted_gini(&left_counts, left_weight);
                let right_gini = self.weighted_gini(&right_counts, right_weight);

                let weighted_gini =
                    (left_weight / total_weight) * left_gini
                    + (right_weight / total_weight) * right_gini;

                let gain = parent_gini - weighted_gini;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }

        // If no good split found, make leaf
        if best_gain <= 0.0 {
            let majority_class = weighted_counts
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            nodes.push(TreeNode {
                feature_idx: None,
                threshold: 0.0,
                left_child: None,
                right_child: None,
                class_prediction: Some(majority_class),
            });
            return;
        }

        // Split data
        let node_idx = nodes.len();
        nodes.push(TreeNode {
            feature_idx: Some(best_feature),
            threshold: best_threshold,
            left_child: None,
            right_child: None,
            class_prediction: None,
        });

        let mut left_features = Vec::new();
        let mut left_labels = Vec::new();
        let mut left_weights = Vec::new();
        let mut right_features = Vec::new();
        let mut right_labels = Vec::new();
        let mut right_weights = Vec::new();

        for ((f, &l), &w) in features.iter().zip(labels.iter()).zip(weights.iter()) {
            if f[best_feature] <= best_threshold {
                left_features.push(f.clone());
                left_labels.push(l);
                left_weights.push(w);
            } else {
                right_features.push(f.clone());
                right_labels.push(l);
                right_weights.push(w);
            }
        }

        // Recursively build children
        if !left_features.is_empty() {
            let left_idx = nodes.len();
            self.build_node(&left_features, &left_labels, &left_weights, nodes, max_depth, depth + 1);
            nodes[node_idx].left_child = Some(left_idx);
        }

        if !right_features.is_empty() {
            let right_idx = nodes.len();
            self.build_node(&right_features, &right_labels, &right_weights, nodes, max_depth, depth + 1);
            nodes[node_idx].right_child = Some(right_idx);
        }
    }

    fn weighted_gini(&self, counts: &[f32], total: f32) -> f32 {
        if total <= 0.0 {
            return 0.0;
        }
        let mut sum_sq = 0.0;
        for &c in counts {
            let p = c / total;
            sum_sq += p * p;
        }
        1.0 - sum_sq
    }

    fn predict(&self, features: &[f32]) -> (usize, String) {
        // Normalize
        let normalized: Vec<f32> = features
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - self.feature_means[i]) / self.feature_stds[i])
            .collect();

        // Vote from all trees
        let mut votes: HashMap<usize, usize> = HashMap::new();
        for tree in &self.trees {
            let pred = self.predict_tree(&normalized, tree);
            *votes.entry(pred).or_insert(0) += 1;
        }

        let (best_class, _) = votes
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .unwrap_or((0, 0));

        let label = self
            .idx_to_label
            .get(best_class)
            .cloned()
            .unwrap_or_else(|| format!("class_{}", best_class));

        (best_class, label)
    }

    fn predict_tree(&self, features: &[f32], tree: &DecisionTree) -> usize {
        let mut node_idx = 0;
        loop {
            let node = &tree.nodes[node_idx];
            if node.feature_idx.is_none() {
                return node.class_prediction.unwrap_or(0);
            }
            let feature_idx = node.feature_idx.unwrap();
            if features[feature_idx] <= node.threshold {
                node_idx = node.left_child.unwrap_or(0);
            } else {
                node_idx = node.right_child.unwrap_or(0);
            }
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn shuffle_slice<T>(slice: &mut [T]) {
    for i in 0..slice.len() {
        let j = (rand_u32() as usize) % slice.len();
        slice.swap(i, j);
    }
}

fn rand_u32() -> u32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x853c49e6748fea9b);

    let mut s = STATE.load(Ordering::Relaxed);
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    STATE.store(s, Ordering::Relaxed);
    (s.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Improved Random Forest Training (112D Cached Features)          ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!("  Trees:         {}", N_TREES);
    println!("  Max Depth:     {}", MAX_DEPTH);
    println!("  Min Leaf:      {}", MIN_SAMPLES_LEAF);
    println!("  Max Features:  {} (~{:.0}% of {}D)", MAX_FEATURES,
        (MAX_FEATURES as f32 / FEATURE_DIM as f32 * 100.0), FEATURE_DIM);
    println!();

    let start = Instant::now();

    // Load manifest
    let manifest_path = "beans_zero_full_manifest.json";
    println!("Loading manifest from: {}", manifest_path);
    let manifest_data = fs::read_to_string(manifest_path)?;
    let manifest: BeansManifest = serde_json::from_str(&manifest_data)?;
    println!("  Total samples: {}", manifest.samples.len());

    // Load cache manifest
    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_manifest_path = cache_dir.join("cache_manifest.json");
    println!("Loading cache manifest from: {:?}", cache_manifest_path);
    let cache_data = fs::read_to_string(&cache_manifest_path)?;
    let cache_manifest: CacheManifest = serde_json::from_str(&cache_data)?;
    println!("  Cached features available: {}", cache_manifest.entries.len());

    // Load all features and labels
    println!("\nLoading features from cache...");
    let mut all_features: Vec<Vec<f32>> = Vec::new();
    let mut all_labels: Vec<String> = Vec::new();
    let mut hits = 0;
    let mut misses = 0;

    for sample in &manifest.samples {
        let audio_file = &sample.audio_file;
        let label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };

        if let Some(cache_file) = cache_manifest.entries.get(audio_file) {
            let full_path = cache_dir.join(cache_file);
            if full_path.exists() {
                if let Ok(file) = fs::File::open(&full_path) {
                    let reader = BufReader::new(file);
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                        if features.len() == FEATURE_DIM {
                            all_features.push(features);
                            all_labels.push(label);
                            hits += 1;
                            continue;
                        }
                    }
                }
            }
        }
        misses += 1;
    }

    println!("  Loaded {} samples (hits: {}, misses: {})", all_features.len(), hits, misses);

    if all_features.is_empty() {
        anyhow::bail!("No features loaded!");
    }

    // Split into train/validation (90/10)
    println!("\nSplitting data: 90% train, 10% validation...");
    let n_train = (all_features.len() as f32 * 0.9) as usize;
    println!("  Train samples: {}", n_train);
    println!("  Val samples: {}", all_features.len() - n_train);

    // Shuffle indices
    let mut indices: Vec<usize> = (0..all_features.len()).collect();
    shuffle_slice(&mut indices);

    let train_indices: Vec<usize> = indices[..n_train].to_vec();
    let val_indices: Vec<usize> = indices[n_train..].to_vec();

    // Create training set
    let train_features: Vec<Vec<f32>> = train_indices.iter().map(|&i| all_features[i].clone()).collect();
    let train_labels: Vec<String> = train_indices.iter().map(|&i| all_labels[i].clone()).collect();

    // Train model
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Training Improved Random Forest                                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let train_start = Instant::now();
    let mut rf_model = RandomForestModel::new();
    rf_model.fit(&train_features, &train_labels);
    println!("  Training time: {:.1}s", train_start.elapsed().as_secs_f32());

    // Validate
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Validation                                                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Training accuracy
    let mut train_correct = 0;
    for (features, label) in train_features.iter().zip(train_labels.iter()) {
        let (_, pred) = rf_model.predict(features);
        if &pred == label {
            train_correct += 1;
        }
    }
    let train_accuracy = train_correct as f32 / train_features.len() as f32 * 100.0;
    println!("  Training Accuracy: {:.2}%", train_accuracy);

    // Validation accuracy
    let mut val_correct = 0;
    for &i in &val_indices {
        let features = &all_features[i];
        let label = &all_labels[i];
        let (_, pred) = rf_model.predict(features);
        if &pred == label {
            val_correct += 1;
        }
    }
    let val_accuracy = val_correct as f32 / val_indices.len() as f32 * 100.0;
    println!("  Validation Accuracy: {:.2}%", val_accuracy);

    // Save model
    println!("\nSaving model to: random_forest_model_112d_improved.json");
    rf_model.save(Path::new("random_forest_model_112d_improved.json"))?;

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Configuration:                                                   ║");
    println!("║    Trees:         {:<46}║", N_TREES);
    println!("║    Max Depth:     {:<46}║", MAX_DEPTH);
    println!("║    Min Leaf:      {:<46}║", MIN_SAMPLES_LEAF);
    println!("║    Max Features:  {:<46}║", MAX_FEATURES);
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Results:                                                         ║");
    println!("║    Training Accuracy:   {:>8.2}%                                ║", train_accuracy);
    println!("║    Validation Accuracy: {:>8.2}%                                ║", val_accuracy);
    println!("║    Total Time:          {:>8.1}s                                 ║", start.elapsed().as_secs_f32());
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    Ok(())
}
