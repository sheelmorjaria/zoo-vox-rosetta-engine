//! BEANS-Zero Model Training Pipeline (112D Full Features)
//! =======================================================
//!
//! Trains and serializes:
//! 1. Random Forest classifier (JSON)
//! 2. Rosetta-Net neural network (JSON format)
//!
//! Usage:
//!   cargo run --release --bin train_beans_models_112d -- /path/to/beans_audio_manifest.json
//!
//! Output:
//!   - random_forest_model_112d.json (Random Forest with 100 trees)
//!   - rosetta_net_model_112d.json (Rosetta-Net with trained weights)
//!
//! This version uses the full 112D RosettaFeatures for better discrimination.

use anyhow::{anyhow, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// Use the full 112D RosettaFeatures from the library
use technical_architecture::MicroDynamicsExtractor;

const FEATURE_DIM: usize = 112;

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
// Feature Cache (Bincode Serialization for faster loading)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheManifest {
    entries: HashMap<String, String>,
    feature_count: usize,
}

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
            fs::create_dir_all(cache_dir)
                .expect(&format!("Failed to create cache directory: {:?}", cache_dir));
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
        let cache_key = format!("{:x}", md5_hash(audio_file));
        let cache_file = format!("features_{}.bin", cache_key);

        let full_path = self.cache_dir.join(&cache_file);
        let file = fs::File::create(&full_path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, features)?;

        self.manifest.entries.insert(audio_file.to_string(), cache_file);
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

fn md5_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
    }
    hash
}

fn rand_u32() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u32
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
    fn new(_n_trees: usize, _max_depth: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_classes: 0,
            label_to_idx: HashMap::new(),
            idx_to_label: Vec::new(),
            feature_means: vec![0.0; FEATURE_DIM],
            feature_stds: vec![1.0; FEATURE_DIM],
        }
    }

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

        // Compute class weights for imbalanced data
        let mut class_counts = vec![0usize; self.n_classes];
        let label_indices: Vec<usize> = labels
            .iter()
            .map(|l| *self.label_to_idx.get(l).unwrap_or(&0))
            .collect();

        for &idx in &label_indices {
            class_counts[idx] += 1;
        }

        let total_samples = labels.len() as f32;
        let max_count = *class_counts.iter().max().unwrap_or(&1);
        let min_count = *class_counts.iter().filter(|&&c| c > 0).min().unwrap_or(&1);
        let imbalance_ratio = max_count as f32 / min_count.max(1) as f32;
        println!(
            "  Class imbalance ratio: {:.1}:1 (max:{}, min:{})",
            imbalance_ratio, max_count, min_count
        );

        // Compute normalization parameters
        let n = features.len() as f32;
        for f in features {
            for (i, &v) in f.iter().enumerate() {
                if i < FEATURE_DIM {
                    self.feature_means[i] += v / n;
                }
            }
        }
        for f in features {
            for (i, &v) in f.iter().enumerate() {
                if i < FEATURE_DIM {
                    self.feature_stds[i] += (v - self.feature_means[i]).powi(2) / n;
                }
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
                    .map(|(i, &v)| {
                        if i < FEATURE_DIM {
                            (v - self.feature_means[i]) / self.feature_stds[i]
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        // Train trees with bootstrapping
        println!("  Training {} decision trees...", n_trees);
        for tree_idx in 0..n_trees {
            let n_samples = normalized.len();
            let mut bootstrap_features = Vec::with_capacity(n_samples);
            let mut bootstrap_labels = Vec::with_capacity(n_samples);

            for _ in 0..n_samples {
                let idx = (rand_u32() as usize) % n_samples;
                bootstrap_features.push(normalized[idx].clone());
                bootstrap_labels.push(label_indices[idx]);
            }

            let tree = Self::train_tree(&bootstrap_features, &bootstrap_labels, max_depth, 0);
            self.trees.push(tree);

            if (tree_idx + 1) % 20 == 0 {
                println!("    Trained {}/{} trees", tree_idx + 1, n_trees);
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
        Self::build_node(features, labels, &mut nodes, max_depth, depth);
        DecisionTree { nodes }
    }

    fn build_node(
        features: &[Vec<f32>],
        labels: &[usize],
        nodes: &mut Vec<TreeNode>,
        max_depth: usize,
        depth_remaining: usize,
    ) {
        // Count classes
        let mut class_counts: Vec<usize> = Vec::new();
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
        if labels.len() < 2 || depth_remaining >= max_depth || n_classes_present <= 1 {
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
            return;
        }

        // Find best split using Gini impurity
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gain = -1.0;

        let parent_gini = Self::gini_impurity(&class_counts);

        for feature_idx in 0..FEATURE_DIM.min(features[0].len()) {
            let mut values: Vec<f32> = features.iter().map(|f| f[feature_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            // Sample thresholds for efficiency
            let step = (values.len() / 10).max(1);
            for threshold_idx in (0..values.len()).step_by(step) {
                let threshold = values[threshold_idx];

                let mut left_counts = vec![0usize; class_counts.len()];
                let mut right_counts = class_counts.clone();

                for (f, &l) in features.iter().zip(labels.iter()) {
                    if f[feature_idx] <= threshold {
                        left_counts[l] += 1;
                        right_counts[l] -= 1;
                    }
                }

                let left_total: usize = left_counts.iter().sum();
                let right_total: usize = right_counts.iter().sum();

                if left_total == 0 || right_total == 0 {
                    continue;
                }

                let left_gini = Self::gini_impurity(&left_counts);
                let right_gini = Self::gini_impurity(&right_counts);

                let n_total = left_total + right_total;
                let weighted_gini =
                    (left_total as f32 / n_total as f32) * left_gini
                    + (right_total as f32 / n_total as f32) * right_gini;

                let gain = parent_gini - weighted_gini;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }

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
            return;
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

        // Create split node
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
            return;
        }

        // Build children
        let left_child_idx = nodes.len();
        Self::build_node(&left_features, &left_labels, nodes, max_depth, depth_remaining + 1);

        let right_child_idx = nodes.len();
        Self::build_node(&right_features, &right_labels, nodes, max_depth, depth_remaining + 1);

        // Update children pointers
        nodes[current_idx].left_child = Some(left_child_idx);
        nodes[current_idx].right_child = Some(right_child_idx);
    }

    fn gini_impurity(counts: &[usize]) -> f32 {
        let total: usize = counts.iter().sum();
        if total == 0 {
            return 0.0;
        }

        let mut impurity = 1.0;
        for &count in counts {
            if count > 0 {
                let p = count as f32 / total as f32;
                impurity -= p * p;
            }
        }
        impurity
    }

    fn predict(&self, features: &[f32]) -> usize {
        if self.trees.is_empty() {
            return 0;
        }

        // Normalize
        let normalized: Vec<f32> = features
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if i < FEATURE_DIM {
                    (v - self.feature_means[i]) / self.feature_stds[i]
                } else {
                    0.0
                }
            })
            .collect();

        // Majority vote
        let mut votes = vec![0usize; self.n_classes];
        for tree in &self.trees {
            let pred = tree.predict(&normalized);
            if pred < votes.len() {
                votes[pred] += 1;
            }
        }

        votes
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn predict_label(&self, features: &[f32]) -> String {
        let idx = self.predict(features);
        self.idx_to_label
            .get(idx)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string())
    }

    fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(&self)?;
        fs::write(path, json)?;
        Ok(())
    }
}

impl DecisionTree {
    fn predict(&self, features: &[f32]) -> usize {
        let mut node_idx = 0;

        loop {
            let node = &self.nodes[node_idx];
            if let Some(class) = node.class_prediction {
                return class;
            }

            let feature_idx = node.feature_idx.unwrap();
            let threshold = node.threshold;

            if features[feature_idx] <= threshold {
                node_idx = node.left_child.unwrap();
            } else {
                node_idx = node.right_child.unwrap();
            }
        }
    }
}

// ============================================================================
// Rosetta-Net (Simple 2-layer neural network)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RosettaNetModel {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    weights_ih: Vec<f32>,
    weights_ho: Vec<f32>,
    bias_h: Vec<f32>,
    bias_o: Vec<f32>,
}

impl RosettaNetModel {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        // He initialization (better for ReLU networks)
        let scale_ih = (2.0 / input_dim as f32).sqrt();
        let scale_ho = (2.0 / hidden_dim as f32).sqrt();

        // Use better random initialization with larger variance
        let weights_ih: Vec<f32> = (0..input_dim * hidden_dim)
            .map(|_| {
                let r = (rand_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
                r * scale_ih
            })
            .collect();

        let weights_ho: Vec<f32> = (0..hidden_dim * output_dim)
            .map(|_| {
                let r = (rand_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
                r * scale_ho
            })
            .collect();

        Self {
            input_dim,
            hidden_dim,
            output_dim,
            weights_ih,
            weights_ho,
            bias_h: vec![0e-3; hidden_dim], // Small positive bias
            bias_o: vec![0.0; output_dim],
        }
    }

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Input -> Hidden (Leaky ReLU to prevent dying neurons)
        let mut hidden = vec![0.0; self.hidden_dim];
        let mut hidden_pre_relu = vec![0.0; self.hidden_dim];
        for h in 0..self.hidden_dim {
            for i in 0..self.input_dim.min(input.len()) {
                hidden_pre_relu[h] += input[i] * self.weights_ih[i * self.hidden_dim + h];
            }
            hidden_pre_relu[h] += self.bias_h[h];
            // Leaky ReLU: max(x, 0.01*x) - allows gradient flow for negative values
            hidden[h] = if hidden_pre_relu[h] > 0.0 {
                hidden_pre_relu[h]
            } else {
                0.01 * hidden_pre_relu[h]
            };
        }

        // Hidden -> Output
        let mut output = vec![0.0; self.output_dim];
        for o in 0..self.output_dim {
            for h in 0..self.hidden_dim {
                output[o] += hidden[h] * self.weights_ho[h * self.output_dim + o];
            }
            output[o] += self.bias_o[o];
        }

        // Stable softmax with temperature scaling
        let max_val = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let temperature = 1.0; // Can tune this
        let sum: f32 = output.iter().map(|o| ((o - max_val) / temperature).exp()).sum();
        for o in output.iter_mut() {
            *o = ((*o - max_val) / temperature).exp() / sum;
            // Clamp to prevent log(0)
            *o = o.clamp(1e-10, 1.0 - 1e-10);
        }

        (output, hidden, hidden_pre_relu)
    }

    fn predict(&self, features: &[f32]) -> usize {
        let (output, _, _) = self.forward(features);
        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn train(&mut self, features: &[Vec<f32>], labels: &[usize], epochs: usize, lr: f32) {
        println!("  Training for {} epochs (lr={})...", epochs, lr);

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut correct = 0;

            // Simple shuffle using time-based seed
            let mut indices: Vec<usize> = (0..features.len()).collect();
            let n = indices.len();
            for i in 0..n {
                let j = (rand_u32() as usize) % (n - i) + i;
                indices.swap(i, j);
            }

            for &i in &indices {
                let input = &features[i];
                let label = labels[i];

                // Forward pass
                let (output, hidden, hidden_pre_relu) = self.forward(input);

                // Cross-entropy loss (with clamped output to prevent log(0))
                if label < output.len() {
                    let prob = output[label].clamp(1e-10, 1.0);
                    total_loss -= prob.ln();

                    // Track accuracy during training
                    let predicted = output.iter().enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    if predicted == label {
                        correct += 1;
                    }
                }

                // Backward pass (gradient descent)
                // Gradient of softmax + cross-entropy: just (output - one_hot)
                let mut grad_output = output.clone();
                if label < grad_output.len() {
                    grad_output[label] -= 1.0;
                }

                // Gradient clipping to prevent explosion
                let grad_norm: f32 = grad_output.iter().map(|g| g * g).sum::<f32>().sqrt().max(1e-6);
                for g in grad_output.iter_mut() {
                    *g = *g / grad_norm;
                }

                // Update output layer
                for o in 0..self.output_dim {
                    for h in 0..self.hidden_dim {
                        self.weights_ho[h * self.output_dim + o] -= lr * grad_output[o] * hidden[h];
                    }
                    self.bias_o[o] -= lr * grad_output[o];
                }

                // Backprop to hidden layer
                let mut grad_hidden = vec![0.0; self.hidden_dim];
                for h in 0..self.hidden_dim {
                    for o in 0..self.output_dim {
                        grad_hidden[h] += grad_output[o] * self.weights_ho[h * self.output_dim + o];
                    }
                    // Leaky ReLU derivative: 1.0 if x > 0, else 0.01
                    if hidden_pre_relu[h] <= 0.0 {
                        grad_hidden[h] *= 0.01;
                    }
                }

                // Update input layer
                for h in 0..self.hidden_dim {
                    for i in 0..self.input_dim.min(input.len()) {
                        self.weights_ih[i * self.hidden_dim + h] -= lr * grad_hidden[h] * input[i];
                    }
                    self.bias_h[h] -= lr * grad_hidden[h];
                }
            }

            if (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
                let avg_loss = total_loss / features.len() as f32;
                let accuracy = correct as f32 / features.len() as f32 * 100.0;
                println!("    Epoch {}/{}: Loss = {:.4}, Accuracy = {:.1}%", epoch + 1, epochs, avg_loss, accuracy);
            }
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: train_beans_models_112d <manifest_path>");
        eprintln!("\nExpected BEANS-Zero manifest format:");
        eprintln!("  {{\"dataset\": \"...\", \"n_samples\": N, \"samples\": [{{\"audio_file\": \"...\", \"labels\": {{\"output\": \"...\"}}}}]}}");
        std::process::exit(1);
    }

    let manifest_path = Path::new(&args[1]);
    if !manifest_path.exists() {
        eprintln!("Error: Manifest file not found: {}", manifest_path.display());
        std::process::exit(1);
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║     BEANS-Zero Model Training (112D RosettaFeatures)             ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Load manifest
    println!("Loading manifest from: {}", manifest_path.display());
    let manifest: BeansManifest = serde_json::from_reader(BufReader::new(fs::File::open(manifest_path)?))?;

    println!("Dataset: {}", manifest.dataset);
    println!("Total samples: {}", manifest.n_samples);
    println!();

    // Initialize feature extractor
    let extractor = MicroDynamicsExtractor::new(44100);
    let mut cache = FeatureCache::new(Path::new("beans_feature_cache_112d"));

    // =========================================================================
    // PHASE 1: Feature Extraction (112D)
    // =========================================================================
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  [Phase 1] Extracting 112D RosettaFeatures                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let mut all_features: Vec<Vec<f32>> = Vec::new();
    let mut all_labels: Vec<String> = Vec::new();

    let samples: Vec<&BeansSample> = manifest.samples.iter().collect();
    println!("Extracting features from {} samples...", samples.len());
    println!("Using full 112D RosettaFeatures");
    println!();

    let start = Instant::now();

    // Process samples in parallel batches
    let batch_size = 100;
    for (batch_idx, batch) in samples.chunks(batch_size).enumerate() {
        print!("\r  Processing batch {}/{} ({} samples)   ",
            batch_idx + 1,
            (samples.len() + batch_size - 1) / batch_size,
            batch.len());

        let batch_results: Vec<(String, Option<Vec<f32>>, String)> = batch
            .par_iter()
            .map(|sample| {
                let audio_path = Path::new(&sample.audio_file);

                // Check cache first
                if let Some(cached) = cache.get(&sample.audio_file) {
                    let label = sample.labels.output.clone().unwrap_or_else(|| "unknown".to_string());
                    return (sample.audio_file.clone(), Some(cached), label);
                }

                // Load audio using hound
                let audio = match load_wav_audio(audio_path) {
                    Ok(a) => a,
                    Err(_) => {
                        return (sample.audio_file.clone(), None, "unknown".to_string());
                    }
                };

                // Extract 112D features
                let features = match extractor.extract_rosetta(&audio) {
                    Ok(f) => f,
                    Err(_) => {
                        return (sample.audio_file.clone(), None, "unknown".to_string());
                    }
                };

                let feature_vec = features.to_array().to_vec();
                let label = sample.labels.output.clone().unwrap_or_else(|| "unknown".to_string());

                (sample.audio_file.clone(), Some(feature_vec), label)
            })
            .collect();

        for (audio_file, features_opt, label) in batch_results {
            if let Some(features) = features_opt {
                // Update cache
                let _ = cache.put(&audio_file, &features);

                all_features.push(features);
                all_labels.push(label);
            }
        }
    }

    println!();
    println!("Extraction completed in {:.1}s", start.elapsed().as_secs_f32());

    if all_features.is_empty() {
        eprintln!("Error: No features extracted!");
        std::process::exit(1);
    }

    println!("Extracted features from {} samples", all_features.len());

    let (hits, misses) = cache.stats();
    println!("Cache stats: {} hits, {} misses", hits, misses);

    cache.save_manifest()?;

    // =========================================================================
    // PHASE 2: Train Random Forest
    // =========================================================================
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  [Phase 2] Training Random Forest (112D features)                ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let start = Instant::now();
    let mut rf_model = RandomForestModel::new(100, 10);
    rf_model.fit(&all_features, &all_labels, 100, 10);

    // Validate Random Forest
    println!();
    println!("Validating Random Forest...");
    let mut correct = 0;
    for (features, label) in all_features.iter().zip(all_labels.iter()) {
        let pred = rf_model.predict_label(features);
        if &pred == label {
            correct += 1;
        }
    }
    let rf_accuracy = correct as f32 / all_features.len() as f32 * 100.0;
    println!("Random Forest Accuracy: {:.2}%", rf_accuracy);
    println!("Training time: {:.1}s", start.elapsed().as_secs_f32());

    // Save Random Forest
    rf_model.save(&Path::new("random_forest_model_112d.json"))?;
    println!("Saved Random Forest to random_forest_model_112d.json");

    // =========================================================================
    // PHASE 3: Train Rosetta-Net
    // =========================================================================
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  [Phase 3] Training Rosetta-Net (112D features)                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let start = Instant::now();
    let n_classes = rf_model.n_classes;
    let label_to_idx = rf_model.label_to_idx.clone();
    let idx_to_label = rf_model.idx_to_label.clone();

    // Normalize features using Random Forest normalization parameters
    let normalized_features: Vec<Vec<f32>> = all_features.iter().map(|f| {
        f.iter().enumerate().map(|(i, &v)| {
            (v - rf_model.feature_means[i]) / rf_model.feature_stds[i]
        }).collect()
    }).collect();

    let mut rosetta_net = RosettaNetModel::new(FEATURE_DIM, 256, n_classes);

    let label_indices: Vec<usize> = all_labels.iter()
        .map(|l| *label_to_idx.get(l).unwrap_or(&0))
        .collect();

    // Use smaller learning rate and more epochs with normalized features
    rosetta_net.train(&normalized_features, &label_indices, 200, 0.01);

    // Validate Rosetta-Net
    println!();
    println!("Validating Rosetta-Net...");
    let mut correct = 0;
    for (features, label) in normalized_features.iter().zip(all_labels.iter()) {
        let pred_idx = rosetta_net.predict(features);
        let pred_label = idx_to_label.get(pred_idx).cloned().unwrap_or_else(|| "unknown".to_string());
        if &pred_label == label {
            correct += 1;
        }
    }
    let nn_accuracy = correct as f32 / all_features.len() as f32 * 100.0;
    println!("Rosetta-Net Accuracy: {:.2}%", nn_accuracy);
    println!("Training time: {:.1}s", start.elapsed().as_secs_f32());

    // Save Rosetta-Net
    rosetta_net.save(&Path::new("rosetta_net_model_112d.json"))?;
    println!("Saved Rosetta-Net to rosetta_net_model_112d.json");

    // =========================================================================
    // Summary
    // =========================================================================
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  TRAINING COMPLETE                                                ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Output files:");
    println!("  - random_forest_model_112d.json ({} trees)", rf_model.trees.len());
    println!("  - rosetta_net_model_112d.json ({}D -> 256 -> {} classes)", FEATURE_DIM, n_classes);
    println!();
    println!("Feature Dimension: {}D (RosettaFeatures)", FEATURE_DIM);
    println!("Random Forest Accuracy: {:.2}%", rf_accuracy);
    println!("Rosetta-Net Accuracy: {:.2}%", nn_accuracy);
    println!();

    Ok(())
}

/// Load WAV audio file using hound crate
fn load_wav_audio(path: &Path) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        }
        hound::SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader.into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Convert to mono if stereo
    let mono_samples: Vec<f32> = if spec.channels == 2 {
        samples.chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    // Resample to 44.1kHz if needed
    let target_rate = 44100u32;
    let resampled = if spec.sample_rate != target_rate {
        let ratio = target_rate as f64 / spec.sample_rate as f64;
        let new_len = (mono_samples.len() as f64 * ratio) as usize;
        (0..new_len)
            .map(|i| {
                let src_idx = (i as f64 / ratio) as usize;
                mono_samples.get(src_idx).copied().unwrap_or(0.0)
            })
            .collect()
    } else {
        mono_samples
    };

    Ok(resampled)
}
