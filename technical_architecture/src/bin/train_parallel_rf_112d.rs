//! Parallel Random Forest Training (112D Features)
//! =================================================
//!
//! Uses Rayon for parallel tree building - major speedup!
//! Trees are independent and can be built in parallel.
//!
//! Usage:
//!   cargo run --release --bin train_parallel_rf_112d

use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const FEATURE_DIM: usize = 112;

// Optimized hyperparameters
const N_TREES: usize = 300;            // 200 trees (parallel building makes this fast)
const MAX_DEPTH: usize = 30;           // Depth 30 (good balance)
const MIN_SAMPLES_LEAF: usize = 1;     // Allow single-sample leaves
const MAX_FEATURES: usize = 20;        // ~18% of 112D

// =============================================================================
// Model Structures
// =============================================================================

#[derive(Debug, Serialize, Deserialize, Clone)]
struct DecisionTree {
    nodes: Vec<TreeNode>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct TreeNode {
    feature_idx: Option<usize>,
    threshold: f32,
    left_child: Option<usize>,
    right_child: Option<usize>,
    class_prediction: Option<usize>,
}

#[derive(Debug, Serialize)]
struct RandomForestModel {
    trees: Vec<DecisionTree>,
    n_classes: usize,
    idx_to_label: Vec<String>,
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
}

// =============================================================================
// Data Structures
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
// Training Data
// =============================================================================

struct TrainingData {
    features: Vec<Vec<f32>>,
    labels: Vec<usize>,
    n_classes: usize,
    class_weights: Vec<f32>,
}

// =============================================================================
// Tree Training (for parallel execution)
// =============================================================================

fn train_single_tree(
    data: &TrainingData,
    bootstrap_indices: Vec<usize>,
    tree_id: usize,
) -> DecisionTree {
    let normalized: Vec<Vec<f32>> = bootstrap_indices
        .iter()
        .map(|&i| {
            data.features[i]
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    // Features already normalized in TrainingData
                    v
                })
                .collect()
        })
        .collect();

    let labels: Vec<usize> = bootstrap_indices.iter().map(|&i| data.labels[i]).collect();
    let weights: Vec<f32> = bootstrap_indices.iter().map(|&i| data.class_weights[data.labels[i]]).collect();

    let mut nodes = Vec::new();
    build_node(&normalized, &labels, &weights, &mut nodes, MAX_DEPTH, 0);

    DecisionTree { nodes }
}

fn build_node(
    features: &[Vec<f32>],
    labels: &[usize],
    weights: &[f32],
    nodes: &mut Vec<TreeNode>,
    max_depth: usize,
    depth: usize,
) {
    // Count classes with weights
    let mut class_counts = HashMap::new();
    let mut total_weight = 0.0f32;
    for (&label, &weight) in labels.iter().zip(weights.iter()) {
        *class_counts.entry(label).or_insert(0.0f32) += weight;
        total_weight += weight;
    }

    let n_classes_present = class_counts.len();

    // Leaf conditions
    if labels.len() <= MIN_SAMPLES_LEAF || depth >= max_depth || n_classes_present <= 1 {
        let majority_class = class_counts
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(c, _)| c)
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
    let parent_gini = gini_impurity(&class_counts, total_weight);

    let mut best_feature = 0;
    let mut best_threshold = 0.0f32;
    let mut best_gain = f32::NEG_INFINITY;

    // Select random subset of features
    let mut feature_indices: Vec<usize> = (0..FEATURE_DIM).collect();
    shuffle_slice(&mut feature_indices);
    let selected_features: Vec<usize> = feature_indices.into_iter().take(MAX_FEATURES).collect();

    for feature_idx in selected_features {
        let mut values: Vec<f32> = features.iter().map(|f| f[feature_idx]).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();

        // Sample thresholds
        let step = (values.len() / 16).max(1);
        for threshold_idx in (0..values.len()).step_by(step) {
            let threshold = values[threshold_idx];

            let mut left_counts = HashMap::new();
            let mut right_counts = class_counts.clone();
            let mut left_weight = 0.0f32;
            let mut right_weight = total_weight;

            for ((f, &label), &weight) in features.iter().zip(labels.iter()).zip(weights.iter()) {
                if f[feature_idx] <= threshold {
                    *left_counts.entry(label).or_insert(0.0f32) += weight;
                    *right_counts.get_mut(&label).unwrap() -= weight;
                    left_weight += weight;
                    right_weight -= weight;
                }
            }

            if left_weight < 0.001 || right_weight < 0.001 {
                continue;
            }

            let left_gini = gini_impurity(&left_counts, left_weight);
            let right_gini = gini_impurity(&right_counts, right_weight);

            let weighted_gini = (left_weight / total_weight) * left_gini
                + (right_weight / total_weight) * right_gini;

            let gain = parent_gini - weighted_gini;

            if gain > best_gain {
                best_gain = gain;
                best_feature = feature_idx;
                best_threshold = threshold;
            }
        }
    }

    // If no good split, make leaf
    if best_gain <= 0.0 {
        let majority_class = class_counts
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(c, _)| c)
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

    // Create split node
    let node_idx = nodes.len();
    nodes.push(TreeNode {
        feature_idx: Some(best_feature),
        threshold: best_threshold,
        left_child: None,
        right_child: None,
        class_prediction: None,
    });

    // Split data
    let mut left_features = Vec::new();
    let mut left_labels = Vec::new();
    let mut left_weights = Vec::new();
    let mut right_features = Vec::new();
    let mut right_labels = Vec::new();
    let mut right_weights = Vec::new();

    for ((f, &label), &weight) in features.iter().zip(labels.iter()).zip(weights.iter()) {
        if f[best_feature] <= best_threshold {
            left_features.push(f.clone());
            left_labels.push(label);
            left_weights.push(weight);
        } else {
            right_features.push(f.clone());
            right_labels.push(label);
            right_weights.push(weight);
        }
    }

    // Recursively build children
    if !left_features.is_empty() {
        let left_idx = nodes.len();
        build_node(&left_features, &left_labels, &left_weights, nodes, max_depth, depth + 1);
        nodes[node_idx].left_child = Some(left_idx);
    }

    if !right_features.is_empty() {
        let right_idx = nodes.len();
        build_node(&right_features, &right_labels, &right_weights, nodes, max_depth, depth + 1);
        nodes[node_idx].right_child = Some(right_idx);
    }
}

fn gini_impurity(class_counts: &HashMap<usize, f32>, total: f32) -> f32 {
    if total <= 0.0 {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for &count in class_counts.values() {
        let p = count / total;
        sum_sq += p * p;
    }
    1.0 - sum_sq
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

fn rand_f32() -> f32 {
    (rand_u32() as f64 / u32::MAX as f64) as f32
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Parallel Random Forest Training (112D Cached Features)         ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!("  Trees:         {} (parallel building)", N_TREES);
    println!("  Max Depth:     {}", MAX_DEPTH);
    println!("  Min Leaf:      {}", MIN_SAMPLES_LEAF);
    println!("  Max Features:  {} (~{:.0}% of {}D)", MAX_FEATURES,
        (MAX_FEATURES as f32 / FEATURE_DIM as f32 * 100.0), FEATURE_DIM);
    println!("  Parallelism:   {} threads", rayon::current_num_threads());
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
                            continue;
                        }
                    }
                }
            }
        }
    }

    println!("  Loaded {} samples", all_features.len());

    if all_features.is_empty() {
        anyhow::bail!("No features loaded!");
    }

    // Build label mapping
    let mut unique_labels: Vec<String> = all_labels.iter().cloned().collect();
    unique_labels.sort();
    unique_labels.dedup();
    let n_classes = unique_labels.len();
    let mut label_to_idx = HashMap::new();
    for (idx, label) in unique_labels.iter().enumerate() {
        label_to_idx.insert(label.clone(), idx);
    }
    println!("  Classes: {}", n_classes);

    // Convert labels to indices
    let label_indices: Vec<usize> = all_labels
        .iter()
        .map(|l| *label_to_idx.get(l).unwrap_or(&0))
        .collect();

    // Compute class weights (class_weight='balanced' - sklearn formula)
    // weight = n_samples / (n_classes * n_samples_for_class)
    let mut class_counts = vec![0usize; n_classes];
    for &idx in &label_indices {
        class_counts[idx] += 1;
    }

    let total_samples = all_labels.len() as f32;
    let class_weights: Vec<f32> = class_counts
        .iter()
        .map(|&count| {
            if count == 0 {
                1.0
            } else {
                // True balanced: n_samples / (n_classes * count)
                // Cap at 100x to prevent extreme weights for tiny classes
                (total_samples / (n_classes as f32 * count as f32)).min(100.0)
            }
        })
        .collect();

    // Print class weight distribution
    let min_weight = class_weights.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_weight = class_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let median_weight = {
        let mut sorted = class_weights.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[n_classes / 2]
    };
    println!("  Class weights: min={:.2}, median={:.2}, max={:.2}",
        min_weight, median_weight, max_weight);

    // Split into train/validation (90/10)
    println!("\nSplitting data: 90% train, 10% validation...");
    let n_train = (all_features.len() as f32 * 0.9) as usize;
    println!("  Train samples: {}", n_train);

    // Shuffle indices
    let mut indices: Vec<usize> = (0..all_features.len()).collect();
    shuffle_slice(&mut indices);

    let train_indices: Vec<usize> = indices[..n_train].to_vec();
    let val_indices: Vec<usize> = indices[n_train..].to_vec();

    // Compute normalization params
    let mut feature_means = vec![0.0f32; FEATURE_DIM];
    let mut feature_stds = vec![0.0f32; FEATURE_DIM];

    for &i in &train_indices {
        for (j, &v) in all_features[i].iter().enumerate() {
            feature_means[j] += v;
        }
    }
    for j in 0..FEATURE_DIM {
        feature_means[j] /= train_indices.len() as f32;
    }

    for &i in &train_indices {
        for (j, &v) in all_features[i].iter().enumerate() {
            let diff = v - feature_means[j];
            feature_stds[j] += diff * diff;
        }
    }
    for j in 0..FEATURE_DIM {
        feature_stds[j] = (feature_stds[j] / train_indices.len() as f32).sqrt().max(1e-8);
    }

    // Normalize all features
    let normalized_features: Vec<Vec<f32>> = all_features
        .iter()
        .map(|f| {
            f.iter()
                .enumerate()
                .map(|(j, &v)| (v - feature_means[j]) / feature_stds[j])
                .collect()
        })
        .collect();

    // Create training data struct
    let train_data = TrainingData {
        features: train_indices.iter().map(|&i| normalized_features[i].clone()).collect(),
        labels: train_indices.iter().map(|&i| label_indices[i]).collect(),
        n_classes,
        class_weights: class_weights.clone(),
    };

    // Generate bootstrap samples (parallel-friendly)
    println!("\nGenerating bootstrap samples...");
    let bootstrap_samples: Vec<Vec<usize>> = (0..N_TREES)
        .map(|_| {
            let n = train_data.features.len();
            (0..n).map(|_| (rand_u32() as usize) % n).collect()
        })
        .collect();

    // PARALLEL TREE BUILDING!
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Training {} Trees in Parallel                                    ║", N_TREES);
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let trained_count = AtomicUsize::new(0);
    let train_start = Instant::now();

    let trees: Vec<DecisionTree> = bootstrap_samples
        .into_par_iter()
        .enumerate()
        .map(|(tree_id, bootstrap)| {
            let tree = train_single_tree(&train_data, bootstrap, tree_id);
            let count = trained_count.fetch_add(1, Ordering::SeqCst) + 1;
            if count % 20 == 0 {
                let elapsed = train_start.elapsed().as_secs_f32();
                let rate = count as f32 / elapsed;
                let remaining = (N_TREES - count) as f32 / rate;
                println!("  Trained {}/{} trees ({:.1}s elapsed, ~{:.0}s remaining)",
                    count, N_TREES, elapsed, remaining);
            }
            tree
        })
        .collect();

    println!("\n  All {} trees trained in {:.1}s", N_TREES, train_start.elapsed().as_secs_f32());

    // Create model
    let model = RandomForestModel {
        trees,
        n_classes,
        idx_to_label: unique_labels,
        feature_means,
        feature_stds,
    };

    // Validation
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Validation                                                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let mut val_correct = 0usize;
    for &i in &val_indices {
        let normalized = &normalized_features[i];
        let true_label = label_indices[i];

        // Vote from all trees
        let mut votes = vec![0usize; n_classes];
        for tree in &model.trees {
            let pred = predict_tree(normalized, tree);
            votes[pred] += 1;
        }

        let pred_class = votes
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap_or(0);

        if pred_class == true_label {
            val_correct += 1;
        }
    }
    let val_accuracy = val_correct as f32 / val_indices.len() as f32 * 100.0;
    println!("  Validation Accuracy: {:.2}%", val_accuracy);

    // Save model
    println!("\nSaving model to: random_forest_model_112d_parallel.json");
    let json = serde_json::to_string_pretty(&model)?;
    fs::write("random_forest_model_112d_parallel.json", json)?;

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Configuration:                                                   ║");
    println!("║    Trees:         {:<46}║", N_TREES);
    println!("║    Max Depth:     {:<46}║", MAX_DEPTH);
    println!("║    Parallelism:   {:<46}║", format!("{} threads", rayon::current_num_threads()));
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Results:                                                         ║");
    println!("║    Validation Accuracy: {:>8.2}%                                ║", val_accuracy);
    println!("║    Total Time:          {:>8.1}s                                 ║", start.elapsed().as_secs_f32());
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    Ok(())
}

fn predict_tree(features: &[f32], tree: &DecisionTree) -> usize {
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
