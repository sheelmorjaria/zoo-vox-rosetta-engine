//! Train Gatekeeper RF (76D) with Full Tree Export
//! ================================================
//!
//! Trains a Random Forest classifier on 76D physics features to predict
//! consolidated taxonomic groups (Bird, Mammal, MarineMammal, Insect, Amphibian, Unknown).
//!
//! Uses rayon for parallel tree training.
//!
//! Usage:
//!   cargo run --release --bin train_gatekeeper_rf

use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
// Reserved for future progress tracking: use std::sync::atomic::{AtomicUsize, Ordering};

use technical_architecture::taxonomic_router::{
    FEATURE_DIM, GATEKEEPER_DIM, ConsolidatedTaxon,
    consolidated_taxon_to_idx, idx_to_consolidated_taxon, consolidated_taxon_labels,
};

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Debug, Deserialize)]
struct BeansManifest { samples: Vec<BeansSample> }

#[derive(Debug, Deserialize, Clone)]
struct BeansSample {
    audio_file: String,
    labels: BeansLabels,
}

#[derive(Debug, Deserialize, Clone)]
struct BeansLabels {
    output: String,
    task: String,
}

#[derive(Debug, Deserialize)]
struct CacheManifest { entries: HashMap<String, String> }

// =============================================================================
// Species to Taxonomic Group Mapping
// =============================================================================

fn species_to_taxon(species: &str, task: &str) -> ConsolidatedTaxon {
    let species_lower = species.to_lowercase();
    let task_lower = task.to_lowercase();

    // Bird species
    if species_lower.contains("sparrow") || species_lower.contains("finch")
        || species_lower.contains("warbler") || species_lower.contains("bird")
        || task_lower.contains("bird") || task_lower.contains("zf-indv")
        || task_lower.contains("cbi") {
        return ConsolidatedTaxon::Bird;
    }

    // Marine mammals (dolphins, whales)
    if species_lower.contains("dolphin") || species_lower.contains("whale")
        || species_lower.contains("cetacean") || species_lower.contains("watkins")
        || task_lower.contains("dolphin") || task_lower.contains("whale") {
        return ConsolidatedTaxon::MarineMammal;
    }

    // Insects (mosquitoes, bees, crickets)
    if species_lower.contains("mosquito") || species_lower.contains("bee")
        || species_lower.contains("cricket") || species_lower.contains("insect")
        || species_lower.contains("humbugdb") || species_lower.contains("cicada")
        || task_lower.contains("insect") || task_lower.contains("humbugdb") {
        return ConsolidatedTaxon::Insect;
    }

    // Amphibians (frogs, toads)
    if species_lower.contains("frog") || species_lower.contains("toad")
        || species_lower.contains("amphibian") || task_lower.contains("frog")
        || task_lower.contains("amphibian") {
        return ConsolidatedTaxon::Amphibian;
    }

    // Mammals (bats, primates)
    if species_lower.contains("bat") || species_lower.contains("marmoset")
        || species_lower.contains("primate") || species_lower.contains("monkey")
        || task_lower.contains("bat") || task_lower.contains("mammal") {
        return ConsolidatedTaxon::Mammal;
    }

    // ESC50 has mixed categories
    if task_lower.contains("esc50") {
        // Try to infer from species name
        if species_lower.contains("dog") || species_lower.contains("cat")
            || species_lower.contains("cow") || species_lower.contains("pig") {
            return ConsolidatedTaxon::Mammal;
        }
        if species_lower.contains("frog") || species_lower.contains("toad") {
            return ConsolidatedTaxon::Amphibian;
        }
        if species_lower.contains("bird") || species_lower.contains("rooster")
            || species_lower.contains("chicken") || species_lower.contains("crow") {
            return ConsolidatedTaxon::Bird;
        }
        if species_lower.contains("cricket") || species_lower.contains("fly")
            || species_lower.contains("mosquito") {
            return ConsolidatedTaxon::Insect;
        }
    }

    // Default to Unknown
    ConsolidatedTaxon::Unknown
}

// =============================================================================
// Gatekeeper RF Structure (with tree export)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TreeNode {
    feature_idx: Option<usize>,
    threshold: f32,
    left_child: Option<usize>,
    right_child: Option<usize>,
    class_prediction: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecisionTree {
    nodes: Vec<TreeNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GatekeeperRFModel {
    trees: Vec<DecisionTree>,
    n_classes: usize,
    idx_to_label: Vec<String>,
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
    train_accuracy: f64,
    val_accuracy: f64,
}

// =============================================================================
// Simple RF Implementation (training)
// =============================================================================

struct SimpleRF {
    trees: Vec<DecisionTree>,
    n_classes: usize,
}

impl SimpleRF {
    fn new(n_classes: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_classes,
        }
    }

    fn train(&mut self, features: &[Vec<f32>], labels: &[usize], n_trees: usize, max_depth: usize) {
        let n_samples = features.len();
        let n_features = features[0].len();
        let mtry = n_features / 3;

        // Parallel tree training with rayon
        let trees: Vec<DecisionTree> = (0..n_trees)
            .into_par_iter()
            .map(|tree_idx| {
                // Each tree gets its own RNG with unique seed
                let mut rng = SimpleRng::seed(42 + tree_idx as u64);

                // Bootstrap sample
                let sample_indices: Vec<usize> = (0..n_samples)
                    .map(|_| rng.next_usize(n_samples))
                    .collect();

                // Train single tree
                self.train_tree(
                    features,
                    labels,
                    &sample_indices,
                    max_depth,
                    mtry,
                    &mut rng,
                )
            })
            .collect();

        self.trees = trees;
    }

    fn train_tree(
        &self,
        features: &[Vec<f32>],
        labels: &[usize],
        sample_indices: &[usize],
        max_depth: usize,
        mtry: usize,
        rng: &mut SimpleRng,
    ) -> DecisionTree {
        let mut nodes: Vec<TreeNode> = Vec::new();

        // Build tree recursively
        self.build_node(
            features,
            labels,
            sample_indices,
            0,
            max_depth,
            mtry,
            rng,
            &mut nodes,
        );

        DecisionTree { nodes }
    }

    fn build_node(
        &self,
        features: &[Vec<f32>],
        labels: &[usize],
        sample_indices: &[usize],
        depth: usize,
        max_depth: usize,
        mtry: usize,
        rng: &mut SimpleRng,
        nodes: &mut Vec<TreeNode>,
    ) -> usize {
        let node_idx = nodes.len();
        nodes.push(TreeNode {
            feature_idx: None,
            threshold: 0.0,
            left_child: None,
            right_child: None,
            class_prediction: None,
        });

        // Check stopping conditions
        if sample_indices.is_empty() || depth >= max_depth {
            // Leaf node - majority vote
            let prediction = self.majority_vote(labels, sample_indices);
            nodes[node_idx].class_prediction = Some(prediction);
            return node_idx;
        }

        // Check purity
        let unique_labels: std::collections::HashSet<usize> = sample_indices
            .iter()
            .map(|&i| labels[i])
            .collect();

        if unique_labels.len() == 1 {
            nodes[node_idx].class_prediction = Some(*unique_labels.iter().next().unwrap());
            return node_idx;
        }

        // Find best split
        let (best_feat, best_thresh, best_gain) = self.find_best_split(
            features,
            labels,
            sample_indices,
            mtry,
            rng,
        );

        if best_gain <= 0.0 {
            nodes[node_idx].class_prediction = Some(self.majority_vote(labels, sample_indices));
            return node_idx;
        }

        // Split
        let mut left_indices: Vec<usize> = Vec::new();
        let mut right_indices: Vec<usize> = Vec::new();

        for &i in sample_indices {
            if features[i][best_feat] <= best_thresh {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        nodes[node_idx].feature_idx = Some(best_feat);
        nodes[node_idx].threshold = best_thresh;

        // Recurse
        let left_child = self.build_node(
            features, labels, &left_indices, depth + 1, max_depth, mtry, rng, nodes,
        );
        let right_child = self.build_node(
            features, labels, &right_indices, depth + 1, max_depth, mtry, rng, nodes,
        );

        nodes[node_idx].left_child = Some(left_child);
        nodes[node_idx].right_child = Some(right_child);

        node_idx
    }

    fn find_best_split(
        &self,
        features: &[Vec<f32>],
        labels: &[usize],
        sample_indices: &[usize],
        mtry: usize,
        rng: &mut SimpleRng,
    ) -> (usize, f32, f64) {
        let n_features = features[0].len();
        let n_try = mtry.min(n_features);

        let mut best_feat = 0;
        let mut best_thresh = 0.0;
        let mut best_gain = f64::NEG_INFINITY;

        let parent_gini = self.gini_impurity(labels, sample_indices);

        // Try random subset of features
        for _ in 0..n_try {
            let feat_idx = rng.next_usize(n_features);

            // Get all values for this feature
            let mut values: Vec<f32> = sample_indices.iter()
                .map(|&i| features[i][feat_idx])
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Try midpoints as thresholds
            for i in 0..values.len().saturating_sub(1) {
                let thresh = (values[i] + values[i + 1]) / 2.0;

                // Split
                let mut left_indices: Vec<usize> = Vec::new();
                let mut right_indices: Vec<usize> = Vec::new();

                for &j in sample_indices {
                    if features[j][feat_idx] <= thresh {
                        left_indices.push(j);
                    } else {
                        right_indices.push(j);
                    }
                }

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                // Calculate gain
                let left_gini = self.gini_impurity(labels, &left_indices);
                let right_gini = self.gini_impurity(labels, &right_indices);

                let n = sample_indices.len() as f64;
                let n_left = left_indices.len() as f64;
                let n_right = right_indices.len() as f64;

                let gain = parent_gini - (n_left / n * left_gini + n_right / n * right_gini);

                if gain > best_gain {
                    best_gain = gain;
                    best_feat = feat_idx;
                    best_thresh = thresh;
                }
            }
        }

        (best_feat, best_thresh, best_gain)
    }

    fn gini_impurity(&self, labels: &[usize], sample_indices: &[usize]) -> f64 {
        if sample_indices.is_empty() {
            return 0.0;
        }

        let mut class_counts = vec![0usize; self.n_classes];
        for &i in sample_indices {
            class_counts[labels[i]] += 1;
        }

        let n = sample_indices.len() as f64;
        let mut gini = 1.0;

        for count in class_counts {
            if count > 0 {
                let p = count as f64 / n;
                gini -= p * p;
            }
        }

        gini
    }

    fn majority_vote(&self, labels: &[usize], sample_indices: &[usize]) -> usize {
        let mut class_counts = vec![0usize; self.n_classes];
        for &i in sample_indices {
            class_counts[labels[i]] += 1;
        }

        class_counts.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn predict(&self, features: &[f32]) -> usize {
        let mut votes = vec![0usize; self.n_classes];
        for tree in &self.trees {
            let pred = self.predict_tree(tree, features, 0);
            votes[pred] += 1;
        }

        votes.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn predict_tree(&self, tree: &DecisionTree, features: &[f32], node_idx: usize) -> usize {
        let node = &tree.nodes[node_idx];

        if let Some(pred) = node.class_prediction {
            return pred;
        }

        let feat_idx = node.feature_idx.unwrap();
        let thresh = node.threshold;

        if features[feat_idx] <= thresh {
            self.predict_tree(tree, features, node.left_child.unwrap())
        } else {
            self.predict_tree(tree, features, node.right_child.unwrap())
        }
    }

    #[allow(dead_code)]
    fn predict_proba(&self, features: &[f32]) -> Vec<f32> {
        let mut votes = vec![0usize; self.n_classes];
        for tree in &self.trees {
            let pred = self.predict_tree(tree, features, 0);
            votes[pred] += 1;
        }

        let total = self.trees.len() as f32;
        votes.iter().map(|&v| v as f32 / total).collect()
    }
}

struct SimpleRng { state: u64 }

impl SimpleRng {
    fn seed(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_usize(&mut self, max: usize) -> usize {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        ((self.state.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as usize) % max
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Gatekeeper RF Training (76D) - Full Tree Export                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Load data
    println!("Loading BEANS-Zero dataset...");
    let manifest: BeansManifest = serde_json::from_str(&fs::read_to_string("beans_zero_full_manifest.json")?)?;
    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_manifest: CacheManifest = serde_json::from_str(&fs::read_to_string(cache_dir.join("cache_manifest.json"))?)?;

    // Collect features and labels
    let mut all_features: Vec<Vec<f32>> = Vec::new();
    let mut all_labels: Vec<usize> = Vec::new();

    for sample in &manifest.samples {
        let species = if sample.labels.output != "None" {
            &sample.labels.output
        } else {
            &sample.labels.task
        };

        let taxon = species_to_taxon(species, &sample.labels.task);
        let label_idx = consolidated_taxon_to_idx(taxon);

        if let Some(cache_file) = cache_manifest.entries.get(&sample.audio_file) {
            let path = cache_dir.join(cache_file);
            if path.exists() {
                if let Ok(file) = fs::File::open(&path) {
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(BufReader::new(file)) {
                        if features.len() == FEATURE_DIM {
                            // Extract 76D gatekeeper features
                            let mut gatekeeper_features = Vec::with_capacity(GATEKEEPER_DIM);
                            gatekeeper_features.extend_from_slice(&features[0..46]); // Base Physics
                            gatekeeper_features.extend_from_slice(&features[46..76]); // Macro Texture

                            all_features.push(gatekeeper_features);
                            all_labels.push(label_idx);
                        }
                    }
                }
            }
        }
    }

    println!("Loaded {} samples", all_features.len());

    // Print class distribution
    let mut class_counts = vec![0usize; 6];
    for &label in &all_labels {
        class_counts[label] += 1;
    }

    println!("\nClass distribution:");
    for (i, count) in class_counts.iter().enumerate() {
        let taxon = idx_to_consolidated_taxon(i);
        println!("  {:?}: {} samples ({:.1}%)", taxon, count, *count as f64 / all_features.len() as f64 * 100.0);
    }

    // Train/validation split (80/20)
    let n_train = (all_features.len() as f32 * 0.8) as usize;
    let train_features = all_features[..n_train].to_vec();
    let train_labels = all_labels[..n_train].to_vec();
    let val_features = all_features[n_train..].to_vec();
    let val_labels = all_labels[n_train..].to_vec();

    println!("\nTrain: {} | Val: {}", train_features.len(), val_features.len());

    // Compute normalization
    let n_features = GATEKEEPER_DIM;
    let mut feature_means = vec![0.0f32; n_features];
    let mut feature_stds = vec![0.0f32; n_features];

    for feat in &train_features {
        for (i, &v) in feat.iter().enumerate() {
            feature_means[i] += v;
        }
    }

    for mean in &mut feature_means {
        *mean /= train_features.len() as f32;
    }

    for feat in &train_features {
        for (i, &v) in feat.iter().enumerate() {
            let diff = v - feature_means[i];
            feature_stds[i] += diff * diff;
        }
    }

    for std in &mut feature_stds {
        *std = (*std / train_features.len() as f32).sqrt().max(1e-8);
    }

    // Normalize features
    let train_features_norm: Vec<Vec<f32>> = train_features.iter()
        .map(|f| f.iter().enumerate()
            .map(|(i, &v)| (v - feature_means[i]) / feature_stds[i])
            .collect())
        .collect();

    let val_features_norm: Vec<Vec<f32>> = val_features.iter()
        .map(|f| f.iter().enumerate()
            .map(|(i, &v)| (v - feature_means[i]) / feature_stds[i])
            .collect())
        .collect();

    // Train RF
    println!("\nTraining Gatekeeper RF (300 trees, max_depth=30)...");
    let mut rf = SimpleRF::new(6);
    rf.train(&train_features_norm, &train_labels, 300, 30);

    // Evaluate
    let mut train_correct = 0;
    for (features, &label) in train_features_norm.iter().zip(train_labels.iter()) {
        if rf.predict(features) == label {
            train_correct += 1;
        }
    }
    let train_accuracy = train_correct as f64 / train_features.len() as f64 * 100.0;

    let mut val_correct = 0;
    for (features, &label) in val_features_norm.iter().zip(val_labels.iter()) {
        if rf.predict(features) == label {
            val_correct += 1;
        }
    }
    let val_accuracy = val_correct as f64 / val_features.len() as f64 * 100.0;

    println!("  Train Accuracy: {:.2}%", train_accuracy);
    println!("  Val Accuracy:   {:.2}%", val_accuracy);

    // Save model
    let model = GatekeeperRFModel {
        trees: rf.trees.clone(),
        n_classes: 6,
        idx_to_label: consolidated_taxon_labels(),
        feature_means,
        feature_stds,
        train_accuracy,
        val_accuracy,
    };

    let json = serde_json::to_string_pretty(&model)?;
    fs::write("gatekeeper_rf_76d_trees.json", &json)?;

    println!("\nModel saved to: gatekeeper_rf_76d_trees.json");
    println!("  Trees: {}", model.trees.len());
    println!("  Classes: {}", model.n_classes);

    Ok(())
}
