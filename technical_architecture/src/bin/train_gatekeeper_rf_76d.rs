//! Train Gatekeeper RF (76D) with Consolidated Taxonomic Classes - Pure Rust
//! ===========================================================================
//!
//! This implements the improved Divide and Conquer architecture:
//! - Input: 76D (Base Physics 46D + Macro Texture 30D)
//! - Output: 6 consolidated classes (Bird, Mammal, MarineMammal, Insect, Amphibian, Unknown)
//!
//! Pure Rust implementation - no Python dependencies.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;

use technical_architecture::taxonomic_router::{
    Taxon, ConsolidatedTaxon, GATEKEEPER_DIM, PHYSICS_DIM,
    consolidate_taxon, consolidated_taxon_to_idx, idx_to_consolidated_taxon,
    consolidated_taxon_labels, map_species_to_taxon, map_task_to_taxon,
};

// =============================================================================
// Simple RNG (XorShift64)
// =============================================================================

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn seed(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }
}

// =============================================================================
// Random Forest Implementation (Pure Rust)
// =============================================================================

/// Decision tree node for 76D features
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TreeNode {
    feature_idx: Option<usize>,
    threshold: Option<f32>,
    left: Option<usize>,
    right: Option<usize>,
    prediction: Option<usize>,
    n_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecisionTree76D {
    nodes: Vec<TreeNode>,
    n_classes: usize,
    feature_dim: usize,
}

impl DecisionTree76D {
    fn new(n_classes: usize) -> Self {
        Self {
            nodes: Vec::new(),
            n_classes,
            feature_dim: GATEKEEPER_DIM,
        }
    }

    fn train_with_indices(
        &mut self,
        data: &[Vec<f32>],
        labels: &[usize],
        bootstrap_indices: &[usize],
        max_depth: usize,
        min_samples_split: usize,
        rng: &mut SimpleRng,
    ) {
        if bootstrap_indices.is_empty() {
            return;
        }

        // Map bootstrap indices to local 0..n range for this tree
        let local_indices: Vec<usize> = (0..bootstrap_indices.len()).collect();
        self.nodes = Vec::new();
        self.build_tree_with_indices(data, labels, bootstrap_indices, &local_indices, 0, max_depth, min_samples_split, rng);
    }

    fn build_tree_with_indices(
        &mut self,
        data: &[Vec<f32>],
        labels: &[usize],
        bootstrap_indices: &[usize],
        local_indices: &[usize],
        depth: usize,
        max_depth: usize,
        min_samples_split: usize,
        rng: &mut SimpleRng,
    ) -> usize {
        let node_idx = self.nodes.len();
        self.nodes.push(TreeNode {
            feature_idx: None,
            threshold: None,
            left: None,
            right: None,
            prediction: None,
            n_samples: local_indices.len(),
        });

        if local_indices.is_empty() {
            self.nodes[node_idx].prediction = Some(0);
            return node_idx;
        }

        // Count classes using bootstrap mapping
        let mut class_counts = vec![0usize; self.n_classes];
        for &local_i in local_indices {
            let data_i = bootstrap_indices[local_i];
            class_counts[labels[data_i]] += 1;
        }

        // Check stopping conditions
        let pure_class = class_counts.iter().position(|&c| c == local_indices.len());
        if pure_class.is_some() || depth >= max_depth || local_indices.len() < min_samples_split {
            let pred = class_counts.iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.nodes[node_idx].prediction = Some(pred);
            return node_idx;
        }

        // Find best split using random feature subset
        let n_features = self.feature_dim;
        let n_candidates = (n_features as f64).sqrt() as usize;
        let mut feature_candidates: Vec<usize> = (0..n_features).collect();
        for i in 0..feature_candidates.len().min(n_candidates) {
            let j = rng.next_usize(feature_candidates.len() - i) + i;
            feature_candidates.swap(i, j);
        }
        let features_to_try = &feature_candidates[..n_candidates.min(feature_candidates.len())];

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_feature = 0;
        let mut best_threshold = 0.0f32;

        let current_gini = gini_impurity(&class_counts, local_indices.len());

        for &feat_idx in features_to_try {
            // Get values using bootstrap mapping
            let mut values: Vec<f32> = local_indices.iter()
                .map(|&local_i| {
                    let data_i = bootstrap_indices[local_i];
                    data[data_i][feat_idx]
                })
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Sample thresholds - try more thresholds for better accuracy
            let max_thresholds = 50.min(values.len().saturating_sub(1));
            let step = (values.len() / max_thresholds).max(1);
            for i in (0..values.len().saturating_sub(1)).step_by(step) {
                let threshold = (values[i] + values[i + 1]) / 2.0;

                let mut left_counts = vec![0usize; self.n_classes];
                let mut right_counts = vec![0usize; self.n_classes];
                let mut n_left = 0;
                let mut n_right = 0;

                for &local_i in local_indices {
                    let data_i = bootstrap_indices[local_i];
                    if data[data_i][feat_idx] < threshold {
                        left_counts[labels[data_i]] += 1;
                        n_left += 1;
                    } else {
                        right_counts[labels[data_i]] += 1;
                        n_right += 1;
                    }
                }

                if n_left == 0 || n_right == 0 {
                    continue;
                }

                let left_gini = gini_impurity(&left_counts, n_left);
                let right_gini = gini_impurity(&right_counts, n_right);
                let weighted_gini = (n_left as f64 * left_gini + n_right as f64 * right_gini)
                    / local_indices.len() as f64;
                let gain = current_gini - weighted_gini;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feat_idx;
                    best_threshold = threshold;
                }
            }
        }

        if best_gain <= 0.0 {
            let pred = class_counts.iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.nodes[node_idx].prediction = Some(pred);
            return node_idx;
        }

        // Split indices
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for &local_i in local_indices {
            let data_i = bootstrap_indices[local_i];
            if data[data_i][best_feature] < best_threshold {
                left_indices.push(local_i);
            } else {
                right_indices.push(local_i);
            }
        }

        self.nodes[node_idx].feature_idx = Some(best_feature);
        self.nodes[node_idx].threshold = Some(best_threshold);

        let left_idx = self.build_tree_with_indices(data, labels, bootstrap_indices, &left_indices, depth + 1, max_depth, min_samples_split, rng);
        let right_idx = self.build_tree_with_indices(data, labels, bootstrap_indices, &right_indices, depth + 1, max_depth, min_samples_split, rng);

        self.nodes[node_idx].left = Some(left_idx);
        self.nodes[node_idx].right = Some(right_idx);

        node_idx
    }

    fn train(
        &mut self,
        data: &[Vec<f32>],
        labels: &[usize],
        max_depth: usize,
        min_samples_split: usize,
        rng: &mut SimpleRng,
    ) {
        if data.is_empty() {
            return;
        }

        let indices: Vec<usize> = (0..data.len()).collect();
        self.nodes = Vec::new();
        self.build_tree(data, labels, &indices, 0, max_depth, min_samples_split, rng);
    }

    fn build_tree(
        &mut self,
        data: &[Vec<f32>],
        labels: &[usize],
        indices: &[usize],
        depth: usize,
        max_depth: usize,
        min_samples_split: usize,
        rng: &mut SimpleRng,
    ) -> usize {
        let node_idx = self.nodes.len();
        self.nodes.push(TreeNode {
            feature_idx: None,
            threshold: None,
            left: None,
            right: None,
            prediction: None,
            n_samples: indices.len(),
        });

        // Check stopping conditions
        if indices.is_empty() {
            self.nodes[node_idx].prediction = Some(0);
            return node_idx;
        }

        // Count classes
        let mut class_counts = vec![0usize; self.n_classes];
        for &i in indices {
            class_counts[labels[i]] += 1;
        }

        // Pure node or max depth or min samples
        let pure_class = class_counts.iter().position(|&c| c == indices.len());
        if pure_class.is_some() || depth >= max_depth || indices.len() < min_samples_split {
            let pred = class_counts.iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.nodes[node_idx].prediction = Some(pred);
            return node_idx;
        }

        // Find best split using random feature subset
        let n_features = self.feature_dim;
        let n_candidates = (n_features as f64).sqrt() as usize; // sqrt(n_features) for RF
        let mut feature_candidates: Vec<usize> = (0..n_features).collect();
        for i in 0..feature_candidates.len().min(n_candidates) {
            let j = rng.next_usize(feature_candidates.len() - i) + i;
            feature_candidates.swap(i, j);
        }
        let features_to_try = &feature_candidates[..n_candidates.min(feature_candidates.len())];

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_feature = 0;
        let mut best_threshold = 0.0f32;

        let current_gini = gini_impurity(&class_counts, indices.len());

        for &feat_idx in features_to_try {
            // Get all unique values for this feature
            let mut values: Vec<f32> = indices.iter().map(|&i| data[i][feat_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Try midpoints between adjacent values
            for i in 0..values.len().saturating_sub(1) {
                let threshold = (values[i] + values[i + 1]) / 2.0;

                // Split
                let mut left_counts = vec![0usize; self.n_classes];
                let mut right_counts = vec![0usize; self.n_classes];
                let mut n_left = 0;
                let mut n_right = 0;

                for &idx in indices {
                    if data[idx][feat_idx] < threshold {
                        left_counts[labels[idx]] += 1;
                        n_left += 1;
                    } else {
                        right_counts[labels[idx]] += 1;
                        n_right += 1;
                    }
                }

                if n_left == 0 || n_right == 0 {
                    continue;
                }

                // Calculate information gain
                let left_gini = gini_impurity(&left_counts, n_left);
                let right_gini = gini_impurity(&right_counts, n_right);
                let weighted_gini = (n_left as f64 * left_gini + n_right as f64 * right_gini)
                    / indices.len() as f64;
                let gain = current_gini - weighted_gini;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feat_idx;
                    best_threshold = threshold;
                }
            }
        }

        // If no good split found, make leaf
        if best_gain <= 0.0 {
            let pred = class_counts.iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.nodes[node_idx].prediction = Some(pred);
            return node_idx;
        }

        // Split indices
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for &i in indices {
            if data[i][best_feature] < best_threshold {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        // Set node values
        self.nodes[node_idx].feature_idx = Some(best_feature);
        self.nodes[node_idx].threshold = Some(best_threshold);

        // Recursively build children
        let left_idx = self.build_tree(data, labels, &left_indices, depth + 1, max_depth, min_samples_split, rng);
        let right_idx = self.build_tree(data, labels, &right_indices, depth + 1, max_depth, min_samples_split, rng);

        self.nodes[node_idx].left = Some(left_idx);
        self.nodes[node_idx].right = Some(right_idx);

        node_idx
    }

    fn predict(&self, features: &[f32]) -> usize {
        if self.nodes.is_empty() { return 0; }
        self.predict_node(0, features)
    }

    fn predict_node(&self, node_idx: usize, features: &[f32]) -> usize {
        let node = &self.nodes[node_idx];
        if let Some(pred) = node.prediction { return pred; }
        if let (Some(feat_idx), Some(thresh), Some(left), Some(right)) =
            (node.feature_idx, node.threshold, node.left, node.right)
        {
            if features[feat_idx] < thresh {
                self.predict_node(left, features)
            } else {
                self.predict_node(right, features)
            }
        } else { 0 }
    }
}

fn gini_impurity(counts: &[usize], total: usize) -> f64 {
    if total == 0 { return 0.0; }
    let mut sum_sq = 0.0;
    for &c in counts {
        let p = c as f64 / total as f64;
        sum_sq += p * p;
    }
    1.0 - sum_sq
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RandomForest76D {
    trees: Vec<DecisionTree76D>,
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
    n_classes: usize,
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
}

impl RandomForest76D {
    fn new(n_estimators: usize, max_depth: usize, min_samples_split: usize, n_classes: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_estimators,
            max_depth,
            min_samples_split,
            n_classes,
            feature_means: vec![0.0; GATEKEEPER_DIM],
            feature_stds: vec![1.0; GATEKEEPER_DIM],
        }
    }

    fn train(&mut self, data: &[Vec<f32>], labels: &[usize]) {
        use rayon::prelude::*;

        println!("Training {} trees with bootstrap sampling (parallel, memory-efficient)...", self.n_estimators);

        let n_samples = data.len();
        let n_classes = self.n_classes;
        let max_depth = self.max_depth;
        let min_samples_split = self.min_samples_split;

        // Pre-generate bootstrap indices (much smaller than cloning data)
        let bootstrap_indices: Vec<Vec<usize>> = (0..self.n_estimators)
            .map(|i| {
                let mut rng = SimpleRng::seed(42 + i as u64);
                let mut indices = Vec::with_capacity(n_samples);
                for _ in 0..n_samples {
                    indices.push(rng.next_usize(n_samples));
                }
                indices
            })
            .collect();

        println!("Bootstrap indices generated. Training trees...");

        // Train trees in parallel using references
        self.trees = bootstrap_indices
            .into_par_iter()
            .enumerate()
            .map(|(i, bootstrap_idx)| {
                let mut tree_rng = SimpleRng::seed(1000 + i as u64);
                let mut tree = DecisionTree76D::new(n_classes);
                // Train using bootstrap indices (no data cloning!)
                tree.train_with_indices(data, labels, &bootstrap_idx, max_depth, min_samples_split, &mut tree_rng);
                tree
            })
            .collect();

        println!("Training complete!");
    }

    /// Predict on pre-normalized features (assumes caller has normalized)
    fn predict_normalized(&self, features: &[f32]) -> usize {
        if self.trees.is_empty() { return 0; }

        // Vote - features already normalized by caller
        let mut votes = vec![0usize; self.n_classes];
        for tree in &self.trees {
            let pred = tree.predict(features);
            votes[pred] += 1;
        }

        votes.iter().enumerate()
            .max_by_key(|(_, &c)| c as i32)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Predict on raw features (normalizes internally)
    fn predict(&self, features: &[f32]) -> usize {
        if self.trees.is_empty() { return 0; }

        // Normalize features
        let normalized: Vec<f32> = features.iter()
            .enumerate()
            .map(|(i, &v)| (v - self.feature_means[i]) / self.feature_stds[i])
            .collect();

        self.predict_normalized(&normalized)
    }

    fn predict_proba(&self, features: &[f32]) -> Vec<f32> {
        if self.trees.is_empty() {
            return vec![0.0; self.n_classes];
        }

        // Normalize features
        let normalized: Vec<f32> = features.iter()
            .enumerate()
            .map(|(i, &v)| (v - self.feature_means[i]) / self.feature_stds[i])
            .collect();

        // Vote
        let mut votes = vec![0usize; self.n_classes];
        for tree in &self.trees {
            let pred = tree.predict(&normalized);
            votes[pred] += 1;
        }

        let total = self.trees.len() as f32;
        votes.iter().map(|&c| c as f32 / total).collect()
    }
}

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Debug, Deserialize)]
struct BeansManifest {
    samples: Vec<BeansSample>,
}

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
struct CacheManifest {
    entries: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
struct GatekeeperMetadata {
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
    class_labels: Vec<String>,
    train_accuracy: f64,
    val_accuracy: f64,
}

// =============================================================================
// Main Training
// =============================================================================

fn main() -> Result<()> {
    println!("=== Gatekeeper RF (76D) Training - Pure Rust ===");
    println!("Consolidated Classes: Bird, Mammal, MarineMammal, Insect, Amphibian, Unknown\n");

    // Load manifest
    let manifest: BeansManifest = serde_json::from_str(
        &fs::read_to_string("beans_zero_full_manifest.json")?
    )?;

    // Load cache manifest
    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_manifest: CacheManifest = serde_json::from_str(
        &fs::read_to_string(cache_dir.join("cache_manifest.json"))?
    )?;

    println!("Loading features and computing consolidated labels...");

    // Collect features and consolidated labels
    let mut all_features: Vec<Vec<f32>> = Vec::new();
    let mut all_labels: Vec<ConsolidatedTaxon> = Vec::new();
    let mut class_counts: HashMap<String, usize> = HashMap::new();

    for sample in &manifest.samples {
        // Determine species label
        let species_label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };

        // Map to detailed taxon, then consolidate
        let taxon = map_species_to_taxon(&species_label);
        let taxon = if taxon == Taxon::Unknown {
            map_task_to_taxon(&species_label.replace("task_", ""))
        } else {
            taxon
        };
        let consolidated = consolidate_taxon(taxon);

        // Load 112D features and slice to 76D
        if let Some(cache_file) = cache_manifest.entries.get(&sample.audio_file) {
            let path = cache_dir.join(cache_file);
            if path.exists() {
                if let Ok(file) = fs::File::open(&path) {
                    let reader = BufReader::new(file);
                    if let Ok(features_112d) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                        if features_112d.len() == 112 {
                            // Slice to 76D: Base Physics (46D) + Macro Texture (30D)
                            let mut features_76d = vec![0.0f32; GATEKEEPER_DIM];
                            features_76d[..PHYSICS_DIM].copy_from_slice(&features_112d[..PHYSICS_DIM]);
                            features_76d[PHYSICS_DIM..GATEKEEPER_DIM].copy_from_slice(&features_112d[46..76]);

                            all_features.push(features_76d);
                            all_labels.push(consolidated);

                            *class_counts.entry(format!("{:?}", consolidated)).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
    }

    println!("Loaded {} samples", all_features.len());
    println!("\nClass distribution:");
    let mut total = 0;
    for (class, count) in &class_counts {
        println!("  {}: {} samples", class, count);
        total += count;
    }
    println!("  Total: {} samples", total);

    if all_features.is_empty() {
        return Err(anyhow::anyhow!("No features loaded!"));
    }

    // Stratified split: 80% train, 20% validation
    // Ensures rare classes are proportionally represented in both sets
    println!("\nPerforming stratified 80/20 split...");
    let n_samples = all_features.len();

    // Group indices by class
    let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, label) in all_labels.iter().enumerate() {
        let class_idx = consolidated_taxon_to_idx(*label);
        class_indices.entry(class_idx).or_default().push(i);
    }

    // Shuffle each class's indices and split 80/20
    let mut rng = SimpleRng::seed(12345);
    let mut train_indices: Vec<usize> = Vec::new();
    let mut val_indices: Vec<usize> = Vec::new();

    for (class_idx, mut indices) in class_indices {
        // Shuffle this class's indices
        for i in 0..indices.len() {
            let j = rng.next_usize(indices.len());
            indices.swap(i, j);
        }

        // Split 80/20
        let n_class_train = (indices.len() as f32 * 0.8) as usize;
        train_indices.extend(indices[..n_class_train].iter().copied());
        val_indices.extend(indices[n_class_train..].iter().copied());

        println!("  Class {:?}: {} train, {} val",
            idx_to_consolidated_taxon(class_idx),
            n_class_train,
            indices.len() - n_class_train
        );
    }

    // Shuffle train and val indices separately
    for i in 0..train_indices.len() {
        let j = rng.next_usize(train_indices.len());
        train_indices.swap(i, j);
    }
    for i in 0..val_indices.len() {
        let j = rng.next_usize(val_indices.len());
        val_indices.swap(i, j);
    }

    let n_train = train_indices.len();
    let n_val = val_indices.len();

    println!("\nTotal - Train: {} samples, Val: {} samples", n_train, n_val);

    // Compute normalization from training set only
    println!("Computing normalization parameters from training set...");
    let mut feature_means = vec![0.0f32; GATEKEEPER_DIM];
    let mut feature_stds = vec![0.0f32; GATEKEEPER_DIM];

    for &i in &train_indices {
        for (j, &v) in all_features[i].iter().enumerate() {
            feature_means[j] += v;
        }
    }
    for m in &mut feature_means { *m /= n_train as f32; }

    for &i in &train_indices {
        for (j, &v) in all_features[i].iter().enumerate() {
            feature_stds[j] += (v - feature_means[j]).powi(2);
        }
    }
    for s in &mut feature_stds { *s = (*s / n_train as f32).sqrt().max(1e-8); }

    // Normalize all features
    let normalized_features: Vec<Vec<f32>> = all_features.iter()
        .map(|f| f.iter().enumerate()
            .map(|(j, &v)| (v - feature_means[j]) / feature_stds[j])
            .collect())
        .collect();

    // Prepare training and validation data using stratified indices
    let train_x: Vec<Vec<f32>> = train_indices.iter()
        .map(|&i| normalized_features[i].to_vec())
        .collect();
    let train_y: Vec<usize> = train_indices.iter()
        .map(|&i| consolidated_taxon_to_idx(all_labels[i]))
        .collect();

    let val_x: Vec<Vec<f32>> = val_indices.iter()
        .map(|&i| normalized_features[i].to_vec())
        .collect();
    let val_y: Vec<usize> = val_indices.iter()
        .map(|&i| consolidated_taxon_to_idx(all_labels[i]))
        .collect();

    // Train Random Forest
    let mut rf = RandomForest76D::new(300, 30, 3, 6);
    rf.feature_means = feature_means.clone();
    rf.feature_stds = feature_stds.clone();
    rf.train(&train_x, &train_y);

    // Evaluate on training set (data is already normalized)
    let mut train_correct = 0usize;
    for (i, features) in train_x.iter().enumerate() {
        let pred = rf.predict_normalized(features);
        if pred == train_y[i] {
            train_correct += 1;
        }
    }
    let train_acc = train_correct as f64 / train_x.len() as f64;

    // Evaluate on validation set (data is already normalized)
    println!("\nEvaluating on validation set...");
    let mut val_correct = 0usize;
    let mut class_correct: HashMap<usize, usize> = HashMap::new();
    let mut class_total: HashMap<usize, usize> = HashMap::new();

    for (i, features) in val_x.iter().enumerate() {
        let pred = rf.predict_normalized(features);
        let true_label = val_y[i];

        *class_total.entry(true_label).or_insert(0) += 1;
        if pred == true_label {
            val_correct += 1;
            *class_correct.entry(true_label).or_insert(0) += 1;
        }
    }

    let val_acc = val_correct as f64 / val_x.len() as f64;

    println!("\n=== Results ===");
    println!("Training Accuracy: {:.2}%", train_acc * 100.0);
    println!("Validation Accuracy: {:.2}%", val_acc * 100.0);
    println!("\nPer-class validation accuracy:");
    for (class_idx, &total) in &class_total {
        let correct = *class_correct.get(class_idx).unwrap_or(&0);
        let class_acc = correct as f64 / total as f64;
        let taxon = idx_to_consolidated_taxon(*class_idx);
        println!("  {:?}: {:.2}% ({}/{})", taxon, class_acc * 100.0, correct, total);
    }

    // Save model
    let model_path = "gatekeeper_rf_76d.bin";
    fs::write(model_path, bincode::serialize(&rf)?)?;
    println!("\nModel saved to: {}", model_path);

    // Save metadata
    let metadata = GatekeeperMetadata {
        n_estimators: 300,
        max_depth: 30,
        min_samples_split: 3,
        feature_means,
        feature_stds,
        class_labels: consolidated_taxon_labels(),
        train_accuracy: train_acc * 100.0,
        val_accuracy: val_acc * 100.0,
    };
    fs::write("gatekeeper_rf_76d.json", serde_json::to_string_pretty(&metadata)?)?;
    println!("Metadata saved to: gatekeeper_rf_76d.json");

    // Analysis
    println!("\n=== Analysis ===");
    if val_acc > 0.85 {
        println!("✓ Validation accuracy > 85%: Soft Veto should be beneficial!");
    } else if val_acc > 0.80 {
        println!("~ Validation accuracy > 80%: Soft Veto may help with higher threshold.");
    } else {
        println!("✗ Validation accuracy < 80%: Need to investigate class confusion.");
    }

    Ok(())
}
