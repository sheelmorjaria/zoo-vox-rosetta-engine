//! Train Taxonomy Gatekeeper Random Forest
//! =========================================
//!
//! Trains an RF on 46D physics features to predict TAXONOMIC GROUPS.
//! This is the "Gatekeeper" in the Hierarchical Veto Ensemble.
//!
//! Usage:
//!   cargo run --release --bin train_taxonomy_gatekeeper --features ml-classical

use anyhow::Result;
use ndarray::{Array1, Array2, s};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use technical_architecture::taxonomic_router::{
    Taxon, FEATURE_DIM, PHYSICS_DIM, slice_physics,
    map_species_to_taxon, map_task_to_taxon,
};

// =============================================================================
// Hyperparameters
// =============================================================================

const N_ESTIMATORS: usize = 300;
const MAX_DEPTH: usize = 30;
const MIN_SAMPLES_SPLIT: usize = 3;
const TRAIN_SPLIT: f32 = 0.9;

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

/// Taxonomy Gatekeeper Model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxonomyGatekeeper {
    /// Number of trees in the forest
    pub n_estimators: usize,
    /// Maximum depth of each tree
    pub max_depth: usize,
    /// Minimum samples to split
    pub min_samples_split: usize,
    /// Feature means for standardization
    pub feature_means: Vec<f32>,
    /// Feature stds for standardization
    pub feature_stds: Vec<f32>,
    /// Taxon labels
    pub taxon_labels: Vec<String>,
    /// Mapping from taxon to index
    pub taxon_to_idx: HashMap<String, usize>,
    /// Training accuracy
    pub train_accuracy: f32,
    /// Validation accuracy
    pub val_accuracy: f32,
}

// =============================================================================
// Decision Tree Node
// =============================================================================

/// A decision tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TreeNode {
    /// Feature index for split (None if leaf)
    feature_idx: Option<usize>,
    /// Threshold for split
    threshold: Option<f32>,
    /// Left child index (None if leaf)
    left: Option<usize>,
    /// Right child index (None if leaf)
    right: Option<usize>,
    /// Predicted class (for leaf nodes)
    prediction: Option<usize>,
    /// Number of samples at this node
    n_samples: usize,
}

impl TreeNode {
    fn leaf(prediction: usize, n_samples: usize) -> Self {
        Self {
            feature_idx: None,
            threshold: None,
            left: None,
            right: None,
            prediction: Some(prediction),
            n_samples,
        }
    }

    fn split(feature_idx: usize, threshold: f32, left: usize, right: usize, n_samples: usize) -> Self {
        Self {
            feature_idx: Some(feature_idx),
            threshold: Some(threshold),
            left: Some(left),
            right: Some(right),
            prediction: None,
            n_samples,
        }
    }

    fn is_leaf(&self) -> bool {
        self.prediction.is_some()
    }
}

// =============================================================================
// Decision Tree Classifier (46D)
// =============================================================================

/// Decision Tree for 46D physics features
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecisionTree46D {
    nodes: Vec<TreeNode>,
    n_classes: usize,
    feature_dim: usize,
}

impl DecisionTree46D {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            n_classes: 0,
            feature_dim: PHYSICS_DIM,
        }
    }

    fn fit(&mut self, x: &Array2<f32>, y: &[usize], max_depth: usize, min_samples_split: usize) {
        self.nodes.clear();
        self.n_classes = y.iter().max().map(|&m| m + 1).unwrap_or(1);

        let n_samples = x.nrows();
        let indices: Vec<usize> = (0..n_samples).collect();

        self.build_tree(x, y, &indices, 0, max_depth, min_samples_split);
    }

    fn build_tree(
        &mut self,
        x: &Array2<f32>,
        y: &[usize],
        indices: &[usize],
        depth: usize,
        max_depth: usize,
        min_samples_split: usize,
    ) -> usize {
        let n_samples = indices.len();

        // Check stopping conditions
        if n_samples < min_samples_split || depth >= max_depth {
            let prediction = self.majority_class(y, indices);
            let node_idx = self.nodes.len();
            self.nodes.push(TreeNode::leaf(prediction, n_samples));
            return node_idx;
        }

        // Check pure node
        let unique_classes: Vec<usize> = indices.iter().map(|&i| y[i]).collect();
        let first_class = unique_classes[0];
        if unique_classes.iter().all(|&c| c == first_class) {
            let node_idx = self.nodes.len();
            self.nodes.push(TreeNode::leaf(first_class, n_samples));
            return node_idx;
        }

        // Find best split
        if let Some((feature_idx, threshold, left_indices, right_indices)) =
            self.find_best_split(x, y, indices)
        {
            if left_indices.is_empty() || right_indices.is_empty() {
                let prediction = self.majority_class(y, indices);
                let node_idx = self.nodes.len();
                self.nodes.push(TreeNode::leaf(prediction, n_samples));
                return node_idx;
            }

            // Create node placeholder
            let node_idx = self.nodes.len();
            self.nodes.push(TreeNode::leaf(0, 0)); // Placeholder

            // Build children
            let left_child = self.build_tree(x, y, &left_indices, depth + 1, max_depth, min_samples_split);
            let right_child = self.build_tree(x, y, &right_indices, depth + 1, max_depth, min_samples_split);

            // Update node with split info
            self.nodes[node_idx] = TreeNode::split(feature_idx, threshold, left_child, right_child, n_samples);

            node_idx
        } else {
            let prediction = self.majority_class(y, indices);
            let node_idx = self.nodes.len();
            self.nodes.push(TreeNode::leaf(prediction, n_samples));
            node_idx
        }
    }

    fn find_best_split(
        &self,
        x: &Array2<f32>,
        y: &[usize],
        indices: &[usize],
    ) -> Option<(usize, f32, Vec<usize>, Vec<usize>)> {
        let n_features = x.ncols();
        let mut best_gain = f32::NEG_INFINITY;
        let mut best_split: Option<(usize, f32, Vec<usize>, Vec<usize>)> = None;

        // Current gini
        let current_gini = self.gini_impurity(y, indices);

        // Sample features for random forest (sqrt(n_features))
        let n_features_to_try = (n_features as f32).sqrt() as usize;
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        let mut rng = rand::thread_rng();
        feature_indices.partial_shuffle(&mut rng, n_features_to_try);

        for &feat_idx in &feature_indices[..n_features_to_try] {
            // Get unique thresholds
            let mut thresholds: Vec<f32> = indices.iter().map(|&i| x[[i, feat_idx]]).collect();
            thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            thresholds.dedup();

            // Sample thresholds (too many is slow)
            let n_thresholds = thresholds.len().min(20);
            let step = (thresholds.len() as f32 / n_thresholds as f32).ceil() as usize;

            for (ti, &threshold) in thresholds.iter().enumerate() {
                if ti % step != 0 && ti != thresholds.len() - 1 {
                    continue;
                }

                let left_indices: Vec<usize> = indices.iter()
                    .filter(|&&i| x[[i, feat_idx]] < threshold)
                    .copied()
                    .collect();

                let right_indices: Vec<usize> = indices.iter()
                    .filter(|&&i| x[[i, feat_idx]] >= threshold)
                    .copied()
                    .collect();

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                // Calculate information gain
                let left_gini = self.gini_impurity(y, &left_indices);
                let right_gini = self.gini_impurity(y, &right_indices);
                let n = indices.len() as f32;
                let weighted_gini = (left_indices.len() as f32 / n) * left_gini
                    + (right_indices.len() as f32 / n) * right_gini;
                let gain = current_gini - weighted_gini;

                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some((feat_idx, threshold, left_indices, right_indices));
                }
            }
        }

        best_split
    }

    fn gini_impurity(&self, y: &[usize], indices: &[usize]) -> f32 {
        if indices.is_empty() {
            return 0.0;
        }

        let mut class_counts = vec![0usize; self.n_classes];
        for &i in indices {
            class_counts[y[i]] += 1;
        }

        let n = indices.len() as f32;
        let mut gini = 1.0;
        for &count in &class_counts {
            if count > 0 {
                let p = count as f32 / n;
                gini -= p * p;
            }
        }

        gini
    }

    fn majority_class(&self, y: &[usize], indices: &[usize]) -> usize {
        let mut class_counts = vec![0usize; self.n_classes];
        for &i in indices {
            class_counts[y[i]] += 1;
        }

        class_counts.iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn predict(&self, features: &Array1<f32>) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        self.predict_node(0, features)
    }

    fn predict_node(&self, node_idx: usize, features: &Array1<f32>) -> usize {
        let node = &self.nodes[node_idx];

        if let Some(pred) = node.prediction {
            return pred;
        }

        if let (Some(feat_idx), Some(thresh), Some(left), Some(right)) =
            (node.feature_idx, node.threshold, node.left, node.right)
        {
            if features[feat_idx] < thresh {
                self.predict_node(left, features)
            } else {
                self.predict_node(right, features)
            }
        } else {
            0
        }
    }
}

// =============================================================================
// Random Forest Classifier (46D)
// =============================================================================

/// Random Forest for 46D physics features
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RandomForest46D {
    trees: Vec<DecisionTree46D>,
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
    n_classes: usize,
}

impl RandomForest46D {
    fn new(n_estimators: usize, max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_estimators,
            max_depth,
            min_samples_split,
            n_classes: 0,
        }
    }

    fn fit(&mut self, x: &Array2<f32>, y: &[usize]) {
        self.trees.clear();
        self.n_classes = y.iter().max().map(|&m| m + 1).unwrap_or(1);

        let n_samples = x.nrows();
        let sample_size = (n_samples as f32 * 0.8) as usize;

        for tree_idx in 0..self.n_estimators {
            let mut rng = rand::rngs::StdRng::seed_from_u64((tree_idx + 42) as u64);

            // Bootstrap sample
            let all_indices: Vec<usize> = (0..n_samples).collect();
            let bootstrap_indices: Vec<usize> = (0..sample_size)
                .map(|_| *all_indices.choose(&mut rng).unwrap())
                .collect();

            // Create bootstrap dataset
            let bootstrap_x = x.select(ndarray::Axis(0), &bootstrap_indices);
            let bootstrap_y: Vec<usize> = bootstrap_indices.iter().map(|&i| y[i]).collect();

            // Train tree
            let mut tree = DecisionTree46D::new();
            tree.fit(&bootstrap_x, &bootstrap_y, self.max_depth, self.min_samples_split);
            self.trees.push(tree);

            if (tree_idx + 1) % 50 == 0 {
                println!("    Trained {}/{} trees", tree_idx + 1, self.n_estimators);
            }
        }
    }

    fn predict(&self, features: &Array1<f32>) -> usize {
        if self.trees.is_empty() {
            return 0;
        }

        // Majority voting
        let mut votes = vec![0usize; self.n_classes];
        for tree in &self.trees {
            let pred = tree.predict(features);
            votes[pred] += 1;
        }

        votes.iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn predict_batch(&self, x: &Array2<f32>) -> Vec<usize> {
        x.rows()
            .into_iter()
            .map(|row| self.predict(&row.to_owned()))
            .collect()
    }

    fn predict_proba(&self, features: &Array1<f32>) -> Array1<f32> {
        if self.trees.is_empty() {
            return Array1::zeros(self.n_classes);
        }

        let mut votes = vec![0usize; self.n_classes];
        for tree in &self.trees {
            let pred = tree.predict(features);
            votes[pred] += 1;
        }

        let total = self.trees.len() as f32;
        Array1::from_vec(votes.iter().map(|&c| c as f32 / total).collect())
    }
}

// =============================================================================
// Main Training Loop
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Taxonomy Gatekeeper RF Training (Rust Implementation)           ║");
    println!("║  46D Physics → Taxonomic Groups                                   ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
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
    let cache_data = fs::read_to_string(&cache_manifest_path)?;
    let cache_manifest: CacheManifest = serde_json::from_str(&cache_data)?;
    println!("  Cached features available: {}", cache_manifest.entries.len());

    // Load all physics features and map to taxonomic groups
    println!("\nLoading physics features and mapping to taxonomic groups...");
    let mut all_physics: Vec<Vec<f32>> = Vec::new();
    let mut all_taxon_names: Vec<String> = Vec::new();

    for sample in &manifest.samples {
        let audio_file = &sample.audio_file;
        let species_label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };

        // Determine taxonomic group
        let taxon = map_species_to_taxon(&species_label);
        let taxon = if taxon == Taxon::Unknown {
            map_task_to_taxon(&sample.labels.task)
        } else {
            taxon
        };

        if let Some(cache_file) = cache_manifest.entries.get(audio_file) {
            let full_path = cache_dir.join(cache_file);
            if full_path.exists() {
                if let Ok(file) = fs::File::open(&full_path) {
                    let reader = BufReader::new(file);
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                        if features.len() == FEATURE_DIM {
                            // Extract only physics features (46D)
                            let physics = slice_physics(&features);
                            all_physics.push(physics.to_vec());
                            all_taxon_names.push(format!("{:?}", taxon));
                        }
                    }
                }
            }
        }
    }

    println!("  Loaded {} samples", all_physics.len());

    if all_physics.is_empty() {
        anyhow::bail!("No features loaded!");
    }

    // Build taxon mapping
    let mut unique_taxons: Vec<String> = all_taxon_names.iter().cloned().collect();
    unique_taxons.sort();
    unique_taxons.dedup();
    let n_classes = unique_taxons.len();

    let mut taxon_to_idx = HashMap::new();
    for (idx, taxon) in unique_taxons.iter().enumerate() {
        taxon_to_idx.insert(taxon.clone(), idx);
    }

    println!("  Taxonomic classes: {}", n_classes);

    // Show taxonomic distribution
    println!("\nTaxonomic Distribution:");
    let mut taxon_counts: HashMap<String, usize> = HashMap::new();
    for taxon in &all_taxon_names {
        *taxon_counts.entry(taxon.clone()).or_insert(0) += 1;
    }
    let total = all_taxon_names.len();

    let mut counts_vec: Vec<_> = taxon_counts.iter().collect();
    counts_vec.sort_by(|a, b| b.1.cmp(a.1));

    for (taxon, count) in &counts_vec {
        let pct = **count as f64 / total as f64 * 100.0;
        println!("  {:20} {:>6} ({:>5.1}%)", taxon, count, pct);
    }

    // Convert to ndarray
    let n_samples = all_physics.len();
    let mut feature_matrix: Array2<f32> = Array2::zeros((n_samples, PHYSICS_DIM));
    let y: Vec<usize> = all_taxon_names.iter()
        .map(|t| *taxon_to_idx.get(t).unwrap_or(&0))
        .collect();

    for (i, physics) in all_physics.iter().enumerate() {
        for (j, &v) in physics.iter().enumerate() {
            feature_matrix[[i, j]] = v;
        }
    }

    // Split into train/validation
    let n_train = (n_samples as f32 * TRAIN_SPLIT) as usize;

    // Shuffle indices with deterministic seed
    let mut indices: Vec<usize> = (0..n_samples).collect();
    for i in 0..indices.len() {
        let j = (rand_u32() as usize) % indices.len();
        indices.swap(i, j);
    }

    let train_indices: Vec<usize> = indices[..n_train].to_vec();
    let val_indices: Vec<usize> = indices[n_train..].to_vec();

    println!("\nSplit: {} train, {} validation", n_train, n_samples - n_train);

    // Compute normalization params from training set
    println!("\nComputing normalization parameters...");
    let mut feature_means = vec![0.0f32; PHYSICS_DIM];
    let mut feature_stds = vec![0.0f32; PHYSICS_DIM];

    for &i in &train_indices {
        for (j, &v) in all_physics[i].iter().enumerate() {
            feature_means[j] += v;
        }
    }
    for j in 0..PHYSICS_DIM {
        feature_means[j] /= train_indices.len() as f32;
    }

    for &i in &train_indices {
        for (j, &v) in all_physics[i].iter().enumerate() {
            let diff = v - feature_means[j];
            feature_stds[j] += diff * diff;
        }
    }
    for j in 0..PHYSICS_DIM {
        feature_stds[j] = (feature_stds[j] / train_indices.len() as f32).sqrt().max(1e-8);
    }

    // Standardize features
    println!("Standardizing features...");
    let mut normalized_matrix = Array2::zeros((n_samples, PHYSICS_DIM));
    for i in 0..n_samples {
        for j in 0..PHYSICS_DIM {
            normalized_matrix[[i, j]] = (feature_matrix[[i, j]] - feature_means[j]) / feature_stds[j];
        }
    }

    // Create train/test splits
    let train_x = normalized_matrix.select(ndarray::Axis(0), &train_indices);
    let train_y: Vec<usize> = train_indices.iter().map(|&i| y[i]).collect();
    let val_x = normalized_matrix.select(ndarray::Axis(0), &val_indices);
    let val_y: Vec<usize> = val_indices.iter().map(|&i| y[i]).collect();

    // Train Random Forest
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Training Taxonomy Gatekeeper RF                                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("  n_estimators: {}", N_ESTIMATORS);
    println!("  max_depth: {}", MAX_DEPTH);
    println!("  min_samples_split: {}", MIN_SAMPLES_SPLIT);
    println!();

    let mut rf = RandomForest46D::new(N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT);
    rf.fit(&train_x, &train_y);

    // Evaluate on training set
    println!("\nEvaluating on training set...");
    let train_preds = rf.predict_batch(&train_x);
    let train_correct = train_preds.iter()
        .zip(train_y.iter())
        .filter(|(p, y)| p == y)
        .count();
    let train_accuracy = train_correct as f32 / train_y.len() as f32 * 100.0;
    println!("  Train accuracy: {:.2}%", train_accuracy);

    // Evaluate on validation set
    println!("\nEvaluating on validation set...");
    let val_preds = rf.predict_batch(&val_x);
    let val_correct = val_preds.iter()
        .zip(val_y.iter())
        .filter(|(p, y)| p == y)
        .count();
    let val_accuracy = val_correct as f32 / val_y.len() as f32 * 100.0;
    println!("  Validation accuracy: {:.2}%", val_accuracy);

    // Per-taxon accuracy
    println!("\nPer-Taxon Validation Accuracy:");
    println!("{:<20} {:>8} {:>8} {:>10}", "Taxon", "Total", "Correct", "Accuracy");
    println!("{}", "-".repeat(50));

    for (taxon_name, &taxon_idx) in taxon_to_idx.iter() {
        let total: usize = val_y.iter().filter(|&&y| y == taxon_idx).count();

        if total > 0 {
            let correct = val_preds.iter()
                .zip(val_y.iter())
                .filter(|(&p, &y)| y == taxon_idx && p == y)
                .count();

            let acc = correct as f64 / total as f64 * 100.0;
            println!("{:<20} {:>8} {:>8} {:>9.1}%", taxon_name, total, correct, acc);
        }
    }

    // Confusion analysis
    println!("\nConfusion Analysis (Top 5 confusions):");
    let mut confusion_pairs: HashMap<(String, String), usize> = HashMap::new();
    for (&pred, &actual) in val_preds.iter().zip(val_y.iter()) {
        if pred != actual {
            let pred_taxon = &unique_taxons[pred];
            let actual_taxon = &unique_taxons[actual];
            *confusion_pairs.entry((actual_taxon.clone(), pred_taxon.clone())).or_insert(0) += 1;
        }
    }
    let mut confusion_vec: Vec<_> = confusion_pairs.into_iter().collect();
    confusion_vec.sort_by(|a, b| b.1.cmp(&a.1));
    for ((pred, actual), count) in confusion_vec.iter().take(5) {
        println!("  {} -> {}: {}", actual, pred, count);
    }

    // Save model
    let model = TaxonomyGatekeeper {
        n_estimators: N_ESTIMATORS,
        max_depth: MAX_DEPTH,
        min_samples_split: MIN_SAMPLES_SPLIT,
        feature_means,
        feature_stds,
        taxon_labels: unique_taxons.clone(),
        taxon_to_idx,
        train_accuracy,
        val_accuracy,
    };

    let model_path = "taxonomy_gatekeeper_rf.json";
    println!("\nSaving model metadata to: {}", model_path);
    fs::write(model_path, serde_json::to_string_pretty(&model)?)?;

    // Save the full RF model (trees)
    let rf_path = "taxonomy_gatekeeper_rf.bin";
    println!("Saving RF model to: {}", rf_path);
    let rf_bytes = bincode::serialize(&rf)?;
    fs::write(rf_path, rf_bytes)?;

    let elapsed = start.elapsed().as_secs_f32();

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Architecture:       Taxonomy Gatekeeper RF                       ║");
    println!("║  Input:             {}D Physics Features                          ║", PHYSICS_DIM);
    println!("║  Output:            {} Taxonomic Groups                            ║", n_classes);
    println!("║  Train Accuracy:    {:>8.2}%                                   ║", train_accuracy);
    println!("║  Val Accuracy:      {:>8.2}%                                   ║", val_accuracy);
    println!("║  Total Time:        {:>8.1}s                                    ║", elapsed);
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    println!("\nHierarchical Veto Ensemble Components:");
    println!("  1. Taxonomy Gatekeeper (RF 46D): {:.2}%", val_accuracy);
    println!("  2. Species Expert (NN 66D):     59.88%");
    println!("  3. Veto Mechanism:              (run eval_hierarchical_veto)");

    println!("\nComparison with Python implementation:");
    println!("  Python RF (sklearn):            67.29%");
    println!("  Rust RF (this):                 {:.2}%", val_accuracy);

    Ok(())
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
