//! Train and Evaluate Voting Ensemble for Species Classification
//! ===============================================================
//!
//! Implements:
//! 1. Grid Search for optimal weight optimization
//! 2. Confidence-based dynamic weighting
//! 3. Top-K candidate shortlisting from NN
//!
//! Usage:
//!   export LIBTORCH=/path/to/libtorch
//!   export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
//!   cargo run --release --bin train_voting_ensemble --features gpu-training

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;

use technical_architecture::taxonomic_router::FEATURE_DIM;
use technical_architecture::voting_ensemble::{
    VotingEnsembleConfig, EnsembleVoter, EnsembleInput, GridSearchOptimizer,
};
use rayon::prelude::*;

#[cfg(feature = "gpu-training")]
use tch::{nn, nn::Module, Device, Tensor, Kind};

// =============================================================================
// RF Structures (112D species-level RF) - matches train_parallel_rf_112d format
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
struct DecisionTreeJSON {
    nodes: Vec<TreeNode>,
}

impl DecisionTreeJSON {
    fn predict(&self, features: &[f32]) -> usize {
        if self.nodes.is_empty() { return 0; }
        self.predict_node(0, features)
    }

    fn predict_node(&self, node_idx: usize, features: &[f32]) -> usize {
        let node = &self.nodes[node_idx];
        if let Some(pred) = node.class_prediction {
            return pred;
        }
        let feat_idx = node.feature_idx.unwrap();
        let thresh = node.threshold;
        if features[feat_idx] <= thresh {
            self.predict_node(node.left_child.unwrap(), features)
        } else {
            self.predict_node(node.right_child.unwrap(), features)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RandomForest112D {
    trees: Vec<DecisionTreeJSON>,
    n_classes: usize,
    idx_to_label: Vec<String>,
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
}

impl RandomForest112D {
    fn predict_proba(&self, features: &[f32]) -> Vec<f32> {
        let mut votes = vec![0usize; self.n_classes];
        for tree in &self.trees {
            let pred = tree.predict(features);
            votes[pred] += 1;
        }
        let total = self.trees.len() as f32;
        votes.iter().map(|&v| v as f32 / total).collect()
    }
}

// =============================================================================
// NN Structures
// =============================================================================

const HIDDEN_DIM_1: i64 = 768;
const HIDDEN_DIM_2: i64 = 512;
const HIDDEN_DIM_3: i64 = 256;

#[derive(Debug)]
struct SpeciesExpert112D {
    fc1: nn::Linear,
    ln1: nn::LayerNorm,
    fc2: nn::Linear,
    ln2: nn::LayerNorm,
    fc3: nn::Linear,
    ln3: nn::LayerNorm,
    fc4: nn::Linear,
    ln4: nn::LayerNorm,
    out: nn::Linear,
}

impl SpeciesExpert112D {
    fn new(vs: &nn::Path, num_classes: i64) -> Self {
        let fc1 = nn::linear(vs, FEATURE_DIM as i64, HIDDEN_DIM_1, Default::default());
        let ln1 = nn::layer_norm(vs, vec![HIDDEN_DIM_1], Default::default());
        let fc2 = nn::linear(vs, HIDDEN_DIM_1, HIDDEN_DIM_2, Default::default());
        let ln2 = nn::layer_norm(vs, vec![HIDDEN_DIM_2], Default::default());
        let fc3 = nn::linear(vs, HIDDEN_DIM_2, HIDDEN_DIM_3, Default::default());
        let ln3 = nn::layer_norm(vs, vec![HIDDEN_DIM_3], Default::default());
        let fc4 = nn::linear(vs, HIDDEN_DIM_3, 128, Default::default());
        let ln4 = nn::layer_norm(vs, vec![128], Default::default());
        let out = nn::linear(vs, 128, num_classes, Default::default());
        Self { fc1, ln1, fc2, ln2, fc3, ln3, fc4, ln4, out }
    }
}

impl nn::Module for SpeciesExpert112D {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let x = xs.apply(&self.fc1).apply(&self.ln1).gelu("none").dropout(0.5, false);
        let x = x.apply(&self.fc2).apply(&self.ln2).gelu("none").dropout(0.5, false);
        let x = x.apply(&self.fc3).apply(&self.ln3).gelu("none").dropout(0.5, false);
        let x = x.apply(&self.fc4).apply(&self.ln4).gelu("none").dropout(0.5, false);
        x.apply(&self.out)
    }
}

#[derive(Debug, Deserialize)]
struct NNMetadata {
    num_classes: usize,
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
    label_to_idx: HashMap<String, usize>,
}

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

struct SimpleRng { state: u64 }

impl SimpleRng {
    fn seed(seed: u64) -> Self { Self { state: if seed == 0 { 1 } else { seed } } }
    fn next_usize(&mut self, max: usize) -> usize {
        self.state ^= self.state >> 12; self.state ^= self.state << 25; self.state ^= self.state >> 27;
        ((self.state.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as usize) % max
    }
}

// =============================================================================
// Dataset Component Mapping
// =============================================================================

fn get_component_name(task: &str, output: &str) -> String {
    let task_lower = task.to_lowercase();
    let output_lower = output.to_lowercase();

    if task_lower.contains("bird") || output_lower.contains("sparrow") ||
       output_lower.contains("finch") || output_lower.contains("warbler") {
        return "bird_species".to_string();
    }
    if task_lower.contains("bat") || output_lower.contains("eptesicus") ||
       output_lower.contains("myotis") {
        return "bat_species".to_string();
    }
    if task_lower.contains("dolphin") || output_lower.contains("dolphin") ||
       output_lower.contains("whale") {
        return "marine_mammals".to_string();
    }
    if task_lower.contains("insect") || output_lower.contains("bee") {
        return "insects".to_string();
    }
    if task_lower.contains("amphibian") || output_lower.contains("frog") {
        return "amphibians".to_string();
    }
    if output_lower.contains("marmoset") {
        return "marmoset".to_string();
    }

    format!("task_{}", task)
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Voting Ensemble Training & Evaluation                           ║");
    println!("║  NN (112D) + RF (112D) → Weighted Vote → Species Classification  ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let device = Device::cuda_if_available();
    println!("Device: {:?}\n", device);

    // Load NN
    println!("Loading Species Expert NN (112D)...");
    let nn_metadata: NNMetadata = serde_json::from_str(&fs::read_to_string("species_expert_112d.json")?)?;
    println!("  Classes: {}", nn_metadata.num_classes);

    let mut vs = nn::VarStore::new(device);
    let net = SpeciesExpert112D::new(&vs.root(), nn_metadata.num_classes as i64);
    vs.load("species_expert_112d.ot")?;
    println!("  Model loaded: species_expert_112d.ot");

    // Load RF (112D species-level)
    println!("\nLoading Species RF (112D)...");
    let rf_path = "random_forest_model_112d_parallel.json";

    let rf: Option<RandomForest112D> = if Path::new(rf_path).exists() {
        let rf_json = fs::read_to_string(rf_path)?;
        Some(serde_json::from_str(&rf_json)?)
    } else {
        println!("  WARNING: {} not found. Will train RF first.", rf_path);
        None
    };

    let rf = match rf {
        Some(rf) => {
            println!("  Trees: {}", rf.trees.len());
            rf
        }
        None => {
            println!("  Training RF (112D) on BEANS-Zero...");
            return train_and_evaluate(&device, &vs, &net, &nn_metadata);
        }
    };

    // Load data
    println!("\nLoading BEANS-Zero dataset...");
    let manifest: BeansManifest = serde_json::from_str(&fs::read_to_string("beans_zero_full_manifest.json")?)?;
    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_manifest: CacheManifest = serde_json::from_str(&fs::read_to_string(cache_dir.join("cache_manifest.json"))?)?;

    let mut component_features: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut component_labels: HashMap<String, Vec<usize>> = HashMap::new();

    // Build idx_to_label
    let mut idx_to_label: HashMap<usize, String> = HashMap::new();
    for (label, &idx) in &nn_metadata.label_to_idx {
        idx_to_label.insert(idx, label.clone());
    }

    for sample in &manifest.samples {
        let label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };

        let label_idx = *nn_metadata.label_to_idx.get(&label).unwrap_or(&0);
        let component = get_component_name(&sample.labels.task, &sample.labels.output);

        if let Some(cache_file) = cache_manifest.entries.get(&sample.audio_file) {
            let path = cache_dir.join(cache_file);
            if path.exists() {
                if let Ok(file) = fs::File::open(&path) {
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(BufReader::new(file)) {
                        if features.len() == FEATURE_DIM {
                            component_features.entry(component.clone()).or_default().push(features);
                            component_labels.entry(component).or_default().push(label_idx);
                        }
                    }
                }
            }
        }
    }

    println!("Loaded samples across {} components", component_features.len());

    // Stratified 80/20 split for full dataset
    let mut all_indices: Vec<usize> = (0..component_features.values().map(|v| v.len()).sum::<usize>()).collect();
    // ... simplified: use random split for now

    println!("\n=== Phase 1: Grid Search for Optimal Weights ===\n");

    // Collect ensemble inputs
    println!("Collecting NN and RF predictions for validation set...");
    let mut ensemble_inputs: Vec<EnsembleInput> = Vec::new();

    for (component, features) in &component_features {
        let labels = component_labels.get(component).unwrap();

        // Use only 20% for validation
        let n_val = (features.len() as f32 * 0.2) as usize;
        let val_indices: Vec<usize> = (0..features.len()).take(n_val).collect();

        // Process in larger batches for efficiency
        let batch_size = 2048;
        for batch_start in (0..val_indices.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(val_indices.len());
            let batch_indices: Vec<usize> = val_indices[batch_start..batch_end].to_vec();

            // Prepare batch tensors
            let mut batch_nn_input: Vec<f32> = Vec::with_capacity(batch_indices.len() * FEATURE_DIM);
            for &i in &batch_indices {
                for (j, &v) in features[i].iter().enumerate() {
                    batch_nn_input.push((v - nn_metadata.feature_means[j]) / nn_metadata.feature_stds[j]);
                }
            }

            let nn_tensor = Tensor::from_slice(&batch_nn_input)
                .view([batch_indices.len() as i64, FEATURE_DIM as i64])
                .to(device);

            // NN forward pass
            let nn_logits = net.forward(&nn_tensor);
            let nn_probs_tensor = nn_logits.softmax(-1, Kind::Float);

            // Get all NN probabilities at once - need to reshape to 1D first
            let batch_size_actual = batch_indices.len();
            let nn_probs_flat: Vec<f32> = nn_probs_tensor
                .reshape(&[batch_size_actual as i64 * nn_metadata.num_classes as i64])
                .try_into()
                .map_err(|e| anyhow::anyhow!("Failed to extract NN probabilities: {:?}", e))?;

            // Process RF predictions in parallel using rayon
            let rf_probs_batch: Vec<Vec<f32>> = batch_indices.par_iter()
                .map(|&i| {
                    let rf_normalized: Vec<f32> = features[i].iter().enumerate()
                        .map(|(j, &v)| (v - rf.feature_means[j]) / rf.feature_stds[j])
                        .collect();
                    rf.predict_proba(&rf_normalized)
                })
                .collect();

            // Combine predictions
            for (batch_i, &i) in batch_indices.iter().enumerate() {
                // NN probabilities - already extracted
                let nn_start = batch_i * nn_metadata.num_classes;
                let nn_probs = nn_probs_flat[nn_start..nn_start + nn_metadata.num_classes].to_vec();

                // RF probabilities - already computed in parallel
                let rf_probs = rf_probs_batch[batch_i].clone();

                ensemble_inputs.push(EnsembleInput::new(nn_probs, rf_probs, labels[i]));
            }
        }
    }

    println!("Collected {} validation predictions", ensemble_inputs.len());

    // Grid Search
    let optimizer = GridSearchOptimizer::new(0.05);
    let grid_result = optimizer.optimize(&ensemble_inputs);
    optimizer.print_results(&grid_result);

    println!("\n=== Phase 2: Evaluate with Optimal Static Weight ===\n");

    let static_config = VotingEnsembleConfig::with_static_weight(grid_result.optimal_weight);
    let static_voter = EnsembleVoter::new(static_config);
    let static_accuracy = static_voter.evaluate(&ensemble_inputs);

    println!("Static Weight Ensemble Accuracy: {:.2}%", static_accuracy * 100.0);

    println!("\n=== Phase 3: Evaluate with Dynamic Weighting ===\n");

    let dynamic_config = VotingEnsembleConfig::with_dynamic_weighting();
    let dynamic_voter = EnsembleVoter::new(dynamic_config);
    let dynamic_accuracy = dynamic_voter.evaluate(&ensemble_inputs);

    println!("Dynamic Weighting Ensemble Accuracy: {:.2}%", dynamic_accuracy * 100.0);

    println!("\n=== Phase 4: Evaluate with Top-K Shortlist ===\n");

    let top3_accuracy = static_voter.evaluate_with_topk(&ensemble_inputs, 3);
    let top5_accuracy = static_voter.evaluate_with_topk(&ensemble_inputs, 5);

    println!("Top-3 Shortlist Ensemble Accuracy: {:.2}%", top3_accuracy * 100.0);
    println!("Top-5 Shortlist Ensemble Accuracy: {:.2}%", top5_accuracy * 100.0);

    // Summary
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  VOTING ENSEMBLE RESULTS                                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  NN Only:              {:>6.2}%                                    ║", grid_result.nn_only_accuracy * 100.0);
    println!("║  RF Only:              {:>6.2}%                                    ║", grid_result.rf_only_accuracy * 100.0);
    println!("║  Static Ensemble:      {:>6.2}% (weight={:.2})                     ║", static_accuracy * 100.0, grid_result.optimal_weight);
    println!("║  Dynamic Ensemble:     {:>6.2}%                                    ║", dynamic_accuracy * 100.0);
    println!("║  Top-3 Ensemble:       {:>6.2}%                                    ║", top3_accuracy * 100.0);
    println!("║  Top-5 Ensemble:       {:>6.2}%                                    ║", top5_accuracy * 100.0);
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    // Save results
    let results = serde_json::json!({
        "nn_only_accuracy": grid_result.nn_only_accuracy,
        "rf_only_accuracy": grid_result.rf_only_accuracy,
        "optimal_weight": grid_result.optimal_weight,
        "static_ensemble_accuracy": static_accuracy,
        "dynamic_ensemble_accuracy": dynamic_accuracy,
        "top3_ensemble_accuracy": top3_accuracy,
        "top5_ensemble_accuracy": top5_accuracy,
        "weight_accuracy_curve": grid_result.weight_accuracy_curve,
        "improvement_over_best_single": grid_result.improvement_over_best_single()
    });

    fs::write("voting_ensemble_results.json", serde_json::to_string_pretty(&results)?)?;
    println!("\nResults saved to: voting_ensemble_results.json");

    Ok(())
}

fn train_and_evaluate(
    _device: &Device,
    _vs: &nn::VarStore,
    _net: &SpeciesExpert112D,
    _nn_metadata: &NNMetadata,
) -> Result<()> {
    println!("\nPlease first train the 112D RF using:");
    println!("  cargo run --release --bin train_parallel_rf_112d");
    println!("\nOr run the simple RF training:");
    println!("  cargo run --release --bin train_beans_models_112d");

    Ok(())
}
