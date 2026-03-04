//! Evaluate Voting Ensemble by Dataset Component
//! ==============================================
//!
//! Uses Static Ensemble with optimal weight=0.05 (95% RF + 5% NN)
//! Evaluates on specified BEANS-Zero dataset components
//!
//! Usage:
//!   export LIBTORCH=/path/to/libtorch
//!   export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
//!   cargo run --release --bin eval_voting_ensemble_by_dataset --features gpu-training

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;

use technical_architecture::taxonomic_router::FEATURE_DIM;

#[cfg(feature = "gpu-training")]
use tch::{nn, nn::Module, Device, Tensor, Kind};

// =============================================================================
// Configuration
// =============================================================================

/// Optimal NN weight from grid search
const OPTIMAL_NN_WEIGHT: f64 = 0.05;

/// Dataset components to evaluate
const EVAL_COMPONENTS: &[&str] = &[
    "esc50",
    "watkins",
    "cbi",
    "humbugdb",
    "unseen-species",
    "unseen-genus",
    "unseen-family",
    "lifestage",
    "call-type",
    "zf-indv",
];

// =============================================================================
// RF Structures (112D species-level RF)
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

    fn predict(&self, features: &[f32]) -> usize {
        let mut votes = vec![0usize; self.n_classes];
        for tree in &self.trees {
            let pred = tree.predict(features);
            votes[pred] += 1;
        }
        votes.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0)
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

// =============================================================================
// Dataset Component Mapping
// =============================================================================

fn get_component_name(sample: &BeansSample) -> String {
    let task_lower = sample.labels.task.to_lowercase();
    let _output_lower = sample.labels.output.to_lowercase();

    // Specific dataset mappings
    if task_lower.contains("esc50") || task_lower == "esc50" {
        return "esc50".to_string();
    }
    if task_lower.contains("watkins") || task_lower == "watkins" {
        return "watkins".to_string();
    }
    if task_lower.contains("cbi") || task_lower == "cbi" {
        return "cbi".to_string();
    }
    if task_lower.contains("humbugdb") || task_lower == "humbugdb" {
        return "humbugdb".to_string();
    }
    if task_lower.contains("unseen-species") || task_lower == "unseen-species" {
        return "unseen-species".to_string();
    }
    if task_lower.contains("unseen-genus") || task_lower == "unseen-genus" {
        return "unseen-genus".to_string();
    }
    if task_lower.contains("unseen-family") || task_lower == "unseen-family" {
        return "unseen-family".to_string();
    }
    if task_lower.contains("lifestage") || task_lower == "lifestage" {
        return "lifestage".to_string();
    }
    if task_lower.contains("call-type") || task_lower == "call-type" {
        return "call-type".to_string();
    }
    if task_lower.contains("zf-indv") || task_lower == "zf-indv" {
        return "zf-indv".to_string();
    }

    // Default: use task name
    format!("task_{}", sample.labels.task)
}

// =============================================================================
// Metrics
// =============================================================================

#[derive(Debug, Serialize)]
struct ComponentMetrics {
    Accuracy: f64,
    Precision: f64,
    Recall: f64,
    #[serde(rename = "F1 Score")]
    F1_Score: f64,
    #[serde(rename = "Top-1 Accuracy")]
    Top1_Accuracy: f64,
}

impl ComponentMetrics {
    fn compute(predictions: &[usize], labels: &[usize], num_classes: usize) -> Self {
        let n = predictions.len();
        if n == 0 {
            return Self {
                Accuracy: 0.0,
                Precision: 0.0,
                Recall: 0.0,
                F1_Score: 0.0,
                Top1_Accuracy: 0.0,
            };
        }

        // Accuracy
        let correct = predictions.iter().zip(labels.iter()).filter(|(p, l)| p == l).count();
        let accuracy = correct as f64 / n as f64;

        // Macro-averaged precision and recall
        let mut precision_sum = 0.0;
        let mut recall_sum = 0.0;
        let mut valid_classes = 0;

        for c in 0..num_classes {
            let tp = predictions.iter().zip(labels.iter())
                .filter(|(p, l)| **p == c && **l == c).count();
            let fp = predictions.iter().zip(labels.iter())
                .filter(|(p, l)| **p == c && **l != c).count();
            let fn_ = predictions.iter().zip(labels.iter())
                .filter(|(p, l)| **p != c && **l == c).count();

            let class_precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
            let class_recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };

            let class_count = labels.iter().filter(|&&l| l == c).count();
            if class_count > 0 {
                precision_sum += class_precision;
                recall_sum += class_recall;
                valid_classes += 1;
            }
        }

        let precision = if valid_classes > 0 { precision_sum / valid_classes as f64 } else { 0.0 };
        let recall = if valid_classes > 0 { recall_sum / valid_classes as f64 } else { 0.0 };
        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Self {
            Accuracy: accuracy,
            Precision: precision,
            Recall: recall,
            F1_Score: f1_score,
            Top1_Accuracy: accuracy, // Same as accuracy for single prediction
        }
    }
}

// =============================================================================
// Ensemble Prediction
// =============================================================================

fn ensemble_predict(nn_probs: &[f32], rf_probs: &[f32], nn_weight: f64) -> usize {
    let rf_weight = 1.0 - nn_weight;

    let mut best_class = 0;
    let mut best_score = f64::NEG_INFINITY;

    for (c, (nn_p, rf_p)) in nn_probs.iter().zip(rf_probs.iter()).enumerate() {
        let ensemble_score = (*nn_p as f64 * nn_weight) + (*rf_p as f64 * rf_weight);
        if ensemble_score > best_score {
            best_score = ensemble_score;
            best_class = c;
        }
    }

    best_class
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Voting Ensemble Evaluation by Dataset Component                  ║");
    println!("║  Static Weight: {}% NN + {}% RF                                    ║",
        OPTIMAL_NN_WEIGHT * 100.0, (1.0 - OPTIMAL_NN_WEIGHT) * 100.0);
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

    // Build idx_to_label
    let mut idx_to_label: HashMap<usize, String> = HashMap::new();
    for (label, &idx) in &nn_metadata.label_to_idx {
        idx_to_label.insert(idx, label.clone());
    }

    // Load RF
    println!("\nLoading Species RF (112D)...");
    let rf_json = fs::read_to_string("random_forest_model_112d_parallel.json")?;
    let rf: RandomForest112D = serde_json::from_str(&rf_json)?;
    println!("  Trees: {}", rf.trees.len());
    println!("  Classes: {}", rf.n_classes);

    // Load data
    println!("\nLoading BEANS-Zero dataset...");
    let manifest: BeansManifest = serde_json::from_str(&fs::read_to_string("beans_zero_full_manifest.json")?)?;
    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_manifest: CacheManifest = serde_json::from_str(&fs::read_to_string(cache_dir.join("cache_manifest.json"))?)?;

    // Organize by component
    let mut component_data: HashMap<String, Vec<(Vec<f32>, usize)>> = HashMap::new();

    for sample in &manifest.samples {
        let label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };

        let label_idx = *nn_metadata.label_to_idx.get(&label).unwrap_or(&0);
        let component = get_component_name(sample);

        if let Some(cache_file) = cache_manifest.entries.get(&sample.audio_file) {
            let path = cache_dir.join(cache_file);
            if path.exists() {
                if let Ok(file) = fs::File::open(&path) {
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(BufReader::new(file)) {
                        if features.len() == FEATURE_DIM {
                            component_data.entry(component).or_default().push((features, label_idx));
                        }
                    }
                }
            }
        }
    }

    // Filter to only requested components
    let eval_components: Vec<String> = EVAL_COMPONENTS.iter().map(|s| s.to_string()).collect();

    println!("\nDataset Components to Evaluate:");
    for comp in &eval_components {
        let n = component_data.get(comp).map(|v| v.len()).unwrap_or(0);
        println!("  {}: {} samples", comp, n);
    }

    // Evaluate each component
    println!("\n=== Evaluating Components ===\n");

    let mut results: HashMap<String, ComponentMetrics> = HashMap::new();
    let batch_size = 2048;

    for component in &eval_components {
        let data = match component_data.get(component) {
            Some(d) if !d.is_empty() => d,
            _ => {
                println!("[{}]: No samples found, skipping", component);
                continue;
            }
        };

        println!("[{}]: {} samples", component, data.len());

        // Use 20% for validation (same as training split)
        let n_val = (data.len() as f32 * 0.2) as usize;
        let val_data: Vec<_> = data.iter().take(n_val).collect();

        if val_data.is_empty() {
            println!("  Skipping (no validation samples)");
            continue;
        }

        let mut predictions: Vec<usize> = Vec::new();
        let mut labels: Vec<usize> = Vec::new();

        // Process in batches
        for batch_start in (0..val_data.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(val_data.len());
            let batch: Vec<_> = val_data[batch_start..batch_end].to_vec();
            let batch_len = batch.len();

            // Prepare NN input
            let mut nn_input: Vec<f32> = Vec::with_capacity(batch_len * FEATURE_DIM);
            for (features, _) in &batch {
                for (j, &v) in features.iter().enumerate() {
                    nn_input.push((v - nn_metadata.feature_means[j]) / nn_metadata.feature_stds[j]);
                }
            }

            // NN forward pass
            let nn_tensor = Tensor::from_slice(&nn_input)
                .view([batch_len as i64, FEATURE_DIM as i64])
                .to(device);
            let nn_logits = net.forward(&nn_tensor);
            let nn_probs_tensor = nn_logits.softmax(-1, Kind::Float);

            // Extract NN probabilities
            let nn_probs_flat: Vec<f32> = nn_probs_tensor
                .reshape([batch_len as i64 * nn_metadata.num_classes as i64])
                .try_into()
                .map_err(|e| anyhow::anyhow!("Failed to extract NN probs: {:?}", e))?;

            // Process each sample
            for (batch_i, (features, label)) in batch.iter().enumerate() {
                let label = *label; // Dereference the reference
                // NN probabilities
                let nn_start = batch_i * nn_metadata.num_classes;
                let nn_probs = &nn_probs_flat[nn_start..nn_start + nn_metadata.num_classes];

                // RF probabilities
                let rf_normalized: Vec<f32> = features.iter().enumerate()
                    .map(|(j, &v)| (v - rf.feature_means[j]) / rf.feature_stds[j])
                    .collect();
                let rf_probs = rf.predict_proba(&rf_normalized);

                // Ensemble prediction with optimal weight
                let pred = ensemble_predict(nn_probs, &rf_probs, OPTIMAL_NN_WEIGHT);

                predictions.push(pred);
                labels.push(label);
            }
        }

        // Compute metrics
        let metrics = ComponentMetrics::compute(&predictions, &labels, nn_metadata.num_classes);

        println!("  Accuracy:  {:.4}", metrics.Accuracy);
        println!("  Precision: {:.4}", metrics.Precision);
        println!("  Recall:    {:.4}", metrics.Recall);
        println!("  F1 Score:  {:.4}", metrics.F1_Score);
        println!();

        results.insert(component.clone(), metrics);
    }

    // Output final results
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  FINAL RESULTS (BEANS Format)                                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let json_output = serde_json::to_string_pretty(&results)?;
    println!("{}", json_output);

    // Save to file
    fs::write("voting_ensemble_by_dataset_results.json", &json_output)?;
    println!("\nResults saved to: voting_ensemble_by_dataset_results.json");

    Ok(())
}
