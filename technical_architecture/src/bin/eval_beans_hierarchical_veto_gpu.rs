//! BEANS-Zero Evaluation with Hierarchical Veto Ensemble (GPU)
//! =============================================================
//!
//! Full evaluation with actual NN inference using tch-rs.
//! Organizes metrics by dataset component (task).
//! Requires LIBTORCH and LD_LIBRARY_PATH to be set.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;

use technical_architecture::taxonomic_router::{
    FEATURE_DIM, GATEKEEPER_DIM, slice_gatekeeper_input,
    Taxon, ConsolidatedTaxon, consolidate_taxon,
    map_species_to_taxon, map_task_to_taxon, consolidated_taxon_to_idx,
};

use tch::{nn, nn::Module, Device, Tensor};

// Dataset component name mapping
fn get_component_name(task: &str, output: &str) -> String {
    // BEANS-Zero uses task IDs, map to readable names
    let task_lower = task.to_lowercase();

    // Common BEANS dataset names
    if task_lower.contains("bird") || output.to_lowercase().contains("bird") {
        return "bird_species".to_string();
    }
    if task_lower.contains("bat") || output.to_lowercase().contains("eptesicus") ||
       output.to_lowercase().contains("myotis") || output.to_lowercase().contains("bat") {
        return "bat_species".to_string();
    }
    if task_lower.contains("marine") || task_lower.contains("dolphin") ||
       output.to_lowercase().contains("dolphin") || output.to_lowercase().contains("whale") {
        return "marine_mammals".to_string();
    }
    if task_lower.contains("insect") || output.to_lowercase().contains("bee") ||
       output.to_lowercase().contains("mosquito") {
        return "insects".to_string();
    }
    if task_lower.contains("amphibian") || output.to_lowercase().contains("frog") ||
       output.to_lowercase().contains("toad") {
        return "amphibians".to_string();
    }
    if task_lower.contains("marmoset") || output.to_lowercase().contains("marmoset") {
        return "marmoset".to_string();
    }

    // Default: use task ID
    format!("task_{}", task)
}

// =============================================================================
// RF Structures
// =============================================================================

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
    fn predict(&self, features: &[f32]) -> usize {
        if self.nodes.is_empty() { return 0; }
        self.predict_node(0, features)
    }

    fn predict_node(&self, node_idx: usize, features: &[f32]) -> usize {
        let node = &self.nodes[node_idx];
        if let Some(pred) = node.prediction {
            return pred;
        }
        let feat_idx = node.feature_idx.unwrap();
        let thresh = node.threshold.unwrap();
        if features[feat_idx] <= thresh {
            self.predict_node(node.left.unwrap(), features)
        } else {
            self.predict_node(node.right.unwrap(), features)
        }
    }
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
    fn predict_normalized(&self, features: &[f32], num_classes: usize) -> (usize, Vec<f32>) {
        let mut votes = vec![0usize; num_classes];
        for tree in &self.trees {
            let pred = tree.predict(features);
            votes[pred] += 1;
        }
        let total = self.trees.len() as f32;
        let probs: Vec<f32> = votes.iter().map(|&v| v as f32 / total).collect();
        let mut max_idx = 0;
        let mut max_val = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            if p > max_val {
                max_val = p;
                max_idx = i;
            }
        }
        (max_idx, probs)
    }
}

#[derive(Debug, Deserialize)]
struct RFMetadata {
    class_labels: Vec<String>,
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
// Metrics
// =============================================================================

#[derive(Debug, Serialize)]
struct EvaluationMetrics {
    Accuracy: f64,
    Precision: f64,
    Recall: f64,
    #[serde(rename = "F1 Score")]
    F1_Score: f64,
    #[serde(rename = "Top-1 Accuracy")]
    Top1_Accuracy: f64,
}

impl EvaluationMetrics {
    fn compute(predictions: &[usize], labels: &[usize], num_classes: usize) -> Self {
        let n = predictions.len();
        if n == 0 {
            return Self { Accuracy: 0.0, Precision: 0.0, Recall: 0.0, F1_Score: 0.0, Top1_Accuracy: 0.0 };
        }

        let correct = predictions.iter().zip(labels.iter()).filter(|(p, l)| p == l).count();
        let accuracy = correct as f64 / n as f64;

        let mut precision_sum = 0.0;
        let mut recall_sum = 0.0;
        let mut valid_classes = 0;

        for c in 0..num_classes {
            let tp = predictions.iter().zip(labels.iter()).filter(|(p, l)| **p == c && **l == c).count();
            let fp = predictions.iter().zip(labels.iter()).filter(|(p, l)| **p == c && **l != c).count();
            let fn_ = predictions.iter().zip(labels.iter()).filter(|(p, l)| **p != c && **l == c).count();

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
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };

        Self { Accuracy: accuracy, Precision: precision, Recall: recall, F1_Score: f1, Top1_Accuracy: accuracy }
    }
}

// =============================================================================
// Species to Taxonomic Group Mapping
// =============================================================================

fn species_to_consolidated_taxon(label: &str) -> ConsolidatedTaxon {
    let taxon = map_species_to_taxon(label);
    if taxon != Taxon::Unknown {
        return consolidate_taxon(taxon);
    }
    let task_name = label.replace("task_", "");
    consolidate_taxon(map_task_to_taxon(&task_name))
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  BEANS-Zero Hierarchical Veto Ensemble (GPU)                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let confidence_threshold = 0.85;
    let device = Device::cuda_if_available();
    println!("Device: {:?}", device);
    println!("RF Confidence Threshold: {}%\n", confidence_threshold * 100.0);

    // Load RF
    println!("Loading Gatekeeper RF (76D)...");
    let rf: RandomForest76D = bincode::deserialize(&fs::read("gatekeeper_rf_76d.bin")?)?;
    let rf_metadata: RFMetadata = serde_json::from_str(&fs::read_to_string("gatekeeper_rf_76d.json")?)?;
    println!("  Trees: {}, Classes: {:?}", rf.n_estimators, rf_metadata.class_labels);

    // Load NN
    println!("\nLoading Species Expert NN (112D)...");
    let nn_metadata: NNMetadata = serde_json::from_str(&fs::read_to_string("species_expert_112d.json")?)?;
    println!("  Classes: {}", nn_metadata.num_classes);

    // Build idx_to_label
    let mut idx_to_label: HashMap<usize, String> = HashMap::new();
    for (label, &idx) in &nn_metadata.label_to_idx {
        idx_to_label.insert(idx, label.clone());
    }

    // Load NN model
    let mut vs = nn::VarStore::new(device);
    let net = SpeciesExpert112D::new(&vs.root(), nn_metadata.num_classes as i64);
    vs.load("species_expert_112d.ot")?;
    println!("  Model loaded: species_expert_112d.ot");

    // Load data
    println!("\nLoading BEANS-Zero dataset...");
    let manifest: BeansManifest = serde_json::from_str(&fs::read_to_string("beans_zero_full_manifest.json")?)?;
    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_manifest: CacheManifest = serde_json::from_str(&fs::read_to_string(cache_dir.join("cache_manifest.json"))?)?;

    // Store features by component
    let mut component_features: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut component_labels: HashMap<String, Vec<String>> = HashMap::new();

    for sample in &manifest.samples {
        let label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };

        // Determine component name
        let component = get_component_name(&sample.labels.task, &sample.labels.output);

        if let Some(cache_file) = cache_manifest.entries.get(&sample.audio_file) {
            let path = cache_dir.join(cache_file);
            if path.exists() {
                if let Ok(file) = fs::File::open(&path) {
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(BufReader::new(file)) {
                        if features.len() == FEATURE_DIM {
                            component_features.entry(component.clone()).or_default().push(features);
                            component_labels.entry(component).or_default().push(label);
                        }
                    }
                }
            }
        }
    }

    // Print component summary
    println!("\nDataset Components:");
    let mut components: Vec<String> = component_features.keys().cloned().collect();
    components.sort();
    for comp in &components {
        let n = component_features.get(comp).map(|v| v.len()).unwrap_or(0);
        println!("  {}: {} samples", comp, n);
    }

    let num_taxon_classes = rf_metadata.class_labels.len();

    // Evaluate each component
    println!("\n=== Evaluating by Component ===");

    let mut all_component_results: HashMap<String, serde_json::Value> = HashMap::new();

    for component in &components {
        let features = component_features.get(component).unwrap();
        let labels = component_labels.get(component).unwrap();

        println!("\n[{}]: {} samples", component, features.len());

        // Stratified 80/20 split
        let mut class_indices: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, label) in labels.iter().enumerate() {
            class_indices.entry(label.clone()).or_default().push(i);
        }

        let mut rng = SimpleRng::seed(42);
        let mut val_indices: Vec<usize> = Vec::new();

        for (_, mut indices) in class_indices {
            for i in 0..indices.len() {
                let j = rng.next_usize(indices.len());
                indices.swap(i, j);
            }
            let n_train = (indices.len() as f32 * 0.8) as usize;
            val_indices.extend(indices[n_train..].iter().copied());
        }

        if val_indices.is_empty() {
            println!("  Skipping (no validation samples)");
            continue;
        }

        let mut rf_only_preds: Vec<usize> = Vec::new();
        let mut veto_preds: Vec<usize> = Vec::new();
        let mut true_labels: Vec<usize> = Vec::new();
        let mut rf_used = 0;
        let mut nn_used = 0;

        // Process in batches
        let batch_size = 512;
        for batch_start in (0..val_indices.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(val_indices.len());
            let batch_indices: Vec<usize> = val_indices[batch_start..batch_end].to_vec();

            let mut rf_features_batch: Vec<Vec<f32>> = Vec::new();
            let mut nn_features_batch: Vec<Vec<f32>> = Vec::new();
            let mut batch_labels: Vec<usize> = Vec::new();

            for &i in &batch_indices {
                let features_112d = &features[i];
                let true_label = &labels[i];
                let true_taxon = species_to_consolidated_taxon(true_label);
                let true_idx = consolidated_taxon_to_idx(true_taxon);

                let features_76d = slice_gatekeeper_input(features_112d);
                let rf_normalized: Vec<f32> = features_76d.iter().enumerate()
                    .map(|(j, &v)| (v - rf.feature_means[j]) / rf.feature_stds[j])
                    .collect();

                let nn_normalized: Vec<f32> = features_112d.iter().enumerate()
                    .map(|(j, &v)| (v - nn_metadata.feature_means[j]) / nn_metadata.feature_stds[j])
                    .collect();

                rf_features_batch.push(rf_normalized);
                nn_features_batch.push(nn_normalized);
                batch_labels.push(true_idx);
            }

            let mut rf_preds_batch: Vec<usize> = Vec::new();
            let mut rf_conf_batch: Vec<f32> = Vec::new();

            for rf_feat in &rf_features_batch {
                let (pred, probs) = rf.predict_normalized(rf_feat, num_taxon_classes);
                rf_preds_batch.push(pred);
                rf_conf_batch.push(probs[pred]);
            }

            let nn_needed: Vec<usize> = rf_conf_batch.iter().enumerate()
                .filter(|(_, &conf)| conf <= confidence_threshold)
                .map(|(i, _)| i)
                .collect();

            let mut nn_species_preds: HashMap<usize, usize> = HashMap::new();
            if !nn_needed.is_empty() {
                let nn_input: Vec<f32> = nn_needed.iter()
                    .flat_map(|&i| nn_features_batch[i].clone())
                    .collect();

                let nn_tensor = Tensor::from_slice(&nn_input)
                    .view([nn_needed.len() as i64, FEATURE_DIM as i64])
                    .to(device);

                let nn_logits = net.forward(&nn_tensor);
                let nn_preds_tensor = nn_logits.argmax(-1, false);

                for (batch_i, &local_i) in nn_needed.iter().enumerate() {
                    let species_idx = nn_preds_tensor.get(batch_i as i64).int64_value(&[]) as usize;
                    nn_species_preds.insert(local_i, species_idx);
                }
            }

            for (local_i, (&rf_pred, &true_idx)) in rf_preds_batch.iter().zip(batch_labels.iter()).enumerate() {
                rf_only_preds.push(rf_pred);
                true_labels.push(true_idx);

                if rf_conf_batch[local_i] > confidence_threshold {
                    veto_preds.push(rf_pred);
                    rf_used += 1;
                } else {
                    if let Some(&species_idx) = nn_species_preds.get(&local_i) {
                        let species_name = idx_to_label.get(&species_idx).map(|s| s.as_str()).unwrap_or("Unknown");
                        let nn_taxon = species_to_consolidated_taxon(species_name);
                        veto_preds.push(consolidated_taxon_to_idx(nn_taxon));
                        nn_used += 1;
                    } else {
                        veto_preds.push(rf_pred);
                    }
                }
            }
        }

        let rf_metrics = EvaluationMetrics::compute(&rf_only_preds, &true_labels, num_taxon_classes);
        let veto_metrics = EvaluationMetrics::compute(&veto_preds, &true_labels, num_taxon_classes);

        println!("  RF Only:   Acc={:.4} F1={:.4}", rf_metrics.Accuracy, rf_metrics.F1_Score);
        println!("  Veto:      Acc={:.4} F1={:.4} (RF:{:.1}% NN:{:.1}%)",
            veto_metrics.Accuracy, veto_metrics.F1_Score,
            rf_used as f64 / val_indices.len() as f64 * 100.0,
            nn_used as f64 / val_indices.len() as f64 * 100.0);

        // Store results for this component
        all_component_results.insert(component.clone(), serde_json::json!({
            "Accuracy": veto_metrics.Accuracy,
            "Precision": veto_metrics.Precision,
            "Recall": veto_metrics.Recall,
            "F1 Score": veto_metrics.F1_Score,
            "Top-1 Accuracy": veto_metrics.Top1_Accuracy
        }));
    }

    // ==========================================================================
    // Output Final Results
    // ==========================================================================
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  FINAL RESULTS (BEANS Format)                                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // Convert to sorted JSON object
    let mut sorted_results = serde_json::Map::new();
    for comp in &components {
        if let Some(metrics) = all_component_results.get(comp) {
            sorted_results.insert(comp.clone(), metrics.clone());
        }
    }

    let results = serde_json::Value::Object(sorted_results);

    println!("{}", serde_json::to_string_pretty(&results)?);

    fs::write("beans_hierarchical_veto_results.json", serde_json::to_string_pretty(&results)?)?;
    println!("\nResults saved to: beans_hierarchical_veto_results.json");

    Ok(())
}
