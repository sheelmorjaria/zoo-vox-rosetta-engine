//! Evaluate Hierarchical Veto Ensemble - JSON Output Format
//! ================================================================
//!
//! Evaluates the combined Taxonomy Gatekeeper RF + Species Expert NN ensemble.
//! Outputs results in JSON format broken down by BEANS-Zero taxonomic subsets.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;
use tch::{nn, nn::Module, Device, Tensor};

use technical_architecture::taxonomic_router::{
    Taxon, FEATURE_DIM, PHYSICS_DIM, TEXTURE_DIM,
    slice_physics, slice_texture, map_species_to_taxon, map_task_to_taxon,
};

// =============================================================================
// Configuration
// =============================================================================

const VETO_CONFIDENCE_THRESHOLD: f32 = 0.85;

// =============================================================================
// Output Structures
// =============================================================================

#[derive(Debug, Serialize)]
struct SubsetMetrics {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1_score: f64,
    top1_accuracy: f64,
    n_samples: usize,
    n_correct: usize,
}

#[derive(Debug, Serialize)]
struct EvaluationResults {
    beans_zero: HashMap<String, SubsetMetrics>,
    overall: OverallMetrics,
    comparison: ComparisonMetrics,
}

#[derive(Debug, Serialize)]
struct OverallMetrics {
    nn_species_accuracy: f64,
    rf_taxonomic_accuracy: f64,
    soft_veto_accuracy: f64,
    total_samples: usize,
}

#[derive(Debug, Serialize)]
struct ComparisonMetrics {
    nn_only_floor: f64,
    soft_veto_result: f64,
    net_improvement: f64,
    veto_improved: usize,
    veto_hurt: usize,
}

// =============================================================================
// Neural Network (66D Texture → Species)
// =============================================================================

const HIDDEN_DIM_1: i64 = 512;
const HIDDEN_DIM_2: i64 = 256;
const HIDDEN_DIM_3: i64 = 128;
const DROPOUT: f64 = 0.4;

#[derive(Debug)]
struct GlobalSpeciesExpert {
    fc1: nn::Linear,
    ln1: nn::LayerNorm,
    fc2: nn::Linear,
    ln2: nn::LayerNorm,
    fc3: nn::Linear,
    ln3: nn::LayerNorm,
    out: nn::Linear,
}

impl GlobalSpeciesExpert {
    fn new(vs: &nn::Path, num_classes: i64) -> GlobalSpeciesExpert {
        let fc1 = nn::linear(vs, TEXTURE_DIM as i64, HIDDEN_DIM_1, Default::default());
        let ln1 = nn::layer_norm(vs, vec![HIDDEN_DIM_1], Default::default());
        let fc2 = nn::linear(vs, HIDDEN_DIM_1, HIDDEN_DIM_2, Default::default());
        let ln2 = nn::layer_norm(vs, vec![HIDDEN_DIM_2], Default::default());
        let fc3 = nn::linear(vs, HIDDEN_DIM_2, HIDDEN_DIM_3, Default::default());
        let ln3 = nn::layer_norm(vs, vec![HIDDEN_DIM_3], Default::default());
        let out = nn::linear(vs, HIDDEN_DIM_3, num_classes, Default::default());
        GlobalSpeciesExpert { fc1, ln1, fc2, ln2, fc3, ln3, out }
    }
}

impl nn::Module for GlobalSpeciesExpert {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let x = xs.apply(&self.fc1).apply(&self.ln1).gelu("none").dropout(DROPOUT, false);
        let x = x.apply(&self.fc2).apply(&self.ln2).gelu("none").dropout(DROPOUT, false);
        let x = x.apply(&self.fc3).apply(&self.ln3).gelu("none").dropout(DROPOUT, false);
        x.apply(&self.out)
    }
}

// =============================================================================
// Random Forest (46D Physics → Taxonomy)
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
struct DecisionTree46D {
    nodes: Vec<TreeNode>,
    n_classes: usize,
    feature_dim: usize,
}

impl DecisionTree46D {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RandomForest46D {
    trees: Vec<DecisionTree46D>,
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
    n_classes: usize,
}

impl RandomForest46D {
    fn predict(&self, features: &[f32]) -> usize {
        if self.trees.is_empty() { return 0; }
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

    fn predict_proba(&self, features: &[f32]) -> Vec<f32> {
        if self.trees.is_empty() {
            return vec![0.0; self.n_classes];
        }
        let mut votes = vec![0usize; self.n_classes];
        for tree in &self.trees {
                let pred = tree.predict(features);
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
struct CacheManifest {
    entries: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct TaxonomyGatekeeperMetadata {
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
    taxon_labels: Vec<String>,
}

// =============================================================================
// Main Evaluation
// =============================================================================

fn main() -> Result<()> {
    let start = Instant::now();
    let device = Device::cuda_if_available();

    // Load models
    let rf_metadata: TaxonomyGatekeeperMetadata =
        serde_json::from_str(&fs::read_to_string("taxonomy_gatekeeper_rf.json")?)?;
    let rf: RandomForest46D = bincode::deserialize(&fs::read("taxonomy_gatekeeper_rf.bin")?)?;

    let vs = nn::VarStore::new(device);
    let manifest: BeansManifest = serde_json::from_str(&fs::read_to_string("beans_zero_full_manifest.json")?)?;

    // Build label mapping
    let mut unique_labels: Vec<String> = Vec::new();
    for sample in &manifest.samples {
        let label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };
        if !unique_labels.contains(&label) {
            unique_labels.push(label);
        }
    }
    unique_labels.sort();
    let n_classes = unique_labels.len() as i64;

    let mut label_to_idx = HashMap::new();
    for (idx, label) in unique_labels.iter().enumerate() {
        label_to_idx.insert(label.clone(), idx);
    }
    let idx_to_label: HashMap<usize, String> = unique_labels.iter()
        .enumerate().map(|(idx, label)| (idx, label.clone())).collect();

    // Build species → taxon mapping
    let mut species_to_taxon: HashMap<String, Taxon> = HashMap::new();
    for label in &unique_labels {
        let taxon = map_species_to_taxon(label);
        let taxon = if taxon == Taxon::Unknown { map_task_to_taxon(&label.replace("task_", "")) } else { taxon };
        species_to_taxon.insert(label.clone(), taxon);
    }

    // Load NN model
    let mut vs = nn::VarStore::new(device);
    let _net = GlobalSpeciesExpert::new(&vs.root(), n_classes);
    vs.load("global_species_expert.nn.ot")?;

    // Load features
    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_manifest: CacheManifest =
        serde_json::from_str(&fs::read_to_string(cache_dir.join("cache_manifest.json"))?)?;

    let mut all_features: Vec<Vec<f32>> = Vec::new();
    let mut all_labels: Vec<String> = Vec::new();

    for sample in &manifest.samples {
        let label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };
        if let Some(cache_file) = cache_manifest.entries.get(&sample.audio_file) {
            let path = cache_dir.join(cache_file);
            if path.exists() {
                if let Ok(file) = fs::File::open(&path) {
                    let reader = BufReader::new(file);
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                        if features.len() == FEATURE_DIM {
                            all_features.push(features);
                            all_labels.push(label);
                        }
                    }
                }
            }
        }
    }

    // Split data (same seed as training)
    let n_samples = all_features.len();
    let n_train = (n_samples as f32 * 0.9) as usize;
    let mut indices: Vec<usize> = (0..n_samples).collect();
    for i in 0..indices.len() {
        let j = (rand_u32() as usize) % indices.len();
        indices.swap(i, j);
    }
    let test_indices: Vec<usize> = indices[n_train..].to_vec();

    // Load/create normalization params
    let texture_means: Vec<f32> = if Path::new("global_species_expert_means.bin").exists() {
        bincode::deserialize(&fs::read("global_species_expert_means.bin")?)?
    } else {
        let mut means = vec![0.0f32; TEXTURE_DIM];
        for &i in &indices[..n_train] {
            let texture = slice_texture(&all_features[i]);
            for (j, &v) in texture.iter().enumerate() { means[j] += v; }
        }
        for m in &mut means { *m /= n_train as f32; }
        fs::write("global_species_expert_means.bin", bincode::serialize(&means)?)?;
        means
    };

    let texture_stds: Vec<f32> = if Path::new("global_species_expert_stds.bin").exists() {
        bincode::deserialize(&fs::read("global_species_expert_stds.bin")?)?
    } else {
        let mut stds = vec![0.0f32; TEXTURE_DIM];
        for &i in &indices[..n_train] {
            let texture = slice_texture(&all_features[i]);
            for (j, &v) in texture.iter().enumerate() {
                stds[j] += (v - texture_means[j]).powi(2);
            }
        }
        for s in &mut stds { *s = (*s / n_train as f32).sqrt().max(1e-8); }
        fs::write("global_species_expert_stds.bin", bincode::serialize(&stds)?)?;
        stds
    };

    // Create test tensor
    let test_size = test_indices.len();
    let test_data: Vec<f32> = test_indices.iter()
        .flat_map(|&i| {
            let texture = slice_texture(&all_features[i]);
            texture.iter().enumerate()
                .map(|(j, &v)| (v - texture_means[j]) / texture_stds[j])
                .collect::<Vec<_>>()
        }).collect();

    let test_x = Tensor::from_slice(&test_data).view([test_size as i64, TEXTURE_DIM as i64]).to(device);
    let test_y: Vec<usize> = test_indices.iter().map(|&i| label_to_idx[&all_labels[i]]).collect();

    // Load NN and evaluate
    let mut vs_eval = nn::VarStore::new(device);
    let net = GlobalSpeciesExpert::new(&vs_eval.root(), n_classes);
    vs_eval.load("global_species_expert.nn.ot")?;

    let logits = net.forward(&test_x);
    let predictions = logits.argmax(-1, false);
    let preds_vec: Vec<i64> = predictions.iter::<i64>()?.collect();

    // Track per-taxon metrics
    let mut subset_metrics: HashMap<String, SubsetMetrics> = HashMap::new();
    let mut subset_tp: HashMap<String, usize> = HashMap::new();  // true positives
    let mut subset_fp: HashMap<String, usize> = HashMap::new();  // false positives
    let mut subset_fn: HashMap<String, usize> = HashMap::new();  // false negatives
    let mut subset_total: HashMap<String, usize> = HashMap::new();
    let mut subset_correct: HashMap<String, usize> = HashMap::new();

    let mut nn_correct = 0usize;
    let mut rf_correct = 0usize;
    let mut soft_veto_correct = 0usize;
    let mut veto_improved = 0usize;
    let mut veto_hurt = 0usize;

    // Get NN probabilities
    let probs = logits.softmax(-1, tch::Kind::Float).to_kind(tch::Kind::Double);
    let probs_data: Vec<f64> = probs.view([-1]).iter::<f64>()?.collect();

    for (sample_idx, &data_idx) in test_indices.iter().enumerate() {
        let true_label = &all_labels[data_idx];
        let true_species_idx = label_to_idx[true_label];
        let true_taxon = species_to_taxon[true_label];
        let taxon_name = format!("{:?}", true_taxon);

        // RF prediction
        let physics = slice_physics(&all_features[data_idx]);
        let mut physics_norm = vec![0.0f32; PHYSICS_DIM];
        for j in 0..PHYSICS_DIM {
            physics_norm[j] = (physics[j] - rf_metadata.feature_means[j]) / rf_metadata.feature_stds[j];
        }
        let pred_taxon_idx = rf.predict(&physics_norm);
        let rf_confidence = rf.predict_proba(&physics_norm)[pred_taxon_idx];
        let pred_taxon = parse_taxon(&rf_metadata.taxon_labels[pred_taxon_idx]);

        // NN prediction
        let nn_pred_idx = preds_vec[sample_idx] as usize;
        let nn_pred_label = &idx_to_label[&nn_pred_idx];

        // Track NN accuracy
        if nn_pred_idx == true_species_idx { nn_correct += 1; }

        // Track RF accuracy
        if pred_taxon == true_taxon { rf_correct += 1; }

        // Soft veto: only override when RF is confident
        let final_pred = if rf_confidence > VETO_CONFIDENCE_THRESHOLD {
            // Find first NN candidate matching RF's taxon
            let mut species_probs: Vec<(usize, f32)> = (0..n_classes as usize)
                .map(|i| (i, probs_data[sample_idx * n_classes as usize + i] as f32))
                .collect();
            species_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut found = nn_pred_idx;
            for (idx, _) in species_probs.iter().take(10) {
                let label = &idx_to_label[idx];
                if species_to_taxon[label] == pred_taxon {
                    found = *idx;
                    break;
                }
            }
            found
        } else {
            nn_pred_idx
        };

        // Track soft veto accuracy
        if final_pred == true_species_idx {
            soft_veto_correct += 1;
            if final_pred != nn_pred_idx { veto_improved += 1; }
        } else if final_pred != nn_pred_idx && nn_pred_idx == true_species_idx {
            veto_hurt += 1;
        }

        // Track per-taxon metrics for NN
        *subset_total.entry(taxon_name.clone()).or_insert(0) += 1;
        if nn_pred_idx == true_species_idx {
            *subset_correct.entry(taxon_name.clone()).or_insert(0) += 1;
            *subset_tp.entry(taxon_name.clone()).or_insert(0) += 1;
        } else {
            *subset_fn.entry(taxon_name.clone()).or_insert(0) += 1;
            let pred_taxon_name = format!("{:?}", species_to_taxon[nn_pred_label]);
            *subset_fp.entry(pred_taxon_name).or_insert(0) += 1;
        }
    }

    // Compute per-subset metrics
    for (taxon, &total) in subset_total.iter() {
        let correct = *subset_correct.get(taxon).unwrap_or(&0);
        let tp = *subset_tp.get(taxon).unwrap_or(&0);
        let fp = *subset_fp.get(taxon).unwrap_or(&0);
        let fn_count = *subset_fn.get(taxon).unwrap_or(&0);

        let accuracy = correct as f64 / total as f64;
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };

        subset_metrics.insert(taxon.clone(), SubsetMetrics {
            accuracy,
            precision,
            recall,
            f1_score: f1,
            top1_accuracy: accuracy,
            n_samples: total,
            n_correct: correct,
        });
    }

    // Build final results
    let results = EvaluationResults {
        beans_zero: subset_metrics,
        overall: OverallMetrics {
            nn_species_accuracy: nn_correct as f64 / test_size as f64,
            rf_taxonomic_accuracy: rf_correct as f64 / test_size as f64,
            soft_veto_accuracy: soft_veto_correct as f64 / test_size as f64,
            total_samples: test_size,
        },
        comparison: ComparisonMetrics {
            nn_only_floor: nn_correct as f64 / test_size as f64,
            soft_veto_result: soft_veto_correct as f64 / test_size as f64,
            net_improvement: (soft_veto_correct as f64 - nn_correct as f64) / test_size as f64,
            veto_improved,
            veto_hurt,
        },
    };

    // Output JSON
    println!("{}", serde_json::to_string_pretty(&results)?);

    // Also print summary to stderr for human reading
    eprintln!("\n=== Summary ===");
    eprintln!("NN Species Accuracy: {:.2}%", results.overall.nn_species_accuracy * 100.0);
    eprintln!("RF Taxonomic Accuracy: {:.2}%", results.overall.rf_taxonomic_accuracy * 100.0);
    eprintln!("Soft Veto Accuracy: {:.2}%", results.overall.soft_veto_accuracy * 100.0);
    eprintln!("Net Improvement: {:+.2}%", results.comparison.net_improvement * 100.0);
    eprintln!("Time: {:.1}s", start.elapsed().as_secs_f32());

    Ok(())
}

fn parse_taxon(s: &str) -> Taxon {
    match s.trim() {
        "Songbird" => Taxon::Songbird,
        "Mammal" => Taxon::Mammal,
        "Cetacean" => Taxon::Cetacean,
        "Mysticete" => Taxon::Mysticete,
        "NonPasserine" => Taxon::NonPasserine,
        "Amphibian" => Taxon::Amphibian,
        "Insect" => Taxon::Insect,
        "Pinniped" => Taxon::Pinniped,
        _ => Taxon::Unknown,
    }
}

fn rand_u32() -> u32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x853c49e6748fea9b);
    let mut s = STATE.load(Ordering::Relaxed);
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    STATE.store(s, Ordering::Relaxed);
    (s.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
}
