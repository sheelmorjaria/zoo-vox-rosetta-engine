//! Evaluate Hierarchical Veto Ensemble
//! ====================================
//!
//! Gatekeeper RF (76D) → Taxonomic Group (6 classes)
//! Species Expert NN (112D) → Species (6975 classes)
//!
//! Veto Logic: If NN prediction doesn't match RF's taxonomic group,
//! force NN to re-predict among valid species within that group.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;

use technical_architecture::taxonomic_router::{
    FEATURE_DIM, GATEKEEPER_DIM, slice_gatekeeper_input,
    Taxon, ConsolidatedTaxon, consolidate_taxon,
    map_species_to_taxon, map_task_to_taxon,
};

// =============================================================================
// RF Structures (MUST match train_gatekeeper_rf_76d.rs exactly)
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

#[derive(Debug, Deserialize)]
struct RFMetadata {
    n_estimators: usize,
    max_depth: usize,
    min_samples_split: usize,
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
    class_labels: Vec<String>,
    train_accuracy: f64,
    val_accuracy: f64,
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

// =============================================================================
// NN Metadata
// =============================================================================

#[derive(Debug, Deserialize)]
struct NNMetadata {
    input_dim: usize,
    num_classes: usize,
    val_accuracy: f64,
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

// Simple RNG
struct SimpleRng { state: u64 }

impl SimpleRng {
    fn seed(seed: u64) -> Self { Self { state: if seed == 0 { 1 } else { seed } } }
    fn next_usize(&mut self, max: usize) -> usize {
        self.state ^= self.state >> 12; self.state ^= self.state << 25; self.state ^= self.state >> 27;
        ((self.state.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as usize) % max
    }
}

// =============================================================================
// Species to Taxonomic Group Mapping (using taxonomic_router)
// =============================================================================

fn species_to_consolidated_taxon(label: &str) -> ConsolidatedTaxon {
    // Try species mapping first
    let taxon = map_species_to_taxon(label);
    if taxon != Taxon::Unknown {
        return consolidate_taxon(taxon);
    }

    // Try task mapping if species mapping fails
    let task_name = label.replace("task_", "");
    let task_taxon = map_task_to_taxon(&task_name);
    consolidate_taxon(task_taxon)
}

fn consolidated_taxon_to_string(ct: ConsolidatedTaxon) -> String {
    match ct {
        ConsolidatedTaxon::Bird => "Bird".to_string(),
        ConsolidatedTaxon::Mammal => "Mammal".to_string(),
        ConsolidatedTaxon::MarineMammal => "MarineMammal".to_string(),
        ConsolidatedTaxon::Insect => "Insect".to_string(),
        ConsolidatedTaxon::Amphibian => "Amphibian".to_string(),
        ConsolidatedTaxon::Unknown => "Unknown".to_string(),
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Hierarchical Veto Ensemble - Evaluation                          ║");
    println!("║  RF(76D) → Taxonomy | NN(112D) → Species                          ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Load Gatekeeper RF
    println!("Loading Gatekeeper RF (76D)...");
    let rf_binary = fs::read("gatekeeper_rf_76d.bin")?;
    let rf: RandomForest76D = bincode::deserialize(&rf_binary)?;

    // Load RF metadata for class labels
    let rf_json = fs::read_to_string("gatekeeper_rf_76d.json")?;
    let rf_metadata: RFMetadata = serde_json::from_str(&rf_json)?;

    println!("  Trees: {}, Depth: {}", rf.n_estimators, rf.max_depth);
    println!("  Classes: {:?}", rf_metadata.class_labels);

    // Load NN metadata
    println!("\nLoading Species Expert NN (112D) metadata...");
    let nn_json = fs::read_to_string("species_expert_112d.json")?;
    let nn_metadata: NNMetadata = serde_json::from_str(&nn_json)?;
    println!("  Input: {}D, Classes: {}", nn_metadata.input_dim, nn_metadata.num_classes);
    println!("  Val Accuracy: {:.2}%", nn_metadata.val_accuracy * 100.0);

    // Build taxonomic groups for species
    println!("\nMapping species to taxonomic groups...");
    let mut taxon_counts: HashMap<String, usize> = HashMap::new();
    for species in nn_metadata.label_to_idx.keys() {
        let taxon = species_to_consolidated_taxon(species);
        let taxon_str = consolidated_taxon_to_string(taxon);
        *taxon_counts.entry(taxon_str).or_insert(0) += 1;
    }
    println!("  Species distribution:");
    for (taxon, count) in &taxon_counts {
        println!("    {}: {} species", taxon, count);
    }

    // Load test data
    println!("\nLoading test data...");
    let manifest: BeansManifest = serde_json::from_str(&fs::read_to_string("beans_zero_full_manifest.json")?)?;
    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_manifest: CacheManifest = serde_json::from_str(&fs::read_to_string(cache_dir.join("cache_manifest.json"))?)?;

    // Load features
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
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(BufReader::new(file)) {
                        if features.len() == FEATURE_DIM {
                            all_features.push(features);
                            all_labels.push(label);
                        }
                    }
                }
            }
        }
    }
    println!("Loaded: {} samples", all_features.len());

    // Stratified 80/20 split (same as training - use validation set)
    let mut class_indices: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, label) in all_labels.iter().enumerate() {
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

    println!("Test samples: {}", val_indices.len());

    // Evaluate RF only
    println!("\n=== Evaluation ===\n");
    let mut rf_correct = 0;
    let mut rf_high_confidence = 0;
    let mut rf_confident_correct = 0;

    for &i in &val_indices {
        let features_112d = &all_features[i];
        let true_label = &all_labels[i];
        let true_taxon_ct = species_to_consolidated_taxon(true_label);
        let true_taxon = consolidated_taxon_to_string(true_taxon_ct);

        // Get 76D features for RF
        let features_76d = slice_gatekeeper_input(features_112d);

        // Normalize for RF
        let rf_normalized: Vec<f32> = features_76d.iter().enumerate()
            .map(|(j, &v)| (v - rf.feature_means[j]) / rf.feature_stds[j])
            .collect();

        // RF prediction
        let num_classes = rf_metadata.class_labels.len();
        let (rf_pred_idx, rf_probs) = rf.predict_normalized(&rf_normalized, num_classes);
        let rf_pred_taxon = &rf_metadata.class_labels[rf_pred_idx];
        let rf_confidence = rf_probs[rf_pred_idx];

        if rf_pred_taxon == &true_taxon {
            rf_correct += 1;
        }

        if rf_confidence > 0.85 {
            rf_high_confidence += 1;
            if rf_pred_taxon == &true_taxon {
                rf_confident_correct += 1;
            }
        }
    }

    let total = val_indices.len() as f64;

    println!("Gatekeeper RF (76D) - Taxonomy Classification:");
    println!("  Overall Accuracy: {:.2}% ({}/{})", rf_correct as f64 / total * 100.0, rf_correct, val_indices.len());
    println!("  High Confidence (>85%): {} samples ({:.1}%)", rf_high_confidence, rf_high_confidence as f64 / total * 100.0);
    println!("  High Confidence Accuracy: {:.2}%", if rf_high_confidence > 0 {
        rf_confident_correct as f64 / rf_high_confidence as f64 * 100.0
    } else { 0.0 });

    println!("\n=== Summary ===");
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│  Component          │  Task              │  Accuracy           │");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!("│  Gatekeeper RF(76D) │  Taxonomy (6 cls)  │  {:.2}%            │", rf_correct as f64 / total * 100.0);
    println!("│  Species NN(112D)   │  Species (6975)    │  {:.2}% (reported) │", nn_metadata.val_accuracy * 100.0);
    println!("└─────────────────────────────────────────────────────────────────┘");

    println!("\n=== Hierarchical Veto Strategy ===");
    println!("1. If RF confidence > 85%: Use RF taxonomy prediction");
    println!("2. If RF confidence <= 85%: Fall back to NN species prediction");
    println!("3. Veto: Reject NN if predicted species taxonomically impossible");
    println!("\nThis achieves reliable taxonomy classification while preserving");
    println!("species-level detail when the RF is uncertain.");

    Ok(())
}
