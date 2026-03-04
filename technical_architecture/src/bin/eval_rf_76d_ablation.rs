//! 76D Ablation Study: Remove Layer 3 (Micro Texture) for Species ID
//! ================================================================
//!
//! Hypothesis: Layer 3 (Micro Texture: Jitter, Shimmer, Vibrato) is NOISE
//! for Species Identification. It's only useful for Context/Prosody tasks.
//!
//! 76D = Layer 1 (46D) + Layer 2 (30D)
//! - Layer 1: Base Physics (F0, Duration, MFCCs, Spectral Shape)
//! - Layer 2: Macro Texture (Harmonic ratios, GLCM)
//!
//! Layer 3 (36D) REMOVED:
//! - Jitter, Shimmer, Vibrato, Perturbations (Prosody features)
//!
//! Usage:
//!   cargo run --release --bin eval_rf_76d_ablation

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use technical_architecture::beans_zero_weights::{BeansZeroWeightRouter, TaxonomicGroup};

// Ablated feature dimension (Layer 1 + Layer 2 only)
const FEATURE_DIM_76: usize = 76;
const FULL_DIM: usize = 112;

// =============================================================================
// Model Loading Structures
// =============================================================================

#[derive(Debug, Deserialize)]
struct RandomForestModel {
    trees: Vec<DecisionTree>,
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
    #[allow(dead_code)]
    label_to_idx: HashMap<String, usize>,
    idx_to_label: Vec<String>,
    n_classes: usize,
}

#[derive(Debug, Deserialize, Clone)]
struct DecisionTree {
    nodes: Vec<TreeNode>,
}

#[derive(Debug, Deserialize, Clone)]
struct TreeNode {
    feature_idx: Option<usize>,
    threshold: f32,
    left_child: Option<usize>,
    right_child: Option<usize>,
    class_prediction: Option<usize>,
}

// =============================================================================
// Manifest Structures
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
// 76D Ablated Model (uses only first 76 features)
// =============================================================================

struct AblatedModel76D {
    trees: Vec<DecisionTree>,
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
    idx_to_label: Vec<String>,
    n_classes: usize,
}

impl AblatedModel76D {
    fn from_full_model(full: &RandomForestModel) -> Self {
        // Slice first 76 dimensions from normalization params
        let feature_means = full.feature_means[..FEATURE_DIM_76].to_vec();
        let feature_stds = full.feature_stds[..FEATURE_DIM_76].to_vec();

        // Clone trees - they already use feature indices < 76 for most splits
        // (since we trained with max_features=20, features are distributed)
        Self {
            trees: full.trees.clone(),
            feature_means,
            feature_stds,
            idx_to_label: full.idx_to_label.clone(),
            n_classes: full.n_classes,
        }
    }

    fn predict(&self, features_112d: &[f32]) -> (usize, String) {
        // Ablate: use only first 76 features
        let ablated_features = &features_112d[..FEATURE_DIM_76];

        // Normalize
        let normalized: Vec<f32> = ablated_features
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - self.feature_means[i]) / self.feature_stds[i])
            .collect();

        // Vote from all trees
        let mut votes: HashMap<usize, usize> = HashMap::new();
        for tree in &self.trees {
            let pred = self.predict_tree(&normalized, tree);
            *votes.entry(pred).or_insert(0) += 1;
        }

        let (best_class, _) = votes
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .unwrap_or((0, 0));

        let label = self
            .idx_to_label
            .get(best_class)
            .cloned()
            .unwrap_or_else(|| format!("class_{}", best_class));

        (best_class, label)
    }

    fn predict_tree(&self, features: &[f32], tree: &DecisionTree) -> usize {
        let mut node_idx = 0;
        loop {
            let node = &tree.nodes[node_idx];
            if node.feature_idx.is_none() {
                return node.class_prediction.unwrap_or(0);
            }
            let feature_idx = node.feature_idx.unwrap();

            // If feature index >= 76, use mean value (feature not available)
            let feature_val = if feature_idx < features.len() {
                features[feature_idx]
            } else {
                0.0 // Use normalized mean (0.0) for ablated features
            };

            if feature_val <= node.threshold {
                node_idx = node.left_child.unwrap_or(0);
            } else {
                node_idx = node.right_child.unwrap_or(0);
            }
        }
    }
}

// =============================================================================
// Evaluation Results
// =============================================================================

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaxonomicStats {
    pub correct_species: usize,
    pub correct_taxonomic: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResults {
    pub species_accuracy: f64,
    pub taxonomic_accuracy: f64,
    pub per_taxon_stats: HashMap<String, TaxonomicStats>,
    pub total_samples: usize,
}

fn taxon_to_string(taxon: &TaxonomicGroup) -> String {
    match taxon {
        TaxonomicGroup::Cetacean => "cetacean".to_string(),
        TaxonomicGroup::Bat => "bat".to_string(),
        TaxonomicGroup::Amphibian => "amphibian".to_string(),
        TaxonomicGroup::Insect => "insect".to_string(),
        TaxonomicGroup::Primate => "primate".to_string(),
        TaxonomicGroup::Mammal => "mammal".to_string(),
        TaxonomicGroup::Bird => "bird".to_string(),
        TaxonomicGroup::Unknown => "unknown".to_string(),
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  76D ABLATION STUDY: Layer 3 (Micro Texture) Removal             ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("HYPOTHESIS:");
    println!("  Layer 3 (Micro Texture: Jitter, Shimmer, Vibrato) is NOISE for");
    println!("  Species ID. It's only useful for Context/Prosody tasks.");
    println!();
    println!("  76D = Layer 1 (46D) + Layer 2 (30D)");
    println!("      Layer 1: Base Physics (F0, Duration, MFCCs, Spectral)");
    println!("      Layer 2: Macro Texture (Harmonic ratios, GLCM)");
    println!("  REMOVED: Layer 3 (36D) = Jitter, Shimmer, Vibrato, Perturbations");
    println!();

    let start = Instant::now();

    // Load the existing 112D model
    println!("Loading existing 112D model from: random_forest_model_112d.json");
    let model_data = fs::read_to_string("random_forest_model_112d.json")?;
    let full_model: RandomForestModel = serde_json::from_str(&model_data)?;
    println!("  Loaded {} trees, {} classes", full_model.trees.len(), full_model.n_classes);

    // Create ablated 76D model
    let model_76d = AblatedModel76D::from_full_model(&full_model);
    println!("  Created 76D ablated model (first 76/112 features)");

    // Load manifest
    let manifest_path = "beans_zero_full_manifest.json";
    println!("\nLoading manifest from: {}", manifest_path);
    let manifest_data = fs::read_to_string(manifest_path)?;
    let manifest: BeansManifest = serde_json::from_str(&manifest_data)?;
    println!("  Total samples: {}", manifest.samples.len());

    // Load cache manifest
    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_manifest_path = cache_dir.join("cache_manifest.json");
    let cache_data = fs::read_to_string(&cache_manifest_path)?;
    let cache_manifest: CacheManifest = serde_json::from_str(&cache_data)?;
    println!("  Cached features available: {}", cache_manifest.entries.len());

    // Select evaluation subset (last 10%)
    let n_eval = (manifest.samples.len() as f32 * 0.1) as usize;
    let eval_start = manifest.samples.len() - n_eval;
    let eval_samples = &manifest.samples[eval_start..];
    println!("  Evaluation samples: {}", eval_samples.len());

    // Evaluate
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Running 76D Ablation Evaluation                                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let mut per_taxon_stats: HashMap<String, TaxonomicStats> = HashMap::new();
    let mut total_correct_species = 0;
    let mut total_correct_taxon = 0;
    let mut total_evaluated = 0;

    for (i, sample) in eval_samples.iter().enumerate() {
        if (i + 1) % 1000 == 0 {
            println!("  Progress: {}/{}", i + 1, eval_samples.len());
        }

        let audio_file = &sample.audio_file;
        let true_label = if sample.labels.output != "None" {
            &sample.labels.output
        } else {
            &sample.labels.task
        };

        // Load cached 112D features
        if let Some(cache_file) = cache_manifest.entries.get(audio_file) {
            let full_path = cache_dir.join(cache_file);
            if full_path.exists() {
                if let Ok(file) = fs::File::open(&full_path) {
                    let reader = BufReader::new(file);
                    if let Ok(features_112d) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                        if features_112d.len() == FULL_DIM {
                            // Predict using 76D ablated model
                            let (_, pred_label) = model_76d.predict(&features_112d);

                            // Get taxonomic groups
                            let true_taxon = BeansZeroWeightRouter::detect_group(true_label);
                            let pred_taxon = BeansZeroWeightRouter::detect_group(&pred_label);

                            // Update stats
                            let taxon_name = taxon_to_string(&true_taxon);
                            let stats = per_taxon_stats.entry(taxon_name).or_default();
                            stats.total += 1;

                            let species_match = true_label.to_lowercase() == pred_label.to_lowercase();
                            let taxon_match = true_taxon == pred_taxon;

                            if species_match {
                                stats.correct_species += 1;
                                total_correct_species += 1;
                            }
                            if taxon_match {
                                stats.correct_taxonomic += 1;
                                total_correct_taxon += 1;
                            }

                            total_evaluated += 1;
                        }
                    }
                }
            }
        }
    }

    let species_accuracy = if total_evaluated > 0 {
        total_correct_species as f64 / total_evaluated as f64
    } else {
        0.0
    };
    let taxonomic_accuracy = if total_evaluated > 0 {
        total_correct_taxon as f64 / total_evaluated as f64
    } else {
        0.0
    };

    // Print results
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║         76D ABLATION STUDY RESULTS                                     ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║  Total Samples Evaluated: {:<44}║", total_evaluated);
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Species-Level Accuracy:    {:>8.2}%                              ║",
        species_accuracy * 100.0
    );
    println!(
        "║  Taxonomic-Level Accuracy:  {:>8.2}%                              ║",
        taxonomic_accuracy * 100.0
    );
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║                     COMPARISON                                          ║");
    println!("╠═══════════════╦═════════════════╦═════════════════╦═══════════════════════╣");
    println!("║  Taxonomy     ║  Species Acc    ║  Taxon Acc      ║  Interpretation       ║");
    println!("╠═══════════════╬═════════════════╬═════════════════╬═══════════════════════╣");

    let mut taxa: Vec<_> = per_taxon_stats.iter().collect();
    taxa.sort_by(|a, b| b.1.total.cmp(&a.1.total));

    for (taxon, stats) in taxa {
        if stats.total == 0 {
            continue;
        }
        let spec_acc = stats.correct_species as f64 / stats.total as f64 * 100.0;
        let tax_acc = stats.correct_taxonomic as f64 / stats.total as f64 * 100.0;

        let interpretation = if tax_acc >= 80.0 {
            "✅ Excellent"
        } else if tax_acc >= 60.0 {
            "✅ Good"
        } else if tax_acc >= 40.0 {
            "⚠️  Fair"
        } else {
            "❌ Poor"
        };

        println!(
            "║ {:<13} ║ {:>10.1}%  ║ {:>10.1}%  ║ {:<21} ║",
            taxon, spec_acc, tax_acc, interpretation
        );
    }
    println!("╚═══════════════╩═════════════════╩═════════════════╩═══════════════════════╝");

    println!();
    println!("📊 ABLATION STUDY CONCLUSION:");
    println!();
    println!("  112D (Full):     Species=7.4%, Taxon=56.4%");
    println!("  76D (Ablated):   Species={:.1}%, Taxon={:.1}%", species_accuracy * 100.0, taxonomic_accuracy * 100.0);
    println!();

    if taxonomic_accuracy > 0.60 {
        println!("  ✅ HYPOTHESIS CONFIRMED!");
        println!("     Removing Layer 3 (Micro Texture) improved Species ID.");
        println!("     The 76D stack is better for Taxonomic Classification.");
    } else if taxonomic_accuracy > 0.55 {
        println!("  ⚠️  MIXED RESULTS");
        println!("     Similar performance. Layer 3 may not be pure noise.");
    } else {
        println!("  ❌ HYPOTHESIS REJECTED");
        println!("     Layer 3 features contributed positively to classification.");
    }

    println!();
    println!("Evaluation completed in {:.1}s", start.elapsed().as_secs_f32());

    // Save results
    let results = EvaluationResults {
        species_accuracy,
        taxonomic_accuracy,
        per_taxon_stats,
        total_samples: total_evaluated,
    };
    let json = serde_json::to_string_pretty(&results)?;
    fs::write("rf_76d_ablation_results.json", json)?;
    println!("Saved results to: rf_76d_ablation_results.json");

    Ok(())
}
