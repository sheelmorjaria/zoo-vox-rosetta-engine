//! Evaluate Trained Random Forest (112D) on BEANS-Zero Benchmark
//! ================================================================
//!
//! Uses cached 112D features - no re-extraction needed.
//! Uses the same TaxonomicGroup detection as beans_zero_eval.rs
//!
//! Usage:
//!   cargo run --release --bin eval_rf_112d_cached

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use technical_architecture::beans_zero_weights::{BeansZeroWeightRouter, TaxonomicGroup};

const FEATURE_DIM: usize = 112;

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

#[derive(Debug, Deserialize)]
struct DecisionTree {
    nodes: Vec<TreeNode>,
}

#[derive(Debug, Deserialize)]
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
// Evaluation Results (matching beans_zero_eval.rs structure)
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
    pub taxonomic_confusion: HashMap<String, HashMap<String, usize>>,
}

impl EvaluationResults {
    pub fn print_summary(&self) {
        println!();
        println!("╔═══════════════════════════════════════════════════════════════════════╗");
        println!("║         BEANS-Zero Random Forest (112D) Evaluation Results            ║");
        println!("╠═══════════════════════════════════════════════════════════════════════╣");
        println!("║  Total Samples: {:<54}║", self.total_samples);
        println!("╠═══════════════════════════════════════════════════════════════════════╣");
        println!(
            "║  Species-Level Accuracy:    {:>8.2}%                              ║",
            self.species_accuracy * 100.0
        );
        println!(
            "║  Taxonomic-Level Accuracy:  {:>8.2}%                              ║",
            self.taxonomic_accuracy * 100.0
        );
        println!("╠═══════════════════════════════════════════════════════════════════════╣");
        println!("║                     TAXONOMIC BREAKDOWN                                ║");
        println!("╠═══════════════╦═══════════════╦═══════════════╦═══════════════════════╣");
        println!("║  Taxonomy     ║  Species Acc  ║  Taxon Acc    ║  Interpretation       ║");
        println!("╠═══════════════╬═══════════════╬═══════════════╬═══════════════════════╣");

        let mut taxa: Vec<_> = self.per_taxon_stats.iter().collect();
        taxa.sort_by(|a, b| b.1.total.cmp(&a.1.total));

        for (taxon, stats) in taxa {
            if stats.total == 0 {
                continue;
            }
            let species_acc = stats.correct_species as f64 / stats.total as f64 * 100.0;
            let taxon_acc = stats.correct_taxonomic as f64 / stats.total as f64 * 100.0;

            let interpretation = if taxon_acc >= 80.0 {
                "✅ Excellent"
            } else if taxon_acc >= 60.0 {
                "✅ Good"
            } else if taxon_acc >= 40.0 {
                "⚠️  Fair"
            } else {
                "❌ Poor"
            };

            println!(
                "║ {:<13} ║ {:>10.1}%  ║ {:>10.1}%  ║ {:<21} ║",
                taxon, species_acc, taxon_acc, interpretation
            );
        }
        println!("╚═══════════════╩═══════════════╩═══════════════╩═══════════════════════╝");

        println!();
        println!("📊 INTERPRETATION:");
        if self.species_accuracy < 0.10 && self.taxonomic_accuracy > 0.50 {
            println!("   🔬 The model understands the BIOLOGY (high taxonomic accuracy)");
            println!("   but struggles with exact SPECIES NAMES (low species accuracy).");
            println!("   This is EXPECTED for 6,975 classes with imbalanced data.");
        } else if self.taxonomic_accuracy > 0.70 {
            println!("   ✅ Strong taxonomic understanding - model knows birds from whales!");
        } else {
            println!("   ⚠️  Both accuracies need improvement - check feature pipeline.");
        }
    }
}

// =============================================================================
// Random Forest Prediction
// =============================================================================

impl DecisionTree {
    fn predict(&self, features: &[f32]) -> usize {
        let mut node_idx = 0;

        loop {
            let node = &self.nodes[node_idx];

            if node.feature_idx.is_none() {
                return node.class_prediction.unwrap_or(0);
            }

            let feature_idx = node.feature_idx.unwrap();
            let threshold = node.threshold;

            if features[feature_idx] <= threshold {
                node_idx = node.left_child.unwrap_or(0);
            } else {
                node_idx = node.right_child.unwrap_or(0);
            }
        }
    }
}

impl RandomForestModel {
    fn predict(&self, features: &[f32]) -> (usize, String) {
        // Normalize features using stored normalization params
        let normalized: Vec<f32> = features
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let mean = self.feature_means.get(i).copied().unwrap_or(0.0);
                let std = self.feature_stds.get(i).copied().unwrap_or(1.0);
                let std_safe = if std < 1e-10 { 1.0 } else { std };
                (v - mean) / std_safe
            })
            .collect();

        // Vote from all trees
        let mut votes: HashMap<usize, usize> = HashMap::new();
        for tree in &self.trees {
            let pred = tree.predict(&normalized);
            *votes.entry(pred).or_insert(0) += 1;
        }

        // Find majority vote
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
}

// =============================================================================
// Taxonomic Group Helper (using BeansZeroWeightRouter)
// =============================================================================

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
// Main Evaluation
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  BEANS-Zero Random Forest Evaluation (112D Cached Features)       ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let start = Instant::now();

    // Load model
    println!("Loading trained model from: random_forest_model_112d.json");
    let model_data = fs::read_to_string("random_forest_model_112d.json")?;
    let model: RandomForestModel = serde_json::from_str(&model_data)?;
    println!("  Loaded {} trees, {} classes", model.trees.len(), model.n_classes);

    // Load manifest
    let manifest_path = "beans_zero_full_manifest.json";
    println!("\nLoading manifest from: {}", manifest_path);
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

    // Split into train/eval (use 10% for evaluation)
    println!("\nSplitting data: 90% train, 10% evaluation...");
    let n_eval = (manifest.samples.len() as f32 * 0.1) as usize;
    println!("  Evaluation samples: {}", n_eval);

    // Use last 10% for evaluation (consistent split)
    let eval_start = manifest.samples.len() - n_eval;
    let eval_samples = &manifest.samples[eval_start..];
    println!("  Selected {} samples for evaluation", eval_samples.len());

    // Evaluate
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Running Evaluation                                                ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let mut per_taxon_stats: HashMap<String, TaxonomicStats> = HashMap::new();
    let mut taxonomic_confusion: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut total_correct_species = 0;
    let mut total_correct_taxon = 0;
    let mut total_evaluated = 0;
    let mut cache_hits = 0;
    let mut cache_misses = 0;

    for (i, sample) in eval_samples.iter().enumerate() {
        if (i + 1) % 1000 == 0 {
            println!(
                "  Progress: {}/{} ({:.1}%)",
                i + 1,
                eval_samples.len(),
                (i + 1) as f64 / eval_samples.len() as f64 * 100.0
            );
        }

        let audio_file = &sample.audio_file;
        let true_label = if sample.labels.output != "None" {
            &sample.labels.output
        } else {
            &sample.labels.task
        };

        // Load cached features
        let features = if let Some(cache_file) = cache_manifest.entries.get(audio_file) {
            let full_path = cache_dir.join(cache_file);
            if full_path.exists() {
                if let Ok(file) = fs::File::open(&full_path) {
                    let reader = BufReader::new(file);
                    if let Ok(feats) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                        cache_hits += 1;
                        feats
                    } else {
                        cache_misses += 1;
                        continue;
                    }
                } else {
                    cache_misses += 1;
                    continue;
                }
            } else {
                cache_misses += 1;
                continue;
            }
        } else {
            cache_misses += 1;
            continue;
        };

        if features.len() != FEATURE_DIM {
            continue;
        }

        // Predict
        let (_, pred_label) = model.predict(&features);

        // Get taxonomic groups using BeansZeroWeightRouter (same as beans_zero_eval.rs)
        let true_taxon = BeansZeroWeightRouter::detect_group(true_label);
        let pred_taxon = BeansZeroWeightRouter::detect_group(&pred_label);

        // Update stats
        let taxon_name = taxon_to_string(&true_taxon);
        let stats = per_taxon_stats.entry(taxon_name.clone()).or_default();
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

        // Update confusion matrix
        let pred_taxon_name = taxon_to_string(&pred_taxon);
        *taxonomic_confusion
            .entry(taxon_name)
            .or_default()
            .entry(pred_taxon_name)
            .or_default() += 1;

        total_evaluated += 1;
    }

    println!("\n  Cache hits: {}, misses: {}", cache_hits, cache_misses);
    println!("  Total evaluated: {}", total_evaluated);

    // Compute final results
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

    let results = EvaluationResults {
        species_accuracy,
        taxonomic_accuracy,
        per_taxon_stats,
        total_samples: total_evaluated,
        taxonomic_confusion,
    };

    // Print results
    results.print_summary();

    // Save results
    let results_json = serde_json::to_string_pretty(&results)?;
    fs::write("rf_112d_evaluation_results.json", results_json)?;
    println!("\nSaved results to: rf_112d_evaluation_results.json");

    println!("\nEvaluation completed in {:.1}s", start.elapsed().as_secs_f32());

    Ok(())
}
