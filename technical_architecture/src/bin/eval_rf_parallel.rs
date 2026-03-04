//! Evaluate Parallel RF Model on BEANS-Zero Benchmark
//! ====================================================

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use technical_architecture::beans_zero_weights::{BeansZeroWeightRouter, TaxonomicGroup};

const FEATURE_DIM: usize = 112;

#[derive(Debug, Deserialize)]
struct RandomForestModel {
    trees: Vec<DecisionTree>,
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
    #[allow(dead_code)]
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

impl RandomForestModel {
    fn predict(&self, features: &[f32]) -> (usize, String) {
        let normalized: Vec<f32> = features
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - self.feature_means[i]) / self.feature_stds[i])
            .collect();

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
            if features[feature_idx] <= node.threshold {
                node_idx = node.left_child.unwrap_or(0);
            } else {
                node_idx = node.right_child.unwrap_or(0);
            }
        }
    }
}

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Parallel Random Forest Evaluation (112D Cached Features)        ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let start = Instant::now();

    println!("Loading trained model from: random_forest_model_112d_parallel.json");
    let model_data = fs::read_to_string("random_forest_model_112d_parallel.json")?;
    let model: RandomForestModel = serde_json::from_str(&model_data)?;
    println!("  Loaded {} trees, {} classes", model.trees.len(), model.n_classes);

    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_manifest_path = cache_dir.join("cache_manifest.json");
    let cache_data = fs::read_to_string(&cache_manifest_path)?;
    let cache_manifest: CacheManifest = serde_json::from_str(&cache_data)?;
    println!("  Cached features available: {}", cache_manifest.entries.len());

    // Load full manifest
    let manifest_data = fs::read_to_string("beans_zero_full_manifest.json")?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_data)?;
    let samples = manifest["samples"].as_array().unwrap();
    let n_samples = samples.len();
    println!("  Total samples: {}", n_samples);

    // Use last 10% for evaluation
    let n_eval = (n_samples as f32 * 0.1) as usize;
    let eval_start = n_samples - n_eval;
    println!("  Evaluation samples: {}", n_eval);

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Running Evaluation                                                ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    let mut per_taxon_stats: HashMap<String, (usize, usize, usize)> = HashMap::new();
    let mut total_correct_species = 0;
    let mut total_correct_taxon = 0;
    let mut total_evaluated = 0;

    for i in eval_start..n_samples {
        if (i - eval_start + 1) % 1000 == 0 {
            println!("  Progress: {}/{}", i - eval_start + 1, n_eval);
        }

        let sample = &samples[i];
        let audio_file = sample["audio_file"].as_str().unwrap();
        let output = sample["labels"]["output"].as_str().unwrap();
        let task = sample["labels"]["task"].as_str().unwrap();
        let true_label = if output != "None" { output } else { task };

        if let Some(cache_file) = cache_manifest.entries.get(audio_file) {
            let full_path = cache_dir.join(cache_file);
            if let Ok(file) = fs::File::open(&full_path) {
                let reader = BufReader::new(file);
                if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                    if features.len() == FEATURE_DIM {
                        let (_, pred_label) = model.predict(&features);

                        let true_taxon = BeansZeroWeightRouter::detect_group(true_label);
                        let pred_taxon = BeansZeroWeightRouter::detect_group(&pred_label);

                        let taxon_name = taxon_to_string(&true_taxon);
                        let (sp, tx, tot) = per_taxon_stats.entry(taxon_name).or_insert((0, 0, 0));
                        *tot += 1;

                        if true_label.to_lowercase() == pred_label.to_lowercase() {
                            *sp += 1;
                            total_correct_species += 1;
                        }
                        if true_taxon == pred_taxon {
                            *tx += 1;
                            total_correct_taxon += 1;
                        }
                        total_evaluated += 1;
                    }
                }
            }
        }
    }

    let species_accuracy = total_correct_species as f64 / total_evaluated as f64;
    let taxonomic_accuracy = total_correct_taxon as f64 / total_evaluated as f64;

    println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║         PARALLEL RANDOM FOREST EVALUATION RESULTS                     ║");
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║  Total Samples: {:<54}║", total_evaluated);
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║  Species-Level Accuracy:    {:>8.2}%                              ║", species_accuracy * 100.0);
    println!("║  Taxonomic-Level Accuracy:  {:>8.2}%                              ║", taxonomic_accuracy * 100.0);
    println!("╠═══════════════════════════════════════════════════════════════════════╣");
    println!("║                     TAXONOMIC BREAKDOWN                                ║");
    println!("╠═══════════════╦═════════════════╦═════════════════╦═══════════════════════╣");
    println!("║  Taxonomy     ║  Species Acc    ║  Taxon Acc      ║  Interpretation       ║");
    println!("╠═══════════════╬═════════════════╬═════════════════╬═══════════════════════╣");

    let mut taxa: Vec<_> = per_taxon_stats.iter().collect();
    taxa.sort_by(|a, b| b.1.2.cmp(&a.1.2));

    for (taxon, (sp, tx, tot)) in taxa {
        if *tot == 0 { continue; }
        let spec_acc = *sp as f64 / *tot as f64 * 100.0;
        let tax_acc = *tx as f64 / *tot as f64 * 100.0;
        let interpretation = if tax_acc >= 80.0 { "✅ Excellent" }
            else if tax_acc >= 60.0 { "✅ Good" }
            else if tax_acc >= 40.0 { "⚠️  Fair" }
            else { "❌ Poor" };
        println!("║ {:<13} ║ {:>10.1}%  ║ {:>10.1}%  ║ {:<21} ║",
            taxon, spec_acc, tax_acc, interpretation);
    }
    println!("╚═══════════════╩═════════════════╩═════════════════╩═══════════════════════╝");

    println!("\nEvaluation completed in {:.1}s", start.elapsed().as_secs_f32());

    Ok(())
}
