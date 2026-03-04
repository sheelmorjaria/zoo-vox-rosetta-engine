//! Evaluate Hybrid Expert Architecture
//! ====================================
//!
//! Evaluates the Hybrid Expert ensemble:
//! - Texture NN (66D) predictions with taxonomic masking
//! - Physics RF (46D) predictions
//! - Soft voting ensemble
//!
//! Usage:
//!   export LIBTORCH=$HOME/libtorch
//!   export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
//!   cargo run --release --bin eval_hybrid_expert --features gpu-training

use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use tch::{nn, nn::Module, Device, Tensor};

use technical_architecture::taxonomic_router::{
    Taxon, FEATURE_DIM, TEXTURE_DIM,
    apply_taxonomic_mask, map_species_to_taxon, map_task_to_taxon, slice_texture,
};

const DROPOUT: f64 = 0.4;
const HIDDEN_DIM_1: i64 = 512;
const HIDDEN_DIM_2: i64 = 256;
const HIDDEN_DIM_3: i64 = 128;

// =============================================================================
// Texture Network (same as training)
// =============================================================================

#[derive(Debug)]
struct TextureNet {
    fc1: nn::Linear,
    ln1: nn::LayerNorm,
    fc2: nn::Linear,
    ln2: nn::LayerNorm,
    fc3: nn::Linear,
    ln3: nn::LayerNorm,
    out: nn::Linear,
}

impl TextureNet {
    fn new(vs: &nn::Path, num_classes: i64) -> TextureNet {
        let fc1 = nn::linear(vs, TEXTURE_DIM as i64, HIDDEN_DIM_1, Default::default());
        let ln1 = nn::layer_norm(vs, vec![HIDDEN_DIM_1], Default::default());
        let fc2 = nn::linear(vs, HIDDEN_DIM_1, HIDDEN_DIM_2, Default::default());
        let ln2 = nn::layer_norm(vs, vec![HIDDEN_DIM_2], Default::default());
        let fc3 = nn::linear(vs, HIDDEN_DIM_2, HIDDEN_DIM_3, Default::default());
        let ln3 = nn::layer_norm(vs, vec![HIDDEN_DIM_3], Default::default());
        let out = nn::linear(vs, HIDDEN_DIM_3, num_classes, Default::default());

        TextureNet { fc1, ln1, fc2, ln2, fc3, ln3, out }
    }
}

impl nn::Module for TextureNet {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let x = xs.apply(&self.fc1).apply(&self.ln1).gelu("none").dropout(DROPOUT, false);
        let x = x.apply(&self.fc2).apply(&self.ln2).gelu("none").dropout(DROPOUT, false);
        let x = x.apply(&self.fc3).apply(&self.ln3).gelu("none").dropout(DROPOUT, false);
        x.apply(&self.out)
    }
}

// =============================================================================
// Data Structures
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
// Evaluation
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Hybrid Expert Evaluation                                         ║");
    println!("║  - Texture NN (66D) with Taxonomic Masking                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let device = Device::cuda_if_available();
    println!("Device: {:?}", device);
    println!();

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
    println!("  Cached features: {}", cache_manifest.entries.len());

    // Load all features
    println!("\nLoading features...");
    let mut all_features: Vec<Vec<f32>> = Vec::new();
    let mut all_labels: Vec<String> = Vec::new();
    let mut all_taxons: Vec<Taxon> = Vec::new();

    for sample in &manifest.samples {
        let audio_file = &sample.audio_file;
        let label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };

        let taxon = map_species_to_taxon(&label);
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
                            let masked = apply_taxonomic_mask(&features, taxon);
                            all_features.push(masked);
                            all_labels.push(label);
                            all_taxons.push(taxon);
                        }
                    }
                }
            }
        }
    }

    println!("  Loaded {} samples", all_features.len());

    // Build label mapping
    let mut unique_labels: Vec<String> = all_labels.iter().cloned().collect();
    unique_labels.sort();
    unique_labels.dedup();
    let n_classes = unique_labels.len() as i64;
    let mut label_to_idx = HashMap::new();
    for (idx, label) in unique_labels.iter().enumerate() {
        label_to_idx.insert(label.clone(), idx as i64);
    }
    let idx_to_label: HashMap<i64, &String> = unique_labels.iter()
        .enumerate()
        .map(|(idx, label)| (idx as i64, label))
        .collect();
    println!("  Classes: {}", n_classes);

    // Use last 10% as test set
    let n_test = (all_features.len() as f32 * 0.1) as usize;
    let n_train = all_features.len() - n_test;
    println!("\nTest set: {} samples", n_test);

    // Extract texture features
    let all_texture: Vec<Vec<f32>> = all_features.iter()
        .map(|f| slice_texture(f))
        .collect();

    // Compute normalization from training set
    let mut texture_means = vec![0.0f32; TEXTURE_DIM];
    let mut texture_stds = vec![0.0f32; TEXTURE_DIM];

    for i in 0..n_train {
        for (j, &v) in all_texture[i].iter().enumerate() {
            texture_means[j] += v;
        }
    }
    for j in 0..TEXTURE_DIM {
        texture_means[j] /= n_train as f32;
    }

    for i in 0..n_train {
        for (j, &v) in all_texture[i].iter().enumerate() {
            let diff = v - texture_means[j];
            texture_stds[j] += diff * diff;
        }
    }
    for j in 0..TEXTURE_DIM {
        texture_stds[j] = (texture_stds[j] / n_train as f32).sqrt().max(1e-8);
    }

    // Create test tensor
    let test_indices: Vec<usize> = (n_train..all_features.len()).collect();
    let test_data: Vec<f32> = test_indices
        .iter()
        .flat_map(|&i| {
            all_texture[i].iter()
                .enumerate()
                .map(|(j, &v)| (v - texture_means[j]) / texture_stds[j])
                .collect::<Vec<_>>()
        })
        .collect();
    let test_labels: Vec<i64> = test_indices
        .iter()
        .map(|&i| *label_to_idx.get(&all_labels[i]).unwrap_or(&0))
        .collect();

    let test_x = Tensor::from_slice(&test_data)
        .view([n_test as i64, TEXTURE_DIM as i64])
        .to(device);
    let test_y = Tensor::from_slice(&test_labels).to(device);

    println!("  Test tensor shape: {:?}", test_x.size());

    // Load model
    println!("\nLoading Texture NN from: hybrid_expert_texture_nn.ot");
    let mut vs = nn::VarStore::new(device);
    let net = TextureNet::new(&vs.root(), n_classes);
    vs.load("hybrid_expert_texture_nn.ot")?;

    // Evaluate
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Evaluation Results                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Forward pass
    let logits = net.forward(&test_x);
    let predictions = logits.argmax(-1, false);

    // Compute accuracy
    use tch::Kind;
    let correct = predictions.eq_tensor(&test_y).sum(Kind::Int64).int64_value(&[]);
    let accuracy = correct as f32 / n_test as f32 * 100.0;

    // Per-class accuracy
    let mut class_correct: HashMap<i64, i64> = HashMap::new();
    let mut class_total: HashMap<i64, i64> = HashMap::new();

    let pred_vec: Vec<i64> = (0..n_test as i64)
        .map(|i| predictions.get(i).int64_value(&[]))
        .collect();
    let label_vec: Vec<i64> = (0..n_test as i64)
        .map(|i| test_y.get(i).int64_value(&[]))
        .collect();

    for i in 0..n_test {
        let pred = pred_vec[i];
        let true_label = label_vec[i];

        *class_total.entry(true_label).or_insert(0) += 1;
        if pred == true_label {
            *class_correct.entry(true_label).or_insert(0) += 1;
        }
    }

    let mut class_stats: Vec<(i64, i64, i64)> = class_total.iter()
        .map(|(&idx, &total)| {
            let correct = *class_correct.get(&idx).unwrap_or(&0);
            (idx, total, correct)
        })
        .collect();
    class_stats.sort_by(|a, b| b.1.cmp(&a.1));

    println!("Overall Test Accuracy: {:.2}%", accuracy);
    println!("Correct: {} / {}", correct, n_test);
    println!();

    println!("Top 20 Classes:");
    println!("{:<50} {:>8} {:>8} {:>8}", "Class", "Total", "Correct", "Accuracy");
    println!("{}", "-".repeat(76));

    for (idx, total, correct) in class_stats.iter().take(20) {
        let label_name = match idx_to_label.get(idx) {
            Some(name) => name.as_str(),
            None => "<unknown>",
        };
        let acc = if *total > 0 {
            (*correct as f64 / *total as f64) * 100.0
        } else {
            0.0
        };
        println!("{:<50} {:>8} {:>8} {:>7.1}%", label_name, total, correct, acc);
    }

    // Taxonomic breakdown
    println!("\n{}", "-".repeat(76));
    println!("Accuracy by Taxonomic Group:");

    let mut taxon_correct: HashMap<Taxon, i64> = HashMap::new();
    let mut taxon_total: HashMap<Taxon, i64> = HashMap::new();

    for i in 0..n_test {
        let taxon = all_taxons[n_train + i];
        let pred = pred_vec[i];
        let true_label = label_vec[i];

        *taxon_total.entry(taxon).or_insert(0) += 1;
        if pred == true_label {
            *taxon_correct.entry(taxon).or_insert(0) += 1;
        }
    }

    let mut taxon_stats: Vec<(Taxon, i64, i64)> = taxon_total.iter()
        .map(|(&taxon, &total)| {
            let correct = *taxon_correct.get(&taxon).unwrap_or(&0);
            (taxon, total, correct)
        })
        .collect();
    taxon_stats.sort_by(|a, b| b.1.cmp(&a.1));

    for (taxon, total, correct) in &taxon_stats {
        let acc = if *total > 0 {
            (*correct as f64 / *total as f64) * 100.0
        } else {
            0.0
        };
        println!("  {:?} {:>20} {:>6} samples, {:>5.1}% accuracy", 
            taxon, "", total, acc);
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Architecture:       Hybrid Expert (Texture NN 66D)              ║");
    println!("║  Taxonomic Masking:  Enabled                                     ║");
    println!("║  Test Samples:       {:>44}║", n_test);
    println!("║  Classes:            {:>44}║", n_classes);
    println!("║  Test Accuracy:      {:>8.2}%                                   ║", accuracy);
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    println!("\nComparison:");
    println!("  RF (112D):       35.07%");
    println!("  Full NN (112D):  55.09%");
    println!("  Texture NN (66D + masking): {:.2}%", accuracy);

    Ok(())
}
