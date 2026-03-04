//! Evaluate Neural Network (112D Features) - GPU Model
//! ====================================================
//!
//! Evaluates the trained RosettaNet model on the BEANS-Zero test set.
//!
//! Usage:
//!   export LIBTORCH=$HOME/libtorch
//!   export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
//!   cargo run --release --bin eval_nn_112d_gpu --features gpu-training

use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use tch::{nn, nn::Module, Device, Kind, Tensor};

const FEATURE_DIM: i64 = 112;
const HIDDEN_DIM_1: i64 = 1024;
const HIDDEN_DIM_2: i64 = 512;
const HIDDEN_DIM_3: i64 = 256;
const HIDDEN_DIM_4: i64 = 128;
const DROPOUT: f64 = 0.5;

// =============================================================================
// Neural Network Definition (same as training)
// =============================================================================

#[derive(Debug)]
struct RosettaNet {
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

impl RosettaNet {
    fn new(vs: &nn::Path, num_classes: i64) -> RosettaNet {
        let fc1 = nn::linear(vs, FEATURE_DIM, HIDDEN_DIM_1, Default::default());
        let ln1 = nn::layer_norm(vs, vec![HIDDEN_DIM_1], Default::default());
        let fc2 = nn::linear(vs, HIDDEN_DIM_1, HIDDEN_DIM_2, Default::default());
        let ln2 = nn::layer_norm(vs, vec![HIDDEN_DIM_2], Default::default());
        let fc3 = nn::linear(vs, HIDDEN_DIM_2, HIDDEN_DIM_3, Default::default());
        let ln3 = nn::layer_norm(vs, vec![HIDDEN_DIM_3], Default::default());
        let fc4 = nn::linear(vs, HIDDEN_DIM_3, HIDDEN_DIM_4, Default::default());
        let ln4 = nn::layer_norm(vs, vec![HIDDEN_DIM_4], Default::default());
        let out = nn::linear(vs, HIDDEN_DIM_4, num_classes, Default::default());

        RosettaNet { fc1, ln1, fc2, ln2, fc3, ln3, fc4, ln4, out }
    }
}

impl nn::Module for RosettaNet {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // Block 1: Linear -> LN -> GELU -> Dropout
        let x = xs.apply(&self.fc1).apply(&self.ln1).gelu("none").dropout(DROPOUT, false);

        // Block 2: Linear -> LN -> GELU -> Dropout
        let x = x.apply(&self.fc2).apply(&self.ln2).gelu("none").dropout(DROPOUT, false);

        // Block 3: Linear -> LN -> GELU -> Dropout
        let x = x.apply(&self.fc3).apply(&self.ln3).gelu("none").dropout(DROPOUT, false);

        // Block 4: Linear -> LN -> GELU -> Dropout
        let x = x.apply(&self.fc4).apply(&self.ln4).gelu("none").dropout(DROPOUT, false);

        // Output layer
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
    println!("║  GPU Neural Network Evaluation (112D Features)                   ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Check for GPU
    let device = Device::cuda_if_available();
    println!("Device: {:?}", device);
    if !device.is_cuda() {
        println!("WARNING: CUDA not available, falling back to CPU!");
    }
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
    println!("Loading cache manifest from: {:?}", cache_manifest_path);
    let cache_data = fs::read_to_string(&cache_manifest_path)?;
    let cache_manifest: CacheManifest = serde_json::from_str(&cache_data)?;
    println!("  Cached features available: {}", cache_manifest.entries.len());

    // Load all features and labels
    println!("\nLoading features from cache...");
    let mut all_features: Vec<Vec<f32>> = Vec::new();
    let mut all_labels: Vec<String> = Vec::new();
    let mut all_tasks: Vec<String> = Vec::new();

    for sample in &manifest.samples {
        let audio_file = &sample.audio_file;
        let label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };

        if let Some(cache_file) = cache_manifest.entries.get(audio_file) {
            let full_path = cache_dir.join(cache_file);
            if full_path.exists() {
                if let Ok(file) = fs::File::open(&full_path) {
                    let reader = BufReader::new(file);
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                        if features.len() == FEATURE_DIM as usize {
                            all_features.push(features);
                            all_labels.push(label);
                            all_tasks.push(sample.labels.task.clone());
                        }
                    }
                }
            }
        }
    }

    println!("  Loaded {} samples", all_features.len());

    if all_features.is_empty() {
        anyhow::bail!("No features loaded!");
    }

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

    // Use last 10% as test set (same split as training)
    let n_test = (all_features.len() as f32 * 0.1) as usize;
    let n_train = all_features.len() - n_test;
    println!("\nTest set: {} samples (last 10%)", n_test);

    // Compute normalization params from training set (first 90%)
    let mut feature_means = vec![0.0f32; FEATURE_DIM as usize];
    let mut feature_stds = vec![0.0f32; FEATURE_DIM as usize];

    for i in 0..n_train {
        for (j, &v) in all_features[i].iter().enumerate() {
            feature_means[j] += v;
        }
    }
    for j in 0..FEATURE_DIM as usize {
        feature_means[j] /= n_train as f32;
    }

    for i in 0..n_train {
        for (j, &v) in all_features[i].iter().enumerate() {
            let diff = v - feature_means[j];
            feature_stds[j] += diff * diff;
        }
    }
    for j in 0..FEATURE_DIM as usize {
        feature_stds[j] = (feature_stds[j] / n_train as f32).sqrt().max(1e-8);
    }

    // Create test tensor
    println!("\nCreating test tensors...");
    let test_indices: Vec<usize> = (n_train..all_features.len()).collect();
    let test_data: Vec<f32> = test_indices
        .iter()
        .flat_map(|&i| {
            all_features[i].iter()
                .enumerate()
                .map(|(j, &v)| (v - feature_means[j]) / feature_stds[j])
                .collect::<Vec<_>>()
        })
        .collect();
    let test_labels: Vec<i64> = test_indices
        .iter()
        .map(|&i| *label_to_idx.get(&all_labels[i]).unwrap_or(&0))
        .collect();

    let test_x = Tensor::from_slice(&test_data)
        .view([n_test as i64, FEATURE_DIM])
        .to(device);
    let test_y = Tensor::from_slice(&test_labels).to(device);

    println!("  Test tensor shape: {:?}", test_x.size());

    // Load model
    println!("\nLoading model from: rosetta_net_112d_gpu.ot");
    let mut vs = nn::VarStore::new(device);
    let net = RosettaNet::new(&vs.root(), n_classes);
    vs.load("rosetta_net_112d_gpu.ot")?;

    // Evaluate
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Evaluation Results                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Forward pass on test set
    let logits = net.forward(&test_x);
    let predictions = logits.argmax(-1, false);

    // Compute accuracy
    let correct = predictions.eq_tensor(&test_y).sum(Kind::Int64).int64_value(&[]);
    let accuracy = correct as f32 / n_test as f32 * 100.0;

    // Compute per-class accuracy for top classes
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

    // Find top 20 classes by sample count
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

    println!("Top 20 Classes by Sample Count:");
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

    // Compute taxonomic accuracy (by task)
    println!("\n{}", "-".repeat(76));
    println!("Accuracy by Task (Taxonomic):");

    let mut task_correct: HashMap<String, i64> = HashMap::new();
    let mut task_total: HashMap<String, i64> = HashMap::new();

    for i in 0..n_test {
        let true_task = &all_tasks[n_train + i];
        let pred = pred_vec[i];
        let true_label = label_vec[i];

        *task_total.entry(true_task.clone()).or_insert(0) += 1;
        if pred == true_label {
            *task_correct.entry(true_task.clone()).or_insert(0) += 1;
        }
    }

    let mut task_stats: Vec<(String, i64, i64)> = task_total.iter()
        .map(|(task, &total)| {
            let correct = *task_correct.get(task).unwrap_or(&0);
            (task.clone(), total, correct)
        })
        .collect();
    task_stats.sort_by(|a, b| b.1.cmp(&a.1));

    let mut total_task_correct = 0i64;
    let mut total_task_samples = 0i64;

    for (task, total, correct) in &task_stats {
        let acc = if *total > 0 {
            (*correct as f64 / *total as f64) * 100.0
        } else {
            0.0
        };
        println!("  {:<30} {:>6} samples, {:>5.1}% accuracy", task, total, acc);
        total_task_correct += correct;
        total_task_samples += total;
    }

    let taxonomic_acc = if total_task_samples > 0 {
        total_task_correct as f64 / total_task_samples as f64 * 100.0
    } else {
        0.0
    };

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Device:             {:<44}║", format!("{:?}", device));
    println!("║  Test Samples:       {:>44}║", n_test);
    println!("║  Classes:            {:>44}║", n_classes);
    println!("║  Species Accuracy:   {:>8.2}%                                   ║", accuracy);
    println!("║  Taxonomic Accuracy: {:>8.2}%                                   ║", taxonomic_acc);
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    // Comparison with baseline
    println!();
    println!("Comparison with Random Forest Baseline:");
    println!("  RF Species:   35.07%");
    println!("  RF Taxonomic: 88.18%");
    println!("  NN Species:   {:.2}% ({:+.2}%)", accuracy, accuracy - 35.07);
    println!("  NN Taxonomic: {:.2}% ({:+.2}%)", taxonomic_acc, taxonomic_acc - 88.18);

    Ok(())
}
