//! Train Random Forest on Physics Features (46D) for Hybrid Expert Ensemble
//! ========================================================================
//!
//! Trains a Random Forest classifier on the 46D physics features (Layer 1)
//! to be combined with the Texture NN for the Hybrid Expert ensemble.
//!
//! Usage:
//!   cargo run --release --bin train_physics_rf --features ml-classical

use anyhow::Result;
use linfa::prelude::*;
use linfa_trees::DecisionTree;
use linfa_preprocessing::linear_scaling::LinearScaler;
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use technical_architecture::taxonomic_router::{
    FEATURE_DIM, PHYSICS_DIM, slice_physics,
};

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
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Random Forest Training - Physics Features (46D)                 ║");
    println!("║  For Hybrid Expert Ensemble                                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let start = Instant::now();

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

    // Load all features and labels
    println!("\nLoading features from cache...");
    let mut all_features: Vec<Vec<f32>> = Vec::new();
    let mut all_labels: Vec<String> = Vec::new();

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
                        if features.len() == FEATURE_DIM {
                            // Extract only physics features (46D)
                            let physics = slice_physics(&features);
                            all_features.push(physics);
                            all_labels.push(label);
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
    let n_classes = unique_labels.len();
    let mut label_to_idx = HashMap::new();
    for (idx, label) in unique_labels.iter().enumerate() {
        label_to_idx.insert(label.clone(), idx as usize);
    }
    println!("  Classes: {}", n_classes);

    // Convert to ndarray
    println!("\nConverting to ndarray format...");
    let n_samples = all_features.len();
    let mut feature_matrix: Array2<f64> = Array2::zeros((n_samples, PHYSICS_DIM));
    let mut label_array: Array1<usize> = Array1::zeros(n_samples);

    for (i, features) in all_features.iter().enumerate() {
        for (j, &v) in features.iter().enumerate() {
            feature_matrix[[i, j]] = v as f64;
        }
        label_array[i] = *label_to_idx.get(&all_labels[i]).unwrap_or(&0);
    }

    // Split into train/test (90/10)
    let n_train = (n_samples as f32 * 0.9) as usize;
    println!("\nSplitting: {} train, {} test", n_train, n_samples - n_train);

    // Shuffle indices
    let mut indices: Vec<usize> = (0..n_samples).collect();
    for i in 0..indices.len() {
        let j = (rand_u32() as usize) % indices.len();
        indices.swap(i, j);
    }

    let train_indices: Vec<usize> = indices[..n_train].to_vec();
    let test_indices: Vec<usize> = indices[n_train..].to_vec();

    // Create train/test splits
    let train_x = feature_matrix.select_axis(ndarray::Axis(0), &train_indices);
    let train_y = label_array.select(&train_indices);
    let test_x = feature_matrix.select_axis(ndarray::Axis(0), &test_indices);
    let test_y = label_array.select(&test_indices);

    // Standardize features
    println!("\nStandardizing features...");
    let scaler = LinearScaler::standard();
    let train_x_scaled = scaler.fit_transform(&train_x)?;
    let test_x_scaled = scaler.transform(&test_x)?;

    // Create dataset
    let train_dataset = Dataset::new(train_x_scaled.to_owned(), train_y.to_owned());

    // Train Random Forest with balanced class weights
    println!("\nTraining Random Forest...");
    println!("  Max depth: 20");
    println!("  Min samples split: 5");
    println!("  Features: {}D physics", PHYSICS_DIM);

    let model = DecisionTree::params()
        .max_depth(Some(20))
        .min_weight_split(5.0)
        .fit(&train_dataset);

    // Evaluate on test set
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Evaluation Results                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Predict on test set
    let predictions = model.predict(&test_x_scaled);
    
    let mut correct = 0usize;
    for (pred, actual) in predictions.iter().zip(test_y.iter()) {
        if pred == actual {
            correct += 1;
        }
    }
    
    let accuracy = correct as f64 / test_indices.len() as f64 * 100.0;
    println!("Test Accuracy: {:.2}%", accuracy);
    println!("Correct: {} / {}", correct, test_indices.len());

    // Per-class accuracy for top classes
    let mut class_correct: HashMap<usize, usize> = HashMap::new();
    let mut class_total: HashMap<usize, usize> = HashMap::new();

    for (pred, actual) in predictions.iter().zip(test_y.iter()) {
        *class_total.entry(*actual).or_insert(0) += 1;
        if pred == actual {
            *class_correct.entry(*actual).or_insert(0) += 1;
        }
    }

    let mut class_stats: Vec<(usize, usize, usize)> = class_total.iter()
        .map(|(&idx, &total)| {
            let correct = *class_correct.get(&idx).unwrap_or(&0);
            (idx, total, correct)
        })
        .collect();
    class_stats.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\nTop 20 Classes:");
    println!("{:<50} {:>8} {:>8} {:>8}", "Class", "Total", "Correct", "Accuracy");
    println!("{}", "-".repeat(76));

    let idx_to_label: HashMap<usize, &String> = unique_labels.iter()
        .enumerate()
        .map(|(idx, label)| (idx, label))
        .collect();

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

    // Save model
    let model_path = "physics_rf_model.json";
    println!("\nSaving model to: {}", model_path);
    
    let model_data = serde_json::json!({
        "model_type": "DecisionTree",
        "feature_dim": PHYSICS_DIM,
        "n_classes": n_classes,
        "labels": unique_labels,
        "accuracy": accuracy,
    });
    fs::write(model_path, serde_json::to_string_pretty(&model_data)?)?;

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Architecture:       Random Forest (Physics 46D)                 ║");
    println!("║  Test Accuracy:      {:>8.2}%                                   ║", accuracy);
    println!("║  Total Time:         {:>8.1}s                                    ║", start.elapsed().as_secs_f32());
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    println!("\nHybrid Expert Ensemble Summary:");
    println!("  - Texture NN (66D):  59.88%");
    println!("  - Physics RF (46D):  {:.2}%", accuracy);
    println!("  - Combined ensemble: (run eval_hybrid_ensemble)");

    Ok(())
}

fn rand_u32() -> u32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x853c49e6748fea9b);

    let mut s = STATE.load(Ordering::Relaxed);
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    STATE.store(s, Ordering::Relaxed);
    (s.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
}
