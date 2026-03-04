//! Train Species Expert NN (112D) with GPU - Universal Fallback
//! ===================================================================
//!
//! GPU-accelerated training using tch (libtorch).
//! This NN has access to ALL 112D features and serves as the universal fallback.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor, Kind};

use technical_architecture::taxonomic_router::FEATURE_DIM;

// Hyperparameters
const HIDDEN_DIM_1: i64 = 768;
const HIDDEN_DIM_2: i64 = 512;
const HIDDEN_DIM_3: i64 = 256;
const LEARNING_RATE: f64 = 1e-4;
const WEIGHT_DECAY: f64 = 0.01;
const EPOCHS: i64 = 100;
const BATCH_SIZE: i64 = 256;
const PATIENCE: i64 = 15;
const DROPOUT: f64 = 0.5;  // Increased from 0.3 for better regularization

// Neural Network
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
        let x = xs.apply(&self.fc1).apply(&self.ln1).gelu("none").dropout(DROPOUT, false);
        let x = x.apply(&self.fc2).apply(&self.ln2).gelu("none").dropout(DROPOUT, false);
        let x = x.apply(&self.fc3).apply(&self.ln3).gelu("none").dropout(DROPOUT, false);
        let x = x.apply(&self.fc4).apply(&self.ln4).gelu("none").dropout(DROPOUT, false);
        x.apply(&self.out)
    }
}

// Data structures
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

#[derive(Debug, Serialize)]
struct ModelMetadata {
    input_dim: usize,
    num_classes: usize,
    val_accuracy: f64,
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,
    label_to_idx: HashMap<String, usize>,
}

// RNG
struct SimpleRng { state: u64 }

impl SimpleRng {
    fn seed(seed: u64) -> Self { Self { state: if seed == 0 { 1 } else { seed } } }
    fn next_usize(&mut self, max: usize) -> usize {
        self.state ^= self.state >> 12; self.state ^= self.state << 25; self.state ^= self.state >> 27;
        ((self.state.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as usize) % max
    }
}

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Species Expert NN (112D) - GPU Training                               ║");
    println!("║  Universal Fallback with FULL feature access                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let device = Device::cuda_if_available();
    println!("Device: {:?}", device);
    if !device.is_cuda() {
        println!("WARNING: CUDA not available, using CPU!");
    }
    println!();

    let start = Instant::now();

    // Load data
    println!("Loading BEANS-Zero dataset...");
    let manifest: BeansManifest = serde_json::from_str(&fs::read_to_string("beans_zero_full_manifest.json")?)?;
    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_manifest: CacheManifest = serde_json::from_str(&fs::read_to_string(cache_dir.join("cache_manifest.json"))?)?;

    // Build labels
    let mut unique_labels: Vec<String> = Vec::new();
    for sample in &manifest.samples {
        let label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };
        if !unique_labels.contains(&label) { unique_labels.push(label); }
    }
    unique_labels.sort();

    let num_classes = unique_labels.len();
    println!("Classes: {}", num_classes);

    let mut label_to_idx: HashMap<String, usize> = HashMap::new();
    for (idx, label) in unique_labels.iter().enumerate() {
        label_to_idx.insert(label.clone(), idx);
    }

    // Load features
    println!("Loading features...");
    let mut all_features: Vec<Vec<f32>> = Vec::new();
    let mut all_labels: Vec<usize> = Vec::new();

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
                            all_labels.push(*label_to_idx.get(&label).unwrap());
                        }
                    }
                }
            }
        }
    }
    println!("Loaded: {} samples", all_features.len());

    // Stratified 80/20 split
    let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &label) in all_labels.iter().enumerate() {
        class_indices.entry(label).or_default().push(i);
    }

    let mut rng = SimpleRng::seed(42);
    let mut train_indices: Vec<usize> = Vec::new();
    let mut val_indices: Vec<usize> = Vec::new();

    for (_, mut indices) in class_indices {
        for i in 0..indices.len() {
            let j = rng.next_usize(indices.len());
            indices.swap(i, j);
        }
        let n_train = (indices.len() as f32 * 0.8) as usize;
        train_indices.extend(indices[..n_train].iter().copied());
        val_indices.extend(indices[n_train..].iter().copied());
    }

    println!("Train: {}, Val: {}", train_indices.len(), val_indices.len());

    // Normalize
    let mut feature_means = vec![0.0f32; FEATURE_DIM];
    let mut feature_stds = vec![1.0f32; FEATURE_DIM];

    for &i in &train_indices {
        for (j, &v) in all_features[i].iter().enumerate() {
            feature_means[j] += v;
        }
    }
    for m in &mut feature_means { *m /= train_indices.len() as f32; }

    for &i in &train_indices {
        for (j, &v) in all_features[i].iter().enumerate() {
            feature_stds[j] += (v - feature_means[j]).powi(2);
        }
    }
    for s in &mut feature_stds { *s = (*s / train_indices.len() as f32).sqrt().max(1e-8); }

    // Create tensors
    println!("Creating tensors...");
    let train_size = train_indices.len();
    let val_size = val_indices.len();

    let train_data: Vec<f32> = train_indices.iter()
        .flat_map(|&i| all_features[i].iter().enumerate()
            .map(|(j, &v)| (v - feature_means[j]) / feature_stds[j]))
        .collect();
    let train_labels: Vec<i64> = train_indices.iter().map(|&i| all_labels[i] as i64).collect();

    let val_data: Vec<f32> = val_indices.iter()
        .flat_map(|&i| all_features[i].iter().enumerate()
            .map(|(j, &v)| (v - feature_means[j]) / feature_stds[j]))
        .collect();
    let val_labels: Vec<i64> = val_indices.iter().map(|&i| all_labels[i] as i64).collect();

    let train_x = Tensor::from_slice(&train_data).view([train_size as i64, FEATURE_DIM as i64]).to(device);
    let train_y = Tensor::from_slice(&train_labels).to(device);
    let val_x = Tensor::from_slice(&val_data).view([val_size as i64, FEATURE_DIM as i64]).to(device);
    let val_y = Tensor::from_slice(&val_labels).to(device);

    println!("Train: {:?}, Val: {:?}", train_x.size(), val_x.size());

    // Create model
    let mut vs = nn::VarStore::new(device);
    let net = SpeciesExpert112D::new(&vs.root(), num_classes as i64);
    let mut opt = nn::Adam {
        wd: WEIGHT_DECAY,
        ..nn::Adam::default()
    }.build(&vs, LEARNING_RATE)?;

    println!("\nTraining...");
    let mut best_val_acc = 0.0f64;
    let mut patience_counter = 0i64;

    for epoch in 0..EPOCHS {
        // Shuffle
        let perm = Tensor::randperm(train_size as i64, (Kind::Int64, device));
        let shuffled_x = train_x.index_select(0, &perm);
        let shuffled_y = train_y.index_select(0, &perm);

        // Train batches
        for batch_start in (0..train_size as i64).step_by(BATCH_SIZE as usize) {
            let batch_end = (batch_start + BATCH_SIZE).min(train_size as i64);
            let batch_x = shuffled_x.narrow(0, batch_start, batch_end - batch_start);
            let batch_y = shuffled_y.narrow(0, batch_start, batch_end - batch_start);

            let logits = net.forward(&batch_x);
            let loss = logits.cross_entropy_for_logits(&batch_y);

            opt.backward_step(&loss);
        }

        // Evaluate
        let val_logits = net.forward(&val_x);
        let val_preds = val_logits.argmax(-1, false);
        let val_correct = val_preds.eq_tensor(&val_y).sum(Kind::Int64);
        let val_acc = val_correct.double_value(&[]) / val_size as f64;

        let train_logits = net.forward(&train_x);
        let train_preds = train_logits.argmax(-1, false);
        let train_correct = train_preds.eq_tensor(&train_y).sum(Kind::Int64);
        let train_acc = train_correct.double_value(&[]) / train_size as f64;

        if epoch % 5 == 0 {
            println!("Epoch {:3}: Train={:.2}% Val={:.2}%", epoch, train_acc * 100.0, val_acc * 100.0);
        }

        // Early stopping
        if val_acc > best_val_acc {
            best_val_acc = val_acc;
            patience_counter = 0;
            vs.save("species_expert_112d.ot")?;
        } else {
            patience_counter += 1;
            if patience_counter >= PATIENCE {
                println!("Early stopping at epoch {}", epoch);
                break;
            }
        }
    }

    // Final evaluation
    vs.load("species_expert_112d.ot")?;
    let final_logits = net.forward(&val_x);
    let final_preds = final_logits.argmax(-1, false);
    let final_correct = final_preds.eq_tensor(&val_y).sum(Kind::Int64);
    let final_acc = final_correct.double_value(&[]) / val_size as f64;

    println!("\n=== Results ===");
    println!("Best Val Accuracy: {:.2}%", best_val_acc * 100.0);
    println!("Final Val Accuracy: {:.2}%", final_acc * 100.0);
    println!("Training Time: {:.1}s", start.elapsed().as_secs_f32());

    // Save metadata
    let metadata = ModelMetadata {
        input_dim: FEATURE_DIM,
        num_classes,
        val_accuracy: final_acc,
        feature_means,
        feature_stds,
        label_to_idx,
    };
    fs::write("species_expert_112d.json", serde_json::to_string_pretty(&metadata)?)?;

    println!("\nSaved: species_expert_112d.ot, species_expert_112d.json");

    Ok(())
}
