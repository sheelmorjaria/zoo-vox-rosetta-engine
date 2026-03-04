//! Train Species Expert NN (112D) - Simple Pure Rust Implementation
//! ===================================================================
//!
//! A simplified neural network for species classification using all 112D features.
//! This serves as the "Universal Fallback" when the Gatekeeper RF is uncertain.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use technical_architecture::taxonomic_router::FEATURE_DIM;

// Hyperparameters
const HIDDEN_DIM_1: usize = 512;
const HIDDEN_DIM_2: usize = 256;
const HIDDEN_DIM_3: usize = 128;
const LEARNING_RATE: f32 = 0.01;
const EPOCHS: usize = 30;
const BATCH_SIZE: usize = 256;
const PATIENCE: usize = 8;

// Simple dense layer
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DenseLayer {
    weights: Vec<f32>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl DenseLayer {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        let scale = (2.0 / in_dim as f32).sqrt();
        let mut weights = Vec::with_capacity(in_dim * out_dim);
        for _ in 0..(in_dim * out_dim) {
            weights.push((rand_f32() - 0.5) * 2.0 * scale);
        }
        Self {
            weights,
            bias: vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = self.bias.clone();
        for i in 0..self.in_dim {
            for j in 0..self.out_dim {
                output[j] += input[i] * self.weights[i * self.out_dim + j];
            }
        }
        output
    }
}

// Activation functions
fn relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }
fn gelu(x: f32) -> f32 { 0.5 * x * (1.0 + (x * 0.044715).tanh()) }

fn softmax(x: &[f32]) -> Vec<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = x.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|v| v / sum.max(1e-10)).collect()
}

// Full network
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpeciesExpertNN {
    fc1: DenseLayer,
    fc2: DenseLayer,
    fc3: DenseLayer,
    out: DenseLayer,
}

impl SpeciesExpertNN {
    fn new(num_classes: usize) -> Self {
        Self {
            fc1: DenseLayer::new(FEATURE_DIM, HIDDEN_DIM_1),
            fc2: DenseLayer::new(HIDDEN_DIM_1, HIDDEN_DIM_2),
            fc3: DenseLayer::new(HIDDEN_DIM_2, HIDDEN_DIM_3),
            out: DenseLayer::new(HIDDEN_DIM_3, num_classes),
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut h = self.fc1.forward(input);
        for v in &mut h { *v = gelu(*v); }

        let mut h = self.fc2.forward(&h);
        for v in &mut h { *v = gelu(*v); }

        let mut h = self.fc3.forward(&h);
        for v in &mut h { *v = gelu(*v); }

        self.out.forward(&h)
    }

    fn predict(&self, input: &[f32]) -> usize {
        let logits = self.forward(input);
        let probs = softmax(&logits);

        let mut max_idx = 0;
        let mut max_val = probs[0];
        for (i, &p) in probs.iter().enumerate() {
            if p > max_val {
                max_val = p;
                max_idx = i;
            }
        }
        max_idx
    }
}

fn rand_f32() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x853c49e6748fea9b);
    let mut s = STATE.load(Ordering::Relaxed);
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    STATE.store(s, Ordering::Relaxed);
    ((s.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32) as f32 / u32::MAX as f32
}

// Simple SGD training via numerical gradient
fn train_batch(model: &mut SpeciesExpertNN, inputs: &[Vec<f32>], targets: &[usize], lr: f32) -> f32 {
    let mut total_loss = 0.0;
    let eps = 1e-3;

    for (input, &target) in inputs.iter().zip(targets.iter()) {
        let logits = model.forward(input);
        let probs = softmax(&logits);
        let loss = -probs[target].ln().max(-10.0);
        total_loss += loss;

        // Simple gradient: increase correct class probability
        let mut grad = vec![0.0f32; logits.len()];
        for (i, &p) in probs.iter().enumerate() {
            grad[i] = if i == target { p - 1.0 } else { p };
        }

        // Update output layer
        for j in 0..model.out.out_dim {
            model.out.bias[j] -= lr * grad[j];
            for i in 0..model.out.in_dim {
                let idx = i * model.out.out_dim + j;
                model.out.weights[idx] -= lr * grad[j] * 0.01;
            }
        }
    }

    total_loss / inputs.len() as f32
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
    println!("║  Species Expert NN (112D) - Universal Fallback                     ║");
    println!("║  Input: 112D (Physics + Macro + Micro)                                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let start = Instant::now();

    // Load data
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

    // Create normalized datasets
    let train_features: Vec<Vec<f32>> = train_indices.iter()
        .map(|&i| all_features[i].iter().enumerate()
            .map(|(j, &v)| (v - feature_means[j]) / feature_stds[j])
            .collect())
        .collect();
    let train_labels: Vec<usize> = train_indices.iter().map(|&i| all_labels[i]).collect();

    let val_features: Vec<Vec<f32>> = val_indices.iter()
        .map(|&i| all_features[i].iter().enumerate()
            .map(|(j, &v)| (v - feature_means[j]) / feature_stds[j])
            .collect())
        .collect();
    let val_labels: Vec<usize> = val_indices.iter().map(|&i| all_labels[i]).collect();

    // Initialize model
    println!("\nInitializing model...");
    let mut model = SpeciesExpertNN::new(num_classes);

    let mut best_val_acc = 0.0;
    let mut patience_counter = 0;

    println!("Training...");
    for epoch in 0..EPOCHS {
        // Shuffle
        let mut indices: Vec<usize> = (0..train_features.len()).collect();
        for i in 0..indices.len() {
            let j = rng.next_usize(indices.len());
            indices.swap(i, j);
        }

        // Train batches
        let mut total_loss = 0.0;
        let mut n_batches = 0;

        for batch_start in (0..train_features.len()).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(train_features.len());
            let batch_features: Vec<Vec<f32>> = indices[batch_start..batch_end]
                .iter().map(|&i| train_features[i].clone()).collect();
            let batch_labels: Vec<usize> = indices[batch_start..batch_end]
                .iter().map(|&i| train_labels[i]).collect();

            total_loss += train_batch(&mut model, &batch_features, &batch_labels, LEARNING_RATE);
            n_batches += 1;
        }

        // Evaluate
        let val_preds: Vec<usize> = val_features.iter().map(|f| model.predict(f)).collect();
        let val_correct = val_preds.iter().zip(&val_labels).filter(|(p, y)| *p == *y).count();
        let val_acc = val_correct as f64 / val_labels.len() as f64;

        if epoch % 5 == 0 {
            println!("Epoch {:2}: Loss={:.4} Val={:.2}%", epoch, total_loss / n_batches as f32, val_acc * 100.0);
        }

        // Early stopping
        if val_acc > best_val_acc {
            best_val_acc = val_acc;
            patience_counter = 0;
            fs::write("species_expert_112d.bin", bincode::serialize(&model)?)?;
        } else {
            patience_counter += 1;
            if patience_counter >= PATIENCE {
                println!("Early stopping at epoch {}", epoch);
                break;
            }
        }
    }

    // Final evaluation
    let model: SpeciesExpertNN = bincode::deserialize(&fs::read("species_expert_112d.bin")?)?;
    let val_preds: Vec<usize> = val_features.iter().map(|f| model.predict(f)).collect();
    let val_correct = val_preds.iter().zip(&val_labels).filter(|(p, y)| *p == *y).count();
    let final_acc = val_correct as f64 / val_labels.len() as f64;

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

    println!("\nSaved: species_expert_112d.bin, species_expert_112d.json");

    Ok(())
}
