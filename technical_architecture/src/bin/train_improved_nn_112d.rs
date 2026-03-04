//! Train Improved Neural Network (112D Features with Cached Data)
//! ===============================================================
//!
//! Implements recommended improvements for NN training:
//! - Deeper architecture: 112 -> 512 -> 256 -> 128 -> output
//! - LeakyReLU activation (prevents dead neurons)
//! - AdamW optimizer with weight decay
//! - Learning rate scheduling (ReduceLROnPlateau)
//! - Weighted loss for class imbalance
//! - Batch normalization (simplified)
//! - Dropout (0.1)
//!
//! Usage:
//!   cargo run --release --bin train_improved_nn_112d

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

const FEATURE_DIM: usize = 112;

// Improved hyperparameters
const HIDDEN_DIM_1: usize = 512;     // First hidden layer
const HIDDEN_DIM_2: usize = 256;     // Second hidden layer
const HIDDEN_DIM_3: usize = 128;     // Third hidden layer
const LEARNING_RATE: f32 = 1e-4;     // Adam learning rate
const WEIGHT_DECAY: f32 = 0.01;      // AdamW weight decay
const DROPOUT_RATE: f32 = 0.1;       // Dropout probability
const LEAKY_RELU_SLOPE: f32 = 0.01;  // Negative slope for LeakyReLU
const EPOCHS: usize = 100;           // Maximum epochs
const PATIENCE: usize = 10;          // Early stopping patience
const LR_DECAY_FACTOR: f32 = 0.5;    // LR reduction factor
const LR_DECAY_PATIENCE: usize = 5;  // Epochs without improvement before LR decay

// Adam constants
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;

// =============================================================================
// Model Structures
// =============================================================================

#[derive(Debug, Serialize)]
struct ImprovedRosettaNet {
    // Layer 1: Input -> Hidden1
    weights_1: Vec<Vec<f32>>,
    bias_1: Vec<f32>,
    // BatchNorm1 parameters
    bn1_gamma: Vec<f32>,
    bn1_beta: Vec<f32>,
    bn1_running_mean: Vec<f32>,
    bn1_running_var: Vec<f32>,

    // Layer 2: Hidden1 -> Hidden2
    weights_2: Vec<Vec<f32>>,
    bias_2: Vec<f32>,
    // BatchNorm2 parameters
    bn2_gamma: Vec<f32>,
    bn2_beta: Vec<f32>,
    bn2_running_mean: Vec<f32>,
    bn2_running_var: Vec<f32>,

    // Layer 3: Hidden2 -> Hidden3
    weights_3: Vec<Vec<f32>>,
    bias_3: Vec<f32>,

    // Output layer
    weights_out: Vec<Vec<f32>>,
    bias_out: Vec<f32>,

    // Normalization parameters for input
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,

    // Label mapping
    n_classes: usize,
    idx_to_label: Vec<String>,

    #[serde(skip)]
    // Training state
    hidden1: Vec<f32>,
    hidden1_bn: Vec<f32>,
    hidden2: Vec<f32>,
    hidden2_bn: Vec<f32>,
    hidden3: Vec<f32>,

    // Adam state
    m_weights_1: Vec<Vec<f32>>,
    v_weights_1: Vec<Vec<f32>>,
    m_bias_1: Vec<f32>,
    v_bias_1: Vec<f32>,
    m_weights_2: Vec<Vec<f32>>,
    v_weights_2: Vec<Vec<f32>>,
    m_bias_2: Vec<f32>,
    v_bias_2: Vec<f32>,
    m_weights_3: Vec<Vec<f32>>,
    v_weights_3: Vec<Vec<f32>>,
    m_bias_3: Vec<f32>,
    v_bias_3: Vec<f32>,
    m_weights_out: Vec<Vec<f32>>,
    v_weights_out: Vec<Vec<f32>>,
    m_bias_out: Vec<f32>,
    v_bias_out: Vec<f32>,
    adam_t: usize,  // Time step for Adam
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
// Helper Functions
// =============================================================================

fn rand_f32() -> f32 {
    (rand_u32() as f64 / u32::MAX as f64) as f32
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

fn he_init(fan_in: usize) -> f32 {
    let scale = (2.0 / fan_in as f64).sqrt() as f32;
    (rand_f32() * 2.0 - 1.0) * scale
}

fn leaky_relu(x: f32, slope: f32) -> f32 {
    if x > 0.0 { x } else { x * slope }
}

fn leaky_relu_derivative(x: f32, slope: f32) -> f32 {
    if x > 0.0 { 1.0 } else { slope }
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|&v| v / sum).collect()
}

fn adamw_update(
    weights: &mut Vec<Vec<f32>>,
    biases: &mut Vec<f32>,
    m_weights: &mut Vec<Vec<f32>>,
    v_weights: &mut Vec<Vec<f32>>,
    m_biases: &mut Vec<f32>,
    v_biases: &mut Vec<f32>,
    grad: &[f32],
    input: &[f32],
    lr: f32,
    t: usize,
) {
    let bias_correction1 = 1.0 - BETA1.powi(t as i32);
    let bias_correction2 = 1.0 - BETA2.powi(t as i32);

    for i in 0..weights.len() {
        for j in 0..weights[0].len() {
            let g = grad[i] * input[j] + WEIGHT_DECAY * weights[i][j];

            m_weights[i][j] = BETA1 * m_weights[i][j] + (1.0 - BETA1) * g;
            v_weights[i][j] = BETA2 * v_weights[i][j] + (1.0 - BETA2) * g * g;

            let m_hat = m_weights[i][j] / bias_correction1;
            let v_hat = v_weights[i][j] / bias_correction2;

            weights[i][j] -= lr * m_hat / (v_hat.sqrt() + EPS);
        }

        let g = grad[i];
        m_biases[i] = BETA1 * m_biases[i] + (1.0 - BETA1) * g;
        v_biases[i] = BETA2 * v_biases[i] + (1.0 - BETA2) * g * g;

        let m_hat = m_biases[i] / bias_correction1;
        let v_hat = v_biases[i] / bias_correction2;

        biases[i] -= lr * m_hat / (v_hat.sqrt() + EPS);
    }
}

// =============================================================================
// ImprovedRosettaNet Implementation
// =============================================================================

impl ImprovedRosettaNet {
    fn new(input_dim: usize, n_classes: usize, idx_to_label: Vec<String>) -> Self {
        // Initialize weights with He initialization
        let weights_1: Vec<Vec<f32>> = (0..HIDDEN_DIM_1)
            .map(|_| (0..input_dim).map(|_| he_init(input_dim)).collect())
            .collect();
        let bias_1 = vec![0.0; HIDDEN_DIM_1];

        // BatchNorm1: Initialize gamma=1, beta=0
        let bn1_gamma = vec![1.0; HIDDEN_DIM_1];
        let bn1_beta = vec![0.0; HIDDEN_DIM_1];
        let bn1_running_mean = vec![0.0; HIDDEN_DIM_1];
        let bn1_running_var = vec![1.0; HIDDEN_DIM_1];

        let weights_2: Vec<Vec<f32>> = (0..HIDDEN_DIM_2)
            .map(|_| (0..HIDDEN_DIM_1).map(|_| he_init(HIDDEN_DIM_1)).collect())
            .collect();
        let bias_2 = vec![0.0; HIDDEN_DIM_2];

        let bn2_gamma = vec![1.0; HIDDEN_DIM_2];
        let bn2_beta = vec![0.0; HIDDEN_DIM_2];
        let bn2_running_mean = vec![0.0; HIDDEN_DIM_2];
        let bn2_running_var = vec![1.0; HIDDEN_DIM_2];

        let weights_3: Vec<Vec<f32>> = (0..HIDDEN_DIM_3)
            .map(|_| (0..HIDDEN_DIM_2).map(|_| he_init(HIDDEN_DIM_2)).collect())
            .collect();
        let bias_3 = vec![0.0; HIDDEN_DIM_3];

        let weights_out: Vec<Vec<f32>> = (0..n_classes)
            .map(|_| (0..HIDDEN_DIM_3).map(|_| he_init(HIDDEN_DIM_3)).collect())
            .collect();
        let bias_out = vec![0.0; n_classes];

        // Initialize Adam state with zeros
        let m_weights_1 = vec![vec![0.0; input_dim]; HIDDEN_DIM_1];
        let v_weights_1 = vec![vec![0.0; input_dim]; HIDDEN_DIM_1];
        let m_bias_1 = vec![0.0; HIDDEN_DIM_1];
        let v_bias_1 = vec![0.0; HIDDEN_DIM_1];

        let m_weights_2 = vec![vec![0.0; HIDDEN_DIM_1]; HIDDEN_DIM_2];
        let v_weights_2 = vec![vec![0.0; HIDDEN_DIM_1]; HIDDEN_DIM_2];
        let m_bias_2 = vec![0.0; HIDDEN_DIM_2];
        let v_bias_2 = vec![0.0; HIDDEN_DIM_2];

        let m_weights_3 = vec![vec![0.0; HIDDEN_DIM_2]; HIDDEN_DIM_3];
        let v_weights_3 = vec![vec![0.0; HIDDEN_DIM_2]; HIDDEN_DIM_3];
        let m_bias_3 = vec![0.0; HIDDEN_DIM_3];
        let v_bias_3 = vec![0.0; HIDDEN_DIM_3];

        let m_weights_out = vec![vec![0.0; HIDDEN_DIM_3]; n_classes];
        let v_weights_out = vec![vec![0.0; HIDDEN_DIM_3]; n_classes];
        let m_bias_out = vec![0.0; n_classes];
        let v_bias_out = vec![0.0; n_classes];

        Self {
            weights_1, bias_1, bn1_gamma, bn1_beta, bn1_running_mean, bn1_running_var,
            weights_2, bias_2, bn2_gamma, bn2_beta, bn2_running_mean, bn2_running_var,
            weights_3, bias_3,
            weights_out, bias_out,
            feature_means: vec![0.0; input_dim],
            feature_stds: vec![1.0; input_dim],
            n_classes,
            idx_to_label,
            hidden1: vec![0.0; HIDDEN_DIM_1],
            hidden1_bn: vec![0.0; HIDDEN_DIM_1],
            hidden2: vec![0.0; HIDDEN_DIM_2],
            hidden2_bn: vec![0.0; HIDDEN_DIM_2],
            hidden3: vec![0.0; HIDDEN_DIM_3],
            m_weights_1, v_weights_1, m_bias_1, v_bias_1,
            m_weights_2, v_weights_2, m_bias_2, v_bias_2,
            m_weights_3, v_weights_3, m_bias_3, v_bias_3,
            m_weights_out, v_weights_out, m_bias_out, v_bias_out,
            adam_t: 0,
        }
    }

    fn forward(&mut self, input: &[f32], training: bool) -> Vec<f32> {
        // === Layer 1: Linear -> BatchNorm -> LeakyReLU -> Dropout ===
        let mut z1 = self.bias_1.clone();
        for i in 0..HIDDEN_DIM_1 {
            for (j, &x) in input.iter().enumerate() {
                z1[i] += self.weights_1[i][j] * x;
            }
        }

        // BatchNorm1 (simplified - use running stats during inference)
        for i in 0..HIDDEN_DIM_1 {
            self.hidden1_bn[i] = self.bn1_gamma[i] * (z1[i] - self.bn1_running_mean[i])
                / (self.bn1_running_var[i] + 1e-5).sqrt() + self.bn1_beta[i];
        }

        // LeakyReLU
        for i in 0..HIDDEN_DIM_1 {
            self.hidden1[i] = leaky_relu(self.hidden1_bn[i], LEAKY_RELU_SLOPE);
        }

        // Dropout (only during training)
        if training {
            for i in 0..HIDDEN_DIM_1 {
                if rand_f32() < DROPOUT_RATE {
                    self.hidden1[i] = 0.0;
                }
            }
        }

        // === Layer 2: Linear -> BatchNorm -> LeakyReLU ===
        let mut z2 = self.bias_2.clone();
        for i in 0..HIDDEN_DIM_2 {
            for (j, &h) in self.hidden1.iter().enumerate() {
                z2[i] += self.weights_2[i][j] * h;
            }
        }

        for i in 0..HIDDEN_DIM_2 {
            self.hidden2_bn[i] = self.bn2_gamma[i] * (z2[i] - self.bn2_running_mean[i])
                / (self.bn2_running_var[i] + 1e-5).sqrt() + self.bn2_beta[i];
        }

        for i in 0..HIDDEN_DIM_2 {
            self.hidden2[i] = leaky_relu(self.hidden2_bn[i], LEAKY_RELU_SLOPE);
        }

        // === Layer 3: Linear -> LeakyReLU ===
        let mut z3 = self.bias_3.clone();
        for i in 0..HIDDEN_DIM_3 {
            for (j, &h) in self.hidden2.iter().enumerate() {
                z3[i] += self.weights_3[i][j] * h;
            }
        }

        for i in 0..HIDDEN_DIM_3 {
            self.hidden3[i] = leaky_relu(z3[i], LEAKY_RELU_SLOPE);
        }

        // === Output Layer ===
        let mut output = self.bias_out.clone();
        for i in 0..self.n_classes {
            for (j, &h) in self.hidden3.iter().enumerate() {
                output[i] += self.weights_out[i][j] * h;
            }
        }

        output
    }

    fn predict(&mut self, input: &[f32]) -> (usize, String) {
        // Normalize input
        let normalized: Vec<f32> = input
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - self.feature_means[i]) / self.feature_stds[i].max(1e-8))
            .collect();

        let output = self.forward(&normalized, false);
        let pred_class = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let label = self.idx_to_label
            .get(pred_class)
            .cloned()
            .unwrap_or_else(|| format!("class_{}", pred_class));

        (pred_class, label)
    }

    fn train_step(&mut self, input: &[f32], label: usize, class_weight: f32, lr: f32) -> f32 {
        self.adam_t += 1;
        let t = self.adam_t;

        // Forward pass
        let output = self.forward(input, true);

        // Compute softmax
        let probs = softmax(&output);

        // Compute weighted cross-entropy loss
        let loss = -class_weight * (probs[label] + 1e-10).ln();

        // Output layer gradient (with weight decay in AdamW)
        let mut output_grad = probs.clone();
        output_grad[label] -= 1.0;
        for g in &mut output_grad {
            *g *= class_weight;
        }

        // Backprop through output layer
        let mut hidden3_grad = vec![0.0; HIDDEN_DIM_3];
        for i in 0..self.n_classes {
            for j in 0..HIDDEN_DIM_3 {
                hidden3_grad[j] += output_grad[i] * self.weights_out[i][j];
            }
        }

        // AdamW update for output layer (inline)
        adamw_update(
            &mut self.weights_out, &mut self.bias_out,
            &mut self.m_weights_out, &mut self.v_weights_out,
            &mut self.m_bias_out, &mut self.v_bias_out,
            &output_grad, &self.hidden3, lr, t,
        );

        // Backprop through layer 3 (LeakyReLU)
        for i in 0..HIDDEN_DIM_3 {
            hidden3_grad[i] *= leaky_relu_derivative(self.hidden3[i], LEAKY_RELU_SLOPE);
        }

        // Backprop through layer 3 weights
        let mut hidden2_grad = vec![0.0; HIDDEN_DIM_2];
        for i in 0..HIDDEN_DIM_3 {
            for j in 0..HIDDEN_DIM_2 {
                hidden2_grad[j] += hidden3_grad[i] * self.weights_3[i][j];
            }
        }

        adamw_update(
            &mut self.weights_3, &mut self.bias_3,
            &mut self.m_weights_3, &mut self.v_weights_3,
            &mut self.m_bias_3, &mut self.v_bias_3,
            &hidden3_grad, &self.hidden2, lr, t,
        );

        // Backprop through layer 2 (LeakyReLU + BN)
        for i in 0..HIDDEN_DIM_2 {
            hidden2_grad[i] *= leaky_relu_derivative(self.hidden2[i], LEAKY_RELU_SLOPE);
        }

        // Backprop through BN2 (simplified)
        let mut hidden1_grad = vec![0.0; HIDDEN_DIM_1];
        for i in 0..HIDDEN_DIM_2 {
            for j in 0..HIDDEN_DIM_1 {
                hidden1_grad[j] += hidden2_grad[i] * self.weights_2[i][j];
            }
        }

        adamw_update(
            &mut self.weights_2, &mut self.bias_2,
            &mut self.m_weights_2, &mut self.v_weights_2,
            &mut self.m_bias_2, &mut self.v_bias_2,
            &hidden2_grad, &self.hidden1, lr, t,
        );

        // Backprop through layer 1 (LeakyReLU + BN)
        for i in 0..HIDDEN_DIM_1 {
            hidden1_grad[i] *= leaky_relu_derivative(self.hidden1[i], LEAKY_RELU_SLOPE);
        }

        adamw_update(
            &mut self.weights_1, &mut self.bias_1,
            &mut self.m_weights_1, &mut self.v_weights_1,
            &mut self.m_bias_1, &mut self.v_bias_1,
            &hidden1_grad, input, lr, t,
        );

        loss
    }

    fn save(&self, path: &Path) -> Result<()> {
        // Skip training state when saving
        let model_for_save = serde_json::to_string_pretty(self)?;
        fs::write(path, model_for_save)?;
        Ok(())
    }
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Improved Neural Network Training (112D Cached Features)         ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Architecture:");
    println!("  Input:      {}D", FEATURE_DIM);
    println!("  Hidden 1:   {}D (BN + LeakyReLU + Dropout)", HIDDEN_DIM_1);
    println!("  Hidden 2:   {}D (BN + LeakyReLU)", HIDDEN_DIM_2);
    println!("  Hidden 3:   {}D (LeakyReLU)", HIDDEN_DIM_3);
    println!("  Optimizer:  AdamW (lr={}, weight_decay={})", LEARNING_RATE, WEIGHT_DECAY);
    println!("  Dropout:    {}", DROPOUT_RATE);
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
    println!("Loading cache manifest from: {:?}", cache_manifest_path);
    let cache_data = fs::read_to_string(&cache_manifest_path)?;
    let cache_manifest: CacheManifest = serde_json::from_str(&cache_data)?;
    println!("  Cached features available: {}", cache_manifest.entries.len());

    // Load all features and labels
    println!("\nLoading features from cache...");
    let mut all_features: Vec<Vec<f32>> = Vec::new();
    let mut all_labels: Vec<String> = Vec::new();
    let mut hits = 0;

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
                            all_features.push(features);
                            all_labels.push(label);
                            hits += 1;
                            continue;
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
        label_to_idx.insert(label.clone(), idx);
    }
    println!("  Classes: {}", n_classes);

    // Compute class weights
    let mut class_counts = vec![0usize; n_classes];
    let label_indices: Vec<usize> = all_labels
        .iter()
        .map(|l| *label_to_idx.get(l).unwrap_or(&0))
        .collect();
    for &idx in &label_indices {
        class_counts[idx] += 1;
    }

    let total_samples = all_labels.len() as f32;
    let class_weights: Vec<f32> = class_counts
        .iter()
        .map(|&count| {
            if count == 0 { 1.0 } else {
                (total_samples / (n_classes as f32 * count as f32)).sqrt().min(10.0)
            }
        })
        .collect();

    // Split into train/validation (90/10)
    println!("\nSplitting data: 90% train, 10% validation...");
    let n_train = (all_features.len() as f32 * 0.9) as usize;

    // Shuffle
    let mut indices: Vec<usize> = (0..all_features.len()).collect();
    for i in 0..indices.len() {
        let j = (rand_u32() as usize) % indices.len();
        indices.swap(i, j);
    }

    let train_indices: Vec<usize> = indices[..n_train].to_vec();
    let val_indices: Vec<usize> = indices[n_train..].to_vec();

    // Compute normalization params from training set
    let mut feature_means = vec![0.0f32; FEATURE_DIM];
    let mut feature_stds = vec![0.0f32; FEATURE_DIM];

    for &i in &train_indices {
        for (j, &v) in all_features[i].iter().enumerate() {
            feature_means[j] += v;
        }
    }
    for j in 0..FEATURE_DIM {
        feature_means[j] /= train_indices.len() as f32;
    }

    for &i in &train_indices {
        for (j, &v) in all_features[i].iter().enumerate() {
            let diff = v - feature_means[j];
            feature_stds[j] += diff * diff;
        }
    }
    for j in 0..FEATURE_DIM {
        feature_stds[j] = (feature_stds[j] / train_indices.len() as f32).sqrt().max(1e-8);
    }

    // Initialize model
    let mut model = ImprovedRosettaNet::new(FEATURE_DIM, n_classes, unique_labels);
    model.feature_means = feature_means.clone();
    model.feature_stds = feature_stds.clone();

    // Training
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Training                                                         ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let mut best_val_acc = 0.0f32;
    let mut best_epoch = 0;
    let mut patience_counter = 0;
    let mut lr_decay_counter = 0;
    let mut current_lr = LEARNING_RATE;

    for epoch in 0..EPOCHS {
        // Shuffle training indices
        let mut epoch_indices = train_indices.clone();
        for i in 0..epoch_indices.len() {
            let j = (rand_u32() as usize) % epoch_indices.len();
            epoch_indices.swap(i, j);
        }

        // Train epoch
        let mut total_loss = 0.0f32;
        let mut train_correct = 0usize;

        for &i in &epoch_indices {
            let features: Vec<f32> = all_features[i]
                .iter()
                .enumerate()
                .map(|(j, &v)| (v - feature_means[j]) / feature_stds[j])
                .collect();

            let label_idx = label_indices[i];
            let weight = class_weights[label_idx];

            let loss = model.train_step(&features, label_idx, weight, current_lr);
            total_loss += loss;

            let (_, pred_label) = model.predict(&all_features[i]);
            if &pred_label == &all_labels[i] {
                train_correct += 1;
            }
        }

        let train_acc = train_correct as f32 / train_indices.len() as f32 * 100.0;
        let avg_loss = total_loss / train_indices.len() as f32;

        // Validation
        let mut val_correct = 0usize;
        for &i in &val_indices {
            let (_, pred_label) = model.predict(&all_features[i]);
            if &pred_label == &all_labels[i] {
                val_correct += 1;
            }
        }
        let val_acc = val_correct as f32 / val_indices.len() as f32 * 100.0;

        // Check for improvement
        if val_acc > best_val_acc {
            best_val_acc = val_acc;
            best_epoch = epoch + 1;
            patience_counter = 0;
            lr_decay_counter = 0;

            // Save best model
            model.save(Path::new("rosetta_net_112d_improved.json"))?;
        } else {
            patience_counter += 1;
            lr_decay_counter += 1;
        }

        // Learning rate decay
        if lr_decay_counter >= LR_DECAY_PATIENCE {
            current_lr *= LR_DECAY_FACTOR;
            lr_decay_counter = 0;
            println!("  Reducing learning rate to {:.2e}", current_lr);
        }

        // Print progress
        if (epoch + 1) % 5 == 0 || epoch == 0 {
            println!("  Epoch {:3}/{}: Loss={:.4}, Train={:.1}%, Val={:.1}% | Best={:.1}% (epoch {})",
                epoch + 1, EPOCHS, avg_loss, train_acc, val_acc, best_val_acc, best_epoch);
        }

        // Early stopping
        if patience_counter >= PATIENCE {
            println!("\n  Early stopping at epoch {} (no improvement for {} epochs)",
                epoch + 1, PATIENCE);
            break;
        }
    }

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Best Validation Accuracy: {:>8.2}%                             ║", best_val_acc);
    println!("║  Best Epoch:               {:<8}                                ║", best_epoch);
    println!("║  Total Time:               {:>8.1}s                              ║", start.elapsed().as_secs_f32());
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    println!("\nSaved best model to: rosetta_net_112d_improved.json");

    Ok(())
}
