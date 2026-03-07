//! Curriculum Neural Network Training (112D Features)
//! ===================================================
//!
//! Implements Progressive/Hierarchical Training:
//! - Phase 1 (Physics): Train only on features 0-45 (46D base physics)
//! - Phase 2 (Macro): Freeze physics, train on features 46-75 (30D macro texture)
//! - Phase 3 (Micro): Freeze physics+macro, train on features 76-111 (36D micro texture)
//!
//! This prevents the model from overfitting to high-variance micro features
//! before learning robust physics-based taxonomy discrimination.
//!
//! Usage:
//!   cargo run --release --bin train_curriculum_nn_112d

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::useless_vec)]

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

// =============================================================================
// Feature Dimensions (Hierarchical)
// =============================================================================

const PHYSICS_DIM: usize = 46; // Features 0-45: Base physics (F0, duration, MFCCs, etc.)
const MACRO_DIM: usize = 30; // Features 46-75: Macro texture (harmonics, pitch geometry)
const MICRO_DIM: usize = 36; // Features 76-111: Micro texture (AM/FM spectrum, rhythm)
const FEATURE_DIM: usize = 112; // Total: 46 + 30 + 36 = 112

// Hidden dimensions for each block
const PHYSICS_HIDDEN: usize = 256;
const MACRO_HIDDEN: usize = 128;
const MICRO_HIDDEN: usize = 64;
const OUTPUT_HIDDEN: usize = 64;

// Hyperparameters
const LEARNING_RATE: f32 = 1e-4;
const WEIGHT_DECAY: f32 = 0.01;
const DROPOUT_RATE: f32 = 0.1;
const LEAKY_RELU_SLOPE: f32 = 0.01;

// Curriculum phases
const PHYSICS_EPOCHS: usize = 30;
const MACRO_EPOCHS: usize = 30;
const MICRO_EPOCHS: usize = 40;
const PATIENCE: usize = 8;
const LR_DECAY_FACTOR: f32 = 0.5;
const LR_DECAY_PATIENCE: usize = 4;

// Adam constants
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;

// =============================================================================
// Model Structures
// =============================================================================

#[derive(Debug, Serialize)]
struct CurriculumNet {
    // Physics Block (Layer 1): processes features 0-45
    physics_weights: Vec<Vec<f32>>,
    physics_bias: Vec<f32>,
    physics_bn_gamma: Vec<f32>,
    physics_bn_beta: Vec<f32>,
    physics_bn_mean: Vec<f32>,
    physics_bn_var: Vec<f32>,

    // Macro Block (Layer 2): physics_hidden + macro_features -> macro_hidden
    macro_weights: Vec<Vec<f32>>,
    macro_bias: Vec<f32>,
    macro_bn_gamma: Vec<f32>,
    macro_bn_beta: Vec<f32>,
    macro_bn_mean: Vec<f32>,
    macro_bn_var: Vec<f32>,

    // Micro Block (Layer 3): macro_hidden + micro_features -> micro_hidden
    micro_weights: Vec<Vec<f32>>,
    micro_bias: Vec<f32>,
    micro_bn_gamma: Vec<f32>,
    micro_bn_beta: Vec<f32>,
    micro_bn_mean: Vec<f32>,
    micro_bn_var: Vec<f32>,

    // Output Block: micro_hidden -> output_hidden -> n_classes
    output_weights_1: Vec<Vec<f32>>,
    output_bias_1: Vec<f32>,
    output_weights_2: Vec<Vec<f32>>,
    output_bias_2: Vec<f32>,

    // Normalization
    feature_means: Vec<f32>,
    feature_stds: Vec<f32>,

    // Metadata
    n_classes: usize,
    idx_to_label: Vec<String>,

    // Training state (not serialized)
    #[serde(skip)]
    physics_hidden: Vec<f32>,
    #[serde(skip)]
    macro_hidden: Vec<f32>,
    #[serde(skip)]
    micro_hidden: Vec<f32>,
    #[serde(skip)]
    output_hidden: Vec<f32>,

    // Frozen flags for curriculum
    #[serde(skip)]
    physics_frozen: bool,
    #[serde(skip)]
    macro_frozen: bool,

    // Adam state (simplified - store per layer)
    #[serde(skip)]
    adam_state: AdamState,
}

#[derive(Debug, Default)]
struct AdamState {
    t: usize,
    // Physics
    physics_m_w: Vec<Vec<f32>>,
    physics_v_w: Vec<Vec<f32>>,
    physics_m_b: Vec<f32>,
    physics_v_b: Vec<f32>,
    // Macro
    macro_m_w: Vec<Vec<f32>>,
    macro_v_w: Vec<Vec<f32>>,
    macro_m_b: Vec<f32>,
    macro_v_b: Vec<f32>,
    // Micro
    micro_m_w: Vec<Vec<f32>>,
    micro_v_w: Vec<Vec<f32>>,
    micro_m_b: Vec<f32>,
    micro_v_b: Vec<f32>,
    // Output
    output_m_w1: Vec<Vec<f32>>,
    output_v_w1: Vec<Vec<f32>>,
    output_m_b1: Vec<f32>,
    output_v_b1: Vec<f32>,
    output_m_w2: Vec<Vec<f32>>,
    output_v_w2: Vec<Vec<f32>>,
    output_m_b2: Vec<f32>,
    output_v_b2: Vec<f32>,
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
    if x > 0.0 {
        x
    } else {
        x * slope
    }
}

fn leaky_relu_derivative(x: f32, slope: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        slope
    }
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|&v| v / sum).collect()
}

// =============================================================================
// CurriculumNet Implementation
// =============================================================================

impl CurriculumNet {
    fn new(n_classes: usize, idx_to_label: Vec<String>) -> Self {
        // Physics block: PHYSICS_DIM -> PHYSICS_HIDDEN
        let physics_weights: Vec<Vec<f32>> = (0..PHYSICS_HIDDEN)
            .map(|_| (0..PHYSICS_DIM).map(|_| he_init(PHYSICS_DIM)).collect())
            .collect();
        let physics_bias = vec![0.0; PHYSICS_HIDDEN];
        let physics_bn_gamma = vec![1.0; PHYSICS_HIDDEN];
        let physics_bn_beta = vec![0.0; PHYSICS_HIDDEN];
        let physics_bn_mean = vec![0.0; PHYSICS_HIDDEN];
        let physics_bn_var = vec![1.0; PHYSICS_HIDDEN];

        // Macro block: (PHYSICS_HIDDEN + MACRO_DIM) -> MACRO_HIDDEN
        let macro_input_dim = PHYSICS_HIDDEN + MACRO_DIM;
        let macro_weights: Vec<Vec<f32>> = (0..MACRO_HIDDEN)
            .map(|_| (0..macro_input_dim).map(|_| he_init(macro_input_dim)).collect())
            .collect();
        let macro_bias = vec![0.0; MACRO_HIDDEN];
        let macro_bn_gamma = vec![1.0; MACRO_HIDDEN];
        let macro_bn_beta = vec![0.0; MACRO_HIDDEN];
        let macro_bn_mean = vec![0.0; MACRO_HIDDEN];
        let macro_bn_var = vec![1.0; MACRO_HIDDEN];

        // Micro block: (MACRO_HIDDEN + MICRO_DIM) -> MICRO_HIDDEN
        let micro_input_dim = MACRO_HIDDEN + MICRO_DIM;
        let micro_weights: Vec<Vec<f32>> = (0..MICRO_HIDDEN)
            .map(|_| (0..micro_input_dim).map(|_| he_init(micro_input_dim)).collect())
            .collect();
        let micro_bias = vec![0.0; MICRO_HIDDEN];
        let micro_bn_gamma = vec![1.0; MICRO_HIDDEN];
        let micro_bn_beta = vec![0.0; MICRO_HIDDEN];
        let micro_bn_mean = vec![0.0; MICRO_HIDDEN];
        let micro_bn_var = vec![1.0; MICRO_HIDDEN];

        // Output block: MICRO_HIDDEN -> OUTPUT_HIDDEN -> n_classes
        let output_weights_1: Vec<Vec<f32>> = (0..OUTPUT_HIDDEN)
            .map(|_| (0..MICRO_HIDDEN).map(|_| he_init(MICRO_HIDDEN)).collect())
            .collect();
        let output_bias_1 = vec![0.0; OUTPUT_HIDDEN];
        let output_weights_2: Vec<Vec<f32>> = (0..n_classes)
            .map(|_| (0..OUTPUT_HIDDEN).map(|_| he_init(OUTPUT_HIDDEN)).collect())
            .collect();
        let output_bias_2 = vec![0.0; n_classes];

        // Initialize Adam state
        let adam_state = AdamState {
            t: 0,
            physics_m_w: vec![vec![0.0; PHYSICS_DIM]; PHYSICS_HIDDEN],
            physics_v_w: vec![vec![0.0; PHYSICS_DIM]; PHYSICS_HIDDEN],
            physics_m_b: vec![0.0; PHYSICS_HIDDEN],
            physics_v_b: vec![0.0; PHYSICS_HIDDEN],
            macro_m_w: vec![vec![0.0; macro_input_dim]; MACRO_HIDDEN],
            macro_v_w: vec![vec![0.0; macro_input_dim]; MACRO_HIDDEN],
            macro_m_b: vec![0.0; MACRO_HIDDEN],
            macro_v_b: vec![0.0; MACRO_HIDDEN],
            micro_m_w: vec![vec![0.0; micro_input_dim]; MICRO_HIDDEN],
            micro_v_w: vec![vec![0.0; micro_input_dim]; MICRO_HIDDEN],
            micro_m_b: vec![0.0; MICRO_HIDDEN],
            micro_v_b: vec![0.0; MICRO_HIDDEN],
            output_m_w1: vec![vec![0.0; MICRO_HIDDEN]; OUTPUT_HIDDEN],
            output_v_w1: vec![vec![0.0; MICRO_HIDDEN]; OUTPUT_HIDDEN],
            output_m_b1: vec![0.0; OUTPUT_HIDDEN],
            output_v_b1: vec![0.0; OUTPUT_HIDDEN],
            output_m_w2: vec![vec![0.0; OUTPUT_HIDDEN]; n_classes],
            output_v_w2: vec![vec![0.0; OUTPUT_HIDDEN]; n_classes],
            output_m_b2: vec![0.0; n_classes],
            output_v_b2: vec![0.0; n_classes],
        };

        Self {
            physics_weights,
            physics_bias,
            physics_bn_gamma,
            physics_bn_beta,
            physics_bn_mean,
            physics_bn_var,
            macro_weights,
            macro_bias,
            macro_bn_gamma,
            macro_bn_beta,
            macro_bn_mean,
            macro_bn_var,
            micro_weights,
            micro_bias,
            micro_bn_gamma,
            micro_bn_beta,
            micro_bn_mean,
            micro_bn_var,
            output_weights_1,
            output_bias_1,
            output_weights_2,
            output_bias_2,
            feature_means: vec![0.0; FEATURE_DIM],
            feature_stds: vec![1.0; FEATURE_DIM],
            n_classes,
            idx_to_label,
            physics_hidden: vec![0.0; PHYSICS_HIDDEN],
            macro_hidden: vec![0.0; MACRO_HIDDEN],
            micro_hidden: vec![0.0; MICRO_HIDDEN],
            output_hidden: vec![0.0; OUTPUT_HIDDEN],
            physics_frozen: false,
            macro_frozen: false,
            adam_state,
        }
    }

    fn forward(&mut self, physics_input: &[f32], macro_input: &[f32], micro_input: &[f32], training: bool) -> Vec<f32> {
        // === Physics Block ===
        let mut z1 = self.physics_bias.clone();
        for i in 0..PHYSICS_HIDDEN {
            for (j, &x) in physics_input.iter().enumerate() {
                z1[i] += self.physics_weights[i][j] * x;
            }
        }

        // BatchNorm + LeakyReLU
        for i in 0..PHYSICS_HIDDEN {
            self.physics_hidden[i] = leaky_relu(
                self.physics_bn_gamma[i] * (z1[i] - self.physics_bn_mean[i]) / (self.physics_bn_var[i] + 1e-5).sqrt()
                    + self.physics_bn_beta[i],
                LEAKY_RELU_SLOPE,
            );
        }

        // Dropout
        if training && !self.physics_frozen {
            for i in 0..PHYSICS_HIDDEN {
                if rand_f32() < DROPOUT_RATE {
                    self.physics_hidden[i] = 0.0;
                }
            }
        }

        // === Macro Block: concat(physics_hidden, macro_features) ===
        let mut macro_concat = Vec::with_capacity(PHYSICS_HIDDEN + MACRO_DIM);
        macro_concat.extend_from_slice(&self.physics_hidden);
        macro_concat.extend_from_slice(macro_input);

        let mut z2 = self.macro_bias.clone();
        for i in 0..MACRO_HIDDEN {
            for (j, &x) in macro_concat.iter().enumerate() {
                z2[i] += self.macro_weights[i][j] * x;
            }
        }

        for i in 0..MACRO_HIDDEN {
            self.macro_hidden[i] = leaky_relu(
                self.macro_bn_gamma[i] * (z2[i] - self.macro_bn_mean[i]) / (self.macro_bn_var[i] + 1e-5).sqrt()
                    + self.macro_bn_beta[i],
                LEAKY_RELU_SLOPE,
            );
        }

        if training && !self.macro_frozen {
            for i in 0..MACRO_HIDDEN {
                if rand_f32() < DROPOUT_RATE {
                    self.macro_hidden[i] = 0.0;
                }
            }
        }

        // === Micro Block: concat(macro_hidden, micro_features) ===
        let mut micro_concat = Vec::with_capacity(MACRO_HIDDEN + MICRO_DIM);
        micro_concat.extend_from_slice(&self.macro_hidden);
        micro_concat.extend_from_slice(micro_input);

        let mut z3 = self.micro_bias.clone();
        for i in 0..MICRO_HIDDEN {
            for (j, &x) in micro_concat.iter().enumerate() {
                z3[i] += self.micro_weights[i][j] * x;
            }
        }

        for i in 0..MICRO_HIDDEN {
            self.micro_hidden[i] = leaky_relu(
                self.micro_bn_gamma[i] * (z3[i] - self.micro_bn_mean[i]) / (self.micro_bn_var[i] + 1e-5).sqrt()
                    + self.micro_bn_beta[i],
                LEAKY_RELU_SLOPE,
            );
        }

        if training {
            for i in 0..MICRO_HIDDEN {
                if rand_f32() < DROPOUT_RATE {
                    self.micro_hidden[i] = 0.0;
                }
            }
        }

        // === Output Block ===
        let mut z4 = self.output_bias_1.clone();
        for i in 0..OUTPUT_HIDDEN {
            for (j, &h) in self.micro_hidden.iter().enumerate() {
                z4[i] += self.output_weights_1[i][j] * h;
            }
        }

        for i in 0..OUTPUT_HIDDEN {
            self.output_hidden[i] = leaky_relu(z4[i], LEAKY_RELU_SLOPE);
        }

        // Final output layer
        let mut output = self.output_bias_2.clone();
        for i in 0..self.n_classes {
            for (j, &h) in self.output_hidden.iter().enumerate() {
                output[i] += self.output_weights_2[i][j] * h;
            }
        }

        output
    }

    fn predict(&mut self, full_input: &[f32]) -> (usize, String) {
        // Split input into hierarchical components
        let physics_input: Vec<f32> = full_input[0..PHYSICS_DIM]
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - self.feature_means[i]) / self.feature_stds[i].max(1e-8))
            .collect();

        let macro_input: Vec<f32> = full_input[PHYSICS_DIM..(PHYSICS_DIM + MACRO_DIM)]
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - self.feature_means[PHYSICS_DIM + i]) / self.feature_stds[PHYSICS_DIM + i].max(1e-8))
            .collect();

        let micro_input: Vec<f32> = full_input[(PHYSICS_DIM + MACRO_DIM)..]
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                (v - self.feature_means[PHYSICS_DIM + MACRO_DIM + i])
                    / self.feature_stds[PHYSICS_DIM + MACRO_DIM + i].max(1e-8)
            })
            .collect();

        let output = self.forward(&physics_input, &macro_input, &micro_input, false);
        let pred_class = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let label = self
            .idx_to_label
            .get(pred_class)
            .cloned()
            .unwrap_or_else(|| format!("class_{}", pred_class));

        (pred_class, label)
    }

    fn train_step(
        &mut self,
        physics_input: &[f32],
        macro_input: &[f32],
        micro_input: &[f32],
        label: usize,
        class_weight: f32,
        lr: f32,
    ) -> f32 {
        self.adam_state.t += 1;
        let t = self.adam_state.t;

        // Forward pass
        let output = self.forward(physics_input, macro_input, micro_input, true);
        let probs = softmax(&output);
        let loss = -class_weight * (probs[label] + 1e-10).ln();

        // Output layer gradient
        let mut output_grad = probs.clone();
        output_grad[label] -= 1.0;
        for g in &mut output_grad {
            *g *= class_weight;
        }

        // Backprop through output layer 2
        let mut hidden4_grad = vec![0.0; OUTPUT_HIDDEN];
        for i in 0..self.n_classes {
            for j in 0..OUTPUT_HIDDEN {
                hidden4_grad[j] += output_grad[i] * self.output_weights_2[i][j];
            }
        }

        adam_update_layer(
            &mut self.output_weights_2,
            &mut self.output_bias_2,
            &mut self.adam_state.output_m_w2,
            &mut self.adam_state.output_v_w2,
            &mut self.adam_state.output_m_b2,
            &mut self.adam_state.output_v_b2,
            &output_grad,
            &self.output_hidden,
            lr,
            t,
        );

        // Backprop through LeakyReLU
        for i in 0..OUTPUT_HIDDEN {
            hidden4_grad[i] *= leaky_relu_derivative(self.output_hidden[i], LEAKY_RELU_SLOPE);
        }

        // Backprop through output layer 1
        let mut micro_grad = vec![0.0; MICRO_HIDDEN];
        for i in 0..OUTPUT_HIDDEN {
            for j in 0..MICRO_HIDDEN {
                micro_grad[j] += hidden4_grad[i] * self.output_weights_1[i][j];
            }
        }

        adam_update_layer(
            &mut self.output_weights_1,
            &mut self.output_bias_1,
            &mut self.adam_state.output_m_w1,
            &mut self.adam_state.output_v_w1,
            &mut self.adam_state.output_m_b1,
            &mut self.adam_state.output_v_b1,
            &hidden4_grad,
            &self.micro_hidden,
            lr,
            t,
        );

        // Backprop through micro block (LeakyReLU + BN)
        for i in 0..MICRO_HIDDEN {
            micro_grad[i] *= leaky_relu_derivative(self.micro_hidden[i], LEAKY_RELU_SLOPE);
        }

        // Micro input gradient (for backprop to macro)
        let micro_input_dim = MACRO_HIDDEN + MICRO_DIM;
        let mut micro_concat_grad = vec![0.0; micro_input_dim];
        for i in 0..MICRO_HIDDEN {
            for j in 0..micro_input_dim {
                micro_concat_grad[j] += micro_grad[i] * self.micro_weights[i][j];
            }
        }

        // Only update micro weights if not frozen
        if !self.macro_frozen {
            let micro_concat: Vec<f32> = [self.macro_hidden.as_slice(), micro_input].concat();
            adam_update_layer(
                &mut self.micro_weights,
                &mut self.micro_bias,
                &mut self.adam_state.micro_m_w,
                &mut self.adam_state.micro_v_w,
                &mut self.adam_state.micro_m_b,
                &mut self.adam_state.micro_v_b,
                &micro_grad,
                &micro_concat,
                lr,
                t,
            );
        }

        // Gradient for macro hidden (first part of micro_concat_grad)
        let mut macro_grad: Vec<f32> = micro_concat_grad[0..MACRO_HIDDEN].to_vec();

        // Backprop through macro block
        for i in 0..MACRO_HIDDEN {
            macro_grad[i] *= leaky_relu_derivative(self.macro_hidden[i], LEAKY_RELU_SLOPE);
        }

        // Macro input gradient
        let macro_input_dim = PHYSICS_HIDDEN + MACRO_DIM;
        let mut macro_concat_grad = vec![0.0; macro_input_dim];
        for i in 0..MACRO_HIDDEN {
            for j in 0..macro_input_dim {
                macro_concat_grad[j] += macro_grad[i] * self.macro_weights[i][j];
            }
        }

        // Only update macro weights if physics is not frozen
        if !self.physics_frozen {
            let macro_concat: Vec<f32> = [self.physics_hidden.as_slice(), macro_input].concat();
            adam_update_layer(
                &mut self.macro_weights,
                &mut self.macro_bias,
                &mut self.adam_state.macro_m_w,
                &mut self.adam_state.macro_v_w,
                &mut self.adam_state.macro_m_b,
                &mut self.adam_state.macro_v_b,
                &macro_grad,
                &macro_concat,
                lr,
                t,
            );
        }

        // Gradient for physics hidden (first part of macro_concat_grad)
        let mut physics_grad: Vec<f32> = macro_concat_grad[0..PHYSICS_HIDDEN].to_vec();

        // Backprop through physics block
        for i in 0..PHYSICS_HIDDEN {
            physics_grad[i] *= leaky_relu_derivative(self.physics_hidden[i], LEAKY_RELU_SLOPE);
        }

        // Only update physics weights if not frozen
        if !self.physics_frozen {
            adam_update_layer(
                &mut self.physics_weights,
                &mut self.physics_bias,
                &mut self.adam_state.physics_m_w,
                &mut self.adam_state.physics_v_w,
                &mut self.adam_state.physics_m_b,
                &mut self.adam_state.physics_v_b,
                &physics_grad,
                physics_input,
                lr,
                t,
            );
        }

        loss
    }

    fn save(&self, path: &Path) -> Result<()> {
        let model_json = serde_json::to_string_pretty(self)?;
        fs::write(path, model_json)?;
        Ok(())
    }
}

fn adam_update_layer(
    weights: &mut [Vec<f32>],
    biases: &mut [f32],
    m_w: &mut [Vec<f32>],
    v_w: &mut [Vec<f32>],
    m_b: &mut [f32],
    v_b: &mut [f32],
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

            m_w[i][j] = BETA1 * m_w[i][j] + (1.0 - BETA1) * g;
            v_w[i][j] = BETA2 * v_w[i][j] + (1.0 - BETA2) * g * g;

            let m_hat = m_w[i][j] / bias_correction1;
            let v_hat = v_w[i][j] / bias_correction2;

            weights[i][j] -= lr * m_hat / (v_hat.sqrt() + EPS);
        }

        let g = grad[i];
        m_b[i] = BETA1 * m_b[i] + (1.0 - BETA1) * g;
        v_b[i] = BETA2 * v_b[i] + (1.0 - BETA2) * g * g;

        let m_hat = m_b[i] / bias_correction1;
        let v_hat = v_b[i] / bias_correction2;

        biases[i] -= lr * m_hat / (v_hat.sqrt() + EPS);
    }
}

// =============================================================================
// Training Functions
// =============================================================================

struct TrainingData {
    features: Vec<Vec<f32>>,
    labels: Vec<usize>,
    label_indices: Vec<String>,
    n_classes: usize,
    idx_to_label: Vec<String>,
    class_weights: Vec<f32>,
    train_indices: Vec<usize>,
    val_indices: Vec<usize>,
}

fn load_data() -> Result<TrainingData> {
    let manifest_path = "beans_zero_full_manifest.json";
    println!("Loading manifest from: {}", manifest_path);
    let manifest_data = fs::read_to_string(manifest_path)?;
    let manifest: BeansManifest = serde_json::from_str(&manifest_data)?;
    println!("  Total samples: {}", manifest.samples.len());

    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_manifest_path = cache_dir.join("cache_manifest.json");
    println!("Loading cache manifest from: {:?}", cache_manifest_path);
    let cache_data = fs::read_to_string(&cache_manifest_path)?;
    let cache_manifest: CacheManifest = serde_json::from_str(&cache_data)?;
    println!("  Cached features available: {}", cache_manifest.entries.len());

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
                            all_features.push(features);
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
    let mut unique_labels: Vec<String> = all_labels.to_vec();
    unique_labels.sort();
    unique_labels.dedup();
    let n_classes = unique_labels.len();
    let mut label_to_idx = HashMap::new();
    for (idx, label) in unique_labels.iter().enumerate() {
        label_to_idx.insert(label.clone(), idx);
    }
    let idx_to_label: Vec<String> = unique_labels.clone();
    println!("  Classes: {}", n_classes);

    // Compute class weights
    let mut class_counts = vec![0usize; n_classes];
    let label_indices: Vec<usize> = all_labels.iter().map(|l| *label_to_idx.get(l).unwrap_or(&0)).collect();
    for &idx in &label_indices {
        class_counts[idx] += 1;
    }

    let total_samples = all_labels.len() as f32;
    let class_weights: Vec<f32> = class_counts
        .iter()
        .map(|&count| {
            if count == 0 {
                1.0
            } else {
                (total_samples / (n_classes as f32 * count as f32)).sqrt().min(10.0)
            }
        })
        .collect();

    // Split
    let n_train = (all_features.len() as f32 * 0.9) as usize;

    let mut indices: Vec<usize> = (0..all_features.len()).collect();
    for i in 0..indices.len() {
        let j = (rand_u32() as usize) % indices.len();
        indices.swap(i, j);
    }

    let train_indices: Vec<usize> = indices[..n_train].to_vec();
    let val_indices: Vec<usize> = indices[n_train..].to_vec();

    Ok(TrainingData {
        features: all_features,
        labels: label_indices,
        label_indices: all_labels,
        n_classes,
        idx_to_label,
        class_weights,
        train_indices,
        val_indices,
    })
}

fn train_phase(
    model: &mut CurriculumNet,
    data: &TrainingData,
    phase_name: &str,
    epochs: usize,
    lr: f32,
    feature_means: &[f32],
    feature_stds: &[f32],
) -> Result<f32> {
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  {} Phase{:<55}║", phase_name, "");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Physics Frozen: {:<5}  Macro Frozen: {:<5}                ║",
        model.physics_frozen, model.macro_frozen
    );
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    let mut best_val_acc = 0.0f32;
    let mut best_epoch = 0;
    let mut patience_counter = 0;
    let mut lr_decay_counter = 0;
    let mut current_lr = lr;

    for epoch in 0..epochs {
        // Shuffle
        let mut epoch_indices = data.train_indices.clone();
        for i in 0..epoch_indices.len() {
            let j = (rand_u32() as usize) % epoch_indices.len();
            epoch_indices.swap(i, j);
        }

        // Train
        let mut total_loss = 0.0f32;
        let mut train_correct = 0usize;

        for &i in &epoch_indices {
            let full_features = &data.features[i];

            // Normalize and split features
            let physics_input: Vec<f32> = full_features[0..PHYSICS_DIM]
                .iter()
                .enumerate()
                .map(|(j, &v)| (v - feature_means[j]) / feature_stds[j].max(1e-8))
                .collect();

            let macro_input: Vec<f32> = full_features[PHYSICS_DIM..(PHYSICS_DIM + MACRO_DIM)]
                .iter()
                .enumerate()
                .map(|(j, &v)| (v - feature_means[PHYSICS_DIM + j]) / feature_stds[PHYSICS_DIM + j].max(1e-8))
                .collect();

            let micro_input: Vec<f32> = full_features[(PHYSICS_DIM + MACRO_DIM)..]
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    (v - feature_means[PHYSICS_DIM + MACRO_DIM + j])
                        / feature_stds[PHYSICS_DIM + MACRO_DIM + j].max(1e-8)
                })
                .collect();

            let label_idx = data.labels[i];
            let weight = data.class_weights[label_idx];

            let loss = model.train_step(
                &physics_input,
                &macro_input,
                &micro_input,
                label_idx,
                weight,
                current_lr,
            );
            total_loss += loss;

            let (_, pred_label) = model.predict(&data.features[i]);
            if pred_label == data.label_indices[i] {
                train_correct += 1;
            }
        }

        let train_acc = train_correct as f32 / data.train_indices.len() as f32 * 100.0;
        let avg_loss = total_loss / data.train_indices.len() as f32;

        // Validate
        let mut val_correct = 0usize;
        for &i in &data.val_indices {
            let (_, pred_label) = model.predict(&data.features[i]);
            if pred_label == data.label_indices[i] {
                val_correct += 1;
            }
        }
        let val_acc = val_correct as f32 / data.val_indices.len() as f32 * 100.0;

        // Check improvement
        if val_acc > best_val_acc {
            best_val_acc = val_acc;
            best_epoch = epoch + 1;
            patience_counter = 0;
            lr_decay_counter = 0;
        } else {
            patience_counter += 1;
            lr_decay_counter += 1;
        }

        // LR decay
        if lr_decay_counter >= LR_DECAY_PATIENCE {
            current_lr *= LR_DECAY_FACTOR;
            lr_decay_counter = 0;
            println!("  Reducing learning rate to {:.2e}", current_lr);
        }

        // Progress
        if (epoch + 1) % 5 == 0 || epoch == 0 {
            println!(
                "  Epoch {:3}/{}: Loss={:.4}, Train={:.1}%, Val={:.1}% | Best={:.1}% (epoch {})",
                epoch + 1,
                epochs,
                avg_loss,
                train_acc,
                val_acc,
                best_val_acc,
                best_epoch
            );
        }

        // Early stopping
        if patience_counter >= PATIENCE {
            println!(
                "\n  Early stopping at epoch {} (no improvement for {} epochs)",
                epoch + 1,
                PATIENCE
            );
            break;
        }
    }

    println!("\n  Phase complete. Best Val Acc: {:.2}%", best_val_acc);
    Ok(best_val_acc)
}

// =============================================================================
// Main
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Curriculum Neural Network Training (112D Features)              ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Architecture:");
    println!(
        "  Physics Block: {}D -> {}D (Base physics features)",
        PHYSICS_DIM, PHYSICS_HIDDEN
    );
    println!(
        "  Macro Block:   {}D + {}D -> {}D (Texture features)",
        PHYSICS_HIDDEN, MACRO_DIM, MACRO_HIDDEN
    );
    println!(
        "  Micro Block:   {}D + {}D -> {}D (Fine features)",
        MACRO_HIDDEN, MICRO_DIM, MICRO_HIDDEN
    );
    println!("  Output:        {}D -> {}D -> n_classes", MICRO_HIDDEN, OUTPUT_HIDDEN);
    println!();
    println!("Curriculum:");
    println!(
        "  Phase 1 (Physics): {} epochs - Train physics block only",
        PHYSICS_EPOCHS
    );
    println!(
        "  Phase 2 (Macro):   {} epochs - Freeze physics, train macro",
        MACRO_EPOCHS
    );
    println!(
        "  Phase 3 (Micro):   {} epochs - Freeze physics+macro, train micro",
        MICRO_EPOCHS
    );
    println!();

    let start = Instant::now();

    // Load data
    let data = load_data()?;

    // Compute normalization
    let mut feature_means = vec![0.0f32; FEATURE_DIM];
    let mut feature_stds = vec![0.0f32; FEATURE_DIM];

    for &i in &data.train_indices {
        for (j, &v) in data.features[i].iter().enumerate() {
            feature_means[j] += v;
        }
    }
    for j in 0..FEATURE_DIM {
        feature_means[j] /= data.train_indices.len() as f32;
    }

    for &i in &data.train_indices {
        for (j, &v) in data.features[i].iter().enumerate() {
            let diff = v - feature_means[j];
            feature_stds[j] += diff * diff;
        }
    }
    for j in 0..FEATURE_DIM {
        feature_stds[j] = (feature_stds[j] / data.train_indices.len() as f32).sqrt().max(1e-8);
    }

    // Initialize model
    let mut model = CurriculumNet::new(data.n_classes, data.idx_to_label.clone());
    model.feature_means = feature_means.clone();
    model.feature_stds = feature_stds.clone();

    // === PHASE 1: Train Physics Block ===
    model.physics_frozen = false;
    model.macro_frozen = true;
    train_phase(
        &mut model,
        &data,
        "Physics",
        PHYSICS_EPOCHS,
        LEARNING_RATE,
        &feature_means,
        &feature_stds,
    )?;

    // === PHASE 2: Train Macro Block (Physics frozen) ===
    model.physics_frozen = true;
    model.macro_frozen = false;
    train_phase(
        &mut model,
        &data,
        "Macro",
        MACRO_EPOCHS,
        LEARNING_RATE * 0.5,
        &feature_means,
        &feature_stds,
    )?;

    // === PHASE 3: Train Micro Block (Physics + Macro frozen) ===
    model.physics_frozen = true;
    model.macro_frozen = true;
    let final_acc = train_phase(
        &mut model,
        &data,
        "Micro",
        MICRO_EPOCHS,
        LEARNING_RATE * 0.25,
        &feature_means,
        &feature_stds,
    )?;

    // Save model
    model.save(Path::new("rosetta_net_112d_curriculum.json"))?;

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Curriculum Training Complete                                     ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Final Validation Accuracy: {:>8.2}%                           ║",
        final_acc
    );
    println!(
        "║  Total Time:                {:>8.1}s                            ║",
        start.elapsed().as_secs_f32()
    );
    println!("║  Model saved to: rosetta_net_112d_curriculum.json                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    Ok(())
}
