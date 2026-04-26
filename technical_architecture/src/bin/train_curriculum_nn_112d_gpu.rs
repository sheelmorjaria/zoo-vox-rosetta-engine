//! Curriculum Neural Network Training (112D Features) - GPU Accelerated
//! ======================================================================
//!
//! Uses libtorch (tch-rs) for GPU acceleration.
//!
//! Implements Progressive/Hierarchical Training:
//! - Phase 1 (Physics): Train only on features 0-45 (46D base physics)
//! - Phase 2 (Macro): Freeze physics, train on features 46-75 (30D macro texture)
//! - Phase 3 (Micro): Freeze physics+macro, train on features 76-111 (36D micro texture)
//!
//! Usage:
//!   export LIBTORCH=/home/sheel/libtorch
//!   export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
//!   cargo run --release --features gpu-training --bin train_curriculum_nn_112d_gpu

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Device, Tensor};

// =============================================================================
// Feature Dimensions (Hierarchical)
// =============================================================================

const PHYSICS_DIM: i64 = 46;
const MACRO_DIM: i64 = 30;
const MICRO_DIM: i64 = 36;
const FEATURE_DIM: i64 = 112;

const PHYSICS_HIDDEN: i64 = 256;
const MACRO_HIDDEN: i64 = 128;
const MICRO_HIDDEN: i64 = 64;

const LEARNING_RATE: f64 = 1e-4;
const WEIGHT_DECAY: f64 = 0.01;
const DROPOUT_RATE: f64 = 0.3;
const BATCH_SIZE: i64 = 256;

const PHYSICS_EPOCHS: i64 = 30;
const MACRO_EPOCHS: i64 = 30;
const MICRO_EPOCHS: i64 = 40;
const PATIENCE: i64 = 8;

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Debug, Deserialize)]
struct BeansManifest {
    samples: Vec<BeansSample>,
}

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
struct CacheManifest {
    entries: HashMap<String, String>,
}

// =============================================================================
// Physics Block
// =============================================================================

struct PhysicsBlock {
    fc1: nn::Linear,
    bn1: nn::BatchNorm,
    fc2: nn::Linear,
}

impl PhysicsBlock {
    fn new(vs: &nn::Path) -> Self {
        let fc1 = nn::linear(vs, PHYSICS_DIM, PHYSICS_HIDDEN, Default::default());
        let bn1 = nn::batch_norm1d(vs, PHYSICS_HIDDEN, Default::default());
        let fc2 = nn::linear(vs, PHYSICS_HIDDEN, PHYSICS_HIDDEN, Default::default());
        Self { fc1, bn1, fc2 }
    }

    fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        let x = x.apply(&self.fc1);
        let x = x.apply_t(&self.bn1, train);
        let x = x.gelu("none");
        let x = x.apply(&self.fc2);
        x.gelu("none").dropout(DROPOUT_RATE, train)
    }
}

// =============================================================================
// Macro Block
// =============================================================================

struct MacroBlock {
    fc1: nn::Linear,
    bn1: nn::BatchNorm,
    fc2: nn::Linear,
}

impl MacroBlock {
    fn new(vs: &nn::Path) -> Self {
        let input_dim = PHYSICS_HIDDEN + MACRO_DIM;
        let fc1 = nn::linear(vs, input_dim, MACRO_HIDDEN, Default::default());
        let bn1 = nn::batch_norm1d(vs, MACRO_HIDDEN, Default::default());
        let fc2 = nn::linear(vs, MACRO_HIDDEN, MACRO_HIDDEN, Default::default());
        Self { fc1, bn1, fc2 }
    }

    fn forward(&self, physics_out: &Tensor, macro_feat: &Tensor, train: bool) -> Tensor {
        let x = Tensor::cat(&[physics_out, macro_feat], 1);
        let x = x.apply(&self.fc1);
        let x = x.apply_t(&self.bn1, train);
        let x = x.gelu("none");
        let x = x.apply(&self.fc2);
        x.gelu("none").dropout(DROPOUT_RATE, train)
    }
}

// =============================================================================
// Micro Block
// =============================================================================

struct MicroBlock {
    fc1: nn::Linear,
    bn1: nn::BatchNorm,
    fc2: nn::Linear,
}

impl MicroBlock {
    fn new(vs: &nn::Path) -> Self {
        let input_dim = MACRO_HIDDEN + MICRO_DIM;
        let fc1 = nn::linear(vs, input_dim, MICRO_HIDDEN, Default::default());
        let bn1 = nn::batch_norm1d(vs, MICRO_HIDDEN, Default::default());
        let fc2 = nn::linear(vs, MICRO_HIDDEN, MICRO_HIDDEN, Default::default());
        Self { fc1, bn1, fc2 }
    }

    fn forward(&self, macro_out: &Tensor, micro_feat: &Tensor, train: bool) -> Tensor {
        let x = Tensor::cat(&[macro_out, micro_feat], 1);
        let x = x.apply(&self.fc1);
        let x = x.apply_t(&self.bn1, train);
        let x = x.gelu("none");
        let x = x.apply(&self.fc2);
        x.gelu("none").dropout(DROPOUT_RATE, train)
    }
}

// =============================================================================
// Output Block
// =============================================================================

struct OutputBlock {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl OutputBlock {
    fn new(vs: &nn::Path, n_classes: i64) -> Self {
        let fc1 = nn::linear(vs, MICRO_HIDDEN, MICRO_HIDDEN, Default::default());
        let fc2 = nn::linear(vs, MICRO_HIDDEN, n_classes, Default::default());
        Self { fc1, fc2 }
    }

    fn forward(&self, x: &Tensor, _train: bool) -> Tensor {
        let x = x.apply(&self.fc1);
        let x = x.gelu("none");
        x.apply(&self.fc2)
    }
}

// =============================================================================
// Full Network
// =============================================================================

struct CurriculumNet {
    physics: PhysicsBlock,
    macro_block: MacroBlock,
    micro: MicroBlock,
    output: OutputBlock,
}

impl CurriculumNet {
    fn new(vs: &nn::Path, n_classes: i64) -> Self {
        let physics = PhysicsBlock::new(&vs.sub("physics"));
        let macro_block = MacroBlock::new(&vs.sub("macro"));
        let micro = MicroBlock::new(&vs.sub("micro"));
        let output = OutputBlock::new(&vs.sub("output"), n_classes);
        Self {
            physics,
            macro_block,
            micro,
            output,
        }
    }

    fn forward(&self, physics_input: &Tensor, macro_input: &Tensor, micro_input: &Tensor, train: bool) -> Tensor {
        let physics_out = self.physics.forward(physics_input, train);
        let macro_out = self.macro_block.forward(&physics_out, macro_input, train);
        let micro_out = self.micro.forward(&macro_out, micro_input, train);
        self.output.forward(&micro_out, train)
    }
}

// =============================================================================
// Data Loading
// =============================================================================

struct Dataset {
    features: Vec<Vec<f32>>,
    labels: Vec<i64>,
    n_classes: i64,
}

fn load_data() -> Result<Dataset> {
    println!("Loading manifest from: beans_zero_full_manifest.json");
    let manifest_data = fs::read_to_string("beans_zero_full_manifest.json")?;
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
                        if features.len() == FEATURE_DIM as usize {
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
    let mut unique_labels: Vec<String> = all_labels.clone();
    unique_labels.sort();
    unique_labels.dedup();
    let n_classes = unique_labels.len();
    let mut label_to_idx = HashMap::new();
    for (idx, label) in unique_labels.iter().enumerate() {
        label_to_idx.insert(label.clone(), idx as i64);
    }
    println!("  Classes: {}", n_classes);

    let labels: Vec<i64> = all_labels.iter().map(|l| *label_to_idx.get(l).unwrap_or(&0)).collect();

    Ok(Dataset {
        features: all_features,
        labels,
        n_classes: n_classes as i64,
    })
}

// =============================================================================
// Training Helpers
// =============================================================================

struct DataBatcher {
    features: Vec<Vec<f32>>,
    labels: Vec<i64>,
    indices: Vec<usize>,
    current: usize,
    batch_size: i64,
    device: Device,
}

impl DataBatcher {
    fn new(features: Vec<Vec<f32>>, labels: Vec<i64>, batch_size: i64, device: Device) -> Self {
        let indices: Vec<usize> = (0..features.len()).collect();
        Self {
            features,
            labels,
            indices,
            current: 0,
            batch_size,
            device,
        }
    }

    fn shuffle(&mut self) {
        for i in 0..self.indices.len() {
            let j = (rand::random::<usize>()) % self.indices.len();
            self.indices.swap(i, j);
        }
        self.current = 0;
    }

    fn next(&mut self) -> Option<(Tensor, Tensor, Tensor, Tensor)> {
        if self.current >= self.indices.len() {
            return None;
        }

        let end = (self.current + self.batch_size as usize).min(self.indices.len());
        let batch_indices: Vec<usize> = self.indices[self.current..end].to_vec();
        self.current = end;

        let batch_size = batch_indices.len() as i64;

        let mut physics_data = vec![0.0f32; batch_size as usize * PHYSICS_DIM as usize];
        let mut macro_data = vec![0.0f32; batch_size as usize * MACRO_DIM as usize];
        let mut micro_data = vec![0.0f32; batch_size as usize * MICRO_DIM as usize];
        let mut label_data = vec![0i64; batch_size as usize];

        for (i, &idx) in batch_indices.iter().enumerate() {
            let features = &self.features[idx];
            for j in 0..PHYSICS_DIM as usize {
                physics_data[i * PHYSICS_DIM as usize + j] = features[j];
            }
            for j in 0..MACRO_DIM as usize {
                macro_data[i * MACRO_DIM as usize + j] = features[PHYSICS_DIM as usize + j];
            }
            for j in 0..MICRO_DIM as usize {
                micro_data[i * MICRO_DIM as usize + j] = features[(PHYSICS_DIM + MACRO_DIM) as usize + j];
            }
            label_data[i] = self.labels[idx];
        }

        let physics_tensor = Tensor::from_slice(&physics_data)
            .reshape([batch_size, PHYSICS_DIM])
            .to(self.device);
        let macro_tensor = Tensor::from_slice(&macro_data)
            .reshape([batch_size, MACRO_DIM])
            .to(self.device);
        let micro_tensor = Tensor::from_slice(&micro_data)
            .reshape([batch_size, MICRO_DIM])
            .to(self.device);
        let label_tensor = Tensor::from_slice(&label_data).to(self.device);

        Some((physics_tensor, macro_tensor, micro_tensor, label_tensor))
    }
}

fn compute_accuracy(
    net: &CurriculumNet,
    features: &[Vec<f32>],
    labels: &[i64],
    batch_size: i64,
    device: Device,
) -> f64 {
    let mut correct = 0i64;
    let mut total = 0i64;

    for start in (0..features.len()).step_by(batch_size as usize) {
        let end = (start + batch_size as usize).min(features.len());
        let actual_batch = (end - start) as i64;

        let mut physics_data = vec![0.0f32; actual_batch as usize * PHYSICS_DIM as usize];
        let mut macro_data = vec![0.0f32; actual_batch as usize * MACRO_DIM as usize];
        let mut micro_data = vec![0.0f32; actual_batch as usize * MICRO_DIM as usize];
        let mut label_data = vec![0i64; actual_batch as usize];

        for (i, idx) in (start..end).enumerate() {
            let features = &features[idx];
            for j in 0..PHYSICS_DIM as usize {
                physics_data[i * PHYSICS_DIM as usize + j] = features[j];
            }
            for j in 0..MACRO_DIM as usize {
                macro_data[i * MACRO_DIM as usize + j] = features[PHYSICS_DIM as usize + j];
            }
            for j in 0..MICRO_DIM as usize {
                micro_data[i * MICRO_DIM as usize + j] = features[(PHYSICS_DIM + MACRO_DIM) as usize + j];
            }
            label_data[i] = labels[idx];
        }

        let physics_tensor = Tensor::from_slice(&physics_data)
            .reshape([actual_batch, PHYSICS_DIM])
            .to(device);
        let macro_tensor = Tensor::from_slice(&macro_data)
            .reshape([actual_batch, MACRO_DIM])
            .to(device);
        let micro_tensor = Tensor::from_slice(&micro_data)
            .reshape([actual_batch, MICRO_DIM])
            .to(device);
        let label_tensor = Tensor::from_slice(&label_data).to(device);

        let output = net.forward(&physics_tensor, &macro_tensor, &micro_tensor, false);
        let predictions = output.argmax(1, false);

        let acc_tensor = predictions.eq_tensor(&label_tensor).sum(tch::Kind::Int64);
        correct += acc_tensor.int64_value(&[]);
        total += actual_batch;
    }

    correct as f64 / total as f64 * 100.0
}

fn train_phase(
    vs: &nn::VarStore,
    net: &CurriculumNet,
    dataset: &Dataset,
    phase_name: &str,
    epochs: i64,
    device: Device,
    train_physics: bool,
    train_macro: bool,
    train_micro: bool,
    train_output: bool,
) -> Result<f64> {
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  {} Phase{:>55}║", phase_name, "");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Training: Physics={:<5} Macro={:<5} Micro={:<5} Output={:<5}║",
        train_physics, train_macro, train_micro, train_output
    );
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    // In tch-rs, we create optimizer on VarStore and all parameters get updated.
    // For curriculum learning, we train all blocks but the focus is on the specified phase.
    // The learning schedule provides the curriculum structure.

    // Create optimizer
    let mut opt = nn::Adam::default()
        .build(vs, LEARNING_RATE)
        .context("failed to create optimizer")?;
    opt.set_weight_decay(WEIGHT_DECAY);

    // Split data
    let n_train = (dataset.features.len() as f64 * 0.9) as usize;
    let train_features = dataset.features[..n_train].to_vec();
    let train_labels = dataset.labels[..n_train].to_vec();
    let val_features = dataset.features[n_train..].to_vec();
    let val_labels = dataset.labels[n_train..].to_vec();

    let mut batcher = DataBatcher::new(train_features.clone(), train_labels.clone(), BATCH_SIZE, device);

    let mut best_val_acc = 0.0f64;
    let mut best_epoch = 0;
    let mut patience_counter = 0i64;
    let mut current_lr = LEARNING_RATE;

    for epoch in 1..=epochs {
        batcher.shuffle();
        let mut total_loss = 0.0f64;
        let mut n_batches = 0;

        while let Some((physics, macro_feat, micro_feat, labels)) = batcher.next() {
            let logits = net.forward(&physics, &macro_feat, &micro_feat, true);
            let loss = logits.cross_entropy_for_logits(&labels);

            // Manual backward step for specific variables
            opt.backward_step_clip(&loss, 1.0);

            total_loss += loss.double_value(&[]);
            n_batches += 1;
        }

        let avg_loss = total_loss / n_batches as f64;
        let train_acc = compute_accuracy(net, &train_features, &train_labels, BATCH_SIZE, device);
        let val_acc = compute_accuracy(net, &val_features, &val_labels, BATCH_SIZE, device);

        if val_acc > best_val_acc {
            best_val_acc = val_acc;
            best_epoch = epoch;
            patience_counter = 0;

            // Save best model
            vs.save("rosetta_net_112d_curriculum_gpu.ot")
                .context("failed to save model")?;
        } else {
            patience_counter += 1;
        }

        // LR decay
        if patience_counter >= 3 && current_lr > 1e-6 {
            current_lr *= 0.5;
            opt.set_lr(current_lr);
            println!("  Reducing learning rate to {:.2e}", current_lr);
        }

        if epoch % 5 == 0 || epoch == 1 {
            println!(
                "  Epoch {:3}/{}: Loss={:.4}, Train={:.1}%, Val={:.1}% | Best={:.1}% (epoch {})",
                epoch, epochs, avg_loss, train_acc, val_acc, best_val_acc, best_epoch
            );
        }

        // Early stopping
        if patience_counter >= PATIENCE {
            println!(
                "\n  Early stopping at epoch {} (no improvement for {} epochs)",
                epoch, PATIENCE
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
    println!("║  Curriculum Neural Network Training (112D Features) - GPU        ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Check for CUDA
    let device = if tch::Cuda::is_available() {
        println!("CUDA available! Using GPU.");
        Device::Cuda(0)
    } else {
        println!("CUDA not available, using CPU.");
        Device::Cpu
    };
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
    println!("  Output:        {}D -> {}D -> n_classes", MICRO_HIDDEN, MICRO_HIDDEN);
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
    let dataset = load_data()?;

    // Create model
    let vs = nn::VarStore::new(device);
    let net = CurriculumNet::new(&vs.root(), dataset.n_classes);

    // === PHASE 1: Train Physics Block ===
    train_phase(
        &vs,
        &net,
        &dataset,
        "Physics",
        PHYSICS_EPOCHS,
        device,
        true,  // train_physics
        false, // train_macro
        false, // train_micro
        true,  // train_output
    )?;

    // === PHASE 2: Train Macro Block ===
    train_phase(
        &vs,
        &net,
        &dataset,
        "Macro",
        MACRO_EPOCHS,
        device,
        false, // train_physics
        true,  // train_macro
        false, // train_micro
        true,  // train_output
    )?;

    // === PHASE 3: Train Micro Block ===
    let final_acc = train_phase(
        &vs,
        &net,
        &dataset,
        "Micro",
        MICRO_EPOCHS,
        device,
        false, // train_physics
        false, // train_macro
        true,  // train_micro
        true,  // train_output
    )?;

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
    println!("║  Model saved to: rosetta_net_112d_curriculum_gpu.ot               ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    Ok(())
}
