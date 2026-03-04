//! Train Hybrid Expert Architecture (Taxonomic-Aware NN + RF)
//! ============================================================
//!
//! Implements the Hybrid Expert Architecture:
//! - Input A (Physics Vector - 46D): Random Forest
//! - Input B (Texture Vector - 66D): Neural Network with Taxonomic Masking
//!
//! Usage:
//!   export LIBTORCH=$HOME/libtorch
//!   export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
//!   cargo run --release --bin train_hybrid_expert --features gpu-training

use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

use technical_architecture::taxonomic_router::{
    Taxon, FEATURE_DIM, PHYSICS_DIM, TEXTURE_DIM,
    apply_taxonomic_mask, map_species_to_taxon, map_task_to_taxon, slice_texture,
};

// NN Hyperparameters for 66D texture input
const HIDDEN_DIM_1: i64 = 512;
const HIDDEN_DIM_2: i64 = 256;
const HIDDEN_DIM_3: i64 = 128;
const LEARNING_RATE: f64 = 3e-4;
const WEIGHT_DECAY: f64 = 0.01;
const EPOCHS: i64 = 200;
const BATCH_SIZE: i64 = 128;
const PATIENCE: i64 = 40;
const LABEL_SMOOTHING: f64 = 0.1;
const DROPOUT: f64 = 0.4;
const GRADIENT_CLIP: f64 = 0.5;
const WARMUP_EPOCHS: i64 = 5;

// =============================================================================
// Neural Network for Texture Features (66D)
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
// Main Training Loop
// =============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Hybrid Expert Architecture Training                              ║");
    println!("║  - Physics (46D) → Random Forest                                  ║");
    println!("║  - Texture (66D) → Neural Network + Taxonomic Masking             ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let device = Device::cuda_if_available();
    println!("Device: {:?}", device);
    if !device.is_cuda() {
        println!("WARNING: CUDA not available, falling back to CPU!");
    }
    println!();

    println!("NN Architecture (Texture 66D):");
    println!("  Input:      {}D", TEXTURE_DIM);
    println!("  Hidden 1:   {}D (LN + GELU + Dropout {})", HIDDEN_DIM_1, DROPOUT);
    println!("  Hidden 2:   {}D (LN + GELU + Dropout {})", HIDDEN_DIM_2, DROPOUT);
    println!("  Hidden 3:   {}D (LN + GELU + Dropout {})", HIDDEN_DIM_3, DROPOUT);
    println!("  Optimizer:  AdamW (lr={}, wd={})", LEARNING_RATE, WEIGHT_DECAY);
    println!("  Batch Size: {}", BATCH_SIZE);
    println!("  Label Smoothing: {}", LABEL_SMOOTHING);
    println!("  Taxonomic Masking: Enabled");
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

    // Load all features, labels, and taxons
    println!("\nLoading features from cache...");
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

        // Determine taxonomic group
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
                            // Apply taxonomic mask
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
    println!("  Taxonomic distribution:");
    let mut taxon_counts: HashMap<Taxon, usize> = HashMap::new();
    for t in &all_taxons {
        *taxon_counts.entry(*t).or_insert(0) += 1;
    }
    for (taxon, count) in &taxon_counts {
        println!("    {:?}: {} ({:.1}%)", taxon, count, *count as f64 / all_features.len() as f64 * 100.0);
    }

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
    println!("  Classes: {}", n_classes);

    // Split into train/validation (90/10)
    let n_train = (all_features.len() as f32 * 0.9) as usize;
    println!("\nSplitting data: 90% train, 10% validation...");

    // Shuffle indices using deterministic seed
    let mut indices: Vec<usize> = (0..all_features.len()).collect();
    for i in 0..indices.len() {
        let j = (rand_u32() as usize) % indices.len();
        indices.swap(i, j);
    }

    let train_indices: Vec<usize> = indices[..n_train].to_vec();
    let val_indices: Vec<usize> = indices[n_train..].to_vec();
    println!("  Train samples: {}", train_indices.len());
    println!("  Val samples: {}", val_indices.len());

    // Extract texture features (66D) from masked 112D
    let all_texture: Vec<Vec<f32>> = all_features.iter()
        .map(|f| slice_texture(f))
        .collect();

    // Compute normalization params from training set
    let mut texture_means = vec![0.0f32; TEXTURE_DIM];
    let mut texture_stds = vec![0.0f32; TEXTURE_DIM];

    for &i in &train_indices {
        for (j, &v) in all_texture[i].iter().enumerate() {
            texture_means[j] += v;
        }
    }
    for j in 0..TEXTURE_DIM {
        texture_means[j] /= train_indices.len() as f32;
    }

    for &i in &train_indices {
        for (j, &v) in all_texture[i].iter().enumerate() {
            let diff = v - texture_means[j];
            texture_stds[j] += diff * diff;
        }
    }
    for j in 0..TEXTURE_DIM {
        texture_stds[j] = (texture_stds[j] / train_indices.len() as f32).sqrt().max(1e-8);
    }

    // Create tensors
    println!("\nCreating tensors...");

    // Normalize texture features
    let normalized_texture: Vec<Vec<f32>> = all_texture.iter()
        .map(|t| t.iter()
            .enumerate()
            .map(|(j, &v)| (v - texture_means[j]) / texture_stds[j])
            .collect())
        .collect();

    // Create training tensor
    let train_size = train_indices.len();
    let train_data: Vec<f32> = train_indices
        .iter()
        .flat_map(|&i| normalized_texture[i].clone())
        .collect();
    let train_labels: Vec<i64> = train_indices
        .iter()
        .map(|&i| *label_to_idx.get(&all_labels[i]).unwrap_or(&0))
        .collect();

    // Create validation tensor
    let val_size = val_indices.len();
    let val_data: Vec<f32> = val_indices
        .iter()
        .flat_map(|&i| normalized_texture[i].clone())
        .collect();
    let val_labels: Vec<i64> = val_indices
        .iter()
        .map(|&i| *label_to_idx.get(&all_labels[i]).unwrap_or(&0))
        .collect();

    // Move to device
    let train_x = Tensor::from_slice(&train_data)
        .view([train_size as i64, TEXTURE_DIM as i64])
        .to(device);
    let train_y = Tensor::from_slice(&train_labels).to(device);

    let val_x = Tensor::from_slice(&val_data)
        .view([val_size as i64, TEXTURE_DIM as i64])
        .to(device);
    let val_y = Tensor::from_slice(&val_labels).to(device);

    println!("  Train tensor shape: {:?}", train_x.size());
    println!("  Val tensor shape: {:?}", val_x.size());

    // Create model
    let vs = nn::VarStore::new(device);
    let net = TextureNet::new(&vs.root(), n_classes);

    // AdamW optimizer
    let mut opt = nn::AdamW {
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        wd: WEIGHT_DECAY,
        amsgrad: false,
    }
    .build(&vs, LEARNING_RATE)?;

    // Training
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Training Hybrid Expert (Texture NN)                              ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let mut best_val_acc = 0.0f32;
    let mut best_epoch = 0;
    let mut patience_counter = 0;
    let n_batches = (train_size as i64 + BATCH_SIZE - 1) / BATCH_SIZE;

    for epoch in 0..EPOCHS {
        // Cosine annealing learning rate with warmup
        let lr = if epoch < WARMUP_EPOCHS {
            LEARNING_RATE * (epoch + 1) as f64 / WARMUP_EPOCHS as f64
        } else {
            let progress = (epoch - WARMUP_EPOCHS) as f64 / (EPOCHS - WARMUP_EPOCHS) as f64;
            let cosine_factor = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            LEARNING_RATE * cosine_factor
        };
        opt.set_lr(lr);

        // Shuffle training data
        let perm = Tensor::randperm(train_size as i64, (Kind::Int64, device));
        let shuffled_x = train_x.index_select(0, &perm);
        let shuffled_y = train_y.index_select(0, &perm);

        let mut total_loss = 0.0f64;
        let mut train_correct = 0i64;

        for batch in 0..n_batches {
            let start_idx = batch * BATCH_SIZE;
            let end_idx = std::cmp::min(start_idx + BATCH_SIZE, train_size as i64);

            let batch_x = shuffled_x.narrow(0, start_idx, end_idx - start_idx);
            let batch_y = shuffled_y.narrow(0, start_idx, end_idx - start_idx);

            // Forward pass
            let logits = net.forward(&batch_x);

            // Cross-entropy loss with label smoothing
            let loss = logits.cross_entropy_loss::<Tensor>(
                &batch_y,
                None,
                tch::Reduction::Mean,
                -100i64,
                LABEL_SMOOTHING
            );

            // Backward pass with gradient clipping
            opt.backward_step_clip(&loss, GRADIENT_CLIP);

            total_loss += loss.double_value(&[]);

            // Count correct
            let predictions = logits.argmax(-1, false);
            train_correct += predictions.eq_tensor(&batch_y).sum(Kind::Int64).int64_value(&[]);
        }

        let avg_loss = total_loss / n_batches as f64;

        // Validation
        let val_logits = net.forward(&val_x);
        let val_predictions = val_logits.argmax(-1, false);
        let val_correct = val_predictions.eq_tensor(&val_y).sum(Kind::Int64).int64_value(&[]);
        let val_acc = val_correct as f32 / val_size as f32 * 100.0;

        let train_acc = train_correct as f32 / train_size as f32 * 100.0;

        // Check for improvement
        if val_acc > best_val_acc {
            best_val_acc = val_acc;
            best_epoch = epoch + 1;
            patience_counter = 0;
            vs.save("hybrid_expert_texture_nn.ot")?;
        } else {
            patience_counter += 1;
        }

        // Print progress
        if (epoch + 1) % 5 == 0 || epoch == 0 {
            println!(
                "  Epoch {:3}/{}: Loss={:.4}, LR={:.6}, Train={:.1}%, Val={:.1}% | Best={:.1}% (epoch {})",
                epoch + 1, EPOCHS, avg_loss, lr, train_acc, val_acc, best_val_acc, best_epoch
            );
        }

        // Early stopping
        if patience_counter >= PATIENCE {
            println!("\n  Early stopping at epoch {} (no improvement for {} epochs)", epoch + 1, PATIENCE);
            break;
        }
    }

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                          ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Device:             {:<44}║", format!("{:?}", device));
    println!("║  Architecture:       {:<44}║", "Hybrid Expert (Texture NN 66D)");
    println!("║  Taxonomic Masking:  {:<44}║", "Enabled");
    println!("║  Best Val Accuracy:  {:>8.2}%                                   ║", best_val_acc);
    println!("║  Best Epoch:         {:<44}║", best_epoch);
    println!("║  Total Time:         {:>8.1}s                                    ║", start.elapsed().as_secs_f32());
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    println!("\nSaved Texture NN to: hybrid_expert_texture_nn.ot");
    println!("\nNext step: Train Random Forest on Physics (46D) features");

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
