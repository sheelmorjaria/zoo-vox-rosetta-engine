//! Rosetta-Net MLP Training with Burn Framework 0.16
//!
//! Architecture: MLP (105D features → 256D hidden → 128D latent → classification)
//! Uses LibTorch backend for reliable CPU support
//!
//! Usage:
//!   export LIBTORCH=/home/sheel/libtorch
//!   export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
//!   cargo run --release --bin train_rosetta_gpu

use anyhow::Result;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::activation::relu;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};
use burn_tch::{LibTorch, LibTorchDevice};
use std::collections::HashMap;
use std::time::Instant;

// ============================================================================
// Configuration - LARGER MODEL (512D Hidden)
// ============================================================================

pub const INPUT_DIM: usize = 105;
pub const HIDDEN_DIM: usize = 512; // Increased from 256
pub const LATENT_DIM: usize = 256; // Increased from 128
pub const BATCH_SIZE: usize = 64; // Smaller batch for better gradients
pub const LEARNING_RATE: f64 = 0.0005; // Slightly lower for stability
pub const NUM_EPOCHS: usize = 200; // More epochs for larger model
pub const TRAIN_SPLIT: f64 = 0.8;

type MyBackend = burn::backend::Autodiff<LibTorch<f32>>;

// ============================================================================
// Model Configuration
// ============================================================================

#[derive(Config, Debug)]
pub struct RosettaNetConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
    pub num_classes: usize,
}

// ============================================================================
// MLP Model - Deeper Architecture (4 layers for larger model)
// ============================================================================

#[derive(Module, Debug)]
pub struct RosettaNet<B: burn::tensor::backend::Backend> {
    encoder: Linear<B>,
    hidden1: Linear<B>,
    hidden2: Linear<B>,
    latent: Linear<B>,
    classifier: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> RosettaNet<B> {
    pub fn init(config: &RosettaNetConfig, device: &B::Device) -> Self {
        let encoder = LinearConfig::new(config.input_dim, config.hidden_dim).init::<B>(device);
        let hidden1 = LinearConfig::new(config.hidden_dim, config.hidden_dim).init::<B>(device);
        let hidden2 = LinearConfig::new(config.hidden_dim, config.latent_dim).init::<B>(device);
        let latent = LinearConfig::new(config.latent_dim, config.latent_dim).init::<B>(device);
        let classifier = LinearConfig::new(config.latent_dim, config.num_classes).init::<B>(device);

        Self {
            encoder,
            hidden1,
            hidden2,
            latent,
            classifier,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = relu(self.encoder.forward(x));
        let x = relu(self.hidden1.forward(x));
        let x = relu(self.hidden2.forward(x));
        let latent = relu(self.latent.forward(x));
        let logits = self.classifier.forward(latent.clone());
        (latent, logits)
    }
}

impl<B: AutodiffBackend> RosettaNet<B> {
    pub fn training_step(
        &self,
        features: Tensor<B, 2>,
        targets: Tensor<B, 1, burn::tensor::Int>,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let (_, logits) = self.forward(features);
        let ce = burn::nn::loss::CrossEntropyLoss::new(None, device);
        ce.forward(logits, targets)
    }
}

// ============================================================================
// Data Loading (Memory Efficient)
// ============================================================================

#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub features: Vec<f32>,
    pub class_label: usize,
}

pub struct BeanzZeroDataset {
    pub samples: Vec<TrainingSample>,
    pub num_classes: usize,
    pub class_names: Vec<String>,
}

impl BeanzZeroDataset {
    pub fn load_or_synthetic() -> Result<Self> {
        let cache_path = "beans_zero_cache/feature_cache_eval/all_features.bin";
        let manifest_path = "beans_zero_cache/beans_audio_manifest.json";

        if std::path::Path::new(cache_path).exists() && std::path::Path::new(manifest_path).exists()
        {
            println!("Loading features from: {}", cache_path);
            match Self::load_from_cache(cache_path, manifest_path) {
                Ok(dataset) => return Ok(dataset),
                Err(e) => {
                    eprintln!("ERROR loading cache: {:?}", e);
                    eprintln!("Falling back to synthetic data...");
                }
            }
        } else {
            eprintln!("Cache file not found: {}", cache_path);
        }

        Self::create_synthetic()
    }

    fn load_from_cache(cache_path: &str, manifest_path: &str) -> Result<Self> {
        use std::io::BufReader;

        // Load manifest
        let manifest_json = std::fs::read_to_string(manifest_path)?;
        let manifest: serde_json::Value = serde_json::from_str(&manifest_json)?;
        let samples_meta = manifest
            .get("samples")
            .and_then(|s| s.as_array())
            .ok_or_else(|| anyhow::anyhow!("No samples in manifest"))?;

        // Build ID -> species label mapping (BEANS-Zero aligned)
        // Use "output" field which contains the exact label used in evaluation
        let mut id_to_label: HashMap<String, String> = HashMap::new();
        for sample in samples_meta {
            if let (Some(id_val), Some(labels)) = (sample.get("id"), sample.get("labels")) {
                let id_str = id_val.as_str().unwrap_or("");

                // Use "output" field exactly as BEANS-Zero expects it
                // This is the ground truth label used in evaluation
                let label = if let Some(output) = labels.get("output").and_then(|s| s.as_str()) {
                    if output != "None" && !output.is_empty() {
                        // Clean up: keep the full species/description name
                        // Remove trailing punctuation and normalize whitespace
                        let cleaned = output
                            .trim()
                            .trim_end_matches('.')
                            .trim_end_matches(',')
                            .split_whitespace()
                            .collect::<Vec<_>>()
                            .join(" ");
                        cleaned
                    } else {
                        // Fallback to source_dataset for samples without species labels
                        labels
                            .get("source_dataset")
                            .and_then(|s| s.as_str())
                            .unwrap_or("unknown")
                            .to_string()
                    }
                } else {
                    // Fallback to source_dataset
                    labels
                        .get("source_dataset")
                        .and_then(|s| s.as_str())
                        .unwrap_or("unknown")
                        .to_string()
                };

                id_to_label.insert(id_str.to_string(), label);
            }
        }
        println!("  Manifest: {} samples with labels", id_to_label.len());

        // Load features
        let file = std::fs::File::open(cache_path)?;
        let reader = BufReader::new(file);
        let cache: HashMap<String, Vec<f32>> = bincode::deserialize_from(reader)?;
        println!("  Cached features: {}", cache.len());

        let mut samples = Vec::new();
        let mut class_map: HashMap<String, usize> = HashMap::new();
        let mut next_class = 0;

        for (cache_key, features) in &cache {
            if features.len() < INPUT_DIM {
                continue;
            }

            // Extract sample ID from cache key
            let sample_id = cache_key
                .rsplit('/')
                .next()
                .unwrap_or(cache_key)
                .trim_end_matches(".rawi")
                .trim_end_matches(".raw")
                .to_string();

            let class_name = id_to_label
                .get(&sample_id)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());

            let class_idx = *class_map.entry(class_name.clone()).or_insert_with(|| {
                let idx = next_class;
                next_class += 1;
                idx
            });

            samples.push(TrainingSample {
                features: features[..INPUT_DIM].to_vec(),
                class_label: class_idx,
            });
        }

        // Build class names list
        let mut class_names: Vec<(String, usize)> =
            class_map.iter().map(|(k, v)| (k.clone(), *v)).collect();
        class_names.sort_by_key(|(_, idx)| *idx);
        let class_names: Vec<String> = class_names.into_iter().map(|(k, _)| k).collect();

        let mem_mb = samples.len() * INPUT_DIM * 4 / 1024 / 1024;
        println!(
            "  Loaded {} samples, {} classes (~{} MB)",
            samples.len(),
            next_class,
            mem_mb
        );
        println!(
            "  First 10 classes: {:?}",
            class_names.iter().take(10).collect::<Vec<_>>()
        );

        Ok(Self {
            samples,
            num_classes: next_class,
            class_names,
        })
    }

    fn create_synthetic() -> Result<Self> {
        println!("Creating synthetic dataset (10,000 samples, 50 classes)...");
        let mut rng = rand::thread_rng();
        let num_classes = 50;

        let samples = (0..10000)
            .map(|i| TrainingSample {
                features: (0..INPUT_DIM)
                    .map(|_| rand::Rng::gen::<f32>(&mut rng) * 2.0 - 1.0)
                    .collect(),
                class_label: i % num_classes,
            })
            .collect();

        let class_names = (0..num_classes).map(|i| format!("class_{}", i)).collect();

        Ok(Self {
            samples,
            num_classes,
            class_names,
        })
    }

    pub fn split(&self, ratio: f64) -> (Vec<&TrainingSample>, Vec<&TrainingSample>) {
        let split = (self.samples.len() as f64 * ratio) as usize;
        let (train, val) = self.samples.split_at(split);
        (train.iter().collect(), val.iter().collect())
    }
}

// ============================================================================
// Checkpointing
// ============================================================================

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Checkpoint {
    pub epoch: usize,
    pub train_loss: f32,
    pub val_loss: f32,
    pub train_acc: f32,
    pub val_acc: f32,
}

impl Checkpoint {
    pub fn save(&self, path: &str) -> Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

// ============================================================================
// Model Weight Export
// ============================================================================

#[derive(serde::Serialize)]
pub struct ModelConfigJson {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
    pub num_classes: usize,
    pub class_names: Vec<String>,
}

pub fn save_model<B: burn::tensor::backend::Backend>(
    model: &RosettaNet<B>,
    config: &RosettaNetConfig,
    class_names: &[String],
    path: &str,
) -> Result<()> {
    let record = model.clone().into_record();
    CompactRecorder::new()
        .record(record, format!("{}.mpk", path).into())
        .map_err(|e| anyhow::anyhow!("Failed to save model: {:?}", e))?;

    let config_json = ModelConfigJson {
        input_dim: config.input_dim,
        hidden_dim: config.hidden_dim,
        latent_dim: config.latent_dim,
        num_classes: config.num_classes,
        class_names: class_names.to_vec(),
    };
    std::fs::write(
        format!("{}_config.json", path),
        serde_json::to_string_pretty(&config_json)?,
    )?;

    println!("  Saved: {}.mpk, {}_config.json", path, path);
    Ok(())
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║       ROSETTA-NET MLP TRAINING WITH BURN FRAMEWORK 0.16              ║");
    println!(
        "║       {}D Features | {}D Hidden | {}D Latent | LibTorch Backend        ║",
        INPUT_DIM, HIDDEN_DIM, LATENT_DIM
    );
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    let start = Instant::now();

    println!("Initializing LibTorch CPU device...");
    let device = LibTorchDevice::Cpu;

    // Load data
    let dataset = BeanzZeroDataset::load_or_synthetic()?;
    let (train, val) = dataset.split(TRAIN_SPLIT);
    println!(
        "  Train: {} | Val: {} | Classes: {}\n",
        train.len(),
        val.len(),
        dataset.num_classes
    );

    // Create model
    let config = RosettaNetConfig::new(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, dataset.num_classes);
    let mut model = RosettaNet::<MyBackend>::init(&config, &device);

    // Initialize optimizer
    let mut optim = AdamConfig::new().init::<MyBackend, RosettaNet<MyBackend>>();

    println!("✓ Model & optimizer initialized");
    println!(
        "  Input: {}D | Hidden: {}D | Latent: {}D | Classes: {}",
        INPUT_DIM, HIDDEN_DIM, LATENT_DIM, dataset.num_classes
    );
    println!(
        "  Batch size: {} | Learning rate: {}",
        BATCH_SIZE, LEARNING_RATE
    );

    println!(
        "\n{}\nSTARTING TRAINING\n{}",
        "=".repeat(70),
        "=".repeat(70)
    );

    let mut best_val_loss = f32::MAX;
    let mut history: Vec<Checkpoint> = Vec::new();

    for epoch in 0..NUM_EPOCHS {
        let epoch_start = Instant::now();
        let mut train_loss = 0.0;
        let mut train_correct = 0;
        let n_train_batches = (train.len() + BATCH_SIZE - 1) / BATCH_SIZE;

        // Training
        for batch_start in (0..train.len()).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(train.len());
            let bs = batch_end - batch_start;

            let (features, labels) = build_batch(&train, batch_start, batch_end);
            let labels_for_acc = labels.clone();

            let feat_t = Tensor::<MyBackend, 2>::from_data(
                TensorData::new(features, [bs, INPUT_DIM]),
                &device,
            );
            let label_t = Tensor::<MyBackend, 1, burn::tensor::Int>::from_data(
                TensorData::new(labels, [bs]),
                &device,
            );

            let loss = model.training_step(feat_t, label_t.clone(), &device);
            let grads = loss.backward();

            let loss_val = loss.into_data().to_vec::<f32>().unwrap_or_default()[0];
            train_loss += loss_val;

            // Calculate accuracy
            let (_, logits) = model.forward(Tensor::<MyBackend, 2>::from_data(
                TensorData::new(
                    build_batch(&train, batch_start, batch_end).0,
                    [bs, INPUT_DIM],
                ),
                &device,
            ));
            let preds = logits
                .argmax(1)
                .into_data()
                .to_vec::<i64>()
                .unwrap_or_default();
            for (p, l) in preds.iter().zip(labels_for_acc.iter()) {
                if *p == *l {
                    train_correct += 1;
                }
            }

            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(LEARNING_RATE, model, grads_params);
        }

        // Validation
        let mut val_loss = 0.0;
        let mut val_correct = 0;
        let n_val_batches = (val.len() + BATCH_SIZE - 1) / BATCH_SIZE;

        for batch_start in (0..val.len()).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(val.len());
            let bs = batch_end - batch_start;

            let (features, labels) = build_batch(&val, batch_start, batch_end);
            let labels_for_acc = labels.clone();

            let feat_t = Tensor::<MyBackend, 2>::from_data(
                TensorData::new(features, [bs, INPUT_DIM]),
                &device,
            );
            let label_t = Tensor::<MyBackend, 1, burn::tensor::Int>::from_data(
                TensorData::new(labels, [bs]),
                &device,
            );

            let (_, logits) = model.forward(feat_t);

            let ce = burn::nn::loss::CrossEntropyLoss::new(None, &device);
            let loss = ce.forward(logits.clone(), label_t);
            val_loss += loss.into_data().to_vec::<f32>().unwrap_or_default()[0];

            let preds = logits
                .argmax(1)
                .into_data()
                .to_vec::<i64>()
                .unwrap_or_default();
            for (p, l) in preds.iter().zip(labels_for_acc.iter()) {
                if *p == *l {
                    val_correct += 1;
                }
            }
        }

        let avg_train_loss = train_loss / n_train_batches as f32;
        let avg_val_loss = val_loss / n_val_batches as f32;
        let train_acc = train_correct as f32 / train.len() as f32 * 100.0;
        let val_acc = val_correct as f32 / val.len() as f32 * 100.0;
        let time = epoch_start.elapsed();

        println!(
            "Epoch {:3}/{} | Train: {:.4} ({:.1}%) | Val: {:.4} ({:.1}%) | Time: {:5.1}s{}",
            epoch + 1,
            NUM_EPOCHS,
            avg_train_loss,
            train_acc,
            avg_val_loss,
            val_acc,
            time.as_secs_f32(),
            if avg_val_loss < best_val_loss {
                " ← BEST"
            } else {
                ""
            }
        );

        let ckpt = Checkpoint {
            epoch: epoch + 1,
            train_loss: avg_train_loss,
            val_loss: avg_val_loss,
            train_acc,
            val_acc,
        };
        history.push(ckpt.clone());

        if avg_val_loss < best_val_loss {
            best_val_loss = avg_val_loss;
            ckpt.save("rosetta_net_best_checkpoint.json")?;
            save_model(&model, &config, &dataset.class_names, "rosetta_net_best")?;
        }
    }

    // Save final model
    println!("\nSaving final model...");
    save_model(&model, &config, &dataset.class_names, "rosetta_net_final")?;

    history
        .last()
        .unwrap()
        .save("rosetta_net_final_checkpoint.json")?;
    std::fs::write(
        "rosetta_net_training_history.json",
        serde_json::to_string_pretty(&history)?,
    )?;

    println!(
        "\n{}\nTRAINING COMPLETE\n{}",
        "=".repeat(70),
        "=".repeat(70)
    );
    println!(
        "  Time: {:.1} min | Best Val Loss: {:.4}",
        start.elapsed().as_secs_f32() / 60.0,
        best_val_loss
    );
    println!("  Files: rosetta_net_best.mpk, rosetta_net_final.mpk");
    println!("         rosetta_net_best_config.json, rosetta_net_final_config.json");

    Ok(())
}

fn build_batch(samples: &[&TrainingSample], start: usize, end: usize) -> (Vec<f32>, Vec<i64>) {
    let bs = end - start;
    let mut features = Vec::with_capacity(bs * INPUT_DIM);
    let mut labels = Vec::with_capacity(bs);

    for idx in start..end {
        let s = &samples[idx];
        features.extend_from_slice(&s.features);
        labels.push(s.class_label as i64);
    }

    (features, labels)
}
