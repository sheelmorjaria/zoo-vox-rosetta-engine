//! Evaluate trained Rosetta-Net model on BEANS-Zero dataset
//!
//! Usage:
//!   export LIBTORCH=/home/sheel/libtorch
//!   export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
//!   cargo run --release --bin evaluate_rosetta_net

use anyhow::Result;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::activation::relu;
use burn::tensor::{Tensor, TensorData};
use burn_tch::{LibTorch, LibTorchDevice};
use std::collections::HashMap;
use std::io::BufReader;

// ============================================================================
// Configuration (must match training)
// ============================================================================

pub const INPUT_DIM: usize = 105;

type MyBackend = LibTorch<f32>;

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
// MLP Model (must match training architecture)
// ============================================================================

#[derive(Module, Debug)]
pub struct RosettaNet<B: burn::tensor::backend::Backend> {
    encoder: Linear<B>,
    hidden: Linear<B>,
    latent: Linear<B>,
    classifier: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> RosettaNet<B> {
    pub fn init(config: &RosettaNetConfig, device: &B::Device) -> Self {
        let encoder = LinearConfig::new(config.input_dim, config.hidden_dim).init::<B>(device);
        let hidden = LinearConfig::new(config.hidden_dim, config.latent_dim).init::<B>(device);
        let latent = LinearConfig::new(config.latent_dim, config.latent_dim).init::<B>(device);
        let classifier = LinearConfig::new(config.latent_dim, config.num_classes).init::<B>(device);

        Self {
            encoder,
            hidden,
            latent,
            classifier,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = relu(self.encoder.forward(x));
        let x = relu(self.hidden.forward(x));
        let latent = relu(self.latent.forward(x));
        let logits = self.classifier.forward(latent.clone());
        (latent, logits)
    }
}

// ============================================================================
// Model Config JSON (with class names)
// ============================================================================

#[derive(Debug, serde::Deserialize)]
pub struct ModelConfigJson {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
    pub num_classes: usize,
    #[serde(default)]
    pub class_names: Vec<String>,
}

// ============================================================================
// Data Loading
// ============================================================================

#[derive(Debug, Clone)]
pub struct TestSample {
    pub features: Vec<f32>,
    pub class_label: usize,
}

pub struct TestDataset {
    pub samples: Vec<TestSample>,
    pub num_classes: usize,
    pub class_names: Vec<String>,
}

impl TestDataset {
    pub fn load() -> Result<Self> {
        let cache_path = "beans_zero_cache/feature_cache_eval/all_features.bin";
        let manifest_path = "beans_zero_cache/beans_audio_manifest.json";

        if !std::path::Path::new(cache_path).exists() {
            anyhow::bail!("Cache file not found: {}", cache_path);
        }

        // Load manifest
        let manifest_json = std::fs::read_to_string(manifest_path)?;
        let manifest: serde_json::Value = serde_json::from_str(&manifest_json)?;
        let samples_meta = manifest
            .get("samples")
            .and_then(|s| s.as_array())
            .ok_or_else(|| anyhow::anyhow!("No samples in manifest"))?;

        // Build ID -> species label mapping (same as training)
        let mut id_to_label: HashMap<String, String> = HashMap::new();
        for sample in samples_meta {
            if let (Some(id_val), Some(labels)) = (sample.get("id"), sample.get("labels")) {
                let id_str = id_val.as_str().unwrap_or("");

                // Use "output" field which contains species names (same as training)
                let label = if let Some(output) = labels.get("output").and_then(|s| s.as_str()) {
                    if output != "None" && !output.is_empty() {
                        let cleaned = output
                            .split_whitespace()
                            .take(4)
                            .collect::<Vec<_>>()
                            .join(" ");
                        cleaned
                    } else {
                        labels
                            .get("source_dataset")
                            .and_then(|s| s.as_str())
                            .unwrap_or("unknown")
                            .to_string()
                    }
                } else {
                    labels
                        .get("source_dataset")
                        .and_then(|s| s.as_str())
                        .unwrap_or("unknown")
                        .to_string()
                };

                id_to_label.insert(id_str.to_string(), label);
            }
        }

        // Load features
        let file = std::fs::File::open(cache_path)?;
        let reader = BufReader::new(file);
        let cache: HashMap<String, Vec<f32>> = bincode::deserialize_from(reader)?;

        let mut samples = Vec::new();
        let mut class_map: HashMap<String, usize> = HashMap::new();
        let mut next_class = 0;

        for (cache_key, features) in &cache {
            if features.len() < INPUT_DIM {
                continue;
            }

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

            samples.push(TestSample {
                features: features[..INPUT_DIM].to_vec(),
                class_label: class_idx,
            });
        }

        // Build class names list
        let mut class_names: Vec<(String, usize)> =
            class_map.iter().map(|(k, v)| (k.clone(), *v)).collect();
        class_names.sort_by_key(|(_, idx)| *idx);
        let class_names: Vec<String> = class_names.into_iter().map(|(k, _)| k).collect();

        println!("  Loaded {} samples, {} classes", samples.len(), next_class);
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
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║              ROSETTA-NET MODEL EVALUATION                             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    let device = LibTorchDevice::Cpu;

    // Load model config
    println!("Loading model configuration...");
    let config_json = std::fs::read_to_string("rosetta_net_best_config.json")?;
    let model_config: ModelConfigJson = serde_json::from_str(&config_json)?;
    println!(
        "  Input: {}D | Hidden: {}D | Latent: {}D | Classes: {}",
        model_config.input_dim,
        model_config.hidden_dim,
        model_config.latent_dim,
        model_config.num_classes
    );

    let config = RosettaNetConfig::new(
        model_config.input_dim,
        model_config.hidden_dim,
        model_config.latent_dim,
        model_config.num_classes,
    );

    // Initialize model
    let mut model = RosettaNet::<MyBackend>::init(&config, &device);

    // Load weights
    println!("\nLoading trained weights from rosetta_net_best.mpk...");
    let record = CompactRecorder::new()
        .load("rosetta_net_best.mpk".into(), &device)
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {:?}", e))?;
    model = model.load_record(record);
    println!("  ✓ Model loaded successfully");

    // Load test dataset
    println!("\nLoading test dataset...");
    let dataset = TestDataset::load()?;
    let class_names = if model_config.class_names.is_empty() {
        dataset.class_names.clone()
    } else {
        model_config.class_names
    };

    // Run evaluation
    println!("\n{}", "=".repeat(70));
    println!("RUNNING EVALUATION");
    println!("{}\n", "=".repeat(70));

    let batch_size = 128;
    let n_samples = dataset.samples.len();
    let n_batches = (n_samples + batch_size - 1) / batch_size;

    let mut correct = 0;
    let mut top5_correct = 0;
    let mut total = 0;
    let mut class_correct: Vec<usize> = vec![0; dataset.num_classes];
    let mut class_total: Vec<usize> = vec![0; dataset.num_classes];

    for batch_idx in 0..n_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(n_samples);
        let bs = end - start;

        // Build batch
        let mut features_flat = Vec::with_capacity(bs * INPUT_DIM);
        let mut labels = Vec::with_capacity(bs);

        for idx in start..end {
            let sample = &dataset.samples[idx];
            features_flat.extend_from_slice(&sample.features);
            labels.push(sample.class_label as i64);
        }

        // Create tensors
        let features = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(features_flat, [bs, INPUT_DIM]),
            &device,
        );

        // Forward pass
        let (_, logits) = model.forward(features);

        // Get predictions - Top-1
        let preds = logits
            .clone()
            .argmax(1)
            .into_data()
            .to_vec::<i64>()
            .unwrap_or_default();

        // Get Top-5 predictions
        let logits_data = logits.into_data().to_vec::<f32>().unwrap_or_default();
        let batch_preds: Vec<Vec<(usize, f32)>> = (0..bs)
            .map(|i| {
                let start = i * dataset.num_classes;
                let end = start + dataset.num_classes;
                let mut class_scores: Vec<(usize, f32)> = logits_data[start..end]
                    .iter()
                    .enumerate()
                    .map(|(j, &s)| (j, s))
                    .collect();
                class_scores
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                class_scores.into_iter().take(5).collect()
            })
            .collect();

        // Calculate accuracy
        for (i, (pred, label)) in preds.iter().zip(labels.iter()).enumerate() {
            let pred_idx = *pred as usize;
            let label_idx = *label as usize;

            if pred_idx < dataset.num_classes && label_idx < dataset.num_classes {
                // Top-1 accuracy
                if pred_idx == label_idx {
                    correct += 1;
                    class_correct[label_idx] += 1;
                }

                // Top-5 accuracy
                if let Some(top5) = batch_preds.get(i) {
                    if top5.iter().any(|(idx, _)| *idx == label_idx) {
                        top5_correct += 1;
                    }
                }

                class_total[label_idx] += 1;
                total += 1;
            }
        }
    }

    // Print results
    let accuracy = correct as f64 / total as f64 * 100.0;
    let top5_accuracy = top5_correct as f64 / total as f64 * 100.0;

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                    EVALUATION RESULTS                           ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║  Metric                    │  Value                            ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Total Samples             │  {:>6}                           ║",
        total
    );
    println!(
        "║  Correct Predictions       │  {:>6}                           ║",
        correct
    );
    println!(
        "║  Top-1 Accuracy            │  {:>6.2}%                          ║",
        accuracy
    );
    println!(
        "║  Top-5 Accuracy            │  {:>6.2}%                          ║",
        top5_accuracy
    );
    println!(
        "║  Number of Classes         │  {:>6}                           ║",
        dataset.num_classes
    );
    println!(
        "║  Random Baseline           │  {:>6.2}%                          ║",
        100.0 / dataset.num_classes as f64
    );
    println!(
        "║  Improvement over Random   │  {:.2}x                            ║",
        accuracy / (100.0 / dataset.num_classes as f64)
    );
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Per-class accuracy (top 10)
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                  TOP 10 CLASS ACCURACIES                        ║");
    println!("╠════════════════════════════════════════════════════════════════╣");

    let mut class_acc: Vec<(usize, f64)> = class_total
        .iter()
        .enumerate()
        .map(|(i, &total)| {
            let acc = if total > 0 {
                class_correct[i] as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            (i, acc)
        })
        .collect();
    class_acc.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (idx, acc) in class_acc.iter().take(10) {
        let name = class_names
            .get(*idx)
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        let truncated = if name.len() > 30 { &name[..30] } else { name };
        let samples = class_total.get(*idx).copied().unwrap_or(0);
        println!(
            "║  {:<30} │ {:>5.1}% ({:>4} samples)     ║",
            truncated, acc, samples
        );
    }
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Comparison
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    COMPARISON WITH BASELINES                    ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║  Method                    │  Accuracy   │  Notes              ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Random Baseline           │   {:>5.2}%    │  1/{} classes       ║",
        100.0 / dataset.num_classes as f64,
        dataset.num_classes
    );
    println!("║  Random Forest (45D)       │    3.70%    │  Species Top-1      ║");
    println!(
        "║  Rosetta-Net (105D MLP)    │   {:>5.2}%    │  Species Top-1      ║",
        accuracy
    );
    println!(
        "║  Rosetta-Net (105D MLP)    │   {:>5.2}%    │  Species Top-5      ║",
        top5_accuracy
    );
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Note about what was trained
    println!("\n📊 ANALYSIS:");
    if dataset.num_classes > 100 {
        println!("   • Trained on {} species classes", dataset.num_classes);
        println!(
            "   • Top-1: {:.2}% ({:.1}x better than RF baseline)",
            accuracy,
            accuracy / 3.70
        );
        println!(
            "   • Top-5: {:.2}% (correct species in top 5 guesses)",
            top5_accuracy
        );
        if top5_accuracy > 40.0 {
            println!("   ✓ EXCELLENT: Top-5 > 40% - model has strong discriminative power!");
        } else if top5_accuracy > 20.0 {
            println!("   ✓ GOOD: Top-5 > 20% - model is learning meaningful patterns");
        } else {
            println!("   ⚠ Top-5 < 20% - consider more training or larger model");
        }
    } else {
        println!("   This model was trained on SOURCE DATASETS (context recognition).");
    }

    Ok(())
}
