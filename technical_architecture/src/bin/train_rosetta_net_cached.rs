//! Train Rosetta-Net using Cached Features (Phase 3 only)
//!
//! Features:
//! - Loads cached 112D features (no re-extraction)
//! - Weighted loss for class imbalance
//! - Checkpoint saving
//! - Convergence monitoring

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::Instant;

const FEATURE_DIM: usize = 112;

#[derive(Debug, Deserialize)]
struct BeansManifest {
    dataset: String,
    n_samples: usize,
    samples: Vec<BeansSample>,
}

#[derive(Debug, Deserialize)]
struct BeansSample {
    audio_file: String,
    n_samples: u32,
    labels: BeansLabels,
}

#[derive(Debug, Deserialize, Clone)]
struct BeansLabels {
    output: String,
    task: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheManifest {
    entries: HashMap<String, String>,
    feature_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct TrainingHistory {
    epochs: Vec<usize>,
    train_loss: Vec<f32>,
    train_accuracy: Vec<f32>,
    val_loss: Vec<f32>,
    val_accuracy: Vec<f32>,
}

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Rosetta-Net Training (Using Cached 112D Features)                 ║");
    println!("║  With: Weighted Loss, Checkpoint Saving, Convergence Monitoring    ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let start = Instant::now();

    // Load manifest
    let manifest_path = "beans_zero_full_manifest.json";
    println!("Loading manifest from: {}", manifest_path);
    let manifest_data = fs::read_to_string(manifest_path)?;
    let manifest: BeansManifest = serde_json::from_str(&manifest_data)?;
    println!("Total samples: {}", manifest.n_samples);

    // Load cache manifest
    let cache_dir = Path::new("beans_feature_cache_112d");
    let cache_manifest_path = cache_dir.join("cache_manifest.json");
    println!("Loading cache manifest from: {:?}", cache_manifest_path);

    let cache_data = fs::read_to_string(&cache_manifest_path)?;
    let cache_manifest: CacheManifest = serde_json::from_str(&cache_data)?;
    println!("Cached features available: {}", cache_manifest.entries.len());

    // Build label mapping and count class frequencies
    let mut label_to_idx: HashMap<String, usize> = HashMap::new();
    let mut idx_to_label: Vec<String> = Vec::new();
    let mut class_counts: Vec<usize> = Vec::new();

    for sample in &manifest.samples {
        let label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };
        if !label_to_idx.contains_key(&label) {
            label_to_idx.insert(label.clone(), idx_to_label.len());
            idx_to_label.push(label.clone());
            class_counts.push(0);
        }
        class_counts[label_to_idx[&label]] += 1;
    }
    let n_classes = idx_to_label.len();
    println!("Number of classes: {}", n_classes);

    // Compute class weights (inverse frequency)
    println!();
    println!("Computing class weights for imbalanced data...");
    let total_samples: usize = class_counts.iter().sum();
    let class_weights: Vec<f32> = class_counts
        .iter()
        .map(|&count| {
            if count > 0 {
                (total_samples as f32 / (n_classes as f32 * count as f32)).min(10.0) // Cap at 10x
            } else {
                1.0
            }
        })
        .collect();

    println!("  Total samples: {}", total_samples);
    println!("  Max class weight: {:.2}x", class_weights.iter().cloned().fold(0.0, f32::max));
    println!("  Min class weight: {:.2}x", class_weights.iter().cloned().fold(f32::INFINITY, f32::min));

    // Load all features from cache
    println!();
    println!("Loading cached features...");
    let mut all_features: Vec<Vec<f32>> = Vec::with_capacity(manifest.n_samples);
    let mut all_labels: Vec<usize> = Vec::with_capacity(manifest.n_samples);
    let mut hits = 0;
    let mut misses = 0;

    for sample in &manifest.samples {
        let audio_file = &sample.audio_file;
        let label = if sample.labels.output != "None" {
            sample.labels.output.clone()
        } else {
            format!("task_{}", sample.labels.task)
        };
        let label_idx = label_to_idx[&label];

        if let Some(cache_file) = cache_manifest.entries.get(audio_file) {
            let full_path = cache_dir.join(cache_file);
            if full_path.exists() {
                if let Ok(file) = fs::File::open(&full_path) {
                    let reader = BufReader::new(file);
                    if let Ok(features) = bincode::deserialize_from::<_, Vec<f32>>(reader) {
                        all_features.push(features);
                        all_labels.push(label_idx);
                        hits += 1;
                        continue;
                    }
                }
            }
        }
        misses += 1;
    }

    println!("Loaded {} feature vectors", all_features.len());
    println!("Cache hits: {}, misses: {}", hits, misses);

    if all_features.is_empty() {
        anyhow::bail!("No features loaded from cache!");
    }

    // Split into train/val (90/10)
    println!();
    println!("Splitting data: 90% train, 10% validation...");
    let n_train = (all_features.len() as f32 * 0.9) as usize;
    let n_val = all_features.len() - n_train;

    // Shuffle indices
    let mut indices: Vec<usize> = (0..all_features.len()).collect();
    for i in 0..indices.len() {
        let j = (rand_u32() as usize) % indices.len();
        indices.swap(i, j);
    }

    let train_indices: Vec<usize> = indices[..n_train].to_vec();
    let val_indices: Vec<usize> = indices[n_train..].to_vec();

    println!("  Train samples: {}", n_train);
    println!("  Val samples: {}", n_val);

    // Normalize features using training set statistics
    println!();
    println!("Normalizing features...");
    let feature_means: Vec<f32> = (0..FEATURE_DIM)
        .map(|i| {
            train_indices
                .iter()
                .map(|&idx| all_features[idx].get(i).copied().unwrap_or(0.0))
                .sum::<f32>()
                / n_train as f32
        })
        .collect();

    let feature_stds: Vec<f32> = (0..FEATURE_DIM)
        .map(|i| {
            let mean = feature_means[i];
            let variance: f64 = train_indices
                .iter()
                .map(|&idx| {
                    let v = all_features[idx].get(i).copied().unwrap_or(0.0) - mean;
                    (v as f64) * (v as f64)
                })
                .sum::<f64>()
                / n_train as f64;
            (variance.sqrt() as f32).max(1e-8)
        })
        .collect();

    // Train Rosetta-Net
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Training Rosetta-Net (112D features)                              ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let mut rosetta_net = RosettaNet::new(FEATURE_DIM, 256, n_classes);
    println!("Architecture: {} -> 256 -> {}", FEATURE_DIM, n_classes);
    println!("Training for 200 epochs (lr=0.01, weighted loss)...");

    let n_epochs = 200;
    let learning_rate = 0.01f32;

    let mut history = TrainingHistory {
        epochs: Vec::new(),
        train_loss: Vec::new(),
        train_accuracy: Vec::new(),
        val_loss: Vec::new(),
        val_accuracy: Vec::new(),
    };

    let mut best_val_accuracy = 0.0f32;
    let mut best_epoch = 0;
    let mut patience_counter = 0;
    let patience = 20; // Early stopping patience

    for epoch in 0..n_epochs {
        // Shuffle training indices
        let mut epoch_indices = train_indices.clone();
        for i in 0..epoch_indices.len() {
            let j = (rand_u32() as usize) % epoch_indices.len();
            epoch_indices.swap(i, j);
        }

        // Training
        let mut total_train_loss = 0.0f32;
        let mut train_correct = 0usize;

        for &idx in &epoch_indices {
            let features: Vec<f32> = all_features[idx]
                .iter()
                .enumerate()
                .map(|(i, &v)| (v - feature_means[i]) / feature_stds[i])
                .collect();
            let label = all_labels[idx];
            let weight = class_weights[label];

            // Forward pass
            let output = rosetta_net.forward(&features);

            // Weighted cross-entropy loss
            let max_output = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = output.iter().map(|&o| (o - max_output).exp()).sum();
            let log_sum_exp = max_output + exp_sum.ln();
            let loss = weight * (log_sum_exp - output[label]);
            total_train_loss += loss;

            // Check prediction
            let predicted = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            if predicted == label {
                train_correct += 1;
            }

            // Backward pass with weighted gradients
            let mut output_grad = vec![0.0f32; n_classes];
            for i in 0..n_classes {
                let softmax = (output[i] - max_output).exp() / exp_sum;
                output_grad[i] = weight * if i == label { softmax - 1.0 } else { softmax };
            }

            rosetta_net.backward(&features, &output_grad, learning_rate);
        }

        let train_accuracy = train_correct as f32 / n_train as f32 * 100.0;
        let avg_train_loss = total_train_loss / n_train as f32;

        // Validation
        let mut total_val_loss = 0.0f32;
        let mut val_correct = 0usize;

        for &idx in &val_indices {
            let features: Vec<f32> = all_features[idx]
                .iter()
                .enumerate()
                .map(|(i, &v)| (v - feature_means[i]) / feature_stds[i])
                .collect();
            let label = all_labels[idx];

            let output = rosetta_net.forward(&features);

            let max_output = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = output.iter().map(|&o| (o - max_output).exp()).sum();
            let log_sum_exp = max_output + exp_sum.ln();
            let loss = log_sum_exp - output[label];
            total_val_loss += loss;

            let predicted = output
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            if predicted == label {
                val_correct += 1;
            }
        }

        let val_accuracy = val_correct as f32 / n_val as f32 * 100.0;
        let avg_val_loss = total_val_loss / n_val as f32;

        history.epochs.push(epoch + 1);
        history.train_loss.push(avg_train_loss);
        history.train_accuracy.push(train_accuracy);
        history.val_loss.push(avg_val_loss);
        history.val_accuracy.push(val_accuracy);

        // Check for improvement
        if val_accuracy > best_val_accuracy {
            best_val_accuracy = val_accuracy;
            best_epoch = epoch + 1;
            patience_counter = 0;

            // Save best model checkpoint
            rosetta_net.save(Path::new("rosetta_net_best_checkpoint.json"))?;
        } else {
            patience_counter += 1;
        }

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!(
                "  Epoch {:3}/{}: Train Loss={:.4}, Acc={:.1}% | Val Loss={:.4}, Acc={:.1}% | Best={:.1}% (epoch {})",
                epoch + 1, n_epochs, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy, best_val_accuracy, best_epoch
            );
        }

        // Early stopping
        if patience_counter >= patience {
            println!();
            println!("  Early stopping at epoch {} (no improvement for {} epochs)", epoch + 1, patience);
            break;
        }
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Training Complete                                                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Best Validation Accuracy: {:.2}% (epoch {})", best_val_accuracy, best_epoch);

    // Save final model
    rosetta_net.save(Path::new("rosetta_net_model_112d.json"))?;
    println!("Saved final model to: rosetta_net_model_112d.json");
    println!("Saved best checkpoint to: rosetta_net_best_checkpoint.json");

    // Save training history
    let history_json = serde_json::to_string_pretty(&history)?;
    fs::write("rosetta_net_training_history.json", history_json)?;
    println!("Saved training history to: rosetta_net_training_history.json");

    // Save normalization parameters
    #[derive(Serialize)]
    struct NormalizationParams {
        means: Vec<f32>,
        stds: Vec<f32>,
        label_to_idx: HashMap<String, usize>,
        idx_to_label: Vec<String>,
    }

    let norm_params = NormalizationParams {
        means: feature_means.clone(),
        stds: feature_stds.clone(),
        label_to_idx: label_to_idx.clone(),
        idx_to_label: idx_to_label.clone(),
    };

    let norm_json = serde_json::to_string_pretty(&norm_params)?;
    fs::write("rosetta_net_normalization.json", norm_json)?;
    println!("Saved normalization params to: rosetta_net_normalization.json");

    println!();
    println!("Training completed in {:.1}s", start.elapsed().as_secs_f32());

    Ok(())
}

// Simple neural network
#[derive(Debug, Serialize, Deserialize)]
struct RosettaNet {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    weights_ih: Vec<Vec<f32>>,
    bias_h: Vec<f32>,
    weights_ho: Vec<Vec<f32>>,
    bias_o: Vec<f32>,
    #[serde(skip)]
    hidden: Vec<f32>,
}

impl RosettaNet {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let scale_ih = (2.0 / input_dim as f64).sqrt() as f32;
        let scale_ho = (2.0 / hidden_dim as f64).sqrt() as f32;

        let weights_ih: Vec<Vec<f32>> = (0..hidden_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| {
                        let r = (rand_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
                        r * scale_ih
                    })
                    .collect()
            })
            .collect();

        let bias_h: Vec<f32> = vec![0.0; hidden_dim];

        let weights_ho: Vec<Vec<f32>> = (0..output_dim)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| {
                        let r = (rand_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
                        r * scale_ho
                    })
                    .collect()
            })
            .collect();

        let bias_o: Vec<f32> = vec![0.0; output_dim];

        Self {
            input_dim,
            hidden_dim,
            output_dim,
            weights_ih,
            bias_h,
            weights_ho,
            bias_o,
            hidden: vec![0.0; hidden_dim],
        }
    }

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        // Hidden layer with ReLU
        self.hidden = Vec::with_capacity(self.hidden_dim);
        for i in 0..self.hidden_dim {
            let mut sum = self.bias_h[i];
            for (j, &x) in input.iter().enumerate().take(self.input_dim) {
                sum += self.weights_ih[i][j] * x;
            }
            self.hidden.push(sum.max(0.0)); // ReLU
        }

        // Output layer
        let mut output = Vec::with_capacity(self.output_dim);
        for i in 0..self.output_dim {
            let mut sum = self.bias_o[i];
            for (j, &h) in self.hidden.iter().enumerate() {
                sum += self.weights_ho[i][j] * h;
            }
            output.push(sum);
        }
        output
    }

    fn backward(&mut self, input: &[f32], output_grad: &[f32], lr: f32) {
        // Gradient for hidden layer
        let mut hidden_grad = vec![0.0f32; self.hidden_dim];
        for i in 0..self.output_dim {
            for j in 0..self.hidden_dim {
                hidden_grad[j] += output_grad[i] * self.weights_ho[i][j];
            }
        }

        // ReLU derivative
        for i in 0..self.hidden_dim {
            if self.hidden[i] <= 0.0 {
                hidden_grad[i] = 0.0;
            }
        }

        // Update weights_ho and bias_o
        for i in 0..self.output_dim {
            for j in 0..self.hidden_dim {
                self.weights_ho[i][j] -= lr * output_grad[i] * self.hidden[j];
            }
            self.bias_o[i] -= lr * output_grad[i];
        }

        // Update weights_ih and bias_h
        for i in 0..self.hidden_dim {
            for j in 0..self.input_dim {
                self.weights_ih[i][j] -= lr * hidden_grad[i] * input[j];
            }
            self.bias_h[i] -= lr * hidden_grad[i];
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
}

// Simple XOR-shift RNG
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
