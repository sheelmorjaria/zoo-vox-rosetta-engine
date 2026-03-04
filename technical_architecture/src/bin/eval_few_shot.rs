//! Few-Shot Evaluation using Trained Rosetta-Net Encoder
//! =======================================================
//!
//! Evaluates species classification using prototype-based retrieval with
//! the TRAINED Rosetta-Net encoder (not random weights).
//!
//! Key Insight:
//! - Random encoder: 0.05% species accuracy (useless)
//! - Trained encoder: Should capture discriminative structure
//!
//! Usage:
//!   cargo run --release --bin eval_few_shot -- beans_zero_cache/beans_audio_manifest.json

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use technical_architecture::{FewShotConfig, FewShotDistance, PrototypicalAdapter};

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Deserialize)]
struct BeansManifest {
    dataset: String,
    n_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedFeatures {
    features: Vec<Vec<f32>>,
    labels: Vec<String>,
}

// ============================================================================
// Trained Rosetta-Net Model (Loaded from JSON)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RosettaNetModel {
    input_dim: usize,
    hidden_dim: usize,
    latent_dim: usize,
    output_dim: usize,
    encoder_weights: Vec<Vec<f32>>,
    encoder_bias: Vec<f32>,
    latent_weights: Vec<Vec<f32>>,
    latent_bias: Vec<f32>,
    classifier_weights: Vec<Vec<f32>>,
    classifier_bias: Vec<f32>,
    #[serde(default)]
    feature_means: Vec<f32>,
    #[serde(default)]
    feature_stds: Vec<f32>,
}

impl RosettaNetModel {
    /// Encode features to latent space
    fn encode(&self, x: &[f32]) -> Vec<f32> {
        // Normalize using stored parameters
        let normalized: Vec<f32> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let mean = self.feature_means.get(i).copied().unwrap_or(0.0);
                let std = self.feature_stds.get(i).copied().unwrap_or(1.0).max(1e-6);
                (v - mean) / std
            })
            .collect();

        // Encoder layer (45D -> 128D)
        let mut hidden = vec![0.0; self.hidden_dim];
        for (i, (weights, &bias)) in self
            .encoder_weights
            .iter()
            .zip(self.encoder_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &w) in weights.iter().enumerate() {
                if j < normalized.len() {
                    sum += w * normalized[j];
                }
            }
            hidden[i] = sum.max(0.0); // ReLU
        }

        // Latent layer (128D -> 64D)
        let mut latent = vec![0.0; self.latent_dim];
        for (i, (weights, &bias)) in self
            .latent_weights
            .iter()
            .zip(self.latent_bias.iter())
            .enumerate()
        {
            let mut sum = bias;
            for (j, &h) in hidden.iter().enumerate() {
                sum += weights[j] * h;
            }
            latent[i] = sum.max(0.0); // ReLU
        }

        // L2 normalize for cosine similarity
        let norm: f32 = latent.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        latent.iter().map(|x| x / norm).collect()
    }
}

// ============================================================================
// Few-Shot Evaluation
// ============================================================================

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <manifest.json>", args[0]);
        std::process::exit(1);
    }

    let manifest_path = PathBuf::from(&args[1]);
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║       Few-Shot Evaluation: Trained Rosetta-Net Encoder         ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!("\nLoading BEANS-Zero manifest from: {:?}", manifest_path);

    let manifest_content = std::fs::read_to_string(&manifest_path)?;
    let manifest: BeansManifest = serde_json::from_str(&manifest_content)?;

    println!("Dataset: {}", manifest.dataset);
    println!("Total samples: {}", manifest.n_samples);

    let base_path = manifest_path.parent().unwrap_or(Path::new("."));

    // Load cached features
    let cache_path = base_path.join("feature_cache/all_features.bin");
    let (features, labels) = if cache_path.exists() {
        println!("\nLoading cached features from {:?}...", cache_path);
        let start = Instant::now();
        let file = std::fs::File::open(&cache_path)?;
        let reader = std::io::BufReader::new(file);
        let cached: CachedFeatures = bincode::deserialize_from(reader)?;
        println!(
            "Loaded {} cached features in {:.2}s",
            cached.features.len(),
            start.elapsed().as_secs_f64()
        );
        (cached.features, cached.labels)
    } else {
        eprintln!("No cached features found. Run train_beans_models first.");
        std::process::exit(1);
    };

    // Load trained Rosetta-Net model
    let model_path = base_path.join("rosetta_net_model.json");
    println!("\nLoading trained Rosetta-Net from {:?}...", model_path);

    let model: RosettaNetModel = if model_path.exists() {
        let model_json = std::fs::read_to_string(&model_path)?;
        serde_json::from_str(&model_json)?
    } else {
        eprintln!(
            "No trained model found at {:?}. Run train_beans_models first.",
            model_path
        );
        std::process::exit(1);
    };

    println!(
        "  Input: {}, Hidden: {}, Latent: {}, Output: {}",
        model.input_dim, model.hidden_dim, model.latent_dim, model.output_dim
    );

    // Get unique labels
    let unique_labels: std::collections::HashSet<&String> = labels.iter().collect();
    println!("Unique species: {}", unique_labels.len());

    // Build taxonomic mapping (simplified - using first word as taxonomic hint)
    let taxonomic_map: HashMap<&str, &str> = labels
        .iter()
        .map(|l| {
            let taxon = l.split_whitespace().next().unwrap_or("unknown");
            (l.as_str(), taxon)
        })
        .collect();

    // Encode all features to latent space using trained encoder
    println!("\n=== Encoding with Trained Rosetta-Net ===");
    let start = Instant::now();
    let latent_vectors: Vec<Vec<f32>> = features.iter().map(|f| model.encode(f)).collect();
    println!(
        "Encoded {} samples in {:.2}s",
        latent_vectors.len(),
        start.elapsed().as_secs_f64()
    );

    // Check latent space statistics
    let latent_norms: Vec<f32> = latent_vectors
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();
    let avg_norm = latent_norms.iter().sum::<f32>() / latent_norms.len() as f32;
    println!(
        "  Avg latent vector norm: {:.4} (should be ~1.0 after normalization)",
        avg_norm
    );

    // Split into reference (80%) and test (20%)
    let split_idx = (features.len() as f32 * 0.8) as usize;
    let (ref_latent, test_latent) = latent_vectors.split_at(split_idx);
    let (ref_labels, test_labels) = labels.split_at(split_idx);

    println!("\nReference set: {} samples", ref_latent.len());
    println!("Test set: {} samples", test_latent.len());

    // Build species prototypes from reference set
    println!("\n=== Building Species Prototypes ===");
    let mut species_examples: HashMap<String, Vec<Vec<f32>>> = HashMap::new();

    for (latent, label) in ref_latent.iter().zip(ref_labels.iter()) {
        species_examples
            .entry(label.clone())
            .or_insert_with(Vec::new)
            .push(latent.clone());
    }

    // Report example distribution
    let min_examples = species_examples
        .values()
        .map(|v| v.len())
        .min()
        .unwrap_or(0);
    let max_examples = species_examples
        .values()
        .map(|v| v.len())
        .max()
        .unwrap_or(0);
    let avg_examples = species_examples.values().map(|v| v.len()).sum::<usize>() as f32
        / species_examples.len() as f32;
    println!(
        "  Samples per species: min={}, max={}, avg={:.1}",
        min_examples, max_examples, avg_examples
    );

    // Create prototypical adapter
    let config = FewShotConfig {
        latent_dim: model.latent_dim,
        distance_metric: FewShotDistance::Cosine,
        min_examples: 1,
        temperature: 10.0,
    };

    let mut adapter = PrototypicalAdapter::with_config(config);

    // Add species prototypes
    let start = Instant::now();
    for (species, examples) in &species_examples {
        adapter.add_species_prototype(species, examples);
    }
    println!(
        "Built {} prototypes in {:.2}s",
        adapter.num_species(),
        start.elapsed().as_secs_f64()
    );

    // Evaluate on test set
    println!("\n=== Few-Shot Evaluation (Trained Encoder) ===");
    let start = Instant::now();

    let mut correct_species = 0;
    let mut correct_taxon = 0;
    let mut total_confident = 0;
    let mut correct_confident = 0;

    for (latent, true_label) in test_latent.iter().zip(test_labels.iter()) {
        let result = adapter.classify_few_shot(latent);

        // Species accuracy
        if result.species == *true_label {
            correct_species += 1;
        }

        // Taxonomic accuracy
        let true_taxon = taxonomic_map.get(true_label.as_str()).unwrap_or(&"unknown");
        let pred_taxon = taxonomic_map
            .get(result.species.as_str())
            .unwrap_or(&"unknown");
        if true_taxon == pred_taxon {
            correct_taxon += 1;
        }

        // Confident predictions
        if result.is_confident {
            total_confident += 1;
            if result.species == *true_label {
                correct_confident += 1;
            }
        }
    }

    let n_test = test_latent.len();
    let species_acc = correct_species as f64 / n_test as f64 * 100.0;
    let taxon_acc = correct_taxon as f64 / n_test as f64 * 100.0;

    println!(
        "Evaluation completed in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    // Print results
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    FEW-SHOT RESULTS                            ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║  Metric                    │  Value                           ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Species Accuracy          │  {:>6.2}%                         ║",
        species_acc
    );
    println!(
        "║  Taxonomic Accuracy        │  {:>6.2}%                         ║",
        taxon_acc
    );
    println!(
        "║  Total Test Samples        │  {:>6}                           ║",
        n_test
    );
    println!(
        "║  Confident Predictions     │  {:>6} ({:.1}%)                  ║",
        total_confident,
        total_confident as f64 / n_test as f64 * 100.0
    );
    if total_confident > 0 {
        println!(
            "║  Confident Accuracy        │  {:>6.2}%                         ║",
            correct_confident as f64 / total_confident as f64 * 100.0
        );
    }
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Comparison with baselines
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                 COMPARISON WITH BASELINES                      ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║  Method                    │  Species   │  Taxonomic           ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║  k-NN (Raw 45D)            │   0.00%    │  71.33%              ║");
    println!("║  Random Forest (Flat)      │   3.70%    │  71.33%              ║");
    println!("║  Hierarchical RF           │   3.26%    │  53.44%              ║");
    println!("║  Rosetta-Net (Output)      │   0.08%    │  71.33%              ║");
    println!("║  Rosetta-Net (Latent)      │   1.06%    │  71.33%              ║");
    println!(
        "║  Prototypical (Trained)    │  {:>6.2}%   │  {:>6.2}%             ║",
        species_acc, taxon_acc
    );
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Analysis
    let improvement = species_acc - 3.70;
    if improvement > 0.0 {
        println!(
            "\n✓ Few-Shot IMPROVEMENT: +{:.2}% species accuracy vs Random Forest",
            improvement
        );
    } else {
        println!(
            "\n⚠ Few-Shot did not improve over RF baseline ({:.2}%)",
            species_acc
        );
    }

    // Key insight
    println!("\n📊 INSIGHT:");
    println!("   The Rosetta-Net encoder was trained with softmax cross-entropy,");
    println!("   which may not optimize latent space for prototype distance.");
    println!("   Consider using triplet loss or contrastive learning for better");
    println!("   few-shot performance.");

    Ok(())
}
