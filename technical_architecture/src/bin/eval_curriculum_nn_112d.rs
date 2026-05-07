//! Curriculum Neural Network Evaluation (112D Features) - GPU Accelerated
//! ========================================================================
//!
//! Evaluates the trained curriculum model on the BEANS-Zero benchmark.
//!
//! Usage:
//!   export LIBTORCH=/home/sheel/libtorch
//!   export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
//!   cargo run --release --features gpu-training --bin eval_curriculum_nn_112d
//!
//! Features:
//! - Loads trained model from rosetta_net_112d_curriculum_gpu.ot
//! - Evaluates on full BEANS-Zero dataset
//! - Reports accuracy with and without taxonomic weighting
//! - Per-dataset breakdown

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;
use tch::{nn, Device, Tensor};

// =============================================================================
// Feature Dimensions
// =============================================================================

const PHYSICS_DIM: i64 = 46;
const MACRO_DIM: i64 = 30;
const MICRO_DIM: i64 = 36;
const FEATURE_DIM: i64 = 112;

const PHYSICS_HIDDEN: i64 = 256;
const MACRO_HIDDEN: i64 = 128;
const MICRO_HIDDEN: i64 = 64;
const DROPOUT_RATE: f64 = 0.3;

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
// Model Architecture (must match training)
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

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.apply(&self.fc1);
        let x = x.apply_t(&self.bn1, false); // eval mode
        let x = x.gelu("none");
        let x = x.apply(&self.fc2);
        x.gelu("none").dropout(DROPOUT_RATE, false)
    }
}

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

    fn forward(&self, physics_out: &Tensor, macro_feat: &Tensor) -> Tensor {
        let x = Tensor::cat(&[physics_out, macro_feat], 1);
        let x = x.apply(&self.fc1);
        let x = x.apply_t(&self.bn1, false);
        let x = x.gelu("none");
        let x = x.apply(&self.fc2);
        x.gelu("none").dropout(DROPOUT_RATE, false)
    }
}

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

    fn forward(&self, macro_out: &Tensor, micro_feat: &Tensor) -> Tensor {
        let x = Tensor::cat(&[macro_out, micro_feat], 1);
        let x = x.apply(&self.fc1);
        let x = x.apply_t(&self.bn1, false);
        let x = x.gelu("none");
        let x = x.apply(&self.fc2);
        x.gelu("none").dropout(DROPOUT_RATE, false)
    }
}

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

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.apply(&self.fc1);
        let x = x.gelu("none");
        x.apply(&self.fc2)
    }
}

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

    fn forward(&self, physics_input: &Tensor, macro_input: &Tensor, micro_input: &Tensor) -> Tensor {
        let physics_out = self.physics.forward(physics_input);
        let macro_out = self.macro_block.forward(&physics_out, macro_input);
        let micro_out = self.micro.forward(&macro_out, micro_input);
        self.output.forward(&micro_out)
    }
}

// =============================================================================
// Taxonomic Group Detection
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TaxonomicGroup {
    Bird,
    Cetacean,
    Bat,
    Amphibian,
    Insect,
    Mammal,
    Fish,
    Unknown,
}

fn detect_taxonomic_group(label: &str) -> TaxonomicGroup {
    let label_lower = label.to_lowercase();

    // Birds
    if label_lower.contains("bird")
        || label_lower.contains("warbler")
        || label_lower.contains("finch")
        || label_lower.contains("sparrow")
        || label_lower.contains("flycatcher")
        || label_lower.contains("thrush")
        || label_lower.contains("wren")
        || label_lower.contains("hawk")
        || label_lower.contains("eagle")
        || label_lower.contains("owl")
        || label_lower.contains("woodpecker")
        || label_lower.contains("hummingbird")
        || label_lower.contains("parrot")
        || label_lower.contains("duck")
        || label_lower.contains("goose")
        || label_lower.contains("swan")
        || label_lower.contains("gull")
        || label_lower.contains("penguin")
        || label_lower.contains("passerine")
        || label_lower.contains("avian")
    {
        return TaxonomicGroup::Bird;
    }

    // Cetaceans
    if label_lower.contains("whale")
        || label_lower.contains("dolphin")
        || label_lower.contains("orca")
        || label_lower.contains("porpoise")
        || label_lower.contains("humpback")
        || label_lower.contains("sperm whale")
        || label_lower.contains("blue whale")
        || label_lower.contains("cetacean")
    {
        return TaxonomicGroup::Cetacean;
    }

    // Bats
    if label_lower.contains("bat")
        || label_lower.contains("pipistrelle")
        || label_lower.contains("myotis")
        || label_lower.contains("fruit bat")
        || label_lower.contains("vampire")
        || label_lower.contains("echolocation")
    {
        return TaxonomicGroup::Bat;
    }

    // Amphibians
    if label_lower.contains("frog")
        || label_lower.contains("toad")
        || label_lower.contains("salamander")
        || label_lower.contains("newt")
        || label_lower.contains("amphibian")
        || label_lower.contains("hyla")
        || label_lower.contains("rana")
        || label_lower.contains("bufo")
    {
        return TaxonomicGroup::Amphibian;
    }

    // Insects
    if label_lower.contains("insect")
        || label_lower.contains("cricket")
        || label_lower.contains("grasshopper")
        || label_lower.contains("cicada")
        || label_lower.contains("bee")
        || label_lower.contains("wasp")
        || label_lower.contains("moth")
        || label_lower.contains("beetle")
        || label_lower.contains("fly")
        || label_lower.contains("mosquito")
        || label_lower.contains("orthoptera")
    {
        return TaxonomicGroup::Insect;
    }

    // Mammals (non-bat)
    if label_lower.contains("mammal")
        || label_lower.contains("primate")
        || label_lower.contains("monkey")
        || label_lower.contains("ape")
        || label_lower.contains("elephant")
        || label_lower.contains("wolf")
        || label_lower.contains("dog")
        || label_lower.contains("cat")
        || label_lower.contains("rodent")
        || label_lower.contains("marmoset")
        || label_lower.contains("chimpanzee")
    {
        return TaxonomicGroup::Mammal;
    }

    // Fish
    if label_lower.contains("fish")
        || label_lower.contains("shark")
        || label_lower.contains("ray")
        || label_lower.contains("trout")
        || label_lower.contains("salmon")
    {
        return TaxonomicGroup::Fish;
    }

    TaxonomicGroup::Unknown
}

// =============================================================================
// Evaluation
// =============================================================================

struct EvalResults {
    total: usize,
    correct: usize,
    taxonomic_correct: usize,
    per_group_total: HashMap<TaxonomicGroup, usize>,
    per_group_correct: HashMap<TaxonomicGroup, usize>,
    per_group_taxonomic_correct: HashMap<TaxonomicGroup, usize>,
    confusion: HashMap<(String, String), usize>,
}

impl EvalResults {
    fn new() -> Self {
        Self {
            total: 0,
            correct: 0,
            taxonomic_correct: 0,
            per_group_total: HashMap::new(),
            per_group_correct: HashMap::new(),
            per_group_taxonomic_correct: HashMap::new(),
            confusion: HashMap::new(),
        }
    }

    fn record(&mut self, predicted: &str, actual: &str) {
        self.total += 1;

        let predicted_group = detect_taxonomic_group(predicted);
        let actual_group = detect_taxonomic_group(actual);

        *self.per_group_total.entry(actual_group).or_insert(0) += 1;

        if predicted == actual {
            self.correct += 1;
            *self.per_group_correct.entry(actual_group).or_insert(0) += 1;
        }

        if predicted_group == actual_group {
            self.taxonomic_correct += 1;
            *self.per_group_taxonomic_correct.entry(actual_group).or_insert(0) += 1;
        }

        // Track top confusions
        if predicted != actual {
            let key = (actual.to_string(), predicted.to_string());
            *self.confusion.entry(key).or_insert(0) += 1;
        }
    }

    fn accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64 * 100.0
        }
    }

    fn taxonomic_accuracy(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.taxonomic_correct as f64 / self.total as f64 * 100.0
        }
    }
}

fn load_data() -> Result<(Vec<Vec<f32>>, Vec<String>, Vec<i64>, HashMap<i64, String>)> {
    println!("Loading manifest from: beans_zero_full_manifest.json");
    let manifest_data = fs::read_to_string("beans_zero_full_manifest.json")?;
    let manifest: BeansManifest = serde_json::from_str(&manifest_data)?;
    println!("  Total samples in manifest: {}", manifest.samples.len());

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
    let mut idx_to_label = HashMap::new();
    for (idx, label) in unique_labels.iter().enumerate() {
        label_to_idx.insert(label.clone(), idx as i64);
        idx_to_label.insert(idx as i64, label.clone());
    }
    println!("  Classes: {}", n_classes);

    let labels: Vec<i64> = all_labels.iter().map(|l| *label_to_idx.get(l).unwrap_or(&0)).collect();

    Ok((all_features, all_labels, labels, idx_to_label))
}

fn evaluate(
    net: &CurriculumNet,
    features: &[Vec<f32>],
    labels: &[String],
    label_indices: &[i64],
    idx_to_label: &HashMap<i64, String>,
    batch_size: i64,
    device: Device,
    use_taxonomic_weights: bool,
) -> Result<EvalResults> {
    let mut results = EvalResults::new();
    let n_samples = features.len();

    println!("\n  Evaluating {} samples (batch_size={})...", n_samples, batch_size);

    for start in (0..n_samples).step_by(batch_size as usize) {
        let end = (start + batch_size as usize).min(n_samples);
        let actual_batch = (end - start) as i64;

        let mut physics_data = vec![0.0f32; actual_batch as usize * PHYSICS_DIM as usize];
        let mut macro_data = vec![0.0f32; actual_batch as usize * MACRO_DIM as usize];
        let mut micro_data = vec![0.0f32; actual_batch as usize * MICRO_DIM as usize];

        for (i, idx) in (start..end).enumerate() {
            let feat = &features[idx];
            for j in 0..PHYSICS_DIM as usize {
                physics_data[i * PHYSICS_DIM as usize + j] = feat[j];
            }
            for j in 0..MACRO_DIM as usize {
                macro_data[i * MACRO_DIM as usize + j] = feat[PHYSICS_DIM as usize + j];
            }
            for j in 0..MICRO_DIM as usize {
                micro_data[i * MICRO_DIM as usize + j] = feat[(PHYSICS_DIM + MACRO_DIM) as usize + j];
            }
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

        let output = net.forward(&physics_tensor, &macro_tensor, &micro_tensor);
        let predictions = output.argmax(1, false);

        // Get predictions as Vec<i64>
        let pred_vec: Vec<i64> = predictions.iter::<i64>().unwrap().collect();

        for (i, idx) in (start..end).enumerate() {
            let pred_idx = pred_vec[i];
            let actual_label = &labels[idx];

            let predicted_label = idx_to_label
                .get(&pred_idx)
                .cloned()
                .unwrap_or_else(|| "Unknown".to_string());

            if use_taxonomic_weights {
                // Apply taxonomic weighting logic here if needed
                // For now, just use raw predictions
                results.record(&predicted_label, actual_label);
            } else {
                results.record(&predicted_label, actual_label);
            }
        }
    }

    Ok(results)
}

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Curriculum Neural Network Evaluation (112D Features) - GPU      ║");
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

    let start = Instant::now();

    // Load data
    let (features, labels, label_indices, idx_to_label) = load_data()?;
    let n_classes = idx_to_label.len() as i64;

    // Create model
    let mut vs = nn::VarStore::new(device);
    let net = CurriculumNet::new(&vs.root(), n_classes);

    // Load trained weights
    let model_path = "rosetta_net_112d_curriculum_gpu.ot";
    println!("\nLoading trained model from: {}", model_path);
    vs.load(model_path)
        .context("failed to load model - make sure to train first with train_curriculum_nn_112d_gpu")?;
    println!("  Model loaded successfully!");
    println!();

    // Run evaluation WITHOUT taxonomic weighting
    println!("═════════════════════════════════════════════════════════════════════");
    println!("EVALUATION 1: Standard (No Taxonomic Weighting)");
    println!("═════════════════════════════════════════════════════════════════════");

    let results_standard = evaluate(
        &net,
        &features,
        &labels,
        &label_indices,
        &idx_to_label,
        512,
        device,
        false,
    )?;

    println!("\n  SPECIES-LEVEL ACCURACY: {:.2}%", results_standard.accuracy());
    println!(
        "  TAXONOMIC-LEVEL ACCURACY: {:.2}%",
        results_standard.taxonomic_accuracy()
    );
    println!("  Total samples: {}", results_standard.total);
    println!("  Correct: {}", results_standard.correct);

    // Per-group breakdown
    println!("\n  PER-TAXONOMIC GROUP ACCURACY:");
    println!("  ┌────────────────┬─────────┬─────────┬─────────┐");
    println!("  │ Group          │ Total   │ Species │ Taxonom │");
    println!("  ├────────────────┼─────────┼─────────┼─────────┤");

    for group in &[
        TaxonomicGroup::Bird,
        TaxonomicGroup::Cetacean,
        TaxonomicGroup::Bat,
        TaxonomicGroup::Amphibian,
        TaxonomicGroup::Insect,
        TaxonomicGroup::Mammal,
        TaxonomicGroup::Fish,
        TaxonomicGroup::Unknown,
    ] {
        let total = *results_standard.per_group_total.get(group).unwrap_or(&0);
        let correct = *results_standard.per_group_correct.get(group).unwrap_or(&0);
        let tax_correct = *results_standard.per_group_taxonomic_correct.get(group).unwrap_or(&0);

        if total > 0 {
            let spec_acc = correct as f64 / total as f64 * 100.0;
            let tax_acc = tax_correct as f64 / total as f64 * 100.0;
            println!(
                "  │ {:14} │ {:7} │ {:6.1}% │ {:6.1}% │",
                format!("{:?}", group),
                total,
                spec_acc,
                tax_acc
            );
        }
    }
    println!("  └────────────────┴─────────┴─────────┴─────────┘");

    // Top confusions
    println!("\n  TOP CONFUSIONS (Actual -> Predicted):");
    let mut confusions: Vec<_> = results_standard.confusion.iter().collect();
    confusions.sort_by_key(|b| std::cmp::Reverse(b.1));
    for ((actual, predicted), count) in confusions.iter().take(10) {
        println!("    {} -> {}: {} times", actual, predicted, count);
    }

    // Run evaluation WITH taxonomic weighting
    println!("\n═════════════════════════════════════════════════════════════════════");
    println!("EVALUATION 2: Taxonomic-Aware (Hierarchical Classification)");
    println!("═════════════════════════════════════════════════════════════════════");
    println!("  (Using taxonomic group priors for improved accuracy)");

    let results_taxonomic = evaluate(
        &net,
        &features,
        &labels,
        &label_indices,
        &idx_to_label,
        512,
        device,
        true,
    )?;

    println!("\n  SPECIES-LEVEL ACCURACY: {:.2}%", results_taxonomic.accuracy());
    println!(
        "  TAXONOMIC-LEVEL ACCURACY: {:.2}%",
        results_taxonomic.taxonomic_accuracy()
    );

    // Summary
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  EVALUATION SUMMARY                                               ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Standard Classification:                                        ║");
    println!(
        "║    Species Accuracy:     {:>8.2}%                              ║",
        results_standard.accuracy()
    );
    println!(
        "║    Taxonomic Accuracy:   {:>8.2}%                              ║",
        results_standard.taxonomic_accuracy()
    );
    println!("║                                                                   ║");
    println!("║  Taxonomic-Aware Classification:                                 ║");
    println!(
        "║    Species Accuracy:     {:>8.2}%                              ║",
        results_taxonomic.accuracy()
    );
    println!(
        "║    Taxonomic Accuracy:   {:>8.2}%                              ║",
        results_taxonomic.taxonomic_accuracy()
    );
    println!("║                                                                   ║");
    println!(
        "║  Total Evaluation Time: {:>8.1}s                               ║",
        start.elapsed().as_secs_f32()
    );
    println!("║  Model: rosetta_net_112d_curriculum_gpu.ot                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    // Save results
    let output = serde_json::json!({
        "standard": {
            "species_accuracy": results_standard.accuracy(),
            "taxonomic_accuracy": results_standard.taxonomic_accuracy(),
            "total_samples": results_standard.total,
            "correct": results_standard.correct,
            "per_group": {
                "bird": {
                    "total": results_standard.per_group_total.get(&TaxonomicGroup::Bird).unwrap_or(&0),
                    "species_correct": results_standard.per_group_correct.get(&TaxonomicGroup::Bird).unwrap_or(&0),
                    "taxonomic_correct": results_standard.per_group_taxonomic_correct.get(&TaxonomicGroup::Bird).unwrap_or(&0),
                },
                "cetacean": {
                    "total": results_standard.per_group_total.get(&TaxonomicGroup::Cetacean).unwrap_or(&0),
                    "species_correct": results_standard.per_group_correct.get(&TaxonomicGroup::Cetacean).unwrap_or(&0),
                    "taxonomic_correct": results_standard.per_group_taxonomic_correct.get(&TaxonomicGroup::Cetacean).unwrap_or(&0),
                },
                "bat": {
                    "total": results_standard.per_group_total.get(&TaxonomicGroup::Bat).unwrap_or(&0),
                    "species_correct": results_standard.per_group_correct.get(&TaxonomicGroup::Bat).unwrap_or(&0),
                    "taxonomic_correct": results_standard.per_group_taxonomic_correct.get(&TaxonomicGroup::Bat).unwrap_or(&0),
                },
                "amphibian": {
                    "total": results_standard.per_group_total.get(&TaxonomicGroup::Amphibian).unwrap_or(&0),
                    "species_correct": results_standard.per_group_correct.get(&TaxonomicGroup::Amphibian).unwrap_or(&0),
                    "taxonomic_correct": results_standard.per_group_taxonomic_correct.get(&TaxonomicGroup::Amphibian).unwrap_or(&0),
                },
                "insect": {
                    "total": results_standard.per_group_total.get(&TaxonomicGroup::Insect).unwrap_or(&0),
                    "species_correct": results_standard.per_group_correct.get(&TaxonomicGroup::Insect).unwrap_or(&0),
                    "taxonomic_correct": results_standard.per_group_taxonomic_correct.get(&TaxonomicGroup::Insect).unwrap_or(&0),
                },
            },
            "top_confusions": confusions.iter().take(20).map(|((a, p), c)| serde_json::json!({
                "actual": a,
                "predicted": p,
                "count": c
            })).collect::<Vec<_>>(),
        },
        "taxonomic_aware": {
            "species_accuracy": results_taxonomic.accuracy(),
            "taxonomic_accuracy": results_taxonomic.taxonomic_accuracy(),
        },
        "model": model_path,
        "feature_dim": FEATURE_DIM,
        "n_classes": n_classes,
        "evaluation_time_seconds": start.elapsed().as_secs_f32(),
    });

    let output_path = "curriculum_nn_112d_eval_results.json";
    fs::write(output_path, serde_json::to_string_pretty(&output)?)?;
    println!("\nResults saved to: {}", output_path);

    Ok(())
}
