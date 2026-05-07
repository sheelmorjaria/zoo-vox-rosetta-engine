//! Evaluate RF Feature Stacking Ensemble on BEANS-Zero Benchmark
//! ===============================================================
//!
//! This binary evaluates the FeatureStackingEnsemble (Physics 46D + Full 112D)
//! on the BEANS-Zero bioacoustic benchmark.
//!
//! Usage:
//!   cargo run --release --bin eval_rf_stacking_ensemble -- <manifest.json> [options]
//!
//! Options:
//!   --physics-model <path>  Path to physics RF model JSON (default: physics_rf_model.json)
//!   --full-model <path>     Path to full RF model JSON (default: full_rf_model.json)
//!   --limit <n>             Limit to first n samples
//!   --detection-threshold   Enable detection mode with confidence threshold
//!
//! The evaluation compares:
//!   1. Physics RF alone (46D)
//!   2. Full RF alone (112D)
//!   3. Feature Stacking Ensemble (46D + 112D with confidence weighting)

use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use technical_architecture::{FeatureStackingEnsemble, RFModel, StackingConfig, FULL_DIM, PHYSICS_DIM};

// ============================================================================
// BEANS-Zero Manifest Structures
// ============================================================================

#[derive(Debug, Deserialize)]
struct BeansManifest {
    dataset: String,
    split: String,
    n_samples: usize,
    samples: Vec<BeansSample>,
}

#[derive(Debug, Deserialize)]
struct BeansSample {
    audio_file: String,
    #[allow(dead_code)] // Field exists for JSON deserialization compatibility
    n_samples: u32,
    labels: BeansLabels,
}

#[derive(Debug, Deserialize)]
struct BeansLabels {
    output: String,
    #[allow(dead_code)] // Field exists for JSON deserialization compatibility
    task: String,
}

// ============================================================================
// Feature Cache Structures
// ============================================================================

#[derive(Debug, Deserialize)]
struct CacheManifest {
    entries: HashMap<String, String>,
}

/// Load features from bincode format (Vec<f32>)
fn load_bincode_features(filepath: &Path) -> Result<Vec<f32>> {
    use std::io::Read;

    let mut file = std::fs::File::open(filepath)?;

    // Read varint length
    let mut length: usize = 0;
    let mut shift: u32 = 0;
    loop {
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte)?;
        length |= ((byte[0] & 0x7F) as usize) << shift;
        shift += 7;
        if byte[0] & 0x80 == 0 {
            break;
        }
    }

    // Read features (f32 = 4 bytes each)
    let mut buffer = vec![0u8; length * 4];
    file.read_exact(&mut buffer)?;

    let features: Vec<f32> = buffer
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap_or([0, 0, 0, 0]);
            f32::from_le_bytes(bytes)
        })
        .collect();

    Ok(features)
}

// ============================================================================
// Taxonomic Mapping (simplified for evaluation)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum TaxonomicGroup {
    Cetacean,
    Mysticete,
    Songbird,
    NonPasserine,
    Amphibian,
    Insect,
    Mammal,
    Pinniped,
    Unknown,
}

fn map_species_to_taxon(species: &str) -> TaxonomicGroup {
    let s = species.to_lowercase();

    if s.contains("dolphin")
        || s.contains("porpoise")
        || s.contains("sperm")
        || s.contains("beaked")
        || s.contains("delphinid")
        || s.contains("phocoen")
        || s.contains("orca")
    {
        return TaxonomicGroup::Cetacean;
    }
    if s.contains("humpback")
        || s.contains("blue whale")
        || s.contains("fin whale")
        || s.contains("minke")
        || s.contains("gray whale")
        || s.contains("right whale")
        || s.contains("bowhead")
        || s.contains("balaenopter")
    {
        return TaxonomicGroup::Mysticete;
    }
    if s.contains("sparrow")
        || s.contains("finch")
        || s.contains("warbler")
        || s.contains("thrush")
        || s.contains("robin")
        || s.contains("cardinal")
        || s.contains("towhee")
        || s.contains("ovenbird")
        || s.contains("wren")
        || s.contains("tit")
        || s.contains("swainson")
    {
        return TaxonomicGroup::Songbird;
    }
    if s.contains("parrot")
        || s.contains("owl")
        || s.contains("hawk")
        || s.contains("eagle")
        || s.contains("duck")
        || s.contains("goose")
        || s.contains("gull")
        || s.contains("crow")
        || s.contains("raven")
        || s.contains("penguin")
        || s.contains("psittacid")
        || s.contains("strigid")
    {
        return TaxonomicGroup::NonPasserine;
    }
    if s.contains("frog")
        || s.contains("toad")
        || s.contains("ranid")
        || s.contains("bufonid")
        || s.contains("hylid")
        || s.contains("peeper")
    {
        return TaxonomicGroup::Amphibian;
    }
    if s.contains("cricket")
        || s.contains("mosquito")
        || s.contains("cicada")
        || s.contains("grasshopper")
        || s.contains("katydid")
        || s.contains("bee")
        || s.contains("fly")
        || s.contains("anopheles")
        || s.contains("aedes")
        || s.contains("culex")
        || s.contains("culicid")
    {
        return TaxonomicGroup::Insect;
    }
    if s.contains("bat")
        || s.contains("pteropodid")
        || s.contains("vesper")
        || s.contains("phyllostomid")
        || s.contains("monkey")
        || s.contains("ape")
        || s.contains("gibbon")
        || s.contains("chimp")
        || s.contains("gorilla")
        || s.contains("primate")
    {
        return TaxonomicGroup::Mammal;
    }
    if s.contains("seal")
        || s.contains("sea lion")
        || s.contains("walrus")
        || s.contains("phocid")
        || s.contains("otariid")
    {
        return TaxonomicGroup::Pinniped;
    }

    TaxonomicGroup::Unknown
}

// ============================================================================
// Evaluation Results
// ============================================================================

#[derive(Debug, Default, Serialize, Deserialize)]
struct EvaluationResults {
    // Model accuracies
    physics_accuracy: f64,
    full_accuracy: f64,
    ensemble_accuracy: f64,

    // Taxonomic-level accuracies
    physics_taxonomic_accuracy: f64,
    full_taxonomic_accuracy: f64,
    ensemble_taxonomic_accuracy: f64,

    // Ensemble statistics
    physics_used_count: usize,
    agreement_count: usize,
    total_samples: usize,

    // Detection mode (if enabled)
    detection_precision: f64,
    detection_recall: f64,
    detection_f1: f64,

    // Per-dataset breakdown
    dataset_breakdown: HashMap<String, DatasetMetrics>,

    // Timing
    processing_time_seconds: f64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct DatasetMetrics {
    physics_accuracy: f64,
    full_accuracy: f64,
    ensemble_accuracy: f64,
    total: usize,
}

// ============================================================================
// Main Evaluation Function
// ============================================================================

fn run_evaluation(
    manifest_path: &Path,
    physics_model_path: &Path,
    full_model_path: &Path,
    limit: Option<usize>,
    detection_threshold: Option<f32>,
) -> Result<EvaluationResults> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║     RF Feature Stacking Ensemble Evaluation                       ║");
    println!("║     Physics 46D + Full 112D with Confidence Weighting             ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let start_time = Instant::now();
    let base_path = manifest_path.parent().unwrap_or(Path::new("."));

    // Load manifest
    println!("Loading BEANS-Zero manifest from: {:?}", manifest_path);
    let manifest_content = std::fs::read_to_string(manifest_path)?;
    let manifest: BeansManifest = serde_json::from_str(&manifest_content)?;

    println!("  Dataset: {}", manifest.dataset);
    println!("  Split: {}", manifest.split);
    println!("  Total samples: {}", manifest.n_samples);

    // Load feature cache manifest
    let cache_manifest_path = base_path.join("beans_feature_cache_112d/cache_manifest.json");
    println!("\nLoading feature cache manifest from: {:?}", cache_manifest_path);
    let cache_content = std::fs::read_to_string(&cache_manifest_path)?;
    let cache_manifest: CacheManifest = serde_json::from_str(&cache_content)?;
    println!("  Cached features: {}", cache_manifest.entries.len());

    // Load models
    println!("\nLoading models...");

    println!("  Physics RF (46D): {:?}", physics_model_path);
    let physics_model: RFModel = if physics_model_path.exists() {
        let json_content = std::fs::read_to_string(physics_model_path)?;
        serde_json::from_str(&json_content).context("Failed to parse physics RF model JSON")?
    } else {
        anyhow::bail!("Physics RF model not found at {:?}", physics_model_path);
    };
    println!(
        "    Loaded: {} trees, {} classes, {:.1}% val accuracy",
        physics_model.n_estimators, physics_model.n_classes, physics_model.val_accuracy
    );

    println!("  Full RF (112D): {:?}", full_model_path);
    let full_model: RFModel = if full_model_path.exists() {
        let json_content = std::fs::read_to_string(full_model_path)?;
        serde_json::from_str(&json_content).context("Failed to parse full RF model JSON")?
    } else {
        anyhow::bail!("Full RF model not found at {:?}", full_model_path);
    };
    println!(
        "    Loaded: {} trees, {} classes, {:.1}% val accuracy",
        full_model.n_estimators, full_model.n_classes, full_model.val_accuracy
    );

    // Create ensemble
    let config = StackingConfig {
        physics_preference: 1.5,
        physics_confidence_threshold: 0.6,
        detection_threshold,
        ..Default::default()
    };
    let mut ensemble = FeatureStackingEnsemble::with_config(config);
    ensemble.load_physics_model(physics_model);
    ensemble.load_full_model(full_model);

    println!("  Ensemble config: physics_preference=1.5, confidence_threshold=0.6");

    // Determine samples to process
    let samples_to_process: Vec<_> = if let Some(n) = limit {
        manifest.samples.into_iter().take(n).collect()
    } else {
        manifest.samples
    };
    println!("\nProcessing {} samples...", samples_to_process.len());

    // Process samples in parallel
    println!("\nPhase 1: Loading features from cache...");
    let cache_dir = base_path.join("beans_feature_cache_112d");

    let processed: Vec<_> = samples_to_process
        .par_iter()
        .filter_map(|sample| {
            let cache_file = cache_manifest.entries.get(&sample.audio_file)?;

            let full_path = cache_dir.join(cache_file);
            if !full_path.exists() {
                return None;
            }

            // Load features
            let features = load_bincode_features(&full_path).ok()?;
            if features.len() != FULL_DIM {
                return None;
            }

            // Get label
            let label = sample.labels.output.clone();
            // source_dataset not available in this schema, use placeholder
            let source_dataset = "unknown".to_string();

            Some((features, label, source_dataset))
        })
        .collect();

    println!("  Successfully loaded: {} samples", processed.len());

    // Run evaluation
    println!("\nPhase 2: Running evaluation...");

    let mut results = EvaluationResults::default();
    let mut dataset_stats: HashMap<String, DatasetMetrics> = HashMap::new();

    // Track metrics
    let mut physics_correct = 0usize;
    let mut full_correct = 0usize;
    let mut ensemble_correct = 0usize;
    let mut physics_taxonomic_correct = 0usize;
    let mut full_taxonomic_correct = 0usize;
    let mut ensemble_taxonomic_correct = 0usize;
    let mut physics_used_count = 0usize;
    let mut agreement_count = 0usize;
    let mut total = 0usize;

    // Detection mode metrics
    let mut true_positives = 0usize;
    let mut false_positives = 0usize;
    let mut _true_negatives = 0usize;
    let mut false_negatives = 0usize;

    for (features, true_label, source_dataset) in &processed {
        total += 1;

        // Get ensemble prediction
        let ensemble_result = ensemble.predict(features);

        // Physics-only prediction
        let physics_features: Vec<f32> = features.iter().take(PHYSICS_DIM).copied().collect();
        let physics_pred_idx = ensemble.physics_model.predict(&physics_features);
        let physics_pred = ensemble
            .physics_model
            .class_labels
            .get(physics_pred_idx)
            .map(|s| s.as_str())
            .unwrap_or("<unknown>");

        // Full-only prediction
        let full_pred_idx = ensemble.full_model.predict(features);
        let full_pred = ensemble
            .full_model
            .class_labels
            .get(full_pred_idx)
            .map(|s| s.as_str())
            .unwrap_or("<unknown>");

        // Count correct predictions
        if physics_pred == true_label {
            physics_correct += 1;
        }
        if full_pred == true_label {
            full_correct += 1;
        }
        if ensemble_result.prediction == *true_label {
            ensemble_correct += 1;
        }

        // Taxonomic accuracy
        let true_taxon = map_species_to_taxon(true_label);
        let physics_taxon = map_species_to_taxon(physics_pred);
        let full_taxon = map_species_to_taxon(full_pred);
        let ensemble_taxon = map_species_to_taxon(&ensemble_result.prediction);

        if physics_taxon == true_taxon {
            physics_taxonomic_correct += 1;
        }
        if full_taxon == true_taxon {
            full_taxonomic_correct += 1;
        }
        if ensemble_taxon == true_taxon {
            ensemble_taxonomic_correct += 1;
        }

        // Ensemble statistics
        if ensemble_result.used_physics {
            physics_used_count += 1;
        }
        if ensemble_result.agreement {
            agreement_count += 1;
        }

        // Detection mode metrics
        if let Some(threshold) = detection_threshold {
            let is_detected = ensemble_result.confidence >= threshold;
            let is_positive = true_label != "background" && true_label != "noise";

            if is_detected && is_positive {
                true_positives += 1;
            } else if is_detected && !is_positive {
                false_positives += 1;
            } else if !is_detected && !is_positive {
                _true_negatives += 1;
            } else {
                false_negatives += 1;
            }
        }

        // Update dataset breakdown
        let stats = dataset_stats.entry(source_dataset.clone()).or_default();
        stats.total += 1;
        if physics_pred == *true_label {
            stats.physics_accuracy += 1.0;
        }
        if full_pred == *true_label {
            stats.full_accuracy += 1.0;
        }
        if ensemble_result.prediction == *true_label {
            stats.ensemble_accuracy += 1.0;
        }
    }

    // Calculate final metrics
    if total > 0 {
        results.physics_accuracy = physics_correct as f64 / total as f64;
        results.full_accuracy = full_correct as f64 / total as f64;
        results.ensemble_accuracy = ensemble_correct as f64 / total as f64;
        results.physics_taxonomic_accuracy = physics_taxonomic_correct as f64 / total as f64;
        results.full_taxonomic_accuracy = full_taxonomic_correct as f64 / total as f64;
        results.ensemble_taxonomic_accuracy = ensemble_taxonomic_correct as f64 / total as f64;
    }

    results.physics_used_count = physics_used_count;
    results.agreement_count = agreement_count;
    results.total_samples = total;

    // Detection metrics
    if detection_threshold.is_some() {
        let tp = true_positives as f64;
        let fp = false_positives as f64;
        let fn_ = false_negatives as f64;

        results.detection_precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        results.detection_recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        results.detection_f1 = if results.detection_precision + results.detection_recall > 0.0 {
            2.0 * results.detection_precision * results.detection_recall
                / (results.detection_precision + results.detection_recall)
        } else {
            0.0
        };
    }

    // Finalize dataset breakdown
    for (_, stats) in dataset_stats.iter_mut() {
        if stats.total > 0 {
            stats.physics_accuracy /= stats.total as f64;
            stats.full_accuracy /= stats.total as f64;
            stats.ensemble_accuracy /= stats.total as f64;
        }
    }
    results.dataset_breakdown = dataset_stats;

    results.processing_time_seconds = start_time.elapsed().as_secs_f64();

    Ok(results)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <manifest.json> [options]", args[0]);
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --physics-model <path>  Path to physics RF model JSON");
        eprintln!("  --full-model <path>     Path to full RF model JSON");
        eprintln!("  --limit <n>             Limit to first n samples");
        eprintln!("  --detection-threshold   Enable detection mode with threshold");
        std::process::exit(1);
    }

    let manifest_path = PathBuf::from(&args[1]);
    let base_path = manifest_path.parent().unwrap_or(Path::new(".")).to_path_buf();

    // Parse options
    let mut physics_model_path = base_path.join("physics_rf_model.json");
    let mut full_model_path = base_path.join("full_rf_model.json");
    let mut limit = None;
    let mut detection_threshold = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--physics-model" => {
                i += 1;
                physics_model_path = PathBuf::from(args.get(i).context("--physics-model requires path")?);
            }
            "--full-model" => {
                i += 1;
                full_model_path = PathBuf::from(args.get(i).context("--full-model requires path")?);
            }
            "--limit" => {
                i += 1;
                limit = Some(
                    args.get(i)
                        .context("--limit requires number")?
                        .parse::<usize>()
                        .context("Invalid limit number")?,
                );
            }
            "--detection-threshold" => {
                i += 1;
                detection_threshold = Some(
                    args.get(i)
                        .context("--detection-threshold requires value")?
                        .parse::<f32>()
                        .context("Invalid threshold value")?,
                );
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let results = run_evaluation(
        &manifest_path,
        &physics_model_path,
        &full_model_path,
        limit,
        detection_threshold,
    )?;

    // Print results
    println!("\n{}", "=".repeat(70));
    println!("RF Feature Stacking Ensemble Evaluation Results");
    println!("{}", "=".repeat(70));

    println!("\n--- Species-Level Accuracy ---");
    println!(
        "Physics RF (46D):              {:>7.2}%",
        results.physics_accuracy * 100.0
    );
    println!("Full RF (112D):                {:>7.2}%", results.full_accuracy * 100.0);
    println!(
        "Ensemble (Stacked):            {:>7.2}% ({:+.2}%)",
        results.ensemble_accuracy * 100.0,
        (results.ensemble_accuracy - results.full_accuracy.max(results.physics_accuracy)) * 100.0
    );

    println!("\n--- Taxonomic-Level Accuracy ---");
    println!(
        "Physics RF (46D):              {:>7.2}%",
        results.physics_taxonomic_accuracy * 100.0
    );
    println!(
        "Full RF (112D):                {:>7.2}%",
        results.full_taxonomic_accuracy * 100.0
    );
    println!(
        "Ensemble (Stacked):            {:>7.2}% ({:+.2}%)",
        results.ensemble_taxonomic_accuracy * 100.0,
        (results.ensemble_taxonomic_accuracy - results.full_taxonomic_accuracy.max(results.physics_taxonomic_accuracy))
            * 100.0
    );

    println!("\n--- Ensemble Statistics ---");
    println!("Total samples:                 {}", results.total_samples);
    println!(
        "Physics used as primary:       {} ({:.1}%)",
        results.physics_used_count,
        results.physics_used_count as f64 / results.total_samples as f64 * 100.0
    );
    println!(
        "Model agreement:               {} ({:.1}%)",
        results.agreement_count,
        results.agreement_count as f64 / results.total_samples as f64 * 100.0
    );

    if detection_threshold.is_some() {
        println!("\n--- Detection Mode ---");
        println!(
            "Precision:                     {:>7.2}%",
            results.detection_precision * 100.0
        );
        println!(
            "Recall:                        {:>7.2}%",
            results.detection_recall * 100.0
        );
        println!("F1 Score:                      {:>7.2}%", results.detection_f1 * 100.0);
    }

    println!("\n--- Per-Dataset Breakdown ---");
    let mut datasets: Vec<_> = results.dataset_breakdown.iter().collect();
    datasets.sort_by_key(|b| std::cmp::Reverse(b.1.total));
    for (dataset, metrics) in datasets.iter().take(10) {
        println!(
            "{:<30} n={:>5}  Physics={:>5.1}%  Full={:>5.1}%  Ensemble={:>5.1}%",
            dataset,
            metrics.total,
            metrics.physics_accuracy * 100.0,
            metrics.full_accuracy * 100.0,
            metrics.ensemble_accuracy * 100.0
        );
    }

    println!("\n--- Performance ---");
    println!("Total processing time:         {:.2}s", results.processing_time_seconds);
    println!(
        "Samples/second:                {:.1}",
        results.total_samples as f64 / results.processing_time_seconds
    );

    // Save JSON output
    let output_path = "rf_stacking_ensemble_results.json";
    let output = serde_json::to_string_pretty(&results)?;
    std::fs::write(output_path, output)?;
    println!("\nDetailed results saved to: {}", output_path);

    Ok(())
}
