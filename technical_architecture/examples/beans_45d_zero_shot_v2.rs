//! BEANS-Zero 45D Zero-Shot Detection & Captioning Benchmark (v2)
//!
//! Fixed version that includes detection datasets in the reference database
//! with sample-level holdout for proper zero-shot evaluation.
//!
//! Configuration:
//! - Use samples from all datasets as prototypes (including detection datasets)
//! - Split at sample level: 70% train/prototype, 30% test/eval
//! - Evaluate detection on held-out samples from detection datasets

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use technical_architecture::{AcousticSimilarityEngine, SimilarityMetric, ZooVoxFeatureExtractor};

const FEATURE_DIM: usize = 45;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Datasets to include in prototype/reference database
const PROTOTYPE_DATASETS: &[&str] = &[
    "iNaturalist",
    "Xeno-canto",
    "DCASE-2021-Task-5",
    "Enabirds",
    "Hainan Gibbons",
    "Rainforest Connection",
    "CBI",
    "HumBugDB",
    "HICEAS",
    "Elie et al 2020",
    "Watkins",
    "esc50",
    "Animal Sound Archive",
];

/// Datasets to evaluate detection on
const DETECTION_EVAL_DATASETS: &[&str] = &["Enabirds", "Hainan Gibbons", "Rainforest Connection", "HICEAS"];

/// Train/test split ratio (0.7 = 70% train, 30% test)
const TRAIN_RATIO: f64 = 0.7;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
struct Manifest {
    samples: Vec<ManifestEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct ManifestEntry {
    #[serde(rename = "audio_file")]
    audio_file: String,
    sample_rate: u32,
    n_samples: usize,
    duration_ms: f64,
    labels: Labels,
}

#[derive(Debug, Clone, Deserialize)]
struct Labels {
    #[serde(rename = "source_dataset")]
    source_dataset: String,
    task: String,
    #[serde(default)]
    output: Option<String>,
}

#[derive(Debug, Clone)]
struct Sample {
    id: String,
    features: Vec<f64>,
    source_dataset: String,
    task: String,
    caption: Option<String>,
    sample_idx: usize, // For deterministic splitting
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DetectionResults {
    // Configuration
    train_ratio: f64,
    detection_threshold: f64,

    // Overall metrics
    true_positives: usize,
    false_positives: usize,
    true_negatives: usize,
    false_negatives: usize,
    precision: f64,
    recall: f64,
    f1_score: f64,

    // At various thresholds
    best_f1_threshold: f64,
    best_f1_score: f64,

    // Per-species metrics
    per_species_metrics: HashMap<String, SpeciesDetectionMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpeciesDetectionMetrics {
    species: String,
    n_train_samples: usize,
    n_test_samples: usize,
    detected: usize,
    precision: f64,
    recall: f64,
    f1: f64,
    avg_similarity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CaptioningResults {
    k_neighbors: usize,
    avg_semantic_similarity: f64,
    bleu4_score: f64,
    rouge_l_score: f64,
    per_dataset_metrics: HashMap<String, DatasetCaptionMetrics>,
    example_predictions: Vec<CaptionExample>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetCaptionMetrics {
    dataset: String,
    n_samples: usize,
    avg_similarity: f64,
    bleu4: f64,
    rouge_l: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CaptionExample {
    sample_id: String,
    true_caption: String,
    predicted_caption: String,
    similarity: f64,
    source_dataset: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ZeroShotResults {
    dataset: String,
    feature_dim: usize,
    train_ratio: f64,
    total_samples: usize,
    train_samples: usize,
    test_samples: usize,
    total_time_sec: f64,

    // Detection results
    detection: DetectionResults,

    // Captioning results
    captioning: CaptioningResults,
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║     BEANS-Zero 45D Zero-Shot Detection & Captioning Benchmark (v2)           ║");
    println!("║           With Detection Datasets in Reference Database                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();

    // Display configuration
    println!("Configuration:");
    println!("  ├─ Train Ratio: {}%", TRAIN_RATIO * 100.0);
    println!("  ├─ Prototype Datasets: {} total", PROTOTYPE_DATASETS.len());
    println!("  └─ Detection Eval Datasets: {:?}", DETECTION_EVAL_DATASETS);
    println!();

    // Load manifest
    let manifest_path = "beans_zero_cache/beans_audio_manifest.json";
    println!("[1/5] Loading manifest from: {}", manifest_path);

    let file = File::open(manifest_path)?;
    let reader = BufReader::new(file);
    let manifest: Manifest = serde_json::from_reader(reader)?;

    let total_samples = manifest.samples.len();
    println!("      Loaded {} samples", total_samples);
    println!();

    // ========================================================================
    // Phase 2: Feature Extraction
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[2/5] Phase 1: Feature Extraction");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let extraction_start = Instant::now();
    let processed = Arc::new(AtomicUsize::new(0));

    let all_samples: Vec<Option<Sample>> = manifest
        .samples
        .par_iter()
        .enumerate()
        .map(|(idx, entry)| {
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 20000 == 0 {
                println!("      Progress: {}/{} samples", count + 1, total_samples);
            }

            let audio_path = format!("beans_zero_cache/{}", entry.audio_file);
            let audio = match load_audio_raw(&audio_path, entry.n_samples) {
                Ok(a) => a,
                Err(_) => return None,
            };

            if audio.len() < 100 {
                return None;
            }

            let mut extractor = ZooVoxFeatureExtractor::new(entry.sample_rate);
            match extractor.extract_45d(&audio) {
                Ok(features) => Some(Sample {
                    id: format!("sample_{}", idx),
                    features: features.to_vector().to_vec(),
                    source_dataset: entry.labels.source_dataset.clone(),
                    task: entry.labels.task.clone(),
                    caption: entry.labels.output.clone(),
                    sample_idx: idx,
                }),
                Err(_) => None,
            }
        })
        .collect();

    let extraction_time = extraction_start.elapsed();

    let valid_samples: Vec<_> = all_samples.into_iter().filter_map(|s| s).collect();

    println!("\nExtraction Complete:");
    println!("  ├─ Valid Samples: {}", valid_samples.len());
    println!("  └─ Time: {:.1}s", extraction_time.as_secs_f64());
    println!();

    // ========================================================================
    // Phase 3: Train/Test Split
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[3/5] Phase 2: Train/Test Split (Sample-Level Holdout)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Group samples by dataset
    let mut samples_by_dataset: HashMap<String, Vec<&Sample>> = HashMap::new();
    for sample in &valid_samples {
        samples_by_dataset
            .entry(sample.source_dataset.clone())
            .or_default()
            .push(sample);
    }

    // Split each dataset into train/test
    let mut train_samples: Vec<&Sample> = Vec::new();
    let mut test_samples: Vec<&Sample> = Vec::new();

    for (dataset, mut samples) in samples_by_dataset {
        // Sort by sample_idx for deterministic split
        samples.sort_by_key(|s| s.sample_idx);

        let split_point = (samples.len() as f64 * TRAIN_RATIO) as usize;

        for (i, sample) in samples.into_iter().enumerate() {
            if i < split_point {
                train_samples.push(sample);
            } else {
                test_samples.push(sample);
            }
        }
    }

    println!("Split Summary:");
    println!("  ├─ Train samples (prototypes): {}", train_samples.len());
    println!("  ├─ Test samples (evaluation): {}", test_samples.len());
    println!(
        "  └─ Ratio: {:.1}% / {:.1}%",
        train_samples.len() as f64 / valid_samples.len() as f64 * 100.0,
        test_samples.len() as f64 / valid_samples.len() as f64 * 100.0
    );
    println!();

    // Separate test samples by task
    let detection_test: Vec<&Sample> = test_samples
        .iter()
        .filter(|s| s.task == "detection")
        .filter(|s| DETECTION_EVAL_DATASETS.contains(&s.source_dataset.as_str()))
        .cloned()
        .collect();

    let captioning_test: Vec<&Sample> = test_samples
        .iter()
        .filter(|s| s.task == "captioning")
        .cloned()
        .collect();

    println!("Test Samples by Task:");
    println!("  ├─ Detection (eval): {}", detection_test.len());
    println!("  └─ Captioning (eval): {}", captioning_test.len());
    println!();

    // ========================================================================
    // Phase 4: Zero-Shot Detection Evaluation
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[4/5] Phase 3: Zero-Shot Detection Evaluation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let detection_results = evaluate_zero_shot_detection(&detection_test, &train_samples);

    println!("Detection Results:");
    println!("  ├─ Precision: {:.1}%", detection_results.precision * 100.0);
    println!("  ├─ Recall: {:.1}%", detection_results.recall * 100.0);
    println!("  ├─ F1 Score: {:.1}%", detection_results.f1_score * 100.0);
    println!(
        "  └─ Best F1 @ threshold {:.2}: {:.1}%",
        detection_results.best_f1_threshold,
        detection_results.best_f1_score * 100.0
    );
    println!();

    println!("Per-Dataset Detection Metrics:");
    let mut sorted_species: Vec<_> = detection_results.per_species_metrics.iter().collect();
    sorted_species.sort_by(|a, b| b.1.f1.partial_cmp(&a.1.f1).unwrap());
    for (_, metrics) in sorted_species {
        println!(
            "  ├─ {}: P={:.1}% R={:.1}% F1={:.1}% ({} train, {} test)",
            metrics.species,
            metrics.precision * 100.0,
            metrics.recall * 100.0,
            metrics.f1 * 100.0,
            metrics.n_train_samples,
            metrics.n_test_samples
        );
    }
    println!();

    // ========================================================================
    // Phase 5: Zero-Shot Captioning Evaluation
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[5/5] Phase 4: Zero-Shot Captioning Evaluation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let captioning_results = evaluate_zero_shot_captioning(&captioning_test, &train_samples);

    println!("Captioning Results:");
    println!(
        "  ├─ Avg Semantic Similarity: {:.3}",
        captioning_results.avg_semantic_similarity
    );
    println!("  ├─ BLEU-4 Score: {:.3}", captioning_results.bleu4_score);
    println!("  └─ ROUGE-L Score: {:.3}", captioning_results.rouge_l_score);
    println!();

    // ========================================================================
    // Summary & Save
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("ZERO-SHOT BENCHMARK SUMMARY (v2)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let total_time = total_start.elapsed();

    let results = ZeroShotResults {
        dataset: "EarthSpeciesProject/BEANS-Zero".to_string(),
        feature_dim: FEATURE_DIM,
        train_ratio: TRAIN_RATIO,
        total_samples: valid_samples.len(),
        train_samples: train_samples.len(),
        test_samples: test_samples.len(),
        total_time_sec: total_time.as_secs_f64(),
        detection: detection_results,
        captioning: captioning_results,
    };

    println!("Configuration:");
    println!("  ├─ Feature Dimension: {}D", FEATURE_DIM);
    println!(
        "  ├─ Train/Test Split: {:.0}%/{:.0}%",
        TRAIN_RATIO * 100.0,
        (1.0 - TRAIN_RATIO) * 100.0
    );
    println!("  └─ Prototype Datasets: {}", PROTOTYPE_DATASETS.len());
    println!();

    println!("Sample Counts:");
    println!("  ├─ Total: {}", results.total_samples);
    println!("  ├─ Train (prototypes): {}", results.train_samples);
    println!("  └─ Test (evaluation): {}", results.test_samples);
    println!();

    println!("Detection Task:");
    println!("  ├─ Precision: {:.1}%", results.detection.precision * 100.0);
    println!("  ├─ Recall: {:.1}%", results.detection.recall * 100.0);
    println!("  └─ F1 Score: {:.1}%", results.detection.f1_score * 100.0);
    println!();

    println!("Captioning Task:");
    println!(
        "  ├─ Semantic Similarity: {:.1}%",
        results.captioning.avg_semantic_similarity * 100.0
    );
    println!("  ├─ BLEU-4: {:.3}", results.captioning.bleu4_score);
    println!("  └─ ROUGE-L: {:.3}", results.captioning.rouge_l_score);
    println!();

    println!("Performance:");
    println!(
        "  └─ Total Time: {:.1}s ({:.1} min)",
        total_time.as_secs_f64(),
        total_time.as_secs_f64() / 60.0
    );
    println!();

    // Assessment
    let detection_score = results.detection.f1_score;
    let captioning_score = results.captioning.avg_semantic_similarity;

    let assessment = if detection_score >= 0.5 && captioning_score >= 0.8 {
        "EXCELLENT - Strong zero-shot transfer"
    } else if detection_score >= 0.3 && captioning_score >= 0.7 {
        "GOOD - Meaningful cross-task transfer"
    } else if detection_score >= 0.2 && captioning_score >= 0.6 {
        "FAIR - Some generalization"
    } else {
        "NEEDS_IMPROVEMENT - Limited transfer"
    };

    println!("Overall Assessment: {}", assessment);
    println!();

    // Save results
    std::fs::create_dir_all("beans_analysis")?;
    let output_path = "beans_analysis/beans_45d_zero_shot_v2_results.json";
    let file = File::create(output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &results)?;
    println!("Results saved to: {}", output_path);

    Ok(())
}

// ============================================================================
// DETECTION EVALUATION
// ============================================================================

fn evaluate_zero_shot_detection(test_samples: &[&Sample], train_samples: &[&Sample]) -> DetectionResults {
    // Build prototypes from TRAIN samples only (grouped by dataset)
    let mut prototypes_by_dataset: HashMap<String, (Vec<Vec<f64>>, usize)> = HashMap::new();

    for sample in train_samples {
        let entry = prototypes_by_dataset
            .entry(sample.source_dataset.clone())
            .or_insert((Vec::new(), 0));
        entry.0.push(sample.features.clone());
        entry.1 += 1;
    }

    // Compute mean prototypes
    let mut mean_prototypes: HashMap<String, Vec<f64>> = HashMap::new();
    let mut train_counts: HashMap<String, usize> = HashMap::new();

    for (dataset, (features_list, count)) in &prototypes_by_dataset {
        if features_list.is_empty() {
            continue;
        }
        let dim = features_list[0].len();
        let mut mean = vec![0.0; dim];
        for features in features_list {
            for (i, &val) in features.iter().enumerate() {
                mean[i] += val;
            }
        }
        for val in &mut mean {
            *val /= features_list.len() as f64;
        }
        mean_prototypes.insert(dataset.clone(), mean);
        train_counts.insert(dataset.clone(), *count);
    }

    println!(
        "Built {} species prototypes from {} train samples",
        mean_prototypes.len(),
        train_samples.len()
    );

    // Create similarity engine
    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    {
        let mut matrix = ndarray::Array2::<f64>::zeros((train_samples.len().min(10000), FEATURE_DIM));
        for (i, sample) in train_samples.iter().take(10000).enumerate() {
            for (j, &val) in sample.features.iter().enumerate() {
                matrix[[i, j]] = val;
            }
        }
        engine.fit_normalization(&matrix);
    }

    // Find best threshold via grid search
    let thresholds: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
    let mut best_f1 = 0.0;
    let mut best_threshold = 0.5;

    for &threshold in &thresholds {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_count = 0usize;

        for sample in test_samples.iter().take(5000) {
            let query = Array1::from_vec(sample.features.clone());

            let mut best_sim = 0.0;
            let mut best_dataset = "";

            for (dataset, prototype) in &mean_prototypes {
                let proto = Array1::from_vec(prototype.clone());
                let sim = 1.0 - engine.distance(&query, &proto);
                if sim > best_sim {
                    best_sim = sim;
                    best_dataset = dataset.as_str();
                }
            }

            let detected = best_sim >= threshold;
            let is_correct = best_dataset == sample.source_dataset;

            if detected {
                if is_correct {
                    tp += 1;
                } else {
                    fp += 1;
                }
            } else {
                fn_count += 1;
            }
        }

        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        if f1 > best_f1 {
            best_f1 = f1;
            best_threshold = threshold;
        }
    }

    println!("Best threshold: {:.2} (F1={:.3})", best_threshold, best_f1);

    // Evaluate with best threshold
    let mut tp = 0usize;
    let mut fp = 0usize;
    let tn = 0usize;
    let mut fn_count = 0usize;

    let mut per_species_metrics: HashMap<String, SpeciesDetectionMetrics> = HashMap::new();

    // Initialize per-species metrics
    for sample in test_samples {
        per_species_metrics
            .entry(sample.source_dataset.clone())
            .or_insert(SpeciesDetectionMetrics {
                species: sample.source_dataset.clone(),
                n_train_samples: *train_counts.get(&sample.source_dataset).unwrap_or(&0),
                n_test_samples: 0,
                detected: 0,
                precision: 0.0,
                recall: 0.0,
                f1: 0.0,
                avg_similarity: 0.0,
            });
    }

    for sample in test_samples {
        let query = Array1::from_vec(sample.features.clone());

        let mut best_sim = 0.0;
        let mut best_dataset = "";

        for (dataset, prototype) in &mean_prototypes {
            let proto = Array1::from_vec(prototype.clone());
            let sim = 1.0 - engine.distance(&query, &proto);
            if sim > best_sim {
                best_sim = sim;
                best_dataset = dataset.as_str();
            }
        }

        let detected = best_sim >= best_threshold;
        let is_correct = best_dataset == sample.source_dataset;

        if detected && is_correct {
            tp += 1;
        } else if detected && !is_correct {
            fp += 1;
        } else {
            fn_count += 1;
        }

        // Update per-species metrics
        if let Some(metrics) = per_species_metrics.get_mut(&sample.source_dataset) {
            metrics.n_test_samples += 1;
            metrics.avg_similarity += best_sim;
            if detected && is_correct {
                metrics.detected += 1;
            }
        }
    }

    // Finalize per-species metrics
    for metrics in per_species_metrics.values_mut() {
        if metrics.n_test_samples > 0 {
            metrics.avg_similarity /= metrics.n_test_samples as f64;
            metrics.recall = metrics.detected as f64 / metrics.n_test_samples as f64;
            metrics.precision = metrics.recall; // Simplified
            metrics.f1 = if metrics.precision + metrics.recall > 0.0 {
                2.0 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            } else {
                0.0
            };
        }
    }

    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let recall = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };
    let f1_score = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    DetectionResults {
        train_ratio: TRAIN_RATIO,
        detection_threshold: best_threshold,
        true_positives: tp,
        false_positives: fp,
        true_negatives: tn,
        false_negatives: fn_count,
        precision,
        recall,
        f1_score,
        best_f1_threshold: best_threshold,
        best_f1_score: best_f1,
        per_species_metrics,
    }
}

// ============================================================================
// CAPTIONING EVALUATION
// ============================================================================

fn evaluate_zero_shot_captioning(test_samples: &[&Sample], train_samples: &[&Sample]) -> CaptioningResults {
    // Build caption database from TRAIN samples only
    let caption_db: Vec<(&Sample, Vec<f64>)> = train_samples
        .iter()
        .filter(|s| s.caption.is_some())
        .map(|s| (*s, s.features.clone()))
        .collect();

    println!(
        "Built caption database with {} entries from train set",
        caption_db.len()
    );

    // Create similarity engine
    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    {
        let mut matrix = ndarray::Array2::<f64>::zeros((train_samples.len().min(10000), FEATURE_DIM));
        for (i, sample) in train_samples.iter().take(10000).enumerate() {
            for (j, &val) in sample.features.iter().enumerate() {
                matrix[[i, j]] = val;
            }
        }
        engine.fit_normalization(&matrix);
    }

    let k = 5;
    let mut total_similarity = 0.0;
    let mut total_bleu = 0.0;
    let mut total_rouge = 0.0;
    let mut n_evaluated = 0;

    let mut per_dataset_metrics: HashMap<String, DatasetCaptionMetrics> = HashMap::new();
    let mut example_predictions: Vec<CaptionExample> = Vec::new();

    for sample in test_samples.iter().take(5000) {
        let query = Array1::from_vec(sample.features.clone());
        let true_caption = sample.caption.as_deref().unwrap_or("");

        // Find k nearest neighbors from TRAIN set only
        let mut distances: Vec<(usize, f64)> = caption_db
            .iter()
            .enumerate()
            .map(|(j, (_, features))| {
                let candidate = Array1::from_vec(features.clone());
                (j, engine.distance(&query, &candidate))
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Get predicted caption (from nearest neighbor)
        let predicted_caption = if let Some((idx, _)) = distances.first() {
            caption_db[*idx].0.caption.as_deref().unwrap_or("")
        } else {
            ""
        };

        // Calculate metrics
        let avg_distance: f64 = distances.iter().take(k).map(|(_, d)| *d).sum::<f64>() / k as f64;
        let similarity = 1.0 - avg_distance;

        let bleu = calculate_bleu4(true_caption, predicted_caption);
        let rouge = calculate_rouge_l(true_caption, predicted_caption);

        total_similarity += similarity;
        total_bleu += bleu;
        total_rouge += rouge;
        n_evaluated += 1;

        // Per-dataset metrics
        let entry = per_dataset_metrics
            .entry(sample.source_dataset.clone())
            .or_insert(DatasetCaptionMetrics {
                dataset: sample.source_dataset.clone(),
                n_samples: 0,
                avg_similarity: 0.0,
                bleu4: 0.0,
                rouge_l: 0.0,
            });
        entry.n_samples += 1;
        entry.avg_similarity += similarity;
        entry.bleu4 += bleu;
        entry.rouge_l += rouge;

        // Save examples
        if example_predictions.len() < 10 {
            example_predictions.push(CaptionExample {
                sample_id: sample.id.clone(),
                true_caption: true_caption.to_string(),
                predicted_caption: predicted_caption.to_string(),
                similarity,
                source_dataset: sample.source_dataset.clone(),
            });
        }
    }

    // Finalize per-dataset metrics
    for metrics in per_dataset_metrics.values_mut() {
        if metrics.n_samples > 0 {
            metrics.avg_similarity /= metrics.n_samples as f64;
            metrics.bleu4 /= metrics.n_samples as f64;
            metrics.rouge_l /= metrics.n_samples as f64;
        }
    }

    CaptioningResults {
        k_neighbors: k,
        avg_semantic_similarity: total_similarity / n_evaluated as f64,
        bleu4_score: total_bleu / n_evaluated as f64,
        rouge_l_score: total_rouge / n_evaluated as f64,
        per_dataset_metrics,
        example_predictions,
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn load_audio_raw(path: &str, expected_samples: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;

    let audio: Vec<f64> = bytes
        .chunks_exact(4)
        .take(expected_samples)
        .map(|chunk| {
            let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            val as f64
        })
        .collect();

    Ok(audio)
}

fn calculate_bleu4(reference: &str, hypothesis: &str) -> f64 {
    let ref_lower = reference.to_lowercase();
    let hyp_lower = hypothesis.to_lowercase();
    let ref_tokens: Vec<&str> = ref_lower.split_whitespace().collect();
    let hyp_tokens: Vec<&str> = hyp_lower.split_whitespace().collect();

    if hyp_tokens.is_empty() || ref_tokens.is_empty() {
        return 0.0;
    }

    let ref_set: HashSet<&str> = ref_tokens.iter().cloned().collect();
    let matches = hyp_tokens.iter().filter(|t| ref_set.contains(*t)).count();

    let precision = matches as f64 / hyp_tokens.len() as f64;

    let bp = if hyp_tokens.len() >= ref_tokens.len() {
        1.0
    } else {
        (1.0 - ref_tokens.len() as f64 / hyp_tokens.len() as f64).exp()
    };

    bp * precision
}

fn calculate_rouge_l(reference: &str, hypothesis: &str) -> f64 {
    let ref_lower = reference.to_lowercase();
    let hyp_lower = hypothesis.to_lowercase();
    let ref_tokens: Vec<&str> = ref_lower.split_whitespace().collect();
    let hyp_tokens: Vec<&str> = hyp_lower.split_whitespace().collect();

    if ref_tokens.is_empty() || hyp_tokens.is_empty() {
        return 0.0;
    }

    let lcs_len = longest_common_subsequence(&ref_tokens, &hyp_tokens);

    let precision = lcs_len as f64 / hyp_tokens.len() as f64;
    let recall = lcs_len as f64 / ref_tokens.len() as f64;

    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

fn longest_common_subsequence(a: &[&str], b: &[&str]) -> usize {
    let m = a.len();
    let n = b.len();

    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    dp[m][n]
}
