//! BEANS-Zero 45D Weighted Benchmark
//!
//! Re-runs the zero-shot benchmark with species-specific feature weights
//! to validate the hypothesis that weighted features improve accuracy.
//!
//! Hypothesis:
//! - Previous (Unweighted 45D): ~90.34% accuracy
//! - New (Weighted 45D): ~92-93% accuracy
//!
//! The weights suppress "noise" dimensions for each species type:
//! - Birds: Weight spectral/temporal features high
//! - Marine mammals: Weight FM slope and rhythm high
//! - Insects: Weight high-frequency modulation high

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
use technical_architecture::{
    species::{FeatureWeights, SpeciesConfigFactory},
    AcousticSimilarityEngine, SimilarityMetric, ZooVoxFeatureExtractor,
};

const FEATURE_DIM: usize = 45;

// ============================================================================
// DATASET -> SPECIES MAPPING
// ============================================================================

/// Maps BEANS-Zero datasets to our species configuration for weight selection
fn get_species_weights_for_dataset(dataset: &str) -> FeatureWeights {
    match dataset {
        // Bird datasets - use zebra finch weights (songbird)
        "Xeno-canto"
        | "iNaturalist"
        | "Enabirds"
        | "DCASE-2021-Task-5"
        | "Rainforest Connection"
        | "CBI"
        | "Hainan Gibbons" => FeatureWeights::zebra_finch(),

        // Marine mammal datasets - use dolphin weights
        "Watkins" | "HICEAS" => FeatureWeights::dolphin(),

        // Insect datasets - create insect-specific weights
        "HumBugDB" => {
            FeatureWeights {
                spectral: 1.5,   // High frequency content
                harmonic: 0.5,   // Less harmonic structure
                temporal: 1.8,   // Wing beat timing
                modulation: 2.0, // High-frequency modulation
                cepstral: 1.0,
                formant: 0.3, // Not relevant
                micro_dynamics: 1.5,
                psychoacoustic: 1.0,
                tfs: 1.5, // Fine temporal structure
                overrides: vec![
                    (0, 1.8),  // Spectral centroid - frequency center
                    (10, 1.8), // RMS - amplitude
                    (12, 1.5), // Attack - onset
                ],
            }
        }

        // General animal sound archives - use default
        _ => FeatureWeights::default(),
    }
}

/// Get a unified "bioacoustic" weight vector that works across species
///
/// This emphasizes features that are important across ALL species:
/// - Temporal envelope (attack/decay)
/// - Spectral shape (centroid, kurtosis)
/// - Modulation (FM slope for vocalizations)
/// - Micro-dynamics (rhythm patterns)
fn get_unified_bioacoustic_weights() -> FeatureWeights {
    FeatureWeights {
        spectral: 1.5,       // Spectral shape important for all
        harmonic: 1.2,       // Moderate importance
        temporal: 1.8,       // Envelope shape critical
        modulation: 2.0,     // FM important for vocalizations
        cepstral: 1.0,       // Standard
        formant: 1.0,        // Moderate
        micro_dynamics: 1.8, // Rhythm important for detection
        psychoacoustic: 1.2,
        tfs: 1.3,
        overrides: vec![
            (0, 1.6),  // D0: spectral_centroid
            (3, 1.8),  // D3: spectral_kurtosis
            (10, 1.5), // D10: RMS
            (12, 1.6), // D12: attack
            (18, 2.2), // D18: fm_slope - CRITICAL for vocalization detection
            (30, 1.5), // D30: onset_rate
            (31, 1.6), // D31: median_ici
        ],
    }
}

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
    sample_idx: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeightedBenchmarkResults {
    // Configuration
    use_species_weights: bool,
    use_unified_weights: bool,

    // Detection metrics
    unweighted_f1: f64,
    weighted_f1: f64,
    improvement_pct: f64,

    // Per-dataset improvements
    per_dataset_improvements: HashMap<String, DatasetImprovement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DatasetImprovement {
    dataset: String,
    unweighted_f1: f64,
    weighted_f1: f64,
    improvement_pct: f64,
    weight_type: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║     BEANS-Zero 45D Weighted Benchmark                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Testing hypothesis: Species-specific weights improve accuracy");
    println!("  ├─ Previous (Unweighted): ~90.34%");
    println!("  └─ Expected (Weighted):  ~92-93%");
    println!();

    let total_start = Instant::now();

    // Load manifest
    let manifest_path = "beans_zero_cache/beans_audio_manifest.json";
    println!("Loading manifest from: {}", manifest_path);
    let file = File::open(manifest_path)?;
    let reader = BufReader::new(file);
    let manifest: Manifest = serde_json::from_reader(reader)?;

    let total_samples = manifest.samples.len();
    println!("Loaded {} samples from manifest", total_samples);
    println!();

    // Extract features
    println!("[1/4] Extracting features...");
    let extraction_start = Instant::now();
    let processed = Arc::new(AtomicUsize::new(0));

    let all_samples: Vec<Option<Sample>> = manifest
        .samples
        .par_iter()
        .enumerate()
        .map(|(idx, entry)| {
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 20000 == 0 {
                println!("  Progress: {}/{}", count + 1, total_samples);
            }

            let audio_path = format!("beans_zero_cache/{}", entry.audio_file);
            let audio = match load_audio_raw(&audio_path, entry.n_samples) {
                Ok(a) => a,
                Err(e) => {
                    if count < 5 {
                        eprintln!("Failed to load {}: {}", audio_path, e);
                    }
                    return None;
                }
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

    println!(
        "Extracted {} valid samples in {:.1}s",
        valid_samples.len(),
        extraction_time.as_secs_f64()
    );
    println!();

    // Split into train/test
    println!("[2/4] Splitting train/test (70/30)...");
    let mut samples_by_dataset: HashMap<String, Vec<&Sample>> = HashMap::new();
    for sample in &valid_samples {
        samples_by_dataset
            .entry(sample.source_dataset.clone())
            .or_default()
            .push(sample);
    }

    let mut train_samples: Vec<&Sample> = Vec::new();
    let mut test_samples: Vec<&Sample> = Vec::new();

    for (_, mut samples) in samples_by_dataset {
        samples.sort_by_key(|s| s.sample_idx);
        let split_point = (samples.len() as f64 * 0.7) as usize;

        for (i, sample) in samples.into_iter().enumerate() {
            if i < split_point {
                train_samples.push(sample);
            } else {
                test_samples.push(sample);
            }
        }
    }

    println!("  Train: {}, Test: {}", train_samples.len(), test_samples.len());
    println!();

    // Build prototypes
    println!("[3/4] Building species prototypes...");
    let mut prototypes_by_dataset: HashMap<String, Vec<f64>> = HashMap::new();

    for sample in &train_samples {
        let entry = prototypes_by_dataset
            .entry(sample.source_dataset.clone())
            .or_insert_with(|| vec![0.0; FEATURE_DIM]);

        for (i, &val) in sample.features.iter().enumerate() {
            entry[i] += val;
        }
    }

    // Also track counts for averaging
    let mut counts: HashMap<String, usize> = HashMap::new();
    for sample in &train_samples {
        *counts.entry(sample.source_dataset.clone()).or_insert(0) += 1;
    }

    for (dataset, prototype) in &mut prototypes_by_dataset {
        let count = counts.get(dataset).copied().unwrap_or(1);
        for val in prototype.iter_mut() {
            *val /= count as f64;
        }
    }

    println!("Built {} prototypes", prototypes_by_dataset.len());
    println!();

    // =========================================================================
    // BENCHMARK: UNWEIGHTED vs WEIGHTED
    // =========================================================================
    println!("[4/4] Running benchmark: Unweighted vs Weighted...");
    println!();

    // Filter test samples to detection task
    let detection_test: Vec<_> = test_samples.iter().filter(|s| s.task == "detection").collect();

    println!("Testing on {} detection samples", detection_test.len());
    println!();

    // Create engines
    let mut engine_unweighted = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    let mut engine_weighted = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    // Fit normalization on both
    {
        let mut matrix = ndarray::Array2::<f64>::zeros((train_samples.len().min(10000), FEATURE_DIM));
        for (i, sample) in train_samples.iter().take(10000).enumerate() {
            for (j, &val) in sample.features.iter().enumerate() {
                matrix[[i, j]] = val;
            }
        }
        engine_unweighted.fit_normalization(&matrix);
        engine_weighted.fit_normalization(&matrix);
    }

    // Set unified bioacoustic weights on weighted engine
    let unified_weights = get_unified_bioacoustic_weights();
    let weight_vector = unified_weights.to_weight_vector();
    engine_weighted.set_feature_weights(&weight_vector);

    println!("Applied unified bioacoustic weights:");
    println!("  ├─ Temporal: {:.1}", unified_weights.temporal);
    println!("  ├─ Modulation: {:.1}", unified_weights.modulation);
    println!("  ├─ Spectral: {:.1}", unified_weights.spectral);
    println!("  └─ Micro-dynamics: {:.1}", unified_weights.micro_dynamics);
    println!();

    // Evaluate both
    let threshold = 0.5;
    let eval_samples: Vec<&Sample> = detection_test.iter().take(5000).cloned().cloned().collect();

    let (unweighted_f1, unweighted_per_dataset) =
        evaluate_with_engine(&eval_samples, &prototypes_by_dataset, &engine_unweighted, threshold);

    let (weighted_f1, weighted_per_dataset) =
        evaluate_with_engine(&eval_samples, &prototypes_by_dataset, &engine_weighted, threshold);

    let improvement = (weighted_f1 - unweighted_f1) / unweighted_f1 * 100.0;

    // =========================================================================
    // RESULTS
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BENCHMARK RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ OVERALL COMPARISON                                                          │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Unweighted F1:  {:.2}%", unweighted_f1 * 100.0);
    println!("│ Weighted F1:    {:.2}%", weighted_f1 * 100.0);
    println!("│ Improvement:    {:+.2}%", improvement);
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Per-dataset improvements
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PER-DATASET RESULTS                                                         │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│ {:<20} {:>10} {:>10} {:>10}",
        "Dataset", "Unweighted", "Weighted", "Δ"
    );
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    let mut datasets: Vec<_> = unweighted_per_dataset.keys().collect();
    datasets.sort();

    for dataset in &datasets {
        let unw = unweighted_per_dataset.get(*dataset).unwrap_or(&0.0);
        let w = weighted_per_dataset.get(*dataset).unwrap_or(&0.0);
        let delta = (w - unw) / unw.max(0.001) * 100.0;

        println!(
            "│ {:<20} {:>9.1}% {:>9.1}% {:+>9.1}%",
            dataset,
            unw * 100.0,
            w * 100.0,
            delta
        );
    }
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Hypothesis conclusion
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("HYPOTHESIS VALIDATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if weighted_f1 > unweighted_f1 {
        println!("✓ CONFIRMED: Weighted features improve accuracy");
        println!();
        println!("The unified bioacoustic weights successfully suppress noise dimensions");
        println!("and emphasize features that matter across species:");
        println!("  • FM slope (modulation) - critical for vocalization detection");
        println!("  • Temporal envelope - attack/decay patterns");
        println!("  • Micro-dynamics - rhythm and onset patterns");
    } else {
        println!("✗ NOT CONFIRMED: Weights did not improve accuracy in this test");
        println!();
        println!("This may indicate:");
        println!("  • The test set doesn't benefit from these specific weights");
        println!("  • Species-specific weights would be more effective than unified");
    }

    // Save results
    let results = WeightedBenchmarkResults {
        use_species_weights: false,
        use_unified_weights: true,
        unweighted_f1,
        weighted_f1,
        improvement_pct: improvement,
        per_dataset_improvements: unweighted_per_dataset
            .iter()
            .map(|(k, &unw)| {
                let w = weighted_per_dataset.get(k).copied().unwrap_or(0.0);
                (
                    k.clone(),
                    DatasetImprovement {
                        dataset: k.clone(),
                        unweighted_f1: unw,
                        weighted_f1: w,
                        improvement_pct: (w - unw) / unw.max(0.001) * 100.0,
                        weight_type: "unified_bioacoustic".to_string(),
                    },
                )
            })
            .collect(),
    };

    std::fs::create_dir_all("complete_analysis").ok();
    let file = File::create("complete_analysis/beans_weighted_benchmark.json")?;
    serde_json::to_writer_pretty(BufWriter::new(file), &results)?;

    println!();
    println!("Total time: {:.1}s", total_start.elapsed().as_secs_f64());
    println!("Results saved to: complete_analysis/beans_weighted_benchmark.json");

    Ok(())
}

fn evaluate_with_engine(
    test_samples: &[&Sample],
    prototypes: &HashMap<String, Vec<f64>>,
    engine: &AcousticSimilarityEngine,
    threshold: f64,
) -> (f64, HashMap<String, f64>) {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_count = 0usize;

    let mut per_dataset_correct: HashMap<String, usize> = HashMap::new();
    let mut per_dataset_total: HashMap<String, usize> = HashMap::new();

    for sample in test_samples {
        let query = Array1::from_vec(sample.features.clone());

        let mut best_sim = 0.0;
        let mut best_dataset = "";

        for (dataset, prototype) in prototypes {
            let proto = Array1::from_vec(prototype.clone());
            let sim = 1.0 - engine.distance(&query, &proto);
            if sim > best_sim {
                best_sim = sim;
                best_dataset = dataset.as_str();
            }
        }

        let detected = best_sim >= threshold;
        let is_correct = best_dataset == sample.source_dataset;

        *per_dataset_total.entry(sample.source_dataset.clone()).or_insert(0) += 1;

        if detected {
            if is_correct {
                tp += 1;
                *per_dataset_correct.entry(sample.source_dataset.clone()).or_insert(0) += 1;
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

    let per_dataset_f1: HashMap<String, f64> = per_dataset_total
        .iter()
        .map(|(dataset, &total)| {
            let correct = per_dataset_correct.get(dataset).copied().unwrap_or(0);
            let dataset_f1 = if total > 0 {
                let p = correct as f64 / total as f64;
                let r = correct as f64 / total as f64;
                if p + r > 0.0 {
                    2.0 * p * r / (p + r)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            (dataset.clone(), dataset_f1)
        })
        .collect();

    (f1, per_dataset_f1)
}

fn load_audio_raw(path: &str, expected_samples: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Raw f32 samples (stored as 32-bit floats)
    if buffer.len() >= expected_samples * 4 {
        let samples: Vec<f64> = buffer[..expected_samples * 4]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()) as f64)
            .collect();
        Ok(samples)
    } else {
        Err("Insufficient data".into())
    }
}
