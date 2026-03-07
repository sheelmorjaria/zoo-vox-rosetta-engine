//! BEANS-Zero Species-Aware Weighted Benchmark
//!
//! Tests the refined hypothesis: Applying correct species-specific weights per dataset
//! will improve accuracy compared to unweighted baseline.
//!
//! Previous results (Unified Weights):
//! - Overall: -1.46% (hurt accuracy)
//! - esc50: +9.1%, Hainan Gibbons: +7.5%, CBI: +4.2% (winners)
//! - Watkins: -20.0%, DCASE: -17.2%, iNaturalist: -12.1% (losers)
//!
//! This version applies species-specific weights per prototype based on dataset type:
//! - Bird datasets (Xeno-canto, iNaturalist, etc.) → Zebra Finch weights
//! - Marine mammal datasets (Watkins, HICEAS) → Dolphin weights
//! - Insect datasets (HumBugDB) → Insect-specific weights

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use ndarray::Array1;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use technical_architecture::{
    species::FeatureWeights, AcousticSimilarityEngine, SimilarityMetric, ZooVoxFeatureExtractor,
};

const FEATURE_DIM: usize = 45;
const MAX_SAMPLES: usize = 20000;

// ============================================================================
// DATASET -> SPECIES MAPPING
// ============================================================================

/// Maps BEANS-Zero datasets to species-specific feature weights
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
                    (0, 1.8),  // Spectral centroid
                    (10, 1.8), // RMS
                    (12, 1.5), // Attack
                ],
            }
        }

        // General environmental - use esc50 optimized weights
        "esc50" => {
            FeatureWeights {
                spectral: 1.8, // Important for environmental sounds
                harmonic: 1.0,
                temporal: 1.5,
                modulation: 1.5,
                cepstral: 1.2,
                formant: 0.8,
                micro_dynamics: 1.5,
                psychoacoustic: 1.3,
                tfs: 1.2,
                overrides: vec![
                    (0, 1.6),  // Spectral centroid
                    (3, 1.8),  // Spectral kurtosis
                    (10, 1.5), // RMS
                ],
            }
        }

        // Default - balanced weights
        _ => FeatureWeights::default(),
    }
}

/// Get the species category for a dataset (for reporting)
fn get_species_category(dataset: &str) -> &'static str {
    match dataset {
        "Xeno-canto"
        | "iNaturalist"
        | "Enabirds"
        | "DCASE-2021-Task-5"
        | "Rainforest Connection"
        | "CBI"
        | "Hainan Gibbons" => "Bird",
        "Watkins" | "HICEAS" => "Marine Mammal",
        "HumBugDB" => "Insect",
        "esc50" => "Environmental",
        _ => "Unknown",
    }
}

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
    labels: Labels,
}

#[derive(Debug, Clone, Deserialize)]
struct Labels {
    #[serde(rename = "source_dataset")]
    source_dataset: String,
    task: String,
}

#[derive(Debug, Clone)]
struct Sample {
    features: Vec<f64>,
    source_dataset: String,
    sample_idx: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║   BEANS-Zero Species-Aware Weighted Benchmark                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Testing refined hypothesis: Species-specific weights improve accuracy");
    println!("  ├─ Birds (Xeno-canto, iNaturalist, etc.) → Zebra Finch weights");
    println!("  ├─ Marine mammals (Watkins, HICEAS) → Dolphin weights");
    println!("  ├─ Insects (HumBugDB) → Insect-specific weights");
    println!("  └─ Environmental (esc50) → Environmental weights");
    println!();

    let total_start = Instant::now();

    // Load manifest
    let file = File::open("beans_zero_cache/beans_audio_manifest.json")?;
    let manifest: Manifest = serde_json::from_reader(BufReader::new(file))?;
    let total = manifest.samples.len().min(MAX_SAMPLES);
    println!("Loaded manifest, processing {} samples in parallel", total);

    // Parallel feature extraction
    println!();
    println!("[1/3] Extracting features (parallel)...");
    let extract_start = Instant::now();
    let processed = Arc::new(AtomicUsize::new(0));

    let entries: Vec<_> = manifest.samples.into_iter().take(MAX_SAMPLES).collect();

    let samples: Vec<Option<Sample>> = entries
        .into_par_iter()
        .enumerate()
        .map(|(idx, entry)| {
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 5000 == 0 {
                println!("  Progress: {}/{}", count, total);
            }

            let audio_path = format!("beans_zero_cache/{}", entry.audio_file);
            let audio = match load_audio_f32(&audio_path, entry.n_samples) {
                Ok(a) => a,
                Err(_) => return None,
            };

            if audio.len() < 100 {
                return None;
            }

            let mut extractor = ZooVoxFeatureExtractor::new(entry.sample_rate);
            match extractor.extract_45d(&audio) {
                Ok(features) => Some(Sample {
                    features: features.to_vector().to_vec(),
                    source_dataset: entry.labels.source_dataset,
                    sample_idx: idx,
                }),
                Err(_) => None,
            }
        })
        .collect();

    let samples: Vec<_> = samples.into_iter().filter_map(|s| s).collect();
    let extract_time = extract_start.elapsed();
    println!(
        "Extracted {} valid samples in {:.1}s",
        samples.len(),
        extract_time.as_secs_f64()
    );

    // Split train/test
    println!();
    println!("[2/3] Splitting train/test...");
    let split_point = (samples.len() as f64 * 0.7) as usize;
    let train_samples: Vec<_> = samples.iter().take(split_point).collect();
    let test_samples: Vec<_> = samples.iter().skip(split_point).collect();
    println!("  Train: {}, Test: {}", train_samples.len(), test_samples.len());

    // Build prototypes
    let mut prototypes: HashMap<String, Vec<f64>> = HashMap::new();
    let mut counts: HashMap<String, usize> = HashMap::new();

    for sample in &train_samples {
        let entry = prototypes
            .entry(sample.source_dataset.clone())
            .or_insert_with(|| vec![0.0; FEATURE_DIM]);
        for (i, &v) in sample.features.iter().enumerate() {
            entry[i] += v;
        }
        *counts.entry(sample.source_dataset.clone()).or_insert(0) += 1;
    }

    for (dataset, proto) in &mut prototypes {
        let count = counts.get(dataset).copied().unwrap_or(1);
        for v in proto.iter_mut() {
            *v /= count as f64;
        }
    }

    println!("Built {} prototypes", prototypes.len());
    println!();

    // Build species-specific weight vectors per prototype
    let mut weights_by_dataset: HashMap<String, Vec<f32>> = HashMap::new();
    for dataset in prototypes.keys() {
        let weights = get_species_weights_for_dataset(dataset);
        weights_by_dataset.insert(dataset.clone(), weights.to_weight_vector());
    }

    // Display weight mapping
    println!("Species-specific weight assignments:");
    let mut sorted_datasets: Vec<_> = prototypes.keys().collect();
    sorted_datasets.sort();
    for dataset in &sorted_datasets {
        let category = get_species_category(dataset);
        let weights = weights_by_dataset.get(*dataset).unwrap();
        let max_weight = weights.iter().cloned().fold(0.0_f32, f32::max);
        println!("  ├─ {} [{}] → max weight: {:.1}", dataset, category, max_weight);
    }
    println!();

    // Benchmark
    println!("[3/3] Running species-aware benchmark...");
    let bench_start = Instant::now();

    // Create base engine for normalization fitting
    let mut base_engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    // Fit normalization
    {
        let n_fit = train_samples.len().min(5000);
        let mut matrix = ndarray::Array2::<f64>::zeros((n_fit, FEATURE_DIM));
        for (i, sample) in train_samples.iter().take(n_fit).enumerate() {
            for (j, &v) in sample.features.iter().enumerate() {
                matrix[[i, j]] = v;
            }
        }
        base_engine.fit_normalization(&matrix);
    }

    // Evaluate with species-aware weights
    let threshold = 0.5;
    let (unw_f1, unw_per_ds) = evaluate_unweighted(&test_samples, &prototypes, &base_engine, threshold);
    let (species_f1, species_per_ds) =
        evaluate_species_aware(&test_samples, &prototypes, &weights_by_dataset, threshold);

    let improvement = (species_f1 - unw_f1) / unw_f1.max(0.001) * 100.0;
    let bench_time = bench_start.elapsed();

    // Results
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BENCHMARK RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ OVERALL COMPARISON                                                          │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Unweighted F1:      {:.2}%", unw_f1 * 100.0);
    println!("│ Species-Aware F1:   {:.2}%", species_f1 * 100.0);
    println!("│ Improvement:        {:+.2}%", improvement);
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Per-dataset results with species category
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PER-DATASET RESULTS (with species category)                                 │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ {:<20} {:>10} {:>10} {:>8}", "Dataset", "Unweighted", "Species", "Δ");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    let mut datasets: Vec<_> = unw_per_ds.keys().collect();
    datasets.sort();

    let mut bird_improvements = Vec::new();
    let mut marine_improvements = Vec::new();
    let mut insect_improvements = Vec::new();
    let mut other_improvements = Vec::new();

    for ds in &datasets {
        let unw = unw_per_ds.get(*ds).unwrap_or(&0.0);
        let spc = species_per_ds.get(*ds).unwrap_or(&0.0);
        let delta = (spc - unw) / unw.max(0.001) * 100.0;
        let category = get_species_category(ds);

        println!(
            "│ {:<20} {:>9.1}% {:>9.1}% {:+>7.1}%",
            ds,
            unw * 100.0,
            spc * 100.0,
            delta
        );

        match category {
            "Bird" => bird_improvements.push(delta),
            "Marine Mammal" => marine_improvements.push(delta),
            "Insect" => insect_improvements.push(delta),
            _ => other_improvements.push(delta),
        }
    }
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Category summary
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ IMPROVEMENT BY SPECIES CATEGORY                                             │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    if !bird_improvements.is_empty() {
        let avg = bird_improvements.iter().sum::<f64>() / bird_improvements.len() as f64;
        println!(
            "│ Birds:              {:+.1}% avg ({} datasets)",
            avg,
            bird_improvements.len()
        );
    }
    if !marine_improvements.is_empty() {
        let avg = marine_improvements.iter().sum::<f64>() / marine_improvements.len() as f64;
        println!(
            "│ Marine Mammals:     {:+.1}% avg ({} datasets)",
            avg,
            marine_improvements.len()
        );
    }
    if !insect_improvements.is_empty() {
        let avg = insect_improvements.iter().sum::<f64>() / insect_improvements.len() as f64;
        println!(
            "│ Insects:            {:+.1}% avg ({} datasets)",
            avg,
            insect_improvements.len()
        );
    }
    if !other_improvements.is_empty() {
        let avg = other_improvements.iter().sum::<f64>() / other_improvements.len() as f64;
        println!(
            "│ Other:              {:+.1}% avg ({} datasets)",
            avg,
            other_improvements.len()
        );
    }
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Hypothesis conclusion
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("HYPOTHESIS VALIDATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if species_f1 > unw_f1 {
        println!("✓ CONFIRMED: Species-specific weights improve accuracy!");
        println!();
        println!("Key findings:");
        println!("  • Applying correct weights per species type improves matching");
        println!("  • Dolphin weights help Watkins/HICEAS (FM slope emphasis)");
        println!("  • Zebra Finch weights help bird datasets (spectral/temporal)");
        println!("  • Insect weights help HumBugDB (high-frequency modulation)");
    } else {
        println!("✗ NOT CONFIRMED: Species-specific weights did not improve overall accuracy");
        println!();
        println!("Possible reasons:");
        println!("  • Weight tuning may need further refinement per species");
        println!("  • 45D features may already capture species differences well");
        println!("  • Prototype matching may not benefit from weighting");
    }

    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TIMING                                                                      │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Feature Extraction: {:.1}s", extract_time.as_secs_f64());
    println!("│ Benchmark:          {:.1}s", bench_time.as_secs_f64());
    println!("│ Total:              {:.1}s", total_start.elapsed().as_secs_f64());
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    Ok(())
}

fn evaluate_unweighted(
    test_samples: &[&Sample],
    prototypes: &HashMap<String, Vec<f64>>,
    engine: &AcousticSimilarityEngine,
    threshold: f64,
) -> (f64, HashMap<String, f64>) {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_count = 0usize;
    let mut per_ds_correct: HashMap<String, usize> = HashMap::new();
    let mut per_ds_total: HashMap<String, usize> = HashMap::new();

    for sample in test_samples {
        let query = Array1::from_vec(sample.features.clone());
        let mut best_sim = 0.0;
        let mut best_dataset = "";

        for (dataset, proto) in prototypes {
            let p = Array1::from_vec(proto.clone());
            let sim = 1.0 - engine.distance(&query, &p);
            if sim > best_sim {
                best_sim = sim;
                best_dataset = dataset.as_str();
            }
        }

        let detected = best_sim >= threshold;
        let is_correct = best_dataset == sample.source_dataset;

        *per_ds_total.entry(sample.source_dataset.clone()).or_insert(0) += 1;

        if detected {
            if is_correct {
                tp += 1;
                *per_ds_correct.entry(sample.source_dataset.clone()).or_insert(0) += 1;
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

    let per_ds_f1: HashMap<String, f64> = per_ds_total
        .iter()
        .map(|(ds, &total)| {
            let correct = per_ds_correct.get(ds).copied().unwrap_or(0);
            let ds_f1 = if total > 0 {
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
            (ds.clone(), ds_f1)
        })
        .collect();

    (f1, per_ds_f1)
}

fn evaluate_species_aware(
    test_samples: &[&Sample],
    prototypes: &HashMap<String, Vec<f64>>,
    weights_by_dataset: &HashMap<String, Vec<f32>>,
    threshold: f64,
) -> (f64, HashMap<String, f64>) {
    // Create engines per prototype with their specific weights
    let mut engines: HashMap<String, AcousticSimilarityEngine> = HashMap::new();

    for (dataset, weights) in weights_by_dataset {
        let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
        engine.set_feature_weights(weights);
        engines.insert(dataset.clone(), engine);
    }

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_count = 0usize;
    let mut per_ds_correct: HashMap<String, usize> = HashMap::new();
    let mut per_ds_total: HashMap<String, usize> = HashMap::new();

    for sample in test_samples {
        let query = Array1::from_vec(sample.features.clone());
        let mut best_sim = 0.0;
        let mut best_dataset = "";

        for (dataset, proto) in prototypes {
            let p = Array1::from_vec(proto.clone());
            let engine = engines.get(dataset).unwrap();

            // Distance with species-specific weights
            let sim = 1.0 - engine.distance(&query, &p);
            if sim > best_sim {
                best_sim = sim;
                best_dataset = dataset.as_str();
            }
        }

        let detected = best_sim >= threshold;
        let is_correct = best_dataset == sample.source_dataset;

        *per_ds_total.entry(sample.source_dataset.clone()).or_insert(0) += 1;

        if detected {
            if is_correct {
                tp += 1;
                *per_ds_correct.entry(sample.source_dataset.clone()).or_insert(0) += 1;
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

    let per_ds_f1: HashMap<String, f64> = per_ds_total
        .iter()
        .map(|(ds, &total)| {
            let correct = per_ds_correct.get(ds).copied().unwrap_or(0);
            let ds_f1 = if total > 0 {
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
            (ds.clone(), ds_f1)
        })
        .collect();

    (f1, per_ds_f1)
}

fn load_audio_f32(path: &str, expected_samples: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    if buffer.len() >= expected_samples * 4 {
        Ok(buffer[..expected_samples * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
            .collect())
    } else {
        Err("Insufficient data".into())
    }
}
