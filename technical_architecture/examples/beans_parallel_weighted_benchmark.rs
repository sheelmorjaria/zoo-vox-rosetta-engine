//! Parallelized Weighted Benchmark
//!
//! Uses rayon for parallel feature extraction to speed up processing

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

fn get_unified_bioacoustic_weights() -> FeatureWeights {
    FeatureWeights {
        spectral: 1.5,
        harmonic: 1.2,
        temporal: 1.8,
        modulation: 2.0,
        cepstral: 1.0,
        formant: 1.0,
        micro_dynamics: 1.8,
        psychoacoustic: 1.2,
        tfs: 1.3,
        overrides: vec![
            (0, 1.6),
            (3, 1.8),
            (10, 1.5),
            (12, 1.6),
            (18, 2.2),
            (30, 1.5),
            (31, 1.6),
        ],
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
    println!("║     Parallelized Weighted Benchmark                                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
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
        .map(|entry| {
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
                    sample_idx: count,
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

    // Benchmark
    println!("[3/3] Running benchmark...");
    let bench_start = Instant::now();

    // Create engines
    let mut engine_unw = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    let mut engine_wgt = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    // Fit normalization
    {
        let n_fit = train_samples.len().min(5000);
        let mut matrix = ndarray::Array2::<f64>::zeros((n_fit, FEATURE_DIM));
        for (i, sample) in train_samples.iter().take(n_fit).enumerate() {
            for (j, &v) in sample.features.iter().enumerate() {
                matrix[[i, j]] = v;
            }
        }
        engine_unw.fit_normalization(&matrix);
        engine_wgt.fit_normalization(&matrix);
    }

    // Apply weights
    let weights = get_unified_bioacoustic_weights();
    engine_wgt.set_feature_weights(&weights.to_weight_vector());

    println!("Applied unified bioacoustic weights:");
    println!(
        "  ├─ Temporal: {:.1}, Modulation: {:.1}",
        weights.temporal, weights.modulation
    );
    println!(
        "  └─ Spectral: {:.1}, Micro-dynamics: {:.1}",
        weights.spectral, weights.micro_dynamics
    );
    println!();

    // Evaluate
    let threshold = 0.5;
    let (unw_f1, unw_per_ds) = evaluate_parallel(&test_samples, &prototypes, &engine_unw, threshold);
    let (wgt_f1, wgt_per_ds) = evaluate_parallel(&test_samples, &prototypes, &engine_wgt, threshold);
    let improvement = (wgt_f1 - unw_f1) / unw_f1.max(0.001) * 100.0;
    let bench_time = bench_start.elapsed();

    // Results
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BENCHMARK RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ OVERALL COMPARISON                                                          │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Unweighted F1:  {:.2}%", unw_f1 * 100.0);
    println!("│ Weighted F1:    {:.2}%", wgt_f1 * 100.0);
    println!("│ Improvement:    {:+.2}%", improvement);
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Per-dataset results
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PER-DATASET RESULTS                                                         │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ {:<25} {:>10} {:>10} {:>8}", "Dataset", "Unweighted", "Weighted", "Δ");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    let mut datasets: Vec<_> = unw_per_ds.keys().collect();
    datasets.sort();

    for ds in &datasets {
        let unw = unw_per_ds.get(*ds).unwrap_or(&0.0);
        let wgt = wgt_per_ds.get(*ds).unwrap_or(&0.0);
        let delta = (wgt - unw) / unw.max(0.001) * 100.0;
        println!(
            "│ {:<25} {:>9.1}% {:>9.1}% {:+>7.1}%",
            ds,
            unw * 100.0,
            wgt * 100.0,
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

    if wgt_f1 > unw_f1 {
        println!("✓ CONFIRMED: Weighted features improve accuracy!");
        println!();
        println!("The unified bioacoustic weights successfully emphasize:");
        println!("  • FM slope (modulation) - critical for vocalization detection");
        println!("  • Temporal envelope - attack/decay patterns");
        println!("  • Micro-dynamics - rhythm and onset patterns");
    } else {
        println!("✗ NOT CONFIRMED in this test set");
        println!();
        println!("Possible reasons:");
        println!("  • Test set may not benefit from these specific weights");
        println!("  • Species-specific weights may be more effective than unified");
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

fn evaluate_parallel(
    test_samples: &[&Sample],
    prototypes: &HashMap<String, Vec<f64>>,
    engine: &AcousticSimilarityEngine,
    threshold: f64,
) -> (f64, HashMap<String, f64>) {
    // Parallel evaluation
    let results: Vec<(bool, bool, String)> = test_samples
        .par_iter()
        .map(|sample| {
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

            (detected, is_correct, sample.source_dataset.clone())
        })
        .collect();

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_count = 0usize;
    let mut per_ds_correct: HashMap<String, usize> = HashMap::new();
    let mut per_ds_total: HashMap<String, usize> = HashMap::new();

    for (detected, correct, dataset) in results {
        *per_ds_total.entry(dataset.clone()).or_insert(0) += 1;

        if detected {
            if correct {
                tp += 1;
                *per_ds_correct.entry(dataset).or_insert(0) += 1;
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
