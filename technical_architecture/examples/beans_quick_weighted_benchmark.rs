//! Quick Weighted Benchmark - Uses subset for fast validation
//!
//! Tests the hypothesis with 10,000 samples instead of 92,000

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
const MAX_SAMPLES: usize = 10000; // Quick test

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
    println!("║     Quick Weighted Benchmark (10K samples)                                     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();

    // Load manifest
    let file = File::open("beans_zero_cache/beans_audio_manifest.json")?;
    let manifest: Manifest = serde_json::from_reader(BufReader::new(file))?;
    println!(
        "Loaded manifest with {} samples, using first {}",
        manifest.samples.len(),
        MAX_SAMPLES
    );

    // Extract features
    println!("[1/3] Extracting features from {} samples...", MAX_SAMPLES);
    let processed = Arc::new(AtomicUsize::new(0));

    let samples: Vec<Sample> = manifest
        .samples
        .into_iter()
        .take(MAX_SAMPLES)
        .enumerate()
        .filter_map(|(idx, entry)| {
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 1000 == 0 {
                println!("  Progress: {}/{}", count, MAX_SAMPLES);
            }

            let audio_path = format!("beans_zero_cache/{}", entry.audio_file);
            let audio = load_audio_f32(&audio_path, entry.n_samples).ok()?;

            if audio.len() < 100 {
                return None;
            }

            let mut extractor = ZooVoxFeatureExtractor::new(entry.sample_rate);
            extractor.extract_45d(&audio).ok().map(|f| Sample {
                features: f.to_vector().to_vec(),
                source_dataset: entry.labels.source_dataset,
                sample_idx: idx,
            })
        })
        .collect();

    println!("Extracted {} valid samples", samples.len());

    // Split train/test
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
    println!();

    // Create engines
    let mut engine_unw = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    let mut engine_wgt = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    // Fit normalization
    {
        let mut matrix = ndarray::Array2::<f64>::zeros((train_samples.len().min(5000), FEATURE_DIM));
        for (i, sample) in train_samples.iter().take(5000).enumerate() {
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

    // Evaluate
    let threshold = 0.5;
    let (unw_f1, _) = evaluate(&test_samples, &prototypes, &engine_unw, threshold);
    let (wgt_f1, _) = evaluate(&test_samples, &prototypes, &engine_wgt, threshold);
    let improvement = (wgt_f1 - unw_f1) / unw_f1.max(0.001) * 100.0;

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

    if wgt_f1 > unw_f1 {
        println!("✓ HYPOTHESIS CONFIRMED: Weighted features improve accuracy!");
        println!();
        println!("The unified bioacoustic weights successfully emphasize:");
        println!("  • FM slope (modulation) - vocalization detection");
        println!("  • Temporal envelope - attack/decay patterns");
        println!("  • Micro-dynamics - rhythm patterns");
    } else {
        println!("✗ Hypothesis not confirmed with this sample size.");
    }

    println!();
    println!("Total time: {:.1}s", total_start.elapsed().as_secs_f64());

    Ok(())
}

fn evaluate(
    test_samples: &[&Sample],
    prototypes: &HashMap<String, Vec<f64>>,
    engine: &AcousticSimilarityEngine,
    threshold: f64,
) -> (f64, HashMap<String, f64>) {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_count = 0usize;

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

        if best_sim >= threshold {
            if best_dataset == sample.source_dataset {
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

    (f1, HashMap::new())
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
