//! BEANS-Zero 45D Feature Assessment
//!
//! Compares 30D vs 45D feature performance on the full BEANS-Zero dataset.
//! Tests the new 15 dimensions:
//! - Resonance Factors (6): Formants 1-3, Bandwidths 1-2, Dispersion
//! - Spectral Shape Factors (4): Centroid, Spread, Skewness, Kurtosis
//! - Modulation Factors (3): Tilt, FM Slope, AM Depth
//! - Non-Linear Factors (2): Subharmonic Ratio, Spectral Entropy

use technical_architecture::{
    AcousticSimilarityEngine, SimilarityMetric,
    ZooVoxFeatureExtractor,
};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

const FEATURE_DIM: usize = 45;

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
struct ExtractedFeatures {
    sample_id: String,
    features: Vec<f64>,
    dataset: Option<String>,
    task: Option<String>,
}

#[derive(Debug, Clone)]
struct AcousticType {
    type_id: String,
    centroid: Vec<f64>,
    count: usize,
    sample_ids: Vec<String>,
    avg_distance_to_centroid: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AssessmentResults45D {
    dataset: String,
    total_samples: usize,
    feature_dim: usize,
    total_time_sec: f64,
    throughput_samples_per_sec: f64,
    global_types: usize,
    type_entropy: f64,
    knn_accuracy: f64,
    knn_best_k: usize,
    avg_intra_type_similarity: f64,
    avg_inter_type_distance: f64,
    separation_ratio: Option<f64>,
    source_datasets: HashMap<String, usize>,
    task_types: HashMap<String, usize>,
    // 45D specific metrics
    new_feature_ranges: HashMap<String, (f64, f64)>,
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║   BEANS-Zero 45D Feature Assessment (30D + 15 New Dimensions)             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Load manifest
    let manifest_path = "beans_zero_cache/beans_audio_manifest.json";
    println!("Loading manifest: {}", manifest_path);

    let file = File::open(manifest_path)?;
    let reader = BufReader::new(file);
    let manifest: Manifest = serde_json::from_reader(reader)?;

    let total_samples = manifest.samples.len();
    println!("\nConfiguration:");
    println!("  ├─ Dataset: EarthSpeciesProject/BEANS-Zero");
    println!("  ├─ Split: test");
    println!("  ├─ Total Samples: {}", total_samples);
    println!("  ├─ Feature Dimension: 45D");
    println!("  ├─ Similarity Threshold: 0.85");
    println!("  ├─ k-NN Neighbors: 10");
    println!("  ├─ Parallel: Rayon with 32 threads");
    println!("  └─ Feature Extraction: ZooVoxFeatureExtractor (FFT-based)");
    println!();

    // ========================================================================
    // Phase 1: Parallel Feature Extraction (45D)
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 1: Parallel Feature Extraction (45D)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Extracting {} samples using FFT-based ZooVoxFeatureExtractor...", total_samples);
    println!("  (Parallel with 32 threads)");
    println!();

    let start_time = Instant::now();
    let processed = Arc::new(AtomicUsize::new(0));

    // Extract features in parallel
    let features_results: Vec<Option<ExtractedFeatures>> = manifest.samples
        .par_iter()
        .enumerate()
        .map(|(idx, entry)| {
            // Update progress
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 5000 == 0 {
                println!("  Progress: {}/{} samples", count + 1, total_samples);
            }

            // Build full path
            let audio_path = format!("beans_zero_cache/{}", entry.audio_file);

            // Load audio
            let audio = match load_audio_raw(&audio_path, entry.n_samples) {
                Ok(a) => a,
                Err(_) => return None,
            };

            // Extract 45D features
            let mut extractor = ZooVoxFeatureExtractor::new(entry.sample_rate);
            match extractor.extract_45d(&audio) {
                Ok(features) => Some(ExtractedFeatures {
                    sample_id: format!("sample_{}", idx),
                    features: features.to_vector().to_vec(),
                    dataset: Some(entry.labels.source_dataset.clone()),
                    task: Some(entry.labels.task.clone()),
                }),
                Err(_) => None,
            }
        })
        .collect();

    let extraction_time = start_time.elapsed();
    let valid_features: Vec<_> = features_results.iter()
        .filter_map(|f| f.clone())
        .collect();

    let n_valid = valid_features.len();
    println!("\nExtraction complete:");
    println!("  ├─ Processed: {} samples", total_samples);
    println!("  ├─ Failed: {} samples", total_samples - n_valid);
    println!("  ├─ Time: {:.2}s", extraction_time.as_secs_f64());
    println!("  └─ Throughput: {:.1} samples/sec", total_samples as f64 / extraction_time.as_secs_f64());
    println!();

    // ========================================================================
    // Phase 2: Type Discovery (Acoustic Similarity Engine)
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 2: Type Discovery (Acoustic Similarity Engine)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Track new feature ranges
    let mut new_feature_mins = [f64::MAX; 15];
    let mut new_feature_maxs = [f64::MIN; 15];

    for f in &valid_features {
        for j in 0..15 {
            let val = f.features[30 + j];
            if val < new_feature_mins[j] {
                new_feature_mins[j] = val;
            }
            if val > new_feature_maxs[j] {
                new_feature_maxs[j] = val;
            }
        }
    }

    // Build types using streaming approach
    let types = build_global_types_streaming(&valid_features, 0.85);
    println!();

    // ========================================================================
    // Phase 3: Statistics
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 3: Statistics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let n = valid_features.len();
    let n_types = types.len();

    // Compute entropy
    let type_entropy: f64 = types.iter()
        .map(|t| {
            let p = t.count as f64 / n as f64;
            if p > 0.0 { -p * p.log2() } else { 0.0 }
        })
        .sum();

    println!("Type Discovery:");
    println!("  ├─ Global Types: {}", n_types);
    println!("  └─ Type Entropy: {:.3} bits", type_entropy);
    println!();

    // Similarity statistics
    let (intra_sim, inter_dist) = compute_similarity_stats(&valid_features, &types);
    let separation = if intra_sim > 0.0 { Some(inter_dist / (1.0 - intra_sim + 1e-10)) } else { None };

    println!("Similarity Statistics:");
    println!("  ├─ Avg Intra-Type Similarity: {:.4}", intra_sim);
    println!("  ├─ Avg Inter-Type Distance: {:.4}", inter_dist);
    if let Some(s) = separation {
        if s.is_finite() {
            println!("  └─ Separation Ratio: {:.2}x", s);
        } else {
            println!("  └─ Separation Ratio: infx");
        }
    }
    println!();

    // New feature ranges
    let feature_names = [
        "formant_1_hz", "formant_2_hz", "formant_3_hz",
        "formant_1_bandwidth", "formant_2_bandwidth", "formant_dispersion",
        "spectral_centroid", "spectral_spread", "spectral_skewness", "spectral_kurtosis",
        "spectral_tilt", "fm_slope_hz_per_sec", "am_depth",
        "subharmonic_ratio", "spectral_entropy",
    ];

    println!("New 15D Feature Ranges:");
    let mut new_feature_ranges = HashMap::new();
    for (i, name) in feature_names.iter().enumerate() {
        let min = if new_feature_mins[i] == f64::MAX { 0.0 } else { new_feature_mins[i] };
        let max = if new_feature_maxs[i] == f64::MIN { 0.0 } else { new_feature_maxs[i] };
        println!("  ├─ {}: [{:.2}, {:.2}]", name, min, max);
        new_feature_ranges.insert(name.to_string(), (min, max));
    }
    println!();

    // ========================================================================
    // Phase 4: k-NN Classification
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 4: k-NN Classification");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let knn_accuracy = compute_knn_accuracy(&valid_features, 10);
    println!("k-NN (10-NN): {:.2}% accuracy", knn_accuracy * 100.0);
    println!();

    // ========================================================================
    // Phase 5: Label Analysis
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Phase 5: Label Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Count source datasets and task types
    let mut source_datasets: HashMap<String, usize> = HashMap::new();
    let mut task_types: HashMap<String, usize> = HashMap::new();

    for f in &valid_features {
        if let Some(ref d) = f.dataset {
            *source_datasets.entry(d.clone()).or_insert(0) += 1;
        }
        if let Some(ref t) = f.task {
            *task_types.entry(t.clone()).or_insert(0) += 1;
        }
    }

    // Sort and display
    let mut datasets_sorted: Vec<_> = source_datasets.iter().collect();
    datasets_sorted.sort_by(|a, b| b.1.cmp(a.1));

    println!("Source Datasets:");
    let total: usize = source_datasets.values().sum();
    for (name, count) in datasets_sorted.iter().take(10) {
        let pct = **count as f64 / total as f64 * 100.0;
        println!("  ├─ {}: {} ({:.1}%)", name, count, pct);
    }
    println!();

    let mut tasks_sorted: Vec<_> = task_types.iter().collect();
    tasks_sorted.sort_by(|a, b| b.1.cmp(a.1));

    println!("Task Types:");
    for (name, count) in tasks_sorted {
        let pct = *count as f64 / total as f64 * 100.0;
        println!("  ├─ {}: {} ({:.1}%)", name, count, pct);
    }
    println!();

    // ========================================================================
    // Final Summary
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("FINAL SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let total_time = start_time.elapsed();

    println!("Dataset: EarthSpeciesProject/BEANS-Zero");
    println!("Samples: {}", total_samples);
    println!("Features: 45D");
    println!();
    println!("Performance:");
    println!("  ├─ Total time: {:.1}s ({:.1} min)", total_time.as_secs_f64(), total_time.as_secs_f64() / 60.0);
    println!("  └─ Throughput: {:.1} samples/sec", total_samples as f64 / total_time.as_secs_f64());
    println!();
    println!("Type Discovery:");
    println!("  ├─ Types: {}", n_types);
    println!("  ├─ Entropy: {:.3} bits", type_entropy);
    println!("  ├─ Intra-sim: {:.4}", intra_sim);
    println!("  ├─ Inter-dist: {:.4}", inter_dist);
    if let Some(s) = separation {
        if s.is_finite() {
            println!("  └─ Separation: {:.2}x", s);
        } else {
            println!("  └─ Separation: infx");
        }
    }
    println!();
    println!("Classification: 10-NN @ {:.1}%", knn_accuracy * 100.0);
    println!();

    // Competence assessment
    let competence = if knn_accuracy >= 0.90 {
        "EXCELLENT"
    } else if knn_accuracy >= 0.80 {
        "VERY_GOOD"
    } else if knn_accuracy >= 0.70 {
        "GOOD"
    } else if knn_accuracy >= 0.60 {
        "FAIR"
    } else {
        "NEEDS_IMPROVEMENT"
    };
    println!("Competence: {}", competence);
    println!();

    // Save results
    let results = AssessmentResults45D {
        dataset: "EarthSpeciesProject/BEANS-Zero".to_string(),
        total_samples,
        feature_dim: FEATURE_DIM,
        total_time_sec: total_time.as_secs_f64(),
        throughput_samples_per_sec: total_samples as f64 / total_time.as_secs_f64(),
        global_types: n_types,
        type_entropy,
        knn_accuracy,
        knn_best_k: 10,
        avg_intra_type_similarity: intra_sim,
        avg_inter_type_distance: inter_dist,
        separation_ratio: separation.filter(|s| s.is_finite()),
        source_datasets,
        task_types,
        new_feature_ranges,
    };

    std::fs::create_dir_all("beans_analysis")?;
    let output_path = "beans_analysis/beans_45d_assessment_results.json";
    let file = File::create(output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &results)?;
    println!("Results saved: {}", output_path);

    Ok(())
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn build_global_types_streaming(
    features: &[ExtractedFeatures],
    similarity_threshold: f64,
) -> Vec<AcousticType> {
    if features.is_empty() {
        return Vec::new();
    }

    println!("Building global types using acoustic similarity engine...");
    println!("  (Streaming approach - memory efficient, no O(n²) matrix)");

    let n = features.len();

    // Create feature matrix
    let matrix = {
        let mut m = Array2::<f64>::zeros((n, FEATURE_DIM));
        for (i, f) in features.iter().enumerate() {
            for (j, &val) in f.features.iter().enumerate().take(FEATURE_DIM) {
                m[[i, j]] = val;
            }
        }
        m
    };

    // Create similarity engine
    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    engine.fit_normalization(&matrix);

    // Streaming type assignment
    let mut types: Vec<AcousticType> = Vec::new();
    let max_types_to_check = 1000;

    for i in 0..n {
        let sample = matrix.row(i).to_owned();

        let mut best_type: Option<usize> = None;
        let mut best_sim = 0.0;

        let types_to_check = types.len().min(max_types_to_check);
        for type_idx in 0..types_to_check {
            let centroid = ndarray::Array1::from_vec(types[type_idx].centroid.clone());
            let sim = engine.similarity(&sample, &centroid);

            if sim >= similarity_threshold && sim > best_sim {
                best_sim = sim;
                best_type = Some(type_idx);
            }
        }

        if let Some(type_idx) = best_type {
            let n_in_type = types[type_idx].count + 1;
            types[type_idx].count = n_in_type;
            types[type_idx].sample_ids.push(features[i].sample_id.clone());

            for (j, val) in features[i].features.iter().enumerate().take(FEATURE_DIM) {
                types[type_idx].centroid[j] +=
                    (val - types[type_idx].centroid[j]) / n_in_type as f64;
            }
        } else {
            types.push(AcousticType {
                type_id: format!("type_{}", types.len()),
                centroid: features[i].features.clone(),
                count: 1,
                sample_ids: vec![features[i].sample_id.clone()],
                avg_distance_to_centroid: 0.0,
            });
        }

        if (i + 1) % 10000 == 0 {
            println!("    {}/{} samples, {} types", i + 1, n, types.len());
        }
    }

    // Sort by count
    types.sort_by(|a, b| b.count.cmp(&a.count));

    // Compute avg distances for top types
    for t in types.iter_mut().take(100) {
        if t.count > 1 {
            let centroid = ndarray::Array1::from_vec(t.centroid.clone());
            let mut total_dist = 0.0;
            let mut count = 0;

            for sample_id in t.sample_ids.iter().take(50) {
                if let Some(f) = features.iter().find(|f| &f.sample_id == sample_id) {
                    let sample = ndarray::Array1::from_vec(f.features.clone());
                    total_dist += engine.distance(&centroid, &sample);
                    count += 1;
                }
            }

            if count > 0 {
                t.avg_distance_to_centroid = total_dist / count as f64;
            }
        }
    }

    println!("  Discovered {} types", types.len());
    types
}

fn compute_similarity_stats(features: &[ExtractedFeatures], types: &[AcousticType]) -> (f64, f64) {
    if features.is_empty() || types.is_empty() {
        return (0.0, 0.0);
    }

    // Create feature matrix
    let matrix = {
        let mut m = Array2::<f64>::zeros((features.len(), FEATURE_DIM));
        for (i, f) in features.iter().enumerate() {
            for (j, &val) in f.features.iter().enumerate().take(FEATURE_DIM) {
                m[[i, j]] = val;
            }
        }
        m
    };

    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    engine.fit_normalization(&matrix);

    // Compute intra-type similarity
    let mut intra_sim_sum = 0.0;
    let mut intra_count = 0;

    for t in types.iter().take(100) {
        if t.count > 1 {
            let centroid = ndarray::Array1::from_vec(t.centroid.clone());
            for sample_id in t.sample_ids.iter().take(10) {
                if let Some(f) = features.iter().find(|f| &f.sample_id == sample_id) {
                    let sample = ndarray::Array1::from_vec(f.features.clone());
                    intra_sim_sum += engine.similarity(&centroid, &sample);
                    intra_count += 1;
                }
            }
        }
    }

    let avg_intra_sim = if intra_count > 0 { intra_sim_sum / intra_count as f64 } else { 0.0 };

    // Compute inter-type distance
    let mut inter_dist_sum = 0.0;
    let mut inter_count = 0;

    for i in 0..types.len().min(100) {
        for j in (i + 1)..types.len().min(100) {
            let c1 = ndarray::Array1::from_vec(types[i].centroid.clone());
            let c2 = ndarray::Array1::from_vec(types[j].centroid.clone());
            inter_dist_sum += engine.distance(&c1, &c2);
            inter_count += 1;
        }
    }

    let avg_inter_dist = if inter_count > 0 { inter_dist_sum / inter_count as f64 } else { 0.0 };

    (avg_intra_sim, avg_inter_dist)
}

fn compute_knn_accuracy(features: &[ExtractedFeatures], k: usize) -> f64 {
    if features.len() < 100 {
        return 0.0;
    }

    // Create feature matrix
    let matrix = {
        let mut m = Array2::<f64>::zeros((features.len(), FEATURE_DIM));
        for (i, f) in features.iter().enumerate() {
            for (j, &val) in f.features.iter().enumerate().take(FEATURE_DIM) {
                m[[i, j]] = val;
            }
        }
        m
    };

    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    engine.fit_normalization(&matrix);

    // Simple leave-one-out k-NN on a subset
    let test_size = features.len().min(5000);
    let correct = Arc::new(AtomicUsize::new(0));

    (0..test_size).into_par_iter().for_each(|i| {
        let query = matrix.row(i).to_owned();

        // Find k nearest neighbors (excluding self)
        let mut distances: Vec<(usize, f64)> = (0..features.len())
            .filter(|&j| j != i)
            .map(|j| {
                let sample = matrix.row(j).to_owned();
                (j, engine.distance(&query, &sample))
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Vote among top k
        let neighbors: Vec<&ExtractedFeatures> = distances.iter()
            .take(k)
            .filter_map(|(idx, _)| features.get(*idx))
            .collect();

        // Check if most common dataset matches
        let mut dataset_counts: HashMap<&str, usize> = HashMap::new();
        for n in &neighbors {
            if let Some(ref d) = n.dataset {
                *dataset_counts.entry(d.as_str()).or_insert(0) += 1;
            }
        }

        let predicted = dataset_counts.iter()
            .max_by_key(|(_, &c)| c)
            .map(|(d, _)| *d);

        let actual = features[i].dataset.as_deref();

        if predicted == actual {
            correct.fetch_add(1, Ordering::Relaxed);
        }
    });

    let correct_count = correct.load(Ordering::Relaxed);
    correct_count as f64 / test_size as f64
}

fn load_audio_raw(path: &str, expected_samples: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    // Load raw f32 audio file
    let bytes = std::fs::read(path)?;

    // Convert bytes to f32 samples
    let audio: Vec<f64> = bytes.chunks_exact(4)
        .take(expected_samples)
        .map(|chunk| {
            let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            val as f64
        })
        .collect();

    Ok(audio)
}

fn load_audio_from_npy(path: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    // Check if .npy exists
    let npy_path = path.replace(".flac", ".npy").replace(".wav", ".npy");

    if Path::new(&npy_path).exists() {
        // Load from numpy file
        let bytes = std::fs::read(&npy_path)?;

        // Simple NPY parser for 1D float32 arrays
        if bytes.len() < 10 {
            return Err("NPY file too small".into());
        }

        // Check magic number
        if &bytes[0..6] != b"\x93NUMPY" {
            return Err("Invalid NPY file".into());
        }

        // Get header length (little-endian)
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let data_start = 10 + header_len;

        // Data is float32 little-endian
        let data = &bytes[data_start..];
        let audio: Vec<f64> = data.chunks_exact(4)
            .map(|chunk| {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                val as f64
            })
            .collect();

        return Ok(audio);
    }

    // Fallback: try loading audio directly with symphonia
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if path.ends_with(".flac") {
        hint.with_extension("flac");
    } else if path.ends_with(".wav") {
        hint.with_extension("wav");
    }

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;
    let mut format = probed.format;

    let track = format.default_track().ok_or("No default track")?;
    let track_id = track.id;

    let decoder_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;

    let mut audio_samples: Vec<f64> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let spec = *decoded.spec();
        let duration = decoded.capacity() as u64;
        let mut sample_buf = SampleBuffer::<f32>::new(duration, spec);
        sample_buf.copy_interleaved_ref(decoded);

        for sample in sample_buf.samples() {
            audio_samples.push(*sample as f64);
        }
    }

    Ok(audio_samples)
}
