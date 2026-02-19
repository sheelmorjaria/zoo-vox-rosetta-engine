//! BEANS-Zero 45D Acoustic Similarity Benchmark
//!
//! Comprehensive benchmark testing the 45D acoustic similarity engine
//! on the full BEANS-Zero dataset (91,965 samples).
//!
//! Features tested:
//! - 45D feature extraction (30D base + 15D new dimensions)
//! - Acoustic Similarity Engine with multiple distance metrics
//! - Type discovery via streaming similarity clustering
//! - k-NN classification accuracy
//! - Within-type vs between-type separation analysis
//!
//! New 15D features:
//! - Resonance Factors (6): Formants 1-3, Bandwidths 1-2, Dispersion
//! - Spectral Shape Factors (4): Centroid, Spread, Skewness, Kurtosis
//! - Modulation Factors (3): Tilt, FM Slope, AM Depth
//! - Non-Linear Factors (2): Subharmonic Ratio, Spectral Entropy

use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use technical_architecture::{AcousticSimilarityEngine, SimilarityMetric, ZooVoxFeatureExtractor};

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
    id: Option<String>,
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
struct ExtractedSample {
    sample_id: String,
    features: Vec<f64>,
    source_dataset: String,
    task: String,
    duration_ms: f64,
}

#[derive(Debug, Clone)]
struct AcousticCluster {
    cluster_id: usize,
    centroid: Vec<f64>,
    count: usize,
    sample_ids: Vec<String>,
    source_datasets: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResults {
    // Dataset info
    dataset: String,
    total_samples: usize,
    valid_samples: usize,
    feature_dim: usize,

    // Performance metrics
    extraction_time_sec: f64,
    analysis_time_sec: f64,
    total_time_sec: f64,
    extraction_throughput: f64,
    analysis_throughput: f64,

    // Clustering metrics
    n_clusters: usize,
    cluster_entropy: f64,
    largest_cluster_size: usize,
    smallest_cluster_size: usize,
    avg_cluster_size: f64,

    // Similarity metrics
    avg_intra_cluster_similarity: f64,
    avg_inter_cluster_distance: f64,
    separation_ratio: f64,
    silhouette_approximation: f64,

    // k-NN metrics
    knn_accuracy_5: f64,
    knn_accuracy_10: f64,
    knn_accuracy_15: f64,

    // Feature statistics
    feature_ranges: HashMap<String, (f64, f64)>,
    feature_means: HashMap<String, f64>,
    feature_stds: HashMap<String, f64>,

    // Dataset distribution
    source_dataset_distribution: HashMap<String, usize>,
    task_distribution: HashMap<String, usize>,

    // 45D specific: contribution of new features
    new_features_variance_explained: f64,
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║     BEANS-Zero 45D Acoustic Similarity Benchmark (Full Dataset)              ║");
    println!("║                     91,965 Samples | 45D Features                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();

    // Load manifest
    let manifest_path = "beans_zero_cache/beans_audio_manifest.json";
    println!("[1/6] Loading manifest from: {}", manifest_path);

    let file = File::open(manifest_path)?;
    let reader = BufReader::new(file);
    let manifest: Manifest = serde_json::from_reader(reader)?;

    let total_samples = manifest.samples.len();
    println!("      Loaded {} samples", total_samples);
    println!();

    // Display configuration
    println!("Configuration:");
    println!("  ├─ Dataset: EarthSpeciesProject/BEANS-Zero");
    println!("  ├─ Total Samples: {}", total_samples);
    println!(
        "  ├─ Feature Dimension: {}D (30D base + 15D new)",
        FEATURE_DIM
    );
    println!("  ├─ Similarity Threshold: 0.85");
    println!("  ├─ Distance Metric: Cosine");
    println!("  ├─ k-NN Values: [5, 10, 15]");
    println!("  └─ Parallel: Rayon (all cores)");
    println!();

    // ========================================================================
    // Phase 2: Parallel Feature Extraction (45D)
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[2/6] Phase 1: Parallel 45D Feature Extraction");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let extraction_start = Instant::now();
    let processed = Arc::new(AtomicUsize::new(0));

    let extracted_results: Vec<Option<ExtractedSample>> = manifest
        .samples
        .par_iter()
        .enumerate()
        .map(|(idx, entry)| {
            // Progress update
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 10000 == 0 {
                println!(
                    "      Progress: {}/{} samples ({:.1}%)",
                    count + 1,
                    total_samples,
                    (count + 1) as f64 / total_samples as f64 * 100.0
                );
            }

            // Load audio from raw file
            let audio_path = format!("beans_zero_cache/{}", entry.audio_file);
            let audio = match load_audio_raw(&audio_path, entry.n_samples) {
                Ok(a) => a,
                Err(_) => return None,
            };

            // Skip very short audio
            if audio.len() < 100 {
                return None;
            }

            // Extract 45D features
            let mut extractor = ZooVoxFeatureExtractor::new(entry.sample_rate);
            match extractor.extract_45d(&audio) {
                Ok(features) => Some(ExtractedSample {
                    sample_id: entry
                        .id
                        .clone()
                        .unwrap_or_else(|| format!("sample_{}", idx)),
                    features: features.to_vector().to_vec(),
                    source_dataset: entry.labels.source_dataset.clone(),
                    task: entry.labels.task.clone(),
                    duration_ms: entry.duration_ms,
                }),
                Err(_) => None,
            }
        })
        .collect();

    let extraction_time = extraction_start.elapsed();

    // Filter valid samples
    let valid_samples: Vec<_> = extracted_results.into_iter().filter_map(|s| s).collect();

    let n_valid = valid_samples.len();
    let n_failed = total_samples - n_valid;

    println!();
    println!("Extraction Complete:");
    println!("  ├─ Valid Samples: {}", n_valid);
    println!(
        "  ├─ Failed Samples: {} ({:.1}%)",
        n_failed,
        n_failed as f64 / total_samples as f64 * 100.0
    );
    println!("  ├─ Time: {:.2}s", extraction_time.as_secs_f64());
    println!(
        "  └─ Throughput: {:.1} samples/sec",
        n_valid as f64 / extraction_time.as_secs_f64()
    );
    println!();

    // ========================================================================
    // Phase 3: Feature Statistics
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[3/6] Phase 2: Feature Statistics");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let feature_names_45d = [
        // Base 30D features
        "mean_f0_hz",
        "duration_ms",
        "f0_range_hz",
        "harmonic_to_noise_ratio",
        "spectral_flatness",
        "harmonicity",
        "attack_time_ms",
        "decay_time_ms",
        "sustain_level",
        "vibrato_rate_hz",
        "vibrato_depth",
        "jitter",
        "shimmer",
        "mfcc_1",
        "mfcc_2",
        "mfcc_3",
        "mfcc_4",
        "mfcc_5",
        "mfcc_6",
        "mfcc_7",
        "mfcc_8",
        "mfcc_9",
        "mfcc_10",
        "mfcc_11",
        "mfcc_12",
        "mfcc_13",
        "spectral_flux",
        "median_ici_ms",
        "onset_rate_hz",
        "ici_cv",
        // New 15D features
        "formant_1_hz",
        "formant_2_hz",
        "formant_3_hz",
        "formant_1_bandwidth",
        "formant_2_bandwidth",
        "formant_dispersion",
        "spectral_centroid",
        "spectral_spread",
        "spectral_skewness",
        "spectral_kurtosis",
        "spectral_tilt",
        "fm_slope_hz_per_sec",
        "am_depth",
        "subharmonic_ratio",
        "spectral_entropy",
    ];

    let mut feature_mins = vec![f64::MAX; FEATURE_DIM];
    let mut feature_maxs = vec![f64::MIN; FEATURE_DIM];
    let mut feature_sums = vec![0.0; FEATURE_DIM];
    let mut feature_sq_sums = vec![0.0; FEATURE_DIM];

    for sample in &valid_samples {
        for (j, &val) in sample.features.iter().enumerate() {
            if val < feature_mins[j] {
                feature_mins[j] = val;
            }
            if val > feature_maxs[j] {
                feature_maxs[j] = val;
            }
            feature_sums[j] += val;
            feature_sq_sums[j] += val * val;
        }
    }

    let mut feature_ranges = HashMap::new();
    let mut feature_means = HashMap::new();
    let mut feature_stds = HashMap::new();

    for (i, name) in feature_names_45d.iter().enumerate() {
        let min = if feature_mins[i] == f64::MAX {
            0.0
        } else {
            feature_mins[i]
        };
        let max = if feature_maxs[i] == f64::MIN {
            0.0
        } else {
            feature_maxs[i]
        };
        let mean = feature_sums[i] / n_valid as f64;
        let variance = feature_sq_sums[i] / n_valid as f64 - mean * mean;
        let std = variance.sqrt().max(0.0);

        feature_ranges.insert(name.to_string(), (min, max));
        feature_means.insert(name.to_string(), mean);
        feature_stds.insert(name.to_string(), std);
    }

    // Compute variance explained by new 15D features vs base 30D
    let base_30d_variance: f64 = (0..30)
        .map(|i| feature_stds[feature_names_45d[i]].powi(2))
        .sum();
    let new_15d_variance: f64 = (30..45)
        .map(|i| feature_stds[feature_names_45d[i]].powi(2))
        .sum();
    let total_variance = base_30d_variance + new_15d_variance;
    let new_features_variance_explained = if total_variance > 0.0 {
        new_15d_variance / total_variance
    } else {
        0.0
    };

    println!("Feature Statistics (selected):");
    println!("  Base 30D Variance: {:.2}", base_30d_variance);
    println!("  New 15D Variance: {:.2}", new_15d_variance);
    println!(
        "  New Features Variance Explained: {:.1}%",
        new_features_variance_explained * 100.0
    );
    println!();

    println!("New 15D Feature Ranges:");
    for i in 30..45 {
        let name = feature_names_45d[i];
        let (min, max) = feature_ranges[name];
        let mean = feature_means[name];
        let std = feature_stds[name];
        println!(
            "  ├─ {}: [{:.2}, {:.2}] μ={:.2} σ={:.2}",
            name, min, max, mean, std
        );
    }
    println!();

    // ========================================================================
    // Phase 4: Acoustic Similarity Clustering
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[4/6] Phase 3: Acoustic Similarity Clustering (Streaming)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let analysis_start = Instant::now();

    // Build clusters using streaming similarity
    let clusters = build_clusters_streaming(&valid_samples, 0.85);
    let analysis_time = analysis_start.elapsed();

    let n_clusters = clusters.len();
    println!();
    println!("Clustering Complete:");
    println!("  ├─ Clusters Found: {}", n_clusters);
    println!("  ├─ Time: {:.2}s", analysis_time.as_secs_f64());
    println!(
        "  └─ Throughput: {:.1} samples/sec",
        n_valid as f64 / analysis_time.as_secs_f64()
    );
    println!();

    // Compute cluster statistics
    let cluster_sizes: Vec<usize> = clusters.iter().map(|c| c.count).collect();
    let largest_cluster = cluster_sizes.iter().max().copied().unwrap_or(0);
    let smallest_cluster = cluster_sizes.iter().min().copied().unwrap_or(0);
    let avg_cluster_size =
        cluster_sizes.iter().sum::<usize>() as f64 / cluster_sizes.len().max(1) as f64;

    // Compute cluster entropy
    let cluster_entropy: f64 = clusters
        .iter()
        .map(|c| {
            let p = c.count as f64 / n_valid as f64;
            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum();

    println!("Cluster Statistics:");
    println!("  ├─ Largest Cluster: {} samples", largest_cluster);
    println!("  ├─ Smallest Cluster: {} samples", smallest_cluster);
    println!("  ├─ Avg Cluster Size: {:.1} samples", avg_cluster_size);
    println!("  └─ Cluster Entropy: {:.3} bits", cluster_entropy);
    println!();

    // ========================================================================
    // Phase 5: Similarity Analysis
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[5/6] Phase 4: Within/Between Cluster Similarity Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let (intra_sim, inter_dist, silhouette) = compute_similarity_metrics(&valid_samples, &clusters);

    let separation_ratio = if intra_sim > 0.001 {
        inter_dist / (1.0 - intra_sim)
    } else {
        f64::INFINITY
    };

    println!("Similarity Analysis:");
    println!("  ├─ Avg Intra-Cluster Similarity: {:.4}", intra_sim);
    println!("  ├─ Avg Inter-Cluster Distance: {:.4}", inter_dist);
    println!("  ├─ Separation Ratio: {:.2}x", separation_ratio);
    println!("  └─ Silhouette Approximation: {:.4}", silhouette);
    println!();

    // ========================================================================
    // Phase 6: k-NN Classification
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[6/6] Phase 5: k-NN Classification Evaluation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let knn_5 = compute_knn_accuracy(&valid_samples, 5);
    let knn_10 = compute_knn_accuracy(&valid_samples, 10);
    let knn_15 = compute_knn_accuracy(&valid_samples, 15);

    println!("k-NN Classification Accuracy:");
    println!("  ├─ 5-NN:  {:.2}%", knn_5 * 100.0);
    println!("  ├─ 10-NN: {:.2}%", knn_10 * 100.0);
    println!("  └─ 15-NN: {:.2}%", knn_15 * 100.0);
    println!();

    // ========================================================================
    // Phase 7: Dataset Distribution Analysis
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Dataset Distribution Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let mut source_dataset_distribution: HashMap<String, usize> = HashMap::new();
    let mut task_distribution: HashMap<String, usize> = HashMap::new();

    for sample in &valid_samples {
        *source_dataset_distribution
            .entry(sample.source_dataset.clone())
            .or_insert(0) += 1;
        *task_distribution.entry(sample.task.clone()).or_insert(0) += 1;
    }

    // Sort by count
    let mut datasets_sorted: Vec<_> = source_dataset_distribution.iter().collect();
    datasets_sorted.sort_by(|a, b| b.1.cmp(a.1));

    println!("Source Datasets (top 10):");
    for (name, count) in datasets_sorted.iter().take(10) {
        let pct = **count as f64 / n_valid as f64 * 100.0;
        println!("  ├─ {}: {} ({:.1}%)", name, count, pct);
    }
    println!();

    let mut tasks_sorted: Vec<_> = task_distribution.iter().collect();
    tasks_sorted.sort_by(|a, b| b.1.cmp(a.1));

    println!("Task Types:");
    for (name, count) in tasks_sorted {
        let pct = *count as f64 / n_valid as f64 * 100.0;
        println!("  ├─ {}: {} ({:.1}%)", name, count, pct);
    }
    println!();

    // ========================================================================
    // Final Summary
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("BENCHMARK SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let total_time = total_start.elapsed();

    println!("Dataset: EarthSpeciesProject/BEANS-Zero");
    println!("  ├─ Total Samples: {}", total_samples);
    println!("  └─ Valid Samples: {}", n_valid);
    println!();

    println!("Features: 45D (30D base + 15D new)");
    println!(
        "  ├─ New Features Variance Explained: {:.1}%",
        new_features_variance_explained * 100.0
    );
    println!(
        "  └─ Feature Extraction: {:.1} samples/sec",
        n_valid as f64 / extraction_time.as_secs_f64()
    );
    println!();

    println!("Performance:");
    println!(
        "  ├─ Total Time: {:.1}s ({:.1} min)",
        total_time.as_secs_f64(),
        total_time.as_secs_f64() / 60.0
    );
    println!(
        "  ├─ Extraction Time: {:.1}s",
        extraction_time.as_secs_f64()
    );
    println!("  └─ Analysis Time: {:.1}s", analysis_time.as_secs_f64());
    println!();

    println!("Clustering:");
    println!("  ├─ Clusters: {}", n_clusters);
    println!("  ├─ Entropy: {:.3} bits", cluster_entropy);
    println!("  └─ Avg Size: {:.1} samples", avg_cluster_size);
    println!();

    println!("Similarity Metrics:");
    println!("  ├─ Intra-Cluster Sim: {:.4}", intra_sim);
    println!("  ├─ Inter-Cluster Dist: {:.4}", inter_dist);
    println!("  ├─ Separation Ratio: {:.2}x", separation_ratio);
    println!("  └─ Silhouette: {:.4}", silhouette);
    println!();

    println!("k-NN Accuracy:");
    println!("  ├─ 5-NN:  {:.1}%", knn_5 * 100.0);
    println!("  ├─ 10-NN: {:.1}%", knn_10 * 100.0);
    println!("  └─ 15-NN: {:.1}%", knn_15 * 100.0);
    println!();

    // Assessment
    let avg_knn = (knn_5 + knn_10 + knn_15) / 3.0;
    let competence = if avg_knn >= 0.90 && separation_ratio > 2.0 {
        "EXCELLENT"
    } else if avg_knn >= 0.80 && separation_ratio > 1.5 {
        "VERY_GOOD"
    } else if avg_knn >= 0.70 && separation_ratio > 1.2 {
        "GOOD"
    } else if avg_knn >= 0.60 {
        "FAIR"
    } else {
        "NEEDS_IMPROVEMENT"
    };

    println!("Overall Assessment: {}", competence);
    println!();

    // Save results
    let results = BenchmarkResults {
        dataset: "EarthSpeciesProject/BEANS-Zero".to_string(),
        total_samples,
        valid_samples: n_valid,
        feature_dim: FEATURE_DIM,
        extraction_time_sec: extraction_time.as_secs_f64(),
        analysis_time_sec: analysis_time.as_secs_f64(),
        total_time_sec: total_time.as_secs_f64(),
        extraction_throughput: n_valid as f64 / extraction_time.as_secs_f64(),
        analysis_throughput: n_valid as f64 / analysis_time.as_secs_f64(),
        n_clusters,
        cluster_entropy,
        largest_cluster_size: largest_cluster,
        smallest_cluster_size: smallest_cluster,
        avg_cluster_size,
        avg_intra_cluster_similarity: intra_sim,
        avg_inter_cluster_distance: inter_dist,
        separation_ratio,
        silhouette_approximation: silhouette,
        knn_accuracy_5: knn_5,
        knn_accuracy_10: knn_10,
        knn_accuracy_15: knn_15,
        feature_ranges,
        feature_means,
        feature_stds,
        source_dataset_distribution,
        task_distribution,
        new_features_variance_explained,
    };

    std::fs::create_dir_all("beans_analysis")?;
    let output_path = "beans_analysis/beans_45d_acoustic_similarity_benchmark.json";
    let file = File::create(output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &results)?;
    println!("Results saved to: {}", output_path);

    Ok(())
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn load_audio_raw(
    path: &str,
    expected_samples: usize,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
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

fn build_clusters_streaming(
    samples: &[ExtractedSample],
    similarity_threshold: f64,
) -> Vec<AcousticCluster> {
    if samples.is_empty() {
        return Vec::new();
    }

    println!("Building acoustic similarity clusters (streaming approach)...");

    let n = samples.len();

    // Create feature matrix for fitting
    let _features_matrix: Vec<Vec<f64>> = samples.iter().map(|s| s.features.clone()).collect();

    // Create and fit similarity engine
    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    // Fit normalization
    {
        let mut matrix = ndarray::Array2::<f64>::zeros((n.min(10000), FEATURE_DIM));
        for (i, sample) in samples.iter().take(10000).enumerate() {
            for (j, &val) in sample.features.iter().enumerate() {
                matrix[[i, j]] = val;
            }
        }
        engine.fit_normalization(&matrix);
    }

    let mut clusters: Vec<AcousticCluster> = Vec::new();
    let max_clusters_to_check = 2000;

    for (i, sample) in samples.iter().enumerate() {
        let query = Array1::from_vec(sample.features.clone());

        let mut best_cluster: Option<usize> = None;
        let mut best_sim = 0.0;

        let clusters_to_check = clusters.len().min(max_clusters_to_check);
        for cluster_idx in 0..clusters_to_check {
            let centroid = Array1::from_vec(clusters[cluster_idx].centroid.clone());
            let sim = engine.similarity(&query, &centroid);

            if sim >= similarity_threshold && sim > best_sim {
                best_sim = sim;
                best_cluster = Some(cluster_idx);
            }
        }

        if let Some(cluster_idx) = best_cluster {
            // Add to existing cluster (update centroid incrementally)
            let n_in_cluster = clusters[cluster_idx].count + 1;
            clusters[cluster_idx].count = n_in_cluster;
            clusters[cluster_idx]
                .sample_ids
                .push(sample.sample_id.clone());
            *clusters[cluster_idx]
                .source_datasets
                .entry(sample.source_dataset.clone())
                .or_insert(0) += 1;

            // Incremental centroid update
            for (j, &val) in sample.features.iter().enumerate() {
                clusters[cluster_idx].centroid[j] +=
                    (val - clusters[cluster_idx].centroid[j]) / n_in_cluster as f64;
            }
        } else {
            // Create new cluster
            let mut source_datasets = HashMap::new();
            source_datasets.insert(sample.source_dataset.clone(), 1);

            clusters.push(AcousticCluster {
                cluster_id: clusters.len(),
                centroid: sample.features.clone(),
                count: 1,
                sample_ids: vec![sample.sample_id.clone()],
                source_datasets,
            });
        }

        if (i + 1) % 20000 == 0 {
            println!(
                "      Progress: {}/{} samples, {} clusters",
                i + 1,
                n,
                clusters.len()
            );
        }
    }

    // Sort by count
    clusters.sort_by(|a, b| b.count.cmp(&a.count));

    println!(
        "      Discovered {} clusters from {} samples",
        clusters.len(),
        n
    );
    clusters
}

fn compute_similarity_metrics(
    samples: &[ExtractedSample],
    clusters: &[AcousticCluster],
) -> (f64, f64, f64) {
    if samples.is_empty() || clusters.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let n = samples.len();

    // Create and fit similarity engine
    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    {
        let mut matrix = ndarray::Array2::<f64>::zeros((n.min(10000), FEATURE_DIM));
        for (i, sample) in samples.iter().take(10000).enumerate() {
            for (j, &val) in sample.features.iter().enumerate() {
                matrix[[i, j]] = val;
            }
        }
        engine.fit_normalization(&matrix);
    }

    // Compute intra-cluster similarity (sample to cluster centroid)
    let mut intra_sim_sum = 0.0;
    let mut intra_count = 0;

    for cluster in clusters.iter().take(100) {
        if cluster.count > 1 {
            let centroid = Array1::from_vec(cluster.centroid.clone());

            for sample_id in cluster.sample_ids.iter().take(20) {
                if let Some(sample) = samples.iter().find(|s| &s.sample_id == sample_id) {
                    let query = Array1::from_vec(sample.features.clone());
                    intra_sim_sum += engine.similarity(&query, &centroid);
                    intra_count += 1;
                }
            }
        }
    }

    let avg_intra_sim = if intra_count > 0 {
        intra_sim_sum / intra_count as f64
    } else {
        0.0
    };

    // Compute inter-cluster distance (between centroids)
    let mut inter_dist_sum = 0.0;
    let mut inter_count = 0;

    for i in 0..clusters.len().min(100) {
        for j in (i + 1)..clusters.len().min(100) {
            let c1 = Array1::from_vec(clusters[i].centroid.clone());
            let c2 = Array1::from_vec(clusters[j].centroid.clone());
            inter_dist_sum += engine.distance(&c1, &c2);
            inter_count += 1;
        }
    }

    let avg_inter_dist = if inter_count > 0 {
        inter_dist_sum / inter_count as f64
    } else {
        0.0
    };

    // Silhouette approximation: (inter - intra) / max(inter, intra)
    let silhouette = if avg_inter_dist > 0.0 || avg_intra_sim > 0.0 {
        let intra_dist = 1.0 - avg_intra_sim; // Convert similarity to distance
        (avg_inter_dist - intra_dist) / avg_inter_dist.max(intra_dist).max(1e-10)
    } else {
        0.0
    };

    (avg_intra_sim, avg_inter_dist, silhouette)
}

fn compute_knn_accuracy(samples: &[ExtractedSample], k: usize) -> f64 {
    if samples.len() < 100 {
        return 0.0;
    }

    let n = samples.len();

    // Create and fit similarity engine
    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    {
        let mut matrix = ndarray::Array2::<f64>::zeros((n, FEATURE_DIM));
        for (i, sample) in samples.iter().enumerate() {
            for (j, &val) in sample.features.iter().enumerate() {
                matrix[[i, j]] = val;
            }
        }
        engine.fit_normalization(&matrix);
    }

    // Test on subset
    let test_size = samples.len().min(5000);
    let correct = Arc::new(AtomicUsize::new(0));

    (0..test_size).into_par_iter().for_each(|i| {
        let query = Array1::from_vec(samples[i].features.clone());

        // Find k nearest neighbors (excluding self)
        let mut distances: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let candidate = Array1::from_vec(samples[j].features.clone());
                (j, engine.distance(&query, &candidate))
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Vote among top k
        let mut dataset_counts: HashMap<&str, usize> = HashMap::new();
        for (idx, _) in distances.iter().take(k) {
            *dataset_counts
                .entry(&samples[*idx].source_dataset)
                .or_insert(0) += 1;
        }

        let predicted = dataset_counts
            .iter()
            .max_by_key(|(_, &c)| c)
            .map(|(d, _)| *d);

        let actual = samples[i].source_dataset.as_str();

        if predicted == Some(actual) {
            correct.fetch_add(1, Ordering::Relaxed);
        }
    });

    let correct_count = correct.load(Ordering::Relaxed);
    correct_count as f64 / test_size as f64
}
