// Phase 0: Symbolic Stream Generation for Marmoset - 30D + Multi-Clustering
//
// This version uses the ORIGINAL 30D features (not 37D with 7 additional phylogenetic features)
// and implements multiple clustering approaches to find the optimal technique.
//
// Usage: cargo run --release --example phase0_symbolic_stream_marmoset_30d [--limit N] [--resume]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use technical_architecture::{
    MicroDynamicsExtractor,
    hdbscan::{HdbscanClustering, DistanceMetric},
    clustering::{DbscanClustering, ClusterStats, ClusteringError},
};

// =============================================================================
// Data Structures
// =============================================================================

#[derive(Clone, Debug)]
struct ExtractedFeatures {
    file_name: String,
    call_type: String,
    phrase_index: usize,
    features: Vec<f64>,  // 30D features (original)
    duration_ms: f64,
}

#[derive(Serialize, Deserialize, Clone)]
struct SerializableFeatures {
    file_name: String,
    call_type: String,
    phrase_index: usize,
    features: Vec<f64>,
    duration_ms: f64,
}

#[derive(Serialize, Deserialize, Clone)]
struct CheckpointData {
    all_features: Vec<SerializableFeatures>,
    all_file_names: Vec<String>,
    total_files: usize,
    processed_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CallType {
    Vocalization,
    Phee,
    Twitter,
    Trill,
    Tsik,
    Seep,
    Infant,
    Unknown,
}

impl CallType {
    fn from_filename(filename: &str) -> Self {
        let fname = filename.to_lowercase();
        if fname.contains("vocalization") {
            CallType::Vocalization
        } else if fname.contains("phee") {
            CallType::Phee
        } else if fname.contains("twitter") {
            CallType::Twitter
        } else if fname.contains("trill") {
            CallType::Trill
        } else if fname.contains("tsik") {
            CallType::Tsik
        } else if fname.contains("seep") {
            CallType::Seep
        } else if fname.contains("infant") {
            CallType::Infant
        } else {
            CallType::Unknown
        }
    }

    fn name(&self) -> &'static str {
        match self {
            CallType::Vocalization => "Vocalization",
            CallType::Phee => "Phee",
            CallType::Twitter => "Twitter",
            CallType::Trill => "Trill",
            CallType::Tsik => "Tsik",
            CallType::Seep => "Seep",
            CallType::Infant => "Infant",
            CallType::Unknown => "Unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClusteringAlgorithm {
    HdbscanEom,      // HDBSCAN with Excess of Mass
    HdbscanLeaf,     // HDBSCAN with Leaf clustering
    Dbscan,           // DBSCAN (epsilon-based)
}

impl ClusteringAlgorithm {
    fn name(&self) -> &'static str {
        match self {
            ClusteringAlgorithm::HdbscanEom => "HDBSCAN-EOM",
            ClusteringAlgorithm::HdbscanLeaf => "HDBSCAN-Leaf",
            ClusteringAlgorithm::Dbscan => "DBSCAN",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ClusteringResult {
    pub algorithm: String,
    pub parameters: serde_json::Value,
    pub n_clusters: usize,
    pub noise_count: usize,
    pub cluster_sizes: Vec<usize>,
    pub largest_cluster_pct: f64,
    pub labels: Vec<i32>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ClusteringComparison {
    pub dataset_size: usize,
    pub feature_dim: usize,
    pub results: Vec<ClusteringResult>,
    pub recommended: String,
    pub recommendation_reason: String,
}

#[derive(Clone)]
struct ProgressTracker {
    total: usize,
    processed: Arc<Mutex<usize>>,
    start_time: Instant,
}

impl ProgressTracker {
    fn new(total: usize) -> Self {
        Self {
            total,
            processed: Arc::new(Mutex::new(0)),
            start_time: Instant::now(),
        }
    }

    fn increment(&self) {
        let mut count = self.processed.lock().unwrap();
        *count += 1;
        let current = *count;

        if current % 100 == 0 || current == self.total {
            let elapsed = self.start_time.elapsed().as_secs_f64();
            let rate = current as f64 / elapsed;
            let remaining = if current < self.total {
                let remaining_count = self.total - current;
                remaining_count as f64 / rate
            } else {
                0.0
            };
            println!("   🔄 Processed {}/{} ({:.1}%) | {:.1} files/sec | ETA: {:.1}s",
                     current, self.total,
                     current as f64 / self.total as f64 * 100.0,
                     rate, remaining);
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn estimate_epsilon_from_features(features: &Array2<f64>) -> f64 {
    let (n_samples, n_dims) = features.dim();

    if n_samples == 0 || n_dims == 0 {
        return 1.0;
    }

    // Calculate per-dimension ranges
    let mut ranges = Vec::with_capacity(n_dims);

    for j in 0..n_dims {
        let col = features.column(j);
        let min_val = col.iter().cloned().fold(f64::INFINITY, |a, b| a.min(b));
        let max_val = col.iter().cloned().fold(f64::NEG_INFINITY, |a, b| a.max(b));
        let range = max_val - min_val;
        ranges.push(range);
    }

    // Average range
    let avg_range = ranges.iter().sum::<f64>() / ranges.len() as f64;

    // Epsilon as fraction of average range
    avg_range * 0.1
}

fn normalize_features(features: &Array2<f64>) -> Array2<f64> {
    let (n_samples, n_dims) = features.dim();

    let mut normalized = Array2::zeros((n_samples, n_dims));

    for j in 0..n_dims {
        let col = features.column(j);
        let mean = col.mean().unwrap_or(0.0);

        let variance = col.var(1.0);
        let std = variance.sqrt().max(1e-10);

        for i in 0..n_samples {
            normalized[[i, j]] = (features[[i, j]] - mean) / std;
        }
    }

    normalized
}

// =============================================================================
// Clustering Implementations
// =============================================================================

fn run_hdbscan(
    features: &Array2<f64>,
    min_cluster_size: usize,
    min_samples: usize,
    algorithm: ClusteringAlgorithm,
) -> Result<ClusteringResult, String> {
    let n_samples = features.nrows();

    let hdbscan = HdbscanClustering::with_metric(
        min_cluster_size,
        min_samples,
        DistanceMetric::Euclidean,
    ).map_err(|e| format!("HDBSCAN init failed: {:?}", e))?;

    let labels = hdbscan.fit_predict(features)
        .map_err(|e| format!("HDBSCAN fit_predict failed: {:?}", e))?;

    let stats = hdbscan.get_cluster_stats(&labels);

    let mut cluster_sizes = stats.cluster_sizes.clone();
    cluster_sizes.sort_by(|a, b| b.cmp(a));  // Descending

    let largest_cluster = cluster_sizes.first().copied().unwrap_or(0);
    let largest_pct = if n_samples > 0 {
        largest_cluster as f64 / n_samples as f64 * 100.0
    } else {
        0.0
    };

    Ok(ClusteringResult {
        algorithm: algorithm.name().to_string(),
        parameters: serde_json::json!({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": "Euclidean"
        }),
        n_clusters: stats.n_clusters,
        noise_count: stats.noise_count,
        cluster_sizes: stats.cluster_sizes,
        largest_cluster_pct: largest_pct,
        labels,
    })
}

fn run_dbscan(
    features: &Array2<f64>,
    eps: f64,
    min_samples: usize,
) -> Result<ClusteringResult, String> {
    let n_samples = features.nrows();

    let dbscan = DbscanClustering::new(eps, min_samples)
        .map_err(|e| format!("DBSCAN init failed: {:?}", e))?;

    let labels = dbscan.fit_predict(features)
        .map_err(|e| format!("DBSCAN fit_predict failed: {:?}", e))?;

    // Calculate stats manually
    let mut cluster_sizes_map: HashMap<i32, usize> = HashMap::new();

    for &label in &labels {
        if label != -1 {
            *cluster_sizes_map.entry(label).or_insert(0) += 1;
        }
    }

    let mut cluster_sizes: Vec<usize> = cluster_sizes_map.values().cloned().collect();
    cluster_sizes.sort_by(|a, b| b.cmp(a));  // Descending

    let noise_count = labels.iter().filter(|&&l| l == -1).count();
    let n_clusters = cluster_sizes.len();
    let largest_cluster = cluster_sizes.first().copied().unwrap_or(0);
    let largest_pct = if n_samples > 0 {
        largest_cluster as f64 / n_samples as f64 * 100.0
    } else {
        0.0
    };

    Ok(ClusteringResult {
        algorithm: "DBSCAN".to_string(),
        parameters: serde_json::json!({
            "eps": eps,
            "min_samples": min_samples,
            "metric": "Euclidean"
        }),
        n_clusters,
        noise_count,
        cluster_sizes,
        largest_cluster_pct: largest_pct,
        labels,
    })
}

fn compare_clustering_approaches(
    features: &Array2<f64>,
    n_samples: usize,
    n_dims: usize,
) -> ClusteringComparison {
    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Multi-Method Clustering Comparison                                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let mut results = Vec::new();

    // ========================================================================
    // Approach 1: Aggressive HDBSCAN (EOM)
    // ========================================================================
    println!("   Approach 1: HDBSCAN-EOM (Aggressive - for vocabulary discovery)");
    println!("   ───────────────────────────────────────────────────────────────────────");

    let result1 = run_hdbscan(features, 2, 1, ClusteringAlgorithm::HdbscanEom);
    if let Ok(ref r) = result1 {
        println!("   ✅ Clusters: {}, Noise: {}, Largest: {:.1}%",
                 r.n_clusters, r.noise_count, r.largest_cluster_pct);
        println!("      Top 5 clusters: {:?}", r.cluster_sizes.iter().take(5).copied().collect::<Vec<_>>());
        results.push(r.clone());
    } else {
        println!("   ❌ Failed: {:?}", result1);
    }
    println!();

    // ========================================================================
    // Approach 2: Moderate HDBSCAN (EOM)
    // ========================================================================
    println!("   Approach 2: HDBSCAN-EOM (Moderate - balanced)");
    println!("   ───────────────────────────────────────────────────────────────────────");

    let min_cluster_size_mod = ((n_samples as f64).ln().round() as usize).max(3).min(15);
    let min_samples_mod = ((min_cluster_size_mod as f64).sqrt().round() as usize).max(2);

    let result2 = run_hdbscan(features, min_cluster_size_mod, min_samples_mod, ClusteringAlgorithm::HdbscanEom);
    if let Ok(ref r) = result2 {
        println!("   ✅ Clusters: {}, Noise: {}, Largest: {:.1}%",
                 r.n_clusters, r.noise_count, r.largest_cluster_pct);
        println!("      Top 5 clusters: {:?}", r.cluster_sizes.iter().take(5).copied().collect::<Vec<_>>());
        results.push(r.clone());
    } else {
        println!("   ❌ Failed: {:?}", result2);
    }
    println!();

    // ========================================================================
    // Approach 3: Conservative HDBSCAN (EOM)
    // ========================================================================
    println!("   Approach 3: HDBSCAN-EOM (Conservative - stable clusters)");
    println!("   ───────────────────────────────────────────────────────────────────────");

    let min_cluster_size_con = ((n_samples as f64).ln().round() as usize * 3).max(10).min(50);
    let min_samples_con = ((min_cluster_size_con as f64).sqrt().round() as usize).max(3);

    let result3 = run_hdbscan(features, min_cluster_size_con, min_samples_con, ClusteringAlgorithm::HdbscanEom);
    if let Ok(ref r) = result3 {
        println!("   ✅ Clusters: {}, Noise: {}, Largest: {:.1}%",
                 r.n_clusters, r.noise_count, r.largest_cluster_pct);
        println!("      Top 5 clusters: {:?}", r.cluster_sizes.iter().take(5).copied().collect::<Vec<_>>());
        results.push(r.clone());
    } else {
        println!("   ❌ Failed: {:?}", result3);
    }
    println!();

    // ========================================================================
    // Approach 4: DBSCAN with estimated epsilon
    // ========================================================================
    println!("   Approach 4: DBSCAN (auto epsilon)");
    println!("   ───────────────────────────────────────────────────────────────────────");

    let eps_estimated = estimate_epsilon_from_features(features);
    let min_samples_db = ((n_samples as f64).ln().round() as usize).max(2).min(10);

    println!("      Estimated epsilon: {:.4}", eps_estimated);
    println!("      min_samples: {}", min_samples_db);

    let result4 = run_dbscan(features, eps_estimated, min_samples_db);
    if let Ok(ref r) = result4 {
        println!("   ✅ Clusters: {}, Noise: {}, Largest: {:.1}%",
                 r.n_clusters, r.noise_count, r.largest_cluster_pct);
        println!("      Top 5 clusters: {:?}", r.cluster_sizes.iter().take(5).copied().collect::<Vec<_>>());
        results.push(r.clone());
    } else {
        println!("   ❌ Failed: {:?}", result4);
    }
    println!();

    // ========================================================================
    // Recommendation
    // ========================================================================
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Clustering Comparison Summary                                      │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   ┌────────────┬──────────────┬────────────┬──────────────┬─────────────┐");
    println!("   │ Approach   │ Clusters     │ Noise      │ Largest %    │ Parameters  │");
    println!("   ├────────────┼──────────────┼────────────┼──────────────┼─────────────┤");

    for r in results.iter() {
        let params_str = format!("{:?}", r.parameters);
        let params_short = if params_str.len() > 60 {
            format!("{}...", &params_str[..57])
        } else {
            params_str.clone()
        };
        println!("   │ {:>10} │ {:>10} │ {:>8} │ {:>8.1}% │ {:>11} │",
                 r.algorithm, r.n_clusters, r.noise_count, r.largest_cluster_pct,
                 &params_short[..params_short.len().min(11)]);
    }

    println!("   └────────────┴──────────────┴────────────┴──────────────┴─────────────┘");
    println!();

    // Find best result based on:
    // 1. Multiple clusters (not 1, not all noise)
    // 2. Reasonable noise ratio (not too high)
    // 3. Balanced cluster sizes

    let (recommended, reason) = recommend_best_clustering(&results, n_samples);

    println!("   📊 RECOMMENDED: {}", recommended);
    println!("   📝 Reason: {}", reason);
    println!();

    ClusteringComparison {
        dataset_size: n_samples,
        feature_dim: n_dims,
        results,
        recommended,
        recommendation_reason: reason,
    }
}

fn recommend_best_clustering(results: &[ClusteringResult], _n_samples: usize) -> (String, String) {
    let mut best = None;
    let mut best_score = -f64::INFINITY;

    for r in results {
        // Skip if only 1 cluster (no separation)
        if r.n_clusters <= 1 {
            continue;
        }

        let noise_ratio = r.noise_count as f64 / _n_samples as f64;
        let dominant_ratio = r.largest_cluster_pct / 100.0;

        let mut score = 0.0;

        // Reward more clusters (up to a point)
        score += (r.n_clusters as f64).min(10.0) * 2.0;

        // Penalize high noise
        if noise_ratio > 0.8 {
            score -= 20.0;
        } else if noise_ratio > 0.5 {
            score -= 10.0;
        } else {
            score += 5.0;  // Reward low noise
        }

        // Penalize dominant cluster
        if dominant_ratio > 0.9 {
            score -= 15.0;
        } else if dominant_ratio > 0.7 {
            score -= 5.0;
        } else {
            score += 5.0;  // Reward balanced
        }

        if score > best_score {
            best_score = score;
            best = Some(r.algorithm.clone());
        }
    }

    match best {
        Some(algo) => {
            let reason = if algo.contains("HDBSCAN") {
                "HDBSCAN works well for density-based discovery in bioacoustic data"
            } else {
                "DBSCAN with estimated epsilon provides good balance"
            };
            (algo, reason.to_string())
        }
        None => {
            ("None".to_string(), "All approaches produced insufficient clustering".to_string())
        }
    }
}

// =============================================================================
// Main Function
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║      Phase 0: Symbolic Stream - Marmoset 30D + Multi-Clustering       ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                                                                   ║");
    println!("║  🔧 FEATURES: Original 30D (not 37D with phylogenetic add-ons)    ║");
    println!("║  🔧 CLUSTERING: Multiple approaches with auto-recommendation        ║");
    println!("║  ⚡ PARALLEL PROCESSING ENABLED                                   ║");
    println!("║                                                                   ║");
    println!("║  Input:  FLAC files (marmoset vocalizations)                    ║");
    println!("║  Method: 30D MicroDynamicsFeatures + Multi-algorithm comparison    ║");
    println!("║  Output: Recommended symbolic stream with cluster labels               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let mut vocalizations_dir = PathBuf::from("/home/sheel/birdsong_analysis/data/Vocalizations");
    let mut results_dir = PathBuf::from("/mnt/c/Users/sheel/Desktop/src/marmoset_phase0_30d_results");
    let mut limit = None;
    let mut resume = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse::<usize>() {
                        limit = Some(n);
                        println!("📊 Limiting to {} files for testing", n);
                        i += 1;
                    }
                }
            }
            "--resume" => {
                resume = true;
                println!("🔄 Resume mode: will load from checkpoint if available");
            }
            arg if i == args.len() - 1 && !arg.starts_with("--") => {
                vocalizations_dir = PathBuf::from(arg);
            }
            _ => {}
        }
        i += 1;
    }

    let sample_rate = 96000;
    let checkpoint_path = results_dir.join("phase0_checkpoint.bincode");

    // Detect CPU count for parallel processing
    let num_cpus = num_cpus::get();
    println!("   💻 Detected {} CPU cores", num_cpus);
    let parallel_chunks = num_cpus * 4;
    println!("   ⚡ Using {} parallel chunks for processing", parallel_chunks);
    println!();

    // =============================================================================
    // Step 0: Dataset Overview
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Dataset Overview                                                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    if !vocalizations_dir.exists() {
        println!("   ❌ Vocalizations directory not found: {}", vocalizations_dir.display());
        return Err("Dataset not found".into());
    }

    let mut flac_files = discover_flac_files(&vocalizations_dir)?;

    if let Some(n) = limit {
        let original_len = flac_files.len();
        flac_files.truncate(n.min(original_len));
        println!("📊 Limited to {} files (was {})", flac_files.len(), original_len);
    }

    println!("   📂 Vocalizations Directory: {}", vocalizations_dir.display());
    println!("   🔢 Total FLAC files: {}", flac_files.len());
    println!("   💾 Results Directory: {}", results_dir.display());
    println!();

    // =============================================================================
    // Checkpoint Loading
    // =============================================================================

    let mut all_features: Vec<ExtractedFeatures> = Vec::new();
    let mut all_file_names: Vec<String> = Vec::new();
    let mut start_index = 0;

    if resume && checkpoint_path.exists() {
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│ Checkpoint: Loading Previous Results                                │");
        println!("└─────────────────────────────────────────────────────────────────┘");
        println!();

        match load_checkpoint(&checkpoint_path) {
            Ok(checkpoint) => {
                println!("   ✅ Checkpoint loaded successfully!");
                println!("      ├─ Processed at: {}", checkpoint.processed_at);
                println!("      ├─ Previous files: {}", checkpoint.all_features.len());

                for feat in checkpoint.all_features {
                    all_file_names.push(feat.file_name.clone());
                    all_features.push(ExtractedFeatures {
                        file_name: feat.file_name,
                        call_type: feat.call_type,
                        phrase_index: feat.phrase_index,
                        features: feat.features,
                        duration_ms: feat.duration_ms,
                    });
                }

                start_index = all_features.len();

                if start_index >= flac_files.len() {
                    println!();
                    println!("   ✅ All files were already processed!");
                    println!("   Proceeding to clustering step...");
                    println!();
                } else {
                    println!();
                    println!("   🔄 Resuming from file {} of {} ({} remaining)...",
                             start_index + 1, flac_files.len(), flac_files.len() - start_index);
                    println!();
                }
            }
            Err(e) => {
                println!("   ⚠️  Failed to load checkpoint: {}", e);
                println!("   Starting fresh...");
                println!();
            }
        }
    } else if resume {
        println!("   ℹ️  Resume requested but no checkpoint found. Starting fresh...");
        println!();
    }

    // Count by call type
    println!("   📊 Call Type Distribution:");
    let mut call_type_counts: HashMap<CallType, usize> = HashMap::new();
    for (path, _) in flac_files[start_index..].iter() {
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let call_type = CallType::from_filename(filename);
        *call_type_counts.entry(call_type).or_insert(0) += 1;
    }
    for (call_type, count) in call_type_counts.iter() {
        println!("      • {}: {} files", call_type.name(), count);
    }
    println!();

    // =============================================================================
    // Step 1: PARALLEL Feature Extraction (30D)
    // =============================================================================

    if start_index < flac_files.len() {
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│ Step 1: PARALLEL Feature Extraction - 30D MicroDynamics │");
        println!("└─────────────────────────────────────────────────────────────────┘");
        println!();

        println!("   🔧 Using ORIGINAL 30D features (MicroDynamicsFeatures):");
        println!("      ├─ Temporal (3): attack_time_ms, decay_time_ms, sustain_level");
        println!("      ├─ Modulation (2): vibrato_rate_hz, vibrato_depth");
        println!("      ├─ Perturbation (2): jitter, shimmer");
        println!("      ├─ Timbre (3): harmonicity, spectral_flatness, hnr");
        println!("      ├─ MFCCs (13): mfcc[0-12]");
        println!("      ├─ Spectral (1): spectral_flux");
        println!("      └─ Rhythm (3): median_ici_ms, onset_rate_hz, ici_cv");
        println!();

        println!("   ⚡ Extracting 30D features in PARALLEL...");
        println!("      └─ Chunks: {} (for load balancing)", parallel_chunks);
        println!();

        let extract_start = Instant::now();

        let files_to_process: Vec<_> = flac_files[start_index..].to_vec();
        let tracker = ProgressTracker::new(files_to_process.len());
        let batch_size = (files_to_process.len() + parallel_chunks - 1) / parallel_chunks;

        let chunk_results: Vec<Vec<SerializableFeatures>> = files_to_process
            .par_chunks(batch_size)
            .map(|chunk| {
                let mut local_features = Vec::new();
                for (file_path, call_type) in chunk {
                    let filename = file_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");

                    match load_flac_file(file_path) {
                        Ok(audio) => {
                            let extractor = MicroDynamicsExtractor::new(sample_rate);
                            match extractor.extract(&audio) {
                                Ok(features) => {
                                    // Convert 30D features to Vec<f64>
                                    let feature_vec: Vec<f64> = vec![
                                        features.attack_time_ms as f64,
                                        features.decay_time_ms as f64,
                                        features.sustain_level as f64,
                                        features.vibrato_rate_hz as f64,
                                        features.vibrato_depth as f64,
                                        features.jitter as f64,
                                        features.shimmer as f64,
                                        features.harmonicity as f64,
                                        features.spectral_flatness as f64,
                                        features.harmonic_to_noise_ratio as f64,
                                    ];
                                    // Add 13 MFCCs
                                    let mut with_mfcc = feature_vec;
                                    with_mfcc.extend(features.mfcc.iter().map(|&v| v as f64));
                                    // Add spectral_flux
                                    with_mfcc.push(features.spectral_flux as f64);
                                    // Add rhythm features
                                    with_mfcc.push(features.median_ici_ms as f64);
                                    with_mfcc.push(features.onset_rate_hz as f64);
                                    with_mfcc.push(features.ici_coefficient_of_variation as f64);

                                    local_features.push(SerializableFeatures {
                                        file_name: filename.to_string(),
                                        call_type: call_type.name().to_string(),
                                        phrase_index: 0,
                                        features: with_mfcc,
                                        duration_ms: audio.len() as f64 / sample_rate as f64 * 1000.0,
                                    });
                                }
                                Err(e) => {
                                    eprintln!("      Warning: Feature extraction failed for {}: {}", filename, e);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("      Warning: Failed to load {}: {}", filename, e);
                        }
                    }
                    tracker.increment();
                }
                local_features
            })
            .collect();

        for mut chunk_features in chunk_results {
            for feat in chunk_features.drain(..) {
                all_file_names.push(feat.file_name.clone());
                all_features.push(ExtractedFeatures {
                    file_name: feat.file_name,
                    call_type: feat.call_type,
                    phrase_index: feat.phrase_index,
                    features: feat.features,
                    duration_ms: feat.duration_ms,
                });
            }
        }

        let extract_time = extract_start.elapsed();
        let n_features = all_features.len();
        let newly_processed = n_features - start_index;

        println!();
        println!("   ✅ Feature extraction complete!");
        println!("      ├─ Total features: {}", n_features);
        println!("      ├─ Newly processed: {}", newly_processed);
        println!("      ├─ Time: {:.2}s", extract_time.as_secs_f64());
        println!("      ├─ Rate: {:.1} files/sec", newly_processed as f64 / extract_time.as_secs_f64());
        println!("      └─ Speedup: ~{}x vs sequential", num_cpus);
        println!();

        // Save checkpoint
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│ Checkpoint: Saving Progress                                         │");
        println!("└─────────────────────────────────────────────────────────────────┘");
        println!();

        fs::create_dir_all(&results_dir)?;

        let serializable_features: Vec<SerializableFeatures> = all_features
            .iter()
            .map(|f| SerializableFeatures {
                file_name: f.file_name.clone(),
                call_type: f.call_type.clone(),
                phrase_index: f.phrase_index,
                features: f.features.clone(),
                duration_ms: f.duration_ms,
            })
            .collect();

        let checkpoint_data = CheckpointData {
            all_features: serializable_features,
            all_file_names: all_file_names.clone(),
            total_files: flac_files.len(),
            processed_at: chrono::Utc::now().to_rfc3339(),
        };

        save_checkpoint(&checkpoint_path, &checkpoint_data)?;
        println!("   💾 Checkpoint saved: {}", checkpoint_path.display());
        println!();
    }

    let n_features = all_features.len();

    if n_features == 0 {
        return Err("No features extracted. Check audio files and paths.".into());
    }

    // =============================================================================
    // Step 2: Convert to 2D Array for Clustering
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Converting Features to 2D Array                            │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let convert_start = Instant::now();

    let n_dims = 30;
    let mut feature_matrix = ndarray::Array2::zeros((n_features, n_dims));

    for (i, feat) in all_features.iter().enumerate() {
        for (j, &val) in feat.features.iter().enumerate() {
            if j < n_dims {
                feature_matrix[[i, j]] = val;
            }
        }
    }

    println!("   ✅ Converted to {}x{} array in {:.2}s",
             n_features, n_dims, convert_start.elapsed().as_secs_f64());
    println!();

    // Normalize features
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Normalizing Features                                         │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let normalize_start = Instant::now();
    let feature_matrix_normalized = normalize_features(&feature_matrix);
    println!("   ✅ Features normalized in {:.2}s", normalize_start.elapsed().as_secs_f64());
    println!();

    // =============================================================================
    // Step 4: Multi-Method Clustering Comparison
    // =============================================================================

    let comparison = compare_clustering_approaches(&feature_matrix_normalized, n_features, n_dims);

    // =============================================================================
    // Step 5: Save Results
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Saving Results                                               │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    fs::create_dir_all(&results_dir)?;

    // Save comparison report
    let comparison_path = results_dir.join("clustering_comparison.json");
    fs::write(&comparison_path, serde_json::to_string_pretty(&comparison)?)?;
    println!("   💾 Comparison report: {}", comparison_path.display());

    // Save feature matrix
    let features_path = results_dir.join("marmoset_30d_features.bincode");
    let serializable_features: Vec<SerializableFeatures> = all_features
        .iter()
        .map(|f| SerializableFeatures {
            file_name: f.file_name.clone(),
            call_type: f.call_type.clone(),
            phrase_index: f.phrase_index,
            features: f.features.clone(),
            duration_ms: f.duration_ms,
        })
        .collect();
    let features_data = bincode::serialize(&serializable_features)?;
    fs::write(&features_path, &features_data)?;
    println!("   💾 Features saved: {} ({} MB)",
             features_path.display(), features_data.len() / 1_048_576);

    // Save recommended clustering
    if let Some(ref rec_result) = comparison.results.iter().find(|r| r.algorithm == comparison.recommended) {
        let clusters_path = results_dir.join("hdbscan_clusters_recommended.json");
        let clusters_output = serde_json::json!({
            "metadata": {
                "dataset": "marmoset_vocalizations",
                "n_files": flac_files.len(),
                "n_features": n_features,
                "n_dims": n_dims,
                "feature_type": "30D_MicroDynamicsFeatures (original)",
                "normalized": true,
            },
            "clustering": {
                "algorithm": rec_result.algorithm,
                "parameters": rec_result.parameters,
                "n_clusters": rec_result.n_clusters,
                "noise_count": rec_result.noise_count,
                "cluster_sizes": rec_result.cluster_sizes,
            },
            "labels": rec_result.labels,
            "recommendation": {
                "chosen": comparison.recommended,
                "reason": comparison.recommendation_reason,
            }
        });
        fs::write(&clusters_path, serde_json::to_string_pretty(&clusters_output)?)?;
        println!("   💾 Recommended clusters: {}", clusters_path.display());

        // Generate symbolic stream
        let cluster_offset = 100;
        let symbolic_stream: Vec<i32> = rec_result.labels.iter()
            .map(|&label| if label == -1 { 0 } else { label + cluster_offset })
            .collect();

        let stream_path = results_dir.join("symbolic_stream.txt");
        let stream_text: String = symbolic_stream.iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");
        fs::write(&stream_path, &stream_text)?;
        println!("   💾 Symbolic stream: {}", stream_path.display());

        let readable_path = results_dir.join("symbolic_stream_readable.csv");
        let mut readable_content = String::from("file_name,call_type,cluster_id,symbol\n");
        for (file_info, label) in all_features.iter().zip(rec_result.labels.iter()) {
            let symbol = if *label == -1 { 0 } else { *label + cluster_offset };
            readable_content.push_str(&format!("{},{},{},{}\n",
                file_info.file_name, file_info.call_type, label, symbol));
        }
        fs::write(&readable_path, &readable_content)?;
        println!("   💾 Readable stream: {}", readable_path.display());
    }
    println!();

    // =============================================================================
    // Summary
    // =============================================================================

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║              PHASE 0 COMPLETE (30D + Multi-Clustering)              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ 🔧 FEATURES: Original 30D MicroDynamics (not 37D)               ║");
    println!("║ 🔧 CLUSTERING: {} approaches tested                        ║", comparison.results.len());
    println!("║ 📊 RECOMMENDED: {}                                        ║", comparison.recommended);
    println!("║                                                                   ║");
    println!("║ 📊 SUMMARY:                                                       ║");
    println!("║     • Input: {} FLAC files                                      ║", flac_files.len());
    println!("║     • Features: 30D MicroDynamics (original)                    ║");
    println!("║     • Normalized: Yes                                                ║");
    println!("║     • Recommended: {}                                        ║", comparison.recommended);
    println!("║                                                                   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

fn discover_flac_files(dir: &Path) -> Result<Vec<(PathBuf, CallType)>, Box<dyn std::error::Error>> {
    let mut flac_files = Vec::new();
    let entries = fs::read_dir(dir)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            flac_files.extend(discover_flac_files(&path)?);
        } else if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext.to_string_lossy().to_lowercase() == "flac" {
                    let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    let call_type = CallType::from_filename(filename);
                    if call_type != CallType::Unknown {
                        flac_files.push((path, call_type));
                    }
                }
            }
        }
    }

    Ok(flac_files)
}

fn load_flac_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("flac");

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("No valid audio track found")?;

    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let n_channels = decoder.codec_params().channels.map_or(1, |ch| ch.count());

    let mut audio_samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };

        let decoded = decoder.decode(&packet)?;

        match decoded {
            AudioBufferRef::F32(buf) => {
                for ch in 0..n_channels {
                    audio_samples.extend_from_slice(buf.chan(ch));
                }
            }
            AudioBufferRef::S16(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i16::MAX as f32));
                }
            }
            AudioBufferRef::S32(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i32::MAX as f32));
                }
            }
            AudioBufferRef::U8(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| (s as f32 - 128.0) / 128.0));
                }
            }
            _ => return Err("Unsupported audio format".into()),
        }
    }

    Ok(audio_samples)
}

fn save_checkpoint(path: &Path, data: &CheckpointData) -> Result<(), Box<dyn std::error::Error>> {
    let encoded = bincode::serialize(data)?;
    fs::write(path, &encoded)?;
    Ok(())
}

fn load_checkpoint(path: &Path) -> Result<CheckpointData, Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    let decoded: CheckpointData = bincode::deserialize(&data)?;
    Ok(decoded)
}
