//! Marmoset Graded Vocalization Analysis
//!
//! Analyzes intra-cluster variance to determine if marmoset calls are:
//! - Discrete System: Tight clusters with low variance
//! - Graded System: Loose clusters with high variance (continuum)
//!
//! For graded systems, the trajectory (how the call changes) is the message,
//! not just the type ID.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use technical_architecture::{
    AcousticSimilarityEngine, DynamicSegmenter, DynamicSegmenterConfig, HierarchicalThresholds,
    SimilarityMetric, ZooVoxFeatureExtractor,
};

const FEATURE_DIM: usize = 45;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let max_files: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10000);

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║         Marmoset Graded Vocalization Analysis                                ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let base_dir =
        PathBuf::from(std::env::var("HOME").unwrap()).join("birdsong_analysis/data/Vocalizations");

    // =========================================================================
    // Step 1: Extract Features
    // =========================================================================
    println!("[1/4] Extracting features from {} files...", max_files);

    let files = discover_marmoset_files(&base_dir, max_files);
    println!("Found {} FLAC files", files.len());

    let thresholds = HierarchicalThresholds::marmoset();
    let segmenter_config = DynamicSegmenterConfig::for_syllable_level(&thresholds);
    let sample_rate = 44100u32;

    let processed = Arc::new(AtomicUsize::new(0));
    let total_files = files.len();

    let all_candidates: Vec<(Vec<f64>, f32, String)> = files
        .par_iter()
        .flat_map(|(path, filename, _call_type)| {
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 100 == 0 {
                print!("\r  Progress: {}/{}", count, total_files);
                std::io::stdout().flush().ok();
            }

            let audio = match load_flac_audio(path) {
                Ok(a) => a,
                Err(_) => return Vec::new(),
            };

            if audio.len() < 500 {
                return Vec::new();
            }

            let segmenter = DynamicSegmenter::new(segmenter_config.clone(), sample_rate);
            let extractor = Arc::new(std::sync::Mutex::new(ZooVoxFeatureExtractor::new(
                sample_rate,
            )));

            let extract_fn = |frame: &[f32], _sr: u32| {
                let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
                let mut ext = extractor.lock().unwrap();
                ext.extract_45d(&frame_f64)
                    .ok()
                    .map(|f| f.to_vector().to_vec())
            };

            let result = segmenter.segment(&audio, extract_fn, filename);
            result
                .candidates
                .into_iter()
                .map(|c| (c.features, c.duration_ms, filename.clone()))
                .collect::<Vec<_>>()
        })
        .collect();

    println!("\r  Progress: {}/{}", total_files, total_files);
    println!("Extracted {} candidates", all_candidates.len());

    if all_candidates.is_empty() {
        return Ok(());
    }

    // =========================================================================
    // Step 2: Cluster with Original Threshold
    // =========================================================================
    println!();
    println!("[2/4] Clustering with threshold 0.30...");

    let mut feature_matrix = ndarray::Array2::<f64>::zeros((all_candidates.len(), FEATURE_DIM));
    for (i, (features, _, _)) in all_candidates.iter().enumerate() {
        for (j, &val) in features.iter().take(FEATURE_DIM).enumerate() {
            feature_matrix[[i, j]] = val;
        }
    }

    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    engine.fit_normalization(&feature_matrix);

    let phrase_types = cluster_by_similarity(&all_candidates, &engine, 0.30);
    println!("Discovered {} phrase types", phrase_types.len());

    // =========================================================================
    // Step 3: Compute Intra-Cluster Variance
    // =========================================================================
    println!();
    println!("[3/4] Computing intra-cluster variance...");

    let mut cluster_stats: Vec<ClusterStats> = Vec::new();

    for pt in &phrase_types {
        if pt.indices.len() < 2 {
            continue;
        }

        // Get all feature vectors for this cluster
        let vectors: Vec<Vec<f64>> = pt
            .indices
            .iter()
            .map(|&idx| all_candidates[idx].0.clone())
            .collect();

        // Compute centroid
        let centroid = compute_centroid(&vectors);

        // Compute average distance to centroid (intra-cluster spread)
        let avg_distance: f64 = vectors
            .iter()
            .map(|v| cosine_distance(v, &centroid))
            .sum::<f64>()
            / vectors.len() as f64;

        // Compute max distance (outliers)
        let max_distance = vectors
            .iter()
            .map(|v| cosine_distance(v, &centroid))
            .fold(0.0f64, f64::max);

        // Compute min distance (tightest members)
        let min_distance = vectors
            .iter()
            .map(|v| cosine_distance(v, &centroid))
            .fold(f64::INFINITY, f64::min);

        // Compute variance of distances (cluster coherence)
        let distances: Vec<f64> = vectors
            .iter()
            .map(|v| cosine_distance(v, &centroid))
            .collect();
        let distance_variance = compute_variance(&distances);

        cluster_stats.push(ClusterStats {
            type_id: pt.type_id.clone(),
            instance_count: pt.indices.len(),
            avg_distance_to_centroid: avg_distance,
            max_distance_to_centroid: max_distance,
            min_distance_to_centroid: min_distance,
            distance_variance,
            avg_duration_ms: pt.avg_duration_ms,
        });
    }

    // Sort by instance count
    cluster_stats.sort_by(|a, b| b.instance_count.cmp(&a.instance_count));

    // =========================================================================
    // Step 4: Analyze and Report
    // =========================================================================
    println!();
    println!("[4/4] Analyzing graded vs discrete system...");
    println!();

    // Calculate overall metrics
    let avg_intra_variance: f64 = cluster_stats
        .iter()
        .map(|s| s.avg_distance_to_centroid)
        .sum::<f64>()
        / cluster_stats.len() as f64;

    let max_intra_variance = cluster_stats
        .iter()
        .map(|s| s.avg_distance_to_centroid)
        .fold(0.0f64, f64::max);

    // Determine system type
    let system_classification = if avg_intra_variance > 0.3 {
        "GRADED SYSTEM (high intra-cluster variance)"
    } else if avg_intra_variance > 0.15 {
        "MIXED SYSTEM (moderate variance, partial grading)"
    } else {
        "DISCRETE SYSTEM (tight clusters)"
    };

    // Print results
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("INTRA-CLUSTER VARIANCE ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ OVERALL METRICS                                                             │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│ Average Intra-Cluster Distance:  {:.4}",
        avg_intra_variance
    );
    println!(
        "│ Maximum Intra-Cluster Distance:  {:.4}",
        max_intra_variance
    );
    println!("│ Classification: {}", system_classification);
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PER-TYPE VARIANCE (Top 10 by frequency)                                     │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│ {:<10} {:>8} {:>12} {:>12} {:>12}",
        "Type", "Count", "Avg Dist", "Max Dist", "Variance"
    );
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    for stats in cluster_stats.iter().take(10) {
        println!(
            "│ {:<10} {:>8} {:>12.4} {:>12.4} {:>12.4}",
            stats.type_id,
            stats.instance_count,
            stats.avg_distance_to_centroid,
            stats.max_distance_to_centroid,
            stats.distance_variance
        );
    }
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Interpretation
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("INTERPRETATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if avg_intra_variance > 0.15 {
        println!("⚠ HIGH VARIANCE DETECTED");
        println!();
        println!("The clusters show significant internal spread, indicating:");
        println!("  • Marmoset calls exist on a CONTINUUM (graded system)");
        println!("  • The '23 types' are artificially discrete buckets");
        println!("  • Real structure is more like: Phee ←→ Trill ←→ Tsik");
        println!();
        println!("IMPLICATIONS FOR ANALYSIS:");
        println!("  1. Type IDs alone lose information about call trajectory");
        println!("  2. Consider tracking FEATURE TRAJECTORIES, not just type labels");
        println!("  3. The 'message' may be in HOW a call changes, not WHAT it is");
        println!("  4. Use lower similarity threshold to capture gradations");
        println!("  5. Consider continuous models (GMM, diffusion) instead of discrete clusters");
    } else {
        println!("✓ LOW VARIANCE DETECTED");
        println!();
        println!("The clusters are tight, indicating:");
        println!("  • Marmoset calls are relatively DISCRETE categories");
        println!("  • The 23 types represent genuine acoustic categories");
        println!("  • Type IDs are meaningful tokens for syntax analysis");
    }

    // Save report
    let report = GradedAnalysisReport {
        files_processed: files.len(),
        candidates_extracted: all_candidates.len(),
        phrase_types: phrase_types.len(),
        avg_intra_cluster_distance: avg_intra_variance,
        max_intra_cluster_distance: max_intra_variance,
        system_classification: system_classification.to_string(),
        cluster_stats: cluster_stats.clone(),
    };

    let report_path = "complete_analysis/marmoset_graded_analysis.json";
    std::fs::create_dir_all("complete_analysis").ok();
    let file = std::fs::File::create(report_path)?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &report)?;
    println!();
    println!("Report saved to: {}", report_path);

    Ok(())
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseType {
    type_id: String,
    indices: Vec<usize>,
    avg_duration_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClusterStats {
    type_id: String,
    instance_count: usize,
    avg_distance_to_centroid: f64,
    max_distance_to_centroid: f64,
    min_distance_to_centroid: f64,
    distance_variance: f64,
    avg_duration_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GradedAnalysisReport {
    files_processed: usize,
    candidates_extracted: usize,
    phrase_types: usize,
    avg_intra_cluster_distance: f64,
    max_intra_cluster_distance: f64,
    system_classification: String,
    cluster_stats: Vec<ClusterStats>,
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn discover_marmoset_files(base_dir: &Path, max_files: usize) -> Vec<(PathBuf, String, String)> {
    let mut files = Vec::new();

    fn scan_dir(dir: &Path, files: &mut Vec<(PathBuf, String, String)>, max_files: usize) {
        if files.len() >= max_files {
            return;
        }

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if files.len() >= max_files {
                    break;
                }

                let path = entry.path();
                if path.is_dir() {
                    scan_dir(&path, files, max_files);
                } else if path.extension().map(|e| e == "flac").unwrap_or(false) {
                    let filename = path.file_name().unwrap().to_string_lossy().to_string();
                    let call_type = filename.split('_').next().unwrap_or("Unknown").to_string();
                    files.push((path, filename, call_type));
                }
            }
        }
    }

    scan_dir(base_dir, &mut files, max_files);
    files
}

fn load_flac_audio(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use std::fs::File;
    use symphonia::core::audio::AudioBufferRef;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("flac");

    let mut probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("Probe failed: {}", e))?;

    let track = probed.format.default_track().ok_or("No track")?;
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Decoder failed: {}", e))?;

    let mut samples = Vec::new();

    loop {
        let packet = match probed.format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(AudioBufferRef::F32(buf)) => {
                for plane in buf.as_ref().planes().planes() {
                    samples.extend(plane.iter().copied());
                    break;
                }
            }
            Ok(AudioBufferRef::S16(buf)) => {
                for plane in buf.as_ref().planes().planes() {
                    samples.extend(plane.iter().map(|&s| s as f32 / i16::MAX as f32));
                    break;
                }
            }
            Ok(AudioBufferRef::S32(buf)) => {
                for plane in buf.as_ref().planes().planes() {
                    samples.extend(plane.iter().map(|&s| s as f32 / i32::MAX as f32));
                    break;
                }
            }
            _ => {}
        }
    }

    Ok(samples)
}

fn cluster_by_similarity(
    candidates: &[(Vec<f64>, f32, String)],
    engine: &AcousticSimilarityEngine,
    threshold: f32,
) -> Vec<PhraseType> {
    let n = candidates.len();
    let mut assigned = vec![false; n];
    let mut types: Vec<PhraseType> = Vec::new();

    for i in 0..n {
        if assigned[i] {
            continue;
        }

        let mut indices = vec![i];
        assigned[i] = true;

        let query = ndarray::Array1::from_vec(candidates[i].0.clone());

        for j in (i + 1)..n {
            if !assigned[j] {
                let candidate = ndarray::Array1::from_vec(candidates[j].0.clone());
                let dist = engine.distance(&query, &candidate);
                let similarity = 1.0 - dist.min(1.0);

                if similarity >= threshold as f64 {
                    indices.push(j);
                    assigned[j] = true;
                }
            }
        }

        let avg_dur: f32 =
            indices.iter().map(|&idx| candidates[idx].1).sum::<f32>() / indices.len() as f32;

        types.push(PhraseType {
            type_id: format!("Type_{}", types.len() + 1),
            indices,
            avg_duration_ms: avg_dur,
        });
    }

    types
}

fn compute_centroid(vectors: &[Vec<f64>]) -> Vec<f64> {
    let dim = vectors[0].len();
    let mut centroid = vec![0.0; dim];

    for v in vectors {
        for (i, &val) in v.iter().enumerate() {
            centroid[i] += val;
        }
    }

    let n = vectors.len() as f64;
    for val in &mut centroid {
        *val /= n;
    }

    centroid
}

fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 1.0;
    }

    let similarity = dot / (mag_a * mag_b);
    1.0 - similarity
}

fn compute_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

    variance
}
