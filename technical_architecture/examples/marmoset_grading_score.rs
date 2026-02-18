//! Marmoset Grading Score Analysis
//!
//! Demonstrates the grading score system for distinguishing discrete vs graded calls.
//!
//! Based on variance analysis findings:
//! - Type_1, Type_12: Highly stereotyped (discrete) - emit type ID only
//! - Type_2, Type_3: Higher variance (graded) - emit type ID + 45D vector
//!
//! The grading_score indicates how far an instance is from its type centroid.
//! Low score = typical example, High score = outlier/graded instance.

use technical_architecture::{
    DynamicSegmenter, DynamicSegmenterConfig,
    TypedPhraseCandidate, EmissionStrategy,
    ZooVoxFeatureExtractor,
    AcousticSimilarityEngine, SimilarityMetric,
    HierarchicalThresholds,
    SpeciesConfigFactory,
    species::FeatureWeights,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const FEATURE_DIM: usize = 45;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let max_files: usize = args.get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000);

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║         Marmoset Grading Score Analysis                                       ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Analyzing grading scores to determine emission strategies:");
    println!("  • Discrete Path: Emit type ID only (low variance types)");
    println!("  • Continuous Path: Emit type ID + 45D vector (graded types)");
    println!();

    let base_dir = PathBuf::from(std::env::var("HOME").unwrap())
        .join("birdsong_analysis/data/Vocalizations");

    // Get species config with feature weights
    let species_config = SpeciesConfigFactory::create("marmoset");
    let feature_weights = species_config.feature_weights();

    println!("Feature weights for marmoset:");
    println!("  ├─ Spectral: {:.1}", feature_weights.spectral);
    println!("  ├─ Harmonic: {:.1}", feature_weights.harmonic);
    println!("  ├─ Temporal: {:.1}", feature_weights.temporal);
    println!("  └─ Modulation: {:.1}", feature_weights.modulation);
    println!();

    // =========================================================================
    // Step 1: Extract Candidates
    // =========================================================================
    println!("[1/4] Extracting phrase candidates from {} files...", max_files);

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
            let extractor = Arc::new(std::sync::Mutex::new(ZooVoxFeatureExtractor::new(sample_rate)));

            let extract_fn = |frame: &[f32], _sr: u32| {
                let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
                let mut ext = extractor.lock().unwrap();
                ext.extract_45d(&frame_f64).ok().map(|f| f.to_vector().to_vec())
            };

            let result = segmenter.segment(&audio, extract_fn, filename);
            result.candidates.into_iter()
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
    // Step 2: Cluster and Calculate Centroids
    // =========================================================================
    println!();
    println!("[2/4] Clustering and calculating type centroids...");

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

    // Calculate centroids for each type
    let centroids: Vec<Vec<f64>> = phrase_types.iter()
        .map(|pt| {
            let mut centroid = vec![0.0; FEATURE_DIM];
            for &idx in &pt.indices {
                for (j, &val) in all_candidates[idx].0.iter().take(FEATURE_DIM).enumerate() {
                    centroid[j] += val;
                }
            }
            let n = pt.indices.len() as f64;
            for val in &mut centroid {
                *val /= n;
            }
            centroid
        })
        .collect();

    // =========================================================================
    // Step 3: Calculate Grading Scores
    // =========================================================================
    println!();
    println!("[3/4] Calculating grading scores for each instance...");

    let mut type_variance: HashMap<String, f32> = HashMap::new();
    for (type_idx, pt) in phrase_types.iter().enumerate() {
        let centroid = &centroids[type_idx];
        let distances: Vec<f64> = pt.indices.iter()
            .map(|&idx| cosine_distance(&all_candidates[idx].0, centroid))
            .collect();
        let avg_distance = if distances.is_empty() { 0.0 } else {
            distances.iter().sum::<f64>() / distances.len() as f64
        };
        type_variance.insert(pt.type_id.clone(), avg_distance as f32);
    }

    // Create typed candidates with grading scores
    let mut typed_candidates: Vec<TypedCandidateInfo> = Vec::new();
    for (type_idx, pt) in phrase_types.iter().enumerate() {
        let centroid = &centroids[type_idx];
        let variance = type_variance.get(&pt.type_id).copied().unwrap_or(0.0);

        for &idx in &pt.indices {
            let (features, duration_ms, source_file) = &all_candidates[idx];
            let grading_score = cosine_distance(features, centroid) as f32;
            let is_graded = grading_score > TypedPhraseCandidate::GRADING_THRESHOLD;

            typed_candidates.push(TypedCandidateInfo {
                type_id: pt.type_id.clone(),
                grading_score,
                intra_type_variance: variance,
                is_graded,
                duration_ms: *duration_ms,
                source_file: source_file.clone(),
            });
        }
    }

    // =========================================================================
    // Step 4: Analyze Emission Strategies
    // =========================================================================
    println!();
    println!("[4/4] Analyzing emission strategies...");
    println!();

    // Count discrete vs continuous emissions per type
    let mut type_stats: HashMap<String, EmissionStats> = HashMap::new();
    for tc in &typed_candidates {
        let stats = type_stats.entry(tc.type_id.clone()).or_insert_with(EmissionStats::default);
        stats.total += 1;
        if tc.is_graded {
            stats.graded += 1;
        } else {
            stats.discrete += 1;
        }
        stats.grading_scores.push(tc.grading_score);
    }

    // Sort by total count
    let mut sorted_types: Vec<_> = type_stats.into_iter().collect();
    sorted_types.sort_by(|a, b| b.1.total.cmp(&a.1.total));

    // Print results
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("EMISSION STRATEGY ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TYPE-BY-TYPE EMISSION STRATEGY                                              │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ {:<10} {:>8} {:>10} {:>10} {:>12}", "Type", "Total", "Discrete", "Graded", "Strategy");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    for (type_id, stats) in &sorted_types {
        let discrete_pct = stats.discrete as f64 / stats.total as f64 * 100.0;
        let strategy = if discrete_pct > 95.0 {
            "DISCRETE"
        } else if discrete_pct > 70.0 {
            "MIXED"
        } else {
            "CONTINUOUS"
        };

        println!("│ {:<10} {:>8} {:>10} {:>10} {:>12}",
            type_id, stats.total, stats.discrete, stats.graded, strategy);
    }
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Calculate overall statistics
    let total_candidates = typed_candidates.len();
    let total_discrete = typed_candidates.iter().filter(|tc| !tc.is_graded).count();
    let total_graded = typed_candidates.iter().filter(|tc| tc.is_graded).count();

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ OVERALL EMISSION SUMMARY                                                    │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Total Candidates:    {}", total_candidates);
    println!("│ Discrete Emissions:  {} ({:.1}%)", total_discrete, total_discrete as f64 / total_candidates as f64 * 100.0);
    println!("│ Continuous Emissions: {} ({:.1}%)", total_graded, total_graded as f64 / total_candidates as f64 * 100.0);
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ BANDWIDTH SAVINGS:");
    println!("│ Without grading: {} type IDs", total_candidates);
    println!("│ With grading: {} type IDs + {} vectors", total_discrete, total_graded);
    println!("│ Data reduction: {:.1}%", (1.0 - total_graded as f64 / total_candidates as f64) * 100.0);
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Grading score distribution
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ GRADING SCORE DISTRIBUTION (Top 5 Types)                                    │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    for (type_id, stats) in sorted_types.iter().take(5) {
        let scores = &stats.grading_scores;
        let min = scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = scores.iter().cloned().fold(0.0f32, f32::max);
        let avg = scores.iter().sum::<f32>() / scores.len() as f32;

        // Count by range
        let low = scores.iter().filter(|&&s| s < 0.03).count();
        let medium = scores.iter().filter(|&&s| s >= 0.03 && s < 0.07).count();
        let high = scores.iter().filter(|&&s| s >= 0.07).count();

        println!("│ {}: min={:.3}, max={:.3}, avg={:.3}", type_id, min, max, avg);
        println!("│   Low (<0.03): {}, Medium (0.03-0.07): {}, High (>0.07): {}", low, medium, high);
    }
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Example trajectory tracking
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ EXAMPLE: GRADING SCORE TRAJECTORY                                          │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ For Type_2 (high variance type), tracking grading_score over time reveals:");
    println!("│");
    println!("│   Call 1: score=0.04 (Low Arousal) - Typical Type_2");
    println!("│   Call 2: score=0.06 (Medium Arousal) - Drifting toward Type_3");
    println!("│   Call 3: score=0.08 (High Arousal) - Strong emotional state");
    println!("│");
    println!("│ The Python Cognitive Agent can detect escalation even though phrase_type_id");
    println!("│ remains constant. This is the power of the grading score system.");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    // Save report
    let report = GradingScoreReport {
        total_candidates,
        discrete_emissions: total_discrete,
        continuous_emissions: total_graded,
        type_stats: sorted_types.iter().map(|(k, v)| {
            (k.clone(), TypeReport {
                total: v.total,
                discrete: v.discrete,
                graded: v.graded,
            })
        }).collect(),
    };

    std::fs::create_dir_all("complete_analysis").ok();
    let file = std::fs::File::create("complete_analysis/marmoset_grading_score.json")?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), &report)?;
    println!();
    println!("Report saved to: complete_analysis/marmoset_grading_score.json");

    Ok(())
}

// ============================================================================
// HELPER STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Default)]
struct EmissionStats {
    total: usize,
    discrete: usize,
    graded: usize,
    grading_scores: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TypedCandidateInfo {
    type_id: String,
    grading_score: f32,
    intra_type_variance: f32,
    is_graded: bool,
    duration_ms: f32,
    source_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TypeReport {
    total: usize,
    discrete: usize,
    graded: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GradingScoreReport {
    total_candidates: usize,
    discrete_emissions: usize,
    continuous_emissions: usize,
    type_stats: Vec<(String, TypeReport)>,
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
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;
    use symphonia::core::audio::AudioBufferRef;

    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("flac");

    let mut probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
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

struct PhraseType {
    type_id: String,
    indices: Vec<usize>,
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

        types.push(PhraseType {
            type_id: format!("Type_{}", types.len() + 1),
            indices,
        });
    }

    types
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
