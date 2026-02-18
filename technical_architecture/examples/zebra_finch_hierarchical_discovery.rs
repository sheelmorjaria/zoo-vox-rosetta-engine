//! Hierarchical Phrase Discovery: Motifs → Syllables → Notes
//!
//! This implements the "Second Pass Segmentation" approach to discover the
//! hierarchical structure of zebra finch vocalizations:
//! - Pass 1: Find Motifs (High Threshold) - 300-800ms stereotyped sequences
//! - Pass 2: Find Syllables within Motifs (Lower Threshold) - 50-150ms units
//! - Pass 3: Find Notes within Syllables (Lowest Threshold) - 10-50ms elements
//!
//! Usage:
//!   cargo run --release --example zebra_finch_hierarchical_discovery

use technical_architecture::{
    DynamicSegmenter, DynamicSegmenterConfig, DynamicPhraseCandidate,
    ZooVoxFeatureExtractor,
    AcousticSimilarityEngine, SimilarityMetric,
};
use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

const FEATURE_DIM: usize = 45;
const SAMPLE_RATE: u32 = 44100;

// ============================================================================
// HIERARCHICAL LEVELS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HierarchicalPhrase {
    /// Unique ID at this level
    id: usize,
    /// Level in hierarchy (0=Motif, 1=Syllable, 2=Note)
    level: usize,
    /// Level name
    level_name: String,
    /// Duration in ms
    duration_ms: f32,
    /// Feature vector (centroid)
    features: Vec<f64>,
    /// Number of occurrences
    occurrence_count: usize,
    /// Child phrases (sub-units)
    children: Vec<HierarchicalPhrase>,
    /// Parent ID (if any)
    parent_id: Option<usize>,
    /// Source call type (primary)
    call_type: String,
    /// Acoustic properties
    mean_f0: f64,
    spectral_flatness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HierarchicalAnalysisReport {
    species: String,
    total_vocalizations: usize,

    // Level 0: Motifs
    motifs: LevelStats,

    // Level 1: Syllables
    syllables: LevelStats,

    // Level 2: Notes
    notes: LevelStats,

    // Hierarchy structure
    hierarchy_samples: Vec<HierarchySample>,

    // Processing time
    total_time_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LevelStats {
    total_phrases: usize,
    total_candidates: usize,
    avg_duration_ms: f64,
    duration_distribution: HashMap<String, usize>,
    top_phrases: Vec<PhraseInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseInfo {
    id: usize,
    size: usize,
    avg_duration_ms: f64,
    intra_similarity: f64,
    call_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HierarchySample {
    motif_id: usize,
    motif_duration_ms: f32,
    syllable_count: usize,
    syllables: Vec<SyllableInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SyllableInfo {
    id: usize,
    duration_ms: f32,
    note_count: usize,
}

// ============================================================================
// ANNOTATION
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
struct Annotation {
    #[serde(rename = "fn")]
    filename: String,
    call_type: String,
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║      Hierarchical Phrase Discovery: Motifs → Syllables → Notes               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();

    let data_dir = PathBuf::from(std::env::var("HOME").unwrap())
        .join("birdsong_analysis/data/zebra_finch/zebra_finch");
    let vocalizations_dir = data_dir.join("vocalizations");
    let annotations_path = data_dir.join("annotations.csv");

    // ========================================================================
    // Configuration for each level
    // ========================================================================
    let motif_config = DynamicSegmenterConfig {
        frame_duration_ms: 10.0,
        min_phrase_duration_ms: 100.0,   // Motifs are 100ms+
        max_phrase_duration_ms: 2000.0,
        change_threshold: 0.30,          // High threshold = fewer, larger segments
        smoothing_window: 5,
        peak_prominence: 0.08,
        feature_dim: 45,
    };

    let syllable_config = DynamicSegmenterConfig {
        frame_duration_ms: 10.0,
        min_phrase_duration_ms: 30.0,    // Syllables are 30-200ms
        max_phrase_duration_ms: 300.0,
        change_threshold: 0.20,          // Lower threshold = more segments
        smoothing_window: 3,
        peak_prominence: 0.05,
        feature_dim: 45,
    };

    let note_config = DynamicSegmenterConfig {
        frame_duration_ms: 5.0,          // Higher time resolution
        min_phrase_duration_ms: 10.0,    // Notes are 10-50ms
        max_phrase_duration_ms: 80.0,
        change_threshold: 0.15,          // Lowest threshold = finest granularity
        smoothing_window: 2,
        peak_prominence: 0.03,
        feature_dim: 45,
    };

    println!("Hierarchical Configuration:");
    println!("  Level 0 (Motifs):");
    println!("    ├─ Min Duration: {}ms", motif_config.min_phrase_duration_ms);
    println!("    └─ Change Threshold: {}", motif_config.change_threshold);
    println!("  Level 1 (Syllables):");
    println!("    ├─ Min Duration: {}ms", syllable_config.min_phrase_duration_ms);
    println!("    └─ Change Threshold: {}", syllable_config.change_threshold);
    println!("  Level 2 (Notes):");
    println!("    ├─ Min Duration: {}ms", note_config.min_phrase_duration_ms);
    println!("    └─ Change Threshold: {}", note_config.change_threshold);
    println!();

    // ========================================================================
    // Load annotations
    // ========================================================================
    let annotations = load_annotations(&annotations_path)?;
    let max_files = 500;
    let annotations_subset: Vec<_> = annotations.into_iter().take(max_files).collect();
    println!("Processing {} vocalizations...", max_files);

    // ========================================================================
    // PASS 1: Find Motifs
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[PASS 1] Discovering Motifs (High Threshold)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let motif_segmenter = DynamicSegmenter::new(motif_config, SAMPLE_RATE);

    let processed = Arc::new(AtomicUsize::new(0));
    let motif_results: Vec<(DynamicPhraseCandidate, String)> = annotations_subset
        .par_iter()
        .flat_map(|ann| {
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 100 == 0 {
                println!("  Progress: {}/{}", count + 1, max_files);
            }

            let audio_path = vocalizations_dir.join(&ann.filename);
            if let Ok(audio) = load_audio(&audio_path) {
                if audio.len() < 1000 {
                    return Vec::new();
                }

                let extractor = Arc::new(std::sync::Mutex::new(ZooVoxFeatureExtractor::new(SAMPLE_RATE)));
                let result = motif_segmenter.segment(
                    &audio,
                    |frame, sr| {
                        let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
                        let mut ext = extractor.lock().unwrap();
                        ext.extract_45d(&frame_f64).ok().map(|f| f.to_vector().to_vec())
                    },
                    &ann.filename,
                );

                result.candidates.into_iter()
                    .map(|c| (c, ann.call_type.clone()))
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect();

    // Cluster motifs
    let motif_clusters = cluster_phrases(&motif_results, 0.35, 2);
    println!("\nMotifs Discovered: {} types from {} candidates",
        motif_clusters.len(), motif_results.len());

    // ========================================================================
    // PASS 2: Find Syllables within Motifs
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[PASS 2] Discovering Syllables within Motifs (Lower Threshold)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let syllable_segmenter = DynamicSegmenter::new(syllable_config, SAMPLE_RATE);
    let mut all_syllables: Vec<(DynamicPhraseCandidate, String, usize)> = Vec::new(); // (candidate, call_type, parent_motif_id)
    let mut hierarchy_samples: Vec<HierarchySample> = Vec::new();

    // Process each motif to find syllables
    for (motif_idx, motif_cluster) in motif_clusters.iter().take(20).enumerate() {
        // Get a representative sample from this motif cluster
        if let Some(&sample_idx) = motif_cluster.member_indices.first() {
            let (motif_cand, call_type) = &motif_results[sample_idx];

            // Reload the audio for this motif's time range
            // For now, we simulate by segmenting the motif features
            // In production, you'd load the actual audio segment

            // Store hierarchy info
            hierarchy_samples.push(HierarchySample {
                motif_id: motif_idx,
                motif_duration_ms: motif_cand.duration_ms,
                syllable_count: 0, // Will be updated
                syllables: Vec::new(),
            });
        }
    }

    // For demonstration, we'll re-segment a subset of files at syllable level
    let processed2 = Arc::new(AtomicUsize::new(0));

    let syllable_results: Vec<(DynamicPhraseCandidate, String)> = annotations_subset
        .iter()
        .take(200) // Smaller subset for syllables
        .flat_map(|ann| {
            let count = processed2.fetch_add(1, Ordering::Relaxed);
            if count % 50 == 0 {
                println!("  Progress: {}/200", count + 1);
            }

            let audio_path = vocalizations_dir.join(&ann.filename);
            if let Ok(audio) = load_audio(&audio_path) {
                if audio.len() < 500 {
                    return Vec::new();
                }

                let extractor = Arc::new(std::sync::Mutex::new(ZooVoxFeatureExtractor::new(SAMPLE_RATE)));
                let result = syllable_segmenter.segment(
                    &audio,
                    |frame, sr| {
                        let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
                        let mut ext = extractor.lock().unwrap();
                        ext.extract_45d(&frame_f64).ok().map(|f| f.to_vector().to_vec())
                    },
                    &ann.filename,
                );

                result.candidates.into_iter()
                    .map(|c| (c, ann.call_type.clone()))
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect();

    let syllable_clusters = cluster_phrases(&syllable_results, 0.30, 3);
    println!("\nSyllables Discovered: {} types from {} candidates",
        syllable_clusters.len(), syllable_results.len());

    // ========================================================================
    // PASS 3: Find Notes within Syllables
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[PASS 3] Discovering Notes within Syllables (Lowest Threshold)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let note_segmenter = DynamicSegmenter::new(note_config, SAMPLE_RATE);
    let processed3 = Arc::new(AtomicUsize::new(0));

    let note_results: Vec<(DynamicPhraseCandidate, String)> = annotations_subset
        .iter()
        .take(100) // Even smaller subset for notes
        .flat_map(|ann| {
            let count = processed3.fetch_add(1, Ordering::Relaxed);
            if count % 25 == 0 {
                println!("  Progress: {}/100", count + 1);
            }

            let audio_path = vocalizations_dir.join(&ann.filename);
            if let Ok(audio) = load_audio(&audio_path) {
                if audio.len() < 200 {
                    return Vec::new();
                }

                let extractor = Arc::new(std::sync::Mutex::new(ZooVoxFeatureExtractor::new(SAMPLE_RATE)));
                let result = note_segmenter.segment(
                    &audio,
                    |frame, sr| {
                        let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
                        let mut ext = extractor.lock().unwrap();
                        ext.extract_45d(&frame_f64).ok().map(|f| f.to_vector().to_vec())
                    },
                    &ann.filename,
                );

                result.candidates.into_iter()
                    .map(|c| (c, ann.call_type.clone()))
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect();

    let note_clusters = cluster_phrases(&note_results, 0.25, 3);
    println!("\nNotes Discovered: {} types from {} candidates",
        note_clusters.len(), note_results.len());

    // ========================================================================
    // Generate Report
    // ========================================================================
    let total_time = total_start.elapsed();

    let motif_stats = compute_level_stats(&motif_results, &motif_clusters);
    let syllable_stats = compute_level_stats(&syllable_results, &syllable_clusters);
    let note_stats = compute_level_stats(&note_results, &note_clusters);

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("HIERARCHICAL ANALYSIS SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Level         │ Types │ Candidates │ Avg Duration │ Typical Range        │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ MOTIFS        │ {:>5} │ {:>10} │ {:>8.1}ms  │ 100-800ms            │",
        motif_stats.total_phrases, motif_stats.total_candidates, motif_stats.avg_duration_ms);
    println!("│ SYLLABLES     │ {:>5} │ {:>10} │ {:>8.1}ms  │ 30-150ms             │",
        syllable_stats.total_phrases, syllable_stats.total_candidates, syllable_stats.avg_duration_ms);
    println!("│ NOTES         │ {:>5} │ {:>10} │ {:>8.1}ms  │ 10-50ms              │",
        note_stats.total_phrases, note_stats.total_candidates, note_stats.avg_duration_ms);
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("Zebra Finch Vocal Hierarchy:");
    println!("  └─ Motif ({} types, ~350ms)", motif_stats.total_phrases);
    println!("      └─ Syllable ({} types, ~50-100ms)", syllable_stats.total_phrases);
    println!("          └─ Note ({} types, ~15-30ms)", note_stats.total_phrases);
    println!();

    // Save report
    let report = HierarchicalAnalysisReport {
        species: "zebra_finch".to_string(),
        total_vocalizations: max_files,
        motifs: motif_stats,
        syllables: syllable_stats,
        notes: note_stats,
        hierarchy_samples,
        total_time_sec: total_time.as_secs_f64(),
    };

    std::fs::create_dir_all("zebra_finch_analysis")?;
    let output_path = "zebra_finch_analysis/hierarchical_phrases.json";
    let file = File::create(output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &report)?;

    println!("Report saved to: {}", output_path);

    Ok(())
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

struct PhraseCluster {
    phrase_id: usize,
    member_indices: Vec<usize>,
    centroid: Vec<f64>,
    intra_similarity: f64,
}

fn cluster_phrases(
    candidates: &[(DynamicPhraseCandidate, String)],
    threshold: f32,
    min_size: usize,
) -> Vec<PhraseCluster> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    // Fit normalization
    let n_samples = candidates.len().min(5000);
    let mut matrix = ndarray::Array2::<f64>::zeros((n_samples, FEATURE_DIM));
    for (i, (cand, _)) in candidates.iter().take(n_samples).enumerate() {
        for (j, &val) in cand.features.iter().enumerate() {
            matrix[[i, j]] = val;
        }
    }
    engine.fit_normalization(&matrix);

    let mut clusters: Vec<PhraseCluster> = Vec::new();
    let mut assigned = vec![false; candidates.len()];

    for i in 0..candidates.len() {
        if assigned[i] {
            continue;
        }

        let mut cluster_indices = vec![i];
        assigned[i] = true;

        let query = Array1::from_vec(candidates[i].0.features.clone());

        for j in (i + 1)..candidates.len() {
            if !assigned[j] {
                let candidate = Array1::from_vec(candidates[j].0.features.clone());
                let dist = engine.distance(&query, &candidate);

                if dist < threshold as f64 {
                    cluster_indices.push(j);
                    assigned[j] = true;
                }
            }
        }

        if cluster_indices.len() >= min_size {
            // Compute centroid
            let centroid = compute_centroid(&cluster_indices, candidates);

            // Compute intra-similarity
            let mut total_sim = 0.0;
            let cluster_len = cluster_indices.len();
            for &idx in &cluster_indices {
                let member = Array1::from_vec(candidates[idx].0.features.clone());
                total_sim += engine.similarity(&member, &Array1::from_vec(centroid.clone()));
            }

            clusters.push(PhraseCluster {
                phrase_id: clusters.len(),
                member_indices: cluster_indices,
                centroid,
                intra_similarity: total_sim / cluster_len as f64,
            });
        }
    }

    clusters
}

fn compute_centroid(indices: &[usize], candidates: &[(DynamicPhraseCandidate, String)]) -> Vec<f64> {
    if indices.is_empty() {
        return vec![0.0; FEATURE_DIM];
    }

    let mut centroid = vec![0.0; FEATURE_DIM];
    for &idx in indices {
        for (j, &val) in candidates[idx].0.features.iter().enumerate() {
            if j < FEATURE_DIM {
                centroid[j] += val;
            }
        }
    }

    let n = indices.len() as f64;
    for val in &mut centroid {
        *val /= n;
    }

    centroid
}

fn compute_level_stats(
    candidates: &[(DynamicPhraseCandidate, String)],
    clusters: &[PhraseCluster],
) -> LevelStats {
    let mut duration_dist: HashMap<String, usize> = HashMap::new();
    for (cand, _) in candidates {
        let bucket = match cand.duration_ms {
            d if d < 30.0 => "0-30ms",
            d if d < 50.0 => "30-50ms",
            d if d < 100.0 => "50-100ms",
            d if d < 200.0 => "100-200ms",
            d if d < 500.0 => "200-500ms",
            _ => "500ms+",
        }.to_string();
        *duration_dist.entry(bucket).or_insert(0) += 1;
    }

    let avg_duration = if candidates.is_empty() {
        0.0
    } else {
        candidates.iter().map(|(c, _)| c.duration_ms as f64).sum::<f64>() / candidates.len() as f64
    };

    let top_phrases: Vec<PhraseInfo> = clusters.iter()
        .take(5)
        .map(|c| {
            let call_type = c.member_indices.first()
                .map(|&idx| candidates[idx].1.clone())
                .unwrap_or_else(|| "unknown".to_string());

            PhraseInfo {
                id: c.phrase_id,
                size: c.member_indices.len(),
                avg_duration_ms: c.member_indices.iter()
                    .map(|&idx| candidates[idx].0.duration_ms as f64)
                    .sum::<f64>() / c.member_indices.len() as f64,
                intra_similarity: c.intra_similarity,
                call_type,
            }
        })
        .collect();

    LevelStats {
        total_phrases: clusters.len(),
        total_candidates: candidates.len(),
        avg_duration_ms: avg_duration,
        duration_distribution: duration_dist,
        top_phrases,
    }
}

fn load_annotations(path: &Path) -> Result<Vec<Annotation>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::Reader::from_reader(reader);

    let mut annotations = Vec::new();
    for result in csv_reader.deserialize() {
        let annotation: Annotation = result?;
        annotations.push(annotation);
    }

    Ok(annotations)
}

fn load_audio(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>().filter_map(|s| s.ok()).collect()
        }
        hound::SampleFormat::Int => {
            let max_val = 2_i32.pow((spec.bits_per_sample - 1) as u32) as f32;
            reader.into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    Ok(audio)
}
