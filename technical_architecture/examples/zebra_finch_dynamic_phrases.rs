//! Zebra Finch Atomic Phrase Discovery using Dynamic Segmentation (CPD)
//!
//! This tool discovers "atomic phrases" using Change Point Detection (CPD)
//! instead of fixed-size windowing. It treats vocalizations as continuous
//! landscapes where boundaries are defined by acoustic change, not time increments.
//!
//! Pipeline:
//! 1. Micro-Frame Extraction: Generate 45D vectors at 100Hz (10ms frames)
//! 2. Distance Calculation: Compute acoustic distance between consecutive frames
//! 3. Change Point Detection: Identify peaks in distance curve = phrase boundaries
//! 4. Segment Aggregation: Average features within boundaries to create candidates
//! 5. Atomic Discovery: Cluster candidates to find reusable "Atomic Phrases"
//!
//! Usage:
//!   cargo run --release --example zebra_finch_dynamic_phrases

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
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
use technical_architecture::{
    AcousticSimilarityEngine, AtomicPhraseAnalyzer, AtomicPhraseType, DynamicPhraseCandidate, DynamicSegmenter,
    DynamicSegmenterConfig, SimilarityMetric, ZooVoxFeatureExtractor,
};

const FEATURE_DIM: usize = 45;
const SAMPLE_RATE: u32 = 44100;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
struct Annotation {
    #[serde(rename = "fn")]
    filename: String,
    adult: String,
    name: String,
    date_recorded: String,
    call_type: String,
    rendition_num: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DynamicPhraseReport {
    total_vocalizations: usize,
    total_candidates: usize,
    total_atomic_phrases: usize,
    high_quality_phrases: usize,

    // Segmentation statistics
    avg_phrases_per_vocalization: f64,
    avg_phrase_duration_ms: f64,
    total_change_points: usize,

    // Cluster statistics
    avg_cluster_size: f64,
    avg_intra_similarity: f64,
    avg_inter_distance: f64,
    avg_separation: f64,
    avg_reuse: f64,

    // Duration distribution
    duration_distribution: HashMap<String, usize>,

    // Top phrases
    top_phrases: Vec<PhraseSummary>,

    // Call type distribution
    call_type_distribution: HashMap<String, usize>,

    // Processing time
    segmentation_time_sec: f64,
    clustering_time_sec: f64,
    total_time_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseSummary {
    phrase_id: usize,
    size: usize,
    avg_duration_ms: f64,
    intra_similarity: f64,
    separation: f64,
    reuse_score: f64,
    primary_call_type: String,
    unique_birds: usize,
    unique_files: usize,
    duration_range_ms: (f32, f32),
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║    Zebra Finch Dynamic Phrase Discovery (45D Change Point Detection)         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();

    // Configuration
    let data_dir = PathBuf::from(std::env::var("HOME").unwrap()).join("birdsong_analysis/data/zebra_finch/zebra_finch");
    let vocalizations_dir = data_dir.join("vocalizations");
    let annotations_path = data_dir.join("annotations.csv");

    // Create dynamic segmenter with zebra finch config
    let segmenter_config = DynamicSegmenterConfig::zebra_finch();
    let segmenter = DynamicSegmenter::new(segmenter_config.clone(), SAMPLE_RATE);

    println!("Configuration:");
    println!("  ├─ Data Directory: {:?}", data_dir);
    println!("  ├─ Frame Duration: {}ms", segmenter_config.frame_duration_ms);
    println!(
        "  ├─ Min Phrase Duration: {}ms",
        segmenter_config.min_phrase_duration_ms
    );
    println!(
        "  ├─ Max Phrase Duration: {}ms",
        segmenter_config.max_phrase_duration_ms
    );
    println!("  ├─ Change Threshold: {}", segmenter_config.change_threshold);
    println!("  ├─ Peak Prominence: {}", segmenter_config.peak_prominence);
    println!("  └─ Feature Dimension: {}D", FEATURE_DIM);
    println!();

    // ========================================================================
    // Phase 1: Load Annotations
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[1/4] Loading Annotations");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let annotations = load_annotations(&annotations_path)?;
    println!("Loaded {} annotations", annotations.len());

    // Count call types
    let mut call_type_counts: HashMap<String, usize> = HashMap::new();
    for ann in &annotations {
        *call_type_counts.entry(ann.call_type.clone()).or_insert(0) += 1;
    }

    println!("Call Type Distribution:");
    let mut sorted_types: Vec<_> = call_type_counts.iter().collect();
    sorted_types.sort_by(|a, b| b.1.cmp(a.1));
    for (call_type, count) in sorted_types {
        println!("  ├─ {}: {} samples", call_type, count);
    }
    println!();

    // ========================================================================
    // Phase 2: Dynamic Segmentation
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[2/4] Dynamic Segmentation (Change Point Detection)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let segmentation_start = Instant::now();
    let processed = Arc::new(AtomicUsize::new(0));
    let total_files = annotations.len();

    // Process a subset for demonstration
    let max_files = 1000;
    let annotations_subset: Vec<_> = annotations.into_iter().take(max_files).collect();

    // Segment each vocalization using dynamic segmentation
    let all_candidates: Vec<(DynamicPhraseCandidate, String, String)> = annotations_subset
        .par_iter()
        .flat_map(|ann| {
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 100 == 0 {
                println!("  Progress: {}/{} files", count + 1, max_files.min(total_files));
            }

            let audio_path = vocalizations_dir.join(&ann.filename);

            // Load audio
            if let Ok(audio) = load_audio(&audio_path) {
                if audio.len() < 1000 {
                    return Vec::new();
                }

                // Resample if needed
                let audio = resample_audio(&audio, 44100, SAMPLE_RATE);

                // Create feature extractor for this segment
                let extractor = Arc::new(std::sync::Mutex::new(ZooVoxFeatureExtractor::new(SAMPLE_RATE)));

                // Segment using dynamic approach
                let result = segmenter.segment(
                    &audio,
                    |frame, sr| {
                        // Convert frame to f64 for feature extraction
                        let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
                        let mut ext = extractor.lock().unwrap();
                        ext.extract_45d(&frame_f64).ok().map(|f| f.to_vector().to_vec())
                    },
                    &ann.filename,
                );

                // Add metadata to candidates
                result
                    .candidates
                    .into_iter()
                    .map(|cand| (cand, ann.call_type.clone(), ann.name.clone()))
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect();

    let segmentation_time = segmentation_start.elapsed();

    println!("\nSegmentation Complete:");
    println!("  ├─ Candidates Extracted: {}", all_candidates.len());
    println!("  ├─ Vocalizations Processed: {}", max_files.min(total_files));
    println!("  ├─ Time: {:.1}s", segmentation_time.as_secs_f64());
    println!(
        "  └─ Throughput: {:.1} candidates/sec",
        all_candidates.len() as f64 / segmentation_time.as_secs_f64().max(1.0)
    );
    println!();

    // ========================================================================
    // Phase 3: Discover Atomic Phrases via Clustering
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[3/4] Discovering Atomic Phrases (Acoustic Similarity Clustering)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let clustering_start = Instant::now();

    // Create similarity engine
    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    // Fit normalization
    {
        let n_samples = all_candidates.len().min(10000);
        let mut matrix = ndarray::Array2::<f64>::zeros((n_samples, FEATURE_DIM));
        for (i, (cand, _, _)) in all_candidates.iter().take(n_samples).enumerate() {
            for (j, &val) in cand.features.iter().enumerate() {
                matrix[[i, j]] = val;
            }
        }
        engine.fit_normalization(&matrix);
    }

    // Cluster using similarity threshold
    let similarity_threshold = 0.30; // Cosine distance threshold
    let min_occurrences = 3;

    println!("Building atomic phrase clusters...");
    println!("  ├─ Similarity Threshold: {:.2}", similarity_threshold);
    println!("  └─ Min Occurrences: {}", min_occurrences);
    println!();

    let mut atomic_phrases: Vec<AtomicPhraseCluster> = Vec::new();
    let mut assigned: Vec<bool> = vec![false; all_candidates.len()];

    for i in 0..all_candidates.len() {
        if assigned[i] {
            continue;
        }

        let mut cluster_indices = vec![i];
        assigned[i] = true;

        let query = Array1::from_vec(all_candidates[i].0.features.clone());

        // Find all neighbors
        for j in (i + 1)..all_candidates.len() {
            if !assigned[j] {
                let candidate = Array1::from_vec(all_candidates[j].0.features.clone());
                let dist = engine.distance(&query, &candidate);

                if dist < similarity_threshold {
                    cluster_indices.push(j);
                    assigned[j] = true;
                }
            }
        }

        // Only keep clusters with minimum occurrences
        if cluster_indices.len() >= min_occurrences {
            atomic_phrases.push(AtomicPhraseCluster {
                phrase_id: atomic_phrases.len(),
                member_indices: cluster_indices,
                centroid: vec![0.0; FEATURE_DIM], // Computed below
                intra_similarity: 0.0,            // Computed below
                inter_cluster_distance: 0.0,      // Computed below
                separation_score: 0.0,            // Computed below
            });
        }

        if (i + 1) % 5000 == 0 {
            println!(
                "  Progress: {}/{} candidates, {} phrases",
                i + 1,
                all_candidates.len(),
                atomic_phrases.len()
            );
        }
    }

    println!("  Discovered {} raw phrase clusters", atomic_phrases.len());

    // Compute quality metrics for each phrase
    println!("Computing quality metrics...");
    for phrase in atomic_phrases.iter_mut() {
        // Intra-cluster similarity
        let centroid = compute_centroid(&phrase.member_indices, &all_candidates);
        let mut total_sim = 0.0;
        let mut count = 0;

        for &idx in &phrase.member_indices {
            let member_features = Array1::from_vec(all_candidates[idx].0.features.clone());
            let sim = engine.similarity(&member_features, &Array1::from_vec(centroid.clone()));
            total_sim += sim;
            count += 1;
        }

        phrase.intra_similarity = if count > 0 { total_sim / count as f64 } else { 0.0 };
        phrase.centroid = centroid;
    }

    // Compute inter-cluster distances
    for i in 0..atomic_phrases.len() {
        let mut min_dist = f64::MAX;
        let centroid_i = Array1::from_vec(atomic_phrases[i].centroid.clone());

        for j in 0..atomic_phrases.len() {
            if i != j {
                let centroid_j = Array1::from_vec(atomic_phrases[j].centroid.clone());
                let dist = engine.distance(&centroid_i, &centroid_j);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }

        atomic_phrases[i].inter_cluster_distance = min_dist;
        atomic_phrases[i].separation_score = min_dist / (1.0 - atomic_phrases[i].intra_similarity + 0.001);
    }

    let clustering_time = clustering_start.elapsed();

    println!("\nClustering Complete:");
    println!("  ├─ Atomic Phrases: {}", atomic_phrases.len());
    println!("  └─ Time: {:.1}s", clustering_time.as_secs_f64());
    println!();

    // ========================================================================
    // Phase 4: Generate Report
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[4/4] Dynamic Phrase Analysis Report");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Sort by separation score
    let mut sorted_phrases: Vec<_> = atomic_phrases.iter().collect();
    sorted_phrases.sort_by(|a, b| b.separation_score.partial_cmp(&a.separation_score).unwrap());

    // High quality phrases
    let high_quality_phrases: Vec<_> = sorted_phrases
        .iter()
        .filter(|p| p.intra_similarity > 0.70 && p.separation_score > 1.2)
        .collect();

    // Compute statistics
    let avg_cluster_size = atomic_phrases.iter().map(|p| p.member_indices.len()).sum::<usize>() as f64
        / atomic_phrases.len().max(1) as f64;
    let avg_intra_similarity =
        atomic_phrases.iter().map(|p| p.intra_similarity).sum::<f64>() / atomic_phrases.len().max(1) as f64;
    let avg_inter_distance =
        atomic_phrases.iter().map(|p| p.inter_cluster_distance).sum::<f64>() / atomic_phrases.len().max(1) as f64;
    let avg_separation =
        atomic_phrases.iter().map(|p| p.separation_score).sum::<f64>() / atomic_phrases.len().max(1) as f64;

    // Duration statistics
    let durations: Vec<f32> = all_candidates.iter().map(|(c, _, _)| c.duration_ms).collect();
    let avg_duration = durations.iter().sum::<f32>() as f64 / durations.len().max(1) as f64;
    let avg_phrases_per_voc = all_candidates.len() as f64 / max_files.min(total_files) as f64;

    // Duration distribution
    let mut duration_dist: HashMap<String, usize> = HashMap::new();
    for d in &durations {
        let bucket = match *d {
            d if d < 50.0 => "0-50ms",
            d if d < 100.0 => "50-100ms",
            d if d < 200.0 => "100-200ms",
            d if d < 500.0 => "200-500ms",
            _ => "500ms+",
        }
        .to_string();
        *duration_dist.entry(bucket).or_insert(0) += 1;
    }

    // Reuse statistics
    let reuse_stats: Vec<f64> = atomic_phrases
        .iter()
        .map(|p| {
            let files: HashSet<_> = p
                .member_indices
                .iter()
                .map(|&idx| all_candidates[idx].0.source_file.clone())
                .collect();
            files.len() as f64
        })
        .collect();
    let avg_reuse = reuse_stats.iter().sum::<f64>() / atomic_phrases.len().max(1) as f64;

    println!("DYNAMIC PHRASE STATISTICS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Total Vocalizations: {}", max_files.min(total_files));
    println!("Total Phrase Candidates: {}", all_candidates.len());
    println!("Atomic Phrases Discovered: {}", atomic_phrases.len());
    println!(
        "High Quality Phrases (intra > 0.70, separation > 1.2): {}",
        high_quality_phrases.len()
    );
    println!();

    println!("Segmentation Metrics:");
    println!("  ├─ Avg Phrases per Vocalization: {:.1}", avg_phrases_per_voc);
    println!("  └─ Avg Phrase Duration: {:.1}ms", avg_duration);
    println!();

    println!("Duration Distribution:");
    let mut sorted_duration: Vec<_> = duration_dist.iter().collect();
    sorted_duration.sort_by_key(|(k, _)| match k.as_str() {
        "0-50ms" => 0,
        "50-100ms" => 1,
        "100-200ms" => 2,
        "200-500ms" => 3,
        _ => 4,
    });
    for (bucket, count) in sorted_duration {
        println!("  ├─ {}: {}", bucket, count);
    }
    println!();

    println!("Cluster Quality Metrics:");
    println!("  ├─ Avg Cluster Size: {:.1}", avg_cluster_size);
    println!("  ├─ Avg Intra-Cluster Similarity: {:.3}", avg_intra_similarity);
    println!("  ├─ Avg Inter-Cluster Distance: {:.3}", avg_inter_distance);
    println!("  ├─ Avg Separation Score: {:.2}", avg_separation);
    println!("  └─ Avg Reuse Score: {:.1} files/phrase", avg_reuse);
    println!();

    println!("TOP 10 ATOMIC PHRASES (by separation score)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let top_phrases: Vec<PhraseSummary> = sorted_phrases
        .iter()
        .take(10)
        .map(|p| {
            // Get call types for this phrase
            let mut call_types: HashMap<String, usize> = HashMap::new();
            let mut birds: HashSet<String> = HashSet::new();
            let mut files: HashSet<String> = HashSet::new();
            let mut durations: Vec<f32> = Vec::new();

            for &idx in &p.member_indices {
                let (cand, call_type, bird) = &all_candidates[idx];
                *call_types.entry(call_type.clone()).or_insert(0) += 1;
                birds.insert(bird.clone());
                files.insert(cand.source_file.clone());
                durations.push(cand.duration_ms);
            }

            let primary_call_type = call_types
                .iter()
                .max_by_key(|(_, &c)| c)
                .map(|(ct, _)| ct.clone())
                .unwrap_or_else(|| "Unknown".to_string());

            let min_dur = durations.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_dur = durations.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let avg_dur = durations.iter().sum::<f32>() / durations.len() as f32;

            PhraseSummary {
                phrase_id: p.phrase_id,
                size: p.member_indices.len(),
                avg_duration_ms: avg_dur as f64,
                intra_similarity: p.intra_similarity,
                separation: p.separation_score,
                reuse_score: files.len() as f64,
                primary_call_type,
                unique_birds: birds.len(),
                unique_files: files.len(),
                duration_range_ms: (min_dur, max_dur),
            }
        })
        .collect();

    for (i, summary) in top_phrases.iter().enumerate() {
        println!("{}. Phrase #{}", i + 1, summary.phrase_id);
        println!("   ├─ Size: {} segments", summary.size);
        println!(
            "   ├─ Duration: {:.1}ms (range: {:.1} - {:.1}ms)",
            summary.avg_duration_ms, summary.duration_range_ms.0, summary.duration_range_ms.1
        );
        println!("   ├─ Intra-Similarity: {:.3}", summary.intra_similarity);
        println!("   ├─ Separation: {:.2}", summary.separation);
        println!(
            "   ├─ Reuse: {} files, {} birds",
            summary.unique_files, summary.unique_birds
        );
        println!("   └─ Primary Call Type: {}", summary.primary_call_type);
        println!();
    }

    // Save report
    let total_time = total_start.elapsed();

    // Call type distribution in atomic phrases
    let mut phrase_call_distribution: HashMap<String, usize> = HashMap::new();
    for phrase in &atomic_phrases {
        for &idx in &phrase.member_indices {
            let (_, call_type, _) = &all_candidates[idx];
            *phrase_call_distribution.entry(call_type.clone()).or_insert(0) += 1;
        }
    }

    let report = DynamicPhraseReport {
        total_vocalizations: max_files.min(total_files),
        total_candidates: all_candidates.len(),
        total_atomic_phrases: atomic_phrases.len(),
        high_quality_phrases: high_quality_phrases.len(),
        avg_phrases_per_vocalization: avg_phrases_per_voc,
        avg_phrase_duration_ms: avg_duration,
        total_change_points: 0, // Not tracked at aggregate level
        avg_cluster_size,
        avg_intra_similarity,
        avg_inter_distance,
        avg_separation,
        avg_reuse,
        duration_distribution: duration_dist,
        top_phrases,
        call_type_distribution: phrase_call_distribution,
        segmentation_time_sec: segmentation_time.as_secs_f64(),
        clustering_time_sec: clustering_time.as_secs_f64(),
        total_time_sec: total_time.as_secs_f64(),
    };

    std::fs::create_dir_all("zebra_finch_analysis")?;
    let output_path = "zebra_finch_analysis/dynamic_phrases_report.json";
    let file = File::create(output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &report)?;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("  Vocalizations: {}", max_files.min(total_files));
    println!("  Phrase Candidates: {}", all_candidates.len());
    println!("  Atomic Phrases: {}", atomic_phrases.len());
    println!("  High Quality: {}", high_quality_phrases.len());
    println!("  Avg Duration: {:.1}ms", avg_duration);
    println!("  Total Time: {:.1}s", total_time.as_secs_f64());
    println!();
    println!("  Report saved to: {}", output_path);

    Ok(())
}

// ============================================================================
// HELPER STRUCTURES
// ============================================================================

struct AtomicPhraseCluster {
    phrase_id: usize,
    member_indices: Vec<usize>,
    centroid: Vec<f64>,
    intra_similarity: f64,
    inter_cluster_distance: f64,
    separation_score: f64,
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

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
        hound::SampleFormat::Float => reader.into_samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let max_val = 2_i32.pow((spec.bits_per_sample - 1) as u32) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    Ok(audio)
}

fn resample_audio(audio: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return audio.to_vec();
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let new_len = (audio.len() as f64 * ratio) as usize;

    (0..new_len)
        .map(|i| {
            let src_idx = i as f64 / ratio;
            let idx = src_idx as usize;
            let frac = src_idx - idx as f64;

            let a = audio.get(idx).copied().unwrap_or(0.0);
            let b = audio.get(idx + 1).copied().unwrap_or(a);

            a * (1.0 - frac as f32) + b * frac as f32
        })
        .collect()
}

fn compute_centroid(indices: &[usize], candidates: &[(DynamicPhraseCandidate, String, String)]) -> Vec<f64> {
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
