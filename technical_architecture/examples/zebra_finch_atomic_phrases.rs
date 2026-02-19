//! Zebra Finch Atomic Phrase Discovery using 45D Acoustic Similarity Engine
//!
//! This tool discovers "atomic phrases" - clusters of acoustically similar
//! segments that represent reusable vocal units in zebra finch communication.
//!
//! Atomic Phrase Criteria:
//! 1. High internal coherence (members are similar to each other)
//! 2. Well-separated from other clusters (distinct from other phrases)
//! 3. Represents a reusable vocal unit (appears across multiple vocalizations)
//!
//! Usage:
//!   cargo run --release --example zebra_finch_atomic_phrases

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
use technical_architecture::{AcousticSimilarityEngine, SimilarityMetric, ZooVoxFeatureExtractor};

const FEATURE_DIM: usize = 45;
const SAMPLE_RATE: u32 = 44100;
const SEGMENT_DURATION_MS: f64 = 50.0; // 50ms segments for phrase candidates
const SIMILARITY_THRESHOLD: f64 = 0.70; // For clustering (lowered from 0.85)
const MIN_CLUSTER_SIZE: usize = 3; // Minimum samples to be an atomic phrase (lowered from 5)

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
struct Annotation {
    #[serde(rename = "fn")]
    filename: String,
    adult: String, // "True" or "False" as strings
    name: String,
    date_recorded: String,
    call_type: String,
    rendition_num: String,
}

#[derive(Debug, Clone)]
struct PhraseCandidate {
    id: String,
    features: Vec<f64>,
    source_file: String,
    call_type: String,
    bird_name: String,
    segment_idx: usize,
    start_ms: f64,
    duration_ms: f64,
}

#[derive(Debug, Clone)]
struct AtomicPhrase {
    phrase_id: usize,
    centroid: Vec<f64>,
    members: Vec<String>, // Candidate IDs
    call_types: HashMap<String, usize>,
    bird_names: HashMap<String, usize>,
    source_files: HashSet<String>,

    // Quality metrics
    intra_cluster_similarity: f64,
    inter_cluster_distance: f64,
    separation_score: f64,
    reuse_score: f64, // How many different vocalizations use this phrase
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AtomicPhraseReport {
    total_candidates: usize,
    total_atomic_phrases: usize,
    high_quality_phrases: usize,

    // Phrase statistics
    avg_cluster_size: f64,
    avg_intra_similarity: f64,
    avg_inter_distance: f64,
    avg_separation: f64,
    avg_reuse: f64,

    // Top phrases
    top_phrases: Vec<PhraseSummary>,

    // Call type distribution
    call_type_distribution: HashMap<String, usize>,

    // Processing time
    extraction_time_sec: f64,
    clustering_time_sec: f64,
    total_time_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhraseSummary {
    phrase_id: usize,
    size: usize,
    intra_similarity: f64,
    separation: f64,
    reuse_score: f64,
    primary_call_type: String,
    unique_birds: usize,
    unique_files: usize,
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║        Zebra Finch Atomic Phrase Discovery (45D Acoustic Similarity)         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();

    // Configuration
    let data_dir = PathBuf::from(std::env::var("HOME").unwrap())
        .join("birdsong_analysis/data/zebra_finch/zebra_finch");
    let vocalizations_dir = data_dir.join("vocalizations");
    let annotations_path = data_dir.join("annotations.csv");

    println!("Configuration:");
    println!("  ├─ Data Directory: {:?}", data_dir);
    println!("  ├─ Segment Duration: {}ms", SEGMENT_DURATION_MS);
    println!("  ├─ Similarity Threshold: {}", SIMILARITY_THRESHOLD);
    println!("  ├─ Min Cluster Size: {}", MIN_CLUSTER_SIZE);
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
    // Phase 2: Extract Phrase Candidates
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[2/4] Extracting Phrase Candidates (45D Features)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let extraction_start = Instant::now();
    let processed = Arc::new(AtomicUsize::new(0));
    let total_files = annotations.len();

    // Process a subset for demonstration (limit to 1000 files for speed)
    let max_files = 1000;
    let annotations_subset: Vec<_> = annotations.into_iter().take(max_files).collect();

    let candidates: Vec<PhraseCandidate> = annotations_subset
        .par_iter()
        .flat_map(|ann| {
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 100 == 0 {
                println!(
                    "  Progress: {}/{} files",
                    count + 1,
                    max_files.min(total_files)
                );
            }

            let audio_path = vocalizations_dir.join(&ann.filename);
            extract_phrase_candidates(&audio_path, ann).unwrap_or_default()
        })
        .collect();

    let extraction_time = extraction_start.elapsed();

    println!("\nExtraction Complete:");
    println!("  ├─ Candidates Extracted: {}", candidates.len());
    println!("  ├─ Time: {:.1}s", extraction_time.as_secs_f64());
    println!(
        "  └─ Throughput: {:.1} candidates/sec",
        candidates.len() as f64 / extraction_time.as_secs_f64().max(1.0)
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
        let n_samples = candidates.len().min(10000);
        let mut matrix = ndarray::Array2::<f64>::zeros((n_samples, FEATURE_DIM));
        for (i, cand) in candidates.iter().take(n_samples).enumerate() {
            for (j, &val) in cand.features.iter().enumerate() {
                matrix[[i, j]] = val;
            }
        }
        engine.fit_normalization(&matrix);
    }

    // Build clusters using streaming approach
    let mut atomic_phrases: Vec<AtomicPhrase> = Vec::new();

    println!("Building atomic phrase clusters...");

    for (i, candidate) in candidates.iter().enumerate() {
        let query = Array1::from_vec(candidate.features.clone());

        // Find best matching existing phrase
        let mut best_phrase_idx: Option<usize> = None;
        let mut best_sim = 0.0;

        for (idx, phrase) in atomic_phrases.iter().enumerate() {
            let centroid = Array1::from_vec(phrase.centroid.clone());
            let sim = engine.similarity(&query, &centroid);

            if sim >= SIMILARITY_THRESHOLD && sim > best_sim {
                best_sim = sim;
                best_phrase_idx = Some(idx);
            }
        }

        if let Some(idx) = best_phrase_idx {
            // Add to existing phrase (update centroid incrementally)
            let phrase = &mut atomic_phrases[idx];
            let n = phrase.members.len() + 1;

            // Incremental centroid update
            for (j, &val) in candidate.features.iter().enumerate() {
                phrase.centroid[j] += (val - phrase.centroid[j]) / n as f64;
            }

            phrase.members.push(candidate.id.clone());
            *phrase
                .call_types
                .entry(candidate.call_type.clone())
                .or_insert(0) += 1;
            *phrase
                .bird_names
                .entry(candidate.bird_name.clone())
                .or_insert(0) += 1;
            phrase.source_files.insert(candidate.source_file.clone());
        } else {
            // Create new phrase
            let mut call_types = HashMap::new();
            call_types.insert(candidate.call_type.clone(), 1);

            let mut bird_names = HashMap::new();
            bird_names.insert(candidate.bird_name.clone(), 1);

            let mut source_files = HashSet::new();
            source_files.insert(candidate.source_file.clone());

            atomic_phrases.push(AtomicPhrase {
                phrase_id: atomic_phrases.len(),
                centroid: candidate.features.clone(),
                members: vec![candidate.id.clone()],
                call_types,
                bird_names,
                source_files,
                intra_cluster_similarity: 1.0, // Will be computed later
                inter_cluster_distance: 0.0,
                separation_score: 0.0,
                reuse_score: 0.0,
            });
        }

        if (i + 1) % 5000 == 0 {
            println!(
                "  Progress: {}/{} candidates, {} phrases",
                i + 1,
                candidates.len(),
                atomic_phrases.len()
            );
        }
    }

    println!(
        "  Discovered {} raw phrases from {} candidates",
        atomic_phrases.len(),
        candidates.len()
    );

    // Filter to atomic phrases (min cluster size)
    let mut atomic_phrases: Vec<AtomicPhrase> = atomic_phrases
        .into_iter()
        .filter(|p| p.members.len() >= MIN_CLUSTER_SIZE)
        .collect();

    println!(
        "  Filtered to {} atomic phrases (min size {})",
        atomic_phrases.len(),
        MIN_CLUSTER_SIZE
    );

    // Compute quality metrics for each phrase
    println!("\nComputing quality metrics...");
    for phrase in atomic_phrases.iter_mut() {
        // Intra-cluster similarity
        let centroid = Array1::from_vec(phrase.centroid.clone());
        let mut total_sim = 0.0;
        let mut count = 0;

        for member_id in &phrase.members {
            if let Some(candidate) = candidates.iter().find(|c| &c.id == member_id) {
                let member_features = Array1::from_vec(candidate.features.clone());
                total_sim += engine.similarity(&member_features, &centroid);
                count += 1;
            }
        }

        phrase.intra_cluster_similarity = if count > 0 {
            total_sim / count as f64
        } else {
            0.0
        };

        // Reuse score (how many different files use this phrase)
        phrase.reuse_score = phrase.source_files.len() as f64;
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
        atomic_phrases[i].separation_score =
            min_dist / (1.0 - atomic_phrases[i].intra_cluster_similarity + 0.001);
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
    println!("[4/4] Atomic Phrase Analysis Report");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Sort by separation score (higher = better atomic phrase)
    let mut sorted_phrases: Vec<_> = atomic_phrases.iter().collect();
    sorted_phrases.sort_by(|a, b| b.separation_score.partial_cmp(&a.separation_score).unwrap());

    // High quality phrases (good separation + good internal coherence)
    let high_quality_phrases: Vec<_> = sorted_phrases
        .iter()
        .filter(|p| p.intra_cluster_similarity > 0.85 && p.separation_score > 1.5)
        .collect();

    // Compute statistics
    let avg_cluster_size = atomic_phrases
        .iter()
        .map(|p| p.members.len())
        .sum::<usize>() as f64
        / atomic_phrases.len().max(1) as f64;
    let avg_intra_similarity = atomic_phrases
        .iter()
        .map(|p| p.intra_cluster_similarity)
        .sum::<f64>()
        / atomic_phrases.len().max(1) as f64;
    let avg_inter_distance = atomic_phrases
        .iter()
        .map(|p| p.inter_cluster_distance)
        .sum::<f64>()
        / atomic_phrases.len().max(1) as f64;
    let avg_separation = atomic_phrases
        .iter()
        .map(|p| p.separation_score)
        .sum::<f64>()
        / atomic_phrases.len().max(1) as f64;
    let avg_reuse = atomic_phrases.iter().map(|p| p.reuse_score).sum::<f64>()
        / atomic_phrases.len().max(1) as f64;

    // Call type distribution in atomic phrases
    let mut phrase_call_distribution: HashMap<String, usize> = HashMap::new();
    for phrase in &atomic_phrases {
        for (call_type, _) in &phrase.call_types {
            *phrase_call_distribution
                .entry(call_type.clone())
                .or_insert(0) += 1;
        }
    }

    println!("ATOMIC PHRASE STATISTICS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Total Phrase Candidates: {}", candidates.len());
    println!("Atomic Phrases Discovered: {}", atomic_phrases.len());
    println!(
        "High Quality Phrases (intra > 0.85, separation > 1.5): {}",
        high_quality_phrases.len()
    );
    println!();

    println!("Quality Metrics:");
    println!("  ├─ Avg Cluster Size: {:.1}", avg_cluster_size);
    println!(
        "  ├─ Avg Intra-Cluster Similarity: {:.3}",
        avg_intra_similarity
    );
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
            let primary_call_type = p
                .call_types
                .iter()
                .max_by_key(|(_, &c)| c)
                .map(|(ct, _)| ct.clone())
                .unwrap_or_else(|| "Unknown".to_string());

            PhraseSummary {
                phrase_id: p.phrase_id,
                size: p.members.len(),
                intra_similarity: p.intra_cluster_similarity,
                separation: p.separation_score,
                reuse_score: p.reuse_score,
                primary_call_type,
                unique_birds: p.bird_names.len(),
                unique_files: p.source_files.len(),
            }
        })
        .collect();

    for (i, summary) in top_phrases.iter().enumerate() {
        println!("{}. Phrase #{}", i + 1, summary.phrase_id);
        println!("   ├─ Size: {} segments", summary.size);
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

    let report = AtomicPhraseReport {
        total_candidates: candidates.len(),
        total_atomic_phrases: atomic_phrases.len(),
        high_quality_phrases: high_quality_phrases.len(),
        avg_cluster_size,
        avg_intra_similarity,
        avg_inter_distance,
        avg_separation,
        avg_reuse,
        top_phrases,
        call_type_distribution: phrase_call_distribution,
        extraction_time_sec: extraction_time.as_secs_f64(),
        clustering_time_sec: clustering_time.as_secs_f64(),
        total_time_sec: total_time.as_secs_f64(),
    };

    std::fs::create_dir_all("zebra_finch_analysis")?;
    let output_path = "zebra_finch_analysis/atomic_phrases_report.json";
    let file = File::create(output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &report)?;

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("  Total Candidates: {}", candidates.len());
    println!("  Atomic Phrases: {}", atomic_phrases.len());
    println!("  High Quality: {}", high_quality_phrases.len());
    println!("  Total Time: {:.1}s", total_time.as_secs_f64());
    println!();
    println!("  Report saved to: {}", output_path);

    Ok(())
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

fn extract_phrase_candidates(
    audio_path: &Path,
    annotation: &Annotation,
) -> Result<Vec<PhraseCandidate>, Box<dyn std::error::Error>> {
    // Load audio file using hound
    let reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();
    let file_sample_rate = spec.sample_rate;

    // Convert to f64
    let audio: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .map(|s| s as f64)
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = 2_i32.pow((spec.bits_per_sample - 1) as u32) as f64;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f64 / max_val)
                .collect()
        }
    };

    if audio.len() < 1000 {
        return Ok(Vec::new());
    }

    // Resample if needed
    let audio = if file_sample_rate != SAMPLE_RATE {
        simple_resample(&audio, file_sample_rate, SAMPLE_RATE)
    } else {
        audio
    };

    // Segment into phrase candidates
    let segment_samples = (SAMPLE_RATE as f64 * SEGMENT_DURATION_MS / 1000.0) as usize;
    let hop_samples = segment_samples / 2; // 50% overlap

    let mut candidates = Vec::new();
    let mut segment_idx = 0;

    let mut extractor = ZooVoxFeatureExtractor::new(SAMPLE_RATE);

    for start in (0..audio.len().saturating_sub(segment_samples)).step_by(hop_samples) {
        let segment = &audio[start..start + segment_samples.min(audio.len() - start)];

        if segment.len() < segment_samples / 2 {
            continue;
        }

        // Extract 45D features
        if let Ok(features) = extractor.extract_45d(segment) {
            let start_ms = start as f64 / SAMPLE_RATE as f64 * 1000.0;

            candidates.push(PhraseCandidate {
                id: format!(
                    "{}_seg{}",
                    annotation.filename.replace(".wav", ""),
                    segment_idx
                ),
                features: features.to_vector().to_vec(),
                source_file: annotation.filename.clone(),
                call_type: annotation.call_type.clone(),
                bird_name: annotation.name.clone(),
                segment_idx,
                start_ms,
                duration_ms: SEGMENT_DURATION_MS,
            });

            segment_idx += 1;
        }
    }

    Ok(candidates)
}

fn simple_resample(audio: &[f64], from_rate: u32, to_rate: u32) -> Vec<f64> {
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

            a * (1.0 - frac) + b * frac
        })
        .collect()
}
