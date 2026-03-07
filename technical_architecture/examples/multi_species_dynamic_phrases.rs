//! Multi-Species Dynamic Phrase Discovery using Change Point Detection (CPD)
//!
//! Discovers atomic phrases across multiple species using dynamic segmentation.
//! Supports: zebra_finch, egyptian_bat, marmoset, dolphin
//!
//! Usage:
//!   cargo run --release --example multi_species_dynamic_phrases -- zebra_finch
//!   cargo run --release --example multi_species_dynamic_phrases -- egyptian_bat
//!   cargo run --release --example multi_species_dynamic_phrases -- marmoset
//!   cargo run --release --example multi_species_dynamic_phrases -- dolphin

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use ndarray::Array1;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use technical_architecture::{
    AcousticSimilarityEngine, DynamicPhraseCandidate, DynamicSegmenter, DynamicSegmenterConfig, SimilarityMetric,
    ZooVoxFeatureExtractor,
};

const FEATURE_DIM: usize = 45;

// ============================================================================
// SPECIES CONFIGURATIONS
// ============================================================================

#[derive(Debug, Clone)]
struct SpeciesConfig {
    name: String,
    data_dir: PathBuf,
    sample_rate: u32,
    segmenter_config: DynamicSegmenterConfig,
    annotation_format: AnnotationFormat,
    max_files: usize,
    similarity_threshold: f32,
    min_occurrences: usize,
}

#[derive(Debug, Clone)]
enum AnnotationFormat {
    ZebraFinch,      // CSV with fn, call_type columns
    EgyptianBat,     // CSV with File Name, call_type columns
    Marmoset,        // CSV with File Name, Call Type columns
    DolphinWhistles, // No annotations, numbered files
    RawAudio,        // Just audio files, no annotations
}

impl SpeciesConfig {
    fn zebra_finch(base_dir: &Path) -> Self {
        Self {
            name: "zebra_finch".to_string(),
            data_dir: base_dir.join("zebra_finch/zebra_finch"),
            sample_rate: 44100,
            segmenter_config: DynamicSegmenterConfig::zebra_finch(),
            annotation_format: AnnotationFormat::ZebraFinch,
            max_files: 1000,
            similarity_threshold: 0.30,
            min_occurrences: 3,
        }
    }

    fn egyptian_bat(base_dir: &Path) -> Self {
        Self {
            name: "egyptian_bat".to_string(),
            data_dir: base_dir.join("egyptian_fruit_bat_10k"),
            sample_rate: 250000, // High sample rate for bat calls
            segmenter_config: DynamicSegmenterConfig::bat(),
            annotation_format: AnnotationFormat::EgyptianBat,
            max_files: 1000,
            similarity_threshold: 0.35,
            min_occurrences: 3,
        }
    }

    fn marmoset(base_dir: &Path) -> Self {
        Self {
            name: "marmoset".to_string(),
            data_dir: base_dir.join("marmosets"),
            sample_rate: 44100,
            segmenter_config: DynamicSegmenterConfig::marmoset(),
            annotation_format: AnnotationFormat::Marmoset,
            max_files: 1000,
            similarity_threshold: 0.30,
            min_occurrences: 3,
        }
    }

    fn dolphin(base_dir: &Path) -> Self {
        let mut config = DynamicSegmenterConfig::dolphin();
        config.min_phrase_duration_ms = 50.0; // Shorter for whistles

        Self {
            name: "dolphin".to_string(),
            data_dir: base_dir.join("bottlenose_dolphins"),
            sample_rate: 44100,
            segmenter_config: config,
            annotation_format: AnnotationFormat::DolphinWhistles,
            max_files: 303, // All available whistles
            similarity_threshold: 0.30,
            min_occurrences: 2,
        }
    }
}

// ============================================================================
// ANNOTATION STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
struct ZebraFinchAnnotation {
    #[serde(rename = "fn")]
    filename: String,
    call_type: String,
}

#[derive(Debug, Clone, Deserialize)]
struct BatAnnotation {
    #[serde(rename = "File Name")]
    filename: String,
    call_type: String,
}

#[derive(Debug, Clone, Deserialize)]
struct MarmosetAnnotation {
    #[serde(rename = "File Name")]
    filename: String,
    #[serde(rename = "Call Type")]
    call_type: String,
}

// ============================================================================
// REPORT STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MultiSpeciesReport {
    species: String,
    total_files: usize,
    total_candidates: usize,
    total_atomic_phrases: usize,
    high_quality_phrases: usize,

    avg_phrases_per_file: f64,
    avg_phrase_duration_ms: f64,

    duration_distribution: HashMap<String, usize>,
    call_type_distribution: HashMap<String, usize>,

    avg_cluster_size: f64,
    avg_intra_similarity: f64,
    avg_separation: f64,
    avg_reuse: f64,

    top_phrases: Vec<PhraseSummary>,

    processing_time_sec: f64,
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
    let args: Vec<String> = env::args().collect();
    let species_arg = args.get(1).map(|s| s.as_str()).unwrap_or("zebra_finch");

    let base_dir = PathBuf::from(std::env::var("HOME").unwrap()).join("birdsong_analysis/data");

    let config = match species_arg {
        "zebra_finch" | "finch" => SpeciesConfig::zebra_finch(&base_dir),
        "egyptian_bat" | "bat" => SpeciesConfig::egyptian_bat(&base_dir),
        "marmoset" => SpeciesConfig::marmoset(&base_dir),
        "dolphin" => SpeciesConfig::dolphin(&base_dir),
        _ => {
            eprintln!(
                "Unknown species: {}. Options: zebra_finch, egyptian_bat, marmoset, dolphin",
                species_arg
            );
            std::process::exit(1);
        }
    };

    run_discovery(config)
}

fn run_discovery(config: SpeciesConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║    {} Dynamic Phrase Discovery (45D CPD)                         ║",
        pad_center(&config.name, 33)
    );
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();

    // Create segmenter
    let segmenter = DynamicSegmenter::new(config.segmenter_config.clone(), config.sample_rate);

    println!("Configuration:");
    println!("  ├─ Data Directory: {:?}", config.data_dir);
    println!("  ├─ Sample Rate: {}Hz", config.sample_rate);
    println!("  ├─ Frame Duration: {}ms", config.segmenter_config.frame_duration_ms);
    println!(
        "  ├─ Min Phrase Duration: {}ms",
        config.segmenter_config.min_phrase_duration_ms
    );
    println!(
        "  ├─ Max Phrase Duration: {}ms",
        config.segmenter_config.max_phrase_duration_ms
    );
    println!("  ├─ Change Threshold: {}", config.segmenter_config.change_threshold);
    println!("  └─ Max Files: {}", config.max_files);
    println!();

    // ========================================================================
    // Phase 1: Load Files
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[1/3] Loading Audio Files");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let files = load_files(&config)?;
    println!("Loaded {} files", files.len());

    if !files.is_empty() {
        // Show call type distribution if available
        let call_types: HashMap<String, usize> =
            files
                .iter()
                .filter(|f| f.call_type != "unknown")
                .fold(HashMap::new(), |mut acc, f| {
                    *acc.entry(f.call_type.clone()).or_insert(0) += 1;
                    acc
                });

        if !call_types.is_empty() {
            println!("Call Type Distribution:");
            let mut sorted: Vec<_> = call_types.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));
            for (ct, count) in sorted.iter().take(10) {
                println!("  ├─ {}: {}", ct, count);
            }
        }
    }
    println!();

    // ========================================================================
    // Phase 2: Dynamic Segmentation
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[2/3] Dynamic Segmentation (Change Point Detection)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let processed = Arc::new(AtomicUsize::new(0));
    let total_files = files.len();

    let all_candidates: Vec<(DynamicPhraseCandidate, String)> = files
        .par_iter()
        .flat_map(|file_info| {
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 100 == 0 {
                println!("  Progress: {}/{} files", count + 1, total_files);
            }

            // Load audio
            if let Ok(audio) = load_audio(&file_info.path) {
                if audio.len() < 500 {
                    return Vec::new();
                }

                // Resample if needed
                let audio = resample_audio(&audio, file_info.sample_rate, config.sample_rate);

                // Create feature extractor
                let extractor = Arc::new(std::sync::Mutex::new(ZooVoxFeatureExtractor::new(config.sample_rate)));

                // Segment using dynamic approach
                let result = segmenter.segment(
                    &audio,
                    |frame, sr| {
                        let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
                        let mut ext = extractor.lock().unwrap();
                        ext.extract_45d(&frame_f64).ok().map(|f| f.to_vector().to_vec())
                    },
                    &file_info.filename,
                );

                result
                    .candidates
                    .into_iter()
                    .map(|cand| (cand, file_info.call_type.clone()))
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect();

    println!("\nSegmentation Complete:");
    println!("  ├─ Candidates Extracted: {}", all_candidates.len());
    println!("  ├─ Files Processed: {}", total_files);
    println!("  └─ Time: {:.1}s", total_start.elapsed().as_secs_f64());
    println!();

    // ========================================================================
    // Phase 3: Discover Atomic Phrases
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[3/3] Discovering Atomic Phrases");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Create similarity engine
    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    // Fit normalization
    if !all_candidates.is_empty() {
        let n_samples = all_candidates.len().min(10000);
        let mut matrix = ndarray::Array2::<f64>::zeros((n_samples, FEATURE_DIM));
        for (i, (cand, _)) in all_candidates.iter().take(n_samples).enumerate() {
            for (j, &val) in cand.features.iter().enumerate() {
                matrix[[i, j]] = val;
            }
        }
        engine.fit_normalization(&matrix);
    }

    println!("Building atomic phrase clusters...");
    println!("  ├─ Similarity Threshold: {:.2}", config.similarity_threshold);
    println!("  └─ Min Occurrences: {}", config.min_occurrences);
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

        for j in (i + 1)..all_candidates.len() {
            if !assigned[j] {
                let candidate = Array1::from_vec(all_candidates[j].0.features.clone());
                let dist = engine.distance(&query, &candidate);

                if dist < config.similarity_threshold as f64 {
                    cluster_indices.push(j);
                    assigned[j] = true;
                }
            }
        }

        if cluster_indices.len() >= config.min_occurrences {
            atomic_phrases.push(AtomicPhraseCluster {
                phrase_id: atomic_phrases.len(),
                member_indices: cluster_indices,
                centroid: vec![0.0; FEATURE_DIM],
                intra_similarity: 0.0,
                inter_cluster_distance: 0.0,
                separation_score: 0.0,
            });
        }

        if (i + 1) % 2000 == 0 && i > 0 {
            println!(
                "  Progress: {}/{} candidates, {} phrases",
                i + 1,
                all_candidates.len(),
                atomic_phrases.len()
            );
        }
    }

    println!("  Discovered {} raw phrase clusters", atomic_phrases.len());

    // Compute quality metrics
    println!("Computing quality metrics...");
    for phrase in atomic_phrases.iter_mut() {
        let centroid = compute_centroid(&phrase.member_indices, &all_candidates);
        let mut total_sim = 0.0;

        for &idx in &phrase.member_indices {
            let member_features = Array1::from_vec(all_candidates[idx].0.features.clone());
            let sim = engine.similarity(&member_features, &Array1::from_vec(centroid.clone()));
            total_sim += sim;
        }

        phrase.intra_similarity = total_sim / phrase.member_indices.len() as f64;
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

    let total_time = total_start.elapsed();

    // ========================================================================
    // Generate Report
    // ========================================================================
    generate_report(
        &config,
        &all_candidates,
        &atomic_phrases,
        total_files,
        total_time.as_secs_f64(),
    )
}

// ============================================================================
// HELPER STRUCTURES
// ============================================================================

struct FileInfo {
    path: PathBuf,
    filename: String,
    call_type: String,
    sample_rate: u32,
}

struct AtomicPhraseCluster {
    phrase_id: usize,
    member_indices: Vec<usize>,
    centroid: Vec<f64>,
    intra_similarity: f64,
    inter_cluster_distance: f64,
    separation_score: f64,
}

// ============================================================================
// FILE LOADING
// ============================================================================

fn load_files(config: &SpeciesConfig) -> Result<Vec<FileInfo>, Box<dyn std::error::Error>> {
    match config.annotation_format {
        AnnotationFormat::ZebraFinch => load_zebra_finch_files(config),
        AnnotationFormat::EgyptianBat => load_bat_files(config),
        AnnotationFormat::Marmoset => load_marmoset_files(config),
        AnnotationFormat::DolphinWhistles => load_dolphin_files(config),
        AnnotationFormat::RawAudio => load_raw_audio_files(config),
    }
}

fn load_zebra_finch_files(config: &SpeciesConfig) -> Result<Vec<FileInfo>, Box<dyn std::error::Error>> {
    let annotations_path = config.data_dir.join("annotations.csv");
    let vocalizations_dir = config.data_dir.join("vocalizations");

    let file = File::open(annotations_path)?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::Reader::from_reader(reader);

    let mut files = Vec::new();
    for result in csv_reader.deserialize() {
        let ann: ZebraFinchAnnotation = result?;
        files.push(FileInfo {
            path: vocalizations_dir.join(&ann.filename),
            filename: ann.filename,
            call_type: ann.call_type,
            sample_rate: 44100,
        });
    }

    Ok(files.into_iter().take(config.max_files).collect())
}

fn load_bat_files(config: &SpeciesConfig) -> Result<Vec<FileInfo>, Box<dyn std::error::Error>> {
    let annotations_path = config.data_dir.join("annotations_1k_subset_with_call_types.csv");
    let audio_dir = config.data_dir.join("audio");

    let file = File::open(annotations_path)?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::Reader::from_reader(reader);

    let mut files = Vec::new();
    for result in csv_reader.deserialize() {
        let ann: BatAnnotation = result?;
        files.push(FileInfo {
            path: audio_dir.join(&ann.filename),
            filename: ann.filename,
            call_type: format!("call_type_{}", ann.call_type),
            sample_rate: 250000,
        });
    }

    Ok(files.into_iter().take(config.max_files).collect())
}

fn load_marmoset_files(config: &SpeciesConfig) -> Result<Vec<FileInfo>, Box<dyn std::error::Error>> {
    let annotations_path = config.data_dir.join("marmoset_annotations.csv");

    let file = File::open(annotations_path)?;
    let reader = BufReader::new(file);
    let mut csv_reader = csv::Reader::from_reader(reader);

    let mut files = Vec::new();
    for result in csv_reader.deserialize() {
        let ann: MarmosetAnnotation = result?;
        files.push(FileInfo {
            path: PathBuf::from(&ann.filename),
            filename: PathBuf::from(&ann.filename)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string(),
            call_type: ann.call_type,
            sample_rate: 44100,
        });
    }

    Ok(files.into_iter().take(config.max_files).collect())
}

fn load_dolphin_files(config: &SpeciesConfig) -> Result<Vec<FileInfo>, Box<dyn std::error::Error>> {
    let whistles_dir = config.data_dir.join("single_whistles");

    let mut files = Vec::new();
    for entry in std::fs::read_dir(whistles_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map(|e| e == "wav").unwrap_or(false) {
            files.push(FileInfo {
                filename: path.file_name().unwrap().to_string_lossy().to_string(),
                path,
                call_type: "whistle".to_string(),
                sample_rate: 44100,
            });
        }
    }

    Ok(files.into_iter().take(config.max_files).collect())
}

fn load_raw_audio_files(config: &SpeciesConfig) -> Result<Vec<FileInfo>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(&config.data_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map(|e| e == "wav" || e == "flac").unwrap_or(false) {
            files.push(FileInfo {
                filename: path.file_name().unwrap().to_string_lossy().to_string(),
                path,
                call_type: "unknown".to_string(),
                sample_rate: config.sample_rate,
            });
        }
    }

    Ok(files.into_iter().take(config.max_files).collect())
}

// ============================================================================
// REPORT GENERATION
// ============================================================================

fn generate_report(
    config: &SpeciesConfig,
    all_candidates: &[(DynamicPhraseCandidate, String)],
    atomic_phrases: &[AtomicPhraseCluster],
    total_files: usize,
    total_time_sec: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut sorted_phrases: Vec<_> = atomic_phrases.iter().collect();
    sorted_phrases.sort_by(|a, b| b.separation_score.partial_cmp(&a.separation_score).unwrap());

    let high_quality: Vec<_> = sorted_phrases
        .iter()
        .filter(|p| p.intra_similarity > 0.70 && p.separation_score > 1.2)
        .collect();

    let avg_cluster_size = atomic_phrases.iter().map(|p| p.member_indices.len()).sum::<usize>() as f64
        / atomic_phrases.len().max(1) as f64;
    let avg_intra = atomic_phrases.iter().map(|p| p.intra_similarity).sum::<f64>() / atomic_phrases.len().max(1) as f64;
    let avg_sep = atomic_phrases.iter().map(|p| p.separation_score).sum::<f64>() / atomic_phrases.len().max(1) as f64;

    let durations: Vec<f32> = all_candidates.iter().map(|(c, _)| c.duration_ms).collect();
    let avg_duration = durations.iter().sum::<f32>() as f64 / durations.len().max(1) as f64;

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

    let mut call_type_dist: HashMap<String, usize> = HashMap::new();
    for phrase in atomic_phrases {
        for &idx in &phrase.member_indices {
            let (_, call_type) = &all_candidates[idx];
            *call_type_dist.entry(call_type.clone()).or_insert(0) += 1;
        }
    }

    let top_phrases: Vec<PhraseSummary> = sorted_phrases
        .iter()
        .take(10)
        .map(|p| {
            let mut call_types: HashMap<String, usize> = HashMap::new();
            let mut files_set: HashSet<String> = HashSet::new();
            let mut durations: Vec<f32> = Vec::new();

            for &idx in &p.member_indices {
                let (cand, call_type) = &all_candidates[idx];
                *call_types.entry(call_type.clone()).or_insert(0) += 1;
                files_set.insert(cand.source_file.clone());
                durations.push(cand.duration_ms);
            }

            let primary = call_types
                .iter()
                .max_by_key(|(_, &c)| c)
                .map(|(ct, _)| ct.clone())
                .unwrap_or_else(|| "unknown".to_string());

            let min_dur = durations.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_dur = durations.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let avg_dur = durations.iter().sum::<f32>() / durations.len() as f32;

            PhraseSummary {
                phrase_id: p.phrase_id,
                size: p.member_indices.len(),
                avg_duration_ms: avg_dur as f64,
                intra_similarity: p.intra_similarity,
                separation: p.separation_score,
                reuse_score: files_set.len() as f64,
                primary_call_type: primary,
                unique_birds: files_set.len(),
                unique_files: files_set.len(),
                duration_range_ms: (min_dur, max_dur),
            }
        })
        .collect();

    println!("\nATOMIC PHRASE STATISTICS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Total Files: {}", total_files);
    println!("Total Phrase Candidates: {}", all_candidates.len());
    println!("Atomic Phrases Discovered: {}", atomic_phrases.len());
    println!("High Quality Phrases: {}", high_quality.len());
    println!();
    println!("Segmentation Metrics:");
    println!(
        "  ├─ Avg Phrases per File: {:.1}",
        all_candidates.len() as f64 / total_files.max(1) as f64
    );
    println!("  └─ Avg Phrase Duration: {:.1}ms", avg_duration);
    println!();
    println!("Duration Distribution:");
    let mut sorted_dur: Vec<_> = duration_dist.iter().collect();
    sorted_dur.sort_by_key(|(k, _)| match k.as_str() {
        "0-50ms" => 0,
        "50-100ms" => 1,
        "100-200ms" => 2,
        "200-500ms" => 3,
        _ => 4,
    });
    for (bucket, count) in sorted_dur {
        println!("  ├─ {}: {}", bucket, count);
    }
    println!();
    println!("Cluster Quality:");
    println!("  ├─ Avg Cluster Size: {:.1}", avg_cluster_size);
    println!("  ├─ Avg Intra-Similarity: {:.3}", avg_intra);
    println!("  ├─ Avg Separation: {:.2}", avg_sep);
    println!("  └─ Avg Reuse: {:.1} files/phrase", avg_reuse);

    println!("\nTOP 5 ATOMIC PHRASES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    for (i, s) in top_phrases.iter().take(5).enumerate() {
        println!("\n{}. Phrase #{}", i + 1, s.phrase_id);
        println!(
            "   ├─ Size: {} segments, Duration: {:.1}ms ({:.1}-{:.1}ms)",
            s.size, s.avg_duration_ms, s.duration_range_ms.0, s.duration_range_ms.1
        );
        println!(
            "   ├─ Intra-Sim: {:.3}, Separation: {:.2}",
            s.intra_similarity, s.separation
        );
        println!("   └─ Reuse: {} files, Type: {}", s.unique_files, s.primary_call_type);
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("  Species: {}", config.name);
    println!("  Files Processed: {}", total_files);
    println!("  Phrase Candidates: {}", all_candidates.len());
    println!("  Atomic Phrases: {}", atomic_phrases.len());
    println!("  Total Time: {:.1}s", total_time_sec);

    // Save report
    let report = MultiSpeciesReport {
        species: config.name.clone(),
        total_files,
        total_candidates: all_candidates.len(),
        total_atomic_phrases: atomic_phrases.len(),
        high_quality_phrases: high_quality.len(),
        avg_phrases_per_file: all_candidates.len() as f64 / total_files.max(1) as f64,
        avg_phrase_duration_ms: avg_duration,
        duration_distribution: duration_dist,
        call_type_distribution: call_type_dist,
        avg_cluster_size,
        avg_intra_similarity: avg_intra,
        avg_separation: avg_sep,
        avg_reuse,
        top_phrases,
        processing_time_sec: total_time_sec,
    };

    std::fs::create_dir_all("dynamic_phrase_analysis")?;
    let output_path = format!("dynamic_phrase_analysis/{}_dynamic_phrases.json", config.name);
    let file = File::create(&output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &report)?;

    println!("\n  Report saved to: {}", output_path);

    Ok(())
}

// ============================================================================
// AUDIO UTILITIES
// ============================================================================

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

fn pad_center(s: &str, width: usize) -> String {
    let len = s.chars().count();
    if len >= width {
        s.to_string()
    } else {
        let left = (width - len) / 2;
        let right = width - len - left;
        format!("{}{}{}", " ".repeat(left), s, " ".repeat(right))
    }
}
