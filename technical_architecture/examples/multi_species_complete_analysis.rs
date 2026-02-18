//! Multi-Species Complete Analysis Pipeline
//!
//! Runs the full three-step analysis for multiple species:
//! 1. Hierarchical Segmentation (Motifs → Syllables → Notes)
//! 2. Syntax Analysis (Markov Chains / Bigram transitions)
//! 3. Acoustic Fingerprint (Visualization data)
//!
//! # Species-Dependent Atomic Granularity
//!
//! Different species encode meaning at different hierarchical levels:
//! - **Zebra Finch**: Motifs (~350ms) are the carrier of meaning (song patterns)
//! - **Egyptian Bat**: Syllables (~32ms) are the carrier of meaning (chirp types)
//! - **Dolphin**: Contours (~500ms+) are the carrier of meaning (whistle shapes)
//!
//! Usage:
//!   cargo run --release --example multi_species_complete_analysis -- egyptian_bat
//!   cargo run --release --example multi_species_complete_analysis -- dolphin
//!   cargo run --release --example multi_species_complete_analysis -- zebra_finch

use technical_architecture::{
    DynamicSegmenter, DynamicSegmenterConfig, DynamicPhraseCandidate,
    ZooVoxFeatureExtractor,
    AcousticSimilarityEngine, SimilarityMetric,
    HierarchicalThresholds, AtomicGranularity,
};
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
    /// Species-specific hierarchical thresholds for Motif → Syllable → Note
    hierarchical_thresholds: HierarchicalThresholds,
    /// Which hierarchical level carries semantic meaning
    atomic_granularity: AtomicGranularity,
}

#[derive(Debug, Clone)]
enum AnnotationFormat {
    ZebraFinch,
    EgyptianBat,
    DolphinWhistles,
    Marmoset,
}

impl SpeciesConfig {
    fn zebra_finch(base_dir: &Path) -> Self {
        Self {
            name: "zebra_finch".to_string(),
            data_dir: base_dir.join("zebra_finch/zebra_finch"),
            sample_rate: 44100,
            segmenter_config: DynamicSegmenterConfig::zebra_finch(),
            annotation_format: AnnotationFormat::ZebraFinch,
            max_files: 300,
            similarity_threshold: 0.30,
            min_occurrences: 2,
            hierarchical_thresholds: HierarchicalThresholds::zebra_finch(),
            atomic_granularity: AtomicGranularity::Motif, // MOTIFS carry meaning for songbirds
        }
    }

    fn egyptian_bat(base_dir: &Path) -> Self {
        Self {
            name: "egyptian_bat".to_string(),
            data_dir: base_dir.join("egyptian_fruit_bat_10k"),
            sample_rate: 250000,
            segmenter_config: DynamicSegmenterConfig::bat(),
            annotation_format: AnnotationFormat::EgyptianBat,
            max_files: 500,
            similarity_threshold: 0.35,
            min_occurrences: 2,
            hierarchical_thresholds: HierarchicalThresholds::bat(),
            atomic_granularity: AtomicGranularity::Syllable, // SYLLABLES carry meaning for bats
        }
    }

    fn dolphin(base_dir: &Path) -> Self {
        let mut config = DynamicSegmenterConfig::dolphin();
        config.min_phrase_duration_ms = 100.0;

        Self {
            name: "dolphin".to_string(),
            data_dir: base_dir.join("bottlenose_dolphins"),
            sample_rate: 44100,
            segmenter_config: config,
            annotation_format: AnnotationFormat::DolphinWhistles,
            max_files: 303,
            similarity_threshold: 0.30,
            min_occurrences: 2,
            hierarchical_thresholds: HierarchicalThresholds::dolphin(),
            atomic_granularity: AtomicGranularity::Contour, // CONTOURS carry meaning for dolphins
        }
    }

    fn marmoset(base_dir: &Path) -> Self {
        Self {
            name: "marmoset".to_string(),
            data_dir: base_dir.join("Vocalizations"),
            sample_rate: 44100,
            segmenter_config: DynamicSegmenterConfig::marmoset(),
            annotation_format: AnnotationFormat::Marmoset,
            max_files: 500,
            similarity_threshold: 0.30,
            min_occurrences: 2,
            hierarchical_thresholds: HierarchicalThresholds::marmoset(),
            atomic_granularity: AtomicGranularity::Syllable, // SYLLABLES carry meaning for marmosets
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
    name: String,
}

#[derive(Debug, Clone, Deserialize)]
struct BatAnnotation {
    #[serde(rename = "File Name")]
    filename: String,
    call_type: String,
}

// ============================================================================
// REPORT STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompleteAnalysisReport {
    species: String,
    total_files: usize,
    processing_time_sec: f64,

    // Hierarchical analysis
    hierarchical: HierarchicalResults,

    // Syntax analysis
    syntax: SyntaxResults,

    // Acoustic fingerprint
    fingerprint: FingerprintResults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HierarchicalResults {
    motifs: LevelSummary,
    syllables: LevelSummary,
    notes: LevelSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LevelSummary {
    phrase_types: usize,
    candidates: usize,
    avg_duration_ms: f64,
    duration_distribution: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SyntaxResults {
    vocabulary_size: usize,
    unique_transitions: usize,
    entropy: f64,
    perplexity: f64,
    top_transitions: Vec<TransitionInfo>,
    common_patterns: Vec<PatternInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TransitionInfo {
    from_phrase: usize,
    to_phrase: usize,
    count: usize,
    probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PatternInfo {
    pattern: Vec<usize>,
    occurrences: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FingerprintResults {
    phrase_types: usize,
    call_type_distribution: HashMap<String, usize>,
    duration_range_ms: (f64, f64),
    avg_duration_ms: f64,
    acoustic_niches: Vec<NicheInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NicheInfo {
    name: String,
    call_types: Vec<String>,
    occurrence_percent: f64,
    avg_duration_ms: f64,
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
        "dolphin" => SpeciesConfig::dolphin(&base_dir),
        "marmoset" | "common_marmoset" => SpeciesConfig::marmoset(&base_dir),
        _ => {
            eprintln!("Unknown species: {}. Options: zebra_finch, egyptian_bat, dolphin, marmoset", species_arg);
            std::process::exit(1);
        }
    };

    run_complete_analysis(config)
}

fn run_complete_analysis(config: SpeciesConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║     {} Complete Analysis Pipeline                         ║",
        pad_center(&config.name, 37));
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();

    println!("Configuration:");
    println!("  ├─ Sample Rate: {}Hz", config.sample_rate);
    println!("  ├─ Min Phrase Duration: {}ms", config.segmenter_config.min_phrase_duration_ms);
    println!("  ├─ Change Threshold: {}", config.segmenter_config.change_threshold);
    println!("  └─ Max Files: {}", config.max_files);
    println!();

    // ========================================================================
    // Load files
    // ========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[1/4] Loading Files");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let files = load_files(&config)?;
    println!("Loaded {} files", files.len());

    // ========================================================================
    // Extract all phrases with dynamic segmentation
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[2/4] Dynamic Segmentation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let segmenter = DynamicSegmenter::new(config.segmenter_config.clone(), config.sample_rate);
    let processed = Arc::new(AtomicUsize::new(0));
    let total_files = files.len();

    let all_candidates: Vec<(DynamicPhraseCandidate, String)> = files
        .par_iter()
        .flat_map(|file_info| {
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 50 == 0 {
                println!("  Progress: {}/{}", count + 1, total_files);
            }

            if let Ok(audio) = load_audio(&file_info.path) {
                if audio.len() < 500 {
                    return Vec::new();
                }

                let audio = resample_audio(&audio, file_info.sample_rate, config.sample_rate);
                let extractor = Arc::new(std::sync::Mutex::new(ZooVoxFeatureExtractor::new(config.sample_rate)));

                let result = segmenter.segment(
                    &audio,
                    |frame, sr| {
                        let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
                        let mut ext = extractor.lock().unwrap();
                        ext.extract_45d(&frame_f64).ok().map(|f| f.to_vector().to_vec())
                    },
                    &file_info.filename,
                );

                result.candidates.into_iter()
                    .map(|c| (c, file_info.call_type.clone()))
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect();

    println!("\nExtracted {} phrase candidates", all_candidates.len());

    // ========================================================================
    // Cluster phrases
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[3/4] Clustering Phrases");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let clusters = cluster_phrases(&all_candidates, config.similarity_threshold, config.min_occurrences);
    println!("Discovered {} phrase types", clusters.len());

    // ========================================================================
    // Run all three analyses
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[4/4] Running Three-Step Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Hierarchical analysis (simplified - using different thresholds)
    let hierarchical = analyze_hierarchical(&config, &files)?;

    // Syntax analysis
    let syntax = analyze_syntax(&all_candidates, &clusters, config.similarity_threshold)?;

    // Fingerprint analysis
    let fingerprint = analyze_fingerprint(&all_candidates, &clusters)?;

    let total_time = total_start.elapsed();

    // ========================================================================
    // Generate report
    // ========================================================================
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("COMPLETE ANALYSIS SUMMARY: {}", config.name.to_uppercase());
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ HIERARCHICAL STRUCTURE                                                      │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    // Mark the atomic level (carrier of meaning) with an asterisk
    let motif_marker = if config.atomic_granularity == AtomicGranularity::Motif { " *" } else { "" };
    let syllable_marker = if config.atomic_granularity == AtomicGranularity::Syllable { " *" } else { "" };
    let note_marker = if config.atomic_granularity == AtomicGranularity::Note { " *" } else { "" };

    println!("│ Level     │ Types │ Candidates │ Avg Duration                             │");
    println!("│ Motifs{}   │ {:>5} │ {:>10} │ {:>8.1}ms                               │",
        motif_marker, hierarchical.motifs.phrase_types, hierarchical.motifs.candidates, hierarchical.motifs.avg_duration_ms);
    println!("│ Syllables{}│ {:>5} │ {:>10} │ {:>8.1}ms                               │",
        syllable_marker, hierarchical.syllables.phrase_types, hierarchical.syllables.candidates, hierarchical.syllables.avg_duration_ms);
    println!("│ Notes{}    │ {:>5} │ {:>10} │ {:>8.1}ms                               │",
        note_marker, hierarchical.notes.phrase_types, hierarchical.notes.candidates, hierarchical.notes.avg_duration_ms);
    println!("│                                                                             │");
    println!("│ * = ATOMIC LEVEL (carrier of meaning for {})", config.name);
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ SYNTAX ANALYSIS                                                             │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Vocabulary Size:    {} phrases", syntax.vocabulary_size);
    println!("│ Unique Transitions: {}", syntax.unique_transitions);
    println!("│ Entropy:            {:.3} bits", syntax.entropy);
    println!("│ Perplexity:         {:.2} {}", syntax.perplexity,
        if syntax.perplexity < 5.0 { "(predictable syntax)" } else { "(variable syntax)" });
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ ACOUSTIC FINGERPRINT                                                        │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Phrase Types:       {}", fingerprint.phrase_types);
    println!("│ Duration Range:     {:.0} - {:.0}ms", fingerprint.duration_range_ms.0, fingerprint.duration_range_ms.1);
    println!("│ Avg Duration:       {:.1}ms", fingerprint.avg_duration_ms);
    println!("│ Call Types:         {}", fingerprint.call_type_distribution.len());
    for niche in &fingerprint.acoustic_niches {
        println!("│ • {}: {:.1}%", niche.name, niche.occurrence_percent);
    }
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    // Save report
    let report = CompleteAnalysisReport {
        species: config.name.clone(),
        total_files,
        processing_time_sec: total_time.as_secs_f64(),
        hierarchical: hierarchical.clone(),
        syntax: syntax.clone(),
        fingerprint: fingerprint.clone(),
    };

    std::fs::create_dir_all("complete_analysis")?;
    let output_path = format!("complete_analysis/{}_complete.json", config.name);
    let file = File::create(&output_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &report)?;

    println!("\nReport saved to: {}", output_path);
    println!("Total processing time: {:.1}s", total_time.as_secs_f64());

    Ok(())
}

// ============================================================================
// ANALYSIS FUNCTIONS
// ============================================================================

fn analyze_hierarchical(
    config: &SpeciesConfig,
    files: &[FileInfo],
) -> Result<HierarchicalResults, Box<dyn std::error::Error>> {
    // Use species-specific hierarchical thresholds
    // These are tuned for each species' typical tempo and sample rate
    let thresholds = &config.hierarchical_thresholds;

    // Create segmenters for each level using species-specific thresholds
    let motif_config = DynamicSegmenterConfig::for_motif_level(thresholds);
    let syllable_config = DynamicSegmenterConfig::for_syllable_level(thresholds);
    let note_config = DynamicSegmenterConfig::for_note_level(thresholds);

    let motif_segmenter = DynamicSegmenter::new(motif_config, config.sample_rate);
    let syllable_segmenter = DynamicSegmenter::new(syllable_config, config.sample_rate);
    let note_segmenter = DynamicSegmenter::new(note_config, config.sample_rate);

    let mut motif_candidates = Vec::new();
    let mut syllable_candidates = Vec::new();
    let mut note_candidates = Vec::new();

    for file_info in files.iter().take(100) {
        if let Ok(audio) = load_audio(&file_info.path) {
            if audio.len() < 500 {
                continue;
            }
            let audio = resample_audio(&audio, file_info.sample_rate, config.sample_rate);

            let extractor = Arc::new(std::sync::Mutex::new(ZooVoxFeatureExtractor::new(config.sample_rate)));

            // Extract at each level
            let extract_fn = |frame: &[f32], _sr: u32| {
                let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
                let mut ext = extractor.lock().unwrap();
                ext.extract_45d(&frame_f64).ok().map(|f| f.to_vector().to_vec())
            };

            motif_candidates.extend(motif_segmenter.segment(&audio, extract_fn, &file_info.filename).candidates);
            syllable_candidates.extend(syllable_segmenter.segment(&audio, extract_fn, &file_info.filename).candidates);
            note_candidates.extend(note_segmenter.segment(&audio, extract_fn, &file_info.filename).candidates);
        }
    }

    // Cluster at each level
    let motif_clusters = cluster_phrases_simple(&motif_candidates, config.similarity_threshold, 2);
    let syllable_clusters = cluster_phrases_simple(&syllable_candidates, config.similarity_threshold * 0.9, 2);
    let note_clusters = cluster_phrases_simple(&note_candidates, config.similarity_threshold * 0.8, 2);

    Ok(HierarchicalResults {
        motifs: LevelSummary {
            phrase_types: motif_clusters.len(),
            candidates: motif_candidates.len(),
            avg_duration_ms: if motif_candidates.is_empty() { 0.0 } else {
                motif_candidates.iter().map(|c| c.duration_ms as f64).sum::<f64>() / motif_candidates.len() as f64
            },
            duration_distribution: compute_duration_distribution(&motif_candidates),
        },
        syllables: LevelSummary {
            phrase_types: syllable_clusters.len(),
            candidates: syllable_candidates.len(),
            avg_duration_ms: if syllable_candidates.is_empty() { 0.0 } else {
                syllable_candidates.iter().map(|c| c.duration_ms as f64).sum::<f64>() / syllable_candidates.len() as f64
            },
            duration_distribution: compute_duration_distribution(&syllable_candidates),
        },
        notes: LevelSummary {
            phrase_types: note_clusters.len(),
            candidates: note_candidates.len(),
            avg_duration_ms: if note_candidates.is_empty() { 0.0 } else {
                note_candidates.iter().map(|c| c.duration_ms as f64).sum::<f64>() / note_candidates.len() as f64
            },
            duration_distribution: compute_duration_distribution(&note_candidates),
        },
    })
}

fn analyze_syntax(
    candidates: &[(DynamicPhraseCandidate, String)],
    clusters: &[PhraseCluster],
    threshold: f32,
) -> Result<SyntaxResults, Box<dyn std::error::Error>> {
    if candidates.is_empty() || clusters.is_empty() {
        return Ok(SyntaxResults {
            vocabulary_size: 0,
            unique_transitions: 0,
            entropy: 0.0,
            perplexity: f64::INFINITY,
            top_transitions: Vec::new(),
            common_patterns: Vec::new(),
        });
    }

    // Build phrase ID lookup
    let mut phrase_to_id: HashMap<String, usize> = HashMap::new();
    for cluster in clusters {
        for &idx in &cluster.member_indices {
            if let Some((cand, _)) = candidates.get(idx) {
                phrase_to_id.entry(cand.id.clone()).or_insert(cluster.phrase_id);
            }
        }
    }

    // Group by source file and convert to sequences
    let mut file_sequences: HashMap<String, Vec<usize>> = HashMap::new();
    for (cand, _) in candidates {
        if let Some(&id) = phrase_to_id.get(&cand.id) {
            file_sequences.entry(cand.source_file.clone())
                .or_insert_with(Vec::new)
                .push(id);
        }
    }

    // Build bigram model
    let mut transitions: HashMap<(usize, usize), usize> = HashMap::new();
    let mut from_totals: HashMap<usize, usize> = HashMap::new();
    let mut vocabulary: HashSet<usize> = HashSet::new();

    for seq in file_sequences.values() {
        for window in seq.windows(2) {
            *transitions.entry((window[0], window[1])).or_insert(0) += 1;
            *from_totals.entry(window[0]).or_insert(0) += 1;
            vocabulary.insert(window[0]);
            vocabulary.insert(window[1]);
        }
    }

    // Calculate perplexity
    let mut log_prob = 0.0;
    let mut total = 0;
    for seq in file_sequences.values() {
        for window in seq.windows(2) {
            let from_total = from_totals.get(&window[0]).copied().unwrap_or(0);
            if from_total > 0 {
                let count = transitions.get(&(window[0], window[1])).copied().unwrap_or(0);
                let prob = count as f64 / from_total as f64;
                if prob > 0.0 {
                    log_prob += prob.ln();
                    total += 1;
                }
            }
        }
    }
    let perplexity = if total > 0 { (-log_prob / total as f64).exp() } else { f64::INFINITY };

    // Calculate entropy
    let mut entropy = 0.0;
    for from in &vocabulary {
        let from_total = from_totals.get(from).copied().unwrap_or(0) as f64;
        if from_total > 0.0 {
            for to in &vocabulary {
                let count = transitions.get(&(*from, *to)).copied().unwrap_or(0);
                let prob = count as f64 / from_total;
                if prob > 0.0 {
                    entropy -= prob * prob.log2();
                }
            }
        }
    }
    entropy /= vocabulary.len().max(1) as f64;

    // Top transitions
    let mut top_transitions: Vec<TransitionInfo> = transitions.iter()
        .map(|((from, to), &count)| {
            let from_total = from_totals.get(from).copied().unwrap_or(1);
            TransitionInfo {
                from_phrase: *from,
                to_phrase: *to,
                count,
                probability: count as f64 / from_total as f64,
            }
        })
        .collect();
    top_transitions.sort_by(|a, b| b.count.cmp(&a.count));

    // Common patterns
    let mut pattern_counts: HashMap<Vec<usize>, usize> = HashMap::new();
    for seq in file_sequences.values() {
        for window in seq.windows(3) {
            *pattern_counts.entry(window.to_vec()).or_insert(0) += 1;
        }
    }
    let mut common_patterns: Vec<PatternInfo> = pattern_counts.iter()
        .filter(|(_, &count)| count >= 2)
        .map(|(pattern, &count)| PatternInfo {
            pattern: pattern.clone(),
            occurrences: count,
        })
        .collect();
    common_patterns.sort_by(|a, b| b.occurrences.cmp(&a.occurrences));

    Ok(SyntaxResults {
        vocabulary_size: vocabulary.len(),
        unique_transitions: transitions.len(),
        entropy,
        perplexity,
        top_transitions: top_transitions.into_iter().take(10).collect(),
        common_patterns: common_patterns.into_iter().take(10).collect(),
    })
}

fn analyze_fingerprint(
    candidates: &[(DynamicPhraseCandidate, String)],
    clusters: &[PhraseCluster],
) -> Result<FingerprintResults, Box<dyn std::error::Error>> {
    if candidates.is_empty() {
        return Ok(FingerprintResults {
            phrase_types: 0,
            call_type_distribution: HashMap::new(),
            duration_range_ms: (0.0, 0.0),
            avg_duration_ms: 0.0,
            acoustic_niches: Vec::new(),
        });
    }

    // Call type distribution
    let mut call_type_distribution: HashMap<String, usize> = HashMap::new();
    for (_, call_type) in candidates {
        *call_type_distribution.entry(call_type.clone()).or_insert(0) += 1;
    }

    // Duration stats
    let durations: Vec<f64> = candidates.iter().map(|(c, _)| c.duration_ms as f64).collect();
    let min_dur = durations.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_dur = durations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg_dur = durations.iter().sum::<f64>() / durations.len() as f64;

    // Identify niches
    let total = candidates.len() as f64;

    // Short calls niche
    let short_calls: Vec<_> = candidates.iter()
        .filter(|(c, _)| c.duration_ms < 150.0)
        .collect();
    let short_percent = short_calls.len() as f64 / total * 100.0;

    // Long calls niche
    let long_calls: Vec<_> = candidates.iter()
        .filter(|(c, _)| c.duration_ms > 400.0)
        .collect();
    let long_percent = long_calls.len() as f64 / total * 100.0;

    let mut niches = Vec::new();
    if short_percent > 5.0 {
        let short_types: HashSet<_> = short_calls.iter().map(|(_, ct)| ct.clone()).collect();
        niches.push(NicheInfo {
            name: "Short Calls".to_string(),
            call_types: short_types.into_iter().collect(),
            occurrence_percent: short_percent,
            avg_duration_ms: short_calls.iter().map(|(c, _)| c.duration_ms as f64).sum::<f64>() / short_calls.len().max(1) as f64,
        });
    }
    if long_percent > 5.0 {
        let long_types: HashSet<_> = long_calls.iter().map(|(_, ct)| ct.clone()).collect();
        niches.push(NicheInfo {
            name: "Long Calls".to_string(),
            call_types: long_types.into_iter().collect(),
            occurrence_percent: long_percent,
            avg_duration_ms: long_calls.iter().map(|(c, _)| c.duration_ms as f64).sum::<f64>() / long_calls.len().max(1) as f64,
        });
    }

    Ok(FingerprintResults {
        phrase_types: clusters.len(),
        call_type_distribution,
        duration_range_ms: (min_dur, max_dur),
        avg_duration_ms: avg_dur,
        acoustic_niches: niches,
    })
}

// ============================================================================
// HELPER STRUCTURES AND FUNCTIONS
// ============================================================================

struct FileInfo {
    path: PathBuf,
    filename: String,
    call_type: String,
    sample_rate: u32,
}

struct PhraseCluster {
    phrase_id: usize,
    member_indices: Vec<usize>,
}

fn load_files(config: &SpeciesConfig) -> Result<Vec<FileInfo>, Box<dyn std::error::Error>> {
    match config.annotation_format {
        AnnotationFormat::ZebraFinch => {
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
        AnnotationFormat::EgyptianBat => {
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
                    call_type: format!("type_{}", ann.call_type),
                    sample_rate: 250000,
                });
            }
            Ok(files.into_iter().take(config.max_files).collect())
        }
        AnnotationFormat::DolphinWhistles => {
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
        AnnotationFormat::Marmoset => {
            // Marmoset data is in subdirectories with FLAC files
            // Call types are embedded in filenames: Trill_x.flac, Tsik_x.flac, Seep_x.flac, Vocalization_x.flac
            let mut files = Vec::new();

            // Recursively scan all subdirectories for FLAC files
            fn scan_marmoset_dir(dir: &Path, files: &mut Vec<FileInfo>, max_files: usize) -> std::io::Result<()> {
                if files.len() >= max_files {
                    return Ok(());
                }

                for entry in std::fs::read_dir(dir)? {
                    let entry = entry?;
                    let path = entry.path();

                    if path.is_dir() {
                        scan_marmoset_dir(&path, files, max_files)?;
                    } else if path.extension().map(|e| e == "flac").unwrap_or(false) {
                        let filename = path.file_name().unwrap().to_string_lossy().to_string();

                        // Extract call type from filename (e.g., "Trill_34837.flac" -> "Trill")
                        let call_type = filename.split('_').next().unwrap_or("Unknown").to_string();

                        files.push(FileInfo {
                            filename: filename.clone(),
                            path,
                            call_type,
                            sample_rate: 44100,
                        });

                        if files.len() >= max_files {
                            break;
                        }
                    }
                }
                Ok(())
            }

            scan_marmoset_dir(&config.data_dir, &mut files, config.max_files)?;
            Ok(files)
        }
    }
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
            clusters.push(PhraseCluster {
                phrase_id: clusters.len(),
                member_indices: cluster_indices,
            });
        }
    }

    clusters
}

fn cluster_phrases_simple(
    candidates: &[DynamicPhraseCandidate],
    threshold: f32,
    min_size: usize,
) -> Vec<Vec<usize>> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);

    let n_samples = candidates.len().min(3000);
    let mut matrix = ndarray::Array2::<f64>::zeros((n_samples, FEATURE_DIM));
    for (i, cand) in candidates.iter().take(n_samples).enumerate() {
        for (j, &val) in cand.features.iter().enumerate() {
            matrix[[i, j]] = val;
        }
    }
    engine.fit_normalization(&matrix);

    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut assigned = vec![false; candidates.len()];

    for i in 0..candidates.len() {
        if assigned[i] {
            continue;
        }

        let mut cluster_indices = vec![i];
        assigned[i] = true;

        let query = Array1::from_vec(candidates[i].features.clone());

        for j in (i + 1)..candidates.len() {
            if !assigned[j] {
                let candidate = Array1::from_vec(candidates[j].features.clone());
                let dist = engine.distance(&query, &candidate);

                if dist < threshold as f64 {
                    cluster_indices.push(j);
                    assigned[j] = true;
                }
            }
        }

        if cluster_indices.len() >= min_size {
            clusters.push(cluster_indices);
        }
    }

    clusters
}

fn compute_duration_distribution(candidates: &[DynamicPhraseCandidate]) -> HashMap<String, usize> {
    let mut dist = HashMap::new();
    for c in candidates {
        let bucket = match c.duration_ms {
            d if d < 50.0 => "0-50ms",
            d if d < 100.0 => "50-100ms",
            d if d < 200.0 => "100-200ms",
            d if d < 500.0 => "200-500ms",
            _ => "500ms+",
        }.to_string();
        *dist.entry(bucket).or_insert(0) += 1;
    }
    dist
}

fn load_audio(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let extension = path.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    match extension.as_deref() {
        Some("wav") => load_wav_audio(path),
        Some("flac") => load_flac_audio(path),
        _ => load_wav_audio(path), // Try WAV as fallback
    }
}

fn load_wav_audio(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
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

#[cfg(feature = "symphonia")]
fn load_flac_audio(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use std::fs::File;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;
    use symphonia::core::audio::{AudioBufferRef, Signal};

    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("flac");

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();

    let mut probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| format!("Failed to probe FLAC: {}", e))?;

    let track = probed.format.default_track()
        .ok_or("No default track in FLAC file")?;
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Failed to create FLAC decoder: {}", e))?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match probed.format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(_) => break,
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                match audio_buf {
                    AudioBufferRef::F32(buf) => {
                        for plane in buf.as_ref().planes().planes() {
                            all_samples.extend(plane.iter().copied());
                            break;
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        for plane in buf.as_ref().planes().planes() {
                            all_samples.extend(plane.iter().map(|&s| s as f32 / i16::MAX as f32));
                            break;
                        }
                    }
                    AudioBufferRef::S32(buf) => {
                        for plane in buf.as_ref().planes().planes() {
                            all_samples.extend(plane.iter().map(|&s| s as f32 / i32::MAX as f32));
                            break;
                        }
                    }
                    _ => {} // Skip other formats
                }
            }
            Err(_) => continue,
        }
    }

    Ok(all_samples)
}

#[cfg(not(feature = "symphonia"))]
fn load_flac_audio(_path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Err("FLAC support requires symphonia feature. Rebuild with --features symphonia".into())
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
