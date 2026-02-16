// Phase 1: Phrase-Context Matrix Analysis for Marmoset
//
// This analysis tests the "Sentence Structure" hypothesis by measuring lexical flexibility.
//
// First, we extract proper phrases from the FLAC audio files using within-vocalization
// segmentation, then analyze phrase-context patterns.
//
// Hypothesis: If marmosets use combinatorial syntax, we should observe:
// 1. General-purpose phrases (used in many contexts) - "function words"
// 2. Context-specific phrases (used in few contexts) - "content words"
//
// Marmoset Contexts: Based on call types (Vocalization, Phee, Twitter, Trill, Tsik, Seep, Infant)
//
// Methods:
// - Generality Score: Contexts containing phrase / Total contexts
// - Shannon Entropy: Distribution evenness across contexts
// - Permutation Test: Statistical significance vs random chance
//
// Output: Phrase-Context matrix, generality scores, statistical tests, visualizations
//
// Usage: cargo run --release --example phrase_context_analysis_marmoset_generality

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use technical_architecture::{
    MicroDynamicsExtractor, WithinVocalizationAnalyzer, WithinVocalizationConfig,
};

// ============================================================================
// Data Structures
// ============================================================================

/// Marmoset call type derived from filename
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
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
        if fname.contains("infant") || fname.contains("cry") {
            CallType::Infant
        } else if fname.contains("twitter") {
            CallType::Twitter
        } else if fname.contains("phee") {
            CallType::Phee
        } else if fname.contains("trill") {
            CallType::Trill
        } else if fname.contains("tsik") {
            CallType::Tsik
        } else if fname.contains("seep") {
            CallType::Seep
        } else if fname.contains("vocalization") || fname.starts_with("v") {
            CallType::Vocalization
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

/// Phrase extracted from a vocalization file
#[derive(Debug, Clone)]
struct ExtractedPhrase {
    phrase_id: String,
    file_name: String,
    call_type: CallType,
    start_ms: f64,
    duration_ms: f64,
    features: Vec<f32>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GeneralityMetrics {
    phrase_id: String,
    total_occurrences: usize,
    contexts_used: usize,
    total_contexts: usize,
    generality_score: f64,
    shannon_entropy: f64,
    normalized_entropy: f64,
    classification: PhraseType,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
enum PhraseType {
    UniversalGeneralist,
    Generalist,
    FlexibleSpecialist,
    ContextSpecialist,
    HighlySpecific,
    Rare,
}

#[derive(Debug, Clone, serde::Serialize)]
struct PermutationTestResult {
    observed_mean_generality: f64,
    null_mean_generality: f64,
    null_std_generality: f64,
    p_value: f64,
    z_score: f64,
    significant: bool,
    n_permutations: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
struct AnalysisResults {
    metadata: Metadata,
    generality_metrics: Vec<GeneralityMetrics>,
    permutation_test: PermutationTestResult,
    summary_statistics: SummaryStatistics,
}

#[derive(Debug, Clone, serde::Serialize)]
struct Metadata {
    dataset: String,
    n_phrases: usize,
    n_contexts: usize,
    total_observations: usize,
    n_files: usize,
    analysis_timestamp: String,
}

#[derive(Debug, Clone, serde::Serialize)]
struct SummaryStatistics {
    n_universal_phrases: usize,
    n_generalist_phrases: usize,
    n_flexible_specialist_phrases: usize,
    n_context_specialist_phrases: usize,
    n_highly_specific_phrases: usize,
    mean_generality_score: f64,
    median_generality_score: f64,
    std_generality_score: f64,
    mean_shannon_entropy: f64,
    median_shannon_entropy: f64,
}

// ============================================================================
// Main Function
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║    Phase 1: Phrase-Context Matrix Analysis - Marmoset                        ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  HYPOTHESIS TESTING: Combinatorial Syntax vs Holistic Signals             ║");
    println!("║                                                                           ║");
    println!("║  If combinatorial syntax exists:                                          ║");
    println!("║    • General-purpose phrases (used in many contexts) - function words     ║");
    println!("║    • Context-specific phrases (used in few contexts) - content words       ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let data_dir = Path::new("/home/sheel/birdsong_analysis/data/Vocalizations");
    let results_dir = Path::new("/mnt/c/Users/sheel/Desktop/src/marmoset_phase1_generality_results");

    fs::create_dir_all(&results_dir)?;

    // ========================================================================
    // Step 1: Find All FLAC Files
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Finding Marmoset Vocalization Files                            │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let flac_files = find_flac_files_recursive(data_dir)?;

    // Group by call type
    let mut files_by_type: HashMap<CallType, Vec<PathBuf>> = HashMap::new();
    for path in &flac_files {
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let call_type = CallType::from_filename(filename);
        if call_type != CallType::Unknown {
            files_by_type.entry(call_type).or_insert_with(Vec::new).push(path.clone());
        }
    }

    println!("   📂 Found {} FLAC files", flac_files.len());
    println!();

    println!("   📊 Files by Call Type:");
    let mut types: Vec<_> = files_by_type.iter().collect();
    types.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    for (call_type, files) in &types {
        println!("      • {:15}: {} files", call_type.name(), files.len());
    }
    println!();

    // ========================================================================
    // Step 2: Extract Phrases from Audio Files
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Extracting Phrases using Within-Vocalization Segmentation      │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let sample_rate = 96000u32;
    let config = WithinVocalizationConfig {
        sample_rate,
        min_phrase_duration_ms: 15.0,
        min_pause_duration_ms: 8.0,
        min_f0_change_hz: 1500.0,
        pause_energy_threshold: 0.15,
        frame_size_ms: 5.0,
        hop_size_ms: 2.0,
        require_consensus: false,
        max_phrases: 20,
    };

    let analyzer = WithinVocalizationAnalyzer::new(config);

    // Process files in parallel with limit for testing
    let max_files = 1000usize;  // Adjust as needed
    let all_files: Vec<_> = files_by_type.values().flat_map(|v| v.iter()).cloned().collect();
    let files_to_process: Vec<_> = all_files.into_iter().take(max_files).collect();

    println!("   🔄 Processing {} files (limited from {} total)",
             files_to_process.len(), flac_files.len());
    println!();

    let all_phrases: Arc<Mutex<Vec<ExtractedPhrase>>> = Arc::new(Mutex::new(Vec::new()));
    let progress = Arc::new(Mutex::new(0usize));

    files_to_process.par_iter().for_each(|path| {
        let filename = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let call_type = CallType::from_filename(filename);

        if let Ok(audio) = load_flac_file(path) {
            if let Ok(segmentation) = analyzer.analyze_vocalization(&audio, None) {
                let mut phrases = Vec::new();

                for (i, (&start_ms, &duration_ms)) in segmentation.phrase_starts_ms.iter()
                    .zip(segmentation.phrase_durations_ms.iter()).enumerate()
                {
                    let end_ms = start_ms + duration_ms;
                    let start_sample = (start_ms * sample_rate as f64 / 1000.0) as usize;
                    let end_sample = (end_ms * sample_rate as f64 / 1000.0) as usize;

                    if end_sample > audio.len() || start_sample >= end_sample {
                        continue;
                    }

                    if duration_ms < 10.0 {
                        continue;  // Skip very short phrases
                    }

                    let phrase_audio = &audio[start_sample..end_sample];

                    // Extract 15D features
                    if let Ok(features) = extract_15d_features(phrase_audio, sample_rate) {
                        let phrase_id = format!("{}_phrase_{}", filename, i);
                        phrases.push(ExtractedPhrase {
                            phrase_id: phrase_id.clone(),
                            file_name: filename.to_string(),
                            call_type,
                            start_ms,
                            duration_ms,
                            features,
                        });
                    }
                }

                if !phrases.is_empty() {
                    let mut all = all_phrases.lock().unwrap();
                    all.extend(phrases);
                }
            }
        }

        let mut p = progress.lock().unwrap();
        *p += 1;
        if *p % 100 == 0 {
            println!("      Processed {}/{} files", *p, files_to_process.len());
        }
    });

    let phrases = Arc::try_unwrap(all_phrases).unwrap().into_inner()?;

    println!();
    println!("   ✅ Extracted {} phrases from {} files", phrases.len(), files_to_process.len());
    println!();

    // ========================================================================
    // Step 3: Cluster Phrases to Discover Vocabulary
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Clustering Phrases to Discover Vocabulary                       │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    if phrases.is_empty() {
        println!("   ⚠️  No phrases extracted. Exiting.");
        return Ok(());
    }

    // Cluster phrases based on feature similarity
    let vocabulary = cluster_phrases(&phrases)?;

    println!("   ✅ Discovered {} vocabulary words", vocabulary.len());
    println!();

    // Build phrase-context matrix
    let mut phrase_context_map: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut phrase_totals: HashMap<String, usize> = HashMap::new();
    let mut context_totals: HashMap<String, usize> = HashMap::new();

    for phrase in &phrases {
        // Find which vocabulary word this phrase belongs to
        let word_id = find_closest_vocabulary_word(&phrase.features, &vocabulary);

        let context = phrase.call_type.name().to_string();

        *phrase_context_map.entry(word_id.clone()).or_default().entry(context.clone()).or_insert(0) += 1;
        *phrase_totals.entry(word_id.clone()).or_insert(0) += 1;
        *context_totals.entry(context).or_insert(0) += 1;
    }

    let n_phrases = phrase_context_map.len();
    let n_contexts = context_totals.len();
    let total_obs: usize = phrase_totals.values().sum();

    println!("   📊 Matrix Statistics:");
    println!("      ├─ Unique phrases: {}", n_phrases);
    println!("      ├─ Behavioral contexts: {}", n_contexts);
    println!("      └─ Total observations: {}", total_obs);
    println!();

    // ========================================================================
    // Step 4: Calculate Generality Metrics
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Calculating Generality Metrics                                   │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let mut metrics = calculate_generality_metrics(&phrase_context_map, &phrase_totals, n_contexts)?;
    println!("      └─ Computed metrics for {} phrases", metrics.len());
    println!();

    // ========================================================================
    // Step 5: Classify Phrase Types
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Phrase Type Classification                                      │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    classify_phrases(&mut metrics);

    let type_counts = count_phrase_types(&metrics);
    println!("   🏷️  Phrase Type Distribution:");
    println!("      ┌────────────────────────────┬──────────┬──────────┐");
    println!("      │ Type                       │ Count    │ Percentage│");
    println!("      ├────────────────────────────┼──────────┼──────────┤");
    println!("      │ Universal Generalist       │ {:8} │ {:8.1}│",
             type_counts.0, type_counts.0 as f64 / metrics.len() as f64 * 100.0);
    println!("      │ Generalist                 │ {:8} │ {:8.1}│",
             type_counts.1, type_counts.1 as f64 / metrics.len() as f64 * 100.0);
    println!("      │ Flexible Specialist        │ {:8} │ {:8.1}│",
             type_counts.2, type_counts.2 as f64 / metrics.len() as f64 * 100.0);
    println!("      │ Context Specialist         │ {:8} │ {:8.1}│",
             type_counts.3, type_counts.3 as f64 / metrics.len() as f64 * 100.0);
    println!("      │ Highly Specific            │ {:8} │ {:8.1}│",
             type_counts.4, type_counts.4 as f64 / metrics.len() as f64 * 100.0);
    println!("      └────────────────────────────┴──────────┴──────────┘");
    println!();

    // ========================================================================
    // Step 6: Permutation Test
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 6: Permutation Test (Statistical Significance)                     │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let n_permutations = 1000;
    let perm_result = run_permutation_test(&phrase_context_map, &phrase_totals, n_permutations, n_contexts)?;

    println!();
    println!("   ✅ Permutation Test Results:");
    println!("      ├─ Observed mean generality: {:.4}", perm_result.observed_mean_generality);
    println!("      ├─ Null mean generality:      {:.4} ± {:.4}",
             perm_result.null_mean_generality, perm_result.null_std_generality);
    println!("      ├─ Z-score:                   {:.4}", perm_result.z_score);
    println!("      ├─ P-value:                   {:.6}", perm_result.p_value);
    println!("      └─ Significant (α=0.05):      {}", if perm_result.significant { "YES ✨" } else { "NO" });
    println!();

    if perm_result.significant {
        println!("   🎯 CONCLUSION: Observed phrase reuse is significantly NON-RANDOM.");
    } else {
        println!("   ⚠️  CONCLUSION: Observed pattern could be due to random chance.");
    }
    println!();

    // ========================================================================
    // Step 7: Summary Statistics
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 7: Summary Statistics                                              │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let summary = compute_summary_statistics(&metrics);
    println!("   📊 Generality Score Distribution:");
    println!("      ├─ Mean:   {:.4}", summary.mean_generality_score);
    println!("      ├─ Median: {:.4}", summary.median_generality_score);
    println!("      └─ Std:    {:.4}", summary.std_generality_score);
    println!();

    println!("   📊 Shannon Entropy Distribution:");
    println!("      ├─ Mean:   {:.4} bits", summary.mean_shannon_entropy);
    println!("      └─ Median: {:.4} bits", summary.median_shannon_entropy);
    println!();

    // ========================================================================
    // Step 8: Save Results
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 8: Saving Results                                                   │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let results = AnalysisResults {
        metadata: Metadata {
            dataset: "marmoset".to_string(),
            n_phrases,
            n_contexts,
            total_observations: total_obs,
            n_files: files_to_process.len(),
            analysis_timestamp: chrono::Utc::now().to_rfc3339(),
        },
        generality_metrics: metrics.clone(),
        permutation_test: perm_result.clone(),
        summary_statistics: summary.clone(),
    };

    let results_path = results_dir.join("generality_analysis_results.json");
    fs::write(&results_path, serde_json::to_string_pretty(&results)?)?;
    println!("   💾 Full results: {}", results_path.display());

    let csv_path = results_dir.join("phrase_generality_metrics.csv");
    save_generality_csv(&metrics, &csv_path)?;
    println!("   💾 Generality CSV: {}", csv_path.display());
    println!();

    // ========================================================================
    // Final Summary
    // ========================================================================

    let elapsed = start_time.elapsed();

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    ANALYSIS COMPLETE                                     ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  📊 KEY FINDINGS:                                                         ║");
    println!("║     • Total phrases analyzed: {}                                        ║", n_phrases);
    println!("║     • Call type contexts: {}                                            ║", n_contexts);
    println!("║     • Generalist phrases: {} ({:.1}%)                                   ║",
             type_counts.0 + type_counts.1,
             (type_counts.0 + type_counts.1) as f64 / metrics.len() as f64 * 100.0);
    println!("║     • Specialist phrases: {} ({:.1}%)                                   ║",
             type_counts.3 + type_counts.4,
             (type_counts.3 + type_counts.4) as f64 / metrics.len() as f64 * 100.0);
    println!("║                                                                           ║");
    if perm_result.significant {
        println!("║     ✅ SIGNIFICANT: Phrase reuse is non-random                            ║");
        println!("║     This SUPPORTS the combinatorial syntax hypothesis                   ║");
    } else {
        println!("║     ⚠️  NOT SIGNIFICANT                                                 ║");
    }
    println!("║                                                                           ║");
    println!("║  ⏱️  Analysis time: {:.2}s                                                ║", elapsed.as_secs_f64());
    println!("║                                                                           ║");
    println!("║  📁 Results saved to:                                                     ║");
    println!("║     {}                                              ║", results_dir.display());
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

fn find_flac_files_recursive(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut flac_files = Vec::new();
    let entries = std::fs::read_dir(dir)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            flac_files.extend(find_flac_files_recursive(&path)?);
        } else if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext.to_string_lossy().to_lowercase() == "flac" {
                    flac_files.push(path);
                }
            }
        }
    }

    Ok(flac_files)
}

fn load_flac_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
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
            _ => return Err("Unsupported audio format".into()),
        }
    }

    Ok(audio_samples)
}

fn extract_15d_features(audio: &[f32], sample_rate: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let extractor = MicroDynamicsExtractor::new(sample_rate);
    let features = extractor.extract_15d_marmoset(audio)?;
    Ok(features.to_array().to_vec())
}

fn cluster_phrases(phrases: &[ExtractedPhrase]) -> Result<Vec<(String, Vec<f32>)>, Box<dyn std::error::Error>> {
    use technical_architecture::hdbscan::HdbscanClustering;
    use ndarray::Array2;

    if phrases.is_empty() {
        return Ok(Vec::new());
    }

    let n_samples = phrases.len();
    let n_features = phrases[0].features.len();

    // Build feature matrix
    let mut feature_matrix = Array2::zeros((n_samples, n_features));
    for (i, phrase) in phrases.iter().enumerate() {
        for (j, &val) in phrase.features.iter().enumerate() {
            feature_matrix[[i, j]] = val as f64;
        }
    }

    // Cluster with HDBSCAN
    let min_cluster_size = (n_samples as f64).sqrt().max(5.0) as usize;
    let min_samples = (min_cluster_size * 3) / 4;

    let hdbscan = HdbscanClustering::new(min_cluster_size, min_samples);
    let labels = hdbscan.fit_predict(&feature_matrix)?;

    // Compute cluster centroids
    let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        clusters.entry(label).or_insert_with(Vec::new).push(i);
    }

    let mut vocabulary = Vec::new();
    for (cluster_id, members) in clusters {
        if cluster_id < 0 {
            continue;  // Skip noise
        }

        // Compute centroid
        let mut centroid = vec![0.0f32; n_features];
        for &idx in &members {
            for (j, &val) in phrases[idx].features.iter().enumerate() {
                centroid[j] += val;
            }
        }
        for val in centroid.iter_mut() {
            *val /= members.len() as f32;
        }

        let word_id = format!("word_{}", cluster_id);
        vocabulary.push((word_id, centroid));
    }

    Ok(vocabulary)
}

fn find_closest_vocabulary_word(features: &[f32], vocabulary: &[(String, Vec<f32>)]) -> String {
    let mut best_distance = f32::MAX;
    let mut best_word = "word_0".to_string();

    for (word_id, centroid) in vocabulary {
        let distance = cosine_distance(features, centroid);
        if distance < best_distance {
            best_distance = distance;
            best_word = word_id.clone();
        }
    }

    best_word
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - dot / (norm_a.sqrt() * norm_b.sqrt())
}

fn calculate_generality_metrics(
    pcm: &HashMap<String, HashMap<String, usize>>,
    totals: &HashMap<String, usize>,
    n_contexts: usize,
) -> Result<Vec<GeneralityMetrics>, Box<dyn std::error::Error>> {
    let mut metrics = Vec::new();

    for (phrase_id, context_counts) in pcm {
        let total_occurrences = totals[phrase_id];
        let contexts_used = context_counts.len();

        let generality_score = if n_contexts > 0 {
            contexts_used as f64 / n_contexts as f64
        } else {
            0.0
        };

        let shannon_entropy = calculate_shannon_entropy(context_counts, total_occurrences);
        let max_entropy = (n_contexts as f64).log2();
        let normalized_entropy = if max_entropy > 0.0 {
            shannon_entropy / max_entropy
        } else {
            0.0
        };

        metrics.push(GeneralityMetrics {
            phrase_id: phrase_id.clone(),
            total_occurrences,
            contexts_used,
            total_contexts: n_contexts,
            generality_score,
            shannon_entropy,
            normalized_entropy,
            classification: PhraseType::Rare,
        });
    }

    Ok(metrics)
}

fn calculate_shannon_entropy(context_counts: &HashMap<String, usize>, total: usize) -> f64 {
    let mut entropy = 0.0;
    for &count in context_counts.values() {
        if count > 0 && total > 0 {
            let p = count as f64 / total as f64;
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn classify_phrases(metrics: &mut [GeneralityMetrics]) {
    for m in metrics.iter_mut() {
        m.classification = if m.total_occurrences < 5 {
            PhraseType::Rare
        } else if m.generality_score >= 0.8 {
            PhraseType::UniversalGeneralist
        } else if m.generality_score >= 0.5 {
            PhraseType::Generalist
        } else if m.normalized_entropy >= 0.6 {
            PhraseType::FlexibleSpecialist
        } else if m.generality_score >= 0.2 {
            PhraseType::ContextSpecialist
        } else {
            PhraseType::HighlySpecific
        };
    }
}

fn count_phrase_types(metrics: &[GeneralityMetrics]) -> (usize, usize, usize, usize, usize) {
    let mut counts = (0, 0, 0, 0, 0);
    for m in metrics {
        match m.classification {
            PhraseType::UniversalGeneralist => counts.0 += 1,
            PhraseType::Generalist => counts.1 += 1,
            PhraseType::FlexibleSpecialist => counts.2 += 1,
            PhraseType::ContextSpecialist => counts.3 += 1,
            PhraseType::HighlySpecific | PhraseType::Rare => counts.4 += 1,
        }
    }
    counts
}

fn run_permutation_test(
    pcm: &HashMap<String, HashMap<String, usize>>,
    totals: &HashMap<String, usize>,
    n_permutations: usize,
    n_contexts: usize,
) -> Result<PermutationTestResult, Box<dyn std::error::Error>> {
    use rand::Rng;

    let observed_gens: Vec<f64> = pcm.keys()
        .map(|phrase_id| {
            pcm.get(phrase_id)
                .map(|ctxs| ctxs.len() as f64 / n_contexts as f64)
                .unwrap_or(0.0)
        })
        .collect();

    let observed_mean = observed_gens.iter().sum::<f64>() / observed_gens.len() as f64;

    let mut all_pairs: Vec<(String, String)> = Vec::new();
    for (phrase_id, context_counts) in pcm {
        for (context, &count) in context_counts {
            for _ in 0..count {
                all_pairs.push((context.clone(), phrase_id.clone()));
            }
        }
    }

    let null_means: Vec<f64> = (0..n_permutations)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let mut shuffled_contexts: Vec<String> = all_pairs.iter().map(|(ctx, _)| ctx.clone()).collect();
            shuffled_contexts.shuffle(&mut rng);

            let mut phrase_context_counts: HashMap<String, usize> = HashMap::new();
            for ((_, phrase_id), _) in all_pairs.iter().zip(shuffled_contexts.iter()) {
                *phrase_context_counts.entry(phrase_id.clone()).or_insert(0) += 1;
            }

            let gen_scores: Vec<f64> = phrase_context_counts.values()
                .map(|&n_ctx| n_ctx as f64 / n_contexts as f64)
                .collect();

            gen_scores.iter().sum::<f64>() / gen_scores.len() as f64
        })
        .collect();

    let null_mean = null_means.iter().sum::<f64>() / null_means.len() as f64;
    let null_variance = null_means.iter()
        .map(|&x| (x - null_mean).powi(2))
        .sum::<f64>() / null_means.len() as f64;
    let null_std = null_variance.sqrt();

    let z_score = if null_std > 0.0 {
        (observed_mean - null_mean) / null_std
    } else {
        0.0
    };

    let count_ge_observed = null_means.iter().filter(|&&x| x >= observed_mean).count();
    let p_value = (count_ge_observed + 1) as f64 / (n_permutations + 1) as f64;

    Ok(PermutationTestResult {
        observed_mean_generality: observed_mean,
        null_mean_generality: null_mean,
        null_std_generality: null_std,
        p_value,
        z_score,
        significant: p_value < 0.05,
        n_permutations,
    })
}

fn compute_summary_statistics(metrics: &[GeneralityMetrics]) -> SummaryStatistics {
    let type_counts = count_phrase_types(metrics);

    let gen_scores: Vec<f64> = metrics.iter().map(|m| m.generality_score).collect();
    let mean_gen = gen_scores.iter().sum::<f64>() / gen_scores.len() as f64;
    let mut sorted_gen = gen_scores.clone();
    sorted_gen.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_gen = sorted_gen[sorted_gen.len() / 2];
    let var_gen = gen_scores.iter().map(|x| (x - mean_gen).powi(2)).sum::<f64>() / gen_scores.len() as f64;

    let entropies: Vec<f64> = metrics.iter().map(|m| m.shannon_entropy).collect();
    let mean_ent = entropies.iter().sum::<f64>() / entropies.len() as f64;
    let mut sorted_ent = entropies.clone();
    sorted_ent.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ent = sorted_ent[sorted_ent.len() / 2];

    SummaryStatistics {
        n_universal_phrases: type_counts.0,
        n_generalist_phrases: type_counts.1,
        n_flexible_specialist_phrases: type_counts.2,
        n_context_specialist_phrases: type_counts.3,
        n_highly_specific_phrases: type_counts.4,
        mean_generality_score: mean_gen,
        median_generality_score: median_gen,
        std_generality_score: var_gen.sqrt(),
        mean_shannon_entropy: mean_ent,
        median_shannon_entropy: median_ent,
    }
}

fn save_generality_csv(metrics: &[GeneralityMetrics], path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;

    wtr.write_record(&[
        "phrase_id",
        "total_occurrences",
        "contexts_used",
        "generality_score",
        "shannon_entropy",
        "normalized_entropy",
        "classification",
    ])?;

    for m in metrics {
        let class_str = match m.classification {
            PhraseType::UniversalGeneralist => "Universal Generalist",
            PhraseType::Generalist => "Generalist",
            PhraseType::FlexibleSpecialist => "Flexible Specialist",
            PhraseType::ContextSpecialist => "Context Specialist",
            PhraseType::HighlySpecific => "Highly Specific",
            PhraseType::Rare => "Rare",
        };

        wtr.write_record(&[
            m.phrase_id.to_string(),
            m.total_occurrences.to_string(),
            m.contexts_used.to_string(),
            format!("{:.4}", m.generality_score),
            format!("{:.4}", m.shannon_entropy),
            format!("{:.4}", m.normalized_entropy),
            class_str.to_string(),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

trait Shuffle<T> {
    fn shuffle(&mut self, rng: &mut rand::rngs::ThreadRng);
}

impl<T> Shuffle<T> for [T] {
    fn shuffle(&mut self, rng: &mut rand::rngs::ThreadRng) {
        for i in (1..self.len()).rev() {
            let j = rng.gen_range(0..i + 1);
            self.swap(i, j);
        }
    }
}
