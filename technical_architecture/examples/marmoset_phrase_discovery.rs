//! Marmoset Phrase Discovery Pipeline
//!
//! Runs the unified phrase discovery pipeline on the full marmoset dataset.
//!
//! Pipeline:
//! 1. Dynamic Segmentation (Find acoustic boundaries)
//! 2. Feature Extraction (45D vectors)
//! 3. Acoustic Similarity Clustering (Discover phrase types)
//! 4. Syntax Analysis (Build transition model)
//!
//! Usage:
//!   cargo run --release --example marmoset_phrase_discovery

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use technical_architecture::{
    AcousticSimilarityEngine, DynamicSegmenter, DynamicSegmenterConfig, HierarchicalThresholds, PhraseDiscoveryConfig,
    SimilarityMetric, ZooVoxFeatureExtractor,
};

const FEATURE_DIM: usize = 45;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let max_files: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000); // Default to 1000 files

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║            Marmoset Phrase Discovery Pipeline                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let base_dir = PathBuf::from(std::env::var("HOME").unwrap()).join("birdsong_analysis/data/Vocalizations");

    let config = PhraseDiscoveryConfig::marmoset();
    println!("Configuration:");
    println!("  ├─ Sample Rate: {}Hz", config.sample_rate);
    println!("  ├─ Atomic Granularity: {:?}", config.atomic_granularity);
    println!("  ├─ Similarity Threshold: {}", config.similarity_threshold);
    println!("  └─ Max Files: {}", max_files);
    println!();

    let total_start = Instant::now();

    // =========================================================================
    // Step 1: Discover Files
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[1/4] Discovering Files");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let files = discover_marmoset_files(&base_dir, max_files);
    println!("Found {} FLAC files", files.len());

    if files.is_empty() {
        eprintln!("No files found!");
        return Ok(());
    }

    // =========================================================================
    // Step 2: Dynamic Segmentation (Parallel)
    // =========================================================================
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[2/4] Dynamic Segmentation (Finding Acoustic Boundaries)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let seg_start = Instant::now();
    let thresholds = HierarchicalThresholds::marmoset();
    let segmenter_config = DynamicSegmenterConfig::for_syllable_level(&thresholds);
    let sample_rate = config.sample_rate;

    let processed = Arc::new(AtomicUsize::new(0));
    let total_files = files.len();

    let all_candidates: Vec<(Vec<f64>, f32, String, String)> = files
        .par_iter()
        .flat_map(|(path, filename, call_type)| {
            // Progress tracking
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 100 == 0 {
                print!("\r  Progress: {}/{}", count, total_files);
                std::io::stdout().flush().ok();
            }

            // Load audio
            let audio = match load_flac_audio(path) {
                Ok(a) => a,
                Err(_) => return Vec::new(),
            };

            if audio.len() < 500 {
                return Vec::new();
            }

            // Segment
            let segmenter = DynamicSegmenter::new(segmenter_config.clone(), sample_rate);
            let extractor = Arc::new(std::sync::Mutex::new(ZooVoxFeatureExtractor::new(sample_rate)));

            let extract_fn = |frame: &[f32], _sr: u32| {
                let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
                let mut ext = extractor.lock().unwrap();
                ext.extract_45d(&frame_f64).ok().map(|f| f.to_vector().to_vec())
            };

            let result = segmenter.segment(&audio, extract_fn, filename);

            // Return (features, duration_ms, filename, call_type)
            result
                .candidates
                .into_iter()
                .map(|c| (c.features, c.duration_ms, filename.clone(), call_type.clone()))
                .collect::<Vec<_>>()
        })
        .collect();

    println!("\r  Progress: {}/{}", total_files, total_files);
    let seg_time = seg_start.elapsed();
    println!();
    println!("Extracted {} phrase candidates", all_candidates.len());
    println!("Segmentation time: {:.1}s", seg_time.as_secs_f64());

    if all_candidates.is_empty() {
        eprintln!("No candidates extracted!");
        return Ok(());
    }

    // =========================================================================
    // Step 3: Acoustic Similarity Clustering
    // =========================================================================
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[3/4] Acoustic Similarity Clustering (Discovering Phrase Types)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let cluster_start = Instant::now();

    // Build feature matrix
    let n_samples = all_candidates.len();
    let mut feature_matrix = ndarray::Array2::<f64>::zeros((n_samples, FEATURE_DIM));
    for (i, (features, _, _, _)) in all_candidates.iter().enumerate() {
        for (j, &val) in features.iter().take(FEATURE_DIM).enumerate() {
            feature_matrix[[i, j]] = val;
        }
    }

    // Create similarity engine and fit normalization
    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    engine.fit_normalization(&feature_matrix);

    println!("Fitted normalization on {} samples", n_samples);

    // Cluster using similarity
    let phrase_types = cluster_by_similarity(&all_candidates, &engine, config.similarity_threshold);

    let cluster_time = cluster_start.elapsed();
    println!("Discovered {} phrase types", phrase_types.len());
    println!("Clustering time: {:.1}s", cluster_time.as_secs_f64());

    // Calculate vocabulary reduction
    let vocab_reduction = n_samples as f64 / phrase_types.len().max(1) as f64;
    println!(
        "Vocabulary reduction: {:.1}x ({} → {})",
        vocab_reduction,
        n_samples,
        phrase_types.len()
    );

    // =========================================================================
    // Step 4: Syntax Analysis
    // =========================================================================
    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("[4/4] Syntax Analysis (Building Transition Model)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let syntax_start = Instant::now();

    // Build phrase sequence (by source file)
    let mut file_phrases: HashMap<String, Vec<(usize, usize, String)>> = HashMap::new(); // (position, phrase_id, type)
    for (idx, (_, _, source_file, _)) in all_candidates.iter().enumerate() {
        // Find which type this belongs to
        for (type_idx, pt) in phrase_types.iter().enumerate() {
            if pt.indices.contains(&idx) {
                file_phrases.entry(source_file.clone()).or_insert_with(Vec::new).push((
                    idx,
                    type_idx,
                    pt.type_id.clone(),
                ));
                break;
            }
        }
    }

    // Build transitions
    let mut transitions: HashMap<(String, String), usize> = HashMap::new();
    for (_, mut phrases) in file_phrases {
        phrases.sort_by_key(|(pos, _, _)| *pos);
        for window in phrases.windows(2) {
            let from = &window[0].2;
            let to = &window[1].2;
            *transitions.entry((from.clone(), to.clone())).or_insert(0) += 1;
        }
    }

    // Calculate entropy
    let total_transitions: usize = transitions.values().sum();
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for ((_, to), _) in &transitions {
        *type_counts.entry(to.clone()).or_insert(0) += 1;
    }

    let type_entropy = if total_transitions > 0 {
        type_counts
            .values()
            .map(|&count| {
                let p = count as f64 / total_transitions as f64;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum()
    } else {
        0.0
    };

    let syntax_time = syntax_start.elapsed();
    println!("Unique transitions: {}", transitions.len());
    println!("Type entropy: {:.3} bits", type_entropy);
    println!("Syntax analysis time: {:.1}s", syntax_time.as_secs_f64());

    // =========================================================================
    // Summary
    // =========================================================================
    let total_time = total_start.elapsed();

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("COMPLETE ANALYSIS SUMMARY: MARMOSET PHRASE DISCOVERY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PIPELINE STATISTICS                                                         │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Files Processed:      {}", files.len());
    println!("│ Candidates Extracted: {}", n_samples);
    println!("│ Phrase Types Found:   {}", phrase_types.len());
    println!("│ Vocabulary Reduction: {:.1}x", vocab_reduction);
    println!("│ Type Entropy:         {:.3} bits", type_entropy);
    println!("│ Unique Transitions:   {}", transitions.len());
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Duration statistics
    let durations: Vec<f32> = all_candidates.iter().map(|(_, d, _, _)| *d).collect();
    let avg_dur = durations.iter().sum::<f32>() / durations.len() as f32;
    let min_dur = durations.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_dur = durations.iter().cloned().fold(0.0f32, f32::max);

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ ACOUSTIC CHARACTERISTICS                                                    │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Duration Range:       {:.1} - {:.1}ms", min_dur, max_dur);
    println!("│ Average Duration:     {:.1}ms", avg_dur);
    println!("│ Atomic Level:         Syllables (carrier of meaning)");
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Top phrase types
    let mut sorted_types: Vec<_> = phrase_types.iter().collect();
    sorted_types.sort_by(|a, b| b.indices.len().cmp(&a.indices.len()));

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TOP 10 PHRASE TYPES                                                         │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    for (i, pt) in sorted_types.iter().take(10).enumerate() {
        println!(
            "│ {}. {} - {} instances, avg {:.1}ms",
            i + 1,
            pt.type_id,
            pt.indices.len(),
            pt.avg_duration_ms
        );
    }
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Top transitions
    let mut sorted_transitions: Vec<_> = transitions.iter().collect();
    sorted_transitions.sort_by(|a, b| b.1.cmp(a.1));

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TOP 10 TRANSITIONS                                                          │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    for ((from, to), count) in sorted_transitions.iter().take(10) {
        println!("│ {} → {} ({} occurrences)", from, to, count);
    }
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Timing summary
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TIMING                                                                      │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│ Segmentation:  {:.1}s", seg_time.as_secs_f64());
    println!("│ Clustering:    {:.1}s", cluster_time.as_secs_f64());
    println!("│ Syntax:        {:.1}s", syntax_time.as_secs_f64());
    println!("│ Total:         {:.1}s", total_time.as_secs_f64());
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    // Save results
    let report = PhraseDiscoveryReport {
        files_processed: files.len(),
        candidates_extracted: n_samples,
        phrase_types: phrase_types.len(),
        vocabulary_reduction: vocab_reduction,
        type_entropy,
        unique_transitions: transitions.len(),
        duration_range_ms: (min_dur, max_dur),
        avg_duration_ms: avg_dur,
        total_time_sec: total_time.as_secs_f64(),
    };

    let report_path = "complete_analysis/marmoset_phrase_discovery.json";
    std::fs::create_dir_all("complete_analysis").ok();
    let file = File::create(report_path)?;
    serde_json::to_writer_pretty(BufWriter::new(file), &report)?;
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
struct PhraseDiscoveryReport {
    files_processed: usize,
    candidates_extracted: usize,
    phrase_types: usize,
    vocabulary_reduction: f64,
    type_entropy: f64,
    unique_transitions: usize,
    duration_range_ms: (f32, f32),
    avg_duration_ms: f32,
    total_time_sec: f64,
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

fn cluster_by_similarity(
    candidates: &[(Vec<f64>, f32, String, String)],
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

        // Calculate average duration
        let avg_dur: f32 = indices.iter().map(|&idx| candidates[idx].1).sum::<f32>() / indices.len() as f32;

        types.push(PhraseType {
            type_id: format!("Type_{}", types.len() + 1),
            indices,
            avg_duration_ms: avg_dur,
        });
    }

    types
}
