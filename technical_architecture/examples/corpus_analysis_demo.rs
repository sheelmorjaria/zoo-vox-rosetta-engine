//! Full Corpus Analysis Pipeline - Egyptian Fruit Bats
//!
//! This example performs end-to-end corpus analysis:
//! 1. Load raw audio vocalizations
//! 2. Extract NBD (Neural Boundary Detection) segments
//! 3. Extract RosettaFeatures (112D) from each segment
//! 4. Cluster segments to get symbolic labels
//! 5. Build symbolic sequences per vocalization
//! 6. Compute corpus-wide n-gram statistics

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// Import from parent crate
use technical_architecture::{
    NgramConfig, NgramCorpusStats,
    NeuralBoundaryDetector, BoundaryDetectorConfig,
    segment_phrases_by_boundaries,
    MicroDynamicsExtractor,
};

/// Report structure for saving analysis results
#[derive(Serialize, Deserialize)]
struct AnalysisReport {
    total_vocalizations: usize,
    total_segments: usize,
    unique_segment_types: usize,
    unique_ngrams: usize,
    avg_segments_per_vocalization: f64,
    top_bigrams: Vec<(Vec<u32>, usize)>,
    top_trigrams: Vec<(Vec<u32>, usize)>,
    top_4grams: Vec<(Vec<u32>, usize)>,
    top_5grams: Vec<(Vec<u32>, usize)>,
    analysis_timestamp: String,
}

/// Segment with extracted features
#[derive(Debug, Clone)]
struct Segment {
    vocalization_id: String,
    segment_idx: usize,
    start_sample: usize,
    end_sample: usize,
    features: Vec<f32>,
}

fn main() -> Result<()> {
    println!("╔═════════════════════════════════════════════════════════════════════════════════╗");
    println!("║     Egyptian Fruit Bat - Full Corpus Analysis Pipeline                          ║");
    println!("║     NBD Segmentation → Feature Extraction → Clustering → N-gram Analysis       ║");
    println!("╚═════════════════════════════════════════════════════════════════════════════════╝");

    // Paths
    let audio_dir = PathBuf::from("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio");
    let output_dir = PathBuf::from("corpus_analysis_output");

    // Create output directory
    fs::create_dir_all(&output_dir)?;

    // Configuration
    let sample_rate: u32 = 44100;
    let max_files: Option<usize> = None; // None = process all 91k files

    println!("\n[Phase 1] NBD Segmentation & Feature Extraction");
    println!("═════════════════════════════════════════════════════════════════════════════════");

    // Initialize NBD detector
    let nbd_config = BoundaryDetectorConfig {
        hop_size: 512,
        sample_rate,
        min_phrase_duration_ms: 30.0,  // Minimum 30ms segments
        threshold: 0.3,                 // Lower threshold for more sensitive detection
        smoothing_frames: 3,
    };
    let mut nbd_detector = NeuralBoundaryDetector::with_config(nbd_config);

    // Initialize feature extractor
    let feature_extractor = MicroDynamicsExtractor::new(sample_rate);

    // Collect all segments with features
    let mut all_segments: Vec<Segment> = Vec::new();
    let mut vocalization_sequences: HashMap<String, Vec<usize>> = HashMap::new();

    // Get list of audio files
    let audio_files: Vec<PathBuf> = fs::read_dir(&audio_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|ext| ext == "wav").unwrap_or(false))
        .take(max_files.unwrap_or(usize::MAX))
        .collect();

    println!("Found {} audio files to process", audio_files.len());

    // Process each vocalization
    for (idx, audio_path) in audio_files.iter().enumerate() {
        let vocalization_id = audio_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Load audio
        if let Ok(audio) = load_wav_audio(&audio_path, sample_rate) {
            if audio.is_empty() {
                continue;
            }

            // Detect boundaries using NBD
            nbd_detector.reset();
            let boundaries = nbd_detector.detect_boundaries(&audio);

            // Segment into phrases
            let phrases = segment_phrases_by_boundaries(&audio, &boundaries, sample_rate);

            // Extract features from each segment
            for (seg_idx, phrase_audio) in phrases.iter().enumerate() {
                let min_samples = (sample_rate as f32 * 10.0 / 1000.0) as usize;
                if phrase_audio.len() < min_samples {
                    // Skip segments shorter than 10ms
                    continue;
                }

                // Extract RosettaFeatures (112D)
                if let Ok(features_112d) = feature_extractor.extract_rosetta(phrase_audio) {
                    let feature_vec = features_112d.to_array().to_vec();

                    let segment = Segment {
                        vocalization_id: vocalization_id.clone(),
                        segment_idx: seg_idx,
                        start_sample: 0, // Would need to track from boundaries
                        end_sample: phrase_audio.len(),
                        features: feature_vec,
                    };

                    all_segments.push(segment);
                }
            }

            if idx > 0 && idx % 1000 == 0 {
                println!("  Processed {}/{} files, {} total segments extracted",
                    idx, audio_files.len(), all_segments.len());
            }
        }
    }

    println!("\n  Total segments extracted: {}", all_segments.len());

    // Phase 2: Clustering
    println!("\n[Phase 2] Feature Clustering");
    println!("═════════════════════════════════════════════════════════════════════════════════");

    // Simple clustering: hash features to discrete labels
    // For production, use HDBSCAN or k-means on the feature vectors
    let cluster_labels = cluster_features_simple(&all_segments);
    let unique_clusters: std::collections::HashSet<u32> = cluster_labels.iter().copied().collect();

    println!("  Unique cluster types: {}", unique_clusters.len());

    // Build sequences per vocalization
    println!("\n[Phase 3] Building Symbolic Sequences");
    println!("═════════════════════════════════════════════════════════════════════════════════");

    let mut vocalization_to_sequence: HashMap<String, Vec<u32>> = HashMap::new();
    for (seg, &label) in all_segments.iter().zip(cluster_labels.iter()) {
        vocalization_to_sequence
            .entry(seg.vocalization_id.clone())
            .or_insert_with(Vec::new)
            .push(label);
    }

    println!("  Vocalizations with segments: {}", vocalization_to_sequence.len());

    // Phase 4: Corpus Analysis
    println!("\n[Phase 4] N-gram Corpus Analysis");
    println!("═════════════════════════════════════════════════════════════════════════════════");

    // Create corpus stats with configurable N-gram sizes (supports arbitrary N)
    let config = NgramConfig {
        min_ngram_size: 2,
        max_ngram_size: 5,  // Can be increased for longer patterns
        track_occurrences: true,
        track_contexts: true,
    };
    let stats = Arc::new(NgramCorpusStats::with_config(config));

    // Process each vocalization's sequence
    for (vocalization_id, sequence) in &vocalization_to_sequence {
        if sequence.len() >= 2 {
            stats.process_file(vocalization_id, sequence, None);
        }
    }

    // Get summary
    let summary = stats.summary();
    let avg_len = if summary.total_files > 0 {
        summary.total_segments as f64 / summary.total_files as f64
    } else {
        0.0
    };

    println!("\n╔═════════════════════════════════════════════════════════════════════════════════╗");
    println!("║     CORPUS ANALYSIS RESULTS                                                    ║");
    println!("╚═════════════════════════════════════════════════════════════════════════════════╝");
    println!("  Vocalizations analyzed: {}", summary.total_files);
    println!("  Total NBD segments: {}", summary.total_segments);
    println!("  Unique segment types: {}", summary.unique_segments);
    println!("  Unique n-grams (2-5): {}", summary.unique_ngrams);
    println!("  Avg segments/vocalization: {:.2}", avg_len);
    println!("  ─────────────────────────────────────────────────────────────────────────────");
    println!("  SYNTACTIC DEPTH (Longest Repeated N-gram): {}", summary.max_ngram_length);

    // Find the actual longest repeated n-gram
    if let Some((pattern, count)) = stats.find_longest_repeated_ngram(2, 20) {
        let pattern_str = pattern.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
        println!("  Longest pattern: [{}] (appears {} times)", pattern_str, count);
    }

    // Top patterns by size
    for ngram_size in 2..=5 {
        let top_patterns = stats.get_top_ngrams(10, Some(ngram_size));
        if !top_patterns.is_empty() {
            println!("\n=== Top 10 {}-grams ===", ngram_size);
            for (i, (pattern, count)) in top_patterns.iter().enumerate() {
                let pattern_str = format!("[{}]", pattern.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
                let files_with_pattern = stats.get_files_with_pattern(&pattern);
                let prevalence = if summary.total_files > 0 {
                    (files_with_pattern.len() as f64 / summary.total_files as f64) * 100.0
                } else {
                    0.0
                };
                println!(
                    "  {}. {} - Count: {}, In {} files ({:.1}% prevalence)",
                    i + 1, pattern_str, count, files_with_pattern.len(), prevalence
                );
            }
        }
    }

    // Save report
    let report = AnalysisReport {
        total_vocalizations: summary.total_files,
        total_segments: summary.total_segments,
        unique_segment_types: summary.unique_segments,
        unique_ngrams: summary.unique_ngrams,
        avg_segments_per_vocalization: avg_len,
        top_bigrams: stats.get_top_ngrams(50, Some(2)),
        top_trigrams: stats.get_top_ngrams(50, Some(3)),
        top_4grams: stats.get_top_ngrams(50, Some(4)),
        top_5grams: stats.get_top_ngrams(50, Some(5)),
        analysis_timestamp: chrono::Utc::now().to_rfc3339(),
    };

    let output_path = output_dir.join("corpus_analysis_report.json");
    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&output_path, json)?;

    println!("\n╔═════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  Report saved to: corpus_analysis_output/corpus_analysis_report.json           ║");
    println!("╚═════════════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Load audio from WAV file
fn load_wav_audio(path: &Path, target_sample_rate: u32) -> Result<Vec<f32>> {
    use hound::WavReader;

    let reader = WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.into_samples::<f32>().filter_map(|s| s.ok()).collect()
        }
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader.into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Simple resampling if needed (linear interpolation)
    if sample_rate != target_sample_rate {
        let ratio = sample_rate as f64 / target_sample_rate as f64;
        let output_len = (samples.len() as f64 / ratio) as usize;
        let mut resampled = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f64 * ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(samples.len() - 1);
            let frac = src_idx - idx0 as f64;

            let sample = samples[idx0] * (1.0 - frac as f32) + samples[idx1] * frac as f32;
            resampled.push(sample);
        }

        Ok(resampled)
    } else {
        Ok(samples)
    }
}

/// Simple feature clustering using quantization
/// For production, use HDBSCAN from the clustering module
fn cluster_features_simple(segments: &[Segment]) -> Vec<u32> {
    // Quantize features to discrete labels
    segments.iter().map(|seg| {
        // Simple hash-based clustering using first few features
        if seg.features.len() >= 4 {
            let f0 = (seg.features[0] * 100.0) as i32;  // mean_f0
            let dur = (seg.features[1] * 10.0) as i32;  // duration
            let hnr = (seg.features[3] * 10.0) as i32;  // HNR

            // Create discrete label from quantized features
            let hash = (f0.abs() * 1000 + dur.abs() * 100 + hnr.abs()) as u32;
            hash % 10000  // Limit to 10k clusters for simplicity
        } else {
            0
        }
    }).collect()
}
