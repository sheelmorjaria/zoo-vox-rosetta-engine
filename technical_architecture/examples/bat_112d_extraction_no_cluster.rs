//! Egyptian Fruit Bat 112D Feature Extraction (No Clustering)
//! ============================================================
//!
//! Fast extraction of 112D RosettaFeatures from bat audio files.
//! Clustering is done separately in Python for better scalability.
//!
//! Usage:
//!   cargo run --release --example bat_112d_extraction_no_cluster

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use technical_architecture::{MicroDynamicsExtractor, RosettaFeatures};

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Serialize)]
struct ExtractedSegment {
    file_name: String,
    start_sample: usize,
    segment_index: usize,
    features_112d: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
struct ExtractionResults {
    total_files: usize,
    total_segments: usize,
    feature_dimension: usize,
    segments: Vec<ExtractedSegment>,
}

// ============================================================================
// Audio Loading
// ============================================================================

fn load_audio_file(path: &Path) -> Result<Vec<f32>> {
    use hound::WavReader;

    let reader = WavReader::open(path).with_context(|| format!("Failed to open audio file: {:?}", path))?;

    let spec = reader.spec();
    let samples: Vec<f32> = reader.into_samples::<f32>().map(|s| s.unwrap_or(0.0)).collect();

    // Convert to mono if stereo
    if spec.channels == 2 {
        let mono: Vec<f32> = samples.chunks(2).map(|pair| (pair[0] + pair[1]) / 2.0).collect();
        Ok(mono)
    } else {
        Ok(samples)
    }
}

// ============================================================================
// Main Extraction Pipeline
// ============================================================================

fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Egyptian Fruit Bat 112D Feature Extraction (No Clustering)         ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = data_dir.join("audio");
    let output_dir = data_dir.join("extraction_112d");

    fs::create_dir_all(&output_dir)?;

    // ========================================================================
    // Step 1: List audio files
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Audio Files                                            │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let mut audio_files: Vec<_> = fs::read_dir(&audio_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "wav").unwrap_or(false))
        .collect();

    audio_files.sort_by_key(|e| e.path().file_name().unwrap_or_default().to_string_lossy().to_string());

    // Process subset for demo
    let max_files = std::env::var("MAX_FILES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let audio_files: Vec<_> = audio_files.into_iter().take(max_files).collect();
    let total_to_process = audio_files.len().min(max_files);

    println!(
        "  Found {} audio files (processing {})",
        audio_files.len(),
        total_to_process
    );
    println!();

    // ========================================================================
    // Step 2: Extract 112D Features
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Extracting 112D RosettaFeatures                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let sample_rate = 48000;
    let extractor = MicroDynamicsExtractor::new(sample_rate);

    let mut all_segments = Vec::new();
    let segment_ms = 100.0; // 100ms segments
    let hop_ms = 50.0; // 50ms hop

    for (i, entry) in audio_files.iter().enumerate() {
        if i % 1000 == 0 {
            println!("  [{}/{}] Processing...", i, audio_files.len());
        }

        let file_name = entry.file_name().to_string_lossy().to_string();
        let audio_path = entry.path();

        match load_audio_file(&audio_path) {
            Ok(audio) => {
                let hop_samples = (sample_rate as f32 * hop_ms / 1000.0) as usize;
                let segment_samples = (sample_rate as f32 * segment_ms / 1000.0) as usize;

                for (start, chunk) in audio.chunks(segment_samples).enumerate() {
                    if chunk.len() < segment_samples / 2 {
                        break; // Skip short segments
                    }

                    let start_sample = start * hop_samples;

                    match extractor.extract(chunk) {
                        Ok(features) => {
                            let features_vec = features.to_vec();

                            all_segments.push(ExtractedSegment {
                                file_name: file_name.clone(),
                                start_sample,
                                segment_index: start,
                                features_112d: features_vec,
                            });
                        }
                        Err(e) => {
                            eprintln!("    Warning: Failed to extract features from {}: {}", file_name, e);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("    Warning: Failed to load audio {}: {}", file_name, e);
            }
        }
    }

    println!();
    println!("  ✅ Extracted {} segments", all_segments.len());
    println!();

    // ========================================================================
    // Step 3: Normalize Features (Z-score)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Normalizing Features (Z-score)                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let n_dims = 112;
    let n_segments = all_segments.len();

    // Compute mean and std for each dimension
    let mut means = vec![0.0f32; n_dims];
    let mut stds = vec![0.0f32; n_dims];

    println!(
        "  Computing statistics for {} dimensions across {} segments...",
        n_dims, n_segments
    );

    for dim in 0..n_dims {
        let values: Vec<f32> = all_segments.iter().map(|s| s.features_112d[dim]).collect();

        let mean = values.iter().sum::<f32>() / n_segments.max(1) as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n_segments.max(1) as f32;
        let std = variance.sqrt();

        means[dim] = mean;
        stds[dim] = std.max(0.001); // Avoid division by zero
    }

    // Normalize
    println!("  Normalizing features...");
    for segment in &mut all_segments {
        for dim in 0..n_dims {
            segment.features_112d[dim] = (segment.features_112d[dim] - means[dim]) / stds[dim];
        }
    }

    println!("  ✅ Features normalized");
    println!();

    // ========================================================================
    // Step 4: Export Results (No Clustering)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Exporting Results (No Clustering)                              │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let results = ExtractionResults {
        total_files: audio_files.len(),
        total_segments: all_segments.len(),
        feature_dimension: n_dims,
        segments: all_segments,
    };

    let output_path = output_dir.join("extraction_112d_no_cluster.json");
    println!("  Writing to: {:?}", output_path);

    // Write in chunks to avoid memory issues
    use std::io::Write;
    let mut file = fs::File::create(&output_path)?;
    write!(file, "{{\n")?;
    write!(file, "  \"total_files\": {},\n", results.total_files)?;
    write!(file, "  \"total_segments\": {},\n", results.total_segments)?;
    write!(file, "  \"feature_dimension\": {},\n", results.feature_dimension)?;
    write!(file, "  \"segments\": [\n")?;

    for (i, segment) in results.segments.iter().enumerate() {
        let json = serde_json::to_string(segment)?;
        write!(file, "    {}", json)?;
        if i < results.segments.len() - 1 {
            write!(file, ",")?;
        }
        write!(file, "\n")?;

        if (i + 1) % 100000 == 0 {
            println!("  Written {}/{} segments...", i + 1, results.segments.len());
            file.flush()?;
        }
    }

    write!(file, "  ]\n")?;
    write!(file, "}}\n")?;
    file.flush()?;

    println!("  ✅ Results exported to: {:?}", output_path);
    println!();

    // Also export normalization parameters
    let norm_path = output_dir.join("normalization_params.json");
    let norm_data = serde_json::json!({
        "means": means,
        "stds": stds,
        "n_dims": n_dims,
        "n_segments": n_segments
    });
    fs::write(&norm_path, serde_json::to_string_pretty(&norm_data)?)?;
    println!("  ✅ Normalization params exported to: {:?}", norm_path);
    println!();

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Extraction Complete! (Clustering skipped)                            ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Next steps:");
    println!("  1. Run Python clustering script");
    println!("  2. Perform PCFG analysis on clustered results");

    Ok(())
}
