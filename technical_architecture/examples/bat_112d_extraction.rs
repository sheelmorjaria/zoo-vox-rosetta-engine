//! Egyptian Fruit Bat 112D Feature Extraction
//! ==========================================
//!
//! Fresh extraction of 112D RosettaFeatures from bat audio files,
//! followed by HDBSCAN clustering for phrase discovery.
//!
//! Pipeline:
//! 1. Load audio files from data/egyptian_fruit_bats/audio/
//! 2. Extract 112D features using MicroDynamicsExtractor
//! 3. Apply HDBSCAN clustering
//! 4. Export results with phrase assignments
//!
//! Usage:
//!   cargo run --release --example bat_112d_extraction

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use technical_architecture::hdbscan::{DistanceMetric, HdbscanClustering};
use technical_architecture::{MicroDynamicsExtractor, RosettaFeatures};

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Serialize)]
struct ExtractedSegment {
    file_name: String,
    start_sample: usize,
    end_sample: usize,
    duration_ms: f64,
    features_112d: Vec<f32>,
    cluster_id: Option<i32>,
}

#[derive(Debug, Clone, Serialize)]
struct ExtractionResults {
    total_files: usize,
    total_segments: usize,
    feature_dimension: usize,
    cluster_count: usize,
    noise_count: usize,
    segments: Vec<ExtractedSegment>,
    cluster_stats: HashMap<i32, ClusterStats>,
}

#[derive(Debug, Clone, Serialize)]
struct ClusterStats {
    cluster_id: i32,
    segment_count: usize,
    centroid: Vec<f32>,
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
    println!("║     Egyptian Fruit Bat 112D Feature Extraction                             ║");
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
        .unwrap_or(usize::MAX); // Process all files by default

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
        let file_name = entry.file_name().to_string_lossy().to_string();
        println!("  [{}/{}] Processing {}...", i + 1, audio_files.len(), file_name);

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
                    let end_sample = start_sample + chunk.len();

                    match extractor.extract(chunk) {
                        Ok(features) => {
                            let features_vec = features.to_vec();

                            all_segments.push(ExtractedSegment {
                                file_name: file_name.clone(),
                                start_sample,
                                end_sample,
                                duration_ms: segment_ms as f64,
                                features_112d: features_vec,
                                cluster_id: None,
                            });
                        }
                        Err(e) => {
                            eprintln!("    Warning: Failed to extract features: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("    Warning: Failed to load audio: {}", e);
            }
        }
    }

    println!();
    println!("  ✅ Extracted {} segments", all_segments.len());
    println!();

    // ========================================================================
    // Step 3: Normalize Features
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Normalizing Features (Z-score)                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Compute mean and std for each dimension
    let n_dims = 112;
    let n_segments = all_segments.len();

    let mut means = vec![0.0f32; n_dims];
    let mut stds = vec![0.0f32; n_dims];

    for dim in 0..n_dims {
        let values: Vec<f32> = all_segments.iter().map(|s| s.features_112d[dim]).collect();

        let mean = values.iter().sum::<f32>() / n_segments.max(1) as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n_segments.max(1) as f32;
        let std = variance.sqrt();

        means[dim] = mean;
        stds[dim] = std.max(0.001); // Avoid division by zero
    }

    // Normalize
    for segment in &mut all_segments {
        for dim in 0..n_dims {
            segment.features_112d[dim] = (segment.features_112d[dim] - means[dim]) / stds[dim];
        }
    }

    println!("  ✅ Features normalized");
    println!();

    // ========================================================================
    // Step 4: HDBSCAN Clustering
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: HDBSCAN Clustering                                              │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Convert to ndarray format expected by HDBSCAN
    let mut data_vec = Vec::new();
    for segment in &all_segments {
        data_vec.push(segment.features_112d.iter().map(|&v| v as f64).collect::<Vec<f64>>());
    }

    // Find max dimension
    let max_dim = data_vec.iter().map(|v| v.len()).max().unwrap_or(0);

    // Pad all vectors to same length
    for vec in &mut data_vec {
        while vec.len() < max_dim {
            vec.push(0.0);
        }
    }

    let flattened: Vec<f64> = data_vec.into_iter().flatten().collect();
    let data_array = Array2::from_shape_vec((all_segments.len(), max_dim), flattened)
        .with_context(|| "Failed to create data array")?;

    let hdbscan = HdbscanClustering::new(5, 3)?;

    println!("  Running HDBSCAN (min_cluster_size=5, min_samples=3)...");

    let labels = hdbscan.fit_predict(&data_array)?;

    // Assign clusters
    let mut cluster_counts: HashMap<i32, usize> = HashMap::new();
    for (segment, label) in all_segments.iter_mut().zip(labels.iter()) {
        segment.cluster_id = Some(*label);
        *cluster_counts.entry(*label).or_insert(0) += 1;
    }

    // Filter out noise (-1) for cluster count
    let cluster_count = cluster_counts.iter().filter(|(&k, _)| k >= 0).count();
    let noise_count = cluster_counts.get(&-1).copied().unwrap_or(0);

    println!("  ✅ Clustering complete");
    println!("     Clusters found: {}", cluster_count);
    println!("     Noise points: {}", noise_count);
    println!();

    // ========================================================================
    // Step 5: Export Results
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Exporting Results                                                │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let results = ExtractionResults {
        total_files: audio_files.len(),
        total_segments: all_segments.len(),
        feature_dimension: n_dims,
        cluster_count,
        noise_count,
        segments: all_segments,
        cluster_stats: HashMap::new(),
    };

    let output_path = output_dir.join("extraction_112d_results.json");
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&output_path, json)?;

    println!("  ✅ Results exported to: {:?}", output_path);
    println!();

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Extraction Complete!                                                ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}
