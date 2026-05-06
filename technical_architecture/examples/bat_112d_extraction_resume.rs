//! Egyptian Fruit Bat 112D Feature Extraction (With Resume Support)
//! ===================================================================
//!
//! Resumable extraction of 112D RosettaFeatures from bat audio files.
//! Can be interrupted and resumed without losing progress.
//!
//! Usage:
//!   cargo run --release --example bat_112d_extraction_resume

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use technical_architecture::{MicroDynamicsExtractor, RosettaFeatures};

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExtractedSegment {
    file_name: String,
    start_sample: usize,
    segment_index: usize,
    features_112d: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointData {
    processed_files: Vec<String>,
    total_segments: usize,
    last_update: String,
}

// ============================================================================
// Checkpoint Management
// ============================================================================

fn load_checkpoint(checkpoint_path: &Path) -> Result<Option<CheckpointData>> {
    if !checkpoint_path.exists() {
        return Ok(None);
    }

    let file = File::open(checkpoint_path)
        .context("Failed to open checkpoint file")?;
    let reader = BufReader::new(file);

    // Try to load as JSON first
    if let Ok(content) = std::io::read_to_string(reader) {
        if let Ok(checkpoint) = serde_json::from_str::<CheckpointData>(&content) {
            return Ok(Some(checkpoint));
        }
    }

    Ok(None)
}

fn save_checkpoint(checkpoint_path: &Path, checkpoint: &CheckpointData) -> Result<()> {
    let json = serde_json::to_string_pretty(checkpoint)?;
    fs::write(checkpoint_path, json)?;
    Ok(())
}

fn append_segment(output_file: &mut File, segment: &ExtractedSegment) -> Result<()> {
    let json = serde_json::to_string(segment)?;
    writeln!(output_file, "{}", json)?;
    output_file.flush()?;
    Ok(())
}

// ============================================================================
// Audio Loading
// ============================================================================

fn load_audio_file(path: &Path) -> Result<Vec<f32>> {
    use hound::WavReader;

    let reader = WavReader::open(path)
        .with_context(|| format!("Failed to open audio file: {:?}", path))?;

    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .into_samples::<f32>()
        .map(|s| s.unwrap_or(0.0))
        .collect();

    // Convert to mono if stereo
    if spec.channels == 2 {
        let mono: Vec<f32> = samples
            .chunks(2)
            .map(|pair| (pair[0] + pair[1]) / 2.0)
            .collect();
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
    println!("║     Egyptian Fruit Bat 112D Feature Extraction (Resumable)             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = data_dir.join("audio");
    let output_dir = data_dir.join("extraction_112d");

    fs::create_dir_all(&output_dir)?;

    let checkpoint_path = output_dir.join("extraction_checkpoint.json");
    let segments_path = output_dir.join("extraction_segments_partial.jsonl");

    // ========================================================================
    // Step 1: Load checkpoint if exists
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Checking for Previous Progress                                  │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let mut processed_files: HashMap<String, bool> = HashMap::new();
    let mut existing_segment_count = 0;

    if let Some(checkpoint) = load_checkpoint(&checkpoint_path)? {
        println!("  📋 Found checkpoint from: {}", checkpoint.last_update);
        println!("  ✅ Previously processed: {} files", checkpoint.processed_files.len());
        println!("  ✅ Previously extracted: {} segments", checkpoint.total_segments);

        for file_name in &checkpoint.processed_files {
            processed_files.insert(file_name.clone(), true);
        }
        existing_segment_count = checkpoint.total_segments;
        println!();
    } else {
        println!("  🆕 No checkpoint found - starting fresh extraction");
        println!();
    }

    // ========================================================================
    // Step 2: List audio files
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Loading Audio Files                                            │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let mut audio_files: Vec<_> = fs::read_dir(&audio_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "wav").unwrap_or(false))
        .collect();

    audio_files.sort_by_key(|e| e.path().file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string());

    // Process subset for demo
    let max_files = std::env::var("MAX_FILES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let audio_files: Vec<_> = audio_files.into_iter().take(max_files).collect();

    let total_files = audio_files.len();
    let already_processed = processed_files.len();
    let remaining = total_files - already_processed;

    println!("  Total files: {}", total_files);
    println!("  Already processed: {}", already_processed);
    println!("  Remaining: {}", remaining);
    println!("  Progress: {:.1}%", (already_processed as f32 / total_files as f32) * 100.0);
    println!();

    if remaining == 0 {
        println!("  ✅ All files already processed! Proceeding to finalization...");
        println!();
        return finalize_extraction(&output_dir, total_files, existing_segment_count);
    }

    // ========================================================================
    // Step 3: Extract 112D Features (with resume)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Extracting 112D RosettaFeatures                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let sample_rate = 48000;
    let extractor = MicroDynamicsExtractor::new(sample_rate);

    let segment_ms = 100.0; // 100ms segments
    let hop_ms = 50.0;      // 50ms hop

    // Open segments file for appending
    let segments_file_exists = segments_path.exists();
    let mut segments_file = OpenOptions::new()
        .write(true)
        .create(true)
        .append(true)
        .open(&segments_path)?;

    // Track new segments
    let new_segment_count = Arc::new(Mutex::new(0usize));
    let checkpoint_interval = 100; // Save checkpoint every 100 files

    let start_time = Instant::now();

    for (i, entry) in audio_files.iter().enumerate() {
        let file_name = entry.file_name().to_string_lossy().to_string();

        // Skip if already processed
        if processed_files.contains_key(&file_name) {
            continue;
        }

        // Progress indicator
        if i % 100 == 0 || i == total_files - 1 {
            let elapsed = start_time.elapsed().as_secs_f32();
            let rate = i as f32 / elapsed.max(0.1);
            let eta = (remaining as f32 - i as f32) / rate.max(0.1);
            println!("  [{}/{}] {} - ETA: {:.1}m", i + 1, total_files, file_name, eta / 60.0);
        }

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

                            let segment = ExtractedSegment {
                                file_name: file_name.clone(),
                                start_sample,
                                segment_index: start,
                                features_112d: features_vec,
                            };

                            // Append to segments file
                            append_segment(&mut segments_file, &segment)?;

                            *new_segment_count.lock().unwrap() += 1;
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

        // Mark as processed
        processed_files.insert(file_name.clone(), true);

        // Save checkpoint periodically
        if (i + 1) % checkpoint_interval == 0 || i == total_files - 1 {
            let checkpoint = CheckpointData {
                processed_files: processed_files.keys().cloned().collect(),
                total_segments: existing_segment_count + *new_segment_count.lock().unwrap(),
                last_update: chrono::Utc::now().to_rfc3339(),
            };
            save_checkpoint(&checkpoint_path, &checkpoint)?;
        }
    }

    let total_segments = existing_segment_count + *new_segment_count.lock().unwrap();
    let elapsed = start_time.elapsed();

    println!();
    println!("  ✅ Extracted {} new segments (total: {})", new_segment_count.lock().unwrap(), total_segments);
    println!("  ⏱️  Time: {:.1}s ({:.1} files/sec)", elapsed.as_secs_f32(), remaining as f32 / elapsed.as_secs_f32());
    println!();

    // ========================================================================
    // Step 4: Finalize extraction
    // ========================================================================

    finalize_extraction(&output_dir, total_files, total_segments)
}

fn finalize_extraction(output_dir: &Path, total_files: usize, total_segments: usize) -> Result<()> {
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Normalizing and Merging Results                                │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let segments_path = output_dir.join("extraction_segments_partial.jsonl");
    let output_path = output_dir.join("extraction_112d_no_cluster.json");

    println!("  Loading {} segments from partial file...", total_segments);

    // Load all segments from the JSONL file
    let mut all_segments = Vec::new();
    let file = File::open(&segments_path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        if let Ok(json_str) = line {
            if let Ok(segment) = serde_json::from_str::<ExtractedSegment>(&json_str) {
                all_segments.push(segment);
            }
        }
    }

    println!("  Loaded {} segments", all_segments.len());

    // Compute normalization statistics
    let n_dims = 112;
    let n_segments = all_segments.len();

    println!("  Computing statistics for {} dimensions...", n_dims);

    let mut means = vec![0.0f32; n_dims];
    let mut stds = vec![0.0f32; n_dims];

    for dim in 0..n_dims {
        let values: Vec<f32> = all_segments.iter()
            .map(|s| s.features_112d[dim])
            .collect();

        let mean = values.iter().sum::<f32>() / n_segments.max(1) as f32;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>() / n_segments.max(1) as f32;
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

    // Write final output
    println!("  Writing final output...");
    let mut file = File::create(&output_path)?;
    write!(file, "{{\n")?;
    write!(file, "  \"total_files\": {},\n", total_files)?;
    write!(file, "  \"total_segments\": {},\n", total_segments)?;
    write!(file, "  \"feature_dimension\": {},\n", n_dims)?;
    write!(file, "  \"segments\": [\n")?;

    for (i, segment) in all_segments.iter().enumerate() {
        let json = serde_json::to_string(segment)?;
        write!(file, "    {}", json)?;
        if i < all_segments.len() - 1 {
            write!(file, ",")?;
        }
        write!(file, "\n")?;

        if (i + 1) % 100000 == 0 {
            println!("  Written {}/{} segments...", i + 1, all_segments.len());
            file.flush()?;
        }
    }

    write!(file, "  ]\n")?;
    write!(file, "}}\n")?;
    file.flush()?;

    println!("  ✅ Results exported to: {:?}", output_path);
    println!();

    // Export normalization parameters
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

    // Clean up checkpoint (only after successful export)
    let checkpoint_path = output_dir.join("extraction_checkpoint.json");
    if checkpoint_path.exists() {
        fs::remove_file(checkpoint_path)?;
        println!("  🧹 Cleaned up checkpoint file");
    }

    // Keep partial segments file as backup
    let backup_path = output_dir.join("extraction_segments_backup.jsonl");
    if segments_path.exists() {
        fs::copy(&segments_path, &backup_path)?;
        println!("  💾 Backed up partial segments file to: {:?}", backup_path);
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Extraction Complete!                                                ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}
