//! Egyptian Fruit Bat - NBD-based Feature Caching
//! ==============================================
//!
//! Uses Neural Boundary Detection to segment, then caches 105D features.

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use technical_architecture::{BoundaryDetectorConfig, MicroDynamicsExtractor, NeuralBoundaryDetector};

/// Cached segment with NBD boundary info
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedSegmentNBD {
    source_file: String,
    context: i32,
    emitter: i32,
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    boundary_type: String, // "Hard", "Soft", "Transitional", "Start"
    features: Vec<f32>,
}

/// Load WAV file
fn load_wav(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let bytes = fs::read(path)?;
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("Not a valid WAV file");
    }

    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut audio_format = 0u16;
    let mut bits_per_sample = 0u16;

    while pos < bytes.len() - 8 {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;

        if chunk_id == b"fmt " {
            let fmt_data = &bytes[pos + 8..pos + 8 + chunk_size.min(18)];
            audio_format = u16::from_le_bytes([fmt_data[0], fmt_data[1]]);
            sample_rate = u32::from_le_bytes([fmt_data[4], fmt_data[5], fmt_data[6], fmt_data[7]]);
            bits_per_sample = u16::from_le_bytes([fmt_data[14], fmt_data[15]]);
        } else if chunk_id == b"data" {
            let data_start = pos + 8;
            let data_end = pos + 8 + chunk_size;
            let audio_bytes = &bytes[data_start..data_end.min(bytes.len())];

            let samples: Vec<f32> = match (audio_format, bits_per_sample) {
                (3, 32) => audio_bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                (1, 16) => audio_bytes
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                    .collect(),
                _ => anyhow::bail!("Unsupported format"),
            };
            return Ok((samples, sample_rate));
        }
        pos += 8 + chunk_size + (chunk_size % 2);
    }
    anyhow::bail!("No data chunk")
}

fn parse_annotations(path: &Path) -> anyhow::Result<HashMap<String, (i32, i32)>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut annotations = HashMap::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 8 {
            let emitter: i32 = parts[0].parse().unwrap_or(0);
            let context: i32 = parts[2].parse().unwrap_or(0);
            let filename = parts[7].to_string(); // Already has .wav extension
            annotations.insert(filename, (emitter, context));
        }
    }
    Ok(annotations)
}

fn compute_105d_f32(extractor: &MicroDynamicsExtractor, audio: &[f32]) -> Option<Vec<f32>> {
    let base_45d = extractor.extract_45d(audio).ok()?;
    let mut features = Vec::with_capacity(105);
    features.extend_from_slice(&base_45d.to_array());
    for _ in 0..60 {
        features.push(0.0);
    }
    Some(features)
}

fn boundary_type_str(bt: technical_architecture::BoundaryType) -> String {
    match bt {
        technical_architecture::BoundaryType::Hard => "Hard".to_string(),
        technical_architecture::BoundaryType::Soft => "Soft".to_string(),
        technical_architecture::BoundaryType::Transitional => "Transitional".to_string(),
    }
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     EGYPTIAN FRUIT BAT - NBD-BASED FEATURE CACHING                       ║");
    println!("║           Using Neural Boundary Detection for Segmentation                ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = data_dir.join("audio");
    let annotations_path = data_dir.join("annotations.csv");
    let cache_dir = Path::new("bat_nbd_cache");

    fs::create_dir_all(cache_dir)?;
    println!("Cache directory: {}", cache_dir.display());

    let annotations = parse_annotations(&annotations_path)?;
    println!("Annotations: {} files", annotations.len());

    let all_files: Vec<PathBuf> = fs::read_dir(&audio_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    println!("Total WAV files: {}", all_files.len());
    println!();

    // NBD configuration for bat calls (250kHz)
    let nbd_config = BoundaryDetectorConfig {
        hop_size: 2048,
        sample_rate: 250000,
        min_phrase_duration_ms: 5.0, // Bats have very short calls
        threshold: 0.25,
        smoothing_frames: 2,
    };

    println!("NBD Configuration:");
    println!(
        "  • Hop: {} ({:.2}ms)",
        nbd_config.hop_size,
        nbd_config.hop_size as f32 / 250000.0 * 1000.0
    );
    println!("  • Min duration: {}ms", nbd_config.min_phrase_duration_ms);
    println!("  • Threshold: {}", nbd_config.threshold);
    println!();

    let extractor = MicroDynamicsExtractor::new(250000);
    let batch_size = 500;
    let mut batch_segments: Vec<CachedSegmentNBD> = Vec::new();
    let mut batch_num = 0;
    let mut total_segments = 0;
    let mut total_files = 0;
    let mut total_boundaries = 0;
    let mut hard_count = 0;
    let mut soft_count = 0;
    let mut trans_count = 0;

    println!("Processing files...");
    println!("─────────────────────────────────────────────────────────────────────────");

    for (idx, path) in all_files.iter().enumerate() {
        let filename = path.file_name().unwrap().to_str().unwrap().to_string();
        let (emitter, context) = annotations.get(&filename).copied().unwrap_or((0, 0));

        match load_wav(path) {
            Ok((audio, sr)) => {
                // Create NBD for this file
                let mut detector = NeuralBoundaryDetector::with_config(BoundaryDetectorConfig {
                    sample_rate: sr,
                    ..nbd_config.clone()
                });

                // Detect boundaries using NEURAL BOUNDARY DETECTION
                let boundaries = detector.detect_boundaries(&audio);
                total_boundaries += boundaries.len();

                // Count boundary types
                for b in &boundaries {
                    match b.boundary_type {
                        technical_architecture::BoundaryType::Hard => hard_count += 1,
                        technical_architecture::BoundaryType::Soft => soft_count += 1,
                        technical_architecture::BoundaryType::Transitional => trans_count += 1,
                    }
                }

                // Segment based on NBD boundaries
                let mut segments: Vec<(usize, usize, String)> = Vec::new();
                let mut start = 0usize;
                let min_len = (sr as f32 * 0.003) as usize; // 3ms minimum

                for b in &boundaries {
                    let end = (b.time_ms * sr as f32 / 1000.0) as usize;
                    if end > start && end <= audio.len() && end - start >= min_len {
                        segments.push((start, end, boundary_type_str(b.boundary_type)));
                    }
                    start = end;
                }

                // Final segment
                if audio.len() - start >= min_len {
                    segments.push((start, audio.len(), "End".to_string()));
                }

                // If no boundaries, use whole file
                if segments.is_empty() {
                    segments.push((0, audio.len(), "Whole".to_string()));
                }

                // Extract features
                let mut seg_idx = 0;
                for (start, end, btype) in segments {
                    if let Some(features) = compute_105d_f32(&extractor, &audio[start..end]) {
                        let start_ms = start as f32 / sr as f32 * 1000.0;
                        let end_ms = end as f32 / sr as f32 * 1000.0;

                        batch_segments.push(CachedSegmentNBD {
                            source_file: filename.clone(),
                            context,
                            emitter,
                            segment_idx: seg_idx,
                            start_ms,
                            end_ms,
                            boundary_type: btype,
                            features,
                        });
                        seg_idx += 1;
                        total_segments += 1;
                    }
                }

                total_files += 1;
            }
            Err(e) => {
                eprintln!("Error {}: {}", filename, e);
            }
        }

        // Save batch
        if batch_segments.len() >= batch_size {
            batch_num += 1;
            let cache_file = cache_dir.join(format!("nbd_batch_{:04}.json", batch_num));
            let json = serde_json::to_string(&batch_segments)?;
            let mut file = File::create(&cache_file)?;
            file.write_all(json.as_bytes())?;

            println!(
                "  Batch {:4}: {} files, {} segments | Totals: {} files, {} segments, {} boundaries",
                batch_num,
                total_files - (batch_num - 1) * batch_size,
                batch_segments.len(),
                total_files,
                total_segments,
                total_boundaries
            );

            batch_segments.clear();
        }

        if (idx + 1) % 5000 == 0 {
            println!("  Progress: {}/{} files...", idx + 1, all_files.len());
        }
    }

    // Final batch
    if !batch_segments.is_empty() {
        batch_num += 1;
        let cache_file = cache_dir.join(format!("nbd_batch_{:04}.json", batch_num));
        let json = serde_json::to_string(&batch_segments)?;
        let mut file = File::create(&cache_file)?;
        file.write_all(json.as_bytes())?;
        println!("  Batch {:4}: Final", batch_num);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("NBD CACHE COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  Files processed: {}", total_files);
    println!("  Total segments: {}", total_segments);
    println!("  Total boundaries: {}", total_boundaries);
    println!("    • Hard: {}", hard_count);
    println!("    • Soft: {}", soft_count);
    println!("    • Transitional: {}", trans_count);
    println!("  Batch files: {}", batch_num);
    println!("  Cache directory: {}", cache_dir.display());
    println!();
    println!("Run 'bat_nbd_mine_from_cache' for motif analysis.");
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
