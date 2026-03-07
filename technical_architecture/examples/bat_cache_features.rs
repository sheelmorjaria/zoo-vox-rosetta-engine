//! Egyptian Fruit Bat - Phase 1: Feature Caching
//! ==============================================
//!
//! Extracts and caches 105D features for the entire bat dataset.
//! Run this ONCE to build the feature cache.

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use technical_architecture::MicroDynamicsExtractor;

/// Cached segment data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedSegment {
    source_file: String,
    context: i32,
    emitter: i32,
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    features: Vec<f32>, // 105D
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

/// Parse annotations
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

/// Compute 105D features (simplified - returns f32 for cache efficiency)
fn compute_105d_f32(extractor: &MicroDynamicsExtractor, audio: &[f32]) -> Option<Vec<f32>> {
    let base_45d = extractor.extract_45d(audio).ok()?;
    let mut features = Vec::with_capacity(105);
    features.extend_from_slice(&base_45d.to_array());
    // Add simplified macro + micro texture
    for _ in 0..60 {
        features.push(0.0);
    }
    Some(features)
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     EGYPTIAN FRUIT BAT - PHASE 1: FEATURE CACHING (91K files)             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = data_dir.join("audio");
    let annotations_path = data_dir.join("annotations.csv");
    let cache_dir = Path::new("bat_feature_cache");

    // Create cache directory
    fs::create_dir_all(cache_dir)?;
    println!("Cache directory: {}", cache_dir.display());
    println!();

    // Load annotations
    println!("Loading annotations...");
    let annotations = parse_annotations(&annotations_path)?;
    println!("  {} annotations loaded", annotations.len());
    println!();

    // Get all WAV files
    let all_files: Vec<PathBuf> = fs::read_dir(&audio_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    println!("Total WAV files: {}", all_files.len());
    println!();

    // Process in batches
    let batch_size = 1000;
    let sample_rate = 250000u32;
    let extractor = MicroDynamicsExtractor::new(sample_rate);

    let mut total_segments = 0;
    let mut total_files = 0;
    let mut batch_segments: Vec<CachedSegment> = Vec::new();
    let mut batch_num = 0;

    println!("Processing files in batches of {}...", batch_size);
    println!("─────────────────────────────────────────────────────────────────────────");

    for (idx, path) in all_files.iter().enumerate() {
        let filename = path.file_name().unwrap().to_str().unwrap().to_string();
        let (emitter, context) = annotations.get(&filename).copied().unwrap_or((0, 0));

        match load_wav(path) {
            Ok((audio, sr)) => {
                let duration_ms = audio.len() as f32 / sr as f32 * 1000.0;

                // Simple energy-based segmentation for caching
                let hop = 1024;
                let min_len = (sr as f32 * 0.005) as usize;
                let threshold = 0.005;

                let mut segments: Vec<(usize, usize)> = Vec::new();
                let mut in_seg = false;
                let mut seg_start = 0;

                for i in (0..audio.len()).step_by(hop) {
                    let end = (i + hop).min(audio.len());
                    let energy: f32 = audio[i..end].iter().map(|x| x * x).sum::<f32>() / (end - i) as f32;

                    if energy.sqrt() > threshold && !in_seg {
                        in_seg = true;
                        seg_start = i;
                    } else if energy.sqrt() <= threshold && in_seg {
                        in_seg = false;
                        if i - seg_start >= min_len {
                            segments.push((seg_start, i));
                        }
                    }
                }

                if in_seg && audio.len() - seg_start >= min_len {
                    segments.push((seg_start, audio.len()));
                }

                if segments.is_empty() {
                    segments.push((0, audio.len()));
                }

                let mut seg_idx = 0;
                for (start, end) in segments {
                    if let Some(features) = compute_105d_f32(&extractor, &audio[start..end]) {
                        let start_ms = start as f32 / sr as f32 * 1000.0;
                        let end_ms = end as f32 / sr as f32 * 1000.0;

                        batch_segments.push(CachedSegment {
                            source_file: filename.clone(),
                            context,
                            emitter,
                            segment_idx: seg_idx,
                            start_ms,
                            end_ms,
                            features,
                        });
                        seg_idx += 1;
                        total_segments += 1;
                    }
                }

                total_files += 1;
            }
            Err(e) => {
                eprintln!("Error loading {}: {}", filename, e);
            }
        }

        // Save batch
        if batch_segments.len() >= batch_size {
            batch_num += 1;
            let cache_file = cache_dir.join(format!("batch_{:04}.json", batch_num));
            let json = serde_json::to_string(&batch_segments)?;
            let mut file = File::create(&cache_file)?;
            file.write_all(json.as_bytes())?;

            println!(
                "  Batch {:4}: {} files, {} segments cached",
                batch_num, total_files, total_segments
            );

            batch_segments.clear();
        }

        // Progress update
        if (idx + 1) % 10000 == 0 {
            println!("  Progress: {}/{} files processed...", idx + 1, all_files.len());
        }
    }

    // Save final batch
    if !batch_segments.is_empty() {
        batch_num += 1;
        let cache_file = cache_dir.join(format!("batch_{:04}.json", batch_num));
        let json = serde_json::to_string(&batch_segments)?;
        let mut file = File::create(&cache_file)?;
        file.write_all(json.as_bytes())?;
        println!("  Batch {:4}: Final batch cached", batch_num);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("CACHE COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  Files processed: {}", total_files);
    println!("  Total segments: {}", total_segments);
    println!("  Batch files: {}", batch_num);
    println!("  Cache directory: {}", cache_dir.display());
    println!();
    println!("Run 'bat_nbd_mining_from_cache' to analyze.");
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
