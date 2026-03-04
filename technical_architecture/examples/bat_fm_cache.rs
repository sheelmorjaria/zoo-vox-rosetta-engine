//! Egyptian Fruit Bat - FM Shape Cache Generation (Parallel)
//! ==========================================================
//!
//! Caches 105D features PLUS f0_start and f0_end for FM sweep analysis.
//! Uses Rayon for parallel file processing.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use technical_architecture::{
    BoundaryDetectorConfig, MicroDynamicsExtractor, NeuralBoundaryDetector,
};

/// Cached segment with FM sweep info
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedSegmentFM {
    source_file: String,
    context: i32,
    emitter: i32,
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    duration_ms: f32,
    boundary_type: String,
    f0_start: f32,
    f0_end: f32,
    f0_mean: f32,
    sweep_rate: f32,
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
        let chunk_size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;

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
            let filename = parts[7].to_string();
            annotations.insert(filename, (emitter, context));
        }
    }
    Ok(annotations)
}

/// Extract features plus FM sweep parameters
fn extract_features_with_fm(
    extractor: &MicroDynamicsExtractor,
    audio: &[f32],
    sr: u32,
) -> Option<(Vec<f32>, f32, f32, f32, f32)> {
    let base_45d = extractor.extract_45d(audio).ok()?;
    let mut features = Vec::with_capacity(105);
    features.extend_from_slice(&base_45d.to_array());
    for _ in 0..60 {
        features.push(0.0);
    }

    let window_samples = ((sr as f32 * 0.005) as usize)
        .max(audio.len() / 10)
        .min(audio.len() / 2);
    let start_window = &audio[..window_samples.min(audio.len())];
    let (f0_start, _, conf_start) = extractor.estimate_f0(start_window);

    let end_start = audio.len().saturating_sub(window_samples);
    let end_window = &audio[end_start..];
    let (f0_end, _, conf_end) = extractor.estimate_f0(end_window);

    let (f0_mean, _, _) = extractor.estimate_f0(audio);

    let f0_start = if conf_start > 0.2 { f0_start } else { 0.0 };
    let f0_end = if conf_end > 0.2 { f0_end } else { 0.0 };

    let duration_ms = audio.len() as f32 / sr as f32 * 1000.0;

    let sweep_rate = if duration_ms > 0.0 && f0_start > 0.0 && f0_end > 0.0 {
        (f0_end - f0_start) / duration_ms
    } else {
        0.0
    };

    Some((features, f0_start, f0_end, f0_mean, sweep_rate))
}

fn boundary_type_str(bt: technical_architecture::NeuralBoundaryType) -> String {
    match bt {
        technical_architecture::NeuralBoundaryType::Hard => "Hard".to_string(),
        technical_architecture::NeuralBoundaryType::Soft => "Soft".to_string(),
        technical_architecture::NeuralBoundaryType::Transitional => "Transitional".to_string(),
    }
}

/// Process a single file and return segments
fn process_file(
    path: &Path,
    annotations: &HashMap<String, (i32, i32)>,
    nbd_config: &BoundaryDetectorConfig,
) -> Vec<CachedSegmentFM> {
    let filename = path.file_name().unwrap().to_str().unwrap().to_string();
    let (emitter, context) = annotations.get(&filename).copied().unwrap_or((0, 0));

    let audio = match load_wav(path) {
        Ok((a, _)) => a,
        Err(_) => return Vec::new(),
    };
    let sr = nbd_config.sample_rate;

    let extractor = MicroDynamicsExtractor::new(sr);
    let mut detector = NeuralBoundaryDetector::with_config(nbd_config.clone());
    let boundaries = detector.detect_boundaries(&audio);

    // Segment based on NBD boundaries
    let mut segments: Vec<(usize, usize, String)> = Vec::new();
    let mut start = 0usize;
    let min_len = (sr as f32 * 0.003) as usize;

    for b in &boundaries {
        let end = (b.time_ms * sr as f32 / 1000.0) as usize;
        if end > start && end <= audio.len() && end - start >= min_len {
            segments.push((start, end, boundary_type_str(b.boundary_type)));
        }
        start = end;
    }

    if audio.len() - start >= min_len {
        segments.push((start, audio.len(), "End".to_string()));
    }

    if segments.is_empty() {
        segments.push((0, audio.len(), "Whole".to_string()));
    }

    // Extract features
    let mut result = Vec::new();
    for (seg_idx, (start, end, btype)) in segments.into_iter().enumerate() {
        if let Some((features, f0_start, f0_end, f0_mean, sweep_rate)) =
            extract_features_with_fm(&extractor, &audio[start..end], sr)
        {
            let duration_ms = (end - start) as f32 / sr as f32 * 1000.0;
            let start_ms = start as f32 / sr as f32 * 1000.0;
            let end_ms = end as f32 / sr as f32 * 1000.0;

            result.push(CachedSegmentFM {
                source_file: filename.clone(),
                context,
                emitter,
                segment_idx: seg_idx,
                start_ms,
                end_ms,
                duration_ms,
                boundary_type: btype,
                f0_start,
                f0_end,
                f0_mean,
                sweep_rate,
                features,
            });
        }
    }

    result
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     EGYPTIAN FRUIT BAT - FM SHAPE CACHE GENERATION (PARALLEL)            ║");
    println!("║     Extracting F0 trajectories for Sweep Shape Mining                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = data_dir.join("audio");
    let annotations_path = data_dir.join("annotations.csv");
    let cache_dir = Path::new("bat_fm_cache");

    fs::create_dir_all(cache_dir)?;
    println!("Cache directory: {}", cache_dir.display());

    let annotations = Arc::new(parse_annotations(&annotations_path)?);
    println!("Annotations: {} files", annotations.len());

    let all_files: Vec<PathBuf> = fs::read_dir(&audio_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    let total_files = all_files.len();
    println!("Total WAV files: {}", total_files);
    println!();

    let nbd_config = BoundaryDetectorConfig {
        hop_size: 2048,
        sample_rate: 250000,
        min_phrase_duration_ms: 5.0,
        threshold: 0.25,
        smoothing_frames: 2,
    };
    let nbd_config = Arc::new(nbd_config);

    println!("Processing {} files in parallel...", total_files);
    println!("─────────────────────────────────────────────────────────────────────────");

    // Process files in parallel
    let progress = Arc::new(Mutex::new(0usize));
    let all_segments: Vec<CachedSegmentFM> = all_files
        .par_iter()
        .flat_map(|path| {
            // Update progress
            if let Ok(mut p) = progress.lock() {
                *p += 1;
                if *p % 1000 == 0 {
                    println!("  Progress: {}/{} files...", *p, total_files);
                }
            }

            process_file(path, &annotations, &nbd_config)
        })
        .collect();

    let total_segments = all_segments.len();
    println!();
    println!(
        "  Processed {} files, {} segments",
        total_files, total_segments
    );

    // Count sweep types
    let up_sweeps = all_segments.iter().filter(|s| s.sweep_rate > 100.0).count();
    let down_sweeps = all_segments
        .iter()
        .filter(|s| s.sweep_rate < -100.0)
        .count();
    let flat_sweeps = all_segments
        .iter()
        .filter(|s| s.sweep_rate.abs() <= 100.0 && s.f0_start > 0.0)
        .count();
    let no_pitch = all_segments
        .iter()
        .filter(|s| s.f0_start <= 0.0 || s.f0_end <= 0.0)
        .count();

    // Save to batches
    println!();
    println!("Saving to cache batches...");

    let batch_size = 500;
    let mut batch_num = 0;

    for chunk in all_segments.chunks(batch_size) {
        batch_num += 1;
        let cache_file = cache_dir.join(format!("fm_batch_{:04}.json", batch_num));
        let json = serde_json::to_string(chunk)?;
        let mut file = File::create(&cache_file)?;
        file.write_all(json.as_bytes())?;
    }

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("FM SHAPE CACHE COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  Files processed: {}", total_files);
    println!("  Total segments: {}", total_segments);
    println!("  Batch files: {}", batch_num);
    println!();
    println!("  FM Sweep Classification:");
    println!(
        "    • Up-sweeps   (>+100 Hz/ms): {:7} ({:.1}%)",
        up_sweeps,
        up_sweeps as f64 / total_segments as f64 * 100.0
    );
    println!(
        "    • Down-sweeps (<-100 Hz/ms): {:7} ({:.1}%)",
        down_sweeps,
        down_sweeps as f64 / total_segments as f64 * 100.0
    );
    println!(
        "    • Flat        (±100 Hz/ms):  {:7} ({:.1}%)",
        flat_sweeps,
        flat_sweeps as f64 / total_segments as f64 * 100.0
    );
    println!(
        "    • No pitch    (unvoiced):    {:7} ({:.1}%)",
        no_pitch,
        no_pitch as f64 / total_segments as f64 * 100.0
    );
    println!();
    println!("  Cache directory: {}", cache_dir.display());
    println!();
    println!("Run 'bat_fm_shape_mining' for sweep motif analysis.");
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
