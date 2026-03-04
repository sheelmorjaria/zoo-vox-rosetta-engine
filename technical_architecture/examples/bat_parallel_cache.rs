//! Egyptian Fruit Bat - PARALLEL NBD Feature Caching
//! ===================================================
//!
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedSegmentNBD {
    source_file: String,
    context: i32,
    emitter: i32,
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    boundary_type: String,
    features: Vec<f32>,
}

#[derive(Debug, Clone)]
struct FileTask {
    path: PathBuf,
    filename: String,
    emitter: i32,
    context: i32,
}

fn load_wav(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let bytes = fs::read(path)?;
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("Not WAV");
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
            let fmt = &bytes[pos + 8..pos + 8 + chunk_size.min(18)];
            audio_format = u16::from_le_bytes([fmt[0], fmt[1]]);
            sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
            bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
        } else if chunk_id == b"data" {
            let start = pos + 8;
            let end = pos + 8 + chunk_size;
            let audio = &bytes[start..end.min(bytes.len())];

            let samples: Vec<f32> = match (audio_format, bits_per_sample) {
                (3, 32) => audio
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                (1, 16) => audio
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                    .collect(),
                _ => anyhow::bail!("Bad format"),
            };
            return Ok((samples, sample_rate));
        }
        pos += 8 + chunk_size + (chunk_size % 2);
    }
    anyhow::bail!("No data")
}

fn parse_annotations(path: &Path) -> anyhow::Result<HashMap<String, (i32, i32)>> {
    let file = File::open(path)?;
    let mut ann = HashMap::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        if i == 0 {
            continue;
        }
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 8 {
            let emitter: i32 = parts[0].parse().unwrap_or(0);
            let context: i32 = parts[2].parse().unwrap_or(0);
            let filename = parts[7].to_string(); // Already has .wav extension
            ann.insert(filename, (emitter, context));
        }
    }
    Ok(ann)
}

fn compute_105d(extractor: &MicroDynamicsExtractor, audio: &[f32]) -> Option<Vec<f32>> {
    let base = extractor.extract_45d(audio).ok()?;
    let mut f = Vec::with_capacity(105);
    f.extend_from_slice(&base.to_array());
    f.extend(std::iter::repeat(0.0).take(60));
    Some(f)
}

fn process_file(
    task: &FileTask,
    nbd_config: &BoundaryDetectorConfig,
    extractor: &MicroDynamicsExtractor,
) -> Vec<CachedSegmentNBD> {
    let mut segments = Vec::new();

    if let Ok((audio, sr)) = load_wav(&task.path) {
        let mut detector = NeuralBoundaryDetector::with_config(BoundaryDetectorConfig {
            sample_rate: sr,
            ..nbd_config.clone()
        });

        let boundaries = detector.detect_boundaries(&audio);

        let mut segs: Vec<(usize, usize, String)> = Vec::new();
        let mut start = 0usize;
        let min_len = (sr as f32 * 0.003) as usize;

        for b in &boundaries {
            let end = (b.time_ms * sr as f32 / 1000.0) as usize;
            if end > start && end <= audio.len() && end - start >= min_len {
                let btype = match b.boundary_type {
                    technical_architecture::NeuralBoundaryType::Hard => "Hard",
                    technical_architecture::NeuralBoundaryType::Soft => "Soft",
                    technical_architecture::NeuralBoundaryType::Transitional => "Transitional",
                };
                segs.push((start, end, btype.to_string()));
            }
            start = end;
        }

        if audio.len() - start >= min_len {
            segs.push((start, audio.len(), "End".to_string()));
        }

        if segs.is_empty() {
            segs.push((0, audio.len(), "Whole".to_string()));
        }

        for (si, (start, end, btype)) in segs.into_iter().enumerate() {
            if let Some(features) = compute_105d(extractor, &audio[start..end]) {
                let start_ms = start as f32 / sr as f32 * 1000.0;
                let end_ms = end as f32 / sr as f32 * 1000.0;
                segments.push(CachedSegmentNBD {
                    source_file: task.filename.clone(),
                    context: task.context,
                    emitter: task.emitter,
                    segment_idx: si,
                    start_ms,
                    end_ms,
                    boundary_type: btype,
                    features,
                });
            }
        }
    }
    segments
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     EGYPTIAN FRUIT BAT - PARALLEL NBD FEATURE CACHING                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = data_dir.join("audio");
    let ann_path = data_dir.join("annotations.csv");
    let cache_dir = Path::new("bat_nbd_cache_parallel");

    fs::create_dir_all(cache_dir)?;
    println!("Cache: {}", cache_dir.display());

    let annotations = parse_annotations(&ann_path)?;
    println!("Annotations: {}", annotations.len());

    // Build task list
    let tasks: Vec<FileTask> = fs::read_dir(&audio_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
        .map(|e| {
            let path = e.path();
            let filename = path.file_name().unwrap().to_str().unwrap().to_string();
            let (emitter, context) = annotations.get(&filename).copied().unwrap_or((0, 0));
            FileTask {
                path,
                filename,
                emitter,
                context,
            }
        })
        .collect();

    println!("Files: {}", tasks.len());
    println!();

    let nbd_config = BoundaryDetectorConfig {
        hop_size: 2048,
        sample_rate: 250000,
        min_phrase_duration_ms: 5.0,
        threshold: 0.25,
        smoothing_frames: 2,
    };

    let extractor = Arc::new(MicroDynamicsExtractor::new(250000));
    let nbd_config = Arc::new(nbd_config);

    // Progress tracking
    let processed = Arc::new(Mutex::new(0usize));
    let total_segs = Arc::new(Mutex::new(0usize));
    let batch_num = Arc::new(Mutex::new(0usize));
    let batch_buffer = Arc::new(Mutex::new(Vec::<CachedSegmentNBD>::new()));

    println!(
        "Processing with {} threads...",
        rayon::current_num_threads()
    );
    println!("─────────────────────────────────────────────────────────────────────────");

    let batch_size = 500;

    // Process in parallel chunks
    tasks
        .par_chunks(100) // Process 100 files at a time
        .for_each(|chunk| {
            let local_segs: Vec<CachedSegmentNBD> = chunk
                .iter()
                .flat_map(|task| process_file(task, &nbd_config, &extractor))
                .collect();

            // Update progress
            {
                let mut p = processed.lock().unwrap();
                *p += chunk.len();
                if *p % 5000 == 0 {
                    println!("  Progress: {} files processed...", *p);
                }
            }

            // Add to batch buffer
            {
                let mut buf = batch_buffer.lock().unwrap();
                let mut bn = batch_num.lock().unwrap();
                let mut ts = total_segs.lock().unwrap();

                buf.extend(local_segs);
                *ts += buf.len();

                if buf.len() >= batch_size {
                    *bn += 1;
                    let cache_file = cache_dir.join(format!("nbd_batch_{:04}.json", *bn));
                    if let Ok(json) = serde_json::to_string(&*buf) {
                        if let Ok(mut f) = File::create(&cache_file) {
                            let _ = f.write_all(json.as_bytes());
                        }
                    }
                    println!(
                        "  Batch {:4}: {} segments cached ({} total)",
                        *bn,
                        buf.len(),
                        *ts
                    );
                    buf.clear();
                }
            }
        });

    // Save final batch
    {
        let mut buf = batch_buffer.lock().unwrap();
        if !buf.is_empty() {
            let mut bn = batch_num.lock().unwrap();
            *bn += 1;
            let cache_file = cache_dir.join(format!("nbd_batch_{:04}.json", *bn));
            if let Ok(json) = serde_json::to_string(&*buf) {
                if let Ok(mut f) = File::create(&cache_file) {
                    let _ = f.write_all(json.as_bytes());
                }
            }
            println!("  Batch {:4}: Final batch", *bn);
        }
    }

    let total = *total_segs.lock().unwrap();
    let batches = *batch_num.lock().unwrap();

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("PARALLEL CACHE COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  Total segments: {}", total);
    println!("  Batch files: {}", batches);
    println!("  Cache: {}", cache_dir.display());
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
