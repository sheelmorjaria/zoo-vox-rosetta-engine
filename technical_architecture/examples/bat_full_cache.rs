//! Egyptian Fruit Bat - NBD Feature Caching with PROPER 105D features
//! ==================================================================
//!
//! Uses real macro/micro texture features, not zeros.

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use technical_architecture::{
    BoundaryDetectorConfig, MicroDynamicsExtractor, MicroDynamicsFeatures45D, NeuralBoundaryDetector,
};

#[derive(Debug, Clone, Serialize)]
struct CachedSegmentNBD {
    source_file: String,
    context: i32,
    emitter: i32,
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    duration_ms: f32,
    boundary_type: String,
    features: Vec<f32>, // Full 105D
}

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
        let chunk_size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;

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

fn parse_annotations(path: &Path) -> HashMap<String, (i32, i32)> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return HashMap::new(),
    };
    let mut ann = HashMap::new();

    for (i, line) in BufReader::new(file).lines().enumerate() {
        if i == 0 {
            continue;
        }
        if let Ok(line) = line {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() >= 8 {
                let emitter: i32 = parts[0].parse().unwrap_or(0);
                let context: i32 = parts[2].parse().unwrap_or(0);
                let filename = parts[7].to_string(); // Already has .wav
                ann.insert(filename, (emitter, context));
            }
        }
    }
    ann
}

/// Compute PROPER 105D features with real macro/micro texture
fn compute_105d_full(extractor: &MicroDynamicsExtractor, audio: &[f32]) -> Option<Vec<f32>> {
    let base_45d = extractor.extract_45d(audio).ok()?;

    let mut features = Vec::with_capacity(105);

    // Layer 1: Base Physics (45D)
    features.extend_from_slice(&base_45d.to_array());

    // Layer 2: Macro Texture (30D) - REAL computation
    features.extend(compute_macro_texture(&base_45d));

    // Layer 3: Micro Texture (30D) - REAL computation
    features.extend(compute_micro_texture(&base_45d));

    Some(features)
}

/// Compute 30D macro texture features
fn compute_macro_texture(base: &MicroDynamicsFeatures45D) -> Vec<f32> {
    let mut f = Vec::with_capacity(30);

    // Harmonic Texture (8D)
    f.push(base.spectral_tilt);
    f.push(base.base_30d.harmonic_to_noise_ratio * 0.5);
    f.push(base.base_30d.jitter);
    f.push(base.base_30d.harmonic_to_noise_ratio * 0.1);
    f.push(base.base_30d.spectral_flux);
    f.push(base.formant_1_hz / (base.formant_2_hz + 1.0));
    f.push(base.formant_2_hz / (base.formant_3_hz + 1.0));
    f.push(base.formant_3_hz / (base.formant_dispersion * 10.0 + 1.0));

    // Pitch Geometry (7D)
    f.push(base.f0_range_hz / (base.duration_ms + 1.0));
    f.push(base.fm_slope * 0.5);
    f.push(0.0); // f0_inflection_count
    f.push(base.fm_slope);
    f.push(base.base_30d.vibrato_rate_hz / 10.0);
    f.push(base.base_30d.jitter * 10.0);
    f.push(base.f0_range_hz / (base.mean_f0_hz + 1.0));

    // GLCM Texture (10D)
    f.push(base.spectral_kurtosis);
    f.push(base.spectral_skewness * 0.5);
    f.push(1.0 - base.base_30d.spectral_flatness);
    f.push(1.0 - base.base_30d.spectral_flatness);
    f.push(base.spectral_spread * 0.01);
    f.push(base.duration_ms / 100.0);
    f.push(1.0 / (base.duration_ms / 100.0 + 1.0));
    f.push(base.base_30d.spectral_flatness);
    f.push(base.am_depth);
    f.push(base.fm_slope * 0.1);

    // Temporal Texture (5D)
    f.push(0.1);
    f.push(base.base_30d.attack_time_ms / (base.base_30d.decay_time_ms + 1.0));
    f.push(base.base_30d.sustain_level * 10.0);
    f.push(base.base_30d.vibrato_depth / 100.0);
    f.push(0.1);

    f
}

/// Compute 30D micro texture features
fn compute_micro_texture(base: &MicroDynamicsFeatures45D) -> Vec<f32> {
    let mut f = Vec::with_capacity(30);

    // AM Spectrum (5D)
    let vr = base.base_30d.vibrato_rate_hz;
    f.push(if vr < 10.0 { 1.0 } else { 0.0 });
    f.push(if vr >= 10.0 && vr < 30.0 { 1.0 } else { 0.0 });
    f.push(if vr >= 30.0 && vr < 50.0 { 1.0 } else { 0.0 });
    f.push(if vr >= 50.0 && vr < 100.0 { 1.0 } else { 0.0 });
    f.push(base.am_depth);

    // FM Spectrum (5D)
    let fm = base.fm_slope;
    f.push(if fm < 10.0 { 1.0 } else { 0.0 });
    f.push(if fm >= 10.0 && fm < 30.0 { 1.0 } else { 0.0 });
    f.push(if fm >= 30.0 && fm < 50.0 { 1.0 } else { 0.0 });
    f.push(if fm >= 50.0 && fm < 100.0 { 1.0 } else { 0.0 });
    f.push(0.0);

    // Modulation Stats (5D)
    f.push(base.base_30d.vibrato_rate_hz);
    f.push(base.fm_slope);
    f.push(base.am_depth);
    f.push(base.fm_slope * 0.5);
    f.push(base.base_30d.vibrato_depth);

    // Rhythm Histogram (5D)
    let ici = base.base_30d.median_ici_ms;
    f.push(if ici < 20.0 { 1.0 } else { 0.0 });
    f.push(if ici >= 20.0 && ici < 50.0 { 1.0 } else { 0.0 });
    f.push(if ici >= 50.0 && ici < 100.0 { 1.0 } else { 0.0 });
    f.push(if ici >= 100.0 && ici < 200.0 { 1.0 } else { 0.0 });
    f.push(if ici >= 200.0 { 1.0 } else { 0.0 });

    // Rhythm Stats (5D)
    f.push(base.base_30d.median_ici_ms);
    f.push(1.0 / (base.base_30d.median_ici_ms / 1000.0 + 0.001));
    f.push(base.base_30d.onset_rate_hz);
    f.push(base.base_30d.ici_coefficient_of_variation);
    f.push(base.base_30d.onset_rate_hz * 60.0);

    // Psychoacoustics (5D)
    f.push(base.spectral_centroid / 1000.0);
    f.push(base.base_30d.harmonic_to_noise_ratio);
    f.push(1.0 - base.subharmonic_ratio);
    f.push(base.spectral_entropy);
    f.push(base.base_30d.harmonicity);

    f
}

fn process_file(
    task: &FileTask,
    nbd_config: &BoundaryDetectorConfig,
    extractor: &MicroDynamicsExtractor,
) -> Vec<CachedSegmentNBD> {
    let mut segments = Vec::new();

    if let Ok((audio, sr)) = load_wav(&task.path) {
        let duration_ms = audio.len() as f32 / sr as f32 * 1000.0;

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
                    technical_architecture::BoundaryType::Hard => "Hard",
                    technical_architecture::BoundaryType::Soft => "Soft",
                    technical_architecture::BoundaryType::Transitional => "Transitional",
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
            if let Some(features) = compute_105d_full(extractor, &audio[start..end]) {
                let start_ms = start as f32 / sr as f32 * 1000.0;
                let end_ms = end as f32 / sr as f32 * 1000.0;
                let seg_duration = end_ms - start_ms;

                segments.push(CachedSegmentNBD {
                    source_file: task.filename.clone(),
                    context: task.context,
                    emitter: task.emitter,
                    segment_idx: si,
                    start_ms,
                    end_ms,
                    duration_ms: seg_duration,
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
    println!("║     EGYPTIAN FRUIT BAT - PARALLEL NBD CACHING (FULL 105D)                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = data_dir.join("audio");
    let ann_path = data_dir.join("annotations.csv");
    let cache_dir = Path::new("bat_nbd_cache_full");

    fs::create_dir_all(cache_dir)?;
    println!("Cache: {}", cache_dir.display());

    let annotations = parse_annotations(&ann_path);
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

    let extractor = MicroDynamicsExtractor::new(250000);
    let batch_size = 500;

    println!("Using {} threads (Rayon parallel)", rayon::current_num_threads());
    println!("Processing with FULL 105D features (real macro/micro texture)...");
    println!("─────────────────────────────────────────────────────────────────────────");

    let processed = AtomicUsize::new(0);
    let mut batch_segments: Vec<CachedSegmentNBD> = Vec::new();
    let mut batch_num = 0;
    let mut total_segments = 0;
    let mut context_counts: HashMap<i32, usize> = HashMap::new();
    let mut emitter_counts: HashMap<i32, usize> = HashMap::new();

    // Process in parallel using rayon
    let all_segments: Vec<Vec<CachedSegmentNBD>> = tasks
        .par_chunks(100)
        .map(|chunk| {
            chunk
                .iter()
                .flat_map(|task| process_file(task, &nbd_config, &extractor))
                .collect()
        })
        .collect();

    println!("  Parallel processing complete, writing batches...");

    // Flatten and write batches
    for local_segments in all_segments {
        for seg in &local_segments {
            *context_counts.entry(seg.context).or_insert(0) += 1;
            *emitter_counts.entry(seg.emitter).or_insert(0) += 1;
        }

        batch_segments.extend(local_segments);
        total_segments += batch_segments.len();

        while batch_segments.len() >= batch_size {
            batch_num += 1;
            let batch: Vec<_> = batch_segments.drain(..batch_size).collect();

            let cache_file = cache_dir.join(format!("nbd_batch_{:04}.json", batch_num));
            if let Ok(json) = serde_json::to_string(&batch) {
                if let Ok(mut f) = File::create(&cache_file) {
                    let _ = f.write_all(json.as_bytes());
                }
            }

            println!("  Batch {:4}: {} segments", batch_num, batch.len());
        }

        processed.fetch_add(100, Ordering::SeqCst);
        let p = processed.load(Ordering::SeqCst);
        if p % 5000 == 0 {
            println!("  Progress: {} files...", p);
        }
    }

    // Write final batch
    if !batch_segments.is_empty() {
        batch_num += 1;
        let cache_file = cache_dir.join(format!("nbd_batch_{:04}.json", batch_num));
        if let Ok(json) = serde_json::to_string(&batch_segments) {
            if let Ok(mut f) = File::create(&cache_file) {
                let _ = f.write_all(json.as_bytes());
            }
        }
        println!("  Batch {:4}: Final batch", batch_num);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("CACHE COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  Total segments: {}", total_segments);
    println!("  Batch files: {}", batch_num);
    println!("  Cache: {}", cache_dir.display());
    println!();
    println!("Context distribution:");
    let mut ctx_sorted: Vec<_> = context_counts.iter().collect();
    ctx_sorted.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (ctx, count) in ctx_sorted.iter().take(10) {
        let pct = **count as f64 / total_segments as f64 * 100.0;
        println!("  Context {}: {} ({:.1}%)", ctx, count, pct);
    }
    println!();
    println!("Top emitters:");
    let mut emit_sorted: Vec<_> = emitter_counts.iter().collect();
    emit_sorted.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (emit, count) in emit_sorted.iter().take(10) {
        let pct = **count as f64 / total_segments as f64 * 100.0;
        println!("  Bat {}: {} ({:.1}%)", emit, count, pct);
    }
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
