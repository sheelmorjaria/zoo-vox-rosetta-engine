//! Egyptian Fruit Bat - PROPERLY PARALLEL NBD Feature Caching
//! ============================================================
//!
//! Uses channels for lock-free parallel processing.

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use crossbeam::channel::{bounded, Receiver, Sender};
use serde::Serialize;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
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
    boundary_type: String,
    features: Vec<f32>,
}

struct WorkItem {
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

    // Layer 1: Base Physics (45D)
    f.extend_from_slice(&base.to_array());

    // Layer 2: Macro Texture (30D)
    f.extend(compute_macro_texture(&base));

    // Layer 3: Micro Texture (30D)
    f.extend(compute_micro_texture(&base));

    Some(f)
}

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
    f.push(0.0);
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

// Worker thread function - processes files independently
fn worker(
    id: usize,
    work_rx: Receiver<WorkItem>,
    result_tx: Sender<Vec<CachedSegmentNBD>>,
    nbd_config: BoundaryDetectorConfig,
) {
    let extractor = MicroDynamicsExtractor::new(250000);

    while let Ok(work) = work_rx.recv() {
        let mut segments = Vec::new();

        if let Ok((audio, sr)) = load_wav(&work.path) {
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
                if let Some(features) = compute_105d(&extractor, &audio[start..end]) {
                    let start_ms = start as f32 / sr as f32 * 1000.0;
                    let end_ms = end as f32 / sr as f32 * 1000.0;
                    segments.push(CachedSegmentNBD {
                        source_file: work.filename.clone(),
                        context: work.context,
                        emitter: work.emitter,
                        segment_idx: si,
                        start_ms,
                        end_ms,
                        boundary_type: btype,
                        features,
                    });
                }
            }
        }

        if result_tx.send(segments).is_err() {
            break;
        }
    }

    println!("  Worker {} finished", id);
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     EGYPTIAN FRUIT BAT - CHANNEL-BASED PARALLEL NBD CACHING              ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = data_dir.join("audio");
    let ann_path = data_dir.join("annotations.csv");
    let cache_dir = Path::new("bat_nbd_cache_channel");

    fs::create_dir_all(cache_dir)?;
    println!("Cache: {}", cache_dir.display());

    let annotations = parse_annotations(&ann_path)?;
    println!("Annotations: {}", annotations.len());

    // Build work queue
    let work_items: Vec<WorkItem> = fs::read_dir(&audio_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
        .map(|e| {
            let path = e.path();
            let filename = path.file_name().unwrap().to_str().unwrap().to_string();
            let (emitter, context) = annotations.get(&filename).copied().unwrap_or((0, 0));
            WorkItem {
                path,
                filename,
                emitter,
                context,
            }
        })
        .collect();

    let total_files = work_items.len();
    println!("Files: {}", total_files);
    println!();

    let nbd_config = BoundaryDetectorConfig {
        hop_size: 2048,
        sample_rate: 250000,
        min_phrase_duration_ms: 5.0,
        threshold: 0.25,
        smoothing_frames: 2,
    };

    // Create channels
    let num_workers = rayon::current_num_threads().min(16);
    let (work_tx, work_rx) = bounded::<WorkItem>(num_workers * 2);
    let (result_tx, result_rx) = bounded::<Vec<CachedSegmentNBD>>(num_workers * 2);

    println!("Starting {} worker threads...", num_workers);
    println!("─────────────────────────────────────────────────────────────────────────");

    // Clone senders for workers before we use work_tx
    let work_rx_for_workers = work_rx.clone();
    let result_tx_for_workers = result_tx.clone();

    // Spawn worker threads
    let workers: Vec<_> = (0..num_workers)
        .map(|id| {
            let work_rx = work_rx_for_workers.clone();
            let result_tx = result_tx_for_workers.clone();
            let config = nbd_config.clone();
            thread::spawn(move || {
                worker(id, work_rx, result_tx, config);
            })
        })
        .collect();

    // Drop the receiver clone used for workers
    drop(work_rx_for_workers);
    drop(result_tx_for_workers);

    // Spawn a sender thread to avoid deadlock
    // (main thread will be receiving results while this sends work)
    let sender_handle = thread::spawn(move || {
        println!("Sending {} work items to workers...", total_files);
        for item in work_items {
            if work_tx.send(item).is_err() {
                break; // Workers finished
            }
        }
        // Drop the sender so workers know there's no more work
        drop(work_tx);
        println!("All work items sent.");
    });

    // Main thread: collect results and write batches (runs in parallel with sender)
    let batch_size = 500;
    let mut batch: Vec<CachedSegmentNBD> = Vec::new();
    let mut batch_num = 0;
    let mut total_segments = 0;
    let mut files_processed = 0;

    while let Ok(segments) = result_rx.recv() {
        files_processed += 1;
        batch.extend(segments);

        if batch.len() >= batch_size {
            batch_num += 1;
            total_segments += batch.len();

            let cache_file = cache_dir.join(format!("nbd_batch_{:04}.json", batch_num));
            if let Ok(json) = serde_json::to_string(&batch) {
                if let Ok(mut f) = File::create(&cache_file) {
                    let _ = f.write_all(json.as_bytes());
                }
            }

            println!(
                "  Batch {:4}: {} segments ({} files, {:.0}% done)",
                batch_num,
                batch.len(),
                files_processed,
                files_processed as f64 / total_files as f64 * 100.0
            );

            batch.clear();
        }
    }

    // Wait for sender to finish
    let _ = sender_handle.join();

    // Wait for workers to finish
    for worker in workers {
        let _ = worker.join();
    }

    // Write final batch
    if !batch.is_empty() {
        batch_num += 1;
        total_segments += batch.len();

        let cache_file = cache_dir.join(format!("nbd_batch_{:04}.json", batch_num));
        if let Ok(json) = serde_json::to_string(&batch) {
            if let Ok(mut f) = File::create(&cache_file) {
                let _ = f.write_all(json.as_bytes());
            }
        }
        println!("  Batch {:4}: Final batch", batch_num);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("PARALLEL CACHE COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  Files processed: {}", files_processed);
    println!("  Total segments: {}", total_segments);
    println!("  Batch files: {}", batch_num);
    println!("  Cache: {}", cache_dir.display());
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
