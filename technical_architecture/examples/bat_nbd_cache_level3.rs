//! Bat NBD Cache with Interaction Data (Level 3)
//! ==============================================
//!
//! Extracts NBD segments WITH Addressee ID for turn-taking model support.
//!
//! Level 3 Capabilities:
//! - Sequential audio (multiple segments per file)
//! - Context (behavioral state)
//! - Emitter ID (who is calling)
//! - Addressee ID (who is being addressed) ← NEW
//! - Timestamps (when)
//!
//! This enables:
//! - Turn-Taking Models: "If Bat A calls, Bat B replies within 200ms"
//! - Addressing Models: "Who is being talked to"
//! - Conversational Rules: "When to respond"

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

/// Cached segment with full interaction data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedSegmentInteraction {
    source_file: String,
    context: i32,
    emitter: i32,
    addressee: i32, // NEW: Receiver ID
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    duration_ms: f32,
    boundary_type: String,
    // Interaction metadata
    is_self_addressed: bool, // Emitter == Addressee
    features: Vec<f32>,
}

/// Annotation from CSV
#[derive(Debug, Clone)]
struct Annotation {
    emitter: i32,
    addressee: i32,
    context: i32,
    file_name: String,
}

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

fn parse_annotations(path: &Path) -> anyhow::Result<HashMap<String, Annotation>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut annotations = HashMap::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue;
        } // Skip header
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 8 {
            let emitter: i32 = parts[0].parse().unwrap_or(0);
            let addressee: i32 = parts[1].parse().unwrap_or(0);
            let context: i32 = parts[2].parse().unwrap_or(0);
            let filename = parts[7].to_string();

            annotations.insert(
                filename,
                Annotation {
                    emitter,
                    addressee,
                    context,
                    file_name: parts[7].to_string(),
                },
            );
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

fn boundary_type_str(bt: technical_architecture::NeuralBoundaryType) -> String {
    match bt {
        technical_architecture::NeuralBoundaryType::Hard => "Hard".to_string(),
        technical_architecture::NeuralBoundaryType::Soft => "Soft".to_string(),
        technical_architecture::NeuralBoundaryType::Transitional => "Transitional".to_string(),
    }
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     BAT NBD CACHE - LEVEL 3 (INTERACTION-READY)                          ║");
    println!("║     Extracting Emitter + Addressee for Turn-Taking Models                ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = data_dir.join("audio");
    let annotations_path = data_dir.join("annotations.csv");
    let cache_dir = Path::new("bat_nbd_cache_level3");

    fs::create_dir_all(cache_dir)?;
    println!("Cache directory: {}", cache_dir.display());

    // Load annotations
    let annotations = parse_annotations(&annotations_path)?;
    println!("Annotations: {} files", annotations.len());

    // Get all WAV files
    let all_files: Vec<PathBuf> = fs::read_dir(&audio_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    let total_files = all_files.len();
    println!("Total WAV files: {}", total_files);
    println!();

    // NBD configuration
    let nbd_config = BoundaryDetectorConfig {
        hop_size: 2048,
        sample_rate: 250000,
        min_phrase_duration_ms: 5.0,
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

    // Statistics (thread-safe)
    let emitter_counts = Arc::new(Mutex::new(HashMap::<i32, usize>::new()));
    let addressee_counts = Arc::new(Mutex::new(HashMap::<i32, usize>::new()));
    let context_counts = Arc::new(Mutex::new(HashMap::<i32, usize>::new()));
    let interaction_pairs = Arc::new(Mutex::new(HashMap::<(i32, i32), usize>::new()));
    let self_addressed = Arc::new(Mutex::new(0usize));
    let total_segments = Arc::new(Mutex::new(0usize));
    let files_processed = Arc::new(Mutex::new(0usize));
    let files_with_addressee = Arc::new(Mutex::new(0usize));

    let batch_size = 500;
    let progress = Arc::new(Mutex::new(0usize));
    let nbd_config = Arc::new(nbd_config);
    let annotations = Arc::new(annotations);

    println!("Processing {} files in parallel...", total_files);
    println!("─────────────────────────────────────────────────────────────────────────");

    // Process files in parallel
    let all_segments: Vec<Vec<CachedSegmentInteraction>> = all_files
        .par_iter()
        .map(|path| {
            let filename = path.file_name().unwrap().to_str().unwrap().to_string();

            // Get annotation
            let annotation = annotations.get(&filename);
            let (emitter, addressee, context) = match annotation {
                Some(a) => {
                    *files_with_addressee.lock().unwrap() += 1;
                    (a.emitter, a.addressee, a.context)
                }
                None => (0, 0, 0),
            };

            let mut result = Vec::new();

            match load_wav(path) {
                Ok((audio, sr)) => {
                    let config = BoundaryDetectorConfig {
                        sample_rate: sr,
                        hop_size: nbd_config.hop_size,
                        min_phrase_duration_ms: nbd_config.min_phrase_duration_ms,
                        threshold: nbd_config.threshold,
                        smoothing_frames: nbd_config.smoothing_frames,
                    };
                    let mut detector = NeuralBoundaryDetector::with_config(config);

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
                    let extractor = MicroDynamicsExtractor::new(sr);

                    for (seg_idx, (start, end, btype)) in segments.into_iter().enumerate() {
                        if let Some(features) = compute_105d_f32(&extractor, &audio[start..end]) {
                            let start_ms = start as f32 / sr as f32 * 1000.0;
                            let end_ms = end as f32 / sr as f32 * 1000.0;
                            let duration_ms = end_ms - start_ms;
                            let is_self = emitter == addressee;

                            result.push(CachedSegmentInteraction {
                                source_file: filename.clone(),
                                context,
                                emitter,
                                addressee,
                                segment_idx: seg_idx,
                                start_ms,
                                end_ms,
                                duration_ms,
                                boundary_type: btype,
                                is_self_addressed: is_self,
                                features,
                            });

                            // Update stats
                            *emitter_counts.lock().unwrap().entry(emitter).or_insert(0) += 1;
                            *addressee_counts
                                .lock()
                                .unwrap()
                                .entry(addressee)
                                .or_insert(0) += 1;
                            *context_counts.lock().unwrap().entry(context).or_insert(0) += 1;
                            if !is_self {
                                *interaction_pairs
                                    .lock()
                                    .unwrap()
                                    .entry((emitter, addressee))
                                    .or_insert(0) += 1;
                            } else {
                                *self_addressed.lock().unwrap() += 1;
                            }
                            *total_segments.lock().unwrap() += 1;
                        }
                    }

                    *files_processed.lock().unwrap() += 1;
                }
                Err(_) => {}
            }

            // Update progress
            let mut prog = progress.lock().unwrap();
            *prog += 1;
            if *prog % 5000 == 0 {
                println!("  Progress: {}/{} files...", *prog, total_files);
            }

            result
        })
        .collect();

    let total_segments_val = *total_segments.lock().unwrap();
    let files_processed_val = *files_processed.lock().unwrap();
    let files_with_addressee_val = *files_with_addressee.lock().unwrap();
    let self_addressed_val = *self_addressed.lock().unwrap();

    // Flatten and save batches
    println!();
    println!("  Saving batches...");

    let all_segments_flat: Vec<_> = all_segments.into_iter().flatten().collect();
    let total_flat = all_segments_flat.len();

    let mut batch_num = 0;
    for chunk in all_segments_flat.chunks(batch_size) {
        batch_num += 1;
        let cache_file = cache_dir.join(format!("level3_batch_{:04}.json", batch_num));
        let json = serde_json::to_string(chunk)?;
        let mut file = File::create(&cache_file)?;
        file.write_all(json.as_bytes())?;
    }

    // Final batch
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("LEVEL 3 CACHE COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  Files processed: {}", files_processed_val);
    println!(
        "  Files with addressee: {} ({:.1}%)",
        files_with_addressee_val,
        files_with_addressee_val as f64 / files_processed_val as f64 * 100.0
    );
    println!("  Total segments: {}", total_flat);
    println!("  Batch files: {}", batch_num);
    println!();

    // Get final statistics
    let emitter_counts = Arc::try_unwrap(emitter_counts)
        .unwrap()
        .into_inner()
        .unwrap();
    let addressee_counts = Arc::try_unwrap(addressee_counts)
        .unwrap()
        .into_inner()
        .unwrap();
    let context_counts = Arc::try_unwrap(context_counts)
        .unwrap()
        .into_inner()
        .unwrap();
    let interaction_pairs = Arc::try_unwrap(interaction_pairs)
        .unwrap()
        .into_inner()
        .unwrap();

    // Interaction statistics
    println!("  INTERACTION STATISTICS:");
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");

    // Top emitters
    let mut sorted_emitters: Vec<_> = emitter_counts.iter().collect();
    sorted_emitters.sort_by(|a, b| b.1.cmp(a.1));
    println!("  │  Top Emitters:                                                          │");
    for (emit, count) in sorted_emitters.iter().take(5) {
        let pct = **count as f64 / total_flat as f64 * 100.0;
        println!(
            "  │    • Bat {:4}: {:6} calls ({:.1}%)                               │",
            emit, count, pct
        );
    }

    // Top addressees
    let mut sorted_addressees: Vec<_> = addressee_counts.iter().collect();
    sorted_addressees.sort_by(|a, b| b.1.cmp(a.1));
    println!("  │  Top Addressees:                                                        │");
    for (addr, count) in sorted_addressees.iter().take(5) {
        let pct = **count as f64 / total_flat as f64 * 100.0;
        println!(
            "  │    • Bat {:4}: {:6} received ({:.1}%)                             │",
            addr, count, pct
        );
    }

    println!("  │                                                                          │");
    println!(
        "  │  Self-addressed: {} ({:.1}%)                                            │",
        self_addressed_val,
        self_addressed_val as f64 / total_flat as f64 * 100.0
    );

    // Top interaction pairs
    let mut sorted_pairs: Vec<_> = interaction_pairs.iter().collect();
    sorted_pairs.sort_by(|a, b| b.1.cmp(a.1));
    println!("  │  Top Interaction Pairs (Emitter → Addressee):                          │");
    for ((emit, addr), count) in sorted_pairs.iter().take(5) {
        let pct = **count as f64 / total_flat as f64 * 100.0;
        println!(
            "  │    • {:4} → {:4}: {:6} calls ({:.1}%)                           │",
            emit, addr, count, pct
        );
    }

    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Level capability summary
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  LEVEL 3 CAPABILITY SUMMARY                                             │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │                                                                          │");
    println!(
        "  │  ✓ Sequential Audio: {} segments from {} files                      ",
        total_flat, files_processed_val
    );
    println!(
        "  │  ✓ Context: {} behavioral contexts                                        ",
        context_counts.len()
    );
    println!(
        "  │  ✓ Emitter ID: {} unique callers                                         ",
        emitter_counts.len()
    );
    println!(
        "  │  ✓ Addressee ID: {} unique receivers                                     ",
        addressee_counts.len()
    );
    println!("  │  ✓ Timestamps: start_ms, end_ms per segment                             │");
    println!(
        "  │  ✓ Interaction Pairs: {} unique (Emitter, Addressee) pairs               ",
        interaction_pairs.len()
    );
    println!("  │                                                                          │");
    println!("  │  ═══════════════════════════════════════════════════════════════════    │");
    println!("  │                                                                          │");
    println!("  │  NOW ENABLED:                                                            │");
    println!("  │  • Turn-Taking Models: \"If Bat A calls, Bat B replies\"                  │");
    println!("  │  • Addressing Models: \"Who is being talked to\"                           │");
    println!("  │  • Conversational Timing: \"When to respond\"                             │");
    println!("  │  • Social Network Analysis: Interaction graphs                          │");
    println!("  │                                                                          │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("  Cache directory: {}", cache_dir.display());
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
