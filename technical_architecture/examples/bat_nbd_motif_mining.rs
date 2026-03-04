//! Neural Boundary Detection Motif Mining on Egyptian Fruit Bats
//! ==============================================================
//!
//! Tests the "Hidden Discrete Motifs" hypothesis on bat FM sweeps.
//!
//! Expected Results for Egyptian Fruit Bats:
//!   - Purity: 10-20% (LOW)
//!   - Noise Ratio: 80-90% (HIGH)
//!   - Interpretation: Prosodic modulation - FM sweeps are unique events

use ndarray::Array2;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use technical_architecture::{
    BoundaryDetectorConfig, HdbscanClustering, MicroDynamicsExtractor, MicroDynamicsFeatures45D,
    NeuralBoundaryDetector,
};

/// Segment metadata
#[derive(Debug, Clone)]
struct SegmentInfo {
    source_file: String,
    context: i32,
    emitter: i32,
    features: Vec<f64>,
}

/// Load WAV file (32-bit float format)
fn load_wav(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let bytes = fs::read(path)?;

    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("Not a valid WAV file");
    }

    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut num_channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut audio_format = 0u16;

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
            num_channels = u16::from_le_bytes([fmt_data[2], fmt_data[3]]);
            sample_rate = u32::from_le_bytes([fmt_data[4], fmt_data[5], fmt_data[6], fmt_data[7]]);
            bits_per_sample = u16::from_le_bytes([fmt_data[14], fmt_data[15]]);
        } else if chunk_id == b"data" {
            let data_start = pos + 8;
            let data_end = pos + 8 + chunk_size;
            let audio_bytes = &bytes[data_start..data_end.min(bytes.len())];

            let samples: Vec<f32> = match (audio_format, bits_per_sample) {
                (3, 32) => {
                    // IEEE float
                    audio_bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect()
                }
                (1, 16) => audio_bytes
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
                    .collect(),
                _ => anyhow::bail!(
                    "Unsupported format: format={}, bits={}",
                    audio_format,
                    bits_per_sample
                ),
            };

            let mono_samples = if num_channels == 2 {
                samples
                    .chunks_exact(2)
                    .map(|c| (c[0] + c[1]) / 2.0)
                    .collect()
            } else {
                samples
            };

            return Ok((mono_samples, sample_rate));
        }

        pos += 8 + chunk_size + (chunk_size % 2);
    }

    anyhow::bail!("No data chunk found")
}

/// Parse annotations CSV
fn parse_annotations(path: &Path) -> anyhow::Result<HashMap<String, (i32, i32)>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut annotations = HashMap::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 {
            continue; // Skip header
        }

        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 8 {
            let emitter: i32 = parts[0].parse().unwrap_or(0);
            let context: i32 = parts[2].parse().unwrap_or(0);
            let filename = format!("{}.wav", parts[7]);
            annotations.insert(filename, (emitter, context));
        }
    }

    Ok(annotations)
}

/// Compute 105D features
fn compute_105d_features(extractor: &MicroDynamicsExtractor, audio: &[f32]) -> Option<Vec<f64>> {
    let base_45d = extractor.extract_45d(audio).ok()?;
    let mut features = Vec::with_capacity(105);
    features.extend(base_45d.to_array().iter().map(|&v| v as f64));
    features.extend(compute_macro_texture(&base_45d));
    features.extend(compute_micro_texture(&base_45d));
    Some(features)
}

fn compute_macro_texture(base_45d: &MicroDynamicsFeatures45D) -> Vec<f64> {
    let mut f = Vec::with_capacity(30);
    f.push(base_45d.spectral_tilt as f64);
    f.push(base_45d.base_30d.harmonic_to_noise_ratio as f64 * 0.5);
    f.push(base_45d.base_30d.jitter as f64);
    f.push(base_45d.base_30d.harmonic_to_noise_ratio as f64 * 0.1);
    f.push(base_45d.base_30d.spectral_flux as f64);
    f.push(base_45d.formant_1_hz as f64 / (base_45d.formant_2_hz as f64 + 1.0));
    f.push(base_45d.formant_2_hz as f64 / (base_45d.formant_3_hz as f64 + 1.0));
    f.push(base_45d.formant_3_hz as f64 / (base_45d.formant_dispersion as f64 * 10.0 + 1.0));
    f.push(base_45d.f0_range_hz as f64 / (base_45d.duration_ms as f64 + 1.0));
    f.push(base_45d.fm_slope as f64 * 0.5);
    f.push(0.0);
    f.push(base_45d.fm_slope as f64);
    f.push(base_45d.base_30d.vibrato_rate_hz as f64 / 10.0);
    f.push(base_45d.base_30d.jitter as f64 * 10.0);
    f.push(base_45d.f0_range_hz as f64 / (base_45d.mean_f0_hz as f64 + 1.0));
    f.push(base_45d.spectral_kurtosis as f64);
    f.push(base_45d.spectral_skewness as f64 * 0.5);
    f.push(1.0 - base_45d.base_30d.spectral_flatness as f64);
    f.push(1.0 - base_45d.base_30d.spectral_flatness as f64);
    f.push(base_45d.spectral_spread as f64 * 0.01);
    f.push(base_45d.duration_ms as f64 / 100.0);
    f.push(1.0 / (base_45d.duration_ms as f64 / 100.0 + 1.0));
    f.push(base_45d.base_30d.spectral_flatness as f64);
    f.push(base_45d.am_depth as f64);
    f.push(base_45d.fm_slope as f64 * 0.1);
    f.push(0.1);
    f.push(
        base_45d.base_30d.attack_time_ms as f64 / (base_45d.base_30d.decay_time_ms as f64 + 1.0),
    );
    f.push(base_45d.base_30d.sustain_level as f64 * 10.0);
    f.push(base_45d.base_30d.vibrato_depth as f64 / 100.0);
    f.push(0.1);
    f
}

fn compute_micro_texture(base_45d: &MicroDynamicsFeatures45D) -> Vec<f64> {
    let mut f = Vec::with_capacity(30);
    let vibrato_rate = base_45d.base_30d.vibrato_rate_hz as f64;
    f.push(if vibrato_rate < 10.0 { 1.0 } else { 0.0 });
    f.push(if vibrato_rate >= 10.0 && vibrato_rate < 30.0 {
        1.0
    } else {
        0.0
    });
    f.push(if vibrato_rate >= 30.0 && vibrato_rate < 50.0 {
        1.0
    } else {
        0.0
    });
    f.push(if vibrato_rate >= 50.0 && vibrato_rate < 100.0 {
        1.0
    } else {
        0.0
    });
    f.push(base_45d.am_depth as f64);
    let fm_rate = base_45d.fm_slope as f64;
    f.push(if fm_rate < 10.0 { 1.0 } else { 0.0 });
    f.push(if fm_rate >= 10.0 && fm_rate < 30.0 {
        1.0
    } else {
        0.0
    });
    f.push(if fm_rate >= 30.0 && fm_rate < 50.0 {
        1.0
    } else {
        0.0
    });
    f.push(if fm_rate >= 50.0 && fm_rate < 100.0 {
        1.0
    } else {
        0.0
    });
    f.push(0.0);
    f.push(base_45d.base_30d.vibrato_rate_hz as f64);
    f.push(base_45d.fm_slope as f64);
    f.push(base_45d.am_depth as f64);
    f.push(base_45d.fm_slope as f64 * 0.5);
    f.push(base_45d.base_30d.vibrato_depth as f64);
    let ici = base_45d.base_30d.median_ici_ms as f64;
    f.push(if ici < 20.0 { 1.0 } else { 0.0 });
    f.push(if ici >= 20.0 && ici < 50.0 { 1.0 } else { 0.0 });
    f.push(if ici >= 50.0 && ici < 100.0 { 1.0 } else { 0.0 });
    f.push(if ici >= 100.0 && ici < 200.0 {
        1.0
    } else {
        0.0
    });
    f.push(if ici >= 200.0 { 1.0 } else { 0.0 });
    f.push(base_45d.base_30d.median_ici_ms as f64);
    f.push(1.0 / (base_45d.base_30d.median_ici_ms as f64 / 1000.0 + 0.001));
    f.push(base_45d.base_30d.onset_rate_hz as f64);
    f.push(base_45d.base_30d.ici_coefficient_of_variation as f64);
    f.push(base_45d.base_30d.onset_rate_hz as f64 * 60.0);
    f.push(base_45d.spectral_centroid as f64 / 1000.0);
    f.push(base_45d.base_30d.harmonic_to_noise_ratio as f64);
    f.push(1.0 - base_45d.subharmonic_ratio as f64);
    f.push(base_45d.spectral_entropy as f64);
    f.push(base_45d.base_30d.harmonicity as f64);
    f
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║    NEURAL BOUNDARY DETECTION - EGYPTIAN FRUIT BAT MOTIF MINING            ║");
    println!("║              Testing 'Hidden Discrete Motifs' Hypothesis                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = data_dir.join("audio");
    let annotations_path = data_dir.join("annotations.csv");

    // Parse annotations
    println!("Loading annotations...");
    let annotations = parse_annotations(&annotations_path)?;
    println!("  Loaded {} annotations", annotations.len());
    println!();

    // Configuration - Note: Bat audio is 250kHz!
    let sample_rate = 250000u32;

    let nbd_config = BoundaryDetectorConfig {
        hop_size: 2048, // Larger hop for high sample rate
        sample_rate,
        min_phrase_duration_ms: 10.0, // Bats have short calls
        threshold: 0.25,              // Sensitive for FM sweep detection
        smoothing_frames: 2,
    };

    println!("Configuration:");
    println!("  • Sample Rate: {} Hz (250 kHz)", sample_rate);
    println!(
        "  • Hop Size: {} samples ({:.2}ms)",
        nbd_config.hop_size,
        nbd_config.hop_size as f32 / sample_rate as f32 * 1000.0
    );
    println!(
        "  • Min Phrase Duration: {}ms",
        nbd_config.min_phrase_duration_ms
    );
    println!("  • Threshold: {}", nbd_config.threshold);
    println!();

    // Sample files - get a stratified sample across contexts
    let mut files_by_context: HashMap<i32, Vec<PathBuf>> = HashMap::new();

    for (filename, (_, context)) in &annotations {
        let path = audio_dir.join(filename);
        if path.exists() {
            files_by_context.entry(*context).or_default().push(path);
        }
    }

    // Sample up to 50 files per context, max 500 total
    let mut selected_files: Vec<(PathBuf, i32)> = Vec::new();
    let max_per_context = 50;
    let max_total = 500;

    for (context, mut files) in files_by_context {
        // Shuffle and take
        files.sort_by_key(|p| {
            p.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .parse::<String>()
                .unwrap_or_default()
        });
        for path in files.into_iter().take(max_per_context) {
            if selected_files.len() >= max_total {
                break;
            }
            selected_files.push((path, context));
        }
        if selected_files.len() >= max_total {
            break;
        }
    }

    println!("Selected {} files for analysis", selected_files.len());
    println!();

    // NBD and feature extraction
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("STEP 1: Neural Boundary Detection + 105D Feature Extraction");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let extractor = MicroDynamicsExtractor::new(sample_rate);
    let mut all_segments: Vec<SegmentInfo> = Vec::new();
    let mut context_counts: HashMap<i32, usize> = HashMap::new();
    let mut total_boundaries = 0;
    let mut files_processed = 0;

    for (path, context) in selected_files {
        let filename = path.file_name().unwrap().to_str().unwrap();

        match load_wav(&path) {
            Ok((audio, sr)) => {
                let duration_ms = audio.len() as f32 / sr as f32 * 1000.0;

                let mut detector = NeuralBoundaryDetector::with_config(BoundaryDetectorConfig {
                    sample_rate: sr,
                    ..nbd_config.clone()
                });

                let boundaries = detector.detect_boundaries(&audio);
                total_boundaries += boundaries.len();

                // Segment
                let mut segments_audio: Vec<Vec<f32>> = Vec::new();
                let mut start = 0usize;

                for b in &boundaries {
                    let end = (b.time_ms * sr as f32 / 1000.0) as usize;
                    if end > start && end <= audio.len() {
                        segments_audio.push(audio[start..end].to_vec());
                    }
                    start = end;
                }
                if start < audio.len() {
                    segments_audio.push(audio[start..].to_vec());
                }

                if segments_audio.is_empty() {
                    segments_audio.push(audio);
                }

                // Extract features
                let min_len = (sr as f32 * 0.005) as usize; // 5ms minimum
                let mut seg_count = 0;

                for seg_audio in segments_audio {
                    if seg_audio.len() < min_len {
                        continue;
                    }

                    if let Some(features) = compute_105d_features(&extractor, &seg_audio) {
                        all_segments.push(SegmentInfo {
                            source_file: filename.to_string(),
                            context,
                            emitter: 0, // Not tracking per-segment
                            features,
                        });
                        *context_counts.entry(context).or_insert(0) += 1;
                        seg_count += 1;
                    }
                }

                files_processed += 1;
                if files_processed % 50 == 0 {
                    println!(
                        "  Processed {}/{} files...",
                        files_processed,
                        selected_files.len()
                    );
                }
            }
            Err(e) => {
                println!("  Error loading {}: {}", filename, e);
            }
        }
    }

    println!();
    println!("  Files processed: {}", files_processed);
    println!("  Total boundaries detected: {}", total_boundaries);
    println!("  Total segments extracted: {}", all_segments.len());
    println!("  Segments by context:");
    for (ctx, count) in context_counts.iter() {
        println!("    • Context {}: {}", ctx, count);
    }
    println!();

    // Clustering
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("STEP 2: HDBSCAN Clustering");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    if all_segments.len() < 10 {
        println!("Insufficient segments for clustering");
        return Ok(());
    }

    let n_segments = all_segments.len();
    let n_features = 105;
    let min_cluster_size = (n_segments / 20).max(5).min(20);

    let mut feature_matrix = Array2::<f64>::zeros((n_segments, n_features));
    for (i, seg) in all_segments.iter().enumerate() {
        for (j, &val) in seg.features.iter().enumerate().take(n_features) {
            feature_matrix[[i, j]] = val;
        }
    }

    println!(
        "  Feature matrix: {} segments × {} features",
        n_segments, n_features
    );
    println!("  min_cluster_size: {}", min_cluster_size);
    println!();

    let hdbscan = HdbscanClustering::new(min_cluster_size, 3)?;
    let labels = hdbscan.fit_predict(&feature_matrix)?;

    let stats = hdbscan.get_cluster_stats(&labels);
    let noise_count = labels.iter().filter(|&&l| l == -1).count();
    let purity = (n_segments - noise_count) as f64 / n_segments as f64;
    let noise_ratio = noise_count as f64 / n_segments as f64;

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("STEP 3: Results");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  Total segments: {}", n_segments);
    println!("  Clusters found: {}", stats.n_clusters);
    println!(
        "  Noise points: {} ({:.1}%)",
        noise_count,
        noise_ratio * 100.0
    );
    println!("  Purity: {:.1}%", purity * 100.0);
    println!();

    // Cluster composition
    let mut cluster_members: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        cluster_members.entry(label).or_default().push(idx);
    }

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  CLUSTER COMPOSITION BY CONTEXT                                          │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    let mut sorted_clusters: Vec<_> = cluster_members.iter().collect();
    sorted_clusters.sort_by_key(|(&label, _)| if label == -1 { 999 } else { label });

    for (&label, member_indices) in sorted_clusters.iter().take(10) {
        if label == -1 {
            println!(
                "  │  NOISE ({})                                                          ",
                member_indices.len()
            );
        } else {
            let mut ctx_counts: HashMap<i32, usize> = HashMap::new();
            for &idx in member_indices.iter() {
                let ctx = all_segments[idx].context;
                *ctx_counts.entry(ctx).or_insert(0) += 1;
            }

            println!(
                "  │  CLUSTER {} ({} segments)                                       ",
                label,
                member_indices.len()
            );
            let mut sorted_ctx: Vec<_> = ctx_counts.iter().collect();
            sorted_ctx.sort_by(|a, b| b.1.cmp(a.1));
            for (ctx, count) in sorted_ctx.iter().take(3) {
                let pct = *count as f64 / member_indices.len() as f64 * 100.0;
                println!("  │    • Context {:2}: {} ({:.0}%)", ctx, count, pct);
            }
        }
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Interpretation
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("INTERPRETATION");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  Expected for Egyptian Fruit Bats:");
    println!("    • Purity: 10-20%");
    println!("    • Noise: 80-90%");
    println!("    • Interpretation: Prosodic modulation (unique FM sweeps)");
    println!();

    println!("  Observed:");
    println!("    • Purity: {:.1}%", purity * 100.0);
    println!("    • Noise: {:.1}%", noise_ratio * 100.0);
    println!();

    if purity < 0.25 {
        println!("  ✓ CONFIRMS HYPOTHESIS: LOW MOTIF REUSE");
        println!();
        println!("  Egyptian fruit bat vocalizations are predominantly unique events.");
        println!("  FM sweeps show prosodic modulation - each call is a 'solo performance'.");
        println!();
        println!("  → Use Direct 105D similarity (Bag-of-Phrases will FAIL)");
    } else if purity < 0.50 {
        println!("  ~ MODERATE MOTIF REUSE");
        println!();
        println!("  Some acoustic patterns are reused, but many are unique.");
        println!("  This suggests a MIXED system.");
    } else {
        println!("  ⚠ UNEXPECTED: HIGH MOTIF REUSE");
        println!();
        println!("  This is higher than expected for bats. Possible explanations:");
        println!("    1. Dataset contains many similar context calls");
        println!("    2. NBD is grouping by energy patterns, not semantic content");
        println!("    3. Bats may reuse more patterns than expected");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
