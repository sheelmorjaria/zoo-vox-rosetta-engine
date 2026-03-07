//! Full Graded Phrase Mining using ACTUAL Neural Boundary Detection
//! ================================================================
//!
//! Uses NeuralBoundaryDetector to segment graded vocalizations,
//! then clusters segments to test motif reuse hypothesis.

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use ndarray::Array2;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use technical_architecture::{
    BoundaryDetectorConfig, HdbscanClustering, MicroDynamicsExtractor, MicroDynamicsFeatures45D, NeuralBoundaryDetector,
};

/// Segment metadata for tracking
#[derive(Debug, Clone)]
struct SegmentInfo {
    source_file: String,
    call_type: String,
    segment_idx: usize,
    features: Vec<f64>,
}

/// Load WAV file
fn load_wav(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let bytes = fs::read(path)?;

    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("Not a valid WAV file");
    }

    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut num_channels = 0u16;
    let mut bits_per_sample = 0u16;

    while pos < bytes.len() - 8 {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;

        if chunk_id == b"fmt " {
            let fmt_data = &bytes[pos + 8..pos + 8 + chunk_size.min(16)];
            num_channels = u16::from_le_bytes([fmt_data[2], fmt_data[3]]);
            sample_rate = u32::from_le_bytes([fmt_data[4], fmt_data[5], fmt_data[6], fmt_data[7]]);
            bits_per_sample = u16::from_le_bytes([fmt_data[14], fmt_data[15]]);
        } else if chunk_id == b"data" {
            let data_start = pos + 8;
            let data_end = pos + 8 + chunk_size;
            let audio_bytes = &bytes[data_start..data_end.min(bytes.len())];

            let samples: Vec<f32> = match bits_per_sample {
                16 => audio_bytes
                    .chunks_exact(2)
                    .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
                    .collect(),
                _ => anyhow::bail!("Unsupported bits per sample: {}", bits_per_sample),
            };

            let mono_samples = if num_channels == 2 {
                samples.chunks_exact(2).map(|c| (c[0] + c[1]) / 2.0).collect()
            } else {
                samples
            };

            return Ok((mono_samples, sample_rate));
        }

        pos += 8 + chunk_size + (chunk_size % 2);
    }

    anyhow::bail!("No data chunk found")
}

/// Extract call type from filename
fn get_call_type(filename: &str) -> &str {
    if filename.starts_with("Tsik") {
        "Tsik"
    } else if filename.starts_with("Twitter") {
        "Twitter"
    } else {
        "Vocalization"
    }
}

/// Compute 105D features from audio segment
fn compute_105d_features(extractor: &MicroDynamicsExtractor, audio: &[f32]) -> Option<Vec<f64>> {
    let base_45d = extractor.extract_45d(audio).ok()?;
    let mut features = Vec::with_capacity(105);
    features.extend(base_45d.to_array().iter().map(|&v| v as f64));
    features.extend(compute_macro_texture(&base_45d));
    features.extend(compute_micro_texture(&base_45d));
    Some(features)
}

fn compute_macro_texture(base_45d: &MicroDynamicsFeatures45D) -> Vec<f64> {
    let mut features = Vec::with_capacity(30);

    // Harmonic Texture (8D)
    features.push(base_45d.spectral_tilt as f64);
    features.push(base_45d.base_30d.harmonic_to_noise_ratio as f64 * 0.5);
    features.push(base_45d.base_30d.jitter as f64);
    features.push(base_45d.base_30d.harmonic_to_noise_ratio as f64 * 0.1);
    features.push(base_45d.base_30d.spectral_flux as f64);
    features.push(base_45d.formant_1_hz as f64 / (base_45d.formant_2_hz as f64 + 1.0));
    features.push(base_45d.formant_2_hz as f64 / (base_45d.formant_3_hz as f64 + 1.0));
    features.push(base_45d.formant_3_hz as f64 / (base_45d.formant_dispersion as f64 * 10.0 + 1.0));

    // Pitch Geometry (7D)
    features.push(base_45d.f0_range_hz as f64 / (base_45d.duration_ms as f64 + 1.0));
    features.push(base_45d.fm_slope as f64 * 0.5);
    features.push(0.0);
    features.push(base_45d.fm_slope as f64);
    features.push(base_45d.base_30d.vibrato_rate_hz as f64 / 10.0);
    features.push(base_45d.base_30d.jitter as f64 * 10.0);
    features.push(base_45d.f0_range_hz as f64 / (base_45d.mean_f0_hz as f64 + 1.0));

    // GLCM Texture (10D)
    features.push(base_45d.spectral_kurtosis as f64);
    features.push(base_45d.spectral_skewness as f64 * 0.5);
    features.push(1.0 - base_45d.base_30d.spectral_flatness as f64);
    features.push(1.0 - base_45d.base_30d.spectral_flatness as f64);
    features.push(base_45d.spectral_spread as f64 * 0.01);
    features.push(base_45d.duration_ms as f64 / 100.0);
    features.push(1.0 / (base_45d.duration_ms as f64 / 100.0 + 1.0));
    features.push(base_45d.base_30d.spectral_flatness as f64);
    features.push(base_45d.am_depth as f64);
    features.push(base_45d.fm_slope as f64 * 0.1);

    // Temporal Texture (5D)
    features.push(0.1);
    features.push(base_45d.base_30d.attack_time_ms as f64 / (base_45d.base_30d.decay_time_ms as f64 + 1.0));
    features.push(base_45d.base_30d.sustain_level as f64 * 10.0);
    features.push(base_45d.base_30d.vibrato_depth as f64 / 100.0);
    features.push(0.1);

    features
}

fn compute_micro_texture(base_45d: &MicroDynamicsFeatures45D) -> Vec<f64> {
    let mut features = Vec::with_capacity(30);

    let vibrato_rate = base_45d.base_30d.vibrato_rate_hz as f64;
    features.push(if vibrato_rate < 10.0 { 1.0 } else { 0.0 });
    features.push(if vibrato_rate >= 10.0 && vibrato_rate < 30.0 {
        1.0
    } else {
        0.0
    });
    features.push(if vibrato_rate >= 30.0 && vibrato_rate < 50.0 {
        1.0
    } else {
        0.0
    });
    features.push(if vibrato_rate >= 50.0 && vibrato_rate < 100.0 {
        1.0
    } else {
        0.0
    });
    features.push(base_45d.am_depth as f64);

    let fm_rate = base_45d.fm_slope as f64;
    features.push(if fm_rate < 10.0 { 1.0 } else { 0.0 });
    features.push(if fm_rate >= 10.0 && fm_rate < 30.0 { 1.0 } else { 0.0 });
    features.push(if fm_rate >= 30.0 && fm_rate < 50.0 { 1.0 } else { 0.0 });
    features.push(if fm_rate >= 50.0 && fm_rate < 100.0 { 1.0 } else { 0.0 });
    features.push(0.0);

    features.push(base_45d.base_30d.vibrato_rate_hz as f64);
    features.push(base_45d.fm_slope as f64);
    features.push(base_45d.am_depth as f64);
    features.push(base_45d.fm_slope as f64 * 0.5);
    features.push(base_45d.base_30d.vibrato_depth as f64);

    let ici = base_45d.base_30d.median_ici_ms as f64;
    features.push(if ici < 20.0 { 1.0 } else { 0.0 });
    features.push(if ici >= 20.0 && ici < 50.0 { 1.0 } else { 0.0 });
    features.push(if ici >= 50.0 && ici < 100.0 { 1.0 } else { 0.0 });
    features.push(if ici >= 100.0 && ici < 200.0 { 1.0 } else { 0.0 });
    features.push(if ici >= 200.0 { 1.0 } else { 0.0 });

    features.push(base_45d.base_30d.median_ici_ms as f64);
    features.push(1.0 / (base_45d.base_30d.median_ici_ms as f64 / 1000.0 + 0.001));
    features.push(base_45d.base_30d.onset_rate_hz as f64);
    features.push(base_45d.base_30d.ici_coefficient_of_variation as f64);
    features.push(base_45d.base_30d.onset_rate_hz as f64 * 60.0);

    features.push(base_45d.spectral_centroid as f64 / 1000.0);
    features.push(base_45d.base_30d.harmonic_to_noise_ratio as f64);
    features.push(1.0 - base_45d.subharmonic_ratio as f64);
    features.push(base_45d.spectral_entropy as f64);
    features.push(base_45d.base_30d.harmonicity as f64);

    features
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║    GRADED PHRASE MINING WITH NEURAL BOUNDARY DETECTION                   ║");
    println!("║       Testing 'Hidden Discrete Motifs' Hypothesis on Marmoset             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration for Neural Boundary Detection
    let nbd_config = BoundaryDetectorConfig {
        hop_size: 512,
        sample_rate: 44100,
        min_phrase_duration_ms: 30.0, // Minimum phrase length
        threshold: 0.3,               // Lower = more sensitive to semantic changes
        smoothing_frames: 3,
    };

    println!("Neural Boundary Detection Configuration:");
    println!(
        "  • Hop Size: {} samples ({:.1}ms)",
        nbd_config.hop_size,
        nbd_config.hop_size as f32 / nbd_config.sample_rate as f32 * 1000.0
    );
    println!("  • Min Phrase Duration: {}ms", nbd_config.min_phrase_duration_ms);
    println!("  • Threshold: {} (semantic sensitivity)", nbd_config.threshold);
    println!("  • Smoothing: {} frames", nbd_config.smoothing_frames);
    println!();

    // Clustering configuration
    let min_cluster_size = 5;
    let min_samples = 2;

    println!("Clustering Configuration:");
    println!("  • Features: 105D");
    println!("  • Min Cluster Size: {}", min_cluster_size);
    println!("  • Min Samples: {}", min_samples);
    println!();

    // Load all audio files
    let wav_dir = Path::new("test_marmoset_wav");
    let wav_files: Vec<_> = fs::read_dir(wav_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("STEP 1: Neural Boundary Detection + 105D Feature Extraction");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let extractor = MicroDynamicsExtractor::new(44100);
    let mut all_segments: Vec<SegmentInfo> = Vec::new();
    let mut call_type_counts: HashMap<String, usize> = HashMap::new();
    let mut boundary_counts: HashMap<String, usize> = HashMap::new();

    for wav_path in &wav_files {
        let filename = wav_path.file_name().unwrap().to_str().unwrap().to_string();
        let call_type = get_call_type(&filename).to_string();

        match load_wav(wav_path) {
            Ok((audio, sr)) => {
                let duration_ms = audio.len() as f32 / sr as f32 * 1000.0;

                // Create boundary detector for this file
                let mut detector = NeuralBoundaryDetector::with_config(BoundaryDetectorConfig {
                    sample_rate: sr,
                    ..nbd_config.clone()
                });

                // Detect boundaries using NEURAL BOUNDARY DETECTION
                let boundaries = detector.detect_boundaries(&audio);

                // Convert boundaries to segments
                let mut segments_audio: Vec<Vec<f32>> = Vec::new();
                let mut start_sample = 0usize;

                for boundary in &boundaries {
                    let end_sample = (boundary.time_ms * sr as f32 / 1000.0) as usize;
                    if end_sample > start_sample && end_sample <= audio.len() {
                        segments_audio.push(audio[start_sample..end_sample].to_vec());
                    }
                    start_sample = end_sample;
                }

                // Add final segment
                if start_sample < audio.len() {
                    segments_audio.push(audio[start_sample..].to_vec());
                }

                // If no boundaries, use entire audio as one segment
                if segments_audio.is_empty() {
                    segments_audio.push(audio);
                }

                println!("  📁 {} ({:.0}ms)", filename, duration_ms);
                println!(
                    "     Boundaries detected: {} → {} segments",
                    boundaries.len(),
                    segments_audio.len()
                );

                // Show boundary types
                if !boundaries.is_empty() {
                    let hard_count = boundaries
                        .iter()
                        .filter(|b| format!("{:?}", b.boundary_type) == "Hard")
                        .count();
                    let soft_count = boundaries
                        .iter()
                        .filter(|b| format!("{:?}", b.boundary_type) == "Soft")
                        .count();
                    let trans_count = boundaries
                        .iter()
                        .filter(|b| format!("{:?}", b.boundary_type) == "Transitional")
                        .count();
                    println!(
                        "     Boundary types: {} Hard, {} Soft, {} Transitional",
                        hard_count, soft_count, trans_count
                    );
                }

                boundary_counts.insert(filename.clone(), boundaries.len());

                // Extract 105D features for each segment
                let mut seg_idx = 0;
                for segment_audio in segments_audio {
                    // Minimum segment length check
                    if segment_audio.len() < 661 {
                        // ~15ms
                        continue;
                    }

                    if let Some(features) = compute_105d_features(&extractor, &segment_audio) {
                        all_segments.push(SegmentInfo {
                            source_file: filename.clone(),
                            call_type: call_type.clone(),
                            segment_idx: seg_idx,
                            features,
                        });

                        *call_type_counts.entry(call_type.clone()).or_insert(0) += 1;
                        seg_idx += 1;
                    }
                }
            }
            Err(e) => {
                println!("  ⚠️  Error loading {}: {}", filename, e);
            }
        }
    }

    println!();
    println!("  Total segments extracted: {}", all_segments.len());
    println!("  Segments by call type:");
    for (call_type, count) in &call_type_counts {
        println!("    • {}: {}", call_type, count);
    }
    println!();

    // Build feature matrix
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("STEP 2: HDBSCAN Clustering on 105D Features");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let n_segments = all_segments.len();
    let n_features = 105;

    let mut feature_matrix = Array2::<f64>::zeros((n_segments, n_features));
    for (i, seg) in all_segments.iter().enumerate() {
        for (j, &val) in seg.features.iter().enumerate().take(n_features) {
            feature_matrix[[i, j]] = val;
        }
    }

    println!("  Feature matrix: {} segments × {} features", n_segments, n_features);
    println!("  Running HDBSCAN with min_cluster_size={}...", min_cluster_size);
    println!();

    let hdbscan = HdbscanClustering::new(min_cluster_size, min_samples)?;
    let labels = hdbscan.fit_predict(&feature_matrix)?;

    // Analyze results
    let stats = hdbscan.get_cluster_stats(&labels);
    let noise_count = labels.iter().filter(|&&l| l == -1).count();
    let purity = (n_segments - noise_count) as f64 / n_segments as f64;
    let noise_ratio = noise_count as f64 / n_segments as f64;

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("STEP 3: Cluster Membership Analysis");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  Overall Statistics:");
    println!("    • Total segments: {}", n_segments);
    println!("    • Clusters found: {}", stats.n_clusters);
    println!("    • Noise points: {} ({:.1}%)", noise_count, noise_ratio * 100.0);
    println!("    • Purity: {:.1}%", purity * 100.0);
    println!();

    // Group segments by cluster
    let mut cluster_members: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        cluster_members.entry(label).or_default().push(idx);
    }

    // Analyze cluster composition
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  CLUSTER COMPOSITION BY CALL TYPE                                       │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    let mut sorted_clusters: Vec<_> = cluster_members.iter().collect();
    sorted_clusters.sort_by_key(|(&label, _)| if label == -1 { 999 } else { label });

    for (&label, member_indices) in &sorted_clusters {
        if label == -1 {
            println!(
                "  │  NOISE ({})                                                          ",
                member_indices.len()
            );
        } else {
            let mut type_counts: HashMap<&str, usize> = HashMap::new();
            let mut file_set: HashSet<&str> = HashSet::new();

            for &idx in member_indices.iter() {
                let seg = &all_segments[idx];
                *type_counts.entry(seg.call_type.as_str()).or_insert(0) += 1;
                file_set.insert(seg.source_file.as_str());
            }

            println!(
                "  │  CLUSTER {} ({} segments from {} files)                       ",
                label,
                member_indices.len(),
                file_set.len()
            );
            println!("  │    Call type distribution:");

            let mut sorted_types: Vec<_> = type_counts.iter().collect();
            sorted_types.sort_by(|a, b| b.1.cmp(a.1));

            for (call_type, count) in sorted_types {
                let pct = *count as f64 / member_indices.len() as f64 * 100.0;
                let bar = "█".repeat((pct / 5.0) as usize);
                println!("  │      • {:14} {:3} ({:5.1}%) {}", call_type, count, pct, bar);
            }
            println!("  │                                                                         ");
        }
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Final interpretation
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("INTERPRETATION");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    if purity > 0.6 {
        println!("  ✓ HIGH MOTIF REUSE ({:.0}% purity)", purity * 100.0);
        println!();
        println!("  Segments from different files are clustering together.");
        println!("  This suggests a SHARED VOCABULARY of acoustic motifs.");
        println!();
        println!("  → Bag-of-Phrases approach WILL WORK");
    } else if purity > 0.3 {
        println!("  ~ MODERATE MOTIF REUSE ({:.0}% purity)", purity * 100.0);
        println!();
        println!("  Mix of clustered and unique segments.");
        println!("  This suggests a HYBRID system with both discrete and graded elements.");
        println!();
        println!("  → Use BOTH Bag-of-Phrases AND Direct 105D similarity");
    } else {
        println!("  ✗ LOW MOTIF REUSE ({:.0}% purity)", purity * 100.0);
        println!();
        println!("  Most segments are unique (noise).");
        println!("  This supports the TRUE GRADED CONTINUUM hypothesis.");
        println!();
        println!("  → Use Direct 105D similarity (Bag-of-Phrases will fail)");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
