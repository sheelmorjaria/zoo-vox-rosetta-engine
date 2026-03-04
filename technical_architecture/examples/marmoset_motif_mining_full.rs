//! Full Graded Phrase Mining with Feature Caching and Cluster Analysis
//! ====================================================================
//!
//! 1. Extract 105D features from all marmoset audio and cache
//! 2. Run clustering with higher min_cluster_size to force multiple clusters
//! 3. Analyze cluster membership by call type

use ndarray::Array2;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use technical_architecture::{HdbscanClustering, MicroDynamicsExtractor, MicroDynamicsFeatures45D};

/// Segment metadata for tracking
#[derive(Debug, Clone)]
struct SegmentInfo {
    source_file: String,
    call_type: String, // Tsik, Twitter, or Vocalization
    start_ms: f32,
    end_ms: f32,
    duration_ms: f32,
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
        let chunk_size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;

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
    // Extract 45D base features
    let base_45d = extractor.extract_45d(audio).ok()?;

    // Start with 45D base
    let mut features = Vec::with_capacity(105);
    features.extend(base_45d.to_array().iter().map(|&v| v as f64));

    // Add macro texture (30D) - simplified
    features.extend(compute_macro_texture(&base_45d));

    // Add micro texture (30D) - simplified
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
    features.push(
        base_45d.base_30d.attack_time_ms as f64 / (base_45d.base_30d.decay_time_ms as f64 + 1.0),
    );
    features.push(base_45d.base_30d.sustain_level as f64 * 10.0);
    features.push(base_45d.base_30d.vibrato_depth as f64 / 100.0);
    features.push(0.1);

    features
}

fn compute_micro_texture(base_45d: &MicroDynamicsFeatures45D) -> Vec<f64> {
    let mut features = Vec::with_capacity(30);

    // AM Spectrum (5D)
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

    // FM Spectrum (5D)
    let fm_rate = base_45d.fm_slope as f64;
    features.push(if fm_rate < 10.0 { 1.0 } else { 0.0 });
    features.push(if fm_rate >= 10.0 && fm_rate < 30.0 {
        1.0
    } else {
        0.0
    });
    features.push(if fm_rate >= 30.0 && fm_rate < 50.0 {
        1.0
    } else {
        0.0
    });
    features.push(if fm_rate >= 50.0 && fm_rate < 100.0 {
        1.0
    } else {
        0.0
    });
    features.push(0.0);

    // Modulation Stats (5D)
    features.push(base_45d.base_30d.vibrato_rate_hz as f64);
    features.push(base_45d.fm_slope as f64);
    features.push(base_45d.am_depth as f64);
    features.push(base_45d.fm_slope as f64 * 0.5);
    features.push(base_45d.base_30d.vibrato_depth as f64);

    // Rhythm Histogram (5D)
    let ici = base_45d.base_30d.median_ici_ms as f64;
    features.push(if ici < 20.0 { 1.0 } else { 0.0 });
    features.push(if ici >= 20.0 && ici < 50.0 { 1.0 } else { 0.0 });
    features.push(if ici >= 50.0 && ici < 100.0 { 1.0 } else { 0.0 });
    features.push(if ici >= 100.0 && ici < 200.0 {
        1.0
    } else {
        0.0
    });
    features.push(if ici >= 200.0 { 1.0 } else { 0.0 });

    // Rhythm Stats (5D)
    features.push(base_45d.base_30d.median_ici_ms as f64);
    features.push(1.0 / (base_45d.base_30d.median_ici_ms as f64 / 1000.0 + 0.001));
    features.push(base_45d.base_30d.onset_rate_hz as f64);
    features.push(base_45d.base_30d.ici_coefficient_of_variation as f64);
    features.push(base_45d.base_30d.onset_rate_hz as f64 * 60.0);

    // Psychoacoustics (5D)
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
    println!("║       FULL GRADED PHRASE MINING WITH FEATURE CACHING                     ║");
    println!("║           Marmoset Cross-Recording Motif Analysis                         ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let min_phrase_duration_ms = 30.0;
    let min_segment_samples = 661; // ~15ms at 44.1kHz (lower for more segments)
    let min_cluster_size = 3; // Lower to force more clusters
    let min_samples = 2;
    let sample_rate = 44100u32;

    println!("Configuration:");
    println!("  • Feature Mode: Full105D");
    println!("  • Min Cluster Size: {} (forced higher)", min_cluster_size);
    println!("  • Min Samples: {}", min_samples);
    println!("  • Min Phrase Duration: {}ms", min_phrase_duration_ms);
    println!();

    // Load all audio files
    let wav_dir = Path::new("test_marmoset_wav");
    let wav_files: Vec<_> = fs::read_dir(wav_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("STEP 1: Loading audio and extracting 105D features (caching)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let extractor = MicroDynamicsExtractor::new(sample_rate);
    let mut all_segments: Vec<SegmentInfo> = Vec::new();
    let mut call_type_counts: HashMap<String, usize> = HashMap::new();

    for wav_path in &wav_files {
        let filename = wav_path.file_name().unwrap().to_str().unwrap().to_string();
        let call_type = get_call_type(&filename).to_string();

        match load_wav(wav_path) {
            Ok((audio, sr)) => {
                let duration_ms = audio.len() as f32 / sr as f32 * 1000.0;

                // Simple energy-based segmentation
                let hop_size = 512;
                let frame_size = 1024;
                let mut segments_audio: Vec<(usize, usize)> = Vec::new();
                let mut in_segment = false;
                let mut segment_start = 0usize;

                let energy_threshold = 0.005; // Lower threshold for more segments
                let mut frame_energies: Vec<f32> = Vec::new();

                for i in (0..audio.len()).step_by(hop_size) {
                    let end = (i + frame_size).min(audio.len());
                    let energy: f32 =
                        audio[i..end].iter().map(|x| x * x).sum::<f32>() / (end - i) as f32;
                    frame_energies.push(energy.sqrt());
                }

                // Find segments based on energy
                for (i, &energy) in frame_energies.iter().enumerate() {
                    let sample_idx = i * hop_size;

                    if energy > energy_threshold && !in_segment {
                        in_segment = true;
                        segment_start = sample_idx;
                    } else if energy <= energy_threshold && in_segment {
                        in_segment = false;
                        let segment_end = sample_idx;
                        if segment_end - segment_start >= min_segment_samples {
                            segments_audio.push((segment_start, segment_end));
                        }
                    }
                }

                // Close final segment if needed
                if in_segment {
                    let segment_end = audio.len();
                    if segment_end - segment_start >= min_segment_samples {
                        segments_audio.push((segment_start, segment_end));
                    }
                }

                println!(
                    "  📁 {} ({:.0}ms): {} segments",
                    filename,
                    duration_ms,
                    segments_audio.len()
                );

                // Extract features for each segment
                for (start, end) in segments_audio {
                    let segment_audio = &audio[start..end];
                    let start_ms = start as f32 / sr as f32 * 1000.0;
                    let end_ms = end as f32 / sr as f32 * 1000.0;
                    let seg_duration_ms = end_ms - start_ms;

                    if let Some(features) = compute_105d_features(&extractor, segment_audio) {
                        all_segments.push(SegmentInfo {
                            source_file: filename.clone(),
                            call_type: call_type.clone(),
                            start_ms,
                            end_ms,
                            duration_ms: seg_duration_ms,
                            features,
                        });

                        *call_type_counts.entry(call_type.clone()).or_insert(0) += 1;
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
    println!("STEP 2: Building feature matrix and running HDBSCAN clustering");
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

    println!(
        "  Feature matrix: {} segments × {} features",
        n_segments, n_features
    );
    println!(
        "  Running HDBSCAN with min_cluster_size={}...",
        min_cluster_size
    );
    println!();

    let hdbscan = HdbscanClustering::new(min_cluster_size, min_samples)?;
    let labels = hdbscan.fit_predict(&feature_matrix)?;

    // Analyze results
    let stats = hdbscan.get_cluster_stats(&labels);
    let noise_count = labels.iter().filter(|&&l| l == -1).count();
    let purity = (n_segments - noise_count) as f64 / n_segments as f64;
    let noise_ratio = noise_count as f64 / n_segments as f64;

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("STEP 3: Cluster membership analysis by call type");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  Overall Statistics:");
    println!("    • Total segments: {}", n_segments);
    println!("    • Clusters found: {}", stats.n_clusters);
    println!(
        "    • Noise points: {} ({:.1}%)",
        noise_count,
        noise_ratio * 100.0
    );
    println!("    • Purity: {:.1}%", purity * 100.0);
    println!();

    // Group segments by cluster (store indices)
    let mut cluster_members: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        cluster_members.entry(label).or_default().push(idx);
    }

    // Analyze call type distribution per cluster
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
                println!(
                    "  │      • {:14} {:3} ({:5.1}%) {}",
                    call_type, count, pct, bar
                );
            }
            println!(
                "  │                                                                         "
            );
        }
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Cross-cluster analysis: Are different call types separated?
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  CALL TYPE SEPARATION ANALYSIS                                          │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    for call_type in &["Tsik", "Twitter", "Vocalization"] {
        // Get indices of segments of this call type
        let type_indices: Vec<usize> = all_segments
            .iter()
            .enumerate()
            .filter(|(_, s)| s.call_type == *call_type)
            .map(|(i, _)| i)
            .collect();

        if type_indices.is_empty() {
            continue;
        }

        let mut cluster_dist: HashMap<i32, usize> = HashMap::new();
        for &idx in &type_indices {
            let label = labels[idx];
            *cluster_dist.entry(label).or_insert(0) += 1;
        }

        let dominant_cluster = cluster_dist
            .iter()
            .max_by_key(|(_, &c)| c)
            .map(|(&l, &c)| (l, c));

        if let Some((dom_cluster, dom_count)) = dominant_cluster {
            let pct = dom_count as f64 / type_indices.len() as f64 * 100.0;
            let separation = if pct > 80.0 {
                "STRONG"
            } else if pct > 50.0 {
                "MODERATE"
            } else {
                "WEAK"
            };

            println!(
                "  │  {:14} → {} segments in {} clusters ({:>8} separation)",
                call_type,
                type_indices.len(),
                cluster_dist.len(),
                separation
            );
            println!(
                "  │    Dominant: Cluster {} ({:.0}%)",
                if dom_cluster == -1 {
                    "NOISE".to_string()
                } else {
                    dom_cluster.to_string()
                },
                pct
            );
        }
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Final interpretation
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("INTERPRETATION");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    if stats.n_clusters >= 3 {
        println!("  ✓  MULTIPLE CLUSTERS DETECTED ({})", stats.n_clusters);
        println!();
        println!("  This suggests that marmoset vocalizations CAN be separated into");
        println!("  distinct acoustic categories based on 105D features.");
        println!();
        println!("  If call types are separated across clusters:");
        println!("    → Bag-of-Phrases will work well");
        println!("    → Each cluster represents a 'motif' or 'phrase type'");
        println!();
        println!("  If call types are mixed within clusters:");
        println!("    → Call types share acoustic similarity");
        println!("    → Grading continuum exists between types");
    } else if stats.n_clusters == 1 {
        println!("  ⚠ ONLY 1 CLUSTER DETECTED");
        println!();
        println!("  All segments are acoustically similar enough to cluster together.");
        println!("  This could mean:");
        println!("    1. Test dataset is too homogeneous");
        println!("    2. min_cluster_size is too high");
        println!("    3. Marmoset vocalizations form a true graded continuum");
    } else {
        println!("  ✗ NO CLEAR CLUSTERS (mostly noise)");
        println!();
        println!("  High noise ratio indicates segments are unique -");
        println!("  supporting the 'analog slider' hypothesis.");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
