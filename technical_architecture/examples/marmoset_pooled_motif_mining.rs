//! Pooled Graded Phrase Mining - All Marmoset Audio Combined
//! ==========================================================
//!
//! This pools ALL segments from ALL recordings together to test if
//! there's a cross-recording vocabulary of motifs.

use std::fs;
use std::path::Path;
use technical_architecture::{
    FeatureMode, GradedMiningConfig, GradedPhraseMiner, ProcessingApproach,
};

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

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║     POOLED GRADED PHRASE MINING - ALL MARMOSET AUDIO                ║");
    println!("║         Testing Cross-Recording Motif Reuse                         ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Load all audio files
    let wav_dir = Path::new("test_marmoset_wav");
    let wav_files: Vec<_> = fs::read_dir(wav_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    println!("Step 1: Loading all marmoset audio files...");
    println!("─────────────────────────────────────────────────────────────────────────");

    // Load all audio and concatenate
    let mut all_audio: Vec<f32> = Vec::new();
    let sample_rate = 44100u32; // All marmoset files are 44.1kHz
    let gap_samples = (sample_rate as f32 * 0.2) as usize; // 200ms gap between files

    for wav_path in &wav_files {
        match load_wav(wav_path) {
            Ok((audio, sr)) => {
                let filename = wav_path.file_name().unwrap().to_str().unwrap();
                let duration_ms = audio.len() as f32 / sr as f32 * 1000.0;
                println!("  Loaded: {} ({:.0}ms)", filename, duration_ms);

                // Resample if needed (simple linear interpolation)
                if sr != sample_rate {
                    let ratio = sample_rate as f32 / sr as f32;
                    let new_len = (audio.len() as f32 * ratio) as usize;
                    let mut resampled = Vec::with_capacity(new_len);
                    for i in 0..new_len {
                        let src_idx = i as f32 / ratio;
                        let idx0 = src_idx as usize;
                        let idx1 = (idx0 + 1).min(audio.len() - 1);
                        let frac = src_idx - idx0 as f32;
                        resampled.push(audio[idx0] * (1.0 - frac) + audio[idx1] * frac);
                    }
                    all_audio.extend(resampled);
                } else {
                    all_audio.extend(audio);
                }

                // Add gap between files
                all_audio.extend(vec![0.0f32; gap_samples]);
            }
            Err(e) => {
                println!("  Error loading {}: {}", wav_path.display(), e);
            }
        }
    }

    let total_duration_s = all_audio.len() as f32 / sample_rate as f32;
    println!();
    println!(
        "Total pooled audio: {:.1} seconds ({} samples)",
        total_duration_s,
        all_audio.len()
    );
    println!();

    // Now run the Graded Phrase Mining on the pooled audio
    println!("Step 2: Running Graded Phrase Mining on pooled audio...");
    println!("─────────────────────────────────────────────────────────────────────────");

    let config = GradedMiningConfig {
        min_phrase_duration_ms: 30.0,
        boundary_threshold: 0.35,
        min_cluster_size: 5, // Need larger clusters for pooled data
        min_samples: 3,
        feature_mode: FeatureMode::Full105D,
        min_segment_samples: 1323, // ~30ms at 44.1kHz
    };

    println!("Configuration:");
    println!("  • Feature Mode: {:?}", config.feature_mode);
    println!("  • Min Cluster Size: {}", config.min_cluster_size);
    println!("  • Min Samples: {}", config.min_samples);
    println!();

    let mut miner = GradedPhraseMiner::new(config);

    match miner.analyze(&all_audio, sample_rate) {
        Ok(report) => {
            println!();
            println!("═══════════════════════════════════════════════════════════════════════");
            println!("                    POOLED ANALYSIS RESULTS                             ");
            println!("═══════════════════════════════════════════════════════════════════════");
            println!();
            println!("  Total Segments Extracted: {}", report.total_segments);
            println!("  Clusters Found: {}", report.num_clusters);
            println!("  Noise Points: {}", report.noise_count);
            println!();

            println!("  ┌─────────────────────────────────────────────────────────────────┐");
            println!("  │  MOTIF REUSE METRICS (Cross-Recording)                          │");
            println!("  ├─────────────────────────────────────────────────────────────────┤");
            println!(
                "  │  Purity:             {:>6.1}%                                    │",
                report.purity * 100.0
            );
            println!(
                "  │  Noise Ratio:        {:>6.1}%                                    │",
                report.noise_ratio * 100.0
            );
            println!(
                "  │  Avg Cohesion:       {:>6.3}                                    │",
                report.avg_cohesion
            );
            println!("  └─────────────────────────────────────────────────────────────────┘");
            println!();

            // Interpret results
            println!("  ┌─────────────────────────────────────────────────────────────────┐");
            println!("  │  INTERPRETATION                                                  │");
            println!("  ├─────────────────────────────────────────────────────────────────┤");
            println!(
                "  │  {}",
                report.interpretation.lines().next().unwrap_or("")
            );
            println!("  │                                                                 │");

            if report.purity > 0.6 {
                println!("  │  Segments from DIFFERENT recordings are clustering together!    │");
                println!("  │  This suggests a SHARED vocabulary of acoustic motifs.          │");
            } else if report.purity > 0.3 {
                println!("  │  Some cross-recording clustering, but also unique segments.     │");
                println!("  │  Marmosets use BOTH discrete motifs AND graded transitions.     │");
            } else {
                println!("  │  Most segments are UNIQUE - no shared vocabulary detected.      │");
                println!("  │  Marmosets use a TRUE GRADED CONTINUUM (analog slider).         │");
            }
            println!("  └─────────────────────────────────────────────────────────────────┘");
            println!();

            println!("  ┌─────────────────────────────────────────────────────────────────┐");
            println!(
                "  │  RECOMMENDED APPROACH: {:?}",
                format!("{:?}", report.recommended_approach).pad_to_width(27)
            );
            println!("  └─────────────────────────────────────────────────────────────────┘");
            println!();

            // Show cluster details
            if !report.cluster_stats.is_empty() {
                println!("  ┌─────────────────────────────────────────────────────────────────┐");
                println!("  │  TOP CLUSTERS                                                    │");
                println!("  ├─────────────────────────────────────────────────────────────────┤");
                for (i, cluster) in report.cluster_stats.iter().take(5).enumerate() {
                    println!(
                        "  │  Cluster {}: {} segments, cohesion: {:.3}              │",
                        i + 1,
                        cluster.size,
                        cluster.avg_cohesion
                    );
                }
                println!("  └─────────────────────────────────────────────────────────────────┘");
            }
        }
        Err(e) => {
            println!("Analysis error: {}", e);
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    Ok(())
}

trait PadToWidth {
    fn pad_to_width(&self, width: usize) -> String;
}

impl PadToWidth for str {
    fn pad_to_width(&self, width: usize) -> String {
        format!("{:width$}", self, width = width)
    }
}
