//! Graded Phrase Mining Integration Test on Marmoset Audio
//! ========================================================
//!
//! Tests the "Hidden Discrete Motifs" hypothesis on real marmoset vocalizations
//! using the full 105D feature pipeline.

use std::fs;
use std::path::Path;
use technical_architecture::{
    FeatureMode, GradedMiningConfig, GradedPhraseMiner, MotifReport, ProcessingApproach,
};

/// Load WAV file and return samples as f32
fn load_wav(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let bytes = fs::read(path)?;

    // Parse WAV header
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
    println!("║     GRADED PHRASE MINING - MARMOSET VOCALIZATION ANALYSIS           ║");
    println!("║            Testing the 'Hidden Discrete Motifs' Hypothesis          ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let config = GradedMiningConfig {
        min_phrase_duration_ms: 30.0, // Detect shorter phrases
        boundary_threshold: 0.35,     // Sensitive boundary detection
        min_cluster_size: 3,          // Smaller clusters for limited data
        min_samples: 2,
        feature_mode: FeatureMode::Full105D, // 105D for maximum discriminative power
        min_segment_samples: 2205,           // ~50ms at 44.1kHz
    };

    println!("Configuration:");
    println!("  • Feature Mode: {:?}", config.feature_mode);
    println!(
        "  • Min Phrase Duration: {}ms",
        config.min_phrase_duration_ms
    );
    println!("  • Boundary Threshold: {}", config.boundary_threshold);
    println!("  • Min Cluster Size: {}", config.min_cluster_size);
    println!();

    let mut miner = GradedPhraseMiner::new(config);

    // Find marmoset WAV files
    let wav_dir = Path::new("test_marmoset_wav");
    let wav_files: Vec<_> = fs::read_dir(wav_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    println!("Found {} marmoset audio files", wav_files.len());
    println!();
    println!("Processing files...");
    println!("─────────────────────────────────────────────────────────────────────────");

    let mut all_reports: Vec<(String, MotifReport)> = Vec::new();

    for wav_path in &wav_files {
        let filename = wav_path.file_name().unwrap().to_str().unwrap();

        match load_wav(wav_path) {
            Ok((audio, sample_rate)) => {
                let duration_ms = audio.len() as f32 / sample_rate as f32 * 1000.0;

                match miner.analyze(&audio, sample_rate) {
                    Ok(report) => {
                        println!("  📁 {}", filename);
                        println!(
                            "     Duration: {:.0}ms | Segments: {} | Clusters: {}",
                            duration_ms, report.total_segments, report.num_clusters
                        );
                        println!(
                            "     Purity: {:.1}% | Noise: {:.1}%",
                            report.purity * 100.0,
                            report.noise_ratio * 100.0
                        );

                        all_reports.push((filename.to_string(), report));
                        miner.reset();
                    }
                    Err(e) => {
                        println!("  ⚠️  {}: Analysis error - {}", filename, e);
                    }
                }
            }
            Err(e) => {
                println!("  ⚠️  {}: Load error - {}", filename, e);
            }
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("                         AGGREGATE RESULTS                              ");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    if !all_reports.is_empty() {
        let total_segments: usize = all_reports.iter().map(|(_, r)| r.total_segments).sum();
        let total_clusters: usize = all_reports.iter().map(|(_, r)| r.num_clusters).sum();
        let avg_purity: f64 =
            all_reports.iter().map(|(_, r)| r.purity).sum::<f64>() / all_reports.len() as f64;
        let avg_noise: f64 =
            all_reports.iter().map(|(_, r)| r.noise_ratio).sum::<f64>() / all_reports.len() as f64;

        println!("  Files Analyzed: {}", all_reports.len());
        println!("  Total Segments Extracted: {}", total_segments);
        println!("  Total Clusters Found: {}", total_clusters);
        println!();
        println!("  ┌─────────────────────────────────────────────────────────────────┐");
        println!("  │  MOTIF REUSE METRICS                                            │");
        println!("  ├─────────────────────────────────────────────────────────────────┤");
        println!(
            "  │  Average Purity:     {:>6.1}%                                    │",
            avg_purity * 100.0
        );
        println!(
            "  │  Average Noise:      {:>6.1}%                                    │",
            avg_noise * 100.0
        );
        println!("  └─────────────────────────────────────────────────────────────────┘");
        println!();

        // Interpret results
        let interpretation = if avg_purity > 0.6 {
            "HIGH MOTIF REUSE - Hidden vocabulary detected"
        } else if avg_purity > 0.3 {
            "HYBRID SYSTEM - Mix of discrete motifs and graded transitions"
        } else {
            "TRUE GRADED CONTINUUM - Analog slider without discrete units"
        };

        println!("  ┌─────────────────────────────────────────────────────────────────┐");
        println!("  │  INTERPRETATION                                                  │");
        println!("  ├─────────────────────────────────────────────────────────────────┤");
        println!("  │  {}", interpretation);
        if avg_purity > 0.3 && avg_purity <= 0.6 {
            println!("  │                                                                 │");
            println!("  │  Marmosets use BOTH:                                            │");
            println!("  │  • Discrete alarm chirps (reusable motifs)                      │");
            println!("  │  • Graded Phee transitions (continuous modulation)              │");
        }
        println!("  └─────────────────────────────────────────────────────────────────┘");
        println!();

        // Recommendation
        let recommended = if avg_purity > 0.6 {
            ProcessingApproach::BagOfPhrases
        } else if avg_purity > 0.3 {
            ProcessingApproach::HybridDiscreteGraded
        } else {
            ProcessingApproach::Direct105D
        };

        println!("  ┌─────────────────────────────────────────────────────────────────┐");
        println!("  │  RECOMMENDED PROCESSING APPROACH                                │");
        println!("  ├─────────────────────────────────────────────────────────────────┤");
        match recommended {
            ProcessingApproach::BagOfPhrases => {
                println!("  │  BAG-OF-PHRASES                                                  │");
                println!("  │  Discrete vocabulary found - phrase-based classification works  │");
            }
            ProcessingApproach::HybridDiscreteGraded => {
                println!("  │  HYBRID DISCRETE + GRADED                                        │");
                println!("  │  Use BOTH Bag-of-Phrases AND Direct 105D similarity             │");
            }
            ProcessingApproach::Direct105D => {
                println!("  │  DIRECT 105D SIMILARITY                                          │");
                println!("  │  No discrete vocabulary - use continuous feature matching       │");
            }
            ProcessingApproach::InsufficientData => {
                println!("  │  INSUFFICIENT DATA                                               │");
            }
        }
        println!("  └─────────────────────────────────────────────────────────────────┘");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!();

    Ok(())
}
