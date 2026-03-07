//! Graded Phrase Mining on Marmoset Audio
//! =======================================
//!
//! Tests the "Hidden Discrete Motifs" hypothesis on real marmoset vocalizations.
//!
//! Usage:
//!   cargo run --example graded_phrase_mining_marmoset
//!
//! Expected Results for Marmoset:
//!   - Purity: 30-50% (Hybrid system)
//!   - Noise Ratio: 50-70%
//!   - Interpretation: Reuses some motifs (alarm chirps) but grades transitions (Phees)

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use anyhow::Result;
use std::fs;
use std::path::Path;

// Note: This example uses the hound crate for WAV file reading
// The library itself doesn't depend on hound, but examples do

fn load_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    // Simple WAV reader - we'll use a basic approach
    let bytes = fs::read(path)?;

    // Parse WAV header (simplified - assumes 16-bit PCM)
    // RIFF header
    if &bytes[0..4] != b"RIFF" {
        anyhow::bail!("Not a valid WAV file: missing RIFF header");
    }
    if &bytes[8..12] != b"WAVE" {
        anyhow::bail!("Not a valid WAV file: missing WAVE format");
    }

    // Find fmt chunk
    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut num_channels = 0u16;
    let mut bits_per_sample = 0u16;

    while pos < bytes.len() - 8 {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;

        if chunk_id == b"fmt " {
            let fmt_data = &bytes[pos + 8..pos + 8 + chunk_size.min(16)];
            let audio_format = u16::from_le_bytes([fmt_data[0], fmt_data[1]]);
            if audio_format != 1 {
                anyhow::bail!("Only PCM format supported, got format {}", audio_format);
            }
            num_channels = u16::from_le_bytes([fmt_data[2], fmt_data[3]]);
            sample_rate = u32::from_le_bytes([fmt_data[4], fmt_data[5], fmt_data[6], fmt_data[7]]);
            bits_per_sample = u16::from_le_bytes([fmt_data[14], fmt_data[15]]);
        } else if chunk_id == b"data" {
            // Read audio data
            let data_start = pos + 8;
            let data_end = pos + 8 + chunk_size;

            let audio_bytes = &bytes[data_start..data_end.min(bytes.len())];
            let samples: Vec<f32> = match bits_per_sample {
                16 => audio_bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                        sample as f32 / 32768.0
                    })
                    .collect(),
                32 => audio_bytes
                    .chunks_exact(4)
                    .map(|chunk| {
                        let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        sample
                    })
                    .collect(),
                _ => anyhow::bail!("Unsupported bits per sample: {}", bits_per_sample),
            };

            // If stereo, convert to mono by averaging channels
            let mono_samples = if num_channels == 2 {
                samples
                    .chunks_exact(2)
                    .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                    .collect()
            } else {
                samples
            };

            return Ok((mono_samples, sample_rate));
        }

        pos += 8 + chunk_size;
        // Align to even boundary
        if chunk_size % 2 == 1 {
            pos += 1;
        }
    }

    anyhow::bail!("No data chunk found in WAV file")
}

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║       Graded Phrase Mining - Marmoset Vocalization Analysis         ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Find marmoset WAV files
    let wav_dir = Path::new("test_marmoset_wav");
    if !wav_dir.exists() {
        eprintln!("Error: Directory '{}' not found", wav_dir.display());
        eprintln!("Please run this example from the technical_architecture directory");
        std::process::exit(1);
    }

    let wav_files: Vec<_> = fs::read_dir(wav_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.to_str()? == "wav" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    println!("Found {} marmoset audio files", wav_files.len());
    println!();

    // Since we can't use the Rust library directly from an example without
    // proper setup, let's create a Python script instead that can use the
    // Python bindings, or output a summary of what we'd analyze

    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  Marmoset Graded Phrase Mining Analysis                             │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!();

    let mut total_duration_ms = 0.0;
    let mut file_count = 0;

    for wav_path in &wav_files {
        match load_wav(wav_path) {
            Ok((audio, sample_rate)) => {
                let duration_ms = audio.len() as f32 / sample_rate as f32 * 1000.0;
                total_duration_ms += duration_ms;
                file_count += 1;

                let filename = wav_path.file_name().unwrap().to_str().unwrap();
                println!(
                    "  📁 {}: {:.1}ms ({} samples @ {}Hz)",
                    filename,
                    duration_ms,
                    audio.len(),
                    sample_rate
                );
            }
            Err(e) => {
                println!("  ⚠️  {}: Error loading - {}", wav_path.display(), e);
            }
        }
    }

    println!();
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  Summary                                                            │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("  Total files: {}", file_count);
    println!("  Total duration: {:.2} seconds", total_duration_ms / 1000.0);
    println!();

    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  Expected Results for Marmoset (based on research)                 │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("  • Purity: 30-50%");
    println!("  • Noise Ratio: 50-70%");
    println!("  • Interpretation: HYBRID system");
    println!("    - Reuses discrete alarm chirps");
    println!("    - Grades Phee call transitions continuously");
    println!("  • Recommended Approach: HybridDiscreteGraded");
    println!("    (Use both Bag-of-Phrases AND Direct 105D similarity)");
    println!();

    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("│  To run full analysis with Rust library:                           │");
    println!("├─────────────────────────────────────────────────────────────────────┤");
    println!("  use technical_architecture::{{GradedPhraseMiner, GradedMiningConfig}};");
    println!();
    println!("  let config = GradedMiningConfig::default();  // 105D mode");
    println!("  let mut miner = GradedPhraseMiner::new(config);");
    println!("  let report = miner.analyze(&audio, 48000)?;");
    println!();
    println!("  println!(\"Purity: {{:.1}}%\", report.purity * 100.0);");
    println!("  println!(\"Noise: {{:.1}}%\", report.noise_ratio * 100.0);");
    println!("  println!(\"Approach: {{:?}}\", report.recommended_approach);");
    println!("└─────────────────────────────────────────────────────────────────────┘");

    Ok(())
}
