//! Bird Syllable-Level NBD + N-gram Mining
//! =========================================
//!
//! Segments each vocalization file into SYLLABLES using NBD,
//! then performs N-gram mining on syllable sequences.
//!
//! This solves the "no sequence data" problem by treating
//! each file as a sequence of syllables.
//!
//! Datasets analyzed:
//! - 11905533 (Adult/Chick vocalizations): 3,433 files, 216KB avg
//! - zebra_finch: 3,405 files, 216KB avg
//! - bird_songs: 687 files, 391KB avg

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
struct SyllableSegment {
    source_file: String,
    syllable_idx: usize,
    state_id: u32,
    start_ms: f32,
    end_ms: f32,
    duration_ms: f32,
}

#[derive(Debug, Clone, Serialize)]
struct DatasetAnalysis {
    dataset: String,
    total_files: usize,
    total_syllables: usize,
    avg_syllables_per_file: f64,
    unique_states: usize,
    recurrence_rate: f64,
    bigram_total: usize,
    bigram_unique: usize,
    bigram_reuse_rate: f64,
    trigram_reuse_rate: f64,
    top_bigrams: Vec<(String, usize)>,
    has_discrete_syntax: bool,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     BIRD SYLLABLE-LEVEL NBD + SYNTAX MINING                               ║");
    println!("║     Segments files into syllables → N-gram mining on syllable sequences    ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let datasets = vec![
        (
            "11905533 (Bird)",
            dirs::home_dir().unwrap().join("birdsong_analysis/data/11905533"),
        ),
        (
            "zebra_finch",
            dirs::home_dir().unwrap().join("birdsong_analysis/data/zebra_finch"),
        ),
        (
            "Whistle_Signals (Dolphin)",
            dirs::home_dir().unwrap().join("birdsong_analysis/data/Whistle_Signals"),
        ),
        (
            "Dominica (Sperm Whale)",
            dirs::home_dir()
                .unwrap()
                .join("birdsong_analysis/data/Dominica_dataset/Signal_parts"),
        ),
    ];

    let output_dir = Path::new("bird_syllable_results");
    fs::create_dir_all(output_dir)?;

    let mut all_results = Vec::new();

    for (dataset_name, dataset_path) in &datasets {
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!("DATASET: {}", dataset_name);
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!();

        // Find audio files
        let audio_files: Vec<PathBuf> = fs::read_dir(dataset_path)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
            .map(|e| e.path())
            .take(300) // Limit for performance
            .collect();

        if audio_files.is_empty() {
            // Try subdirectories
            let mut sub_files: Vec<PathBuf> = Vec::new();

            if let Ok(entries) = fs::read_dir(dataset_path) {
                for entry in entries.flatten() {
                    let sub_path = entry.path();
                    if sub_path.is_dir() {
                        if let Ok(sub_entries) = fs::read_dir(&sub_path) {
                            for sub_entry in sub_entries.flatten() {
                                if sub_entry.path().extension().map(|x| x == "wav").unwrap_or(false) {
                                    sub_files.push(sub_entry.path());
                                    if sub_files.len() >= 300 {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if sub_files.len() >= 300 {
                        break;
                    }
                }
            }

            if sub_files.is_empty() {
                println!("  No audio files found");
                continue;
            }

            println!("  Found {} audio files (subdirectories)", sub_files.len());
            analyze_dataset(dataset_name, sub_files, &mut all_results)?;
        } else {
            println!("  Found {} audio files", audio_files.len());
            analyze_dataset(dataset_name, audio_files, &mut all_results)?;
        }
    }

    // Comparison table
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("CROSS-DATASET COMPARISON");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  Dataset         │ Files │ Syllables │ Syl/File │ Bigram Reuse │ Syntax│");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    all_results.sort_by(|a, b| b.bigram_reuse_rate.partial_cmp(&a.bigram_reuse_rate).unwrap());

    for r in &all_results {
        let syntax = if r.has_discrete_syntax {
            "✓ DISCR"
        } else {
            "✗ GRADE"
        };
        println!(
            "  │  {:15} │ {:5} │ {:9} │ {:8.1} │     {:5.1}%    │ {} │",
            r.dataset,
            r.total_files,
            r.total_syllables,
            r.avg_syllables_per_file,
            r.bigram_reuse_rate * 100.0,
            syntax
        );
    }

    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Reference
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  REFERENCE (Egyptian Fruit Bat)                                        │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │  91,066 files → 1,567,640 syllables (17.2/file)                         │");
    println!("  │  Bigram reuse: 87.9% → DISCRETE SYNTAX                                 │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Save
    let json = serde_json::to_string_pretty(&all_results)?;
    fs::write(output_dir.join("bird_syllable_analysis.json"), json)?;

    println!(
        "  Results saved to: {}/bird_syllable_analysis.json",
        output_dir.display()
    );
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}

fn analyze_dataset(
    dataset_name: &str,
    audio_files: Vec<PathBuf>,
    results: &mut Vec<DatasetAnalysis>,
) -> anyhow::Result<()> {
    println!("  Extracting syllables with NBD...");

    // Process all files - each file becomes a sequence of syllables
    let file_syllables: Vec<Vec<SyllableSegment>> = audio_files
        .par_iter()
        .map(|path| extract_syllables(path).unwrap_or_default())
        .collect();

    let total_files = file_syllables.len();
    let total_syllables: usize = file_syllables.iter().map(|v| v.len()).sum();
    let avg_syllables = total_syllables as f64 / total_files as f64;

    println!("  Extracted {} syllables from {} files", total_syllables, total_files);
    println!("  Average {:.1} syllables per file", avg_syllables);
    println!();

    if avg_syllables < 3.0 {
        println!("  ⚠ Too few syllables per file for N-gram mining");
        return Ok(());
    }

    // Count states and recurrences
    let all_syllables: Vec<&SyllableSegment> = file_syllables.iter().flatten().collect();
    let unique_states: std::collections::HashSet<u32> = all_syllables.iter().map(|s| s.state_id).collect();
    let recurrence_rate = 1.0 - (unique_states.len() as f64 / all_syllables.len() as f64);

    println!("  Unique states: {}", unique_states.len());
    println!("  Recurrence rate: {:.1}%", recurrence_rate * 100.0);
    println!();

    // N-gram mining on syllable sequences
    println!("  Mining N-grams on syllable sequences...");

    let mut bigram_counts: HashMap<(u32, u32), usize> = HashMap::new();
    let mut trigram_counts: HashMap<(u32, u32, u32), usize> = HashMap::new();
    let mut bigram_total = 0usize;
    let mut trigram_total = 0usize;

    for syllables in &file_syllables {
        if syllables.len() < 2 {
            continue;
        }

        let seq: Vec<u32> = syllables.iter().map(|s| s.state_id).collect();

        // Bigrams
        for i in 0..seq.len() - 1 {
            *bigram_counts.entry((seq[i], seq[i + 1])).or_insert(0) += 1;
            bigram_total += 1;
        }

        // Trigrams
        if seq.len() >= 3 {
            for i in 0..seq.len() - 2 {
                *trigram_counts.entry((seq[i], seq[i + 1], seq[i + 2])).or_insert(0) += 1;
                trigram_total += 1;
            }
        }
    }

    let bigram_unique = bigram_counts.len();
    let trigram_unique = trigram_counts.len();

    let bigram_reuse_rate = if bigram_total > 0 {
        1.0 - (bigram_unique as f64 / bigram_total as f64)
    } else {
        0.0
    };

    let trigram_reuse_rate = if trigram_total > 0 {
        1.0 - (trigram_unique as f64 / trigram_total as f64)
    } else {
        0.0
    };

    println!(
        "  Bigrams: {} total, {} unique, {:.1}% reuse",
        bigram_total,
        bigram_unique,
        bigram_reuse_rate * 100.0
    );
    println!(
        "  Trigrams: {} total, {} unique, {:.1}% reuse",
        trigram_total,
        trigram_unique,
        trigram_reuse_rate * 100.0
    );
    println!();

    // Top bigrams
    let mut top_bigrams: Vec<_> = bigram_counts.into_iter().collect();
    top_bigrams.sort_by(|a, b| b.1.cmp(&a.1));

    println!("  Top bigrams:");
    for ((a, b), count) in top_bigrams.iter().take(5) {
        println!("    {} → {} : {} occurrences", a % 10, b % 10, count);
    }
    println!();

    let has_discrete_syntax = bigram_reuse_rate > 0.5;

    println!(
        "  Classification: {}",
        if has_discrete_syntax {
            "✓ DISCRETE SYNTAX"
        } else {
            "✗ GRADED SYNTAX"
        }
    );
    println!();

    let top_bigrams_str: Vec<(String, usize)> = top_bigrams
        .into_iter()
        .take(5)
        .map(|((a, b), count)| (format!("{}→{}", a % 10, b % 10), count))
        .collect();

    results.push(DatasetAnalysis {
        dataset: dataset_name.to_string(),
        total_files,
        total_syllables,
        avg_syllables_per_file: avg_syllables,
        unique_states: unique_states.len(),
        recurrence_rate,
        bigram_total,
        bigram_unique,
        bigram_reuse_rate,
        trigram_reuse_rate,
        top_bigrams: top_bigrams_str,
        has_discrete_syntax,
    });

    Ok(())
}

fn extract_syllables(path: &Path) -> anyhow::Result<Vec<SyllableSegment>> {
    let (audio, sr) = load_wav_with_sr(path)?;
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    let window = (sr as f32 * 0.01) as usize; // 10ms windows
    let min_syllable = (sr as f32 * 0.003) as usize; // 3ms minimum

    // Adaptive threshold based on signal energy
    let max_amp = audio.iter().cloned().fold(0.0f32, f32::max);
    let mean_energy: f32 = audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32;

    // Use lower threshold for quiet recordings
    let threshold = (max_amp * 0.05).max(0.001).min(mean_energy * 3.0);

    let mut syllables = Vec::new();
    let mut in_syllable = false;
    let mut syllable_start = 0usize;
    let mut syllable_count = 0usize;

    for i in 0..audio.len() / window {
        let start = i * window;
        let end = (start + window).min(audio.len());

        let energy: f32 = audio[start..end].iter().map(|x| x * x).sum::<f32>() / window as f32;

        if energy > threshold && !in_syllable {
            in_syllable = true;
            syllable_start = start;
        } else if energy <= threshold && in_syllable {
            in_syllable = false;
            let syllable_end = i * window;

            if syllable_end - syllable_start >= min_syllable {
                let segment_audio = &audio[syllable_start..syllable_end];
                let state_id = compute_state(segment_audio);

                syllables.push(SyllableSegment {
                    source_file: path.file_name().unwrap().to_str().unwrap().to_string(),
                    syllable_idx: syllable_count,
                    state_id,
                    start_ms: syllable_start as f32 / sr as f32 * 1000.0,
                    end_ms: syllable_end as f32 / sr as f32 * 1000.0,
                    duration_ms: (syllable_end - syllable_start) as f32 / sr as f32 * 1000.0,
                });

                syllable_count += 1;
            }
        }
    }

    // Handle last syllable
    if in_syllable && audio.len() - syllable_start >= min_syllable {
        let segment_audio = &audio[syllable_start..];
        let state_id = compute_state(segment_audio);

        syllables.push(SyllableSegment {
            source_file: path.file_name().unwrap().to_str().unwrap().to_string(),
            syllable_idx: syllable_count,
            state_id,
            start_ms: syllable_start as f32 / sr as f32 * 1000.0,
            end_ms: audio.len() as f32 / sr as f32 * 1000.0,
            duration_ms: (audio.len() - syllable_start) as f32 / sr as f32 * 1000.0,
        });
    }

    Ok(syllables)
}

fn load_wav(path: &Path) -> anyhow::Result<Vec<f32>> {
    let (samples, _sr) = load_wav_with_sr(path)?;
    Ok(samples)
}

fn load_wav_with_sr(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let bytes = fs::read(path)?;
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("Not a valid WAV file");
    }

    let mut pos = 12;
    let mut data_start = 0usize;
    let mut data_size = 0usize;
    let mut sample_rate = 44100u32;

    while pos < bytes.len() - 8 {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;

        if chunk_id == b"fmt " {
            let fmt_data = &bytes[pos + 8..pos + 8 + chunk_size.min(18)];
            sample_rate = u32::from_le_bytes([fmt_data[4], fmt_data[5], fmt_data[6], fmt_data[7]]);
        } else if chunk_id == b"data" {
            data_start = pos + 8;
            data_size = chunk_size;
            break;
        }
        pos += 8 + chunk_size + (chunk_size % 2);
    }

    if data_size == 0 {
        return Ok((Vec::new(), sample_rate));
    }

    let audio_bytes = &bytes[data_start..data_start + data_size.min(bytes.len() - data_start)];

    Ok((
        audio_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        sample_rate,
    ))
}

fn compute_state(audio: &[f32]) -> u32 {
    let energy = audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32;
    let zcr = audio.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count() as f32 / audio.len() as f32;

    let energy_bin = (energy * 1000.0) as u32 % 20;
    let zcr_bin = (zcr * 100.0) as u32 % 10;
    let dur_bin = (audio.len() as f32 / 44100.0 * 100.0) as u32 % 10;

    energy_bin * 100 + zcr_bin * 10 + dur_bin
}
