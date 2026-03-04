//! Marmoset Vocalization Syllable-Level NBD + N-gram Mining
//! =======================================================
//!
//! Segments each marmoset vocalization into syllables,
//! then performs N-gram mining on syllable sequences.
//!
//! Dataset: ~/birdsong_analysis/data/Vocalizations (871,045 FLAC files)
//!
//! Each file = 1 vocalization call
//! Each call = sequence of syllables
//! Syllable sequences = N-gram patterns

use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
struct Syllable {
    source_file: String,
    call_type: String, // Trill, Phee, Tsik, etc.
    syllable_idx: usize,
    state_id: u32,
    duration_ms: f32,
}

#[derive(Debug, Clone, Serialize)]
struct AnalysisResult {
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
    call_type_stats: HashMap<String, usize>,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     MARMOSET VOCALIZATION SYLLABLE-LEVEL NBD + SYNTAX MINING              ║");
    println!("║     871,045 vocalization files → syllable sequences → N-gram mining        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let base_dir = dirs::home_dir()
        .unwrap()
        .join("birdsong_analysis/data/Vocalizations");

    // Find FLAC files
    println!("  Finding FLAC files...");
    let mut all_flac: Vec<PathBuf> = Vec::new();

    for entry in fs::read_dir(&base_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            for sub_entry in fs::read_dir(&path)? {
                let sub = sub_entry?;
                if sub.path().extension().map(|x| x == "flac").unwrap_or(false) {
                    all_flac.push(sub.path());
                }
            }
        }
    }

    println!("  Found {} FLAC files", all_flac.len());

    // Sample for analysis (full dataset would take too long)
    let sample_size = 5000.min(all_flac.len());
    all_flac.truncate(sample_size);
    println!("  Processing {} files (sample)", all_flac.len());
    println!();

    // Extract syllables
    println!("  Extracting syllables with NBD...");
    let file_syllables: Vec<Vec<Syllable>> = all_flac
        .par_iter()
        .map(|path| extract_syllables(path).unwrap_or_default())
        .collect();

    let total_files = file_syllables.len();
    let total_syllables: usize = file_syllables.iter().map(|v| v.len()).sum();
    let avg_syllables = total_syllables as f64 / total_files as f64;

    println!(
        "  Extracted {} syllables from {} files",
        total_syllables, total_files
    );
    println!("  Average {:.2} syllables per vocalization", avg_syllables);
    println!();

    // Count call types
    let mut call_type_stats: HashMap<String, usize> = HashMap::new();
    for syllables in &file_syllables {
        for syl in syllables {
            *call_type_stats.entry(syl.call_type.clone()).or_insert(0) += 1;
        }
    }

    // Count states
    let all_syllables: Vec<&Syllable> = file_syllables.iter().flatten().collect();
    let unique_states: std::collections::HashSet<u32> =
        all_syllables.iter().map(|s| s.state_id).collect();
    let recurrence_rate = 1.0 - (unique_states.len() as f64 / all_syllables.len() as f64);

    println!("  Unique syllable states: {}", unique_states.len());
    println!("  Recurrence rate: {:.1}%", recurrence_rate * 100.0);
    println!();

    // N-gram mining
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
                *trigram_counts
                    .entry((seq[i], seq[i + 1], seq[i + 2]))
                    .or_insert(0) += 1;
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

    println!("  Top syllable bigrams:");
    for ((a, b), count) in top_bigrams.iter().take(10) {
        println!("    State {} → State {}: {} occurrences", a, b, count);
    }
    println!();

    let has_discrete_syntax = bigram_reuse_rate > 0.5;

    // Results
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("RESULTS SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  MARMOSET VOCALIZATION ANALYSIS                                         │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Files processed: {:>8}                                              ",
        total_files
    );
    println!(
        "  │  Total syllables: {:>8}                                              ",
        total_syllables
    );
    println!(
        "  │  Avg syllables/file: {:>6.2}                                           ",
        avg_syllables
    );
    println!(
        "  │  Unique states: {:>8}                                                 ",
        unique_states.len()
    );
    println!(
        "  │  Recurrence rate: {:6.1}%                                             ",
        recurrence_rate * 100.0
    );
    println!(
        "  │  Bigram reuse: {:6.1}%                                                ",
        bigram_reuse_rate * 100.0
    );
    println!(
        "  │  Trigram reuse: {:6.1}%                                               ",
        trigram_reuse_rate * 100.0
    );
    println!(
        "  │  Discrete syntax: {}                                               ",
        if has_discrete_syntax {
            "✓ YES"
        } else {
            "✗ NO"
        }
    );
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Call type distribution
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  CALL TYPE DISTRIBUTION                                                 │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    let mut call_types: Vec<_> = call_type_stats.iter().collect();
    call_types.sort_by(|a, b| b.1.cmp(a.1));

    for (call_type, count) in call_types.iter().take(10) {
        let pct = **count as f64 / total_syllables as f64 * 100.0;
        println!(
            "  │  {:20} {:8} ({:5.1}%)                                ",
            call_type, count, pct
        );
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Comparison
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  COMPARISON TO OTHER SPECIES                                            │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │  Species          │ Syl/File │ Bigram Reuse │ Syntax                   │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Marmoset (this)  │ {:7.1} │ {:10.1}%  │ {}                   │",
        avg_syllables,
        bigram_reuse_rate * 100.0,
        if has_discrete_syntax {
            "DISCR"
        } else {
            "GRADED"
        }
    );
    println!("  │  Sperm Whale      │  1032.3  │       97.4%  │ DISCR                    │");
    println!("  │  Egyptian Bat     │    17.2  │       87.9%  │ DISCR                    │");
    println!("  │  Giant Otter      │     9.4  │       28.9%  │ GRADED                   │");
    println!("  │  Orcas            │     3.2  │        6.1%  │ GRADED                   │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Save
    let top_bigrams_str: Vec<(String, usize)> = top_bigrams
        .into_iter()
        .take(10)
        .map(|((a, b), count)| (format!("{}→{}", a, b), count))
        .collect();

    let result = AnalysisResult {
        dataset: "Marmoset_Vocalizations".to_string(),
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
        call_type_stats,
    };

    let output_dir = Path::new("marmoset_syllable_results");
    fs::create_dir_all(output_dir)?;
    let json = serde_json::to_string_pretty(&result)?;
    fs::write(output_dir.join("marmoset_syllable_analysis.json"), json)?;

    println!(
        "  Results saved to: {}/marmoset_syllable_analysis.json",
        output_dir.display()
    );
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}

fn extract_syllables(path: &Path) -> anyhow::Result<Vec<Syllable>> {
    let (audio, sr) = load_flac(path)?;
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    // Extract call type from filename
    let filename = path.file_name().unwrap().to_str().unwrap();
    let call_type = extract_call_type(filename);

    let window = (sr as f32 * 0.005) as usize; // 5ms windows
    let min_syllable = (sr as f32 * 0.003) as usize; // 3ms minimum

    // Adaptive threshold
    let max_amp = audio.iter().cloned().fold(0.0f32, f32::max);
    let mean_energy: f32 = audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32;
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

                syllables.push(Syllable {
                    source_file: filename.to_string(),
                    call_type: call_type.clone(),
                    syllable_idx: syllable_count,
                    state_id,
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

        syllables.push(Syllable {
            source_file: filename.to_string(),
            call_type: call_type.clone(),
            syllable_idx: syllable_count,
            state_id,
            duration_ms: (audio.len() - syllable_start) as f32 / sr as f32 * 1000.0,
        });
    }

    Ok(syllables)
}

fn extract_call_type(filename: &str) -> String {
    // Extract call type from filename like "Trill_12345.flac"
    if filename.contains("Trill") {
        return "Trill".to_string();
    }
    if filename.contains("Phee") {
        return "Phee".to_string();
    }
    if filename.contains("Tsik") {
        return "Tsik".to_string();
    }
    if filename.contains("Twitter") {
        return "Twitter".to_string();
    }
    if filename.contains("Infant") || filename.contains("Infant_cry") {
        return "Infant_cry".to_string();
    }
    if filename.contains("Vocalization") {
        return "Vocalization".to_string();
    }
    "Unknown".to_string()
}

fn load_flac(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    // Use FFmpeg to decode FLAC to raw samples
    use std::process::Command;

    let output = Command::new("ffmpeg")
        .args([
            "-i",
            path.to_str().unwrap(),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-",
        ])
        .output()?;

    if !output.status.success() {
        return Ok((Vec::new(), 44100));
    }

    let bytes = &output.stdout;
    let sr = 44100u32;

    let samples: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();

    Ok((samples, sr))
}

fn compute_state(audio: &[f32]) -> u32 {
    let energy = audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32;
    let zcr = audio
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count() as f32
        / audio.len() as f32;

    let energy_bin = (energy * 1000.0) as u32 % 20;
    let zcr_bin = (zcr * 100.0) as u32 % 10;
    let dur_bin = (audio.len() as f32 / 44100.0 * 100.0) as u32 % 10;

    energy_bin * 100 + zcr_bin * 10 + dur_bin
}
