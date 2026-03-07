//! Bird Species NBD + Syntax Mining
//! ==================================
//!
//! Performs Neural Boundary Detection and N-gram mining on bird song datasets.
//!
//! Datasets:
//! - bird_songs: 687 multi-species bird recordings
//!
//! Tests whether birds have:
//! - Discrete motifs (atomic level)
//! - Discrete syntax (N-gram level)

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct Segment {
    source_file: String,
    segment_idx: usize,
    state_id: u32,
    duration_ms: f32,
    species: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpeciesStats {
    species: String,
    total_files: usize,
    total_segments: usize,
    unique_states: usize,
    recurrence_rate: f64,
    bigram_reuse_rate: f64,
    trigram_reuse_rate: f64,
    has_discrete_motifs: bool,
    has_discrete_syntax: bool,
    top_bigrams: Vec<(Vec<u32>, usize)>,
}

#[derive(Debug, Clone, Deserialize)]
struct SegmentInfo {
    source_file: String,
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    duration_ms: f32,
    boundary_type: String,
    feature_hash: u64,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     BIRD SPECIES NBD + SYNTAX MINING                                       ║");
    println!("║     Testing for Discrete Motifs and Syntax in Birds                        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let bird_songs_dir = dirs::home_dir()
        .unwrap()
        .join("birdsong_analysis/data/bird_songs/audio");
    let annotations_path = dirs::home_dir()
        .unwrap()
        .join("birdsong_analysis/data/bird_songs/annotations.csv");

    // Load species annotations
    let species_map = load_species_annotations(&annotations_path)?;
    println!("  Loaded species annotations for {} files", species_map.len());

    // Get audio files
    let audio_files: Vec<PathBuf> = fs::read_dir(&bird_songs_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "wav").unwrap_or(false))
        .map(|e| e.path())
        .take(500) // Limit for performance
        .collect();

    println!("  Processing {} audio files", audio_files.len());
    println!();

    // Process all files and collect segments
    println!("  Extracting segments with NBD...");
    let all_segments: Vec<Segment> = audio_files
        .par_iter()
        .flat_map(|path| {
            let species = species_map
                .get(path.file_name().unwrap().to_str().unwrap())
                .cloned()
                .unwrap_or_else(|| "Unknown".to_string());
            process_audio_file(path, &species).unwrap_or_default()
        })
        .collect();

    let total_segments = all_segments.len();
    println!("  Extracted {} total segments", total_segments);
    println!();

    // Group by species
    let mut species_segments: HashMap<String, Vec<&Segment>> = HashMap::new();
    for seg in &all_segments {
        species_segments.entry(seg.species.clone()).or_default().push(seg);
    }

    println!("  Found {} species", species_segments.len());
    println!();

    // Analyze each species
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("PER-SPECIES ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let mut all_stats = Vec::new();

    for (species, segments) in species_segments.iter() {
        if segments.len() < 50 {
            continue;
        } // Skip species with too few segments

        println!("  {} ({} segments)", species, segments.len());

        // Recurrence analysis
        let unique_states: std::collections::HashSet<u32> = segments.iter().map(|s| s.state_id).collect();
        let recurrence_rate = 1.0 - (unique_states.len() as f64 / segments.len() as f64);

        // N-gram analysis
        let (bigram_reuse, trigram_reuse, top_bigrams) = analyze_ngrams(segments);

        let has_discrete_motifs = recurrence_rate > 0.5;
        let has_discrete_syntax = bigram_reuse > 0.5;

        println!(
            "    Recurrence: {:.1}%  Bigram reuse: {:.1}%  Trigram reuse: {:.1}%",
            recurrence_rate * 100.0,
            bigram_reuse * 100.0,
            trigram_reuse * 100.0
        );
        println!(
            "    Discrete motifs: {}  Discrete syntax: {}",
            if has_discrete_motifs { "✓" } else { "✗" },
            if has_discrete_syntax { "✓" } else { "✗" }
        );
        println!();

        all_stats.push(SpeciesStats {
            species: species.clone(),
            total_files: segments
                .iter()
                .map(|s| s.source_file.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len(),
            total_segments: segments.len(),
            unique_states: unique_states.len(),
            recurrence_rate,
            bigram_reuse_rate: bigram_reuse,
            trigram_reuse_rate: trigram_reuse,
            has_discrete_motifs,
            has_discrete_syntax,
            top_bigrams,
        });
    }

    // Sort by bigram reuse
    all_stats.sort_by(|a, b| b.bigram_reuse_rate.partial_cmp(&a.bigram_reuse_rate).unwrap());

    // Comparison table
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("CROSS-SPECIES COMPARISON");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  Species           │ Segments │ Recur.  │ Bigram │ Motifs │ Syntax    │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    for stats in &all_stats {
        let motifs = if stats.has_discrete_motifs { "✓" } else { "✗" };
        let syntax = if stats.has_discrete_syntax { "✓" } else { "✗" };
        println!(
            "  │  {:17} │ {:8} │ {:5.1}%  │ {:5.1}% │   {}    │    {}     │",
            stats.species,
            stats.total_segments,
            stats.recurrence_rate * 100.0,
            stats.bigram_reuse_rate * 100.0,
            motifs,
            syntax
        );
    }

    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Reference species
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  REFERENCE SPECIES                                                       │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │  Bengalese Finch: Discrete motifs + Discrete syntax (100% both)        │");
    println!("  │  European Starling: Graded atoms + Discrete syntax (~60% reuse)       │");
    println!("  │  Egyptian Fruit Bat: Graded atoms + Discrete syntax (87.9% reuse)      │");
    println!("  │  Zebra Finch: Discrete motifs + Discrete syntax                         │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Save results
    let output_dir = Path::new("bird_nbd_results");
    fs::create_dir_all(output_dir)?;
    let json = serde_json::to_string_pretty(&all_stats)?;
    fs::write(output_dir.join("bird_species_analysis.json"), json)?;

    println!(
        "  Results saved to: {}/bird_species_analysis.json",
        output_dir.display()
    );
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}

fn load_species_annotations(path: &Path) -> anyhow::Result<HashMap<String, String>> {
    let content = fs::read_to_string(path)?;
    let mut map = HashMap::new();

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Ok(map);
    }

    // Parse header to get species names
    let header: Vec<&str> = lines[0].split(',').collect();
    let species_names: Vec<&str> = header[1..].to_vec();

    // Parse each row
    for line in &lines[1..] {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.is_empty() {
            continue;
        }

        let filename = parts[0].to_string();

        // Find the species with highest count (or first 1)
        for (i, &val) in parts[1..].iter().enumerate() {
            if val == "1" && i < species_names.len() {
                // Extract species name from column header (e.g., "Parcae_song" -> "Parcae")
                let species_full = species_names[i];
                let species = species_full.split('_').next().unwrap_or(species_full);
                map.insert(filename, species.to_string());
                break;
            }
        }
    }

    Ok(map)
}

fn process_audio_file(path: &Path, species: &str) -> anyhow::Result<Vec<Segment>> {
    let audio = load_wav(path)?;
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    let sr = 44100; // Assume 44.1kHz for bird songs
    let min_samples = (sr as f32 * 0.005) as usize; // 5ms minimum

    // Simple energy-based segmentation
    let window = (sr as f32 * 0.01) as usize; // 10ms windows
    let threshold = 0.02; // Energy threshold

    let mut segments = Vec::new();
    let mut in_segment = false;
    let mut segment_start = 0usize;

    for i in 0..audio.len() / window {
        let start = i * window;
        let end = (start + window).min(audio.len());

        let energy: f32 = audio[start..end].iter().map(|x| x * x).sum::<f32>() / window as f32;

        if energy > threshold && !in_segment {
            in_segment = true;
            segment_start = start;
        } else if energy <= threshold && in_segment {
            in_segment = false;
            let segment_end = i * window;

            if segment_end - segment_start >= min_samples {
                let segment_audio = &audio[segment_start..segment_end];
                let state_id = compute_state(segment_audio);

                segments.push(Segment {
                    source_file: path.file_name().unwrap().to_str().unwrap().to_string(),
                    segment_idx: segments.len(),
                    state_id,
                    duration_ms: (segment_end - segment_start) as f32 / sr as f32 * 1000.0,
                    species: species.to_string(),
                });
            }
        }
    }

    // Handle last segment
    if in_segment && audio.len() - segment_start >= min_samples {
        let segment_audio = &audio[segment_start..];
        let state_id = compute_state(segment_audio);

        segments.push(Segment {
            source_file: path.file_name().unwrap().to_str().unwrap().to_string(),
            segment_idx: segments.len(),
            state_id,
            duration_ms: (audio.len() - segment_start) as f32 / sr as f32 * 1000.0,
            species: species.to_string(),
        });
    }

    Ok(segments)
}

fn load_wav(path: &Path) -> anyhow::Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("Not a valid WAV file");
    }

    let mut pos = 12;
    let mut data_start = 0usize;
    let mut data_size = 0usize;

    while pos < bytes.len() - 8 {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;

        if chunk_id == b"data" {
            data_start = pos + 8;
            data_size = chunk_size;
            break;
        }
        pos += 8 + chunk_size + (chunk_size % 2);
    }

    if data_size == 0 {
        return Ok(Vec::new());
    }

    let audio_bytes = &bytes[data_start..data_start + data_size.min(bytes.len() - data_start)];

    // Assume 16-bit PCM
    let samples: Vec<f32> = audio_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();

    Ok(samples)
}

fn compute_state(audio: &[f32]) -> u32 {
    // Compute simple acoustic state from statistics
    let mean = audio.iter().sum::<f32>() / audio.len() as f32;
    let energy = audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32;
    let zero_crossings =
        audio.windows(2).filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0)).count() as f32 / audio.len() as f32;

    // Discretize into state bins
    let energy_bin = (energy * 100.0) as u32 % 10;
    let zc_bin = (zero_crossings * 100.0) as u32 % 10;

    energy_bin * 10 + zc_bin
}

fn analyze_ngrams(segments: &[&Segment]) -> (f64, f64, Vec<(Vec<u32>, usize)>) {
    // Group by file
    let mut file_sequences: HashMap<&str, Vec<u32>> = HashMap::new();
    for seg in segments {
        file_sequences.entry(&seg.source_file).or_default().push(seg.state_id);
    }

    // Count bigrams
    let mut bigram_counts: HashMap<(u32, u32), usize> = HashMap::new();
    let mut bigram_total = 0usize;

    for (_file, seq) in &file_sequences {
        if seq.len() < 2 {
            continue;
        }
        for i in 0..seq.len() - 1 {
            *bigram_counts.entry((seq[i], seq[i + 1])).or_insert(0) += 1;
            bigram_total += 1;
        }
    }

    let bigram_unique = bigram_counts.len();
    let bigram_reuse = if bigram_total > 0 {
        1.0 - (bigram_unique as f64 / bigram_total as f64)
    } else {
        0.0
    };

    // Count trigrams
    let mut trigram_counts: HashMap<(u32, u32, u32), usize> = HashMap::new();
    let mut trigram_total = 0usize;

    for (_file, seq) in &file_sequences {
        if seq.len() < 3 {
            continue;
        }
        for i in 0..seq.len() - 2 {
            *trigram_counts.entry((seq[i], seq[i + 1], seq[i + 2])).or_insert(0) += 1;
            trigram_total += 1;
        }
    }

    let trigram_unique = trigram_counts.len();
    let trigram_reuse = if trigram_total > 0 {
        1.0 - (trigram_unique as f64 / trigram_total as f64)
    } else {
        0.0
    };

    // Top bigrams
    let mut top_bigrams: Vec<_> = bigram_counts
        .into_iter()
        .map(|((a, b), count)| (vec![a, b], count))
        .collect();
    top_bigrams.sort_by(|a, b| b.1.cmp(&a.1));
    top_bigrams.truncate(5);

    (bigram_reuse, trigram_reuse, top_bigrams)
}
