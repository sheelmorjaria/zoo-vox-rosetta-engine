//! Multi-Species NBD Segmentation + Recurrence Analysis
//! =====================================================
//!
//! Performs Neural Boundary Detection (NBD) on multiple species datasets
//! and searches for segment recurrences (motif mining).
//!
//! Species analyzed:
//! - Sperm Whale (Dominica dataset)
//! - Dolphin (Whistle Signals)
//! - Orcas
//! - Giant Otter
//! - Meerkat
//! - Marmoset (for comparison)

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use technical_architecture::{BoundaryDetectorConfig, MicroDynamicsExtractor, NeuralBoundaryDetector};

#[derive(Debug, Clone)]
struct SpeciesConfig {
    name: String,
    data_dir: PathBuf,
    file_pattern: String,
    sample_rate: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SegmentInfo {
    source_file: String,
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    duration_ms: f32,
    boundary_type: String,
    feature_hash: u64, // For recurrence detection
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RecurrenceStats {
    total_segments: usize,
    unique_segments: usize,
    recurrence_rate: f64,
    top_patterns: Vec<RecurrencePattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RecurrencePattern {
    feature_hash: u64,
    count: usize,
    first_occurrence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpeciesAnalysis {
    species: String,
    total_files: usize,
    total_segments: usize,
    avg_segments_per_file: f64,
    recurrence_stats: RecurrenceStats,
    sample_rate: u32,
    nbd_config: String,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     MULTI-SPECIES NBD SEGMENTATION + RECURRENCE ANALYSIS                  ║");
    println!("║     Testing for Discrete Motifs Across Species                             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Define species configurations
    let species_configs = vec![
        SpeciesConfig {
            name: "Sperm Whale".to_string(),
            data_dir: PathBuf::from(
                dirs::home_dir()
                    .unwrap()
                    .join("birdsong_analysis/data/Dominica_dataset/Signal_parts"),
            ),
            file_pattern: "*.wav".to_string(),
            sample_rate: 96000,
        },
        SpeciesConfig {
            name: "Dolphin".to_string(),
            data_dir: PathBuf::from(dirs::home_dir().unwrap().join("birdsong_analysis/data/Whistle_Signals")),
            file_pattern: "*.wav".to_string(),
            sample_rate: 192000,
        },
        SpeciesConfig {
            name: "Orcas".to_string(),
            data_dir: PathBuf::from(dirs::home_dir().unwrap().join("birdsong_analysis/data/orcas/audio")),
            file_pattern: "*.wav".to_string(),
            sample_rate: 44100,
        },
        SpeciesConfig {
            name: "Giant Otter".to_string(),
            data_dir: PathBuf::from(
                dirs::home_dir()
                    .unwrap()
                    .join("birdsong_analysis/data/giant_otter/giant_otters/Audio_S1"),
            ),
            file_pattern: "*.wav".to_string(),
            sample_rate: 44100,
        },
        SpeciesConfig {
            name: "Meerkat".to_string(),
            data_dir: PathBuf::from("/mnt/c/Users/sheel/Desktop/data/MeerKAT_10s_2024-06-12/wav"),
            file_pattern: "*.wav".to_string(),
            sample_rate: 44100,
        },
        SpeciesConfig {
            name: "Marmoset".to_string(),
            data_dir: PathBuf::from(
                dirs::home_dir()
                    .unwrap()
                    .join("birdsong_analysis/data/marmoset_wav_subset"),
            ),
            file_pattern: "*.wav".to_string(),
            sample_rate: 44100,
        },
    ];

    let output_dir = Path::new("multi_species_nbd_results");
    fs::create_dir_all(output_dir)?;

    // Process each species
    let mut all_results = Vec::new();

    for config in &species_configs {
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!("Processing: {}", config.name);
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!();

        match analyze_species(config, output_dir) {
            Ok(analysis) => {
                println!("  ✓ {} files processed", analysis.total_files);
                println!("  ✓ {} segments extracted", analysis.total_segments);
                println!("  ✓ {:.1} segments/file average", analysis.avg_segments_per_file);
                println!(
                    "  ✓ Recurrence rate: {:.1}%",
                    analysis.recurrence_stats.recurrence_rate * 100.0
                );
                println!();
                all_results.push(analysis);
            }
            Err(e) => {
                println!("  ✗ Error: {}", e);
                println!();
            }
        }
    }

    // Final comparison
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("CROSS-SPECIES COMPARISON");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  Species       │ Files │ Segments │ Seg/File │ Recurrence │           │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    all_results.sort_by(|a, b| {
        b.recurrence_stats
            .recurrence_rate
            .partial_cmp(&a.recurrence_stats.recurrence_rate)
            .unwrap()
    });

    for result in &all_results {
        println!(
            "  │  {:14} │ {:5} │ {:8} │ {:8.1} │ {:8.1}% │           │",
            result.species,
            result.total_files,
            result.total_segments,
            result.avg_segments_per_file,
            result.recurrence_stats.recurrence_rate * 100.0
        );
    }

    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Classification
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  CLASSIFICATION (Recurrence > 50% = Discrete, < 50% = Graded)          │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    for result in &all_results {
        let class = if result.recurrence_stats.recurrence_rate > 0.5 {
            "DISCRETE MOTIFS"
        } else {
            "GRADED CONTINUUM"
        };
        println!(
            "  │  {:14}: {:^20}                                  │",
            result.species, class
        );
    }

    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Save results
    let results_json = serde_json::to_string_pretty(&all_results)?;
    fs::write(output_dir.join("all_species_analysis.json"), results_json)?;

    println!("  Results saved to: {}", output_dir.display());
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}

fn analyze_species(config: &SpeciesConfig, output_dir: &Path) -> anyhow::Result<SpeciesAnalysis> {
    // Find audio files
    let audio_files: Vec<PathBuf> = fs::read_dir(&config.data_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "wav" || ext == "WAV")
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .take(500) // Limit for performance
        .collect();

    let total_files = audio_files.len();

    if total_files == 0 {
        anyhow::bail!("No audio files found in {}", config.data_dir.display());
    }

    println!("  Data directory: {}", config.data_dir.display());
    println!("  Files found: {}", total_files);
    println!("  Sample rate: {} Hz", config.sample_rate);
    println!();

    // NBD Configuration
    let nbd_config = BoundaryDetectorConfig {
        hop_size: 2048,
        sample_rate: config.sample_rate,
        min_phrase_duration_ms: 5.0,
        threshold: 0.25,
        smoothing_frames: 2,
    };

    // Process files and collect segments
    println!("  Processing files...");

    let all_segments: Vec<SegmentInfo> = audio_files
        .par_iter()
        .flat_map(|path| process_audio_file(path, &nbd_config).unwrap_or_default())
        .collect();

    let total_segments = all_segments.len();
    let avg_segments = if total_files > 0 {
        total_segments as f64 / total_files as f64
    } else {
        0.0
    };

    println!("  Extracted {} segments", total_segments);
    println!("  Average {:.1} segments per file", avg_segments);
    println!();

    // Compute recurrence stats
    let recurrence_stats = compute_recurrence(&all_segments);

    println!("  Recurrence analysis:");
    println!("    • Unique patterns: {}", recurrence_stats.unique_segments);
    println!(
        "    • Recurrence rate: {:.1}%",
        recurrence_stats.recurrence_rate * 100.0
    );

    if !recurrence_stats.top_patterns.is_empty() {
        println!("    • Top recurring patterns:");
        for (i, pattern) in recurrence_stats.top_patterns.iter().take(5).enumerate() {
            println!(
                "      {}.{:>3} occurrences of {:016x}",
                i + 1,
                pattern.count,
                pattern.feature_hash
            );
        }
    }

    // Save species-specific results
    let species_dir = output_dir.join(&config.name.to_lowercase().replace(' ', "_"));
    fs::create_dir_all(&species_dir)?;

    let segments_json = serde_json::to_string_pretty(&all_segments)?;
    fs::write(species_dir.join("segments.json"), segments_json)?;

    Ok(SpeciesAnalysis {
        species: config.name.clone(),
        total_files,
        total_segments,
        avg_segments_per_file: avg_segments,
        recurrence_stats,
        sample_rate: config.sample_rate,
        nbd_config: format!("{:?}", nbd_config),
    })
}

fn process_audio_file(path: &Path, nbd_config: &BoundaryDetectorConfig) -> anyhow::Result<Vec<SegmentInfo>> {
    let audio = load_wav(path)?;
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    let sr = nbd_config.sample_rate as f32;
    let mut detector = NeuralBoundaryDetector::with_config(nbd_config.clone());
    let boundaries = detector.detect_boundaries(&audio);

    let mut segments = Vec::new();
    let min_samples = (sr * 0.003) as usize; // 3ms minimum
    let mut start = 0usize;

    for b in &boundaries {
        let end = (b.time_ms * sr / 1000.0) as usize;
        if end > start && end <= audio.len() && end - start >= min_samples {
            let segment_audio = &audio[start..end];
            let feature_hash = compute_feature_hash(segment_audio);

            segments.push(SegmentInfo {
                source_file: path.file_name().unwrap().to_str().unwrap().to_string(),
                segment_idx: segments.len(),
                start_ms: start as f32 / sr * 1000.0,
                end_ms: end as f32 / sr * 1000.0,
                duration_ms: (end - start) as f32 / sr * 1000.0,
                boundary_type: format!("{:?}", b.boundary_type),
                feature_hash,
            });
        }
        start = end;
    }

    // Final segment
    if audio.len() - start >= min_samples {
        let segment_audio = &audio[start..];
        let feature_hash = compute_feature_hash(segment_audio);

        segments.push(SegmentInfo {
            source_file: path.file_name().unwrap().to_str().unwrap().to_string(),
            segment_idx: segments.len(),
            start_ms: start as f32 / sr * 1000.0,
            end_ms: audio.len() as f32 / sr * 1000.0,
            duration_ms: (audio.len() - start) as f32 / sr * 1000.0,
            boundary_type: "End".to_string(),
            feature_hash,
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
    let mut sample_rate = 0u32;
    let mut audio_format = 0u16;
    let mut bits_per_sample = 0u16;
    let mut data_start = 0usize;
    let mut data_size = 0usize;

    while pos < bytes.len() - 8 {
        let chunk_id = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]]) as usize;

        if chunk_id == b"fmt " {
            let fmt_data = &bytes[pos + 8..pos + 8 + chunk_size.min(18)];
            audio_format = u16::from_le_bytes([fmt_data[0], fmt_data[1]]);
            sample_rate = u32::from_le_bytes([fmt_data[4], fmt_data[5], fmt_data[6], fmt_data[7]]);
            bits_per_sample = u16::from_le_bytes([fmt_data[14], fmt_data[15]]);
        } else if chunk_id == b"data" {
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

    let samples: Vec<f32> = match (audio_format, bits_per_sample) {
        (3, 32) => audio_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        (1, 16) => audio_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
        _ => return Ok(Vec::new()),
    };

    Ok(samples)
}

fn compute_feature_hash(audio: &[f32]) -> u64 {
    // Simple hash based on energy envelope statistics
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Compute envelope statistics
    let envelope: Vec<f32> = audio
        .windows(101)
        .map(|w| w.iter().map(|x| x.abs()).sum::<f32>() / 101.0)
        .collect();

    if envelope.is_empty() {
        return 0;
    }

    // Quantize statistics for hash
    let mean = envelope.iter().sum::<f32>() / envelope.len() as f32;
    let max = envelope.iter().cloned().fold(0.0f32, f32::max);
    let std = {
        let variance: f32 = envelope.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / envelope.len() as f32;
        variance.sqrt()
    };

    // Quantize to create hash
    let q_mean = (mean * 1000.0) as u32;
    let q_max = (max * 1000.0) as u32;
    let q_std = (std * 1000.0) as u32;
    let q_len = (audio.len() / 100) as u32;

    let mut hasher = DefaultHasher::new();
    (q_mean, q_max, q_std, q_len).hash(&mut hasher);
    hasher.finish()
}

fn compute_recurrence(segments: &[SegmentInfo]) -> RecurrenceStats {
    let total_segments = segments.len();
    if total_segments == 0 {
        return RecurrenceStats {
            total_segments: 0,
            unique_segments: 0,
            recurrence_rate: 0.0,
            top_patterns: Vec::new(),
        };
    }

    // Count occurrences of each feature hash
    let mut hash_counts: HashMap<u64, usize> = HashMap::new();
    let mut first_occurrence: HashMap<u64, String> = HashMap::new();

    for seg in segments {
        *hash_counts.entry(seg.feature_hash).or_insert(0) += 1;
        first_occurrence
            .entry(seg.feature_hash)
            .or_insert_with(|| seg.source_file.clone());
    }

    let unique_segments = hash_counts.len();

    // Recurrence rate = 1 - (unique / total)
    let recurrence_rate = 1.0 - (unique_segments as f64 / total_segments as f64);

    // Find top patterns
    let mut patterns: Vec<_> = hash_counts.into_iter().collect();
    patterns.sort_by(|a, b| b.1.cmp(&a.1));

    let top_patterns: Vec<RecurrencePattern> = patterns
        .into_iter()
        .take(10)
        .map(|(hash, count)| RecurrencePattern {
            feature_hash: hash,
            count,
            first_occurrence: first_occurrence.get(&hash).cloned().unwrap_or_default(),
        })
        .collect();

    RecurrenceStats {
        total_segments,
        unique_segments,
        recurrence_rate,
        top_patterns,
    }
}
