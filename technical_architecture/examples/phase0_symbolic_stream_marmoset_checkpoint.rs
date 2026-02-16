// Phase 0: Symbolic Stream Generation for Marmoset
//
// This is prerequisite analysis that converts raw audio into a sequence of
// discrete symbols (Cluster IDs) using HDBSCAN clustering.
//
// Input:  Corpus of FLAC files (marmoset vocalizations)
// Feature Extraction: extract_15d_marmoset() via MicroDynamicsExtractor
// Discovery: HDBSCAN (hierarchical density-based clustering)
// Output: A long sequence of Cluster IDs representing discovered "words" or "syllables"
//
// Usage: cargo run --release --example phase0_symbolic_stream_marmoset [--limit N]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use serde::{Serialize, Deserialize};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use technical_architecture::{
    MicroDynamicsExtractor,
    hdbscan::{HdbscanClustering, DistanceMetric},
};

// =============================================================================
// Data Structures
// =============================================================================

/// Features extracted from a single phrase
#[derive(Clone, Debug)]
struct ExtractedFeatures {
    file_name: String,
    call_type: String,
    phrase_index: usize,
    features: Vec<f64>,  // 15D features
    duration_ms: f64,
}

/// Serializable version for saving
#[derive(Serialize, Deserialize, Clone)]
struct SerializableFeatures {
    file_name: String,
    call_type: String,
    phrase_index: usize,
    features: Vec<f64>,
    duration_ms: f64,
}

/// Checkpoint data structure for resume capability
#[derive(Serialize, Deserialize, Clone)]
struct CheckpointData {
    all_features: Vec<SerializableFeatures>,
    all_file_names: Vec<String>,
    total_files: usize,
    processed_at: String,  // ISO 8601 timestamp
}

/// Checkpoint statistics for display
#[derive(Serialize, Deserialize, Clone)]
struct CheckpointStats {
    resumed_from: usize,
    skipped_already_processed: bool,
}

/// Marmoset call types (contexts)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CallType {
    Vocalization,
    Phee,
    Twitter,
    Trill,
    Tsik,
    Seep,
    Infant,
    Unknown,
}

impl CallType {
    fn from_filename(filename: &str) -> Self {
        let fname = filename.to_lowercase();
        if fname.contains("vocalization") {
            CallType::Vocalization
        } else if fname.contains("phee") {
            CallType::Phee
        } else if fname.contains("twitter") {
            CallType::Twitter
        } else if fname.contains("trill") {
            CallType::Trill
        } else if fname.contains("tsik") {
            CallType::Tsik
        } else if fname.contains("seep") {
            CallType::Seep
        } else if fname.contains("infant") {
            CallType::Infant
        } else {
            CallType::Unknown
        }
    }

    fn name(&self) -> &'static str {
        match self {
            CallType::Vocalization => "Vocalization",
            CallType::Phee => "Phee",
            CallType::Twitter => "Twitter",
            CallType::Trill => "Trill",
            CallType::Tsik => "Tsik",
            CallType::Seep => "Seep",
            CallType::Infant => "Infant",
            CallType::Unknown => "Unknown",
        }
    }
}

/// Cluster size classification for interpretation
fn classify_cluster_by_size(size: usize) -> &'static str {
    if size >= 100 {
        "Frequent Word"
    } else if size >= 20 {
        "Common Word"
    } else if size >= 5 {
        "Rare Word"
    } else {
        "Unique Word"
    }
}

// =============================================================================
// Main Function
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║         Phase 0: Symbolic Stream Generation - Marmoset                ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                                                                   ║");
    println!("║  GOAL: Convert raw audio → discrete symbol sequence (Cluster IDs)║");
    println!("║                                                                   ║");
    println!("║  Input:  FLAC files (marmoset vocalizations)                    ║");
    println!("║  Method: HDBSCAN clustering on 15D Goldilocks features         ║");
    println!("║  Output: Symbolic stream [101, 105, 101, 105, 200, 101, 105, 300...] ║");
    println!("╚═════════════════════════════════════════════════════════════════╝");
    println!();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let mut vocalizations_dir = PathBuf::from("/home/sheel/birdsong_analysis/data/Vocalizations");
    let mut results_dir = PathBuf::from("/mnt/c/Users/sheel/Desktop/src/marmoset_phase0_results");
    let mut limit = None; // Limit number of files to process (for testing)

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => {
                if i + 1 < args.len() {
                    if let Ok(n) = args[i + 1].parse::<usize>() {
                        limit = Some(n);
                        println!("📊 Limiting to {} files for testing", n);
                        i += 1;
                    }
                }
            }
            arg if i == args.len() - 1 && !arg.starts_with("--") => {
                // Last argument is the directory
                vocalizations_dir = PathBuf::from(arg);
            }
            _ => {}
        }
        i += 1;
    }

    let sample_rate = 96000; // Common for marmoset recordings

    // =============================================================================
    // Step 0: Dataset Overview
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Dataset Overview                                                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    if !vocalizations_dir.exists() {
        println!("   ❌ Vocalizations directory not found: {}", vocalizations_dir.display());
        return Err("Dataset not found".into());
    }

    let mut flac_files = discover_flac_files(&vocalizations_dir)?;
    let total_files = flac_files.len();

    // Apply limit if specified
    if let Some(n) = limit {
        let original_len = flac_files.len();
        flac_files.truncate(n.min(original_len));
        println!("📊 Limited to {} files (was {})", flac_files.len(), original_len);
    }

    println!("   📂 Vocalizations Directory: {}", vocalizations_dir.display());
    println!("   🔢 Total FLAC files: {}", flac_files.len());
    println!("   💾 Results Directory: {}", results_dir.display());
    println!();

    // Count by call type
    let mut call_type_counts: HashMap<CallType, usize> = HashMap::new();
    for (path, _) in &flac_files {
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        let call_type = CallType::from_filename(filename);
        *call_type_counts.entry(call_type).or_insert(0) += 1;
    }

    println!("   📊 Call Type Distribution:");
    for (call_type, count) in call_type_counts.iter() {
        println!("      • {}: {} files", call_type.name(), count);
    }
    println!();

    // =============================================================================
    // Step 1: Load Audio and Extract 15D Features
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Feature Extraction - 15D Goldilocks Subset              │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   📊 Extracting 15D Goldilocks features from FLAC files...");
    println!("      └─ Features: RFE-optimized for marmoset call types");
    println!();

    let extract_start = Instant::now();

    let mut all_features: Vec<ExtractedFeatures> = Vec::new();
    let mut all_file_names: Vec<String> = Vec::new();

    for (i, (file_path, call_type)) in flac_files.iter().enumerate() {
        let filename = file_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        if i % 100 == 0 || i == flac_files.len() - 1 {
            println!("   🔄 Processing {}/{} ({:.1}%)...",
                     i + 1, flac_files.len(), (i + 1) as f64 / flac_files.len() as f64 * 100.0);
        }

        match load_flac_file(file_path) {
            Ok(audio) => {
                // Extract 15D Goldilocks features
                let extractor = MicroDynamicsExtractor::new(sample_rate);
                match extractor.extract_15d_marmoset(&audio) {
                    Ok(features) => {
                        let feature_vec = features.to_array().to_vec();

                        all_file_names.push(filename.to_string());

                        // Each vocalization treated as one "phrase" for Phase 0
                        // In Phase 1, we'll do within-vocalization segmentation
                        all_features.push(ExtractedFeatures {
                            file_name: filename.to_string(),
                            call_type: call_type.name().to_string(),
                            phrase_index: 0,
                            features: feature_vec.into_iter().map(|v| v as f64).collect(),
                            duration_ms: audio.len() as f64 / sample_rate as f64 * 1000.0,
                        });
                    }
                    Err(e) => {
                        eprintln!("      Warning: Feature extraction failed for {}: {}", filename, e);
                    }
                }
            }
            Err(e) => {
                eprintln!("      Warning: Failed to load {}: {}", filename, e);
            }
        }
    }

    let extract_time = extract_start.elapsed();
    let n_features = all_features.len();

    println!();
    println!("   ✅ Feature extraction complete!");
    println!("      ├─ Total features: {}", n_features);
    println!("      ├─ Time: {:.2}s ({:.1} files/sec)",
             extract_time.as_secs_f64(),
             flac_files.len() as f64 / extract_time.as_secs_f64());
    println!();

    if n_features == 0 {
        return Err("No features extracted. Check audio files and paths.".into());
    }

    // =============================================================================
    // Step 2: Convert to 2D Array for HDBSCAN
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Converting Features to 2D Array                            │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let convert_start = Instant::now();

    let n_dims = 15; // 15D Goldilocks features
    let mut feature_matrix = ndarray::Array2::zeros((n_features, n_dims));

    for (i, feat) in all_features.iter().enumerate() {
        for (j, &val) in feat.features.iter().enumerate() {
            if j < n_dims {
                feature_matrix[[i, j]] = val;
            }
        }
    }

    println!("   ✅ Converted to {}x{} array in {:.2}s",
             n_features, n_dims, convert_start.elapsed().as_secs_f64());
    println!();

    // =============================================================================
    // Step 3: Save Features (Checkpoint)
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Saving Feature Checkpoint                                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    fs::create_dir_all(&results_dir)?;

    let features_path = results_dir.join("marmoset_15d_features.bincode");
    let file_names_path = results_dir.join("marmoset_file_names.json");

    // Save features
    let serializable_features: Vec<SerializableFeatures> = all_features
        .iter()
        .map(|f| SerializableFeatures {
            file_name: f.file_name.clone(),
            call_type: f.call_type.clone(),
            phrase_index: f.phrase_index,
            features: f.features.clone(),
            duration_ms: f.duration_ms,
        })
        .collect();

    let features_data = bincode::serialize(&serializable_features)?;
    fs::write(&features_path, &features_data)?;

    // Save file names for sequence reconstruction
    let file_names_json = serde_json::to_string_pretty(&all_file_names)?;
    fs::write(&file_names_path, &file_names_json)?;

    println!("   💾 Features saved: {} ({} MB)",
             features_path.display(),
             features_data.len() / 1_048_576);
    println!("   💾 File names saved: {}", file_names_path.display());
    println!();

    // =============================================================================
    // Step 4: HDBSCAN Clustering - Discover Discrete Symbols
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: HDBSCAN Clustering - Discovering Discrete Symbols        │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    // HDBSCAN configuration for marmoset vocalizations
    // min_cluster_size: Controls minimum cluster size (higher = fewer, larger clusters)
    // min_samples: Controls cluster density (higher = more conservative clustering)
    let min_cluster_size = (n_features as f64).sqrt().max(5.0) as usize;
    let min_samples = (min_cluster_size * 3) / 4;

    println!("   🏗️  HDBSCAN Configuration:");
    println!("      ├─ min_cluster_size: {} (minimum phrases per word type)", min_cluster_size);
    println!("      ├─ min_samples: {} (density threshold)", min_samples);
    println!("      ├─ metric: Euclidean distance");
    println!("      └─ Output: Cluster IDs (each ID = discovered word/syllable type)");
    println!();

    let cluster_start = Instant::now();

    let hdbscan = HdbscanClustering::with_metric(
        min_cluster_size,
        min_samples,
        DistanceMetric::Euclidean,
    )?;

    println!("   🔍 Running HDBSCAN...");
    let labels = hdbscan.fit_predict(&feature_matrix)?;

    let cluster_time = cluster_start.elapsed();
    println!("   ✅ Clustering complete in {:.2}s ({:.3}ms per sample)",
             cluster_time.as_secs_f64(),
             cluster_time.as_millis() as f64 / n_features as f64);
    println!();

    // =============================================================================
    // Step 5: Cluster Analysis
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Cluster Analysis - Discovered Vocabulary                 │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let stats = hdbscan.get_cluster_stats(&labels);

    println!("   📊 Clustering Results:");
    println!("      ├─ Total vocalizations analyzed: {}", n_features);
    println!("      ├─ Vocabulary items discovered: {}", stats.n_clusters);
    println!("      ├─ Noise points (unclassified): {}", stats.noise_count);
    println!("      └─ Classified vocalizations: {}", n_features - stats.noise_count);
    println!();

    if !stats.cluster_sizes.is_empty() {
        let total_clustered: usize = stats.cluster_sizes.iter().sum();
        let avg_size = total_clustered as f64 / stats.cluster_sizes.len() as f64;
        let max_size = *stats.cluster_sizes.iter().max().unwrap_or(&0);
        let min_size = *stats.cluster_sizes.iter().min().unwrap_or(&0);

        println!("   📈 Cluster Statistics:");
        println!("      ├─ Size range: {} - {} vocalizations", min_size, max_size);
        println!("      ├─ Average: {:.1} vocalizations per word type", avg_size);
        println!();

        // Top 20 discovered word types
        let mut sorted_clusters: Vec<(i32, usize)> = stats.cluster_sizes.iter()
            .enumerate()
            .map(|(i, size)| (i as i32, *size))
            .collect();
        sorted_clusters.sort_by(|a, b| b.1.cmp(&a.1));

        println!("   🎯 Top 20 Discovered Word Types (by frequency):");
        println!("      ┌──────────┬───────────┬────────┬────────────────┐");
        println!("      │ Cluster │ Occurs    │   %    │ Type           │");
        println!("      ├──────────┼───────────┼────────┼────────────────┤");

        for (i, (cluster_id, size)) in sorted_clusters.iter().take(20).enumerate() {
            let percentage = *size as f64 / n_features as f64 * 100.0;
            let word_type = classify_cluster_by_size(*size);
            println!("      │   {:3}    │    {:5}  │ {:5.1} │ {:14} │",
                     i + 1, cluster_id, percentage, word_type);
        }
        println!("      └──────────┴───────────┴────────┴────────────────┘");
    }
    println!();

    // =============================================================================
    // Step 6: Generate Symbolic Stream Output
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 6: Symbolic Stream Generation                                  │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    // Create symbolic stream: Convert cluster IDs to symbolic representation
    // Using offset IDs starting from 100 for readability (100, 101, 102...)
    let cluster_offset = 100;
    let symbolic_stream: Vec<i32> = labels.iter()
        .map(|&label| if label == -1 { 0 } else { label + cluster_offset })
        .collect();

    // Create symbol-to-count mapping
    let mut symbol_counts: HashMap<i32, usize> = HashMap::new();
    for &symbol in &symbolic_stream {
        *symbol_counts.entry(symbol).or_insert(0) += 1;
    }

    println!("   📝 Symbolic Stream Statistics:");
    println!("      ├─ Total symbols: {}", symbolic_stream.len());
    println!("      ├─ Unique symbols: {}", symbol_counts.len());
    println!("      ├─ Symbol 0 (noise): {} occurrences",
             symbol_counts.get(&0).unwrap_or(&0));
    println!("      └─ Symbol range: {} - {}",
             cluster_offset,
             cluster_offset + stats.n_clusters as i32 - 1);
    println!();

    // Display first 100 symbols of stream
    println!("   🔤 Symbolic Stream Preview (first 100 symbols):");
    print!("      ");
    for (i, symbol) in symbolic_stream.iter().take(100).enumerate() {
        print!("{:3} ", symbol);
        if (i + 1) % 20 == 0 {
            println!();
            print!("      ");
        }
    }
    println!();
    println!();

    // Display stream pattern analysis
    println!("   🔍 Stream Pattern Analysis:");

    // Count consecutive sequences
    let mut sequence_patterns: HashMap<String, usize> = HashMap::new();
    for window in symbolic_stream.windows(3) {
        let pattern = format!("{},{},{}", window[0], window[1], window[2]);
        *sequence_patterns.entry(pattern).or_insert(0) += 1;
    }

    let mut sorted_patterns: Vec<_> = sequence_patterns.into_iter().collect();
    sorted_patterns.sort_by(|a, b| b.1.cmp(&a.1));

    println!("      Top 10 Trigram Patterns:");
    for (i, (pattern, count)) in sorted_patterns.iter().take(10).enumerate() {
        println!("         {}. [{}] occurs {} times", i + 1, pattern, count);
    }
    println!();

    // =============================================================================
    // Step 7: Save Results
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 7: Saving Results                                               │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    // Save cluster labels
    let clusters_path = results_dir.join("hdbscan_clusters.json");
    let clusters_output = serde_json::json!({
        "metadata": {
            "dataset": "marmoset_vocalizations",
            "n_files": flac_files.len(),
            "n_features": n_features,
            "n_dims": n_dims,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
        },
        "clustering": {
            "n_clusters": stats.n_clusters,
            "noise_count": stats.noise_count,
            "cluster_sizes": stats.cluster_sizes,
            "labels": labels,
            "clustering_time_sec": cluster_time.as_secs_f64(),
        },
        "symbolic_stream": {
            "cluster_offset": cluster_offset,
            "stream": symbolic_stream,
            "symbol_counts": symbol_counts,
        }
    });

    fs::write(&clusters_path, serde_json::to_string_pretty(&clusters_output)?)?;
    println!("   💾 Clusters saved: {}", clusters_path.display());

    // Save pure symbolic stream (just the sequence)
    let stream_path = results_dir.join("symbolic_stream.txt");
    let stream_text: String = symbolic_stream.iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");
    fs::write(&stream_path, &stream_text)?;
    println!("   💾 Symbolic stream saved: {}", stream_path.display());

    // Save human-readable symbolic stream with file names
    let readable_path = results_dir.join("symbolic_stream_readable.csv");
    let mut readable_content = String::from("file_name,call_type,cluster_id,symbol\n");
    for (file_info, label) in all_features.iter().zip(labels.iter()) {
        let symbol = if *label == -1 { 0 } else { *label + cluster_offset };
        readable_content.push_str(&format!("{},{},{},{}\n",
            file_info.file_name, file_info.call_type, label, symbol));
    }
    fs::write(&readable_path, &readable_content)?;
    println!("   💾 Readable stream saved: {}", readable_path.display());
    println!();

    // =============================================================================
    // Summary
    // =============================================================================

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    PHASE 0 COMPLETE                                 ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                                                                   ║");
    println!("║  ✅ Raw audio converted to symbolic stream                        ║");
    println!("║                                                                   ║");
    println!("║  📊 SUMMARY:                                                       ║");
    println!("║     • Input: {} FLAC files                                      ║", flac_files.len());
    println!("║     • Features: 15D Goldilocks Subset (RFE-optimized)            ║");
    println!("║     • Vocabulary items discovered: {}                          ║", stats.n_clusters);
    println!("║     • Noise points: {} ({:.1}%)                                    ║",
             stats.noise_count,
             stats.noise_count as f64 / n_features as f64 * 100.0);
    println!("║                                                                   ║");
    println!("║  📁 OUTPUT FILES:                                                   ║");
    println!("║     • {:50} ║", features_path.file_name().unwrap_or_default().to_string_lossy());
    println!("║     • {:50} ║", clusters_path.file_name().unwrap_or_default().to_string_lossy());
    println!("║     • {:50} ║", stream_path.file_name().unwrap_or_default().to_string_lossy());
    println!("║     • {:50} ║", readable_path.file_name().unwrap_or_default().to_string_lossy());
    println!("║                                                                   ║");
    println!("║  🎯 SYMBOLIC STREAM FORMAT:                                          ║");
    println!("║     • Each cluster ID + {} = discovered word type                 ║", cluster_offset);
    println!("║     • Example: [101, 105, 101, 105, 200, ...]                    ║");
    println!("║     • 0 = noise (unclassified)                                      ║");
    println!("║                                                                   ║");
    println!("║  🚀 NEXT STEPS:                                                     ║");
    println!("║     • Phase 1: Analyze n-gram distributions                       ║");
    println!("║     • Phase 2: Discover syntax rules                                   ║");
    println!("║     • Phase 3: Build grammar model                                     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Recursively find all FLAC files in a directory
fn discover_flac_files(dir: &Path) -> Result<Vec<(PathBuf, CallType)>, Box<dyn std::error::Error>> {
    let mut flac_files = Vec::new();
    let entries = fs::read_dir(dir)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            // Recurse into subdirectory
            flac_files.extend(discover_flac_files(&path)?);
        } else if path.is_file() {
            // Check if it's a FLAC file
            if let Some(ext) = path.extension() {
                if ext.to_string_lossy().to_lowercase() == "flac" {
                    let filename = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("");

                    let call_type = CallType::from_filename(filename);
                    if call_type != CallType::Unknown {
                        flac_files.push((path, call_type));
                    }
                }
            }
        }
    }

    Ok(flac_files)
}

/// Load a single FLAC file
fn load_flac_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("flac");

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("No valid audio track found")?;

    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let n_channels = decoder.codec_params().channels.map_or(1, |ch| ch.count());

    let mut audio_samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };

        let decoded = decoder.decode(&packet)?;

        match decoded {
            AudioBufferRef::F32(buf) => {
                for ch in 0..n_channels {
                    audio_samples.extend_from_slice(buf.chan(ch));
                }
            }
            AudioBufferRef::S16(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i16::MAX as f32));
                }
            }
            AudioBufferRef::S24(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|s| {
                        let raw = s.0 as f32;
                        raw / (i32::MAX as f32 / 256.0)
                    }));
                }
            }
            AudioBufferRef::S32(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i32::MAX as f32));
                }
            }
            AudioBufferRef::U8(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| (s as f32 - 128.0) / 128.0));
                }
            }
            _ => return Err("Unsupported audio format".into()),
        }
    }

    Ok(audio_samples)
}
