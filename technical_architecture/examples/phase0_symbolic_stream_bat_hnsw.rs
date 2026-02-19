// Phase 0: Symbolic Stream Generation for Egyptian Fruit Bat - HNSW-OPTIMIZED VERSION
//
// This version implements the "Approximate Nearest Neighbors" fix using HNSW
// (Hierarchical Navigable Small World) graphs for O(log n) memory-efficient clustering.
//
// KEY FEATURES:
// - Uses HNSW instead of exact KNN for 10-100x memory reduction
// - Includes timestamp tracking for each discovered word (for later audio extraction)
// - Outputs symbolic stream + cluster metadata + timestamps in JSON format
//
// Usage: cargo run --release --example phase0_symbolic_stream_bat_hnsw

use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuration
    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = data_dir.join("audio");
    let results_dir = data_dir.join("phase0_symbolic_stream_results_hnsw");

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Phase 0: Symbolic Stream - HNSW-OPTIMIZED VERSION                    ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  Using HNSW (Hierarchical Navigable Small World) for O(log n) ANN        ║");
    println!("║  Memory: O(n log n) instead of O(n²) - ~10-100x memory reduction         ║");
    println!("║  Includes timestamp tracking for audio extraction                         ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Create results directory
    fs::create_dir_all(&results_dir)?;

    // ========================================================================
    // Step 0: Dataset Overview
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Dataset Overview                                                         │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    if !audio_dir.exists() {
        println!("❌ Audio directory not found: {}", audio_dir.display());
        return Err("Dataset not found".into());
    }

    let wav_files = discover_wav_files(&audio_dir)?;
    let total_files = wav_files.len();

    println!("   📂 Audio Directory: {}", audio_dir.display());
    println!("   🔢 Total WAV files: {}", total_files);
    println!("   💾 Results Directory: {}", results_dir.display());
    println!();

    // ========================================================================
    // Step 1: Feature Extraction with Metadata Tracking
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Feature Extraction with Timestamp Tracking                       │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   📊 Extracting 30D features + metadata from audio files...");
    println!("   └─ Features: Time-domain + Frequency-domain + Temporal dynamics");
    println!();

    let extract_start = Instant::now();

    // Smaller batch size for memory efficiency
    let batch_size = 5000; // Process 5K files at a time
    let mut all_metadata: Vec<AudioMetadata> = Vec::new();
    let mut all_features: Vec<Vec<f64>> = Vec::new();
    let mut all_file_names: Vec<String> = Vec::new();

    for (batch_idx, batch_start) in (0..total_files).step_by(batch_size).enumerate() {
        let batch_end = (batch_start + batch_size).min(total_files);
        let batch_files: Vec<_> = wav_files[batch_start..batch_end].iter().collect();

        println!(
            "   🔄 Processing batch {} (files {}-{})...",
            batch_idx + 1,
            batch_start,
            batch_end - 1
        );

        let batch_results = extract_features_with_metadata_batch(batch_files.clone())?;
        let valid_count = batch_results.len();

        for result in batch_results {
            all_file_names.push(result.file_name.clone());
            all_features.push(result.features);
            all_metadata.push(result.metadata);
        }

        println!(
            "      └─ Extracted {} features ({:.1}% success rate)",
            valid_count,
            valid_count as f64 / batch_files.len() as f64 * 100.0
        );
    }

    let extract_time = extract_start.elapsed();
    let n_features = all_features.len();

    println!();
    println!("   ✅ Feature extraction complete!");
    println!("      ├─ Total features: {}", n_features);
    println!(
        "      ├─ Time: {:.2}s ({:.1} files/sec)",
        extract_time.as_secs_f64(),
        total_files as f64 / extract_time.as_secs_f64()
    );
    println!();

    // ========================================================================
    // Step 2: Convert to 2D Array for HDBSCAN
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Converting Features to 2D Array                                  │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let convert_start = Instant::now();

    let n_dims = 30; // 30D base features
    let mut feature_matrix = ndarray::Array2::zeros((n_features, n_dims));

    for (i, feat) in all_features.iter().enumerate() {
        for (j, &val) in feat.iter().enumerate() {
            if j < n_dims {
                feature_matrix[[i, j]] = val;
            }
        }
    }

    println!(
        "   ✅ Converted to {}x{} array in {:.2}s",
        n_features,
        n_dims,
        convert_start.elapsed().as_secs_f64()
    );
    println!(
        "   📊 Memory: ~{} MB for feature matrix",
        (n_features * n_dims * 8) / 1_048_576
    );
    println!();

    // ========================================================================
    // Step 3: HDBSCAN Clustering with HNSW Optimization
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: HDBSCAN Clustering - HNSW OPTIMIZED                              │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // HDBSCAN configuration for bat vocalizations
    let min_cluster_size = 100; // Larger for large datasets
    let min_samples = 30;

    println!("   🏗️  HDBSCAN Configuration:");
    println!(
        "      ├─ min_cluster_size: {} (minimum phrases per word type)",
        min_cluster_size
    );
    println!("      ├─ min_samples: {} (density threshold)", min_samples);
    println!("      ├─ Algorithm: Hierarchical Density-Based with HNSW ANN");
    println!("      └─ Output: Cluster IDs (each ID = discovered word/syllable type)");
    println!();

    let cluster_start = Instant::now();

    // Use HNSW-optimized HDBSCAN
    let hdbscan =
        technical_architecture::hdbscan::HdbscanClustering::new(min_cluster_size, min_samples)?;

    println!("   🔍 Running HDBSCAN with HNSW optimization...");
    println!("      └─ Memory-efficient O(log n) nearest neighbor queries");
    let labels = hdbscan.fit_predict_hnsw(&feature_matrix)?;

    let cluster_time = cluster_start.elapsed();
    println!(
        "   ✅ Clustering complete in {:.2}s ({:.3}ms per sample)",
        cluster_time.as_secs_f64(),
        cluster_time.as_millis() as f64 / n_features as f64
    );
    println!();

    // ========================================================================
    // Step 4: Cluster Analysis
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Cluster Analysis - Discovered Vocabulary                         │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let stats = hdbscan.get_cluster_stats(&labels);

    println!("   📊 Clustering Results:");
    println!("      ├─ Total phrases analyzed: {}", n_features);
    println!("      ├─ Vocabulary items discovered: {}", stats.n_clusters);
    println!(
        "      ├─ Noise points (unclassified): {}",
        stats.noise_count
    );
    println!(
        "      └─ Classified phrases: {}",
        n_features - stats.noise_count
    );
    println!();

    if !stats.cluster_sizes.is_empty() {
        let total_clustered: usize = stats.cluster_sizes.iter().sum();
        let avg_size = total_clustered as f64 / stats.cluster_sizes.len() as f64;
        let max_size = *stats.cluster_sizes.iter().max().unwrap_or(&0);
        let min_size = *stats.cluster_sizes.iter().min().unwrap_or(&0);

        println!("   📈 Cluster Statistics:");
        println!("      ├─ Size range: {} - {} phrases", min_size, max_size);
        println!("      ├─ Average: {:.1} phrases per word type", avg_size);
        println!();

        // Top 20 discovered word types
        let mut sorted_clusters: Vec<(i32, usize)> = stats
            .cluster_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| (i as i32, size))
            .collect();
        sorted_clusters.sort_by(|a, b| b.1.cmp(&a.1));

        println!("   🎯 Top 20 Discovered Word Types (by frequency):");
        println!("      ┌──────────┬──────────────┬──────────┬────────────┐");
        println!("      │ Cluster  │ Occurrences  │   %      │ Type       │");
        println!("      ├──────────┼──────────────┼──────────┼────────────┤");

        for (i, (cluster_id, size)) in sorted_clusters.iter().take(20).enumerate() {
            let percentage = size.clone() as f64 / n_features as f64 * 100.0;
            let word_type = classify_cluster_by_size(*size);
            println!(
                "      │   {:3}    │    {:5}     │  {:5.2}  │ {:10} │",
                i + 1,
                size,
                percentage,
                word_type
            );
        }
        println!("      └──────────┴──────────────┴──────────┴────────────┘");
    }
    println!();

    // ========================================================================
    // Step 5: Generate Symbolic Stream with Timestamps
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Symbolic Stream Generation with Timestamps                     │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Create symbolic stream: Convert cluster IDs to symbolic representation
    // Using offset IDs starting from 100 for readability (100, 101, 102...)
    let cluster_offset = 100;

    // Build rich symbolic stream entries with metadata
    let mut symbolic_stream_entries: Vec<SymbolicStreamEntry> = Vec::new();
    let mut symbol_counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();

    for (i, (((file_name, &label), metadata))) in all_file_names
        .iter()
        .zip(labels.iter())
        .zip(all_metadata.iter())
        .enumerate()
    {
        let symbol = if label == -1 {
            0
        } else {
            label + cluster_offset
        };
        *symbol_counts.entry(symbol).or_insert(0) += 1;

        symbolic_stream_entries.push(SymbolicStreamEntry {
            index: i,
            file_name: file_name.clone(),
            cluster_id: label,
            symbol: symbol,
            timestamp_ms: metadata.start_time_ms,
            duration_ms: metadata.duration_ms,
            sample_rate: metadata.sample_rate,
        });
    }

    println!("   📝 Symbolic Stream Statistics:");
    println!("      ├─ Total symbols: {}", symbolic_stream_entries.len());
    println!("      ├─ Unique symbols: {}", symbol_counts.len());
    println!(
        "      ├─ Symbol 0 (noise): {} occurrences",
        symbol_counts.get(&0).unwrap_or(&0)
    );
    println!(
        "      └─ Symbol range: {} - {}",
        cluster_offset,
        cluster_offset + stats.n_clusters as i32 - 1
    );
    println!();

    // Display first 50 symbols of the stream with timestamps
    println!("   🔤 Symbolic Stream Preview (first 50 symbols with timestamps):");
    println!("      ┌──────┬────────────┬─────────┬──────────────┬────────────┐");
    println!("      │ Idx  │ File       │ Symbol  │ Time (ms)    │ Dur (ms)   │");
    println!("      ├──────┼────────────┼─────────┼──────────────┼────────────┤");

    for entry in symbolic_stream_entries.iter().take(50) {
        println!(
            "      │ {:4} │ {:10} │   {:3}  │ {:8}     │ {:8}   │",
            entry.index,
            truncate(&entry.file_name, 10),
            entry.symbol,
            entry.timestamp_ms as i64,
            entry.duration_ms as i64
        );
    }
    println!("      └──────┴────────────┴─────────┴──────────────┴────────────┘");
    println!();

    // ========================================================================
    // Step 6: Save Results with Timestamps
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 6: Saving Results with Full Metadata                               │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Save full results with timestamps
    let results_path = results_dir.join("hnsw_symbolic_stream_with_timestamps.json");
    let results_output = serde_json::json!({
        "metadata": {
            "dataset": "egyptian_fruit_bat",
            "algorithm": "HDBSCAN-HNSW",
            "n_files": total_files,
            "n_features": n_features,
            "n_dims": n_dims,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "hnsw_parameters": {
                "nb_connection": 15,
                "ef_construction": 100,
                "max_layer": 16,
                "ef_search": 50,
                "distance_metric": "L2"
            }
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
            "symbol_counts": symbol_counts,
        },
        "stream_entries": symbolic_stream_entries,
        "timestamp_info": {
            "description": "Each entry includes timestamp_ms for audio extraction",
            "timestamp_origin": "Start time within each WAV file (ms from file start)",
            "use_case": "Load WAV file, seek to timestamp_ms, extract duration_ms"
        }
    });

    fs::write(
        &results_path,
        serde_json::to_string_pretty(&results_output)?,
    )?;
    println!(
        "   💾 Full results with timestamps: {}",
        results_path.display()
    );

    // Save pure symbolic stream (just the sequence)
    let stream_path = results_dir.join("symbolic_stream.txt");
    let stream_text: String = symbolic_stream_entries
        .iter()
        .map(|e| e.symbol.to_string())
        .collect::<Vec<_>>()
        .join(",");
    fs::write(&stream_path, &stream_text)?;
    println!(
        "   💾 Symbolic stream (sequence only): {}",
        stream_path.display()
    );

    // Save timestamp map for audio extraction
    let timestamp_map_path = results_dir.join("timestamp_map.json");
    let timestamp_map: serde_json::Value = serde_json::json!({
        "description": "Map of symbol IDs to audio file locations for extraction",
        "format": "symbol_id -> [{file_name, start_ms, duration_ms}, ...]",
        "data": {
            "by_symbol": build_symbol_to_entries_map(&symbolic_stream_entries),
            "by_file": build_file_to_symbols_map(&symbolic_stream_entries),
        }
    });
    fs::write(
        &timestamp_map_path,
        serde_json::to_string_pretty(&timestamp_map)?,
    )?;
    println!(
        "   💾 Timestamp map for extraction: {}",
        timestamp_map_path.display()
    );
    println!();

    // ========================================================================
    // Summary
    // ========================================================================

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PHASE 0 COMPLETE - HNSW OPTIMIZED                     ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  ✅ Raw audio converted to symbolic stream with timestamps                 ║");
    println!("║                                                                           ║");
    println!("║  📊 SUMMARY:                                                              ║");
    println!(
        "║     • Input:  {} WAV files                                           ║",
        total_files
    );
    println!("║     • Features: 30D MicroDynamics                                     ║");
    println!(
        "║     • Vocabulary items discovered: {}                               ║",
        stats.n_clusters
    );
    println!(
        "║     • Noise points: {} ({:.1}%)                                    ║",
        stats.noise_count,
        stats.noise_count as f64 / n_features as f64 * 100.0
    );
    println!(
        "║     • Clustering time: {:.2}s                                          ║",
        cluster_time.as_secs_f64()
    );
    println!("║                                                                           ║");
    println!("║  🎯 TIMESTAMP INFO:                                                       ║");
    println!("║     • Each discovered word includes:                                     ║");
    println!("║       - file_name: Source WAV file                                      ║");
    println!("║       - timestamp_ms: Start time within file (ms)                        ║");
    println!("║       - duration_ms: Word duration (ms)                                  ║");
    println!("║       - sample_rate: Audio sample rate                                   ║");
    println!("║                                                                           ║");
    println!("║  🔧 AUDIO EXTRACTION:                                                     ║");
    println!("║     1. Load timestamp_map.json                                           ║");
    println!("║     2. Select symbol_id to extract                                       ║");
    println!("║     3. For each entry:                                                   ║");
    println!("║        - Load WAV file                                                   ║");
    println!("║        - Seek to timestamp_ms                                             ║");
    println!("║        - Extract duration_ms samples                                      ║");
    println!("║                                                                           ║");
    println!("║  🚀 NEXT STEPS:                                                           ║");
    println!("║     • Phase 1: Analyze n-gram distributions                              ║");
    println!("║     • Phase 2: Discover syntax rules                                     ║");
    println!("║     • Phase 3: Build grammar model                                       ║");
    println!("║     • Audio Extraction: Use timestamp_map.json                           ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Clone, serde::Serialize)]
struct AudioMetadata {
    start_time_ms: f64,
    duration_ms: f64,
    sample_rate: u32,
}

#[derive(Clone, serde::Serialize)]
struct ExtractedFeatureWithMetadata {
    file_name: String,
    features: Vec<f64>, // 30D features
    metadata: AudioMetadata,
}

#[derive(Clone, serde::Serialize)]
struct SymbolicStreamEntry {
    index: usize,
    file_name: String,
    cluster_id: i32,
    symbol: i32,
    timestamp_ms: f64, // Start time within the WAV file
    duration_ms: f64,  // Duration of this audio segment
    sample_rate: u32,
}

// ============================================================================
// Helper Functions
// ============================================================================

fn discover_wav_files(dir: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut wav_files = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext.eq_ignore_ascii_case("wav") {
                    if let Some(file_name) = path.file_name() {
                        wav_files.push(file_name.to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    wav_files.sort();
    Ok(wav_files)
}

fn extract_features_with_metadata_batch(
    file_names: Vec<&String>,
) -> Result<Vec<ExtractedFeatureWithMetadata>, Box<dyn std::error::Error>> {
    use rayon::prelude::*;

    let audio_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio");

    let features: Vec<_> = file_names
        .par_iter()
        .filter_map(|file_name| {
            match extract_single_feature_with_metadata(&audio_dir.join(file_name), file_name) {
                Ok(f) => Some(f),
                Err(e) => {
                    eprintln!("Warning: Failed to extract from {}: {}", file_name, e);
                    None
                }
            }
        })
        .collect();

    Ok(features)
}

fn extract_single_feature_with_metadata(
    file_path: &Path,
    file_name: &str,
) -> Result<ExtractedFeatureWithMetadata, Box<dyn std::error::Error>> {
    use technical_architecture::MicroDynamicsExtractor;

    // Load WAV file
    let reader = hound::WavReader::open(file_path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    // Read samples
    let audio: Vec<f32> = reader
        .into_samples::<f32>()
        .filter_map(|s| s.ok())
        .collect();

    if audio.is_empty() {
        return Err("No audio samples".into());
    }

    // Convert to mono if stereo
    let audio_mono = if spec.channels == 2 {
        audio.chunks_exact(2).map(|c| (c[0] + c[1]) / 2.0).collect()
    } else {
        audio
    };

    let duration_ms = (audio_mono.len() as f64 / sample_rate as f64) * 1000.0;

    // Extract 30D base features
    let extractor = MicroDynamicsExtractor::new(sample_rate);
    let features_56d = extractor.extract_56d(&audio_mono)?;

    // Convert to 30D Vector30D
    let vector30d = features_56d.base_30d.to_vector30d(
        10000.0, // mean_f0_hz for bat
        duration_ms as f32,
        5000.0, // f0_range_hz for bat
    );

    let features_30d: Vec<f64> = vector30d.to_array().iter().map(|&x| x as f64).collect();

    Ok(ExtractedFeatureWithMetadata {
        file_name: file_name.to_string(),
        features: features_30d,
        metadata: AudioMetadata {
            start_time_ms: 0.0, // Start of file (could be adjusted for segmented audio)
            duration_ms,
            sample_rate,
        },
    })
}

fn classify_cluster_by_size(size: usize) -> &'static str {
    match size {
        s if s >= 5000 => "VERY_COMMON",
        s if s >= 2000 => "COMMON",
        s if s >= 500 => "MODERATE",
        s if s >= 100 => "RARE",
        _ => "VERY_RARE",
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

fn build_symbol_to_entries_map(entries: &[SymbolicStreamEntry]) -> serde_json::Value {
    let mut map: std::collections::HashMap<i32, Vec<serde_json::Value>> =
        std::collections::HashMap::new();

    for entry in entries {
        map.entry(entry.symbol)
            .or_insert_with(Vec::new)
            .push(serde_json::json!({
                "file": entry.file_name,
                "start_ms": entry.timestamp_ms,
                "duration_ms": entry.duration_ms,
            }));
    }

    serde_json::to_value(map).unwrap()
}

fn build_file_to_symbols_map(entries: &[SymbolicStreamEntry]) -> serde_json::Value {
    let mut map: std::collections::HashMap<String, Vec<serde_json::Value>> =
        std::collections::HashMap::new();

    for entry in entries {
        map.entry(entry.file_name.clone())
            .or_insert_with(Vec::new)
            .push(serde_json::json!({
                "symbol": entry.symbol,
                "cluster_id": entry.cluster_id,
                "start_ms": entry.timestamp_ms,
                "duration_ms": entry.duration_ms,
            }));
    }

    serde_json::to_value(map).unwrap()
}
