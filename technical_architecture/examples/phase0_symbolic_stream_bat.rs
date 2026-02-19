// Phase 0: Symbolic Stream Generation for Egyptian Fruit Bat
//
// This is the prerequisite analysis that converts raw audio into a sequence of
// discrete symbols (Cluster IDs) using HDBSCAN clustering.
//
// Input:  Corpus of audio files (91,080 WAV files)
// Feature Extraction: extract_30d_features() via MicroDynamicsExtractor
// Discovery: HDBSCAN (hierarchical density-based clustering)
// Output: A long sequence of Cluster IDs representing discovered "words" or "syllables"
//
// Usage: cargo run --release --example phase0_symbolic_stream_bat

use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuration
    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = data_dir.join("audio");
    let results_dir = data_dir.join("phase0_symbolic_stream_results");

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║         Phase 0: Symbolic Stream Generation - Egyptian Fruit Bat         ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  GOAL: Convert raw audio → discrete symbol sequence (Cluster IDs)        ║");
    println!("║                                                                           ║");
    println!("║  Input:  91,080 WAV files (individual vocalizations)                     ║");
    println!("║  Method: HDBSCAN clustering on 30D features                              ║");
    println!("║  Output: Symbolic stream [101, 105, 101, 105, 200, 101, 105, 300...]     ║");
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
    // Step 1: Feature Extraction (30D MicroDynamics)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Feature Extraction - 30D MicroDynamics                          │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("   📊 Extracting 30D features from audio files...");
    println!("   └─ Features: Time-domain + Frequency-domain + Temporal dynamics");
    println!();

    let extract_start = Instant::now();

    // Process files in batches for memory efficiency
    let batch_size = 10000; // Process 10K files at a time
    let mut all_features: Vec<ExtractedFeatures> = Vec::new();
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

        let batch_features = extract_features_from_files_batch(batch_files.clone())?;
        let valid_count = batch_features.len();

        for feat in batch_features {
            all_file_names.push(feat.file_name.clone());
            all_features.push(feat);
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
        for (j, &val) in feat.features.iter().enumerate() {
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
    println!();

    // ========================================================================
    // Step 3: Save Features (Checkpoint)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Saving Feature Checkpoint                                       │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let features_path = results_dir.join("bat_30d_features.bincode");
    let file_names_path = results_dir.join("bat_file_names.json");

    // Save features
    let serializable_features: Vec<SerializableFeatures> = all_features
        .iter()
        .map(|f| SerializableFeatures {
            file_name: f.file_name.clone(),
            features: f.features.clone(),
            duration_ms: f.duration_ms,
        })
        .collect();

    let features_data = bincode::serialize(&serializable_features)?;
    fs::write(&features_path, &features_data)?;

    // Save file names for sequence reconstruction
    let file_names_json = serde_json::to_string_pretty(&all_file_names)?;
    fs::write(&file_names_path, &file_names_json)?;

    println!(
        "   💾 Features saved: {} ({} MB)",
        features_path.display(),
        features_data.len() / 1_048_576
    );
    println!("   💾 File names saved: {}", file_names_path.display());
    println!();

    // ========================================================================
    // Step 4: HDBSCAN Clustering - Discover Discrete Symbols
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: HDBSCAN Clustering - Discovering Discrete Symbols                │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // HDBSCAN configuration for bat vocalizations
    // min_cluster_size: Controls minimum cluster size (higher = fewer, larger clusters)
    // min_samples: Controls cluster density (higher = more conservative clustering)
    let min_cluster_size = 100; // Larger for large datasets
    let min_samples = 30;

    println!("   🏗️  HDBSCAN Configuration:");
    println!(
        "      ├─ min_cluster_size: {} (minimum phrases per word type)",
        min_cluster_size
    );
    println!("      ├─ min_samples: {} (density threshold)", min_samples);
    println!("      ├─ Algorithm: Hierarchical Density-Based");
    println!("      └─ Output: Cluster IDs (each ID = discovered word/syllable type)");
    println!();

    let cluster_start = Instant::now();

    let hdbscan =
        technical_architecture::hdbscan::HdbscanClustering::new(min_cluster_size, min_samples)?;

    println!("   🔍 Running HDBSCAN...");
    let labels = hdbscan.fit_predict(&feature_matrix)?;

    let cluster_time = cluster_start.elapsed();
    println!(
        "   ✅ Clustering complete in {:.2}s ({:.3}ms per sample)",
        cluster_time.as_secs_f64(),
        cluster_time.as_millis() as f64 / n_features as f64
    );
    println!();

    // ========================================================================
    // Step 5: Cluster Analysis
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Cluster Analysis - Discovered Vocabulary                         │");
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
    // Step 6: Generate Symbolic Stream Output
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 6: Symbolic Stream Generation                                       │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Create symbolic stream: Convert cluster IDs to symbolic representation
    // Using offset IDs starting from 100 for readability (100, 101, 102...)
    let cluster_offset = 100;
    let symbolic_stream: Vec<i32> = labels
        .iter()
        .map(|&label| {
            if label == -1 {
                0
            } else {
                label + cluster_offset
            }
        })
        .collect();

    // Create symbol-to-count mapping
    let mut symbol_counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    for &symbol in &symbolic_stream {
        *symbol_counts.entry(symbol).or_insert(0) += 1;
    }

    println!("   📝 Symbolic Stream Statistics:");
    println!("      ├─ Total symbols: {}", symbolic_stream.len());
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

    // Display first 100 symbols of the stream
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
    let mut sequence_patterns: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
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

    // ========================================================================
    // Step 7: Save Results
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 7: Saving Results                                                   │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Save cluster labels
    let clusters_path = results_dir.join("hdbscan_clusters.json");
    let clusters_output = serde_json::json!({
        "metadata": {
            "dataset": "egyptian_fruit_bat",
            "n_files": total_files,
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

    fs::write(
        &clusters_path,
        serde_json::to_string_pretty(&clusters_output)?,
    )?;
    println!("   💾 Clusters saved: {}", clusters_path.display());

    // Save pure symbolic stream (just the sequence)
    let stream_path = results_dir.join("symbolic_stream.txt");
    let stream_text: String = symbolic_stream
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");
    fs::write(&stream_path, &stream_text)?;
    println!("   💾 Symbolic stream saved: {}", stream_path.display());

    // Save human-readable symbolic stream with file names
    let readable_path = results_dir.join("symbolic_stream_readable.csv");
    let mut readable_content = String::from("file_name,cluster_id,symbol\n");
    for (i, (file_name, &label)) in all_file_names.iter().zip(labels.iter()).enumerate() {
        let symbol = if label == -1 {
            0
        } else {
            label + cluster_offset
        };
        readable_content.push_str(&format!("{},{},{}\n", file_name, label, symbol));
    }
    fs::write(&readable_path, &readable_content)?;
    println!("   💾 Readable stream saved: {}", readable_path.display());
    println!();

    // ========================================================================
    // Summary
    // ========================================================================

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PHASE 0 COMPLETE                                      ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  ✅ Raw audio converted to symbolic stream                                ║");
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
    println!("║                                                                           ║");
    println!("║  📁 OUTPUT FILES:                                                         ║");
    println!(
        "║     • {:50}                                              ║",
        features_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    );
    println!(
        "║     • {:50}                                              ║",
        clusters_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    );
    println!(
        "║     • {:50}                                                 ║",
        stream_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    );
    println!(
        "║     • {:50}                                        ║",
        readable_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    );
    println!("║                                                                           ║");
    println!("║  🎯 SYMBOLIC STREAM FORMAT:                                                ║");
    println!(
        "║     • Each cluster ID + {} = discovered word type                       ║",
        cluster_offset
    );
    println!("║     • Example: [101, 105, 101, 105, 200, ...]                             ║");
    println!("║     • 0 = noise (unclassified)                                           ║");
    println!("║                                                                           ║");
    println!("║  🚀 NEXT STEPS:                                                           ║");
    println!("║     • Phase 1: Analyze n-gram distributions                              ║");
    println!("║     • Phase 2: Discover syntax rules                                     ║");
    println!("║     • Phase 3: Build grammar model                                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    Ok(())
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Clone)]
struct ExtractedFeatures {
    file_name: String,
    features: Vec<f64>, // 30D features
    duration_ms: f64,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SerializableFeatures {
    file_name: String,
    features: Vec<f64>,
    duration_ms: f64,
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

fn extract_features_from_files_batch(
    file_names: Vec<&String>,
) -> Result<Vec<ExtractedFeatures>, Box<dyn std::error::Error>> {
    use rayon::prelude::*;

    let audio_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio");

    let features: Vec<_> = file_names
        .par_iter()
        .filter_map(|file_name| {
            match extract_single_feature(&audio_dir.join(file_name), file_name) {
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

fn extract_single_feature(
    file_path: &Path,
    file_name: &str,
) -> Result<ExtractedFeatures, Box<dyn std::error::Error>> {
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

    Ok(ExtractedFeatures {
        file_name: file_name.to_string(),
        features: features_30d,
        duration_ms,
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
