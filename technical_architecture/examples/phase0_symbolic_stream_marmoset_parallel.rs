// Phase 0: Symbolic Stream Generation for Marmoset - PARALLEL VERSION
//
// This is prerequisite analysis that converts raw audio into a sequence of
// discrete symbols (Cluster IDs) using HDBSCAN clustering.
//
// Input:  Corpus of FLAC files (marmoset vocalizations)
// Feature Extraction: extract_15d_marmoset() via MicroDynamicsExtractor
// Discovery: HDBSCAN (hierarchical density-based clustering)
// Output: A long sequence of Cluster IDs representing discovered "words" or "syllables"
//
// Usage: cargo run --release --example phase0_symbolic_stream_marmoset_parallel [--limit N] [--resume]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;

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

/// Progress tracking for parallel processing
#[derive(Clone)]
struct ProgressTracker {
    total: usize,
    processed: Arc<Mutex<usize>>,
    start_time: Instant,
}

impl ProgressTracker {
    fn new(total: usize) -> Self {
        Self {
            total,
            processed: Arc::new(Mutex::new(0)),
            start_time: Instant::now(),
        }
    }

    fn increment(&self) {
        let mut count = self.processed.lock().unwrap();
        *count += 1;
        let current = *count;

        // Print progress every 100 items or at completion
        if current % 100 == 0 || current == self.total {
            let elapsed = self.start_time.elapsed().as_secs_f64();
            let rate = current as f64 / elapsed;
            let remaining = if current < self.total {
                let remaining_count = self.total - current;
                remaining_count as f64 / rate
            } else {
                0.0
            };
            println!("   🔄 Processed {}/{} ({:.1}%) | {:.1} files/sec | ETA: {:.1}s",
                     current, self.total,
                     current as f64 / self.total as f64 * 100.0,
                     rate, remaining);
        }
    }

    fn count(&self) -> usize {
        *self.processed.lock().unwrap()
    }
}

// =============================================================================
// Main Function
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║    Phase 0: Symbolic Stream - Marmoset (PARALLEL)              ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                                                                   ║");
    println!("║  🚀 PARALLEL PROCESSING ENABLED                                   ║");
    println!("║  GOAL: Convert raw audio → discrete symbol sequence (Cluster IDs)║");
    println!("║                                                                   ║");
    println!("║  Input:  FLAC files (marmoset vocalizations)                    ║");
    println!("║  Method: HDBSCAN clustering on 15D Goldilocks features         ║");
    println!("║  Output: Symbolic stream [101, 105, 101, 105, 200, ...]         ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let mut vocalizations_dir = PathBuf::from("/home/sheel/birdsong_analysis/data/Vocalizations");
    let mut results_dir = PathBuf::from("/mnt/c/Users/sheel/Desktop/src/marmoset_phase0_results");
    let mut limit = None;
    let mut resume = false;

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
            "--resume" => {
                resume = true;
                println!("🔄 Resume mode: will load from checkpoint if available");
            }
            arg if i == args.len() - 1 && !arg.starts_with("--") => {
                vocalizations_dir = PathBuf::from(arg);
            }
            _ => {}
        }
        i += 1;
    }

    let sample_rate = 96000;
    let checkpoint_path = results_dir.join("phase0_checkpoint.bincode");

    // Detect CPU count for parallel processing
    let num_cpus = num_cpus::get();
    println!("   💻 Detected {} CPU cores", num_cpus);
    let parallel_chunks = num_cpus * 4; // Process 4 batches per CPU for better load balancing
    println!("   ⚡ Using {} parallel chunks for processing", parallel_chunks);
    println!();

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

    // =============================================================================
    // Checkpoint Loading
    // =============================================================================

    let mut all_features: Vec<ExtractedFeatures> = Vec::new();
    let mut all_file_names: Vec<String> = Vec::new();
    let mut start_index = 0;

    if resume && checkpoint_path.exists() {
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│ Checkpoint: Loading Previous Results                                │");
        println!("└─────────────────────────────────────────────────────────────────┘");
        println!();

        match load_checkpoint(&checkpoint_path) {
            Ok(checkpoint) => {
                println!("   ✅ Checkpoint loaded successfully!");
                println!("      ├─ Processed at: {}", checkpoint.processed_at);
                println!("      ├─ Previous files: {}", checkpoint.all_features.len());
                println!("      └─ Total files in checkpoint: {}", checkpoint.total_files);

                // Convert to internal format
                for feat in checkpoint.all_features {
                    all_file_names.push(feat.file_name.clone());
                    all_features.push(ExtractedFeatures {
                        file_name: feat.file_name,
                        call_type: feat.call_type,
                        phrase_index: feat.phrase_index,
                        features: feat.features,
                        duration_ms: feat.duration_ms,
                    });
                }

                start_index = all_features.len();

                // Check if we need to continue processing
                if start_index >= flac_files.len() {
                    println!();
                    println!("   ✅ All files were already processed!");
                    println!("   Proceeding to clustering step...");
                    println!();
                } else {
                    println!();
                    println!("   🔄 Resuming from file {} of {} ({} remaining)...",
                             start_index + 1, flac_files.len(), flac_files.len() - start_index);
                    println!();
                }
            }
            Err(e) => {
                println!("   ⚠️  Failed to load checkpoint: {}", e);
                println!("   Starting fresh...");
                println!();
            }
        }
    } else if resume {
        println!("   ℹ️  Resume requested but no checkpoint found. Starting fresh...");
        println!();
    }

    // Count by call type (only for new files if resuming)
    let range_start = if start_index > 0 { start_index } else { 0 };
    println!("   📊 Call Type Distribution:");
    let mut call_type_counts: HashMap<CallType, usize> = HashMap::new();
    for (path, _) in flac_files[range_start..].iter() {
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        let call_type = CallType::from_filename(filename);
        *call_type_counts.entry(call_type).or_insert(0) += 1;
    }
    for (call_type, count) in call_type_counts.iter() {
        println!("      • {}: {} files", call_type.name(), count);
    }
    println!();

    // =============================================================================
    // Step 1: PARALLEL Feature Extraction
    // =============================================================================

    if start_index < flac_files.len() {
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│ Step 1: PARALLEL Feature Extraction - 15D Goldilocks Subset │");
        println!("└─────────────────────────────────────────────────────────────────┘");
        println!();

        println!("   ⚡ Extracting 15D Goldilocks features in PARALLEL...");
        println!("      └─ Features: RFE-optimized for marmoset call types");
        println!("      └─ Chunks: {} (for load balancing)", parallel_chunks);
        println!();

        let extract_start = Instant::now();

        // Get files that need processing
        let files_to_process: Vec<_> = flac_files[start_index..].to_vec();

        // Create progress tracker
        let tracker = ProgressTracker::new(files_to_process.len());

        // Process files in parallel chunks
        let batch_size = (files_to_process.len() + parallel_chunks - 1) / parallel_chunks;

        let chunk_results: Vec<Vec<SerializableFeatures>> = files_to_process
            .par_chunks(batch_size)
            .map(|chunk| {
                let mut local_features = Vec::new();
                for (file_path, call_type) in chunk {
                    let filename = file_path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown");

                    match load_flac_file(file_path) {
                        Ok(audio) => {
                            let extractor = MicroDynamicsExtractor::new(sample_rate);
                            match extractor.extract_15d_marmoset(&audio) {
                                Ok(features) => {
                                    let feature_vec = features.to_array().to_vec();

                                    local_features.push(SerializableFeatures {
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
                    tracker.increment();
                }
                local_features
            })
            .collect();

        // Merge results from all chunks
        for mut chunk_features in chunk_results {
            for feat in chunk_features.drain(..) {
                all_file_names.push(feat.file_name.clone());
                all_features.push(ExtractedFeatures {
                    file_name: feat.file_name,
                    call_type: feat.call_type,
                    phrase_index: feat.phrase_index,
                    features: feat.features,
                    duration_ms: feat.duration_ms,
                });
            }
        }

        let extract_time = extract_start.elapsed();
        let n_features = all_features.len();
        let newly_processed = n_features - start_index;

        println!();
        println!("   ✅ Feature extraction complete!");
        println!("      ├─ Total features: {}", n_features);
        println!("      ├─ Newly processed: {}", newly_processed);
        println!("      ├─ Time: {:.2}s", extract_time.as_secs_f64());
        println!("      ├─ Rate: {:.1} files/sec", newly_processed as f64 / extract_time.as_secs_f64());
        println!("      └─ Speedup: ~{}x vs sequential", num_cpus);
        println!();

        // Save checkpoint after extraction
        println!("┌─────────────────────────────────────────────────────────────────┐");
        println!("│ Checkpoint: Saving Progress                                         │");
        println!("└─────────────────────────────────────────────────────────────────┘");
        println!();

        fs::create_dir_all(&results_dir)?;

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

        let checkpoint_data = CheckpointData {
            all_features: serializable_features,
            all_file_names: all_file_names.clone(),
            total_files: flac_files.len(),
            processed_at: chrono::Utc::now().to_rfc3339(),
        };

        save_checkpoint(&checkpoint_path, &checkpoint_data)?;
        println!("   💾 Checkpoint saved: {}", checkpoint_path.display());
        println!();
    }

    let n_features = all_features.len();

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

    let n_dims = 15;
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
    // Step 3: Save Features
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Saving Feature Checkpoint                                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let features_path = results_dir.join("marmoset_15d_features.bincode");
    let file_names_path = results_dir.join("marmoset_file_names.json");

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

    let file_names_json = serde_json::to_string_pretty(&all_file_names)?;
    fs::write(&file_names_path, &file_names_json)?;

    println!("   💾 Features saved: {} ({} MB)",
             features_path.display(),
             features_data.len() / 1_048_576);
    println!("   💾 File names saved: {}", file_names_path.display());
    println!();

    // =============================================================================
    // Step 4: HDBSCAN Clustering
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: HDBSCAN Clustering - Discovering Discrete Symbols        │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let min_cluster_size = (n_features as f64).sqrt().max(5.0) as usize;
    let min_samples = (min_cluster_size * 3) / 4;

    println!("   🏗️  HDBSCAN Configuration:");
    println!("      ├─ min_cluster_size: {}", min_cluster_size);
    println!("      ├─ min_samples: {}", min_samples);
    println!("      └─ metric: Euclidean distance");
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
    println!("   ✅ Clustering complete in {:.2}s", cluster_time.as_secs_f64());
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
    println!("      ├─ Total vocalizations: {}", n_features);
    println!("      ├─ Vocabulary items: {}", stats.n_clusters);
    println!("      ├─ Noise points: {}", stats.noise_count);
    println!("      └─ Classified: {}", n_features - stats.noise_count);
    println!();

    // =============================================================================
    // Step 6: Generate Symbolic Stream
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 6: Symbolic Stream Generation                                  │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let cluster_offset = 100;
    let symbolic_stream: Vec<i32> = labels.iter()
        .map(|&label| if label == -1 { 0 } else { label + cluster_offset })
        .collect();

    println!("   📝 Symbolic Stream Statistics:");
    println!("      ├─ Total symbols: {}", symbolic_stream.len());
    println!("      ├─ Unique symbols: {}", symbolic_stream.iter().collect::<std::collections::HashSet<_>>().len());
    println!("      └─ Symbol range: {} - {}",
             cluster_offset,
             cluster_offset + stats.n_clusters as i32 - 1);
    println!();

    // =============================================================================
    // Step 7: Save Results
    // =============================================================================

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Step 7: Saving Results                                               │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let clusters_path = results_dir.join("hdbscan_clusters.json");
    let clusters_output = serde_json::json!({
        "metadata": {
            "dataset": "marmoset_vocalizations",
            "n_files": flac_files.len(),
            "n_features": n_features,
            "n_dims": n_dims,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "parallel_mode": true,
            "num_cpus": num_cpus,
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
        }
    });

    fs::write(&clusters_path, serde_json::to_string_pretty(&clusters_output)?)?;
    println!("   💾 Clusters saved: {}", clusters_path.display());

    let stream_path = results_dir.join("symbolic_stream.txt");
    let stream_text: String = symbolic_stream.iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");
    fs::write(&stream_path, &stream_text)?;
    println!("   💾 Symbolic stream saved: {}", stream_path.display());

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
    println!("║                    PHASE 0 COMPLETE (PARALLEL)                    ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                                                                   ║");
    println!("║  ✅ Raw audio converted to symbolic stream                        ║");
    println!("║  ⚡ Parallel processing enabled (~{}x speedup)                      ║", num_cpus);
    println!("║                                                                   ║");
    println!("║  📊 SUMMARY:                                                       ║");
    println!("║     • Input: {} FLAC files                                      ║", flac_files.len());
    println!("║     • Features: 15D Goldilocks Subset                            ║");
    println!("║     • Vocabulary items: {}                                       ║", stats.n_clusters);
    println!("║                                                                   ║");
    println!("║  🔄 CHECKPOINT:                                                    ║");
    println!("║     • Use --resume to continue from checkpoint                    ║");
    println!("║     • Checkpoint: {:38} ║", checkpoint_path.file_name().unwrap_or_default().to_string_lossy());
    println!("║                                                                   ║");
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
            flac_files.extend(discover_flac_files(&path)?);
        } else if path.is_file() {
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

/// Save checkpoint data
fn save_checkpoint(path: &Path, data: &CheckpointData) -> Result<(), Box<dyn std::error::Error>> {
    let encoded = bincode::serialize(data)?;
    fs::write(path, &encoded)?;
    Ok(())
}

/// Load checkpoint data
fn load_checkpoint(path: &Path) -> Result<CheckpointData, Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    let decoded: CheckpointData = bincode::deserialize(&data)?;
    Ok(decoded)
}
