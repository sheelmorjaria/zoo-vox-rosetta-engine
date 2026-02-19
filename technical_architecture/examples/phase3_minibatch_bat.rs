// Phase 3: MiniBatch K-Means Discovery for Egyptian Fruit Bat
//
// This example extracts features from bat WAV files and runs MiniBatch K-Means
// clustering to discover the vocabulary of bat vocalizations.
//
// MiniBatch K-Means scales linearly O(n) and can process all 91K files efficiently.
//
// Usage: cargo run --release --example phase3_minibatch_bat

use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let audio_dir = data_dir.join("audio");
    let results_dir =
        Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/lexicon_to_syntax_results");
    let features_path = results_dir.join("bat_features.bincode");
    let output_path = results_dir.join("minibatch_clusters.json");

    println!("🦇 Phase 3: MiniBatch K-Means Discovery - Egyptian Fruit Bat");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Audio Directory: {}", audio_dir.display());
    println!("   Features:       {}", features_path.display());
    println!("   Output:         {}", output_path.display());
    println!();

    // Create results directory
    std::fs::create_dir_all(&results_dir)?;

    // ========================================================================
    // Step 1: Extract features from all WAV files
    // ========================================================================

    println!("📂 Step 1: Extracting 56D MicroDynamics features from WAV files...");
    println!();

    let extract_start = Instant::now();

    // Discover all WAV files
    let wav_files = discover_wav_files(&audio_dir)?;
    let n_files = wav_files.len();

    println!("   Found {} WAV files", n_files);
    println!("   Processing ALL files (this will take ~2-3 hours)");
    println!();

    // Process ALL files for full dataset analysis
    let wav_files_all: Vec<&String> = wav_files.iter().collect();

    // Extract features from WAV files
    let features = extract_features_from_files_progress(wav_files_all, n_files)?;

    let extract_time = extract_start.elapsed();
    println!(
        "   └─ Extracted {} features in {:.2}s ({:.1} files/sec)",
        features.len(),
        extract_time.as_secs_f64(),
        features.len() as f64 / extract_time.as_secs_f64()
    );
    println!();

    // ========================================================================
    // Step 2: Save features to disk
    // ========================================================================

    println!("💾 Step 2: Saving features to disk...");
    println!();

    let save_start = Instant::now();

    // Save features as bincode
    let serializable_features: Vec<SerializableFeatures> = features
        .into_iter()
        .map(|f| SerializableFeatures {
            file_name: f.file_name,
            features: f.features,
            duration_ms: f.duration_ms,
            sample_rate: f.sample_rate,
        })
        .collect();

    let features_data = bincode::serialize(&serializable_features)?;
    std::fs::write(&features_path, &features_data)?;

    let save_time = save_start.elapsed();
    println!(
        "   └─ Saved {} features ({} MB) in {:.2}s",
        serializable_features.len(),
        features_data.len() / 1_048_576,
        save_time.as_secs_f64()
    );
    println!();

    // ========================================================================
    // Step 3: Convert to 2D array for clustering
    // ========================================================================

    println!("🔄 Step 3: Converting features to 2D array...");
    println!();

    let convert_start = Instant::now();

    let n_features = serializable_features.len();
    let n_dims = 56; // 56D MicroDynamics features (base_30d + 13 mfcc_delta + 13 mfcc_delta_delta)

    let mut feature_matrix = ndarray::Array2::zeros((n_features, n_dims));

    for (i, sf) in serializable_features.iter().enumerate() {
        for (j, &val) in sf.features.iter().enumerate() {
            if j < n_dims {
                feature_matrix[[i, j]] = val;
            }
        }
    }

    println!(
        "   └─ Converted to {}x{} array in {:.2}s",
        n_features,
        n_dims,
        convert_start.elapsed().as_secs_f64()
    );
    println!();

    // ========================================================================
    // Step 4: Run MiniBatch K-Means clustering
    // ========================================================================

    println!("🏗️  Step 4: Running MiniBatch K-Means clustering...");
    println!();

    // Configuration for bat vocalizations
    let n_clusters = 50; // Number of vocabulary items to discover
    let batch_size = 1000; // Mini-batch size
    let max_iter = 100; // Maximum iterations
    let tol = 1e-4; // Convergence tolerance

    println!("   Configuration:");
    println!("   ├─ n_clusters: {}", n_clusters);
    println!("   ├─ batch_size: {}", batch_size);
    println!("   ├─ max_iter: {}", max_iter);
    println!("   ├─ tol: {}", tol);
    println!("   └─ Using O(n) linear-time algorithm");
    println!();

    let cluster_start = Instant::now();

    // Create MiniBatch K-Means clusterer
    let kmeans = technical_architecture::clustering::MiniBatchKMeans::new(
        n_clusters,
        batch_size,
        max_iter,
        tol,
        Some(42), // Random seed for reproducibility
    )?;

    // Run clustering
    let labels = kmeans.fit_predict(&feature_matrix)?;

    let cluster_time = cluster_start.elapsed();
    let ms_per_sample = cluster_time.as_millis() as f64 / n_features as f64;

    println!(
        "   └─ Clustering completed in {:.2}s ({:.3}ms per sample)",
        cluster_time.as_secs_f64(),
        ms_per_sample
    );
    println!();

    // ========================================================================
    // Step 5: Analyze results
    // ========================================================================

    println!("📊 Step 5: Cluster Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let stats = kmeans.get_cluster_stats(&labels);

    println!("   Total phrases:        {}", n_features);
    println!("   Clusters found:       {}", stats.n_clusters);
    println!("   Noise points:         {}", stats.noise_count);
    println!(
        "   Clustered phrases:    {}",
        n_features - stats.noise_count
    );
    println!();

    if !stats.cluster_sizes.is_empty() {
        let total_clustered: usize = stats.cluster_sizes.iter().sum();
        let avg_size = total_clustered as f64 / stats.cluster_sizes.len() as f64;
        let max_size = *stats.cluster_sizes.iter().max().unwrap_or(&0);
        let min_size = *stats.cluster_sizes.iter().min().unwrap_or(&0);

        println!("   Cluster size range:  {} - {}", min_size, max_size);
        println!("   Average cluster:     {:.1} phrases", avg_size);
        println!();

        // Top 15 clusters
        let mut sorted_clusters: Vec<(usize, usize)> = stats
            .cluster_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| (i, size))
            .collect();
        sorted_clusters.sort_by(|a, b| b.1.cmp(&a.1));

        println!("   Top 15 Clusters:");
        for (i, (cluster_id, size)) in sorted_clusters.iter().take(15).enumerate() {
            let percentage = size.clone() as f64 / n_features as f64 * 100.0;
            println!(
                "      {}. Cluster {}: {} phrases ({:.2}%)",
                i + 1,
                cluster_id,
                size,
                percentage
            );
        }
    }
    println!();

    // ========================================================================
    // Step 6: Save results
    // ========================================================================

    println!("💾 Step 6: Saving cluster labels...");
    println!();

    let output_json = serde_json::json!({
        "n_features": n_features,
        "n_clusters": stats.n_clusters,
        "noise_count": stats.noise_count,
        "cluster_sizes": stats.cluster_sizes,
        "labels": labels,
        "n_clusters_requested": n_clusters,
        "batch_size": batch_size,
        "max_iter": max_iter,
        "clustering_time_sec": cluster_time.as_secs_f64(),
        "ms_per_sample": ms_per_sample,
    });

    std::fs::write(&output_path, output_json.to_string())?;
    println!("   └─ Saved to {}", output_path.display());
    println!();

    // ========================================================================
    // Summary
    // ========================================================================

    println!("✅ Phase 3 Complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    println!("📊 SUMMARY:");
    println!("   Species: Egyptian Fruit Bat (Rousettus aegyptiacus)");
    println!("   Feature extraction: 56D MicroDynamics (30D base + 13 Δ + 13 ΔΔ)");
    println!("   Audio files processed: {} / {}", n_features, n_files);
    println!("   Vocabulary items discovered: {}", stats.n_clusters);
    println!(
        "   Clustering time: {:.2}s ({:.3}ms per sample)",
        cluster_time.as_secs_f64(),
        ms_per_sample
    );
    println!();

    println!("📁 Output:");
    println!("   Features: {}", features_path.display());
    println!("   Clusters: {}", output_path.display());
    println!();

    println!("🎉 Next Steps:");
    println!("   1. Analyze cluster characteristics");
    println!("   2. Map clusters to behavioral contexts");
    println!("   3. Run Phase 4: GMM-HMM refinement");
    println!();

    Ok(())
}

// ============================================================================
// Feature Extraction
// ============================================================================

struct ExtractedFeatures {
    file_name: String,
    features: Vec<f64>,
    duration_ms: f64,
    sample_rate: u32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SerializableFeatures {
    file_name: String,
    features: Vec<f64>,
    duration_ms: f64,
    sample_rate: u32,
}

fn discover_wav_files(dir: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut wav_files = Vec::new();

    for entry in std::fs::read_dir(dir)? {
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

fn extract_features_from_files(
    file_names: &[String],
) -> Result<Vec<ExtractedFeatures>, Box<dyn std::error::Error>> {
    use rayon::prelude::*;

    let audio_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio");

    let features: Vec<_> = file_names
        .par_iter()
        .filter_map(|file_name| {
            match extract_single_feature(&audio_dir.join(file_name), file_name) {
                Ok(f) => Some(f),
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to extract features from {}: {}",
                        file_name, e
                    );
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

    // Load WAV file using hound
    let reader = hound::WavReader::open(file_path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    // Read samples as f32
    let audio: Vec<f32> = reader
        .into_samples::<f32>()
        .filter_map(|s| s.ok())
        .collect();

    if audio.is_empty() {
        return Err("No audio samples found".into());
    }

    // Convert to mono if stereo
    let audio_mono = if spec.channels == 2 {
        audio.chunks_exact(2).map(|c| (c[0] + c[1]) / 2.0).collect()
    } else {
        audio
    };

    // Calculate duration
    let duration_ms = (audio_mono.len() as f64 / sample_rate as f64) * 1000.0;

    // Extract 56D MicroDynamics features (full delta preservation)
    let extractor = MicroDynamicsExtractor::new(sample_rate);
    let features_56d = extractor.extract_56d(&audio_mono)?;

    // Convert 56D features to flat Vec<f64>
    // Structure: 30D base + 13 mfcc_delta + 13 mfcc_delta_delta = 56D
    let vector30d = features_56d.base_30d.to_vector30d(
        10000.0, // mean_f0_hz (estimated for bat FM sweeps)
        duration_ms as f32,
        5000.0, // f0_range_hz (estimated)
    );

    let mut features_vec: Vec<f64> = vector30d.to_array().iter().map(|&x| x as f64).collect();

    // Append 13 mfcc_delta features
    for delta in &features_56d.mfcc_delta {
        features_vec.push(*delta as f64);
    }

    // Append 13 mfcc_delta_delta features
    for delta_delta in &features_56d.mfcc_delta_delta {
        features_vec.push(*delta_delta as f64);
    }

    // Final dimension: 30 + 13 + 13 = 56

    Ok(ExtractedFeatures {
        file_name: file_name.to_string(),
        features: features_vec,
        duration_ms,
        sample_rate,
    })
}

/// Extract features from files with progress reporting
fn extract_features_from_files_progress(
    file_names: Vec<&String>,
    total_files: usize,
) -> Result<Vec<ExtractedFeatures>, Box<dyn std::error::Error>> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let audio_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio");
    let progress = AtomicUsize::new(0);
    let start_time = std::time::Instant::now();
    let report_interval = 1000; // Report every 1000 files

    let features: Vec<_> = file_names
        .par_iter()
        .filter_map(|file_name| {
            // Update progress counter
            let current = progress.fetch_add(1, Ordering::Relaxed) + 1;

            // Report progress at intervals
            if current % report_interval == 0 || current == total_files {
                let elapsed = start_time.elapsed().as_secs_f64();
                let rate = current as f64 / elapsed;
                let remaining = (total_files - current) as f64 / rate;
                let pct = (current as f64 / total_files as f64) * 100.0;

                println!(
                    "   Progress: {}/{} files ({:.1}%) | {:.1} files/sec | {:.1}s remaining",
                    current, total_files, pct, rate, remaining
                );
            }

            match extract_single_feature(&audio_dir.join(file_name), file_name) {
                Ok(f) => Some(f),
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to extract features from {}: {}",
                        file_name, e
                    );
                    None
                }
            }
        })
        .collect();

    Ok(features)
}
