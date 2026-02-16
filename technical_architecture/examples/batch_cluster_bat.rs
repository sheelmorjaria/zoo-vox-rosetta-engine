// Batch Processing with DBSCAN Clustering: Egyptian Fruit Bat Dataset
//
// This example demonstrates:
// 1. Real audio loading from WAV files
// 2. DBSCAN clustering to discover reusable phrase types
// 3. Batch processing with checkpointing for memory efficiency
//
// Usage:
//   cargo run --example batch_cluster_bat --release --features parallel-extraction

use std::path::{Path, PathBuf};
use technical_architecture::{
    batch_process_and_cluster, ExtractionPhraseCandidate,
    ParallelExtractionPipeline, ClusteredPhrase,
};

// Configuration
const BAT_AUDIO_DIR: &str = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio";
const CHECKPOINT_DIR: &str = "/tmp/bat_checkpoint";
const BATCH_SIZE: usize = 1000; // Process 1000 files per batch
const DBSCAN_EPS: f64 = 0.35; // Tighter clusters for more phrase types (was 0.5 -> 205 clusters)
const MIN_SAMPLES: usize = 10; // Require more evidence (was 5 -> 205 clusters)

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║   Batch Processing with DBSCAN Clustering: Egyptian Fruit Bat Dataset     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let audio_dir = Path::new(BAT_AUDIO_DIR);
    let checkpoint_dir = Path::new(CHECKPOINT_DIR);

    if !audio_dir.exists() {
        println!("❌ Audio directory not found: {}", BAT_AUDIO_DIR);
        return Err("Audio directory not found".into());
    }

    println!("📂 Configuration:");
    println!("   Audio directory: {}", BAT_AUDIO_DIR);
    println!("   Checkpoint directory: {}", CHECKPOINT_DIR);
    println!("   Batch size: {}", BATCH_SIZE);
    println!("   DBSCAN eps: {}", DBSCAN_EPS);
    println!("   Min samples: {}", MIN_SAMPLES);
    println!();

    // Run batch processing and clustering
    let start = std::time::Instant::now();

    let (clustered_phrases, vocalization_results) = batch_process_and_cluster(
        audio_dir,
        BATCH_SIZE,
        DBSCAN_EPS,
        MIN_SAMPLES,
        checkpoint_dir,
        None, // Process all files
    )?;

    let duration = start.elapsed();

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                        CLUSTERING RESULTS                             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Count unique clusters
    let unique_clusters: std::collections::HashSet<i32> = clustered_phrases
        .iter()
        .map(|cp| cp.cluster_id)
        .collect();

    println!("📊 Clustering Statistics:");
    println!("   Total clustered phrases: {}", clustered_phrases.len());
    println!("   Unique phrase types: {}", unique_clusters.len());
    println!("   Processing time: {:.2}s", duration.as_secs_f64());
    println!();

    // Analyze cluster sizes
    let mut cluster_sizes: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    for cp in &clustered_phrases {
        *cluster_sizes.entry(cp.cluster_id).or_insert(0) += 1;
    }

    let mut cluster_sizes_vec: Vec<_> = cluster_sizes.iter().collect();
    cluster_sizes_vec.sort_by(|a, b| b.1.cmp(a.1));

    println!("🔢 Top 10 Largest Clusters:");
    for (i, (cluster_id, size)) in cluster_sizes_vec.iter().take(10).enumerate() {
        println!("   {:2}. Cluster {:5}: {} members", i + 1, cluster_id, size);
    }
    println!();

    // Calculate Zipf's Law
    calculate_zipf_law(&clustered_phrases);

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PIPELINE COMPLETE                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("✅ Checkpoint saved to: {}", checkpoint_dir.display());
    println!();

    Ok(())
}

/// Calculate Zipf's Law from clustered phrases
fn calculate_zipf_law(clustered_phrases: &[ClusteredPhrase]) {
    use std::collections::HashMap;

    // Count phrase frequency
    let mut phrase_freq: HashMap<i32, usize> = HashMap::new();
    for cp in clustered_phrases {
        *phrase_freq.entry(cp.cluster_id).or_insert(0) += 1;
    }

    // Sort by frequency (descending)
    let mut freq_vec: Vec<_> = phrase_freq.iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(a.1));

    if freq_vec.is_empty() {
        return;
    }

    // Calculate Zipf's Law slope
    let n = freq_vec.len() as f64;
    let mut sum_log_rank = 0.0;
    let mut sum_log_freq = 0.0;
    let mut sum_log_rank_log_freq = 0.0;
    let mut sum_log_rank_sq = 0.0;

    for (rank, (_cluster_id, freq)) in freq_vec.iter().enumerate() {
        let rank_f = (rank + 1) as f64;
        let freq_f = **freq as f64;

        let log_rank = rank_f.ln();
        let log_freq = freq_f.ln();

        sum_log_rank += log_rank;
        sum_log_freq += log_freq;
        sum_log_rank_log_freq += log_rank * log_freq;
        sum_log_rank_sq += log_rank * log_rank;
    }

    let numerator = n * sum_log_rank_log_freq - sum_log_rank * sum_log_freq;
    let denominator = n * sum_log_rank_sq - sum_log_rank * sum_log_rank;

    let slope = if denominator.abs() > 1e-10 {
        numerator / denominator
    } else {
        0.0
    };

    // Calculate R²
    let mean_log_freq = sum_log_freq / n;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;

    for (rank, (_cluster_id, freq)) in freq_vec.iter().enumerate() {
        let rank_f = (rank + 1) as f64;
        let freq_f = **freq as f64;
        let log_freq = freq_f.ln();

        let predicted = slope * rank_f.ln();
        let residual = log_freq - predicted;
        let variation = log_freq - mean_log_freq;

        ss_res += residual * residual;
        ss_tot += variation * variation;
    }

    let r_squared = if ss_tot > 1e-10 {
        1.0 - (ss_res / ss_tot)
    } else {
        0.0
    };

    println!("📈 Zipf's Law Analysis:");
    println!("   Slope (α): {:.4}", slope);
    println!("   Correlation (R²): {:.4}", r_squared);
    println!("   Unique phrase types: {}", freq_vec.len());
    println!();

    // Interpret results
    if slope.abs() > 0.5 && r_squared > 0.7 {
        println!("   ✅ NATURAL LANGUAGE STRUCTURE DETECTED!");
        println!("   The dataset follows Zipf's Law, indicating natural communication.");
    } else if slope.abs() > 0.3 {
        println!("   ⚠️  MODERATE Zipf's Law compliance");
        println!("   Some structure detected but not strong natural language patterns.");
    } else {
        println!("   ❌ Flat distribution - no natural language structure");
        println!("   The dataset does not follow Zipf's Law.");
    }
}
