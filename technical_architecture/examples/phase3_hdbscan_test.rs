// Phase 3: HDBSCAN Discovery - TEST VERSION (10K samples)
//
// Test HDBSCAN with a smaller subset before running on full dataset
//
// Usage: cargo run --release --example phase3_hdbscan_test

use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Paths to checkpoint data
    let results_dir = Path::new("/home/sheel/birdsong_analysis/data/marmoset_lexicon_to_syntax_results");
    let features_path = results_dir.join("phrase_features.bincode");
    let output_path = results_dir.join("hdbscan_clusters_test_10k.json");

    let subset_size = 10000; // Test with 10K samples first

    println!("🔬 Phase 3: HDBSCAN Discovery (TEST - {} samples)", subset_size);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Features: {}", features_path.display());
    println!("   Output:   {}", output_path.display());
    println!();

    // Load features
    println!("📂 Loading features from disk...");
    let load_start = Instant::now();

    let features_data = std::fs::read(&features_path)?;
    println!("   ├─ Loaded {} MB of feature data", features_data.len() / 1_048_576);

    // Deserialize features
    let serializable_features: Vec<technical_architecture::lexicon_to_syntax::PhraseFeaturesSerializable> =
        bincode::deserialize(&features_data)?;

    let n_features = serializable_features.len();
    println!("   ├─ Total features available: {}", n_features);
    println!("   └─ Using subset: {}", subset_size);
    println!();

    // Take subset
    let subset_features = &serializable_features[..subset_size.min(n_features)];

    // Convert to 2D array for HDBSCAN
    println!("🔄 Converting features to 2D array...");
    let convert_start = Instant::now();

    let n_dims = 30; // MicroDynamics features
    let mut feature_matrix = ndarray::Array2::zeros((subset_features.len(), n_dims));

    for (i, pf) in subset_features.iter().enumerate() {
        for (j, &val) in pf.features_flat.iter().enumerate() {
            if j < n_dims {
                feature_matrix[[i, j]] = val;
            }
        }
    }

    println!("   └─ Converted to {}x{} array in {:.2}s",
        subset_features.len(), n_dims, convert_start.elapsed().as_secs_f64());
    println!();

    // Configure HDBSCAN
    let min_cluster_size = 20; // Smaller for test dataset
    let min_samples = 10;

    println!("🏗️  Running HDBSCAN clustering...");
    println!("   ├─ min_cluster_size: {}", min_cluster_size);
    println!("   ├─ min_samples: {}", min_samples);
    println!("   └─ Using MEMORY-EFFICIENT on-the-fly distance computation");
    println!();

    let cluster_start = Instant::now();

    // Create HDBSCAN clusterer
    let hdbscan = technical_architecture::hdbscan::HdbscanClustering::new(min_cluster_size, min_samples)?;

    // Run clustering
    let labels = hdbscan.fit_predict(&feature_matrix)?;

    let cluster_time = cluster_start.elapsed();
    println!("   └─ Clustering completed in {:.2}s ({:.2}ms per sample)",
        cluster_time.as_secs_f64(),
        cluster_time.as_millis() as f64 / subset_features.len() as f64);
    println!();

    // Analyze results
    println!("📊 Cluster Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let stats = hdbscan.get_cluster_stats(&labels);

    println!("   Total phrases:        {}", subset_features.len());
    println!("   Clusters found:       {}", stats.n_clusters);
    println!("   Noise points:         {}", stats.noise_count);
    println!("   Clustered phrases:    {}", subset_features.len() - stats.noise_count);
    println!("   Clustering rate:      {:.1}%",
        (subset_features.len() - stats.noise_count) as f64 / subset_features.len() as f64 * 100.0);
    println!();

    if !stats.cluster_sizes.is_empty() {
        let total_clustered: usize = stats.cluster_sizes.iter().sum();
        let avg_size = total_clustered as f64 / stats.cluster_sizes.len() as f64;
        let max_size = *stats.cluster_sizes.iter().max().unwrap_or(&0);
        let min_size = *stats.cluster_sizes.iter().min().unwrap_or(&0);

        println!("   Cluster size range:  {} - {}", min_size, max_size);
        println!("   Average cluster:     {:.1} phrases", avg_size);
        println!();

        // Top 10 clusters
        let mut sorted_clusters: Vec<(i32, usize)> = stats.cluster_sizes.iter()
            .enumerate()
            .map(|(i, &size)| (i as i32, size))
            .collect();
        sorted_clusters.sort_by(|a, b| b.1.cmp(&a.1));

        println!("   Top 10 Clusters:");
        for (i, (cluster_id, size)) in sorted_clusters.iter().take(10).enumerate() {
            let percentage = size.clone() as f64 / subset_features.len() as f64 * 100.0;
            println!("      {}. Cluster {}: {} phrases ({:.1}%)",
                i + 1, cluster_id, size, percentage);
        }
    }

    println!();
    println!("💾 Saving cluster labels...");
    let output_json = serde_json::json!({
        "n_features": subset_features.len(),
        "n_clusters": stats.n_clusters,
        "noise_count": stats.noise_count,
        "cluster_sizes": stats.cluster_sizes,
        "labels": labels,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "clustering_time_sec": cluster_time.as_secs_f64(),
        "ms_per_sample": cluster_time.as_millis() as f64 / subset_features.len() as f64,
    });

    std::fs::write(&output_path, output_json.to_string())?;
    println!("   └─ Saved to {}", output_path.display());
    println!();

    // Extrapolate to full dataset
    let total_samples = n_features;
    let estimated_time_sec = (total_samples as f64 / subset_features.len() as f64) * cluster_time.as_secs_f64();
    let estimated_time_hours = estimated_time_sec / 3600.0;

    println!("📈 Performance Extrapolation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Test samples:           {}", subset_features.len());
    println!("   Test time:              {:.2}s", cluster_time.as_secs_f64());
    println!("   Per-sample time:        {:.3}ms", cluster_time.as_millis() as f64 / subset_features.len() as f64);
    println!();
    println!("   Full dataset:           {}", total_samples);
    println!("   Estimated full time:    {:.1} hours ({:.0} minutes)",
        estimated_time_hours, estimated_time_sec / 60.0);
    println!();

    println!("✅ Phase 3 Test Complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Discovered {} vocabulary items from {} phrases",
        stats.n_clusters, subset_features.len());
    println!();

    if estimated_time_hours < 24.0 {
        println!("   🚀 Ready to run on full dataset!");
    } else {
        println!("   ⚠️  Full dataset will take > 24 hours");
        println!("   Consider: using HDBSCAN with sample, or MiniBatch K-Means instead");
    }
    println!();

    Ok(())
}
