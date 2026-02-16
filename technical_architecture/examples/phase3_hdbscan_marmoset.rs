// Phase 3: HDBSCAN Discovery for Marmoset Vocalizations
//
// This example loads the pre-computed features from Phase 2 and runs
// parallel HDBSCAN clustering to discover the vocabulary.
//
// Usage: cargo run --release --example phase3_hdbscan_marmoset

use std::path::Path;
use std::time::Instant;
use technical_architecture::lexicon_to_syntax::PhraseFeaturesSerializable;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Paths to checkpoint data
    let results_dir = Path::new("/home/sheel/birdsong_analysis/data/marmoset_lexicon_to_syntax_results");
    let features_path = results_dir.join("phrase_features.bincode");
    let output_path = results_dir.join("hdbscan_clusters.json");

    println!("🔍 Phase 3: HDBSCAN Discovery");
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
    let serializable_features: Vec<PhraseFeaturesSerializable> =
        bincode::deserialize(&features_data)?;

    let n_features = serializable_features.len();
    println!("   └─ {} features loaded in {:.2}s", n_features, load_start.elapsed().as_secs_f64());
    println!();

    // Convert to 2D array for HDBSCAN
    println!("🔄 Converting features to 2D array...");
    let convert_start = Instant::now();

    let n_dims = 30; // MicroDynamics features
    let mut feature_matrix = ndarray::Array2::zeros((n_features, n_dims));

    for (i, pf) in serializable_features.iter().enumerate() {
        for (j, &val) in pf.features_flat.iter().enumerate() {
            if j < n_dims {
                feature_matrix[[i, j]] = val;
            }
        }
    }

    println!("   └─ Converted to {}x{} array in {:.2}s",
        n_features, n_dims, convert_start.elapsed().as_secs_f64());
    println!();

    // Configure HDBSCAN
    // min_cluster_size: minimum size of clusters (5-50 is typical)
    // min_samples: minimum samples for core point (usually same or smaller than min_cluster_size)
    let min_cluster_size = 50; // Larger for large dataset
    let min_samples = 20;

    println!("🏗️  Running HDBSCAN clustering...");
    println!("   ├─ min_cluster_size: {}", min_cluster_size);
    println!("   ├─ min_samples: {}", min_samples);
    println!("   └─ Using parallel Rayon implementation");
    println!();

    let cluster_start = Instant::now();

    // Create HDBSCAN clusterer
    let hdbscan = technical_architecture::hdbscan::HdbscanClustering::new(min_cluster_size, min_samples)?;

    // Run clustering
    let labels = hdbscan.fit_predict(&feature_matrix)?;

    let cluster_time = cluster_start.elapsed();
    println!("   └─ Clustering completed in {:.2}s", cluster_time.as_secs_f64());
    println!();

    // Analyze results
    println!("📊 Cluster Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let stats = hdbscan.get_cluster_stats(&labels);

    println!("   Total phrases:        {}", n_features);
    println!("   Clusters found:       {}", stats.n_clusters);
    println!("   Noise points:         {}", stats.noise_count);
    println!("   Clustered phrases:    {}", n_features - stats.noise_count);
    println!("   Clustering rate:      {:.1}%",
        (n_features - stats.noise_count) as f64 / n_features as f64 * 100.0);
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
            let percentage = size.clone() as f64 / n_features as f64 * 100.0;
            println!("      {}. Cluster {}: {} phrases ({:.1}%)",
                i + 1, cluster_id, size, percentage);
        }
    }

    println!();
    println!("💾 Saving cluster labels...");
    let output_json = serde_json::json!({
        "n_features": n_features,
        "n_clusters": stats.n_clusters,
        "noise_count": stats.noise_count,
        "cluster_sizes": stats.cluster_sizes,
        "labels": labels,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "clustering_time_sec": cluster_time.as_secs_f64(),
    });

    std::fs::write(&output_path, output_json.to_string())?;
    println!("   └─ Saved to {}", output_path.display());
    println!();

    println!("✅ Phase 3 Complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("   Discovered {} vocabulary items from {} phrases",
        stats.n_clusters, n_features);
    println!();

    Ok(())
}
