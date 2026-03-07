// Phase 3: MiniBatch K-Means Discovery for Marmoset Vocalizations
//
// This example loads the pre-computed features from Phase 2 and runs
// MiniBatch K-Means clustering to discover the vocabulary.
//
// MiniBatch K-Means scales linearly O(n) and is much faster than HDBSCAN O(n²)
//
// Usage: cargo run --release --example phase3_minibatch_marmoset

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Paths to checkpoint data
    let results_dir = Path::new("/home/sheel/birdsong_analysis/data/marmoset_lexicon_to_syntax_results");
    let features_path = results_dir.join("phrase_features.bincode");
    let output_path = results_dir.join("minibatch_clusters.json");

    println!("🚀 Phase 3: MiniBatch K-Means Discovery");
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
    println!(
        "   └─ {} features loaded in {:.2}s",
        n_features,
        load_start.elapsed().as_secs_f64()
    );
    println!();

    // Convert to 2D array for clustering
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

    println!(
        "   └─ Converted to {}x{} array in {:.2}s",
        n_features,
        n_dims,
        convert_start.elapsed().as_secs_f64()
    );
    println!();

    // Configure MiniBatch K-Means
    // Based on 300-file test, we found 25 vocabulary items
    // For 1.4M phrases, let's try 50 clusters (can be adjusted)
    let n_clusters = 50;
    let batch_size = 1000; // Larger batch for faster convergence
    let max_iter = 100; // Maximum iterations
    let tol = 1e-4; // Convergence tolerance

    println!("🏗️  Running MiniBatch K-Means clustering...");
    println!("   ├─ n_clusters: {}", n_clusters);
    println!("   ├─ batch_size: {}", batch_size);
    println!("   ├─ max_iter: {}", max_iter);
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

    // Analyze results
    println!("📊 Cluster Analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let stats = kmeans.get_cluster_stats(&labels);

    println!("   Total phrases:        {}", n_features);
    println!("   Clusters found:       {}", stats.n_clusters);
    println!("   Noise points:         {}", stats.noise_count);
    println!("   Clustered phrases:    {}", n_features - stats.noise_count);
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
    println!("💾 Saving cluster labels...");
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

    println!("✅ Phase 3 Complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "   Discovered {} vocabulary items from {} phrases",
        stats.n_clusters, n_features
    );
    println!(
        "   Completed in {:.2}s ({:.1}x faster than HDBSCAN)",
        cluster_time.as_secs_f64(),
        60.0 * 3600.0 / cluster_time.as_secs_f64()
    );
    println!();

    Ok(())
}
