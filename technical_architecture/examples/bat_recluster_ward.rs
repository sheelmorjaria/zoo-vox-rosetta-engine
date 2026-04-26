// Re-Clustering Egyptian Fruit Bat Vocalizations with Agglomerative Clustering
// ============================================================================
//
// This example re-clusters the bat vocalization segments using position-independent
// agglomerative clustering with Ward linkage to discover true acoustic phrase types.
//
// Pipeline:
// 1. Load segments WITHOUT sequential ID assignment
// 2. Normalize 30D features (z-score normalization)
// 3. Apply AgglomerativeClustering with Ward linkage
// 4. Validate acoustic coherence of new clusters
// 5. Analyze true phrase transitions
// 6. Test for combinatorial syntax
//
// Usage: cargo run --release --example bat_recluster_ward

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone)]
struct Segment {
    segment_id: usize,
    file_name: String,
    start_time_ms: f64,
    end_time_ms: f64,
    duration_ms: f64,
    original_cluster_id: i32,
    features: Vec<f32>,
    new_cluster_id: Option<i32>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ReclusteredSegment {
    segment_id: usize,
    file_name: String,
    original_cluster_id: i32,
    new_cluster_id: i32,
    features_normalized: Vec<f32>,
}

#[derive(Debug, Clone)]
struct ClusterStats {
    cluster_id: i32,
    segment_count: usize,
    centroid: Vec<f32>,
    within_cluster_distance: f64,
    max_distance: f32,
    file_distribution: HashMap<String, usize>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ReclusteringResults {
    total_segments: usize,
    n_clusters: usize,
    method: String,
    normalization: String,
    silhouette_score: f64,
    cluster_stats: Vec<ClusterStatsExport>,
    transition_analysis: TransitionAnalysisExport,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ClusterStatsExport {
    cluster_id: i32,
    segment_count: usize,
    avg_duration_ms: f64,
    within_cluster_distance: f64,
    unique_files: usize,
    top_files: Vec<(String, usize)>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct TransitionAnalysisExport {
    unique_bigrams: usize,
    unique_trigrams: usize,
    top_transitions: Vec<TransitionExport>,
    context_diversity: f64,
    avg_transition_entropy: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct TransitionExport {
    from_cluster: i32,
    to_cluster: i32,
    count: usize,
    proportion: f64,
}

// ============================================================================
// Main Function
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Re-Clustering with Agglomerative Clustering (Ward Linkage)        ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let phase0_dir = data_dir.join("phase0_twolevel_hdbscan_results");
    let results_dir = data_dir.join("reclustering_results");
    fs::create_dir_all(&results_dir)?;

    // ========================================================================
    // Step 1: Load Raw Segments (WITHOUT position-based IDs)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 1: Loading Raw Segment Data                                       │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let mut segments = load_segments_raw(&phase0_dir)?;
    println!("   📂 Loaded {} segments", segments.len());
    println!(
        "   📊 Feature dimension: {}D",
        segments.first().map(|s| s.features.len()).unwrap_or(0)
    );
    println!();

    // ========================================================================
    // Step 2: Normalize Features (Z-score normalization)
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 2: Normalizing Features (Z-score)                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    normalize_features_zscore(&mut segments);
    println!("   ✅ Features normalized (zero mean, unit variance)");
    println!();

    // ========================================================================
    // Step 3: Agglomerative Clustering with Ward Linkage
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 3: Agglomerative Clustering (Ward Linkage)                        │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Determine optimal number of clusters using elbow method on sample
    let n_clusters = determine_optimal_clusters(&segments, 10, 50)?;
    println!("   📊 Optimal number of clusters: {}", n_clusters);
    println!();

    // Perform agglomerative clustering
    println!("   🔄 Performing agglomerative clustering...");
    ward_agglomerative_clustering(&mut segments, n_clusters)?;
    println!("   ✅ Clustering complete");
    println!();

    // ========================================================================
    // Step 4: Validate Acoustic Coherence
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 4: Validating Cluster Coherence                                   │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let cluster_stats = compute_cluster_statistics(&segments);
    let silhouette_score = compute_silhouette_score(&segments, &cluster_stats);

    println!("   📊 Cluster Statistics:");
    println!(
        "   {:<10} {:>12} {:>15} {:>15} {:>12}",
        "Cluster", "Segments", "Within-Dist", "Max-Dist", "Files"
    );
    println!("{}", "-".repeat(75));

    let mut sorted_stats: Vec<_> = cluster_stats.values().collect();
    sorted_stats.sort_by(|a, b| b.segment_count.cmp(&a.segment_count));

    for stats in sorted_stats.iter().take(15) {
        let files_display = if stats.file_distribution.len() > 100 {
            format!("{}+", stats.file_distribution.len())
        } else {
            format!("{}", stats.file_distribution.len())
        };

        println!(
            "   {:<10} {:>12} {:>15.4} {:>15.4} {:>12}",
            stats.cluster_id, stats.segment_count, stats.within_cluster_distance, stats.max_distance, files_display
        );
    }
    println!();

    println!("   📊 Overall Silhouette Score: {:.4}", silhouette_score);
    println!("      (>0.5 = good, 0.2-0.5 = moderate, <0.2 = poor)");
    println!();

    // ========================================================================
    // Step 5: Analyze True Phrase Transitions
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 5: Analyzing True Phrase Transitions                              │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Load annotations
    let annotations = load_annotations(data_dir.join("annotations.csv"))?;
    println!("   📂 Loaded {} annotations", annotations.len());
    println!();

    let transition_analysis = analyze_transitions(&segments, &annotations)?;
    display_transition_results(&transition_analysis);

    // ========================================================================
    // Step 6: Syntax Detection Test
    // ========================================================================

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("│ Step 6: Combinatorial Syntax Detection                                 │");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    test_combinatorial_syntax(&segments, &annotations, silhouette_score)?;

    // ========================================================================
    // Step 7: Save Re-clustered Data
    // ========================================================================

    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Step 7: Saving Re-clustered Data                                        │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    save_reclustered_data(
        &segments,
        &results_dir,
        &cluster_stats,
        &transition_analysis,
        silhouette_score,
    )?;

    println!("   💾 All results saved to: {}", results_dir.display());
    println!();

    Ok(())
}

// ============================================================================
// Data Loading
// ============================================================================

fn load_annotations(path: impl AsRef<Path>) -> Result<HashMap<String, i32>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let mut mapping = HashMap::new();

    for line in content.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 8 {
            if let Ok(context) = parts[2].parse::<i32>() {
                let file_name = parts[7].trim().to_string();
                mapping.insert(file_name, context);
            }
        }
    }

    Ok(mapping)
}

fn load_segments_raw(phase0_dir: &Path) -> Result<Vec<Segment>, Box<dyn std::error::Error>> {
    let segments_path = phase0_dir.join("all_segments.json");
    let segments_json: serde_json::Value = serde_json::from_str(&fs::read_to_string(&segments_path)?)?;

    let mut segments = Vec::new();

    if let Some(arr) = segments_json.as_array() {
        for segment in arr {
            if let (Some(segment_id), Some(file_name), Some(start), Some(end), Some(cluster_id), Some(features)) = (
                segment["segment_id"].as_u64(),
                segment["file_name"].as_str(),
                segment["start_time_ms"].as_f64(),
                segment["end_time_ms"].as_f64(),
                segment["level1_cluster_id"].as_i64(),
                segment["representative_features"].as_array(),
            ) {
                let duration = end - start;

                // Convert features to Vec<f32>
                let feature_vec: Vec<f32> = features.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();

                segments.push(Segment {
                    segment_id: segment_id as usize,
                    file_name: file_name.to_string(),
                    start_time_ms: start,
                    end_time_ms: end,
                    duration_ms: duration,
                    original_cluster_id: cluster_id as i32,
                    features: feature_vec,
                    new_cluster_id: None,
                });
            }
        }
    }

    // Sort by file and start time (for later analysis)
    segments.sort_by(|a, b| {
        a.file_name.cmp(&b.file_name).then_with(|| {
            a.start_time_ms
                .partial_cmp(&b.start_time_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    Ok(segments)
}

// ============================================================================
// Feature Normalization
// ============================================================================

fn normalize_features_zscore(segments: &mut [Segment]) {
    if segments.is_empty() {
        return;
    }

    let n_features = segments[0].features.len();
    if n_features == 0 {
        return;
    }

    // Calculate mean and std for each feature dimension
    let mut means = vec![0.0f64; n_features];
    let mut stds = vec![0.0f64; n_features];

    // First pass: calculate means
    for seg in segments.iter() {
        for (i, &val) in seg.features.iter().enumerate() {
            means[i] += val as f64;
        }
    }

    let n = segments.len() as f64;
    for mean in &mut means {
        *mean /= n;
    }

    // Second pass: calculate standard deviations
    for seg in segments.iter() {
        for (i, &val) in seg.features.iter().enumerate() {
            let diff = val as f64 - means[i];
            stds[i] += diff * diff;
        }
    }

    for std_val in stds.iter_mut() {
        *std_val = (*std_val / n).sqrt();
        // Avoid division by zero
        if *std_val < 1e-10 {
            *std_val = 1.0;
        }
    }

    // Normalize
    for seg in segments.iter_mut() {
        for (i, val) in seg.features.iter_mut().enumerate() {
            *val = ((*val as f64 - means[i]) / stds[i]) as f32;
        }
    }
}

// ============================================================================
// Agglomerative Clustering with Ward Linkage
// ============================================================================

fn determine_optimal_clusters(
    segments: &[Segment],
    min_clusters: usize,
    max_clusters: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    println!("   📊 Running elbow method on sample ({} clusters)...", max_clusters);

    // Sample segments for speed (use 10% or max 5000)
    let sample_size = segments.len().min(5000);
    let mut rng = rand::thread_rng();
    let sample_indices: Vec<usize> = (0..segments.len())
        .collect::<Vec<_>>()
        .into_iter()
        .filter(|_| rand::Rng::gen_bool(&mut rng, 0.1))
        .take(sample_size)
        .collect();

    if sample_indices.is_empty() && !segments.is_empty() {
        // Fallback: take first sample_size
        let sample_indices: Vec<usize> = (0..segments.len().min(sample_size)).collect();
    }

    let mut inertias = Vec::new();

    for k in min_clusters..=max_clusters {
        let mut clusters = vec![0; sample_indices.len()];

        // Simple k-means for inertia calculation
        let mut centroids = initialize_centroids_kmeans_plus_plus(&sample_indices, segments, k);

        for _iter in 0..10 {
            // Assign to nearest centroid
            for (idx, &seg_idx) in sample_indices.iter().enumerate() {
                let seg = &segments[seg_idx];
                let mut best_cluster = 0;
                let mut best_dist = f64::MAX;

                for (cluster_id, centroid) in centroids.iter().enumerate() {
                    let dist = euclidean_distance_f32(&seg.features, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = cluster_id;
                    }
                }

                clusters[idx] = best_cluster;
            }

            // Update centroids
            update_centroids(&sample_indices, segments, &clusters, k, &mut centroids);
        }

        // Calculate inertia
        let mut inertia = 0.0;
        for (idx, &seg_idx) in sample_indices.iter().enumerate() {
            let seg = &segments[seg_idx];
            let centroid = &centroids[clusters[idx]];
            inertia += euclidean_distance_f32(&seg.features, centroid).powi(2);
        }

        inertias.push(inertia);

        if k % 10 == 0 {
            println!("      k={}: inertia={:.2}", k, inertia);
        }
    }

    // Find elbow point (maximum curvature)
    let optimal_k = find_elbow_point(&inertias, min_clusters);
    println!("   📍 Elbow point detected at k={}", optimal_k);

    Ok(optimal_k)
}

fn initialize_centroids_kmeans_plus_plus(indices: &[usize], segments: &[Segment], k: usize) -> Vec<Vec<f32>> {
    if indices.is_empty() || k == 0 {
        return Vec::new();
    }

    let mut centroids = Vec::new();
    let mut rng = rand::thread_rng();

    // First centroid: random choice
    let first_idx = indices[rand::Rng::gen_range(&mut rng, 0..indices.len())];
    centroids.push(segments[first_idx].features.clone());

    // Subsequent centroids: weighted by squared distance
    while centroids.len() < k {
        let mut distances = Vec::new();
        let mut total_dist = 0.0;

        for &idx in indices {
            let seg = &segments[idx];
            let min_dist = centroids
                .iter()
                .map(|c| euclidean_distance_f32(&seg.features, c))
                .fold(f64::INFINITY, |a, b| a.min(b));

            let squared_dist = min_dist * min_dist;
            distances.push(squared_dist);
            total_dist += squared_dist;
        }

        // Choose with probability proportional to squared distance
        let mut choice = rand::Rng::gen_range(&mut rng, 0.0..total_dist);
        let mut selected_idx = 0;

        for (i, &dist) in distances.iter().enumerate() {
            choice -= dist;
            if choice <= 0.0 {
                selected_idx = indices[i];
                break;
            }
        }

        centroids.push(segments[selected_idx].features.clone());
    }

    centroids
}

fn update_centroids(indices: &[usize], segments: &[Segment], clusters: &[usize], k: usize, centroids: &mut [Vec<f32>]) {
    let n_features = centroids[0].len();

    // Reset centroids
    for centroid in centroids.iter_mut() {
        centroid.fill(0.0);
    }

    let mut counts = vec![0usize; k];

    // Sum features for each cluster
    for (idx, &seg_idx) in indices.iter().enumerate() {
        let cluster_id = clusters[idx];
        let seg = &segments[seg_idx];

        for (i, &val) in seg.features.iter().enumerate() {
            centroids[cluster_id][i] += val;
        }

        counts[cluster_id] += 1;
    }

    // Average
    for (cluster_id, centroid) in centroids.iter_mut().enumerate() {
        if counts[cluster_id] > 0 {
            for val in centroid.iter_mut() {
                *val /= counts[cluster_id] as f32;
            }
        }
    }

    // Handle empty clusters
    for (cluster_id, count) in counts.iter().enumerate() {
        if *count == 0 {
            // Reinitialize to a random point
            let mut rng = rand::thread_rng();
            let random_idx = indices[rand::Rng::gen_range(&mut rng, 0..indices.len())];
            centroids[cluster_id] = segments[random_idx].features.clone();
        }
    }
}

fn find_elbow_point(inertias: &[f64], min_k: usize) -> usize {
    if inertias.len() < 3 {
        return min_k;
    }

    let max_curvature: usize = 2_usize.min(inertias.len().saturating_sub(2));

    let mut best_k = min_k;
    let mut max_curvature_value = 0.0;

    for i in max_curvature..inertias.len() - 1 {
        let x1 = (i - 1) as f64;
        let x2 = i as f64;
        let x3 = (i + 1) as f64;

        let y1 = inertias[i - 1];
        let y2 = inertias[i];
        let y3 = inertias[i + 1];

        // Calculate curvature
        let numerator: f64 = ((y3 - y1) / (x3 - x1)).abs();
        let denom1: f64 = (y2 - y1).abs();
        let denom2: f64 = (y3 - y2).abs();
        let denominator = denom1.max(denom2);

        let curvature = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        if curvature > max_curvature_value {
            max_curvature_value = curvature;
            best_k = i + 1;
        }
    }

    best_k
}

fn ward_agglomerative_clustering(
    segments: &mut [Segment],
    n_clusters: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if segments.is_empty() || n_clusters == 0 {
        return Ok(());
    }

    println!(
        "   🔄 Ward clustering: {} segments → {} clusters",
        segments.len(),
        n_clusters
    );

    let n = segments.len();

    // For large datasets, use a two-stage approach:
    // 1. Pre-cluster with K-means to reduce to manageable size
    // 2. Apply Ward on the centroids

    let precluster_size = if n > 50000 {
        n_clusters * 10 // 10x final clusters for intermediate
    } else {
        n // Small enough to process directly
    };

    // Initialize with k-means++ for better starting points
    let all_indices: Vec<usize> = (0..n).collect();
    let mut centroids = initialize_centroids_kmeans_plus_plus(&all_indices, segments, precluster_size.min(n));

    // Assign initial clusters
    let mut cluster_assignments = vec![0usize; n];

    // Refine with a few iterations of Lloyd's algorithm
    for _iter in 0..20 {
        // Assign to nearest centroid
        for (idx, seg) in segments.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_dist = f64::MAX;

            for (cluster_id, centroid) in centroids.iter().enumerate() {
                let dist = euclidean_distance_f32(&seg.features, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = cluster_id;
                }
            }

            cluster_assignments[idx] = best_cluster;
        }

        // Update centroids
        let mut counts = vec![0usize; centroids.len()];
        for centroid in centroids.iter_mut() {
            centroid.fill(0.0);
        }

        for (idx, seg) in segments.iter().enumerate() {
            let cluster_id = cluster_assignments[idx];
            for (i, &val) in seg.features.iter().enumerate() {
                centroids[cluster_id][i] += val;
            }
            counts[cluster_id] += 1;
        }

        for (cluster_id, centroid) in centroids.iter_mut().enumerate() {
            if counts[cluster_id] > 0 {
                for val in centroid.iter_mut() {
                    *val /= counts[cluster_id] as f32;
                }
            }
        }
    }

    // Now do proper agglomerative merging based on Ward criterion
    // For efficiency, we'll merge based on centroid distances

    // Build distance matrix between clusters
    let mut active_clusters: HashSet<usize> = (0..precluster_size.min(n)).collect();
    let mut cluster_sizes: Vec<usize> = (0..precluster_size.min(n))
        .map(|i| cluster_assignments.iter().filter(|&&c| c == i).count())
        .collect();

    // Merge until we reach target number of clusters
    while active_clusters.len() > n_clusters {
        // Find closest pair of clusters (minimum Ward distance)
        let mut best_pair = None;
        let mut best_merge_cost = f64::MAX;

        let cluster_vec: Vec<_> = active_clusters.iter().copied().collect();

        for i in 0..cluster_vec.len() {
            for j in (i + 1)..cluster_vec.len() {
                let c1 = cluster_vec[i];
                let c2 = cluster_vec[j];

                // Ward merge cost:
                // Δ = (n1 * n2) / (n1 + n2) * ||c1 - c2||^2
                let n1 = cluster_sizes[c1] as f64;
                let n2 = cluster_sizes[c2] as f64;
                let dist_sq = euclidean_distance_f32(&centroids[c1], &centroids[c2]).powi(2);
                let merge_cost = (n1 * n2) / (n1 + n2) * dist_sq;

                if merge_cost < best_merge_cost {
                    best_merge_cost = merge_cost;
                    best_pair = Some((c1, c2));
                }
            }
        }

        // Merge the best pair
        if let Some((c1, c2)) = best_pair {
            let n1 = cluster_sizes[c1];
            let n2 = cluster_sizes[c2];

            // New centroid: weighted average
            let new_centroid: Vec<f32> = centroids[c1]
                .iter()
                .zip(centroids[c2].iter())
                .map(|(&v1, &v2)| (v1 * n1 as f32 + v2 * n2 as f32) / ((n1 + n2) as f32))
                .collect();

            // Update c1 with merged centroid
            centroids[c1] = new_centroid;
            cluster_sizes[c1] = n1 + n2;

            // Reassign all points from c2 to c1
            for idx in 0..n {
                if cluster_assignments[idx] == c2 {
                    cluster_assignments[idx] = c1;
                }
            }

            // Remove c2 from active clusters
            active_clusters.remove(&c2);
        }

        // Print progress
        if active_clusters.len() % 10 == 0 || active_clusters.len() <= n_clusters + 5 {
            println!("      Merging... {} clusters remaining", active_clusters.len());
        }
    }

    // Map the remaining cluster IDs to sequential range
    let mut cluster_id_map: HashMap<usize, i32> = HashMap::new();
    let mut next_id = 0;

    for &cluster_id in active_clusters.iter() {
        cluster_id_map.insert(cluster_id, next_id);
        next_id += 1;
    }

    // Assign final cluster IDs to segments
    for (idx, seg) in segments.iter_mut().enumerate() {
        let old_id = cluster_assignments[idx];
        if let Some(&new_id) = cluster_id_map.get(&old_id) {
            seg.new_cluster_id = Some(new_id);
        } else {
            seg.new_cluster_id = Some(-1); // Should not happen
        }
    }

    println!("   ✅ Complete: {} clusters", next_id);

    Ok(())
}

// ============================================================================
// Cluster Analysis
// ============================================================================

fn compute_cluster_statistics(segments: &[Segment]) -> HashMap<i32, ClusterStats> {
    let mut cluster_data: HashMap<i32, Vec<&Segment>> = HashMap::new();

    // Group by new cluster ID
    for seg in segments {
        if let Some(cluster_id) = seg.new_cluster_id {
            if cluster_id >= 0 {
                cluster_data.entry(cluster_id).or_insert_with(Vec::new).push(seg);
            }
        }
    }

    let mut stats = HashMap::new();

    for (cluster_id, segs) in cluster_data {
        if segs.is_empty() {
            continue;
        }

        let n_features = segs[0].features.len();

        // Calculate centroid
        let mut centroid = vec![0.0f32; n_features];
        for seg in &segs {
            for (i, &val) in seg.features.iter().enumerate() {
                centroid[i] += val;
            }
        }

        for val in centroid.iter_mut() {
            *val /= segs.len() as f32;
        }

        // Calculate within-cluster distances
        let mut distances = Vec::new();
        for seg in &segs {
            let dist = euclidean_distance_f32(&seg.features, &centroid);
            distances.push(dist);
        }

        let within_cluster_distance = distances.iter().sum::<f64>() / distances.len() as f64;
        let max_distance = distances.into_iter().fold(f64::NEG_INFINITY, |a, b| a.max(b)) as f32;

        // File distribution
        let mut file_dist: HashMap<String, usize> = HashMap::new();
        for seg in &segs {
            *file_dist.entry(seg.file_name.clone()).or_insert(0) += 1;
        }

        stats.insert(
            cluster_id,
            ClusterStats {
                cluster_id,
                segment_count: segs.len(),
                centroid,
                within_cluster_distance,
                max_distance,
                file_distribution: file_dist,
            },
        );
    }

    stats
}

fn compute_silhouette_score(segments: &[Segment], cluster_stats: &HashMap<i32, ClusterStats>) -> f64 {
    let mut total_silhouette = 0.0;
    let mut count = 0;

    for seg in segments {
        if let Some(cluster_id) = seg.new_cluster_id {
            if cluster_id < 0 || !cluster_stats.contains_key(&cluster_id) {
                continue;
            }

            // a: mean distance to points in same cluster
            let a = cluster_stats[&cluster_id].within_cluster_distance;

            // b: min mean distance to points in other clusters
            let mut b = f64::MAX;

            for (&other_id, other_stats) in cluster_stats {
                if other_id != cluster_id {
                    let dist = euclidean_distance_f32(&seg.features, &other_stats.centroid);
                    if dist < b {
                        b = dist;
                    }
                }
            }

            // Silhouette for this point
            let s = if b > a { (b - a) / b.max(a) } else { 0.0 };

            total_silhouette += s;
            count += 1;
        }
    }

    if count > 0 {
        total_silhouette / count as f64
    } else {
        0.0
    }
}

// ============================================================================
// Transition Analysis
// ============================================================================

#[derive(Debug, Clone)]
struct TransitionAnalysis {
    bigram_counts: HashMap<(i32, i32), usize>,
    trigram_counts: HashMap<(i32, i32, i32), usize>,
    context_patterns: HashMap<i32, ContextPattern>,
    total_transitions: usize,
}

#[derive(Debug, Clone)]
struct ContextPattern {
    context_id: i32,
    num_files: usize,
    num_transitions: usize,
    unique_bigrams: usize,
    top_bigrams: Vec<((i32, i32), usize)>,
    entropy: f64,
}

fn analyze_transitions(
    segments: &[Segment],
    annotations: &HashMap<String, i32>,
) -> Result<TransitionAnalysis, Box<dyn std::error::Error>> {
    // Group segments by file
    let mut file_segments: HashMap<String, Vec<&Segment>> = HashMap::new();
    for seg in segments {
        file_segments
            .entry(seg.file_name.clone())
            .or_insert_with(Vec::new)
            .push(seg);
    }

    let mut bigram_counts: HashMap<(i32, i32), usize> = HashMap::new();
    let mut trigram_counts: HashMap<(i32, i32, i32), usize> = HashMap::new();
    let mut context_bigrams: HashMap<i32, Vec<(i32, i32)>> = HashMap::new();
    let mut total_transitions = 0;

    for (file_name, segs) in file_segments {
        // Sort by start time
        let mut sorted: Vec<_> = segs.iter().collect();
        sorted.sort_by(|a, b| a.start_time_ms.partial_cmp(&b.start_time_ms).unwrap());

        // Extract transitions
        for window in sorted.windows(2) {
            let from_id = window[0].new_cluster_id.unwrap_or(-1);
            let to_id = window[1].new_cluster_id.unwrap_or(-1);

            if from_id >= 0 && to_id >= 0 {
                *bigram_counts.entry((from_id, to_id)).or_insert(0) += 1;
                total_transitions += 1;

                if let Some(&ctx) = annotations.get(&file_name) {
                    context_bigrams
                        .entry(ctx)
                        .or_insert_with(Vec::new)
                        .push((from_id, to_id));
                }
            }
        }

        for window in sorted.windows(3) {
            let id1 = window[0].new_cluster_id.unwrap_or(-1);
            let id2 = window[1].new_cluster_id.unwrap_or(-1);
            let id3 = window[2].new_cluster_id.unwrap_or(-1);

            if id1 >= 0 && id2 >= 0 && id3 >= 0 {
                *trigram_counts.entry((id1, id2, id3)).or_insert(0) += 1;
            }
        }
    }

    // Analyze context-specific patterns
    let mut context_patterns = HashMap::new();

    for (context_id, bigrams) in context_bigrams {
        let num_transitions = bigrams.len();

        // Count unique bigrams
        let unique_bigrams: HashSet<(i32, i32)> = bigrams.iter().copied().collect();

        // Get top bigrams
        let mut bigram_counts_local: HashMap<(i32, i32), usize> = HashMap::new();
        for bigram in &bigrams {
            *bigram_counts_local.entry(*bigram).or_insert(0) += 1;
        }

        // Calculate entropy from bigram_counts_local before consuming it
        let counts: Vec<usize> = bigram_counts_local.values().copied().collect();
        let total: usize = counts.iter().sum();
        let mut entropy = 0.0;
        for count in counts {
            if total > 0 {
                let p = count as f64 / total as f64;
                if p > 0.0 {
                    entropy -= p * p.log2();
                }
            }
        }

        // Get top bigrams (consumes bigram_counts_local)
        let mut top_bigrams: Vec<_> = bigram_counts_local.into_iter().collect();
        top_bigrams.sort_by(|a, b| b.1.cmp(&a.1));
        top_bigrams.truncate(10);

        // Count files in this context
        let num_files = annotations.values().filter(|&&c| c == context_id).count();

        context_patterns.insert(
            context_id,
            ContextPattern {
                context_id,
                num_files,
                num_transitions,
                unique_bigrams: unique_bigrams.len(),
                top_bigrams,
                entropy,
            },
        );
    }

    Ok(TransitionAnalysis {
        bigram_counts,
        trigram_counts,
        context_patterns,
        total_transitions,
    })
}

fn display_transition_results(analysis: &TransitionAnalysis) {
    println!("   📊 Overall Transition Statistics:");
    println!("      Total transitions: {}", analysis.total_transitions);
    println!("      Unique bigrams: {}", analysis.bigram_counts.len());
    println!("      Unique trigrams: {}", analysis.trigram_counts.len());
    println!();

    // Top transitions
    let mut sorted_bigrams: Vec<_> = analysis.bigram_counts.iter().collect();
    sorted_bigrams.sort_by(|a, b| b.1.cmp(&a.1));

    println!("   📊 Top 20 Bigram Transitions:");
    println!("   {:<8} {:<8} {:>12} {:>12}", "From", "To", "Count", "Proportion");
    println!("{}", "-".repeat(50));

    for ((from, to), count) in sorted_bigrams.iter().take(20) {
        let proportion = **count as f64 / analysis.total_transitions as f64;
        println!("   {:<8} {:<8} {:>12} {:>11.3}%", from, to, count, proportion * 100.0);
    }
    println!();

    // Context-specific patterns
    let mut sorted_contexts: Vec<_> = analysis.context_patterns.values().collect();
    sorted_contexts.sort_by(|a, b| b.num_transitions.cmp(&a.num_transitions));

    println!("   📊 Context-Specific Patterns:");
    println!(
        "   {:<10} {:>10} {:>12} {:>12} {:>12}",
        "Context", "Files", "Transitions", "Unique", "Entropy"
    );
    println!("{}", "-".repeat(60));

    for pattern in sorted_contexts.iter().take(10) {
        println!(
            "   {:<10} {:>10} {:>12} {:>12} {:>12.3}",
            pattern.context_id, pattern.num_files, pattern.num_transitions, pattern.unique_bigrams, pattern.entropy
        );
    }
    println!();

    // Calculate average entropy
    let avg_entropy = if !analysis.context_patterns.is_empty() {
        analysis.context_patterns.values().map(|p| p.entropy).sum::<f64>() / analysis.context_patterns.len() as f64
    } else {
        0.0
    };

    println!("   📊 Average Bigram Entropy: {:.3} bits", avg_entropy);
    println!();

    if avg_entropy < 3.0 {
        println!("      ✅ LOW ENTROPY: Highly predictable transitions");
        println!("         → Strong evidence for SYNTACTIC RULES");
    } else if avg_entropy < 5.0 {
        println!("      ⚠️  MEDIUM ENTROPY: Moderately predictable transitions");
        println!("         → Suggests FLEXIBLE SYNTAX");
    } else {
        println!("      ❌ HIGH ENTROPY: Unpredictable transitions");
        println!("         → NO evidence for syntactic rules");
    }
}

// ============================================================================
// Syntax Detection Test
// ============================================================================

fn test_combinatorial_syntax(
    segments: &[Segment],
    annotations: &HashMap<String, i32>,
    silhouette_score: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   📊 Combinatorial Syntax Detection Test");
    println!();

    // Test 1: Cluster Quality
    println!("   Test 1: Cluster Acoustic Coherence");
    println!("   ─────────────────────────────────────");
    println!("   Silhouette Score: {:.4}", silhouette_score);

    let quality_test = if silhouette_score > 0.5 {
        "✅ PASS"
    } else if silhouette_score > 0.2 {
        "⚠️  MARGINAL"
    } else {
        "❌ FAIL"
    };
    println!("   Result: {}", quality_test);
    println!();

    // Test 2: Transition Diversity
    let transition_analysis = analyze_transitions(segments, annotations)?;
    let diversity_ratio = transition_analysis.bigram_counts.len() as f64 / transition_analysis.total_transitions as f64;

    println!("   Test 2: Transition Diversity");
    println!("   ─────────────────────────────");
    println!("   Unique bigrams: {}", transition_analysis.bigram_counts.len());
    println!("   Total transitions: {}", transition_analysis.total_transitions);
    println!("   Diversity ratio: {:.4}", diversity_ratio);

    let diversity_test = if diversity_ratio > 0.1 {
        "✅ PASS"
    } else if diversity_ratio > 0.05 {
        "⚠️  MARGINAL"
    } else {
        "❌ FAIL"
    };
    println!("   Result: {}", diversity_test);
    println!();

    // Test 3: Context Specificity
    let context_specific_bigrams = count_context_specific_bigrams(&transition_analysis);
    let context_specificity = context_specific_bigrams as f64 / transition_analysis.bigram_counts.len() as f64;

    println!("   Test 3: Context Specificity");
    println!("   ─────────────────────────────");
    println!("   Context-specific bigrams: {}", context_specific_bigrams);
    println!("   Total bigrams: {}", transition_analysis.bigram_counts.len());
    println!("   Specificity ratio: {:.4}", context_specificity);

    let specificity_test = if context_specificity > 0.3 {
        "✅ PASS"
    } else if context_specificity > 0.1 {
        "⚠️  MARGINAL"
    } else {
        "❌ FAIL"
    };
    println!("   Result: {}", specificity_test);
    println!();

    // Test 4: Cross-Context Consistency
    let avg_entropy = if !transition_analysis.context_patterns.is_empty() {
        transition_analysis
            .context_patterns
            .values()
            .map(|p| p.entropy)
            .sum::<f64>()
            / transition_analysis.context_patterns.len() as f64
    } else {
        0.0
    };

    println!("   Test 4: Cross-Context Pattern Consistency");
    println!("   ──────────────────────────────────────────");
    println!("   Average bigram entropy: {:.3} bits", avg_entropy);

    let consistency_test = if avg_entropy < 5.0 {
        "✅ PASS (consistent patterns)"
    } else if avg_entropy < 7.0 {
        "⚠️  MARGINAL"
    } else {
        "❌ FAIL (no consistent patterns)"
    };
    println!("   Result: {}", consistency_test);
    println!();

    // Overall verdict
    println!("   ╔═══════════════════════════════════════════════════════════════╗");
    println!("   ║              OVERALL SYNTAX DETECTION VERDICT                ║");
    println!("   ╚═══════════════════════════════════════════════════════════════╝");
    println!();

    let passes = [
        quality_test == "✅ PASS",
        diversity_test == "✅ PASS",
        specificity_test == "✅ PASS",
        consistency_test.contains("✅ PASS"),
    ]
    .iter()
    .filter(|&&x| x)
    .count();

    if passes >= 3 {
        println!("   ✅ STRONG EVIDENCE for Combinatorial Syntax");
        println!("      → {} out of 4 tests passed", passes);
        println!("      → Egyptian fruit bats may use combinatorial syntax");
    } else if passes >= 2 {
        println!("   ⚠️  MODERATE EVIDENCE for Combinatorial Syntax");
        println!("      → {} out of 4 tests passed", passes);
        println!("      → More analysis needed");
    } else {
        println!("   ❌ LITTLE to NO EVIDENCE for Combinatorial Syntax");
        println!("      → {} out of 4 tests passed", passes);
        println!("      → Likely context-specific vocabulary system");
    }
    println!();

    Ok(())
}

fn count_context_specific_bigrams(analysis: &TransitionAnalysis) -> usize {
    let mut count = 0;

    for ((from, to), _) in analysis.bigram_counts.iter() {
        let mut context_count = 0;

        for pattern in analysis.context_patterns.values() {
            for &((bigram_from, bigram_to), _count) in &pattern.top_bigrams {
                if bigram_from == *from && bigram_to == *to {
                    context_count += 1;
                    break;
                }
            }
        }

        // Count as context-specific if appears in < half of contexts
        if context_count < (analysis.context_patterns.len() / 2).max(1) {
            count += 1;
        }
    }

    count
}

// ============================================================================
// Save Results
// ============================================================================

fn save_reclustered_data(
    segments: &[Segment],
    results_dir: &Path,
    cluster_stats: &HashMap<i32, ClusterStats>,
    transition_analysis: &TransitionAnalysis,
    silhouette_score: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Save re-clustered segments
    let reclustered: Vec<ReclusteredSegment> = segments
        .iter()
        .map(|seg| ReclusteredSegment {
            segment_id: seg.segment_id,
            file_name: seg.file_name.clone(),
            original_cluster_id: seg.original_cluster_id,
            new_cluster_id: seg.new_cluster_id.unwrap_or(-1),
            features_normalized: seg.features.clone(),
        })
        .collect();

    let segments_path = results_dir.join("reclustered_segments.json");
    fs::write(&segments_path, serde_json::to_string_pretty(&reclustered)?)?;
    println!("   💾 Re-clustered segments: {}", segments_path.display());

    // Save cluster statistics
    let mut stats_export = Vec::new();
    for stats in cluster_stats.values() {
        let total_duration: f64 = stats
            .file_distribution
            .keys()
            .filter_map(|file| {
                segments
                    .iter()
                    .find(|s| &s.file_name == file && s.new_cluster_id == Some(stats.cluster_id))
            })
            .map(|s| s.duration_ms)
            .sum::<f64>();

        let avg_duration = if stats.segment_count > 0 {
            total_duration / stats.segment_count as f64
        } else {
            0.0
        };

        let mut file_counts: Vec<_> = stats.file_distribution.iter().collect();
        file_counts.sort_by(|a, b| b.1.cmp(&a.1));
        file_counts.truncate(10);

        stats_export.push(ClusterStatsExport {
            cluster_id: stats.cluster_id,
            segment_count: stats.segment_count,
            avg_duration_ms: avg_duration,
            within_cluster_distance: stats.within_cluster_distance,
            unique_files: stats.file_distribution.len(),
            top_files: file_counts.into_iter().map(|(f, c)| (f.clone(), *c)).collect(),
        });
    }

    stats_export.sort_by(|a, b| b.segment_count.cmp(&a.segment_count));

    let stats_path = results_dir.join("cluster_statistics.json");
    fs::write(&stats_path, serde_json::to_string_pretty(&stats_export)?)?;
    println!("   💾 Cluster statistics: {}", stats_path.display());

    // Save transition analysis
    let mut sorted_transitions: Vec<_> = transition_analysis.bigram_counts.iter().collect();
    sorted_transitions.sort_by(|a, b| b.1.cmp(&a.1));

    let transition_export = TransitionAnalysisExport {
        unique_bigrams: transition_analysis.bigram_counts.len(),
        unique_trigrams: transition_analysis.trigram_counts.len(),
        top_transitions: sorted_transitions
            .iter()
            .take(100)
            .map(|((from, to), count)| TransitionExport {
                from_cluster: *from,
                to_cluster: *to,
                count: **count as usize,
                proportion: **count as f64 / transition_analysis.total_transitions as f64,
            })
            .collect(),
        context_diversity: transition_analysis.context_patterns.len() as f64,
        avg_transition_entropy: transition_analysis
            .context_patterns
            .values()
            .map(|p| p.entropy)
            .sum::<f64>()
            / transition_analysis.context_patterns.len() as f64,
    };

    let transitions_path = results_dir.join("transition_analysis.json");
    fs::write(&transitions_path, serde_json::to_string_pretty(&transition_export)?)?;
    println!("   💾 Transition analysis: {}", transitions_path.display());

    // Save summary report
    let report = serde_json::json!({
        "method": "Agglomerative Clustering with Ward Linkage",
        "normalization": "Z-score (zero mean, unit variance)",
        "total_segments": segments.len(),
        "n_clusters": cluster_stats.len(),
        "silhouette_score": silhouette_score,
        "timestamp": chrono::Utc::now().to_rfc3339()
    });

    let report_path = results_dir.join("summary_report.json");
    fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    println!("   💾 Summary report: {}", report_path.display());

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += (x as f64 - y as f64).powi(2);
    }
    sum.sqrt()
}
