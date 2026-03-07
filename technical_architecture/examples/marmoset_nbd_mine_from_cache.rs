//! Marmoset NBD Motif Mining from Normalized Cache
//! =================================================
//!
//! Loads pre-normalized 105D features and runs HDBSCAN clustering.
//! Uses the same parallel/subsampling approach as bat_nbd_mine.

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use ndarray::Array2;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use technical_architecture::HdbscanClustering;

/// Normalized segment from cache
#[derive(Debug, Clone, Deserialize)]
struct CachedSegmentNBD {
    audio_file: String,
    call_type: String,
    label_id: i32,
    segment_idx: usize,
    features: Vec<f32>,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     MARMOSET - NBD MOTIF MINING FROM NORMALIZED CACHE                    ║");
    println!("║           Testing 'Hidden Discrete Motifs' with Normalized 105D          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let cache_dir = Path::new("marmoset_nbd_cache_normalized");

    if !cache_dir.exists() {
        eprintln!("Error: Normalized cache not found: {}", cache_dir.display());
        eprintln!("Run 'marmoset_normalize_cache' first.");
        std::process::exit(1);
    }

    // ---------------------------------------------------------
    // STEP 1: PARALLEL FILE LOADING
    // ---------------------------------------------------------
    println!("[1/3] Loading Normalized 105D Feature Cache (Parallel)...");
    println!("─────────────────────────────────────────────────────────────────────────");

    // Collect file paths first
    let cache_files: Vec<_> = fs::read_dir(cache_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    println!("  Found {} cache files", cache_files.len());
    println!("  Parsing files in parallel...");

    // Use Rayon to read and parse files in parallel
    let mut all_segments: Vec<CachedSegmentNBD> = cache_files
        .par_iter()
        .flat_map(|file| {
            let json = match fs::read_to_string(file) {
                Ok(j) => j,
                Err(_) => return None,
            };
            let batch: Vec<CachedSegmentNBD> = match serde_json::from_str(&json) {
                Ok(b) => b,
                Err(_) => return None,
            };
            Some(batch)
        })
        .flatten()
        .collect();

    if all_segments.is_empty() {
        eprintln!("No segments found.");
        return Ok(());
    }

    let total_loaded = all_segments.len();
    println!("  Loaded {} segments", total_loaded);

    // Count FULL dataset stats before subsampling
    let full_calltype_counts: HashMap<String, usize> = all_segments.iter().fold(HashMap::new(), |mut acc, seg| {
        *acc.entry(seg.call_type.clone()).or_insert(0) += 1;
        acc
    });

    // ---------------------------------------------------------
    // DOWNSAMPLE FOR EFFICIENCY (if needed)
    // ---------------------------------------------------------
    let max_samples = 200_000;

    if all_segments.len() > max_samples {
        println!();
        println!("  ⚠  Large dataset detected ({} segments).", all_segments.len());
        println!("  Downsampling to {} for HDBSCAN efficiency...", max_samples);
        println!();

        let mut rng = rand::thread_rng();
        all_segments.shuffle(&mut rng);
        all_segments.truncate(max_samples);
    }

    let n_samples = all_segments.len();

    // Build feature matrix in parallel
    let n_features = 105;
    let mut feature_matrix = Array2::<f64>::zeros((n_samples, n_features));

    {
        let matrix_slice = feature_matrix.as_slice_mut().unwrap();
        matrix_slice
            .par_chunks_mut(n_features)
            .zip(all_segments.par_iter())
            .for_each(|(row, seg)| {
                for (j, &val) in seg.features.iter().enumerate().take(n_features) {
                    row[j] = val as f64;
                }
            });
    }

    println!("  Matrix shape: {} × {}", n_samples, n_features);
    println!();

    // Print summary
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    DATA SUMMARY                                           ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Total samples:  {:>8}                                              ",
        total_loaded
    );
    if total_loaded != n_samples {
        println!(
            "║  Subsampled:     {:>8}                                              ",
            n_samples
        );
    }
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("Call Type Distribution:");
    let mut sorted_types: Vec<_> = full_calltype_counts.iter().collect();
    sorted_types.sort_by(|a, b| b.1.cmp(a.1));
    for (call_type, count) in &sorted_types {
        let pct = **count as f64 / total_loaded as f64 * 100.0;
        println!("  • {:14}: {:5} ({:5.1}%)", call_type, count, pct);
    }
    println!();

    // ---------------------------------------------------------
    // STEP 2: HDBSCAN CLUSTERING
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[2/3] HDBSCAN Clustering (Normalized 105D Space)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let min_cluster_size = 10;
    let min_samples = 5;

    println!("  min_cluster_size: {}", min_cluster_size);
    println!("  min_samples: {}", min_samples);
    println!("  Algorithm: HNSW (Hierarchical Navigable Small World)");
    println!();

    let hdbscan = HdbscanClustering::new(min_cluster_size, min_samples)?;
    let labels = hdbscan.fit_predict_hnsw(&feature_matrix)?;

    // Results
    let stats = hdbscan.get_cluster_stats(&labels);
    let noise_count = labels.iter().filter(|&&l| l == -1).count();
    let purity = (n_samples - noise_count) as f64 / n_samples as f64;
    let noise_ratio = noise_count as f64 / n_samples as f64;

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[3/3] Results");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  OVERALL STATISTICS                                                      │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Samples:        {:>8}                                              ",
        n_samples
    );
    println!(
        "  │  Clusters:       {:>8}                                              ",
        stats.n_clusters
    );
    println!(
        "  │  Noise:          {:>8} ({:>5.1}%)                                    ",
        noise_count,
        noise_ratio * 100.0
    );
    println!(
        "  │  Purity:         {:>8.1}%                                            ",
        purity * 100.0
    );
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Cluster composition
    let mut cluster_members: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        cluster_members.entry(label).or_default().push(idx);
    }

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  TOP CLUSTERS (by size)                                                  │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    let mut sorted_clusters: Vec<_> = cluster_members.iter().filter(|(&l, _)| l != -1).collect();
    sorted_clusters.sort_by_key(|(_, m)| std::cmp::Reverse(m.len()));

    for (label, member_indices) in sorted_clusters.iter().take(10) {
        let mut type_counts: HashMap<&str, usize> = HashMap::new();

        for &idx in member_indices.iter() {
            let call_type = all_segments[idx].call_type.as_str();
            *type_counts.entry(call_type).or_insert(0) += 1;
        }

        let unique_files: std::collections::HashSet<_> = member_indices
            .iter()
            .map(|&idx| all_segments[idx].audio_file.as_str())
            .collect();

        println!(
            "  │  Cluster {} ({} samples, {} files)                        ",
            label,
            member_indices.len(),
            unique_files.len()
        );

        // Show top 3 call types
        let mut sorted_types: Vec<_> = type_counts.iter().collect();
        sorted_types.sort_by(|a, b| b.1.cmp(a.1));

        for (call_type, count) in sorted_types.iter().take(3) {
            let pct = **count as f64 / member_indices.len() as f64 * 100.0;
            println!("  │    • {:14}: {:4} ({:.0}%)", call_type, count, pct);
        }
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // DISSECT GIANT CLUSTER if present
    if let Some(cluster_0_indices) = cluster_members.get(&0) {
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!("DISSECTING THE GIANT: Cluster 0 Analysis");
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!();

        println!(
            "  Cluster 0 contains {} samples ({:.1}% of all data)",
            cluster_0_indices.len(),
            cluster_0_indices.len() as f64 / n_samples as f64 * 100.0
        );
        println!();

        // Count call types WITHIN Cluster 0
        let mut cluster_0_types: HashMap<&str, usize> = HashMap::new();
        for &idx in cluster_0_indices {
            let call_type = all_segments[idx].call_type.as_str();
            *cluster_0_types.entry(call_type).or_insert(0) += 1;
        }

        // Sort by frequency
        let mut type_vec: Vec<_> = cluster_0_types.iter().collect();
        type_vec.sort_by(|a, b| b.1.cmp(a.1));

        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!("  │  CALL TYPE DISTRIBUTION INSIDE CLUSTER 0                                │");
        println!("  ├─────────────────────────────────────────────────────────────────────────┤");

        let top_type_count = type_vec.first().map(|(_, &c)| c).unwrap_or(0);
        let top_type_pct = top_type_count as f64 / cluster_0_indices.len() as f64 * 100.0;

        for (call_type, count) in type_vec.iter() {
            let pct = **count as f64 / cluster_0_indices.len() as f64 * 100.0;
            let bar_len = (pct / 2.5) as usize;
            let bar = "█".repeat(bar_len);
            println!("  │  {:14}: {:5} ({:5.1}%) {:<40}", call_type, count, pct, bar);
        }
        println!("  └─────────────────────────────────────────────────────────────────────────┘");
        println!();

        // Calculate entropy
        let total = cluster_0_indices.len() as f64;
        let entropy: f64 = type_vec
            .iter()
            .map(|(_, &count)| {
                let p = count as f64 / total;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();

        let max_entropy = (type_vec.len() as f64).log2();
        let normalized_entropy = if max_entropy > 0.0 { entropy / max_entropy } else { 0.0 };

        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!("  │  DIAGNOSTIC INTERPRETATION                                               │");
        println!("  ├─────────────────────────────────────────────────────────────────────────┤");
        println!("  │  Call Type Diversity Metrics:                                           ");
        println!(
            "  │    Unique types:   {:3}                                                 ",
            type_vec.len()
        );
        println!(
            "  │    Entropy:         {:.3} bits (max: {:.3})                            ",
            entropy, max_entropy
        );
        println!(
            "  │    Normalized:      {:.1}% (100% = uniform, 0% = single type)          ",
            normalized_entropy * 100.0
        );
        println!(
            "  │    Top type:        {:.1}% of cluster                                   ",
            top_type_pct
        );
        println!("  │                                                                          ");

        if top_type_pct > 70.0 {
            let dominant_type = type_vec.first().map(|(t, _)| **t).unwrap_or("Unknown");
            println!(
                "  │  → CLUSTERING WORKED: Dominated by {:14}                        ",
                dominant_type
            );
            println!("  │    Found a semantically meaningful cluster.                            ");
        } else if normalized_entropy > 0.8 {
            println!("  │  → FEATURES TOO SIMILAR: Uniform type mix                              ");
            println!("  │    Normalized 105D features don't discriminate between call types.     ");
            println!("  │    Marmosets use GRADED CONTINUUM, not discrete motifs.               ");
            println!("  │                                                                          ");
            println!("  │    ⚠ THIS IS THE 'HOLY GRAIL' FINDING! ⚠                               ");
            println!("  │    Same acoustic substrate used for different call types.             ");
        } else {
            println!("  │  → PARTIAL DISCRIMINATION: Mixed but not uniform                       ");
            println!("  │    Some acoustic patterns correlate with call type.                    ");
        }
        println!("  └─────────────────────────────────────────────────────────────────────────┘");
        println!();
    }

    // Interpretation
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("INTERPRETATION");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  Expected for Marmosets (graded vocal system):");
    println!("    • Purity: 30-50%  •  Noise: 50-70%");
    println!();

    println!("  Observed:");
    println!(
        "    • Purity: {:.1}%  •  Noise: {:.1}%",
        purity * 100.0,
        noise_ratio * 100.0
    );
    println!();

    if purity > 0.7 {
        println!("  ⚠ HIGH PURITY - But check cluster entropy!");
        println!("  If entropy is high (>80%), clustering failed to discriminate.");
    } else if purity > 0.5 {
        println!("  ~ MODERATE motif reuse - hybrid system");
        println!("  → Use BOTH Bag-of-Phrases AND Direct 105D similarity");
    } else {
        println!("  ✗ LOW MOTIF REUSE - True graded continuum");
        println!("  → Use Direct 105D similarity (Bag-of-Phrases will fail)");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
