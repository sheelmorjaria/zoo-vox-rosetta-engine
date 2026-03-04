//! Marmoset - Pitch Geometry Shape Mining
//! =======================================
//!
//! Extracts the "Shape Slice" (Pitch Geometry) from cached 105D features
//! and clusters to detect reusable pitch contours (e.g., 'Rising Arch', 'Falling Sweep').
//!
//! 105D Stack Layout:
//! - Layer 1 (0-44): Base Physics
//! - Layer 2 (45-74): Macro Texture
//!   - 45-47: Harmonic Texture
//!   - 53-59: Pitch Geometry (Slope, Curvature, Inflections) <-- THIS
//! - Layer 3 (75-104): Micro Texture

use ndarray::Array2;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use technical_architecture::HdbscanClustering;

#[derive(Debug, Clone, Deserialize)]
struct CachedSegmentNBD {
    audio_file: String,
    call_type: String,
    #[allow(dead_code)]
    label_id: i32,
    #[allow(dead_code)]
    segment_idx: usize,
    features: Vec<f32>,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     MARMOSET - PITCH GEOMETRY SHAPE MINING                                ║");
    println!("║     Testing 'Reusable Pitch Contours' (Slope, Curvature, Inflection)      ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let cache_dir = Path::new("marmoset_nbd_cache_normalized");

    if !cache_dir.exists() {
        eprintln!("Error: Cache not found: {}", cache_dir.display());
        std::process::exit(1);
    }

    // ---------------------------------------------------------
    // STEP 1: LOAD CACHE
    // ---------------------------------------------------------
    println!("[1/4] Loading Normalized Cache...");
    println!("─────────────────────────────────────────────────────────────────────────");

    let cache_files: Vec<_> = fs::read_dir(cache_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    println!("  Found {} cache files", cache_files.len());

    let all_segments: Vec<CachedSegmentNBD> = cache_files
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

    let n_samples = all_segments.len();
    println!("  Loaded {} segments", n_samples);
    println!();

    // Count call types
    let call_type_counts: HashMap<String, usize> =
        all_segments.iter().fold(HashMap::new(), |mut acc, seg| {
            *acc.entry(seg.call_type.clone()).or_insert(0) += 1;
            acc
        });

    println!("Call Type Distribution:");
    let mut sorted_types: Vec<_> = call_type_counts.iter().collect();
    sorted_types.sort_by(|a, b| b.1.cmp(a.1));
    for (call_type, count) in &sorted_types {
        let pct = **count as f64 / n_samples as f64 * 100.0;
        println!("  • {:14}: {:5} ({:5.1}%)", call_type, count, pct);
    }
    println!();

    // ---------------------------------------------------------
    // STEP 2: EXTRACT PITCH GEOMETRY SLICE
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[2/4] Extracting Pitch Geometry (Shape Slice)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // 105D Stack Definition:
    // Layer 2: Macro Texture (45-74)
    //   - 45-52: Harmonic Texture (8D)
    //   - 53-59: Pitch Geometry (7D) <-- Shape features
    //   - 60-69: GLCM Texture (10D)
    //   - 70-74: Temporal Texture (5D)

    let shape_start = 53;
    let shape_end = 60; // exclusive
    let n_shape_features = shape_end - shape_start;

    println!("  105D Feature Stack Layout:");
    println!("    • Layer 1 (0-44):  Base Physics");
    println!("    • Layer 2 (45-74): Macro Texture");
    println!("        - 45-52: Harmonic Texture (8D)");
    println!("        - 53-59: Pitch Geometry (7D) <-- EXTRACTING");
    println!("        - 60-69: GLCM Texture (10D)");
    println!("        - 70-74: Temporal Texture (5D)");
    println!("    • Layer 3 (75-104): Micro Texture");
    println!();
    println!("  Shape Features (indices {}..{}):", shape_start, shape_end);
    println!("    • Pitch slope ratio");
    println!("    • FM slope modulated");
    println!("    • Pitch trajectory curvature");
    println!("    • Inflection indicators");
    println!();

    // Build Shape Matrix
    let mut feature_matrix = Array2::<f64>::zeros((n_samples, n_shape_features));

    {
        let matrix_slice = feature_matrix.as_slice_mut().unwrap();
        matrix_slice
            .par_chunks_mut(n_shape_features)
            .zip(all_segments.par_iter())
            .for_each(|(row, seg)| {
                for i in 0..n_shape_features {
                    let idx = shape_start + i;
                    if idx < seg.features.len() {
                        row[i] = seg.features[idx] as f64;
                    }
                }
            });
    }

    println!("  Shape Matrix: {} × {}", n_samples, n_shape_features);
    println!();

    // ---------------------------------------------------------
    // STEP 3: HDBSCAN CLUSTERING
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[3/4] HDBSCAN Clustering on Pitch Geometry");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let min_cluster_size = 20;
    let min_samples = 10;

    println!("  min_cluster_size: {}", min_cluster_size);
    println!("  min_samples: {}", min_samples);
    println!();

    let hdbscan = HdbscanClustering::new(min_cluster_size, min_samples)?;
    let labels = hdbscan.fit_predict_hnsw(&feature_matrix)?;

    let stats = hdbscan.get_cluster_stats(&labels);
    let noise_count = labels.iter().filter(|&&l| l == -1).count();
    let coverage = (n_samples - noise_count) as f64 / n_samples as f64;
    let noise_ratio = noise_count as f64 / n_samples as f64;

    // ---------------------------------------------------------
    // STEP 4: RESULTS
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[4/4] Results - Pitch Contour Analysis");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  SHAPE CLUSTERING RESULTS                                                │");
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
        "  │  Coverage:       {:>8.1}%                                            ",
        coverage * 100.0
    );
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Cluster composition
    let mut cluster_members: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        cluster_members.entry(label).or_default().push(idx);
    }

    // Find largest cluster
    let largest_cluster = cluster_members
        .iter()
        .filter(|(&l, _)| l != -1)
        .max_by_key(|(_, m)| m.len());

    if let Some((&cluster_id, members)) = largest_cluster {
        println!("  Largest Cluster ({}) Analysis:", cluster_id);
        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!("  │  CALL TYPE DISTRIBUTION                                                  │");
        println!("  ├─────────────────────────────────────────────────────────────────────────┤");

        let mut type_counts: HashMap<&str, usize> = HashMap::new();
        for &idx in members {
            let call_type = all_segments[idx].call_type.as_str();
            *type_counts.entry(call_type).or_insert(0) += 1;
        }

        let mut sorted: Vec<_> = type_counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        for (call_type, count) in &sorted {
            let pct = **count as f64 / members.len() as f64 * 100.0;
            let bar_len = (pct / 2.5) as usize;
            let bar = "█".repeat(bar_len);
            println!(
                "  │  {:14}: {:5} ({:5.1}%) {:<40}",
                call_type, count, pct, bar
            );
        }
        println!("  └─────────────────────────────────────────────────────────────────────────┘");
        println!();

        // Entropy calculation
        let total = members.len() as f64;
        let entropy: f64 = sorted
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

        let max_entropy = (sorted.len() as f64).log2();
        let normalized_entropy = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        };

        // Top type percentage
        let top_type_pct = sorted
            .first()
            .map(|(_, &c)| c as f64 / total * 100.0)
            .unwrap_or(0.0);

        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!("  │  DIAGNOSTIC INTERPRETATION                                               │");
        println!("  ├─────────────────────────────────────────────────────────────────────────┤");
        println!("  │  Pitch Geometry Diversity:                                              ");
        println!(
            "  │    Unique call types: {:3}                                              ",
            sorted.len()
        );
        println!(
            "  │    Entropy:          {:.3} bits (max: {:.3})                            ",
            entropy, max_entropy
        );
        println!(
            "  │    Normalized:       {:.1}% (100% = uniform, 0% = single type)          ",
            normalized_entropy * 100.0
        );
        println!(
            "  │    Top type:         {:.1}% of cluster                                  ",
            top_type_pct
        );
        println!("  │                                                                          ");

        if stats.n_clusters > 5 && normalized_entropy < 0.7 {
            println!(
                "  │  ✓ DISCRETE PITCH CONTOURS FOUND!                                       │"
            );
            println!(
                "  │    {} distinct shape clusters detected                                   ",
                stats.n_clusters
            );
            println!(
                "  │    → Marmosets reuse specific pitch trajectories                        │"
            );
            println!(
                "  │    → Bag-of-Shapes approach WILL WORK                                    │"
            );
        } else if normalized_entropy > 0.9 {
            println!(
                "  │  ✗ NO DISCRETE SHAPES: Uniform contour mix                              │"
            );
            println!(
                "  │                                                                          │"
            );
            println!(
                "  │  Even 'Slope' and 'Curvature' are mixed across call types.              │"
            );
            println!(
                "  │  → Marmosets use CONTINUOUS pitch modulation.                           │"
            );
            println!(
                "  │  → No reusable 'Arch' or 'Sweep' templates found.                      │"
            );
            println!(
                "  │  → This confirms the Graded Continuum hypothesis.                      │"
            );
        } else {
            println!(
                "  │  ~ PARTIAL DISCRIMINATION: Mixed but not uniform                        │"
            );
            println!(
                "  │    Some pitch contour structure may exist                               │"
            );
        }
        println!("  └─────────────────────────────────────────────────────────────────────────┘");
    }

    // Show top clusters if multiple
    if stats.n_clusters > 1 {
        println!();
        println!("  All Clusters:");

        let mut sorted_clusters: Vec<_> =
            cluster_members.iter().filter(|(&l, _)| l != -1).collect();
        sorted_clusters.sort_by_key(|(_, m)| std::cmp::Reverse(m.len()));

        for (label, members) in sorted_clusters.iter().take(10) {
            let mut type_counts: HashMap<&str, usize> = HashMap::new();
            for &idx in members.iter() {
                *type_counts
                    .entry(all_segments[idx].call_type.as_str())
                    .or_insert(0) += 1;
            }
            let mut sorted: Vec<_> = type_counts.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));

            print!("    Cluster {} ({}): ", label, members.len());
            for (ct, cnt) in sorted.iter().take(2) {
                let pct = **cnt as f64 / members.len() as f64 * 100.0;
                print!("{}={:.0}% ", ct, pct);
            }
            println!();
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
