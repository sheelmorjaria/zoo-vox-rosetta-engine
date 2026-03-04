//! Delta Feature Mining - Detecting "Continuous Reusable Motifs"
//! ==============================================================
//!
//! Instead of clustering static feature vectors, this analyzes VELOCITY vectors
//! (directions of change) to detect if animals reuse specific trajectories
//! through acoustic space.
//!
//! Key Insight:
//! - Static Clustering: "What note are you playing?"
//! - Delta Clustering: "What direction are you moving?"
//!
//! If animals reuse a "sweep trajectory" (e.g., always slide down at 5kHz/ms),
//! the positions may be different, but the VELOCITIES will cluster.

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use technical_architecture::HdbscanClustering;

/// Segment from cache
#[derive(Debug, Clone, Deserialize)]
struct CachedSegment {
    source_file: String,
    #[allow(dead_code)]
    context: i32,
    #[allow(dead_code)]
    emitter: i32,
    #[allow(dead_code)]
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    #[allow(dead_code)]
    boundary_type: String,
    features: Vec<f32>,
}

/// Compute delta features between consecutive segments
/// Delta[i] = features[i+1] - features[i]
fn compute_delta_features(segments: &[CachedSegment]) -> Vec<(String, Vec<f32>)> {
    // Group segments by source file
    let mut file_segments: HashMap<String, Vec<(f32, Vec<f32>)>> = HashMap::new();

    for seg in segments {
        file_segments
            .entry(seg.source_file.clone())
            .or_default()
            .push((seg.start_ms, seg.features.clone()));
    }

    // Sort by start time and compute deltas
    let mut deltas: Vec<(String, Vec<f32>)> = Vec::new();

    for (file, mut segs) in file_segments {
        segs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for i in 0..segs.len().saturating_sub(1) {
            let (t1, f1) = &segs[i];
            let (t2, f2) = &segs[i + 1];

            let dt = t2 - t1;

            // Only compute delta for close segments (< 500ms apart)
            if dt < 500.0 && dt > 0.0 {
                let delta: Vec<f32> = f1
                    .iter()
                    .zip(f2.iter())
                    .map(|(a, b)| (b - a) / dt) // Velocity per ms
                    .collect();

                deltas.push((file.clone(), delta));
            }
        }
    }

    deltas
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     DELTA FEATURE MINING - Detecting Continuous Reusable Motifs          ║");
    println!("║     'What direction are you moving?' not 'Where are you?'                ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Try bat cache first, then marmoset
    let bat_cache = Path::new("bat_nbd_cache_normalized");
    let marmoset_cache = Path::new("marmoset_nbd_cache_normalized");

    let (cache_dir, species) = if bat_cache.exists() {
        (bat_cache, "Egyptian Fruit Bat")
    } else if marmoset_cache.exists() {
        (marmoset_cache, "Marmoset")
    } else {
        eprintln!("Error: No normalized cache found");
        std::process::exit(1);
    };

    println!("Species: {}", species);
    println!("Cache: {}", cache_dir.display());
    println!();

    // ---------------------------------------------------------
    // STEP 1: LOAD SEGMENTS
    // ---------------------------------------------------------
    println!("[1/4] Loading Feature Cache...");
    println!("─────────────────────────────────────────────────────────────────────────");

    let cache_files: Vec<_> = fs::read_dir(cache_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    println!("  Found {} cache files", cache_files.len());

    let all_segments: Vec<CachedSegment> = cache_files
        .par_iter()
        .flat_map(|file| {
            let json = match fs::read_to_string(file) {
                Ok(j) => j,
                Err(_) => return None,
            };
            let batch: Vec<CachedSegment> = match serde_json::from_str(&json) {
                Ok(b) => b,
                Err(_) => return None,
            };
            Some(batch)
        })
        .flatten()
        .collect();

    let n_segments = all_segments.len();
    println!("  Loaded {} segments", n_segments);
    println!();

    // ---------------------------------------------------------
    // STEP 2: COMPUTE DELTA FEATURES
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[2/4] Computing Delta Features (Velocities)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  Computing velocity vectors: ΔF = (F[t+1] - F[t]) / Δt");
    println!("  Only considering consecutive segments < 500ms apart");
    println!();

    let deltas = compute_delta_features(&all_segments);
    let n_deltas = deltas.len();

    println!(
        "  Computed {} delta vectors from {} segments",
        n_deltas, n_segments
    );
    println!(
        "  Delta ratio: {:.1}%",
        n_deltas as f64 / n_segments as f64 * 100.0
    );
    println!();

    if n_deltas < 100 {
        eprintln!("Error: Not enough consecutive segments for delta analysis");
        return Ok(());
    }

    // Subsample if needed
    let max_deltas = 50_000;
    let deltas: Vec<_> = if n_deltas > max_deltas {
        println!("  Subsampling to {} deltas...", max_deltas);
        deltas.into_iter().take(max_deltas).collect()
    } else {
        deltas
    };
    let n_deltas = deltas.len();

    // ---------------------------------------------------------
    // STEP 3: CLUSTER DELTA VECTORS
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[3/4] HDBSCAN Clustering on Delta Vectors");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Build delta matrix
    let n_features = deltas[0].1.len();
    let mut delta_matrix = Array2::<f64>::zeros((n_deltas, n_features));

    for (i, (_, delta)) in deltas.iter().enumerate() {
        for (j, &val) in delta.iter().enumerate() {
            if j < n_features {
                delta_matrix[[i, j]] = val as f64;
            }
        }
    }

    println!("  Delta matrix: {} × {}", n_deltas, n_features);
    println!();

    // Cluster delta vectors
    let min_cluster_size = 20;
    let min_samples = 10;

    println!("  min_cluster_size: {}", min_cluster_size);
    println!("  min_samples: {}", min_samples);
    println!();

    let hdbscan = HdbscanClustering::new(min_cluster_size, min_samples)?;
    let labels = hdbscan.fit_predict_hnsw(&delta_matrix)?;

    let stats = hdbscan.get_cluster_stats(&labels);
    let noise_count = labels.iter().filter(|&&l| l == -1).count();
    let noise_ratio = noise_count as f64 / n_deltas as f64;

    // ---------------------------------------------------------
    // STEP 4: RESULTS
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[4/4] Results - Trajectory Motif Detection");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  DELTA CLUSTERING RESULTS                                                │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Delta vectors:    {:>8}                                              ",
        n_deltas
    );
    println!(
        "  │  Clusters:         {:>8}                                              ",
        stats.n_clusters
    );
    println!(
        "  │  Noise:            {:>8} ({:>5.1}%)                                    ",
        noise_count,
        noise_ratio * 100.0
    );
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Show cluster sizes
    let mut cluster_members: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        cluster_members.entry(label).or_default().push(idx);
    }

    let mut sorted_clusters: Vec<_> = cluster_members.iter().filter(|(&l, _)| l != -1).collect();
    sorted_clusters.sort_by_key(|(_, m)| std::cmp::Reverse(m.len()));

    if stats.n_clusters > 0 {
        println!("  Top Trajectory Clusters:");
        for (label, members) in sorted_clusters.iter().take(10) {
            let pct = members.len() as f64 / n_deltas as f64 * 100.0;
            println!(
                "    • Trajectory {:3}: {:6} deltas ({:.1}%)",
                label,
                members.len(),
                pct
            );
        }
        println!();
    }

    // Interpretation
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  SCIENTIFIC INTERPRETATION                                               │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    if stats.n_clusters > 5 && noise_ratio < 0.5 {
        println!("  │  ✓ CONTINUOUS MOTIFS DETECTED!                                          │");
        println!(
            "  │    {} trajectory clusters found                                          ",
            stats.n_clusters
        );
        println!("  │    Animals reuse SPECIFIC DIRECTIONS through acoustic space            │");
        println!("  │                                                                          │");
        println!("  │  → The 'Giant Cluster' was actually a STRUCTURED MANIFOLD!              │");
        println!("  │  → Bag-of-Trajectories approach is recommended                          │");
    } else if stats.n_clusters > 1 && noise_ratio < 0.7 {
        println!("  │  ~ PARTIAL TRAJECTORY STRUCTURE                                         │");
        println!(
            "  │    {} clusters with {:.0}% noise                                          ",
            stats.n_clusters,
            noise_ratio * 100.0
        );
        println!("  │    Some reusable trajectories exist                                     │");
    } else {
        println!("  │  ✗ NO TRAJECTORY STRUCTURE DETECTED                                     │");
        println!(
            "  │    Clusters: {}  |  Noise: {:.0}%                                          ",
            stats.n_clusters,
            noise_ratio * 100.0
        );
        println!("  │                                                                          │");
        println!("  │  → Velocities are random (not reusable)                                 │");
        println!("  │  → This confirms TRUE GRADED CONTINUUM                                  │");
        println!("  │  → Each vocalization is acoustically unique                             │");
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Comparison
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  COMPARISON: Static vs Delta Clustering                                 │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │  Static Features:  1 giant cluster (no discrete motifs)                 │");
    println!(
        "  │  Delta Features:   {} clusters, {:.0}% noise                                ",
        stats.n_clusters.max(1),
        noise_ratio * 100.0
    );
    println!("  │                                                                          │");
    if stats.n_clusters > 5 && noise_ratio < 0.5 {
        println!("  │  → STATIC analysis missed the trajectory structure!                     │");
        println!("  │  → DELTA analysis revealed hidden continuous motifs                     │");
    } else {
        println!("  │  → Both analyses agree: No reusable structure detected                  │");
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
