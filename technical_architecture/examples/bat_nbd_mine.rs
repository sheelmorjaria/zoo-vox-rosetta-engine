//! Egyptian Fruit Bat - NBD Motif Mining from Cache (Parallelized)
//! ================================================================
//!
//! Loads NBD-segmented 105D features and runs HDBSCAN clustering.
//! Uses Rayon for parallel file loading and matrix construction.
//! Uses random subsampling for statistical efficiency.

use ndarray::Array2;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use technical_architecture::HdbscanClustering;

#[derive(Debug, Clone, Deserialize)]
struct CachedSegmentNBD {
    source_file: String,
    context: i32,
    emitter: i32,
    #[allow(dead_code)]
    segment_idx: usize,
    #[allow(dead_code)]
    start_ms: f32,
    #[allow(dead_code)]
    end_ms: f32,
    boundary_type: String,
    features: Vec<f32>,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     EGYPTIAN FRUIT BAT - NBD MOTIF MINING (PARALLEL)                     ║");
    println!("║           Testing 'Hidden Discrete Motifs' with NBD Segments             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let cache_dir = Path::new("bat_nbd_cache_normalized");

    if !cache_dir.exists() {
        eprintln!("Error: NBD cache not found: {}", cache_dir.display());
        eprintln!("Run 'bat_nbd_cache' first.");
        std::process::exit(1);
    }

    // ---------------------------------------------------------
    // STEP 1: PARALLEL FILE LOADING
    // ---------------------------------------------------------
    println!("[1/3] Loading NBD-cached features (Parallel)...");
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
            // Try to read and parse, return empty vec on error for this file
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
    let (full_context_counts, full_emitter_counts, full_boundary_counts) =
        count_stats_parallel(&all_segments);

    // ---------------------------------------------------------
    // DOWNSAMPLE FOR EFFICIENCY
    // ---------------------------------------------------------
    let max_samples = 200_000; // 200k is plenty for statistical significance

    if all_segments.len() > max_samples {
        println!();
        println!(
            "  ⚠  Large dataset detected ({} segments).",
            all_segments.len()
        );
        println!(
            "  Downsampling to {} for HDBSCAN efficiency...",
            max_samples
        );
        println!("  (If discrete motifs exist, they will form clusters in any random sample)");
        println!();

        let mut rng = rand::thread_rng();
        all_segments.shuffle(&mut rng);
        all_segments.truncate(max_samples);
    }

    println!();

    // Print full dataset summary
    print_full_stats(
        total_loaded,
        all_segments.len(),
        &full_context_counts,
        &full_emitter_counts,
        &full_boundary_counts,
    );

    // Note: subsampled stats are similar to full dataset due to random sampling

    // ---------------------------------------------------------
    // STEP 2: PARALLEL MATRIX CONSTRUCTION
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[2/3] Building Feature Matrix (Parallel)");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let n_segments = all_segments.len();
    let n_features = 105;

    // Create matrix and fill in parallel using chunk-based iteration
    let mut feature_matrix = Array2::<f64>::zeros((n_segments, n_features));

    // Get mutable slice and work with chunks (each row is a chunk)
    {
        let matrix_slice = feature_matrix.as_slice_mut().unwrap();

        // Parallel chunk processing - each thread gets distinct chunks
        matrix_slice
            .par_chunks_mut(n_features)
            .zip(all_segments.par_iter())
            .for_each(|(row, seg)| {
                for (j, &val) in seg.features.iter().enumerate().take(n_features) {
                    row[j] = val as f64;
                }
            });
    }

    println!("  Matrix shape: {} × {}", n_segments, n_features);
    println!();

    // ---------------------------------------------------------
    // STEP 3: HDBSCAN CLUSTERING
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[3/3] HDBSCAN Clustering (HNSW)");
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
    let purity = (n_segments - noise_count) as f64 / n_segments as f64;
    let noise_ratio = noise_count as f64 / n_segments as f64;

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("RESULTS");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  OVERALL STATISTICS                                                      │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Segments:       {:>8}                                              ",
        n_segments
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
        let mut ctx_counts: HashMap<i32, usize> = HashMap::new();
        let mut btype_counts: HashMap<String, usize> = HashMap::new();

        for &idx in member_indices.iter() {
            let ctx = all_segments[idx].context;
            let btype = &all_segments[idx].boundary_type;
            *ctx_counts.entry(ctx).or_insert(0) += 1;
            *btype_counts.entry(btype.clone()).or_insert(0) += 1;
        }

        let unique_files: std::collections::HashSet<_> = member_indices
            .iter()
            .map(|&idx| &all_segments[idx].source_file)
            .collect();

        println!(
            "  │  Cluster {} ({} seg, {} files)                        ",
            label,
            member_indices.len(),
            unique_files.len()
        );

        // Show top context
        if let Some((ctx, cnt)) = ctx_counts.iter().max_by_key(|(_, c)| *c) {
            let pct = *cnt as f64 / member_indices.len() as f64 * 100.0;
            print!("  │    Context {}: {:.0}%", ctx, pct);
        }
        // Show top boundary type
        if let Some((btype, cnt)) = btype_counts.iter().max_by_key(|(_, c)| *c) {
            let pct = *cnt as f64 / member_indices.len() as f64 * 100.0;
            print!(" | {}: {:.0}%", btype, pct);
        }
        println!();
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // ---------------------------------------------------------
    // DISSECTING THE GIANT: Cluster 0 Analysis
    // ---------------------------------------------------------
    if let Some(cluster_0_indices) = cluster_members.get(&0) {
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!("DISSECTING THE GIANT: Cluster 0 Analysis");
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!();

        println!(
            "  Cluster 0 contains {} segments ({:.1}% of all data)",
            cluster_0_indices.len(),
            cluster_0_indices.len() as f64 / n_segments as f64 * 100.0
        );
        println!();

        // Count contexts WITHIN Cluster 0
        let mut cluster_0_contexts: HashMap<i32, usize> = HashMap::new();
        let mut cluster_0_emitters: HashMap<i32, usize> = HashMap::new();
        let mut cluster_0_btypes: HashMap<String, usize> = HashMap::new();

        for &idx in cluster_0_indices {
            let ctx = all_segments[idx].context;
            let emit = all_segments[idx].emitter;
            let btype = &all_segments[idx].boundary_type;
            *cluster_0_contexts.entry(ctx).or_insert(0) += 1;
            *cluster_0_emitters.entry(emit).or_insert(0) += 1;
            *cluster_0_btypes.entry(btype.clone()).or_insert(0) += 1;
        }

        // Sort contexts by frequency
        let mut ctx_vec: Vec<_> = cluster_0_contexts.iter().collect();
        ctx_vec.sort_by(|a, b| b.1.cmp(a.1));

        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!("  │  CONTEXT DISTRIBUTION INSIDE CLUSTER 0                                  │");
        println!("  ├─────────────────────────────────────────────────────────────────────────┤");

        let top_ctx_count = ctx_vec.first().map(|(_, &c)| c).unwrap_or(0);
        let top_ctx_pct = top_ctx_count as f64 / cluster_0_indices.len() as f64 * 100.0;

        for (ctx, count) in ctx_vec.iter() {
            let pct = **count as f64 / cluster_0_indices.len() as f64 * 100.0;
            let bar_len = (pct / 2.5) as usize; // Max 40 chars
            let bar = "█".repeat(bar_len);
            println!(
                "  │  Context {:2}: {:6} ({:5.1}%) {:<40}",
                ctx, count, pct, bar
            );
        }
        println!("  └─────────────────────────────────────────────────────────────────────────┘");
        println!();

        // Boundary type distribution
        let mut btype_vec: Vec<_> = cluster_0_btypes.iter().collect();
        btype_vec.sort_by(|a, b| b.1.cmp(a.1));

        println!("  Boundary Types in Cluster 0:");
        for (btype, count) in btype_vec.iter() {
            let pct = **count as f64 / cluster_0_indices.len() as f64 * 100.0;
            println!("    • {:14}: {:6} ({:5.1}%)", btype, count, pct);
        }
        println!();

        // Top emitters in Cluster 0
        let mut emit_vec: Vec<_> = cluster_0_emitters.iter().collect();
        emit_vec.sort_by(|a, b| b.1.cmp(a.1));

        println!("  Top Bats in Cluster 0:");
        for (emit, count) in emit_vec.iter().take(5) {
            let pct = **count as f64 / cluster_0_indices.len() as f64 * 100.0;
            println!("    • Bat {:5}: {:6} ({:5.1}%)", emit, count, pct);
        }
        println!();

        // Diagnostic interpretation
        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!("  │  DIAGNOSTIC INTERPRETATION                                               │");
        println!("  ├─────────────────────────────────────────────────────────────────────────┤");

        // Calculate entropy of context distribution
        let total = cluster_0_indices.len() as f64;
        let entropy: f64 = ctx_vec
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

        let max_entropy = (ctx_vec.len() as f64).log2(); // Max entropy if uniform
        let normalized_entropy = entropy / max_entropy;

        println!("  │  Context Diversity Metrics:                                             ");
        println!(
            "  │    Unique contexts: {:3}                                                ",
            ctx_vec.len()
        );
        println!(
            "  │    Entropy:          {:.3} bits (max: {:.3})                            ",
            entropy, max_entropy
        );
        println!(
            "  │    Normalized:       {:.1}% (100% = uniform, 0% = single context)       ",
            normalized_entropy * 100.0
        );
        println!(
            "  │    Top context:      {:.1}% of cluster                                  ",
            top_ctx_pct
        );
        println!("  │                                                                          ");

        if top_ctx_pct > 70.0 {
            println!(
                "  │  → CLUSTERING WORKED: Dominated by Context {:2}                        ",
                ctx_vec.first().map(|(c, _)| **c).unwrap_or(-1)
            );
            println!("  │    Found a semantically meaningful cluster.                            ");
        } else if normalized_entropy > 0.8 {
            println!("  │  → FEATURES TOO SIMILAR: Uniform context mix                           ");
            println!(
                "  │    105D features don't discriminate between behavioral contexts.        "
            );
            println!("  │    Context is encoded in RATE/DYNAMICS, not TEXTURE.                   ");
            println!(
                "  │                                                                          "
            );
            println!("  │    ⚠ THIS IS THE 'HOLY GRAIL' FINDING! ⚠                               ");
            println!("  │    Same acoustic substrate used for different behaviors.               ");
        } else {
            println!("  │  → PARTIAL DISCRIMINATION: Mixed but not uniform                       ");
            println!("  │    Some acoustic patterns correlate with context.                      ");
        }
        println!("  └─────────────────────────────────────────────────────────────────────────┘");
        println!();
    }

    // Interpretation
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("INTERPRETATION");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  Expected for Egyptian Fruit Bats (prosodic modulation):");
    println!("    • Purity: 10-20%  •  Noise: 80-90%");
    println!();

    println!("  Observed with NBD segmentation:");
    println!(
        "    • Purity: {:.1}%  •  Noise: {:.1}%",
        purity * 100.0,
        noise_ratio * 100.0
    );
    println!();

    if purity < 0.25 {
        println!("  ╔═════════════════════════════════════════════════════════════════════════╗");
        println!("  ║  ✓ CONFIRMED: LOW MOTIF REUSE (PROSODIC MODULATION)                   ║");
        println!("  ╠═════════════════════════════════════════════════════════════════════════╣");
        println!("  ║  Egyptian fruit bats use CONTINUOUSLY MODULATED FM sweeps.            ║");
        println!("  ║  Each call is unique - no reusable acoustic vocabulary.               ║");
        println!("  ║                                                                        ║");
        println!("  ║  → Use DIRECT 105D SIMILARITY                                          ║");
        println!("  ║  → Bag-of-Phrases will FAIL                                            ║");
        println!("  ╚═════════════════════════════════════════════════════════════════════════╝");
    } else if purity < 0.50 {
        println!("  ~ MODERATE motif reuse - hybrid approach recommended");
    } else {
        println!("  ⚠ HIGH motif reuse - unexpected for bats!");
        println!("  This suggests acoustic patterns are reused more than expected.");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}

/// Count stats in parallel using fold/reduce pattern
fn count_stats_parallel(
    segments: &[CachedSegmentNBD],
) -> (
    HashMap<i32, usize>,
    HashMap<i32, usize>,
    HashMap<String, usize>,
) {
    use std::sync::Mutex;

    let context_counts = Mutex::new(HashMap::new());
    let emitter_counts = Mutex::new(HashMap::new());
    let boundary_counts = Mutex::new(HashMap::new());

    segments.par_iter().for_each(|seg| {
        if let Ok(mut ctx) = context_counts.lock() {
            *ctx.entry(seg.context).or_insert(0) += 1;
        }
        if let Ok(mut emit) = emitter_counts.lock() {
            *emit.entry(seg.emitter).or_insert(0) += 1;
        }
        if let Ok(mut btype) = boundary_counts.lock() {
            *btype.entry(seg.boundary_type.clone()).or_insert(0) += 1;
        }
    });

    (
        context_counts.into_inner().unwrap(),
        emitter_counts.into_inner().unwrap(),
        boundary_counts.into_inner().unwrap(),
    )
}

fn print_full_stats(
    total_loaded: usize,
    subsampled: usize,
    context_counts: &HashMap<i32, usize>,
    emitter_counts: &HashMap<i32, usize>,
    boundary_counts: &HashMap<String, usize>,
) {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    CACHED DATA SUMMARY                                    ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Total segments:    {:>8}                                             ",
        total_loaded
    );
    if total_loaded != subsampled {
        println!(
            "║  Subsampled:        {:>8}                                             ",
            subsampled
        );
    }
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("Segments by Boundary Type (NBD):");
    let mut b_sorted: Vec<_> = boundary_counts.iter().collect();
    b_sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (btype, count) in &b_sorted {
        let pct = **count as f64 / total_loaded as f64 * 100.0;
        println!("  • {:14}: {:6} ({:5.1}%)", btype, count, pct);
    }
    println!();

    println!("Top Contexts:");
    let mut ctx_sorted: Vec<_> = context_counts.iter().collect();
    ctx_sorted.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (ctx, count) in ctx_sorted.iter().take(8) {
        let pct = **count as f64 / total_loaded as f64 * 100.0;
        println!("  • Context {:2}: {:6} ({:5.1}%)", ctx, count, pct);
    }
    println!();

    println!("Top Bats (Emitters):");
    let mut emit_sorted: Vec<_> = emitter_counts.iter().collect();
    emit_sorted.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (emit, count) in emit_sorted.iter().take(8) {
        let pct = **count as f64 / total_loaded as f64 * 100.0;
        println!("  • Bat {:5}: {:6} ({:5.1}%)", emit, count, pct);
    }
    println!();
}
