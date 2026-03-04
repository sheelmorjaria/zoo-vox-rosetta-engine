//! Egyptian Fruit Bat - Phase 2: Motif Mining from Cache
//! =====================================================
//!
//! Loads cached 105D features and runs HDBSCAN to test motif reuse.
//! Expected: LOW purity (10-20%), HIGH noise (80-90%) - prosodic modulation

use ndarray::Array2;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use technical_architecture::HdbscanClustering;

/// Cached segment data
#[derive(Debug, Clone, Deserialize)]
struct CachedSegment {
    source_file: String,
    context: i32,
    emitter: i32,
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    features: Vec<f32>,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     EGYPTIAN FRUIT BAT - PHASE 2: MOTIF MINING FROM CACHE                 ║");
    println!("║              Testing 'Hidden Discrete Motifs' Hypothesis                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let cache_dir = Path::new("bat_feature_cache");

    // Check cache exists
    if !cache_dir.exists() {
        eprintln!("Error: Cache directory not found: {}", cache_dir.display());
        eprintln!("Run 'bat_cache_features' first to build the cache.");
        std::process::exit(1);
    }

    // Load all cached segments
    println!("Loading cached features...");
    println!("─────────────────────────────────────────────────────────────────────────");

    let mut all_segments: Vec<CachedSegment> = Vec::new();
    let mut context_counts: HashMap<i32, usize> = HashMap::new();
    let mut emitter_counts: HashMap<i32, usize> = HashMap::new();

    let cache_files: Vec<_> = fs::read_dir(cache_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    println!("Found {} cache files", cache_files.len());

    for cache_file in &cache_files {
        let json = fs::read_to_string(cache_file)?;
        let batch: Vec<CachedSegment> = serde_json::from_str(&json)?;

        for seg in &batch {
            *context_counts.entry(seg.context).or_insert(0) += 1;
            *emitter_counts.entry(seg.emitter).or_insert(0) += 1;
        }

        all_segments.extend(batch);
    }

    println!("  Total segments loaded: {}", all_segments.len());
    println!();

    println!("Segments by Context:");
    let mut ctx_sorted: Vec<_> = context_counts.iter().collect();
    ctx_sorted.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (ctx, count) in ctx_sorted.iter().take(10) {
        let pct = *count as f64 / all_segments.len() as f64 * 100.0;
        println!("  • Context {:2}: {:6} ({:5.1}%)", ctx, count, pct);
    }
    println!();

    println!("Top Emitters (Bats):");
    let mut emit_sorted: Vec<_> = emitter_counts.iter().collect();
    emit_sorted.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (emit, count) in emit_sorted.iter().take(10) {
        let pct = *count as f64 / all_segments.len() as f64 * 100.0;
        println!("  • Bat {:4}: {:6} ({:5.1}%)", emit, count, pct);
    }
    println!();

    // Build feature matrix
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("STEP 2: Building Feature Matrix and HDBSCAN Clustering");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let n_segments = all_segments.len();
    let n_features = 105;

    let mut feature_matrix = Array2::<f64>::zeros((n_segments, n_features));
    for (i, seg) in all_segments.iter().enumerate() {
        for (j, &val) in seg.features.iter().enumerate().take(n_features) {
            feature_matrix[[i, j]] = val as f64;
        }
    }

    println!(
        "  Feature matrix: {} segments × {} features",
        n_segments, n_features
    );

    // Adaptive min_cluster_size based on dataset size
    let min_cluster_size = (n_segments / 500).max(20).min(100);
    let min_samples = (min_cluster_size / 2).max(5);

    println!("  min_cluster_size: {}", min_cluster_size);
    println!("  min_samples: {}", min_samples);
    println!();
    println!("  Running HDBSCAN on {} segments...", n_segments);
    println!("  (This may take a few minutes...)");
    println!();

    let hdbscan = HdbscanClustering::new(min_cluster_size, min_samples)?;
    let labels = hdbscan.fit_predict(&feature_matrix)?;

    let stats = hdbscan.get_cluster_stats(&labels);
    let noise_count = labels.iter().filter(|&&l| l == -1).count();
    let purity = (n_segments - noise_count) as f64 / n_segments as f64;
    let noise_ratio = noise_count as f64 / n_segments as f64;

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("STEP 3: Results Analysis");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  OVERALL STATISTICS                                                      │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Total segments: {:>8}                                              ",
        n_segments
    );
    println!(
        "  │  Clusters found: {:>8}                                              ",
        stats.n_clusters
    );
    println!(
        "  │  Noise points:   {:>8} ({:>5.1}%)                                    ",
        noise_count,
        noise_ratio * 100.0
    );
    println!(
        "  │  Purity:         {:>8.1}%                                            ",
        purity * 100.0
    );
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Cluster composition by context
    let mut cluster_members: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        cluster_members.entry(label).or_default().push(idx);
    }

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  CLUSTER COMPOSITION BY CONTEXT                                          │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    let mut sorted_clusters: Vec<_> = cluster_members.iter().collect();
    sorted_clusters.sort_by_key(|(&label, members)| {
        if label == -1 {
            (999, 0)
        } else {
            (label, -(members.len() as i32))
        }
    });

    for (&label, member_indices) in sorted_clusters.iter().take(15) {
        if label == -1 {
            println!(
                "  │  NOISE ({}) segments ({:.1}%)                              ",
                member_indices.len(),
                member_indices.len() as f64 / n_segments as f64 * 100.0
            );
        } else {
            let mut ctx_counts: HashMap<i32, usize> = HashMap::new();
            let mut emit_counts: HashMap<i32, usize> = HashMap::new();

            for &idx in member_indices.iter() {
                let ctx = all_segments[idx].context;
                let emit = all_segments[idx].emitter;
                *ctx_counts.entry(ctx).or_insert(0) += 1;
                *emit_counts.entry(emit).or_insert(0) += 1;
            }

            let unique_files: std::collections::HashSet<_> = member_indices
                .iter()
                .map(|&idx| &all_segments[idx].source_file)
                .collect();

            println!(
                "  │  CLUSTER {} ({} seg, {} files)                        ",
                label,
                member_indices.len(),
                unique_files.len()
            );

            let mut sorted_ctx: Vec<_> = ctx_counts.iter().collect();
            sorted_ctx.sort_by(|a, b| b.1.cmp(a.1));
            for (ctx, count) in sorted_ctx.iter().take(3) {
                let pct = *count as f64 / member_indices.len() as f64 * 100.0;
                print!("  │    Ctx{:2}: {:3} ({:.0}%) ", ctx, count, pct);
            }
            println!();
        }
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Context separation analysis
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  CONTEXT SEPARATION ANALYSIS                                             │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    for (ctx, _) in ctx_sorted.iter().take(8) {
        let ctx_segments: Vec<usize> = all_segments
            .iter()
            .enumerate()
            .filter(|(_, s)| s.context == **ctx)
            .map(|(i, _)| i)
            .collect();

        if ctx_segments.is_empty() {
            continue;
        }

        let mut cluster_dist: HashMap<i32, usize> = HashMap::new();
        for &idx in &ctx_segments {
            let label = labels[idx];
            *cluster_dist.entry(label).or_insert(0) += 1;
        }

        let dominant = cluster_dist
            .iter()
            .max_by_key(|(_, &c)| c)
            .map(|(&l, &c)| (l, c));

        if let Some((dom_cluster, dom_count)) = dominant {
            let pct = dom_count as f64 / ctx_segments.len() as f64 * 100.0;
            let separation = if pct > 80.0 {
                "STRONG"
            } else if pct > 50.0 {
                "MODERATE"
            } else {
                "WEAK"
            };

            let num_clusters = cluster_dist.len();
            let noise_in_ctx = cluster_dist.get(&-1).copied().unwrap_or(0);

            println!(
                "  │  Context {:2}: {} seg in {} clusters [{:>8}]        ",
                ctx,
                ctx_segments.len(),
                num_clusters,
                separation
            );
            println!(
                "  │    Dominant: {} ({:.0}%), Noise: {:.0}%",
                if dom_cluster == -1 {
                    "NOISE"
                } else {
                    &dom_cluster.to_string()
                },
                pct,
                noise_in_ctx as f64 / ctx_segments.len() as f64 * 100.0
            );
        }
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Final interpretation
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("INTERPRETATION");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  Expected for Egyptian Fruit Bats:");
    println!("    • Purity: 10-20% (LOW)");
    println!("    • Noise:  80-90% (HIGH)");
    println!("    • Interpretation: Prosodic modulation - unique FM sweeps");
    println!();

    println!("  Observed:");
    println!("    • Purity: {:.1}%", purity * 100.0);
    println!("    • Noise:  {:.1}%", noise_ratio * 100.0);
    println!();

    if purity < 0.25 {
        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!("  │  ✓ CONFIRMS HYPOTHESIS: LOW MOTIF REUSE                                 │");
        println!("  ├─────────────────────────────────────────────────────────────────────────┤");
        println!("  │                                                                         │");
        println!("  │  Egyptian fruit bat vocalizations are predominantly UNIQUE events.     │");
        println!("  │  FM sweeps show PROSODIC MODULATION - each call is a 'solo performance'.│");
        println!("  │                                                                         │");
        println!("  │  Bats modulate frequency sweeps continuously based on:                 │");
        println!("  │    • Emotional state                                                    │");
        println!("  │    • Social context                                                     │");
        println!("  │    • Individual identity                                                │");
        println!("  │                                                                         │");
        println!("  │  → Use Direct 105D similarity approach                                  │");
        println!("  │  → Bag-of-Phrases will FAIL (no reusable vocabulary)                   │");
        println!("  └─────────────────────────────────────────────────────────────────────────┘");
    } else if purity < 0.50 {
        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!("  │  ~ MODERATE MOTIF REUSE                                                  │");
        println!("  ├─────────────────────────────────────────────────────────────────────────┤");
        println!("  │  Some acoustic patterns are reused across contexts.                    │");
        println!("  │  Consider HYBRID approach with both discrete and continuous methods.   │");
        println!("  └─────────────────────────────────────────────────────────────────────────┘");
    } else {
        println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
        println!("  │  ⚠ UNEXPECTED: HIGH MOTIF REUSE                                         │");
        println!("  ├─────────────────────────────────────────────────────────────────────────┤");
        println!("  │  Higher than expected! Possible explanations:                          │");
        println!("  │    1. Dataset contains similar context calls grouped together          │");
        println!("  │    2. Clustering is grouping by energy patterns, not semantic content  │");
        println!("  │    3. Bats may reuse more acoustic patterns than expected              │");
        println!("  └─────────────────────────────────────────────────────────────────────────┘");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
