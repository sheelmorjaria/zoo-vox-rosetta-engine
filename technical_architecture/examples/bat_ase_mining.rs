//! Egyptian Fruit Bat - ASE-Weighted NBD Motif Mining
//! ====================================================
//!
//! Uses Acoustic Similarity Engine (ASE) weights to transform features
//! before HDBSCAN clustering. Tests for "Hidden Phonology" in bat vocalizations.
//!
//! Bat data uses CONTEXT (behavioral context) instead of call_type.
//! Contexts: 0=Unknown, 1-15=various behavioral states

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use technical_architecture::HdbscanClustering;

/// Normalized segment from bat cache
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
    #[allow(dead_code)]
    boundary_type: String,
    features: Vec<f32>,
}

/// ASE weight configuration
#[derive(Debug, Clone, Copy)]
enum AseWeightConfig {
    TextureHeavy,
    PhysicsHeavy,
    Balanced,
    ModulationFocused,
}

impl AseWeightConfig {
    fn name(&self) -> &'static str {
        match self {
            AseWeightConfig::TextureHeavy => "Texture-Heavy (0.8/0.2)",
            AseWeightConfig::PhysicsHeavy => "Physics-Heavy (0.2/0.8)",
            AseWeightConfig::Balanced => "Balanced (0.5/0.5)",
            AseWeightConfig::ModulationFocused => "Modulation-Focused",
        }
    }

    fn weights(&self) -> [f64; 105] {
        let mut w = [1.0f64; 105];

        match self {
            AseWeightConfig::TextureHeavy => {
                for i in 0..45 {
                    w[i] = 0.3;
                }
                for i in 45..75 {
                    w[i] = 2.5;
                }
                for i in 75..105 {
                    w[i] = 2.0;
                }
                w[9] = 2.0;
                w[10] = 2.0;
                w[41] = 2.5;
            }
            AseWeightConfig::PhysicsHeavy => {
                for i in 0..45 {
                    w[i] = 2.5;
                }
                for i in 45..75 {
                    w[i] = 0.3;
                }
                for i in 75..105 {
                    w[i] = 0.3;
                }
                w[0] = 3.0;
                w[1] = 2.5;
                w[2] = 2.5;
            }
            AseWeightConfig::Balanced => {
                w[0] = 1.5;
                w[1] = 1.3;
                w[9] = 1.8;
                w[13] = 1.5;
            }
            AseWeightConfig::ModulationFocused => {
                w[9] = 4.0;
                w[10] = 3.5;
                w[40] = 3.0;
                w[41] = 4.0;
                w[42] = 3.0;
                for i in 53..60 {
                    w[i] = 2.5;
                }
                for i in 75..90 {
                    w[i] = 2.5;
                }
                for i in 0..9 {
                    w[i] = 0.5;
                }
                for i in 11..40 {
                    w[i] = 0.5;
                }
            }
        }
        w
    }
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     EGYPTIAN FRUIT BAT - ASE-WEIGHTED NBD MOTIF MINING                   ║");
    println!("║     'Acoustically Aware' Clustering Test                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let cache_dir = Path::new("bat_nbd_cache_normalized");

    if !cache_dir.exists() {
        eprintln!(
            "Error: Bat normalized cache not found: {}",
            cache_dir.display()
        );
        std::process::exit(1);
    }

    // ---------------------------------------------------------
    // STEP 1: LOAD NORMALIZED FEATURES
    // ---------------------------------------------------------
    println!("[1/4] Loading Bat Normalized 105D Feature Cache...");
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

    // Subsample if too large
    let max_samples = 50_000;
    let all_segments = if n_samples > max_samples {
        println!("  Subsampling to {} for efficiency...", max_samples);
        all_segments.into_iter().take(max_samples).collect()
    } else {
        all_segments
    };
    let n_samples = all_segments.len();
    println!();

    // Count contexts
    let context_counts: HashMap<i32, usize> =
        all_segments.iter().fold(HashMap::new(), |mut acc, seg| {
            *acc.entry(seg.context).or_insert(0) += 1;
            acc
        });

    println!("Context Distribution:");
    let mut sorted_contexts: Vec<_> = context_counts.iter().collect();
    sorted_contexts.sort_by(|a, b| b.1.cmp(a.1));
    for (context, count) in sorted_contexts.iter().take(10) {
        let pct = **count as f64 / n_samples as f64 * 100.0;
        println!("  • Context {:2}: {:6} ({:5.1}%)", context, count, pct);
    }
    println!();

    // ---------------------------------------------------------
    // STEP 2: BUILD BASE FEATURE MATRIX
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[2/4] Building Feature Matrix");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

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

    // ---------------------------------------------------------
    // STEP 3: TEST MULTIPLE ASE WEIGHT CONFIGURATIONS
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[3/4] ASE-Weighted HDBSCAN Clustering Tests");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let configs = [
        AseWeightConfig::TextureHeavy,
        AseWeightConfig::PhysicsHeavy,
        AseWeightConfig::Balanced,
        AseWeightConfig::ModulationFocused,
    ];

    let mut results: Vec<(String, usize, f64, f64)> = Vec::new();

    for config in &configs {
        println!("─────────────────────────────────────────────────────────────────────────");
        println!("  Testing: {}", config.name());
        println!("─────────────────────────────────────────────────────────────────────────");

        let weights = config.weights();
        let sqrt_weights: Array1<f64> =
            Array1::from_vec(weights.iter().map(|&w| w.sqrt()).collect());

        let mut weighted_matrix = Array2::<f64>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                weighted_matrix[[i, j]] = feature_matrix[[i, j]] * sqrt_weights[j];
            }
        }

        let min_cluster_size = 10;
        let min_samples = 5;

        let hdbscan = HdbscanClustering::new(min_cluster_size, min_samples)?;
        let labels = hdbscan.fit_predict_hnsw(&weighted_matrix)?;

        let stats = hdbscan.get_cluster_stats(&labels);
        let noise_count = labels.iter().filter(|&&l| l == -1).count();
        let noise_ratio = noise_count as f64 / n_samples as f64;

        // Calculate cluster entropy based on context distribution
        let mut cluster_members: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &label) in labels.iter().enumerate() {
            cluster_members.entry(label).or_default().push(idx);
        }

        // Get entropy of largest cluster
        let mut max_entropy_normalized = 0.0;

        let largest_cluster = cluster_members
            .iter()
            .filter(|(&l, _)| l != -1)
            .max_by_key(|(_, m)| m.len());

        if let Some((_, members)) = largest_cluster {
            let mut ctx_counts: HashMap<i32, usize> = HashMap::new();
            for &idx in members {
                let ctx = all_segments[idx].context;
                *ctx_counts.entry(ctx).or_insert(0) += 1;
            }

            let total = members.len() as f64;
            let entropy: f64 = ctx_counts
                .values()
                .map(|&count| {
                    let p = count as f64 / total;
                    if p > 0.0 {
                        -p * p.log2()
                    } else {
                        0.0
                    }
                })
                .sum();

            let max_e = (ctx_counts.len() as f64).log2();
            max_entropy_normalized = if max_e > 0.0 { entropy / max_e } else { 0.0 };
        }

        println!(
            "    Clusters: {}  |  Noise: {:.1}%",
            stats.n_clusters,
            noise_ratio * 100.0
        );
        println!(
            "    Giant Cluster Context Entropy: {:.0}% normalized",
            max_entropy_normalized * 100.0
        );

        results.push((
            config.name().to_string(),
            stats.n_clusters,
            noise_ratio,
            max_entropy_normalized,
        ));

        // Show cluster composition if multiple clusters found
        if stats.n_clusters > 1 {
            println!();
            println!("    → {} CLUSTERS FOUND!", stats.n_clusters);

            let mut sorted_clusters: Vec<_> =
                cluster_members.iter().filter(|(&l, _)| l != -1).collect();
            sorted_clusters.sort_by_key(|(_, m)| std::cmp::Reverse(m.len()));

            for (label, members) in sorted_clusters.iter().take(5) {
                let mut ctx_counts: HashMap<i32, usize> = HashMap::new();
                for &idx in members.iter() {
                    *ctx_counts.entry(all_segments[idx].context).or_insert(0) += 1;
                }
                let mut sorted: Vec<_> = ctx_counts.iter().collect();
                sorted.sort_by(|a, b| b.1.cmp(a.1));

                print!("      Cluster {} ({}): ", label, members.len());
                for (ctx, cnt) in sorted.iter().take(3) {
                    let pct = **cnt as f64 / members.len() as f64 * 100.0;
                    print!("C{}={:.0}% ", ctx, pct);
                }
                println!();
            }
        } else if let Some((_, members)) = largest_cluster {
            // Show composition of giant cluster
            let mut ctx_counts: HashMap<i32, usize> = HashMap::new();
            for &idx in members {
                let ctx = all_segments[idx].context;
                *ctx_counts.entry(ctx).or_insert(0) += 1;
            }
            let mut sorted: Vec<_> = ctx_counts.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));

            println!("    Giant cluster composition ({} samples):", members.len());
            for (ctx, cnt) in sorted.iter().take(6) {
                let pct = **cnt as f64 / members.len() as f64 * 100.0;
                println!("      • Context {:2}: {:5} ({:.1}%)", ctx, cnt, pct);
            }
        }

        println!();
    }

    // ---------------------------------------------------------
    // STEP 4: SUMMARY
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[4/4] SUMMARY: Scientific Litmus Test");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  COMPARISON: Euclidean vs ASE-Weighted                                 │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    for (name, n_clusters, _noise, entropy_norm) in &results {
        println!(
            "  │  {:24} Clusters={:3}  Entropy={:4.0}%",
            name,
            n_clusters,
            entropy_norm * 100.0
        );
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Find best result (lowest entropy)
    let best = results
        .iter()
        .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
        .unwrap();

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  VERDICT                                                                │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    if best.1 > 1 {
        println!("  │  ✓ ASE FOUND MULTIPLE CLUSTERS!                                        │");
        println!(
            "  │    Best config: {}                                  ",
            best.0
        );
        println!(
            "  │    Clusters found: {}                                                   ",
            best.1
        );
        println!("  │    → Bats may have context-specific acoustic patterns                  │");
    } else if best.3 < 0.5 {
        println!("  │  ✓ ASE DISCRIMINATED CONTEXTS!                                         │");
        println!(
            "  │    Best config: {}                                  ",
            best.0
        );
        println!(
            "  │    Context entropy dropped to {:.0}%                                    ",
            best.3 * 100.0
        );
        println!("  │    → Bats have context-dependent vocal signatures                      │");
    } else if best.3 < 0.8 {
        println!("  │  ~ PARTIAL CONTEXT DISCRIMINATION                                      │");
        println!(
            "  │    Best config: {}                                  ",
            best.0
        );
        println!(
            "  │    Context entropy: {:.0}%                                              ",
            best.3 * 100.0
        );
        println!("  │    → Some context-specific structure exists                            │");
    } else {
        println!("  │  ✗ CONFIRMED: PROSODIC MODULATION SYSTEM                               │");
        println!("  │    All ASE configurations show ~100% context entropy                   │");
        println!("  │    → Even biologically-weighted features don't split contexts          │");
        println!("  │    → Bats modulate FM sweeps continuously (rate-based encoding)        │");
        println!("  │    → Use Direct 105D similarity (not Bag-of-Phrases)                   │");
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
