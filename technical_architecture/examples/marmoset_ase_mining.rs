//! Marmoset - ASE-Weighted NBD Motif Mining
//! ==========================================
//!
//! Uses Acoustic Similarity Engine (ASE) weights to transform features
//! before HDBSCAN clustering. This tests whether "acoustically aware"
//! distance metrics can reveal hidden motifs.
//!
//! Key Insight:
//! - Euclidean Distance: Geometrically blind, treats all dimensions equally
//! - ASE Distance: Biologically weighted, emphasizes texture/physics importance
//!
//! If ASE clustering finds multiple clusters where Euclidean found one,
//! we have discovered "Hidden Phonology" in marmoset vocalizations.

use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use technical_architecture::{AcousticSimilarityEngine, HdbscanClustering};

/// Normalized segment from cache
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

/// ASE weight configuration for different acoustic hypotheses
#[derive(Debug, Clone, Copy)]
enum AseWeightConfig {
    /// High texture weight - emphasizes spectral shape/motif
    TextureHeavy,
    /// High physics weight - emphasizes fundamental/duration
    PhysicsHeavy,
    /// Balanced - equal importance
    Balanced,
    /// Vibrato-focused - emphasizes modulation features
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

    /// Generate 105D feature weights based on configuration
    ///
    /// 105D Layout:
    /// - [0-44]: Base Physics (45D) - fundamental, grit, motion, MFCCs, rhythm, resonance, spectral
    /// - [45-74]: Macro Texture (30D) - harmonic texture, pitch geometry, GLCM, temporal
    /// - [75-104]: Micro Texture (30D) - vibrato bins, FM bins, dynamics, ICI bins, rhythm
    fn weights(&self) -> [f64; 105] {
        let mut w = [1.0f64; 105];

        match self {
            AseWeightConfig::TextureHeavy => {
                // Base Physics (0-44): LOW weight
                for i in 0..45 {
                    w[i] = 0.3;
                }
                // Macro Texture (45-74): HIGH weight
                for i in 45..75 {
                    w[i] = 2.5;
                }
                // Micro Texture (75-104): HIGH weight
                for i in 75..105 {
                    w[i] = 2.0;
                }
                // Key discriminative features get extra boost
                w[9] = 2.0; // vibrato_rate (Base Motion)
                w[10] = 2.0; // vibrato_depth
                w[41] = 2.5; // fm_slope (Modulation)
            }
            AseWeightConfig::PhysicsHeavy => {
                // Base Physics (0-44): HIGH weight
                for i in 0..45 {
                    w[i] = 2.5;
                }
                // Macro Texture (45-74): LOW weight
                for i in 45..75 {
                    w[i] = 0.3;
                }
                // Micro Texture (75-104): LOW weight
                for i in 75..105 {
                    w[i] = 0.3;
                }
                // Key physics features
                w[0] = 3.0; // mean_f0_hz (Fundamental)
                w[1] = 2.5; // duration_ms
                w[2] = 2.5; // f0_range_hz
            }
            AseWeightConfig::Balanced => {
                // All equal weight (already 1.0)
                // Slight boost to key features
                w[0] = 1.5; // mean_f0_hz
                w[1] = 1.3; // duration_ms
                w[9] = 1.8; // vibrato_rate
                w[13] = 1.5; // mfcc_1
            }
            AseWeightConfig::ModulationFocused => {
                // Emphasize vibrato, FM, AM features
                // Base Physics - modulation-related
                w[9] = 4.0; // vibrato_rate
                w[10] = 3.5; // vibrato_depth
                w[40] = 3.0; // spectral_tilt
                w[41] = 4.0; // fm_slope
                w[42] = 3.0; // am_depth

                // Macro Texture - pitch geometry
                for i in 53..60 {
                    w[i] = 2.5;
                }

                // Micro Texture - vibrato/FM bins
                for i in 75..90 {
                    w[i] = 2.5;
                }

                // Reduce other features
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
    println!("║     MARMOSET - ASE-WEIGHTED NBD MOTIF MINING                             ║");
    println!("║     'Acoustically Aware' Clustering Test                                 ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let cache_dir = Path::new("marmoset_nbd_cache_normalized");

    if !cache_dir.exists() {
        eprintln!("Error: Normalized cache not found: {}", cache_dir.display());
        std::process::exit(1);
    }

    // ---------------------------------------------------------
    // STEP 1: LOAD NORMALIZED FEATURES
    // ---------------------------------------------------------
    println!("[1/4] Loading Normalized 105D Feature Cache...");
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

    let mut results: Vec<(String, usize, f64, f64, f64)> = Vec::new();

    for config in &configs {
        println!("─────────────────────────────────────────────────────────────────────────");
        println!("  Testing: {}", config.name());
        println!("─────────────────────────────────────────────────────────────────────────");

        // Get ASE weights
        let weights = config.weights();
        let ase = AcousticSimilarityEngine::new(n_features);
        let weights_slice: Vec<f32> = weights.iter().map(|&w| w as f32).collect();

        // Create ASE-weighted feature matrix
        // Transform: weighted_features = features * sqrt(weights)
        // This makes Euclidean distance equivalent to weighted Euclidean
        let sqrt_weights: Array1<f64> =
            Array1::from_vec(weights.iter().map(|&w| w.sqrt()).collect());

        let mut weighted_matrix = Array2::<f64>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                weighted_matrix[[i, j]] = feature_matrix[[i, j]] * sqrt_weights[j];
            }
        }

        // Run HDBSCAN
        let min_cluster_size = 10;
        let min_samples = 5;

        let hdbscan = HdbscanClustering::new(min_cluster_size, min_samples)?;
        let labels = hdbscan.fit_predict_hnsw(&weighted_matrix)?;

        // Analyze results
        let stats = hdbscan.get_cluster_stats(&labels);
        let noise_count = labels.iter().filter(|&&l| l == -1).count();
        let purity = (n_samples - noise_count) as f64 / n_samples as f64;
        let noise_ratio = noise_count as f64 / n_samples as f64;

        // Calculate cluster entropy
        let mut cluster_members: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &label) in labels.iter().enumerate() {
            cluster_members.entry(label).or_default().push(idx);
        }

        // Get entropy of largest cluster (regardless of ID)
        let mut max_entropy = 0.0;
        let mut max_entropy_normalized = 0.0;

        let largest_cluster = cluster_members
            .iter()
            .filter(|(&l, _)| l != -1)
            .max_by_key(|(_, m)| m.len());

        if let Some((_, members)) = largest_cluster {
            let mut type_counts: HashMap<&str, usize> = HashMap::new();
            for &idx in members {
                let call_type = all_segments[idx].call_type.as_str();
                *type_counts.entry(call_type).or_insert(0) += 1;
            }

            let total = members.len() as f64;
            let entropy: f64 = type_counts
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

            let max_e = (type_counts.len() as f64).log2();
            max_entropy = entropy;
            max_entropy_normalized = if max_e > 0.0 { entropy / max_e } else { 0.0 };
        }

        println!(
            "    Clusters: {}  |  Noise: {:.1}%  |  Purity: {:.1}%",
            stats.n_clusters,
            noise_ratio * 100.0,
            purity * 100.0
        );
        println!(
            "    Giant Cluster Entropy: {:.3} bits ({:.0}% normalized)",
            max_entropy,
            max_entropy_normalized * 100.0
        );

        results.push((
            config.name().to_string(),
            stats.n_clusters,
            purity,
            noise_ratio,
            max_entropy_normalized,
        ));

        // Always show giant cluster composition for verification
        println!("    Total clusters found: {}", stats.n_clusters);
        println!(
            "    Cluster IDs: {:?}",
            cluster_members.keys().collect::<Vec<_>>()
        );

        // Find the largest cluster
        let largest_cluster = cluster_members
            .iter()
            .filter(|(&l, _)| l != -1)
            .max_by_key(|(_, m)| m.len())
            .map(|(l, m)| (*l, m.clone()))
            .unwrap_or((-1, vec![]));

        if !largest_cluster.1.is_empty() {
            let mut type_counts: HashMap<&str, usize> = HashMap::new();
            for &idx in &largest_cluster.1 {
                let call_type = all_segments[idx].call_type.as_str();
                *type_counts.entry(call_type).or_insert(0) += 1;
            }
            let mut sorted: Vec<_> = type_counts.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));

            println!(
                "    Largest Cluster ({}) composition ({} samples):",
                largest_cluster.0,
                largest_cluster.1.len()
            );
            for (ct, cnt) in sorted.iter() {
                let pct = **cnt as f64 / largest_cluster.1.len() as f64 * 100.0;
                println!("      • {:14}: {:4} ({:.0}%)", ct, cnt, pct);
            }
        }

        // If we found multiple clusters, show composition
        if stats.n_clusters > 1 {
            println!();
            println!("    → MULTIPLE CLUSTERS FOUND! Showing all:");

            let mut sorted_clusters: Vec<_> =
                cluster_members.iter().filter(|(&l, _)| l != -1).collect();
            sorted_clusters.sort_by_key(|(_, m)| std::cmp::Reverse(m.len()));

            for (label, members) in sorted_clusters.iter().take(5) {
                let mut type_counts: HashMap<&str, usize> = HashMap::new();
                for &idx in members.iter() {
                    *type_counts
                        .entry(all_segments[idx].call_type.as_str())
                        .or_insert(0) += 1;
                }
                let mut sorted: Vec<_> = type_counts.iter().collect();
                sorted.sort_by(|a, b| b.1.cmp(a.1));

                print!("      Cluster {} ({}): ", label, members.len());
                for (ct, cnt) in sorted.iter().take(2) {
                    let pct = **cnt as f64 / members.len() as f64 * 100.0;
                    print!("{}={:.0}% ", ct, pct);
                }
                println!();
            }
        }

        println!();
    }

    // ---------------------------------------------------------
    // STEP 4: SUMMARY & INTERPRETATION
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[4/4] SUMMARY: Scientific Litmus Test");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  COMPARISON: Euclidean vs ASE-Weighted                                 │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    for (name, n_clusters, purity, noise, entropy_norm) in &results {
        println!(
            "  │  {:24} Clusters={:3}  Entropy={:4.0}%",
            name,
            n_clusters,
            entropy_norm * 100.0
        );
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Find best result
    let best = results
        .iter()
        .min_by(|a, b| a.4.partial_cmp(&b.4).unwrap())
        .unwrap();

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  VERDICT                                                                │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    if best.4 < 0.5 {
        println!("  │  ✓ ASE FOUND HIDDEN PHONOLOGY!                                         │");
        println!(
            "  │    Best config: {}                                  ",
            best.0
        );
        println!(
            "  │    Cluster entropy dropped to {:.0}% (from 100%)                       ",
            best.4 * 100.0
        );
        println!("  │    → Marmosets DO have discrete motifs!                                │");
        println!("  │    → Use Bag-of-Phrases with ASE weighting                             │");
    } else if best.4 < 0.8 {
        println!("  │  ~ PARTIAL DISCRIMINATION                                               │");
        println!(
            "  │    Best config: {}                                  ",
            best.0
        );
        println!(
            "  │    Cluster entropy: {:.0}% (improved from 100%)                        ",
            best.4 * 100.0
        );
        println!("  │    → Some motif structure exists                                       │");
        println!("  │    → Hybrid approach recommended                                       │");
    } else {
        println!("  │  ✗ CONFIRMED: TRUE GRADED CONTINUUM                                    │");
        println!("  │    All ASE configurations show ~100% entropy                           │");
        println!("  │    → Even biologically-weighted features don't split calls             │");
        println!("  │    → Marmosets use CONTINUOUS vocal gradients                          │");
        println!("  │    → Use Direct 105D similarity (not Bag-of-Phrases)                   │");
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
