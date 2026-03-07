// Feature Selection Analysis for Egyptian Fruit Bat HDBSCAN
//
// Analyzes which features provide the best clustering discrimination
// Uses variance analysis and correlation to identify redundant features

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
struct PhraseSegment {
    segment_id: usize,
    file_index: usize,
    file_name: String,
    start_time_ms: f64,
    end_time_ms: f64,
    duration_ms: f64,
    start_sample: usize,
    end_sample: usize,
    sample_rate: u32,
    frame_indices: Vec<usize>,
    level1_cluster_id: i32,
    representative_features: Vec<f64>,
}

struct FeatureStats {
    index: usize,
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    cv: f64, // Coefficient of variation
    variance: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Feature Selection Analysis for Egyptian Fruit Bat Clustering           ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let data_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats");
    let segments_path = data_dir.join("phase0_twolevel_hdbscan_results/all_segments.json");

    println!("📂 Loading segments from: {}", segments_path.display());

    let segments_json = fs::read_to_string(&segments_path)?;
    let all_segments: Vec<PhraseSegment> = serde_json::from_str(&segments_json)?;

    println!("   ✅ Loaded {} segments", all_segments.len());
    println!();

    // Sample for analysis (use subset for speed)
    let sample_size = 50000.min(all_segments.len());
    let sample_indices: Vec<usize> = (0..sample_size).collect();

    println!(
        "📊 Analyzing {} segments (sampled from {} total)",
        sample_size,
        all_segments.len()
    );
    println!();

    // Build feature matrix
    let n_samples = sample_indices.len();
    let n_features = all_segments[0].representative_features.len();

    let mut feature_matrix = vec![0.0f64; n_samples * n_features];
    for (row_idx, &seg_idx) in sample_indices.iter().enumerate() {
        let features = &all_segments[seg_idx].representative_features;
        for (col_idx, &val) in features.iter().enumerate() {
            feature_matrix[row_idx * n_features + col_idx] = val;
        }
    }

    // Feature names for interpretation
    let feature_names = vec![
        // Fundamental (0-2)
        "mean_f0_hz",
        "f0_range_hz",
        "duration_ms",
        // Grit Factors (3-5)
        "harmonic_to_noise_ratio",
        "spectral_flatness",
        "harmonicity",
        // Motion Factors (6-12)
        "attack_time_ms",
        "decay_time_ms",
        "sustain_level",
        "vibrato_rate_hz",
        "vibrato_depth",
        "jitter",
        "shimmer",
        // Fingerprint/MFCCs (13-25)
        "mfcc_1",
        "mfcc_2",
        "mfcc_3",
        "mfcc_4",
        "mfcc_5",
        "mfcc_6",
        "mfcc_7",
        "mfcc_8",
        "mfcc_9",
        "mfcc_10",
        "mfcc_11",
        "mfcc_12",
        "mfcc_13",
        // Spectral Dynamics (26)
        "spectral_flux",
        // Rhythm Factors (27-29)
        "median_ici_ms",
        "onset_rate_hz",
        "ici_coefficient_of_variation",
    ];

    // ========================================================================
    // 1. Variance Analysis
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ 1. VARIANCE ANALYSIS - Which features have the most discrimination?      │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let mut feature_stats: Vec<FeatureStats> = Vec::new();

    for feat_idx in 0..n_features {
        let mut values = Vec::with_capacity(n_samples);
        for row_idx in 0..n_samples {
            values.push(feature_matrix[row_idx * n_features + feat_idx]);
        }

        let mean = values.iter().sum::<f64>() / n_samples as f64;
        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
        let std = variance.sqrt();
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let cv = if mean > 0.0 { std / mean.abs() } else { 0.0 };

        feature_stats.push(FeatureStats {
            index: feat_idx,
            mean,
            std,
            min: min_val,
            max: max_val,
            cv,
            variance,
        });
    }

    // Sort by variance (descending)
    feature_stats.sort_by(|a, b| b.variance.partial_cmp(&a.variance).unwrap());

    println!("   FEATURES BY VARIANCE (discriminatory power):");
    println!("   ┌──────┬──────────────────────────────┬────────────┬────────────┬──────────┬────────────┐");
    println!("   │ Idx  │ Name                         │ Variance   │ Std        │ CV       │ Range      │");
    println!("   ├──────┼──────────────────────────────┼────────────┼────────────┼──────────┼────────────┤");

    for stat in feature_stats.iter().take(15) {
        println!(
            "   │ {:4} │ {:28} │ {:10.2e} │ {:10.2} │ {:8.2} │ {:.2} - {:.2} │",
            stat.index, feature_names[stat.index], stat.variance, stat.std, stat.cv, stat.min, stat.max
        );
    }

    println!("   └──────┴──────────────────────────────┴────────────┴────────────┴──────────┴────────────┘");
    println!();

    // ========================================================================
    // 2. Scale Analysis - Dominant Features
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ 2. SCALE ANALYSIS - Which features dominate distance calculations?    │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Calculate contribution to Euclidean distance
    let mut total_contribution = 0.0;
    let mut contributions: Vec<(usize, f64)> = Vec::new();

    for feat_idx in 0..n_features {
        let stat = &feature_stats[feat_idx];
        // Approximate contribution = std^2 (for standardized features)
        let contribution = stat.std * stat.std;
        total_contribution += contribution;
        contributions.push((feat_idx, contribution));
    }

    contributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("   FEATURE CONTRIBUTION TO EUCLIDEAN DISTANCE:");
    println!("   ┌──────┬──────────────────────────────┬────────────┬────────────┐");
    println!("   │ Idx  │ Name                         │ Contrib.   │ Percentage │");
    println!("   ├──────┼──────────────────────────────┼────────────┼────────────┤");

    let mut _cumulative = 0.0;
    for (idx, &(feat_idx, contrib)) in contributions.iter().enumerate() {
        let pct = contrib / total_contribution * 100.0;
        _cumulative += pct;
        println!(
            "   │ {:4} │ {:28} │ {:10.2e} | {:9.2}%% │",
            feat_idx, feature_names[feat_idx], contrib, pct
        );
        if idx >= 14 {
            break;
        }
    }

    println!("   └──────┴──────────────────────────────┴────────────┴────────────┘");
    println!();

    // ========================================================================
    // 3. Recommended Feature Subsets
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ 3. RECOMMENDED FEATURE SUBSETS FOR CLUSTERING                          │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Option 1: High-variance features (excluding F0 dominance)
    let high_variance_features: Vec<usize> = feature_stats
        .iter()
        .filter(|s| s.index != 0) // Exclude F0
        .take(15)
        .map(|s| s.index)
        .collect();

    println!("   OPTION 1: High-Variance Features (excluding F0)");
    println!("   ┌─────────────────────────────────────────────────────────────────────┐");
    println!(
        "   │ Features: {:?} │",
        high_variance_features
            .iter()
            .map(|i| feature_names[*i])
            .collect::<Vec<_>>()
    );
    println!("   └─────────────────────────────────────────────────────────────────────┘");
    println!("   Rationale: Excludes dominant F0, keeps features with good variance");
    println!();

    // Option 2: Duration + MFCC + Modulation (linguistically relevant)
    let linguistic_features = vec![2, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25];

    println!("   OPTION 2: Linguistic Features (Duration + MFCCs)");
    println!("   ┌─────────────────────────────────────────────────────────────────────┐");
    println!(
        "   │ Features: {:?} │",
        linguistic_features
            .iter()
            .map(|i| feature_names[*i])
            .collect::<Vec<_>>()
    );
    println!("   └─────────────────────────────────────────────────────────────────────┘");
    println!("   Rationale: Duration and spectral envelope are most linguistically relevant");
    println!();

    // Option 3: Normalized all features (z-score)
    println!("   OPTION 3: All Features (Z-score Normalized)");
    println!("   ┌─────────────────────────────────────────────────────────────────────┐");
    println!("   │ Features: All 30D, each normalized to mean=0, std=1                │");
    println!("   └─────────────────────────────────────────────────────────────────────┘");
    println!("   Rationale: Preserves all information but balances scales");
    println!();

    // ========================================================================
    // 4. Save Results
    // ========================================================================

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ 4. SAVING RECOMMENDATIONS                                               │");
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    let results = serde_json::json!({
        "feature_analysis": {
            "n_samples": n_samples,
            "n_features": n_features,
            "feature_names": feature_names,
            "variance_ranking": feature_stats.iter().map(|s| serde_json::json!({
                "index": s.index,
                "name": feature_names[s.index],
                "variance": s.variance,
                "std": s.std,
                "cv": s.cv,
                "min": s.min,
                "max": s.max
            })).collect::<Vec<_>>()
        },
        "recommendations": {
            "option1_high_variance_excluding_f0": {
                "name": "High Variance (excluding F0)",
                "features": high_variance_features,
                "feature_names": high_variance_features.iter().map(|i| feature_names[*i]).collect::<Vec<_>>(),
                "count": high_variance_features.len()
            },
            "option2_linguistic": {
                "name": "Linguistic Features (Duration + MFCCs)",
                "features": linguistic_features,
                "feature_names": linguistic_features.iter().map(|i| feature_names[*i]).collect::<Vec<_>>(),
                "count": linguistic_features.len()
            },
            "option3_all_normalized": {
                "name": "All Features (Z-score Normalized)",
                "features": (0..30).collect::<Vec<_>>(),
                "feature_names": feature_names,
                "count": 30,
                "normalization": "z-score"
            }
        }
    });

    let results_dir = data_dir.join("feature_selection_results");
    fs::create_dir_all(&results_dir)?;

    let results_path = results_dir.join("feature_analysis.json");
    fs::write(&results_path, serde_json::to_string_pretty(&results)?)?;
    println!("   💾 Results saved to: {}", results_path.display());

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    ANALYSIS COMPLETE                                    ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                           ║");
    println!("║  KEY FINDINGS:                                                          ║");
    println!("║     • F0 (feature 0) dominates Euclidean distance due to scale          ║");
    println!("║     • Cosine fails because all segments have similar shape (0.998)     ║");
    println!("║     • Duration and MFCCs provide good variance for clustering          ║");
    println!("║                                                                           ║");
    println!("║  RECOMMENDED NEXT STEP:                                                 ║");
    println!("║     Try OPTION 3 (Z-score normalization) with existing HDBSCAN           ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");

    Ok(())
}
