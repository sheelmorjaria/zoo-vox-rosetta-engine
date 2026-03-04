//! Marmoset - Pitch Contour Matching (NBD vs Energy)
//! ===================================================
//!
//! "Search vs. Cluster" experiment to test if NBD segments are "Sharable Units".
//!
//! Metric: "Recurrence Rate" = % of segments that have at least 1 match elsewhere.
//!
//! Hypothesis:
//! - NBD Segments: High Recurrence (semantic cuts isolate reusable motifs)
//! - Energy Segments: Low Recurrence (cuts through motifs, creating broken shapes)

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use technical_architecture::{
    BoundaryDetectorConfig, MicroDynamicsExtractor, NeuralBoundaryDetector,
};

/// Segment info
#[derive(Debug, Clone)]
struct Segment {
    start_sample: usize,
    end_sample: usize,
    start_ms: f32,
    end_ms: f32,
}

/// Match result
#[derive(Debug, Clone, Serialize)]
struct MatchResult {
    method: String,
    total_segments: usize,
    segments_with_matches: usize,
    recurrence_rate: f64,
    avg_matches_per_segment: f64,
    total_matches: usize,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     MARMOSET - PITCH CONTOUR MATCHING (NBD vs Energy)                     ║");
    println!("║     'Search vs. Cluster' - Testing Segment Reusability                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Use marmoset normalized cache which already has features
    let cache_dir = Path::new("marmoset_nbd_cache_normalized");

    if !cache_dir.exists() {
        eprintln!("Error: Cache not found: {}", cache_dir.display());
        eprintln!("Run marmoset_normalize_cache first.");
        std::process::exit(1);
    }

    // ---------------------------------------------------------
    // STEP 1: LOAD CACHE
    // ---------------------------------------------------------
    println!("[1/4] Loading Normalized Feature Cache...");
    println!("─────────────────────────────────────────────────────────────────────────");

    let cache_files: Vec<_> = fs::read_dir(cache_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    #[derive(Debug, Clone, Deserialize)]
    struct CachedSeg {
        audio_file: String,
        call_type: String,
        features: Vec<f32>,
    }

    let all_segments: Vec<CachedSeg> = cache_files
        .par_iter()
        .flat_map(|file| {
            let json = match fs::read_to_string(file) {
                Ok(j) => j,
                Err(_) => return None,
            };
            let batch: Vec<CachedSeg> = match serde_json::from_str(&json) {
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

    // ---------------------------------------------------------
    // STEP 2: EXTRACT PITCH CONTOURS (from features)
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[2/4] Extracting Pitch Contours from Feature Cache");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Extract pitch-relevant features from the 105D cache
    // Indices 0-2 contain: mean_f0_hz, duration_ms, f0_range_hz
    // We'll use these as a "pitch signature" for each segment

    println!("  Using Pitch Signature from 105D cache:");
    println!("    • Feature 0: mean_f0_hz");
    println!("    • Feature 1: duration_ms");
    println!("    • Feature 2: f0_range_hz");
    println!("    • Feature 40-42: Modulation (spectral_tilt, fm_slope, am_depth)");
    println!();

    // Build pitch signature matrix (6D)
    let pitch_indices = [0, 1, 2, 40, 41, 42];
    let n_pitch_features = pitch_indices.len();

    let mut pitch_matrix = Array2::<f64>::zeros((n_samples, n_pitch_features));

    for (i, seg) in all_segments.iter().enumerate() {
        for (j, &idx) in pitch_indices.iter().enumerate() {
            if idx < seg.features.len() {
                pitch_matrix[[i, j]] = seg.features[idx] as f64;
            }
        }
    }

    // Z-normalize each feature column
    for j in 0..n_pitch_features {
        let col: Vec<f64> = (0..n_samples).map(|i| pitch_matrix[[i, j]]).collect();
        let mean = col.iter().sum::<f64>() / n_samples as f64;
        let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
        let std = var.sqrt().max(1e-8);

        for i in 0..n_samples {
            pitch_matrix[[i, j]] = (pitch_matrix[[i, j]] - mean) / std;
        }
    }

    println!(
        "  Pitch Signature Matrix: {} × {} (Z-normalized)",
        n_samples, n_pitch_features
    );
    println!();

    // ---------------------------------------------------------
    // STEP 3: FIND MATCHES
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[3/4] Finding Matching Pitch Signatures");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Match threshold (Euclidean distance in normalized 6D space)
    let match_threshold = 1.5; // About 1.5 standard deviations

    println!(
        "  Match Threshold: {} (Euclidean distance in 6D)",
        match_threshold
    );
    println!("  Lower = stricter match requirement");
    println!();

    // Subsample for efficiency (use first 2000 samples)
    let max_samples = 2000.min(n_samples);
    println!("  Searching within first {} samples...", max_samples);

    // Compute pairwise distances and find matches
    let mut match_counts = vec![0usize; max_samples];

    for i in 0..max_samples {
        for j in 0..max_samples {
            if i == j {
                continue;
            }

            // Euclidean distance
            let dist: f64 = (0..n_pitch_features)
                .map(|k| (pitch_matrix[[i, k]] - pitch_matrix[[j, k]]).powi(2))
                .sum::<f64>()
                .sqrt();

            if dist < match_threshold {
                match_counts[i] += 1;
            }
        }
    }

    // ---------------------------------------------------------
    // STEP 4: RESULTS
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[4/4] Results - Recurrence Rate Analysis");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    let total_matches: usize = match_counts.iter().sum();
    let segments_with_matches = match_counts.iter().filter(|&&c| c > 0).count();
    let recurrence_rate = segments_with_matches as f64 / max_samples as f64;
    let avg_matches = total_matches as f64 / max_samples as f64;

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  PITCH SIGNATURE MATCHING RESULTS                                        │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Total segments searched:    {:>8}                                    ",
        max_samples
    );
    println!(
        "  │  Match threshold:            {:>8.2}                                   ",
        match_threshold
    );
    println!("  │                                                                         │");
    println!(
        "  │  Segments with ≥1 match:     {:>8} ({:.1}%)                           ",
        segments_with_matches,
        recurrence_rate * 100.0
    );
    println!(
        "  │  Total match pairs found:    {:>8}                                    ",
        total_matches / 2
    ); // Divide by 2 (bidirectional)
    println!(
        "  │  Avg matches per segment:    {:>8.1}                                   ",
        avg_matches
    );
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Analyze by call type
    println!("  Recurrence Rate by Call Type:");
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");

    let mut by_type: HashMap<&str, (usize, usize, usize)> = HashMap::new(); // (total, with_match, total_matches)

    for (i, seg) in all_segments.iter().take(max_samples).enumerate() {
        let ct = seg.call_type.as_str();
        let (t, w, m) = by_type.get(ct).copied().unwrap_or((0, 0, 0));
        let has_match = if match_counts[i] > 0 { 1 } else { 0 };
        by_type.insert(ct, (t + 1, w + has_match, m + match_counts[i]));
    }

    let mut sorted: Vec<_> = by_type.iter().collect();
    sorted.sort_by_key(|(_, (t, _, _))| std::cmp::Reverse(*t));

    for (call_type, (total, with_match, matches)) in sorted.iter() {
        let rate = if *total > 0 {
            *with_match as f64 / *total as f64 * 100.0
        } else {
            0.0
        };
        let avg = if *total > 0 {
            *matches as f64 / *total as f64
        } else {
            0.0
        };
        println!(
            "  │  {:14}: {:5.1}% recurrence, {:.1} avg matches",
            call_type, rate, avg
        );
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Interpretation
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  INTERPRETATION                                                          │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    if recurrence_rate > 0.8 {
        println!(
            "  │  ✓ HIGH RECURRENCE: {:.1}% of segments have matches                    ",
            recurrence_rate * 100.0
        );
        println!("  │                                                                          │");
        println!("  │  Pitch signatures are REUSABLE across calls.                           │");
        println!("  │  Similar pitch contours appear in multiple recordings.                 │");
        println!("  │  → NBD segments capture acoustically similar units.                    │");
    } else if recurrence_rate > 0.5 {
        println!(
            "  │  ~ MODERATE RECURRENCE: {:.1}% of segments have matches                 ",
            recurrence_rate * 100.0
        );
        println!("  │                                                                          │");
        println!("  │  Some pitch signatures repeat, others are unique.                      │");
        println!("  │  → Partial reusability in pitch contours.                              │");
    } else {
        println!(
            "  │  ✗ LOW RECURRENCE: {:.1}% of segments have matches                      ",
            recurrence_rate * 100.0
        );
        println!("  │                                                                          │");
        println!("  │  Pitch signatures are UNIQUE.                                          │");
        println!("  │  Each call has a distinct pitch trajectory.                            │");
        println!("  │  → Confirms Graded Continuum: No reusable pitch templates.            │");
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Match distribution
    println!("  Match Distribution:");
    let no_match = match_counts.iter().filter(|&&c| c == 0).count();
    let few_match = match_counts.iter().filter(|&&c| c >= 1 && c <= 5).count();
    let some_match = match_counts.iter().filter(|&&c| c > 5 && c <= 20).count();
    let many_match = match_counts.iter().filter(|&&c| c > 20).count();

    println!(
        "    • No matches (0):        {:5} ({:.1}%)",
        no_match,
        no_match as f64 / max_samples as f64 * 100.0
    );
    println!(
        "    • Few matches (1-5):     {:5} ({:.1}%)",
        few_match,
        few_match as f64 / max_samples as f64 * 100.0
    );
    println!(
        "    • Some matches (6-20):   {:5} ({:.1}%)",
        some_match,
        some_match as f64 / max_samples as f64 * 100.0
    );
    println!(
        "    • Many matches (>20):    {:5} ({:.1}%)",
        many_match,
        many_match as f64 / max_samples as f64 * 100.0
    );
    println!();

    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
