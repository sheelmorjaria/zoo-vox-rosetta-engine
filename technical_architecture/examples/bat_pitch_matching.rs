//! Egyptian Fruit Bat - Pitch Contour Matching
//! ==============================================
//!
//! "Search vs. Cluster" experiment to test segment reusability.

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
struct CachedSeg {
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

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     EGYPTIAN FRUIT BAT - PITCH CONTOUR MATCHING                           ║");
    println!("║     'Search vs. Cluster' - Testing Segment Reusability                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let cache_dir = Path::new("bat_nbd_cache_normalized");

    if !cache_dir.exists() {
        eprintln!("Error: Cache not found: {}", cache_dir.display());
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

    println!("  Found {} cache files", cache_files.len());

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

    let total_samples = all_segments.len();
    let max_samples = 50000.min(total_samples);
    let all_segments: Vec<_> = all_segments.into_iter().take(max_samples).collect();
    let n_samples = all_segments.len();

    println!("  Loaded {} segments (using {} for analysis)", total_samples, n_samples);
    println!();

    // Count contexts
    let context_counts: HashMap<i32, usize> = all_segments.iter().fold(HashMap::new(), |mut acc, seg| {
        *acc.entry(seg.context).or_insert(0) += 1;
        acc
    });

    println!("Context Distribution:");
    let mut sorted_contexts: Vec<_> = context_counts.iter().collect();
    sorted_contexts.sort_by(|a, b| b.1.cmp(a.1));
    for (context, count) in sorted_contexts.iter().take(8) {
        let pct = **count as f64 / n_samples as f64 * 100.0;
        println!("  • Context {:2}: {:6} ({:5.1}%)", context, count, pct);
    }
    println!();

    // ---------------------------------------------------------
    // STEP 2: EXTRACT PITCH SIGNATURES
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[2/4] Extracting Pitch Signatures from Feature Cache");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  Using Pitch Signature (6D):");
    println!("    • Feature 0: mean_f0_hz");
    println!("    • Feature 1: duration_ms");
    println!("    • Feature 2: f0_range_hz");
    println!("    • Feature 40-42: Modulation");
    println!();

    let pitch_indices: Vec<usize> = vec![0, 1, 2, 40, 41, 42];
    let n_pitch_features = pitch_indices.len();

    // Extract pitch signatures
    let mut pitch_signatures: Vec<Vec<f64>> = all_segments
        .iter()
        .map(|seg| {
            pitch_indices
                .iter()
                .map(|&idx| {
                    if idx < seg.features.len() {
                        seg.features[idx] as f64
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();

    // Z-normalize each feature
    for j in 0..n_pitch_features {
        let col: Vec<f64> = pitch_signatures.iter().map(|sig| sig[j]).collect();
        let mean = col.iter().sum::<f64>() / n_samples as f64;
        let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
        let std = var.sqrt().max(1e-8);

        for sig in pitch_signatures.iter_mut() {
            sig[j] = (sig[j] - mean) / std;
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

    let match_threshold = 1.5;
    let match_samples = 5000.min(n_samples);

    println!("  Match Threshold: {} (Euclidean distance in 6D)", match_threshold);
    println!("  Searching within {} samples...", match_samples);
    println!();

    let mut match_counts = vec![0usize; match_samples];

    for i in 0..match_samples {
        for j in 0..match_samples {
            if i == j {
                continue;
            }

            let dist: f64 = pitch_signatures[i]
                .iter()
                .zip(pitch_signatures[j].iter())
                .map(|(a, b)| (a - b).powi(2))
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
    let recurrence_rate = segments_with_matches as f64 / match_samples as f64;
    let avg_matches = total_matches as f64 / match_samples as f64;

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  PITCH SIGNATURE MATCHING RESULTS                                        │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Total segments searched:    {:>8}                                    ",
        match_samples
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
    );
    println!(
        "  │  Avg matches per segment:    {:>8.1}                                   ",
        avg_matches
    );
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // By context
    println!("  Recurrence Rate by Behavioral Context:");
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");

    let mut by_context: HashMap<i32, (usize, usize, usize)> = HashMap::new();

    for (i, seg) in all_segments.iter().take(match_samples).enumerate() {
        let ctx = seg.context;
        let (t, w, m) = by_context.get(&ctx).copied().unwrap_or((0, 0, 0));
        let has_match = if match_counts[i] > 0 { 1 } else { 0 };
        by_context.insert(ctx, (t + 1, w + has_match, m + match_counts[i]));
    }

    let mut sorted: Vec<_> = by_context.iter().collect();
    sorted.sort_by_key(|(_, (t, _, _))| std::cmp::Reverse(*t));

    for (context, (total, with_match, matches)) in sorted.iter().take(8) {
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
            "  │  Context {:2}:   {:5.1}% recurrence, {:.0} avg matches",
            context, rate, avg
        );
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
        no_match as f64 / match_samples as f64 * 100.0
    );
    println!(
        "    • Few matches (1-5):     {:5} ({:.1}%)",
        few_match,
        few_match as f64 / match_samples as f64 * 100.0
    );
    println!(
        "    • Some matches (6-20):   {:5} ({:.1}%)",
        some_match,
        some_match as f64 / match_samples as f64 * 100.0
    );
    println!(
        "    • Many matches (>20):    {:5} ({:.1}%)",
        many_match,
        many_match as f64 / match_samples as f64 * 100.0
    );
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
    } else if recurrence_rate > 0.5 {
        println!(
            "  │  ~ MODERATE RECURRENCE: {:.1}% of segments have matches                 ",
            recurrence_rate * 100.0
        );
        println!("  │                                                                          │");
        println!("  │  Some pitch signatures repeat, others are unique.                      │");
    } else {
        println!(
            "  │  ✗ LOW RECURRENCE: {:.1}% of segments have matches                      ",
            recurrence_rate * 100.0
        );
        println!("  │                                                                          │");
        println!("  │  Pitch signatures are UNIQUE.                                          │");
        println!("  │  → Confirms Prosodic Modulation: No reusable templates.                │");
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
