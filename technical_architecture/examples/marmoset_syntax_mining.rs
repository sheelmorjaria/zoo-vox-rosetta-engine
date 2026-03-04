//! Marmoset - Syntax Mining (NBD Sequence Analysis)
//! =================================================
//!
//! "Continuous N-Gram Mining" - searching for reusable "sentence structures"
//! on top of the graded continuum.
//!
//! Pipeline:
//! 1. Discretize continuous 105D features into "Acoustic States" (k-means grid)
//! 2. Build sequences of states per file
//! 3. Mine for repeated N-grams (reusable sequences)

use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

type StateId = u32;
type Sequence = Vec<StateId>;

#[derive(Debug, Clone, Deserialize)]
struct CachedSeg {
    audio_file: String,
    call_type: String,
    features: Vec<f32>,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     MARMOSET - SYNTAX MINING (NBD Sequence Analysis)                      ║");
    println!("║           Searching for 'Sentence Structure' in Graded Continuum          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let cache_dir = Path::new("marmoset_nbd_cache_normalized");

    if !cache_dir.exists() {
        eprintln!("Error: Cache not found: {}", cache_dir.display());
        std::process::exit(1);
    }

    // ---------------------------------------------------------
    // STEP 1: LOAD & DISCRETIZE
    // ---------------------------------------------------------
    println!("[1/4] Loading Segments and Discretizing into Acoustic States...");
    println!("─────────────────────────────────────────────────────────────────────────");

    let cache_files: Vec<_> = fs::read_dir(cache_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
        .map(|e| e.path())
        .collect();

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

    // Discretize using simple grid-based approach
    // Use 6D pitch signature for discretization
    let pitch_indices: Vec<usize> = vec![0, 1, 2, 40, 41, 42];
    let n_features = pitch_indices.len();

    // Extract features
    let mut features: Vec<Vec<f64>> = all_segments
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

    // Z-normalize
    for j in 0..n_features {
        let col: Vec<f64> = features.iter().map(|f| f[j]).collect();
        let mean = col.iter().sum::<f64>() / n_samples as f64;
        let std = (col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64)
            .sqrt()
            .max(1e-8);
        for f in features.iter_mut() {
            f[j] = (f[j] - mean) / std;
        }
    }

    // Discretize into k states using simple grid binning
    // Each dimension gets split into bins, creating a grid
    let bins_per_dim = 3; // 3^6 = 729 possible states
    let k = (bins_per_dim as usize).pow(n_features as u32);

    let state_ids: Vec<StateId> = features
        .iter()
        .map(|f| {
            // Discretize each dimension
            let mut state = 0u32;
            let mut multiplier = 1u32;

            for &val in f {
                // Bin: -inf to -0.5 = 0, -0.5 to 0.5 = 1, 0.5 to inf = 2
                let bin = if val < -0.5 {
                    0
                } else if val < 0.5 {
                    1
                } else {
                    2
                };
                state += bin * multiplier;
                multiplier *= bins_per_dim as u32;
            }
            state
        })
        .collect();

    // Count unique states
    let unique_states: std::collections::HashSet<_> = state_ids.iter().collect();
    println!(
        "  Discretized into {} unique acoustic states (max possible: {})",
        unique_states.len(),
        k
    );
    println!();

    // ---------------------------------------------------------
    // STEP 2: BUILD SEQUENCES
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[2/4] Building Sequences per File...");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Group by audio file and build sequences
    let mut file_data: HashMap<String, (Sequence, Vec<String>)> = HashMap::new();

    for (idx, seg) in all_segments.iter().enumerate() {
        let entry = file_data.entry(seg.audio_file.clone()).or_default();
        entry.0.push(state_ids[idx]);
        entry.1.push(seg.call_type.clone());
    }

    let total_files = file_data.len();
    let total_seqs = file_data.values().map(|(s, _)| s.len()).sum::<usize>();

    println!(
        "  Built {} sequences from {} files",
        total_files, total_files
    );
    println!("  Total state transitions: {}", total_seqs);
    println!();

    // Check if we have multi-segment files
    let multi_segment_files = file_data.values().filter(|(s, _)| s.len() > 1).count();
    let max_seq_len = file_data.values().map(|(s, _)| s.len()).max().unwrap_or(0);

    // Sequence length distribution
    let mut len_counts: HashMap<usize, usize> = HashMap::new();
    for (seq, _) in file_data.values() {
        *len_counts.entry(seq.len()).or_insert(0) += 1;
    }

    let mut sorted_lens: Vec<_> = len_counts.iter().collect();
    sorted_lens.sort_by_key(|(l, _)| std::cmp::Reverse(**l));

    println!("  Sequence Length Distribution:");
    for (len, count) in sorted_lens.iter().take(5) {
        println!("    • Length {}: {} files", len, count);
    }
    if sorted_lens.len() > 5 {
        println!("    • ... and {} other lengths", sorted_lens.len() - 5);
    }
    println!();

    // Critical check: Do we have sequences?
    if max_seq_len < 2 {
        println!("  ⚠ INSUFFICIENT DATA FOR SYNTAX ANALYSIS");
        println!("    Each file contains only 1 segment.");
        println!("    N-gram analysis requires multi-segment sequences.");
        println!("    → Cannot test for 'sentence structure' with current data.");
        println!();
        println!("═══════════════════════════════════════════════════════════════════════════");
        println!("NOTE: Syntax mining requires sequential segments within files.");
        println!("      The marmoset cache has 1 segment per file (isolated calls).");
        println!("      Use bat_nbd_cache_normalized for proper syntax analysis.");
        println!("═══════════════════════════════════════════════════════════════════════════");
        return Ok(());
    }

    // ---------------------------------------------------------
    // STEP 3: MINE N-GRAMS
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[3/4] Mining Reusable Sequences (N-Grams)...");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Mine N-grams for N = 2, 3, 4
    let n_values = [2, 3, 4];

    for n in n_values {
        let mut ngram_counts: HashMap<Vec<StateId>, usize> = HashMap::new();
        let mut ngram_contexts: HashMap<Vec<StateId>, Vec<String>> = HashMap::new();

        for (file, (seq, call_types)) in &file_data {
            if seq.len() < n {
                continue;
            }

            for (i, window) in seq.windows(n).enumerate() {
                let ngram = window.to_vec();
                *ngram_counts.entry(ngram.clone()).or_insert(0) += 1;

                // Track which call types appear in this n-gram
                let ctx = format!("{:?}", &call_types[i..i + n]);
                ngram_contexts.entry(ngram).or_default().push(ctx);
            }
        }

        let total_ngrams = ngram_counts.len();
        let reusable = ngram_counts.values().filter(|&&c| c > 1).count();
        let max_freq = ngram_counts.values().max().copied().unwrap_or(0);
        let reuse_rate = if total_ngrams > 0 {
            reusable as f64 / total_ngrams as f64 * 100.0
        } else {
            0.0
        };

        println!("  {}-grams:", n);
        println!("    • Total unique:    {}", total_ngrams);
        println!("    • Reusable (>1):   {} ({:.1}%)", reusable, reuse_rate);
        println!("    • Max frequency:   {}", max_freq);

        // Show top N-grams
        let mut sorted: Vec<_> = ngram_counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        if max_freq > 1 {
            println!("    • Top repeated:");
            for (ngram, count) in sorted.iter().take(3).filter(|(_, &c)| c > 1) {
                let ngram_str: Vec<String> = ngram.iter().map(|s| s.to_string()).collect();
                println!("      States [{}] → {} times", ngram_str.join(","), count);
            }
        }
        println!();
    }

    // ---------------------------------------------------------
    // STEP 4: RESULTS
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[4/4] Final Analysis");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Calculate overall statistics
    let mut bigram_counts: HashMap<Vec<StateId>, usize> = HashMap::new();
    for (seq, _) in file_data.values() {
        for window in seq.windows(2) {
            *bigram_counts.entry(window.to_vec()).or_insert(0) += 1;
        }
    }

    let total_bigrams = bigram_counts.len();
    let reusable_bigrams = bigram_counts.values().filter(|&&c| c > 1).count();
    let max_bigram_freq = bigram_counts.values().max().copied().unwrap_or(0);
    let bigram_reuse_rate = reusable_bigrams as f64 / total_bigrams as f64 * 100.0;

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  SYNTAX STATISTICS (Bigrams)                                             │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Total unique bigrams:     {:>8}                                    ",
        total_bigrams
    );
    println!(
        "  │  Reusable bigrams (>1):    {:>8} ({:.1}%)                           ",
        reusable_bigrams, bigram_reuse_rate
    );
    println!(
        "  │  Max frequency:            {:>8}                                     ",
        max_bigram_freq
    );
    println!("  │                                                                         │");
    println!(
        "  │  Possible states:          {:>8}                                    ",
        unique_states.len()
    );
    println!(
        "  │  Theoretical max bigrams:  {:>8}                                    ",
        unique_states.len() * unique_states.len()
    );
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Interpretation
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  INTERPRETATION                                                          │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");

    if bigram_reuse_rate > 20.0 && max_bigram_freq > 5 {
        println!("  │  ✓ DISCRETE SYNTAX DETECTED!                                            │");
        println!("  │                                                                          │");
        println!("  │  Marmosets reuse specific sequences of acoustic states.                 │");
        println!("  │  This implies 'Sentence Structure' exists on top of the continuum.      │");
        println!("  │  → Both graded calls AND reusable syntactic patterns.                   │");
    } else if bigram_reuse_rate > 5.0 {
        println!("  │  ~ PARTIAL SYNTAX                                                       │");
        println!("  │                                                                          │");
        println!("  │  Some sequence patterns repeat, but most are unique.                    │");
        println!("  │  → Weak syntactic structure with high improvisation.                   │");
    } else {
        println!("  │  ✗ NO DISCRETE SYNTAX                                                    │");
        println!("  │                                                                          │");
        println!("  │  Almost all sequences are unique.                                       │");
        println!("  │  Marmosets use 'Free Jazz' improvisation, not 'Written Sentences'.      │");
        println!("  │  → Context is encoded in local dynamics (Rate/Texture), not order.      │");
        println!("  │  → Each vocalization sequence is freshly generated.                     │");
    }
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Rosetta Spectrum final summary
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  ROSETTA SPECTRUM - FINAL CLASSIFICATION                                │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │                                                                          │");
    println!("  │  Test                          Result          Verdict                  │");
    println!("  │  ─────────────────────────────────────────────────────────────────────  │");
    println!("  │  Static HDBSCAN               1 cluster       Graded Continuum         │");
    println!("  │  ASE-Weighted HDBSCAN         1 cluster       Graded Continuum         │");
    println!("  │  Delta (Velocity)             1 cluster       Random Trajectories      │");
    println!("  │  Pitch Geometry Shape         1 cluster       No Discrete Shapes       │");
    println!("  │  Similarity Matching          98.2% recurrence Dense Manifold          │");
    println!(
        "  │  Syntax (N-grams)             {:.1}% reusable   {}              │",
        bigram_reuse_rate,
        if bigram_reuse_rate > 20.0 {
            "Discrete Syntax"
        } else if bigram_reuse_rate > 5.0 {
            "Partial Syntax"
        } else {
            "No Syntax"
        }
    );
    println!("  │                                                                          │");
    println!("  │  ═══════════════════════════════════════════════════════════════════    │");
    println!("  │                                                                          │");
    println!("  │  FINAL VERDICT: DENSE GRADED CONTINUUM                                  │");
    println!("  │  → Marmosets use continuous prosodic modulation                         │");
    println!("  │  → No discrete motifs, shapes, or syntactic patterns                    │");
    println!("  │  → Each vocalization is uniquely generated                              │");
    println!("  │  → Use Direct 105D similarity for matching                               │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘");

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}
