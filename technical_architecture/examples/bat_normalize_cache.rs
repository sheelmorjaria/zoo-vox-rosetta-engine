//! Normalize Bat 105D Feature Cache
//! ==================================
//!
//! Loads existing bat_nbd_cache_channel and normalizes features
//! so all 105 dimensions contribute equally to distance calculations.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CachedSegmentNBD {
    source_file: String,
    context: i32,
    emitter: i32,
    segment_idx: usize,
    start_ms: f32,
    end_ms: f32,
    boundary_type: String,
    features: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct NormalizationParams {
    means: Vec<f64>,
    stds: Vec<f64>,
    n_samples: usize,
    n_features: usize,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     BAT CACHE NORMALIZATION (105D -> Unit Variance)                      ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let input_dir = Path::new("bat_nbd_cache_channel");
    let output_dir = Path::new("bat_nbd_cache_normalized");

    if !input_dir.exists() {
        eprintln!("Error: Input cache not found: {}", input_dir.display());
        std::process::exit(1);
    }

    fs::create_dir_all(output_dir)?;

    // =====================================================
    // STEP 1: LOAD ALL FEATURES
    // =====================================================
    println!("[1/3] Loading existing cache...");
    println!("─────────────────────────────────────────────────────────────────────────");

    let cache_files: Vec<_> = fs::read_dir(input_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
        .map(|e| e.path())
        .collect();

    println!("  Found {} cache files", cache_files.len());

    // Load all segments in parallel
    let all_segments: Vec<CachedSegmentNBD> = cache_files
        .par_iter()
        .flat_map(|file| {
            let json = fs::read_to_string(file).ok()?;
            let batch: Vec<CachedSegmentNBD> = serde_json::from_str(&json).ok()?;
            Some(batch)
        })
        .flatten()
        .collect();

    let n_samples = all_segments.len();
    let n_features = 105;

    println!("  Loaded {} segments", n_samples);
    println!();

    // =====================================================
    // STEP 2: COMPUTE NORMALIZATION PARAMETERS
    // =====================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[2/3] Computing Normalization Parameters");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Compute mean and std for each dimension
    let mut means = vec![0.0f64; n_features];
    let mut stds = vec![0.0f64; n_features];

    // First pass: compute means
    for seg in &all_segments {
        for (j, &val) in seg.features.iter().enumerate().take(n_features) {
            means[j] += val as f64;
        }
    }
    for m in means.iter_mut() {
        *m /= n_samples as f64;
    }

    // Second pass: compute stds
    for seg in &all_segments {
        for (j, &val) in seg.features.iter().enumerate().take(n_features) {
            stds[j] += (val as f64 - means[j]).powi(2);
        }
    }
    for s in stds.iter_mut() {
        *s = (*s / n_samples as f64).sqrt().max(1e-8);
    }

    // Report statistics by section
    let base_mean: f64 = means[..45].iter().sum::<f64>() / 45.0;
    let base_std: f64 = stds[..45].iter().sum::<f64>() / 45.0;
    let base_var: f64 = stds[..45].iter().map(|s| s.powi(2)).sum();

    let macro_mean: f64 = means[45..75].iter().sum::<f64>() / 30.0;
    let macro_std: f64 = stds[45..75].iter().sum::<f64>() / 30.0;
    let macro_var: f64 = stds[45..75].iter().map(|s| s.powi(2)).sum();

    let micro_mean: f64 = means[75..105].iter().sum::<f64>() / 30.0;
    let micro_std: f64 = stds[75..105].iter().sum::<f64>() / 30.0;
    let micro_var: f64 = stds[75..105].iter().map(|s| s.powi(2)).sum();

    let total_var = base_var + macro_var + micro_var;

    println!("  BEFORE NORMALIZATION:");
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!(
        "  │  Base 45D:   mean={:>10.2}, std={:>10.2}, var_contrib={:>5.1}%",
        base_mean,
        base_std,
        base_var / total_var * 100.0
    );
    println!(
        "  │  Macro 30D:  mean={:>10.4}, std={:>10.4}, var_contrib={:>5.2}%",
        macro_mean,
        macro_std,
        macro_var / total_var * 100.0
    );
    println!(
        "  │  Micro 30D:  mean={:>10.4}, std={:>10.4}, var_contrib={:>5.2}%",
        micro_mean,
        micro_std,
        micro_var / total_var * 100.0
    );
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // =====================================================
    // STEP 3: NORMALIZE AND SAVE
    // =====================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[3/3] Normalizing and Saving");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Normalize and collect
    let normalized_segments: Vec<CachedSegmentNBD> = all_segments
        .into_iter()
        .map(|mut seg| {
            for j in 0..n_features {
                seg.features[j] = ((seg.features[j] as f64 - means[j]) / stds[j]) as f32;
            }
            seg
        })
        .collect();

    // Verify normalization
    let mut new_means = vec![0.0f64; n_features];
    let mut new_stds = vec![0.0f64; n_features];

    for seg in &normalized_segments {
        for (j, &val) in seg.features.iter().enumerate().take(n_features) {
            new_means[j] += val as f64;
        }
    }
    for m in new_means.iter_mut() {
        *m /= n_samples as f64;
    }

    for seg in &normalized_segments {
        for (j, &val) in seg.features.iter().enumerate().take(n_features) {
            new_stds[j] += (val as f64 - new_means[j]).powi(2);
        }
    }
    for s in new_stds.iter_mut() {
        *s = (*s / n_samples as f64).sqrt();
    }

    let new_base_var: f64 = new_stds[..45].iter().map(|s| s.powi(2)).sum();
    let new_macro_var: f64 = new_stds[45..75].iter().map(|s| s.powi(2)).sum();
    let new_micro_var: f64 = new_stds[75..105].iter().map(|s| s.powi(2)).sum();
    let new_total_var = new_base_var + new_macro_var + new_micro_var;

    println!("  AFTER NORMALIZATION:");
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!(
        "  │  Base 45D:   var_contrib={:>5.1}%",
        new_base_var / new_total_var * 100.0
    );
    println!(
        "  │  Macro 30D:  var_contrib={:>5.1}%",
        new_macro_var / new_total_var * 100.0
    );
    println!(
        "  │  Micro 30D:  var_contrib={:>5.1}%",
        new_micro_var / new_total_var * 100.0
    );
    println!("  │                                                                         │");
    println!("  │  All 105 dimensions now contribute EQUALLY!                            │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Save in batches
    let batch_size = 500;
    let n_batches = (normalized_segments.len() + batch_size - 1) / batch_size;

    println!(
        "  Saving {} batches to {}...",
        n_batches,
        output_dir.display()
    );

    for (i, chunk) in normalized_segments.chunks(batch_size).enumerate() {
        let filename = format!("{}/nbd_batch_{:04}.json", output_dir.display(), i + 1);
        let file = File::create(&filename)?;
        serde_json::to_writer(BufWriter::new(file), &chunk)?;

        if (i + 1) % 500 == 0 {
            println!("    Saved batch {}/{}", i + 1, n_batches);
        }
    }

    // Save normalization parameters
    let params = NormalizationParams {
        means: means.clone(),
        stds: stds.clone(),
        n_samples,
        n_features,
    };
    let params_file = File::create(output_dir.join("normalization_params.json"))?;
    serde_json::to_writer(BufWriter::new(params_file), &params)?;

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("NORMALIZATION COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!("  Output: {}", output_dir.display());
    println!("  Segments: {}", normalized_segments.len());
    println!("  Batches: {}", n_batches);
    println!();

    Ok(())
}
