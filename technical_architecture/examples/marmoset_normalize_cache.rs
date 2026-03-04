//! Normalize Marmoset 105D Feature Cache
//! =======================================
//!
//! Loads existing marmoset_train_cache and normalizes features
//! so all 105 dimensions contribute equally to distance calculations.

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CachedSegment {
    audio_file: String,
    n_samples: usize,
    labels: LabelInfo,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct LabelInfo {
    call_type: String,
    label_id: i32,
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
    println!("║     MARMOSET CACHE NORMALIZATION (105D -> Unit Variance)                ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let input_cache = Path::new("marmoset_train_cache/feature_cache_eval/all_features.bin");
    let input_manifest = Path::new("marmoset_train_manifest.json");
    let output_dir = Path::new("marmoset_nbd_cache_normalized");

    if !input_cache.exists() {
        eprintln!("Error: Input cache not found: {}", input_cache.display());
        std::process::exit(1);
    }

    fs::create_dir_all(output_dir)?;

    // =====================================================
    // STEP 1: LOAD MANIFEST AND FEATURES
    // =====================================================
    println!("[1/3] Loading existing cache...");
    println!("─────────────────────────────────────────────────────────────────────────");

    // Load manifest
    let manifest_json = fs::read_to_string(input_manifest)?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_json)?;

    let n_samples = manifest["n_samples"].as_u64().unwrap_or(6000) as usize;
    let num_classes = manifest["num_classes"].as_u64().unwrap_or(6) as usize;

    println!("  Manifest: {} samples, {} classes", n_samples, num_classes);

    // Load binary features
    let data = fs::read(input_cache)?;
    println!("  Binary cache: {} bytes", data.len());

    // Parse features (skip 3-float = 12 byte header)
    let n_features = 105;
    let n_floats = (data.len() - 12) / 4;

    println!(
        "  Total floats: {} (expected {})",
        n_floats,
        n_samples * n_features
    );

    // Build feature matrix
    let mut feature_matrix: Vec<Vec<f32>> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut row = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let offset = 12 + (i * n_features + j) * 4;
            if offset + 4 <= data.len() {
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ];
                row.push(f32::from_le_bytes(bytes));
            }
        }
        feature_matrix.push(row);
    }

    println!("  Feature matrix: {} × {}", n_samples, n_features);
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
    for row in &feature_matrix {
        for (j, &val) in row.iter().enumerate().take(n_features) {
            means[j] += val as f64;
        }
    }
    for m in means.iter_mut() {
        *m /= n_samples as f64;
    }

    // Second pass: compute stds
    for row in &feature_matrix {
        for (j, &val) in row.iter().enumerate().take(n_features) {
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

    // Normalize
    let normalized: Vec<Vec<f32>> = feature_matrix
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &val)| ((val as f64 - means[j]) / stds[j]) as f32)
                .collect()
        })
        .collect();

    // Verify normalization
    let mut new_means = vec![0.0f64; n_features];
    let mut new_stds = vec![0.0f64; n_features];

    for row in &normalized {
        for (j, &val) in row.iter().enumerate().take(n_features) {
            new_means[j] += val as f64;
        }
    }
    for m in new_means.iter_mut() {
        *m /= n_samples as f64;
    }

    for row in &normalized {
        for (j, &val) in row.iter().enumerate().take(n_features) {
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

    // Save normalized features in JSON batches
    let batch_size = 500;
    let n_batches = (n_samples + batch_size - 1) / batch_size;

    println!(
        "  Saving {} batches to {}...",
        n_batches,
        output_dir.display()
    );

    // Get call types from manifest
    let empty_vec = Vec::new();
    let samples = manifest["samples"].as_array().unwrap_or(&empty_vec);

    for (batch_idx, chunk) in normalized.chunks(batch_size).enumerate() {
        let mut batch_data = Vec::new();

        for (i, features) in chunk.iter().enumerate() {
            let global_idx = batch_idx * batch_size + i;
            let sample = samples.get(global_idx);

            let call_type = sample
                .and_then(|s| s["labels"]["call_type"].as_str())
                .unwrap_or("Unknown")
                .to_string();

            let label_id = sample
                .and_then(|s| s["labels"]["label_id"].as_i64())
                .unwrap_or(0) as i32;

            let audio_file = sample
                .and_then(|s| s["audio_file"].as_str())
                .unwrap_or("unknown")
                .to_string();

            batch_data.push(serde_json::json!({
                "audio_file": audio_file,
                "call_type": call_type,
                "label_id": label_id,
                "segment_idx": global_idx,
                "features": features
            }));
        }

        let filename = format!(
            "{}/nbd_batch_{:04}.json",
            output_dir.display(),
            batch_idx + 1
        );
        let file = File::create(&filename)?;
        serde_json::to_writer(BufWriter::new(file), &batch_data)?;
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
    println!("  Samples: {}", n_samples);
    println!("  Batches: {}", n_batches);
    println!("  Feature dims: {}", n_features);
    println!();

    Ok(())
}
