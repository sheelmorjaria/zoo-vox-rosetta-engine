//! Convert RF JSON models to compact binary format (bincode)
//! ==============================================================
//!
//! This script converts large JSON model files to bincode format,
//! which is 3-5x smaller and loads much faster with less memory.
//!
//! Usage:
//!   cargo run --release --bin convert_rf_to_bincode

use anyhow::Result;
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use technical_architecture::classical_ml::RandomForestClassifier;

fn convert_model(json_path: &Path, bincode_path: &Path) -> Result<()> {
    let file_size = fs::metadata(json_path)?.len();
    let size_gb = file_size as f64 / (1024.0 * 1024.0 * 1024.0);

    println!(
        "Converting: {} ({:.2} GB)",
        json_path.file_name().unwrap().to_string_lossy(),
        size_gb
    );

    // Use buffered reader for large files
    let file = fs::File::open(json_path)?;
    let reader = BufReader::new(file);

    println!("  Loading JSON...");
    let model: RandomForestClassifier = serde_json::from_reader(reader)?;

    println!(
        "  Model loaded: {} trees, {} classes",
        model.n_trees(),
        model.n_classes()
    );

    // Write as bincode
    println!("  Writing bincode...");
    let out_file = fs::File::create(bincode_path)?;
    let writer = BufWriter::new(out_file);
    bincode::serialize_into(writer, &model)?;

    // Report compression
    let new_size = fs::metadata(bincode_path)?.len();
    let new_size_gb = new_size as f64 / (1024.0 * 1024.0 * 1024.0);
    let compression_ratio = file_size as f64 / new_size as f64;

    println!(
        "  Done: {:.2} GB -> {:.2} GB ({:.1}x compression)",
        size_gb, new_size_gb, compression_ratio
    );

    Ok(())
}

fn main() -> Result<()> {
    println!("==============================================================");
    println!("  RF Model JSON to Bincode Converter");
    println!("==============================================================\n");

    let models_dir = Path::new("specialist_rf_models");

    // Find all JSON model files
    let entries: Vec<_> = fs::read_dir(models_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "json").unwrap_or(false))
        .collect();

    println!("Found {} JSON model files\n", entries.len());

    let mut total_original = 0u64;
    let mut total_compressed = 0u64;
    let mut converted = 0usize;
    let mut failed = 0usize;

    for entry in &entries {
        let json_path = entry.path();
        let bincode_path = json_path.with_extension("bincode");

        // Skip if bincode already exists and is newer
        if bincode_path.exists() {
            let json_time = fs::metadata(&json_path)?.modified()?;
            let bin_time = fs::metadata(&bincode_path)?.modified()?;
            if bin_time > json_time {
                println!(
                    "Skipping: {} (bincode up to date)",
                    json_path.file_name().unwrap().to_string_lossy()
                );
                continue;
            }
        }

        match convert_model(&json_path, &bincode_path) {
            Ok(()) => {
                total_original += fs::metadata(&json_path)?.len();
                total_compressed += fs::metadata(&bincode_path)?.len();
                converted += 1;
            }
            Err(e) => {
                println!("  ERROR: {}", e);
                failed += 1;
            }
        }
        println!();
    }

    println!("==============================================================");
    println!("  Conversion Summary");
    println!("==============================================================");
    println!("  Converted: {}", converted);
    println!("  Failed:    {}", failed);

    if converted > 0 {
        let orig_gb = total_original as f64 / (1024.0 * 1024.0 * 1024.0);
        let comp_gb = total_compressed as f64 / (1024.0 * 1024.0 * 1024.0);
        let ratio = total_original as f64 / total_compressed as f64;

        println!();
        println!("  Total original:    {:.2} GB", orig_gb);
        println!("  Total compressed:  {:.2} GB", comp_gb);
        println!("  Compression ratio: {:.1}x", ratio);
    }

    Ok(())
}
