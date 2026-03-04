//! Fast BEANS-Zero Manifest Creator (Rust Implementation)
//! =======================================================
//!
//! Reads HuggingFace Arrow files directly and creates manifest + WAV files.
//! Much faster than Python version due to:
//! - Zero-copy Arrow reading
//! - Rayon parallel processing
//! - Faster I/O
//!
//! Usage:
//!   cargo run --release --bin beans_create_manifest -- /path/to/beans_zero_test /output/manifest.json

use anyhow::{Context, Result};
use arrow::array::{Array, Float32Array, LargeListArray, ListArray, RecordBatch, StringArray, StructArray};
use arrow::datatypes::DataType;
use arrow::ipc::reader::StreamReader;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// Constants
const SAMPLE_RATE: u32 = 44100;

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
struct BeansManifest {
    dataset: String,
    n_samples: usize,
    samples: Vec<ManifestSample>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ManifestSample {
    audio_file: String,
    n_samples: u32,
    labels: Labels,
}

#[derive(Debug, Serialize, Deserialize)]
struct Labels {
    output: String,
    task: String,
}

// ============================================================================
// WAV Writer (simple implementation)
// ============================================================================

fn write_wav_file(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    // Convert f32 to i16
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let int_sample = (clamped * 32767.0) as i16;
        writer.write_sample(int_sample)?;
    }

    writer.finalize()?;
    Ok(())
}

// ============================================================================
// Arrow File Processing
// ============================================================================

fn process_arrow_file(
    arrow_path: &Path,
    audio_dir: &Path,
    start_idx: usize,
    file_idx: usize,
) -> Result<Vec<ManifestSample>> {
    let file = File::open(arrow_path)?;

    // Use BufReader for better performance
    let buf_reader = BufReader::new(file);

    // HuggingFace uses streaming Arrow format (no footer)
    let reader = StreamReader::try_new(buf_reader, None)?;

    let mut samples = Vec::new();
    let mut batch_count = 0;

    for batch_result in reader {
        let batch = batch_result?;
        batch_count += 1;

        // Get column indices from schema
        let schema = batch.schema();

        // Debug: print schema on first file
        if file_idx == 0 && batch_count == 1 {
            eprintln!("Schema fields: {:?}", schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>());
            eprintln!("First batch has {} rows", batch.num_rows());
        }

        // Find columns - HuggingFace uses different names
        let audio_col_idx = find_column_idx(&schema, &["audio", "audio_array", "array"]);
        let output_col_idx = find_column_idx(&schema, &["output", "label", "class", "category"]);
        let dataset_name_idx = find_column_idx(&schema, &["dataset_name", "source", "dataset"]);

        if audio_col_idx.is_none() {
            if file_idx == 0 && batch_count == 1 {
                eprintln!("No audio column found!");
            }
            continue;
        }

        let audio_col_idx = audio_col_idx.unwrap();
        let num_rows = batch.num_rows();

        for row in 0..num_rows {
            // Extract audio data
            let audio_data = extract_audio_from_column(&batch, audio_col_idx, row);

            if audio_data.is_empty() {
                continue;
            }

            // Extract labels
            let output_label = output_col_idx
                .and_then(|idx| extract_string_value(&batch, idx, row))
                .unwrap_or_else(|| "unknown".to_string());

            let task = dataset_name_idx
                .and_then(|idx| extract_string_value(&batch, idx, row))
                .unwrap_or_else(|| "unknown".to_string());

            // Create WAV file with unique index
            let sample_idx = start_idx + samples.len();
            let audio_filename = format!("sample_{:06}.wav", sample_idx);
            let audio_path = audio_dir.join(&audio_filename);

            if let Err(e) = write_wav_file(&audio_path, &audio_data, SAMPLE_RATE) {
                eprintln!("Error writing WAV file {}: {}", audio_path.display(), e);
                continue;
            }

            samples.push(ManifestSample {
                audio_file: audio_path.to_string_lossy().to_string(),
                n_samples: audio_data.len() as u32,
                labels: Labels {
                    output: output_label,
                    task,
                },
            });
        }
    }

    Ok(samples)
}

fn find_column_idx(schema: &arrow::datatypes::Schema, possible_names: &[&str]) -> Option<usize> {
    for name in possible_names {
        if let Ok(idx) = schema.index_of(name) {
            return Some(idx);
        }
    }
    None
}

fn extract_string_value(batch: &RecordBatch, col_idx: usize, row: usize) -> Option<String> {
    let column = batch.column(col_idx);

    // Try StringArray
    if let Some(str_array) = column.as_any().downcast_ref::<StringArray>() {
        if row < str_array.len() && !str_array.is_null(row) {
            return Some(str_array.value(row).to_string());
        }
    }

    // Try LargeStringArray
    if let Some(str_array) = column.as_any().downcast_ref::<arrow::array::LargeStringArray>() {
        if row < str_array.len() && !str_array.is_null(row) {
            return Some(str_array.value(row).to_string());
        }
    }

    None
}

fn extract_audio_from_column(batch: &RecordBatch, col_idx: usize, row: usize) -> Vec<f32> {
    let column = batch.column(col_idx);

    match column.data_type() {
        DataType::Struct(fields) => {
            // HuggingFace audio format: struct with 'array' and 'sampling_rate'
            let struct_array = column.as_any().downcast_ref::<StructArray>().unwrap();

            // Find 'array' field
            for (i, field) in fields.iter().enumerate() {
                if field.name() == "array" {
                    let array_field = struct_array.column(i);
                    return extract_float_list(array_field, row);
                }
            }
            Vec::new()
        }
        DataType::List(_) => {
            // BEANS-Zero stores audio directly as List(float64)
            extract_float_list(column, row)
        }
        DataType::LargeList(_) => {
            extract_float_list_from_large(column, row)
        }
        DataType::FixedSizeList(_, _) => {
            extract_float_list(column, row)
        }
        _ => {
            eprintln!("Unknown audio column type: {:?}", column.data_type());
            Vec::new()
        }
    }
}

fn extract_float_list(array: &dyn Array, row: usize) -> Vec<f32> {
    // Try ListArray<Float64> (BEANS-Zero uses float64)
    if let Some(list_array) = array.as_any().downcast_ref::<ListArray>() {
        if row < list_array.len() && !list_array.is_null(row) {
            let value = list_array.value(row);

            // Try Float64 first (BEANS-Zero format)
            if let Some(float_array) = value.as_any().downcast_ref::<arrow::array::Float64Array>() {
                return float_array.values().iter().map(|&v| v as f32).collect();
            }

            // Try Float32
            if let Some(float_array) = value.as_any().downcast_ref::<Float32Array>() {
                return float_array.values().to_vec();
            }
        }
    }

    // Try FixedSizeList
    if let Some(fixed_list) = array.as_any().downcast_ref::<arrow::array::FixedSizeListArray>() {
        if row < fixed_list.len() && !fixed_list.is_null(row) {
            let value = fixed_list.value(row);

            if let Some(float_array) = value.as_any().downcast_ref::<arrow::array::Float64Array>() {
                return float_array.values().iter().map(|&v| v as f32).collect();
            }

            if let Some(float_array) = value.as_any().downcast_ref::<Float32Array>() {
                return float_array.values().to_vec();
            }
        }
    }

    Vec::new()
}

fn extract_float_list_from_large(array: &dyn Array, row: usize) -> Vec<f32> {
    if let Some(large_list) = array.as_any().downcast_ref::<LargeListArray>() {
        if row < large_list.len() && !large_list.is_null(row) {
            let value = large_list.value(row);

            // Try Float64 first
            if let Some(float_array) = value.as_any().downcast_ref::<arrow::array::Float64Array>() {
                return float_array.values().iter().map(|&v| v as f32).collect();
            }

            // Try Float32
            if let Some(float_array) = value.as_any().downcast_ref::<Float32Array>() {
                return float_array.values().to_vec();
            }
        }
    }
    Vec::new()
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <dataset_dir> <output_manifest>", args[0]);
        eprintln!("Example: {} beans_zero_data/beans_zero_test beans_zero_manifest.json", args[0]);
        std::process::exit(1);
    }

    let dataset_dir = PathBuf::from(&args[1]);
    let output_manifest = PathBuf::from(&args[2]);

    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║     BEANS-Zero Manifest Creator (Rust - Fast)                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    // Read state.json to get list of Arrow files
    let state_path = dataset_dir.join("state.json");
    let state_data = fs::read_to_string(&state_path)
        .context("Failed to read state.json - is this a HuggingFace dataset?")?;

    #[derive(Deserialize)]
    struct State {
        _data_files: Vec<HashMap<String, String>>,
    }

    let state: State = serde_json::from_str(&state_data)?;
    let arrow_files: Vec<PathBuf> = state
        ._data_files
        .iter()
        .map(|f| dataset_dir.join(&f["filename"]))
        .collect();

    println!("Dataset directory: {}", dataset_dir.display());
    println!("Found {} Arrow files", arrow_files.len());

    // Create audio output directory
    let audio_dir = output_manifest.parent().unwrap_or(Path::new(".")).join("beans_audio_rust");
    fs::create_dir_all(&audio_dir)?;
    println!("Audio directory: {}", audio_dir.display());

    let start_time = Instant::now();
    let total_samples = AtomicUsize::new(0);

    println!();
    println!("Processing {} Arrow files...", arrow_files.len());

    // Estimate samples per file for unique naming
    // BEANS-Zero has ~91,965 samples across 291 files = ~316 samples per file
    let samples_per_file = 500; // Conservative estimate

    // Process Arrow files sequentially first (debug)
    let mut all_samples: Vec<ManifestSample> = Vec::new();

    for (file_idx, arrow_path) in arrow_files.iter().enumerate() {
        let start_idx = file_idx * samples_per_file;

        match process_arrow_file(arrow_path, &audio_dir, start_idx, file_idx) {
            Ok(samples) => {
                let count = samples.len();
                total_samples.fetch_add(count, Ordering::SeqCst);
                all_samples.extend(samples);

                if (file_idx + 1) % 20 == 0 {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let rate = (file_idx + 1) as f64 / elapsed;
                    let eta = (arrow_files.len() - file_idx - 1) as f64 / rate;
                    println!(
                        "  Processed {}/{} files ({} samples) - ETA: {:.0}s",
                        file_idx + 1,
                        arrow_files.len(),
                        total_samples.load(Ordering::SeqCst),
                        eta
                    );
                    // Force flush
                    let _ = std::io::stdout().flush();
                }
            }
            Err(e) => {
                eprintln!("Error processing {}: {}", arrow_path.display(), e);
            }
        }
    }

    let mut all_samples_flat = all_samples;

    println!();
    println!("Total samples extracted: {}", all_samples_flat.len());
    println!("Re-numbering files...");

    // Re-number files sequentially
    for (idx, sample) in all_samples_flat.iter_mut().enumerate() {
        let old_path = PathBuf::from(&sample.audio_file);
        let new_filename = format!("sample_{:06}.wav", idx);
        let new_path = audio_dir.join(&new_filename);

        if old_path != new_path {
            if let Err(e) = fs::rename(&old_path, &new_path) {
                // If rename fails (cross-device), try copy + delete
                if fs::copy(&old_path, &new_path).is_ok() {
                    let _ = fs::remove_file(&old_path);
                } else {
                    eprintln!("Warning: Could not rename {} to {}: {}", old_path.display(), new_path.display(), e);
                }
            }
            sample.audio_file = new_path.to_string_lossy().to_string();
        }
    }

    let elapsed = start_time.elapsed();
    let total = all_samples_flat.len();
    let rate = total as f64 / elapsed.as_secs_f64();

    println!();
    println!("Processing completed in {:.1}s ({:.1} samples/s)", elapsed.as_secs_f64(), rate);

    // Create manifest
    let manifest = BeansManifest {
        dataset: "BEANS-Zero".to_string(),
        n_samples: total,
        samples: all_samples_flat,
    };

    // Save manifest
    let manifest_json = serde_json::to_string_pretty(&manifest)?;
    let mut file = BufWriter::new(File::create(&output_manifest)?);
    file.write_all(manifest_json.as_bytes())?;

    println!();
    println!("Created manifest: {}", output_manifest.display());
    println!("  Total samples: {}", total);
    println!("  Audio directory: {}", audio_dir.display());

    // Count unique labels
    let mut label_counts: HashMap<String, usize> = HashMap::new();
    for sample in &manifest.samples {
        *label_counts.entry(sample.labels.output.clone()).or_insert(0) += 1;
    }

    println!("  Unique labels: {}", label_counts.len());
    println!();
    println!("Top 20 labels by frequency:");

    let mut sorted_labels: Vec<_> = label_counts.into_iter().collect();
    sorted_labels.sort_by(|a, b| b.1.cmp(&a.1));

    for (label, count) in sorted_labels.into_iter().take(20) {
        println!("    {}: {}", label, count);
    }

    Ok(())
}
