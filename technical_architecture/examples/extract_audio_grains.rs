//! Extract Audio Grains for Granular Synthesis
//! =============================================
//!
//! Reads the synthesis library manifest and extracts actual audio segments
//! for each grain and template sequence.
//!
//! Output:
//! - grains/state_XXX.wav      - Individual grain audio files
//! - templates/context_X/YYY.wav - Concatenated template sequences

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
struct GrainEntry {
    id: usize,
    state_id: u32,
    source_file: String,
    start_ms: f32,
    end_ms: f32,
    duration_ms: f32,
    #[allow(dead_code)]
    centroid_features: Vec<f32>,
    dominant_context: i32,
    #[allow(dead_code)]
    context_purity: f64,
    #[allow(dead_code)]
    context_distribution: HashMap<i32, f64>,
    #[allow(dead_code)]
    sample_count: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct SyntaxTemplate {
    id: usize,
    pattern: Vec<u32>,
    pattern_str: String,
    n: usize,
    dominant_context: i32,
    purity: f64,
    total_occurrences: usize,
    #[allow(dead_code)]
    context_distribution: HashMap<i32, f64>,
    grain_ids: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct LibraryManifest {
    grains: Vec<GrainEntry>,
    templates: Vec<SyntaxTemplate>,
}

fn main() -> anyhow::Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║     AUDIO GRAIN EXTRACTOR FOR GRANULAR SYNTHESIS                         ║");
    println!("║     Creating playable audio files from synthesis library                  ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let manifest_path = Path::new("rosetta_synthesis_library/library_manifest.json");
    let output_dir = Path::new("rosetta_grains");
    let bat_audio_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio");

    if !manifest_path.exists() {
        eprintln!("Error: Manifest not found. Run build_synthesis_library first.");
        std::process::exit(1);
    }

    // Load manifest
    let manifest_json = fs::read_to_string(manifest_path)?;
    let manifest: LibraryManifest = serde_json::from_str(&manifest_json)?;

    println!(
        "  Loaded {} grains, {} templates",
        manifest.grains.len(),
        manifest.templates.len()
    );
    println!();

    // Create output directories
    let grains_dir = output_dir.join("grains");
    let templates_dir = output_dir.join("templates");
    fs::create_dir_all(&grains_dir)?;
    fs::create_dir_all(&templates_dir)?;

    // ---------------------------------------------------------
    // PHASE 1: EXTRACT INDIVIDUAL GRAINS
    // ---------------------------------------------------------
    println!("[1/3] Extracting Individual Grains...");
    println!("─────────────────────────────────────────────────────────────────────────");

    // Build grain lookup by state_id
    let mut grain_by_state: HashMap<u32, &GrainEntry> = HashMap::new();
    for grain in &manifest.grains {
        grain_by_state.insert(grain.state_id, grain);
    }

    let mut extracted_count = 0;
    let mut failed_count = 0;

    for grain in &manifest.grains {
        let source_path = bat_audio_dir.join(&grain.source_file);

        if !source_path.exists() {
            println!("  ⚠ Source not found: {}", grain.source_file);
            failed_count += 1;
            continue;
        }

        let output_path = grains_dir.join(format!("state_{:03}.wav", grain.state_id));

        match extract_audio_segment(&source_path, &output_path, grain.start_ms, grain.end_ms) {
            Ok(duration) => {
                extracted_count += 1;
                if extracted_count <= 10 {
                    println!(
                        "  ✓ State {:3}: {} ({:.1}ms) -> {}",
                        grain.state_id,
                        grain.source_file,
                        grain.duration_ms,
                        output_path.display()
                    );
                }
            }
            Err(e) => {
                println!("  ✗ State {:3}: {} - {}", grain.state_id, grain.source_file, e);
                failed_count += 1;
            }
        }
    }

    println!();
    println!("  Extracted: {} grains", extracted_count);
    println!("  Failed:    {} grains", failed_count);
    println!();

    // ---------------------------------------------------------
    // PHASE 2: BUILD TEMPLATE SEQUENCES
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[2/3] Building Template Sequences...");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    // Group templates by context
    let mut templates_by_context: HashMap<i32, Vec<&SyntaxTemplate>> = HashMap::new();
    for template in &manifest.templates {
        templates_by_context
            .entry(template.dominant_context)
            .or_default()
            .push(template);
    }

    let mut template_count = 0;

    for (context, templates) in templates_by_context.iter() {
        // Sort by purity
        let mut sorted = templates.clone();
        sorted.sort_by(|a, b| b.purity.partial_cmp(&a.purity).unwrap());

        // Create context directory
        let context_dir = templates_dir.join(format!("context_{}", context));
        fs::create_dir_all(&context_dir)?;

        println!("  Context {} ({} templates):", context, sorted.len());

        // Take top 5 templates per context
        for (i, template) in sorted.iter().take(5).enumerate() {
            let output_path = context_dir.join(format!("template_{:02}_purity_{:.0}.wav", i, template.purity * 100.0));

            // Build list of grain files to concatenate
            let grain_files: Vec<PathBuf> = template
                .pattern
                .iter()
                .filter_map(|&state_id| Some(grains_dir.join(format!("state_{:03}.wav", state_id))))
                .collect();

            // Check all grain files exist
            let all_exist = grain_files.iter().all(|p| p.exists());

            if all_exist && grain_files.len() == template.pattern.len() {
                match concatenate_wav_files(&grain_files, &output_path) {
                    Ok(_) => {
                        template_count += 1;
                        println!(
                            "    ✓ Template {:2}: {} -> {:.0}% pure, {} occurrences",
                            i,
                            template.pattern_str,
                            template.purity * 100.0,
                            template.total_occurrences
                        );
                    }
                    Err(e) => {
                        println!("    ✗ Template {:2}: Failed - {}", i, e);
                    }
                }
            } else {
                println!(
                    "    ⚠ Template {:2}: Missing {} grain files",
                    i,
                    template.pattern.len() - grain_files.iter().filter(|p| p.exists()).count()
                );
            }
        }
        println!();
    }

    // ---------------------------------------------------------
    // PHASE 3: CREATE SUMMARY
    // ---------------------------------------------------------
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("[3/3] Summary");
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();

    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  GRANULAR SYNTHESIS DATABASE                                             │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "  │  Grains extracted:    {:>5}                                         ",
        extracted_count
    );
    println!(
        "  │  Templates built:     {:>5}                                         ",
        template_count
    );
    println!("  │                                                                          │");
    println!("  │  Output structure:                                                       │");
    println!("  │    rosetta_grains/                                                       │");
    println!("  │    ├── grains/                                                           │");
    println!("  │    │   ├── state_336.wav  (Context 12, 39% pure)                        │");
    println!("  │    │   ├── state_391.wav  (Context 6, 33% pure)                         │");
    println!("  │    │   └── ...                                                            │");
    println!("  │    └── templates/                                                        │");
    println!("  │        ├── context_6/                                                    │");
    println!("  │        │   ├── template_00_purity_82.wav                                 │");
    println!("  │        │   └── ...                                                        │");
    println!("  │        └── context_12/                                                   │");
    println!("  │            └── ...                                                        │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    // Usage
    println!("  ┌─────────────────────────────────────────────────────────────────────────┐");
    println!("  │  HOW TO USE FOR GRANULAR SYNTHESIS                                       │");
    println!("  ├─────────────────────────────────────────────────────────────────────────┤");
    println!("  │                                                                          │");
    println!("  │  1. PLAY A GRAIN:                                                        │");
    println!("  │     aplay rosetta_grains/grains/state_391.wav                           │");
    println!("  │                                                                          │");
    println!("  │  2. PLAY A TEMPLATE (Context 6 = Territorial):                          │");
    println!("  │     aplay rosetta_grains/templates/context_6/template_00_purity_82.wav   │");
    println!("  │                                                                          │");
    println!("  │  3. BUILD NEW SEQUENCE:                                                  │");
    println!("  │     sox state_391.wav state_391.wav state_391.wav new_sequence.wav       │");
    println!("  │     aplay new_sequence.wav                                               │");
    println!("  │                                                                          │");
    println!("  └─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("═══════════════════════════════════════════════════════════════════════════");

    Ok(())
}

/// Extract a segment from a WAV file (handles both float and integer formats)
fn extract_audio_segment(source_path: &Path, output_path: &Path, start_ms: f32, end_ms: f32) -> anyhow::Result<f32> {
    let reader = hound::WavReader::open(source_path)?;
    let spec = reader.spec();

    let start_sample = ((start_ms / 1000.0) * spec.sample_rate as f32) as u64;
    let end_sample = ((end_ms / 1000.0) * spec.sample_rate as f32) as u64;
    let n_samples = end_sample.saturating_sub(start_sample);

    // Create output spec (always 16-bit for compatibility)
    let out_spec = hound::WavSpec {
        channels: spec.channels,
        sample_rate: spec.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(output_path, out_spec)?;

    // Read samples based on source format
    match spec.sample_format {
        hound::SampleFormat::Float => {
            // Float samples (32-bit)
            let mut reader = hound::WavReader::open(source_path)?;

            // Skip to start
            for _ in 0..start_sample {
                if reader.samples::<f32>().next().is_none() {
                    break;
                }
            }

            // Copy samples, converting to 16-bit int
            let mut count = 0u64;
            for sample in reader.samples::<f32>() {
                if count >= n_samples {
                    break;
                }
                let val = sample?;
                let int_val = (val.clamp(-1.0, 1.0) * 32767.0) as i16;
                writer.write_sample(int_val)?;
                count += 1;
            }

            writer.finalize()?;
            Ok((count as f32 / spec.sample_rate as f32) * 1000.0)
        }
        hound::SampleFormat::Int => {
            let mut reader = hound::WavReader::open(source_path)?;

            // Skip to start
            for _ in 0..start_sample {
                if reader.samples::<i32>().next().is_none() {
                    break;
                }
            }

            // Copy samples
            let mut count = 0u64;
            for sample in reader.samples::<i32>() {
                if count >= n_samples {
                    break;
                }
                let val = sample?;
                // Scale to 16-bit
                let int_val = (val as i64 * 32767 / (1i64 << (spec.bits_per_sample - 1))) as i16;
                writer.write_sample(int_val)?;
                count += 1;
            }

            writer.finalize()?;
            Ok((count as f32 / spec.sample_rate as f32) * 1000.0)
        }
    }
}

/// Concatenate multiple WAV files (assumes 16-bit format from our grain extractor)
fn concatenate_wav_files(input_files: &[PathBuf], output_path: &Path) -> anyhow::Result<()> {
    if input_files.is_empty() {
        anyhow::bail!("No input files");
    }

    // Read first file to get format (should be 16-bit from our extractor)
    let first_reader = hound::WavReader::open(&input_files[0])?;
    let spec = first_reader.spec();

    // Create output
    let mut writer = hound::WavWriter::create(output_path, spec)?;

    // Copy all files
    for input_file in input_files {
        let mut reader = hound::WavReader::open(input_file)?;
        for sample in reader.samples::<i16>() {
            writer.write_sample(sample?)?;
        }
    }

    writer.finalize()?;
    Ok(())
}
