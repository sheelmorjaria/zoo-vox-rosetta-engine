//! Phrase Typing Ensemble CLI
//!
//! Combines Closed-Set (k-NN) + Open-Set (HDBSCAN) for stable phrase typing
//!
//! # Usage
//!
//! ```bash
//! # Build phrase library from known calls
//! phrase_typing build-library --input known_phrases/ --output library.json
//!
//! # Classify new segments
//! phrase_typing classify --library library.json --input segments/ --output labels.json
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "phrase_typing")]
#[command(about = "Phrase Typing Ensemble - Stable + Discovery")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Build phrase library from known examples
    BuildLibrary {
        /// Input directory with labeled phrase examples
        #[arg(short, long)]
        input: PathBuf,

        /// Output library file
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Classify segments using ensemble
    Classify {
        /// Phrase library file
        #[arg(short, long)]
        library: PathBuf,

        /// Input segments (JSON with features)
        #[arg(short, long)]
        input: PathBuf,

        /// Output labels file
        #[arg(short, long)]
        output: PathBuf,

        /// Match confidence threshold (0.0-1.0)
        #[arg(long, default_value = "0.85")]
        threshold: f32,
    },

    /// Analyze audio file and extract phrases
    Analyze {
        /// Audio file to analyze
        #[arg(short, long)]
        input: PathBuf,

        /// Phrase library file
        #[arg(long)]
        library: Option<PathBuf>,

        /// Output file for phrase labels
        #[arg(short, long)]
        output: PathBuf,

        /// Match threshold
        #[arg(long, default_value = "0.85")]
        threshold: f32,
    },

    /// Show library statistics
    Stats {
        /// Library file
        #[arg(short, long)]
        library: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::BuildLibrary { input, output } => {
            build_library(&input, &output)?;
        }
        Commands::Classify {
            library,
            input,
            output,
            threshold,
        } => {
            classify_segments(&library, &input, &output, threshold)?;
        }
        Commands::Analyze {
            input,
            library,
            output,
            threshold,
        } => {
            analyze_audio(&input, library.as_deref(), &output, threshold)?;
        }
        Commands::Stats { library } => {
            show_stats(&library)?;
        }
    }

    Ok(())
}

fn build_library(input: &PathBuf, output: &PathBuf) -> Result<()> {
    use technical_architecture::phrase_typing_ensemble::PhraseLibrary;

    println!("Building phrase library from: {:?}", input);

    let mut library = PhraseLibrary::new();

    // Scan input directory for phrase examples
    for entry in std::fs::read_dir(input)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|e| e == "json").unwrap_or(false) {
            // Load phrase features from JSON
            let content = std::fs::read_to_string(&path)?;
            let phrase_data: serde_json::Value = serde_json::from_str(&content)?;

            if let (Some(id), Some(features)) = (
                phrase_data.get("id").and_then(|v| v.as_str()),
                phrase_data.get("features").and_then(|v| v.as_array()),
            ) {
                let features: Vec<f32> = features
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();

                library.add_phrase(id.to_string(), features);
                println!("  Added phrase: {}", id);
            }
        }
    }

    println!("\nSaving library with {} phrases", library.phrases.len());
    library.save(output)?;
    println!("Saved to: {:?}", output);

    Ok(())
}

fn classify_segments(
    library_path: &PathBuf,
    input_path: &PathBuf,
    output_path: &PathBuf,
    threshold: f32,
) -> Result<()> {
    use technical_architecture::phrase_typing_ensemble::{PhraseLabel, PhraseTypingEnsemble};

    println!("Loading phrase library: {:?}", library_path);
    let mut ensemble = PhraseTypingEnsemble::new(library_path.to_str().unwrap(), threshold)?;

    println!("Loading segments from: {:?}", input_path);
    let content = std::fs::read_to_string(input_path)?;
    let segments: serde_json::Value = serde_json::from_str(&content)?;

    let segment_list = segments
        .as_array()
        .with_context(|| "Input must be a JSON array of segments")?;

    println!("\nClassifying {} segments...", segment_list.len());

    let mut results = Vec::new();
    let mut known_count = 0;
    let mut discovered_count = 0;
    let mut noise_count = 0;

    for seg in segment_list {
        let id = seg.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
        let features: Vec<f32> = seg
            .get("features")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();

        let label = ensemble.classify(&features, id);

        let label_str = match &label {
            PhraseLabel::Known(id) => {
                known_count += 1;
                format!("known:{}", id)
            }
            PhraseLabel::Discovered(id) => {
                discovered_count += 1;
                format!("discovered:{}", id)
            }
            PhraseLabel::Noise => {
                noise_count += 1;
                "noise".to_string()
            }
            PhraseLabel::Uncertain { best_match, score } => {
                format!("uncertain:{}:{:.2}", best_match, score)
            }
        };

        results.push(serde_json::json!({
            "id": id,
            "label": label_str,
        }));
    }

    println!("\nResults:");
    println!("  Known:      {}", known_count);
    println!("  Discovered: {}", discovered_count);
    println!("  Noise:      {}", noise_count);

    // Save results
    let output = serde_json::to_string_pretty(&results)?;
    std::fs::write(output_path, output)?;
    println!("\nSaved to: {:?}", output_path);

    // Save updated library
    ensemble.save_library(library_path)?;
    println!("Library updated");

    Ok(())
}

fn analyze_audio(
    input_path: &PathBuf,
    library_path: Option<&std::path::Path>,
    output_path: &PathBuf,
    threshold: f32,
) -> Result<()> {
    use technical_architecture::phrase_typing_ensemble::{PhraseLabel, PhraseTypingEnsemble};

    println!("Analyzing audio: {:?}", input_path);

    // Create ensemble (with or without library)
    let mut ensemble = if let Some(lib_path) = library_path {
        println!("Using library: {:?}", lib_path);
        PhraseTypingEnsemble::new(lib_path.to_str().unwrap(), threshold)?
    } else {
        println!("Starting fresh (no library)");
        PhraseTypingEnsemble::new("/tmp/temp_library.json", threshold)?
    };

    // Load audio
    let audio = load_audio(input_path)?;

    // Segment using smart segmenter
    let segments = segment_audio(&audio, 44100)?;

    println!("Found {} segments", segments.len());

    // Extract features and classify
    let mut results = Vec::new();

    for (i, segment) in segments.iter().enumerate() {
        let features = extract_features(segment, 44100)?;
        let label = ensemble.classify(&features, &format!("seg_{}", i));

        let label_str = match &label {
            PhraseLabel::Known(id) => format!("known:{}", id),
            PhraseLabel::Discovered(id) => format!("discovered:{}", id),
            PhraseLabel::Noise => "noise".to_string(),
            PhraseLabel::Uncertain { best_match, score } => {
                format!("uncertain:{}:{:.2}", best_match, score)
            }
        };

        results.push(serde_json::json!({
            "segment_id": i,
            "label": label_str,
            "duration_ms": segment.len() as f32 / 44100.0 * 1000.0,
        }));
    }

    // Save results
    let output = serde_json::to_string_pretty(&results)?;
    std::fs::write(output_path, output)?;
    println!(
        "\nSaved {} phrase labels to: {:?}",
        results.len(),
        output_path
    );

    let stats = ensemble.stats();
    println!("\nEnsemble Stats:");
    println!("  Known phrases:    {}", stats.known_phrases);
    println!("  Discovered:       {}", stats.discovery_counter);

    Ok(())
}

fn show_stats(library_path: &PathBuf) -> Result<()> {
    use technical_architecture::phrase_typing_ensemble::PhraseLibrary;

    let library = PhraseLibrary::load(library_path)?;

    println!("Phrase Library Statistics");
    println!("=========================");
    println!("Version:    {}", library.version);
    println!("Created:    {}", library.created);
    println!("Phrases:    {}", library.phrases.len());
    println!();

    if !library.phrases.is_empty() {
        println!("Phrase Details:");
        for (id, template) in &library.phrases {
            println!("  {} - {} samples", id, template.sample_count);
        }
    }

    Ok(())
}

// Helper functions
fn load_audio(path: &std::path::Path) -> Result<Vec<f32>> {
    // Use hound for WAV loading
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max_val)
                .collect()
        }
    };

    // Convert to mono if stereo
    let mono = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|ch| (ch[0] + ch.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok(mono)
}

fn segment_audio(audio: &[f32], sr: u32) -> Result<Vec<Vec<f32>>> {
    // Simple energy-based segmentation
    let frame_size = (sr as f32 * 0.02) as usize; // 20ms frames
    let hop = frame_size / 2;

    let mut segments = Vec::new();
    let mut in_segment = false;
    let mut segment_start = 0;

    let energy_threshold = 0.01;

    for i in (0..audio.len() - frame_size).step_by(hop) {
        let frame = &audio[i..i + frame_size];
        let rms = (frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32).sqrt();

        if rms > energy_threshold && !in_segment {
            in_segment = true;
            segment_start = i;
        } else if rms <= energy_threshold && in_segment {
            // End segment
            let segment = audio[segment_start..i].to_vec();
            if segment.len() > sr as usize / 10 {
                // Min 100ms
                segments.push(segment);
            }
            in_segment = false;
        }
    }

    // Capture last segment
    if in_segment {
        let segment = audio[segment_start..].to_vec();
        if segment.len() > sr as usize / 10 {
            segments.push(segment);
        }
    }

    Ok(segments)
}

fn extract_features(audio: &[f32], sr: u32) -> Result<Vec<f32>> {
    // Simplified 105D feature extraction
    // In production, would use micro_dynamics_extractor

    let mut features = vec![0.0f32; 105];

    if audio.is_empty() {
        return Ok(features);
    }

    // Basic features
    features[0] = audio.len() as f32 / sr as f32 * 1000.0; // duration_ms
    features[1] = (audio.iter().map(|x| x * x).sum::<f32>() / audio.len() as f32).sqrt(); // RMS

    // Spectral centroid approximation
    let mut centroid = 0.0;
    for (i, &s) in audio.iter().enumerate() {
        centroid += (i as f32 * s.abs()) / audio.len() as f32;
    }
    features[2] = centroid / (sr as f32) * 1000.0;

    // Fill remaining with simple stats
    let mean = audio.iter().sum::<f32>() / audio.len() as f32;
    let var = audio.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / audio.len() as f32;
    features[3] = mean;
    features[4] = var.sqrt();

    Ok(features)
}
