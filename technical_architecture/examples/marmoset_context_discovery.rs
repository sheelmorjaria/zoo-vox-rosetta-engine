//! Marmoset Context Discovery Pipeline
//!
//! Applies the "Anchor and Propagate" workflow to the marmoset vocalization dataset.
//! Labels are extracted from filenames (Phee, Twitter, Tsik, Trill, Infant_cry, Seep).
//!
//! Dataset: ~/birdsong_analysis/data/Vocalizations/

use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use technical_architecture::{
    species::FeatureWeights, AcousticSimilarityEngine, SimilarityMetric, ZooVoxFeatureExtractor,
};

const FEATURE_DIM: usize = 45;
const MAX_FILES: usize = 5000; // Process first 5000 files

// =============================================================================
// Marmoset Call Types (from filenames)
// =============================================================================

fn get_marmoset_call_type(filename: &str) -> String {
    // Extract call type from filename (e.g., "Phee_12345.flac" -> "Phee")
    let name = Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Unknown");

    let parts: Vec<&str> = name.split('_').collect();
    if parts.len() >= 2 {
        // Check for known call types
        let call_type = parts[0];
        match call_type {
            "Phee" | "Twitter" | "Tsik" | "Trill" | "Infant" | "Seep" | "Vocalization" => {
                if call_type == "Infant" && parts.len() > 1 && parts[1] == "cry" {
                    return "Infant_cry".to_string();
                }
                return call_type.to_string();
            }
            _ => {}
        }
    }
    "Unknown".to_string()
}

// =============================================================================
// Discovered Phrase
// =============================================================================

#[derive(Debug, Clone)]
struct DiscoveredPhrase {
    features: Vec<f32>,
    label: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║        Marmoset Context Discovery Pipeline                                     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();

    // =========================================================================
    // STEP 1: Find Audio Files
    // =========================================================================
    println!("[1/4] Scanning for marmoset vocalizations...");

    let marmoset_data_path = dirs::home_dir()
        .map(|h| h.join("birdsong_analysis/data/Vocalizations"))
        .expect("Could not find home directory");

    let all_files: Vec<_> = walkdir::WalkDir::new(&marmoset_data_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "flac" || ext == "wav")
        })
        .take(MAX_FILES)
        .collect();

    let total_files = all_files.len();
    println!("Found {} audio files to process", total_files);

    // Show label distribution
    let mut label_counts: HashMap<String, usize> = HashMap::new();
    for entry in &all_files {
        let filename = entry.file_name().to_str().unwrap_or("");
        let label = get_marmoset_call_type(filename);
        *label_counts.entry(label).or_insert(0) += 1;
    }
    println!("\nCall type distribution:");
    let mut labels: Vec<_> = label_counts.iter().collect();
    labels.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (label, count) in labels.iter().take(10) {
        println!("  {}: {} files", label, count);
    }
    println!();

    if total_files == 0 {
        println!("No files found. Exiting.");
        return Ok(());
    }

    // =========================================================================
    // STEP 2: Feature Extraction
    // =========================================================================
    println!(
        "[2/4] Extracting 45D features from {} files...",
        total_files
    );

    let extract_start = Instant::now();
    let processed = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));

    // Parallel processing
    let all_phrases: Vec<Option<DiscoveredPhrase>> = all_files
        .par_iter()
        .map(|entry| {
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 500 == 0 {
                println!("  Progress: {}/{}", count, total_files);
            }

            let path = entry.path();

            // Load audio
            let audio = match load_audio_file(path) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("  Error loading {:?}: {}", path, e);
                    errors.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            };

            if audio.samples.len() < 100 {
                return None;
            }

            // Extract 45D features
            let mut extractor = ZooVoxFeatureExtractor::new(audio.sample_rate);
            match extractor.extract_45d(&audio.samples) {
                Ok(features) => {
                    let label = get_marmoset_call_type(entry.file_name().to_str().unwrap_or(""));
                    Some(DiscoveredPhrase {
                        features: features.to_vector().iter().map(|&f| f as f32).collect(),
                        label,
                    })
                }
                Err(e) => {
                    eprintln!("  Error extracting features from {:?}: {}", path, e);
                    None
                }
            }
        })
        .collect();

    let extract_time = extract_start.elapsed();

    // Flatten and count
    let all_phrases: Vec<_> = all_phrases.into_iter().filter_map(|p| p).collect();
    let total_phrases = all_phrases.len();
    let error_count = errors.load(Ordering::Relaxed);

    println!(
        "\nExtracted {} features from {} files in {:.1}s",
        total_phrases,
        total_files - error_count,
        extract_time.as_secs_f64()
    );
    println!("  ({} files had errors)", error_count);
    println!();

    if total_phrases == 0 {
        println!("No features extracted. Exiting.");
        return Ok(());
    }

    // =========================================================================
    // STEP 3: Build Semantic Dictionary via Clustering
    // =========================================================================
    println!("[3/4] Building semantic phrase dictionary...");

    let dict_start = Instant::now();

    // Create similarity engine with marmoset-specific weights
    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    let marmoset_weights = FeatureWeights::marmoset();
    engine.set_feature_weights(&marmoset_weights.to_weight_vector());

    // Fit normalization on sample
    {
        let n_fit = total_phrases.min(500);
        let mut matrix = ndarray::Array2::<f64>::zeros((n_fit, FEATURE_DIM));
        for (i, phrase) in all_phrases.iter().take(n_fit).enumerate() {
            for (j, &f) in phrase.features.iter().enumerate() {
                if j < FEATURE_DIM {
                    matrix[[i, j]] = f as f64;
                }
            }
        }
        engine.fit_normalization(&matrix);
    }

    // Simple clustering using similarity
    let similarity_threshold = 0.85;
    println!(
        "  Clustering with similarity threshold {:.0}%...",
        similarity_threshold * 100.0
    );

    let mut phrase_types: Vec<Vec<usize>> = Vec::new();
    let mut type_centroids: Vec<Vec<f32>> = Vec::new();
    let mut assignments = vec![0usize; total_phrases];

    for (phrase_idx, phrase) in all_phrases.iter().enumerate() {
        let query = ndarray::Array1::from_vec(phrase.features.iter().map(|&f| f as f64).collect());

        // Find best matching type
        let mut best_type = None;
        let mut best_sim = 0.0;

        for (type_idx, centroid) in type_centroids.iter().enumerate() {
            let proto = ndarray::Array1::from_vec(centroid.iter().map(|&f| f as f64).collect());
            let sim = 1.0 - engine.distance(&query, &proto);

            if sim > similarity_threshold && sim > best_sim {
                best_sim = sim;
                best_type = Some(type_idx);
            }
        }

        if let Some(type_idx) = best_type {
            assignments[phrase_idx] = type_idx;
            phrase_types[type_idx].push(phrase_idx);
        } else {
            let new_type_idx = phrase_types.len();
            assignments[phrase_idx] = new_type_idx;
            phrase_types.push(vec![phrase_idx]);
            type_centroids.push(phrase.features.clone());
        }
    }

    let num_types = phrase_types.len();
    println!(
        "  Found {} phrase types from {} phrases",
        num_types, total_phrases
    );

    // Build semantic dictionary
    let mut type_to_labels: HashMap<String, HashMap<String, usize>> = HashMap::new();

    for (phrase_idx, type_idx) in assignments.iter().enumerate() {
        let phrase = &all_phrases[phrase_idx];
        let type_id = format!("Type_{}", type_idx);

        *type_to_labels
            .entry(type_id.clone())
            .or_default()
            .entry(phrase.label.clone())
            .or_insert(0) += 1;
    }

    // Convert to probabilities
    let mut type_to_label_probs: HashMap<String, HashMap<String, f32>> = HashMap::new();
    for (type_id, label_counts) in &type_to_labels {
        let total: usize = label_counts.values().sum();
        if total > 0 {
            let probs: HashMap<String, f32> = label_counts
                .iter()
                .map(|(label, &count)| (label.clone(), count as f32 / total as f32))
                .collect();
            type_to_label_probs.insert(type_id.clone(), probs);
        }
    }

    let dict_time = dict_start.elapsed();
    println!("  Built dictionary in {:.1}s", dict_time.as_secs_f64());
    println!();

    // =========================================================================
    // STEP 4: Display Results
    // =========================================================================
    println!("[4/4] Results:");
    println!();

    // Sort types by occurrence count
    let mut type_sizes: Vec<_> = phrase_types
        .iter()
        .enumerate()
        .map(|(i, members)| (i, members.len()))
        .collect();
    type_sizes.sort_by_key(|(_, size)| std::cmp::Reverse(*size));

    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!("│ TOP 15 SEMANTIC PHRASE TYPES                                                 │");
    println!("├─────────────────────────────────────────────────────────────────────────────┤");

    for (type_idx, size) in type_sizes.iter().take(15) {
        let type_id = format!("Type_{}", type_idx);
        if let Some(label_probs) = type_to_label_probs.get(&type_id) {
            let mut labels: Vec<_> = label_probs.iter().collect();
            labels.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

            let primary = labels
                .first()
                .map(|(l, p)| format!("{} ({:.0}%)", l, **p * 100.0))
                .unwrap_or_else(|| "Unknown".to_string());

            println!("│ {:<12} {:>5} occurrences  {}", type_id, size, primary);
        }
    }
    println!("└─────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // =========================================================================
    // Save Dictionary
    // =========================================================================
    let output_dir = "marmoset_guided_results";
    std::fs::create_dir_all(output_dir)?;

    // Save semantic dictionary
    let dict_path = format!("{}/marmoset_semantic_dictionary.json", output_dir);
    let dict_json = serde_json::to_string_pretty(&type_to_label_probs)?;
    let mut file = File::create(&dict_path)?;
    file.write_all(dict_json.as_bytes())?;
    println!("Saved: {}", dict_path);

    // Save type centroids for propagation
    let centroids_path = format!("{}/marmoset_type_centroids.json", output_dir);
    let centroids_data: HashMap<String, Vec<f32>> = type_centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (format!("Type_{}", i), c.clone()))
        .collect();
    let centroids_json = serde_json::to_string_pretty(&centroids_data)?;
    let mut file = File::create(&centroids_path)?;
    file.write_all(centroids_json.as_bytes())?;
    println!("Saved: {}", centroids_path);

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("Files processed: {}", total_files - error_count);
    println!("Features extracted: {}", total_phrases);
    println!("Phrase types: {}", num_types);
    println!("Total time: {:.1}s", total_start.elapsed().as_secs_f64());
    println!();
    println!("The semantic dictionary can now be used for:");
    println!("  • Propagating labels to unlabeled marmoset recordings");
    println!("  • Classifying new recordings in real-time");
    println!("  • Analyzing context-specific vocalization patterns");

    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

struct AudioData {
    samples: Vec<f64>,
    sample_rate: u32,
}

fn load_audio_file(path: &Path) -> Result<AudioData, Box<dyn std::error::Error>> {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match extension {
        "wav" => load_wav_file(path),
        "flac" => load_flac_file(path),
        _ => Err(format!("Unsupported format: {}", extension).into()),
    }
}

fn load_wav_file(path: &Path) -> Result<AudioData, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    std::io::Read::read_to_end(&mut file, &mut buffer)?;

    if buffer.len() < 44 {
        return Err("File too small".into());
    }

    // Check RIFF header
    if &buffer[0..4] != b"RIFF" {
        return Err("Not a valid WAV file".into());
    }

    // Parse WAV header
    let sample_rate = u32::from_le_bytes([buffer[24], buffer[25], buffer[26], buffer[27]]);
    let bits_per_sample = u16::from_le_bytes([buffer[34], buffer[35]]) as usize;
    let data_start = 44;

    let samples: Vec<f64> = if bits_per_sample == 16 {
        buffer[data_start..]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64 / 32768.0)
            .collect()
    } else if bits_per_sample == 32 {
        buffer[data_start..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect()
    } else {
        return Err(format!("Unsupported bits per sample: {}", bits_per_sample).into());
    };

    Ok(AudioData {
        samples,
        sample_rate,
    })
}

fn load_flac_file(path: &Path) -> Result<AudioData, Box<dyn std::error::Error>> {
    // Use symphonia for FLAC decoding
    use symphonia::core::audio::AudioBufferRef;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("flac");

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed =
        symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;
    let mut format = probed.format;

    let track = format.default_track().ok_or("No default track")?;
    let sample_rate = track.codec_params.sample_rate.ok_or("No sample rate")?;

    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &decoder_opts)?;

    let mut all_samples = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                // Convert audio buffer to interleaved f32 samples
                match audio_buf {
                    AudioBufferRef::F32(buf) => {
                        for plane in buf.planes().planes() {
                            for sample in plane.iter() {
                                all_samples.push(*sample as f64);
                            }
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        for plane in buf.planes().planes() {
                            for sample in plane.iter() {
                                all_samples.push(*sample as f64 / 32768.0);
                            }
                        }
                    }
                    AudioBufferRef::S24(buf) => {
                        for plane in buf.planes().planes() {
                            for sample in plane.iter() {
                                // i24 is a newtype wrapper around i32
                                all_samples.push(sample.inner() as f64 / 8388608.0);
                            }
                        }
                    }
                    AudioBufferRef::S32(buf) => {
                        for plane in buf.planes().planes() {
                            for sample in plane.iter() {
                                all_samples.push(*sample as f64 / 2147483648.0);
                            }
                        }
                    }
                    AudioBufferRef::F64(buf) => {
                        for plane in buf.planes().planes() {
                            for sample in plane.iter() {
                                all_samples.push(*sample);
                            }
                        }
                    }
                    _ => {
                        // For other formats, skip
                        continue;
                    }
                }
            }
            Err(_) => break,
        }
    }

    Ok(AudioData {
        samples: all_samples,
        sample_rate,
    })
}
