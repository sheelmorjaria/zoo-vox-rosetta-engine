//! Human-Guided Context Discovery Pipeline
//!
//! Applies the "Anchor and Propagate" workflow to datasets with file-level annotations.
//!
//! Datasets:
//! - Egyptian Fruit Bats: /mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/
//! - Marmoset Vocalizations: ~/birdsong_analysis/data/Vocalizations/

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use technical_architecture::{
    species::FeatureWeights, AcousticSimilarityEngine, SimilarityMetric, ZooVoxFeatureExtractor,
};

const FEATURE_DIM: usize = 45;
const MAX_FILES: usize = 2000; // Process first 2000 files

// =============================================================================
// Egyptian Fruit Bat Context Codes
// =============================================================================

fn get_bat_context_name(code: i32) -> &'static str {
    match code {
        1 => "Sleeping",
        2 => "Resting",
        3 => "Fighting",
        4 => "Grooming",
        5 => "Suckling",
        6 => "Protest",
        7 => "Isolation",
        8 => "Kissing",
        9 => "Mating",
        10 => "Landing",
        11 => "Unknown",
        12 => "Background",
        _ => "Unknown",
    }
}

// =============================================================================
// Annotation Parsers
// =============================================================================

#[derive(Debug, Clone, Deserialize)]
struct BatAnnotation {
    #[serde(rename = "Emitter")]
    emitter: i32,
    #[serde(rename = "Addressee")]
    addressee: i32,
    #[serde(rename = "Context")]
    context: i32,
    #[serde(rename = "File Name")]
    filename: String,
}

// =============================================================================
// Discovered Phrase
// =============================================================================

#[derive(Debug, Clone)]
struct DiscoveredPhrase {
    file_idx: usize,
    features: Vec<f32>,
    label: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║        Human-Guided Context Discovery Pipeline                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();

    // =========================================================================
    // STEP 1: Load Annotations
    // =========================================================================
    println!("[1/4] Loading annotations...");

    let bat_data_path = "/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats";
    let bat_audio_path = format!("{}/audio", bat_data_path);
    let bat_ann_path = format!("{}/annotations.csv", bat_data_path);

    let annotations = load_bat_annotations(&bat_ann_path)?;
    let total_annotations = annotations.len();
    let process_count = total_annotations.min(MAX_FILES);
    println!(
        "Loaded {} bat annotations, processing first {}",
        total_annotations, process_count
    );

    // Show context distribution
    let mut context_counts: HashMap<i32, usize> = HashMap::new();
    for ann in &annotations {
        *context_counts.entry(ann.context).or_insert(0) += 1;
    }
    println!("\nContext distribution:");
    let mut contexts: Vec<_> = context_counts.iter().collect();
    contexts.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (ctx, count) in contexts.iter().take(10) {
        println!("  {} ({}): {} files", get_bat_context_name(**ctx), ctx, count);
    }
    println!();

    // =========================================================================
    // STEP 2: Feature Extraction (Whole File)
    // =========================================================================
    println!("[2/4] Extracting 45D features from {} files...", process_count);

    let extract_start = Instant::now();
    let processed = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));

    let annotations_subset: Vec<_> = annotations.into_iter().take(MAX_FILES).collect();

    // Parallel processing
    let all_phrases: Vec<Option<DiscoveredPhrase>> = annotations_subset
        .par_iter()
        .map(|ann| {
            let count = processed.fetch_add(1, Ordering::Relaxed);
            if count % 500 == 0 {
                println!("  Progress: {}/{}", count, process_count);
            }

            // Load audio
            let audio_path = format!("{}/{}", bat_audio_path, ann.filename);
            let audio = match load_wav_f32(&audio_path) {
                Ok(a) => a,
                Err(_) => {
                    errors.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            };

            if audio.samples.len() < 100 {
                return None;
            }

            // Extract 45D features for entire file
            let mut extractor = ZooVoxFeatureExtractor::new(audio.sample_rate);
            match extractor.extract_45d(&audio.samples) {
                Ok(features) => Some(DiscoveredPhrase {
                    file_idx: count,
                    features: features.to_vector().iter().map(|&f| f as f32).collect(),
                    label: get_bat_context_name(ann.context).to_string(),
                }),
                Err(_) => None,
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
        process_count - error_count,
        extract_time.as_secs_f64()
    );
    println!("  ({}) files had errors)", error_count);
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

    // Create similarity engine with bat-specific weights
    let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, SimilarityMetric::Cosine);
    let bat_weights = FeatureWeights::bat();
    engine.set_feature_weights(&bat_weights.to_weight_vector());

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
    println!("  Found {} phrase types from {} phrases", num_types, total_phrases);

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
    let output_dir = "human_guided_results";
    std::fs::create_dir_all(output_dir)?;

    // Save semantic dictionary
    let dict_path = format!("{}/bat_semantic_dictionary.json", output_dir);
    let dict_json = serde_json::to_string_pretty(&type_to_label_probs)?;
    let mut file = File::create(&dict_path)?;
    file.write_all(dict_json.as_bytes())?;
    println!("Saved: {}", dict_path);

    // Save type centroids for propagation
    let centroids_path = format!("{}/bat_type_centroids.json", output_dir);
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
    println!("Files processed: {}", process_count - error_count);
    println!("Features extracted: {}", total_phrases);
    println!("Phrase types: {}", num_types);
    println!("Total time: {:.1}s", total_start.elapsed().as_secs_f64());
    println!();
    println!("The semantic dictionary can now be used for:");
    println!("  • Propagating labels to unlabeled bat recordings");
    println!("  • Classifying new recordings in real-time");
    println!("  • Analyzing context-specific vocalization patterns");

    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

fn load_bat_annotations(path: &str) -> Result<Vec<BatAnnotation>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = csv::Reader::from_reader(BufReader::new(file));

    let mut annotations = Vec::new();
    for result in reader.deserialize() {
        let ann: BatAnnotation = result?;
        annotations.push(ann);
    }

    Ok(annotations)
}

struct AudioData {
    samples: Vec<f64>,
    sample_rate: u32,
}

fn load_wav_f32(path: &str) -> Result<AudioData, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    if buffer.len() < 44 {
        return Err("File too small".into());
    }

    // Check RIFF header
    if &buffer[0..4] != b"RIFF" {
        // Try raw float format
        let samples: Vec<f64> = buffer
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect();
        return Ok(AudioData {
            samples,
            sample_rate: 250000, // Default bat sample rate
        });
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

    Ok(AudioData { samples, sample_rate })
}
