// Recursive Feature Elimination (RFE) Analysis: Marmoset Vocalizations
// ===================================================================
//
// This example identifies the most discriminative 30D MicroDynamics features
// for distinguishing between marmoset call types using RFE.
//
// Features analyzed (30D):
// 1. attack_time_ms      11. mfcc[0]         21. mfcc[10]
// 2. decay_time_ms       12. mfcc[1]         22. mfcc[11]
// 3. sustain_level       13. mfcc[2]         23. mfcc[12]
// 4. vibrato_rate_hz     14. mfcc[3]         24. spectral_flux
// 5. vibrato_depth       15. mfcc[4]         25. median_ici_ms
// 6. jitter              16. mfcc[5]         26. onset_rate_hz
// 7. shimmer             17. mfcc[6]         27. ici_cv
// 8. harmonicity         18. mfcc[7]         28. duration_ms
// 9. spectral_flatness   19. mfcc[8]         29. rms_energy
// 10. hnr                20. mfcc[9]

use std::collections::HashMap;
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use technical_architecture::micro_dynamics_extractor::MicroDynamicsExtractor;

/// Feature names for 29D MicroDynamics (excluding vibrato_depth which is calculated separately)
const FEATURE_NAMES: &[&str] = &[
    "attack_time_ms",    // 0
    "decay_time_ms",     // 1
    "sustain_level",     // 2
    "vibrato_rate_hz",   // 3
    "jitter",            // 4
    "shimmer",           // 5
    "harmonicity",       // 6
    "spectral_flatness", // 7
    "hnr",               // 8
    "mfcc_0",            // 9
    "mfcc_1",            // 10
    "mfcc_2",            // 11
    "mfcc_3",            // 12
    "mfcc_4",            // 13
    "mfcc_5",            // 14
    "mfcc_6",            // 15
    "mfcc_7",            // 16
    "mfcc_8",            // 17
    "mfcc_9",            // 18
    "mfcc_10",           // 19
    "mfcc_11",           // 20
    "mfcc_12",           // 21
    "spectral_flux",     // 22
    "median_ici_ms",     // 23
    "onset_rate_hz",     // 24
    "ici_cv",            // 25
    "duration_ms",       // 26
    "rms_energy",        // 27
    "vibrato_depth",     // 28 (extracted but may be 0.0 for some calls)
];

/// Call type label
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CallType {
    Phee,
    Twitter,
    Trill,
    Tsik,
    Seep,
    Infant,
    Vocalization,
}

impl CallType {
    fn from_filename(filename: &str) -> Option<Self> {
        if filename.starts_with("Phee") {
            Some(CallType::Phee)
        } else if filename.starts_with("Twitter") {
            Some(CallType::Twitter)
        } else if filename.starts_with("Trill") {
            Some(CallType::Trill)
        } else if filename.starts_with("Tsik") {
            Some(CallType::Tsik)
        } else if filename.starts_with("Seep") {
            Some(CallType::Seep)
        } else if filename.starts_with("Infant") {
            Some(CallType::Infant)
        } else if filename.starts_with("Vocalization") {
            Some(CallType::Vocalization)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        match self {
            CallType::Phee => "Phee",
            CallType::Twitter => "Twitter",
            CallType::Trill => "Trill",
            CallType::Tsik => "Tsik",
            CallType::Seep => "Seep",
            CallType::Infant => "Infant_cry",
            CallType::Vocalization => "Vocalization",
        }
    }
}

/// Labeled feature vector
#[derive(Debug, Clone)]
struct LabeledFeatures {
    features: Vec<f32>, // 30D
    label: CallType,
    file_id: String,
}

/// Load a single FLAC file and return audio samples
fn load_flac_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("flac");

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("No valid audio track found")?;

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let n_channels = decoder.codec_params().channels.map_or(1, |ch| ch.count());

    let mut audio_samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };

        let decoded = decoder.decode(&packet)?;

        match decoded {
            AudioBufferRef::F64(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32));
                }
            }
            AudioBufferRef::F32(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend_from_slice(samples);
                }
            }
            AudioBufferRef::S32(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i32::MAX as f32));
                }
            }
            AudioBufferRef::S24(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(
                        samples
                            .iter()
                            .map(|&s| s.inner() as f32 / (i32::MAX >> 8) as f32),
                    );
                }
            }
            AudioBufferRef::S16(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i16::MAX as f32));
                }
            }
            AudioBufferRef::S8(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i8::MAX as f32));
                }
            }
            AudioBufferRef::U8(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| (s as f32 - 128.0) / 128.0));
                }
            }
            AudioBufferRef::U16(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| (s as f32 - 32768.0) / 32768.0));
                }
            }
            AudioBufferRef::U24(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(
                        samples
                            .iter()
                            .map(|&s| (s.inner() as f32 - 8388608.0) / 8388608.0),
                    );
                }
            }
            AudioBufferRef::U32(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(
                        samples
                            .iter()
                            .map(|&s| (s as f32 - 2147483648.0) / 2147483648.0),
                    );
                }
            }
        }
    }

    Ok(audio_samples)
}

/// Extract 29D features from audio
fn extract_30d_features(
    audio: &[f32],
    sample_rate: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let extractor = MicroDynamicsExtractor::new(sample_rate);
    let features = extractor.extract(audio)?;

    Ok(vec![
        features.attack_time_ms,          // 0
        features.decay_time_ms,           // 1
        features.sustain_level,           // 2
        features.vibrato_rate_hz,         // 3
        features.jitter,                  // 4
        features.shimmer,                 // 5
        features.harmonicity,             // 6
        features.spectral_flatness,       // 7
        features.harmonic_to_noise_ratio, // 8
        // 13 MFCC coefficients
        features.mfcc[0],                      // 9
        features.mfcc[1],                      // 10
        features.mfcc[2],                      // 11
        features.mfcc[3],                      // 12
        features.mfcc[4],                      // 13
        features.mfcc[5],                      // 14
        features.mfcc[6],                      // 15
        features.mfcc[7],                      // 16
        features.mfcc[8],                      // 17
        features.mfcc[9],                      // 18
        features.mfcc[10],                     // 19
        features.mfcc[11],                     // 20
        features.mfcc[12],                     // 21
        features.spectral_flux,                // 22
        features.median_ici_ms,                // 23
        features.onset_rate_hz,                // 24
        features.ici_coefficient_of_variation, // 25
        features.vibrato_depth,                // 26
        // Duration and energy
        (audio.len() as f32 / sample_rate as f32) * 1000.0, // 27
        (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32).sqrt(), // 28
    ])
}

/// Normalize features to zero mean and unit variance
fn normalize_features(data: &mut [LabeledFeatures]) {
    let n_features = 29; // Updated to match actual feature count
    let n_samples = data.len();

    // Compute mean
    let mut mean = vec![0.0f32; n_features];
    for sample in data.iter() {
        for (i, &val) in sample.features.iter().enumerate() {
            mean[i] += val;
        }
    }
    for val in mean.iter_mut() {
        *val /= n_samples as f32;
    }

    // Compute std
    let mut std = vec![0.0f32; n_features];
    for sample in data.iter() {
        for (i, &val) in sample.features.iter().enumerate() {
            std[i] += (val - mean[i]).powi(2);
        }
    }
    for val in std.iter_mut() {
        *val = (*val / n_samples as f32).sqrt();
        if *val < 1e-6 {
            *val = 1.0; // Avoid division by zero
        }
    }

    // Normalize
    for sample in data.iter_mut() {
        for (i, val) in sample.features.iter_mut().enumerate() {
            *val = (*val - mean[i]) / std[i];
        }
    }
}

/// Compute Fisher score for a single feature
/// Higher score = more discriminative
fn fisher_score(data: &[LabeledFeatures], feature_idx: usize) -> f64 {
    let mut class_stats: HashMap<CallType, (Vec<f64>, f64)> = HashMap::new();

    // Organize data by class
    for sample in data {
        let val = sample.features[feature_idx] as f64;
        class_stats
            .entry(sample.label)
            .or_insert_with(|| (Vec::new(), 0.0))
            .0
            .push(val);
    }

    if class_stats.len() < 2 {
        return 0.0;
    }

    // Compute class means and overall mean
    let mut class_means: Vec<f64> = Vec::new();
    let mut class_sizes: Vec<usize> = Vec::new();
    let mut overall_mean = 0.0;
    let mut total_samples = 0;

    for (values, _) in class_stats.values() {
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        class_means.push(mean);
        class_sizes.push(values.len());
        overall_mean += mean * values.len() as f64;
        total_samples += values.len();
    }
    overall_mean /= total_samples as f64;

    // Compute between-class variance
    let mut between_var = 0.0;
    for (i, &mean) in class_means.iter().enumerate() {
        between_var += class_sizes[i] as f64 * (mean - overall_mean).powi(2);
    }

    // Compute within-class variance
    let mut within_var = 0.0;
    for (values, _) in class_stats.values() {
        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        within_var += values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>();
    }

    if within_var < 1e-10 {
        return 0.0;
    }

    between_var / within_var
}

/// Recursive Feature Elimination
fn rfe_analysis(data: &[LabeledFeatures], n_features_to_select: usize) -> Vec<(usize, f64)> {
    let mut remaining_features: Vec<usize> = (0..29).collect(); // Updated to 29
    let mut rankings: Vec<(usize, f64)> = Vec::new();

    println!("RFE Progress:");
    println!("==============");

    while remaining_features.len() > n_features_to_select {
        // Compute Fisher scores for remaining features
        let mut scores: Vec<(usize, f64)> = remaining_features
            .iter()
            .map(|&idx| (idx, fisher_score(data, idx)))
            .collect();

        // Sort by score (ascending - lowest score gets eliminated first)
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Eliminate the feature with lowest score
        let (worst_idx, worst_score) = scores[0];
        rankings.push((worst_idx, worst_score));

        // Remove from remaining features
        remaining_features.retain(|&x| x != worst_idx);

        println!(
            "  Eliminated: {:20} (Score: {:.6}) | Remaining: {}",
            FEATURE_NAMES[worst_idx],
            worst_score,
            remaining_features.len()
        );
    }

    // Add final remaining features with their scores
    for &idx in &remaining_features {
        let score = fisher_score(data, idx);
        rankings.push((idx, score));
    }

    // Sort rankings by score (descending - most important first)
    rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    rankings
}

/// Load and extract features from marmoset dataset
fn load_marmoset_dataset(
    vocalizations_dir: &Path,
    samples_per_class: usize,
) -> Result<Vec<LabeledFeatures>, Box<dyn std::error::Error>> {
    let mut all_data: Vec<LabeledFeatures> = Vec::new();

    let entries = std::fs::read_dir(vocalizations_dir)?;
    let mut class_counts: HashMap<CallType, usize> = HashMap::new();

    for entry in entries {
        let entry = entry?;
        let dir_path = entry.path();

        if !dir_path.is_dir() {
            continue;
        }

        let file_entries = std::fs::read_dir(&dir_path)?;
        for file_entry in file_entries {
            let file_entry = file_entry?;
            let file_path = file_entry.path();

            if !file_path.is_file() {
                continue;
            }

            let filename = file_path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            if !filename.ends_with(".flac") {
                continue;
            }

            if let Some(call_type) = CallType::from_filename(filename) {
                let count = *class_counts.get(&call_type).unwrap_or(&0);
                if count >= samples_per_class {
                    continue;
                }

                match load_flac_file(&file_path) {
                    Ok(audio) => match extract_30d_features(&audio, 96000) {
                        Ok(features) => {
                            let file_id = filename.to_string();
                            all_data.push(LabeledFeatures {
                                features,
                                label: call_type,
                                file_id,
                            });
                            *class_counts.entry(call_type).or_insert(0) += 1;
                        }
                        Err(_) => continue,
                    },
                    Err(_) => continue,
                }
            }
        }

        // Check if we have enough samples
        let min_count = *class_counts.values().min().unwrap_or(&0);
        if min_count >= samples_per_class {
            break;
        }
    }

    println!(
        "Loaded {} samples across {} call types",
        all_data.len(),
        class_counts.len()
    );

    for (call_type, count) in &class_counts {
        println!("  {:20}: {} samples", call_type.name(), count);
    }

    Ok(all_data)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║   Recursive Feature Elimination (RFE): Marmoset Vocalizations             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Identifying the most discriminative MicroDynamics features");
    println!("for distinguishing between marmoset call types.");
    println!();

    let vocalizations_dir = Path::new("/home/sheel/birdsong_analysis/data/Vocalizations");

    if !vocalizations_dir.exists() {
        println!("❌ Directory not found: {}", vocalizations_dir.display());
        return Err("Dataset directory not found".into());
    }

    // Load dataset with balanced samples per class
    println!("Loading marmoset dataset (balanced samples)...");
    println!("---");
    let samples_per_class = 200;
    let mut data = load_marmoset_dataset(vocalizations_dir, samples_per_class)?;

    if data.is_empty() {
        return Err("No data loaded".into());
    }

    println!();
    println!("Normalizing features...");
    normalize_features(&mut data);

    println!();
    println!("Running Recursive Feature Elimination (RFE)...");
    println!("Goal: Select top 15 most discriminative features");
    println!();

    let rankings = rfe_analysis(&data, 15);

    println!();
    println!("========================================================================");
    println!("RFE RESULTS: Top 15 Most Discriminative Features");
    println!("========================================================================");
    println!();

    println!(
        "{:<3} {:<20} {:>15} {:>20}",
        "Rank", "Feature", "Fisher Score", "Interpretation"
    );
    println!("{}", "-".repeat(70));

    let interpretations = [
        "Temporal envelope shape",
        "Modulation rate",
        "Frequency modulation depth",
        "Micro-variation (pitch)",
        "Micro-variation (amplitude)",
        "Harmonic structure",
        "Spectral envelope (MFCC)",
        "Rhythm timing",
        "Spectral noisiness",
        "Energy/loudness",
    ];

    for (i, (feat_idx, score)) in rankings.iter().take(15).enumerate() {
        let interpretation = if *feat_idx < 3 {
            interpretations[0]
        } else if *feat_idx == 3 {
            interpretations[1]
        } else if *feat_idx == 4 {
            interpretations[2]
        } else if *feat_idx == 5 {
            interpretations[3]
        } else if *feat_idx == 6 {
            interpretations[4]
        } else if *feat_idx < 10 {
            interpretations[5]
        } else if (10..23).contains(feat_idx) {
            interpretations[6]
        } else if *feat_idx == 24 || *feat_idx == 25 || *feat_idx == 26 {
            interpretations[7]
        } else if *feat_idx == 8 {
            interpretations[8]
        } else {
            interpretations[9]
        };

        println!(
            "{:<3} {:<20} {:>15.6}  {}",
            i + 1,
            FEATURE_NAMES[*feat_idx],
            score,
            interpretation
        );
    }

    println!();
    println!("========================================================================");
    println!("FEATURE CATEGORY ANALYSIS");
    println!("========================================================================");
    println!();

    // Group by category
    let mut category_scores: HashMap<&str, Vec<f64>> = HashMap::new();

    for (feat_idx, score) in &rankings {
        let category = if *feat_idx < 3 {
            "Temporal Envelope"
        } else if *feat_idx == 3 || *feat_idx == 4 {
            "Modulation"
        } else if *feat_idx == 5 || *feat_idx == 6 {
            "Perturbation"
        } else if *feat_idx == 7 || *feat_idx == 8 || *feat_idx == 9 {
            "Timbre/Harmonics"
        } else if (10..23).contains(feat_idx) {
            "MFCC (Spectral Envelope)"
        } else if *feat_idx == 24 {
            "Rhythm (ICI)"
        } else if *feat_idx == 25 || *feat_idx == 26 {
            "Rhythm (Onset)"
        } else {
            "Energy"
        };

        category_scores
            .entry(category)
            .or_insert_with(Vec::new)
            .push(*score);
    }

    println!(
        "{:<25} {:>15} {:>15}",
        "Category", "Avg Fisher Score", "Rank"
    );
    println!("{}", "-".repeat(55));

    let mut category_rankings: Vec<(&str, f64)> = category_scores
        .iter()
        .map(|(cat, scores)| {
            let avg_score: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
            (*cat, avg_score)
        })
        .collect();

    category_rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (category, avg_score)) in category_rankings.iter().enumerate() {
        println!("{:<25} {:>15.6}         {}", category, avg_score, i + 1);
    }

    println!();
    println!("========================================================================");
    println!("SCIENTIFIC INTERPRETATION");
    println!("========================================================================");
    println!();

    let top_features: Vec<_> = rankings.iter().take(5).collect();

    println!("Most Discriminative Features for Marmoset Call Types:");
    println!("=======================================================");
    for (i, (feat_idx, score)) in top_features.iter().enumerate() {
        println!(
            "  {}. {} (Fisher Score: {:.6})",
            i + 1,
            FEATURE_NAMES[*feat_idx],
            score
        );
    }

    println!();
    println!("Key Insights:");
    println!("===============");
    println!("• High Fisher score indicates strong between-class variance");
    println!("• Features with high scores are most useful for call type classification");
    println!("• MFCC features (10-22) typically capture spectral envelope shape");
    println!("• Temporal features (0-2) capture attack/decay/sustain characteristics");

    Ok(())
}
