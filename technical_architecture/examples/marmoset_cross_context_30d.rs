// Cross-Context Syntactic Analysis: Marmoset Vocalizations (30D Features)
// ====================================================================
//
// This example analyzes how syntactic structure varies across different
// call types using 30D MicroDynamics features for phrase similarity.
//
// Research Question: Does marmoset vocalization syntax change with call type?
//
// Uses 30D MicroDynamics features:
// - Temporal: attack_time_ms, decay_time_ms, sustain_level
// - Modulation: vibrato_rate_hz, vibrato_depth
// - Perturbation: jitter, shimmer
// - Timbre: harmonicity, spectral_flatness, harmonic_to_noise_ratio
// - Spectral: 13×MFCC, spectral_flux
// - Rhythm: median_ici_ms, onset_rate_hz, ici_coefficient_of_variation

use std::collections::HashMap;
use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use technical_architecture::micro_dynamics_extractor::MicroDynamicsExtractor;

/// Call type extracted from filename
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
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

/// 30D phrase features for similarity comparison
#[derive(Debug, Clone)]
struct PhraseFeatures30D {
    phrase_id: String,
    features: Vec<f32>, // 30D feature vector
}

/// Word type discovered from clustering
#[derive(Debug, Clone)]
struct WordType30D {
    word_id: usize,
    representative_features: Vec<f32>,
    member_phrases: Vec<String>,
}

/// Word sequence for PMI calculation
#[derive(Debug, Clone)]
struct WordSequence30D {
    words: Vec<usize>, // Word IDs
    phrase_id: String,
}

/// Cross-context analysis results
#[derive(Debug, Clone)]
struct ContextAnalysis30D {
    context_id: String,
    num_vocalizations: usize,
    total_phrases: usize,
    vocabulary_size: usize,
    avg_sequence_length: f64,
    avg_pmi: f64,
    max_pmi: f64,
}

/// Simple 30D phrase extractor with fixed segmentation
fn extract_phrases_30d(
    audio: &[f32],
    sample_rate: u32,
    phrase_id: &str,
) -> Result<Vec<PhraseFeatures30D>, Box<dyn std::error::Error>> {
    let extractor = MicroDynamicsExtractor::new(sample_rate);

    // Fixed phrase segmentation: divide into 50ms phrases
    let phrase_duration_ms = 50.0;
    let samples_per_phrase = (sample_rate as f32 * phrase_duration_ms / 1000.0) as usize;

    let mut phrases = Vec::new();
    let mut phrase_idx = 0;

    for chunk in audio.chunks(samples_per_phrase) {
        if chunk.len() < samples_per_phrase / 2 {
            continue; // Skip too short chunks
        }

        // Extract 30D features
        let features = extractor.extract(chunk)?;

        // Convert to 30D vector
        let feature_vec = vec![
            features.attack_time_ms,
            features.decay_time_ms,
            features.sustain_level,
            features.vibrato_rate_hz,
            features.vibrato_depth,
            features.jitter,
            features.shimmer,
            features.harmonicity,
            features.spectral_flatness,
            features.harmonic_to_noise_ratio,
            // 13 MFCC coefficients
            features.mfcc[0],
            features.mfcc[1],
            features.mfcc[2],
            features.mfcc[3],
            features.mfcc[4],
            features.mfcc[5],
            features.mfcc[6],
            features.mfcc[7],
            features.mfcc[8],
            features.mfcc[9],
            features.mfcc[10],
            features.mfcc[11],
            features.mfcc[12],
            features.spectral_flux,
            features.median_ici_ms,
            features.onset_rate_hz,
            features.ici_coefficient_of_variation,
            // Duration (computed from chunk size)
            (chunk.len() as f32 / sample_rate as f32) * 1000.0,
            // Energy (RMS)
            (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32).sqrt(),
        ];

        phrases.push(PhraseFeatures30D {
            phrase_id: format!("{}_{}", phrase_id, phrase_idx),
            features: feature_vec,
        });

        phrase_idx += 1;
    }

    Ok(phrases)
}

/// Discover vocabulary by clustering 30D features
fn discover_vocabulary_30d(
    all_phrases: &[Vec<PhraseFeatures30D>],
    similarity_threshold: f32,
) -> Result<Vec<WordType30D>, Box<dyn std::error::Error>> {
    let mut all_features: Vec<(Vec<f32>, String)> = Vec::new();

    // Collect all phrases
    for phrases in all_phrases {
        for phrase in phrases {
            all_features.push((phrase.features.clone(), phrase.phrase_id.clone()));
        }
    }

    if all_features.is_empty() {
        return Ok(Vec::new());
    }

    // Simple clustering: group by cosine similarity
    let mut clusters: Vec<Vec<(Vec<f32>, String)>> = Vec::new();

    for (features, phrase_id) in all_features {
        let mut assigned = false;

        for cluster in &mut clusters {
            if let Some((rep, _)) = cluster.first() {
                // Compute cosine similarity
                let dot_product: f32 = features.iter().zip(rep.iter()).map(|(a, b)| a * b).sum();
                let norm_a: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = rep.iter().map(|x| x * x).sum::<f32>().sqrt();

                if norm_a > 0.0 && norm_b > 0.0 {
                    let similarity = dot_product / (norm_a * norm_b);
                    if similarity > similarity_threshold {
                        cluster.push((features.clone(), phrase_id.clone()));
                        assigned = true;
                        break;
                    }
                }
            }
        }

        if !assigned {
            clusters.push(vec![(features, phrase_id)]);
        }
    }

    // Convert to word types
    let mut words = Vec::new();
    for (word_id, cluster) in clusters.iter().enumerate() {
        // Compute centroid
        let n_features = cluster[0].0.len();
        let mut centroid = vec![0.0f32; n_features];

        for (features, _) in cluster {
            for (i, &val) in features.iter().enumerate() {
                centroid[i] += val;
            }
        }

        for val in centroid.iter_mut() {
            *val /= cluster.len() as f32;
        }

        let member_phrases: Vec<String> = cluster.iter().map(|(_, id)| id.clone()).collect();

        words.push(WordType30D {
            word_id,
            representative_features: centroid,
            member_phrases,
        });
    }

    Ok(words)
}

/// Extract word sequences from phrases
fn extract_sequences_30d(
    all_phrases: Vec<Vec<PhraseFeatures30D>>,
    word_types: &[WordType30D],
    similarity_threshold: f32,
) -> Result<Vec<WordSequence30D>, Box<dyn std::error::Error>> {
    let mut sequences = Vec::new();

    for phrases in &all_phrases {
        if phrases.is_empty() {
            continue;
        }

        let mut word_ids = Vec::new();

        for phrase in phrases {
            let mut assigned = false;

            // Find matching word type
            for word in word_types {
                let dot_product: f32 = phrase
                    .features
                    .iter()
                    .zip(word.representative_features.iter())
                    .map(|(a, b)| a * b)
                    .sum();

                let norm_a: f32 = phrase.features.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = word
                    .representative_features
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt();

                if norm_a > 0.0 && norm_b > 0.0 {
                    let similarity = dot_product / (norm_a * norm_b);
                    if similarity > similarity_threshold {
                        word_ids.push(word.word_id);
                        assigned = true;
                        break;
                    }
                }
            }

            if !assigned {
                // No match found - create new word on-the-fly
                word_ids.push(word_types.len() + word_ids.len());
            }
        }

        sequences.push(WordSequence30D {
            words: word_ids,
            phrase_id: phrases[0].phrase_id.clone(),
        });
    }

    Ok(sequences)
}

/// Calculate PMI from word sequences
fn calculate_pmi_30d(sequences: &[WordSequence30D]) -> (f64, f64) {
    if sequences.is_empty() {
        return (0.0, 0.0);
    }

    // Count word occurrences and transitions
    let mut word_counts: HashMap<usize, usize> = HashMap::new();
    let mut transition_counts: HashMap<(usize, usize), usize> = HashMap::new();
    let mut total_transitions = 0;

    for seq in sequences {
        for word in &seq.words {
            *word_counts.entry(*word).or_insert(0) += 1;
        }

        for i in 0..seq.words.len().saturating_sub(1) {
            let transition = (seq.words[i], seq.words[i + 1]);
            *transition_counts.entry(transition).or_insert(0) += 1;
            total_transitions += 1;
        }
    }

    if total_transitions == 0 || word_counts.is_empty() {
        return (0.0, 0.0);
    }

    // Calculate PMI for each transition
    let mut pmi_sum = 0.0;
    let mut pmi_count = 0;
    let mut max_pmi: f64 = 0.0;

    for ((w1, w2), &count) in &transition_counts {
        let p_w1 = *word_counts.get(w1).unwrap_or(&0) as f64 / sequences.len() as f64;
        let p_w2 = *word_counts.get(w2).unwrap_or(&0) as f64 / sequences.len() as f64;
        let p_w1w2 = count as f64 / total_transitions as f64;

        if p_w1 > 0.0 && p_w2 > 0.0 && p_w1w2 > 0.0 {
            let pmi = (p_w1w2 / (p_w1 * p_w2)).ln();
            pmi_sum += pmi * count as f64;
            pmi_count += count;
            max_pmi = max_pmi.max(pmi);
        }
    }

    let avg_pmi = if pmi_count > 0 {
        pmi_sum / pmi_count as f64
    } else {
        0.0
    };

    (avg_pmi, max_pmi)
}

fn analyze_context_30d(
    file_paths: &[String],
    context_name: &str,
    sample_size: usize,
) -> Result<ContextAnalysis30D, Box<dyn std::error::Error>> {
    let sample_files: Vec<_> = file_paths.iter().take(sample_size).collect();

    println!("  {}: {} vocalizations", context_name, sample_files.len());

    let mut all_phrases = Vec::new();

    // Extract phrases from vocalizations
    for (i, file_path) in sample_files.iter().enumerate() {
        if i % 100 == 0 {
            println!("    Processed {} / {}", i, sample_files.len());
        }

        let path = Path::new(file_path);
        if !path.exists() {
            continue;
        }

        match load_flac_file(path) {
            Ok(audio) => {
                let phrase_id = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown");

                match extract_phrases_30d(&audio, 96000, phrase_id) {
                    Ok(phrases) => {
                        if !phrases.is_empty() {
                            all_phrases.push(phrases);
                        }
                    }
                    Err(_) => continue,
                }
            }
            Err(_) => continue,
        }
    }

    if all_phrases.is_empty() {
        return Err("No phrases extracted".into());
    }

    // Discover vocabulary using 30D features
    let word_types = discover_vocabulary_30d(&all_phrases, 0.7)?;

    // Save metrics before moving
    let num_vocalizations = all_phrases.len();
    let total_phrases = all_phrases.iter().map(|p| p.len()).sum();

    // Extract sequences
    let sequences = extract_sequences_30d(all_phrases, &word_types, 0.7)?;

    if sequences.is_empty() {
        return Err("No sequences extracted".into());
    }

    // Calculate PMI
    let (avg_pmi, max_pmi) = calculate_pmi_30d(&sequences);

    Ok(ContextAnalysis30D {
        context_id: context_name.to_string(),
        num_vocalizations,
        total_phrases,
        vocabulary_size: word_types.len(),
        avg_sequence_length: sequences.iter().map(|s| s.words.len()).sum::<usize>() as f64
            / sequences.len() as f64,
        avg_pmi,
        max_pmi,
    })
}

/// Group files by call type
fn discover_files_by_context(
    vocalizations_dir: &Path,
) -> Result<HashMap<CallType, Vec<String>>, Box<dyn std::error::Error>> {
    let mut context_files: HashMap<CallType, Vec<String>> = HashMap::new();

    let entries = std::fs::read_dir(vocalizations_dir)?;
    let mut total_files = 0;

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
                let full_path = file_path.to_str().ok_or("Invalid path")?.to_string();

                context_files
                    .entry(call_type)
                    .or_insert_with(Vec::new)
                    .push(full_path);

                total_files += 1;
            }
        }
    }

    println!(
        "Discovered {} FLAC files across {} call types",
        total_files,
        context_files.len()
    );

    Ok(context_files)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================================================");
    println!("Cross-Context Syntactic Analysis (30D): Marmoset Vocalizations");
    println!("========================================================================");
    println!();
    println!("Research Question: Does vocalization syntax vary by call type?");
    println!("Using 30D MicroDynamics features for phrase similarity");
    println!();

    let vocalizations_dir = Path::new("/home/sheel/birdsong_analysis/data/Vocalizations");

    if !vocalizations_dir.exists() {
        println!("❌ Directory not found: {}", vocalizations_dir.display());
        return Err("Dataset directory not found".into());
    }

    // Discover files by call type
    println!("Discovering marmoset vocalizations by call type...");
    println!("---");
    let context_files = discover_files_by_context(vocalizations_dir)?;

    // Display file counts per context
    println!();
    println!("Call Type Distribution:");
    println!("=======================");
    let all_call_types = [
        CallType::Phee,
        CallType::Twitter,
        CallType::Trill,
        CallType::Tsik,
        CallType::Seep,
        CallType::Infant,
        CallType::Vocalization,
    ];

    for call_type in &all_call_types {
        if let Some(files) = context_files.get(call_type) {
            println!("  {:20} {:>8} files", call_type.name(), files.len());
        }
    }
    println!();

    // Analyze each context
    let mut results = Vec::new();
    let sample_size = 500;

    println!(
        "Analyzing call types (max {} vocalizations each)...",
        sample_size
    );
    println!("---");

    for call_type in &all_call_types {
        if let Some(files) = context_files.get(call_type) {
            match analyze_context_30d(files, call_type.name(), sample_size) {
                Ok(result) => {
                    results.push(result);
                }
                Err(e) => {
                    println!("  {}: Skipped ({})", call_type.name(), e);
                }
            }
        }
    }

    println!();
    println!("========================================================================");
    println!("CROSS-CONTEXT COMPARISON RESULTS (30D Features)");
    println!("========================================================================");
    println!();

    if results.is_empty() {
        println!("❌ No results generated.");
        return Ok(());
    }

    results.sort_by(|a, b| a.context_id.cmp(&b.context_id));

    // Summary table
    println!("Summary Table:");
    println!("===============");
    println!(
        "{:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "Call Type", "Vocs", "Phrases", "Vocab", "AvgLen", "AvgPMI", "MaxPMI"
    );
    println!("{}", "-".repeat(80));

    for r in &results {
        println!(
            "{:<20} {:<10} {:<10} {:<10} {:<10.2} {:<10.3} {:<10.3}",
            r.context_id,
            r.num_vocalizations,
            r.total_phrases,
            r.vocabulary_size,
            r.avg_sequence_length,
            r.avg_pmi,
            r.max_pmi
        );
    }

    println!();

    // Find contexts with strongest syntax evidence
    println!("Syntax Strength Ranking (by Average PMI):");
    println!("==========================================");
    let mut sorted_by_pmi = results.clone();
    sorted_by_pmi.sort_by(|a, b| b.avg_pmi.partial_cmp(&a.avg_pmi).unwrap());

    for (i, r) in sorted_by_pmi.iter().enumerate() {
        let status = if r.avg_pmi > 2.0 {
            "STRONG"
        } else if r.avg_pmi > 0.5 {
            "MODERATE"
        } else {
            "LIMITED"
        };
        println!(
            "  {}. {:20} PMI={:.3} ({})",
            i + 1,
            r.context_id,
            r.avg_pmi,
            status
        );
    }

    println!();

    // Vocabulary richness comparison
    println!("Vocabulary Richness Ranking:");
    println!("============================");
    let mut sorted_by_vocab = results.clone();
    sorted_by_vocab.sort_by(|a, b| b.vocabulary_size.cmp(&a.vocabulary_size));

    for (i, r) in sorted_by_vocab.iter().enumerate() {
        println!(
            "  {}. {:20} {} word types",
            i + 1,
            r.context_id,
            r.vocabulary_size
        );
    }

    println!();

    // Statistical summary
    let avg_pmi_mean: f64 = results.iter().map(|r| r.avg_pmi).sum::<f64>() / results.len() as f64;
    let avg_pmi_std: f64 = {
        let mean = avg_pmi_mean;
        let variance = results
            .iter()
            .map(|r| (r.avg_pmi - mean).powi(2))
            .sum::<f64>()
            / results.len() as f64;
        variance.sqrt()
    };

    let vocab_mean: f64 = results
        .iter()
        .map(|r| r.vocabulary_size as f64)
        .sum::<f64>()
        / results.len() as f64;

    println!("Statistical Summary Across Call Types:");
    println!("======================================");
    println!("  Call types analyzed: {}", results.len());
    println!("  Average PMI: {:.3} ± {:.3}", avg_pmi_mean, avg_pmi_std);
    println!("  Average vocabulary size: {:.1} words", vocab_mean);
    println!();

    // Research interpretation
    println!("Research Interpretation:");
    println!("=======================");
    if avg_pmi_std > 0.5 {
        println!("  ✓ SIGNIFICANT VARIATION in syntax across call types");
        println!(
            "    PMI varies by {:.3} across call types (std dev)",
            avg_pmi_std
        );
        println!("    → Marmoset vocalization syntax DOES change with call type");
    } else {
        println!("  ~ MINIMAL VARIATION in syntax across call types");
        println!("    PMI relatively consistent across call types");
        println!("    → Syntax may be universal across call types");
    }

    println!();
    if !sorted_by_pmi.is_empty() {
        println!(
            "  Highest PMI call type: {} ({:.3})",
            sorted_by_pmi[0].context_id, sorted_by_pmi[0].avg_pmi
        );
        println!(
            "  Lowest PMI call type: {} ({:.3})",
            sorted_by_pmi.last().unwrap().context_id,
            sorted_by_pmi.last().unwrap().avg_pmi
        );
    }

    Ok(())
}
