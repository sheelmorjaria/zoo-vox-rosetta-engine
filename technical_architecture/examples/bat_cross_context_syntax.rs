// Cross-Context Syntactic Analysis: Egyptian Fruit Bats
// =====================================================
//
// This example analyzes how syntactic structure varies across different
// behavioral contexts (e.g., aggression, food sharing, mating, etc.)
//
// Research Question: Does bat vocalization syntax change with context?
//
// Contexts in dataset:
// - 0-12: Different behavioral contexts
// - Distribution varies: Context 11 (29,627), Context 12 (33,997), etc.

use std::collections::HashMap;
use std::path::Path;
use technical_architecture::phrase_sequence_analyzer::{PhraseSequenceAnalyzer, PMIAnalysis};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Context annotation from CSV
#[derive(Debug, Clone)]
struct ContextAnnotation {
    emitter: i32,
    addressee: i32,
    context: u32,
    file_name: String,
}

/// Load a single WAV file and return audio samples
fn load_wav_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let hint = Hint::new();
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("No valid audio track found")?;

    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let n_channels = decoder.codec_params().channels.map_or(1, |ch| ch.count());

    let mut audio_samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };

        let decoded = decoder.decode(&packet)?;
        match decoded {
            AudioBufferRef::F32(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend_from_slice(samples);
                }
            }
            AudioBufferRef::S16(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    for &sample in samples.iter() {
                        audio_samples.push(sample as f32 / i16::MAX as f32);
                    }
                }
            }
            _ => {
                return Err("Unsupported audio format".into());
            }
        }
    }

    Ok(audio_samples)
}

/// Load context annotations from CSV
fn load_annotations(annotations_path: &Path) -> Result<Vec<ContextAnnotation>, Box<dyn std::error::Error>> {
    let mut annotations = Vec::new();
    let content = std::fs::read_to_string(annotations_path)?;

    // Skip header
    for line in content.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 8 {
            let emitter = parts[0].parse::<i32>().unwrap_or(0);
            let addressee = parts[1].parse::<i32>().unwrap_or(0);
            let context = parts[2].parse::<u32>().unwrap_or(0);
            let file_name = parts[7].to_string();

            annotations.push(ContextAnnotation {
                emitter,
                addressee,
                context,
                file_name,
            });
        }
    }

    Ok(annotations)
}

/// Cross-context analysis results
#[derive(Debug, Clone)]
struct ContextAnalysis {
    context_id: u32,
    num_vocalizations: usize,
    total_phrases: usize,
    vocabulary_size: usize,
    avg_sequence_length: f64,
    avg_pmi: f64,
    max_pmi: f64,
    high_pmi_count: usize,
}

fn analyze_context(
    audio_dir: &Path,
    annotations: &[ContextAnnotation],
    target_context: u32,
    sample_size: usize,
) -> Result<ContextAnalysis, Box<dyn std::error::Error>> {
    // Filter annotations by context
    let context_annotations: Vec<_> = annotations
        .iter()
        .filter(|a| a.context == target_context)
        .take(sample_size)
        .collect();

    if context_annotations.is_empty() {
        return Err(format!("No vocalizations found for context {}", target_context).into());
    }

    println!("  Context {}: {} vocalizations", target_context, context_annotations.len());

    let sequence_analyzer = PhraseSequenceAnalyzer::with_threshold(0.2);
    let mut all_phrases = Vec::new();

    // Extract phrases from vocalizations
    for annotation in &context_annotations {
        let audio_path = audio_dir.join(&annotation.file_name);
        if !audio_path.exists() {
            continue;
        }

        match load_wav_file(&audio_path) {
            Ok(audio) => {
                match sequence_analyzer.extract_phrases(&audio, 250000) {
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

    // Discover vocabulary
    let word_types = sequence_analyzer.discover_vocabulary(all_phrases.clone())?;

    // Save metrics before moving
    let num_vocalizations = all_phrases.len();
    let total_phrases = all_phrases.iter().map(|p| p.len()).sum();

    // Extract sequences
    let sequences = sequence_analyzer.extract_sequences(all_phrases, &word_types)?;

    if sequences.is_empty() {
        return Err("No sequences extracted".into());
    }

    // Calculate PMI
    let pmi = sequence_analyzer.calculate_pmi(&sequences);

    Ok(ContextAnalysis {
        context_id: target_context,
        num_vocalizations,
        total_phrases,
        vocabulary_size: word_types.len(),
        avg_sequence_length: sequences.iter().map(|s| s.words.len()).sum::<usize>() as f64 / sequences.len() as f64,
        avg_pmi: pmi.avg_pmi,
        max_pmi: pmi.max_pmi,
        high_pmi_count: pmi.high_pmi_transitions.len(),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================================================");
    println!("Cross-Context Syntactic Analysis: Egyptian Fruit Bats");
    println!("========================================================================");
    println!();
    println!("Research Question: Does vocalization syntax vary by context?");
    println!();

    let audio_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio");
    let annotations_path = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/annotations.csv");

    // Load annotations
    println!("Loading context annotations...");
    let annotations = load_annotations(annotations_path)?;
    println!("Loaded {} annotations", annotations.len());
    println!();

    // Get unique contexts
    let mut contexts: Vec<u32> = annotations.iter().map(|a| a.context).collect();
    contexts.sort();
    contexts.dedup();

    println!("Found {} unique contexts: {:?}", contexts.len(), contexts);
    println!();

    // Analyze each context
    let mut results = Vec::new();
    let sample_size = 500; // Max vocalizations per context

    println!("Analyzing contexts (max {} vocalizations each)...", sample_size);
    println!("---");

    for context_id in contexts {
        match analyze_context(audio_dir, &annotations, context_id, sample_size) {
            Ok(result) => {
                results.push(result);
            }
            Err(e) => {
                println!("  Context {}: Skipped ({})", context_id, e);
            }
        }
    }

    println!();
    println!("========================================================================");
    println!("CROSS-CONTEXT COMPARISON RESULTS");
    println!("========================================================================");
    println!();

    // Sort by context ID
    results.sort_by_key(|r| r.context_id);

    // Summary table
    println!("Summary Table:");
    println!("===============");
    println!("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "Context", "Vocs", "Phrases", "Vocab", "AvgLen", "AvgPMI", "MaxPMI");
    println!("{}", "-".repeat(80));

    for r in &results {
        println!("{:<10} {:<10} {:<10} {:<10} {:<10.2} {:<10.3} {:<10.3}",
            r.context_id,
            r.num_vocalizations,
            r.total_phrases,
            r.vocabulary_size,
            r.avg_sequence_length,
            r.avg_pmi,
            r.max_pmi);
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
        println!("  {}. Context {}: PMI={:.3} ({})", i + 1, r.context_id, r.avg_pmi, status);
    }

    println!();

    // Vocabulary richness comparison
    println!("Vocabulary Richness Ranking:");
    println!("============================");
    let mut sorted_by_vocab = results.clone();
    sorted_by_vocab.sort_by(|a, b| b.vocabulary_size.cmp(&a.vocabulary_size));

    for (i, r) in sorted_by_vocab.iter().enumerate() {
        println!("  {}. Context {}: {} word types", i + 1, r.context_id, r.vocabulary_size);
    }

    println!();

    // Statistical summary
    if !results.is_empty() {
        let avg_pmi_mean: f64 = results.iter().map(|r| r.avg_pmi).sum::<f64>() / results.len() as f64;
        let avg_pmi_std: f64 = {
            let mean = avg_pmi_mean;
            let variance = results.iter().map(|r| (r.avg_pmi - mean).powi(2)).sum::<f64>() / results.len() as f64;
            variance.sqrt()
        };

        let vocab_mean: f64 = results.iter().map(|r| r.vocabulary_size as f64).sum::<f64>() / results.len() as f64;

        println!("Statistical Summary Across Contexts:");
        println!("====================================");
        println!("  Contexts analyzed: {}", results.len());
        println!("  Average PMI: {:.3} ± {:.3}", avg_pmi_mean, avg_pmi_std);
        println!("  Average vocabulary size: {:.1} words", vocab_mean);
        println!();

        // Research interpretation
        println!("Research Interpretation:");
        println!("=======================");
        if avg_pmi_std > 0.5 {
            println!("  ✓ SIGNIFICANT VARIATION in syntax across contexts");
            println!("    PMI varies by {:.3} across contexts (std dev)", avg_pmi_std);
            println!("    → Bat vocalization syntax DOES change with context");
            println!("    → Different contexts may have different communicative functions");
        } else {
            println!("  ~ MINIMAL VARIATION in syntax across contexts");
            println!("    PMI relatively consistent across contexts");
            println!("    → Syntax may be universal across behavioral contexts");
        }

        println!();
        println!("  Highest PMI context: {} ({:.3})",
            sorted_by_pmi[0].context_id, sorted_by_pmi[0].avg_pmi);
        println!("  Lowest PMI context: {} ({:.3})",
            sorted_by_pmi.last().unwrap().context_id, sorted_by_pmi.last().unwrap().avg_pmi);
    }

    Ok(())
}
