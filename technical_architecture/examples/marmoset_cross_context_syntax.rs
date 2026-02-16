// Cross-Context Syntactic Analysis: Marmoset Vocalizations
// ===========================================================
//
// This example analyzes how syntactic structure varies across different
// call types (e.g., Phee, Twitter, Trill, Tsik, Seep, Infant_cry)
//
// Research Question: Does marmoset vocalization syntax change with call type?
//
// Call types in dataset:
// - Phee: Long-distance contact calls
// - Twitter: Short-range social calls
// - Trill: Aggressive/excitement calls
// - Tsik: Alarm calls
// - Seep: Close contact calls
// - Infant_cry: Juvenile vocalizations
// - Vocalization: Unclassified/general

use std::collections::HashMap;
use std::path::Path;
use technical_architecture::phrase_sequence_analyzer::PhraseSequenceAnalyzer;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

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

    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let n_channels = decoder.codec_params().channels.map_or(1, |ch| ch.count());
    let _sample_rate = decoder.codec_params().sample_rate.unwrap_or(96000);

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
                    audio_samples.extend(samples.iter().map(|&s| s.inner() as f32 / (i32::MAX >> 8) as f32));
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
                    audio_samples.extend(samples.iter().map(|&s| (s.inner() as f32 - 8388608.0) / 8388608.0));
                }
            }
            AudioBufferRef::U32(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| (s as f32 - 2147483648.0) / 2147483648.0));
                }
            }
        }
    }

    Ok(audio_samples)
}

/// Cross-context analysis results
#[derive(Debug, Clone)]
struct ContextAnalysis {
    context_id: String,
    num_vocalizations: usize,
    total_phrases: usize,
    vocabulary_size: usize,
    avg_sequence_length: f64,
    avg_pmi: f64,
    max_pmi: f64,
    high_pmi_count: usize,
}

/// Group files by call type
fn discover_files_by_context(vocalizations_dir: &Path) -> Result<HashMap<CallType, Vec<String>>, Box<dyn std::error::Error>> {
    let mut context_files: HashMap<CallType, Vec<String>> = HashMap::new();

    // Read all subdirectories (date-based: 2021_1_0, 2020_10_0, etc.)
    let entries = std::fs::read_dir(vocalizations_dir)?;
    let mut total_files = 0;

    for entry in entries {
        let entry = entry?;
        let dir_path = entry.path();

        if !dir_path.is_dir() {
            continue;
        }

        // Read FLAC files in this directory
        let file_entries = std::fs::read_dir(&dir_path)?;
        for file_entry in file_entries {
            let file_entry = file_entry?;
            let file_path = file_entry.path();

            if !file_path.is_file() {
                continue;
            }

            let filename = file_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            if !filename.ends_with(".flac") {
                continue;
            }

            // Extract call type from filename
            if let Some(call_type) = CallType::from_filename(filename) {
                let full_path = file_path.to_str()
                    .ok_or("Invalid path")?
                    .to_string();

                context_files.entry(call_type)
                    .or_insert_with(Vec::new)
                    .push(full_path);

                total_files += 1;
            }
        }
    }

    println!("Discovered {} FLAC files across {} call types",
        total_files, context_files.len());

    Ok(context_files)
}

fn analyze_context(
    file_paths: &[String],
    context_name: &str,
    sample_size: usize,
) -> Result<ContextAnalysis, Box<dyn std::error::Error>> {
    if file_paths.is_empty() {
        return Err(format!("No files found for context {}", context_name).into());
    }

    // Take sample_size files
    let sample_files: Vec<_> = file_paths.iter().take(sample_size).collect();

    println!("  {}: {} vocalizations", context_name, sample_files.len());

    let sequence_analyzer = PhraseSequenceAnalyzer::with_threshold(0.2);
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
                // Marmoset sample rate is typically 96kHz
                match sequence_analyzer.extract_phrases(&audio, 96000) {
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
        context_id: context_name.to_string(),
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
    println!("Cross-Context Syntactic Analysis: Marmoset Vocalizations");
    println!("========================================================================");
    println!();
    println!("Research Question: Does vocalization syntax vary by call type?");
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
    let sample_size = 500; // Max vocalizations per context

    println!("Analyzing call types (max {} vocalizations each)...", sample_size);
    println!("---");

    for call_type in &all_call_types {
        if let Some(files) = context_files.get(call_type) {
            match analyze_context(files, call_type.name(), sample_size) {
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
    println!("CROSS-CONTEXT COMPARISON RESULTS");
    println!("========================================================================");
    println!();

    if results.is_empty() {
        println!("❌ No results generated. Check file paths and formats.");
        return Ok(());
    }

    // Sort by context ID
    results.sort_by(|a, b| a.context_id.cmp(&b.context_id));

    // Summary table
    println!("Summary Table:");
    println!("===============");
    println!("{:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "Call Type", "Vocs", "Phrases", "Vocab", "AvgLen", "AvgPMI", "MaxPMI");
    println!("{}", "-".repeat(80));

    for r in &results {
        println!("{:<20} {:<10} {:<10} {:<10} {:<10.2} {:<10.3} {:<10.3}",
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
        println!("  {}. {:20} PMI={:.3} ({})", i + 1, r.context_id, r.avg_pmi, status);
    }

    println!();

    // Vocabulary richness comparison
    println!("Vocabulary Richness Ranking:");
    println!("============================");
    let mut sorted_by_vocab = results.clone();
    sorted_by_vocab.sort_by(|a, b| b.vocabulary_size.cmp(&a.vocabulary_size));

    for (i, r) in sorted_by_vocab.iter().enumerate() {
        println!("  {}. {:20} {} word types", i + 1, r.context_id, r.vocabulary_size);
    }

    println!();

    // Statistical summary
    let avg_pmi_mean: f64 = results.iter().map(|r| r.avg_pmi).sum::<f64>() / results.len() as f64;
    let avg_pmi_std: f64 = {
        let mean = avg_pmi_mean;
        let variance = results.iter().map(|r| (r.avg_pmi - mean).powi(2)).sum::<f64>() / results.len() as f64;
        variance.sqrt()
    };

    let vocab_mean: f64 = results.iter().map(|r| r.vocabulary_size as f64).sum::<f64>() / results.len() as f64;

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
        println!("    PMI varies by {:.3} across call types (std dev)", avg_pmi_std);
        println!("    → Marmoset vocalization syntax DOES change with call type");
        println!("    → Different call types may have different communicative functions");
    } else {
        println!("  ~ MINIMAL VARIATION in syntax across call types");
        println!("    PMI relatively consistent across call types");
        println!("    → Syntax may be universal across call types");
    }

    println!();
    if !sorted_by_pmi.is_empty() {
        println!("  Highest PMI call type: {} ({:.3})",
            sorted_by_pmi[0].context_id, sorted_by_pmi[0].avg_pmi);
        println!("  Lowest PMI call type: {} ({:.3})",
            sorted_by_pmi.last().unwrap().context_id, sorted_by_pmi.last().unwrap().avg_pmi);
    }

    Ok(())
}
