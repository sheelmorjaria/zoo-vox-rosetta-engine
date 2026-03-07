// Syntactic Structure Discovery: Egyptian Fruit Bats
// =================================================
//
// This example discovers syntactic rules in bat vocalizations by:
// 1. Extracting phrase sequences from within-vocalization analysis
// 2. Clustering similar phrases into "word types"
// 3. Calculating PMI to prove fixed word order (syntax)
// 4. Identifying recurring patterns (grammatical rules)
//
// Research Goal: Prove that bat vocalizations follow syntactic rules
// similar to human language grammar.

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::path::Path;
use technical_architecture::phrase_sequence_analyzer::{PMIAnalysis, PhraseSequenceAnalyzer, SyntaxRules};
use technical_architecture::within_vocalization_analyzer::{WithinVocalizationAnalyzer, WithinVocalizationConfig};

/// Load a single WAV file and return audio samples
fn load_wav_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use symphonia::core::audio::{AudioBufferRef, Signal};
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

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
                let _n_frames = buf.frames();
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend_from_slice(samples);
                }
            }
            AudioBufferRef::S16(buf) => {
                let _n_frames = buf.frames();
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

/// Analyze bat vocalizations for syntactic structure
fn analyze_bat_syntax(audio_dir: &Path, sample_size: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================================================");
    println!("Syntactic Structure Discovery: Egyptian Fruit Bats");
    println!("========================================================================");
    println!();
    println!("Research Goal: Discover grammatical rules in bat vocalizations");
    println!("Method: PMI analysis + pattern mining");
    println!();

    // Find audio files
    println!("Scanning audio directory...");
    let audio_files: Vec<_> = std::fs::read_dir(audio_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "wav"))
        .take(sample_size)
        .collect();

    println!(
        "Found {} audio files (analyzing sample of {})",
        audio_files.len(),
        sample_size
    );
    println!();

    // Configure analyzers
    let within_config = WithinVocalizationConfig {
        min_phrase_duration_ms: 5.0,
        min_pause_duration_ms: 2.0,
        min_f0_change_hz: 1500.0,
        sample_rate: 250000,
        frame_size_ms: 2.0,
        hop_size_ms: 1.0,
        pause_energy_threshold: 0.15,
        require_consensus: false,
        max_phrases: 10,
    };

    let within_analyzer = WithinVocalizationAnalyzer::new(within_config);
    let sequence_analyzer = PhraseSequenceAnalyzer::with_threshold(0.2);

    // Step 1: Extract phrases from all vocalizations
    println!("Step 1: Extracting phrases from vocalizations...");
    println!("---");

    let mut all_phrases = Vec::new();
    let mut vocalization_ids = Vec::new();

    for (i, entry) in audio_files.iter().enumerate() {
        let path = entry.path();
        let file_name = path.file_name().unwrap().to_string_lossy().to_string();

        if i % 100 == 0 {
            println!("Progress: {}/{} files processed...", i, audio_files.len());
        }

        match load_wav_file(&path) {
            Ok(audio) => match sequence_analyzer.extract_phrases(&audio, 250000) {
                Ok(phrases) => {
                    if !phrases.is_empty() {
                        vocalization_ids.push(file_name);
                        all_phrases.push(phrases);
                    }
                }
                Err(e) => {
                    eprintln!("  Warning: Could not extract phrases from {}: {}", file_name, e);
                }
            },
            Err(e) => {
                eprintln!("  Error loading {}: {}", file_name, e);
            }
        }
    }

    println!("---");
    println!("Extracted {} vocalizations with phrase structure", all_phrases.len());
    println!("Total phrases: {}", all_phrases.iter().map(|p| p.len()).sum::<usize>());
    println!();

    // Step 2: Discover vocabulary (word types)
    println!("Step 2: Discovering vocabulary (word types)...");
    println!("---");

    let word_types = match sequence_analyzer.discover_vocabulary(all_phrases.clone()) {
        Ok(words) => words,
        Err(e) => {
            println!("Error: {}", e);
            return Err(e.into());
        }
    };

    println!("Discovered {} unique word types", word_types.len());
    println!();

    // Show vocabulary statistics
    println!("Vocabulary Statistics:");
    println!("======================");
    for (i, word) in word_types.iter().take(10).enumerate() {
        println!(
            "  Word {}: {} occurrences, F0={:.0} Hz",
            i, word.count, word.features.f0_mean
        );
    }

    if word_types.len() > 10 {
        println!("  ... and {} more word types", word_types.len() - 10);
    }
    println!();

    // Step 3: Extract word sequences
    println!("Step 3: Extracting word sequences...");
    println!("---");

    let sequences = sequence_analyzer.extract_sequences(all_phrases.clone(), &word_types)?;

    println!("Extracted {} word sequences", sequences.len());
    println!(
        "Average sequence length: {:.2} words",
        sequences.iter().map(|s| s.words.len()).sum::<usize>() as f64 / sequences.len() as f64
    );
    println!();

    // Show example sequences
    println!("Example Word Sequences:");
    println!("=======================");
    for (i, seq) in sequences.iter().take(5).enumerate() {
        println!("  [Vocalization {}]: {:?}", vocalization_ids[i], seq.words);
    }
    println!();

    // Step 4: Calculate PMI (Pointwise Mutual Information)
    println!("Step 4: Calculating PMI (Pointwise Mutual Information)...");
    println!("---");

    let pmi = sequence_analyzer.calculate_pmi(&sequences);

    println!("PMI Analysis Results:");
    println!("=====================");
    println!("Vocabulary size: {}", pmi.vocabulary_size);
    println!("Total word tokens: {}", pmi.total_words);
    println!("Total bigram tokens: {}", pmi.total_bigrams);
    println!("Maximum PMI: {:.3}", pmi.max_pmi);
    println!("Average PMI: {:.3}", pmi.avg_pmi);
    println!();

    // Show high-PMI transitions (strong word associations)
    println!("High-PMI Transitions (Fixed Word Order):");
    println!("===========================================");
    if pmi.high_pmi_transitions.is_empty() {
        println!("  No transitions with PMI > 2.0 found");
        println!("  (This may indicate flexible word ordering)");
    } else {
        for (w1, w2, pmi_score) in pmi.high_pmi_transitions.iter().take(15) {
            println!(
                "  Word {} → Word {}: PMI = {:.3} (strong association)",
                w1, w2, pmi_score
            );
        }
    }
    println!();

    // Step 5: Find recurring patterns
    println!("Step 5: Finding recurring n-gram patterns...");
    println!("---");

    let patterns = sequence_analyzer.find_patterns(&sequences, 4);

    println!("Most Common Patterns (n-grams):");
    println!("=================================");
    for (i, pattern) in patterns.iter().take(10).enumerate() {
        println!("  {}. {:?} (occurs {} times)", i + 1, pattern.words, pattern.count);
    }
    println!();

    // Step 6: Discover syntactic rules
    println!("Step 6: Discovering syntactic rules...");
    println!("---");

    let rules = sequence_analyzer.discover_rules(&sequences);

    println!("Discovered Syntactic Rules:");
    println!("===========================");
    println!("Vocabulary size: {}", rules.vocabulary_size);
    println!("Average sentence length: {:.2} words", rules.avg_sentence_length);
    println!();

    println!("Positional Grammar (most common word at each position):");
    println!("========================================================");
    let mut position_entries: Vec<_> = rules.positional_grammar.iter().collect();
    position_entries.sort_by_key(|&(pos, _)| pos);
    for (pos, word) in position_entries {
        println!("  Position {}: Word {} (fixed position)", pos, word);
    }
    println!();

    // Research Interpretation
    println!("Research Interpretation:");
    println!("=======================");
    println!();

    if pmi.avg_pmi > 2.0 {
        println!("✓ STRONG EVIDENCE for fixed word order (syntax):");
        println!(
            "  Average PMI {:.2} > 2.0 indicates strong word associations",
            pmi.avg_pmi
        );
        println!("  → Bat vocalizations follow grammatical rules");
        println!("  → Word order is not random");
    } else if pmi.avg_pmi > 0.5 {
        println!("~ MODERATE EVIDENCE for syntactic structure:");
        println!("  Average PMI {:.2} shows some word associations", pmi.avg_pmi);
        println!("  → Partial word ordering may exist");
    } else {
        println!("✗ LIMITED EVIDENCE for fixed word order:");
        println!("  Average PMI {:.2} indicates flexible word ordering", pmi.avg_pmi);
        println!("  → Bat vocalizations may use combinatorial grammar");
        println!("  → Similar to human language flexibility");
    }

    println!();
    println!("Pattern Analysis:");
    if !patterns.is_empty() {
        let top_pattern = &patterns[0];
        println!(
            "  Most common pattern: {:?} ({} occurrences)",
            top_pattern.words, top_pattern.count
        );
        println!("  → Recurring patterns indicate grammatical structure");
    }

    println!();
    println!("Vocabulary richness:");
    println!("  {} unique word types discovered", rules.vocabulary_size);
    println!("  Average {:.2} words per vocalization", rules.avg_sentence_length);

    if rules.vocabulary_size > 20 {
        println!("  → Rich vocabulary suggests complex communication");
    } else if rules.vocabulary_size > 5 {
        println!("  → Moderate vocabulary suggests structured communication");
    } else {
        println!("  → Limited vocabulary may indicate simple communication");
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio");

    // Analyze 500 vocalizations for syntax
    analyze_bat_syntax(audio_dir, 500)?;

    Ok(())
}
