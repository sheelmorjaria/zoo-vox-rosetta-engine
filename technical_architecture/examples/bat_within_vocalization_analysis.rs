// Within-Vocalization Phrase Analysis: Egyptian Fruit Bats
// =========================================================
//
// Research Goal: Prove or refute the hypothesis that individual bat vocalizations
// contain multi-phrase structure [Word A] + [Word B] rather than being holistic units.
//
// This example analyzes the Egyptian fruit bat dataset to detect:
// 1. Micro-pauses within vocalizations (phrase boundaries)
// 2. F0 (fundamental frequency) changes
// 3. Spectral content changes (for seamless concatenation)
//
// Dataset: /mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/
// - 91,080 vocalizations
// - 250kHz sample rate
// - Mono, IEEE Float format

use std::path::Path;
use technical_architecture::within_vocalization_analyzer::{
    CorpusPhraseAnalyzer, PhraseSegmentation, WithinVocalizationAnalyzer, WithinVocalizationConfig,
};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Load a single WAV file and return audio samples
fn load_wav_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let hint = Hint::new();
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &fmt_opts,
        &meta_opts,
    )?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("No valid audio track found")?;

    let mut decoder = symphonia::default::get_codecs().make(
        &track.codec_params,
        &DecoderOptions::default(),
    )?;

    // Get number of channels from the decoder's spec
    let n_channels = decoder.codec_params().channels.map_or(1, |ch| ch.count());

    // Decode entire file
    let mut audio_samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };

        let decoded = decoder.decode(&packet)?;
        match decoded {
            AudioBufferRef::F32(buf) => {
                // Process f32 audio buffer
                let n_frames = buf.frames();
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    for &sample in samples.iter() {
                        audio_samples.push(sample);
                    }
                }
            }
            AudioBufferRef::S16(buf) => {
                // Process i16 audio buffer
                let n_frames = buf.frames();
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

/// Analyze a sample of bat vocalizations for within-vocalization phrase structure
fn analyze_bat_corpus(
    audio_dir: &Path,
    sample_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("========================================================================");
    println!("Within-Vocalization Phrase Analysis: Egyptian Fruit Bats");
    println!("========================================================================");
    println!();
    println!("Research Hypothesis: Bat vocalizations contain [Word A] + [Word B] structure");
    println!("Analysis Goal: Detect multi-phrase structure within individual vocalizations");
    println!();

    // Configuration for bat vocalizations (250kHz, high-frequency detection)
    let config = WithinVocalizationConfig {
        min_phrase_duration_ms: 5.0,   // Short phrases (5ms minimum)
        min_pause_duration_ms: 2.0,    // Very brief pauses (2ms minimum)
        min_f0_change_hz: 1500.0,      // F0 changes (1.5kHz threshold for bats)
        sample_rate: 250000,           // Bat audio sample rate
        frame_size_ms: 2.0,            // Fine-grained analysis (2ms frames)
        hop_size_ms: 1.0,              // Overlapping frames (1ms hop)
        pause_energy_threshold: 0.15,  // Energy threshold for pause detection
        require_consensus: false,      // Don't require consensus (seamless concatenation)
        max_phrases: 8,                // Maximum phrases per vocalization
    };

    println!("Configuration:");
    println!("  - Sample rate: {} kHz", config.sample_rate / 1000);
    println!("  - Min phrase duration: {} ms", config.min_phrase_duration_ms);
    println!("  - Min pause duration: {} ms", config.min_pause_duration_ms);
    println!("  - Min F0 change: {} Hz", config.min_f0_change_hz);
    println!("  - Frame size: {} ms", config.frame_size_ms);
    println!("  - Hop size: {} ms", config.hop_size_ms);
    println!("  - Require consensus: {}", config.require_consensus);
    println!();

    let analyzer = WithinVocalizationAnalyzer::new(config.clone());
    let corpus_analyzer = CorpusPhraseAnalyzer::new(config);

    // Find audio files
    println!("Scanning audio directory: {}", audio_dir.display());
    let audio_files: Vec<_> = std::fs::read_dir(audio_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "wav"))
        .take(sample_size)
        .collect();

    println!("Found {} audio files (analyzing sample of {})", audio_files.len(), sample_size);
    println!();

    // Load and analyze vocalizations
    println!("Analyzing vocalizations...");
    println!("---");

    let mut vocalizations = Vec::new();
    let mut f0_contours = Vec::new();
    let mut multi_phrase_examples = Vec::new();
    let mut single_phrase_examples = Vec::new();

    for (i, entry) in audio_files.iter().enumerate() {
        let path = entry.path();
        let file_name = path.file_name().unwrap().to_string_lossy();

        if i % 100 == 0 {
            println!("Progress: {}/{} files loaded...", i, audio_files.len());
        }

        match load_wav_file(&path) {
            Ok(audio) => {
                // Analyze for multi-phrase structure
                match analyzer.analyze_vocalization(&audio, None) {
                    Ok(segmentation) => {
                        if segmentation.num_phrases > 1 {
                            multi_phrase_examples.push((file_name.to_string(), segmentation.clone()));
                            if multi_phrase_examples.len() <= 5 {
                                println!("  ✓ [{}] {} phrases detected: {}",
                                    file_name, segmentation.num_phrases,
                                    format_bounds(&segmentation)
                                );
                            }
                        } else {
                            if single_phrase_examples.len() < 3 {
                                single_phrase_examples.push((file_name.to_string(), segmentation));
                            }
                        }

                        // Collect for corpus analysis
                        // Note: We need to keep the audio alive for references
                        vocalizations.push(audio);
                        f0_contours.push(None);
                    }
                    Err(e) => {
                        eprintln!("  ✗ Error analyzing {}: {}", file_name, e);
                    }
                }
            }
            Err(e) => {
                eprintln!("  ✗ Error loading {}: {}", file_name, e);
            }
        }

        // Stop after reaching sample size
        if vocalizations.len() >= sample_size {
            break;
        }
    }

    println!("---");
    println!();

    // Perform corpus-level analysis
    if !vocalizations.is_empty() {
        println!("Corpus-Level Statistics:");
        println!("=========================");

        let vocalizations_refs: Vec<&[f32]> = vocalizations.iter().map(|v| v.as_slice()).collect();

        match corpus_analyzer.analyze_corpus(vocalizations_refs, f0_contours) {
            Ok(stats) => {
                println!("Total vocalizations analyzed: {}", stats.total_vocalizations);
                println!("Multi-phrase vocalizations: {}", stats.multi_phrase_count);
                println!("Multi-phrase detection rate: {:.2}%", stats.multi_phrase_rate * 100.0);
                println!("Average phrases per vocalization: {:.2}", stats.avg_phrases_per_vocalization);
                println!("Total boundaries detected: {}", stats.total_boundaries);
                println!();

                // Research interpretation
                println!("Research Interpretation:");
                println!("========================");

                if stats.multi_phrase_rate > 0.30 {
                    println!("✓ STRONG EVIDENCE for multi-phrase structure:");
                    println!("  {:.1}% of vocalizations show internal phrase boundaries", stats.multi_phrase_rate * 100.0);
                    println!("  Average {:.2} phrases per vocalization", stats.avg_phrases_per_vocalization);
                    println!("  → Supports hypothesis: [Word A] + [Word B] structure exists");
                } else if stats.multi_phrase_rate > 0.10 {
                    println!("~ MODERATE EVIDENCE for multi-phrase structure:");
                    println!("  {:.1}% of vocalizations show internal phrase boundaries", stats.multi_phrase_rate * 100.0);
                    println!("  Average {:.2} phrases per vocalization", stats.avg_phrases_per_vocalization);
                    println!("  → Suggests some vocalizations have multi-phrase structure");
                } else {
                    println!("✗ LIMITED EVIDENCE for multi-phrase structure:");
                    println!("  {:.1}% of vocalizations show internal phrase boundaries", stats.multi_phrase_rate * 100.0);
                    println!("  Average {:.2} phrases per vocalization", stats.avg_phrases_per_vocalization);
                    println!("  → Most vocalizations appear to be holistic units");
                    println!("  → May need to adjust detection thresholds or use different features");
                }
            }
            Err(e) => {
                eprintln!("Error analyzing corpus: {}", e);
            }
        }
    }

    println!();
    println!("Example Multi-Phrase Vocalizations:");
    println!("====================================");
    for (file_name, segmentation) in multi_phrase_examples.iter().take(10) {
        println!("  [{}]: {} phrases, confidence: {:.2}",
            file_name,
            segmentation.num_phrases,
            segmentation.confidence
        );
        println!("    Boundaries: {}", format_bounds(segmentation));
        println!("    Durations: {}", format_durations(segmentation));
    }

    Ok(())
}

fn format_bounds(segmentation: &PhraseSegmentation) -> String {
    if segmentation.boundaries.is_empty() {
        "none".to_string()
    } else {
        segmentation.boundaries.iter()
            .map(|b| format!("{:.1}ms", b.position_ms))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

fn format_durations(segmentation: &PhraseSegmentation) -> String {
    segmentation.phrase_durations_ms.iter()
        .map(|d| format!("{:.1}ms", d))
        .collect::<Vec<_>>()
        .join(" + ")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio");

    // Analyze a sample of 1000 vocalizations
    analyze_bat_corpus(audio_dir, 1000)?;

    Ok(())
}
