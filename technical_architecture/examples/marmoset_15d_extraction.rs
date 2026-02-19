// Marmoset-Optimized 15D Feature Extraction Example
// ================================================
//
// This example demonstrates the use of the RFE-optimized 15D feature extraction
// specifically designed for marmoset call type classification.
//
// **Research Context:**
// The 15 features were selected via Recursive Feature Elimination (RFE) using
// Fisher scores as the discriminative metric. They are optimized for distinguishing
// between marmoset call types: Phee, Twitter, Trill, Tsik, Seep, and Infant cries.
//
// **Feature Breakdown:**
// - Energy (2D): rms_energy, vibrato_depth
// - MFCC (4D): mfcc_0, mfcc_1, mfcc_3, mfcc_4
// - Timbre (2D): spectral_flux, hnr
// - Temporal (3D): decay_time_ms, sustain_level, attack_time_ms
// - Rhythm (2D): ici_cv, onset_rate_hz
// - Modulation (1D): vibrato_rate_hz
// - Perturbation (1D): shimmer
//
// Usage: cargo run --example marmoset_15d_extraction --release

use std::path::Path;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use technical_architecture::{MicroDynamicsExtractor, MicroDynamicsFeatures15D};

/// Marmoset call types for demonstration
#[derive(Debug, Clone, Copy)]
enum MarmosetCallType {
    Phee,    // Long-distance harmonic communication
    Twitter, // Rapid high-pitched social calls
    Trill,   // Rapid frequency-modulated calls
    Tsik,    // Short sharp alarm calls
    Seep,    // Soft contact calls
    Infant,  // Infant distress calls
}

impl MarmosetCallType {
    fn name(&self) -> &'static str {
        match self {
            MarmosetCallType::Phee => "Phee",
            MarmosetCallType::Twitter => "Twitter",
            MarmosetCallType::Trill => "Trill",
            MarmosetCallType::Tsik => "Tsik",
            MarmosetCallType::Seep => "Seep",
            MarmosetCallType::Infant => "Infant",
        }
    }

    /// Describe typical acoustic characteristics for this call type
    fn typical_characteristics(&self) -> &'static str {
        match self {
            MarmosetCallType::Phee => {
                "Long duration (200-500ms), high HNR, slow attack/decay, low spectral flux"
            }
            MarmosetCallType::Twitter => {
                "Short duration (50-100ms), high onset rate, high-frequency MFCCs, rhythmic"
            }
            MarmosetCallType::Trill => {
                "FM-dominated, high vibrato depth, moderate spectral flux, variable rhythm"
            }
            MarmosetCallType::Tsik => {
                "Very short (<50ms), sharp attack, rapid decay, low sustain, noisy (low HNR)"
            }
            MarmosetCallType::Seep => {
                "Soft (low RMS energy), low frequency, moderate duration, steady amplitude"
            }
            MarmosetCallType::Infant => {
                "High pitch, variable duration, high shimmer, irregular rhythm (high ICI CV)"
            }
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
            AudioBufferRef::F32(buf) => {
                for ch in 0..n_channels {
                    audio_samples.extend_from_slice(buf.chan(ch));
                }
            }
            AudioBufferRef::S16(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i16::MAX as f32));
                }
            }
            _ => return Err("Unsupported audio format".into()),
        }
    }

    Ok(audio_samples)
}

/// Analyze 15D features and provide interpretation
fn interpret_features(features: &MicroDynamicsFeatures15D) -> MarmosetCallType {
    // Simple heuristic classification based on RFE rankings
    // In practice, you would use a trained classifier (SVM, Random Forest, etc.)

    // High HNR + long decay + low spectral flux → Phee
    if features.hnr > 15.0 && features.decay_time_ms > 150.0 && features.spectral_flux < 0.5 {
        return MarmosetCallType::Phee;
    }

    // High onset rate + high-frequency MFCCs → Twitter
    if features.onset_rate_hz > 10.0 && features.mfcc_0 > 100.0 {
        return MarmosetCallType::Twitter;
    }

    // High vibrato depth + moderate spectral flux → Trill
    if features.vibrato_depth > 100.0 && features.spectral_flux > 0.5 {
        return MarmosetCallType::Trill;
    }

    // Short duration + sharp attack + low sustain → Tsik
    if features.attack_time_ms < 10.0
        && features.sustain_level < 0.3
        && features.decay_time_ms < 50.0
    {
        return MarmosetCallType::Tsik;
    }

    // Low RMS energy + low frequency → Seep
    if features.rms_energy < 0.2 {
        return MarmosetCallType::Seep;
    }

    // High shimmer + irregular rhythm → Infant
    if features.shimmer > 0.05 && features.ici_cv > 0.4 {
        return MarmosetCallType::Infant;
    }

    // Default
    MarmosetCallType::Phee
}

/// Display features with interpretation
fn display_features(features: &MicroDynamicsFeatures15D, filename: &str) {
    let call_type = interpret_features(features);

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!(
        "│ File: {} (30 chars max)                         │",
        &filename[..filename.len().min(30)]
    );
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("Predicted Call Type: {}", call_type.name());
    println!("Characteristics: {}", call_type.typical_characteristics());
    println!();

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    15D RFE-OPTIMIZED FEATURES                             ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ ENERGY FEATURES (2D)                                                     │");
    println!("├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  RMS Energy        (Fisher: 1.914 #1)    {:>10.4}                       │",
        features.rms_energy
    );
    println!(
        "│  Vibrato Depth     (Fisher: 0.631 #6)    {:>10.4} cents                │",
        features.vibrato_depth
    );
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ MFCC FEATURES (4D)                                                       │");
    println!("├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  MFCC[0]           (Fisher: 1.844 #2)    {:>10.4}                       │",
        features.mfcc_0
    );
    println!(
        "│  MFCC[1]           (Fisher: 1.389 #3)    {:>10.4}                       │",
        features.mfcc_1
    );
    println!(
        "│  MFCC[3]           (Fisher: 0.268 #8)    {:>10.4}                       │",
        features.mfcc_3
    );
    println!(
        "│  MFCC[4]           (Fisher: 0.257 #9)    {:>10.4}                       │",
        features.mfcc_4
    );
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ TIMBRE FEATURES (2D)                                                      │");
    println!("├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Spectral Flux     (Fisher: 0.701 #4)    {:>10.4}                       │",
        features.spectral_flux
    );
    println!(
        "│  HNR               (Fisher: 0.639 #5)    {:>10.4} dB                    │",
        features.hnr
    );
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ TEMPORAL FEATURES (3D)                                                    │");
    println!("├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Decay Time        (Fisher: 0.427 #7)    {:>10.4} ms                   │",
        features.decay_time_ms
    );
    println!(
        "│  Sustain Level     (Fisher: 0.192 #11)   {:>10.4}                       │",
        features.sustain_level
    );
    println!(
        "│  Attack Time       (Fisher: 0.184 #13)   {:>10.4} ms                   │",
        features.attack_time_ms
    );
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ RHYTHM FEATURES (2D)                                                      │");
    println!("├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  ICI CV            (Fisher: 0.215 #10)   {:>10.4}                       │",
        features.ici_cv
    );
    println!(
        "│  Onset Rate        (Fisher: 0.190 #12)   {:>10.4} Hz                    │",
        features.onset_rate_hz
    );
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ MODULATION FEATURES (1D)                                                  │");
    println!("├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Vibrato Rate      (Fisher: 0.154 #14)   {:>10.4} Hz                    │",
        features.vibrato_rate_hz
    );
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ PERTURBATION FEATURES (1D)                                               │");
    println!("├─────────────────────────────────────────────────────────────────────────┤");
    println!(
        "│  Shimmer           (Fisher: 0.140 #15)   {:>10.4}                       │",
        features.shimmer
    );
    println!("└─────────────────────────────────────────────────────────────────────────┘");
    println!();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║         Marmoset-Optimized 15D Feature Extraction Demo                     ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("This example demonstrates RFE-optimized feature extraction for marmoset");
    println!("call type classification using the 15 most discriminative features.");
    println!();
    println!("Features ranked by Fisher score (discriminative power):");
    println!("  1. rms_energy        (1.914)  9.  decay_time_ms    (0.427)");
    println!("  2. mfcc_0            (1.844) 10.  ici_cv           (0.215)");
    println!("  3. mfcc_1            (1.389) 11.  sustain_level    (0.192)");
    println!("  4. spectral_flux     (0.701) 12.  onset_rate_hz    (0.190)");
    println!("  5. hnr               (0.639) 13.  attack_time_ms   (0.184)");
    println!("  6. vibrato_depth     (0.631) 14.  vibrato_rate_hz  (0.154)");
    println!("  7. mfcc_3            (0.268) 15.  shimmer          (0.140)");
    println!("  8. mfcc_4            (0.257)");
    println!();

    // Create extractor
    let sample_rate = 96000; // Common for marmoset recordings
    let extractor = MicroDynamicsExtractor::new(sample_rate);

    println!(
        "Created MicroDynamicsExtractor with sample rate: {} Hz",
        sample_rate
    );
    println!();

    // Check if we have a specific file to analyze
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 {
        // Analyze specific file
        let file_path = Path::new(&args[1]);
        if !file_path.exists() {
            println!("❌ File not found: {}", file_path.display());
            return Ok(());
        }

        println!("Analyzing file: {}", file_path.display());
        println!("---");

        match load_flac_file(file_path) {
            Ok(audio) => {
                println!(
                    "Loaded {} samples ({} ms)",
                    audio.len(),
                    audio.len() as f32 / sample_rate as f32 * 1000.0
                );
                println!();

                // Extract 15D features
                match extractor.extract_15d_marmoset(&audio) {
                    Ok(features) => {
                        // Validate features
                        if let Err(e) = features.validate() {
                            println!("⚠️  Feature validation warning: {}", e);
                            println!();
                        }

                        // Display features
                        display_features(
                            &features,
                            file_path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("unknown"),
                        );
                    }
                    Err(e) => {
                        println!("❌ Feature extraction failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("❌ Failed to load audio: {}", e);
            }
        }
    } else {
        // Generate synthetic examples for each call type
        println!("No file provided. Generating synthetic examples for each call type...");
        println!("---");
        println!();

        for call_type in [
            MarmosetCallType::Phee,
            MarmosetCallType::Twitter,
            MarmosetCallType::Trill,
            MarmosetCallType::Tsik,
            MarmosetCallType::Seep,
            MarmosetCallType::Infant,
        ] {
            // Generate synthetic audio for this call type
            let (duration_ms, frequency_hz, modulation) = match call_type {
                MarmosetCallType::Phee => (300.0, 9000.0, 5.0),
                MarmosetCallType::Twitter => (80.0, 11000.0, 15.0),
                MarmosetCallType::Trill => (150.0, 9500.0, 25.0),
                MarmosetCallType::Tsik => (40.0, 10000.0, 0.0),
                MarmosetCallType::Seep => (200.0, 7000.0, 2.0),
                MarmosetCallType::Infant => (250.0, 12000.0, 10.0),
            };

            let num_samples = (duration_ms / 1000.0 * sample_rate as f32) as usize;
            let mut audio = vec![0.0f32; num_samples];

            for (i, sample) in audio.iter_mut().enumerate() {
                let t = i as f32 / sample_rate as f32;
                let base = (2.0 * std::f32::consts::PI * frequency_hz * t).sin();
                let mod_depth = match call_type {
                    MarmosetCallType::Trill => 0.3,
                    MarmosetCallType::Infant => 0.2,
                    _ => 0.05,
                };
                let modulation = if modulation > 0.0 {
                    1.0 + mod_depth * (2.0 * std::f32::consts::PI * modulation * t).sin()
                } else {
                    1.0
                };
                *sample = base * modulation * 0.5; // Scale to reasonable amplitude
            }

            // Extract features
            match extractor.extract_15d_marmoset(&audio) {
                Ok(features) => {
                    display_features(&features, &format!("SYNTHETIC_{}.flac", call_type.name()));
                }
                Err(e) => {
                    println!(
                        "❌ Feature extraction failed for {}: {}",
                        call_type.name(),
                        e
                    );
                }
            }

            println!();
            println!("---");
            println!();
        }
    }

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                              SUMMARY                                      ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("✅ Marmoset-optimized 15D feature extraction complete!");
    println!();
    println!("Key Benefits:");
    println!("  • 50% dimensionality reduction vs 30D (15 vs 30 features)");
    println!("  • Optimized for marmoset call type discrimination");
    println!("  • Faster computation for real-time applications");
    println!("  • Reduced overfitting potential in ML models");
    println!();
    println!("Usage Examples:");
    println!("  • Call type classification (Phee, Twitter, Trill, Tsik, Seep, Infant)");
    println!("  • Cross-call-type syntactic analysis");
    println!("  • Real-time marmoset vocalization monitoring");
    println!();

    Ok(())
}
