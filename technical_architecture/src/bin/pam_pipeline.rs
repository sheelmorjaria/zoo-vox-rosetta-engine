//! Passive Acoustic Monitoring Pipeline
//! =====================================
//!
//! Integration binary that combines all PAM phases:
//! - Phase 1: Real-time stream ingestion with system clock timestamps
//! - Phase 2: 112D feature extraction and hierarchical routing
//! - Phase 3: Confidence threshold filtering
//! - Phase 4: Active learning flagging and JSON output
//!
//! # Usage
//!
//! ```bash
//! # Process audio from stdin (real-time mode)
//! cargo run --bin pam_pipeline -- --real-time
//!
//! # Process audio file
//! cargo run --bin pam_pipeline -- --input audio.wav
//!
//! # Custom confidence threshold
//! cargo run --bin pam_pipeline -- --threshold 1.5 --input audio.wav
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use std::io::{self, BufRead};
use std::path::PathBuf;

use technical_architecture::{
    flag_for_active_learning,
    // Phase 2: Routing
    AcousticGroup,
    // Phase 4: Active Learning
    ActiveLearningConfig,
    BoundaryDetectorConfig,
    // Phase 1: Streaming
    DebounceTimer,
    DetectionMode,
    DetectionPayload,
    // Phase 2: 112D Feature Extraction
    MicroDynamicsExtractor,
    NeuralBoundaryDetector,
    PAMResult,
    PAMRouter,
    PAMRouterConfig,
    RealTimeTimestamp,
    StreamingBuffer,
    StreamingConfig,
};

/// PAM Pipeline Configuration
#[derive(Parser, Debug)]
#[command(name = "pam_pipeline", about = "Passive Acoustic Monitoring Pipeline", version)]
struct Args {
    /// Input audio file (raw f32 samples)
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Real-time mode (read from stdin)
    #[arg(short, long)]
    real_time: bool,

    /// Confidence threshold for detection
    #[arg(short, long, default_value = "1.5")]
    threshold: f32,

    /// Sample rate in Hz
    #[arg(long, default_value = "44100")]
    sample_rate: u32,

    /// Hop size in samples
    #[arg(long, default_value = "512")]
    hop_size: usize,

    /// Minimum phrase duration in ms
    #[arg(long, default_value = "50.0")]
    min_phrase_duration_ms: f32,

    /// Active learning lower margin
    #[arg(long, default_value = "1.4")]
    al_low: f32,

    /// Active learning upper margin
    #[arg(long, default_value = "1.5")]
    al_high: f32,

    /// Output format
    #[arg(long, value_enum, default_value = "jsonl")]
    format: OutputFormat,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum OutputFormat {
    /// JSON Lines (one JSON object per line)
    Jsonl,
    /// Compact single-line JSON
    Json,
    /// Human-readable text
    Text,
}

/// PAM Pipeline State
struct PipelineState {
    /// Streaming buffer for audio ingestion
    buffer: StreamingBuffer,
    /// Neural boundary detector
    boundary_detector: NeuralBoundaryDetector,
    /// PAM router for classification
    router: PAMRouter,
    /// Active learning configuration
    al_config: ActiveLearningConfig,
    /// Debounce timer for detection
    debounce: DebounceTimer,
    /// 112D feature extractor (MicroDynamicsExtractor)
    feature_extractor: MicroDynamicsExtractor,
    /// Sample rate
    sample_rate: u32,
    /// Verbose mode
    verbose: bool,
}

impl PipelineState {
    fn new(args: &Args) -> Result<Self> {
        let streaming_config = StreamingConfig {
            hop_size: args.hop_size,
            sample_rate: args.sample_rate,
            buffer_duration_secs: 60.0,
            min_phrase_duration_ms: args.min_phrase_duration_ms,
        };

        let buffer = StreamingBuffer::with_config(streaming_config);

        let boundary_config = BoundaryDetectorConfig {
            hop_size: args.hop_size,
            sample_rate: args.sample_rate,
            min_phrase_duration_ms: args.min_phrase_duration_ms,
            threshold: 0.3, // Lower threshold for better detection
            smoothing_frames: 3,
            mode: DetectionMode::Phrase,
            max_phrase_duration_ms: 5000.0,
            smoothing_window_ms: 20.0,
            energy_weight: 0.5,
            spectral_weight: 0.5,
        };
        let boundary_detector = NeuralBoundaryDetector::with_config(boundary_config);

        let router_config = PAMRouterConfig {
            confidence_threshold: args.threshold,
            active_learning_low: args.al_low,
            active_learning_high: args.al_high,
            models_dir: PathBuf::from("specialist_rf_models"),
        };
        let router = PAMRouter::with_config(router_config).context("Failed to initialize PAM router")?;

        let al_config = ActiveLearningConfig::new(args.al_low, args.al_high);

        let debounce = DebounceTimer::new(args.min_phrase_duration_ms, args.sample_rate);

        // Initialize 112D feature extractor using MicroDynamicsExtractor
        let feature_extractor = MicroDynamicsExtractor::new(args.sample_rate);

        Ok(Self {
            buffer,
            boundary_detector,
            router,
            al_config,
            debounce,
            feature_extractor,
            sample_rate: args.sample_rate,
            verbose: args.verbose,
        })
    }

    /// Process a chunk of audio samples
    fn process_chunk(&mut self, samples: &[f32]) -> Result<Vec<DetectionPayload>> {
        let mut detections = Vec::new();

        // Phase 1: Add samples to streaming buffer
        let timestamp = self.buffer.add_samples(samples);

        if self.verbose {
            eprintln!(
                "[{:?}] Ingested {} samples ({:.1}ms)",
                timestamp.system_time,
                samples.len(),
                timestamp.duration_ms
            );
        }

        // Get current buffer for analysis
        let buffer_samples = self.buffer.get_all_samples();

        if buffer_samples.len() < self.sample_rate as usize {
            // Not enough samples yet
            return Ok(detections);
        }

        // Phase 1: Detect phrase boundaries
        let boundaries = self.boundary_detector.detect_boundaries(&buffer_samples);

        if boundaries.is_empty() {
            return Ok(detections);
        }

        if self.verbose {
            eprintln!("[Boundary] Detected {} phrase boundaries", boundaries.len());
        }

        // Process each phrase segment
        let phrases = technical_architecture::segment_into_phrases(&buffer_samples, &boundaries, self.sample_rate);

        for phrase_samples in phrases {
            // Debounce check
            let current_sample = self.buffer.total_samples().saturating_sub(phrase_samples.len());
            if !self.debounce.check_and_update(current_sample) {
                continue;
            }

            // Phase 2: Extract 112D Rosetta features using MicroDynamicsExtractor
            let features_112d = match extract_112d_features(&self.feature_extractor, &phrase_samples) {
                Ok(f) => f,
                Err(e) => {
                    if self.verbose {
                        eprintln!("[Error] Feature extraction failed: {}", e);
                    }
                    continue;
                }
            };

            // Route to acoustic specialist WITHOUT segment-to-segment bias
            // Uses ONLY the current segment's features to determine the group
            let group = route_acoustic_group_from_features(&features_112d);

            // Phase 2 & 3: Classify with threshold filtering
            match self.router.classify(&features_112d, group) {
                Ok(Some(result)) => {
                    // Phase 4: Create detection payload
                    let mut payload = create_detection_payload(&result, &timestamp);

                    // Phase 4: Check for active learning flagging
                    if flag_for_active_learning(result.confidence, &self.al_config) {
                        payload.flag_for_learning(Some(format!(
                            "uncertain_samples/{}_{}.bin",
                            payload.species.replace(' ', "_"),
                            payload.timestamp_ms
                        )));

                        if self.verbose {
                            eprintln!(
                                "[ActiveLearning] Flagged {} (confidence {:.2})",
                                payload.species, payload.confidence
                            );
                        }
                    }

                    detections.push(payload);
                }
                Ok(None) => {
                    // Below confidence threshold
                    if self.verbose {
                        eprintln!("[Filtered] Detection below threshold");
                    }
                }
                Err(e) => {
                    if self.verbose {
                        eprintln!("[Error] Classification error: {}", e);
                    }
                }
            }
        }

        Ok(detections)
    }
}

/// Extract 112D features using MicroDynamicsExtractor::extract_rosetta
///
/// Uses the full 112D Rosetta feature extraction pipeline:
/// - Layer 1 (46D): Base Physics - F0, duration, energy, harmonicity, envelope, MFCCs, spectral shape
/// - Layer 2 (30D): Macro Texture - Harmonic texture, pitch geometry, GLCM texture
/// - Layer 3 (36D): Micro Texture - Spectral derivatives, FM/AM bins, ICI bins, rhythm histogram
fn extract_112d_features(extractor: &MicroDynamicsExtractor, audio: &[f32]) -> Result<Vec<f32>> {
    if audio.is_empty() {
        // Return zero-initialized 112D vector for empty audio
        return Ok(vec![0.0f32; 112]);
    }

    // Use extract for 112D RosettaFeatures
    let rosetta_features = extractor
        .extract(audio)
        .context("Failed to extract 112D Rosetta features")?;

    // Convert to flat array for routing
    let features_112d = rosetta_features.to_array().to_vec();

    Ok(features_112d)
}

/// Route to acoustic group from 112D features WITHOUT segment-to-segment bias
///
/// This function determines the acoustic group SOLELY from the current segment's
/// 112D features, using the acoustic characteristics defined for each group.
///
/// Feature Indices (from RosettaFeatures):
/// - Index 0: mean_f0_hz - Fundamental frequency
/// - Index 1: duration_ms - Segment duration
/// - Index 4: zero_crossing_rate - Proxy for frequency content
/// - Index 26: spectral_centroid - Spectral brightness
///
/// Acoustic Groups are defined by:
/// - Frequency range (F0 / spectral centroid)
/// - Duration range
/// - Modulation patterns
fn route_acoustic_group_from_features(features: &[f32]) -> AcousticGroup {
    // Extract key features for routing (indices from RosettaFeatures::to_array)
    // Layer 1: Base Physics indices
    let mean_f0_hz = features.first().copied().unwrap_or(0.0);
    let duration_ms = features.get(1).copied().unwrap_or(0.0);
    let rms_energy = features.get(3).copied().unwrap_or(0.0);
    let zcr = features.get(4).copied().unwrap_or(0.0);
    let spectral_centroid = features.get(26).copied().unwrap_or(0.0);

    // Use spectral centroid (Hz) and F0 to determine frequency band
    // F0 is more reliable for tonal sounds, centroid for broadband
    let effective_freq = if mean_f0_hz > 100.0 {
        mean_f0_hz
    } else {
        spectral_centroid // Already in Hz from RosettaFeatures
    };

    // === ULTRASONIC MAMMALS (Bats): 20-80kHz, 5-100ms ===
    if effective_freq >= 20000.0 && (5.0..=100.0).contains(&duration_ms) {
        return AcousticGroup::UltrasonicMammal;
    }

    // === SONIC LONG MAMMALS (Baleen Whales): 20-5000Hz, 500-5000ms ===
    if (20.0..5000.0).contains(&effective_freq) && duration_ms >= 500.0 {
        return AcousticGroup::SonicLongMammal;
    }

    // === MARINE WHISTLE (Dolphins): 2-24kHz, 100-1000ms ===
    if (2000.0..24000.0).contains(&effective_freq) && (100.0..1000.0).contains(&duration_ms) {
        return AcousticGroup::MarineWhistle;
    }

    // === MARINE CLICK (Porpoises): broadband, impulsive, <2ms ===
    if duration_ms < 2.0 && zcr > 0.4 {
        return AcousticGroup::MarineClick;
    }

    // === MARINE MOAN: Low F0, long duration fallback ===
    if effective_freq < 500.0 && duration_ms >= 1000.0 {
        return AcousticGroup::MarineMoan;
    }

    // === BIRD HIGH FREQ (Songbirds): 4-8kHz, 50-500ms ===
    if (4000.0..8000.0).contains(&effective_freq) && (50.0..=500.0).contains(&duration_ms) {
        return AcousticGroup::BirdHighFreq;
    }

    // === BIRD LOW FREQ (Doves, Owls): 200-1000Hz, 100-1000ms ===
    if (200.0..1000.0).contains(&effective_freq) && duration_ms >= 100.0 {
        return AcousticGroup::BirdLowFreq;
    }

    // === BIRD MECHANICAL (Hummingbirds): broadband, 10-100ms ===
    if zcr > 0.3 && (10.0..=100.0).contains(&duration_ms) {
        return AcousticGroup::BirdMechanical;
    }

    // === INSECT WINGBEAT (Mosquitoes): 100-1000Hz, steady tone ===
    if (100.0..1000.0).contains(&effective_freq) && zcr < 0.15 && rms_energy > 0.1 {
        return AcousticGroup::InsectWingbeat;
    }

    // === INSECT STRIDULATION (Crickets): 2-10kHz, broadband pulses ===
    if (2000.0..10000.0).contains(&effective_freq) && zcr > 0.2 && duration_ms < 200.0 {
        return AcousticGroup::InsectStridulation;
    }

    // === AMPHIBIAN: 500-5000Hz, pulsed ===
    if (500.0..5000.0).contains(&effective_freq) && (50.0..=500.0).contains(&duration_ms) {
        return AcousticGroup::Amphibian;
    }

    // === SONIC SHORT MAMMAL (Primates): mid F0, variable ===
    if (100.0..8000.0).contains(&effective_freq) {
        return AcousticGroup::SonicShortMammal;
    }

    // Default to Unknown if no group matches
    AcousticGroup::Unknown
}

/// Create detection payload from PAM result
fn create_detection_payload(result: &PAMResult, timestamp: &RealTimeTimestamp) -> DetectionPayload {
    DetectionPayload::new(
        timestamp.as_millis_since_epoch(),
        result.species.clone(),
        result.confidence,
        result.acoustic_group.to_string(),
        format!("{:?}", result.taxon),
        result.inference_time_us,
    )
}

/// Output detection in the specified format
fn output_detection(detection: &DetectionPayload, format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Jsonl => {
            println!("{}", detection.to_json()?);
        }
        OutputFormat::Json => {
            let json = serde_json::to_string(detection)?;
            println!("{}", json);
        }
        OutputFormat::Text => {
            println!(
                "[{:.3}] {} ({}) - confidence: {:.2}, group: {}, inference: {}us{}",
                detection.timestamp_ms as f64 / 1000.0,
                detection.species,
                detection.taxon,
                detection.confidence,
                detection.acoustic_group,
                detection.inference_time_us,
                if detection.active_learning {
                    " [ACTIVE_LEARNING]"
                } else {
                    ""
                }
            );
        }
    }
    Ok(())
}

/// Run in real-time mode (read from stdin)
fn run_real_time(state: &mut PipelineState, format: OutputFormat) -> Result<()> {
    let stdin = io::stdin();
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = stdin.lock().read_line(&mut line)?;
        if bytes_read == 0 {
            break; // EOF
        }

        // Parse audio samples from input
        // Format: space-separated f32 values
        let samples: Vec<f32> = line.split_whitespace().filter_map(|s| s.parse::<f32>().ok()).collect();

        if samples.is_empty() {
            continue;
        }

        // Process chunk
        let detections = state.process_chunk(&samples)?;

        // Output detections
        for detection in detections {
            output_detection(&detection, format)?;
        }
    }

    Ok(())
}

/// Run in file mode
fn run_file(state: &mut PipelineState, input_path: &PathBuf, format: OutputFormat) -> Result<()> {
    // Read raw f32 samples from file
    let data = std::fs::read(input_path).with_context(|| format!("Failed to read input file: {:?}", input_path))?;

    // Convert bytes to f32 samples
    let samples: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
            f32::from_le_bytes(bytes)
        })
        .collect();

    if state.verbose {
        eprintln!("Loaded {} samples from {:?}", samples.len(), input_path);
    }

    // Process in chunks
    let chunk_size = state.buffer.config().hop_size * 10; // ~100ms chunks
    for chunk in samples.chunks(chunk_size) {
        let detections = state.process_chunk(chunk)?;

        for detection in detections {
            output_detection(&detection, format)?;
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize pipeline state
    let mut state = PipelineState::new(&args)?;

    if args.verbose {
        eprintln!("PAM Pipeline initialized");
        eprintln!("  Sample rate: {} Hz", args.sample_rate);
        eprintln!("  Hop size: {} samples", args.hop_size);
        eprintln!("  Confidence threshold: {}", args.threshold);
        eprintln!("  Active learning range: [{}, {})", args.al_low, args.al_high);
    }

    // Run appropriate mode
    if args.real_time {
        run_real_time(&mut state, args.format)
    } else if let Some(input_path) = &args.input {
        run_file(&mut state, input_path, args.format)
    } else {
        // Default: real-time mode
        if args.verbose {
            eprintln!("No input specified, running in real-time mode");
        }
        run_real_time(&mut state, args.format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;
    use technical_architecture::taxonomic_router::ConsolidatedTaxon;

    #[test]
    fn test_extract_112d_features() {
        let audio = vec![0.5f32; 1024];
        let extractor = MicroDynamicsExtractor::new(44100);
        let features = extract_112d_features(&extractor, &audio).unwrap();

        assert_eq!(features.len(), 112);
        // Features should be extracted (not all zeros for non-trivial audio)
        assert!(features.iter().any(|&f| f != 0.0));
    }

    #[test]
    fn test_route_acoustic_group() {
        // Test routing with synthetic features
        // 15kHz F0, 100ms duration → MarineWhistle (freq 2-24kHz, duration 100-1000ms)
        let marine = vec![15000.0, 100.0, 5000.0, 0.1, 0.5];
        let group = route_acoustic_group_from_features(&marine);
        assert!(matches!(group, AcousticGroup::MarineWhistle));

        // 500Hz F0, 200ms duration → BirdLowFreq (freq 200-1000Hz, duration >= 100ms)
        let low_freq = vec![500.0, 200.0, 200.0, 0.1, 0.1];
        let group = route_acoustic_group_from_features(&low_freq);
        assert!(matches!(group, AcousticGroup::BirdLowFreq));
    }

    #[test]
    fn test_pipeline_state_creation() {
        let args = Args {
            input: None,
            real_time: false,
            threshold: 1.5,
            sample_rate: 44100,
            hop_size: 512,
            min_phrase_duration_ms: 50.0,
            al_low: 1.4,
            al_high: 1.5,
            format: OutputFormat::Jsonl,
            verbose: false,
        };

        let state = PipelineState::new(&args);
        assert!(state.is_ok());
    }

    #[test]
    fn test_detection_payload_creation() {
        let result = PAMResult {
            species: "Test species".to_string(),
            confidence: 1.45,
            acoustic_group: AcousticGroup::MarineWhistle,
            features_112d: vec![0.0; 112],
            taxon: ConsolidatedTaxon::Mammal,
            inference_time_us: 500,
            active_learning: true,
        };

        let timestamp = RealTimeTimestamp::new(SystemTime::now(), 1000, 10.0);
        let payload = create_detection_payload(&result, &timestamp);

        // Basic fields should be copied
        assert_eq!(payload.species, "Test species");
        assert!((payload.confidence - 1.45).abs() < 0.01);
        // Note: active_learning is set separately via flag_for_learning(), not from PAMResult
        // The payload starts with active_learning: false
        assert!(!payload.active_learning);
    }

    #[test]
    fn test_active_learning_flagging_in_pipeline() {
        let config = ActiveLearningConfig::new(1.4, 1.5);

        // In range
        assert!(flag_for_active_learning(1.45, &config));

        // Out of range
        assert!(!flag_for_active_learning(1.3, &config));
        assert!(!flag_for_active_learning(1.5, &config));
    }
}
