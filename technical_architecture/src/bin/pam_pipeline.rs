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
    // Phase 1: Streaming
    DebounceTimer, NeuralBoundaryDetector, RealTimeTimestamp, StreamingBuffer, StreamingConfig,
    BoundaryDetectorConfig,
    // Phase 2: Routing
    AcousticGroup, PAMRouter, PAMRouterConfig, PAMResult,
    // Phase 4: Active Learning
    ActiveLearningConfig, DetectionPayload, flag_for_active_learning,
};

/// PAM Pipeline Configuration
#[derive(Parser, Debug)]
#[command(
    name = "pam_pipeline",
    about = "Passive Acoustic Monitoring Pipeline",
    version
)]
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
        };
        let boundary_detector = NeuralBoundaryDetector::with_config(boundary_config);

        let router_config = PAMRouterConfig {
            confidence_threshold: args.threshold,
            active_learning_low: args.al_low,
            active_learning_high: args.al_high,
            models_dir: PathBuf::from("specialist_rf_models"),
        };
        let router = PAMRouter::with_config(router_config)
            .context("Failed to initialize PAM router")?;

        let al_config = ActiveLearningConfig::new(args.al_low, args.al_high);

        let debounce = DebounceTimer::new(args.min_phrase_duration_ms, args.sample_rate);

        Ok(Self {
            buffer,
            boundary_detector,
            router,
            al_config,
            debounce,
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
        let phrases = technical_architecture::segment_into_phrases(
            &buffer_samples,
            &boundaries,
            self.sample_rate,
        );

        for phrase_samples in phrases {
            // Debounce check
            let current_sample = self.buffer.total_samples().saturating_sub(phrase_samples.len());
            if !self.debounce.check_and_update(current_sample) {
                continue;
            }

            // Phase 2: Extract features and route
            // (In production, this would use actual 112D feature extraction)
            let features_112d = extract_features_placeholder(&phrase_samples);

            // Infer acoustic group from features
            let group = infer_acoustic_group_from_features(&features_112d);

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

/// Extract 112D features (placeholder implementation)
///
/// In production, this would use the full feature extraction pipeline
/// from taxonomic_router.rs (46D Physics + 30D Macro + 36D Micro)
fn extract_features_placeholder(audio: &[f32]) -> Vec<f32> {
    // Placeholder: return 112 features based on simple audio statistics
    let mut features = vec![0.0f32; 112];

    if audio.is_empty() {
        return features;
    }

    // Basic statistics as placeholder features
    let mean: f32 = audio.iter().sum::<f32>() / audio.len() as f32;
    let variance: f32 = audio.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / audio.len() as f32;
    let rms = variance.sqrt();

    // Fill first few features with basic stats
    features[0] = mean;
    features[1] = rms;
    features[2] = audio.len() as f32;
    features[3] = variance;

    // Compute zero-crossing rate
    let zcr = audio
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count() as f32
        / audio.len() as f32;
    features[4] = zcr;

    features
}

/// Infer acoustic group from features
///
/// In production, this would use a trained gatekeeper classifier
fn infer_acoustic_group_from_features(features: &[f32]) -> AcousticGroup {
    // Placeholder: use ZCR to roughly categorize
    let zcr = features.get(4).copied().unwrap_or(0.0);
    let rms = features.get(1).copied().unwrap_or(0.0);

    // High ZCR suggests high frequency (ultrasonic or birds)
    if zcr > 0.3 && rms > 0.1 {
        AcousticGroup::UltrasonicMammal
    } else if zcr > 0.2 {
        AcousticGroup::BirdHighFreq
    } else if rms > 0.5 {
        AcousticGroup::MarineWhistle
    } else {
        AcousticGroup::Unknown
    }
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
        let samples: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<f32>().ok())
            .collect();

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
    let data = std::fs::read(input_path)
        .with_context(|| format!("Failed to read input file: {:?}", input_path))?;

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
    fn test_extract_features_placeholder() {
        let audio = vec![0.5f32; 1024];
        let features = extract_features_placeholder(&audio);

        assert_eq!(features.len(), 112);
        // Mean should be ~0.5 (index 0)
        assert!((features[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_infer_acoustic_group() {
        let high_zcr = vec![0.0, 0.0, 0.0, 0.0, 0.5]; // High ZCR
        let group = infer_acoustic_group_from_features(&high_zcr);
        assert!(matches!(
            group,
            AcousticGroup::UltrasonicMammal | AcousticGroup::BirdHighFreq
        ));

        let low_features = vec![0.0, 0.05, 0.0, 0.0, 0.1]; // Low ZCR, low RMS
        let group = infer_acoustic_group_from_features(&low_features);
        assert!(matches!(group, AcousticGroup::Unknown));
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
