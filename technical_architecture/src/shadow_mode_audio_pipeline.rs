//! Shadow Mode Audio Pipeline
//! ============================
//!
//! Main orchestration for E2E shadow mode testing. Integrates Predictive NBD,
//! sync pulse injection/detection, BioMAE extraction, and loopback mixing
//! for complete round-trip latency validation.
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

pub use crate::digital_loopback_mixer::{DigitalLoopbackMixer, LoopbackMixerConfig};
pub use crate::soak_test_telemetry::{SoakTestTelemetry, SoakTestTelemetryConfig};
pub use crate::sync_pulse_detector::{DetectedPulse, SyncPulseDetector, SyncPulseDetectorConfig};
pub use crate::sync_pulse_injector::{PulseInjectionRecord, SyncPulseInjector, SyncPulseConfig};

use crate::ptp::PtpTimestamp;
use anyhow::Result;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Configuration for shadow mode pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowModePipelineConfig {
    /// Audio sample rate
    pub sample_rate: u32,

    /// Frame size in milliseconds
    pub frame_size_ms: f32,

    /// Sync pulse configuration
    pub sync_pulse: SyncPulseConfig,

    /// Loopback mixer configuration
    pub loopback: LoopbackMixerConfig,

    /// Predictive NBD configuration (Python side integration)
    pub nbd_boundary_threshold: f32,
    pub nbd_slow_decay: f32,
    pub nbd_fast_decay: f32,

    /// BioMAE model path (if using Rust extraction)
    pub biomae_path: Option<String>,

    /// Enable shadow mode output logging
    pub enable_output_logging: bool,

    /// Output log directory
    pub output_log_dir: Option<String>,
}

impl Default for ShadowModePipelineConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            frame_size_ms: 10.0,
            sync_pulse: SyncPulseConfig::default(),
            loopback: LoopbackMixerConfig::default(),
            nbd_boundary_threshold: 2.5,
            nbd_slow_decay: 0.99,
            nbd_fast_decay: 0.9,
            biomae_path: None,
            enable_output_logging: false,
            output_log_dir: None,
        }
    }
}

/// Output from shadow mode pipeline processing
#[derive(Debug, Clone)]
pub struct ShadowModeOutput {
    /// PTP timestamp of processing
    pub ptp_timestamp: PtpTimestamp,

    /// Detected boundaries from Predictive NBD
    pub boundaries: Vec<BoundaryEvent>,

    /// Extracted 112D features (if using BioMAE)
    pub features_112d: Option<Vec<f32>>,

    /// Confidence score from NBD
    pub confidence: f32,

    /// Sync pulse injection (if any)
    pub pulse_injection: Option<PulseInjectionRecord>,

    /// Processed audio with loopback applied (if enabled)
    pub processed_audio: Vec<f32>,
}

/// Detected semantic boundary event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryEvent {
    /// Timestamp of boundary
    pub timestamp_ns: u64,

    /// Boundary type
    pub boundary_type: String,

    /// Prediction error at boundary
    pub prediction_error: f32,

    /// Confidence score
    pub confidence: f32,

    /// Duration of segment
    pub segment_duration_ms: f32,

    /// Latency from detection
    pub latency_ms: f32,
}

/// Statistics from shadow mode operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowModeStatistics {
    /// Total frames processed
    pub frames_processed: u64,

    /// Total boundaries detected
    pub boundaries_detected: u64,

    /// Average confidence
    pub average_confidence: f32,

    /// Sync pulses injected
    pub pulses_injected: u64,

    /// Sync pulses detected
    pub pulses_detected: u64,

    /// Processing time statistics (ms)
    pub processing_time_p50: f32,
    pub processing_time_p95: f32,
    pub processing_time_p99: f32,
}

/// Shadow mode audio pipeline
///
/// Orchestrates the complete E2E testing pipeline:
/// 1. Sync pulse injection (Rust)
/// 2. Predictive NBD boundary detection (Python integration)
/// 3. BioMAE feature extraction (optional Rust)
/// 4. Digital loopback mixing (for mirror test)
/// 5. Sync pulse detection (Rust)
pub struct ShadowModePipeline {
    config: ShadowModePipelineConfig,
    sync_injector: SyncPulseInjector,
    sync_detector: SyncPulseDetector,
    loopback_mixer: DigitalLoopbackMixer,
    enabled: Arc<AtomicBool>,

    // Statistics
    frames_processed: Arc<AtomicU64>,
    boundaries_detected: Arc<AtomicU64>,
    processing_times: Arc<Mutex<Vec<f32>>>,

    // Output state
    output_buffer: Arc<Mutex<Vec<f32>>>,
}

use std::sync::Mutex;

impl ShadowModePipeline {
    /// Create a new shadow mode pipeline
    pub fn new(config: ShadowModePipelineConfig) -> Result<Self> {
        let sync_injector = SyncPulseInjector::new(config.sync_pulse.clone());

        // Configure detector with matching settings
        let detector_config = SyncPulseDetectorConfig {
            sample_rate: config.sample_rate,
            pulse_frequency_hz: config.sync_pulse.pulse_frequency_hz,
            pulse_duration_ms: config.sync_pulse.pulse_duration_ms,
            ..Default::default()
        };
        let sync_detector = SyncPulseDetector::new(detector_config);

        let loopback_mixer = DigitalLoopbackMixer::new(config.loopback.clone());

        Ok(Self {
            config,
            sync_injector,
            sync_detector,
            loopback_mixer,
            enabled: Arc::new(AtomicBool::new(true)),
            frames_processed: Arc::new(AtomicU64::new(0)),
            boundaries_detected: Arc::new(AtomicU64::new(0)),
            processing_times: Arc::new(Mutex::new(vec![])),
            output_buffer: Arc::new(Mutex::new(vec![])),
        })
    }

    /// Process a single audio frame through the pipeline
    ///
    /// This is the main entry point for shadow mode processing.
    /// In production, this would be called for each 10ms audio frame.
    pub fn process_frame(
        &mut self,
        audio: &[f32],
    ) -> Result<ShadowModeOutput> {
        if !self.enabled.load(Ordering::Relaxed) {
            return Ok(ShadowModeOutput {
                ptp_timestamp: PtpTimestamp::now(),
                boundaries: vec![],
                features_112d: None,
                confidence: 0.0,
                pulse_injection: None,
                processed_audio: audio.to_vec(),
            });
        }

        let start = std::time::Instant::now();
        let ptp_now = PtpTimestamp::now();
        let now_ns = ptp_now.as_nanos() as u64;

        // Create mutable buffer
        let mut audio_buffer = audio.to_vec();

        // Step 1: Inject sync pulse if needed
        let pulse_injection = self.sync_injector.inject_into_buffer(
            &mut audio_buffer,
            ptp_now,
            now_ns,
        );

        // Step 2: Apply loopback if enabled (for mirror test)
        // In production, this would mix synthesized output back in
        if self.loopback_mixer.is_enabled() {
            // For now, skip (would need output from synthesis)
            // self.loopback_mixer.process(&mut audio_buffer, &output)?;
        }

        // Step 3: Process through Predictive NBD (Python integration)
        // In production, this would call Python via ZMQ
        let boundaries = vec![];  // Placeholder

        // Step 4: Extract 112D features (if BioMAE available)
        let features_112d = None;  // Placeholder

        // Step 5: Detect sync pulses in output (if we had output)
        // For now, check in input (bypass detection)
        let _detections = self.sync_detector.detect_pulses(&audio_buffer, now_ns);

        // Track processing time
        let processing_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        {
            let mut times = self.processing_times.lock().unwrap();
            times.push(processing_time_ms as f32);
            // Keep only last 1000 samples
            if times.len() > 1000 {
                times.remove(0);
            }
        }

        self.frames_processed.fetch_add(1, Ordering::Relaxed);

        Ok(ShadowModeOutput {
            ptp_timestamp: ptp_now,
            boundaries,
            features_112d,
            confidence: 0.0,  // Would come from NBD
            pulse_injection,
            processed_audio: audio_buffer,
        })
    }

    /// Enable the pipeline
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
        self.sync_injector.enable();
        self.sync_detector.enable();
        self.loopback_mixer.enable();
        info!("Shadow mode pipeline enabled");
    }

    /// Disable the pipeline
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
        self.sync_injector.disable();
        self.sync_detector.disable();
        self.loopback_mixer.disable();
        info!("Shadow mode pipeline disabled");
    }

    /// Enable digital loopback (for mirror test)
    pub fn enable_loopback(&self) {
        self.loopback_mixer.enable();
        info!("Digital loopback enabled");
    }

    /// Disable digital loopback
    pub fn disable_loopback(&self) {
        self.loopback_mixer.disable();
        info!("Digital loopback disabled");
    }

    /// Get pipeline statistics
    pub fn get_statistics(&self) -> ShadowModeStatistics {
        let times = self.processing_times.lock().unwrap();

        let (p50, p95, p99) = if !times.is_empty() {
            let mut sorted = times.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let len = sorted.len();
            (
                sorted[len * 50 / 100],
                sorted[len * 95 / 100],
                sorted[len * 99 / 100],
            )
        } else {
            (0.0, 0.0, 0.0)
        };

        ShadowModeStatistics {
            frames_processed: self.frames_processed.load(Ordering::Relaxed),
            boundaries_detected: self.boundaries_detected.load(Ordering::Relaxed),
            average_confidence: 0.0,  // Would be calculated
            pulses_injected: self.sync_injector.peek_next_pulse_id(),
            pulses_detected: 0,  // Would track detections
            processing_time_p50: p50,
            processing_time_p95: p95,
            processing_time_p99: p99,
        }
    }

    /// Reset statistics
    pub fn reset_statistics(&self) {
        self.frames_processed.store(0, Ordering::Relaxed);
        self.boundaries_detected.store(0, Ordering::Relaxed);
        self.processing_times.lock().unwrap().clear();
        self.sync_injector.reset_pulse_id();
        self.sync_detector.reset();
        info!("Statistics reset");
    }

    /// Check if pipeline is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Get sync injector reference
    pub fn sync_injector(&self) -> &SyncPulseInjector {
        &self.sync_injector
    }

    /// Get sync detector reference
    pub fn sync_detector(&self) -> &SyncPulseDetector {
        &self.sync_detector
    }

    /// Get loopback mixer reference
    pub fn loopback_mixer(&self) -> &DigitalLoopbackMixer {
        &self.loopback_mixer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let config = ShadowModePipelineConfig::default();
        let pipeline = ShadowModePipeline::new(config);

        assert!(pipeline.is_ok());
        let pipeline = pipeline.unwrap();
        assert!(pipeline.is_enabled());
    }

    #[test]
    fn test_process_frame() {
        let config = ShadowModePipelineConfig::default();
        let mut pipeline = ShadowModePipeline::new(config).unwrap();

        let audio = vec![0.0; 480];  // 10ms @ 48kHz
        let result = pipeline.process_frame(&audio);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.processed_audio.len(), audio.len());
    }

    #[test]
    fn test_enable_disable() {
        let config = ShadowModePipelineConfig::default();
        let pipeline = ShadowModePipeline::new(config).unwrap();

        pipeline.disable();
        assert!(!pipeline.is_enabled());

        pipeline.enable();
        assert!(pipeline.is_enabled());
    }

    #[test]
    fn test_loopback_control() {
        let config = ShadowModePipelineConfig::default();
        let pipeline = ShadowModePipeline::new(config).unwrap();

        assert!(!pipeline.loopback_mixer().is_enabled());

        pipeline.enable_loopback();
        assert!(pipeline.loopback_mixer().is_enabled());

        pipeline.disable_loopback();
        assert!(!pipeline.loopback_mixer().is_enabled());
    }

    #[test]
    fn test_statistics() {
        let config = ShadowModePipelineConfig::default();
        let pipeline = ShadowModePipeline::new(config).unwrap();

        let stats = pipeline.get_statistics();
        assert_eq!(stats.frames_processed, 0);
        assert_eq!(stats.boundaries_detected, 0);
    }

    #[test]
    fn test_reset_statistics() {
        let config = ShadowModePipelineConfig::default();
        let mut pipeline = ShadowModePipeline::new(config).unwrap();

        let audio = vec![0.0; 480];
        let _ = pipeline.process_frame(&audio);

        pipeline.reset_statistics();

        let stats = pipeline.get_statistics();
        assert_eq!(stats.frames_processed, 0);
    }
}
