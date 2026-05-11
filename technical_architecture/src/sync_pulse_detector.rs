//! Ultrasonic Sync Pulse Detector
//! ===============================
//!
//! Detects 80kHz ultrasonic sync pulses in synthesized audio output for
//! round-trip latency measurement. Uses cross-correlation for robust
//! detection under noise.
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use anyhow::Result;
use log::debug;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Configuration for sync pulse detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncPulseDetectorConfig {
    /// Audio sample rate in Hz
    pub sample_rate: u32,

    /// Expected pulse frequency in Hz
    pub pulse_frequency_hz: u32,

    /// Pulse duration in milliseconds
    pub pulse_duration_ms: f32,

    /// Minimum SNR threshold for detection (dB)
    pub min_snr_db: f32,

    /// Correlation threshold (0.0-1.0)
    pub correlation_threshold: f32,

    /// Debounce time in milliseconds (prevent duplicate detections)
    pub debounce_ms: u64,
}

impl Default for SyncPulseDetectorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            pulse_frequency_hz: 80000,
            pulse_duration_ms: 1.0,
            min_snr_db: 10.0,
            correlation_threshold: 0.7,
            debounce_ms: 100,
        }
    }
}

/// Detected sync pulse
#[derive(Debug, Clone)]
pub struct DetectedPulse {
    /// Pulse ID (must be matched with injection records)
    pub pulse_id: Option<u64>,

    /// Detection timestamp in nanoseconds
    pub detection_time_ns: u64,

    /// Sample index where pulse was detected
    pub sample_index: usize,

    /// Detected frequency in Hz
    pub frequency_hz: f32,

    /// Signal amplitude
    pub amplitude: f32,

    /// Signal-to-noise ratio in dB
    pub snr_db: f32,

    /// Detection confidence (0-1)
    pub confidence: f32,
}

/// Ultrasonic sync pulse detector
pub struct SyncPulseDetector {
    config: SyncPulseDetectorConfig,
    enabled: Arc<AtomicBool>,
    last_detection_ns: AtomicU64,
    next_pulse_id_expectation: AtomicU64,

    /// Reference pulse template for correlation
    reference_pulse: Vec<f32>,
}

impl SyncPulseDetector {
    /// Create a new sync pulse detector
    pub fn new(config: SyncPulseDetectorConfig) -> Self {
        let pulse_samples = (config.pulse_duration_ms / 1000.0 * config.sample_rate as f32) as usize;
        let mut reference_pulse = Vec::with_capacity(pulse_samples);

        // Generate reference sine wave with Hann window
        let amplitude = 1.0;
        for i in 0..pulse_samples {
            let t = i as f32 / config.sample_rate as f32;
            let phase = 2.0 * std::f32::consts::PI * config.pulse_frequency_hz as f32 * t;
            let window = Self::hann_window(i, pulse_samples);
            reference_pulse.push(amplitude * phase.sin() * window);
        }

        Self {
            config,
            enabled: Arc::new(AtomicBool::new(true)),
            last_detection_ns: AtomicU64::new(0),
            next_pulse_id_expectation: AtomicU64::new(0),
            reference_pulse,
        }
    }

    /// Generate Hann window
    fn hann_window(i: usize, n: usize) -> f32 {
        if n <= 1 {
            return 1.0;
        }
        0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos())
    }

    /// Detect pulses in audio buffer
    ///
    /// Returns list of detected pulses
    pub fn detect_pulses(
        &self,
        audio: &[f32],
        detection_time_ns: u64,
    ) -> Vec<DetectedPulse> {
        if !self.enabled.load(Ordering::Relaxed) {
            return vec![];
        }

        let debounce_ns = self.config.debounce_ms * 1_000_000;
        let last_detection = self.last_detection_ns.load(Ordering::Relaxed);

        // Check debounce
        if detection_time_ns.saturating_sub(last_detection) < debounce_ns {
            return vec![];
        }

        let mut detections = vec![];

        // Compute cross-correlation
        let correlation = self.cross_correlate(audio);

        // Find peaks
        let peaks = self.find_peaks(&correlation);

        for (sample_idx, corr_value) in peaks {
            if corr_value < self.config.correlation_threshold {
                continue;
            }

            // Extract signal region
            let signal_region = self.extract_region(audio, sample_idx);

            // Calculate SNR
            let snr_db = self.calculate_snr(audio, sample_idx, &signal_region);

            if snr_db < self.config.min_snr_db {
                continue;
            }

            // Calculate amplitude
            let amplitude = signal_region.iter()
                .map(|&v| v.abs())
                .fold(0.0_f32, |a, b| a.max(b));

            // Calculate confidence from SNR
            let confidence = (snr_db - self.config.min_snr_db).max(0.0) / 20.0;
            let confidence = confidence.min(1.0);

            // Expected pulse ID
            let expected_id = self.next_pulse_id_expectation.load(Ordering::Relaxed);

            let detection = DetectedPulse {
                pulse_id: Some(expected_id),
                detection_time_ns,
                sample_index: sample_idx,
                frequency_hz: self.config.pulse_frequency_hz as f32,
                amplitude,
                snr_db,
                confidence,
            };

            detections.push(detection);

            // Update expectation
            self.next_pulse_id_expectation.fetch_add(1, Ordering::Relaxed);
        }

        if !detections.is_empty() {
            self.last_detection_ns.store(detection_time_ns, Ordering::Relaxed);
        }

        for d in &detections {
            debug!(
                "Detected sync pulse {} at sample {} (SNR={:.1}dB, conf={:.2})",
                d.pulse_id.unwrap_or(0), d.sample_index, d.snr_db, d.confidence
            );
        }

        detections
    }

    /// Compute cross-correlation with reference pulse
    fn cross_correlate(&self, audio: &[f32]) -> Vec<f32> {
        let pulse_len = self.reference_pulse.len();
        if audio.len() < pulse_len {
            return vec![0.0];
        }

        let mut correlation = vec![0.0; audio.len() - pulse_len + 1];

        // Normalize reference
        let ref_mean: f32 = self.reference_pulse.iter().sum::<f32>() / pulse_len as f32;
        let ref_std: f32 = (self.reference_pulse.iter()
            .map(|&v| (v - ref_mean).powi(2))
            .sum::<f32>() / pulse_len as f32)
            .sqrt()
            .max(0.001);

        for i in 0..correlation.len() {
            let window = &audio[i..i + pulse_len];

            let mean: f32 = window.iter().sum::<f32>() / pulse_len as f32;
            let std: f32 = (window.iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f32>() / pulse_len as f32)
                .sqrt()
                .max(0.001);

            // Compute correlation
            let corr: f32 = window.iter()
                .zip(self.reference_pulse.iter())
                .map(|(&a, &b)| (a - mean) * (b - ref_mean))
                .sum::<f32>() / (pulse_len as f32 * std * ref_std);

            correlation[i] = corr;
        }

        correlation
    }

    /// Find peaks in correlation array above threshold
    fn find_peaks(&self, correlation: &[f32]) -> Vec<(usize, f32)> {
        let mut peaks = vec![];

        if correlation.len() < 3 {
            return peaks;
        }

        // Simple peak detection
        for i in 1..correlation.len() - 1 {
            if correlation[i] > correlation[i - 1] && correlation[i] > correlation[i + 1] {
                peaks.push((i, correlation[i]));
            }
        }

        // Sort by correlation value
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        peaks
    }

    /// Extract signal region around peak
    fn extract_region(&self, audio: &[f32], center: usize) -> Vec<f32> {
        let pulse_len = self.reference_pulse.len();
        let start = center.saturating_sub(pulse_len / 2);
        let end = (center + pulse_len / 2).min(audio.len());

        audio[start..end].to_vec()
    }

    /// Calculate signal-to-noise ratio
    fn calculate_snr(&self, audio: &[f32], peak_idx: usize, signal: &[f32]) -> f32 {
        // Signal power
        let signal_power: f32 = signal.iter()
            .map(|&v| v.powi(2))
            .sum::<f32>() / signal.len() as f32;

        // Noise power (region before signal)
        let noise_start = peak_idx.saturating_sub(self.reference_pulse.len());
        let noise_end = peak_idx;
        let noise_region = if noise_end > noise_start && noise_start < audio.len() {
            &audio[noise_start..noise_end]
        } else {
            // Fallback: use last part of audio
            let n = audio.len().min(self.reference_pulse.len());
            &audio[audio.len() - n..]
        };

        let noise_power: f32 = if !noise_region.is_empty() {
            noise_region.iter()
                .map(|&v| v.powi(2))
                .sum::<f32>() / noise_region.len() as f32
        } else {
            0.001
        };

        let snr = signal_power / noise_power.max(0.001);
        10.0 * snr.log10().max(0.0)
    }

    /// Enable detection
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    /// Disable detection
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    /// Set expected pulse ID
    pub fn set_expected_pulse_id(&self, pulse_id: u64) {
        self.next_pulse_id_expectation.store(pulse_id, Ordering::Relaxed);
    }

    /// Reset state
    pub fn reset(&self) {
        self.last_detection_ns.store(0, Ordering::Relaxed);
        self.next_pulse_id_expectation.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let config = SyncPulseDetectorConfig::default();
        let detector = SyncPulseDetector::new(config);

        assert!(detector.enabled.load(Ordering::Relaxed));
    }

    #[test]
    fn test_enable_disable() {
        let config = SyncPulseDetectorConfig::default();
        let detector = SyncPulseDetector::new(config);

        detector.disable();
        assert!(!detector.enabled.load(Ordering::Relaxed));

        detector.enable();
        assert!(detector.enabled.load(Ordering::Relaxed));
    }

    #[test]
    fn test_set_expected_pulse_id() {
        let config = SyncPulseDetectorConfig::default();
        let detector = SyncPulseDetector::new(config);

        detector.set_expected_pulse_id(42);
        assert_eq!(detector.next_pulse_id_expectation.load(Ordering::Relaxed), 42);
    }

    #[test]
    fn test_reset() {
        let config = SyncPulseDetectorConfig::default();
        let detector = SyncPulseDetector::new(config);

        detector.last_detection_ns.store(1000, Ordering::Relaxed);
        detector.next_pulse_id_expectation.store(10, Ordering::Relaxed);

        detector.reset();

        assert_eq!(detector.last_detection_ns.load(Ordering::Relaxed), 0);
        assert_eq!(detector.next_pulse_id_expectation.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_cross_correlate() {
        let config = SyncPulseDetectorConfig::default();
        let detector = SyncPulseDetector::new(config);

        // Create a simple test signal
        let mut audio = vec![0.0; 1000];
        // Insert a matching pulse at index 500
        for (i, sample) in detector.reference_pulse.iter().enumerate() {
            if 500 + i < audio.len() {
                audio[500 + i] = *sample;
            }
        }

        let correlation = detector.cross_correlate(&audio);

        // Should have a peak near index 500
        let peak_idx = correlation.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();

        assert!((peak_idx as i32 - 500).abs() < 10);
    }
}
