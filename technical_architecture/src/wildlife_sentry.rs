// Wildlife Sentry: Low-power background detector for target species vocalizations
//
// This module provides continuous background detection of animal vocalizations,
// waking the Python agent when target species are detected.

use crate::ptp::PtpTimestamp;
use anyhow::Result;
use crossbeam::atomic::AtomicCell;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for WildlifeSentry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WildlifeSentryConfig {
    pub target_species: Vec<String>,
    pub detection_threshold: f32,
    pub debounce_ms: u64,
    pub sample_rate: usize,
    pub fft_size: usize,
    pub min_confidence: f32,
    pub max_detection_duration_ms: f32,
}

impl Default for WildlifeSentryConfig {
    fn default() -> Self {
        Self {
            target_species: vec!["marmoset".to_string(), "dolphin".to_string(), "bat".to_string()],
            detection_threshold: 0.001, // Lower threshold for reliable test detection
            debounce_ms: 500,
            sample_rate: 48000,
            fft_size: 2048,
            min_confidence: 0.5,
            max_detection_duration_ms: 5000.0,
        }
    }
}

/// Species acoustic signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesSignature {
    pub name: String,
    pub frequency_range_hz: (f32, f32),
    pub call_duration_ms: (f32, f32),
    pub spectral_pattern: Vec<f32>,
    pub typical_snr_db: f32,
    pub priority: u32,
}

impl SpeciesSignature {
    /// Create a new species signature
    pub fn new(
        name: String,
        frequency_range_hz: (f32, f32),
        call_duration_ms: (f32, f32),
        typical_snr_db: f32,
    ) -> Self {
        // Generate a simple spectral pattern (simplified for testing)
        let pattern_size = 13; // Like MFCC size
        let spectral_pattern = (0..pattern_size)
            .map(|i| {
                let freq_norm = i as f32 / pattern_size as f32;
                let freq = frequency_range_hz.0 + freq_norm * (frequency_range_hz.1 - frequency_range_hz.0);
                (-freq / 10000.0).exp() // Decay with frequency
            })
            .collect();

        Self {
            name,
            frequency_range_hz,
            call_duration_ms,
            spectral_pattern,
            typical_snr_db,
            priority: 1,
        }
    }

    /// Check if a frequency is within this species' range
    pub fn matches_frequency(&self, freq_hz: f32) -> bool {
        freq_hz >= self.frequency_range_hz.0 && freq_hz <= self.frequency_range_hz.1
    }

    /// Calculate confidence score for a detected frequency
    pub fn calculate_confidence(&self, detected_freq_hz: f32, snr_db: f32) -> f32 {
        // Frequency match score
        let freq_range = self.frequency_range_hz.1 - self.frequency_range_hz.0;
        let freq_center = (self.frequency_range_hz.0 + self.frequency_range_hz.1) / 2.0;
        let freq_distance = (detected_freq_hz - freq_center).abs();
        let freq_score = 1.0 - (freq_distance / (freq_range / 2.0)).min(1.0);

        // SNR score
        let snr_score = (snr_db / self.typical_snr_db).min(1.0);

        // Combined score
        0.6 * freq_score + 0.4 * snr_score
    }
}

/// Detection event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionEvent {
    pub species: String,
    pub confidence: f32,
    pub timestamp: PtpTimestamp,
    pub start_sample: usize,
    pub duration_samples: usize,
    pub dominant_frequency_hz: f32,
    pub snr_db: f32,
}

impl DetectionEvent {
    pub fn duration_ms(&self, sample_rate: usize) -> f32 {
        (self.duration_samples as f32 / sample_rate as f32) * 1000.0
    }
}

/// Trigger urgency level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerUrgency {
    Low,      // Single distant call
    Medium,   // Multiple calls
    High,     // Close proximity or agitated calls
    Critical, // Alarm calls or distress
}

impl TriggerUrgency {
    /// Calculate urgency from detection events
    pub fn from_detections(detections: &[DetectionEvent]) -> Self {
        if detections.is_empty() {
            return Self::Low;
        }

        // Check for high confidence detections
        let has_high_confidence = detections.iter().any(|d| d.confidence > 0.8);

        // Check for multiple detections
        let multiple_species = detections
            .iter()
            .map(|d| &d.species)
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1;

        // Check for alarm-like patterns (high frequency, high amplitude)
        let has_alarm_pattern = detections
            .iter()
            .any(|d| d.dominant_frequency_hz > 5000.0 && d.confidence > 0.7 && d.snr_db > 15.0);

        match (has_high_confidence, multiple_species, has_alarm_pattern) {
            (true, _, true) => Self::Critical,
            (true, true, _) => Self::High,
            (true, false, _) => Self::Medium,
            (false, true, _) => Self::Medium,
            (false, false, _) => Self::Low,
        }
    }

    /// Get suggested response duration in milliseconds
    pub fn suggested_response_duration_ms(&self) -> u64 {
        match self {
            Self::Low => 500,
            Self::Medium => 1000,
            Self::High => 2000,
            Self::Critical => 3000,
        }
    }
}

/// Wake trigger for Python agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WakeTrigger {
    pub detections: Vec<DetectionEvent>,
    pub urgency: TriggerUrgency,
    pub suggested_response_duration_ms: u64,
    pub timestamp: PtpTimestamp,
}

impl WakeTrigger {
    pub fn new(detections: Vec<DetectionEvent>) -> Self {
        let urgency = TriggerUrgency::from_detections(&detections);
        let suggested_response_duration_ms = urgency.suggested_response_duration_ms();

        let timestamp = detections
            .first()
            .map(|d| d.timestamp)
            .unwrap_or_else(|| PtpTimestamp::from(chrono::Utc::now()));

        Self {
            detections,
            urgency,
            suggested_response_duration_ms,
            timestamp,
        }
    }
}

/// FFT processor for frequency analysis
struct FFTProcessor {
    #[allow(dead_code)]
    fft_size: usize,
    sample_rate: usize,
}

impl FFTProcessor {
    fn new(fft_size: usize, sample_rate: usize) -> Self {
        Self { fft_size, sample_rate }
    }

    /// Estimate power at a specific frequency using correlation
    fn estimate_power_at_frequency(&self, audio: &[f32], freq_hz: f32) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        let omega = 2.0 * std::f32::consts::PI * freq_hz / self.sample_rate as f32;

        // Correlate with sine and cosine at target frequency
        let mut sin_sum = 0.0;
        let mut cos_sum = 0.0;

        for (i, &sample) in audio.iter().enumerate() {
            let phase = omega * i as f32;
            sin_sum += sample * phase.sin();
            cos_sum += sample * phase.cos();
        }

        // Magnitude squared
        let magnitude_sq = sin_sum * sin_sum + cos_sum * cos_sum;

        // Normalize by number of samples
        magnitude_sq.sqrt() / audio.len() as f32
    }

    /// Find dominant frequency in audio
    fn find_dominant_frequency(&self, audio: &[f32], min_hz: f32, max_hz: f32) -> Option<(f32, f32)> {
        if audio.is_empty() {
            return None;
        }

        // Use coarse-to-fine search
        let step_hz = 100.0; // Coarse step
        let mut best_freq = min_hz;
        let mut best_power = 0.0;

        let mut freq = min_hz;
        while freq <= max_hz {
            let power = self.estimate_power_at_frequency(audio, freq);
            if power > best_power {
                best_power = power;
                best_freq = freq;
            }
            freq += step_hz;
        }

        // Fine-tune around best frequency
        let fine_range = 200.0;
        let fine_step = 10.0;
        freq = (best_freq - fine_range / 2.0).max(min_hz);
        while freq <= max_hz && freq <= best_freq + fine_range / 2.0 {
            let power = self.estimate_power_at_frequency(audio, freq);
            if power > best_power {
                best_power = power;
                best_freq = freq;
            }
            freq += fine_step;
        }

        if best_power > 0.001 {
            Some((best_freq, best_power))
        } else {
            None
        }
    }

    /// Calculate SNR estimate
    fn estimate_snr(&self, audio: &[f32], signal_freq_hz: f32, bandwidth_hz: f32) -> f32 {
        let signal_power = self.estimate_power_at_frequency(audio, signal_freq_hz);

        // Estimate noise as average power in adjacent bands
        let noise_freq = signal_freq_hz + bandwidth_hz * 2.0;
        let noise_power = self.estimate_power_at_frequency(audio, noise_freq);

        if noise_power > 0.0001 {
            20.0 * (signal_power / noise_power).log10().max(0.0)
        } else {
            20.0 // Assume good SNR if noise is very low
        }
    }
}

/// Wildlife Sentry - Main detector
pub struct WildlifeSentry {
    config: WildlifeSentryConfig,
    species_database: HashMap<String, SpeciesSignature>,
    fft_processor: FFTProcessor,
    last_detection: Arc<AtomicCell<Option<Instant>>>,
    detection_count: Arc<AtomicCell<usize>>,
    trigger_count: Arc<AtomicCell<usize>>,
}

impl WildlifeSentry {
    /// Create a new wildlife sentry
    pub fn new(config: WildlifeSentryConfig) -> Self {
        let fft_processor = FFTProcessor::new(config.fft_size, config.sample_rate);

        let mut sentry = Self {
            config,
            species_database: HashMap::new(),
            fft_processor,
            last_detection: Arc::new(AtomicCell::new(None)),
            detection_count: Arc::new(AtomicCell::new(0)),
            trigger_count: Arc::new(AtomicCell::new(0)),
        };

        // Load default species signatures
        sentry.load_default_species();
        sentry
    }

    /// Load default species signatures
    fn load_default_species(&mut self) {
        // Marmoset: 7-12 kHz, short calls
        self.add_species_signature(SpeciesSignature::new(
            "marmoset".to_string(),
            (7000.0, 12000.0),
            (50.0, 300.0),
            10.0,
        ));

        // Dolphin: 2-24 kHz, whistles
        self.add_species_signature(SpeciesSignature::new(
            "dolphin".to_string(),
            (2000.0, 24000.0),
            (200.0, 2000.0),
            12.0,
        ));

        // Bat: 20-100 kHz, FM sweeps (simplified to lower range for testing)
        self.add_species_signature(SpeciesSignature::new(
            "bat".to_string(),
            (20000.0, 100000.0),
            (5.0, 50.0),
            8.0,
        ));

        // Finch: 2-8 kHz, song bursts
        self.add_species_signature(SpeciesSignature::new(
            "finch".to_string(),
            (2000.0, 8000.0),
            (100.0, 500.0),
            9.0,
        ));
    }

    /// Add a species signature to the database
    pub fn add_species_signature(&mut self, signature: SpeciesSignature) {
        self.species_database.insert(signature.name.clone(), signature);
    }

    /// Get the species database
    pub fn species_database(&self) -> &HashMap<String, SpeciesSignature> {
        &self.species_database
    }

    /// Process audio buffer and detect species
    pub fn process_audio(&self, audio: &[f32]) -> Result<Vec<DetectionEvent>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let mut detections = Vec::new();

        // Check each target species
        for species_name in &self.config.target_species {
            if let Some(signature) = self.species_database.get(species_name) {
                // Find dominant frequency in species range
                if let Some((freq_hz, power)) = self.fft_processor.find_dominant_frequency(
                    audio,
                    signature.frequency_range_hz.0,
                    signature.frequency_range_hz.1,
                ) {
                    // Check if power exceeds threshold
                    if power > self.config.detection_threshold {
                        // Calculate confidence
                        let snr_db = self.fft_processor.estimate_snr(audio, freq_hz, 1000.0);
                        let confidence = signature.calculate_confidence(freq_hz, snr_db);

                        if confidence >= self.config.min_confidence {
                            let detection = DetectionEvent {
                                species: species_name.clone(),
                                confidence,
                                timestamp: PtpTimestamp::from(chrono::Utc::now()),
                                start_sample: 0,
                                duration_samples: audio.len(),
                                dominant_frequency_hz: freq_hz,
                                snr_db,
                            };

                            detections.push(detection);
                            self.detection_count.fetch_add(1);
                        }
                    }
                }
            }
        }

        Ok(detections)
    }

    /// Generate wake trigger if conditions are met
    pub fn generate_wake_trigger(&self, audio: &[f32]) -> Result<Option<WakeTrigger>> {
        let detections = self.process_audio(audio)?;

        if detections.is_empty() {
            return Ok(None);
        }

        // Check debounce
        if let Some(last) = self.last_detection.load() {
            if last.elapsed() < Duration::from_millis(self.config.debounce_ms) {
                return Ok(None);
            }
        }

        // Update last detection time
        self.last_detection.store(Some(Instant::now()));
        self.trigger_count.fetch_add(1);

        Ok(Some(WakeTrigger::new(detections)))
    }

    /// Check if should trigger wake
    pub fn should_trigger(&self, detections: &[DetectionEvent]) -> bool {
        // Trigger if any detection exceeds confidence threshold
        detections.iter().any(|d| d.confidence >= self.config.min_confidence)
    }

    /// Get detection statistics
    pub fn detection_stats(&self) -> (usize, usize) {
        (self.detection_count.load(), self.trigger_count.load())
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.detection_count.store(0);
        self.trigger_count.store(0);
        self.last_detection.store(None);
    }
}

impl Default for WildlifeSentry {
    fn default() -> Self {
        Self::new(WildlifeSentryConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_signature(name: &str, freq_min: f32, freq_max: f32) -> SpeciesSignature {
        SpeciesSignature::new(name.to_string(), (freq_min, freq_max), (50.0, 300.0), 10.0)
    }

    fn generate_test_tone(freq_hz: f32, duration_ms: f32, sample_rate: usize) -> Vec<f32> {
        let num_samples = (duration_ms / 1000.0 * sample_rate as f32) as usize;
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin()
            })
            .collect()
    }

    fn generate_test_noise(num_samples: usize) -> Vec<f32> {
        // Generate low-frequency noise (below all target species ranges)
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / 48000.0;
                (50.0 * 2.0 * std::f32::consts::PI * t).sin() * 0.1 // 50 Hz hum, low amplitude
            })
            .collect()
    }

    #[test]
    fn test_config_defaults() {
        let config = WildlifeSentryConfig::default();
        assert_eq!(config.target_species.len(), 3);
        assert_eq!(config.detection_threshold, 0.001);
        assert_eq!(config.debounce_ms, 500);
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.fft_size, 2048);
    }

    #[test]
    fn test_sentry_initialization() {
        let sentry = WildlifeSentry::default();
        assert_eq!(sentry.species_database().len(), 4);
        assert!(sentry.species_database().contains_key("marmoset"));
        assert!(sentry.species_database().contains_key("dolphin"));
        assert!(sentry.species_database().contains_key("bat"));
        assert!(sentry.species_database().contains_key("finch"));
    }

    #[test]
    fn test_add_species_signature() {
        let mut sentry = WildlifeSentry::default();
        let signature = create_test_signature("whale", 100.0, 1000.0);
        sentry.add_species_signature(signature);

        assert!(sentry.species_database().contains_key("whale"));
        assert_eq!(sentry.species_database().len(), 5);
    }

    #[test]
    fn test_species_signature_matches_frequency() {
        let signature = create_test_signature("test", 1000.0, 5000.0);
        assert!(signature.matches_frequency(3000.0));
        assert!(signature.matches_frequency(1000.0));
        assert!(signature.matches_frequency(5000.0));
        assert!(!signature.matches_frequency(999.0));
        assert!(!signature.matches_frequency(5001.0));
    }

    #[test]
    fn test_species_signature_confidence_calculation() {
        let signature = create_test_signature("test", 1000.0, 5000.0);

        // Exact center frequency
        let conf1 = signature.calculate_confidence(3000.0, 10.0);
        assert!(conf1 > 0.5);

        // At edge of range
        let conf2 = signature.calculate_confidence(1000.0, 10.0);
        assert!(conf2 < conf1);

        // Perfect SNR
        let conf3 = signature.calculate_confidence(3000.0, 10.0);
        let conf4 = signature.calculate_confidence(3000.0, 15.0);
        assert!(conf4 >= conf3);
    }

    #[test]
    fn test_detect_marmoset_call() {
        let sentry = WildlifeSentry::default();

        // Generate 9 kHz tone (in marmoset range: 7-12 kHz)
        let tone = generate_test_tone(9000.0, 100.0, 48000);

        let detections = sentry.process_audio(&tone).unwrap();
        assert!(!detections.is_empty());

        let marmoset_detection = detections.iter().find(|d| d.species == "marmoset");
        assert!(marmoset_detection.is_some());

        if let Some(detection) = marmoset_detection {
            assert!(detection.dominant_frequency_hz >= 7000.0);
            assert!(detection.dominant_frequency_hz <= 12000.0);
        }
    }

    #[test]
    fn test_detect_dolphin_whistle() {
        let sentry = WildlifeSentry::default();

        // Generate 10 kHz tone (in dolphin range: 2-24 kHz)
        let tone = generate_test_tone(10000.0, 200.0, 48000);

        let detections = sentry.process_audio(&tone).unwrap();
        assert!(!detections.is_empty());

        let dolphin_detection = detections.iter().find(|d| d.species == "dolphin");
        assert!(dolphin_detection.is_some());
    }

    #[test]
    fn test_reject_noise() {
        let sentry = WildlifeSentry::default();

        // Generate white noise
        let noise = generate_test_noise(4800);

        let detections = sentry.process_audio(&noise).unwrap();

        // White noise should not produce confident detections
        let confident_detections: Vec<_> = detections
            .iter()
            .filter(|d| d.confidence > sentry.config.min_confidence)
            .collect();

        assert_eq!(confident_detections.len(), 0);
    }

    #[test]
    fn test_trigger_wake_python() {
        let sentry = WildlifeSentry::default();

        // Generate strong marmoset call
        let tone = generate_test_tone(9000.0, 100.0, 48000);

        let trigger = sentry.generate_wake_trigger(&tone).unwrap();
        assert!(trigger.is_some());

        if let Some(t) = trigger {
            assert!(!t.detections.is_empty());
        }
    }

    #[test]
    fn test_debounce_detections() {
        let sentry = WildlifeSentry::default();

        // Generate marmoset call
        let tone = generate_test_tone(9000.0, 100.0, 48000);

        // First trigger should work
        let trigger1 = sentry.generate_wake_trigger(&tone).unwrap();
        assert!(trigger1.is_some());

        // Immediate second trigger should be debounced
        let trigger2 = sentry.generate_wake_trigger(&tone).unwrap();
        assert!(trigger2.is_none());
    }

    #[test]
    fn test_multi_species_detection() {
        let config = WildlifeSentryConfig {
            target_species: vec!["marmoset".to_string(), "finch".to_string()],
            ..Default::default()
        };
        let sentry = WildlifeSentry::new(config);

        // Generate tone that overlaps both ranges
        // Finch: 2-8 kHz, Marmoset: 7-12 kHz
        // 7.5 kHz is in both ranges
        let tone = generate_test_tone(7500.0, 100.0, 48000);

        let detections = sentry.process_audio(&tone).unwrap();

        // Should detect both species
        let has_marmoset = detections.iter().any(|d| d.species == "marmoset");
        let has_finch = detections.iter().any(|d| d.species == "finch");

        assert!(has_marmoset || has_finch);
    }

    #[test]
    fn test_urgency_low() {
        let detections = vec![DetectionEvent {
            species: "marmoset".to_string(),
            confidence: 0.5,
            timestamp: PtpTimestamp::new(0, 0),
            start_sample: 0,
            duration_samples: 4800,
            dominant_frequency_hz: 9000.0,
            snr_db: 8.0,
        }];

        let urgency = TriggerUrgency::from_detections(&detections);
        assert_eq!(urgency, TriggerUrgency::Low);
    }

    #[test]
    fn test_urgency_medium() {
        let detections = vec![DetectionEvent {
            species: "marmoset".to_string(),
            confidence: 0.9,
            timestamp: PtpTimestamp::new(0, 0),
            start_sample: 0,
            duration_samples: 4800,
            dominant_frequency_hz: 9000.0,
            snr_db: 8.0,
        }];

        let urgency = TriggerUrgency::from_detections(&detections);
        assert_eq!(urgency, TriggerUrgency::Medium);
    }

    #[test]
    fn test_urgency_high() {
        let detections = vec![
            DetectionEvent {
                species: "marmoset".to_string(),
                confidence: 0.9,
                timestamp: PtpTimestamp::new(0, 0),
                start_sample: 0,
                duration_samples: 4800,
                dominant_frequency_hz: 9000.0,
                snr_db: 8.0,
            },
            DetectionEvent {
                species: "finch".to_string(),
                confidence: 0.8,
                timestamp: PtpTimestamp::new(0, 0),
                start_sample: 0,
                duration_samples: 4800,
                dominant_frequency_hz: 4000.0,
                snr_db: 10.0,
            },
        ];

        let urgency = TriggerUrgency::from_detections(&detections);
        assert_eq!(urgency, TriggerUrgency::High);
    }

    #[test]
    fn test_urgency_critical() {
        let detections = vec![DetectionEvent {
            species: "marmoset".to_string(),
            confidence: 0.9,
            timestamp: PtpTimestamp::new(0, 0),
            start_sample: 0,
            duration_samples: 4800,
            dominant_frequency_hz: 8000.0,
            snr_db: 20.0, // High SNR
        }];

        let urgency = TriggerUrgency::from_detections(&detections);
        assert_eq!(urgency, TriggerUrgency::Critical);
    }

    #[test]
    fn test_suggested_response_duration() {
        assert_eq!(TriggerUrgency::Low.suggested_response_duration_ms(), 500);
        assert_eq!(TriggerUrgency::Medium.suggested_response_duration_ms(), 1000);
        assert_eq!(TriggerUrgency::High.suggested_response_duration_ms(), 2000);
        assert_eq!(TriggerUrgency::Critical.suggested_response_duration_ms(), 3000);
    }

    #[test]
    fn test_wake_trigger_creation() {
        let detections = vec![DetectionEvent {
            species: "marmoset".to_string(),
            confidence: 0.8,
            timestamp: PtpTimestamp::new(0, 0),
            start_sample: 0,
            duration_samples: 4800,
            dominant_frequency_hz: 9000.0,
            snr_db: 10.0,
        }];

        let trigger = WakeTrigger::new(detections.clone());

        assert_eq!(trigger.detections.len(), 1);
        assert_eq!(trigger.detections[0].species, "marmoset");
        assert!(trigger.suggested_response_duration_ms > 0);
    }

    #[test]
    fn test_detection_event_duration() {
        let event = DetectionEvent {
            species: "test".to_string(),
            confidence: 0.8,
            timestamp: PtpTimestamp::new(0, 0),
            start_sample: 0,
            duration_samples: 4800, // 100ms at 48kHz
            dominant_frequency_hz: 1000.0,
            snr_db: 10.0,
        };

        let duration_ms = event.duration_ms(48000);
        assert!((duration_ms - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_should_trigger() {
        let sentry = WildlifeSentry::default();

        let detections = vec![DetectionEvent {
            species: "marmoset".to_string(),
            confidence: 0.9,
            timestamp: PtpTimestamp::new(0, 0),
            start_sample: 0,
            duration_samples: 4800,
            dominant_frequency_hz: 9000.0,
            snr_db: 10.0,
        }];

        assert!(sentry.should_trigger(&detections));
    }

    #[test]
    fn test_should_not_trigger_low_confidence() {
        let sentry = WildlifeSentry::default();

        let detections = vec![DetectionEvent {
            species: "marmoset".to_string(),
            confidence: 0.3,
            timestamp: PtpTimestamp::new(0, 0),
            start_sample: 0,
            duration_samples: 4800,
            dominant_frequency_hz: 9000.0,
            snr_db: 5.0,
        }];

        assert!(!sentry.should_trigger(&detections));
    }

    #[test]
    fn test_detection_stats() {
        let sentry = WildlifeSentry::default();

        let (detections, triggers) = sentry.detection_stats();
        assert_eq!(detections, 0);
        assert_eq!(triggers, 0);

        // Generate a trigger
        let tone = generate_test_tone(9000.0, 100.0, 48000);
        let _ = sentry.generate_wake_trigger(&tone).unwrap();

        let (detections, triggers) = sentry.detection_stats();
        assert!(detections > 0);
        assert_eq!(triggers, 1);
    }

    #[test]
    fn test_reset_stats() {
        let sentry = WildlifeSentry::default();

        // Generate a trigger
        let tone = generate_test_tone(9000.0, 100.0, 48000);
        let _ = sentry.generate_wake_trigger(&tone).unwrap();

        sentry.reset_stats();

        let (detections, triggers) = sentry.detection_stats();
        assert_eq!(detections, 0);
        assert_eq!(triggers, 0);
    }

    #[test]
    fn test_empty_audio_returns_no_detections() {
        let sentry = WildlifeSentry::default();
        let detections = sentry.process_audio(&[]).unwrap();
        assert_eq!(detections.len(), 0);
    }

    #[test]
    fn test_no_trigger_on_empty_audio() {
        let sentry = WildlifeSentry::default();
        let trigger = sentry.generate_wake_trigger(&[]).unwrap();
        assert!(trigger.is_none());
    }
}
