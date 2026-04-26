//! Real-Time Stream Ingestion for Passive Acoustic Monitoring
//! ============================================================
//!
//! Provides streaming audio ingestion with real-time timestamps derived from
//! the system clock rather than synthetic file offsets. This module enables
//! continuous monitoring applications where audio arrives in real-time chunks.
//!
//! # Key Concepts
//!
//! - **RealTimeTimestamp**: Combines system clock time with sample offset
//! - **StreamingBuffer**: Ring buffer for continuous audio ingestion
//! - **DebounceTimer**: Enforces minimum phrase duration to prevent rapid-fire detection
//!
//! # Usage
//!
//! ```rust
//! use technical_architecture::streaming::{StreamingBuffer, StreamingConfig};
//!
//! let config = StreamingConfig {
//!     hop_size: 512,
//!     sample_rate: 44100,
//!     buffer_duration_secs: 60.0,
//! };
//! let mut buffer = StreamingBuffer::with_config(config);
//!
//! // Ingest audio samples from real-time source
//! let audio_chunk = vec![0.0f32; 512];
//! let timestamp = buffer.add_samples(&audio_chunk);
//! println!("Samples ingested at: {:?}", timestamp.system_time);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

/// Configuration for the streaming buffer
#[derive(Debug, Clone, Copy)]
pub struct StreamingConfig {
    /// Hop size in samples (default: 512 for ~11.6ms at 44.1kHz)
    pub hop_size: usize,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Buffer duration in seconds (determines ring buffer size)
    pub buffer_duration_secs: f32,
    /// Minimum phrase duration in milliseconds (debounce)
    pub min_phrase_duration_ms: f32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            hop_size: 512,
            sample_rate: 44100,
            buffer_duration_secs: 60.0,
            min_phrase_duration_ms: 50.0,
        }
    }
}

/// Real-time timestamp combining system clock with sample precision
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RealTimeTimestamp {
    /// System time when samples were ingested
    pub system_time: SystemTime,
    /// Sample offset from start of buffer
    pub sample_offset: usize,
    /// Duration of the samples in milliseconds
    pub duration_ms: f32,
}

impl RealTimeTimestamp {
    /// Create a new real-time timestamp
    pub fn new(system_time: SystemTime, sample_offset: usize, duration_ms: f32) -> Self {
        Self {
            system_time,
            sample_offset,
            duration_ms,
        }
    }

    /// Convert to milliseconds since Unix epoch
    pub fn as_millis_since_epoch(&self) -> u64 {
        self.system_time
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// Get time in seconds as f64
    pub fn as_secs_f64(&self) -> f64 {
        self.system_time
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
}

/// Streaming buffer for continuous audio ingestion
///
/// Uses a ring buffer to maintain a sliding window of audio samples.
/// Each chunk of samples added returns a RealTimeTimestamp based on
/// the system clock at ingestion time.
#[derive(Debug)]
pub struct StreamingBuffer {
    config: StreamingConfig,
    /// Ring buffer storing audio samples
    buffer: VecDeque<f32>,
    /// Total samples ingested (for offset calculation)
    total_samples: usize,
    /// Sample rate for time calculations
    sample_rate: u32,
}

impl StreamingBuffer {
    /// Create a new streaming buffer with default configuration
    pub fn new(hop_size: usize, sample_rate: u32) -> Self {
        Self::with_config(StreamingConfig {
            hop_size,
            sample_rate,
            ..Default::default()
        })
    }

    /// Create a streaming buffer with custom configuration
    pub fn with_config(config: StreamingConfig) -> Self {
        let buffer_size = (config.buffer_duration_secs * config.sample_rate as f32) as usize;
        Self {
            config,
            buffer: VecDeque::with_capacity(buffer_size),
            total_samples: 0,
            sample_rate: config.sample_rate,
        }
    }

    /// Add samples to the buffer, returning a real-time timestamp
    ///
    /// The timestamp is based on the system clock at the moment of ingestion,
    /// not on any synthetic offset from a file.
    pub fn add_samples(&mut self, samples: &[f32]) -> RealTimeTimestamp {
        let system_time = SystemTime::now();
        let sample_offset = self.total_samples;
        let duration_ms = (samples.len() as f32 / self.sample_rate as f32) * 1000.0;

        // Add samples to ring buffer
        for &sample in samples {
            self.buffer.push_back(sample);
        }

        // Maintain buffer size limit
        let max_samples = (self.config.buffer_duration_secs * self.sample_rate as f32) as usize;
        while self.buffer.len() > max_samples {
            self.buffer.pop_front();
        }

        self.total_samples += samples.len();

        RealTimeTimestamp::new(system_time, sample_offset, duration_ms)
    }

    /// Get samples from the buffer within a time range
    ///
    /// Returns samples from `start_ms` to `end_ms` relative to the buffer start.
    pub fn get_samples_in_range(&self, start_ms: f32, end_ms: f32) -> Vec<f32> {
        let sample_rate = self.sample_rate as f32;
        let start_sample = ((start_ms / 1000.0) * sample_rate) as usize;
        let end_sample = ((end_ms / 1000.0) * sample_rate) as usize;

        self.buffer
            .iter()
            .skip(start_sample)
            .take(end_sample.saturating_sub(start_sample))
            .copied()
            .collect()
    }

    /// Get all samples currently in the buffer
    pub fn get_all_samples(&self) -> Vec<f32> {
        self.buffer.iter().copied().collect()
    }

    /// Get the current buffer size in samples
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get the total number of samples ever ingested
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the configuration
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.total_samples = 0;
    }
}

/// Debounce timer for enforcing minimum phrase duration
///
/// Prevents rapid-fire detection of boundaries from transient noise
/// by enforcing a minimum time between detections.
#[derive(Debug, Clone)]
pub struct DebounceTimer {
    /// Minimum duration between triggers in milliseconds
    min_duration_ms: f32,
    /// Last trigger time (in samples since start)
    last_trigger_sample: Option<usize>,
    /// Sample rate for time calculations
    sample_rate: u32,
}

impl DebounceTimer {
    /// Create a new debounce timer
    pub fn new(min_duration_ms: f32, sample_rate: u32) -> Self {
        Self {
            min_duration_ms,
            last_trigger_sample: None,
            sample_rate,
        }
    }

    /// Check if enough time has elapsed since last trigger
    ///
    /// Returns `true` if the trigger is allowed (enough time has passed),
    /// `false` if it should be debounced.
    pub fn check_and_update(&mut self, current_sample: usize) -> bool {
        let min_samples = (self.min_duration_ms / 1000.0 * self.sample_rate as f32) as usize;

        if let Some(last_sample) = self.last_trigger_sample {
            if current_sample.saturating_sub(last_sample) < min_samples {
                return false; // Debounce
            }
        }

        self.last_trigger_sample = Some(current_sample);
        true
    }

    /// Reset the debounce timer
    pub fn reset(&mut self) {
        self.last_trigger_sample = None;
    }

    /// Get the minimum duration in milliseconds
    pub fn min_duration_ms(&self) -> f32 {
        self.min_duration_ms
    }

    /// Get the time until next trigger is allowed (in ms)
    ///
    /// Returns 0.0 if a trigger is currently allowed.
    pub fn time_until_next_trigger(&self, current_sample: usize) -> f32 {
        if let Some(last_sample) = self.last_trigger_sample {
            let min_samples = (self.min_duration_ms / 1000.0 * self.sample_rate as f32) as usize;
            let elapsed = current_sample.saturating_sub(last_sample);
            if elapsed < min_samples {
                let remaining_samples = min_samples - elapsed;
                return (remaining_samples as f32 / self.sample_rate as f32) * 1000.0;
            }
        }
        0.0
    }
}

/// Spectral Change Profile for detecting timbral shifts
///
/// Used by the Neural Boundary Detector to identify phrase boundaries
/// based on spectral characteristics rather than just energy.
#[derive(Debug, Clone, Default)]
pub struct SpectralChangeProfile {
    /// Spectral centroid changes per frame
    pub centroid_changes: Vec<f32>,
    /// Spectral flatness changes per frame
    pub flatness_changes: Vec<f32>,
    /// Zero-crossing rate changes per frame
    pub zcr_changes: Vec<f32>,
    /// Combined change score per frame
    pub combined_score: Vec<f32>,
}

impl SpectralChangeProfile {
    /// Create a new empty spectral change profile
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute spectral change profile from audio
    ///
    /// Analyzes the audio in frames and computes the change in spectral
    /// features between consecutive frames.
    pub fn compute(audio: &[f32], hop_size: usize, sample_rate: u32) -> Self {
        if audio.len() < hop_size * 2 {
            return Self::new();
        }

        let n_frames = audio.len() / hop_size;
        let mut profile = SpectralChangeProfile::new();

        // Pre-allocate
        profile.centroid_changes.reserve(n_frames.saturating_sub(1));
        profile.flatness_changes.reserve(n_frames.saturating_sub(1));
        profile.zcr_changes.reserve(n_frames.saturating_sub(1));
        profile.combined_score.reserve(n_frames.saturating_sub(1));

        // Compute features for each frame
        let mut prev_centroid = 0.0f32;
        let mut prev_flatness = 0.0f32;
        let mut prev_zcr = 0.0f32;

        for i in 0..n_frames.saturating_sub(1) {
            let start = i * hop_size;
            let end = (start + hop_size).min(audio.len());
            let frame = &audio[start..end];

            // Compute spectral centroid (simplified)
            let centroid = compute_spectral_centroid(frame, sample_rate);
            // Compute spectral flatness (simplified)
            let flatness = compute_spectral_flatness(frame);
            // Compute zero-crossing rate
            let zcr = compute_zcr(frame);

            // Compute changes
            let centroid_change = (centroid - prev_centroid).abs();
            let flatness_change = (flatness - prev_flatness).abs();
            let zcr_change = (zcr - prev_zcr).abs();

            profile.centroid_changes.push(centroid_change);
            profile.flatness_changes.push(flatness_change);
            profile.zcr_changes.push(zcr_change);

            // Combined score with weights
            let combined = centroid_change * 0.4 + flatness_change * 0.3 + zcr_change * 0.3;
            profile.combined_score.push(combined);

            prev_centroid = centroid;
            prev_flatness = flatness;
            prev_zcr = zcr;
        }

        profile
    }

    /// Get the maximum combined change score
    pub fn max_change(&self) -> f32 {
        self.combined_score.iter().copied().fold(0.0f32, f32::max)
    }

    /// Find frames where combined score exceeds threshold
    pub fn find_change_points(&self, threshold: f32) -> Vec<usize> {
        self.combined_score
            .iter()
            .enumerate()
            .filter(|(_, &score)| score > threshold)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Compute spectral centroid (simplified)
fn compute_spectral_centroid(frame: &[f32], sample_rate: u32) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }

    // Simple zero-FFT approximation using energy-weighted position
    let mut weighted_sum = 0.0f32;
    let mut energy_sum = 0.0f32;

    for (i, &sample) in frame.iter().enumerate() {
        let energy = sample * sample;
        weighted_sum += i as f32 * energy;
        energy_sum += energy;
    }

    if energy_sum > 1e-10 {
        // Convert sample index to approximate frequency
        let centroid_sample = weighted_sum / energy_sum;
        centroid_sample / frame.len() as f32 * sample_rate as f32 / 2.0
    } else {
        0.0
    }
}

/// Compute spectral flatness (simplified geometric/arithmetic mean ratio)
fn compute_spectral_flatness(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }

    // Simplified: use energy values directly
    let mut geometric_mean = 1.0f32;
    let mut arithmetic_sum = 0.0f32;
    let mut count = 0usize;

    for &sample in frame {
        let energy = sample * sample;
        if energy > 1e-10 {
            geometric_mean *= energy.powf(1.0 / frame.len() as f32);
            arithmetic_sum += energy;
            count += 1;
        }
    }

    if count > 0 && arithmetic_sum > 0.0 {
        let arithmetic_mean = arithmetic_sum / count as f32;
        if arithmetic_mean > 1e-10 {
            return geometric_mean / arithmetic_mean;
        }
    }
    0.0
}

/// Compute zero-crossing rate
fn compute_zcr(frame: &[f32]) -> f32 {
    if frame.len() < 2 {
        return 0.0;
    }

    let mut crossings = 0usize;
    for i in 1..frame.len() {
        if (frame[i - 1] >= 0.0 && frame[i] < 0.0) || (frame[i - 1] < 0.0 && frame[i] >= 0.0) {
            crossings += 1;
        }
    }

    crossings as f32 / (frame.len() - 1) as f32
}

// ============================================================================
// Tests (TDD: Red Phase)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_time_timestamp_uses_system_clock() {
        let mut buffer = StreamingBuffer::new(512, 44100);
        let before = SystemTime::now();
        let ts = buffer.add_samples(&[0.0f32; 512]);
        let after = SystemTime::now();

        assert!(ts.system_time >= before, "Timestamp should be >= before time");
        assert!(ts.system_time <= after, "Timestamp should be <= after time");
    }

    #[test]
    fn test_streaming_buffer_add_samples() {
        let mut buffer = StreamingBuffer::new(512, 44100);
        let samples = vec![0.5f32; 512];
        let ts = buffer.add_samples(&samples);

        assert_eq!(buffer.len(), 512);
        assert_eq!(buffer.total_samples(), 512);
        assert_eq!(ts.sample_offset, 0);
        assert!((ts.duration_ms - 11.61).abs() < 0.1); // ~11.6ms at 44.1kHz
    }

    #[test]
    fn test_streaming_buffer_ring_behavior() {
        let config = StreamingConfig {
            hop_size: 512,
            sample_rate: 44100,
            buffer_duration_secs: 0.1, // Very small buffer (~4410 samples)
            ..Default::default()
        };
        let mut buffer = StreamingBuffer::with_config(config);

        // Add more samples than buffer can hold
        for _ in 0..20 {
            buffer.add_samples(&[1.0f32; 512]);
        }

        // Buffer should be limited to configured size
        let max_samples = (0.1 * 44100.0) as usize;
        assert!(buffer.len() <= max_samples);
        assert_eq!(buffer.total_samples(), 20 * 512); // Total still tracked
    }

    #[test]
    fn test_debounce_timer_allows_first_trigger() {
        let mut timer = DebounceTimer::new(50.0, 44100);
        assert!(timer.check_and_update(0), "First trigger should be allowed");
    }

    #[test]
    fn test_debounce_timer_blocks_rapid_triggers() {
        let mut timer = DebounceTimer::new(50.0, 44100); // 50ms = 2205 samples at 44.1kHz
        assert!(timer.check_and_update(0));

        // Try to trigger too soon
        let blocked = timer.check_and_update(1000); // Only 1000 samples later
        assert!(!blocked, "Rapid trigger should be blocked");
    }

    #[test]
    fn test_debounce_timer_allows_after_cooldown() {
        let mut timer = DebounceTimer::new(50.0, 44100);
        assert!(timer.check_and_update(0));

        // Wait long enough (50ms = 2205 samples)
        let allowed = timer.check_and_update(3000);
        assert!(allowed, "Trigger after cooldown should be allowed");
    }

    #[test]
    fn test_debounce_timer_time_until_next_trigger() {
        let mut timer = DebounceTimer::new(50.0, 44100);
        timer.check_and_update(0);

        // Check time remaining at 1000 samples
        let remaining = timer.time_until_next_trigger(1000);
        assert!(remaining > 0.0, "Should have time remaining");
        assert!(remaining < 50.0, "Should be less than full duration");
    }

    #[test]
    fn test_spectral_change_profile_empty_audio() {
        let profile = SpectralChangeProfile::compute(&[], 512, 44100);
        assert!(profile.centroid_changes.is_empty());
    }

    #[test]
    fn test_spectral_change_profile_detects_changes() {
        // Create audio with distinct sections
        let mut audio = vec![0.0f32; 44100]; // 1 second

        // Section 1: Low frequency (first half)
        for i in 0..22050 {
            audio[i] = (2.0 * std::f32::consts::PI * 100.0 * i as f32 / 44100.0).sin() * 0.5;
        }

        // Section 2: High frequency (second half)
        for i in 22050..44100 {
            audio[i] = (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 44100.0).sin() * 0.5;
        }

        let profile = SpectralChangeProfile::compute(&audio, 512, 44100);
        assert!(!profile.combined_score.is_empty());

        // The transition point should have elevated change score
        let max_change = profile.max_change();
        assert!(max_change > 0.0, "Should detect spectral changes");
    }

    #[test]
    fn test_real_time_timestamp_epoch_conversion() {
        let now = SystemTime::now();
        let ts = RealTimeTimestamp::new(now, 1000, 10.0);

        let millis = ts.as_millis_since_epoch();
        assert!(millis > 0, "Should convert to milliseconds since epoch");

        let secs = ts.as_secs_f64();
        assert!(secs > 0.0, "Should convert to seconds");
    }

    #[test]
    fn test_get_samples_in_range() {
        let mut buffer = StreamingBuffer::new(512, 44100);
        let samples: Vec<f32> = (0..4410).map(|i| i as f32).collect();
        buffer.add_samples(&samples);

        // Get first 100ms (4410 samples at 44.1kHz)
        let result = buffer.get_samples_in_range(0.0, 100.0);
        assert!(!result.is_empty());
        assert!(result.len() <= 4410);
    }
}
