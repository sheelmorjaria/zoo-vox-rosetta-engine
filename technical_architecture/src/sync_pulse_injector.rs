//! Ultrasonic Sync Pulse Injector
//! ==============================
//!
//! Injects 80kHz ultrasonic sync pulses into audio stream for round-trip
//! latency measurement. These inaudible pulses enable precise correlation
//! between input and output timestamps.
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use crate::ptp::PtpTimestamp;
use anyhow::Result;
use log::debug;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Configuration for sync pulse injection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncPulseConfig {
    /// Interval between pulse injections in milliseconds
    pub pulse_interval_ms: u64,

    /// Audio sample rate in Hz
    pub sample_rate: u32,

    /// Pulse frequency in Hz (ultrasonic, typically 80kHz)
    pub pulse_frequency_hz: u32,

    /// Pulse duration in milliseconds
    pub pulse_duration_ms: f32,

    /// Pulse amplitude (0.0-1.0)
    pub pulse_amplitude: f32,
}

impl Default for SyncPulseConfig {
    fn default() -> Self {
        Self {
            pulse_interval_ms: 5000,  // Every 5 seconds
            sample_rate: 48000,
            pulse_frequency_hz: 80000,  // Ultrasonic
            pulse_duration_ms: 1.0,
            pulse_amplitude: 0.5,
        }
    }
}

/// Record of a sync pulse injection
#[derive(Debug, Clone)]
pub struct PulseInjectionRecord {
    /// Unique pulse ID
    pub pulse_id: u64,

    /// PTP timestamp when pulse was injected
    pub ptp_timestamp: PtpTimestamp,

    /// Sample index in buffer where pulse starts
    pub sample_index: usize,

    /// Injection time in nanoseconds
    pub injection_time_ns: u64,
}

/// Ultrasonic sync pulse injector
pub struct SyncPulseInjector {
    config: SyncPulseConfig,
    next_pulse_id: AtomicU64,
    last_injection_ns: AtomicU64,
    enabled: Arc<AtomicBool>,

    /// Pre-computed pulse waveform
    pulse_waveform: Vec<f32>,
}

impl SyncPulseInjector {
    /// Create a new sync pulse injector
    pub fn new(config: SyncPulseConfig) -> Self {
        let pulse_samples = (config.pulse_duration_ms / 1000.0 * config.sample_rate as f32) as usize;
        let mut pulse_waveform = Vec::with_capacity(pulse_samples);

        // Generate sine wave with Hann window
        for i in 0..pulse_samples {
            let t = i as f32 / config.sample_rate as f32;
            let phase = 2.0 * PI * config.pulse_frequency_hz as f32 * t;
            let window = Self::hann_window(i, pulse_samples);
            pulse_waveform.push(config.pulse_amplitude * (phase).sin() * window);
        }

        Self {
            config,
            next_pulse_id: AtomicU64::new(0),
            last_injection_ns: AtomicU64::new(u64::MAX),  // First call always injects
            enabled: Arc::new(AtomicBool::new(true)),
            pulse_waveform,
        }
    }

    /// Generate Hann window for smooth pulse edges
    fn hann_window(i: usize, n: usize) -> f32 {
        if n <= 1 {
            return 1.0;
        }
        0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos())
    }

    /// Check if a pulse should be injected at the given time
    pub fn should_inject(&self, current_time_ns: u64) -> bool {
        if !self.enabled.load(Ordering::Relaxed) {
            return false;
        }

        let interval_ns = self.config.pulse_interval_ms * 1_000_000;
        let last = self.last_injection_ns.load(Ordering::Relaxed);

        // Handle never-injected case (u64::MAX sentinel)
        if last == u64::MAX {
            return true;
        }

        current_time_ns.saturating_sub(last) >= interval_ns
    }

    /// Inject sync pulse into audio buffer
    ///
    /// Returns pulse record if injection occurred
    pub fn inject_into_buffer(
        &self,
        audio: &mut [f32],
        ptp_timestamp: PtpTimestamp,
        injection_time_ns: u64,
    ) -> Option<PulseInjectionRecord> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }

        if !self.should_inject(injection_time_ns) {
            return None;
        }

        // Update last injection time
        self.last_injection_ns.store(injection_time_ns, Ordering::Relaxed);

        // Get pulse ID
        let pulse_id = self.next_pulse_id.fetch_add(1, Ordering::Relaxed);

        // Inject pulse at start of buffer
        let samples_to_inject = self.pulse_waveform.len().min(audio.len());
        for i in 0..samples_to_inject {
            audio[i] += self.pulse_waveform[i];
        }

        // Clamp to prevent clipping
        for sample in audio.iter_mut() {
            *sample = sample.clamp(-1.0, 1.0);
        }

        let record = PulseInjectionRecord {
            pulse_id,
            ptp_timestamp,
            sample_index: 0,
            injection_time_ns,
        };

        debug!(
            "Injected sync pulse {} at PTP={}.{}s",
            pulse_id, ptp_timestamp.seconds, ptp_timestamp.nanos
        );

        Some(record)
    }

    /// Enable pulse injection
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    /// Disable pulse injection
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    /// Check if injection is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Get the next pulse ID without incrementing
    pub fn peek_next_pulse_id(&self) -> u64 {
        self.next_pulse_id.load(Ordering::Relaxed)
    }

    /// Reset pulse ID counter
    pub fn reset_pulse_id(&self) {
        self.next_pulse_id.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulse_injector_creation() {
        let config = SyncPulseConfig::default();
        let injector = SyncPulseInjector::new(config);

        assert!(injector.is_enabled());
        assert_eq!(injector.peek_next_pulse_id(), 0);
    }

    #[test]
    fn test_should_inject_timing() {
        let config = SyncPulseConfig {
            pulse_interval_ms: 1000,  // 1 second
            ..Default::default()
        };
        let injector = SyncPulseInjector::new(config);

        // First injection should happen
        assert!(injector.should_inject(0));

        // Simulate injection
        injector.last_injection_ns.store(0, Ordering::Relaxed);

        // Should not inject again within interval
        assert!(!injector.should_inject(500_000_000));

        // Should inject after interval
        assert!(injector.should_inject(1_000_000_000));
    }

    #[test]
    fn test_inject_into_buffer() {
        let config = SyncPulseConfig::default();
        let injector = SyncPulseInjector::new(config);

        let mut audio = vec![0.0; 1000];
        let ptp_ts = PtpTimestamp::new(100, 0);

        let record = injector.inject_into_buffer(&mut audio, ptp_ts, 0);

        assert!(record.is_some());
        assert_eq!(record.unwrap().pulse_id, 0);

        // Check that some samples were modified
        let max_value = audio.iter().take(100).fold(0.0_f32, |a, &b| a.max(b.abs()));
        assert!(max_value > 0.0);
    }

    #[test]
    fn test_disable_enable() {
        let config = SyncPulseConfig::default();
        let injector = SyncPulseInjector::new(config);

        injector.disable();
        assert!(!injector.is_enabled());

        injector.enable();
        assert!(injector.is_enabled());
    }

    #[test]
    fn test_hann_window() {
        // Window should be 1.0 at edges and symmetric
        let n = 100;
        let w0 = SyncPulseInjector::hann_window(0, n);
        let w_last = SyncPulseInjector::hann_window(n - 1, n);
        let w_mid = SyncPulseInjector::hann_window(n / 2, n);

        assert!((w0 - 0.0).abs() < 0.01);
        assert!((w_last - 0.0).abs() < 0.01);
        assert!((w_mid - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pulse_id_increment() {
        let config = SyncPulseConfig {
            pulse_interval_ms: 500,  // 0.5 seconds for faster testing
            ..Default::default()
        };
        let injector = SyncPulseInjector::new(config);

        let mut audio = vec![0.0; 1000];
        let ptp_ts = PtpTimestamp::new(100, 0);

        // Force injection by setting last_injection_ns back
        injector.last_injection_ns.store(0, Ordering::Relaxed);

        let r1 = injector.inject_into_buffer(&mut audio, ptp_ts, 500_000_000);
        let r2 = injector.inject_into_buffer(&mut audio, ptp_ts, 1_000_000_000);

        assert_eq!(r1.unwrap().pulse_id, 0);
        assert_eq!(r2.unwrap().pulse_id, 1);
    }

    #[test]
    fn test_reset_pulse_id() {
        let config = SyncPulseConfig::default();
        let injector = SyncPulseInjector::new(config);

        injector.reset_pulse_id();
        assert_eq!(injector.peek_next_pulse_id(), 0);
    }
}
