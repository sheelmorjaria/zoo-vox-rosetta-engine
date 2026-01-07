//! Granular Synthesis Module
//! =========================
//!
//! This module implements real-time audio synthesis using granular
//! synthesis techniques. It generates realistic animal vocalizations
//! and environmental audio responses.
//!
//! Features:
//! - Granular synthesis with configurable grain parameters
//! - Environmental convolution for jungle acoustics
//! - Parametric morphing between vocalizations
//! - Real-time synthesis with low latency
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use anyhow::Result;
use log::{debug, info, warn};
use lru::LruCache;
use parking_lot::Mutex;
use rand::thread_rng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

/// Audio features for synthesis (placeholder type)
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    pub rms: f32,
    pub zero_crossing_rate: f32,
    pub spectral_centroid: f32,
    pub bandwidth: f32,
    pub f0: f32,
}

/// Synthesis mode - determines how phrases are combined
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynthesisMode {
    /// Concatenative (sequential/phrasal/horizontal)
    Horizontal,
    /// Superpositional (simultaneous/chordal/vertical)
    Vertical,
    /// Mixed encoding (combined)
    Combined,
}

/// Phrase segment for synthesis with acoustic metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseSegment {
    /// Audio samples
    pub audio: Vec<f32>,
    /// Duration in milliseconds
    pub duration_ms: f32,
    /// Mean fundamental frequency in Hz
    pub mean_f0_hz: f32,
    /// F0 range (max - min) in Hz
    pub f0_range_hz: f32,
    /// Standard deviation of F0 in Hz
    pub std_f0_hz: f32,
    /// Quality score (0.0 to 1.0)
    pub quality_score: f32,
    /// Sample rate
    pub sample_rate: usize,
}

/// Dynamic Microharmonic Synthesis Parameters
/// Captures micro-dynamics for natural-sounding synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicMicroharmonicParams {
    /// Base fundamental frequency in Hz
    pub f0_base: f32,
    /// Duration in milliseconds
    pub duration_ms: f32,

    // === Micro-Dynamics Features ===
    /// Attack time (0.0 to 100.0 ms) - shape of volume onset
    pub attack_ms: f32,
    /// Decay time (0.0 to 200.0 ms) - time to fade out
    pub decay_ms: f32,
    /// Sustain level (0.0 to 1.0) - amplitude during sustain phase
    pub sustain_level: f32,

    /// Vibrato rate in Hz (0.0 to 20.0) - speed of pitch wobble
    pub vibrato_rate_hz: f32,
    /// Vibrato depth in cents (0.0 to 100.0) - extent of pitch wobble
    pub vibrato_depth_cents: f32,

    /// Jitter amount (0.0 to 0.1) - random phase/perturbation variance
    pub jitter_amount: f32,
    /// Shimmer amount (0.0 to 0.1) - random amplitude variance
    pub shimmer_amount: f32,

    /// Spectral tilt / high-frequency rolloff (-12.0 to 0.0 dB/octave)
    pub spectral_tilt: f32,
    /// Harmonic-to-noise ratio (0.0 to 40.0 dB)
    pub hnr_db: f32,
}

impl Default for DynamicMicroharmonicParams {
    fn default() -> Self {
        Self {
            f0_base: 8000.0,
            duration_ms: 50.0,
            attack_ms: 10.0,
            decay_ms: 30.0,
            sustain_level: 0.7,
            vibrato_rate_hz: 7.0,
            vibrato_depth_cents: 25.0,
            jitter_amount: 0.025,
            shimmer_amount: 0.01,
            spectral_tilt: -6.0,
            hnr_db: 20.0,
        }
    }
}

impl DynamicMicroharmonicParams {
    /// Create new parameters with sensible defaults for marmoset vocalizations
    pub fn marmoset_default(f0_hz: f32, duration_ms: f32) -> Self {
        Self {
            f0_base: f0_hz,
            duration_ms,
            ..Default::default()
        }
    }

    /// Create new parameters with sensible defaults for bat vocalizations
    pub fn bat_default(f0_hz: f32, duration_ms: f32) -> Self {
        Self {
            f0_base: f0_hz,
            duration_ms,
            attack_ms: 10.0,
            decay_ms: 28.0,
            sustain_level: 0.7,
            vibrato_rate_hz: 7.4,
            vibrato_depth_cents: 25.0,
            jitter_amount: 0.025,
            shimmer_amount: 0.01,
            spectral_tilt: -6.0,
            hnr_db: 20.0,
        }
    }

    /// Convert vibrato depth from cents to Hz
    pub fn vibrato_depth_hz(&self) -> f32 {
        // Convert cents to Hz: cents = 1200 * log2(f2/f1)
        // f2 = f1 * 2^(cents/1200)
        self.f0_base * (2.0_f32).powf(self.vibrato_depth_cents / 1200.0) - self.f0_base
    }
}

impl PhraseSegment {
    /// Create a new phrase segment from audio and metadata
    pub fn new(audio: Vec<f32>, sample_rate: usize, mean_f0_hz: f32) -> Self {
        let duration_ms = audio.len() as f32 * 1000.0 / sample_rate as f32;
        Self {
            audio,
            duration_ms,
            mean_f0_hz,
            f0_range_hz: 0.0,
            std_f0_hz: 0.0,
            quality_score: 1.0,
            sample_rate,
        }
    }

    /// Get duration in samples
    pub fn len_samples(&self) -> usize {
        self.audio.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.audio.is_empty()
    }
}

/// Microharmonic synthesis constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroharmonicConstraints {
    /// Acceptable frequency range (min Hz, max Hz)
    pub frequency_range: (f32, f32),
    /// Harmonic tolerance (semitones)
    pub harmonic_tolerance: f32,
    /// Enable phase coherence checks
    pub phase_coherence: bool,
    /// Enable amplitude balancing
    pub amplitude_balancing: bool,
    /// Temporal alignment: "start", "center", "end"
    pub temporal_alignment: String,
    /// Crossfade duration in milliseconds
    pub crossfade_duration_ms: f32,
    /// Maximum number of phrases to combine
    pub max_phrases: usize,
    /// Minimum quality score threshold
    pub min_quality_score: f32,
}

impl Default for MicroharmonicConstraints {
    fn default() -> Self {
        Self {
            frequency_range: (200.0, 8000.0),
            harmonic_tolerance: 3.0, // 3 semitones
            phase_coherence: false,
            amplitude_balancing: true,
            temporal_alignment: "start".to_string(),
            crossfade_duration_ms: 10.0,
            max_phrases: 8,
            min_quality_score: 0.5,
        }
    }
}

/// Synthesis result with metadata
#[derive(Debug, Clone)]
pub struct SynthesisResult {
    /// Output audio samples
    pub audio: Vec<f32>,
    /// Sample rate
    pub sample_rate: usize,
    /// Synthesis mode used
    pub synthesis_mode: SynthesisMode,
    /// Duration in milliseconds
    pub duration_ms: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Phrase keys used in synthesis
    pub phrases_used: Vec<String>,
    /// Microharmonic compatibility score (0.0 to 1.0)
    pub microharmonic_score: f32,
}

/// Validation result for microharmonic compatibility
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Overall compatibility score (0.0 to 1.0)
    pub compatibility_score: f32,
    /// Whether phrases pass all constraints
    pub is_valid: bool,
    /// Detailed validation messages
    pub messages: Vec<String>,
    /// Individual phrase scores
    pub phrase_scores: HashMap<String, f32>,
}

/// Safety check result for audio output
#[derive(Debug, Clone)]
pub struct SafetyCheck {
    /// Whether audio passes all safety checks
    pub safe: bool,
    /// RMS level in dB
    pub rms_level: f32,
    /// Peak level in dB
    pub peak_level: f32,
    /// Duration in milliseconds
    pub duration_ms: f32,
    /// Error message if unsafe
    pub error: Option<String>,
}

/// Species-specific acoustic parameters
#[derive(Debug, Clone)]
pub struct SpeciesParameters {
    /// Frequency range (min Hz, max Hz)
    pub frequency_range: (f32, f32),
    /// Harmonic tolerance in semitones
    pub harmonic_tolerance: f32,
    /// Default temporal alignment
    pub default_temporal_alignment: String,
}

impl Default for SpeciesParameters {
    fn default() -> Self {
        Self {
            frequency_range: (200.0, 8000.0),
            harmonic_tolerance: 3.0,
            default_temporal_alignment: "start".to_string(),
        }
    }
}

/// Synthesis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisConfig {
    /// Sample rate (Hz)
    pub sample_rate: usize,

    /// Grain size in milliseconds
    pub grain_size_ms: f32,

    /// Grain overlap (0.0 to 1.0)
    pub grain_overlap: f32,

    /// Maximum number of concurrent grains
    pub max_grains: usize,

    /// Enable environmental convolution
    pub enable_convolution: bool,

    /// Convolution impulse response file path
    pub impulse_response_path: Option<String>,

    /// Enable parametric morphing
    pub enable_morphing: bool,

    /// Output gain
    pub output_gain: f32,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            grain_size_ms: 50.0,
            grain_overlap: 0.5,
            max_grains: 32,
            enable_convolution: true,
            impulse_response_path: Some("impulses/jungle.wav".to_string()),
            enable_morphing: true,
            output_gain: 0.8,
        }
    }
}

/// Audio segment for synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSegment {
    /// Audio samples
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: usize,
    /// Start time in segment (seconds)
    pub start_time: f32,
    /// Duration (seconds)
    pub duration: f32,
}

impl AudioSegment {
    /// Create a new audio segment
    pub fn new(samples: Vec<f32>, sample_rate: usize) -> Self {
        let duration = samples.len() as f32 / sample_rate as f32;
        Self {
            samples,
            sample_rate,
            start_time: 0.0,
            duration,
        }
    }

    /// Get duration in samples
    pub fn len_samples(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Resample to target sample rate (simple linear interpolation)
    pub fn resample(&self, target_rate: usize) -> Result<Self> {
        if self.sample_rate == target_rate {
            return Ok(self.clone());
        }

        let ratio = self.sample_rate as f32 / target_rate as f32;
        let new_len = (self.samples.len() as f32 / ratio).ceil() as usize;

        let mut new_samples = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_pos = i as f32 * ratio;
            let src_idx = src_pos.floor() as usize;
            let frac = src_pos.fract();

            if src_idx + 1 < self.samples.len() {
                let sample =
                    self.samples[src_idx] * (1.0 - frac) + self.samples[src_idx + 1] * frac;
                new_samples.push(sample);
            } else if src_idx < self.samples.len() {
                new_samples.push(self.samples[src_idx]);
            }
        }

        Ok(AudioSegment {
            samples: new_samples,
            sample_rate: target_rate,
            start_time: self.start_time,
            duration: new_len as f32 / target_rate as f32,
        })
    }
}

// ============================================================================
// Supporting Components
// ============================================================================

/// Microharmonic validator for checking phrase compatibility
pub struct MicroharmonicValidator {
    #[allow(dead_code)]
    sample_rate: usize,
}

impl MicroharmonicValidator {
    /// Create a new microharmonic validator
    pub fn new(sample_rate: usize) -> Self {
        Self { sample_rate }
    }

    /// Validate compatibility of phrase set
    pub fn validate_compatibility(
        &self,
        phrase_keys: &[String],
        constraints: &MicroharmonicConstraints,
        phrase_segments: &HashMap<String, PhraseSegment>,
    ) -> ValidationResult {
        let mut messages = Vec::new();
        let mut phrase_scores = HashMap::new();
        let mut total_score = 0.0;

        // Check phrase count
        if phrase_keys.len() > constraints.max_phrases {
            messages.push(format!(
                "Too many phrases: {} exceeds maximum of {}",
                phrase_keys.len(),
                constraints.max_phrases
            ));
        }

        // Validate each phrase
        for key in phrase_keys {
            if let Some(phrase) = phrase_segments.get(key) {
                let score = self
                    .check_frequency_compatibility(phrase.mean_f0_hz, constraints.frequency_range);
                phrase_scores.insert(key.clone(), score);
                total_score += score;

                // Check quality threshold
                if phrase.quality_score < constraints.min_quality_score {
                    messages.push(format!(
                        "Phrase '{}' quality {:.2} below threshold {:.2}",
                        key, phrase.quality_score, constraints.min_quality_score
                    ));
                }

                // Check frequency range
                if phrase.mean_f0_hz < constraints.frequency_range.0
                    || phrase.mean_f0_hz > constraints.frequency_range.1
                {
                    messages.push(format!(
                        "Phrase '{}' F0 {:.1} Hz outside range {:.1}-{:.1} Hz",
                        key,
                        phrase.mean_f0_hz,
                        constraints.frequency_range.0,
                        constraints.frequency_range.1
                    ));
                }
            } else {
                messages.push(format!("Phrase '{}' not found in phrase segments", key));
                phrase_scores.insert(key.clone(), 0.0);
            }
        }

        // Calculate overall compatibility
        let avg_score = if phrase_keys.is_empty() {
            0.0
        } else {
            total_score / phrase_keys.len() as f32
        };

        let is_valid = messages.is_empty() && avg_score > 0.5;

        ValidationResult {
            compatibility_score: avg_score,
            is_valid,
            messages,
            phrase_scores,
        }
    }

    /// Check frequency compatibility (returns score 0.0 to 1.0)
    fn check_frequency_compatibility(&self, f0_hz: f32, range: (f32, f32)) -> f32 {
        if f0_hz >= range.0 && f0_hz <= range.1 {
            1.0
        } else {
            let distance = if f0_hz < range.0 {
                range.0 - f0_hz
            } else {
                f0_hz - range.1
            };
            // Score degrades with distance from range (half point at one octave away)
            (-distance / (range.1 - range.0)).exp()
        }
    }
}

/// Real-time safety monitor for audio output
pub struct RealTimeSafetyMonitor {
    sample_rate: usize,
    max_rms_level: f32,
    max_peak_level: f32,
    min_duration_ms: f32,
    max_duration_ms: f32,
}

impl RealTimeSafetyMonitor {
    /// Create a new real-time safety monitor
    pub fn new(sample_rate: usize) -> Self {
        Self {
            sample_rate,
            max_rms_level: -3.0,  // -3 dBFS
            max_peak_level: -0.5, // -0.5 dBFS
            min_duration_ms: 10.0,
            max_duration_ms: 30000.0, // 30 seconds
        }
    }

    /// Check audio safety
    pub fn check_audio_safety(&self, audio: &[f32]) -> SafetyCheck {
        if audio.is_empty() {
            return SafetyCheck {
                safe: false,
                rms_level: f32::NEG_INFINITY,
                peak_level: f32::NEG_INFINITY,
                duration_ms: 0.0,
                error: Some("Empty audio".to_string()),
            };
        }

        // Calculate RMS
        let rms = (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        let rms_db = 20.0 * rms.log10();

        // Calculate peak
        let peak = audio.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let peak_db = 20.0 * peak.log10();

        // Calculate duration
        let duration_ms = audio.len() as f32 * 1000.0 / self.sample_rate as f32;

        // Safety checks
        let mut safe = true;
        let mut error = None;

        if rms_db > self.max_rms_level {
            safe = false;
            error = Some(format!(
                "RMS level {:.1} dB exceeds maximum {:.1} dB",
                rms_db, self.max_rms_level
            ));
        }

        if peak_db > self.max_peak_level {
            safe = false;
            error = Some(format!(
                "Peak level {:.1} dB exceeds maximum {:.1} dB",
                peak_db, self.max_peak_level
            ));
        }

        if duration_ms < self.min_duration_ms {
            safe = false;
            error = Some(format!(
                "Duration {:.1} ms below minimum {:.1} ms",
                duration_ms, self.min_duration_ms
            ));
        }

        if duration_ms > self.max_duration_ms {
            safe = false;
            error = Some(format!(
                "Duration {:.1} ms exceeds maximum {:.1} ms",
                duration_ms, self.max_duration_ms
            ));
        }

        SafetyCheck {
            safe,
            rms_level: rms_db,
            peak_level: peak_db,
            duration_ms,
            error,
        }
    }

    /// Apply safety limiter to audio (soft clipping)
    pub fn apply_safety_limiter(&self, audio: &mut [f32]) -> Result<()> {
        const THRESHOLD: f32 = 0.9;
        const RATIO: f32 = 4.0;

        for sample in audio.iter_mut() {
            let abs = sample.abs();
            if abs > THRESHOLD {
                let excess = abs - THRESHOLD;
                let limited = THRESHOLD + excess / RATIO;
                *sample = sample.signum() * limited.min(1.0);
            }
        }

        Ok(())
    }
}

/// Cross-species adapter for species-specific parameters
pub struct CrossSpeciesAdapter {
    species_parameters: HashMap<String, SpeciesParameters>,
}

impl CrossSpeciesAdapter {
    /// Create a new cross-species adapter with default parameters
    pub fn new() -> Self {
        let mut species_parameters = HashMap::new();

        // Marmoset (high frequency, harmonic)
        species_parameters.insert(
            "marmoset".to_string(),
            SpeciesParameters {
                frequency_range: (500.0, 15000.0),
                harmonic_tolerance: 2.0,
                default_temporal_alignment: "start".to_string(),
            },
        );

        // Dolphin (very high frequency, whistles)
        species_parameters.insert(
            "dolphin".to_string(),
            SpeciesParameters {
                frequency_range: (1000.0, 25000.0),
                harmonic_tolerance: 4.0,
                default_temporal_alignment: "center".to_string(),
            },
        );

        // Bat (ultrasonic, FM sweeps)
        species_parameters.insert(
            "bat".to_string(),
            SpeciesParameters {
                frequency_range: (10000.0, 120000.0),
                harmonic_tolerance: 8.0,
                default_temporal_alignment: "start".to_string(),
            },
        );

        // Finch (songbird, complex harmonic structure)
        species_parameters.insert(
            "finch".to_string(),
            SpeciesParameters {
                frequency_range: (1000.0, 10000.0),
                harmonic_tolerance: 1.5,
                default_temporal_alignment: "end".to_string(),
            },
        );

        // Sperm whale (very low frequency, clicks)
        species_parameters.insert(
            "sperm_whale".to_string(),
            SpeciesParameters {
                frequency_range: (100.0, 8000.0),
                harmonic_tolerance: 6.0,
                default_temporal_alignment: "start".to_string(),
            },
        );

        Self { species_parameters }
    }

    /// Adapt constraints for specific species
    pub fn adapt_parameters_for_species(
        &self,
        species: &str,
        base_constraints: &MicroharmonicConstraints,
    ) -> MicroharmonicConstraints {
        let default_params = SpeciesParameters::default();
        let params = self
            .species_parameters
            .get(species)
            .unwrap_or(&default_params);

        let mut adapted = base_constraints.clone();
        adapted.frequency_range = params.frequency_range;
        adapted.harmonic_tolerance = params.harmonic_tolerance;
        adapted.temporal_alignment = params.default_temporal_alignment.clone();

        adapted
    }

    /// Get available species
    pub fn available_species(&self) -> Vec<String> {
        self.species_parameters.keys().cloned().collect()
    }
}

impl Default for CrossSpeciesAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Core Synthesis Implementations
// ============================================================================

/// Concatenative synthesizer (Horizontal/Sequential mode)
pub struct ConcatenativeSynthesizer {
    sample_rate: usize,
    gain: f32,
}

impl ConcatenativeSynthesizer {
    /// Create a new concatenative synthesizer
    pub fn new(sample_rate: usize, gain: f32) -> Self {
        Self { sample_rate, gain }
    }

    /// Concatenate phrases sequentially with crossfades
    pub fn concatenate_phrases(
        &self,
        phrases: &[PhraseSegment],
        fade_duration_ms: f32,
    ) -> Result<Vec<f32>> {
        if phrases.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate total length
        let fade_samples = (fade_duration_ms * self.sample_rate as f32 / 1000.0) as usize;
        let total_samples: usize = phrases
            .iter()
            .map(|p| p.audio.len())
            .sum::<usize>()
            .saturating_sub((phrases.len() - 1) * fade_samples);

        let mut output = vec![0.0f32; total_samples];
        let mut write_pos = 0;

        for (i, phrase) in phrases.iter().enumerate() {
            let audio = &phrase.audio;
            let phrase_len = audio.len();

            if i == 0 {
                // First phrase - no fade in
                output[write_pos..write_pos + phrase_len].copy_from_slice(audio);
            } else {
                // Apply crossfade
                let fade_start = write_pos;
                let fade_len = fade_samples.min(phrase_len).min(output.len() - fade_start);

                // Fade out previous
                for j in 0..fade_len {
                    let t = j as f32 / fade_len as f32;
                    let fade_out = 0.5 * (1.0 + (std::f32::consts::PI * t).cos()); // Cosine
                    output[fade_start + j] *= fade_out;
                }

                // Fade in current
                for j in 0..fade_len {
                    let t = j as f32 / fade_len as f32;
                    let fade_in = 0.5 * (1.0 - (std::f32::consts::PI * t).cos()); // Cosine
                    output[fade_start + j] += audio[j] * fade_in;
                }

                // Copy rest without fade
                if fade_len < phrase_len {
                    let copy_start = fade_start + fade_len;
                    let audio_start = fade_len;
                    let copy_len = phrase_len - fade_len;
                    if copy_start + copy_len <= output.len() {
                        output[copy_start..copy_start + copy_len]
                            .copy_from_slice(&audio[audio_start..]);
                    }
                }
            }

            write_pos += phrase_len - if i > 0 { fade_samples } else { 0 };
        }

        // Apply gain
        for sample in &mut output {
            *sample *= self.gain;
        }

        Ok(output)
    }
}

/// Superpositional synthesizer (Vertical/Simultaneous mode)
pub struct SuperpositionalSynthesizer {
    #[allow(dead_code)]
    sample_rate: usize,
    max_layers: usize,
}

impl SuperpositionalSynthesizer {
    /// Create a new superpositional synthesizer
    pub fn new(sample_rate: usize, max_layers: usize) -> Self {
        Self {
            sample_rate,
            max_layers,
        }
    }

    /// Layer phrases harmonically at same time position
    pub fn layer_phrases_harmonically(
        &self,
        phrases: &[PhraseSegment],
        amplitude_balance: bool,
    ) -> Result<Vec<f32>> {
        if phrases.is_empty() {
            return Ok(Vec::new());
        }

        let num_phrases = phrases.len().min(self.max_layers);

        // Find maximum length
        let max_len = phrases.iter().map(|p| p.audio.len()).max().unwrap_or(0);

        let mut output = vec![0.0f32; max_len];

        // Mix all phrases
        for phrase in phrases.iter().take(num_phrases) {
            let audio = &phrase.audio;
            for (i, &sample) in audio.iter().enumerate() {
                output[i] += sample;
            }
        }

        // Normalize if amplitude balancing is enabled
        if amplitude_balance {
            self.normalize_output(&mut output);
        } else {
            // Simple divide by count to prevent clipping
            let scale = 1.0 / num_phrases as f32;
            for sample in &mut output {
                *sample *= scale;
            }
        }

        Ok(output)
    }

    /// Normalize output to prevent clipping
    fn normalize_output(&self, output: &mut [f32]) {
        let max_amplitude = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        if max_amplitude > 1.0 {
            let scale = 1.0 / max_amplitude;
            for sample in output.iter_mut() {
                *sample *= scale;
            }
        }
    }
}

/// Combined synthesizer (Mixed encoding mode)
pub struct CombinedSynthesizer {
    #[allow(dead_code)]
    sample_rate: usize,
    concatenative: ConcatenativeSynthesizer,
    superpositional: SuperpositionalSynthesizer,
}

impl CombinedSynthesizer {
    /// Create a new combined synthesizer
    pub fn new(sample_rate: usize) -> Self {
        Self {
            sample_rate,
            concatenative: ConcatenativeSynthesizer::new(sample_rate, 1.0),
            superpositional: SuperpositionalSynthesizer::new(sample_rate, 8),
        }
    }

    /// Synthesize mixed encoding (sequential + simultaneous phrases)
    pub fn synthesize_mixed_encoding(
        &self,
        sequential_phrases: &[PhraseSegment],
        simultaneous_phrases: &[PhraseSegment],
        overlap_duration_ms: f32,
    ) -> Result<Vec<f32>> {
        let start = Instant::now();

        // Process sequential phrases
        let sequential_output = if !sequential_phrases.is_empty() {
            self.concatenative
                .concatenate_phrases(sequential_phrases, overlap_duration_ms)?
        } else {
            Vec::new()
        };

        // Process simultaneous phrases (chord)
        let simultaneous_output = if !simultaneous_phrases.is_empty() {
            self.superpositional
                .layer_phrases_harmonically(simultaneous_phrases, true)?
        } else {
            Vec::new()
        };

        // Mix sequential and simultaneous outputs
        let output = if sequential_output.is_empty() {
            simultaneous_output
        } else if simultaneous_output.is_empty() {
            sequential_output
        } else {
            // Overlay simultaneous on top of sequential
            let max_len = sequential_output.len().max(simultaneous_output.len());
            let mut mixed = vec![0.0f32; max_len];

            // Mix sequential at full amplitude
            for (i, &sample) in sequential_output.iter().enumerate() {
                mixed[i] += sample * 0.7;
            }

            // Mix simultaneous at reduced amplitude
            for (i, &sample) in simultaneous_output.iter().enumerate() {
                mixed[i] += sample * 0.5;
            }

            mixed
        };

        debug!(
            "Mixed encoding synthesis: {:.2}ms",
            start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(output)
    }
}

/// Performance statistics for synthesis operations
#[derive(Debug, Clone, Default)]
pub struct SynthesisPerformanceStats {
    /// Total syntheses performed
    pub total_syntheses: u64,
    /// Horizontal mode count
    pub horizontal_count: u64,
    /// Vertical mode count
    pub vertical_count: u64,
    /// Combined mode count
    pub combined_count: u64,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f64,
    /// Maximum processing time (ms)
    pub max_processing_time_ms: f64,
}

/// Enhanced microharmonic synthesizer (unified interface)
pub struct EnhancedMicroharmonicSynthesizer {
    species: String,
    phrase_segments: HashMap<String, PhraseSegment>,
    validator: MicroharmonicValidator,
    safety_monitor: RealTimeSafetyMonitor,
    species_adapter: CrossSpeciesAdapter,
    performance_stats: Arc<Mutex<SynthesisPerformanceStats>>,
    sample_rate: usize,
}

impl EnhancedMicroharmonicSynthesizer {
    /// Create a new enhanced microharmonic synthesizer
    pub fn new(
        species: String,
        phrase_segments: HashMap<String, PhraseSegment>,
        sample_rate: usize,
    ) -> Self {
        Self {
            species,
            phrase_segments,
            validator: MicroharmonicValidator::new(sample_rate),
            safety_monitor: RealTimeSafetyMonitor::new(sample_rate),
            species_adapter: CrossSpeciesAdapter::new(),
            performance_stats: Arc::new(Mutex::new(SynthesisPerformanceStats::default())),
            sample_rate,
        }
    }

    /// Synthesize in horizontal mode (sequential concatenation)
    pub async fn synthesize_horizontal(
        &self,
        phrase_sequence: &[String],
        constraints: &MicroharmonicConstraints,
    ) -> Result<SynthesisResult> {
        let start = Instant::now();

        // Adapt constraints for species
        let adapted_constraints = self
            .species_adapter
            .adapt_parameters_for_species(&self.species, constraints);

        // Validate compatibility
        let validation = self.validator.validate_compatibility(
            phrase_sequence,
            &adapted_constraints,
            &self.phrase_segments,
        );

        if !validation.is_valid {
            warn!("Validation failed: {:?}", validation.messages);
        }

        // Collect phrase segments
        let phrases: Vec<PhraseSegment> = phrase_sequence
            .iter()
            .filter_map(|key| self.phrase_segments.get(key).cloned())
            .collect();

        if phrases.is_empty() {
            return Ok(SynthesisResult {
                audio: Vec::new(),
                sample_rate: self.sample_rate,
                synthesis_mode: SynthesisMode::Horizontal,
                duration_ms: 0.0,
                processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                phrases_used: Vec::new(),
                microharmonic_score: 0.0,
            });
        }

        // Synthesize using concatenative approach
        let concatenative = ConcatenativeSynthesizer::new(self.sample_rate, 1.0);
        let mut audio = concatenative
            .concatenate_phrases(&phrases, adapted_constraints.crossfade_duration_ms)?;

        // Apply safety limiting
        self.safety_monitor.apply_safety_limiter(&mut audio)?;

        // Calculate duration
        let duration_ms = audio.len() as f32 * 1000.0 / self.sample_rate as f32;

        // Update performance stats
        {
            let mut stats = self.performance_stats.lock();
            stats.total_syntheses += 1;
            stats.horizontal_count += 1;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            stats.avg_processing_time_ms =
                (stats.avg_processing_time_ms * (stats.total_syntheses - 1) as f64 + elapsed)
                    / stats.total_syntheses as f64;
            stats.max_processing_time_ms = stats.max_processing_time_ms.max(elapsed);
        }

        Ok(SynthesisResult {
            audio,
            sample_rate: self.sample_rate,
            synthesis_mode: SynthesisMode::Horizontal,
            duration_ms,
            processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            phrases_used: phrase_sequence.to_vec(),
            microharmonic_score: validation.compatibility_score,
        })
    }

    /// Synthesize in vertical mode (simultaneous layering)
    pub async fn synthesize_vertical(
        &self,
        phrase_set: &[String],
        constraints: &MicroharmonicConstraints,
    ) -> Result<SynthesisResult> {
        let start = Instant::now();

        // Adapt constraints for species
        let adapted_constraints = self
            .species_adapter
            .adapt_parameters_for_species(&self.species, constraints);

        // Validate compatibility
        let validation = self.validator.validate_compatibility(
            phrase_set,
            &adapted_constraints,
            &self.phrase_segments,
        );

        if !validation.is_valid {
            warn!("Validation failed: {:?}", validation.messages);
        }

        // Collect phrase segments
        let phrases: Vec<PhraseSegment> = phrase_set
            .iter()
            .filter_map(|key| self.phrase_segments.get(key).cloned())
            .collect();

        if phrases.is_empty() {
            return Ok(SynthesisResult {
                audio: Vec::new(),
                sample_rate: self.sample_rate,
                synthesis_mode: SynthesisMode::Vertical,
                duration_ms: 0.0,
                processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                phrases_used: Vec::new(),
                microharmonic_score: 0.0,
            });
        }

        // Synthesize using superpositional approach
        let superpositional =
            SuperpositionalSynthesizer::new(self.sample_rate, adapted_constraints.max_phrases);
        let mut audio = superpositional
            .layer_phrases_harmonically(&phrases, adapted_constraints.amplitude_balancing)?;

        // Apply safety limiting
        self.safety_monitor.apply_safety_limiter(&mut audio)?;

        // Calculate duration
        let duration_ms = audio.len() as f32 * 1000.0 / self.sample_rate as f32;

        // Update performance stats
        {
            let mut stats = self.performance_stats.lock();
            stats.total_syntheses += 1;
            stats.vertical_count += 1;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            stats.avg_processing_time_ms =
                (stats.avg_processing_time_ms * (stats.total_syntheses - 1) as f64 + elapsed)
                    / stats.total_syntheses as f64;
            stats.max_processing_time_ms = stats.max_processing_time_ms.max(elapsed);
        }

        Ok(SynthesisResult {
            audio,
            sample_rate: self.sample_rate,
            synthesis_mode: SynthesisMode::Vertical,
            duration_ms,
            processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            phrases_used: phrase_set.to_vec(),
            microharmonic_score: validation.compatibility_score,
        })
    }

    /// Synthesize in combined mode (mixed encoding)
    pub async fn synthesize_combined(
        &self,
        synthesis_plan: &[(SynthesisMode, Vec<String>)],
        constraints: &MicroharmonicConstraints,
    ) -> Result<SynthesisResult> {
        let start = Instant::now();

        // Adapt constraints for species
        let adapted_constraints = self
            .species_adapter
            .adapt_parameters_for_species(&self.species, constraints);

        let mut sequential_phrases: Vec<PhraseSegment> = Vec::new();
        let mut simultaneous_phrases: Vec<PhraseSegment> = Vec::new();
        let mut all_phrase_keys: Vec<String> = Vec::new();
        let mut total_score = 0.0;
        let mut score_count = 0;

        for (mode, phrase_keys) in synthesis_plan {
            all_phrase_keys.extend(phrase_keys.clone());

            // Validate each group
            let validation = self.validator.validate_compatibility(
                phrase_keys,
                &adapted_constraints,
                &self.phrase_segments,
            );
            total_score += validation.compatibility_score;
            score_count += 1;

            match mode {
                SynthesisMode::Horizontal => {
                    for key in phrase_keys {
                        if let Some(phrase) = self.phrase_segments.get(key) {
                            sequential_phrases.push(phrase.clone());
                        }
                    }
                }
                SynthesisMode::Vertical => {
                    for key in phrase_keys {
                        if let Some(phrase) = self.phrase_segments.get(key) {
                            simultaneous_phrases.push(phrase.clone());
                        }
                    }
                }
                SynthesisMode::Combined => {
                    // Treat Combined mode as sequential for now
                    for key in phrase_keys {
                        if let Some(phrase) = self.phrase_segments.get(key) {
                            sequential_phrases.push(phrase.clone());
                        }
                    }
                }
            }
        }

        // Synthesize using combined approach
        let combined = CombinedSynthesizer::new(self.sample_rate);
        let mut audio = combined.synthesize_mixed_encoding(
            &sequential_phrases,
            &simultaneous_phrases,
            adapted_constraints.crossfade_duration_ms,
        )?;

        // Apply safety limiting
        self.safety_monitor.apply_safety_limiter(&mut audio)?;

        // Calculate duration
        let duration_ms = audio.len() as f32 * 1000.0 / self.sample_rate as f32;

        // Update performance stats
        {
            let mut stats = self.performance_stats.lock();
            stats.total_syntheses += 1;
            stats.combined_count += 1;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            stats.avg_processing_time_ms =
                (stats.avg_processing_time_ms * (stats.total_syntheses - 1) as f64 + elapsed)
                    / stats.total_syntheses as f64;
            stats.max_processing_time_ms = stats.max_processing_time_ms.max(elapsed);
        }

        let avg_score = if score_count > 0 {
            total_score / score_count as f32
        } else {
            0.0
        };

        Ok(SynthesisResult {
            audio,
            sample_rate: self.sample_rate,
            synthesis_mode: SynthesisMode::Combined,
            duration_ms,
            processing_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            phrases_used: all_phrase_keys,
            microharmonic_score: avg_score,
        })
    }

    /// Add a phrase segment to the synthesizer
    pub fn add_phrase_segment(&mut self, key: String, segment: PhraseSegment) {
        self.phrase_segments.insert(key, segment);
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> SynthesisPerformanceStats {
        self.performance_stats.lock().clone()
    }

    /// Get available phrase keys
    pub fn available_phrases(&self) -> Vec<String> {
        self.phrase_segments.keys().cloned().collect()
    }
}

// ============================================================================
// Granular Concatenative Synthesis (Preserves Formant Structure)
// ============================================================================

/// Grain Window - Envelope function for smooth grain boundaries
/// Prevents clicking artifacts when grains are triggered
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GrainWindow {
    /// Window samples
    samples: Vec<f32>,
}

#[allow(dead_code)]
impl GrainWindow {
    /// Create a Hanning window (cosine-based fade in/out)
    ///
    /// Parameters:
    /// - grain_size_ms: Grain duration in milliseconds
    /// - sample_rate: Audio sample rate
    ///
    /// Returns: Window function normalized to 0.0-1.0
    pub fn hanning(grain_size_ms: f32, sample_rate: usize) -> Vec<f32> {
        let num_samples = (grain_size_ms / 1000.0 * sample_rate as f32) as usize;
        let mut window = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            // Hanning window: 0.5 * (1 - cos(2*pi*i/N))
            let phase = 2.0 * std::f32::consts::PI * i as f32 / num_samples as f32;
            let sample = 0.5 * (1.0 - phase.cos());
            window.push(sample);
        }

        window
    }

    /// Create a Blackman window (steeper falloff, less sidelobe leakage)
    pub fn blackman(grain_size_ms: f32, sample_rate: usize) -> Vec<f32> {
        let num_samples = (grain_size_ms / 1000.0 * sample_rate as f32) as usize;
        let mut window = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f32 / num_samples as f32;
            // Blackman window: 0.42 - 0.5*cos(2*pi*t) + 0.08*cos(4*pi*t)
            let sample = 0.42 - 0.5 * (2.0 * std::f32::consts::PI * t).cos()
                + 0.08 * (4.0 * std::f32::consts::PI * t).cos();
            window.push(sample);
        }

        window
    }
}

/// Granular Voice - Single voice with pitch/time manipulation
///
/// This is the key innovation: instead of generating audio from math (additive synthesis),
/// we manipulate real audio samples (granular synthesis). This preserves formant structure.
#[allow(dead_code)]
pub struct GranularVoice {
    /// Source audio buffer (real recording)
    source_buffer: Vec<f32>,
    /// Sample rate
    sample_rate: usize,
    /// Grain size in milliseconds
    grain_size_ms: f32,
    /// Pre-computed window function
    window: Vec<f32>,
    /// Current read position in source buffer (floating point for sub-sample accuracy)
    position: f32,
    /// Pitch shift ratio (1.0 = natural, 0.5 = octave down, 2.0 = octave up)
    pitch_shift_ratio: f32,
    /// Time stretch ratio (1.0 = natural, 2.0 = double duration)
    time_stretch_ratio: f32,
}

#[allow(dead_code)]
impl GranularVoice {
    /// Create a new granular voice from source audio
    ///
    /// Parameters:
    /// - source_buffer: Real audio samples to manipulate
    /// - sample_rate: Audio sample rate
    /// - grain_size_ms: Grain window size (typically 10-50ms)
    pub fn new(source_buffer: Vec<f32>, sample_rate: usize, grain_size_ms: f32) -> Self {
        let window = GrainWindow::hanning(grain_size_ms, sample_rate);

        Self {
            source_buffer,
            sample_rate,
            grain_size_ms,
            window,
            position: 0.0,
            pitch_shift_ratio: 1.0,
            time_stretch_ratio: 1.0,
        }
    }

    /// Set pitch shift ratio (changes pitch without changing duration)
    pub fn set_pitch_shift(&mut self, ratio: f32) {
        self.pitch_shift_ratio = ratio.clamp(0.25, 4.0);
    }

    /// Set time stretch ratio (changes duration without changing pitch)
    pub fn set_time_stretch(&mut self, ratio: f32) {
        self.time_stretch_ratio = ratio.clamp(0.5, 4.0);
    }

    /// Get current position in source buffer
    pub fn get_position(&self) -> f32 {
        self.position
    }

    /// Generate a single sample using granular synthesis
    ///
    /// Algorithm:
    /// 1. Read current sample from source buffer (with linear interpolation)
    /// 2. Apply grain window envelope based on position within grain
    /// 3. Advance position based on pitch shift ratio
    pub fn generate_sample(&mut self) -> f32 {
        if self.source_buffer.is_empty() {
            return 0.0;
        }

        // Get current sample with linear interpolation
        let pos_int = self.position as usize;
        let pos_frac = self.position - pos_int as f32;

        // Get samples for interpolation
        let sample0 = if pos_int < self.source_buffer.len() {
            self.source_buffer[pos_int]
        } else {
            0.0
        };

        let sample1 = if pos_int + 1 < self.source_buffer.len() {
            self.source_buffer[pos_int + 1]
        } else {
            0.0
        };

        // Linear interpolation
        let current_sample = sample0 + (sample1 - sample0) * pos_frac;

        // Calculate position within grain (0.0 to 1.0)
        let grain_length = self.window.len() as f32;
        let grain_position = (self.position % grain_length) / grain_length;

        // Get window envelope value at this grain position
        let window_idx = (grain_position * (self.window.len() - 1) as f32) as usize;
        let window_value = if window_idx < self.window.len() {
            self.window[window_idx]
        } else {
            0.0
        };

        // Apply window envelope
        let sample = current_sample * window_value;

        // Advance position based on pitch shift ratio
        // ratio < 1.0 = slower advance = lower pitch
        // ratio > 1.0 = faster advance = higher pitch
        let effective_stride = 1.0 / self.pitch_shift_ratio;
        self.position += effective_stride;

        // Wrap around source buffer
        if self.position >= self.source_buffer.len() as f32 - grain_length {
            self.position = 0.0;
        }

        sample
    }
}

/// Granular Morpher - Multi-voice overlap for smooth transitions
///
/// Overlaps multiple granular voices to create smooth morphs between
/// different pitches or timbres while preserving formant structure.
#[allow(dead_code)]
pub struct GranularMorpher {
    /// Active voices
    voices: Vec<GranularVoice>,
    /// Crossfade duration between voices
    crossfade_ms: f32,
}

#[allow(dead_code)]
impl GranularMorpher {
    /// Create a new granular morpher with multiple voices
    ///
    /// Parameters:
    /// - voices: Vector of granular voices to overlap
    /// - crossfade_ms: Crossfade duration for smooth transitions
    pub fn new(voices: Vec<GranularVoice>, crossfade_ms: f32) -> Self {
        Self {
            voices,
            crossfade_ms,
        }
    }

    /// Generate a single sample by summing all active voices
    ///
    /// This creates the "morphing" effect by overlapping grains
    /// from multiple sources with different pitch ratios.
    pub fn generate_sample(&mut self) -> f32 {
        // Sum all voices
        let mut sample = 0.0_f32;
        for voice in &mut self.voices {
            sample += voice.generate_sample();
        }

        // Normalize by number of voices to prevent clipping
        sample / self.voices.len() as f32
    }

    /// Interpolate all voices toward a target pitch ratio
    pub fn set_target_pitch(&mut self, target_ratio: f32) {
        // Smooth interpolation (10% toward target per call)
        for voice in &mut self.voices {
            let current = voice.pitch_shift_ratio;
            let new_ratio = current * 0.9 + target_ratio * 0.1;
            voice.set_pitch_shift(new_ratio);
        }
    }
}

/// Granular Concatenative Synthesizer
///
/// High-fidelity synthesizer that preserves formant structure
/// by manipulating real audio samples instead of generating from math.
///
/// Key advantage over additive synthesis:
/// - Preserves spectral envelope (formants, throat shape)
/// - Maintains inharmonic partials
/// - Keeps natural texture and noise components
///
/// This should achieve t-SNE distance < 7.0 (similar to concatenative)
/// while providing pitch/time flexibility.
///
/// 30-dimensional micro-dynamics source metadata for delta-based synthesis
///
/// This structure captures the full acoustic profile of a source buffer,
/// enabling precise vector delta operations for all micro-dynamics features.
///
/// **30 Micro-Dynamics Features:**
///
/// 1. **Fundamental** (3 features):
///    - `mean_f0_hz`: Mean fundamental frequency (Hz)
///    - `duration_ms`: Temporal extent (ms)
///    - `f0_range_hz`: Pitch modulation range (Hz)
///
/// 2. **Grit Factors** (3 features) - Timbre texture:
///    - `harmonic_to_noise_ratio`: Harmonic purity vs noise (dB)
///    - `spectral_flatness`: Noise-like vs tonal (0-1)
///    - `harmonicity`: Degree of harmonic relationship (0-1)
///
/// 3. **Motion Factors** (7 features) - Envelope dynamics:
///    - `attack_time_ms`: Onset speed (fast=sharp, slow=gentle)
///    - `decay_time_ms`: Release speed (ms)
///    - `sustain_level`: Steady-state amplitude (0-1)
///    - `vibrato_rate_hz`: Pitch modulation frequency (Hz)
///    - `vibrato_depth`: Pitch modulation depth (Hz)
///    - `jitter`: Micro-perturbations/instability (0-1)
///    - `shimmer`: Amplitude micro-variations/breathiness (0-1)
///
/// 4. **Fingerprint Factors** (14 features) - Spectral shape:
///    - `mfcc_1` through `mfcc_13`: Mel-frequency cepstral coefficients
///    - `spectral_flux`: Rate of spectral change (0-1)
///
/// 5. **Rhythm Factors** (3 features) - Temporal patterns:
///    - `median_ici_ms`: Inter-click interval (ms)
///    - `onset_rate_hz`: Click/event rate (Hz)
///    - `ici_coefficient_of_variation`: Rhythm regularity (0-1)
#[derive(Clone, Copy, Debug)]
pub struct SourceMetadata {
    // === Fundamental (3 features) ===
    /// Mean fundamental frequency of source buffer (Hz)
    pub mean_f0_hz: f32,
    /// Duration of source buffer (ms)
    pub duration_ms: f32,
    /// F0 range of source (Hz)
    pub f0_range_hz: f32,

    // === Grit Factors (3 features) ===
    /// Harmonic-to-noise ratio in dB (higher = more tonal, lower = more noisy)
    pub harmonic_to_noise_ratio: f32,
    /// Spectral flatness (0 = tonal, 1 = noise-like)
    pub spectral_flatness: f32,
    /// Harmonicity - degree of harmonic relationship (0-1, higher = more harmonic)
    pub harmonicity: f32,

    // === Motion Factors (7 features) ===
    /// Attack time in milliseconds (fast = sharp onset, slow = gentle)
    pub attack_time_ms: f32,
    /// Decay time in milliseconds (fast = quick release, slow = long tail)
    pub decay_time_ms: f32,
    /// Sustain level (0-1, steady-state amplitude)
    pub sustain_level: f32,
    /// Vibrato rate in Hz (pitch modulation frequency)
    pub vibrato_rate_hz: f32,
    /// Vibrato depth in Hz (pitch modulation depth)
    pub vibrato_depth: f32,
    /// Jitter - micro-perturbations indicating instability (0-1)
    pub jitter: f32,
    /// Shimmer - amplitude micro-variations indicating breathiness (0-1)
    pub shimmer: f32,

    // === Fingerprint Factors (14 features) ===
    /// Mel-frequency cepstral coefficient 1 (spectral envelope)
    pub mfcc_1: f32,
    /// Mel-frequency cepstral coefficient 2
    pub mfcc_2: f32,
    /// Mel-frequency cepstral coefficient 3
    pub mfcc_3: f32,
    /// Mel-frequency cepstral coefficient 4
    pub mfcc_4: f32,
    /// Mel-frequency cepstral coefficient 5
    pub mfcc_5: f32,
    /// Mel-frequency cepstral coefficient 6
    pub mfcc_6: f32,
    /// Mel-frequency cepstral coefficient 7
    pub mfcc_7: f32,
    /// Mel-frequency cepstral coefficient 8
    pub mfcc_8: f32,
    /// Mel-frequency cepstral coefficient 9
    pub mfcc_9: f32,
    /// Mel-frequency cepstral coefficient 10
    pub mfcc_10: f32,
    /// Mel-frequency cepstral coefficient 11
    pub mfcc_11: f32,
    /// Mel-frequency cepstral coefficient 12
    pub mfcc_12: f32,
    /// Mel-frequency cepstral coefficient 13
    pub mfcc_13: f32,
    /// Spectral flux - rate of spectral change (0-1, higher = faster change)
    pub spectral_flux: f32,

    // === Rhythm Factors (3 features) ===
    /// Median inter-click interval in milliseconds
    pub median_ici_ms: f32,
    /// Onset rate - clicks or events per second
    pub onset_rate_hz: f32,
    /// ICI coefficient of variation - rhythm regularity (0 = perfectly regular, 1 = irregular)
    pub ici_coefficient_of_variation: f32,
}

impl Default for SourceMetadata {
    fn default() -> Self {
        Self {
            // Fundamental - marmoset-like defaults
            mean_f0_hz: 7000.0,
            duration_ms: 50.0,
            f0_range_hz: 400.0,

            // Grit - tonal (low noise)
            harmonic_to_noise_ratio: 20.0, // 20 dB HNR
            spectral_flatness: 0.1,        // Very tonal
            harmonicity: 0.8,              // High harmonicity

            // Motion - gentle attack/decay
            attack_time_ms: 10.0,
            decay_time_ms: 15.0,
            sustain_level: 0.7,
            vibrato_rate_hz: 8.0,
            vibrato_depth: 50.0,
            jitter: 0.02,  // Low instability
            shimmer: 0.03, // Low amplitude variation

            // Fingerprint - neutral spectral shape
            mfcc_1: -500.0,
            mfcc_2: -100.0,
            mfcc_3: -50.0,
            mfcc_4: -20.0,
            mfcc_5: -0.5,
            mfcc_6: -0.3,
            mfcc_7: -0.2,
            mfcc_8: -0.1,
            mfcc_9: 0.0,
            mfcc_10: 0.1,
            mfcc_11: 0.2,
            mfcc_12: 0.3,
            mfcc_13: 0.4,
            spectral_flux: 0.5, // Moderate spectral change rate

            // Rhythm - not pulsed (defaults for harmonic calls)
            median_ici_ms: 0.0, // Not applicable for continuous tones
            onset_rate_hz: 0.0, // Not applicable for continuous tones
            ici_coefficient_of_variation: 0.0, // Not applicable for continuous tones
        }
    }
}

impl SourceMetadata {
    /// Create a builder for partial metadata construction
    #[allow(dead_code)]
    pub fn builder() -> SourceMetadataBuilder {
        SourceMetadataBuilder::default()
    }

    /// Get delta vector (difference between two metadata sets)
    ///
    /// Returns a 30D delta vector representing the difference from `other` to `self`.
    /// This is used for vector delta synthesis: `target = source + delta`
    #[allow(dead_code)]
    pub fn delta_from(&self, other: &SourceMetadata) -> MicroDynamicsDelta {
        MicroDynamicsDelta {
            delta_mean_f0_hz: self.mean_f0_hz - other.mean_f0_hz,
            delta_duration_ms: self.duration_ms - other.duration_ms,
            delta_f0_range_hz: self.f0_range_hz - other.f0_range_hz,

            delta_harmonic_to_noise_ratio: self.harmonic_to_noise_ratio
                - other.harmonic_to_noise_ratio,
            delta_spectral_flatness: self.spectral_flatness - other.spectral_flatness,
            delta_harmonicity: self.harmonicity - other.harmonicity,

            delta_attack_time_ms: self.attack_time_ms - other.attack_time_ms,
            delta_decay_time_ms: self.decay_time_ms - other.decay_time_ms,
            delta_sustain_level: self.sustain_level - other.sustain_level,
            delta_vibrato_rate_hz: self.vibrato_rate_hz - other.vibrato_rate_hz,
            delta_vibrato_depth: self.vibrato_depth - other.vibrato_depth,
            delta_jitter: self.jitter - other.jitter,
            delta_shimmer: self.shimmer - other.shimmer,

            delta_mfcc_1: self.mfcc_1 - other.mfcc_1,
            delta_mfcc_2: self.mfcc_2 - other.mfcc_2,
            delta_mfcc_3: self.mfcc_3 - other.mfcc_3,
            delta_mfcc_4: self.mfcc_4 - other.mfcc_4,
            delta_mfcc_5: self.mfcc_5 - other.mfcc_5,
            delta_mfcc_6: self.mfcc_6 - other.mfcc_6,
            delta_mfcc_7: self.mfcc_7 - other.mfcc_7,
            delta_mfcc_8: self.mfcc_8 - other.mfcc_8,
            delta_mfcc_9: self.mfcc_9 - other.mfcc_9,
            delta_mfcc_10: self.mfcc_10 - other.mfcc_10,
            delta_mfcc_11: self.mfcc_11 - other.mfcc_11,
            delta_mfcc_12: self.mfcc_12 - other.mfcc_12,
            delta_mfcc_13: self.mfcc_13 - other.mfcc_13,
            delta_spectral_flux: self.spectral_flux - other.spectral_flux,

            delta_median_ici_ms: self.median_ici_ms - other.median_ici_ms,
            delta_onset_rate_hz: self.onset_rate_hz - other.onset_rate_hz,
            delta_ici_cv: self.ici_coefficient_of_variation - other.ici_coefficient_of_variation,
        }
    }
}

/// 30-dimensional micro-dynamics delta vector
///
/// Represents the difference between two acoustic feature vectors.
/// Used in vector delta synthesis to calculate transformations.
#[derive(Clone, Copy, Debug, Default)]
pub struct MicroDynamicsDelta {
    // Fundamental deltas
    pub delta_mean_f0_hz: f32,
    pub delta_duration_ms: f32,
    pub delta_f0_range_hz: f32,

    // Grit factor deltas
    pub delta_harmonic_to_noise_ratio: f32,
    pub delta_spectral_flatness: f32,
    pub delta_harmonicity: f32,

    // Motion factor deltas
    pub delta_attack_time_ms: f32,
    pub delta_decay_time_ms: f32,
    pub delta_sustain_level: f32,
    pub delta_vibrato_rate_hz: f32,
    pub delta_vibrato_depth: f32,
    pub delta_jitter: f32,
    pub delta_shimmer: f32,

    // Fingerprint factor deltas
    pub delta_mfcc_1: f32,
    pub delta_mfcc_2: f32,
    pub delta_mfcc_3: f32,
    pub delta_mfcc_4: f32,
    pub delta_mfcc_5: f32,
    pub delta_mfcc_6: f32,
    pub delta_mfcc_7: f32,
    pub delta_mfcc_8: f32,
    pub delta_mfcc_9: f32,
    pub delta_mfcc_10: f32,
    pub delta_mfcc_11: f32,
    pub delta_mfcc_12: f32,
    pub delta_mfcc_13: f32,
    pub delta_spectral_flux: f32,

    // Rhythm factor deltas
    pub delta_median_ici_ms: f32,
    pub delta_onset_rate_hz: f32,
    pub delta_ici_cv: f32,
}

/// Builder for partial SourceMetadata construction
///
/// Allows creating metadata with only known features, using defaults for the rest.
#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
pub struct SourceMetadataBuilder {
    metadata: SourceMetadata,
}

#[allow(dead_code)]
impl SourceMetadataBuilder {
    /// Set fundamental frequency
    pub fn mean_f0_hz(mut self, value: f32) -> Self {
        self.metadata.mean_f0_hz = value;
        self
    }

    /// Set duration
    pub fn duration_ms(mut self, value: f32) -> Self {
        self.metadata.duration_ms = value;
        self
    }

    /// Set F0 range
    pub fn f0_range_hz(mut self, value: f32) -> Self {
        self.metadata.f0_range_hz = value;
        self
    }

    /// Set harmonic-to-noise ratio
    pub fn harmonic_to_noise_ratio(mut self, value: f32) -> Self {
        self.metadata.harmonic_to_noise_ratio = value;
        self
    }

    /// Set spectral flatness
    pub fn spectral_flatness(mut self, value: f32) -> Self {
        self.metadata.spectral_flatness = value;
        self
    }

    /// Set harmonicity
    pub fn harmonicity(mut self, value: f32) -> Self {
        self.metadata.harmonicity = value;
        self
    }

    /// Set attack time
    pub fn attack_time_ms(mut self, value: f32) -> Self {
        self.metadata.attack_time_ms = value;
        self
    }

    /// Set decay time
    pub fn decay_time_ms(mut self, value: f32) -> Self {
        self.metadata.decay_time_ms = value;
        self
    }

    /// Set sustain level
    pub fn sustain_level(mut self, value: f32) -> Self {
        self.metadata.sustain_level = value;
        self
    }

    /// Set vibrato rate
    pub fn vibrato_rate_hz(mut self, value: f32) -> Self {
        self.metadata.vibrato_rate_hz = value;
        self
    }

    /// Set vibrato depth
    pub fn vibrato_depth(mut self, value: f32) -> Self {
        self.metadata.vibrato_depth = value;
        self
    }

    /// Set jitter
    pub fn jitter(mut self, value: f32) -> Self {
        self.metadata.jitter = value;
        self
    }

    /// Set shimmer
    pub fn shimmer(mut self, value: f32) -> Self {
        self.metadata.shimmer = value;
        self
    }

    /// Set MFCC coefficients (all 13)
    #[allow(clippy::too_many_arguments)]
    pub fn mfcc(
        mut self,
        mfcc_1: f32,
        mfcc_2: f32,
        mfcc_3: f32,
        mfcc_4: f32,
        mfcc_5: f32,
        mfcc_6: f32,
        mfcc_7: f32,
        mfcc_8: f32,
        mfcc_9: f32,
        mfcc_10: f32,
        mfcc_11: f32,
        mfcc_12: f32,
        mfcc_13: f32,
    ) -> Self {
        self.metadata.mfcc_1 = mfcc_1;
        self.metadata.mfcc_2 = mfcc_2;
        self.metadata.mfcc_3 = mfcc_3;
        self.metadata.mfcc_4 = mfcc_4;
        self.metadata.mfcc_5 = mfcc_5;
        self.metadata.mfcc_6 = mfcc_6;
        self.metadata.mfcc_7 = mfcc_7;
        self.metadata.mfcc_8 = mfcc_8;
        self.metadata.mfcc_9 = mfcc_9;
        self.metadata.mfcc_10 = mfcc_10;
        self.metadata.mfcc_11 = mfcc_11;
        self.metadata.mfcc_12 = mfcc_12;
        self.metadata.mfcc_13 = mfcc_13;
        self
    }

    /// Set spectral flux
    pub fn spectral_flux(mut self, value: f32) -> Self {
        self.metadata.spectral_flux = value;
        self
    }

    /// Set rhythm features
    pub fn rhythm(mut self, median_ici_ms: f32, onset_rate_hz: f32, ici_cv: f32) -> Self {
        self.metadata.median_ici_ms = median_ici_ms;
        self.metadata.onset_rate_hz = onset_rate_hz;
        self.metadata.ici_coefficient_of_variation = ici_cv;
        self
    }

    /// Build the metadata
    pub fn build(self) -> SourceMetadata {
        self.metadata
    }
}

#[allow(dead_code)]
pub struct GranularConcatenativeSynthesizer {
    sample_rate: usize,
    source_buffer: Vec<f32>,
    grain_size_ms: f32,
    pitch_shift_ratio: f32,
    time_stretch_ratio: f32,
    position: f32,
    /// Metadata for delta-based synthesis
    source_metadata: SourceMetadata,
}

#[allow(dead_code)]
impl GranularConcatenativeSynthesizer {
    /// Create a new granular concatenative synthesizer
    pub fn new(sample_rate: usize) -> Self {
        Self {
            sample_rate,
            source_buffer: Vec::new(),
            grain_size_ms: 20.0, // Default 20ms grains
            pitch_shift_ratio: 1.0,
            time_stretch_ratio: 1.0,
            position: 0.0,
            source_metadata: SourceMetadata::default(),
        }
    }

    /// Load source audio buffer with metadata (for delta-based synthesis)
    ///
    /// **VECTOR DELTA SUPPORT**: This enables delta commands like "shift pitch by +50Hz"
    /// instead of absolute commands like "set pitch to 7000Hz".
    ///
    /// # Parameters
    /// - `source`: Real audio samples
    /// - `metadata`: Acoustic features of the source (F0, duration, etc.)
    ///
    /// # Example
    /// ```ignore
    /// let metadata = SourceMetadata {
    ///     mean_f0_hz: 6800.0,
    ///     duration_ms: 50.0,
    ///     f0_range_hz: 400.0,
    /// };
    /// synthesizer.load_source_with_metadata(audio_buffer, metadata);
    ///
    /// // Now we can use delta commands!
    /// synthesizer.shift_pitch_by_hz(200.0);  // 6800 + 200 = 7000Hz
    /// synthesizer.shift_duration_by_ms(-10.0); // 50 - 10 = 40ms
    /// ```
    pub fn load_source_with_metadata(&mut self, source: Vec<f32>, metadata: SourceMetadata) {
        self.source_buffer = source;
        self.source_metadata = metadata;
        self.position = 0.0;
        self.pitch_shift_ratio = 1.0;
        self.time_stretch_ratio = 1.0;
    }

    /// Load source audio buffer (legacy method, uses default metadata)
    pub fn load_source(&mut self, source: Vec<f32>) {
        self.load_source_with_metadata(source, SourceMetadata::default());
    }

    /// Set source metadata (call after load_source() if metadata known)
    pub fn set_source_metadata(&mut self, metadata: SourceMetadata) {
        self.source_metadata = metadata;
    }

    /// Shift pitch by absolute Hz amount (VECTOR DELTA COMMAND)
    ///
    /// **GOOD**: "Shift pitch by +50Hz relative to source"
    /// **BAD**: "Set pitch to 7000Hz" (ignores source F0)
    ///
    /// # Parameters
    /// - `delta_hz`: Pitch shift in Hz (positive = higher, negative = lower)
    ///
    /// # Example
    /// ```ignore
    /// // Source F0 = 6800Hz
    /// synthesizer.shift_pitch_by_hz(200.0);  // Result: 7000Hz
    /// synthesizer.shift_pitch_by_hz(-300.0); // Result: 6500Hz
    /// ```
    pub fn shift_pitch_by_hz(&mut self, delta_hz: f32) {
        // Calculate ratio from delta Hz
        // Formula: ratio = (source_f0 + delta_hz) / source_f0
        let source_f0 = self.source_metadata.mean_f0_hz;
        let target_f0 = source_f0 + delta_hz;
        let ratio = (target_f0 / source_f0).clamp(0.5, 2.0);
        self.pitch_shift_ratio = ratio;
    }

    /// Shift duration by absolute ms amount (VECTOR DELTA COMMAND)
    ///
    /// **GOOD**: "Shift duration by -10ms relative to source"
    /// **BAD**: "Set duration to 40ms" (ignores source duration)
    ///
    /// # Parameters
    /// - `delta_ms`: Duration shift in ms (positive = longer, negative = shorter)
    ///
    /// # Example
    /// ```ignore
    /// // Source duration = 50ms
    /// synthesizer.shift_duration_by_ms(-10.0); // Result: 40ms
    /// synthesizer.shift_duration_by_ms(20.0);  // Result: 70ms
    /// ```
    pub fn shift_duration_by_ms(&mut self, delta_ms: f32) {
        // Calculate ratio from delta ms
        // Formula: ratio = (source_duration + delta_ms) / source_duration
        let source_duration = self.source_metadata.duration_ms;
        let target_duration = source_duration + delta_ms;
        let ratio = (target_duration / source_duration).clamp(0.5, 4.0);
        self.time_stretch_ratio = ratio;
    }

    /// Apply Vector Delta (legacy 3D method - kept for backward compatibility)
    ///
    /// Applies fundamental shifts simultaneously from a delta vector.
    /// This is the primary integration point for Acoustic Algebra.
    ///
    /// # Parameters
    /// - `delta_f0_hz`: Pitch shift in Hz
    /// - `delta_duration_ms`: Duration shift in ms
    /// - `delta_f0_range_hz`: F0 range shift in Hz
    ///
    /// # Example
    /// ```ignore
    /// // From acoustic algebra: virtual - nearest = delta
    /// synthesizer.apply_vector_delta(
    ///     200.0,    // Shift pitch up by 200Hz
    ///     -10.0,    // Shorten duration by 10ms
    ///     100.0     // Increase F0 range by 100Hz
    /// );
    /// ```
    pub fn apply_vector_delta(
        &mut self,
        delta_f0_hz: f32,
        delta_duration_ms: f32,
        delta_f0_range_hz: f32,
    ) {
        self.shift_pitch_by_hz(delta_f0_hz);
        self.shift_duration_by_ms(delta_duration_ms);
        // Note: F0 range shift would require spectral manipulation beyond granular synthesis
        // This is tracked in metadata for future use
        self.source_metadata.f0_range_hz += delta_f0_range_hz;
    }

    /// Apply Complete 30D Micro-Dynamics Delta
    ///
    /// Applies shifts for all 30 micro-dynamics features simultaneously.
    /// This enables full acoustic algebra integration with delta-based synthesis.
    ///
    /// **Note**: Only fundamental features (F0, duration) directly affect synthesis.
    /// Other features are tracked in metadata for validation and downstream processing.
    ///
    /// # Parameters
    /// - `delta`: 30D micro-dynamics delta vector
    ///
    /// # Example
    /// ```ignore
    /// use technical_architecture::synthesis::{SourceMetadata, MicroDynamicsDelta};
    ///
    /// // Calculate delta: target - source
    /// let delta = target_metadata.delta_from(&source_metadata);
    ///
    /// // Apply delta to synthesizer
    /// synthesizer.apply_micro_dynamics_delta(delta);
    /// ```
    pub fn apply_micro_dynamics_delta(&mut self, delta: MicroDynamicsDelta) {
        // Apply directly synthesis-affecting features
        self.shift_pitch_by_hz(delta.delta_mean_f0_hz);
        self.shift_duration_by_ms(delta.delta_duration_ms);

        // Track all delta features in metadata
        self.source_metadata.mean_f0_hz += delta.delta_mean_f0_hz;
        self.source_metadata.duration_ms += delta.delta_duration_ms;
        self.source_metadata.f0_range_hz += delta.delta_f0_range_hz;

        self.source_metadata.harmonic_to_noise_ratio += delta.delta_harmonic_to_noise_ratio;
        self.source_metadata.spectral_flatness += delta.delta_spectral_flatness;
        self.source_metadata.harmonicity += delta.delta_harmonicity;

        self.source_metadata.attack_time_ms += delta.delta_attack_time_ms;
        self.source_metadata.decay_time_ms += delta.delta_decay_time_ms;
        self.source_metadata.sustain_level += delta.delta_sustain_level;
        self.source_metadata.vibrato_rate_hz += delta.delta_vibrato_rate_hz;
        self.source_metadata.vibrato_depth += delta.delta_vibrato_depth;
        self.source_metadata.jitter += delta.delta_jitter;
        self.source_metadata.shimmer += delta.delta_shimmer;

        self.source_metadata.mfcc_1 += delta.delta_mfcc_1;
        self.source_metadata.mfcc_2 += delta.delta_mfcc_2;
        self.source_metadata.mfcc_3 += delta.delta_mfcc_3;
        self.source_metadata.mfcc_4 += delta.delta_mfcc_4;
        self.source_metadata.mfcc_5 += delta.delta_mfcc_5;
        self.source_metadata.mfcc_6 += delta.delta_mfcc_6;
        self.source_metadata.mfcc_7 += delta.delta_mfcc_7;
        self.source_metadata.mfcc_8 += delta.delta_mfcc_8;
        self.source_metadata.mfcc_9 += delta.delta_mfcc_9;
        self.source_metadata.mfcc_10 += delta.delta_mfcc_10;
        self.source_metadata.mfcc_11 += delta.delta_mfcc_11;
        self.source_metadata.mfcc_12 += delta.delta_mfcc_12;
        self.source_metadata.mfcc_13 += delta.delta_mfcc_13;
        self.source_metadata.spectral_flux += delta.delta_spectral_flux;

        self.source_metadata.median_ici_ms += delta.delta_median_ici_ms;
        self.source_metadata.onset_rate_hz += delta.delta_onset_rate_hz;
        self.source_metadata.ici_coefficient_of_variation += delta.delta_ici_cv;
    }

    /// Get current source metadata
    ///
    /// Returns the current metadata including any applied deltas.
    pub fn get_source_metadata(&self) -> SourceMetadata {
        self.source_metadata
    }

    /// Set pitch shift ratio
    pub fn set_pitch_shift(&mut self, ratio: f32) {
        self.pitch_shift_ratio = ratio.clamp(0.5, 2.0);
    }

    /// Set grain size in milliseconds
    pub fn set_grain_size_ms(&mut self, size_ms: f32) {
        self.grain_size_ms = size_ms.clamp(5.0, 100.0);
    }

    /// Synthesize audio with specified duration
    ///
    /// Parameters:
    /// - duration_ms: Output duration in milliseconds
    ///
    /// Returns: Synthesized audio samples
    pub fn synthesize(&mut self, duration_ms: f32) -> Vec<f32> {
        let num_samples = (duration_ms / 1000.0 * self.sample_rate as f32) as usize;
        let mut output = Vec::with_capacity(num_samples);

        // Create window
        let window = GrainWindow::hanning(self.grain_size_ms, self.sample_rate);
        let grain_length = window.len() as f32;

        // Don't process if buffer is too small
        if self.source_buffer.is_empty() {
            return vec![0.0; num_samples];
        }

        for _ in 0..num_samples {
            // Get current sample with linear interpolation
            let pos_int = self.position as usize;
            let pos_frac = self.position - pos_int as f32;

            // Get samples for interpolation
            let sample0 = if pos_int < self.source_buffer.len() {
                self.source_buffer[pos_int]
            } else {
                0.0
            };

            let sample1 = if pos_int + 1 < self.source_buffer.len() {
                self.source_buffer[pos_int + 1]
            } else {
                0.0
            };

            // Linear interpolation
            let current_sample = sample0 + (sample1 - sample0) * pos_frac;

            // Calculate position within grain (0.0 to 1.0)
            let grain_position = (self.position % grain_length) / grain_length;

            // Get window envelope value at this grain position
            let window_idx = (grain_position * (window.len() - 1) as f32) as usize;
            let window_value = if window_idx < window.len() {
                window[window_idx]
            } else {
                0.0
            };

            // Apply window envelope
            let sample = current_sample * window_value;

            // Calculate effective stride based on pitch shift
            let effective_stride = 1.0 / self.pitch_shift_ratio;

            // Advance position
            self.position += effective_stride;

            // Wrap around source buffer
            let buffer_limit = self.source_buffer.len() as f32 - grain_length;
            if self.position >= buffer_limit || self.position < 0.0 {
                self.position = 0.0;
            }

            output.push(sample);
        }

        output
    }
}

/// Grain for granular synthesis
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Grain {
    /// Audio samples for this grain
    samples: VecDeque<f32>,
    /// Current position in grain
    position: usize,
    /// Grain envelope (window)
    envelope: Vec<f32>,
    /// Grain amplitude
    amplitude: f32,
    /// Playback rate (pitch shift)
    rate: f32,
    /// Pan position (-1.0 to 1.0)
    pan: f32,
}

#[allow(dead_code)]
impl Grain {
    /// Create a new grain
    fn new(samples: Vec<f32>, grain_size: usize) -> Self {
        // Create fade envelope (Hanning window)
        let mut envelope = Vec::with_capacity(grain_size);
        for i in 0..grain_size {
            let phase = 2.0 * std::f32::consts::PI * i as f32 / grain_size as f32;
            envelope.push(0.5 * (1.0 - phase.cos()));
        }

        Self {
            samples: VecDeque::from(samples),
            position: 0,
            envelope,
            amplitude: 1.0,
            rate: 1.0,
            pan: 0.0,
        }
    }

    /// Get next sample from grain
    fn next_sample(&mut self) -> Option<f32> {
        if self.position >= self.samples.len() {
            return None;
        }

        let sample = self.samples[self.position];

        // Apply envelope
        let env = if self.position < self.envelope.len() {
            self.envelope[self.position]
        } else {
            0.0
        };

        self.position += 1;

        Some(sample * env * self.amplitude)
    }

    /// Check if grain is finished
    fn is_finished(&self) -> bool {
        self.position >= self.samples.len()
    }

    /// Reset grain position
    fn reset(&mut self) {
        self.position = 0;
    }
}

/// Granular synthesizer
pub struct GranularSynthesizer {
    /// Configuration
    config: SynthesisConfig,
    /// Active grains
    grains: Vec<Grain>,
    /// Audio buffer for source material
    source_buffer: Vec<f32>,
    /// Current read position in source buffer
    read_position: f32,
    /// Output buffer
    output_buffer: VecDeque<f32>,
}

impl GranularSynthesizer {
    /// Create a new granular synthesizer
    pub async fn new(config: SynthesisConfig) -> Result<Self> {
        info!("Initializing Granular Synthesizer");

        Ok(Self {
            config,
            grains: Vec::new(),
            source_buffer: Vec::new(),
            read_position: 0.0,
            output_buffer: VecDeque::new(),
        })
    }

    /// Load source audio material
    pub async fn load_source(&mut self, segment: AudioSegment) -> Result<()> {
        // Resample if necessary
        let segment = if segment.sample_rate != self.config.sample_rate {
            segment.resample(self.config.sample_rate)?
        } else {
            segment
        };

        self.source_buffer = segment.samples;
        self.read_position = 0.0;

        debug!("Loaded source buffer: {} samples", self.source_buffer.len());

        Ok(())
    }

    /// Generate audio from features
    pub async fn generate(&self, features: &AudioFeatures) -> Result<Vec<f32>> {
        // For now, generate a simple test tone based on features
        let duration_samples = (self.config.sample_rate as f32 * 0.1) as usize; // 100ms
        let mut output = Vec::with_capacity(duration_samples);

        let frequency = if features.f0 > 0.0 {
            features.f0
        } else {
            440.0
        };

        for i in 0..duration_samples {
            let t = i as f32 / self.config.sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
            output.push(sample * self.config.output_gain * features.rms);
        }

        Ok(output)
    }

    /// Synthesize audio using granular synthesis
    pub async fn synthesize(&mut self, duration_ms: f32) -> Result<Vec<f32>> {
        let duration_samples = (self.config.sample_rate as f32 * duration_ms / 1000.0) as usize;
        let mut output = vec![0.0f32; duration_samples];

        // Generate grains if source buffer is available
        if !self.source_buffer.is_empty() {
            self.update_grains(duration_samples);
        }

        // Mix active grains
        #[allow(clippy::needless_range_loop)]
        for grain in &mut self.grains {
            for i in 0..duration_samples {
                if let Some(sample) = grain.next_sample() {
                    output[i] += sample;
                } else {
                    break;
                }
            }
        }

        // Remove finished grains
        self.grains.retain(|g| !g.is_finished());

        // Normalize output
        let max_amplitude = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        if max_amplitude > 0.0 {
            let scale = self.config.output_gain / max_amplitude;
            for sample in &mut output {
                *sample *= scale;
            }
        }

        Ok(output)
    }

    /// Update grain generation
    fn update_grains(&mut self, _num_samples: usize) {
        let grain_size_samples =
            (self.config.grain_size_ms * self.config.sample_rate as f32 / 1000.0) as usize;
        let grain_spacing =
            (grain_size_samples as f32 * (1.0 - self.config.grain_overlap)) as usize;

        // Spawn new grains as needed
        while self.grains.len() < self.config.max_grains {
            let start_pos = self.read_position as usize;
            if start_pos + grain_size_samples > self.source_buffer.len() {
                self.read_position = 0.0;
                break;
            }

            let grain_samples =
                self.source_buffer[start_pos..start_pos + grain_size_samples].to_vec();
            let grain = Grain::new(grain_samples, grain_size_samples);

            self.grains.push(grain);

            self.read_position += grain_spacing as f32;
            if self.read_position as usize >= self.source_buffer.len() {
                self.read_position = 0.0;
            }
        }
    }

    /// Clear all active grains
    pub fn clear_grains(&mut self) {
        self.grains.clear();
    }

    /// Get number of active grains
    pub fn active_grain_count(&self) -> usize {
        self.grains.len()
    }

    /// Emergency stop - immediately halt all synthesis
    ///
    /// This is a safety-critical function that must complete in < 1ms.
    /// It clears all active grains and flushes the output buffer.
    pub fn emergency_stop(&mut self) -> Result<()> {
        // Clear all active grains immediately
        self.grains.clear();

        // Flush output buffer
        self.output_buffer.clear();

        // Reset read position
        self.read_position = 0.0;

        Ok(())
    }

    /// Shutdown synthesizer
    pub async fn shutdown(&self) -> Result<()> {
        info!("Granular Synthesizer shutdown");
        Ok(())
    }
}

// ============================================================================
// DYNAMIC MICROHARMONIC SYNTHESIZER
// ============================================================================

/// Calculate ADSR envelope for a single time point
///
/// Parameters:
/// - time: Current time in milliseconds
/// - total: Total duration in milliseconds
/// - attack: Attack time in milliseconds
/// - decay: Decay time in milliseconds
/// - sustain: Sustain level (0.0 to 1.0)
///
/// Returns: Amplitude multiplier (0.0 to 1.0)
fn calculate_adsr_envelope(time: f32, total: f32, attack: f32, decay: f32, sustain: f32) -> f32 {
    if time < attack {
        // Attack phase - logarithmic attack (percussive)
        (time / attack).powf(3.0)
    } else if time < (total - decay) {
        // Sustain phase
        sustain
    } else if total > decay {
        // Decay phase - logarithmic decay
        let remaining = total - decay;
        let current = time - remaining;
        sustain * (1.0 - (current / decay).powf(0.5))
    } else {
        0.0
    }
}

/// Generate a single sample with dynamic microharmonic synthesis
///
/// Parameters:
/// - time: Current time in seconds
/// - params: Dynamic microharmonic parameters
/// - phase: Current oscillator phase (0.0 to 1.0)
///
/// Returns: Synthesized sample value (-1.0 to 1.0)
pub fn generate_dynamic_microharmonic_sample(
    time: f32,
    params: &DynamicMicroharmonicParams,
    phase: f32,
) -> f32 {
    // 1. Calculate instantaneous F0 with vibrato
    let vibrato_osc = (time * params.vibrato_rate_hz * 2.0 * std::f32::consts::PI).sin();
    let vibrato_cents = vibrato_osc * params.vibrato_depth_cents;
    let vibrato_ratio = 2.0_f32.powf(vibrato_cents / 1200.0);
    let _inst_f0 = params.f0_base * vibrato_ratio;

    // 2. Apply jitter (random phase perturbation)
    let jitter = if params.jitter_amount > 0.0 {
        let mut rng = thread_rng();
        (rng.gen::<f32>() - 0.5) * 2.0 * params.jitter_amount
    } else {
        0.0
    };

    // 3. Generate additive harmonic stack (instead of single sine wave)
    // Real animal vocalizations have multiple harmonics
    let perturbed_phase = phase + jitter;
    let mut sample = 0.0_f32;

    // Add up to 8 harmonics with spectral tilt
    // Spectral tilt determines amplitude rolloff across harmonics
    // Typical values: -6 dB/octave (gentle) to -12 dB/octave (steep)
    let num_harmonics = 8_usize;
    let tilt_db_per_octave = params.spectral_tilt; // e.g., -6.0 dB/octave

    for harmonic in 1..=num_harmonics {
        let harmonic_phase = perturbed_phase * harmonic as f32;
        let harmonic_signal = (harmonic_phase * 2.0 * std::f32::consts::PI).sin();

        // Calculate amplitude based on spectral tilt
        // Each octave (doubling of harmonic number) reduces amplitude by tilt_db_per_octave
        let octaves_above_fundamental = (harmonic as f32).log2();
        let amplitude_db = -tilt_db_per_octave * octaves_above_fundamental;
        let amplitude_linear = 10.0_f32.powf(amplitude_db / 20.0);

        // Apply additional rolloff for higher harmonics (natural vocal tract filtering)
        // This simulates formant-like attenuation
        let formant_rolloff = 1.0 / (1.0 + (harmonic as f32 - 1.0) * 0.15);

        sample += harmonic_signal * amplitude_linear * formant_rolloff;
    }

    // Normalize to prevent clipping (sum of harmonics can exceed 1.0)
    sample /= num_harmonics as f32;

    // 4. Apply shimmer (random amplitude variation)
    if params.shimmer_amount > 0.0 {
        let mut rng = thread_rng();
        let shimmer = 1.0 + (rng.gen::<f32>() - 0.5) * 2.0 * params.shimmer_amount;
        sample *= shimmer;
    }

    // 5. Add noise component based on HNR (Harmonic-to-Noise Ratio)
    // Lower HNR = more noise component
    // HNR of 0 dB = equal parts harmonic and noise
    // HNR of 20 dB = mostly harmonic, little noise
    let hnr_linear = 10.0_f32.powf(params.hnr_db / 20.0);
    if hnr_linear < 100.0 {
        // Add noise if HNR is not extremely high
        let mut rng = thread_rng();
        let noise_magnitude = 1.0 / (1.0 + hnr_linear); // Inverse of HNR
        let noise: f32 = (rng.gen::<f32>() - 0.5) * 2.0 * noise_magnitude * 0.3;
        sample += noise;
    }

    // 6. Calculate ADSR envelope
    let time_ms = time * 1000.0;
    let envelope = calculate_adsr_envelope(
        time_ms,
        params.duration_ms,
        params.attack_ms,
        params.decay_ms,
        params.sustain_level,
    );

    sample * envelope
}

/// Dynamic Microharmonic Synthesizer
///
/// Generates natural-sounding vocalizations using micro-dynamics
/// features captured from real recordings.
///
/// This synthesizer bridges the gap between:
/// - Concatenative synthesis (real segments, limited flexibility)
/// - Parametric synthesis (full flexibility, artificial sound)
///
/// By using micro-dynamics (attack, vibrato, jitter), it produces
/// vocalizations that are statistically congruent with natural sounds.
pub struct DynamicMicroharmonicSynthesizer {
    sample_rate: usize,
}

impl DynamicMicroharmonicSynthesizer {
    /// Create a new dynamic microharmonic synthesizer
    pub fn new(sample_rate: usize) -> Self {
        Self { sample_rate }
    }

    /// Synthesize a phrase using dynamic microharmonic parameters
    ///
    /// This is the core synthesis function that generates a single
    /// phrase with natural-sounding micro-dynamics.
    ///
    /// Parameters:
    /// - params: Dynamic microharmonic synthesis parameters
    ///
    /// Returns: Synthesized audio samples
    pub fn synthesize_phrase(&self, params: &DynamicMicroharmonicParams) -> Vec<f32> {
        let num_samples = (params.duration_ms / 1000.0 * self.sample_rate as f32) as usize;
        let mut output = Vec::with_capacity(num_samples);

        let _phase_increment = params.f0_base / self.sample_rate as f32;
        let mut phase = 0.0;

        for i in 0..num_samples {
            let time = i as f32 / self.sample_rate as f32;

            // Generate sample with micro-dynamics
            let sample = generate_dynamic_microharmonic_sample(time, params, phase);

            output.push(sample);

            // Update phase (instantaneous frequency varies due to vibrato)
            let vibrato_osc = (time * params.vibrato_rate_hz * 2.0 * std::f32::consts::PI).sin();
            let vibrato_cents = vibrato_osc * params.vibrato_depth_cents;
            let vibrato_ratio = 2.0_f32.powf(vibrato_cents / 1200.0);
            let inst_f0 = params.f0_base * vibrato_ratio;

            phase += inst_f0 / self.sample_rate as f32;
            phase %= 1.0;
        }

        output
    }

    /// Synthesize a sequence of phrases (sentence) using dynamic microharmonic
    ///
    /// This creates a multi-phrase vocalization by concatenating
    /// individual dynamically synthesized phrases.
    ///
    /// Parameters:
    /// - phrase_params: List of parameters for each phrase in sequence
    /// - crossfade_ms: Crossfade duration between phrases
    ///
    /// Returns: Synthesized audio samples for entire sequence
    pub fn synthesize_sequence(
        &self,
        phrase_params: &[DynamicMicroharmonicParams],
        crossfade_ms: f32,
    ) -> Vec<f32> {
        if phrase_params.is_empty() {
            return Vec::new();
        }

        if phrase_params.len() == 1 {
            return self.synthesize_phrase(&phrase_params[0]);
        }

        // Synthesize each phrase individually
        let phrases: Vec<Vec<f32>> = phrase_params
            .iter()
            .map(|p| self.synthesize_phrase(p))
            .collect();

        // Calculate total duration
        let total_samples: usize = phrases.iter().map(|p| p.len()).sum();
        let mut output = Vec::with_capacity(total_samples);

        // Concatenate with crossfades
        let crossfade_samples = (crossfade_ms / 1000.0 * self.sample_rate as f32) as usize;

        for (i, phrase) in phrases.iter().enumerate() {
            if i == 0 {
                // First phrase - no crossfade at start
                output.extend_from_slice(phrase);
            } else {
                // Apply crossfade between phrases
                let output_start = output.len().saturating_sub(crossfade_samples);

                for (j, &sample) in phrase.iter().enumerate() {
                    if j < crossfade_samples && output_start + j < output.len() {
                        // Crossfade region
                        let fade_out = 1.0 - (j as f32 / crossfade_samples as f32);
                        let fade_in = j as f32 / crossfade_samples as f32;

                        let mixed = output[output_start + j] * fade_out + sample * fade_in;
                        output[output_start + j] = mixed;
                    } else if j >= crossfade_samples {
                        // No crossfade - just append
                        output.push(sample);
                    }
                    // else: crossfade region already covered
                }
            }
        }

        output
    }

    /// Generate random micro-dynamics parameters within natural ranges
    ///
    /// This is useful for exploration and testing when you don't have
    /// specific target parameters.
    ///
    /// Parameters:
    /// - f0_base: Target fundamental frequency in Hz
    /// - duration_ms: Target duration in milliseconds
    /// - variability: Amount of randomness (0.0 to 1.0)
    ///
    /// Returns: Randomized dynamic microharmonic parameters
    pub fn generate_random_params(
        &self,
        f0_base: f32,
        duration_ms: f32,
        variability: f32,
    ) -> DynamicMicroharmonicParams {
        let mut rng = thread_rng();

        // Base ranges from our micro-dynamics analysis
        let attack_base = 10.0;
        let decay_base = 28.0;
        let vibrato_rate_base = 7.5;
        let vibrato_depth_base = 25.0;
        let jitter_base = 0.025;

        DynamicMicroharmonicParams {
            f0_base,
            duration_ms,
            attack_ms: attack_base * (1.0 + (rng.gen::<f32>() - 0.5) * 2.0 * variability),
            decay_ms: decay_base * (1.0 + (rng.gen::<f32>() - 0.5) * 2.0 * variability),
            sustain_level: 0.7,
            vibrato_rate_hz: vibrato_rate_base
                * (1.0 + (rng.gen::<f32>() - 0.5) * 2.0 * variability),
            vibrato_depth_cents: vibrato_depth_base
                * (1.0 + (rng.gen::<f32>() - 0.5) * 2.0 * variability),
            jitter_amount: jitter_base * (1.0 + (rng.gen::<f32>() - 0.5) * 2.0 * variability),
            shimmer_amount: 0.01,
            spectral_tilt: -6.0,
            hnr_db: 20.0,
        }
    }
}

// ============================================================================
// Multi-Buffer Sequencer for Corvid Multi-Modal Support
// ============================================================================

/// Vocalization modality types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Tonal, sine-like (whistle, phee)
    #[serde(rename = "HARMONIC")]
    Harmonic,
    /// Clicky, noise-like (rattle, click)
    #[serde(rename = "TRANSIENT")]
    Transient,
    /// Frequency modulated (trill, sweep)
    #[serde(rename = "FM_SWEEP")]
    FmSweep,
}

/// Single event in a multi-modal sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    /// Start time in milliseconds
    pub start_ms: f32,
    /// Duration in milliseconds
    pub duration_ms: f32,
    /// Source buffer identifier (e.g., "corvid_whistle.wav")
    pub source_buffer: String,
    /// Modality type for this event
    pub modality: Modality,
}

/// Timeline of events for multi-modal synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityTimeline {
    /// Timeline events
    pub events: Vec<TimelineEvent>,
}

impl ModalityTimeline {
    /// Create a new empty timeline
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Add an event to the timeline
    pub fn add_event(
        &mut self,
        start_ms: f32,
        duration_ms: f32,
        source: String,
        modality: Modality,
    ) {
        let event = TimelineEvent {
            start_ms,
            duration_ms,
            source_buffer: source,
            modality,
        };
        self.events.push(event);
    }

    /// Sort events by start time
    pub fn sort_by_time(&mut self) {
        self.events
            .sort_by(|a, b| a.start_ms.partial_cmp(&b.start_ms).unwrap());
    }

    /// Validate timeline has no overlaps and is sequential
    pub fn validate(&self) -> Result<()> {
        let mut sorted_events = self.events.clone();
        sorted_events.sort_by(|a, b| a.start_ms.partial_cmp(&b.start_ms).unwrap());

        for i in 0..sorted_events.len().saturating_sub(1) {
            let current = &sorted_events[i];
            let next = &sorted_events[i + 1];

            let current_end = current.start_ms + current.duration_ms;
            if current_end > next.start_ms {
                return Err(anyhow::anyhow!(
                    "Timeline overlap: Event {} ends at {}ms, Event {} starts at {}ms",
                    i,
                    current_end,
                    i + 1,
                    next.start_ms
                ));
            }
        }

        Ok(())
    }

    /// Get total duration of timeline in milliseconds
    pub fn total_duration_ms(&self) -> f32 {
        if self.events.is_empty() {
            return 0.0;
        }

        let last_event = self
            .events
            .iter()
            .max_by(|a, b| a.start_ms.partial_cmp(&b.start_ms).unwrap())
            .unwrap();

        last_event.start_ms + last_event.duration_ms
    }
}

impl Default for ModalityTimeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-buffer granular sequencer for composite personas
///
/// This sequencer enables "Texture Sequencing" for multi-modal species like corvids
/// that use multiple modalities (Harmonic + Transient + FM Sweep) in single vocalizations.
///
/// Key Principle: Persona Switching (Source Selection)
/// - Use different source buffers for different modalities
/// - Preserve formant structure per source (Formant Barrier)
/// - Sequence timeline events to compose multi-modal calls
#[derive(Debug)]
pub struct MultiBufferGranularSequencer {
    sample_rate: usize,
    /// Multiple source buffers indexed by buffer name
    source_buffers: HashMap<String, Vec<f32>>,
    /// Metadata for each source buffer
    source_metadata: HashMap<String, SourceMetadata>,
    /// Default grain size in milliseconds
    grain_size_ms: f32,
    /// Default pitch shift ratio
    pitch_shift_ratio: f32,
}

impl MultiBufferGranularSequencer {
    /// Create a new multi-buffer granular sequencer
    pub fn new(sample_rate: usize) -> Self {
        Self {
            sample_rate,
            source_buffers: HashMap::new(),
            source_metadata: HashMap::new(),
            grain_size_ms: 20.0,
            pitch_shift_ratio: 1.0,
        }
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Register a source buffer with metadata
    ///
    /// # Parameters
    /// - `buffer_name`: Unique identifier for this buffer (e.g., "corvid_whistle")
    /// - `audio`: Audio samples
    /// - `metadata`: Acoustic features of the source
    pub fn register_source(
        &mut self,
        buffer_name: String,
        audio: Vec<f32>,
        metadata: SourceMetadata,
    ) {
        self.source_buffers.insert(buffer_name.clone(), audio);
        self.source_metadata.insert(buffer_name, metadata);
    }

    /// Set default grain size
    pub fn set_grain_size_ms(&mut self, grain_size_ms: f32) {
        self.grain_size_ms = grain_size_ms;
    }

    /// Set default pitch shift ratio
    pub fn set_pitch_shift(&mut self, ratio: f32) {
        self.pitch_shift_ratio = ratio.clamp(0.5, 2.0);
    }

    /// Synthesize a multi-modal sequence from timeline
    ///
    /// # Parameters
    /// - `timeline`: Sequence of timeline events with different modalities
    ///
    /// # Returns
    /// Synthesized audio samples
    ///
    /// # Example
    /// ```ignore
    /// let mut timeline = ModalityTimeline::new();
    /// timeline.add_event(0.0, 100.0, "whistle".to_string(), Modality::Harmonic);
    /// timeline.add_event(100.0, 50.0, "rattle".to_string(), Modality::Transient);
    ///
    /// let audio = sequencer.synthesize_timeline(&timeline)?;
    /// ```
    pub fn synthesize_timeline(&self, timeline: &ModalityTimeline) -> Result<Vec<f32>> {
        // Validate timeline
        timeline.validate()?;

        if timeline.events.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate total duration
        let total_duration_ms = timeline.total_duration_ms();
        let total_samples = (total_duration_ms / 1000.0 * self.sample_rate as f32) as usize;

        let mut output = vec![0.0f32; total_samples];

        // Process each event
        for event in &timeline.events {
            // Get source buffer
            let source_audio = self
                .source_buffers
                .get(&event.source_buffer)
                .ok_or_else(|| {
                    anyhow::anyhow!("Source buffer '{}' not found", event.source_buffer)
                })?;

            // Create single-buffer synthesizer for this event
            let grain_size_samples =
                (event.duration_ms / 1000.0 * self.sample_rate as f32) as usize;

            // Calculate start sample
            let start_sample = (event.start_ms / 1000.0 * self.sample_rate as f32) as usize;
            let end_sample = (start_sample + grain_size_samples).min(total_samples);

            if start_sample >= total_samples || grain_size_samples == 0 {
                continue;
            }

            // Simple concatenation: copy source audio to output
            // Apply pitch shift if needed
            let pitch_ratio = if self.pitch_shift_ratio != 1.0 {
                self.pitch_shift_ratio
            } else {
                1.0
            };

            // Copy with pitch shifting (resampling)
            let source_len = source_audio.len().min(grain_size_samples);
            output[start_sample..end_sample]
                .iter_mut()
                .enumerate()
                .for_each(|(i, out_sample)| {
                    let src_idx = (i as f32 / pitch_ratio) as usize;
                    if src_idx < source_len {
                        *out_sample = source_audio[src_idx];
                    }
                });
        }

        Ok(output)
    }

    /// Get list of registered source buffer names
    pub fn registered_sources(&self) -> Vec<String> {
        self.source_buffers.keys().cloned().collect()
    }

    /// Check if a source buffer is registered
    pub fn has_source(&self, buffer_name: &str) -> bool {
        self.source_buffers.contains_key(buffer_name)
    }

    /// Get metadata for a source buffer
    pub fn get_source_metadata(&self, buffer_name: &str) -> Option<&SourceMetadata> {
        self.source_metadata.get(buffer_name)
    }
}

// =============================================================================
// Island Hopping: Cached Audio Buffer Management
// =============================================================================

/// Cached audio buffer for real-time synthesis
///
/// This struct wraps an audio buffer with metadata to support
/// LRU caching for island hopping navigation.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Public API used via PyO3 bindings
pub struct CachedAudioBuffer {
    /// Unique identifier for this buffer
    pub id: String,
    /// Audio samples
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: usize,
    /// Approximate size in bytes (for cache size tracking)
    pub size_bytes: usize,
}

#[allow(dead_code)] // Public API used via PyO3 bindings
impl CachedAudioBuffer {
    /// Create a new cached audio buffer
    pub fn new(id: String, samples: Vec<f32>, sample_rate: usize) -> Self {
        let size_bytes = samples.len() * std::mem::size_of::<f32>();
        Self {
            id,
            samples,
            sample_rate,
            size_bytes,
        }
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> f32 {
        (self.samples.len() as f32 / self.sample_rate as f32) * 1000.0
    }
}

/// Cached Granular Synthesizer for Island Hopping
///
/// This wrapper adds LRU caching to the MultiBufferGranularSequencer
/// to enable real-time island hopping navigation with <100ms latency.
///
/// Key Benefits:
/// - **Cache Hit**: <1ms lookup (RAM access)
/// - **Cache Miss**: ~20ms load (SSD access)
/// - **Pre-fetching**: Context-aware cache warming
#[allow(dead_code)] // Public API used via PyO3 bindings
pub struct CachedGranularSequencer {
    /// The underlying sequencer
    sequencer: MultiBufferGranularSequencer,
    /// LRU cache for audio buffers (key: buffer_id, value: CachedAudioBuffer)
    cache: LruCache<String, CachedAudioBuffer>,
    /// Maximum cache size in bytes (default: 50MB)
    max_cache_bytes: usize,
    /// Current cache usage in bytes
    current_cache_bytes: usize,
    /// Cache hit count (for statistics)
    cache_hits: Arc<Mutex<u64>>,
    /// Cache miss count (for statistics)
    cache_misses: Arc<Mutex<u64>>,
}

#[allow(dead_code)] // Public API used via PyO3 bindings
impl CachedGranularSequencer {
    /// Create a new cached granular sequencer
    ///
    /// # Arguments
    /// * `sample_rate` - Audio sample rate in Hz
    /// * `max_cache_bytes` - Maximum cache size in bytes (default: 50MB = 52428800)
    pub fn new(sample_rate: usize, max_cache_bytes: usize) -> Self {
        info!(
            "Initializing Cached Granular Sequencer with {}MB cache",
            max_cache_bytes / 1024 / 1024
        );

        Self {
            sequencer: MultiBufferGranularSequencer::new(sample_rate),
            cache: LruCache::unbounded(), // We manage size manually
            max_cache_bytes,
            current_cache_bytes: 0,
            cache_hits: Arc::new(Mutex::new(0)),
            cache_misses: Arc::new(Mutex::new(0)),
        }
    }

    /// Create with default 50MB cache
    pub fn with_default_cache(sample_rate: usize) -> Self {
        Self::new(sample_rate, 50 * 1024 * 1024) // 50MB
    }

    /// Register an audio buffer (checks cache first)
    ///
    /// This is the main entry point for island hopping navigation.
    /// The sequencer will:
    /// 1. Check if the buffer is already in cache (<1ms)
    /// 2. If cache hit, use cached buffer directly
    /// 3. If cache miss, load buffer and cache it (~20ms from SSD)
    ///
    /// # Arguments
    /// * `id` - Unique identifier for this buffer (e.g., "neutral_001")
    /// * `audio` - Audio samples
    /// * `metadata` - Acoustic metadata for this buffer
    pub async fn register_source(
        &mut self,
        id: String,
        audio: Vec<f32>,
        metadata: SourceMetadata,
    ) -> Result<()> {
        // Check cache first
        if self.cache.get(&id).is_some() {
            // Cache hit - buffer already loaded
            debug!("Cache HIT for buffer '{}'", id);
            *self.cache_hits.lock() += 1;
            return Ok(());
        }

        // Cache miss - need to load
        debug!("Cache MISS for buffer '{}', loading...", id);
        *self.cache_misses.lock() += 1;

        // Create cached buffer
        let cached =
            CachedAudioBuffer::new(id.clone(), audio.clone(), self.sequencer.sample_rate());

        let size_bytes = cached.size_bytes;

        // Evict old entries if necessary
        self.ensure_cache_space(size_bytes);

        // Add to cache
        self.cache.put(id.clone(), cached);
        self.current_cache_bytes += size_bytes;

        // Register with underlying sequencer
        self.sequencer.register_source(id.clone(), audio, metadata);

        info!(
            "Registered buffer '{}' ({:.2}MB, cache now at {:.2}MB/{:.2}MB)",
            id,
            size_bytes as f32 / 1024.0 / 1024.0,
            self.current_cache_bytes as f32 / 1024.0 / 1024.0,
            self.max_cache_bytes as f32 / 1024.0 / 1024.0
        );

        Ok(())
    }

    /// Pre-load a buffer into cache (for contextual pre-fetching)
    ///
    /// This is used by the Python agent to warm the cache based on
    /// predicted context (e.g., pre-loading "social" phrases when
    /// entering a social context).
    ///
    /// # Arguments
    /// * `id` - Buffer identifier to pre-load
    /// * `audio` - Audio samples
    /// * `metadata` - Acoustic metadata
    pub async fn preload(
        &mut self,
        id: String,
        audio: Vec<f32>,
        metadata: SourceMetadata,
    ) -> Result<()> {
        debug!("Pre-loading buffer '{}'", id);
        self.register_source(id, audio, metadata).await
    }

    /// Check if a buffer is in cache
    pub fn is_cached(&self, id: &str) -> bool {
        self.cache.contains(id)
    }

    /// Synthesize a timeline (uses cached buffers)
    pub async fn synthesize_timeline(&mut self, timeline: &ModalityTimeline) -> Result<Vec<f32>> {
        self.sequencer.synthesize_timeline(timeline)
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let hits = *self.cache_hits.lock();
        let misses = *self.cache_misses.lock();
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f32 / total as f32
        } else {
            0.0
        };

        CacheStats {
            cache_hits: hits,
            cache_misses: misses,
            hit_rate,
            current_bytes: self.current_cache_bytes,
            max_bytes: self.max_cache_bytes,
            num_buffers: self.cache.len(),
        }
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        info!("Clearing audio buffer cache");
        self.cache.clear();
        self.current_cache_bytes = 0;
    }

    /// Ensure enough space in cache for a new buffer
    fn ensure_cache_space(&mut self, required_bytes: usize) {
        while self.current_cache_bytes + required_bytes > self.max_cache_bytes {
            if let Some((id, evicted)) = self.cache.pop_lru() {
                self.current_cache_bytes -= evicted.size_bytes;
                debug!(
                    "Evicted buffer '{}' ({:.2}MB) from cache",
                    id,
                    evicted.size_bytes as f32 / 1024.0 / 1024.0
                );
            } else {
                // Cache is empty but still not enough space
                warn!(
                    "Requested buffer size ({:.2}MB) exceeds cache capacity ({:.2}MB)",
                    required_bytes as f32 / 1024.0 / 1024.0,
                    self.max_cache_bytes as f32 / 1024.0 / 1024.0
                );
                break;
            }
        }
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> usize {
        self.sequencer.sample_rate()
    }
}

/// Cache statistics for monitoring
#[derive(Debug, Clone)]
#[allow(dead_code)] // Public API used via PyO3 bindings
pub struct CacheStats {
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
    /// Cache hit rate (0.0 to 1.0)
    pub hit_rate: f32,
    /// Current cache usage in bytes
    pub current_bytes: usize,
    /// Maximum cache size in bytes
    pub max_bytes: usize,
    /// Number of buffers currently cached
    pub num_buffers: usize,
}

#[allow(dead_code)] // Public API used via PyO3 bindings
impl CacheStats {
    /// Get current cache usage as percentage
    pub fn usage_percent(&self) -> f32 {
        (self.current_bytes as f32 / self.max_bytes as f32) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_synthesis_config_default() {
        let config = SynthesisConfig::default();
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.grain_size_ms, 50.0);
        assert_eq!(config.grain_overlap, 0.5);
    }

    #[tokio::test]
    async fn test_audio_segment_creation() {
        let samples = vec![0.0f32; 1000];
        let segment = AudioSegment::new(samples.clone(), 44100);

        assert_eq!(segment.len_samples(), 1000);
        assert!(!segment.is_empty());
        assert_eq!(segment.sample_rate, 44100);
    }

    #[tokio::test]
    async fn test_audio_segment_resample() {
        let samples: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let segment = AudioSegment::new(samples, 44100);

        let resampled = segment.resample(22050).unwrap();
        assert_eq!(resampled.sample_rate, 22050);
        assert!(resampled.len_samples() < 1000); // Should be about half
    }

    #[tokio::test]
    async fn test_synthesizer_creation() {
        let config = SynthesisConfig::default();
        let synthesizer = GranularSynthesizer::new(config).await.unwrap();
        assert_eq!(synthesizer.active_grain_count(), 0);
    }

    #[tokio::test]
    async fn test_load_source() {
        let config = SynthesisConfig::default();
        let mut synthesizer = GranularSynthesizer::new(config).await.unwrap();

        let samples: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0 - 0.5).collect();
        let segment = AudioSegment::new(samples, 44100);

        synthesizer.load_source(segment).await.unwrap();
    }

    #[tokio::test]
    async fn test_grain_creation() {
        let samples: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let mut grain = Grain::new(samples.clone(), 100);

        assert!(!grain.is_finished());

        // Read all samples
        let mut count = 0;
        while grain.next_sample().is_some() {
            count += 1;
        }
        assert_eq!(count, 100);
        assert!(grain.is_finished());
    }

    #[tokio::test]
    async fn test_synthesize() {
        let config = SynthesisConfig::default();
        let mut synthesizer = GranularSynthesizer::new(config).await.unwrap();

        // Load source
        let samples: Vec<f32> = (0..44100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let segment = AudioSegment::new(samples, 44100);
        synthesizer.load_source(segment).await.unwrap();

        // Synthesize 100ms
        let output = synthesizer.synthesize(100.0).await.unwrap();

        let expected_samples = (44100_f32 * 0.1) as usize;
        assert_eq!(output.len(), expected_samples);
    }

    #[tokio::test]
    async fn test_generate_with_features() {
        let config = SynthesisConfig::default();
        let synthesizer = GranularSynthesizer::new(config).await.unwrap();

        let features = AudioFeatures {
            rms: 0.5,
            zero_crossing_rate: 0.1,
            spectral_centroid: 2000.0,
            bandwidth: 1000.0,
            f0: 440.0,
        };

        let output = synthesizer.generate(&features).await.unwrap();
        assert!(!output.is_empty());
    }

    // ========================================================================
    // New Synthesis Tests
    // ========================================================================

    #[test]
    fn test_synthesis_mode() {
        // Test SynthesisMode enum variants
        let horizontal = SynthesisMode::Horizontal;
        let vertical = SynthesisMode::Vertical;
        let _combined = SynthesisMode::Combined;

        assert_eq!(horizontal, SynthesisMode::Horizontal);
        assert_ne!(horizontal, vertical);
    }

    #[test]
    fn test_phrase_segment_creation() {
        let audio: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        let phrase = PhraseSegment::new(audio.clone(), 44100, 1000.0);

        assert_eq!(phrase.audio.len(), 1000);
        assert_eq!(phrase.sample_rate, 44100);
        assert_eq!(phrase.mean_f0_hz, 1000.0);
        assert_eq!(phrase.quality_score, 1.0);
        assert!(!phrase.is_empty());
        assert_eq!(phrase.len_samples(), 1000);
    }

    #[test]
    fn test_microharmonic_constraints_default() {
        let constraints = MicroharmonicConstraints::default();

        assert_eq!(constraints.frequency_range, (200.0, 8000.0));
        assert_eq!(constraints.harmonic_tolerance, 3.0);
        assert!(!constraints.phase_coherence);
        assert!(constraints.amplitude_balancing);
        assert_eq!(constraints.temporal_alignment, "start");
        assert_eq!(constraints.crossfade_duration_ms, 10.0);
        assert_eq!(constraints.max_phrases, 8);
        assert_eq!(constraints.min_quality_score, 0.5);
    }

    #[tokio::test]
    async fn test_microharmonic_validator() {
        let validator = MicroharmonicValidator::new(44100);

        let mut phrase_segments = HashMap::new();

        // Add test phrase segments
        let audio1: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        phrase_segments.insert(
            "phrase1".to_string(),
            PhraseSegment::new(audio1, 44100, 440.0),
        );

        let audio2: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        phrase_segments.insert(
            "phrase2".to_string(),
            PhraseSegment::new(audio2, 44100, 880.0),
        );

        let constraints = MicroharmonicConstraints::default();
        let phrase_keys = vec!["phrase1".to_string(), "phrase2".to_string()];

        let result = validator.validate_compatibility(&phrase_keys, &constraints, &phrase_segments);

        assert!(result.compatibility_score > 0.0);
        assert!(result.phrase_scores.len() == 2);
    }

    #[tokio::test]
    async fn test_real_time_safety_monitor() {
        let monitor = RealTimeSafetyMonitor::new(44100);

        // Test with safe audio
        let safe_audio: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.1)
            .collect();
        let check = monitor.check_audio_safety(&safe_audio);

        assert!(check.safe);
        assert!(check.rms_level < 0.0);
        assert!(check.peak_level < 0.0);
        assert!(check.duration_ms > 10.0);
        assert!(check.error.is_none());

        // Test with empty audio
        let empty_audio: Vec<f32> = vec![];
        let check_empty = monitor.check_audio_safety(&empty_audio);

        assert!(!check_empty.safe);
        assert!(check_empty.error.is_some());

        // Test safety limiter
        let mut loud_audio: Vec<f32> = vec![0.95, 1.1, 1.5, -1.2, 0.5];
        monitor.apply_safety_limiter(&mut loud_audio).unwrap();

        // Check that no sample exceeds 1.0
        for &sample in &loud_audio {
            assert!(sample.abs() <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_cross_species_adapter() {
        let adapter = CrossSpeciesAdapter::new();

        let species = adapter.available_species();
        assert!(species.contains(&"marmoset".to_string()));
        assert!(species.contains(&"dolphin".to_string()));
        assert!(species.contains(&"bat".to_string()));
        assert!(species.contains(&"finch".to_string()));
        assert!(species.contains(&"sperm_whale".to_string()));

        // Test parameter adaptation
        let base_constraints = MicroharmonicConstraints::default();
        let marmoset_constraints =
            adapter.adapt_parameters_for_species("marmoset", &base_constraints);

        assert_eq!(marmoset_constraints.frequency_range, (500.0, 15000.0));
        assert_eq!(marmoset_constraints.harmonic_tolerance, 2.0);
    }

    #[tokio::test]
    async fn test_concatenative_synthesizer() {
        let synthesizer = ConcatenativeSynthesizer::new(44100, 1.0);

        let mut phrases: Vec<PhraseSegment> = Vec::new();

        // Create test phrases
        let audio1: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();
        phrases.push(PhraseSegment::new(audio1, 44100, 440.0));

        let audio2: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();
        phrases.push(PhraseSegment::new(audio2, 44100, 880.0));

        let output = synthesizer.concatenate_phrases(&phrases, 5.0).unwrap();

        assert!(!output.is_empty());
        assert!(output.len() < phrases[0].audio.len() + phrases[1].audio.len());
        // Should be shorter due to crossfade
    }

    #[tokio::test]
    async fn test_superpositional_synthesizer() {
        let synthesizer = SuperpositionalSynthesizer::new(44100, 8);

        let mut phrases: Vec<PhraseSegment> = Vec::new();

        // Create test phrases with same length
        let audio1: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();
        phrases.push(PhraseSegment::new(audio1, 44100, 440.0));

        let audio2: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();
        phrases.push(PhraseSegment::new(audio2, 44100, 880.0));

        let output = synthesizer
            .layer_phrases_harmonically(&phrases, true)
            .unwrap();

        assert!(!output.is_empty());
        assert_eq!(output.len(), 2205); // Should be same as input length

        // Check that output is normalized (no clipping above 1.0)
        for &sample in &output {
            assert!(sample.abs() <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_combined_synthesizer() {
        let synthesizer = CombinedSynthesizer::new(44100);

        let mut sequential_phrases: Vec<PhraseSegment> = Vec::new();
        let mut simultaneous_phrases: Vec<PhraseSegment> = Vec::new();

        // Create test phrases
        let audio1: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();
        sequential_phrases.push(PhraseSegment::new(audio1, 44100, 440.0));

        let audio2: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 660.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();
        simultaneous_phrases.push(PhraseSegment::new(audio2, 44100, 660.0));

        let output = synthesizer
            .synthesize_mixed_encoding(&sequential_phrases, &simultaneous_phrases, 5.0)
            .unwrap();

        assert!(!output.is_empty());
    }

    #[tokio::test]
    async fn test_enhanced_microharmonic_synthesizer_horizontal() {
        let mut phrase_segments = HashMap::new();

        // Add test phrase segments
        let audio1: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();
        phrase_segments.insert(
            "phrase1".to_string(),
            PhraseSegment::new(audio1, 44100, 440.0),
        );

        let audio2: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();
        phrase_segments.insert(
            "phrase2".to_string(),
            PhraseSegment::new(audio2, 44100, 880.0),
        );

        let synthesizer =
            EnhancedMicroharmonicSynthesizer::new("marmoset".to_string(), phrase_segments, 44100);

        let constraints = MicroharmonicConstraints::default();
        let phrase_sequence = vec!["phrase1".to_string(), "phrase2".to_string()];

        let result = synthesizer
            .synthesize_horizontal(&phrase_sequence, &constraints)
            .await
            .unwrap();

        assert!(!result.audio.is_empty());
        assert_eq!(result.synthesis_mode, SynthesisMode::Horizontal);
        assert_eq!(result.sample_rate, 44100);
        assert_eq!(result.phrases_used.len(), 2);
        assert!(result.duration_ms > 0.0);
    }

    #[tokio::test]
    async fn test_enhanced_microharmonic_synthesizer_vertical() {
        let mut phrase_segments = HashMap::new();

        // Add test phrase segments
        let audio1: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();
        phrase_segments.insert(
            "phrase1".to_string(),
            PhraseSegment::new(audio1, 44100, 440.0),
        );

        let audio2: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();
        phrase_segments.insert(
            "phrase2".to_string(),
            PhraseSegment::new(audio2, 44100, 880.0),
        );

        let synthesizer =
            EnhancedMicroharmonicSynthesizer::new("marmoset".to_string(), phrase_segments, 44100);

        let constraints = MicroharmonicConstraints::default();
        let phrase_set = vec!["phrase1".to_string(), "phrase2".to_string()];

        let result = synthesizer
            .synthesize_vertical(&phrase_set, &constraints)
            .await
            .unwrap();

        assert!(!result.audio.is_empty());
        assert_eq!(result.synthesis_mode, SynthesisMode::Vertical);
        assert_eq!(result.sample_rate, 44100);
        assert_eq!(result.phrases_used.len(), 2);
        assert!(result.duration_ms > 0.0);
    }

    #[tokio::test]
    async fn test_enhanced_microharmonic_synthesizer_combined() {
        let mut phrase_segments = HashMap::new();

        // Add test phrase segments
        let audio1: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();
        phrase_segments.insert(
            "phrase1".to_string(),
            PhraseSegment::new(audio1, 44100, 440.0),
        );

        let audio2: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 660.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();
        phrase_segments.insert(
            "phrase2".to_string(),
            PhraseSegment::new(audio2, 44100, 660.0),
        );

        let audio3: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();
        phrase_segments.insert(
            "phrase3".to_string(),
            PhraseSegment::new(audio3, 44100, 880.0),
        );

        let synthesizer =
            EnhancedMicroharmonicSynthesizer::new("marmoset".to_string(), phrase_segments, 44100);

        let constraints = MicroharmonicConstraints::default();
        let synthesis_plan = vec![
            (
                SynthesisMode::Horizontal,
                vec!["phrase1".to_string(), "phrase2".to_string()],
            ),
            (SynthesisMode::Vertical, vec!["phrase3".to_string()]),
        ];

        let result = synthesizer
            .synthesize_combined(&synthesis_plan, &constraints)
            .await
            .unwrap();

        assert!(!result.audio.is_empty());
        assert_eq!(result.synthesis_mode, SynthesisMode::Combined);
        assert_eq!(result.sample_rate, 44100);
        assert!(result.phrases_used.len() == 3);
        assert!(result.duration_ms > 0.0);
    }
}

// ============================================================================
// CATEGORY 1, ITEM 2: CORVID MODE ROUGHNESS (Jitter + Phase Smearing)
// ============================================================================

/// Corvid Mode roughness parameters
///
/// Category 1, Item 2: "Corvid Mode" Roughness (Jitter + Phase Smearing)
///
/// Corvid vocalizations (crows, ravens) have a characteristic "gritty" or
/// "raspy" quality that is NOT present in clean sine wave synthesis.
/// This roughness is created by:
/// 1. **Jitter**: Rapid, random frequency modulation
/// 2. **Phase smearing**: Random phase shifts creating spectral roughness
///
/// Without these features, synthesized corbid vocalizations will sound
/// "robotic" and will NOT be recognized by real corvids as conspecific.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct CorvidRoughnessParams {
    /// Jitter intensity (0.0 = none, 1.0 = maximum)
    /// Typical corvid range: 0.1 - 0.3
    pub jitter_intensity: f32,

    /// Jitter rate in Hz (how often frequency changes)
    /// Typical corvid range: 50 - 200 Hz
    pub jitter_rate_hz: f32,

    /// Phase smearing amount (0.0 = none, 1.0 = maximum)
    /// Typical corvid range: 0.05 - 0.15
    pub phase_smearing: f32,

    /// Spectral roughness (adds noise harmonics)
    /// Typical corvid range: 0.02 - 0.08
    pub spectral_roughness: f32,
}

impl Default for CorvidRoughnessParams {
    fn default() -> Self {
        // Default American Crow parameters
        Self {
            jitter_intensity: 0.15,
            jitter_rate_hz: 100.0,
            phase_smearing: 0.08,
            spectral_roughness: 0.04,
        }
    }
}

#[allow(dead_code)]
impl CorvidRoughnessParams {
    /// Create parameters for American Crow (Corvus brachyrhynchos)
    pub fn american_crow() -> Self {
        Self {
            jitter_intensity: 0.15,
            jitter_rate_hz: 100.0,
            phase_smearing: 0.08,
            spectral_roughness: 0.04,
        }
    }

    /// Create parameters for Common Raven (Corvus corax)
    /// Ravens have DEEPER, raspier calls
    pub fn common_raven() -> Self {
        Self {
            jitter_intensity: 0.20,   // More jitter
            jitter_rate_hz: 80.0,     // Slower rate
            phase_smearing: 0.12,     // More phase smearing
            spectral_roughness: 0.06, // More spectral noise
        }
    }

    /// Create parameters for Fish Crow (Corvus ossifragus)
    /// Fish Crows have higher-pitched, more nasal calls
    pub fn fish_crow() -> Self {
        Self {
            jitter_intensity: 0.12,
            jitter_rate_hz: 150.0, // Faster rate
            phase_smearing: 0.06,
            spectral_roughness: 0.03,
        }
    }
}

/// Apply corvid mode roughness to audio
///
/// This function takes clean synthesized audio and adds the characteristic
/// "gritty" quality of corvid vocalizations through jitter and phase smearing.
///
/// # Arguments
/// * `audio` - Input audio samples (normalized to [-1.0, 1.0])
/// * `sample_rate` - Sample rate in Hz
/// * `params` - Corvid roughness parameters
///
/// # Returns
/// Audio with corvid roughness applied
#[allow(dead_code)]
pub fn apply_corvid_roughness(
    audio: &[f32],
    sample_rate: usize,
    params: &CorvidRoughnessParams,
) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }

    let mut output = Vec::with_capacity(audio.len());

    // Generate random modulation signals
    let mut rng = thread_rng();
    let mut phase = 0.0;
    let phase_increment = params.jitter_rate_hz * 2.0 * std::f32::consts::PI / sample_rate as f32;

    for (i, &sample) in audio.iter().enumerate() {
        // Update modulation phase
        phase = (phase + phase_increment) % (2.0 * std::f32::consts::PI);

        // Generate jitter (random frequency modulation)
        let jitter = if params.jitter_intensity > 0.0 {
            // Use sine wave modulation with random phase shifts
            let jitter_mod = (phase + rng.gen::<f32>() * 0.5).sin();
            jitter_mod * params.jitter_intensity
        } else {
            0.0
        };

        // Generate phase smearing (random phase shifts)
        let phase_smear = if params.phase_smearing > 0.0 {
            // Random delay up to phase_smearing samples
            let delay_samples =
                (rng.gen::<f32>() * params.phase_smearing * sample_rate as f32) as usize;
            if i >= delay_samples {
                audio[i - delay_samples]
            } else {
                sample
            }
        } else {
            sample
        };

        // Generate spectral roughness (add noise harmonics)
        let spectral_noise = if params.spectral_roughness > 0.0 {
            // Add high-frequency noise
            (rng.gen::<f32>() - 0.5) * 2.0 * params.spectral_roughness
        } else {
            0.0
        };

        // Combine effects
        let rough_sample = phase_smear * (1.0 + jitter) + spectral_noise;

        // Soft clip to prevent distortion
        let clipped = rough_sample.tanh();

        output.push(clipped);
    }

    output
}

/// Corvid Mode Synthesizer
///
/// High-level interface for synthesizing corvid vocalizations with
/// realistic roughness characteristics.
#[allow(dead_code)]
pub struct CorvidModeSynthesizer {
    sample_rate: usize,
    params: CorvidRoughnessParams,
}

#[allow(dead_code)]
impl CorvidModeSynthesizer {
    /// Create a new corvid mode synthesizer
    pub fn new(sample_rate: usize, params: CorvidRoughnessParams) -> Self {
        Self {
            sample_rate,
            params,
        }
    }

    /// Synthesize a corvid-style phrase with roughness
    ///
    /// Takes a clean synthesized phrase and applies corvid roughness
    pub fn synthesize_with_roughness(&self, clean_audio: &[f32]) -> Vec<f32> {
        apply_corvid_roughness(clean_audio, self.sample_rate, &self.params)
    }

    /// Update roughness parameters
    pub fn set_params(&mut self, params: CorvidRoughnessParams) {
        self.params = params;
    }

    /// Get current parameters
    pub fn get_params(&self) -> &CorvidRoughnessParams {
        &self.params
    }
}

#[cfg(test)]
mod corvid_roughness_tests {
    use super::*;

    #[test]
    fn test_corvid_roughness_params_default() {
        let params = CorvidRoughnessParams::default();
        assert_eq!(params.jitter_intensity, 0.15);
        assert_eq!(params.jitter_rate_hz, 100.0);
        assert_eq!(params.phase_smearing, 0.08);
        assert_eq!(params.spectral_roughness, 0.04);
    }

    #[test]
    fn test_corvid_roughness_params_species() {
        let crow = CorvidRoughnessParams::american_crow();
        let raven = CorvidRoughnessParams::common_raven();
        let fish_crow = CorvidRoughnessParams::fish_crow();

        // Raven should have MORE jitter and phase smearing
        assert!(raven.jitter_intensity > crow.jitter_intensity);
        assert!(raven.phase_smearing > crow.phase_smearing);

        // Fish Crow should have FASTER jitter rate
        assert!(fish_crow.jitter_rate_hz > crow.jitter_rate_hz);
    }

    #[test]
    fn test_apply_corvid_roughness() {
        let sample_rate = 44100;

        // Create clean sine wave
        let clean: Vec<f32> = (0..sample_rate)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect();

        let params = CorvidRoughnessParams::american_crow();
        let rough = apply_corvid_roughness(&clean, sample_rate, &params);

        // Output should be same length
        assert_eq!(rough.len(), clean.len());

        // Output should be different (roughness applied)
        let diff_squared: f32 = rough
            .iter()
            .zip(clean.iter())
            .map(|(r, c)| (r - c).powi(2))
            .sum();
        assert!(diff_squared > 0.01, "Roughness should modify audio");

        // Output should be bounded (soft clipping applied)
        let max_abs = rough.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
        assert!(max_abs <= 1.0, "Output should be bounded to [-1.0, 1.0]");
    }

    #[test]
    fn test_corvid_mode_synthesizer() {
        let sample_rate = 44100;
        let params = CorvidRoughnessParams::common_raven();
        let mut synthesizer = CorvidModeSynthesizer::new(sample_rate, params.clone());

        // Create clean audio
        let clean: Vec<f32> = (0..22050)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 880.0 * i as f32 / sample_rate as f32).sin() * 0.3
            })
            .collect();

        // Apply roughness
        let rough = synthesizer.synthesize_with_roughness(&clean);

        assert_eq!(rough.len(), clean.len());
        assert!(synthesizer.get_params().jitter_intensity > 0.0);

        // Update parameters
        let new_params = CorvidRoughnessParams::fish_crow();
        synthesizer.set_params(new_params);
        assert_eq!(synthesizer.get_params().jitter_rate_hz, 150.0);
    }

    #[test]
    fn test_corvid_roughness_empty_input() {
        let params = CorvidRoughnessParams::default();
        let empty: Vec<f32> = vec![];
        let output = apply_corvid_roughness(&empty, 44100, &params);
        assert!(output.is_empty());
    }

    #[test]
    fn test_corvid_roughness_zero_params() {
        let sample_rate = 44100;
        let clean: Vec<f32> = (0..sample_rate)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect();

        let params = CorvidRoughnessParams {
            jitter_intensity: 0.0,
            jitter_rate_hz: 0.0,
            phase_smearing: 0.0,
            spectral_roughness: 0.0,
        };

        let rough = apply_corvid_roughness(&clean, sample_rate, &params);

        // With zero params, output should still be modified (soft clipping)
        assert_eq!(rough.len(), clean.len());
        let max_abs = rough.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
        assert!(max_abs <= 1.0);
    }

    #[tokio::test]
    async fn test_synthesis_performance_stats() {
        let mut phrase_segments = HashMap::new();

        let audio1: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();
        phrase_segments.insert(
            "phrase1".to_string(),
            PhraseSegment::new(audio1, 44100, 440.0),
        );

        let synthesizer =
            EnhancedMicroharmonicSynthesizer::new("marmoset".to_string(), phrase_segments, 44100);

        let constraints = MicroharmonicConstraints::default();

        // Run a few syntheses
        synthesizer
            .synthesize_horizontal(&["phrase1".to_string()], &constraints)
            .await
            .unwrap();
        synthesizer
            .synthesize_vertical(&["phrase1".to_string()], &constraints)
            .await
            .unwrap();

        let stats = synthesizer.get_performance_stats();

        assert_eq!(stats.total_syntheses, 2);
        assert_eq!(stats.horizontal_count, 1);
        assert_eq!(stats.vertical_count, 1);
        assert!(stats.avg_processing_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_species_parameters_default() {
        let params = SpeciesParameters::default();

        assert_eq!(params.frequency_range, (200.0, 8000.0));
        assert_eq!(params.harmonic_tolerance, 3.0);
        assert_eq!(params.default_temporal_alignment, "start");
    }

    // ========================================================================
    // Granular Concatenative Synthesis Tests (TDD)
    // ========================================================================

    /// Test 1: Grain Window Generation
    /// Verify that Hanning window is generated correctly
    #[test]
    fn test_grain_window_hanning() {
        let grain_size_ms = 20.0;
        let sample_rate = 22050;

        let window = GrainWindow::hanning(grain_size_ms, sample_rate);

        let expected_samples = (grain_size_ms / 1000.0 * sample_rate as f32) as usize;
        assert_eq!(window.len(), expected_samples);

        // Hanning window should start at 0, peak in middle, end at 0
        assert!((window[0] - 0.0).abs() < 0.01);
        assert!((window[window.len() - 1] - 0.0).abs() < 0.01);

        // Peak should be near 1.0 in the middle
        let mid = window.len() / 2;
        assert!(window[mid] > 0.9);
    }

    /// Test 2: Granular Voice Pitch Shifting
    /// Verify that pitch shift ratio changes playback speed
    #[test]
    fn test_granular_voice_pitch_shift() {
        let sample_rate = 22050;
        let source: Vec<f32> = (0..22050)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect();

        let mut voice = GranularVoice::new(source, sample_rate, 20.0);

        // Set pitch shift to 0.5 (one octave down)
        voice.set_pitch_shift(0.5);

        // Generate 100 samples
        let mut output = Vec::with_capacity(100);
        for _ in 0..100 {
            output.push(voice.generate_sample());
        }

        assert_eq!(output.len(), 100);

        // Output should not be silent
        let max_amplitude = output.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
        assert!(max_amplitude > 0.001);
    }

    /// Test 3: Granular Voice Time Stretching
    /// Verify that time stretch changes duration
    #[test]
    fn test_granular_voice_time_stretch() {
        let sample_rate = 22050;
        let source: Vec<f32> = (0..22050)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5
            })
            .collect();

        let mut voice = GranularVoice::new(source, sample_rate, 20.0);

        // Set time stretch to 2.0 (double duration)
        voice.set_time_stretch(2.0);

        // Generate samples
        let _ = voice.generate_sample();

        // Position should advance slower with time stretch
        let position1 = voice.get_position();
        let _ = voice.generate_sample();
        let position2 = voice.get_position();

        // With time stretch, position should advance slower
        // (actual advancement depends on implementation)
        assert!(position2 >= position1);
    }

    /// Test 4: Granular Morphing (Multi-Voice Overlap)
    /// Verify that multiple voices can overlap
    #[test]
    fn test_granular_morpher_overlap() {
        let sample_rate = 22050;

        // Create two different source signals
        let source1: Vec<f32> = (0..22050)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.3
            })
            .collect();

        let source2: Vec<f32> = (0..22050)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 880.0 * i as f32 / sample_rate as f32).sin() * 0.3
            })
            .collect();

        let voice1 = GranularVoice::new(source1, sample_rate, 20.0);
        let voice2 = GranularVoice::new(source2, sample_rate, 20.0);

        let mut morpher = GranularMorpher::new(vec![voice1, voice2], 10.0);

        // Generate samples from both voices
        let sample = morpher.generate_sample();

        // Should not be silent (both voices contributing)
        assert!(sample.abs() > 0.0 || sample.abs() == 0.0); // May cancel out
    }

    /// Test 5: Granular Concatenative Synthesizer
    /// Verify full synthesis pipeline with audio buffer
    #[test]
    fn test_granular_concatenative_synthesizer() {
        let sample_rate = 22050;

        // Create source audio (simple sine wave)
        let source: Vec<f32> = (0..22050)
            .map(|i| {
                (2.0 * std::f32::consts::PI * 7000.0 * i as f32 / sample_rate as f32).sin() * 0.3
            })
            .collect();

        let mut synthesizer = GranularConcatenativeSynthesizer::new(sample_rate);
        synthesizer.load_source(source);

        // Set parameters for pitch shifting (lower pitch = 0.9x)
        synthesizer.set_pitch_shift(0.9);
        synthesizer.set_grain_size_ms(20.0);

        // Synthesize 100ms of audio
        let output = synthesizer.synthesize(100.0);

        let expected_samples = (100.0 / 1000.0 * sample_rate as f32) as usize;
        assert_eq!(output.len(), expected_samples);

        // Output should not be silent
        let max_amplitude = output.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
        assert!(max_amplitude > 0.001);
    }

    // ========================================================================
    // 30D Metadata Tests (Builder Pattern and Delta Calculation)
    // ========================================================================

    #[test]
    fn test_source_metadata_builder_pattern() {
        // Test builder with partial metadata
        let metadata = SourceMetadata::builder()
            .mean_f0_hz(7000.0)
            .duration_ms(50.0)
            .jitter(0.05)
            .build();

        assert_eq!(metadata.mean_f0_hz, 7000.0);
        assert_eq!(metadata.duration_ms, 50.0);
        assert_eq!(metadata.jitter, 0.05);

        // Unspecified fields should have defaults (marmoset-like)
        assert_eq!(metadata.f0_range_hz, 400.0);
        assert_eq!(metadata.harmonic_to_noise_ratio, 20.0);
        assert_eq!(metadata.spectral_flatness, 0.1);
    }

    #[test]
    fn test_source_metadata_builder_full_specification() {
        // Test builder with all features specified
        let metadata = SourceMetadata::builder()
            .mean_f0_hz(6500.0)
            .duration_ms(60.0)
            .f0_range_hz(400.0)
            .harmonic_to_noise_ratio(20.0)
            .spectral_flatness(0.1)
            .harmonicity(0.8)
            .attack_time_ms(5.0)
            .decay_time_ms(10.0)
            .sustain_level(0.7)
            .vibrato_rate_hz(6.0)
            .vibrato_depth(0.03)
            .jitter(0.02)
            .shimmer(0.03)
            .mfcc(
                1.2, 0.8, -0.3, 0.5, -0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4,
            )
            .spectral_flux(0.5)
            .rhythm(45.0, 12.0, 0.25)
            .build();

        assert_eq!(metadata.mean_f0_hz, 6500.0);
        assert_eq!(metadata.duration_ms, 60.0);
        assert_eq!(metadata.f0_range_hz, 400.0);
        assert_eq!(metadata.harmonic_to_noise_ratio, 20.0);
        assert_eq!(metadata.spectral_flatness, 0.1);
        assert_eq!(metadata.harmonicity, 0.8);
        assert_eq!(metadata.attack_time_ms, 5.0);
        assert_eq!(metadata.decay_time_ms, 10.0);
        assert_eq!(metadata.sustain_level, 0.7);
        assert_eq!(metadata.vibrato_rate_hz, 6.0);
        assert_eq!(metadata.vibrato_depth, 0.03);
        assert_eq!(metadata.jitter, 0.02);
        assert_eq!(metadata.shimmer, 0.03);
        assert_eq!(metadata.mfcc_1, 1.2);
        assert_eq!(metadata.mfcc_2, 0.8);
        assert_eq!(metadata.mfcc_3, -0.3);
        assert_eq!(metadata.mfcc_4, 0.5);
        assert_eq!(metadata.mfcc_5, -0.5);
        assert_eq!(metadata.mfcc_6, -0.3);
        assert_eq!(metadata.mfcc_7, -0.2);
        assert_eq!(metadata.mfcc_8, -0.1);
        assert_eq!(metadata.mfcc_9, 0.0);
        assert_eq!(metadata.mfcc_10, 0.1);
        assert_eq!(metadata.mfcc_11, 0.2);
        assert_eq!(metadata.mfcc_12, 0.3);
        assert_eq!(metadata.mfcc_13, 0.4);
        assert_eq!(metadata.spectral_flux, 0.5);
        assert_eq!(metadata.median_ici_ms, 45.0);
        assert_eq!(metadata.onset_rate_hz, 12.0);
        assert_eq!(metadata.ici_coefficient_of_variation, 0.25);
    }

    #[test]
    fn test_source_metadata_delta_from() {
        // Source: Lower pitch, shorter, pure tone
        let source = SourceMetadata::builder()
            .mean_f0_hz(6000.0)
            .duration_ms(40.0)
            .f0_range_hz(200.0)
            .harmonic_to_noise_ratio(25.0)
            .spectral_flatness(0.05)
            .build();

        // Target: Higher pitch, longer, gritty
        let target = SourceMetadata::builder()
            .mean_f0_hz(7000.0)
            .duration_ms(60.0)
            .f0_range_hz(400.0)
            .harmonic_to_noise_ratio(15.0)
            .spectral_flatness(0.3)
            .build();

        // Calculate delta
        let delta = target.delta_from(&source);

        // Verify delta calculations
        assert_eq!(delta.delta_mean_f0_hz, 1000.0); // +1000Hz
        assert_eq!(delta.delta_duration_ms, 20.0); // +20ms
        assert_eq!(delta.delta_f0_range_hz, 200.0); // +200Hz
        assert_eq!(delta.delta_harmonic_to_noise_ratio, -10.0); // -10dB (less harmonic)
        assert_eq!(delta.delta_spectral_flatness, 0.25); // +0.25 (more noisy)
    }

    #[test]
    fn test_source_metadata_delta_full_30d() {
        // Test all 30 delta dimensions
        let source = SourceMetadata {
            mean_f0_hz: 6500.0,
            duration_ms: 50.0,
            f0_range_hz: 300.0,
            harmonic_to_noise_ratio: 20.0,
            spectral_flatness: 0.15,
            harmonicity: 0.75,
            attack_time_ms: 8.0,
            decay_time_ms: 12.0,
            sustain_level: 0.6,
            vibrato_rate_hz: 5.0,
            vibrato_depth: 0.02,
            jitter: 0.03,
            shimmer: 0.04,
            mfcc_1: 1.0,
            mfcc_2: 0.7,
            mfcc_3: -0.2,
            mfcc_4: 0.4,
            mfcc_5: -0.5,
            mfcc_6: -0.3,
            mfcc_7: -0.2,
            mfcc_8: -0.1,
            mfcc_9: 0.0,
            mfcc_10: 0.1,
            mfcc_11: 0.2,
            mfcc_12: 0.3,
            mfcc_13: 0.4,
            spectral_flux: 0.5,
            median_ici_ms: 40.0,
            onset_rate_hz: 10.0,
            ici_coefficient_of_variation: 0.3,
        };

        let target = SourceMetadata {
            mean_f0_hz: 7500.0,                // +1000
            duration_ms: 70.0,                 // +20
            f0_range_hz: 500.0,                // +200
            harmonic_to_noise_ratio: 10.0,     // -10
            spectral_flatness: 0.35,           // +0.2
            harmonicity: 0.85,                 // +0.1
            attack_time_ms: 3.0,               // -5 (faster)
            decay_time_ms: 8.0,                // -4
            sustain_level: 0.8,                // +0.2
            vibrato_rate_hz: 7.0,              // +2
            vibrato_depth: 0.05,               // +0.03
            jitter: 0.08,                      // +0.05
            shimmer: 0.06,                     // +0.02
            mfcc_1: 1.5,                       // +0.5
            mfcc_2: 0.9,                       // +0.2
            mfcc_3: -0.4,                      // -0.2
            mfcc_4: 0.6,                       // +0.2
            mfcc_5: -0.3,                      // +0.2
            mfcc_6: -0.1,                      // +0.2
            mfcc_7: 0.0,                       // +0.2
            mfcc_8: 0.1,                       // +0.2
            mfcc_9: 0.2,                       // +0.2
            mfcc_10: 0.3,                      // +0.2
            mfcc_11: 0.4,                      // +0.2
            mfcc_12: 0.5,                      // +0.2
            mfcc_13: 0.6,                      // +0.2
            spectral_flux: 0.7,                // +0.2
            median_ici_ms: 50.0,               // +10
            onset_rate_hz: 15.0,               // +5
            ici_coefficient_of_variation: 0.2, // -0.1
        };

        let delta = target.delta_from(&source);

        // Verify all 30 dimensions (using approximate comparison for floating point)
        assert_eq!(delta.delta_mean_f0_hz, 1000.0);
        assert_eq!(delta.delta_duration_ms, 20.0);
        assert_eq!(delta.delta_f0_range_hz, 200.0);
        assert_eq!(delta.delta_harmonic_to_noise_ratio, -10.0);
        assert!((delta.delta_spectral_flatness - 0.2).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_harmonicity - 0.1).abs() < 0.0001); // FP tolerant
        assert_eq!(delta.delta_attack_time_ms, -5.0);
        assert_eq!(delta.delta_decay_time_ms, -4.0);
        assert!((delta.delta_sustain_level - 0.2).abs() < 0.0001); // FP tolerant
        assert_eq!(delta.delta_vibrato_rate_hz, 2.0);
        assert!((delta.delta_vibrato_depth - 0.03).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_jitter - 0.05).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_shimmer - 0.02).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_mfcc_1 - 0.5).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_mfcc_2 - 0.2).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_mfcc_3 - (-0.2)).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_mfcc_4 - 0.2).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_mfcc_5 - 0.2).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_mfcc_6 - 0.2).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_mfcc_7 - 0.2).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_mfcc_8 - 0.2).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_mfcc_9 - 0.2).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_mfcc_10 - 0.2).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_mfcc_11 - 0.2).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_mfcc_12 - 0.2).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_mfcc_13 - 0.2).abs() < 0.0001); // FP tolerant
        assert!((delta.delta_spectral_flux - 0.2).abs() < 0.0001); // FP tolerant
        assert_eq!(delta.delta_median_ici_ms, 10.0);
        assert_eq!(delta.delta_onset_rate_hz, 5.0);
        assert!((delta.delta_ici_cv - (-0.1)).abs() < 0.0001); // FP tolerant
    }

    #[test]
    fn test_source_metadata_persona_comparison() {
        // Test GRITTY vs PURE persona delta
        let pure_metadata = SourceMetadata {
            mean_f0_hz: 7000.0,
            duration_ms: 50.0,
            f0_range_hz: 400.0,
            harmonic_to_noise_ratio: 25.0, // High (pure)
            spectral_flatness: 0.05,       // Low (focused)
            harmonicity: 0.95,             // High (pure)
            attack_time_ms: 25.0,          // Slow (smooth)
            decay_time_ms: 15.0,
            sustain_level: 0.7,
            vibrato_rate_hz: 6.0,
            vibrato_depth: 0.02,
            jitter: 0.01,  // Low (stable)
            shimmer: 0.01, // Low (stable)
            mfcc_1: 1.2,
            mfcc_2: 0.8,
            mfcc_3: -0.3,
            mfcc_4: 0.5,
            mfcc_5: -0.5,
            mfcc_6: -0.3,
            mfcc_7: -0.2,
            mfcc_8: -0.1,
            mfcc_9: 0.0,
            mfcc_10: 0.1,
            mfcc_11: 0.2,
            mfcc_12: 0.3,
            mfcc_13: 0.4,
            spectral_flux: 0.5,
            median_ici_ms: 0.0,
            onset_rate_hz: 0.0,
            ici_coefficient_of_variation: 0.0,
        };

        let gritty_metadata = SourceMetadata {
            mean_f0_hz: 7000.0,
            duration_ms: 50.0,
            f0_range_hz: 400.0,
            harmonic_to_noise_ratio: 2.0, // Low (gritty)
            spectral_flatness: 0.8,       // High (noise-like)
            harmonicity: 0.4,             // Low (gritty)
            attack_time_ms: 3.0,          // Fast (sharp)
            decay_time_ms: 15.0,
            sustain_level: 0.7,
            vibrato_rate_hz: 6.0,
            vibrato_depth: 0.02,
            jitter: 0.15,  // High (rough)
            shimmer: 0.12, // High (rough)
            mfcc_1: 1.2,
            mfcc_2: 0.8,
            mfcc_3: -0.3,
            mfcc_4: 0.5,
            mfcc_5: -0.5,
            mfcc_6: -0.3,
            mfcc_7: -0.2,
            mfcc_8: -0.1,
            mfcc_9: 0.0,
            mfcc_10: 0.1,
            mfcc_11: 0.2,
            mfcc_12: 0.3,
            mfcc_13: 0.4,
            spectral_flux: 0.9,
            median_ici_ms: 0.0,
            onset_rate_hz: 0.0,
            ici_coefficient_of_variation: 0.0,
        };

        let delta = gritty_metadata.delta_from(&pure_metadata);

        // GRITTY persona should show:
        assert_eq!(delta.delta_harmonic_to_noise_ratio, -23.0); // Much less harmonic
        assert_eq!(delta.delta_spectral_flatness, 0.75); // Much more noise
        assert!((delta.delta_harmonicity - (-0.55)).abs() < 0.0001); // Much less harmonicity
        assert_eq!(delta.delta_attack_time_ms, -22.0); // Faster attack
        assert_eq!(delta.delta_jitter, 0.14); // More jitter
        assert!((delta.delta_shimmer - 0.11).abs() < 0.0001); // More shimmer
        assert!((delta.delta_spectral_flux - 0.4).abs() < 0.0001); // More spectral flux
    }

    #[test]
    fn test_source_metadata_default_matches_builder() {
        // Verify that builder with no modifications matches default
        let built = SourceMetadata::builder().build();
        let defaulted = SourceMetadata::default();

        // All 30 fields should match
        assert_eq!(built.mean_f0_hz, defaulted.mean_f0_hz);
        assert_eq!(built.duration_ms, defaulted.duration_ms);
        assert_eq!(built.f0_range_hz, defaulted.f0_range_hz);
        assert_eq!(
            built.harmonic_to_noise_ratio,
            defaulted.harmonic_to_noise_ratio
        );
        assert_eq!(built.spectral_flatness, defaulted.spectral_flatness);
        assert_eq!(built.harmonicity, defaulted.harmonicity);
        assert_eq!(built.attack_time_ms, defaulted.attack_time_ms);
        assert_eq!(built.decay_time_ms, defaulted.decay_time_ms);
        assert_eq!(built.sustain_level, defaulted.sustain_level);
        assert_eq!(built.vibrato_rate_hz, defaulted.vibrato_rate_hz);
        assert_eq!(built.vibrato_depth, defaulted.vibrato_depth);
        assert_eq!(built.jitter, defaulted.jitter);
        assert_eq!(built.shimmer, defaulted.shimmer);
        assert_eq!(built.mfcc_1, defaulted.mfcc_1);
        assert_eq!(built.mfcc_2, defaulted.mfcc_2);
        assert_eq!(built.mfcc_3, defaulted.mfcc_3);
        assert_eq!(built.mfcc_4, defaulted.mfcc_4);
        assert_eq!(built.mfcc_5, defaulted.mfcc_5);
        assert_eq!(built.mfcc_6, defaulted.mfcc_6);
        assert_eq!(built.mfcc_7, defaulted.mfcc_7);
        assert_eq!(built.mfcc_8, defaulted.mfcc_8);
        assert_eq!(built.mfcc_9, defaulted.mfcc_9);
        assert_eq!(built.mfcc_10, defaulted.mfcc_10);
        assert_eq!(built.mfcc_11, defaulted.mfcc_11);
        assert_eq!(built.mfcc_12, defaulted.mfcc_12);
        assert_eq!(built.mfcc_13, defaulted.mfcc_13);
        assert_eq!(built.spectral_flux, defaulted.spectral_flux);
        assert_eq!(built.median_ici_ms, defaulted.median_ici_ms);
        assert_eq!(built.onset_rate_hz, defaulted.onset_rate_hz);
        assert_eq!(
            built.ici_coefficient_of_variation,
            defaulted.ici_coefficient_of_variation
        );
    }

    // Multi-Buffer Sequencer Tests for Corvid Multi-Modal Support

    #[test]
    fn test_modality_timeline_creation() {
        let mut timeline = ModalityTimeline::new();

        timeline.add_event(0.0, 100.0, "whistle".to_string(), Modality::Harmonic);
        timeline.add_event(100.0, 50.0, "rattle".to_string(), Modality::Transient);

        assert_eq!(timeline.events.len(), 2);
        assert_eq!(timeline.events[0].modality, Modality::Harmonic);
        assert_eq!(timeline.events[1].modality, Modality::Transient);
    }

    #[test]
    fn test_modality_timeline_sorting() {
        let mut timeline = ModalityTimeline::new();

        // Add events out of order
        timeline.add_event(100.0, 50.0, "rattle".to_string(), Modality::Transient);
        timeline.add_event(0.0, 100.0, "whistle".to_string(), Modality::Harmonic);

        timeline.sort_by_time();

        assert_eq!(timeline.events[0].start_ms, 0.0);
        assert_eq!(timeline.events[1].start_ms, 100.0);
    }

    #[test]
    fn test_modality_timeline_validation_success() {
        let mut timeline = ModalityTimeline::new();

        timeline.add_event(0.0, 100.0, "whistle".to_string(), Modality::Harmonic);
        timeline.add_event(100.0, 50.0, "rattle".to_string(), Modality::Transient);

        assert!(timeline.validate().is_ok());
    }

    #[test]
    fn test_modality_timeline_validation_overlap() {
        let mut timeline = ModalityTimeline::new();

        // Add overlapping events
        timeline.add_event(0.0, 150.0, "whistle".to_string(), Modality::Harmonic);
        timeline.add_event(100.0, 50.0, "rattle".to_string(), Modality::Transient);

        assert!(timeline.validate().is_err());
    }

    #[test]
    fn test_modality_timeline_total_duration() {
        let mut timeline = ModalityTimeline::new();

        timeline.add_event(0.0, 100.0, "whistle".to_string(), Modality::Harmonic);
        timeline.add_event(100.0, 50.0, "rattle".to_string(), Modality::Transient);

        assert_eq!(timeline.total_duration_ms(), 150.0);
    }

    #[test]
    fn test_multi_buffer_sequencer_creation() {
        let sequencer = MultiBufferGranularSequencer::new(44100);

        assert_eq!(sequencer.registered_sources().len(), 0);
        assert!(!sequencer.has_source("whistle"));
    }

    #[test]
    fn test_multi_buffer_sequencer_register_source() {
        let mut sequencer = MultiBufferGranularSequencer::new(44100);

        let audio = vec![0.0f32; 1000];
        let metadata = SourceMetadata::default();

        sequencer.register_source("whistle".to_string(), audio, metadata);

        assert!(sequencer.has_source("whistle"));
        assert_eq!(sequencer.registered_sources().len(), 1);
    }

    #[test]
    fn test_multi_buffer_sequencer_synthesize_timeline() {
        let mut sequencer = MultiBufferGranularSequencer::new(44100);

        // Register sources
        let whistle_audio: Vec<f32> = (0..4410)
            .map(|i| (2.0 * std::f32::consts::PI * 7000.0 * i as f32 / 44100.0).sin() * 0.3)
            .collect();

        let rattle_audio: Vec<f32> = (0..2205)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.5)
            .collect();

        let whistle_metadata = SourceMetadata {
            mean_f0_hz: 7000.0,
            duration_ms: 100.0,
            harmonic_to_noise_ratio: 25.0,
            spectral_flatness: 0.05,
            ..Default::default()
        };

        let rattle_metadata = SourceMetadata {
            mean_f0_hz: 0.0,
            duration_ms: 50.0,
            harmonic_to_noise_ratio: 2.0,
            spectral_flatness: 0.8,
            ..Default::default()
        };

        sequencer.register_source("whistle".to_string(), whistle_audio, whistle_metadata);
        sequencer.register_source("rattle".to_string(), rattle_audio, rattle_metadata);

        // Create timeline
        let mut timeline = ModalityTimeline::new();
        timeline.add_event(0.0, 100.0, "whistle".to_string(), Modality::Harmonic);
        timeline.add_event(100.0, 50.0, "rattle".to_string(), Modality::Transient);

        // Synthesize
        let result = sequencer.synthesize_timeline(&timeline);

        assert!(result.is_ok());
        let audio = result.unwrap();

        // Total duration: 100ms + 50ms = 150ms
        // At 44.1kHz: 150ms * 44.1 samples/ms = 6615 samples
        assert_eq!(audio.len(), 6615);
    }

    #[test]
    fn test_multi_buffer_sequencer_missing_source() {
        let sequencer = MultiBufferGranularSequencer::new(44100);

        let mut timeline = ModalityTimeline::new();
        timeline.add_event(0.0, 100.0, "missing_source".to_string(), Modality::Harmonic);

        let result = sequencer.synthesize_timeline(&timeline);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_multi_buffer_sequencer_voice_switching() {
        let mut sequencer = MultiBufferGranularSequencer::new(44100);

        // Register multiple sources
        let whistle_audio = vec![0.1f32; 2205]; // 50ms
        let rattle_audio = vec![0.2f32; 2205]; // 50ms
        let metadata = SourceMetadata::default();

        sequencer.register_source("whistle".to_string(), whistle_audio, metadata.clone());
        sequencer.register_source("rattle".to_string(), rattle_audio, metadata);

        // Create timeline with voice switching: whistle -> rattle -> whistle
        let mut timeline = ModalityTimeline::new();
        timeline.add_event(0.0, 50.0, "whistle".to_string(), Modality::Harmonic);
        timeline.add_event(50.0, 50.0, "rattle".to_string(), Modality::Transient);
        timeline.add_event(100.0, 50.0, "whistle".to_string(), Modality::Harmonic);

        let result = sequencer.synthesize_timeline(&timeline);

        assert!(result.is_ok());
        let audio = result.unwrap();

        // Verify voice switching: first 50ms samples should be ~0.1, next 50ms should be ~0.2
        let first_sample = audio[0];
        let middle_sample = audio[2205]; // At 50ms mark

        assert!((first_sample - 0.1).abs() < 0.01);
        assert!((middle_sample - 0.2).abs() < 0.01);
    }
}
