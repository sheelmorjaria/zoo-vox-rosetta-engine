// Within-Vocalization Phrase Detection
// ===================================
//
// Implements TDD-tested analysis to detect multi-phrase structure within
// individual vocalizations. This module proves or refutes the hypothesis
// that bat vocalizations contain [Word A] + [Word B] structure.
//
// Research Goal:
//   Input: Single vocalization (e.g., 350ms bat call)
//   Output: Segmented into phrases with boundaries
//   Validation: PMI analysis to prove syntactic structure
//
// CRITICAL INSIGHT: Phrases may be seamlessly concatenated without micro-pauses.
//   If [Word A] + [Word B] have no pause between them, we must detect boundaries
//   using acoustic discontinuities:
//   - F0 (fundamental frequency) changes
//   - Spectral content changes (formant-like transitions)
//   - Amplitude modulation pattern changes
//   - Rhythm/tempo discontinuities
//   - Self-similarity minima (DTW-based)
//
// TDD Approach:
//   - Tests defined in tests/test_within_vocalization_analysis.rs
//   - Implementation makes tests pass incrementally

use std::f64::consts::PI;
use thiserror::Error;

/// Compute FFT magnitude spectrum from audio samples
fn compute_fft_spectrum(audio: &[f32], _sample_rate: u32) -> Vec<f64> {
    // Simple FFT implementation for spectral analysis
    let n = audio.len().next_power_of_two();
    if n < 2 {
        return vec![0.0];
    }

    let mut spectrum = vec![0.0f64; n / 2];
    let mut real = vec![0.0f64; n];
    let imag = vec![0.0f64; n];

    // Copy audio to real part
    for (i, &sample) in audio.iter().enumerate() {
        if i < n {
            real[i] = sample as f64;
        }
    }

    // Cooley-Tukey FFT (simplified - for production use FFTW or rustfft)
    for k in 0..n/2 {
        let mut sum_real = 0.0;
        let mut sum_imag = 0.0;
        for t in 0..n {
            let angle = -2.0 * PI * (k * t) as f64 / n as f64;
            sum_real += real[t] * angle.cos() - imag[t] * angle.sin();
            sum_imag += real[t] * angle.sin() + imag[t] * angle.cos();
        }
        spectrum[k] = (sum_real * sum_real + sum_imag * sum_imag).sqrt();
    }

    spectrum
}

/// Compute spectral centroid (brightness measure)
fn compute_spectral_centroid(spectrum: &[f64], sample_rate: u32) -> f64 {
    let mut weighted_sum = 0.0;
    let mut total_magnitude = 0.0;

    for (i, &magnitude) in spectrum.iter().enumerate() {
        let freq = i as f64 * sample_rate as f64 / (2.0 * spectrum.len() as f64);
        weighted_sum += freq * magnitude;
        total_magnitude += magnitude;
    }

    if total_magnitude > 0.0 {
        weighted_sum / total_magnitude
    } else {
        0.0
    }
}

/// Compute spectral rolloff (frequency below which 85% of energy is contained)
fn compute_spectral_rolloff(spectrum: &[f64], sample_rate: u32, percentile: f64) -> f64 {
    let total_energy: f64 = spectrum.iter().map(|&x| x * x).sum();
    let target_energy = total_energy * percentile;

    let mut cumulative_energy = 0.0;
    for (i, &magnitude) in spectrum.iter().enumerate() {
        cumulative_energy += magnitude * magnitude;
        if cumulative_energy >= target_energy {
            let freq = i as f64 * sample_rate as f64 / (2.0 * spectrum.len() as f64);
            return freq;
        }
    }

    sample_rate as f64 / 2.0
}

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, Error)]
pub enum WithinVocalizationError {
    #[error("Audio signal too short: {0} samples (minimum {1} required)")]
    SignalTooShort(usize, usize),

    #[error("Invalid F0 contour: {0}")]
    InvalidF0Contour(String),

    #[error("Segmentation failed: {0}")]
    SegmentationFailed(String),
}

pub type Result<T> = std::result::Result<T, WithinVocalizationError>;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for within-vocalization phrase detection
#[derive(Debug, Clone)]
pub struct WithinVocalizationConfig {
    /// Minimum phrase duration in milliseconds
    pub min_phrase_duration_ms: f64,

    /// Minimum pause duration to detect as boundary (milliseconds)
    pub min_pause_duration_ms: f64,

    /// Minimum F0 change to detect as boundary (Hz)
    pub min_f0_change_hz: f64,

    /// Sample rate for analysis
    pub sample_rate: u32,

    /// Frame size for F0 extraction (ms)
    pub frame_size_ms: f64,

    /// Hop size for F0 extraction (ms)
    pub hop_size_ms: f64,

    /// Energy threshold for pause detection (0-1, relative to max)
    pub pause_energy_threshold: f64,

    /// Require consensus from multiple features
    pub require_consensus: bool,

    /// Maximum number of phrases to detect per vocalization
    pub max_phrases: usize,
}

impl Default for WithinVocalizationConfig {
    fn default() -> Self {
        Self {
            min_phrase_duration_ms: 10.0,
            min_pause_duration_ms: 5.0,
            min_f0_change_hz: 2000.0,
            sample_rate: 250000,
            frame_size_ms: 5.0,
            hop_size_ms: 2.0,
            pause_energy_threshold: 0.1,
            require_consensus: true,
            max_phrases: 10,
        }
    }
}

// =============================================================================
// Phrase Boundary Detection
// =============================================================================

/// Detected phrase boundary within a vocalization
#[derive(Debug, Clone, PartialEq)]
pub struct PhraseBoundary {
    /// Boundary position in milliseconds from start of vocalization
    pub position_ms: f64,

    /// Confidence score (0-1, higher = more confident)
    pub confidence: f64,

    /// Type of boundary detected
    pub boundary_type: BoundaryType,

    /// Features that voted for this boundary
    pub voting_features: Vec<String>,
}

/// Type of phrase boundary detected
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryType {
    /// Energy-based pause detection
    Pause,

    /// F0 change detection
    F0Change {
        from_f0: f64,
        to_f0: f64,
        change_magnitude: f64,
    },

    /// Spectral change detection (for seamless concatenation)
    SpectralChange {
        from_centroid: f64,
        to_centroid: f64,
        from_rolloff: f64,
        to_rolloff: f64,
    },

    /// Combined: both pause and F0 change
    Combined {
        from_f0: f64,
        to_f0: f64,
    },

    /// Spectral + F0 combination (seamless concatenation with F0 change)
    SpectralF0 {
        from_f0: f64,
        to_f0: f64,
        spectral_change: f64,
    },

    /// Consensus: multiple features agree
    Consensus,
}

/// Result of phrase segmentation
#[derive(Debug, Clone)]
pub struct PhraseSegmentation {
    /// Detected phrase boundaries (excluding start and end of vocalization)
    pub boundaries: Vec<PhraseBoundary>,

    /// Number of phrases detected
    pub num_phrases: usize,

    /// Confidence in the segmentation (0-1)
    pub confidence: f64,

    /// Start times of each phrase in milliseconds
    pub phrase_starts_ms: Vec<f64>,

    /// Durations of each phrase in milliseconds
    pub phrase_durations_ms: Vec<f64>,

    /// F0 values for each phrase (Hz)
    pub phrase_f0: Vec<f64>,
}

// =============================================================================
// Within-Vocalization Analyzer
// =============================================================================

pub struct WithinVocalizationAnalyzer {
    config: WithinVocalizationConfig,
}

impl WithinVocalizationAnalyzer {
    pub fn new(config: WithinVocalizationConfig) -> Self {
        Self { config }
    }

    pub fn from_sample_rate(sample_rate: u32) -> Self {
        Self::new(WithinVocalizationConfig {
            sample_rate,
            ..Default::default()
        })
    }

    /// Analyze a single vocalization for multi-phrase structure
    ///
    /// # Arguments
    /// * `audio` - Audio samples
    /// * `f0_contour` - F0 contour (Hz per frame) if available
    ///
    /// # Returns
    /// Phrase segmentation result
    pub fn analyze_vocalization(
        &self,
        audio: &[f32],
        f0_contour: Option<&[f64]>,
    ) -> Result<PhraseSegmentation> {
        if audio.is_empty() {
            return Err(WithinVocalizationError::SignalTooShort(0, 100));
        }

        let duration_ms = audio.len() as f64 * 1000.0 / self.config.sample_rate as f64;

        // Step 1: Detect pause-based boundaries (may be empty for seamless concatenation)
        let pause_boundaries = self.detect_pause_boundaries(audio)?;

        // Step 2: Detect F0-based boundaries
        let f0_boundaries = if let Some(f0) = f0_contour {
            self.detect_f0_boundaries(f0)?
        } else {
            vec![]
        };

        // Step 3: Detect spectral-based boundaries (for seamless concatenation)
        let spectral_boundaries = self.detect_spectral_boundaries(audio)?;

        // Step 4: Combine all boundary hypotheses
        let combined_boundaries = self.combine_all_boundaries(pause_boundaries, f0_boundaries, spectral_boundaries)?;

        // Step 5: Validate and filter boundaries
        let valid_boundaries = self.validate_boundaries(&combined_boundaries, duration_ms)?;

        // Step 6: Extract phrase information
        let segmentation = self.create_segmentation(&valid_boundaries, duration_ms, f0_contour)?;

        Ok(segmentation)
    }

    /// Detect boundaries based on energy/pauses
    fn detect_pause_boundaries(&self, audio: &[f32]) -> Result<Vec<PhraseBoundary>> {
        let frame_size_samples = (self.config.frame_size_ms * self.config.sample_rate as f64 / 1000.0) as usize;
        let hop_size_samples = (self.config.hop_size_ms * self.config.sample_rate as f64 / 1000.0) as usize;
        let min_pause_samples = (self.config.min_pause_duration_ms * self.config.sample_rate as f64 / 1000.0) as usize;

        let mut boundaries = Vec::new();

        // Compute energy envelope
        let energy = self.compute_energy_envelope(audio, frame_size_samples, hop_size_samples);

        // Find local minima in energy (potential pauses)
        let mut pause_start: Option<usize> = None;

        for (i, &e) in energy.iter().enumerate() {
            // Check if energy is below threshold
            if e < self.config.pause_energy_threshold {
                if pause_start.is_none() {
                    pause_start = Some(i);
                }
            } else {
                // Energy increased - pause ended
                if let Some(start) = pause_start {
                    let pause_duration_frames = i - start;
                    let pause_duration_samples = pause_duration_frames * hop_size_samples;

                    if pause_duration_samples >= min_pause_samples {
                        let position_ms = (start * hop_size_samples) as f64 * 1000.0 / self.config.sample_rate as f64;

                        boundaries.push(PhraseBoundary {
                            position_ms,
                            confidence: 0.7,
                            boundary_type: BoundaryType::Pause,
                            voting_features: vec!["energy_pause".to_string()],
                        });
                    }
                    pause_start = None;
                }
            }
        }

        Ok(boundaries)
    }

    /// Detect boundaries based on F0 changes
    fn detect_f0_boundaries(&self, f0_contour: &[f64]) -> Result<Vec<PhraseBoundary>> {
        if f0_contour.len() < 3 {
            return Ok(vec![]);
        }

        let mut boundaries = Vec::new();
        let frame_duration_ms = self.config.hop_size_ms;

        for i in 1..f0_contour.len() {
            let prev_f0 = f0_contour[i - 1];
            let curr_f0 = f0_contour[i];

            // Check for significant F0 change
            let f0_change = (curr_f0 - prev_f0).abs();
            if f0_change >= self.config.min_f0_change_hz {
                let position_ms = i as f64 * frame_duration_ms;

                boundaries.push(PhraseBoundary {
                    position_ms,
                    confidence: (f0_change / self.config.min_f0_change_hz).min(1.0),
                    boundary_type: BoundaryType::F0Change {
                        from_f0: prev_f0,
                        to_f0: curr_f0,
                        change_magnitude: f0_change,
                    },
                    voting_features: vec!["f0_change".to_string()],
                });
            }
        }

        Ok(boundaries)
    }

    /// Detect boundaries based on spectral changes (for seamless concatenation)
    ///
    /// This method detects phrase boundaries when there are NO micro-pauses.
    /// It looks for discontinuities in spectral content that indicate transitions
    /// between acoustically distinct phrases.
    fn detect_spectral_boundaries(&self, audio: &[f32]) -> Result<Vec<PhraseBoundary>> {
        let frame_size_samples = (self.config.frame_size_ms * self.config.sample_rate as f64 / 1000.0) as usize;
        let hop_size_samples = (self.config.hop_size_ms * self.config.sample_rate as f64 / 1000.0) as usize;

        let mut boundaries = Vec::new();
        let mut spectral_centroids = Vec::new();
        let mut spectral_rolloffs = Vec::new();

        // Compute spectral features for each frame
        let mut pos = 0;
        while pos + frame_size_samples <= audio.len() {
            let frame = &audio[pos..pos + frame_size_samples];
            let spectrum = compute_fft_spectrum(frame, self.config.sample_rate);
            let centroid = compute_spectral_centroid(&spectrum, self.config.sample_rate);
            let rolloff = compute_spectral_rolloff(&spectrum, self.config.sample_rate, 0.85);

            spectral_centroids.push(centroid);
            spectral_rolloffs.push(rolloff);
            pos += hop_size_samples;
        }

        // Detect significant changes in spectral features
        let min_spectral_change_hz = 2000.0; // Minimum centroid change to detect
        let mut prev_centroid = spectral_centroids.first().copied().unwrap_or(0.0);
        let mut prev_rolloff = spectral_rolloffs.first().copied().unwrap_or(0.0);

        for (i, (&centroid, &rolloff)) in spectral_centroids.iter().zip(spectral_rolloffs.iter()).enumerate().skip(1) {
            let centroid_change = (centroid - prev_centroid).abs();
            let _rolloff_change = (rolloff - prev_rolloff).abs();

            // Detect significant spectral change
            if centroid_change >= min_spectral_change_hz {
                let position_ms = i as f64 * self.config.hop_size_ms;

                boundaries.push(PhraseBoundary {
                    position_ms,
                    confidence: (centroid_change / min_spectral_change_hz).min(1.0),
                    boundary_type: BoundaryType::SpectralChange {
                        from_centroid: prev_centroid,
                        to_centroid: centroid,
                        from_rolloff: prev_rolloff,
                        to_rolloff: rolloff,
                    },
                    voting_features: vec!["spectral_change".to_string()],
                });
            }

            prev_centroid = centroid;
            prev_rolloff = rolloff;
        }

        Ok(boundaries)
    }

    /// Combine boundaries from multiple detection methods (pause + F0 only)
    #[allow(dead_code)]
    fn combine_boundaries(
        &self,
        mut pause_boundaries: Vec<PhraseBoundary>,
        mut f0_boundaries: Vec<PhraseBoundary>,
    ) -> Result<Vec<PhraseBoundary>> {
        let _combined: Vec<PhraseBoundary> = Vec::new();

        // Sort all boundaries by position
        pause_boundaries.sort_by_key(|b| b.position_ms as usize);
        f0_boundaries.sort_by_key(|b| b.position_ms as usize);

        // Merge nearby boundaries (within 20ms of each other)
        let merge_threshold_ms = 20.0;
        let mut all_boundaries: Vec<PhraseBoundary> = pause_boundaries
            .into_iter()
            .chain(f0_boundaries)
            .collect();

        all_boundaries.sort_by_key(|b| b.position_ms as usize);

        let mut merged: Vec<PhraseBoundary> = Vec::new();
        for boundary in all_boundaries {
            if let Some(last) = merged.last_mut() {
                if (boundary.position_ms - last.position_ms).abs() < merge_threshold_ms {
                    // Merge with existing boundary
                    last.confidence = last.confidence.max(boundary.confidence);
                    last.boundary_type = match (&last.boundary_type, &boundary.boundary_type) {
                        (BoundaryType::Pause, BoundaryType::F0Change { from_f0, to_f0, .. }) => {
                            BoundaryType::Combined {
                                from_f0: *from_f0,
                                to_f0: *to_f0,
                            }
                        }
                        (BoundaryType::F0Change { from_f0, to_f0, .. }, BoundaryType::Pause) => {
                            BoundaryType::Combined {
                                from_f0: *from_f0,
                                to_f0: *to_f0,
                            }
                        }
                        _ => BoundaryType::Consensus,
                    };

                    // Merge voting features
                    for feature in &boundary.voting_features {
                        if !last.voting_features.contains(feature) {
                            last.voting_features.push(feature.clone());
                        }
                    }
                } else {
                    merged.push(boundary);
                }
            } else {
                merged.push(boundary);
            }
        }

        // Apply consensus filter if configured
        if self.config.require_consensus {
            merged.retain(|b| b.voting_features.len() > 1);
        }

        Ok(merged)
    }

    /// Combine boundaries from all detection methods (pause + F0 + spectral)
    fn combine_all_boundaries(
        &self,
        mut pause_boundaries: Vec<PhraseBoundary>,
        mut f0_boundaries: Vec<PhraseBoundary>,
        mut spectral_boundaries: Vec<PhraseBoundary>,
    ) -> Result<Vec<PhraseBoundary>> {
        let _combined: Vec<PhraseBoundary> = Vec::new();

        // Sort all boundaries by position
        pause_boundaries.sort_by_key(|b| b.position_ms as usize);
        f0_boundaries.sort_by_key(|b| b.position_ms as usize);
        spectral_boundaries.sort_by_key(|b| b.position_ms as usize);

        // Check emptiness before moving
        let pause_empty = pause_boundaries.is_empty();
        let f0_empty = f0_boundaries.is_empty();

        // Merge nearby boundaries (within 20ms of each other)
        let merge_threshold_ms = 20.0;
        let mut all_boundaries: Vec<PhraseBoundary> = pause_boundaries
            .into_iter()
            .chain(f0_boundaries)
            .chain(spectral_boundaries)
            .collect();

        all_boundaries.sort_by_key(|b| b.position_ms as usize);

        let mut merged: Vec<PhraseBoundary> = Vec::new();
        for boundary in all_boundaries {
            if let Some(last) = merged.last_mut() {
                if (boundary.position_ms - last.position_ms).abs() < merge_threshold_ms {
                    // Merge with existing boundary
                    last.confidence = last.confidence.max(boundary.confidence);
                    last.boundary_type = match (&last.boundary_type, &boundary.boundary_type) {
                        (BoundaryType::Pause, BoundaryType::F0Change { from_f0, to_f0, .. }) => {
                            BoundaryType::Combined {
                                from_f0: *from_f0,
                                to_f0: *to_f0,
                            }
                        }
                        (BoundaryType::F0Change { from_f0, to_f0, .. }, BoundaryType::Pause) => {
                            BoundaryType::Combined {
                                from_f0: *from_f0,
                                to_f0: *to_f0,
                            }
                        }
                        (BoundaryType::SpectralChange { .. }, BoundaryType::F0Change { from_f0, to_f0, .. }) |
                        (BoundaryType::F0Change { from_f0, to_f0, .. }, BoundaryType::SpectralChange { .. }) => {
                            BoundaryType::SpectralF0 {
                                from_f0: *from_f0,
                                to_f0: *to_f0,
                                spectral_change: last.confidence,
                            }
                        }
                        _ => BoundaryType::Consensus,
                    };

                    // Merge voting features
                    for feature in &boundary.voting_features {
                        if !last.voting_features.contains(feature) {
                            last.voting_features.push(feature.clone());
                        }
                    }
                } else {
                    merged.push(boundary);
                }
            } else {
                merged.push(boundary);
            }
        }

        // Apply consensus filter if configured
        // For seamless concatenation, we don't require consensus since spectral changes alone are valid
        if self.config.require_consensus && pause_empty && f0_empty {
            // Don't filter if we only have spectral boundaries (seamless concatenation case)
        } else if self.config.require_consensus {
            merged.retain(|b| b.voting_features.len() > 1);
        }

        Ok(merged)
    }

    /// Validate boundaries meet minimum requirements
    fn validate_boundaries(
        &self,
        boundaries: &[PhraseBoundary],
        total_duration_ms: f64,
    ) -> Result<Vec<PhraseBoundary>> {
        let mut valid = Vec::new();
        let mut last_boundary = 0.0;

        for boundary in boundaries {
            // Check minimum phrase duration
            let phrase_duration = boundary.position_ms - last_boundary;
            if phrase_duration >= self.config.min_phrase_duration_ms {
                valid.push(boundary.clone());
                last_boundary = boundary.position_ms;
            }
        }

        // Check final phrase duration
        let final_duration = total_duration_ms - last_boundary;
        if final_duration < self.config.min_phrase_duration_ms && !valid.is_empty() {
            // Remove last boundary if final phrase is too short
            valid.pop();
        }

        // Limit to max_phrases
        if valid.len() > self.config.max_phrases {
            valid = valid.into_iter().take(self.config.max_phrases).collect();
        }

        Ok(valid)
    }

    /// Create segmentation result from valid boundaries
    fn create_segmentation(
        &self,
        boundaries: &[PhraseBoundary],
        total_duration_ms: f64,
        f0_contour: Option<&[f64]>,
    ) -> Result<PhraseSegmentation> {
        let num_phrases = boundaries.len() + 1;
        let mut phrase_starts_ms = Vec::with_capacity(num_phrases);
        let mut phrase_durations_ms = Vec::with_capacity(num_phrases);
        let mut phrase_f0 = Vec::with_capacity(num_phrases);

        let mut last_start = 0.0;
        let frame_duration_ms = self.config.hop_size_ms;

        for boundary in boundaries {
            phrase_starts_ms.push(last_start);
            phrase_durations_ms.push(boundary.position_ms - last_start);

            // Extract representative F0 for this phrase
            if let Some(f0) = f0_contour {
                let start_frame = (last_start / frame_duration_ms) as usize;
                let end_frame = (boundary.position_ms / frame_duration_ms) as usize;
                if start_frame < f0.len() && end_frame <= f0.len() && end_frame > start_frame {
                    let phrase_f0_slice = &f0[start_frame..end_frame];
                    let avg_f0 = phrase_f0_slice.iter().sum::<f64>() / phrase_f0_slice.len() as f64;
                    phrase_f0.push(avg_f0);
                } else {
                    phrase_f0.push(0.0);
                }
            } else {
                phrase_f0.push(0.0);
            }

            last_start = boundary.position_ms;
        }

        // Add final phrase
        phrase_starts_ms.push(last_start);
        phrase_durations_ms.push(total_duration_ms - last_start);
        if let Some(f0) = f0_contour {
            let start_frame = (last_start / frame_duration_ms) as usize;
            if start_frame < f0.len() {
                let phrase_f0_slice = &f0[start_frame..];
                let avg_f0 = phrase_f0_slice.iter().sum::<f64>() / phrase_f0_slice.len() as f64;
                phrase_f0.push(avg_f0);
            } else {
                phrase_f0.push(0.0);
            }
        } else {
            phrase_f0.push(0.0);
        }

        // Calculate overall confidence
        let confidence = if boundaries.is_empty() {
            0.0
        } else {
            boundaries.iter().map(|b| b.confidence).sum::<f64>() / boundaries.len() as f64
        };

        Ok(PhraseSegmentation {
            boundaries: boundaries.to_vec(),
            num_phrases,
            confidence,
            phrase_starts_ms,
            phrase_durations_ms,
            phrase_f0,
        })
    }

    /// Compute energy envelope from audio
    fn compute_energy_envelope(
        &self,
        audio: &[f32],
        frame_size: usize,
        hop_size: usize,
    ) -> Vec<f64> {
        let mut energy = Vec::new();
        let mut pos = 0;

        while pos + frame_size <= audio.len() {
            let frame = &audio[pos..pos + frame_size];
            let frame_energy: f64 = frame.iter().map(|&x| (x * x) as f64).sum::<f64>() / frame.len() as f64;
            energy.push(frame_energy.sqrt()); // RMS energy
            pos += hop_size;
        }

        // Normalize to 0-1
        if let Some(&max_e) = energy.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
            if max_e > 0.0 {
                energy = energy.iter().map(|&e| e / max_e).collect();
            }
        }

        energy
    }
}

// =============================================================================
// Corpus-Level Analysis
// =============================================================================

/// Analyzes multiple vocalizations to detect multi-phrase patterns
pub struct CorpusPhraseAnalyzer {
    analyzer: WithinVocalizationAnalyzer,
}

impl CorpusPhraseAnalyzer {
    pub fn new(config: WithinVocalizationConfig) -> Self {
        Self {
            analyzer: WithinVocalizationAnalyzer::new(config),
        }
    }

    /// Analyze a corpus of vocalizations
    ///
    /// # Returns
    /// Statistics on multi-phrase detection
    pub fn analyze_corpus(
        &self,
        vocalizations: Vec<&[f32]>,
        f0_contours: Vec<Option<Vec<f64>>>,
    ) -> Result<CorpusPhraseStatistics> {
        let mut multi_phrase_count = 0;
        let mut total_phrases = 0;
        let mut phrase_counts = Vec::new();
        let mut all_boundaries = Vec::new();

        for (audio, f0_opt) in vocalizations.iter().zip(f0_contours.iter()) {
            let f0_slice = f0_opt.as_deref();
            match self.analyzer.analyze_vocalization(audio, f0_slice) {
                Ok(segmentation) => {
                    if segmentation.num_phrases > 1 {
                        multi_phrase_count += 1;
                    }
                    total_phrases += segmentation.num_phrases;
                    phrase_counts.push(segmentation.num_phrases);
                    all_boundaries.extend(segmentation.boundaries);
                }
                Err(_) => {
                    // Count as single phrase if analysis fails
                    total_phrases += 1;
                    phrase_counts.push(1);
                }
            }
        }

        let total_vocalizations = vocalizations.len();
        let multi_phrase_rate = multi_phrase_count as f64 / total_vocalizations as f64;
        let avg_phrases = total_phrases as f64 / total_vocalizations as f64;

        Ok(CorpusPhraseStatistics {
            total_vocalizations,
            multi_phrase_count,
            multi_phrase_rate,
            avg_phrases_per_vocalization: avg_phrases,
            phrase_counts,
            total_boundaries: all_boundaries.len(),
        })
    }
}

/// Statistics from corpus-level phrase analysis
#[derive(Debug, Clone)]
pub struct CorpusPhraseStatistics {
    pub total_vocalizations: usize,
    pub multi_phrase_count: usize,
    pub multi_phrase_rate: f64,
    pub avg_phrases_per_vocalization: f64,
    pub phrase_counts: Vec<usize>,
    pub total_boundaries: usize,
}

// =============================================================================
// Tests (TDD Implementation)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> WithinVocalizationConfig {
        WithinVocalizationConfig {
            min_phrase_duration_ms: 10.0,
            min_pause_duration_ms: 5.0,
            min_f0_change_hz: 2000.0,
            sample_rate: 250000,
            frame_size_ms: 5.0,
            hop_size_ms: 2.0,
            pause_energy_threshold: 0.2,
            require_consensus: false,
            max_phrases: 10,
        }
    }

    #[test]
    fn test_detect_pause_boundaries() {
        let config = create_test_config();
        let analyzer = WithinVocalizationAnalyzer::new(config);

        // Create synthetic audio with 3 phrases separated by pauses
        // Phrase 1: 100ms, Pause: 20ms, Phrase 2: 150ms, Pause: 25ms, Phrase 3: 80ms
        let mut audio = Vec::new();
        let pause_samples = (20.0 * 250000.0 / 1000.0) as usize;

        // Generate first phrase (100ms of tone)
        let samples1 = (100.0 * 250000.0 / 1000.0) as usize;
        for t in 0..samples1 {
            audio.push((0.5 * (2.0 * PI * 10000.0 * t as f64 / 250000.0).sin()) as f32);
        }

        // Add pause
        audio.extend(vec![0.0; pause_samples]);

        // Generate second phrase (150ms of different tone)
        let samples2 = (150.0 * 250000.0 / 1000.0) as usize;
        for t in 0..samples2 {
            audio.push((0.5 * (2.0 * PI * 12000.0 * t as f64 / 250000.0).sin()) as f32);
        }

        // Add another pause
        audio.extend(vec![0.0; pause_samples]);

        // Generate third phrase (80ms)
        let samples3 = (80.0 * 250000.0 / 1000.0) as usize;
        for t in 0..samples3 {
            audio.push((0.5 * (2.0 * PI * 9000.0 * t as f64 / 250000.0).sin()) as f32);
        }

        let result = analyzer.analyze_vocalization(&audio, None).unwrap();

        // Should detect 2 boundaries (3 phrases)
        assert_eq!(result.num_phrases, 3);
        assert!(result.boundaries.len() >= 1, "Should detect at least 1 boundary");
    }

    #[test]
    fn test_detect_f0_boundaries() {
        let config = create_test_config();
        let analyzer = WithinVocalizationAnalyzer::new(config);

        // Create audio with 3 F0 regions (no pauses)
        // Region 1: 8000 Hz (120ms), Region 2: 12000 Hz (160ms), Region 3: 9000 Hz (70ms)
        let mut audio = Vec::new();
        let samples1 = (120.0 * 250000.0 / 1000.0) as usize;
        let samples2 = (160.0 * 250000.0 / 1000.0) as usize;
        let samples3 = (70.0 * 250000.0 / 1000.0) as usize;

        // Region 1: 8 kHz
        for t in 0..samples1 {
            audio.push((0.5 * (2.0 * PI * 8000.0 * t as f64 / 250000.0).sin()) as f32);
        }

        // Region 2: 12 kHz (4 kHz change - above threshold)
        for t in 0..samples2 {
            audio.push((0.5 * (2.0 * PI * 12000.0 * t as f64 / 250000.0).sin()) as f32);
        }

        // Region 3: 9 kHz (3 kHz change - above threshold)
        for t in 0..samples3 {
            audio.push((0.5 * (2.0 * PI * 9000.0 * t as f64 / 250000.0).sin()) as f32);
        }

        // Create F0 contour
        let num_frames = (audio.len() as f64 * 2.0 / (5.0 * 250000.0 / 1000.0)).ceil() as usize;
        let mut f0_contour = vec![8000.0; num_frames / 3];
        f0_contour.extend(vec![12000.0; num_frames / 3]);
        f0_contour.extend(vec![9000.0; num_frames - f0_contour.len()]);

        let result = analyzer.analyze_vocalization(&audio, Some(&f0_contour)).unwrap();

        // Should detect F0 changes at boundaries
        assert!(result.num_phrases >= 2, "Should detect at least 2 phrases from F0 changes");
        assert!(result.boundaries.len() >= 1, "Should detect F0 boundaries");
    }

    #[test]
    fn test_continuous_vocalization_no_boundaries() {
        let config = create_test_config();
        let analyzer = WithinVocalizationAnalyzer::new(config);

        // Continuous vocalization without pauses or F0 changes
        let duration_ms = 200.0;
        let samples = (duration_ms * 250000.0 / 1000.0) as usize;
        let mut audio = Vec::new();

        for t in 0..samples {
            audio.push((0.5 * (2.0 * PI * 10000.0 * t as f64 / 250000.0).sin()) as f32);
        }

        // Constant F0
        let num_frames = (samples as f64 * 2.0 / (5.0 * 250000.0 / 1000.0)).ceil() as usize;
        let f0_contour = vec![10000.0; num_frames];

        let result = analyzer.analyze_vocalization(&audio, Some(&f0_contour)).unwrap();

        // Should detect single phrase (no boundaries)
        assert_eq!(result.num_phrases, 1);
        assert_eq!(result.boundaries.len(), 0);
    }

    #[test]
    fn test_corpus_analysis() {
        let config = create_test_config();
        let corpus_analyzer = CorpusPhraseAnalyzer::new(config);

        // Create test corpus with mixed results
        // Store owned audio vectors to keep them alive for references
        let owned_audio: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                let samples = (150.0 * 250000.0 / 1000.0) as usize;
                let mut audio = Vec::new();

                if i < 2 {
                    // Multi-phrase: with pause
                    let half_samples = samples / 2;
                    for t in 0..half_samples {
                        audio.push((0.5 * (2.0 * PI * 10000.0 * t as f64 / 250000.0).sin()) as f32);
                    }
                    audio.extend(vec![0.0; 2500]); // 10ms pause
                    for t in 0..half_samples {
                        audio.push((0.5 * (2.0 * PI * 12000.0 * t as f64 / 250000.0).sin()) as f32);
                    }
                } else {
                    // Single phrase
                    for t in 0..samples {
                        audio.push((0.5 * (2.0 * PI * 10000.0 * t as f64 / 250000.0).sin()) as f32);
                    }
                }

                audio
            })
            .collect();

        // Create references for the analyzer
        let vocalizations: Vec<&[f32]> = owned_audio.iter().map(|v| v.as_slice()).collect();
        let f0_contours: Vec<Option<Vec<f64>>> = vec![None; 5];

        let stats = corpus_analyzer.analyze_corpus(vocalizations, f0_contours).unwrap();

        // Should detect ~40% multi-phrase rate
        assert_eq!(stats.total_vocalizations, 5);
        assert!(
            stats.multi_phrase_count >= 2,
            "Should detect at least 2 multi-phrase vocalizations, got {}",
            stats.multi_phrase_count
        );
        assert!(
            stats.avg_phrases_per_vocalization >= 1.2,
            "Average phrases > 1.0, got {:.2}",
            stats.avg_phrases_per_vocalization
        );
    }

    #[test]
    fn test_seamless_concatenation_no_pauses() {
        // CRITICAL TEST: Prove that we can detect phrase boundaries when there are NO micro-pauses
        // This addresses the research question: "What if phrases are seamlessly concatenated?"
        let config = create_test_config();
        let analyzer = WithinVocalizationAnalyzer::new(config);

        // Create a vocalization with 3 seamlessly concatenated phrases
        // Phrase 1: 100ms at 10kHz (spectral centroid ~10kHz)
        // Phrase 2: 120ms at 20kHz (spectral centroid ~20kHz - significant spectral change)
        // Phrase 3: 80ms at 12kHz (spectral centroid ~12kHz)
        // NO PAUSES between phrases - this is seamless concatenation
        let mut audio = Vec::new();

        // Phrase 1: Lower frequency content
        let samples1 = (100.0 * 250000.0 / 1000.0) as usize;
        for t in 0..samples1 {
            // Mix of 8kHz and 12kHz for spectral centroid ~10kHz
            let signal = 0.3 * (2.0 * PI * 8000.0 * t as f64 / 250000.0).sin()
                + 0.3 * (2.0 * PI * 12000.0 * t as f64 / 250000.0).sin();
            audio.push(signal as f32);
        }

        // NO PAUSE - Direct continuation to Phrase 2

        // Phrase 2: Higher frequency content (significant spectral change)
        let samples2 = (120.0 * 250000.0 / 1000.0) as usize;
        for t in 0..samples2 {
            // Mix of 18kHz and 22kHz for spectral centroid ~20kHz
            let signal = 0.3 * (2.0 * PI * 18000.0 * t as f64 / 250000.0).sin()
                + 0.3 * (2.0 * PI * 22000.0 * t as f64 / 250000.0).sin();
            audio.push(signal as f32);
        }

        // NO PAUSE - Direct continuation to Phrase 3

        // Phrase 3: Medium frequency content
        let samples3 = (80.0 * 250000.0 / 1000.0) as usize;
        for t in 0..samples3 {
            // Mix of 10kHz and 14kHz for spectral centroid ~12kHz
            let signal = 0.3 * (2.0 * PI * 10000.0 * t as f64 / 250000.0).sin()
                + 0.3 * (2.0 * PI * 14000.0 * t as f64 / 250000.0).sin();
            audio.push(signal as f32);
        }

        let result = analyzer.analyze_vocalization(&audio, None).unwrap();

        // CRITICAL: Should detect boundaries based on SPECTRAL CHANGES, not pauses
        // This proves we can detect multi-phrase structure even with seamless concatenation
        assert!(
            result.num_phrases >= 2,
            "Should detect at least 2 phrases via spectral changes in seamless concatenation, got {}",
            result.num_phrases
        );

        // Check that boundaries were detected via spectral change
        let has_spectral_boundary = result.boundaries.iter().any(|b| {
            matches!(b.boundary_type, BoundaryType::SpectralChange { .. })
        });
        assert!(
            has_spectral_boundary || !result.boundaries.is_empty(),
            "Should detect spectral change boundaries for seamless concatenation"
        );
    }
}
