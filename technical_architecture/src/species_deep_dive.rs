//! Species-Specific Deep Dive Modules
//!
//! Provides specialized analysis modules for species with unique vocalization
//! characteristics that require domain-specific signal processing techniques.
//!
//! **Macaque Module**: Spectral derivative analysis for rapid FM sweeps
//! - Uses time-frequency derivative to track fast frequency modulations
//! - Essential for analyzing rapid call transitions in macaque vocalizations
//!
//! **Dolphin Module**: Bispectrum analysis for click/whistle interactions
//! - Detects quadratic phase coupling between frequency components
//! - Identifies non-linear interactions in dolphin echolocation and whistles

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// COMMON TYPES
// ============================================================================

/// Result of a species-specific analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesAnalysisResult {
    /// Species analyzed
    pub species: String,
    /// Analysis type performed
    pub analysis_type: String,
    /// Confidence in the analysis result
    pub confidence: f64,
    /// Extracted features
    pub features: HashMap<String, f64>,
    /// Detected events
    pub events: Vec<DetectedEvent>,
    /// Timestamp of analysis
    pub timestamp_ms: u64,
}

/// A detected event in the vocalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedEvent {
    /// Event type
    pub event_type: String,
    /// Start time in milliseconds
    pub start_ms: f64,
    /// Duration in milliseconds
    pub duration_ms: f64,
    /// Event strength/intensity
    pub intensity: f64,
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
}

// ============================================================================
// MACAQUE MODULE: Spectral Derivative Analysis
// ============================================================================

/// Macaque-specific analysis using spectral derivatives
///
/// Spectral derivative dF/dt measures how quickly frequency changes over time.
/// This is critical for macaque vocalizations which feature rapid FM sweeps
/// that convey emotional and contextual information.
#[derive(Debug, Clone)]
pub struct MacaqueSpectralDerivative {
    /// Sample rate
    sample_rate: u32,
    /// FFT size for spectral analysis
    fft_size: usize,
    /// Hop size between frames
    hop_size: usize,
    /// Minimum FM sweep rate to detect (Hz/ms)
    min_sweep_rate: f64,
    /// Maximum FM sweep rate (Hz/ms)
    max_sweep_rate: f64,
}

impl MacaqueSpectralDerivative {
    /// Create new spectral derivative analyzer
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            fft_size: 1024,
            hop_size: 256,
            min_sweep_rate: 10.0,
            max_sweep_rate: 5000.0,
        }
    }

    /// Configure parameters
    pub fn with_params(
        mut self,
        fft_size: usize,
        hop_size: usize,
        min_sweep_rate: f64,
        max_sweep_rate: f64,
    ) -> Self {
        self.fft_size = fft_size;
        self.hop_size = hop_size;
        self.min_sweep_rate = min_sweep_rate;
        self.max_sweep_rate = max_sweep_rate;
        self
    }

    /// Compute spectral derivative from spectrogram
    ///
    /// Returns the rate of frequency change at each time-frequency point
    pub fn compute_spectral_derivative(&self, spectrogram: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if spectrogram.is_empty() || spectrogram[0].is_empty() {
            return Vec::new();
        }

        let n_frames = spectrogram.len();
        let n_bins = spectrogram[0].len();
        let mut derivative = vec![vec![0.0; n_bins]; n_frames];

        // Compute time derivative at each frequency bin
        for frame_idx in 1..n_frames - 1 {
            for bin_idx in 0..n_bins {
                // Central difference for time derivative
                let dt =
                    (frame_idx as f64 * self.hop_size as f64) / self.sample_rate as f64 * 1000.0;
                let df = (spectrogram[frame_idx + 1][bin_idx]
                    - spectrogram[frame_idx - 1][bin_idx])
                    / (2.0 * dt);

                derivative[frame_idx][bin_idx] = df;
            }
        }

        derivative
    }

    /// Detect FM sweep events
    ///
    /// Finds regions where frequency is changing rapidly (FM sweeps)
    pub fn detect_fm_sweeps(
        &self,
        spectrogram: &[Vec<f64>],
        frequencies: &[f64],
    ) -> Vec<FmSweepEvent> {
        let derivative = self.compute_spectral_derivative(spectrogram);
        let mut sweeps = Vec::new();

        if derivative.is_empty() {
            return sweeps;
        }

        let hop_ms = (self.hop_size as f64 / self.sample_rate as f64) * 1000.0;
        let freq_resolution =
            frequencies.get(1).unwrap_or(&0.0) - frequencies.get(0).unwrap_or(&0.0);

        // Find peaks in derivative (rapid FM regions)
        let mut in_sweep = false;
        let mut sweep_start = 0;
        let mut sweep_direction = 0i32; // 1 = up, -1 = down
        let mut sweep_freq_start = 0.0f64;
        let mut sweep_freq_end = 0.0f64;
        let mut sweep_max_rate = 0.0f64;

        for (frame_idx, frame) in derivative.iter().enumerate() {
            // Find maximum derivative at this frame
            let (max_bin, &max_deriv) = frame
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));

            // Convert to sweep rate (Hz/ms)
            let sweep_rate = max_deriv.abs() * freq_resolution.abs();

            if sweep_rate >= self.min_sweep_rate && sweep_rate <= self.max_sweep_rate {
                if !in_sweep {
                    // Start new sweep
                    in_sweep = true;
                    sweep_start = frame_idx;
                    sweep_direction = if max_deriv > 0.0 { 1 } else { -1 };
                    sweep_freq_start = *frequencies.get(max_bin).unwrap_or(&0.0);
                    sweep_freq_end = sweep_freq_start;
                    sweep_max_rate = sweep_rate;
                } else {
                    // Continue sweep
                    sweep_freq_end = *frequencies.get(max_bin).unwrap_or(&0.0);
                    sweep_max_rate = sweep_max_rate.max(sweep_rate);
                }
            } else if in_sweep {
                // End sweep
                let duration_ms = (frame_idx - sweep_start) as f64 * hop_ms;
                if duration_ms > 5.0 {
                    sweeps.push(FmSweepEvent {
                        start_ms: sweep_start as f64 * hop_ms,
                        duration_ms,
                        start_freq_hz: sweep_freq_start,
                        end_freq_hz: sweep_freq_end,
                        sweep_rate_hz_ms: sweep_max_rate,
                        direction: if sweep_direction > 0 {
                            SweepDirection::Up
                        } else {
                            SweepDirection::Down
                        },
                        intensity: sweep_max_rate / self.max_sweep_rate,
                    });
                }
                in_sweep = false;
            }
        }

        // Handle sweep at end
        if in_sweep {
            let duration_ms = (derivative.len() - sweep_start) as f64 * hop_ms;
            if duration_ms > 5.0 {
                sweeps.push(FmSweepEvent {
                    start_ms: sweep_start as f64 * hop_ms,
                    duration_ms,
                    start_freq_hz: sweep_freq_start,
                    end_freq_hz: sweep_freq_end,
                    sweep_rate_hz_ms: sweep_max_rate,
                    direction: if sweep_direction > 0 {
                        SweepDirection::Up
                    } else {
                        SweepDirection::Down
                    },
                    intensity: sweep_max_rate / self.max_sweep_rate,
                });
            }
        }

        sweeps
    }

    /// Analyze macaque vocalization for FM characteristics
    pub fn analyze(
        &self,
        spectrogram: &[Vec<f64>],
        frequencies: &[f64],
        timestamp_ms: u64,
    ) -> SpeciesAnalysisResult {
        let sweeps = self.detect_fm_sweeps(spectrogram, frequencies);
        let derivative = self.compute_spectral_derivative(spectrogram);

        // Compute aggregate statistics
        let total_sweep_duration: f64 = sweeps.iter().map(|s| s.duration_ms).sum();
        let avg_sweep_rate = if !sweeps.is_empty() {
            sweeps.iter().map(|s| s.sweep_rate_hz_ms).sum::<f64>() / sweeps.len() as f64
        } else {
            0.0
        };

        let up_sweeps = sweeps
            .iter()
            .filter(|s| s.direction == SweepDirection::Up)
            .count();
        let down_sweeps = sweeps
            .iter()
            .filter(|s| s.direction == SweepDirection::Down)
            .count();

        let mut features = HashMap::new();
        features.insert("sweep_count".to_string(), sweeps.len() as f64);
        features.insert("total_sweep_duration_ms".to_string(), total_sweep_duration);
        features.insert("avg_sweep_rate_hz_ms".to_string(), avg_sweep_rate);
        features.insert("up_sweep_count".to_string(), up_sweeps as f64);
        features.insert("down_sweep_count".to_string(), down_sweeps as f64);
        features.insert(
            "sweep_ratio".to_string(),
            if down_sweeps > 0 {
                up_sweeps as f64 / down_sweeps as f64
            } else {
                up_sweeps as f64
            },
        );

        // Max derivative across all frames
        let max_deriv: f64 = derivative
            .iter()
            .flat_map(|f| f.iter())
            .map(|d| d.abs())
            .fold(0.0, |a, b| a.max(b));
        features.insert("max_spectral_derivative".to_string(), max_deriv);

        let events: Vec<DetectedEvent> = sweeps
            .iter()
            .map(|s| DetectedEvent {
                event_type: "fm_sweep".to_string(),
                start_ms: s.start_ms,
                duration_ms: s.duration_ms,
                intensity: s.intensity,
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("start_freq_hz".to_string(), s.start_freq_hz);
                    p.insert("end_freq_hz".to_string(), s.end_freq_hz);
                    p.insert("sweep_rate_hz_ms".to_string(), s.sweep_rate_hz_ms);
                    p
                },
            })
            .collect();

        SpeciesAnalysisResult {
            species: "macaque".to_string(),
            analysis_type: "spectral_derivative".to_string(),
            confidence: if sweeps.is_empty() { 0.3 } else { 0.85 },
            features,
            events,
            timestamp_ms,
        }
    }
}

/// FM Sweep direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SweepDirection {
    /// Frequency increasing (upsweep)
    Up,
    /// Frequency decreasing (downsweep)
    Down,
}

/// Detected FM sweep event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FmSweepEvent {
    /// Start time in milliseconds
    pub start_ms: f64,
    /// Duration in milliseconds
    pub duration_ms: f64,
    /// Starting frequency
    pub start_freq_hz: f64,
    /// Ending frequency
    pub end_freq_hz: f64,
    /// Sweep rate (Hz/ms)
    pub sweep_rate_hz_ms: f64,
    /// Sweep direction
    pub direction: SweepDirection,
    /// Intensity (0-1)
    pub intensity: f64,
}

// ============================================================================
// DOLPHIN MODULE: Bispectrum Analysis
// ============================================================================

/// Dolphin-specific analysis using bispectrum
///
/// The bispectrum B(f1, f2) measures quadratic phase coupling between
/// frequency components. This is essential for analyzing dolphin
/// echolocation clicks and whistle interactions.
///
/// Quadratic Phase Coupling (QPC):
/// When two frequency components f1 and f2 interact non-linearly,
/// they generate sum and difference frequencies with phase coherence.
/// The bispectrum detects this coherence.
#[derive(Debug, Clone)]
pub struct DolphinBispectrumAnalyzer {
    /// Sample rate
    sample_rate: u32,
    /// FFT size
    fft_size: usize,
    /// Maximum frequency to analyze
    max_freq: f64,
    /// QPC detection threshold
    qpc_threshold: f64,
}

impl DolphinBispectrumAnalyzer {
    /// Create new bispectrum analyzer
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            fft_size: 2048,
            max_freq: 150000.0, // Dolphins can produce up to 150kHz
            qpc_threshold: 0.3,
        }
    }

    /// Configure parameters
    pub fn with_params(mut self, fft_size: usize, max_freq: f64, qpc_threshold: f64) -> Self {
        self.fft_size = fft_size;
        self.max_freq = max_freq;
        self.qpc_threshold = qpc_threshold;
        self
    }

    /// Compute bispectrum from FFT
    ///
    /// B(f1, f2) = E[X(f1) * X(f2) * X*(f1+f2)]
    ///
    /// Where X(f) is the FFT at frequency f, and X* is the complex conjugate.
    /// For real signals, we use magnitude for simplicity.
    pub fn compute_bispectrum(
        &self,
        fft_magnitudes: &[f64],
        fft_phases: &[f64],
    ) -> BispectrumResult {
        let n = fft_magnitudes.len();
        let freq_resolution = self.sample_rate as f64 / (2 * n) as f64;

        // Bispectrum is computed for f1, f2 where f1 + f2 < fmax
        let max_bin = ((self.max_freq / freq_resolution) as usize).min(n / 2);

        let mut bispectrum_magnitude = vec![vec![0.0; max_bin]; max_bin];
        let mut bispectrum_phase = vec![vec![0.0; max_bin]; max_bin];

        // Compute bispectrum B(f1, f2)
        for f1_bin in 0..max_bin {
            for f2_bin in 0..(max_bin - f1_bin) {
                let f3_bin = f1_bin + f2_bin;

                if f3_bin < n {
                    // B(f1, f2) = |X(f1)| * |X(f2)| * |X(f3)| * exp(i*(phi1 + phi2 - phi3))
                    let m1 = fft_magnitudes[f1_bin];
                    let m2 = fft_magnitudes[f2_bin];
                    let m3 = fft_magnitudes[f3_bin];

                    let p1 = fft_phases.get(f1_bin).copied().unwrap_or(0.0);
                    let p2 = fft_phases.get(f2_bin).copied().unwrap_or(0.0);
                    let p3 = fft_phases.get(f3_bin).copied().unwrap_or(0.0);

                    // Bispectrum magnitude (simplified)
                    bispectrum_magnitude[f1_bin][f2_bin] = m1 * m2 * m3;

                    // Bispectrum phase (phi1 + phi2 - phi3)
                    bispectrum_phase[f1_bin][f2_bin] = p1 + p2 - p3;
                }
            }
        }

        BispectrumResult {
            magnitude: bispectrum_magnitude,
            phase: bispectrum_phase,
            freq_resolution,
            max_freq: self.max_freq,
        }
    }

    /// Detect Quadratic Phase Coupling (QPC)
    ///
    /// QPC occurs when the bispectrum magnitude is high and the biphase
    /// (phi1 + phi2 - phi3) is near zero (phase coherent).
    pub fn detect_qpc(&self, bispectrum: &BispectrumResult) -> Vec<QpcEvent> {
        let mut qpc_events = Vec::new();

        let n = bispectrum.magnitude.len();
        if n == 0 {
            return qpc_events;
        }

        // Find maximum for normalization
        let max_bispec: f64 = bispectrum
            .magnitude
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .fold(0.0_f64, |a, b| a.max(b));

        if max_bispec == 0.0 {
            return qpc_events;
        }

        // Threshold for QPC detection
        let threshold = max_bispec * self.qpc_threshold;

        for (f1_bin, row) in bispectrum.magnitude.iter().enumerate() {
            for (f2_bin, &magnitude) in row.iter().enumerate() {
                if magnitude > threshold {
                    let f1 = f1_bin as f64 * bispectrum.freq_resolution;
                    let f2 = f2_bin as f64 * bispectrum.freq_resolution;
                    let f3 = (f1_bin + f2_bin) as f64 * bispectrum.freq_resolution;

                    // Biphase coherence (closer to 0 mod 2pi = more coherent)
                    let biphase = bispectrum.phase[f1_bin][f2_bin];
                    let coherence = ((biphase % (2.0 * std::f64::consts::PI)).abs()
                        / std::f64::consts::PI)
                        .min(1.0);

                    qpc_events.push(QpcEvent {
                        f1_hz: f1,
                        f2_hz: f2,
                        f3_hz: f3,
                        bispectrum_magnitude: magnitude / max_bispec,
                        biphase_coherence: 1.0 - coherence,
                        coupling_type: self.classify_coupling(f1, f2, f3),
                    });
                }
            }
        }

        // Sort by magnitude
        qpc_events.sort_by(|a, b| {
            b.bispectrum_magnitude
                .partial_cmp(&a.bispectrum_magnitude)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top events
        qpc_events.into_iter().take(50).collect()
    }

    /// Classify the type of frequency coupling
    fn classify_coupling(&self, f1: f64, f2: f64, f3: f64) -> CouplingType {
        // Click-dominated: high frequency components
        // Whistle-dominated: low frequency components
        // Mixed: both high and low

        let click_threshold = 50000.0; // 50kHz
        let whistle_threshold = 20000.0; // 20kHz

        let has_click = f1 > click_threshold || f2 > click_threshold || f3 > click_threshold;
        let has_whistle =
            f1 < whistle_threshold || f2 < whistle_threshold || f3 < whistle_threshold;

        if has_click && has_whistle {
            CouplingType::ClickWhistleInteraction
        } else if has_click {
            CouplingType::ClickClickInteraction
        } else if has_whistle {
            CouplingType::WhistleWhistleInteraction
        } else {
            CouplingType::Unknown
        }
    }

    /// Analyze dolphin vocalization for bispectrum characteristics
    pub fn analyze(
        &self,
        fft_magnitudes: &[f64],
        fft_phases: &[f64],
        timestamp_ms: u64,
    ) -> SpeciesAnalysisResult {
        let bispectrum = self.compute_bispectrum(fft_magnitudes, fft_phases);
        let qpc_events = self.detect_qpc(&bispectrum);

        // Compute aggregate statistics
        let total_qpc = qpc_events.len();
        let avg_magnitude = if !qpc_events.is_empty() {
            qpc_events
                .iter()
                .map(|e| e.bispectrum_magnitude)
                .sum::<f64>()
                / qpc_events.len() as f64
        } else {
            0.0
        };
        let avg_coherence = if !qpc_events.is_empty() {
            qpc_events.iter().map(|e| e.biphase_coherence).sum::<f64>() / qpc_events.len() as f64
        } else {
            0.0
        };

        let click_interactions = qpc_events
            .iter()
            .filter(|e| e.coupling_type == CouplingType::ClickClickInteraction)
            .count();
        let whistle_interactions = qpc_events
            .iter()
            .filter(|e| e.coupling_type == CouplingType::WhistleWhistleInteraction)
            .count();
        let mixed_interactions = qpc_events
            .iter()
            .filter(|e| e.coupling_type == CouplingType::ClickWhistleInteraction)
            .count();

        let mut features = HashMap::new();
        features.insert("qpc_count".to_string(), total_qpc as f64);
        features.insert("avg_bispectrum_magnitude".to_string(), avg_magnitude);
        features.insert("avg_biphase_coherence".to_string(), avg_coherence);
        features.insert(
            "click_click_interactions".to_string(),
            click_interactions as f64,
        );
        features.insert(
            "whistle_whistle_interactions".to_string(),
            whistle_interactions as f64,
        );
        features.insert(
            "click_whistle_interactions".to_string(),
            mixed_interactions as f64,
        );

        // Bispectrum entropy (measure of distribution)
        let entropy = self.compute_bispectrum_entropy(&bispectrum);
        features.insert("bispectrum_entropy".to_string(), entropy);

        let events: Vec<DetectedEvent> = qpc_events
            .iter()
            .map(|e| DetectedEvent {
                event_type: "qpc".to_string(),
                start_ms: 0.0, // QPC is frequency-domain, no temporal location
                duration_ms: 0.0,
                intensity: e.bispectrum_magnitude,
                parameters: {
                    let mut p = HashMap::new();
                    p.insert("f1_hz".to_string(), e.f1_hz);
                    p.insert("f2_hz".to_string(), e.f2_hz);
                    p.insert("f3_hz".to_string(), e.f3_hz);
                    p.insert("biphase_coherence".to_string(), e.biphase_coherence);
                    p
                },
            })
            .collect();

        SpeciesAnalysisResult {
            species: "dolphin".to_string(),
            analysis_type: "bispectrum".to_string(),
            confidence: if total_qpc > 5 {
                0.9
            } else if total_qpc > 0 {
                0.6
            } else {
                0.3
            },
            features,
            events,
            timestamp_ms,
        }
    }

    /// Compute entropy of bispectrum magnitude distribution
    fn compute_bispectrum_entropy(&self, bispectrum: &BispectrumResult) -> f64 {
        let total: f64 = bispectrum.magnitude.iter().flat_map(|row| row.iter()).sum();

        if total == 0.0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for row in &bispectrum.magnitude {
            for &mag in row {
                if mag > 0.0 {
                    let p = mag / total;
                    entropy -= p * p.log2();
                }
            }
        }

        entropy
    }
}

/// Result of bispectrum computation
#[derive(Debug, Clone)]
pub struct BispectrumResult {
    /// Bispectrum magnitude B(f1, f2)
    pub magnitude: Vec<Vec<f64>>,
    /// Bispectrum phase (biphase)
    pub phase: Vec<Vec<f64>>,
    /// Frequency resolution in Hz
    pub freq_resolution: f64,
    /// Maximum frequency analyzed
    pub max_freq: f64,
}

/// Type of frequency coupling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouplingType {
    /// Interaction between two click components
    ClickClickInteraction,
    /// Interaction between two whistle components
    WhistleWhistleInteraction,
    /// Interaction between click and whistle components
    ClickWhistleInteraction,
    /// Unknown coupling type
    Unknown,
}

/// Quadratic Phase Coupling event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QpcEvent {
    /// First frequency
    pub f1_hz: f64,
    /// Second frequency
    pub f2_hz: f64,
    /// Sum frequency (f1 + f2)
    pub f3_hz: f64,
    /// Normalized bispectrum magnitude
    pub bispectrum_magnitude: f64,
    /// Biphase coherence (0-1, higher = more coherent)
    pub biphase_coherence: f64,
    /// Type of coupling
    pub coupling_type: CouplingType,
}

// ============================================================================
// SPECIES DEEP DIVE MANAGER
// ============================================================================

/// Manager for species-specific deep dive analyses
pub struct SpeciesDeepDiveManager {
    macaque_analyzer: MacaqueSpectralDerivative,
    dolphin_analyzer: DolphinBispectrumAnalyzer,
    sample_rate: u32,
}

impl SpeciesDeepDiveManager {
    /// Create new manager
    pub fn new(sample_rate: u32) -> Self {
        Self {
            macaque_analyzer: MacaqueSpectralDerivative::new(sample_rate),
            dolphin_analyzer: DolphinBispectrumAnalyzer::new(sample_rate),
            sample_rate,
        }
    }

    /// Get the macaque analyzer
    pub fn macaque(&self) -> &MacaqueSpectralDerivative {
        &self.macaque_analyzer
    }

    /// Get the dolphin analyzer
    pub fn dolphin(&self) -> &DolphinBispectrumAnalyzer {
        &self.dolphin_analyzer
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl Default for SpeciesDeepDiveManager {
    fn default() -> Self {
        Self::new(48000)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== MACAQUE TESTS ====================

    #[test]
    fn test_macaque_analyzer_creation() {
        let analyzer = MacaqueSpectralDerivative::new(48000);
        assert_eq!(analyzer.sample_rate, 48000);
        assert_eq!(analyzer.fft_size, 1024);
    }

    #[test]
    fn test_macaque_custom_params() {
        let analyzer = MacaqueSpectralDerivative::new(48000).with_params(2048, 512, 20.0, 3000.0);
        assert_eq!(analyzer.fft_size, 2048);
        assert_eq!(analyzer.hop_size, 512);
    }

    #[test]
    fn test_spectral_derivative_empty() {
        let analyzer = MacaqueSpectralDerivative::new(48000);
        let result = analyzer.compute_spectral_derivative(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_spectral_derivative_single_frame() {
        let analyzer = MacaqueSpectralDerivative::new(48000);
        let spectrogram = vec![vec![1.0, 2.0, 3.0]];
        let result = analyzer.compute_spectral_derivative(&spectrogram);
        // Single frame returns zero-filled result (no central difference possible)
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], vec![0.0, 0.0, 0.0]); // All zeros since no derivative computed
    }

    #[test]
    fn test_spectral_derivative_basic() {
        let analyzer = MacaqueSpectralDerivative::new(48000);
        let spectrogram = vec![
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0],
            vec![4.0, 4.0, 4.0],
            vec![5.0, 5.0, 5.0],
        ];
        let result = analyzer.compute_spectral_derivative(&spectrogram);

        // Should have derivative at middle frames
        assert!(!result.is_empty());
    }

    #[test]
    fn test_detect_fm_sweeps_empty() {
        let analyzer = MacaqueSpectralDerivative::new(48000);
        let frequencies = vec![1000.0, 2000.0, 3000.0];
        let sweeps = analyzer.detect_fm_sweeps(&[], &frequencies);
        assert!(sweeps.is_empty());
    }

    #[test]
    fn test_macaque_analyze() {
        let analyzer = MacaqueSpectralDerivative::new(48000);

        // Create spectrogram with some variation
        let spectrogram: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![1.0 + i as f64 * 0.1, 2.0, 3.0])
            .collect();
        let frequencies = vec![1000.0, 2000.0, 3000.0];

        let result = analyzer.analyze(&spectrogram, &frequencies, 0);

        assert_eq!(result.species, "macaque");
        assert_eq!(result.analysis_type, "spectral_derivative");
        assert!(result.features.contains_key("sweep_count"));
    }

    #[test]
    fn test_fm_sweep_event() {
        let sweep = FmSweepEvent {
            start_ms: 100.0,
            duration_ms: 50.0,
            start_freq_hz: 2000.0,
            end_freq_hz: 4000.0,
            sweep_rate_hz_ms: 40.0,
            direction: SweepDirection::Up,
            intensity: 0.8,
        };

        assert_eq!(sweep.direction, SweepDirection::Up);
        assert_eq!(sweep.end_freq_hz - sweep.start_freq_hz, 2000.0);
    }

    #[test]
    fn test_sweep_direction() {
        let up = SweepDirection::Up;
        let down = SweepDirection::Down;
        assert_ne!(up, down);
    }

    // ==================== DOLPHIN TESTS ====================

    #[test]
    fn test_dolphin_analyzer_creation() {
        let analyzer = DolphinBispectrumAnalyzer::new(192000);
        assert_eq!(analyzer.sample_rate, 192000);
        assert_eq!(analyzer.max_freq, 150000.0);
    }

    #[test]
    fn test_dolphin_custom_params() {
        let analyzer = DolphinBispectrumAnalyzer::new(192000).with_params(4096, 100000.0, 0.4);
        assert_eq!(analyzer.fft_size, 4096);
        assert_eq!(analyzer.max_freq, 100000.0);
        assert!((analyzer.qpc_threshold - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_bispectrum_empty() {
        let analyzer = DolphinBispectrumAnalyzer::new(192000);
        let result = analyzer.compute_bispectrum(&[], &[]);
        assert!(result.magnitude.is_empty() || result.magnitude[0].is_empty());
    }

    #[test]
    fn test_bispectrum_basic() {
        let analyzer = DolphinBispectrumAnalyzer::new(192000);

        // Simple FFT magnitudes and phases
        let magnitudes: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let phases: Vec<f64> = (0..256).map(|_| 0.0).collect();

        let result = analyzer.compute_bispectrum(&magnitudes, &phases);

        assert!(!result.magnitude.is_empty());
        assert!(result.freq_resolution > 0.0);
    }

    #[test]
    fn test_detect_qpc_empty() {
        let analyzer = DolphinBispectrumAnalyzer::new(192000);
        let bispectrum = BispectrumResult {
            magnitude: vec![],
            phase: vec![],
            freq_resolution: 750.0,
            max_freq: 150000.0,
        };

        let qpc = analyzer.detect_qpc(&bispectrum);
        assert!(qpc.is_empty());
    }

    #[test]
    fn test_detect_qpc_with_data() {
        let analyzer = DolphinBispectrumAnalyzer::new(192000);

        // Create bispectrum with some peaks
        let mut magnitude = vec![vec![0.0; 100]; 100];
        magnitude[10][20] = 1.0; // Peak at (f1, f2)
        magnitude[30][40] = 0.8;

        let bispectrum = BispectrumResult {
            magnitude,
            phase: vec![vec![0.0; 100]; 100],
            freq_resolution: 750.0,
            max_freq: 150000.0,
        };

        let qpc = analyzer.detect_qpc(&bispectrum);
        assert!(!qpc.is_empty());
    }

    #[test]
    fn test_dolphin_analyze() {
        let analyzer = DolphinBispectrumAnalyzer::new(192000);

        // Simple FFT data
        let magnitudes: Vec<f64> = (0..256)
            .map(|i| {
                if i < 50 {
                    0.5
                } else if i < 100 {
                    0.3
                } else {
                    0.1
                }
            })
            .collect();
        let phases: Vec<f64> = (0..256).map(|_| 0.0).collect();

        let result = analyzer.analyze(&magnitudes, &phases, 0);

        assert_eq!(result.species, "dolphin");
        assert_eq!(result.analysis_type, "bispectrum");
        assert!(result.features.contains_key("qpc_count"));
    }

    #[test]
    fn test_coupling_type_classification() {
        let analyzer = DolphinBispectrumAnalyzer::new(192000);

        assert_eq!(
            analyzer.classify_coupling(10000.0, 15000.0, 25000.0),
            CouplingType::WhistleWhistleInteraction
        );
        assert_eq!(
            analyzer.classify_coupling(60000.0, 70000.0, 130000.0),
            CouplingType::ClickClickInteraction
        );
        assert_eq!(
            analyzer.classify_coupling(10000.0, 60000.0, 70000.0),
            CouplingType::ClickWhistleInteraction
        );
    }

    #[test]
    fn test_qpc_event() {
        let qpc = QpcEvent {
            f1_hz: 10000.0,
            f2_hz: 20000.0,
            f3_hz: 30000.0,
            bispectrum_magnitude: 0.75,
            biphase_coherence: 0.9,
            coupling_type: CouplingType::WhistleWhistleInteraction,
        };

        assert_eq!(qpc.f3_hz, qpc.f1_hz + qpc.f2_hz);
        assert_eq!(qpc.coupling_type, CouplingType::WhistleWhistleInteraction);
    }

    #[test]
    fn test_bispectrum_entropy() {
        let analyzer = DolphinBispectrumAnalyzer::new(192000);

        // Uniform distribution
        let uniform = BispectrumResult {
            magnitude: vec![vec![0.5; 10]; 10],
            phase: vec![vec![0.0; 10]; 10],
            freq_resolution: 750.0,
            max_freq: 150000.0,
        };

        let entropy = analyzer.compute_bispectrum_entropy(&uniform);
        assert!(entropy > 0.0);
    }

    // ==================== MANAGER TESTS ====================

    #[test]
    fn test_manager_creation() {
        let manager = SpeciesDeepDiveManager::new(192000);
        assert_eq!(manager.sample_rate(), 192000);
    }

    #[test]
    fn test_manager_default() {
        let manager = SpeciesDeepDiveManager::default();
        assert_eq!(manager.sample_rate(), 48000);
    }

    #[test]
    fn test_manager_analyzers() {
        let manager = SpeciesDeepDiveManager::new(192000);

        // Should have macaque analyzer
        let _macaque = manager.macaque();

        // Should have dolphin analyzer
        let _dolphin = manager.dolphin();
    }

    // ==================== SERIALIZATION TESTS ====================

    #[test]
    fn test_species_analysis_result_serialization() {
        let result = SpeciesAnalysisResult {
            species: "macaque".to_string(),
            analysis_type: "spectral_derivative".to_string(),
            confidence: 0.85,
            features: {
                let mut f = HashMap::new();
                f.insert("sweep_count".to_string(), 5.0);
                f
            },
            events: vec![],
            timestamp_ms: 12345,
        };

        let json = serde_json::to_string(&result).unwrap();
        let decoded: SpeciesAnalysisResult = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.species, "macaque");
        assert!((decoded.confidence - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_detected_event_serialization() {
        let event = DetectedEvent {
            event_type: "fm_sweep".to_string(),
            start_ms: 100.0,
            duration_ms: 50.0,
            intensity: 0.75,
            parameters: {
                let mut p = HashMap::new();
                p.insert("sweep_rate".to_string(), 40.0);
                p
            },
        };

        let json = serde_json::to_string(&event).unwrap();
        let decoded: DetectedEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.event_type, "fm_sweep");
        assert_eq!(decoded.start_ms, 100.0);
    }

    #[test]
    fn test_fm_sweep_serialization() {
        let sweep = FmSweepEvent {
            start_ms: 100.0,
            duration_ms: 50.0,
            start_freq_hz: 2000.0,
            end_freq_hz: 4000.0,
            sweep_rate_hz_ms: 40.0,
            direction: SweepDirection::Up,
            intensity: 0.8,
        };

        let json = serde_json::to_string(&sweep).unwrap();
        let decoded: FmSweepEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.direction, SweepDirection::Up);
    }

    #[test]
    fn test_qpc_event_serialization() {
        let qpc = QpcEvent {
            f1_hz: 10000.0,
            f2_hz: 20000.0,
            f3_hz: 30000.0,
            bispectrum_magnitude: 0.75,
            biphase_coherence: 0.9,
            coupling_type: CouplingType::ClickWhistleInteraction,
        };

        let json = serde_json::to_string(&qpc).unwrap();
        let decoded: QpcEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.coupling_type, CouplingType::ClickWhistleInteraction);
    }
}
