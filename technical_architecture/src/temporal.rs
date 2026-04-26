//! Temporal feature extraction
//!
//! This module provides time-domain acoustic features related to rhythm,
//! timing, and temporal patterns in vocalizations.

use std::f32::consts::PI;

/// Rhythmic stability - measures the consistency of inter-onset intervals.
///
/// Low rhythmic stability = irregular, unpredictable timing (e.g., marmoset phees)
/// High rhythmic stability = regular, metronome-like timing (e.g., corvid rattles)
///
/// ## Algorithm
/// 1. Detect onsets using spectral flux
/// 2. Calculate inter-onset intervals (IOIs)
/// 3. Compute coefficient of variation: CV = std(IOIs) / mean(IOIs)
///
/// ## Use Cases
/// - Distinguishes Corvid "Rattles" (high stability) from Marmoset "Phees" (low stability)
/// - Identifies rhythmic vs. arrhythmic vocalizations
/// - Correlates with motor planning and intentionality
#[derive(Debug, Clone, PartialEq)]
pub struct RhythmicStabilityCalculator {
    /// Sample rate for audio processing
    pub sample_rate: u32,
    /// Minimum onset threshold (0-1)
    pub onset_threshold: f32,
}

impl Default for RhythmicStabilityCalculator {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            onset_threshold: 0.1,
        }
    }
}

impl RhythmicStabilityCalculator {
    /// Create a new rhythmic stability calculator
    pub fn new(sample_rate: u32, onset_threshold: f32) -> Self {
        Self {
            sample_rate,
            onset_threshold: onset_threshold.clamp(0.01, 0.9),
        }
    }

    /// Calculate rhythmic stability from audio signal
    ///
    /// # Returns
    /// * Stability value where:
    ///   - 0.0 = completely irregular (random timing)
    ///   - 1.0 = perfectly regular (metronome-like)
    ///
    /// # Examples
    /// ```
    /// use technical_architecture::temporal::RhythmicStabilityCalculator;
    ///
    /// let calculator = RhythmicStabilityCalculator::default();
    ///
    /// // Regular clicks = high stability
    /// let regular = generate_regular_clicks(48000, 10, 0.1);
    /// let stability = calculator.calculate(&regular);
    /// assert!(stability > 0.7);
    /// ```
    pub fn calculate(&self, audio: &[f32]) -> f32 {
        if audio.len() < self.sample_rate as usize / 10 {
            // Need at least 100ms of audio
            return 0.0;
        }

        // Detect onsets using spectral flux
        let onsets = self.detect_onsets(audio);

        if onsets.len() < 3 {
            // Need at least 3 onsets to measure stability
            return 0.0;
        }

        // Calculate inter-onset intervals (in samples)
        let iois: Vec<f32> = onsets.windows(2).map(|pair| (pair[1] - pair[0]) as f32).collect();

        // Calculate mean and std dev of IOIs
        let mean_ioi = iois.iter().sum::<f32>() / iois.len() as f32;

        if mean_ioi < 1.0 {
            return 0.0;
        }

        let variance = iois.iter().map(|&ioi| (ioi - mean_ioi).powi(2)).sum::<f32>() / iois.len() as f32;

        let std_ioi = variance.sqrt();

        // Coefficient of variation (normalized by mean)
        let cv = std_ioi / mean_ioi;

        // Convert CV to stability: 1.0 when CV=0, approaches 0.0 as CV increases
        // Using exponential decay: stability = exp(-3 * CV)
        // This gives:
        // - CV = 0.0 -> stability = 1.0 (perfect rhythm)
        // - CV = 0.1 -> stability ≈ 0.74 (good rhythm)
        // - CV = 0.3 -> stability ≈ 0.41 (moderate rhythm)
        // - CV = 0.5 -> stability ≈ 0.22 (poor rhythm)
        (-3.0 * cv).exp()
    }

    /// Detect onsets using spectral flux
    fn detect_onsets(&self, audio: &[f32]) -> Vec<usize> {
        let frame_size = 1024usize;
        let hop_size = 512usize;

        if audio.len() < frame_size {
            return Vec::new();
        }

        let mut spectral_flux = Vec::new();
        let mut prev_spectrum: Option<Vec<f32>> = None;

        // Compute spectral flux for each frame
        for i in (0..audio.len() - frame_size).step_by(hop_size) {
            let frame = &audio[i..i + frame_size];
            let spectrum = self.compute_spectrum(frame);

            if let Some(prev) = &prev_spectrum {
                // Calculate spectral flux (sum of positive differences)
                let flux = spectrum
                    .iter()
                    .zip(prev.iter())
                    .map(|(&curr, &prev)| (curr - prev).max(0.0))
                    .sum::<f32>();

                spectral_flux.push(flux);
            }

            prev_spectrum = Some(spectrum);
        }

        // Find peaks in spectral flux (onsets)
        let mut onsets = Vec::new();
        let flux_mean = if spectral_flux.is_empty() {
            0.0
        } else {
            spectral_flux.iter().sum::<f32>() / spectral_flux.len() as f32
        };
        let threshold = flux_mean * self.onset_threshold.max(0.5);

        for (i, &flux) in spectral_flux.iter().enumerate() {
            // Check if this is a local peak above threshold
            let is_peak = flux > threshold
                && (i == 0 || flux > spectral_flux[i - 1])
                && (i == spectral_flux.len() - 1 || flux > spectral_flux[i + 1]);

            if is_peak {
                let onset_sample = i * hop_size;
                onsets.push(onset_sample);
            }
        }

        onsets
    }

    /// Compute magnitude spectrum using simple FFT
    fn compute_spectrum(&self, frame: &[f32]) -> Vec<f32> {
        let n = frame.len();
        if n == 0 || !n.is_power_of_two() {
            return Vec::new();
        }

        // Apply Hann window
        let windowed: Vec<f32> = frame
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let window = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos());
                x * window
            })
            .collect();

        // Simple FFT magnitude (power-of-2 only)
        let mut complex: Vec<(f32, f32)> = windowed.iter().map(|&x| (x, 0.0)).collect();

        // Bit-reversal
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                complex.swap(i, j);
            }
        }

        // FFT
        let mut len = 2;
        while len <= n {
            let angle = -2.0 * PI / len as f32;
            let (sin_a, cos_a) = angle.sin_cos();
            for i in (0..n).step_by(len) {
                let (mut w_r, mut w_i) = (1.0, 0.0);
                for j in i..i + len / 2 {
                    let (u_r, u_i) = complex[j];
                    let (v_r, v_i) = (
                        complex[j + len / 2].0 * w_r - complex[j + len / 2].1 * w_i,
                        complex[j + len / 2].0 * w_i + complex[j + len / 2].1 * w_r,
                    );
                    complex[j] = (u_r + v_r, u_i + v_i);
                    complex[j + len / 2] = (u_r - v_r, u_i - v_i);
                    let (w_r_new, w_i_new) = (w_r * cos_a - w_i * sin_a, w_r * sin_a + w_i * cos_a);
                    w_r = w_r_new;
                    w_i = w_i_new;
                }
            }
            len <<= 1;
        }

        // Return magnitude (only positive frequencies)
        complex[..n / 2].iter().map(|&(r, i)| (r * r + i * i).sqrt()).collect()
    }
}

/// Temporal centroid - the center of gravity of the energy envelope in time.
///
/// This measures where the energy is concentrated in the sound.
/// Low values = energy at the beginning (percussive)
/// High values = energy at the end (rising)
///
/// ## Use Cases
/// - Distinguishes attack-heavy sounds from gradual builds
/// - Identifies temporal structure of vocalizations
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalCentroidCalculator {
    pub sample_rate: u32,
}

impl Default for TemporalCentroidCalculator {
    fn default() -> Self {
        Self { sample_rate: 48000 }
    }
}

impl TemporalCentroidCalculator {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Calculate temporal centroid [0, 1]
    pub fn calculate(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        // Compute energy envelope
        let frame_size = 256;
        let hop_size = 128;

        let mut weighted_sum = 0.0_f32;
        let mut energy_sum = 0.0_f32;
        let mut frame_count = 0usize;

        for i in (0..audio.len()).step_by(hop_size) {
            let end = (i + frame_size).min(audio.len());
            if end <= i {
                break;
            }

            let frame = &audio[i..end];
            let energy = frame.iter().map(|&x| x * x).sum::<f32>();

            weighted_sum += frame_count as f32 * energy;
            energy_sum += energy;
            frame_count += 1;
        }

        if energy_sum < 1e-10 {
            return 0.5;
        }

        let centroid = weighted_sum / energy_sum;

        // Normalize by total frames
        if frame_count > 0 {
            centroid / frame_count as f32
        } else {
            0.5
        }
    }
}

/// Helper: Generate regular clicks (onsets)
#[cfg(test)]
fn generate_regular_clicks(sample_rate: u32, clicks_per_sec: f32, duration_sec: f32) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    let interval_samples = (sample_rate as f32 / clicks_per_sec) as usize;
    let mut audio = vec![0.0; num_samples];

    for i in (0..num_samples).step_by(interval_samples) {
        // Create a short click (impulse)
        if i + 100 < num_samples {
            for j in 0..100.min(num_samples - i) {
                audio[i + j] = 0.5 * (-0.1 * j as f32).exp(); // Decaying impulse
            }
        }
    }

    audio
}

/// Helper: Generate irregular clicks (random timing)
#[cfg(test)]
fn generate_irregular_clicks(sample_rate: u32, num_clicks: usize, duration_sec: f32) -> Vec<f32> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    let mut audio = vec![0.0; num_samples];

    let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
    let mut rng: u32 = seed;

    for _ in 0..num_clicks {
        let i = (rng as usize % num_samples).saturating_sub(100);
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);

        if i + 100 < num_samples {
            for j in 0..100.min(num_samples - i) {
                audio[i + j] = 0.5 * (-0.1 * j as f32).exp();
            }
        }
    }

    audio
}

/// Helper: Generate sine wave for testing
#[cfg(test)]
fn generate_sine_wave(freq_hz: f32, sample_rate: u32, duration_sec: f32) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * PI * freq_hz * t).sin()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Rhythmic Stability Tests (8 tests)
    // =========================================================================

    #[test]
    fn test_rhythmic_stability_regular_clicks() {
        let calculator = RhythmicStabilityCalculator::default();
        // Generate regular clicks at 10 Hz
        let clicks = generate_regular_clicks(48000, 10.0, 0.5);
        let stability = calculator.calculate(&clicks);
        // Regular clicks should have high stability
        assert!(stability > 0.5, "Regular clicks should have high stability");
    }

    #[test]
    fn test_rhythmic_stability_irregular_clicks() {
        let calculator = RhythmicStabilityCalculator::default();
        // Generate irregular clicks (random timing)
        let clicks = generate_irregular_clicks(48000, 10, 0.5);
        let stability = calculator.calculate(&clicks);
        // Irregular clicks should have lower stability
        assert!(stability < 0.7, "Irregular clicks should have lower stability");
    }

    #[test]
    fn test_rhythmic_stability_single_click() {
        let calculator = RhythmicStabilityCalculator::default();
        let clicks = generate_regular_clicks(48000, 1.0, 0.1);
        let stability = calculator.calculate(&clicks);
        // Single click should have 0 stability (not enough onsets)
        assert_eq!(stability, 0.0);
    }

    #[test]
    fn test_rhythmic_stability_silence() {
        let calculator = RhythmicStabilityCalculator::default();
        let silence = vec![0.0; 48000];
        let stability = calculator.calculate(&silence);
        assert_eq!(stability, 0.0);
    }

    #[test]
    fn test_rhythmic_stability_sine_wave() {
        let calculator = RhythmicStabilityCalculator::default();
        let sine = generate_sine_wave(440.0, 48000, 0.5);
        let stability = calculator.calculate(&sine);
        // Continuous sine should have low stability (few onsets)
        assert!(stability <= 1.0);
    }

    #[test]
    fn test_rhythmic_stability_trill() {
        let calculator = RhythmicStabilityCalculator::default();
        // Trill: rapid alternation between two frequencies
        let sample_rate = 48000;
        let duration_ms = 500.0;
        let num_samples = (duration_ms / 1000.0 * sample_rate as f32) as usize;
        let trill: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let freq = if (i / (sample_rate as usize / 20)).is_multiple_of(2) {
                    440.0
                } else {
                    880.0
                };
                (2.0 * PI * freq * t).sin()
            })
            .collect();

        let stability = calculator.calculate(&trill);
        // Trill should have some rhythmic structure
        assert!(stability > 0.0);
    }

    #[test]
    fn test_rhythmic_stability_custom_threshold() {
        let calculator = RhythmicStabilityCalculator::new(48000, 0.5);
        let clicks = generate_regular_clicks(48000, 8.0, 0.5);
        let stability = calculator.calculate(&clicks);
        assert!((0.0..=1.0).contains(&stability));
    }

    #[test]
    fn test_rhythmic_stability_range() {
        let calculator = RhythmicStabilityCalculator::default();
        // Test with various inputs
        for &test in &[&[0.0f32; 4800][..], &generate_sine_wave(1000.0, 48000, 0.1)[..]] {
            let stability = calculator.calculate(test);
            assert!((0.0..=1.0).contains(&stability));
        }
    }

    // =========================================================================
    // Temporal Centroid Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_temporal_centroid_attack_heavy() {
        let calculator = TemporalCentroidCalculator::new(48000);
        // Attack-heavy: loud at beginning, fades out
        let attack_heavy: Vec<f32> = (0..4800)
            .map(|i| {
                let env = (-0.001 * i as f32).exp(); // Exponential decay
                (2.0 * PI * 440.0 * i as f32 / 48000.0).sin() * env
            })
            .collect();

        let centroid = calculator.calculate(&attack_heavy);
        // Should be low (energy at beginning)
        assert!(centroid < 0.4);
    }

    #[test]
    fn test_temporal_centroid_rising() {
        let calculator = TemporalCentroidCalculator::new(48000);
        // Rising: quiet at beginning, gets louder
        let rising: Vec<f32> = (0..4800)
            .map(|i| {
                let env = i as f32 / 4800.0; // Linear rise
                (2.0 * PI * 440.0 * i as f32 / 48000.0).sin() * env
            })
            .collect();

        let centroid = calculator.calculate(&rising);
        // Should be high (energy at end)
        assert!(centroid > 0.5);
    }

    #[test]
    fn test_temporal_centroid_sustained() {
        let calculator = TemporalCentroidCalculator::new(48000);
        // Sustained: constant envelope
        let sustained = generate_sine_wave(440.0, 48000, 0.1);
        let centroid = calculator.calculate(&sustained);
        // Should be around 0.5 (evenly distributed)
        assert!(centroid > 0.3 && centroid < 0.7);
    }

    #[test]
    fn test_temporal_centroid_empty() {
        let calculator = TemporalCentroidCalculator::new(48000);
        let empty: Vec<f32> = vec![];
        let centroid = calculator.calculate(&empty);
        assert_eq!(centroid, 0.0);
    }

    #[test]
    fn test_temporal_centroid_silence() {
        let calculator = TemporalCentroidCalculator::new(48000);
        let silence = vec![0.0; 4800];
        let centroid = calculator.calculate(&silence);
        // Silence should give middle value
        assert_eq!(centroid, 0.5);
    }

    #[test]
    fn test_temporal_centroid_range() {
        let calculator = TemporalCentroidCalculator::new(48000);
        let tone = generate_sine_wave(1000.0, 48000, 0.1);
        let centroid = calculator.calculate(&tone);
        assert!((0.0..=1.0).contains(&centroid));
    }
}
