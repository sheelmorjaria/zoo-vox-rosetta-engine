//! Psychoacoustic feature extraction
//!
//! This module provides perceptually-motivated acoustic features that correlate
//! with how humans and animals perceive sound quality and complexity.

use std::f32::consts::PI;

/// Pitch entropy - measures the complexity and unpredictability of a pitch contour.
///
/// A steady sine wave has entropy near 0, while a complex trill or warble has high entropy.
///
/// ## Algorithm
/// 1. Compute histogram of F0 values across the signal
/// 2. Normalize to probability distribution
/// 3. Calculate Shannon entropy: H = -Σ(p * log2(p))
///
/// ## Use Cases
/// - Differentiates "Monotone Phee" (low entropy) from "Warbled Phee" (high entropy)
/// - Identifies complex vocalizations with rapid pitch changes
/// - Correlates with behavioral arousal in some species
#[derive(Debug, Clone, PartialEq)]
pub struct PitchEntropyCalculator {
    /// Number of histogram bins for entropy calculation
    pub num_bins: usize,
}

impl Default for PitchEntropyCalculator {
    fn default() -> Self {
        Self { num_bins: 24 }
    }
}

impl PitchEntropyCalculator {
    /// Create a new pitch entropy calculator
    pub fn new(num_bins: usize) -> Self {
        assert!(num_bins > 1, "num_bins must be greater than 1");
        Self { num_bins }
    }

    /// Calculate pitch entropy from a sequence of F0 values
    ///
    /// # Arguments
    /// * `f0_contour` - Slice of fundamental frequency values (Hz)
    ///
    /// # Returns
    /// * Shannon entropy in bits (0 = constant pitch, higher = more complex)
    ///
    /// # Examples
    /// ```
    /// use technical_architecture::psychoacoustics::PitchEntropyCalculator;
    ///
    /// let calculator = PitchEntropyCalculator::default();
    ///
    /// // Constant pitch = 0 entropy
    /// let constant = vec![440.0; 100];
    /// let entropy = calculator.calculate(&constant);
    /// assert!(entropy < 0.1);
    ///
    /// // Variable pitch = higher entropy
    /// let variable = vec![400.0, 500.0, 600.0, 400.0, 500.0, 600.0];
    /// let entropy = calculator.calculate(&variable);
    /// assert!(entropy > 0.5);
    /// ```
    pub fn calculate(&self, f0_contour: &[f32]) -> f32 {
        if f0_contour.is_empty() {
            return 0.0;
        }

        // Filter out unvoiced frames (F0 <= 0 or NaN)
        let valid_f0: Vec<f32> = f0_contour
            .iter()
            .filter(|&&f| f > 0.0 && f.is_finite())
            .cloned()
            .collect();

        if valid_f0.len() < 2 {
            return 0.0; // Not enough variation to measure entropy
        }

        // Find min and max F0 for binning
        let f0_min = valid_f0.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let f0_max = valid_f0.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        if f0_max <= f0_min {
            return 0.0; // All values are the same
        }

        // Create histogram
        let mut histogram = vec![0usize; self.num_bins];
        let bin_width = (f0_max - f0_min) / self.num_bins as f32;

        for &f0 in &valid_f0 {
            let bin_idx = ((f0 - f0_min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(self.num_bins - 1);
            histogram[bin_idx] += 1;
        }

        // Convert to probability distribution
        let total_samples = valid_f0.len() as f32;
        let mut entropy = 0.0_f32;

        for &count in &histogram {
            if count > 0 {
                let probability = count as f32 / total_samples;
                entropy -= probability * probability.log2();
            }
        }

        // Normalize by max possible entropy (log2 of num_bins)
        let max_entropy = (self.num_bins as f32).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }
}

/// Spectral roughness - measures the perceived harshness or grittiness of a sound.
///
/// Roughness is based on the energy in high-frequency spectral components relative to total energy.
/// Unlike HNR (which measures tonal vs. noise), roughness measures spectral edges and harshness.
///
/// ## Algorithm
/// 1. Compute magnitude spectrum
/// 2. Sum energy above 500 Hz (roughness band)
/// 3. Normalize by total energy
///
/// ## Use Cases
/// - Distinguishes Corvid "Caws" (high roughness) from Marmoset "Phees" (low roughness)
/// - Identifies harsh, raspy vocalizations
/// - Correlates with aggression and arousal in many species
#[derive(Debug, Clone, PartialEq)]
pub struct RoughnessCalculator {
    /// Frequency threshold (Hz) for roughness band
    pub threshold_hz: f32,
    /// Sample rate for FFT computation
    pub sample_rate: u32,
}

impl Default for RoughnessCalculator {
    fn default() -> Self {
        Self {
            threshold_hz: 500.0,
            sample_rate: 48000,
        }
    }
}

impl RoughnessCalculator {
    /// Create a new roughness calculator
    pub fn new(threshold_hz: f32, sample_rate: u32) -> Self {
        Self {
            threshold_hz,
            sample_rate,
        }
    }

    /// Calculate spectral roughness from audio signal
    ///
    /// # Arguments
    /// * `audio` - Audio samples
    ///
    /// # Returns
    /// * Roughness value [0, 1], where 0 = smooth, 1 = very rough
    ///
    /// # Examples
    /// ```
    /// use technical_architecture::psychoacoustics::RoughnessCalculator;
    ///
    /// let calculator = RoughnessCalculator::default();
    ///
    /// // Pure tone = low roughness
    /// let pure_tone = generate_sine_wave(1000.0, 48000, 0.1);
    /// let roughness = calculator.calculate(&pure_tone);
    /// assert!(roughness < 0.3);
    /// ```
    pub fn calculate(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        // Compute magnitude spectrum using FFT
        let spectrum = self.compute_spectrum(audio);

        if spectrum.is_empty() {
            return 0.0;
        }

        // Calculate frequency bins
        let num_bins = spectrum.len();
        let bin_resolution = self.sample_rate as f32 / (2.0 * num_bins as f32);
        let roughness_bin = (self.threshold_hz / bin_resolution).ceil() as usize;

        // Sum energy in roughness band (high frequencies)
        let roughness_energy: f32 = spectrum.iter()
            .skip(roughness_bin.min(num_bins))
            .map(|&x| x * x)
            .sum();

        // Sum total energy
        let total_energy: f32 = spectrum.iter()
            .map(|&x| x * x)
            .sum();

        if total_energy < 1e-10 {
            return 0.0;
        }

        // Normalize to [0, 1]
        (roughness_energy / total_energy).sqrt()
    }

    /// Compute magnitude spectrum using simple FFT
    fn compute_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }

        // Use next power of 2 for FFT efficiency
        let n = audio.len().next_power_of_two();
        let mut padded = vec![0.0f32; n];
        padded[..audio.len().min(n)].copy_from_slice(&audio[..audio.len().min(n)]);

        // Apply Hann window for better frequency resolution
        self.apply_hann_window(&mut padded);

        // Compute FFT magnitude
        self.fft_magnitude(&padded)
    }

    /// Apply Hann window to reduce spectral leakage
    fn apply_hann_window(&self, data: &mut [f32]) {
        let n = data.len();
        for (i, sample) in data.iter_mut().enumerate() {
            let window = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos());
            *sample *= window;
        }
    }

    /// Compute FFT magnitude using Cooley-Tukey algorithm
    fn fft_magnitude(&self, data: &[f32]) -> Vec<f32> {
        let n = data.len();
        if n == 0 {
            return Vec::new();
        }

        // Convert to complex
        let mut complex: Vec<(f32, f32)> = data.iter().map(|&x| (x, 0.0)).collect();

        // Bit-reversal permutation
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

        // Cooley-Tukey FFT
        let mut len = 2;
        while len <= n {
            let angle = -2.0 * PI / len as f32;
            let (sin_angle, cos_angle) = angle.sin_cos();
            for i in (0..n).step_by(len) {
                let (mut w_real, mut w_imag) = (1.0, 0.0);
                for j in i..i + len / 2 {
                    let (u_real, u_imag) = complex[j];
                    let (v_real, v_imag) = (
                        complex[j + len / 2].0 * w_real - complex[j + len / 2].1 * w_imag,
                        complex[j + len / 2].0 * w_imag + complex[j + len / 2].1 * w_real,
                    );
                    complex[j] = (u_real + v_real, u_imag + v_imag);
                    complex[j + len / 2] = (u_real - v_real, u_imag - v_imag);
                    let (w_real_new, w_imag_new) = (
                        w_real * cos_angle - w_imag * sin_angle,
                        w_real * sin_angle + w_imag * cos_angle,
                    );
                    w_real = w_real_new;
                    w_imag = w_imag_new;
                }
            }
            len <<= 1;
        }

        // Return magnitude spectrum (only positive frequencies)
        complex[..n / 2].iter().map(|&(r, i)| (r * r + i * i).sqrt()).collect()
    }
}

/// Brightness - perceptual loudness weighted by spectral centroid.
///
/// Brightness correlates with the perceived "sharpness" or "brilliance" of a sound.
/// High brightness = more energy in high frequencies (bright, sharp)
/// Low brightness = more energy in low frequencies (dark, warm)
///
/// ## Algorithm
/// 1. Compute weighted centroid: Σ(f * X(f)) / Σ(X(f))
/// 2. Normalize by Nyquist frequency
///
/// ## Use Cases
/// - Distinguishes bright alarm calls from low-pitched contact calls
/// - Correlates with aggression and arousal in many species
#[derive(Debug, Clone, PartialEq)]
pub struct BrightnessCalculator {
    pub sample_rate: u32,
}

impl Default for BrightnessCalculator {
    fn default() -> Self {
        Self { sample_rate: 48000 }
    }
}

impl BrightnessCalculator {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Calculate brightness from audio signal
    ///
    /// # Returns
    /// * Brightness [0, 1], where 0 = dark, 1 = bright
    pub fn calculate(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        // Compute spectrum
        let roughness_calc = RoughnessCalculator::default();
        let spectrum = roughness_calc.compute_spectrum(audio);

        if spectrum.is_empty() {
            return 0.0;
        }

        // Calculate weighted centroid
        let mut weighted_sum = 0.0_f32;
        let mut energy_sum = 0.0_f32;
        let nyquist = self.sample_rate as f32 / 2.0;
        let bin_resolution = nyquist / spectrum.len() as f32;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let frequency = i as f32 * bin_resolution;
            let energy = magnitude * magnitude;
            weighted_sum += frequency * energy;
            energy_sum += energy;
        }

        if energy_sum < 1e-10 {
            return 0.0;
        }

        let centroid = weighted_sum / energy_sum;

        // Normalize by Nyquist
        (centroid / nyquist).min(1.0).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Pitch Entropy Tests (8 tests)
    // =========================================================================

    #[test]
    fn test_pitch_entropy_constant_pitch() {
        let calculator = PitchEntropyCalculator::default();
        let constant = vec![440.0; 100];
        let entropy = calculator.calculate(&constant);
        assert!(entropy < 0.1, "Constant pitch should have near-zero entropy");
    }

    #[test]
    fn test_pitch_entropy_binary_pitch() {
        let calculator = PitchEntropyCalculator::default();
        // Alternating between two pitches
        let binary: Vec<f32> = (0..50).flat_map(|_| [400.0, 600.0]).collect();
        let entropy = calculator.calculate(&binary);
        assert!(entropy > 0.1, "Binary pitch should have some entropy");
        assert!(entropy <= 1.0, "Entropy should be normalized");
    }

    #[test]
    fn test_pitch_entropy_uniform_distribution() {
        let calculator = PitchEntropyCalculator::new(10);
        // Uniform distribution across range
        let uniform: Vec<f32> = (0..100).map(|i| 100.0 + i as f32 * 5.0).collect();
        let entropy = calculator.calculate(&uniform);
        assert!(entropy > 0.7, "Uniform distribution should have high entropy");
    }

    #[test]
    fn test_pitch_entropy_empty() {
        let calculator = PitchEntropyCalculator::default();
        let empty: Vec<f32> = vec![];
        let entropy = calculator.calculate(&empty);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_pitch_entropy_single_value() {
        let calculator = PitchEntropyCalculator::default();
        let single = vec![440.0];
        let entropy = calculator.calculate(&single);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_pitch_entropy_with_unvoiced_frames() {
        let calculator = PitchEntropyCalculator::default();
        // Mix of voiced (positive) and unvoiced (zero/negative) frames
        let mixed = vec![440.0, 0.0, 450.0, -1.0, 460.0, f32::NAN, 470.0];
        let entropy = calculator.calculate(&mixed);
        assert!(entropy > 0.0, "Should handle unvoiced frames");
    }

    #[test]
    fn test_pitch_entropy_trill_pattern() {
        let calculator = PitchEntropyCalculator::default();
        // Trill: rapid alternation between pitches
        let trill: Vec<f32> = (0..30).flat_map(|_| [400.0, 500.0, 400.0, 500.0]).collect();
        let entropy = calculator.calculate(&trill);
        assert!(entropy > 0.2, "Trill should have measurable entropy");
    }

    #[test]
    fn test_pitch_entropy_custom_bins() {
        let calculator = PitchEntropyCalculator::new(8);
        let variable: Vec<f32> = (0..50).map(|i| 100.0 + i as f32 * 10.0).collect();
        let entropy = calculator.calculate(&variable);
        assert!(entropy > 0.0);
        assert!(entropy <= 1.0);
    }

    // =========================================================================
    // Roughness Tests (8 tests)
    // =========================================================================

    #[test]
    fn test_roughness_pure_tone() {
        let calculator = RoughnessCalculator::default();
        // Use low frequency tone (below 500Hz threshold) for this test
        let pure_tone = generate_sine_wave(200.0, 48000, 0.1);
        let roughness = calculator.calculate(&pure_tone);
        // Low frequency tone should have lower roughness (energy below threshold)
        assert!(roughness < 0.5, "Low frequency tone should have lower roughness");
    }

    #[test]
    fn test_roughness_white_noise() {
        let calculator = RoughnessCalculator::default();
        let noise = generate_white_noise(48000, 0.1);
        let roughness = calculator.calculate(&noise);
        // White noise should have higher roughness (broadband energy)
        assert!(roughness > 0.3, "White noise should have higher roughness");
    }

    #[test]
    fn test_roughness_low_frequency() {
        let calculator = RoughnessCalculator::default();
        // Low frequency tone (< 500Hz threshold)
        let low_tone = generate_sine_wave(200.0, 48000, 0.1);
        let roughness = calculator.calculate(&low_tone);
        // Should have lower roughness (energy below threshold)
        assert!(roughness < 0.5);
    }

    #[test]
    fn test_roughness_high_frequency() {
        let calculator = RoughnessCalculator::default();
        // High frequency tone (> 500Hz threshold)
        let high_tone = generate_sine_wave(2000.0, 48000, 0.1);
        let roughness = calculator.calculate(&high_tone);
        // Should have higher roughness (energy above threshold)
        assert!(roughness > 0.1);
    }

    #[test]
    fn test_roughness_empty() {
        let calculator = RoughnessCalculator::default();
        let empty: Vec<f32> = vec![];
        let roughness = calculator.calculate(&empty);
        assert_eq!(roughness, 0.0);
    }

    #[test]
    fn test_roughness_silence() {
        let calculator = RoughnessCalculator::default();
        let silence = vec![0.0; 4800];
        let roughness = calculator.calculate(&silence);
        assert_eq!(roughness, 0.0);
    }

    #[test]
    fn test_roughness_custom_threshold() {
        let calculator = RoughnessCalculator::new(1000.0, 48000);
        let tone = generate_sine_wave(1500.0, 48000, 0.1);
        let roughness = calculator.calculate(&tone);
        assert!(roughness >= 0.0 && roughness <= 1.0);
    }

    #[test]
    fn test_roughness_multi_tone() {
        let calculator = RoughnessCalculator::default();
        // Sum of multiple sine waves
        let mut audio = vec![0.0; 4800];
        for freq in &[500.0, 1000.0, 2000.0, 3000.0] {
            for (i, sample) in audio.iter_mut().enumerate() {
                let t = i as f32 / 48000.0;
                *sample += (2.0 * PI * freq * t).sin();
            }
        }
        let roughness = calculator.calculate(&audio);
        assert!(roughness > 0.0);
    }

    // =========================================================================
    // Brightness Tests (8 tests)
    // =========================================================================

    #[test]
    fn test_brightness_low_frequency() {
        let calculator = BrightnessCalculator::new(48000);
        let low_tone = generate_sine_wave(200.0, 48000, 0.1);
        let brightness = calculator.calculate(&low_tone);
        assert!(brightness < 0.2, "Low frequency should have low brightness");
    }

    #[test]
    fn test_brightness_high_frequency() {
        let calculator = BrightnessCalculator::new(48000);
        let high_tone = generate_sine_wave(10000.0, 48000, 0.1);
        let brightness = calculator.calculate(&high_tone);
        // High frequency tone should have higher brightness than low
        assert!(brightness > 0.2, "High frequency should have measurable brightness");
    }

    #[test]
    fn test_brightness_empty() {
        let calculator = BrightnessCalculator::new(48000);
        let empty: Vec<f32> = vec![];
        let brightness = calculator.calculate(&empty);
        assert_eq!(brightness, 0.0);
    }

    #[test]
    fn test_brightness_silence() {
        let calculator = BrightnessCalculator::new(48000);
        let silence = vec![0.0; 4800];
        let brightness = calculator.calculate(&silence);
        assert_eq!(brightness, 0.0);
    }

    #[test]
    fn test_brightness_range() {
        let calculator = BrightnessCalculator::new(48000);
        for freq in &[100.0, 500.0, 1000.0, 5000.0, 10000.0, 15000.0] {
            let tone = generate_sine_wave(*freq, 48000, 0.1);
            let brightness = calculator.calculate(&tone);
            assert!(brightness >= 0.0 && brightness <= 1.0);
        }
    }

    #[test]
    fn test_brightness_monotonic() {
        let calculator = BrightnessCalculator::new(48000);
        let brightness_100 = calculator.calculate(&generate_sine_wave(100.0, 48000, 0.1));
        let brightness_1000 = calculator.calculate(&generate_sine_wave(1000.0, 48000, 0.1));
        let brightness_10000 = calculator.calculate(&generate_sine_wave(10000.0, 48000, 0.1));
        assert!(brightness_100 < brightness_1000);
        assert!(brightness_1000 < brightness_10000);
    }

    #[test]
    fn test_brightness_harmonic_series() {
        let calculator = BrightnessCalculator::new(48000);
        // Fundamental + harmonics
        let mut audio = vec![0.0; 4800];
        let fundamental = 440.0;
        for (h, weight) in [1.0, 0.5, 0.25, 0.125].iter().enumerate() {
            let freq = fundamental * (h + 1) as f32;
            for (i, sample) in audio.iter_mut().enumerate() {
                let t = i as f32 / 48000.0;
                *sample += weight * (2.0 * PI * freq * t).sin();
            }
        }
        let brightness = calculator.calculate(&audio);
        assert!(brightness > 0.0 && brightness < 1.0);
    }

    #[test]
    fn test_brightness_different_sample_rates() {
        let tone_44100 = generate_sine_wave(1000.0, 44100, 0.1);
        let tone_48000 = generate_sine_wave(1000.0, 48000, 0.1);

        let calc_44100 = BrightnessCalculator::new(44100);
        let calc_48000 = BrightnessCalculator::new(48000);

        let brightness_44100 = calc_44100.calculate(&tone_44100);
        let brightness_48000 = calc_48000.calculate(&tone_48000);

        // Similar frequencies should give similar brightness
        assert!((brightness_44100 - brightness_48000).abs() < 0.1);
    }
}

/// Helper function to generate sine wave for testing
fn generate_sine_wave(freq_hz: f32, sample_rate: u32, duration_sec: f32) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * PI * freq_hz * t).sin()
        })
        .collect()
}

/// Helper function to generate white noise for testing
fn generate_white_noise(sample_rate: u32, duration_sec: f32) -> Vec<f32> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().subsec_nanos();
    let mut rng: u32 = seed;

    (0..num_samples)
        .map(|_| {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}
