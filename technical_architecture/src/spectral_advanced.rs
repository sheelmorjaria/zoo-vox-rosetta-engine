//! Advanced spectral feature extraction
//!
//! This module provides sophisticated spectral analysis features including
//! spectral tilt, advanced envelope analysis, and perceptual spectral features.

use std::f32::consts::PI;

/// Spectral tilt - measures the perceptual brightness/darkness of a sound.
///
/// Unlike spectral_slope (linear regression), spectral tilt measures the
/// perceptual roll-off in dB/octave, correlating with how humans perceive
/// "brightness" or "warmth" in a sound.
///
/// ## Algorithm
/// 1. Compute power spectrum in dB
/// 2. Fit linear regression to log-frequency domain
/// 3. Alpha coefficient (tilt) = slope in dB/octave
///
/// ## Use Cases
/// - "Bright" sounds (e.g., trumpet) have negative tilt (energy decreases with frequency)
/// - "Dark" sounds (e.g., bassoon) have flatter tilt (energy maintained across spectrum)
/// - Correlates with timbre and sound quality
///
/// ## Interpretation
/// - Negative values: Bright sound (high frequency roll-off)
/// - Near zero: Flat/balanced sound
/// - Positive values: Dark sound (low frequency emphasis)
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralTiltCalculator {
    pub sample_rate: u32,
}

impl Default for SpectralTiltCalculator {
    fn default() -> Self {
        Self { sample_rate: 48000 }
    }
}

impl SpectralTiltCalculator {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Calculate spectral tilt (alpha coefficient in dB/octave)
    ///
    /// # Returns
    /// * Tilt value where:
    ///   - Negative = bright (high freq roll-off)
    ///   - Zero = flat spectrum
    ///   - Positive = dark (low freq emphasis)
    pub fn calculate(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        // Compute magnitude spectrum
        let spectrum = self.compute_spectrum(audio);

        if spectrum.len() < 4 {
            return 0.0;
        }

        // Convert to dB and use log-frequency scale
        let nyquist = self.sample_rate as f32 / 2.0;
        let mut log_freqs = Vec::new();
        let mut power_dbs = Vec::new();

        for (i, &magnitude) in spectrum.iter().enumerate() {
            let freq = (i as f32 / spectrum.len() as f32) * nyquist;

            // Skip DC and very low frequencies
            if freq < 50.0 {
                continue;
            }

            // Avoid log(0)
            let power = (magnitude * magnitude).max(1e-10);
            let power_db = 10.0 * power.log10();

            // Log frequency (base 2 for octave scaling)
            let log_freq = (freq / 1000.0).max(0.001).log2(); // Normalize to 1 kHz

            log_freqs.push(log_freq);
            power_dbs.push(power_db);
        }

        if log_freqs.len() < 4 {
            return 0.0;
        }

        // Linear regression: y = mx + b
        // where x = log2(frequency), y = power_db
        let n = log_freqs.len() as f32;

        let sum_x: f32 = log_freqs.iter().sum();
        let sum_y: f32 = power_dbs.iter().sum();
        let sum_xy: f32 = log_freqs
            .iter()
            .zip(power_dbs.iter())
            .map(|(x, y)| x * y)
            .sum();
        let sum_x2: f32 = log_freqs.iter().map(|x| x * x).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;

        if denominator.abs() < 1e-10 {
            return 0.0;
        }

        // Slope (m) in dB/octave
        let slope = (n * sum_xy - sum_x * sum_y) / denominator;

        slope
    }

    /// Compute magnitude spectrum
    fn compute_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        let n = audio.len().next_power_of_two();
        if n < 4 {
            return Vec::new();
        }

        let mut padded = vec![0.0f32; n];
        padded[..audio.len().min(n)].copy_from_slice(&audio[..audio.len().min(n)]);

        // Apply Hann window
        self.apply_hann_window(&mut padded);

        self.fft_magnitude(&padded)
    }

    fn apply_hann_window(&self, data: &mut [f32]) {
        let n = data.len();
        for (i, sample) in data.iter_mut().enumerate() {
            let window = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos());
            *sample *= window;
        }
    }

    fn fft_magnitude(&self, data: &[f32]) -> Vec<f32> {
        let n = data.len();
        let mut complex: Vec<(f32, f32)> = data.iter().map(|&x| (x, 0.0)).collect();

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

        complex[..n / 2]
            .iter()
            .map(|&(r, i)| (r * r + i * i).sqrt())
            .collect()
    }
}

/// Spectral kurtosis - measures the "peakedness" of the spectral distribution.
///
/// High kurtosis = sharp, tonal sound (energy concentrated in few peaks)
/// Low kurtosis = noisy, broadband sound (energy spread out)
///
/// ## Use Cases
/// - Distinguishes tonal calls from noisy calls
/// - Identifies "sharp" vs "diffuse" timbres
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralKurtosisCalculator {
    pub sample_rate: u32,
}

impl Default for SpectralKurtosisCalculator {
    fn default() -> Self {
        Self { sample_rate: 48000 }
    }
}

impl SpectralKurtosisCalculator {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Calculate spectral kurtosis
    pub fn calculate(&self, audio: &[f32]) -> f32 {
        if audio.len() < 256 {
            return 0.0;
        }

        let tilt_calc = SpectralTiltCalculator::new(self.sample_rate);
        let spectrum = tilt_calc.compute_spectrum(audio);

        if spectrum.len() < 4 {
            return 0.0;
        }

        // Convert to power spectrum (normalize)
        let power: Vec<f32> = spectrum.iter().map(|&x| x * x).collect();

        let total_power: f32 = power.iter().sum();
        if total_power < 1e-10 {
            return 0.0;
        }

        let normalized: Vec<f32> = power.iter().map(|&p| p / total_power).collect();

        // Calculate mean
        let mean = normalized.iter().sum::<f32>() / normalized.len() as f32;

        // Calculate variance
        let variance =
            normalized.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / normalized.len() as f32;

        if variance < 1e-10 {
            return 0.0;
        }

        // Calculate kurtosis (fourth moment)
        let kurtosis = normalized
            .iter()
            .map(|&x| ((x - mean) / variance.sqrt()).powi(4))
            .sum::<f32>()
            / normalized.len() as f32;

        // Excess kurtosis (subtract 3 for normal distribution baseline)
        kurtosis - 3.0
    }
}

/// Spectral flatness - measures how "tone-like" vs "noise-like" a sound is.
///
/// Flatness = geometric_mean / arithmetic_mean
/// - 1.0 = white noise (completely flat spectrum)
/// - 0.0 = pure tone (single peak)
///
/// This is already in MicroDynamicsFeatures but provided here for completeness.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralFlatnessCalculator {
    pub sample_rate: u32,
}

impl Default for SpectralFlatnessCalculator {
    fn default() -> Self {
        Self { sample_rate: 48000 }
    }
}

impl SpectralFlatnessCalculator {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Calculate spectral flatness [0, 1]
    pub fn calculate(&self, audio: &[f32]) -> f32 {
        if audio.is_empty() {
            return 0.0;
        }

        let tilt_calc = SpectralTiltCalculator::new(self.sample_rate);
        let spectrum = tilt_calc.compute_spectrum(audio);

        if spectrum.is_empty() {
            return 0.0;
        }

        // Convert to power spectrum
        let power: Vec<f32> = spectrum.iter().map(|&x| (x * x).max(1e-10)).collect();

        // Geometric mean
        let log_sum: f32 = power.iter().map(|&x| x.ln()).sum();
        let geometric_mean = (log_sum / power.len() as f32).exp();

        // Arithmetic mean
        let arithmetic_mean = power.iter().sum::<f32>() / power.len() as f32;

        if arithmetic_mean < 1e-10 {
            return 0.0;
        }

        // Flatness = geometric / arithmetic
        (geometric_mean / arithmetic_mean).min(1.0).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Spectral Tilt Tests (8 tests)
    // =========================================================================

    #[test]
    fn test_spectral_tilt_low_frequency() {
        let calculator = SpectralTiltCalculator::new(48000);
        // Low frequency tone (100 Hz)
        let low_tone = generate_sine_wave(100.0, 48000, 0.1);
        let tilt = calculator.calculate(&low_tone);
        // Low frequency should give flatter spectrum (less negative tilt)
        assert!(tilt > -20.0, "Low frequency should have moderate tilt");
    }

    #[test]
    fn test_spectral_tilt_high_frequency() {
        let calculator = SpectralTiltCalculator::new(48000);
        // High frequency tone (10 kHz)
        let high_tone = generate_sine_wave(10000.0, 48000, 0.1);
        let tilt = calculator.calculate(&high_tone);
        // High frequency should give more negative tilt
        assert!(tilt < 0.0, "High frequency should have negative tilt");
    }

    #[test]
    fn test_spectral_tilt_white_noise() {
        let calculator = SpectralTiltCalculator::new(48000);
        let noise = generate_white_noise(48000, 0.1);
        let tilt = calculator.calculate(&noise);
        // White noise should have flat spectrum (tilt near 0)
        assert!(
            tilt > -10.0 && tilt < 10.0,
            "White noise should have flat tilt"
        );
    }

    #[test]
    fn test_spectral_tilt_empty() {
        let calculator = SpectralTiltCalculator::new(48000);
        let empty: Vec<f32> = vec![];
        let tilt = calculator.calculate(&empty);
        assert_eq!(tilt, 0.0);
    }

    #[test]
    fn test_spectral_tilt_harmonic_series() {
        let calculator = SpectralTiltCalculator::new(48000);
        // Fundamental with harmonics (natural spectrum decay)
        let mut audio = vec![0.0; 4800];
        let fundamental = 220.0;
        for (h, weight) in [1.0, 0.5, 0.33, 0.25, 0.2].iter().enumerate() {
            let freq = fundamental * (h + 1) as f32;
            for (i, sample) in audio.iter_mut().enumerate() {
                let t = i as f32 / 48000.0;
                *sample += weight * (2.0 * PI * freq * t).sin();
            }
        }
        let tilt = calculator.calculate(&audio);
        // Harmonic series should have natural negative tilt
        assert!(tilt < 0.0);
    }

    #[test]
    fn test_spectral_tilt_sawtooth() {
        let calculator = SpectralTiltCalculator::new(48000);
        // Sawtooth wave (1/f spectrum)
        let sawtooth = generate_sawtooth(220.0, 48000, 0.1);
        let tilt = calculator.calculate(&sawtooth);
        // Sawtooth has -6 dB/octave tilt
        assert!(tilt < -3.0 && tilt > -15.0);
    }

    #[test]
    fn test_spectral_tilt_square() {
        let calculator = SpectralTiltCalculator::new(48000);
        // Square wave (odd harmonics only)
        let square = generate_square(220.0, 48000, 0.1);
        let tilt = calculator.calculate(&square);
        // Square has harmonic series with faster decay
        assert!(tilt < 0.0);
    }

    #[test]
    fn test_spectral_tilt_range() {
        let calculator = SpectralTiltCalculator::new(48000);
        // Test various signals
        let sine = generate_sine_wave(1000.0, 48000, 0.1);
        let tilt = calculator.calculate(&sine);
        // Should be finite
        assert!(tilt.is_finite());
    }

    // =========================================================================
    // Spectral Kurtosis Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_spectral_kurtosis_pure_tone() {
        let calculator = SpectralKurtosisCalculator::new(48000);
        let tone = generate_sine_wave(1000.0, 48000, 0.1);
        let kurtosis = calculator.calculate(&tone);
        // Pure tone should have high positive kurtosis (sharp peak)
        assert!(kurtosis > 0.0);
    }

    #[test]
    fn test_spectral_kurtosis_white_noise() {
        let calculator = SpectralKurtosisCalculator::new(48000);
        let noise = generate_white_noise(48000, 0.1);
        let kurtosis = calculator.calculate(&noise);
        // White noise should have lower kurtosis (flat distribution)
        assert!(kurtosis < 10.0);
    }

    #[test]
    fn test_spectral_kurtosis_empty() {
        let calculator = SpectralKurtosisCalculator::new(48000);
        let empty: Vec<f32> = vec![];
        let kurtosis = calculator.calculate(&empty);
        assert_eq!(kurtosis, 0.0);
    }

    #[test]
    fn test_spectral_kurtosis_silence() {
        let calculator = SpectralKurtosisCalculator::new(48000);
        let silence = vec![0.0; 4800];
        let kurtosis = calculator.calculate(&silence);
        assert_eq!(kurtosis, 0.0);
    }

    #[test]
    fn test_spectral_kurtosis_multi_tone() {
        let calculator = SpectralKurtosisCalculator::new(48000);
        let mut audio = vec![0.0; 4800];
        for freq in &[500.0, 1000.0, 1500.0, 2000.0] {
            for (i, sample) in audio.iter_mut().enumerate() {
                let t = i as f32 / 48000.0;
                *sample += (2.0 * PI * freq * t).sin();
            }
        }
        let kurtosis = calculator.calculate(&audio);
        // Multiple tones should give intermediate kurtosis
        assert!(kurtosis.is_finite());
    }

    #[test]
    fn test_spectral_kurtosis_range() {
        let calculator = SpectralKurtosisCalculator::new(48000);
        let tone = generate_sine_wave(440.0, 48000, 0.1);
        let kurtosis = calculator.calculate(&tone);
        assert!(kurtosis.is_finite());
    }

    // =========================================================================
    // Spectral Flatness Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_spectral_flatness_pure_tone() {
        let calculator = SpectralFlatnessCalculator::new(48000);
        let tone = generate_sine_wave(1000.0, 48000, 0.1);
        let flatness = calculator.calculate(&tone);
        // Pure tone should have very low flatness (single peak)
        assert!(flatness < 0.3);
    }

    #[test]
    fn test_spectral_flatness_white_noise() {
        let calculator = SpectralFlatnessCalculator::new(48000);
        let noise = generate_white_noise(48000, 0.1);
        let flatness = calculator.calculate(&noise);
        // White noise should have high flatness (close to 1.0)
        assert!(flatness > 0.5);
    }

    #[test]
    fn test_spectral_flatness_empty() {
        let calculator = SpectralFlatnessCalculator::new(48000);
        let empty: Vec<f32> = vec![];
        let flatness = calculator.calculate(&empty);
        assert_eq!(flatness, 0.0);
    }

    #[test]
    fn test_spectral_flatness_silence() {
        let calculator = SpectralFlatnessCalculator::new(48000);
        let silence = vec![0.0; 4800];
        let flatness = calculator.calculate(&silence);
        assert_eq!(flatness, 0.0);
    }

    #[test]
    fn test_spectral_flatness_range() {
        let calculator = SpectralFlatnessCalculator::new(48000);
        let tone = generate_sine_wave(500.0, 48000, 0.1);
        let flatness = calculator.calculate(&tone);
        assert!(flatness >= 0.0 && flatness <= 1.0);
    }

    #[test]
    fn test_spectral_flatness_harmonic() {
        let calculator = SpectralFlatnessCalculator::new(48000);
        let mut audio = vec![0.0; 4800];
        for h in 1..=5 {
            let freq = 440.0 * h as f32;
            for (i, sample) in audio.iter_mut().enumerate() {
                let t = i as f32 / 48000.0;
                *sample += (2.0 * PI * freq * t).sin() / h as f32;
            }
        }
        let flatness = calculator.calculate(&audio);
        // Harmonic series should have low flatness (peaked spectrum)
        assert!(flatness < 0.5);
    }
}

/// Helper: Generate sine wave
fn generate_sine_wave(freq_hz: f32, sample_rate: u32, duration_sec: f32) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * PI * freq_hz * t).sin()
        })
        .collect()
}

/// Helper: Generate white noise
fn generate_white_noise(sample_rate: u32, duration_sec: f32) -> Vec<f32> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let mut rng: u32 = seed;

    (0..num_samples)
        .map(|_| {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Helper: Generate sawtooth wave
fn generate_sawtooth(freq_hz: f32, sample_rate: u32, duration_sec: f32) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    let period_samples = sample_rate as f32 / freq_hz;

    (0..num_samples)
        .map(|i| {
            let phase = (i as f32 % period_samples) / period_samples;
            2.0 * (phase - 0.5) // Sawtooth: -1 to +1
        })
        .collect()
}

/// Helper: Generate square wave
fn generate_square(freq_hz: f32, sample_rate: u32, duration_sec: f32) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    let period_samples = sample_rate as f32 / freq_hz;

    (0..num_samples)
        .map(|i| {
            let phase = (i as f32 % period_samples) / period_samples;
            if phase < 0.5 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}
