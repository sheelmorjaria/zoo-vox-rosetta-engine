//! Harmonic analysis features
//!
//! This module provides features related to harmonic structure, including
//! harmonic deviation, inharmonicity, and harmonic-to-noise relationships.

use std::f32::consts::PI;

/// Harmonic deviation - measures how much harmonics deviate from perfect integer ratios.
///
/// Perfect harmonics are at integer multiples of F0 (1.0×, 2.0×, 3.0×, etc.)
/// Inharmonicity causes roughness and is characteristic of:
/// - Corvid "rough" vocalizations
/// - Certain bat calls with non-linear propagation effects
/// - Distorted or strained vocalizations
///
/// ## Algorithm
/// 1. Estimate fundamental frequency (F0)
/// 2. Detect harmonic peaks in spectrum
/// 3. Measure deviation from integer multiples
/// 4. Return mean absolute deviation as ratio
///
/// ## Use Cases
/// - Distinguishes "pure" harmonic calls from "rough" calls
/// - Corvid "roughness" is often due to inharmonicity, not just noise
/// - Identifies vocal strain or distortion
///
/// ## Interpretation
/// - 0.0 = perfect harmonics (pure tone)
/// - 0.01-0.03 = slight inharmonicity (normal for biological sounds)
/// - >0.05 = significant inharmonicity (rough sound)
#[derive(Debug, Clone, PartialEq)]
pub struct HarmonicDeviationCalculator {
    pub sample_rate: u32,
    pub max_harmonics: usize,
}

impl Default for HarmonicDeviationCalculator {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            max_harmonics: 10,
        }
    }
}

impl HarmonicDeviationCalculator {
    pub fn new(sample_rate: u32, max_harmonics: usize) -> Self {
        Self {
            sample_rate,
            max_harmonics: max_harmonics.max(2),
        }
    }

    /// Calculate harmonic deviation [0, 1]
    ///
    /// Returns the mean absolute deviation from perfect integer ratios
    pub fn calculate(&self, audio: &[f32]) -> f32 {
        if audio.len() < self.sample_rate as usize / 100 {
            return 0.0;
        }

        // Estimate F0 using autocorrelation
        let f0 = self.estimate_f0(audio);

        if f0 < 50.0 || f0 > self.sample_rate as f32 / 4.0 {
            return 0.0;
        }

        // Compute spectrum
        let spectrum = self.compute_spectrum(audio);

        if spectrum.is_empty() {
            return 0.0;
        }

        // Find harmonic peaks and measure deviation
        let nyquist = self.sample_rate as f32 / 2.0;
        let bin_resolution = nyquist / spectrum.len() as f32;

        let mut deviations = Vec::new();

        for h in 2..=self.max_harmonics.min(20) {
            let expected_freq = f0 * h as f32;

            if expected_freq > nyquist * 0.95 {
                break; // Too close to Nyquist
            }

            // Find peak near expected harmonic frequency
            let expected_bin = (expected_freq / bin_resolution) as usize;
            let search_range = (expected_freq / f0 * 0.1 / bin_resolution) as usize;
            let start = expected_bin.saturating_sub(search_range);
            let end = (expected_bin + search_range + 1).min(spectrum.len());

            let peak_bin = (start..end)
                .into_iter()
                .max_by_key(|&i| spectrum[i] as i64)
                .unwrap_or(expected_bin);

            let actual_freq = peak_bin as f32 * bin_resolution;

            // Calculate deviation from perfect integer ratio
            let actual_ratio = actual_freq / f0;
            let expected_ratio = h as f32;
            let deviation = (actual_ratio - expected_ratio).abs() / expected_ratio;

            deviations.push(deviation);
        }

        if deviations.is_empty() {
            return 0.0;
        }

        // Return mean absolute deviation
        let sum_dev: f32 = deviations.iter().sum();
        sum_dev / deviations.len() as f32
    }

    /// Estimate F0 using autocorrelation
    fn estimate_f0(&self, audio: &[f32]) -> f32 {
        if audio.len() < 100 {
            return 0.0;
        }

        let min_period = (self.sample_rate as f32 / 1000.0) as usize; // Max 1kHz
        let max_period = (self.sample_rate as f32 / 50.0) as usize; // Min 50Hz

        let max_period = max_period.min(audio.len() / 2);

        let mut best_corr = 0.0_f32;
        let mut best_period = min_period;

        for period in min_period..=max_period {
            let mut correlation = 0.0_f32;

            for i in 0..(audio.len() - period) {
                correlation += audio[i] * audio[i + period];
            }

            // Normalize
            correlation /= (audio.len() - period) as f32;

            if correlation > best_corr {
                best_corr = correlation;
                best_period = period;
            }
        }

        if best_corr < 0.01 {
            return 0.0;
        }

        self.sample_rate as f32 / best_period as f32
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

/// Inharmonicity - measures the degree to which partials are NOT harmonic.
///
/// Similar to harmonic deviation but uses a different metric based on
/// the spread of energy around harmonic frequencies.
///
/// ## Algorithm
/// Uses the "spectral dispersion" method: measures energy in between
/// expected harmonic frequencies vs. energy at harmonic frequencies.
///
/// ## Interpretation
/// - 0.0 = perfectly harmonic
/// - 1.0 = completely inharmonic (like a bell or cymbal)
#[derive(Debug, Clone, PartialEq)]
pub struct InharmonicityCalculator {
    pub sample_rate: u32,
}

impl Default for InharmonicityCalculator {
    fn default() -> Self {
        Self { sample_rate: 48000 }
    }
}

impl InharmonicityCalculator {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Calculate inharmonicity [0, 1]
    pub fn calculate(&self, audio: &[f32]) -> f32 {
        if audio.len() < self.sample_rate as usize / 100 {
            return 0.0;
        }

        let harm_calc = HarmonicDeviationCalculator::new(self.sample_rate, 10);
        let f0 = harm_calc.estimate_f0(audio);

        if f0 < 50.0 {
            return 0.0;
        }

        let spectrum = harm_calc.compute_spectrum(audio);

        if spectrum.is_empty() {
            return 0.0;
        }

        let nyquist = self.sample_rate as f32 / 2.0;
        let bin_resolution = nyquist / spectrum.len() as f32;

        let mut harmonic_energy = 0.0_f32;
        let mut inter_harmonic_energy = 0.0_f32;

        // Analyze frequency range up to 5 kHz
        let max_freq = (5000.0_f32).min(nyquist);
        let num_bins = (max_freq / bin_resolution) as usize;

        for h in 1..=20 {
            let harmonic_freq = f0 * h as f32;

            if harmonic_freq > max_freq {
                break;
            }

            let harmonic_bin = (harmonic_freq / bin_resolution) as usize;

            // Energy at harmonic (narrow band)
            let width = (f0 / bin_resolution * 0.2) as usize; // ±20% of F0
            let start = harmonic_bin.saturating_sub(width);
            let end = (harmonic_bin + width + 1).min(num_bins);

            let h_energy: f32 = spectrum[start..end].iter().map(|&x| x * x).sum();

            // Energy between harmonics (midpoint to next harmonic)
            let next_harmonic_freq = f0 * (h + 1) as f32;
            let mid_freq = (harmonic_freq + next_harmonic_freq) / 2.0;

            if mid_freq < max_freq {
                let mid_bin = (mid_freq / bin_resolution) as usize;
                let mid_width = width;
                let mid_start = mid_bin.saturating_sub(mid_width);
                let mid_end = (mid_bin + mid_width + 1).min(num_bins);

                let ih_energy: f32 = spectrum[mid_start..mid_end].iter().map(|&x| x * x).sum();

                harmonic_energy += h_energy;
                inter_harmonic_energy += ih_energy;
            }
        }

        let total_energy = harmonic_energy + inter_harmonic_energy;

        if total_energy < 1e-10 {
            return 0.0;
        }

        // Inharmonicity = inter-harmonic / total
        inter_harmonic_energy / total_energy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Harmonic Deviation Tests (8 tests)
    // =========================================================================

    #[test]
    fn test_harmonic_deviation_pure_sine() {
        let calculator = HarmonicDeviationCalculator::default();
        let pure = generate_sine_wave(440.0, 48000, 0.1);
        let deviation = calculator.calculate(&pure);
        // Pure sine has no harmonics to measure
        assert!(deviation >= 0.0);
    }

    #[test]
    fn test_harmonic_deviation_perfect_harmonics() {
        let calculator = HarmonicDeviationCalculator::new(48000, 5);
        // Perfect harmonic series
        let harmonic = generate_harmonic_series(220.0, 5, 48000, 0.1);
        let deviation = calculator.calculate(&harmonic);
        // Should have low deviation (perfect harmonics)
        assert!(deviation < 0.1);
    }

    #[test]
    fn test_harmonic_deviation_inharmonic() {
        let calculator = HarmonicDeviationCalculator::new(48000, 5);
        // Inharmonic series (slightly detuned harmonics)
        let inharmonic = generate_inharmonic_series(220.0, 5, 0.02, 48000, 0.1);
        let deviation = calculator.calculate(&inharmonic);
        // Should have higher deviation than perfect harmonics
        assert!(deviation > 0.0);
    }

    #[test]
    fn test_harmonic_deviation_empty() {
        let calculator = HarmonicDeviationCalculator::default();
        let empty: Vec<f32> = vec![];
        let deviation = calculator.calculate(&empty);
        assert_eq!(deviation, 0.0);
    }

    #[test]
    fn test_harmonic_deviation_silence() {
        let calculator = HarmonicDeviationCalculator::default();
        let silence = vec![0.0; 4800];
        let deviation = calculator.calculate(&silence);
        assert_eq!(deviation, 0.0);
    }

    #[test]
    fn test_harmonic_deviation_white_noise() {
        let calculator = HarmonicDeviationCalculator::default();
        let noise = generate_white_noise(48000, 0.1);
        let deviation = calculator.calculate(&noise);
        // Noise should give some deviation value
        assert!(deviation >= 0.0 && deviation <= 1.0);
    }

    #[test]
    fn test_harmonic_deviation_low_frequency() {
        let calculator = HarmonicDeviationCalculator::default();
        let low = generate_harmonic_series(100.0, 3, 48000, 0.1);
        let deviation = calculator.calculate(&low);
        assert!(deviation.is_finite());
    }

    #[test]
    fn test_harmonic_deviation_high_frequency() {
        let calculator = HarmonicDeviationCalculator::default();
        let high = generate_harmonic_series(2000.0, 3, 48000, 0.1);
        let deviation = calculator.calculate(&high);
        assert!(deviation.is_finite());
    }

    // =========================================================================
    // Inharmonicity Tests (8 tests)
    // =========================================================================

    #[test]
    fn test_inharmonicity_pure_tone() {
        let calculator = InharmonicityCalculator::new(48000);
        let pure = generate_sine_wave(440.0, 48000, 0.1);
        let inharm = calculator.calculate(&pure);
        // Pure tone has no inter-harmonic energy
        assert!(inharm < 0.3);
    }

    #[test]
    fn test_inharmonicity_perfect_harmonics() {
        let calculator = InharmonicityCalculator::new(48000);
        let harmonic = generate_harmonic_series(220.0, 5, 48000, 0.1);
        let inharm = calculator.calculate(&harmonic);
        // Perfect harmonics should have low inharmonicity
        assert!(inharm < 0.2);
    }

    #[test]
    fn test_inharmonicity_bell_like() {
        let calculator = InharmonicityCalculator::new(48000);
        // Bell-like: inharmonic partials
        let inharmonic = generate_inharmonic_series(440.0, 8, 0.05, 48000, 0.1);
        let inharm = calculator.calculate(&inharmonic);
        // Inharmonic series should have higher inharmonicity
        assert!(inharm > 0.0);
    }

    #[test]
    fn test_inharmonicity_white_noise() {
        let calculator = InharmonicityCalculator::new(48000);
        let noise = generate_white_noise(48000, 0.1);
        let inharm = calculator.calculate(&noise);
        // Noise is completely inharmonic
        assert!(inharm > 0.3);
    }

    #[test]
    fn test_inharmonicity_empty() {
        let calculator = InharmonicityCalculator::new(48000);
        let empty: Vec<f32> = vec![];
        let inharm = calculator.calculate(&empty);
        assert_eq!(inharm, 0.0);
    }

    #[test]
    fn test_inharmonicity_silence() {
        let calculator = InharmonicityCalculator::new(48000);
        let silence = vec![0.0; 4800];
        let inharm = calculator.calculate(&silence);
        assert_eq!(inharm, 0.0);
    }

    #[test]
    fn test_inharmonicity_range() {
        let calculator = InharmonicityCalculator::new(48000);
        let harmonic = generate_harmonic_series(440.0, 5, 48000, 0.1);
        let inharm = calculator.calculate(&harmonic);
        assert!(inharm >= 0.0 && inharm <= 1.0);
    }

    #[test]
    fn test_inharmonicity_two_tones() {
        let calculator = InharmonicityCalculator::new(48000);
        // Two non-harmonically related tones
        let mut audio = vec![0.0; 4800];
        for (i, sample) in audio.iter_mut().enumerate() {
            let t = i as f32 / 48000.0;
            *sample = 0.5 * (2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 523.25 * t).sin();
            // 440 Hz + C5 (not harmonic)
        }
        let inharm = calculator.calculate(&audio);
        // Two non-harmonic tones should have some inharmonicity
        assert!(inharm > 0.0);
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

/// Helper: Generate perfect harmonic series
fn generate_harmonic_series(
    f0: f32,
    num_harmonics: usize,
    sample_rate: u32,
    duration_sec: f32,
) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    let mut audio = vec![0.0; num_samples];

    for h in 1..=num_harmonics {
        let freq = f0 * h as f32;
        let amplitude = 1.0 / h as f32; // Natural amplitude decay
        for (i, sample) in audio.iter_mut().enumerate() {
            let t = i as f32 / sample_rate as f32;
            *sample += amplitude * (2.0 * PI * freq * t).sin();
        }
    }

    audio
}

/// Helper: Generate inharmonic series (detuned harmonics)
fn generate_inharmonic_series(
    f0: f32,
    num_partials: usize,
    detune: f32,
    sample_rate: u32,
    duration_sec: f32,
) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    let mut audio = vec![0.0; num_samples];

    for h in 1..=num_partials {
        // Add slight detuning to create inharmonicity
        let freq = f0 * h as f32 * (1.0 + detune * h as f32);
        let amplitude = 1.0 / h as f32;
        for (i, sample) in audio.iter_mut().enumerate() {
            let t = i as f32 / sample_rate as f32;
            *sample += amplitude * (2.0 * PI * freq * t).sin();
        }
    }

    audio
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
