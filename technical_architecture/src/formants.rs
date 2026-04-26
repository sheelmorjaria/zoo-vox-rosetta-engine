//! Formant analysis features
//!
//! This module provides formant-related features including formant frequency
//! extraction and spectral peak analysis. Formants are resonant frequencies
//! of the vocal tract that shape the timbre of vocalizations.

use std::f32::consts::PI;

/// Formant frequencies - the peaks in the spectral envelope.
///
/// Formants correspond to resonant frequencies of the vocal tract and are
/// crucial for timbre and sound quality. Unlike MFCCs (which are compressed),
/// formant frequencies represent the actual physical peaks in the spectrum.
///
/// ## Algorithm
/// 1. Compute smoothed power spectrum
/// 2. Find local maxima (peaks)
/// 3. Sort by magnitude and return top N
/// 4. Optional: Apply LPC for better formant tracking
///
/// ## Use Cases
/// - Distinguishes vocal tract shapes across species
/// - Enables formant-based filtering for synthesis
/// - Correlates with vowel quality in some species
///
/// ## Synthesis Power
/// You can set a bandpass filter exactly at a formant frequency to "shape" a sound,
/// making this valuable for concatenative synthesis and voice transformation.
#[derive(Debug, Clone, PartialEq)]
pub struct FormantExtractor {
    pub sample_rate: u32,
    pub num_formants: usize,
    pub smoothing_width: usize,
}

impl Default for FormantExtractor {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            num_formants: 3,
            smoothing_width: 5,
        }
    }
}

impl FormantExtractor {
    pub fn new(sample_rate: u32, num_formants: usize) -> Self {
        Self {
            sample_rate,
            num_formants: num_formants.max(1).min(10),
            smoothing_width: 5,
        }
    }

    /// Extract formant frequencies in Hz
    ///
    /// Returns a vector of formant frequencies sorted by magnitude (strongest first)
    pub fn extract(&self, audio: &[f32]) -> Vec<f32> {
        if audio.len() < self.sample_rate as usize / 100 {
            return vec![0.0; self.num_formants];
        }

        // Compute power spectrum
        let spectrum = self.compute_spectrum(audio);

        if spectrum.is_empty() {
            return vec![0.0; self.num_formants];
        }

        // Smooth spectrum to reduce noise
        let smoothed = self.smooth_spectrum(&spectrum);

        // Find peaks
        let peaks = self.find_peaks(&smoothed);

        // Sort by magnitude and take top N
        let mut sorted_peaks: Vec<(usize, f32)> = peaks.into_iter().collect();
        sorted_peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Convert bin indices to frequencies
        let nyquist = self.sample_rate as f32 / 2.0;
        let bin_resolution = nyquist / spectrum.len() as f32;

        let mut formants = Vec::new();
        for (bin, _magnitude) in sorted_peaks.into_iter().take(self.num_formants) {
            let freq = bin as f32 * bin_resolution;
            formants.push(freq);
        }

        // Pad with zeros if needed
        while formants.len() < self.num_formants {
            formants.push(0.0);
        }

        formants
    }

    /// Extract formant frequencies with magnitudes
    pub fn extract_with_magnitudes(&self, audio: &[f32]) -> Vec<(f32, f32)> {
        if audio.len() < self.sample_rate as usize / 100 {
            return vec![(0.0, 0.0); self.num_formants];
        }

        let spectrum = self.compute_spectrum(audio);

        if spectrum.is_empty() {
            return vec![(0.0, 0.0); self.num_formants];
        }

        let smoothed = self.smooth_spectrum(&spectrum);
        let peaks = self.find_peaks(&smoothed);

        let mut sorted_peaks: Vec<(usize, f32)> = peaks.into_iter().collect();
        sorted_peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let nyquist = self.sample_rate as f32 / 2.0;
        let bin_resolution = nyquist / spectrum.len() as f32;

        let mut formants = Vec::new();
        for (bin, magnitude) in sorted_peaks.into_iter().take(self.num_formants) {
            let freq = bin as f32 * bin_resolution;
            formants.push((freq, magnitude));
        }

        while formants.len() < self.num_formants {
            formants.push((0.0, 0.0));
        }

        formants
    }

    /// Compute magnitude spectrum
    fn compute_spectrum(&self, audio: &[f32]) -> Vec<f32> {
        let n = audio.len().next_power_of_two();
        if n < 4 {
            return Vec::new();
        }

        let mut padded = vec![0.0f32; n];
        padded[..audio.len().min(n)].copy_from_slice(&audio[..audio.len().min(n)]);

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

        complex[..n / 2].iter().map(|&(r, i)| (r * r + i * i).sqrt()).collect()
    }

    /// Smooth spectrum using moving average
    fn smooth_spectrum(&self, spectrum: &[f32]) -> Vec<f32> {
        let width = self.smoothing_width;
        let mut smoothed = Vec::with_capacity(spectrum.len());

        for i in 0..spectrum.len() {
            let start = i.saturating_sub(width / 2);
            let end = (i + width / 2 + 1).min(spectrum.len());

            let sum: f32 = spectrum[start..end].iter().sum();
            let count = (end - start) as f32;
            smoothed.push(sum / count);
        }

        smoothed
    }

    /// Find peaks in spectrum
    fn find_peaks(&self, spectrum: &[f32]) -> Vec<(usize, f32)> {
        let mut peaks = Vec::new();

        // Skip very low frequencies (DC to 100 Hz)
        let min_bin = ((100.0 / (self.sample_rate as f32 / 2.0)) * spectrum.len() as f32) as usize;
        let min_bin = min_bin.min(5);

        for i in min_bin..spectrum.len() - 1 {
            let is_peak = spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1];

            if is_peak && spectrum[i] > 0.01 {
                // Parabolic interpolation for sub-bin accuracy
                let y_prev = spectrum[i - 1];
                let y_curr = spectrum[i];
                let y_next = spectrum[i + 1];

                let denominator = 2.0 * (y_prev - 2.0 * y_curr + y_next);

                let offset = if denominator.abs() > 1e-10 {
                    (y_prev - y_next) / denominator
                } else {
                    0.0
                }
                .clamp(-0.5, 0.5);

                let interpolated_bin = i as f32 + offset;
                let interpolated_mag = y_curr - 0.25 * (y_prev - y_next) * offset;

                peaks.push((interpolated_bin as usize, interpolated_mag));
            }
        }

        peaks
    }
}

/// Formant bandwidth - the width of formant peaks.
///
/// Narrow bandwidth = clear, resonant peak (high quality)
/// Wide bandwidth = damped, diffuse peak (low quality)
///
/// This can indicate:
/// - Vocal tract damping (e.g., nasalization)
/// - Recording environment acoustics
/// - Sound source characteristics
#[derive(Debug, Clone, PartialEq)]
pub struct FormantBandwidthCalculator {
    pub sample_rate: u32,
    pub num_formants: usize,
}

impl Default for FormantBandwidthCalculator {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            num_formants: 3,
        }
    }
}

impl FormantBandwidthCalculator {
    pub fn new(sample_rate: u32, num_formants: usize) -> Self {
        Self {
            sample_rate,
            num_formants: num_formants.max(1).min(10),
        }
    }

    /// Calculate formant bandwidths in Hz
    pub fn calculate(&self, audio: &[f32]) -> Vec<f32> {
        let extractor = FormantExtractor::new(self.sample_rate, self.num_formants);
        let formants = extractor.extract_with_magnitudes(audio);

        if formants.is_empty() {
            return vec![0.0; self.num_formants];
        }

        let spectrum = extractor.compute_spectrum(audio);
        if spectrum.is_empty() {
            return vec![0.0; self.num_formants];
        }

        let nyquist = self.sample_rate as f32 / 2.0;
        let bin_resolution = nyquist / spectrum.len() as f32;

        let mut bandwidths = Vec::new();

        for (freq, _mag) in formants.iter().take(self.num_formants) {
            if *freq < 100.0 {
                bandwidths.push(0.0);
                continue;
            }

            let center_bin = (*freq / bin_resolution) as usize;

            // Find -3 dB points on each side of the peak
            let peak_mag = spectrum.get(center_bin).unwrap_or(&0.0);
            let half_power = *peak_mag / 2.0_f32.sqrt();

            // Find lower -3 dB point
            let mut lower_bin = center_bin;
            for i in (0..center_bin).rev() {
                if spectrum[i] <= half_power {
                    lower_bin = i;
                    break;
                }
            }

            // Find upper -3 dB point
            let mut upper_bin = center_bin;
            for i in center_bin + 1..spectrum.len() {
                if spectrum[i] <= half_power {
                    upper_bin = i;
                    break;
                }
            }

            let bandwidth_hz = (upper_bin - lower_bin) as f32 * bin_resolution;
            bandwidths.push(bandwidth_hz);
        }

        while bandwidths.len() < self.num_formants {
            bandwidths.push(0.0);
        }

        bandwidths
    }
}

/// Helper: Generate sine wave
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

/// Helper: Generate harmonic series
#[cfg(test)]
fn generate_harmonic_series(f0: f32, num_harmonics: usize, sample_rate: u32, duration_sec: f32) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    let mut audio = vec![0.0; num_samples];

    for h in 1..=num_harmonics {
        let freq = f0 * h as f32;
        let amplitude = 1.0 / h as f32;
        for (i, sample) in audio.iter_mut().enumerate() {
            let t = i as f32 / sample_rate as f32;
            *sample += amplitude * (2.0 * PI * freq * t).sin();
        }
    }

    audio
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Formant Extraction Tests (8 tests)
    // =========================================================================

    #[test]
    fn test_formant_extraction_pure_tone() {
        let extractor = FormantExtractor::new(48000, 3);
        let tone = generate_sine_wave(1000.0, 48000, 0.1);
        let formants = extractor.extract(&tone);

        assert_eq!(formants.len(), 3);
        // First formant should be near 1000 Hz
        assert!((formants[0] - 1000.0).abs() < 100.0);
    }

    #[test]
    fn test_formant_extraction_multi_tone() {
        let extractor = FormantExtractor::new(48000, 3);
        // Three tones at distinct frequencies
        let mut audio = vec![0.0; 4800];
        for freq in &[500.0, 1500.0, 3000.0] {
            for (i, sample) in audio.iter_mut().enumerate() {
                let t = i as f32 / 48000.0;
                *sample += (2.0 * PI * freq * t).sin();
            }
        }

        let formants = extractor.extract(&audio);

        // Should detect peaks near our test frequencies
        assert!(formants.iter().any(|&f| (f - 500.0).abs() < 200.0));
        assert!(formants.iter().any(|&f| (f - 1500.0).abs() < 200.0));
    }

    #[test]
    fn test_formant_extraction_harmonic_series() {
        let extractor = FormantExtractor::new(48000, 3);
        let harmonic = generate_harmonic_series(220.0, 5, 48000, 0.1);
        let formants = extractor.extract(&harmonic);

        // First formant should be near fundamental
        assert!((formants[0] - 220.0).abs() < 100.0);
    }

    #[test]
    fn test_formant_extraction_empty() {
        let extractor = FormantExtractor::new(48000, 3);
        let empty: Vec<f32> = vec![];
        let formants = extractor.extract(&empty);
        assert_eq!(formants, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_formant_extraction_silence() {
        let extractor = FormantExtractor::new(48000, 3);
        let silence = vec![0.0; 4800];
        let formants = extractor.extract(&silence);
        // All formants should be 0
        assert!(formants.iter().all(|&f| f == 0.0));
    }

    #[test]
    fn test_formant_extraction_with_magnitudes() {
        let extractor = FormantExtractor::new(48000, 2);
        let tone = generate_sine_wave(1000.0, 48000, 0.1);
        let formants = extractor.extract_with_magnitudes(&tone);

        assert_eq!(formants.len(), 2);
        // First formant should have non-zero magnitude
        assert!(formants[0].1 > 0.0);
    }

    #[test]
    fn test_formant_extraction_num_formants() {
        let extractor = FormantExtractor::new(48000, 5);
        let tone = generate_sine_wave(1000.0, 48000, 0.1);
        let formants = extractor.extract(&tone);

        assert_eq!(formants.len(), 5);
    }

    #[test]
    fn test_formant_extraction_custom_smoothing() {
        let mut extractor = FormantExtractor::new(48000, 3);
        extractor.smoothing_width = 3;
        let tone = generate_sine_wave(1000.0, 48000, 0.1);
        let formants = extractor.extract(&tone);

        assert_eq!(formants.len(), 3);
        assert!(formants[0] > 0.0);
    }

    // =========================================================================
    // Formant Bandwidth Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_formant_bandwidth_pure_tone() {
        let calculator = FormantBandwidthCalculator::new(48000, 2);
        let tone = generate_sine_wave(1000.0, 48000, 0.1);
        let bandwidths = calculator.calculate(&tone);

        assert_eq!(bandwidths.len(), 2);
        // Pure tone should have relatively narrow bandwidth
        assert!(bandwidths[0] < 500.0);
    }

    #[test]
    fn test_formant_bandwidth_empty() {
        let calculator = FormantBandwidthCalculator::new(48000, 2);
        let empty: Vec<f32> = vec![];
        let bandwidths = calculator.calculate(&empty);
        assert_eq!(bandwidths, vec![0.0, 0.0]);
    }

    #[test]
    fn test_formant_bandwidth_silence() {
        let calculator = FormantBandwidthCalculator::new(48000, 2);
        let silence = vec![0.0; 4800];
        let bandwidths = calculator.calculate(&silence);
        assert_eq!(bandwidths, vec![0.0, 0.0]);
    }

    #[test]
    fn test_formant_bandwidth_range() {
        let calculator = FormantBandwidthCalculator::new(48000, 3);
        let tone = generate_sine_wave(500.0, 48000, 0.1);
        let bandwidths = calculator.calculate(&tone);

        assert_eq!(bandwidths.len(), 3);
        // Bandwidths should be non-negative
        assert!(bandwidths.iter().all(|&b| b >= 0.0));
    }

    #[test]
    fn test_formant_bandwidth_multi_tone() {
        let calculator = FormantBandwidthCalculator::new(48000, 3);
        let mut audio = vec![0.0; 4800];
        for freq in &[500.0, 1000.0] {
            for (i, sample) in audio.iter_mut().enumerate() {
                let t = i as f32 / 48000.0;
                *sample += (2.0 * PI * freq * t).sin();
            }
        }

        let bandwidths = calculator.calculate(&audio);
        assert_eq!(bandwidths.len(), 3);
    }

    #[test]
    fn test_formant_bandwidth_harmonic() {
        let calculator = FormantBandwidthCalculator::new(48000, 2);
        let harmonic = generate_harmonic_series(440.0, 3, 48000, 0.1);
        let bandwidths = calculator.calculate(&harmonic);

        assert_eq!(bandwidths.len(), 2);
        assert!(bandwidths[0] >= 0.0);
    }
}
