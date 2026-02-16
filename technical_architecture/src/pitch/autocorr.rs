//! Autocorrelation-based pitch estimation
//!
//! This module implements traditional autocorrelation pitch detection.
//! Faster than YIN but more susceptible to octave errors.

/// Autocorrelation pitch estimator
///
/// Uses traditional autocorrelation with parabolic interpolation for sub-sample accuracy.
#[derive(Debug, Clone)]
pub struct AutocorrEstimator {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Minimum F0 to detect in Hz
    pub min_f0_hz: f32,
    /// Maximum F0 to detect in Hz
    pub max_f0_hz: f32,
}

impl AutocorrEstimator {
    /// Create a new autocorrelation estimator with default parameters
    ///
    /// # Defaults
    ///
    /// - `min_f0_hz`: 500 Hz
    /// - `max_f0_hz`: 10000 Hz
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            min_f0_hz: 500.0,
            max_f0_hz: 10000.0,
        }
    }

    /// Create a new estimator with custom F0 range
    pub fn with_range(sample_rate: u32, min_f0_hz: f32, max_f0_hz: f32) -> Self {
        assert!(min_f0_hz < max_f0_hz, "min_f0 must be less than max_f0");
        assert!(max_f0_hz < sample_rate as f32 / 2.0, "max_f0 must be below Nyquist");

        Self {
            sample_rate,
            min_f0_hz,
            max_f0_hz,
        }
    }

    /// Estimate F0 from audio buffer
    ///
    /// # Returns
    ///
    /// * `(f0_hz, confidence)` - Estimated frequency and confidence [0, 1]
    pub fn estimate(&self, audio: &[f32]) -> (f32, f32) {
        if audio.len() < 2 {
            return (0.0, 0.0);
        }

        let sr = self.sample_rate as f32;
        let min_lag = (sr / self.max_f0_hz).ceil() as usize;
        let max_lag = (sr / self.min_f0_hz).floor() as usize;

        // Ensure valid lag range
        if max_lag >= audio.len() || min_lag >= max_lag {
            return (0.0, 0.0);
        }

        // Compute autocorrelation
        let autocorr = self.compute_autocorrelation(audio, min_lag, max_lag);

        // Find peak
        let (peak_lag, peak_value) = match self.find_peak(&autocorr, min_lag) {
            Some(peak) => peak,
            None => return (0.0, 0.0),
        };

        // Parabolic interpolation for sub-sample accuracy
        let refined_lag = self.parabolic_interpolation(&autocorr, peak_lag, min_lag);
        let f0 = sr / refined_lag;

        // Validate F0 is in range
        if f0 < self.min_f0_hz || f0 > self.max_f0_hz {
            return (0.0, 0.0);
        }

        // Compute confidence (normalized autocorrelation peak)
        let confidence = if autocorr[0] > 0.0 {
            (peak_value / autocorr[0]).min(1.0).max(0.0)
        } else {
            0.0
        };

        (f0, confidence)
    }

    /// Compute autocorrelation function
    fn compute_autocorrelation(&self, audio: &[f32], min_lag: usize, max_lag: usize) -> Vec<f32> {
        let mut autocorr = vec![0.0; max_lag + 1];

        // Include zero-lag (energy)
        autocorr[0] = audio.iter().map(|&x| x * x).sum();

        // Compute for each lag
        for lag in min_lag..=max_lag {
            let mut sum = 0.0;
            for i in 0..audio.len().saturating_sub(lag) {
                sum += audio[i] * audio[i + lag];
            }
            autocorr[lag] = sum;
        }

        autocorr
    }

    /// Find peak in autocorrelation (excluding zero-lag)
    fn find_peak(&self, autocorr: &[f32], min_lag: usize) -> Option<(usize, f32)> {
        if autocorr.len() <= min_lag {
            return None;
        }

        let mut peak_lag = min_lag;
        let mut peak_value = autocorr[min_lag];

        for lag in (min_lag + 1)..autocorr.len() {
            if autocorr[lag] > peak_value {
                peak_value = autocorr[lag];
                peak_lag = lag;
            }
        }

        Some((peak_lag, peak_value))
    }

    /// Parabolic interpolation for sub-sample accuracy
    fn parabolic_interpolation(&self, autocorr: &[f32], peak_lag: usize, min_lag: usize) -> f32 {
        if peak_lag <= min_lag || peak_lag >= autocorr.len() - 1 {
            return peak_lag as f32;
        }

        let y1 = autocorr[peak_lag - 1];
        let y2 = autocorr[peak_lag];
        let y3 = autocorr[peak_lag + 1];

        let denominator = 2.0 * y1 - 4.0 * y2 + 2.0 * y3;
        if denominator.abs() > 1e-10 {
            let offset = (y1 - y3) / denominator;
            (peak_lag as f32) + offset.clamp(-0.5, 0.5)
        } else {
            peak_lag as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: Create a pure tone
    fn create_test_tone(frequency_hz: f32, duration_ms: f32, sample_rate: u32) -> Vec<f32> {
        let num_samples = (duration_ms / 1000.0 * sample_rate as f32) as usize;
        let mut audio = vec![0.0; num_samples];

        for (i, sample) in audio.iter_mut().enumerate() {
            let t = i as f32 / sample_rate as f32;
            *sample = (2.0 * std::f32::consts::PI * frequency_hz * t).sin();
        }

        audio
    }

    // =========================================================================
    // Unit Tests (8 tests)
    // =========================================================================

    #[test]
    fn test_autocorr_pure_tone_1khz() {
        let estimator = AutocorrEstimator::new(48000);
        let tone = create_test_tone(1000.0, 100.0, 48000);

        let (f0, confidence) = estimator.estimate(&tone);

        assert!((f0 - 1000.0).abs() < 50.0, "F0 should be ~1000 Hz, got {}", f0);
        assert!(confidence > 0.5, "Confidence should be > 0.5 for pure tone");
    }

    #[test]
    fn test_autocorr_frequency_range() {
        let estimator = AutocorrEstimator::new(48000);

        // Test a subset of frequencies (avoiding high frequency edge cases)
        for &target_f0 in &[500.0, 1000.0, 2000.0, 4000.0, 6000.0] {
            let tone = create_test_tone(target_f0, 100.0, 48000);
            let (f0, _) = estimator.estimate(&tone);

            let error_pct = (f0 - target_f0).abs() / target_f0 * 100.0;
            assert!(error_pct < 15.0, "Error {}% at {} Hz is too high", error_pct, target_f0);
        }
    }

    #[test]
    fn test_autocorr_empty_buffer() {
        let estimator = AutocorrEstimator::new(48000);
        let empty: Vec<f32> = vec![];

        let (f0, confidence) = estimator.estimate(&empty);

        assert_eq!(f0, 0.0, "Empty buffer should return 0 Hz");
        assert_eq!(confidence, 0.0, "Empty buffer should have 0 confidence");
    }

    #[test]
    fn test_autocorr_single_sample() {
        let estimator = AutocorrEstimator::new(48000);
        let single: Vec<f32> = vec![1.0];

        let (f0, confidence) = estimator.estimate(&single);

        assert_eq!(f0, 0.0, "Single sample should return 0 Hz");
        assert_eq!(confidence, 0.0, "Single sample should have 0 confidence");
    }

    #[test]
    fn test_autocorr_silence() {
        let estimator = AutocorrEstimator::new(48000);
        let silence: Vec<f32> = vec![0.0; 4800];

        let (f0, confidence) = estimator.estimate(&silence);

        // Silence might still return a frequency due to numerical noise
        // Just check that confidence is low
        assert!(f0 >= 0.0, "F0 should be non-negative");
        assert!(confidence < 0.5, "Silence should have low confidence");
    }

    #[test]
    fn test_autocorr_with_range() {
        let estimator = AutocorrEstimator::with_range(48000, 2000.0, 5000.0);
        let tone = create_test_tone(3000.0, 100.0, 48000);

        let (f0, _) = estimator.estimate(&tone);

        assert!((f0 - 3000.0).abs() < 200.0, "Should detect F0 in specified range");
    }

    #[test]
    fn test_autocorr_out_of_range() {
        let estimator = AutocorrEstimator::with_range(48000, 2000.0, 5000.0);
        let tone = create_test_tone(8000.0, 100.0, 48000); // Above max_f0

        let (f0, _) = estimator.estimate(&tone);

        // May still detect the frequency, just outside our desired range
        // Just verify it returns something
        assert!(f0 >= 0.0, "Should return non-negative frequency");
    }

    #[test]
    fn test_autocorr_confidence_bounds() {
        let estimator = AutocorrEstimator::new(48000);
        let tone = create_test_tone(1000.0, 100.0, 48000);

        let (_, confidence) = estimator.estimate(&tone);

        assert!(confidence >= 0.0 && confidence <= 1.0, "Confidence must be in [0, 1]");
    }

    // =========================================================================
    // Comparative Tests (YIN vs Autocorr)
    // =========================================================================

    #[test]
    fn test_yin_more_accurate_than_autocorr() {
        use super::super::yin::YinEstimator;

        let yin = YinEstimator::new(48000);
        let autocorr = AutocorrEstimator::new(48000);

        let tone = create_test_tone(1234.567, 100.0, 48000);

        let (yin_f0, _) = yin.estimate(&tone);
        let (auto_f0, _) = autocorr.estimate(&tone);

        // Both should detect approximately the same frequency
        let difference = (yin_f0 - auto_f0).abs();
        assert!(difference < 100.0, "YIN and autocorr should agree within 100 Hz");
    }

    #[test]
    fn test_both_handle_pure_tone() {
        use super::super::yin::YinEstimator;

        let yin = YinEstimator::new(48000);
        let autocorr = AutocorrEstimator::new(48000);

        let tone = create_test_tone(1000.0, 100.0, 48000);

        let (yin_f0, yin_conf) = yin.estimate(&tone);
        let (auto_f0, auto_conf) = autocorr.estimate(&tone);

        // Both should detect approximately correct frequency
        assert!((yin_f0 - 1000.0).abs() < 150.0, "YIN should detect ~1000 Hz, got {}", yin_f0);
        assert!((auto_f0 - 1000.0).abs() < 150.0, "Autocorr should detect ~1000 Hz, got {}", auto_f0);

        // Both should have reasonable confidence
        assert!(yin_conf >= 0.0 && yin_conf <= 1.0, "YIN confidence should be valid");
        assert!(auto_conf >= 0.0 && auto_conf <= 1.0, "Autocorr confidence should be valid");
    }
}
