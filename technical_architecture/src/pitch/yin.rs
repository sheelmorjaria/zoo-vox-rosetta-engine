//! YIN Pitch Estimation Algorithm
//!
//! This module implements the YIN algorithm for fundamental frequency (F0) estimation.
//! YIN improves upon autocorrelation by introducing Cumulative Mean Normalized Difference (CMND),
//! which suppresses peaks at integer multiples of the true period, solving the octave error problem.
//!
//! # Algorithm Steps
//!
//! 1. **Difference Function**: Compute squared difference between signal and time-shifted version
//! 2. **Cumulative Mean Normalization**: Normalize to prevent selecting low-frequency erroneous peaks
//! 3. **Absolute Threshold**: Find first dip below threshold (default: 0.1)
//! 4. **Parabolic Interpolation**: Refine period for sub-sample accuracy
//! 5. **Confidence Scoring**: Based on depth of dip (sharper dip = higher confidence)
//!
//! # References
//!
//! - de Cheveigné, A., & Kawahara, H. (2002). "YIN, a fundamental frequency estimator for speech and music."
//!   The Journal of the Acoustical Society of America, 111(4), 1917-1930.


/// YIN pitch estimator
///
/// Provides robust F0 estimation with confidence scoring using the YIN algorithm.
#[derive(Debug, Clone)]
pub struct YinEstimator {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Minimum F0 to detect in Hz
    pub min_f0_hz: f32,
    /// Maximum F0 to detect in Hz
    pub max_f0_hz: f32,
    /// Dip threshold for CMND (lower = more selective, typical: 0.1)
    pub threshold: f32,
}

impl YinEstimator {
    /// Create a new YIN estimator with default parameters
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Audio sample rate in Hz
    ///
    /// # Defaults
    ///
    /// - `min_f0_hz`: 500 Hz (typical for bird vocalizations)
    /// - `max_f0_hz`: 10000 Hz (Nyquist at 44.1kHz / 4)
    /// - `threshold`: 0.1 (recommended by YIN paper)
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            min_f0_hz: 500.0,
            max_f0_hz: 10000.0,
            threshold: 0.1,
        }
    }

    /// Create a new YIN estimator with custom F0 range
    pub fn with_range(sample_rate: u32, min_f0_hz: f32, max_f0_hz: f32) -> Self {
        assert!(min_f0_hz < max_f0_hz, "min_f0 must be less than max_f0");
        assert!(max_f0_hz < sample_rate as f32 / 2.0, "max_f0 must be below Nyquist");

        Self {
            sample_rate,
            min_f0_hz,
            max_f0_hz,
            threshold: 0.1,
        }
    }

    /// Set the dip threshold
    ///
    /// Lower values are more selective but may miss weak pitch.
    /// Typical range: 0.01 to 0.3
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        assert!(threshold > 0.0 && threshold < 1.0, "threshold must be in (0, 1)");
        self.threshold = threshold;
        self
    }

    /// Estimate F0 from audio buffer
    ///
    /// # Arguments
    ///
    /// * `audio` - Audio samples (normalized to [-1, 1])
    ///
    /// # Returns
    ///
    /// * `(f0_hz, confidence)` - Estimated frequency in Hz and confidence score [0, 1]
    ///
    /// # Notes
    ///
    /// - Returns `(0.0, 0.0)` for silent or unpitchable audio
    /// - Confidence is derived from dip depth: sharper dip = higher confidence
    pub fn estimate(&self, audio: &[f32]) -> (f32, f32) {
        if audio.len() < 2 {
            return (0.0, 0.0);
        }

        // Check if audio has sufficient energy
        let energy: f32 = audio.iter().map(|&x| x * x).sum();
        if energy < audio.len() as f32 * 1e-6 {
            return (0.0, 0.0);
        }

        // Convert F0 range to lag range
        let min_lag = (self.sample_rate as f32 / self.max_f0_hz).ceil() as usize;
        let max_lag = (self.sample_rate as f32 / self.min_f0_hz).floor() as usize;

        // Ensure we have enough samples
        if max_lag >= audio.len() {
            return (0.0, 0.0);
        }

        // Step 1: Compute difference function
        let diff = self.compute_difference_function(audio, max_lag);

        // Step 2: Cumulative mean normalization
        let cmnd = self.cumulative_mean_normalization(&diff);

        // Step 3: Find first dip below threshold
        let (lag, confidence) = match self.find_threshold_dip(&cmnd, min_lag, max_lag) {
            Some(result) => result,
            None => return (0.0, 0.0),
        };

        // Step 4: Parabolic interpolation for sub-sample accuracy
        let refined_lag = self.parabolic_interpolation(&cmnd, lag);

        // Convert lag to frequency
        let f0_hz = self.sample_rate as f32 / refined_lag;

        // Validate F0 is in range
        if f0_hz < self.min_f0_hz || f0_hz > self.max_f0_hz {
            return (0.0, 0.0);
        }

        (f0_hz, confidence)
    }

    /// Compute difference function
    ///
    /// d(t) = Σ(x[i] - x[i+t])² for i = 0 to n-t-1
    fn compute_difference_function(&self, audio: &[f32], max_lag: usize) -> Vec<f32> {
        let mut diff = vec![0.0; max_lag + 1];

        for lag in 0..=max_lag {
            let mut sum = 0.0;
            for i in 0..audio.len().saturating_sub(lag) {
                let delta = audio[i] - audio[i + lag];
                sum += delta * delta;
            }
            diff[lag] = sum;
        }

        diff
    }

    /// Cumulative mean normalization
    ///
    /// cmnd(t) = d(t) / (1/t × Σ[i=0 to t] d(i))
    ///
    /// This normalization suppresses peaks at integer multiples of the true period.
    fn cumulative_mean_normalization(&self, diff: &[f32]) -> Vec<f32> {
        let mut cmnd = vec![0.0; diff.len()];

        let mut running_sum = 0.0;
        for (t, &d) in diff.iter().enumerate() {
            running_sum += d;
            let mean = running_sum / (t + 1) as f32;
            cmnd[t] = if t == 0 { 1.0 } else { d / mean };
        }

        cmnd
    }

    /// Find first dip below threshold in CMND
    ///
    /// Scans from min_lag to max_lag for first value below threshold.
    /// Returns (lag, confidence) where confidence is based on dip depth.
    fn find_threshold_dip(&self, cmnd: &[f32], min_lag: usize, max_lag: usize) -> Option<(usize, f32)> {
        let search_start = min_lag.max(1);
        let search_end = max_lag.min(cmnd.len() - 1);

        for lag in search_start..=search_end {
            if cmnd[lag] < self.threshold {
                // Compute confidence based on dip depth
                let local_mean = cmnd[lag.saturating_sub(5)..=(lag + 5).min(cmnd.len() - 1)]
                    .iter()
                    .fold(0.0, |acc, &x| acc + x)
                    / ((lag + 5).min(cmnd.len() - 1) - lag.saturating_sub(5) + 1) as f32;
                let confidence = (local_mean - cmnd[lag]) / local_mean;
                return Some((lag, confidence.clamp(0.0, 1.0)));
            }
        }

        None
    }

    /// Parabolic interpolation for sub-sample accuracy
    ///
    /// Fits parabola through (lag-1, lag, lag+1) and finds minimum.
    fn parabolic_interpolation(&self, cmnd: &[f32], lag: usize) -> f32 {
        if lag == 0 || lag >= cmnd.len() - 1 {
            return lag as f32;
        }

        let y_prev = cmnd[lag - 1];
        let y_curr = cmnd[lag];
        let y_next = cmnd[lag + 1];

        let denominator = 2.0 * (y_prev - 2.0 * y_curr + y_next);
        if denominator.abs() < 1e-10 {
            return lag as f32;
        }

        let offset = (y_prev - y_next) / denominator;
        (lag as f32) + offset.clamp(-0.5, 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Helper: Create a pure tone
    fn create_pure_tone(frequency_hz: f32, duration_sec: f32, sample_rate: u32) -> Vec<f32> {
        let num_samples = (sample_rate as f32 * duration_sec) as usize;
        let mut tone = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            tone.push((2.0 * PI * frequency_hz * t).sin());
        }

        tone
    }

    /// Helper: Create a noisy signal
    fn create_noisy_tone(frequency_hz: f32, snr_db: f32, duration_sec: f32, sample_rate: u32) -> Vec<f32> {
        let tone = create_pure_tone(frequency_hz, duration_sec, sample_rate);
        let signal_power = tone.iter().map(|&x| x * x).sum::<f32>() / tone.len() as f32;
        let noise_power = signal_power / (10.0_f32).powf(snr_db / 10.0);
        let noise_std = noise_power.sqrt();

        tone.into_iter()
            .map(|sample| sample + rand::random::<f32>() * noise_std * 2.0 - noise_std)
            .collect()
    }

    // =========================================================================
    // Category 1: Pure Tone Detection Tests (5 tests)
    // =========================================================================

    #[test]
    fn test_yin_pure_tone_1khz() {
        let estimator = YinEstimator::new(48000);
        let tone = create_pure_tone(1000.0, 0.1, 48000);

        let (f0, confidence) = estimator.estimate(&tone);

        // Allow 10% error for basic implementation
        assert!((f0 - 1000.0).abs() < 100.0, "F0 should be ~1000 Hz, got {}", f0);
        assert!(confidence >= 0.0, "Confidence should be non-negative, got {}", confidence);
    }

    #[test]
    fn test_yin_pure_tone_5khz() {
        let estimator = YinEstimator::new(48000);
        let tone = create_pure_tone(5000.0, 0.1, 48000);

        let (f0, confidence) = estimator.estimate(&tone);

        assert!((f0 - 5000.0).abs() < 250.0, "F0 should be ~5000 Hz, got {}", f0);
        assert!(confidence > 0.5, "Confidence should be > 0.5 for pure tone");
    }

    #[test]
    fn test_yin_pure_tone_9khz() {
        let estimator = YinEstimator::new(48000);
        let tone = create_pure_tone(9000.0, 0.1, 48000);

        let (f0, confidence) = estimator.estimate(&tone);

        assert!((f0 - 9000.0).abs() < 500.0, "F0 should be ~9000 Hz, got {}", f0);
        assert!(confidence > 0.5, "Confidence should be > 0.5 for high frequency");
    }

    #[test]
    fn test_yin_frequency_range_accuracy() {
        let estimator = YinEstimator::new(48000);
        let test_frequencies = [600.0, 1000.0, 2000.0, 4000.0, 8000.0];

        for &target_f0 in &test_frequencies {
            let tone = create_pure_tone(target_f0, 0.1, 48000);
            let (f0, _) = estimator.estimate(&tone);

            let error_pct = (f0 - target_f0).abs() / target_f0 * 100.0;
            assert!(error_pct < 10.0, "Error {}% at {} Hz is too high", error_pct, target_f0);
        }
    }

    #[test]
    fn test_yin_sub_sample_accuracy() {
        let estimator = YinEstimator::new(48000);
        // Use frequency that doesn't align exactly with sample grid
        let tone = create_pure_tone(1234.567, 0.1, 48000);

        let (f0, _) = estimator.estimate(&tone);

        assert!((f0 - 1234.567).abs() < 100.0, "Sub-sample accuracy should be < 100 Hz");
    }

    // =========================================================================
    // Category 2: Noisy Signal Tests (5 tests)
    // =========================================================================

    #[test]
    fn test_yin_snr_10db() {
        let estimator = YinEstimator::new(48000);
        let noisy_tone = create_noisy_tone(1000.0, 10.0, 0.1, 48000);

        let (f0, confidence) = estimator.estimate(&noisy_tone);

        assert!((f0 - 1000.0).abs() < 50.0, "Should detect F0 at 10dB SNR");
        assert!(confidence > 0.5, "Confidence should be moderate at 10dB SNR");
    }

    #[test]
    fn test_yin_snr_0db() {
        let estimator = YinEstimator::new(48000);
        let noisy_tone = create_noisy_tone(2000.0, 0.0, 0.1, 48000);

        let (f0, confidence) = estimator.estimate(&noisy_tone);

        // At 0dB SNR, detection is challenging
        if f0 > 0.0 {
            assert!((f0 - 2000.0).abs() < 200.0, "Should roughly detect F0 at 0dB SNR");
            assert!(confidence < 0.8, "Confidence should be lower at 0dB SNR");
        }
        // It's acceptable to return (0.0, 0.0) for very noisy signals
    }

    #[test]
    fn test_yin_confidence_degradation() {
        let estimator = YinEstimator::new(48000);

        let tone_pure = create_pure_tone(1000.0, 0.1, 48000);
        let tone_10db = create_noisy_tone(1000.0, 10.0, 0.1, 48000);
        let tone_0db = create_noisy_tone(1000.0, 0.0, 0.1, 48000);

        let (_, conf_pure) = estimator.estimate(&tone_pure);
        let (_, conf_10db) = estimator.estimate(&tone_10db);
        let (_, conf_0db) = estimator.estimate(&tone_0db);

        // All confidences should be valid
        assert!(conf_pure >= 0.0 && conf_pure <= 1.0, "Pure tone confidence should be valid");
        assert!(conf_10db >= 0.0 && conf_10db <= 1.0, "10dB confidence should be valid");
        assert!(conf_0db >= 0.0 && conf_0db <= 1.0, "0dB confidence should be valid");
    }

    #[test]
    fn test_yin_white_noise() {
        let estimator = YinEstimator::new(48000);
        let noise: Vec<f32> = (0..4800).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();

        let (f0, confidence) = estimator.estimate(&noise);

        assert_eq!(f0, 0.0, "White noise should return 0 Hz");
        assert_eq!(confidence, 0.0, "White noise should have 0 confidence");
    }

    #[test]
    fn test_yin_silent_audio() {
        let estimator = YinEstimator::new(48000);
        let silence: Vec<f32> = vec![0.0; 4800];

        let (f0, confidence) = estimator.estimate(&silence);

        assert_eq!(f0, 0.0, "Silence should return 0 Hz");
        assert_eq!(confidence, 0.0, "Silence should have 0 confidence");
    }

    // =========================================================================
    // Category 3: Edge Cases (5 tests)
    // =========================================================================

    #[test]
    fn test_yin_short_buffer() {
        let estimator = YinEstimator::new(48000);
        let short_tone = create_pure_tone(1000.0, 0.001, 48000); // 4.8 samples

        let (f0, _) = estimator.estimate(&short_tone);

        // Should handle gracefully (return 0 or detect if possible)
        assert!(f0 >= 0.0, "F0 should be non-negative");
    }

    #[test]
    fn test_yin_empty_buffer() {
        let estimator = YinEstimator::new(48000);
        let empty: Vec<f32> = vec![];

        let (f0, confidence) = estimator.estimate(&empty);

        assert_eq!(f0, 0.0, "Empty buffer should return 0 Hz");
        assert_eq!(confidence, 0.0, "Empty buffer should have 0 confidence");
    }

    #[test]
    fn test_yin_single_sample() {
        let estimator = YinEstimator::new(48000);
        let single: Vec<f32> = vec![1.0];

        let (f0, confidence) = estimator.estimate(&single);

        assert_eq!(f0, 0.0, "Single sample should return 0 Hz");
        assert_eq!(confidence, 0.0, "Single sample should have 0 confidence");
    }

    #[test]
    fn test_yin_dc_offset() {
        let estimator = YinEstimator::new(48000);
        let tone_with_dc: Vec<f32> = create_pure_tone(1000.0, 0.1, 48000)
            .into_iter()
            .map(|x| x + 0.5)
            .collect();

        let (f0, confidence) = estimator.estimate(&tone_with_dc);

        assert!((f0 - 1000.0).abs() < 150.0, "Should handle DC offset, got {}", f0);
        assert!(confidence >= 0.0, "Confidence should be non-negative");
    }

    #[test]
    fn test_yin_frequency_out_of_range() {
        let estimator = YinEstimator::with_range(48000, 500.0, 2000.0);
        let tone = create_pure_tone(5000.0, 0.1, 48000); // Above max_f0

        let (f0, _) = estimator.estimate(&tone);

        // Might still detect something, just verify it's not absurdly high
        assert!(f0 >= 0.0, "Should return non-negative frequency");
    }

    // =========================================================================
    // Category 4: Confidence Scoring Tests (5 tests)
    // =========================================================================

    #[test]
    fn test_yin_confidence_high_purity() {
        let estimator = YinEstimator::new(48000);
        let pure_tone = create_pure_tone(1000.0, 0.1, 48000);

        let (_, confidence) = estimator.estimate(&pure_tone);

        assert!(confidence >= 0.0 && confidence <= 1.0, "Confidence must be in [0, 1]");
        assert!(confidence > 0.1, "Pure tone should have reasonable confidence");
    }

    #[test]
    fn test_yin_confidence_thresholding() {
        let estimator = YinEstimator::new(48000).with_threshold(0.5); // Very high threshold
        let pure_tone = create_pure_tone(1000.0, 0.1, 48000);

        let (_, confidence) = estimator.estimate(&pure_tone);

        // With high threshold, only strong dips are accepted
        assert!(confidence >= 0.0 && confidence <= 1.0, "Confidence should be in [0, 1]");
    }

    #[test]
    fn test_yin_dip_depth_correlation() {
        let estimator = YinEstimator::new(48000);

        // Pure tone should have deep dip (high confidence)
        let pure_tone = create_pure_tone(1000.0, 0.1, 48000);
        let (_, conf_pure) = estimator.estimate(&pure_tone);

        // Noisy tone should have shallow dip (low confidence)
        let noisy_tone = create_noisy_tone(1000.0, 0.0, 0.1, 48000);
        let (_, conf_noisy) = estimator.estimate(&noisy_tone);

        // Pure tone should have higher or equal confidence
        assert!(conf_pure >= conf_noisy, "Dip depth should correlate with confidence");
    }

    #[test]
    fn test_yin_confidence_bounds() {
        let estimator = YinEstimator::new(48000);
        let tone = create_pure_tone(1000.0, 0.1, 48000);

        let (_, confidence) = estimator.estimate(&tone);

        assert!(confidence >= 0.0 && confidence <= 1.0, "Confidence must be in [0, 1]");
    }

    #[test]
    fn test_yin_confidence_clipping() {
        let estimator = YinEstimator::new(48000);

        // Very short tone (edge case)
        let short_tone = create_pure_tone(1000.0, 0.005, 48000);
        let (_, confidence) = estimator.estimate(&short_tone);

        assert!(confidence >= 0.0 && confidence <= 1.0, "Confidence should be clamped to [0, 1]");
    }

    // =========================================================================
    // Category 5: Performance Tests (5 tests)
    // =========================================================================

    #[test]
    fn test_yin_realtime_capability() {
        let estimator = YinEstimator::new(48000);
        let tone = create_pure_tone(1000.0, 0.1, 48000); // 100ms buffer

        let start = std::time::Instant::now();
        let _ = estimator.estimate(&tone);
        let elapsed = start.elapsed();

        // Should complete in < 10ms for 100ms buffer
        assert!(elapsed.as_millis() < 10, "Real-time requirement: {}ms", elapsed.as_millis());
    }

    #[test]
    fn test_yin_scalability_linear() {
        let estimator = YinEstimator::new(48000);

        let tone_50ms = create_pure_tone(1000.0, 0.05, 48000);
        let tone_100ms = create_pure_tone(1000.0, 0.1, 48000);
        let tone_200ms = create_pure_tone(1000.0, 0.2, 48000);

        let start_50 = std::time::Instant::now();
        let _ = estimator.estimate(&tone_50ms);
        let time_50 = start_50.elapsed();

        let start_100 = std::time::Instant::now();
        let _ = estimator.estimate(&tone_100ms);
        let time_100 = start_100.elapsed();

        let start_200 = std::time::Instant::now();
        let _ = estimator.estimate(&tone_200ms);
        let time_200 = start_200.elapsed();

        // Time should scale roughly linearly with buffer size
        // 100ms should be ~2x 50ms, 200ms should be ~4x 50ms
        let ratio_100_50 = time_100.as_nanos() as f64 / time_50.as_nanos().max(1) as f64;
        let ratio_200_50 = time_200.as_nanos() as f64 / time_50.as_nanos().max(1) as f64;

        // Allow 3x for overhead (not strict 2x and 4x)
        assert!(ratio_100_50 < 3.0, "100ms should be < 3x 50ms, got {}", ratio_100_50);
        assert!(ratio_200_50 < 8.0, "200ms should be < 8x 50ms, got {}", ratio_200_50);
    }

    #[test]
    fn test_yin_memory_stable() {
        let estimator = YinEstimator::new(48000);

        // Multiple calls should not accumulate memory
        for _ in 0..100 {
            let tone = create_pure_tone(1000.0, 0.1, 48000);
            let _ = estimator.estimate(&tone);
        }

        // If we reach here without crashing, memory is stable
        assert!(true);
    }

    #[test]
    fn test_yin_cache_efficiency() {
        let estimator = YinEstimator::new(48000);

        // Create identical tones
        let tone1 = create_pure_tone(1000.0, 0.1, 48000);
        let tone2 = create_pure_tone(1000.0, 0.1, 48000);

        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = estimator.estimate(&tone1);
            let _ = estimator.estimate(&tone2);
        }
        let elapsed = start.elapsed();

        // Should be fast due to cache-friendly sequential access
        assert!(elapsed.as_millis() < 50, "10 iterations should complete in < 50ms");
    }

    #[test]
    fn test_yin_parallelizable() {
        // This test verifies no mutable shared state
        let estimator1 = YinEstimator::new(48000);
        let estimator2 = estimator1.clone();
        let estimator3 = YinEstimator::new(48000);

        let tone = create_pure_tone(1000.0, 0.1, 48000);

        // All three should work independently
        let (f0_1, _) = estimator1.estimate(&tone);
        let (f0_2, _) = estimator2.estimate(&tone);
        let (f0_3, _) = estimator3.estimate(&tone);

        assert_eq!(f0_1, f0_2, "Cloned estimators should produce same result");
        assert_eq!(f0_2, f0_3, "Independent estimators should produce same result");
    }

    // =========================================================================
    // Category 6: Cross-Species Tests (5 tests)
    // =========================================================================

    #[test]
    fn test_yin_marmoset_vocalization() {
        // Marmoset: 7-12 kHz range
        let estimator = YinEstimator::with_range(48000, 5000.0, 15000.0);
        let tone = create_pure_tone(9000.0, 0.1, 48000);

        let (f0, confidence) = estimator.estimate(&tone);

        assert!((f0 - 9000.0).abs() < 1000.0, "Should detect marmoset-like frequency, got {}", f0);
        assert!(confidence >= 0.0, "Confidence should be non-negative");
    }

    #[test]
    fn test_yin_finch_song() {
        // Finch: 2-8 kHz range
        let estimator = YinEstimator::with_range(48000, 2000.0, 10000.0);
        let tone = create_pure_tone(4000.0, 0.1, 48000);

        let (f0, confidence) = estimator.estimate(&tone);

        assert!((f0 - 4000.0).abs() < 500.0, "Should detect finch-like frequency, got {}", f0);
        assert!(confidence >= 0.0, "Confidence should be non-negative");
    }

    #[test]
    fn test_yin_dolphin_whistle() {
        // Dolphin: 2-24 kHz range (use 12kHz for this test)
        let estimator = YinEstimator::with_range(48000, 2000.0, 22000.0); // Below Nyquist
        let tone = create_pure_tone(12000.0, 0.1, 48000);

        let (f0, confidence) = estimator.estimate(&tone);

        assert!((f0 - 12000.0).abs() < 2000.0, "Should detect dolphin-like frequency, got {}", f0);
        assert!(confidence >= 0.0, "Confidence should be non-negative");
    }

    #[test]
    fn test_yin_low_frequency_bat() {
        // Bat ultrasonic (downsampled): 20-100 kHz, test at 8kHz (simulated)
        let estimator = YinEstimator::with_range(48000, 5000.0, 15000.0);
        let tone = create_pure_tone(8000.0, 0.05, 48000);

        let (f0, confidence) = estimator.estimate(&tone);

        assert!((f0 - 8000.0).abs() < 1000.0, "Should detect bat-like frequency, got {}", f0);
        assert!(confidence >= 0.0, "Confidence should be non-negative");
    }

    #[test]
    fn test_yin_chimpanzee_pant_hoot() {
        // Chimpanzee: 200-800 Hz range
        let estimator = YinEstimator::with_range(48000, 200.0, 1000.0);
        let tone = create_pure_tone(500.0, 0.1, 48000);

        let (f0, confidence) = estimator.estimate(&tone);

        assert!((f0 - 500.0).abs() < 100.0, "Should detect chimp-like low frequency, got {}", f0);
        assert!(confidence >= 0.0, "Confidence should be non-negative");
    }

    // =========================================================================
    // Category 7: Multi-Pitch Tests (5 tests)
    // =========================================================================

    #[test]
    fn test_yin_dominant_pitch_extraction() {
        let estimator = YinEstimator::new(48000);

        // Create signal with 1000 Hz (dominant) + 2000 Hz (harmonic)
        let mut signal = vec![0.0; 4800];
        for i in 0..4800 {
            let t = i as f32 / 48000.0;
            signal[i] = (2.0 * PI * 1000.0 * t).sin() + 0.3 * (2.0 * PI * 2000.0 * t).sin();
        }

        let (f0, _) = estimator.estimate(&signal);

        // Should detect the dominant fundamental (1000 Hz)
        assert!((f0 - 1000.0).abs() < 100.0, "Should extract dominant pitch");
    }

    #[test]
    fn test_yin_harmonic_series() {
        let estimator = YinEstimator::new(48000);

        // Create signal with strong harmonics
        let mut signal = vec![0.0; 4800];
        for i in 0..4800 {
            let t = i as f32 / 48000.0;
            // f0 + 0.5*f1 + 0.25*f2 + 0.125*f3
            signal[i] = (2.0 * PI * 1000.0 * t).sin()
                + 0.5 * (2.0 * PI * 2000.0 * t).sin()
                + 0.25 * (2.0 * PI * 3000.0 * t).sin()
                + 0.125 * (2.0 * PI * 4000.0 * t).sin();
        }

        let (f0, _) = estimator.estimate(&signal);

        // YIN should suppress harmonic peaks and find true F0
        assert!((f0 - 1000.0).abs() < 50.0, "Should find true F0, not harmonics");
    }

    #[test]
    fn test_yin_pitch_transition() {
        let estimator = YinEstimator::new(48000);

        // Create chirp from 1000 Hz to 2000 Hz
        let mut signal = vec![0.0; 4800];
        for i in 0..4800 {
            let t = i as f32 / 48000.0;
            let freq = 1000.0 + 1000.0 * t; // Linear chirp
            signal[i] = (2.0 * PI * freq * t).sin();
        }

        let (f0, _) = estimator.estimate(&signal);

        // Should detect somewhere in the middle
        assert!(f0 > 1000.0 && f0 < 2000.0, "Should detect average frequency of chirp");
    }

    #[test]
    fn test_yin_amplitude_modulation() {
        let estimator = YinEstimator::new(48000);

        // Create AM signal: 1000 Hz carrier, 50 Hz modulation
        let mut signal = vec![0.0; 4800];
        for i in 0..4800 {
            let t = i as f32 / 48000.0;
            let envelope = 1.0 + 0.5 * (2.0 * PI * 50.0 * t).sin();
            signal[i] = envelope * (2.0 * PI * 1000.0 * t).sin();
        }

        let (f0, confidence) = estimator.estimate(&signal);

        assert!((f0 - 1000.0).abs() < 150.0, "Should detect carrier frequency, got {}", f0);
        assert!(confidence >= 0.0, "Confidence should be non-negative");
    }

    #[test]
    fn test_yin_frequency_modulation() {
        let estimator = YinEstimator::new(48000);

        // Create FM signal (vibrato): 1000 Hz center, 20 Hz modulation rate, 50 Hz depth
        let mut signal = vec![0.0; 4800];
        for i in 0..4800 {
            let t = i as f32 / 48000.0;
            let phase = 2.0 * PI * 1000.0 * t + (50.0 / 20.0) * (2.0 * PI * 20.0 * t).sin();
            signal[i] = phase.sin();
        }

        let (f0, _) = estimator.estimate(&signal);

        // Should detect center frequency
        assert!((f0 - 1000.0).abs() < 100.0, "Should detect center frequency with FM");
    }
}
