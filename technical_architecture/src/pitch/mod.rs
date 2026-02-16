//! Pitch detection module
//!
//! This module provides various pitch detection algorithms for fundamental frequency (F0) estimation.
//! Currently includes:
//! - **YIN Algorithm**: Robust pitch estimation with confidence scoring
//! - **Autocorrelation**: Traditional autocorrelation-based pitch detection

mod yin;
mod autocorr;

pub use yin::YinEstimator;
pub use autocorr::AutocorrEstimator;

/// Common result type for pitch estimation
pub type F0Estimate = (f32, f32); // (frequency_hz, confidence)

/// Pitch detection algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchAlgorithm {
    /// YIN algorithm (recommended for most use cases)
    YIN,
    /// Traditional autocorrelation (faster, less robust)
    Autocorrelation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_module_compatibility() {
        // Verify both estimators can be created and used
        let yin = YinEstimator::new(48000);
        let autocorr = AutocorrEstimator::new(48000);

        // Test tone
        let tone: Vec<f32> = (0..4800)
            .map(|i| {
                let t = i as f32 / 48000.0;
                (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
            })
            .collect();

        let (yin_f0, yin_conf) = yin.estimate(&tone);
        let (auto_f0, auto_conf) = autocorr.estimate(&tone);

        // Both should detect approximately 1000 Hz
        assert!((yin_f0 - 1000.0).abs() < 150.0, "YIN should detect ~1000 Hz");
        assert!((auto_f0 - 1000.0).abs() < 150.0, "Autocorr should detect ~1000 Hz");

        // Both should have valid confidence
        assert!(yin_conf >= 0.0 && yin_conf <= 1.0, "YIN confidence should be valid");
        assert!(auto_conf >= 0.0 && auto_conf <= 1.0, "Autocorr confidence should be valid");
    }
}
