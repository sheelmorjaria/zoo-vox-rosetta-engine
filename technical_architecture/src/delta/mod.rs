//! Delta features module
//!
//! This module provides temporal delta features for dynamic acoustic analysis.
//! Currently includes:
//! - **Delta MFCCs**: First and second derivatives of MFCCs (Δ and ΔΔ)
//! - **Temporal Delta Features**: Δ/ΔΔ for F0, amplitude, and spectral flux

pub mod mfcc_delta;
pub mod temporal_features;

pub use mfcc_delta::{DeltaRegression, MfccDeltaComputer};
pub use temporal_features::{TemporalDeltaComputer, TemporalFeatureType};

/// Common result type for delta features
pub type DeltaFeatures = (Vec<f32>, Vec<f32>); // (delta, delta_delta)

/// Delta feature configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeltaWidth {
    /// N=2 regression (recommended)
    #[default]
    N2,
    /// N=1 regression (simplest)
    N1,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_width_default() {
        let width = DeltaWidth::default();
        assert_eq!(width, DeltaWidth::N2);
    }

    #[test]
    fn test_delta_module_compatibility() {
        // Verify both delta computers can be created
        let mfcc_computer = MfccDeltaComputer::new(DeltaWidth::N2);
        let temporal_computer = TemporalDeltaComputer::new(DeltaWidth::N2);

        // Test MFCC delta computation
        let mfcc_frames = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![1.2, 2.2, 3.2],
        ];
        let (delta, delta_delta) = mfcc_computer.compute(&mfcc_frames).unwrap();

        assert_eq!(delta.len(), 3);
        assert_eq!(delta_delta.len(), 3);

        // Test temporal delta computation
        let temporal_sequence = vec![100.0, 110.0, 120.0, 130.0];
        let (t_delta, t_delta_delta) = temporal_computer.compute(&temporal_sequence).unwrap();

        assert_eq!(t_delta.len(), 4);
        assert_eq!(t_delta_delta.len(), 4);
    }
}
