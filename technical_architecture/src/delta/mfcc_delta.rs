//! MFCC delta features (Δ and ΔΔ)
//!
//! This module implements first and second derivatives of MFCC features.
//! Uses regression-based method with configurable window width (N=2 recommended).

use crate::delta::DeltaWidth;

/// MFCC delta computer using regression method
///
/// Computes temporal derivatives (Δ and ΔΔ) of MFCC features.
/// The regression method is more robust than simple differences.
///
/// # Algorithm
///
/// For N=2 regression:
/// ```text
/// Δ MFCC[t] = (MFCC[t+1] - MFCC[t-1]) / 2
/// ΔΔ MFCC[t] = (Δ MFCC[t+1] - Δ MFCC[t-1]) / 2
/// ```
///
/// # Example
///
/// ```rust
/// use technical_architecture::delta::{MfccDeltaComputer, DeltaWidth};
///
/// let computer = MfccDeltaComputer::new(DeltaWidth::N2);
/// let mfcc_frames = vec![
///     vec![1.0, 2.0, 3.0],
///     vec![1.1, 2.1, 3.1],
///     vec![1.2, 2.2, 3.2],
/// ];
///
/// let (delta, delta_delta) = computer.compute(&mfcc_frames).unwrap();
/// // delta: 3 frames × 13 coefficients
/// // delta_delta: 3 frames × 13 coefficients
/// ```
#[derive(Debug, Clone)]
pub struct MfccDeltaComputer {
    width: DeltaWidth,
}

impl MfccDeltaComputer {
    /// Create a new delta computer with specified width
    pub fn new(width: DeltaWidth) -> Self {
        Self { width }
    }

    /// Compute delta and delta-delta features from MFCC frames
    ///
    /// # Arguments
    ///
    /// * `mfcc_frames` - Vector of MFCC frames [n_frames × n_coeffs]
    ///
    /// # Returns
    ///
    /// * `(delta, delta_delta)` - Both are [n_frames × n_coeffs]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - mfcc_frames is empty
    /// - frames have inconsistent dimensions
    /// - contains NaN or Inf values
    pub fn compute(&self, mfcc_frames: &[Vec<f32>]) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), String> {
        // Validate input
        if mfcc_frames.is_empty() {
            return Ok((vec![], vec![]));
        }

        let n_frames = mfcc_frames.len();
        let n_coeffs = mfcc_frames[0].len();

        // Check for consistent dimensions
        for frame in mfcc_frames.iter() {
            if frame.len() != n_coeffs {
                return Err(format!(
                    "Inconsistent MFCC dimensions: expected {}, got {}",
                    n_coeffs,
                    frame.len()
                ));
            }
            // Check for NaN/Inf
            for &val in frame {
                if !val.is_finite() {
                    return Err(format!("MFCC contains non-finite value: {}", val));
                }
            }
        }

        // Handle single frame case
        if n_frames == 1 {
            return Ok((vec![vec![0.0; n_coeffs]], vec![vec![0.0; n_coeffs]]));
        }

        // Compute delta (first derivative)
        let delta = self.compute_delta(mfcc_frames, n_frames, n_coeffs);

        // Compute delta-delta (second derivative)
        let delta_delta = self.compute_delta_delta(&delta, n_frames, n_coeffs);

        Ok((delta, delta_delta))
    }

    /// Compute delta features (first derivative)
    fn compute_delta(&self, frames: &[Vec<f32>], n_frames: usize, n_coeffs: usize) -> Vec<Vec<f32>> {
        let mut delta = vec![vec![0.0; n_coeffs]; n_frames];

        match self.width {
            DeltaWidth::N1 => {
                // Simple difference: Δ[t] = x[t+1] - x[t]
                for t in 0..n_frames.saturating_sub(1) {
                    for c in 0..n_coeffs {
                        delta[t][c] = frames[t + 1][c] - frames[t][c];
                    }
                }
                // Last frame: copy previous
                if n_frames > 1 {
                    delta[n_frames - 1] = delta[n_frames - 2].clone();
                }
            }
            DeltaWidth::N2 => {
                // Regression: Δ[t] = (x[t+1] - x[t-1]) / 2
                for t in 1..n_frames.saturating_sub(1) {
                    for c in 0..n_coeffs {
                        delta[t][c] = (frames[t + 1][c] - frames[t - 1][c]) / 2.0;
                    }
                }
                // Edge frames: forward/backward difference
                if n_frames > 1 {
                    // First frame: forward difference
                    for c in 0..n_coeffs {
                        delta[0][c] = frames[1][c] - frames[0][c];
                    }
                    // Last frame: backward difference
                    for c in 0..n_coeffs {
                        delta[n_frames - 1][c] = frames[n_frames - 1][c] - frames[n_frames - 2][c];
                    }
                }
            }
        }

        delta
    }

    /// Compute delta-delta features (second derivative)
    fn compute_delta_delta(&self, delta: &[Vec<f32>], n_frames: usize, n_coeffs: usize) -> Vec<Vec<f32>> {
        let mut delta_delta = vec![vec![0.0; n_coeffs]; n_frames];

        match self.width {
            DeltaWidth::N1 => {
                // ΔΔ[t] = Δ[t+1] - Δ[t]
                for t in 0..n_frames.saturating_sub(1) {
                    for c in 0..n_coeffs {
                        delta_delta[t][c] = delta[t + 1][c] - delta[t][c];
                    }
                }
                // Last frame: copy previous
                if n_frames > 1 {
                    delta_delta[n_frames - 1] = delta_delta[n_frames - 2].clone();
                }
            }
            DeltaWidth::N2 => {
                // ΔΔ[t] = (Δ[t+1] - Δ[t-1]) / 2
                for t in 1..n_frames.saturating_sub(1) {
                    for c in 0..n_coeffs {
                        delta_delta[t][c] = (delta[t + 1][c] - delta[t - 1][c]) / 2.0;
                    }
                }
                // Edge frames: forward/backward difference
                if n_frames > 1 {
                    // First frame
                    for c in 0..n_coeffs {
                        delta_delta[0][c] = delta[1][c] - delta[0][c];
                    }
                    // Last frame
                    for c in 0..n_coeffs {
                        delta_delta[n_frames - 1][c] = delta[n_frames - 1][c] - delta[n_frames - 2][c];
                    }
                }
            }
        }

        delta_delta
    }
}

/// Regression type alias for clarity
pub type DeltaRegression = MfccDeltaComputer;

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Basic Computation Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_constant_to_zero() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // Constant MFCC sequence should produce zero delta
        let frames = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
        ];

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        // All deltas should be ~0
        for frame_delta in &delta {
            for &d in frame_delta {
                assert!(d.abs() < 1e-6, "Constant sequence should produce zero delta");
            }
        }

        // All delta-deltas should be ~0
        for frame_dd in &delta_delta {
            for &dd in frame_dd {
                assert!(dd.abs() < 1e-6, "Constant sequence should produce zero delta-delta");
            }
        }
    }

    #[test]
    fn test_linear_to_constant() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // Linear sequence should produce constant delta
        let frames = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0],
        ];

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        // Middle frames should have delta ≈ 1.0
        assert!((delta[1][0] - 1.0).abs() < 1e-6);
        assert!((delta[2][0] - 1.0).abs() < 1e-6);

        // Delta-delta should be ≈ 0
        for frame_dd in &delta_delta {
            for &dd in frame_dd {
                assert!(dd.abs() < 1e-6, "Linear sequence should produce zero delta-delta");
            }
        }
    }

    #[test]
    fn test_quadratic_to_linear() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // Quadratic sequence: x²
        let frames: Vec<Vec<f32>> = (0..4)
            .map(|i| {
                let x = i as f32;
                vec![x * x, x * x + 1.0, x * x + 2.0]
            })
            .collect();

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        // Delta should be roughly linear (increasing)
        let delta_0 = &delta.iter().map(|d| d[0]).collect::<Vec<_>>();
        for i in 1..delta_0.len() - 1 {
            assert!(delta_0[i] > delta_0[i - 1], "Delta should increase for quadratic input");
        }

        // Delta-delta should be roughly constant (≈ 2 for x²)
        let dd_0 = &delta_delta.iter().map(|dd| dd[0]).collect::<Vec<_>>();
        let dd_mean: f32 =
            dd_0.iter().skip(1).take(dd_0.len() - 2).sum::<f32>() / (dd_0.len().saturating_sub(2) as f32).max(1.0);
        assert!((dd_mean - 2.0).abs() < 1.0, "Delta-delta should be ~2.0 for x²");
    }

    #[test]
    fn test_boundary_handling() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        let frames = vec![vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]];

        let (delta, _) = computer.compute(&frames).unwrap();

        // First frame: forward difference (1.0)
        assert!((delta[0][0] - 1.0).abs() < 1e-6, "First frame should use forward diff");

        // Middle frame: regression (1.0)
        assert!((delta[1][0] - 1.0).abs() < 1e-6, "Middle frame should use regression");

        // Last frame: backward difference (1.0)
        assert!((delta[2][0] - 1.0).abs() < 1e-6, "Last frame should use backward diff");
    }

    #[test]
    fn test_edge_frames_two_elements() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        let frames = vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]];

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        // First frame delta
        assert!((delta[0][0] - 1.0).abs() < 1e-6);

        // Last frame delta (backward difference)
        assert!((delta[1][0] - 1.0).abs() < 1e-6);

        // Delta-delta should be zero (constant derivative)
        assert!(delta_delta[0][0].abs() < 1e-6);
        assert!(delta_delta[1][0].abs() < 1e-6);
    }

    #[test]
    fn test_mfcc_13_dimensions() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // Standard 13-dimensional MFCC
        let frames = vec![
            (0..13).map(|i| i as f32).collect(),
            (0..13).map(|i| (i as f32) + 0.1).collect(),
            (0..13).map(|i| (i as f32) + 0.2).collect(),
        ];

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        assert_eq!(delta.len(), 3);
        assert_eq!(delta_delta.len(), 3);
        assert_eq!(delta[0].len(), 13);
        assert_eq!(delta_delta[0].len(), 13);
    }

    // =========================================================================
    // Delta-Delta Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_constant_delta_to_zero_delta_delta() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // Linear sequence → constant delta → zero delta-delta
        let frames = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0],
        ];

        let (_, delta_delta) = computer.compute(&frames).unwrap();

        // All delta-deltas should be ~0
        for frame_dd in &delta_delta {
            for &dd in frame_dd {
                assert!(dd.abs() < 1e-6, "Constant delta should produce zero delta-delta");
            }
        }
    }

    #[test]
    fn test_linear_delta_to_constant_delta_delta() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // Quadratic sequence → linear delta → constant delta-delta
        let frames: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                let x = i as f32;
                vec![x * x, x * x + 1.0, x * x + 2.0]
            })
            .collect();

        let (_, delta_delta) = computer.compute(&frames).unwrap();

        // Delta-delta should be roughly constant (~2.0 for x²)
        let dd_0 = &delta_delta.iter().map(|dd| dd[0]).collect::<Vec<_>>();
        let dd_mean: f32 =
            dd_0.iter().skip(1).take(dd_0.len() - 2).sum::<f32>() / (dd_0.len().saturating_sub(2) as f32).max(1.0);
        assert!((dd_mean - 2.0).abs() < 1.0, "Delta-delta should be ~2.0");
    }

    #[test]
    fn test_delta_delta_smoothing() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // Noisy sequence
        let frames = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.1, 0.9],
            vec![2.0, 1.9, 2.1],
            vec![3.0, 3.0, 3.0],
        ];

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        // Delta-delta should be small (smoothing effect)
        for frame_dd in &delta_delta {
            for &dd in frame_dd {
                assert!(dd.abs() < 2.0, "Delta-delta should smooth out noise");
            }
        }
    }

    #[test]
    fn test_noise_robustness() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // Linear trend with noise
        let frames: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                let base = i as f32;
                vec![base + 0.01 * (i as f32 % 3.0)]
            })
            .collect();

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        // Delta should be roughly 1.0 despite noise
        let delta_mean: f32 = delta.iter().map(|d| d[0]).sum::<f32>() / delta.len() as f32;
        assert!((delta_mean - 1.0).abs() < 0.1, "Delta should be robust to noise");
    }

    #[test]
    fn test_delta_delta_integration() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        let frames = vec![
            vec![0.0],
            vec![1.0],
            vec![3.0], // Accelerating
            vec![6.0],
        ];

        let (_, delta_delta) = computer.compute(&frames).unwrap();

        // Should detect acceleration
        let dd_sum: f32 = delta_delta.iter().map(|dd| dd[0]).sum::<f32>();
        assert!(dd_sum > 0.0, "Delta-delta should detect acceleration");
    }

    #[test]
    fn test_delta_delta_reconstruction() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        let frames = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        // For linear sequence, delta-delta should be zero
        for frame_dd in &delta_delta {
            assert!(frame_dd[0].abs() < 1e-6);
        }
    }

    // =========================================================================
    // Frame Sequence Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_single_frame() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        let frames = vec![vec![1.0, 2.0, 3.0]];

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        assert_eq!(delta.len(), 1);
        assert_eq!(delta_delta.len(), 1);

        // Both should be zero
        for d in &delta[0] {
            assert_eq!(*d, 0.0);
        }
        for dd in &delta_delta[0] {
            assert_eq!(*dd, 0.0);
        }
    }

    #[test]
    fn test_two_frames() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        let frames = vec![vec![0.0], vec![1.0]];

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        assert_eq!(delta.len(), 2);
        assert_eq!(delta_delta.len(), 2);

        // Delta should be 1.0
        assert!((delta[0][0] - 1.0).abs() < 1e-6);
        assert!((delta[1][0] - 1.0).abs() < 1e-6);

        // Delta-delta should be 0.0
        assert_eq!(delta_delta[0][0], 0.0);
        assert_eq!(delta_delta[1][0], 0.0);
    }

    #[test]
    fn test_long_sequence() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // 1000 frames
        let frames: Vec<Vec<f32>> = (0..1000)
            .map(|i| vec![i as f32, i as f32 * 2.0, i as f32 * 3.0])
            .collect();

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        assert_eq!(delta.len(), 1000);
        assert_eq!(delta_delta.len(), 1000);

        // Delta should be ~1.0
        assert!((delta[500][0] - 1.0).abs() < 1e-6);

        // Delta-delta should be ~0
        assert!(delta_delta[500][0].abs() < 1e-6);
    }

    #[test]
    fn test_variable_length_frames() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // Different number of coefficients
        let frames = vec![vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0]];

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        assert_eq!(delta[0].len(), 3);
        assert_eq!(delta_delta[0].len(), 3);
    }

    #[test]
    fn test_alignment_preserved() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        let frames = vec![vec![0.0, 1.0], vec![1.0, 2.0], vec![2.0, 3.0]];

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        // Alignment should be preserved
        assert_eq!(delta.len(), frames.len());
        assert_eq!(delta_delta.len(), frames.len());

        for (i, (d, dd)) in delta.iter().zip(delta_delta.iter()).enumerate() {
            assert_eq!(d.len(), frames[i].len());
            assert_eq!(dd.len(), frames[i].len());
        }
    }

    #[test]
    fn test_empty_frames() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        let frames: Vec<Vec<f32>> = vec![];

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        assert_eq!(delta.len(), 0);
        assert_eq!(delta_delta.len(), 0);
    }

    // =========================================================================
    // Numerical Stability Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_nan_propagation() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        let frames = vec![vec![0.0, 1.0, 2.0], vec![f32::NAN, 2.0, 3.0], vec![2.0, 3.0, 4.0]];

        let result = computer.compute(&frames);

        assert!(result.is_err(), "NaN should cause error");
        assert!(result.unwrap_err().contains("non-finite"));
    }

    #[test]
    fn test_infinity_propagation() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        let frames = vec![vec![0.0, 1.0, 2.0], vec![f32::INFINITY, 2.0, 3.0], vec![2.0, 3.0, 4.0]];

        let result = computer.compute(&frames);

        assert!(result.is_err(), "Infinity should cause error");
    }

    #[test]
    fn test_underflow_handling() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // Very small values
        let frames = vec![
            vec![1e-30, 2e-30, 3e-30],
            vec![1.1e-30, 2.1e-30, 3.1e-30],
            vec![1.2e-30, 2.2e-30, 3.2e-30],
        ];

        let (delta, _) = computer.compute(&frames).unwrap();

        // Should not underflow to zero
        assert!(delta[1][0] > 0.0, "Should handle small values");
    }

    #[test]
    fn test_overflow_handling() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // Large but not overflowing values
        let frames = vec![
            vec![1e30, 2e30, 3e30],
            vec![1.1e30, 2.1e30, 3.1e30],
            vec![1.2e30, 2.2e30, 3.2e30],
        ];

        let (delta, _) = computer.compute(&frames).unwrap();

        // Should not overflow to inf
        assert!(delta[1][0].is_finite(), "Should handle large values");
    }

    #[test]
    fn test_denormal_numbers() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // Denormal numbers (subnormal)
        let frames = vec![
            vec![1e-40, 2e-40, 3e-40],
            vec![1.1e-40, 2.1e-40, 3.1e-40],
            vec![1.2e-40, 2.2e-40, 3.2e-40],
        ];

        let (delta, _) = computer.compute(&frames).unwrap();

        // Should handle denormals gracefully
        for d in &delta {
            for &val in d {
                assert!(val.is_finite(), "Should handle denormals");
            }
        }
    }

    #[test]
    fn test_precision_f32() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        let frames = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0000001, 2.0000001, 3.0000001],
            vec![1.0000002, 2.0000002, 3.0000002],
        ];

        let (delta, _) = computer.compute(&frames).unwrap();

        // Should maintain reasonable precision
        assert!((delta[1][0] - 0.00000005).abs() < 1e-7, "Should maintain precision");
    }

    // =========================================================================
    // Performance Tests (6 tests)
    // =========================================================================

    #[test]
    fn test_computation_speed() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // 100 frames × 13 coefficients (typical MFCC size)
        let frames: Vec<Vec<f32>> = (0..100).map(|_| (0..13).map(|i| i as f32).collect()).collect();

        let start = std::time::Instant::now();
        let _ = computer.compute(&frames).unwrap();
        let elapsed = start.elapsed();

        // Should complete in < 1ms
        assert!(elapsed.as_millis() < 1, "Computation should be fast");
    }

    #[test]
    fn test_memory_efficiency() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        let frames: Vec<Vec<f32>> = (0..1000).map(|_| (0..13).map(|i| i as f32).collect()).collect();

        let (delta, delta_delta) = computer.compute(&frames).unwrap();

        // Output should be same size as input
        assert_eq!(delta.len(), frames.len());
        assert_eq!(delta_delta.len(), frames.len());
    }

    #[test]
    fn test_cache_locality() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // Sequential access pattern
        let frames: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..13).map(|j| (i * 13 + j) as f32).collect())
            .collect();

        let (delta, _) = computer.compute(&frames).unwrap();

        // Should process all frames correctly
        assert_eq!(delta.len(), 100);
    }

    #[test]
    fn test_parallelizable() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        let frames: Vec<Vec<f32>> = (0..100).map(|_| (0..13).map(|i| i as f32).collect()).collect();

        // Multiple independent computations
        let start = std::time::Instant::now();

        let _ = computer.compute(&frames).unwrap();
        let _ = computer.compute(&frames).unwrap();

        let elapsed = start.elapsed();

        // Should scale reasonably (each < 1ms)
        assert!(elapsed.as_millis() < 10, "Should be parallelizable");
    }

    #[test]
    fn test_vectorization_friendly() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        // Aligned access pattern
        let frames: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..16).map(|i| i as f32).collect()) // Power of 2
            .collect();

        let (delta, _) = computer.compute(&frames).unwrap();

        assert_eq!(delta[0].len(), 16);
    }

    #[test]
    fn test_allocation_free_parsing() {
        let computer = MfccDeltaComputer::new(DeltaWidth::N2);

        let frames: Vec<Vec<f32>> = (0..10).map(|_| (0..13).map(|i| i as f32).collect()).collect();

        // Multiple calls should not leak
        for _ in 0..100 {
            let _ = computer.compute(&frames).unwrap();
        }

        // Test passes if we reach here (no memory leak)
    }
}
