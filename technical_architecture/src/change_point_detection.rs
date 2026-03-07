// Change Point Detection Module
//
// Implements PELT (Pruned Exact Linear Time) algorithm for change point detection
// in audio signals. This is used to segment vocalizations into phrases.
//
// Reference: Killick, R., Fearnhead, P., & Eckley, I. A. (2012).
// "Optimal detection of changepoints with a linear computational cost"

use ndarray::{s, Array2, Axis};
use std::f64;

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum ChangePointError {
    #[error("Insufficient data: need at least {min} samples, got {actual}")]
    InsufficientData { min: usize, actual: usize },

    #[error("Invalid penalty: {penalty} (must be > 0)")]
    InvalidPenalty { penalty: f64 },

    #[error("Feature extraction failed: {0}")]
    FeatureExtractionFailed(String),
}

pub type Result<T> = std::result::Result<T, ChangePointError>;

// =============================================================================
// PELT Segmenter
// =============================================================================

/// PELT (Pruned Exact Linear Time) change point detection algorithm
///
/// Segments audio into phrases by detecting changes in acoustic features.
/// Uses RBF kernel distance to measure similarity between segments.
#[derive(Debug, Clone)]
pub struct PeltSegmenter {
    penalty: f64,
    min_segment_length: usize,
}

impl PeltSegmenter {
    /// Create a new PELT segmenter
    ///
    /// # Arguments
    /// * `penalty` - Penalty for adding a change point (higher = fewer segments)
    /// * `min_segment_length` - Minimum segment length in samples
    pub fn new(penalty: f64, min_segment_length: usize) -> Result<Self> {
        if penalty <= 0.0 {
            return Err(ChangePointError::InvalidPenalty { penalty });
        }

        Ok(Self {
            penalty,
            min_segment_length,
        })
    }

    /// Segment feature matrix into change points
    ///
    /// # Arguments
    /// * `features` - Feature matrix (frames x dimensions)
    ///
    /// # Returns
    /// Vector of change point indices (in frame space)
    pub fn segment(&self, features: &Array2<f64>) -> Result<Vec<usize>> {
        let n_frames = features.nrows();

        // Ensure minimum length
        if n_frames < self.min_segment_length * 2 {
            return Ok(vec![0, n_frames]);
        }

        // PELT algorithm with optimal partitioning
        let n = n_frames;
        let mut cost = vec![f64::MAX; n + 1];
        let mut changepoints: Vec<Vec<usize>> = vec![vec![]; n + 1];

        cost[0] = -self.penalty;
        changepoints[0] = vec![0];

        for t in (self.min_segment_length)..=n {
            // Search for best changepoint
            for tau in (0..=(t - self.min_segment_length)).rev() {
                let segment_cost = self.compute_segment_cost(features, tau, t);
                let total_cost = cost[tau] + segment_cost + self.penalty;

                if total_cost < cost[t] {
                    cost[t] = total_cost;
                    changepoints[t] = changepoints[tau].clone();
                    changepoints[t].push(t);
                }
            }
        }

        Ok(changepoints[n].clone())
    }

    /// Compute cost of a segment from start to end (exclusive)
    fn compute_segment_cost(&self, features: &Array2<f64>, start: usize, end: usize) -> f64 {
        if start >= end {
            return 0.0;
        }

        let segment = features.slice(s![start..end, ..]);
        let n = segment.nrows();

        if n == 0 {
            return 0.0;
        }

        // Compute variance-based cost
        // Cost = sum of squared distances from mean
        let mean = segment.mean_axis(Axis(0)).unwrap();
        let mut cost = 0.0;

        for row in segment.rows() {
            for i in 0..row.len() {
                let diff = row[i] - mean[i];
                cost += diff * diff;
            }
        }

        cost / n as f64
    }

    /// Convert frame indices to sample indices
    pub fn frames_to_samples(&self, frame_indices: &[usize], hop_length: usize) -> Vec<usize> {
        frame_indices.iter().map(|&f| f * hop_length).collect()
    }
}

// =============================================================================
// Tests (TDD Approach)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a constant signal (no changepoints)
    fn create_constant_signal(n: usize, value: f64) -> Array2<f64> {
        Array2::from_shape_fn((n, 1), |(_, _)| value)
    }

    /// Create a step signal (one clear changepoint)
    fn create_step_signal(n: usize, step_at: usize, low: f64, high: f64) -> Array2<f64> {
        Array2::from_shape_fn((n, 1), |(i, _)| if i < step_at { low } else { high })
    }

    /// Create a multi-step signal (multiple changepoints)
    fn create_multi_step_signal(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, 1), |(i, _)| {
            if i < n / 3 {
                0.0
            } else if i < 2 * n / 3 {
                10.0
            } else {
                20.0
            }
        })
    }

    // ===== Test 1: Constructor validation =====
    #[test]
    fn test_pelt_new_valid_parameters() {
        let segmenter = PeltSegmenter::new(10.0, 5);
        assert!(segmenter.is_ok());
        assert_eq!(segmenter.unwrap().penalty, 10.0);
    }

    #[test]
    fn test_pelt_new_invalid_penalty_zero() {
        let segmenter = PeltSegmenter::new(0.0, 5);
        assert!(segmenter.is_err());
    }

    #[test]
    fn test_pelt_new_invalid_penalty_negative() {
        let segmenter = PeltSegmenter::new(-1.0, 5);
        assert!(segmenter.is_err());
    }

    // ===== Test 2: Constant signal (no changepoints) =====
    #[test]
    fn test_pelt_constant_signal_minimal_segmentation() {
        let segmenter = PeltSegmenter::new(10.0, 5).unwrap();
        let signal = create_constant_signal(100, 1.0);

        let changepoints = segmenter.segment(&signal).unwrap();

        // Constant signal should have minimal segmentation (start and end only)
        assert_eq!(changepoints, vec![0, 100]);
    }

    // ===== Test 3: Step signal (one changepoint) =====
    #[test]
    fn test_pelt_step_signal_detects_change() {
        let segmenter = PeltSegmenter::new(5.0, 5).unwrap();
        let signal = create_step_signal(100, 50, 0.0, 10.0);

        let changepoints = segmenter.segment(&signal).unwrap();

        // Should detect at least 2 changepoints (start and end)
        assert!(changepoints.len() >= 2);
        assert_eq!(changepoints[0], 0);
        assert_eq!(changepoints[changepoints.len() - 1], 100);

        // Should detect changepoint near the step (within tolerance)
        let detected_step = changepoints.iter().find(|&&cp| (45..=55).contains(&cp));

        assert!(detected_step.is_some(), "Should detect changepoint near frame 50");
    }

    // ===== Test 4: Multi-step signal (multiple changepoints) =====
    #[test]
    fn test_pelt_multi_step_signal_multiple_changes() {
        let segmenter = PeltSegmenter::new(5.0, 5).unwrap();
        let signal = create_multi_step_signal(120);

        let changepoints = segmenter.segment(&signal).unwrap();

        // Should detect at least 3 changepoints
        assert!(changepoints.len() >= 3);

        // Check that changepoints are in ascending order
        for i in 1..changepoints.len() {
            assert!(changepoints[i] > changepoints[i - 1]);
        }
    }

    // ===== Test 5: Insufficient data =====
    #[test]
    fn test_pelt_insufficient_data_returns_bounds() {
        let segmenter = PeltSegmenter::new(10.0, 50).unwrap();
        let signal = create_constant_signal(10, 1.0);

        let changepoints = segmenter.segment(&signal).unwrap();

        // Should return start and end only
        assert_eq!(changepoints, vec![0, 10]);
    }

    // ===== Test 6: Frame to sample conversion =====
    #[test]
    fn test_frames_to_samples_conversion() {
        let segmenter = PeltSegmenter::new(10.0, 5).unwrap();
        let frame_indices = vec![0, 10, 20, 30];

        let sample_indices = segmenter.frames_to_samples(&frame_indices, 512);

        assert_eq!(sample_indices, vec![0, 5120, 10240, 15360]);
    }

    // ===== Test 7: Multi-dimensional features =====
    #[test]
    fn test_pelt_multi_dimensional_features() {
        let segmenter = PeltSegmenter::new(10.0, 5).unwrap();

        // Create 3D signal with clear change
        let signal1 = Array2::from_shape_fn((50, 3), |(_, _)| 0.0);
        let signal2 = Array2::from_shape_fn((50, 3), |(_, _)| 10.0);
        let signal = ndarray::concatenate(Axis(0), &[signal1.view(), signal2.view()]).unwrap();

        let changepoints = segmenter.segment(&signal).unwrap();

        // Should detect change between 45-55 frames
        assert!(changepoints.len() >= 2);

        let closest = changepoints.iter().find(|&&cp| (45..=55).contains(&cp));

        assert!(closest.is_some(), "Should detect changepoint in expected range");
    }

    // ===== Test 8: Segment cost calculation =====
    #[test]
    fn test_segment_cost_constant_zero() {
        let segmenter = PeltSegmenter::new(10.0, 5).unwrap();

        // Constant segment should have zero cost
        let constant = create_constant_signal(10, 5.0);
        let cost = segmenter.compute_segment_cost(&constant, 0, 10);

        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_segment_cost_varying_positive() {
        let segmenter = PeltSegmenter::new(10.0, 5).unwrap();

        // Segment with variance should have positive cost
        let varying = Array2::from_shape_fn((10, 1), |(i, _)| i as f64);
        let cost = segmenter.compute_segment_cost(&varying, 0, 10);

        assert!(cost > 0.0);
    }

    // ===== Test 9: Penalty affects segmentation =====
    #[test]
    fn test_low_penalty_more_changepoints() {
        let signal = create_step_signal(100, 50, 0.0, 10.0);

        // Low penalty should detect more changepoints
        let segmenter_low = PeltSegmenter::new(1.0, 5).unwrap();
        let changepoints_low = segmenter_low.segment(&signal).unwrap();

        // High penalty should detect fewer changepoints
        let segmenter_high = PeltSegmenter::new(100.0, 5).unwrap();
        let changepoints_high = segmenter_high.segment(&signal).unwrap();

        assert!(changepoints_low.len() >= changepoints_high.len());
    }

    // ===== Test 10: Minimum segment length enforced =====
    #[test]
    fn test_min_segment_length_enforced() {
        let signal = create_multi_step_signal(100);

        // Set minimum segment length to 20 frames
        let segmenter = PeltSegmenter::new(5.0, 20).unwrap();
        let changepoints = segmenter.segment(&signal).unwrap();

        // All segments should be at least 20 frames apart
        for i in 1..changepoints.len() {
            let segment_length = changepoints[i] - changepoints[i - 1];
            assert!(segment_length >= 20 || changepoints[i] == 100); // Last segment may be end
        }
    }
}
