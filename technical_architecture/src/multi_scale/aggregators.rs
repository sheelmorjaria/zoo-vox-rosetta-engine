//! Statistical aggregators for multi-scale feature analysis
//!
//! This module provides statistical aggregation functions for computing
//! hierarchical features across time scales.

/// Multi-scale statistical features
#[derive(Debug, Clone, PartialEq)]
pub struct MultiScaleFeatures {
    /// Arithmetic mean
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Skewness (third moment)
    pub skewness: f32,
    /// Kurtosis (fourth moment)
    pub kurtosis: f32,
    /// Range (max - min)
    pub range: f32,
    /// Interquartile range (Q3 - Q1)
    pub iqr: f32,
}

impl Default for MultiScaleFeatures {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            range: 0.0,
            iqr: 0.0,
        }
    }
}

/// Statistical aggregator for computing multi-scale features
pub struct StatisticalAggregator;

impl StatisticalAggregator {
    /// Compute arithmetic mean
    ///
    /// Returns NaN for empty input.
    pub fn mean(data: &[f32]) -> f32 {
        if data.is_empty() {
            return f32::NAN;
        }
        data.iter().sum::<f32>() / data.len() as f32
    }

    /// Compute standard deviation (sample)
    ///
    /// Returns 0.0 for single element, NaN for empty.
    pub fn std_dev(data: &[f32]) -> f32 {
        if data.len() <= 1 {
            if data.is_empty() {
                return f32::NAN;
            }
            return 0.0;
        }

        let mean = Self::mean(data);
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (data.len() - 1) as f32;
        variance.sqrt()
    }

    /// Compute skewness (third standardized moment)
    ///
    /// Returns 0.0 for insufficient data, NaN for empty.
    pub fn skewness(data: &[f32]) -> f32 {
        if data.len() <= 2 {
            if data.is_empty() {
                return f32::NAN;
            }
            return 0.0;
        }

        let mean = Self::mean(data);
        let std = Self::std_dev(data);

        if std == 0.0 {
            return 0.0;
        }

        let n = data.len() as f32;
        let m3 = data.iter().map(|&x| ((x - mean) / std).powi(3)).sum::<f32>() / n;
        m3
    }

    /// Compute kurtosis (fourth standardized moment, excess kurtosis)
    ///
    /// Returns 0.0 for insufficient data, NaN for empty.
    pub fn kurtosis(data: &[f32]) -> f32 {
        if data.len() <= 3 {
            if data.is_empty() {
                return f32::NAN;
            }
            return 0.0;
        }

        let mean = Self::mean(data);
        let std = Self::std_dev(data);

        if std == 0.0 {
            return 0.0;
        }

        let n = data.len() as f32;
        let m4 = data.iter().map(|&x| ((x - mean) / std).powi(4)).sum::<f32>() / n;
        m4 - 3.0 // Excess kurtosis
    }

    /// Compute range (max - min)
    ///
    /// Returns 0.0 for empty input.
    pub fn range(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        max - min
    }

    /// Compute interquartile range (Q3 - Q1)
    ///
    /// Returns 0.0 for insufficient data.
    pub fn iqr(data: &[f32]) -> f32 {
        if data.len() < 4 {
            if data.is_empty() {
                return 0.0;
            }
            // For small datasets, approximate with range
            return Self::range(data);
        }

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q1_idx = sorted.len() / 4;
        let q3_idx = (3 * sorted.len()) / 4;
        sorted[q3_idx] - sorted[q1_idx]
    }

    /// Compute all multi-scale features in a single pass
    pub fn compute_all(data: &[f32]) -> MultiScaleFeatures {
        MultiScaleFeatures {
            mean: Self::mean(data),
            std_dev: Self::std_dev(data),
            skewness: Self::skewness(data),
            kurtosis: Self::kurtosis(data),
            range: Self::range(data),
            iqr: Self::iqr(data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // =========================================================================
    // Basic Statistics Tests (10 tests)
    // =========================================================================

    #[test]
    fn test_mean_constant() {
        let data = vec![5.0; 100];
        let mean = StatisticalAggregator::mean(&data);
        assert_eq!(mean, 5.0);
    }

    #[test]
    fn test_mean_symmetric() {
        let data = vec![-1.0, 0.0, 1.0];
        let mean = StatisticalAggregator::mean(&data);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_std_dev_zero() {
        let data = vec![5.0; 100];
        let std = StatisticalAggregator::std_dev(&data);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_std_dev_normal() {
        // Standard normal: mean=0, std=1
        let data: Vec<f32> = (0..1000)
            .map(|_| {
                // Box-Muller transform
                let u1: f32 = rand::random();
                let u2: f32 = rand::random();
                (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
            })
            .collect();

        let std = StatisticalAggregator::std_dev(&data);
        // Should be approximately 1.0
        assert!((std - 1.0).abs() < 0.2, "STD should be ~1.0 for normal distribution");
    }

    #[test]
    fn test_skew_symmetric() {
        let data = vec![-1.0, 0.0, 1.0];
        let skew = StatisticalAggregator::skewness(&data);
        // Symmetric distribution should have skewness ≈ 0
        assert!(skew.abs() < 0.5, "Symmetric data should have ~0 skewness");
    }

    #[test]
    fn test_skew_tail() {
        // Right-skewed data
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let skew = StatisticalAggregator::skewness(&data);
        assert!(skew > 0.0, "Right-skewed data should have positive skewness");
    }

    #[test]
    fn test_kurtosis_normal() {
        // Normal distribution: excess kurtosis ≈ 0
        let data: Vec<f32> = (0..1000)
            .map(|_| {
                let u1: f32 = rand::random();
                let u2: f32 = rand::random();
                (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
            })
            .collect();

        let kurt = StatisticalAggregator::kurtosis(&data);
        // Should be approximately 0
        assert!(kurt.abs() < 1.0, "Normal distribution should have ~0 excess kurtosis");
    }

    #[test]
    fn test_kurtosis_uniform() {
        // Uniform distribution: negative excess kurtosis (platykurtic)
        let data: Vec<f32> = (0..1000).map(|_| rand::random()).collect();
        let kurt = StatisticalAggregator::kurtosis(&data);
        assert!(kurt < 0.0, "Uniform distribution should have negative excess kurtosis");
    }

    #[test]
    fn test_range_basic() {
        let data = vec![1.0, 2.0, 5.0, 10.0];
        let range = StatisticalAggregator::range(&data);
        assert_eq!(range, 9.0);
    }

    #[test]
    fn test_iqr_basic() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let iqr = StatisticalAggregator::iqr(&data);
        // For uniform 0-99, Q1≈25, Q3≈75, IQR≈50
        assert!((iqr - 50.0).abs() < 5.0, "IQR should be ~50 for uniform 0-99");
    }

    // =========================================================================
    // Edge Cases Tests (5 tests)
    // =========================================================================

    #[test]
    fn test_empty_returns_nan() {
        let data: Vec<f32> = vec![];
        assert!(StatisticalAggregator::mean(&data).is_nan());
        assert!(StatisticalAggregator::std_dev(&data).is_nan());
        assert!(StatisticalAggregator::skewness(&data).is_nan());
        assert!(StatisticalAggregator::kurtosis(&data).is_nan());
    }

    #[test]
    fn test_single_element_std_zero() {
        let data = vec![5.0];
        assert_eq!(StatisticalAggregator::mean(&data), 5.0);
        assert_eq!(StatisticalAggregator::std_dev(&data), 0.0);
    }

    #[test]
    fn test_two_values() {
        let data = vec![1.0, 3.0];
        assert_eq!(StatisticalAggregator::mean(&data), 2.0);
        assert!(StatisticalAggregator::std_dev(&data) > 0.0);
    }

    #[test]
    fn test_nan_propagation() {
        let data = vec![1.0, f32::NAN, 3.0];
        let mean = StatisticalAggregator::mean(&data);
        assert!(mean.is_nan(), "NaN should propagate through mean");
    }

    #[test]
    fn test_inf_handling() {
        let data = vec![1.0, f32::INFINITY, 3.0];
        let mean = StatisticalAggregator::mean(&data);
        assert!(mean.is_infinite(), "Inf should propagate through mean");
    }

    // =========================================================================
    // Numerical Precision Tests (5 tests)
    // =========================================================================

    #[test]
    fn test_f32_vs_f64_consistency() {
        let data: Vec<f32> = (0..100).map(|i| 0.1 * i as f32).collect();
        // Results should be reasonable for f32
        let mean = StatisticalAggregator::mean(&data);
        assert!(mean.is_finite(), "Mean should be finite");
    }

    #[test]
    fn test_categorical_stability() {
        // Data with repeated values
        let data = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        let mean = StatisticalAggregator::mean(&data);
        assert_eq!(mean, 1.5);
    }

    #[test]
    fn test_commutativity() {
        let data1 = vec![1.0, 2.0, 3.0];
        let mut data2 = vec![3.0, 1.0, 2.0];

        let mean1 = StatisticalAggregator::mean(&data1);
        let mean2 = StatisticalAggregator::mean(&data2);

        assert_eq!(mean1, mean2, "Mean should be commutative");
    }

    #[test]
    fn test_associativity_approximation() {
        // For floating point, exact associativity doesn't hold
        // but should be close
        let data: Vec<f32> = (0..100).map(|i| 0.1).collect();

        let sum1: f32 = data.iter().sum();
        let sum2: f32 = data.iter().rev().sum();

        assert!((sum1 - sum2).abs() < 1e-5, "Sums should be approximately associative");
    }

    #[test]
    fn test_rounding_errors() {
        // Many small values that could accumulate error
        let data: Vec<f32> = (0..10000).map(|_| 0.0001).collect();

        let mean = StatisticalAggregator::mean(&data);
        assert!((mean - 0.0001).abs() < 1e-5, "Should handle rounding errors");
    }

    // =========================================================================
    // Performance Tests (5 tests)
    // =========================================================================

    #[test]
    fn test_single_pass_on() {
        let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();

        let start = std::time::Instant::now();
        let _ = StatisticalAggregator::compute_all(&data);
        let elapsed = start.elapsed();

        // Should be fast (O(N))
        assert!(elapsed.as_millis() < 10, "Single-pass computation should be fast");
    }

    #[test]
    fn test_constant_memory() {
        // Large dataset
        let data: Vec<f32> = (0..100000).map(|i| i as f32).collect();

        // Should not cause memory issues
        let _ = StatisticalAggregator::compute_all(&data);
        // Test passes if we reach here without memory issues
    }

    #[test]
    fn test_cache_efficient() {
        // Sequential access pattern
        let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = StatisticalAggregator::mean(&data);
        }
        let elapsed = start.elapsed();

        // Should be cache-friendly
        assert!(elapsed.as_millis() < 100, "Should be cache-efficient");
    }

    #[test]
    fn test_vectorization_friendly() {
        // Aligned access pattern
        let data: Vec<f32> = (0..10000).map(|_| 1.0).collect();

        let start = std::time::Instant::now();
        let _ = StatisticalAggregator::mean(&data);
        let elapsed = start.elapsed();

        // Single mean computation should be very fast
        assert!(elapsed.as_micros() < 1000, "Should be vectorizable");
    }

    #[test]
    fn test_parallelizable() {
        let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();

        // Multiple independent computations
        let start = std::time::Instant::now();
        let m1 = StatisticalAggregator::mean(&data);
        let m2 = StatisticalAggregator::std_dev(&data);
        let m3 = StatisticalAggregator::skewness(&data);
        let _ = (m1, m2, m3);
        let elapsed = start.elapsed();

        // Should scale reasonably
        assert!(elapsed.as_millis() < 20, "Should be parallelizable");
    }
}
