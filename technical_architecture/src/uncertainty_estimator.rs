//! Uncertainty Quantification for Boundary Detection
//! ================================================
//!
//! Provides Bayesian uncertainty estimates for neural predictions using
//! Monte Carlo dropout sampling. This enables the closed-loop agent to
//! make safer decisions by rejecting predictions with high uncertainty.
//!
//! ## Types of Uncertainty
//!
//! 1. **Epistemic Uncertainty**: Model uncertainty due to limited training data.
//!    Can be reduced with more training data. Captured by variance across
//!    Monte Carlo dropout samples.
//!
//! 2. **Aleatoric Uncertainty**: Inherent noise in the data (e.g., wind, rain).
//!    Cannot be reduced with more data. Captured by predictive variance.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use technical_architecture::uncertainty_estimator::{
//!     MCDropoutUncertaintyEstimator, UncertaintyEstimate,
//! };
//!
//! let estimator = MCDropoutUncertaintyEstimator::new(30, 0.1);
//!
//! // Run N forward passes with dropout enabled
//! let predictions = vec![0.8, 0.82, 0.79, 0.81, 0.83];
//! let uncertainty = estimator.estimate_uncertainty(predictions);
//!
//! assert!(uncertainty.epistemic > 0.0);
//! assert!(uncertainty.aleatoric >= 0.0);
//! assert_eq!(uncertainty.total, uncertainty.epistemic + uncertainty.aleatoric);
//! ```

use serde::{Deserialize, Serialize};
use std::f32;

/// Uncertainty estimates for predictions
///
/// Contains both epistemic (model) and aleatoric (data) uncertainty components.
/// The total uncertainty is the sum of both components.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyEstimate {
    /// Epistemic uncertainty (model knowledge) - can be reduced with more data
    /// This is the variance across Monte Carlo dropout samples
    pub epistemic: f32,

    /// Aleatoric uncertainty (data noise) - irreducible
    /// This is the mean predictive variance (inherent noise in the data)
    pub aleatoric: f32,

    /// Total uncertainty (combined)
    /// total = epistemic + aleatoric
    pub total: f32,
}

impl UncertaintyEstimate {
    /// Create a new uncertainty estimate
    ///
    /// # Arguments
    /// * `epistemic` - Model uncertainty (variance across MC samples)
    /// * `aleatoric` - Data uncertainty (inherent noise)
    ///
    /// # Returns
    /// A new `UncertaintyEstimate` with total = epistemic + aleatoric
    #[must_use]
    pub const fn new(epistemic: f32, aleatoric: f32) -> Self {
        Self {
            epistemic,
            aleatoric,
            total: epistemic + aleatoric,
        }
    }

    /// Create an uncertainty estimate from epistemic only (no aleatoric)
    #[must_use]
    pub const fn from_epistemic(epistemic: f32) -> Self {
        Self {
            epistemic,
            aleatoric: 0.0,
            total: epistemic,
        }
    }

    /// Create a zero uncertainty estimate (certain prediction)
    #[must_use]
    pub const fn certain() -> Self {
        Self {
            epistemic: 0.0,
            aleatoric: 0.0,
            total: 0.0,
        }
    }

    /// Check if total uncertainty exceeds a threshold
    #[must_use]
    pub const fn exceeds_threshold(&self, threshold: f32) -> bool {
        self.total > threshold
    }
}

impl Default for UncertaintyEstimate {
    fn default() -> Self {
        Self::certain()
    }
}

/// Monte Carlo Dropout Uncertainty Estimator
///
/// Estimates uncertainty by running multiple forward passes with dropout
/// enabled at inference time. The variance across samples represents
/// epistemic uncertainty.
///
/// # Algorithm
///
/// 1. Enable dropout at inference time
/// 2. Run N forward passes
/// 3. Compute mean and variance of predictions
/// 4. Epistemic = variance across samples
/// 5. Aleatoric = mean of predictive variance (if available)
/// 6. Total = epistemic + aleatoric
#[derive(Debug, Clone)]
pub struct MCDropoutUncertaintyEstimator {
    /// Number of MC samples to draw
    num_samples: usize,

    /// Dropout rate (0.0-1.0)
    dropout_rate: f32,

    /// Minimum variance threshold (prevents division by zero)
    min_variance: f32,
}

impl MCDropoutUncertaintyEstimator {
    /// Create a new MC dropout uncertainty estimator
    ///
    /// # Arguments
    /// * `num_samples` - Number of forward passes to run (typically 10-30)
    /// * `dropout_rate` - Dropout rate used during training
    ///
    /// # Returns
    /// A new `MCDropoutUncertaintyEstimator`
    #[must_use]
    pub const fn new(num_samples: usize, dropout_rate: f32) -> Self {
        Self {
            num_samples,
            dropout_rate,
            min_variance: 1e-6,
        }
    }

    /// Create with default settings (30 samples, 0.1 dropout)
    #[must_use]
    pub const fn default_settings() -> Self {
        Self {
            num_samples: 30,
            dropout_rate: 0.1,
            min_variance: 1e-6,
        }
    }

    /// Estimate uncertainty from a collection of predictions
    ///
    /// # Arguments
    /// * `predictions` - Vector of predictions from MC dropout samples
    ///
    /// # Returns
    /// An `UncertaintyEstimate` with epistemic and total uncertainty
    ///
    /// # Algorithm
    /// 1. Compute mean of predictions
    /// 2. Compute variance across predictions (epistemic)
    /// 3. Total = epistemic (aleatoric requires per-sample variance)
    #[must_use]
    pub fn estimate_uncertainty(&self, predictions: &[f32]) -> UncertaintyEstimate {
        if predictions.is_empty() {
            return UncertaintyEstimate::certain();
        }

        if predictions.len() == 1 {
            // Single sample: no uncertainty information
            return UncertaintyEstimate::certain();
        }

        // Compute mean
        let mean = predictions.iter().sum::<f32>() / predictions.len() as f32;

        // Compute variance (epistemic uncertainty)
        let variance = predictions
            .iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f32>()
            / (predictions.len() - 1) as f32;

        let epistemic = variance.max(self.min_variance);

        // For now, aleatoric is 0 (requires per-sample predictive variance)
        let aleatoric = 0.0;

        UncertaintyEstimate::new(epistemic, aleatoric)
    }

    /// Estimate uncertainty with aleatoric component
    ///
    /// # Arguments
    /// * `predictions` - Vector of predictions from MC dropout samples
    /// * `predictive_variances` - Per-sample predictive variances (aleatoric)
    ///
    /// # Returns
    /// An `UncertaintyEstimate` with both uncertainty components
    #[must_use]
    pub fn estimate_uncertainty_with_aleatoric(
        &self,
        predictions: &[f32],
        predictive_variances: &[f32],
    ) -> UncertaintyEstimate {
        if predictions.is_empty() || predictive_variances.is_empty() {
            return UncertaintyEstimate::certain();
        }

        // Compute epistemic (variance across predictions)
        let mean = predictions.iter().sum::<f32>() / predictions.len() as f32;
        let epistemic = predictions
            .iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f32>()
            / (predictions.len() - 1).max(1) as f32;

        // Compute aleatoric (mean of predictive variances)
        let aleatoric = predictive_variances.iter().sum::<f32>() / predictive_variances.len() as f32;

        UncertaintyEstimate::new(
            epistemic.max(self.min_variance),
            aleatoric.max(self.min_variance),
        )
    }

    /// Get the number of MC samples
    #[must_use]
    pub const fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Get the dropout rate
    #[must_use]
    pub const fn dropout_rate(&self) -> f32 {
        self.dropout_rate
    }
}

impl Default for MCDropoutUncertaintyEstimator {
    fn default() -> Self {
        Self::default_settings()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncertainty_estimate_fields() {
        let estimate = UncertaintyEstimate::new(0.2, 0.1);
        assert_eq!(estimate.epistemic, 0.2);
        assert_eq!(estimate.aleatoric, 0.1);
        assert_eq!(estimate.total, 0.3);
    }

    #[test]
    fn test_uncertainty_total_is_sum() {
        let estimate = UncertaintyEstimate::new(0.15, 0.25);
        assert_eq!(estimate.total, estimate.epistemic + estimate.aleatoric);
    }

    #[test]
    fn test_uncertainty_certain() {
        let estimate = UncertaintyEstimate::certain();
        assert_eq!(estimate.epistemic, 0.0);
        assert_eq!(estimate.aleatoric, 0.0);
        assert_eq!(estimate.total, 0.0);
    }

    #[test]
    fn test_uncertainty_from_epistemic() {
        let estimate = UncertaintyEstimate::from_epistemic(0.5);
        assert_eq!(estimate.epistemic, 0.5);
        assert_eq!(estimate.aleatoric, 0.0);
        assert_eq!(estimate.total, 0.5);
    }

    #[test]
    fn test_uncertainty_exceeds_threshold() {
        let estimate = UncertaintyEstimate::new(0.5, 0.2);
        assert!(estimate.exceeds_threshold(0.6));
        assert!(!estimate.exceeds_threshold(0.8));
    }

    #[test]
    fn test_uncertainty_default() {
        let estimate = UncertaintyEstimate::default();
        assert_eq!(estimate.total, 0.0);
    }

    #[test]
    fn test_mc_dropout_variance_positive() {
        let estimator = MCDropoutUncertaintyEstimator::new(10, 0.1);
        let predictions = vec![0.7, 0.8, 0.75, 0.82, 0.78];
        let uncertainty = estimator.estimate_uncertainty(&predictions);

        assert!(uncertainty.epistemic > 0.0);
        assert!(uncertainty.total > 0.0);
    }

    #[test]
    fn test_mc_dropout_same_input_low_variance() {
        let estimator = MCDropoutUncertaintyEstimator::new(10, 0.1);
        // Same predictions = low variance
        let predictions = vec![0.8; 10];
        let uncertainty = estimator.estimate_uncertainty(&predictions);

        // Variance should be very close to 0
        assert!(uncertainty.epistemic < 0.01);
    }

    #[test]
    fn test_mc_dropout_empty_predictions() {
        let estimator = MCDropoutUncertaintyEstimator::new(10, 0.1);
        let predictions: Vec<f32> = vec![];
        let uncertainty = estimator.estimate_uncertainty(&predictions);

        assert_eq!(uncertainty.total, 0.0);
    }

    #[test]
    fn test_mc_dropout_single_prediction() {
        let estimator = MCDropoutUncertaintyEstimator::new(10, 0.1);
        let predictions = vec![0.8];
        let uncertainty = estimator.estimate_uncertainty(&predictions);

        assert_eq!(uncertainty.total, 0.0);
    }

    #[test]
    fn test_mc_dropout_with_aleatoric() {
        let estimator = MCDropoutUncertaintyEstimator::new(10, 0.1);
        let predictions = vec![0.7, 0.8, 0.75, 0.82, 0.78];
        let variances = vec![0.01, 0.02, 0.015, 0.018, 0.012];
        let uncertainty = estimator.estimate_uncertainty_with_aleatoric(&predictions, &variances);

        assert!(uncertainty.epistemic > 0.0);
        assert!(uncertainty.aleatoric > 0.0);
        assert_eq!(uncertainty.total, uncertainty.epistemic + uncertainty.aleatoric);
    }

    #[test]
    fn test_mc_dropout_num_samples() {
        let estimator = MCDropoutUncertaintyEstimator::new(20, 0.15);
        assert_eq!(estimator.num_samples(), 20);
        assert_eq!(estimator.dropout_rate(), 0.15);
    }

    #[test]
    fn test_mc_dropout_default_settings() {
        let estimator = MCDropoutUncertaintyEstimator::default_settings();
        assert_eq!(estimator.num_samples(), 30);
        assert_eq!(estimator.dropout_rate(), 0.1);
    }

    #[test]
    fn test_uncertainty_serialization() {
        let estimate = UncertaintyEstimate::new(0.2, 0.1);
        let json = serde_json::to_string(&estimate).unwrap();
        let decoded: UncertaintyEstimate = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.epistemic, 0.2);
        assert_eq!(decoded.aleatoric, 0.1);
        assert_eq!(decoded.total, 0.3);
    }
}
