// Gaussian Mixture Model (GMM)
//
// Implements GMM for acoustic feature modeling and clustering.
//
// GMM is a probabilistic model that assumes all data points are generated
// from a mixture of a finite number of Gaussian distributions with unknown
// parameters.
//
// Reference: Bishop, C. M. (2006). "Pattern Recognition and Machine Learning"

use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand::Rng;
use rand::SeedableRng;
use std::f64::consts::PI;

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum GmmError {
    #[error("Insufficient data: need at least {min} samples, got {actual}")]
    InsufficientData { min: usize, actual: usize },

    #[error("Invalid number of components: {n} (must be >= 1)")]
    InvalidComponents { n: usize },

    #[error("Feature dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Failed to converge: {0}")]
    ConvergenceFailed(String),
}

pub type Result<T> = std::result::Result<T, GmmError>;

// =============================================================================
// Gaussian Mixture Model
// =============================================================================

/// Gaussian Mixture Model for clustering acoustic features
#[derive(Debug, Clone)]
pub struct GaussianMixtureModel {
    n_components: usize,
    means: Array2<f64>,            // (n_components, n_features)
    covariances: Vec<Array2<f64>>, // (n_features, n_features) per component
    weights: Array1<f64>,          // (n_components,) - mixing coefficients
    converged: bool,
}

impl GaussianMixtureModel {
    /// Create a new GMM with random initialization
    ///
    /// # Arguments
    /// * `n_components` - Number of Gaussian components (clusters)
    /// * `n_features` - Dimensionality of feature vectors
    /// * `seed` - Random seed for reproducibility
    pub fn new(n_components: usize, n_features: usize, seed: u64) -> Result<Self> {
        if n_components < 1 {
            return Err(GmmError::InvalidComponents { n: n_components });
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Initialize means randomly
        let mut means = Array2::zeros((n_components, n_features));
        for i in 0..n_components {
            for j in 0..n_features {
                means[[i, j]] = rng.gen_range(-1.0..1.0);
            }
        }

        // Initialize covariances as identity matrices
        let mut covariances = Vec::new();
        for _ in 0..n_components {
            let mut cov = Array2::zeros((n_features, n_features));
            for i in 0..n_features {
                cov[[i, i]] = 1.0; // Identity matrix
            }
            covariances.push(cov);
        }

        // Initialize weights uniformly
        let weights = Array1::from(vec![1.0 / n_components as f64; n_components]);

        Ok(Self {
            n_components,
            means,
            covariances,
            weights,
            converged: false,
        })
    }

    /// Fit GMM to data using Expectation-Maximization (EM)
    ///
    /// # Algorithm
    /// 1. E-step: Compute responsibilities (posterior probabilities)
    /// 2. M-step: Update means, covariances, and weights
    /// 3. Repeat until convergence or max iterations
    ///
    /// # Arguments
    /// * `features` - Feature matrix (n_samples, n_features)
    /// * `max_iterations` - Maximum EM iterations (default: 100)
    /// * `tolerance` - Convergence tolerance (default: 1e-6)
    pub fn fit(&mut self, features: &Array2<f64>, max_iterations: usize, tolerance: f64) -> Result<()> {
        let n_samples = features.nrows();
        let n_features = features.ncols();

        if n_samples < self.n_components {
            return Err(GmmError::InsufficientData {
                min: self.n_components,
                actual: n_samples,
            });
        }

        if n_features != self.means.ncols() {
            return Err(GmmError::DimensionMismatch {
                expected: self.means.ncols(),
                actual: n_features,
            });
        }

        let mut prev_log_likelihood = f64::NEG_INFINITY;

        for iteration in 0..max_iterations {
            // E-step: Compute responsibilities
            let responsibilities = self.e_step(features)?;

            // M-step: Update parameters
            self.m_step(features, &responsibilities)?;

            // Compute log-likelihood
            let log_likelihood = self.compute_log_likelihood(features)?;

            // Check convergence
            let delta = (log_likelihood - prev_log_likelihood).abs();
            if delta < tolerance {
                self.converged = true;
                break;
            }

            prev_log_likelihood = log_likelihood;

            // Prevent infinite loops
            if log_likelihood.is_nan() || log_likelihood.is_infinite() {
                return Err(GmmError::ConvergenceFailed(format!(
                    "Numerical instability at iteration {}",
                    iteration
                )));
            }
        }

        Ok(())
    }

    /// E-step: Compute responsibilities (posterior probabilities)
    fn e_step(&self, features: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = features.nrows();
        let mut responsibilities = Array2::zeros((n_samples, self.n_components));

        for i in 0..n_samples {
            let sample = features.row(i);
            let mut log_prob = Array1::zeros(self.n_components);

            for k in 0..self.n_components {
                log_prob[k] = self.weights[k].ln() + self.log_gaussian_pdf(sample, k)?;
            }

            // Log-sum-exp trick for numerical stability
            let log_sum_exp = self.log_sum_exp(&log_prob)?;
            let max_log = log_prob.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            for k in 0..self.n_components {
                responsibilities[[i, k]] = (log_prob[k] - max_log).exp() * (-log_sum_exp + max_log).exp();
            }

            // Normalize
            let sum: f64 = responsibilities.row(i).sum();
            if sum > 0.0 {
                for k in 0..self.n_components {
                    responsibilities[[i, k]] /= sum;
                }
            }
        }

        Ok(responsibilities)
    }

    /// M-step: Update parameters given responsibilities
    fn m_step(&mut self, features: &Array2<f64>, responsibilities: &Array2<f64>) -> Result<()> {
        let n_samples = features.nrows();
        let n_features = features.ncols();

        // Compute effective number of points per component
        let nk: Array1<f64> = responsibilities.sum_axis(Axis(0));

        // Update weights
        for k in 0..self.n_components {
            self.weights[k] = nk[k] / n_samples as f64;
        }

        // Update means
        for k in 0..self.n_components {
            if nk[k] > 0.0 {
                let mut mean = Array1::<f64>::zeros(n_features);
                for i in 0..n_samples {
                    let sample = features.row(i);
                    for j in 0..n_features {
                        mean[j] += responsibilities[[i, k]] * sample[j];
                    }
                }

                for j in 0..n_features {
                    self.means[[k, j]] = mean[j] / nk[k];
                }
            }
        }

        // Update covariances
        for k in 0..self.n_components {
            if nk[k] > 1.0 {
                let mut cov = Array2::zeros((n_features, n_features));

                for i in 0..n_samples {
                    let sample = features.row(i);
                    for j1 in 0..n_features {
                        for j2 in 0..n_features {
                            let diff1 = sample[j1] - self.means[[k, j1]];
                            let diff2 = sample[j2] - self.means[[k, j2]];
                            cov[[j1, j2]] += responsibilities[[i, k]] * diff1 * diff2;
                        }
                    }
                }

                // Add regularization for numerical stability
                for j in 0..n_features {
                    cov[[j, j]] += 1e-6;
                }

                // Normalize
                for j1 in 0..n_features {
                    for j2 in 0..n_features {
                        cov[[j1, j2]] /= nk[k];
                    }
                }

                self.covariances[k] = cov;
            }
        }

        Ok(())
    }

    /// Predict cluster labels for samples
    pub fn predict(&self, features: &Array2<f64>) -> Result<Vec<usize>> {
        let responsibilities = self.e_step(features)?;
        let mut labels = Vec::new();

        for i in 0..features.nrows() {
            let resp_row = responsibilities.row(i);
            let max_idx = resp_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            labels.push(max_idx);
        }

        Ok(labels)
    }

    /// Compute log Gaussian PDF for a sample
    fn log_gaussian_pdf(&self, sample: ArrayView1<f64>, component: usize) -> Result<f64> {
        let mean = self.means.row(component);
        let cov = &self.covariances[component];
        let n_features = mean.len();

        // Compute (x - μ)^T Σ^-1 (x - μ)
        let mut diff = Vec::with_capacity(n_features);
        for j in 0..n_features {
            diff.push(sample[j] - mean[j]);
        }

        // Use diagonal approximation for simplicity
        let mut log_det = 0.0;
        let mut mahalanobis = 0.0;

        for j in 0..n_features {
            let var = cov[[j, j]];
            log_det += var.ln();
            mahalanobis += diff[j] * diff[j] / var;
        }

        // Log PDF: -0.5 * (n_features * ln(2π) + ln|Σ| + (x-μ)^T Σ^-1 (x-μ))
        let log_pdf = -0.5 * (n_features as f64 * (2.0 * PI).ln() + log_det + mahalanobis);

        Ok(log_pdf)
    }

    /// Compute log-likelihood of data under model
    fn compute_log_likelihood(&self, features: &Array2<f64>) -> Result<f64> {
        let mut log_likelihood = 0.0;

        for i in 0..features.nrows() {
            let sample = features.row(i);
            let mut weighted_log_prob = Array1::zeros(self.n_components);

            for k in 0..self.n_components {
                weighted_log_prob[k] = self.weights[k].ln() + self.log_gaussian_pdf(sample, k)?;
            }

            log_likelihood += self.log_sum_exp(&weighted_log_prob)?;
        }

        Ok(log_likelihood)
    }

    /// Log-sum-exp trick for numerical stability
    fn log_sum_exp(&self, log_probs: &Array1<f64>) -> Result<f64> {
        let max_val = log_probs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f64 = log_probs.iter().map(|&x| (x - max_val).exp()).sum();
        Ok(max_val + sum.ln())
    }

    /// Get AIC (Akaike Information Criterion) for model selection
    pub fn aic(&self, features: &Array2<f64>) -> Result<f64> {
        let log_likelihood = self.compute_log_likelihood(features)?;
        let n_params = self.n_components * (1 + self.means.ncols() + self.means.ncols() * self.means.ncols());
        Ok(2.0 * n_params as f64 - 2.0 * log_likelihood)
    }

    /// Get BIC (Bayesian Information Criterion) for model selection
    pub fn bic(&self, features: &Array2<f64>) -> Result<f64> {
        let log_likelihood = self.compute_log_likelihood(features)?;
        let n_params = self.n_components * (1 + self.means.ncols() + self.means.ncols() * self.means.ncols());
        let n_samples = features.nrows() as f64;
        Ok(n_params as f64 * (n_samples).ln() - 2.0 * log_likelihood)
    }
}

// =============================================================================
// Tests (TDD Approach)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    /// Test 1: GMM converges on well-separated clusters
    #[test]
    fn test_gmm_convergence() {
        // Create two well-separated clusters
        let features = arr2(&[
            // Cluster 0: around (0, 0)
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            // Cluster 1: around (5, 5)
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [5.1, 5.1],
        ]);

        let mut gmm = GaussianMixtureModel::new(2, 2, 42).unwrap();
        gmm.fit(&features, 100, 1e-6).unwrap();

        assert!(gmm.converged, "GMM should converge");
    }

    /// Test 2: GMM rejects invalid parameters
    #[test]
    fn test_gmm_invalid_params() {
        // n_components too small
        let result = GaussianMixtureModel::new(0, 2, 42);
        assert!(result.is_err());

        // n_components is 1 (should work)
        let result = GaussianMixtureModel::new(1, 2, 42);
        assert!(result.is_ok());
    }

    /// Test 3: GMM predicts correct cluster assignments
    #[test]
    fn test_gmm_predict() {
        let features = arr2(&[
            // Cluster 0
            [0.0, 0.0],
            [0.1, 0.0],
            // Cluster 1
            [5.0, 5.0],
            [5.1, 5.0],
        ]);

        let mut gmm = GaussianMixtureModel::new(2, 2, 42).unwrap();
        gmm.fit(&features, 100, 1e-6).unwrap();

        let labels = gmm.predict(&features).unwrap();

        // First two should be in same cluster
        assert_eq!(labels[0], labels[1]);
        // Last two should be in same cluster
        assert_eq!(labels[2], labels[3]);
        // Clusters should be different
        assert_ne!(labels[0], labels[2]);
    }

    /// Test 4: GMM handles insufficient data
    #[test]
    fn test_gmm_insufficient_data() {
        let features = arr2(&[[0.0, 0.0]]);
        let mut gmm = GaussianMixtureModel::new(2, 2, 42).unwrap();

        let result = gmm.fit(&features, 100, 1e-6);
        assert!(result.is_err());
    }

    /// Test 5: GMM computes AIC and BIC
    #[test]
    fn test_gmm_information_criteria() {
        let features = arr2(&[[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]]);

        let mut gmm = GaussianMixtureModel::new(2, 2, 42).unwrap();
        gmm.fit(&features, 100, 1e-6).unwrap();

        let aic = gmm.aic(&features).unwrap();
        let bic = gmm.bic(&features).unwrap();

        // Both should be finite numbers
        assert!(aic.is_finite());
        assert!(bic.is_finite());
    }

    /// Test 6: GMM is deterministic with same seed
    #[test]
    fn test_gmm_deterministic() {
        let features = arr2(&[[0.0, 0.0], [5.0, 5.0]]);

        let mut gmm1 = GaussianMixtureModel::new(2, 2, 42).unwrap();
        gmm1.fit(&features, 10, 1e-6).unwrap();

        let mut gmm2 = GaussianMixtureModel::new(2, 2, 42).unwrap();
        gmm2.fit(&features, 10, 1e-6).unwrap();

        let labels1 = gmm1.predict(&features).unwrap();
        let labels2 = gmm2.predict(&features).unwrap();

        assert_eq!(labels1, labels2, "Results should be deterministic with same seed");
    }
}
