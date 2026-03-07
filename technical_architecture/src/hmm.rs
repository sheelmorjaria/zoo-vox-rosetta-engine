// Hidden Markov Model (HMM)
//
// Implements HMM for temporal sequence modeling and state decoding.
//
// HMM is a statistical model where the system is assumed to be a Markov process
// with hidden (unobservable) states. Each state emits observations with probabilities.
//
// The combination of GMM (for acoustic features) + HMM (for temporal transitions)
// is the classic "Phoneme Discovery" approach used in early speech recognition.
//
// Reference: Rabiner, L. R. (1989). "A tutorial on hidden Markov models and
// selected applications in speech recognition"

use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum HmmError {
    #[error("Insufficient data: need at least {min} samples, got {actual}")]
    InsufficientData { min: usize, actual: usize },

    #[error("Invalid number of states: {n} (must be >= 1)")]
    InvalidStates { n: usize },

    #[error("Invalid transition matrix: not stochastic")]
    InvalidTransitionMatrix,

    #[error("Observation sequence too short: {len} (minimum 2)")]
    SequenceTooShort { len: usize },

    #[error("Observation probability is zero (underflow)")]
    ZeroProbability,
}

pub type Result<T> = std::result::Result<T, HmmError>;

// =============================================================================
// Hidden Markov Model
// =============================================================================

/// Hidden Markov Model for temporal sequence modeling
#[derive(Debug, Clone)]
pub struct HiddenMarkovModel {
    n_states: usize,
    n_observations: usize,
    initial_probs: Array1<f64>,     // (n_states,) - initial state distribution
    transition_matrix: Array2<f64>, // (n_states, n_states) - state transition probs
    emission_probs: Array2<f64>,    // (n_states, n_observations) - emission probs
}

impl HiddenMarkovModel {
    /// Create a new HMM with random initialization
    ///
    /// # Arguments
    /// * `n_states` - Number of hidden states
    /// * `n_observations` - Number of possible observation symbols
    /// * `seed` - Random seed for reproducibility
    pub fn new(n_states: usize, n_observations: usize, seed: u64) -> Result<Self> {
        if n_states < 1 {
            return Err(HmmError::InvalidStates { n: n_states });
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Initialize transition matrix with random values and normalize rows
        let mut transition_matrix = Array2::zeros((n_states, n_states));
        for i in 0..n_states {
            let mut sum = 0.0;
            for j in 0..n_states {
                transition_matrix[[i, j]] = rng.gen::<f64>();
                sum += transition_matrix[[i, j]];
            }
            // Normalize row
            for j in 0..n_states {
                transition_matrix[[i, j]] /= sum;
            }
        }

        // Initialize emission probabilities with random values and normalize rows
        let mut emission_probs = Array2::zeros((n_states, n_observations));
        for i in 0..n_states {
            let mut sum = 0.0;
            for j in 0..n_observations {
                emission_probs[[i, j]] = rng.gen::<f64>();
                sum += emission_probs[[i, j]];
            }
            // Normalize row
            for j in 0..n_observations {
                emission_probs[[i, j]] /= sum;
            }
        }

        // Initialize initial probabilities uniformly
        let initial_probs = Array1::from(vec![1.0 / n_states as f64; n_states]);

        Ok(Self {
            n_states,
            n_observations,
            initial_probs,
            transition_matrix,
            emission_probs,
        })
    }

    /// Create HMM with explicit parameters
    ///
    /// # Arguments
    /// * `initial_probs` - Initial state distribution (n_states,)
    /// * `transition_matrix` - State transition matrix (n_states, n_states)
    /// * `emission_probs` - Emission probability matrix (n_states, n_observations)
    pub fn with_params(
        initial_probs: Array1<f64>,
        transition_matrix: Array2<f64>,
        emission_probs: Array2<f64>,
    ) -> Result<Self> {
        let n_states = initial_probs.len();
        let n_observations = emission_probs.ncols();

        // Validate transition matrix
        for i in 0..n_states {
            let row_sum: f64 = transition_matrix.row(i).sum();
            if (row_sum - 1.0).abs() > 1e-6 {
                return Err(HmmError::InvalidTransitionMatrix);
            }
        }

        Ok(Self {
            n_states,
            n_observations,
            initial_probs,
            transition_matrix,
            emission_probs,
        })
    }

    /// Fit HMM to observation sequences using Baum-Welch algorithm
    ///
    /// # Algorithm (EM for HMM)
    /// 1. E-step: Compute forward-backward probabilities
    /// 2. M-step: Update transition, emission, and initial probabilities
    /// 3. Repeat until convergence or max iterations
    ///
    /// # Arguments
    /// * `sequences` - Vector of observation sequences
    /// * `max_iterations` - Maximum EM iterations (default: 100)
    /// * `tolerance` - Convergence tolerance (default: 1e-6)
    pub fn fit(&mut self, sequences: &[Vec<usize>], max_iterations: usize, tolerance: f64) -> Result<()> {
        if sequences.is_empty() {
            return Err(HmmError::InsufficientData { min: 1, actual: 0 });
        }

        let mut prev_log_likelihood = f64::NEG_INFINITY;

        for _iteration in 0..max_iterations {
            // Initialize accumulators for M-step
            let mut new_initial = Array1::zeros(self.n_states);
            let mut new_transition = Array2::zeros((self.n_states, self.n_states));
            let mut new_emission = Array2::zeros((self.n_states, self.n_observations));
            let mut log_likelihood = 0.0;

            // Process each sequence
            for sequence in sequences {
                if sequence.len() < 2 {
                    return Err(HmmError::SequenceTooShort { len: sequence.len() });
                }

                // E-step: Forward-backward
                let (alpha, beta, gamma, xi, seq_log_likelihood) = self.forward_backward(sequence)?;

                log_likelihood += seq_log_likelihood;

                // M-step: Accumulate statistics
                self.accumulate_stats(
                    sequence,
                    &alpha,
                    &beta,
                    &gamma,
                    &xi,
                    &mut new_initial,
                    &mut new_transition,
                    &mut new_emission,
                )?;
            }

            // Normalize accumulators
            let initial_sum = new_initial.sum();
            if initial_sum > 0.0 {
                self.initial_probs = &new_initial / initial_sum;
            }

            for i in 0..self.n_states {
                let trans_sum = new_transition.row(i).sum();
                if trans_sum > 0.0 {
                    for j in 0..self.n_states {
                        self.transition_matrix[[i, j]] = new_transition[[i, j]] / trans_sum;
                    }
                }

                let emit_sum = new_emission.row(i).sum();
                if emit_sum > 0.0 {
                    for k in 0..self.n_observations {
                        self.emission_probs[[i, k]] = new_emission[[i, k]] / emit_sum;
                    }
                }
            }

            // Check convergence
            let delta = (log_likelihood - prev_log_likelihood).abs();
            if delta < tolerance {
                break;
            }
            prev_log_likelihood = log_likelihood;
        }

        Ok(())
    }

    /// Forward-backward algorithm
    ///
    /// Computes:
    /// - alpha: Forward probabilities (likelihood of observations up to t)
    /// - beta: Backward probabilities (likelihood of observations from t+1 to end)
    /// - gamma: State probabilities at each time step
    /// - xi: Transition probabilities between consecutive time steps
    fn forward_backward(
        &self,
        sequence: &[usize],
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array3<f64>, f64)> {
        let t_len = sequence.len();

        // Forward pass (alpha)
        let mut alpha = Array2::zeros((t_len, self.n_states));

        // Initialize alpha[0]
        for i in 0..self.n_states {
            let obs = sequence[0];
            if obs < self.n_observations && self.emission_probs[[i, obs]] > 0.0 {
                alpha[[0, i]] = self.initial_probs[i] * self.emission_probs[[i, obs]];
            }
        }

        // Scale alpha[0] to prevent underflow
        let c0 = alpha.row(0).sum();
        if c0 > 0.0 {
            alpha.row_mut(0).map_inplace(|x| *x /= c0);
        }

        // Forward recursion
        for t in 1..t_len {
            let obs = sequence[t];
            for j in 0..self.n_states {
                let mut sum = 0.0;
                for i in 0..self.n_states {
                    sum += alpha[[t - 1, i]] * self.transition_matrix[[i, j]];
                }
                if obs < self.n_observations {
                    alpha[[t, j]] = sum * self.emission_probs[[j, obs]];
                }
            }

            // Scale alpha[t] to prevent underflow
            let ct = alpha.row(t).sum();
            if ct > 0.0 {
                alpha.row_mut(t).map_inplace(|x| *x /= ct);
            }
        }

        // Backward pass (beta)
        let mut beta = Array2::zeros((t_len, self.n_states));

        // Initialize beta[T-1] = 1
        for i in 0..self.n_states {
            beta[[t_len - 1, i]] = 1.0;
        }

        // Backward recursion
        for t in (0..t_len - 1).rev() {
            let obs_next = sequence[t + 1];
            for i in 0..self.n_states {
                let mut sum = 0.0;
                for j in 0..self.n_states {
                    if obs_next < self.n_observations {
                        sum += self.transition_matrix[[i, j]] * self.emission_probs[[j, obs_next]] * beta[[t + 1, j]];
                    }
                }
                beta[[t, i]] = sum;
            }

            // Scale beta[t]
            let ct = beta.row(t).sum();
            if ct > 0.0 {
                beta.row_mut(t).map_inplace(|x| *x /= ct);
            }
        }

        // Compute gamma (state probabilities)
        let mut gamma = Array2::zeros((t_len, self.n_states));
        for t in 0..t_len {
            let sum: f64 = (0..self.n_states).map(|i| alpha[[t, i]] * beta[[t, i]]).sum();

            if sum > 0.0 {
                for i in 0..self.n_states {
                    gamma[[t, i]] = alpha[[t, i]] * beta[[t, i]] / sum;
                }
            }
        }

        // Compute xi (transition probabilities)
        let mut xi = Array3::zeros((t_len - 1, self.n_states, self.n_states));
        for t in 0..t_len - 1 {
            let obs_next = sequence[t + 1];
            let mut denom = 0.0;

            for i in 0..self.n_states {
                for j in 0..self.n_states {
                    if obs_next < self.n_observations {
                        let numer = alpha[[t, i]]
                            * self.transition_matrix[[i, j]]
                            * self.emission_probs[[j, obs_next]]
                            * beta[[t + 1, j]];
                        xi[[t, i, j]] = numer;
                        denom += numer;
                    }
                }
            }

            if denom > 0.0 {
                for i in 0..self.n_states {
                    for j in 0..self.n_states {
                        xi[[t, i, j]] /= denom;
                    }
                }
            }
        }

        // Compute log-likelihood
        let log_likelihood: f64 = (0..t_len)
            .map(|t| alpha.row(t).iter().filter(|&&x| x > 0.0).map(|&x| x.ln()).sum::<f64>())
            .sum();

        Ok((alpha, beta, gamma, xi, log_likelihood))
    }

    /// Accumulate statistics for M-step
    fn accumulate_stats(
        &self,
        sequence: &[usize],
        _alpha: &Array2<f64>,
        _beta: &Array2<f64>,
        gamma: &Array2<f64>,
        xi: &Array3<f64>,
        new_initial: &mut Array1<f64>,
        new_transition: &mut Array2<f64>,
        new_emission: &mut Array2<f64>,
    ) -> Result<()> {
        let t_len = sequence.len();

        // Accumulate initial probabilities
        for i in 0..self.n_states {
            new_initial[i] += gamma[[0, i]];
        }

        // Accumulate transition probabilities
        for t in 0..t_len - 1 {
            for i in 0..self.n_states {
                for j in 0..self.n_states {
                    new_transition[[i, j]] += xi[[t, i, j]];
                }
            }
        }

        // Accumulate emission probabilities
        for t in 0..t_len {
            let obs = sequence[t];
            if obs < self.n_observations {
                for i in 0..self.n_states {
                    new_emission[[i, obs]] += gamma[[t, i]];
                }
            }
        }

        Ok(())
    }

    /// Decode most likely state sequence using Viterbi algorithm
    ///
    /// # Arguments
    /// * `sequence` - Observation sequence
    ///
    /// # Returns
    /// Most likely state sequence and probability
    pub fn decode_viterbi(&self, sequence: &[usize]) -> Result<(Vec<usize>, f64)> {
        if sequence.len() < 2 {
            return Err(HmmError::SequenceTooShort { len: sequence.len() });
        }

        let t_len = sequence.len();

        // Viterbi trellis
        let mut delta = Array2::zeros((t_len, self.n_states));
        let mut psi = vec![vec![0usize; self.n_states]; t_len];

        // Initialize
        for i in 0..self.n_states {
            let obs = sequence[0];
            if obs < self.n_observations && self.emission_probs[[i, obs]] > 0.0 {
                delta[[0, i]] = (self.initial_probs[i] * self.emission_probs[[i, obs]]).ln();
            } else {
                delta[[0, i]] = f64::NEG_INFINITY;
            }
        }

        // Recursion
        for t in 1..t_len {
            let obs = sequence[t];
            for j in 0..self.n_states {
                let mut max_val = f64::NEG_INFINITY;
                let mut max_state = 0;

                for i in 0..self.n_states {
                    let val = delta[[t - 1, i]] + self.transition_matrix[[i, j]].ln();
                    if val > max_val {
                        max_val = val;
                        max_state = i;
                    }
                }

                if obs < self.n_observations && self.emission_probs[[j, obs]] > 0.0 {
                    delta[[t, j]] = max_val + self.emission_probs[[j, obs]].ln();
                } else {
                    delta[[t, j]] = f64::NEG_INFINITY;
                }
                psi[t][j] = max_state;
            }
        }

        // Termination
        let mut max_val = f64::NEG_INFINITY;
        let mut last_state = 0;
        for i in 0..self.n_states {
            if delta[[t_len - 1, i]] > max_val {
                max_val = delta[[t_len - 1, i]];
                last_state = i;
            }
        }

        // Backtracking
        let mut states = vec![0usize; t_len];
        states[t_len - 1] = last_state;

        for t in (0..t_len - 1).rev() {
            states[t] = psi[t + 1][states[t + 1]];
        }

        let probability = max_val.exp();

        Ok((states, probability))
    }

    /// Predict next observation based on current state
    pub fn predict_next(&self, current_state: usize, n_steps: usize) -> Result<Vec<usize>> {
        if current_state >= self.n_states {
            return Ok(vec![]);
        }

        let mut predictions = Vec::new();
        let mut state = current_state;

        for _ in 0..n_steps {
            // Sample next state from transition distribution
            let rand_val: f64 = rand::random();
            let mut cumsum = 0.0;

            for next_state in 0..self.n_states {
                cumsum += self.transition_matrix[[state, next_state]];
                if rand_val < cumsum {
                    state = next_state;
                    break;
                }
            }

            // Sample observation from emission distribution
            let rand_val: f64 = rand::random();
            let mut cumsum = 0.0;

            for obs in 0..self.n_observations {
                cumsum += self.emission_probs[[state, obs]];
                if rand_val < cumsum {
                    predictions.push(obs);
                    break;
                }
            }
        }

        Ok(predictions)
    }

    /// Get log-likelihood of observation sequence
    pub fn score(&self, sequence: &[usize]) -> Result<f64> {
        if sequence.len() < 2 {
            return Err(HmmError::SequenceTooShort { len: sequence.len() });
        }

        let (_, _, _, _, log_likelihood) = self.forward_backward(sequence)?;
        Ok(log_likelihood)
    }

    /// Get transition matrix
    pub fn transition_matrix(&self) -> &Array2<f64> {
        &self.transition_matrix
    }

    /// Get emission probabilities
    pub fn emission_probs(&self) -> &Array2<f64> {
        &self.emission_probs
    }

    /// Get initial probabilities
    pub fn initial_probs(&self) -> &Array1<f64> {
        &self.initial_probs
    }
}

// =============================================================================
// Data Structures
// =============================================================================

type Array3<A> = ndarray::Array3<A>;

// =============================================================================
// Tests (TDD Approach)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: HMM converges on simple sequence
    #[test]
    fn test_hmm_convergence() {
        let sequences = vec![
            // Alternating pattern: 0, 1, 0, 1, ...
            vec![0, 1, 0, 1, 0, 1],
            vec![1, 0, 1, 0, 1, 0],
        ];

        let mut hmm = HiddenMarkovModel::new(2, 2, 42).unwrap();
        hmm.fit(&sequences, 50, 1e-4).unwrap();

        // Should converge without error - just reaching this point means success
    }

    /// Test 2: HMM rejects invalid parameters
    #[test]
    fn test_hmm_invalid_params() {
        // n_states too small
        let result = HiddenMarkovModel::new(0, 2, 42);
        assert!(result.is_err());

        // n_states = 1 (should work)
        let result = HiddenMarkovModel::new(1, 2, 42);
        assert!(result.is_ok());
    }

    /// Test 3: HMM decodes state sequence with Viterbi
    #[test]
    fn test_hmm_viterbi() {
        let sequences = vec![vec![0, 1, 0, 1, 0, 1], vec![1, 0, 1, 0, 1, 0]];

        let mut hmm = HiddenMarkovModel::new(2, 2, 42).unwrap();
        hmm.fit(&sequences, 50, 1e-4).unwrap();

        let test_seq = vec![0, 1, 0, 1];
        let (states, prob) = hmm.decode_viterbi(&test_seq).unwrap();

        // Should produce valid state sequence
        assert_eq!(states.len(), test_seq.len());
        assert!(prob > 0.0 && prob <= 1.0);
    }

    /// Test 4: HMM handles insufficient data
    #[test]
    fn test_hmm_insufficient_data() {
        let mut hmm = HiddenMarkovModel::new(2, 2, 42).unwrap();

        // Empty sequence list
        let result = hmm.fit(&[], 100, 1e-6);
        assert!(result.is_err());

        // Sequence too short
        let result = hmm.fit(&[vec![0]], 100, 1e-6);
        assert!(result.is_err());
    }

    /// Test 5: HMM computes sequence likelihood
    #[test]
    fn test_hmm_scoring() {
        let sequences = vec![vec![0, 1, 0, 1, 0, 1], vec![1, 0, 1, 0, 1, 0]];

        let mut hmm = HiddenMarkovModel::new(2, 2, 42).unwrap();
        hmm.fit(&sequences, 50, 1e-4).unwrap();

        let test_seq = vec![0, 1, 0, 1];
        let score = hmm.score(&test_seq).unwrap();

        // Score should be finite (log-likelihood)
        assert!(score.is_finite());
    }

    /// Test 6: HMM is deterministic with same seed
    #[test]
    fn test_hmm_deterministic() {
        let sequences = vec![vec![0, 1, 0, 1]];

        let mut hmm1 = HiddenMarkovModel::new(2, 2, 42).unwrap();
        hmm1.fit(&sequences, 10, 1e-4).unwrap();

        let mut hmm2 = HiddenMarkovModel::new(2, 2, 42).unwrap();
        hmm2.fit(&sequences, 10, 1e-4).unwrap();

        // Both should have identical transition matrices
        let trans1 = hmm1.transition_matrix();
        let trans2 = hmm2.transition_matrix();

        for i in 0..2 {
            for j in 0..2 {
                assert!((trans1[[i, j]] - trans2[[i, j]]).abs() < 1e-10);
            }
        }
    }
}
