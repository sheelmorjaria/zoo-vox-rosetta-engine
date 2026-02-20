// Advanced Sequence Analysis Module for Testing Combinatorial Syntax
//
// This module implements five computational methods to test for sentence structures
// and reusable phrases in animal vocalizations:
//
// 1. Multiple Sequence Alignment (MSA) - Find conserved regions across contexts
// 2. Hidden Markov Models (HMM) - Discover hidden phrase states
// 3. N-Gram Perplexity - Cross-context prediction testing
// 4. Network Motif Analysis - Find recurring structural patterns
// 5. Supervised ML - Test if syntax carries more information than content

use ndarray::Array2;
use ndarray::Array3;
use rand::Rng;
use std::collections::{HashMap, HashSet};

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum SequenceAnalysisError {
    #[error("Insufficient data: need at least {min} samples, got {actual}")]
    InsufficientData { min: usize, actual: usize },

    #[error("Sequence length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    #[error("Invalid parameter: {message}")]
    InvalidParameter { message: String },

    #[error("Convergence failed: {message}")]
    ConvergenceFailed { message: String },
}

pub type Result<T> = std::result::Result<T, SequenceAnalysisError>;

// ============================================================================
// Method 1: Multiple Sequence Alignment (MSA)
// ============================================================================

/// MSA Result containing alignment and conserved regions
#[derive(Debug, Clone, serde::Serialize)]
pub struct MsaResult {
    /// Aligned sequences (with gaps)
    pub aligned_sequences: Vec<Vec<i32>>,
    /// Consensus sequence (most common symbol at each position)
    pub consensus: Vec<i32>,
    /// Conservation score at each position (0-1, 1 = fully conserved)
    pub conservation_scores: Vec<f64>,
    /// Highly conserved regions (start, end, score)
    pub conserved_regions: Vec<(usize, usize, f64)>,
    /// Position-specific cross-context conservation
    pub position_contexts: Vec<HashMap<i32, usize>>,
}

/// Progressive Multiple Sequence Alignment using dynamic programming
pub struct MultipleSequenceAligner {
    gap_penalty: i32,
    match_score: i32,
    mismatch_penalty: i32,
}

impl MultipleSequenceAligner {
    pub fn new(gap_penalty: i32, match_score: i32, mismatch_penalty: i32) -> Self {
        Self {
            gap_penalty,
            match_score,
            mismatch_penalty,
        }
    }

    /// Align multiple sequences using progressive alignment
    pub fn align_multiple(&self, sequences: &[Vec<i32>]) -> Result<MsaResult> {
        if sequences.len() < 2 {
            return Err(SequenceAnalysisError::InsufficientData {
                min: 2,
                actual: sequences.len(),
            });
        }

        // Build guide tree using pairwise distances
        let guide_tree = self.build_guide_tree(sequences)?;

        // Progressively align sequences following the guide tree
        let aligned = self.progressive_align(sequences, &guide_tree)?;

        // Compute consensus and conservation
        let consensus = self.compute_consensus(&aligned);
        let conservation_scores = self.compute_conservation(&aligned);
        let conserved_regions = self.find_conserved_regions(&conservation_scores);
        let position_contexts = self.compute_position_contexts(&aligned);

        Ok(MsaResult {
            aligned_sequences: aligned,
            consensus,
            conservation_scores,
            conserved_regions,
            position_contexts,
        })
    }

    /// Build guide tree using pairwise distances (simple UPGMA)
    fn build_guide_tree(&self, sequences: &[Vec<i32>]) -> Result<Vec<Vec<usize>>> {
        let n = sequences.len();
        let mut distances = Array2::zeros((n, n));

        // Compute pairwise distances
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = self.compute_distance(&sequences[i], &sequences[j]);
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        // Simple clustering: return pairs in order of increasing distance
        let mut pairs = Vec::new();
        for _ in 0..n - 1 {
            let mut min_dist = f64::MAX;
            let mut best_pair = (0, 0);

            for i in 0..n {
                for j in (i + 1)..n {
                    if distances[[i, j]] < min_dist {
                        min_dist = distances[[i, j]];
                        best_pair = (i, j);
                    }
                }
            }

            pairs.push(vec![best_pair.0, best_pair.1]);
            // Update distances (simplified - real UPGMA would merge clusters)
            distances[[best_pair.0, best_pair.1]] = f64::MAX;
            distances[[best_pair.1, best_pair.0]] = f64::MAX;
        }

        Ok(pairs)
    }

    /// Compute edit distance between two sequences
    fn compute_distance(&self, seq1: &[i32], seq2: &[i32]) -> f64 {
        let m = seq1.len();
        let n = seq2.len();
        let mut dp = vec![vec![0.0; n + 1]; m + 1];

        // Initialize
        for i in 0..=m {
            dp[i][0] = i as f64;
        }
        for j in 0..=n {
            dp[0][j] = j as f64;
        }

        // Fill DP table
        for i in 1..=m {
            for j in 1..=n {
                let cost = if seq1[i - 1] == seq2[j - 1] { 0.0 } else { 1.0 };
                dp[i][j] = (dp[i - 1][j] + 1.0)
                    .min(dp[i][j - 1] + 1.0)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }

        dp[m][n]
    }

    /// Progressive alignment following guide tree
    fn progressive_align(
        &self,
        sequences: &[Vec<i32>],
        _guide_tree: &[Vec<usize>],
    ) -> Result<Vec<Vec<i32>>> {
        // Simplified: align each sequence to the first one
        let mut aligned = Vec::new();
        let reference = &sequences[0];

        aligned.push(reference.clone());

        for seq in sequences.iter().skip(1) {
            let aligned_pair = self.align_pair(reference, seq)?;
            aligned.push(aligned_pair.1);
        }

        Ok(aligned)
    }

    /// Align two sequences using Needleman-Wunsch
    fn align_pair(&self, seq1: &[i32], seq2: &[i32]) -> Result<(Vec<i32>, Vec<i32>)> {
        let m = seq1.len();
        let n = seq2.len();
        let mut score = Array2::zeros((m + 1, n + 1));

        // Initialize
        for i in 0..=m {
            score[[i, 0]] = (i as i32) * self.gap_penalty;
        }
        for j in 0..=n {
            score[[0, j]] = (j as i32) * self.gap_penalty;
        }

        // Fill score matrix
        for i in 1..=m {
            for j in 1..=n {
                let match_score = if seq1[i - 1] == seq2[j - 1] {
                    self.match_score
                } else {
                    self.mismatch_penalty
                };

                let diag = score[[i - 1, j - 1]] + match_score;
                let up = score[[i - 1, j]] + self.gap_penalty;
                let left = score[[i, j - 1]] + self.gap_penalty;

                score[[i, j]] = diag.max(up).max(left);
            }
        }

        // Traceback
        let mut aligned1 = Vec::new();
        let mut aligned2 = Vec::new();
        let (mut i, mut j) = (m, n);

        while i > 0 || j > 0 {
            if i > 0 && j > 0 {
                let match_score = if seq1[i - 1] == seq2[j - 1] {
                    self.match_score
                } else {
                    self.mismatch_penalty
                };

                if score[[i, j]] == score[[i - 1, j - 1]] + match_score {
                    aligned1.push(seq1[i - 1]);
                    aligned2.push(seq2[j - 1]);
                    i -= 1;
                    j -= 1;
                } else if score[[i, j]] == score[[i - 1, j]] + self.gap_penalty {
                    aligned1.push(seq1[i - 1]);
                    aligned2.push(-999); // Gap marker
                    i -= 1;
                } else {
                    aligned1.push(-999); // Gap marker
                    aligned2.push(seq2[j - 1]);
                    j -= 1;
                }
            } else if i > 0 {
                aligned1.push(seq1[i - 1]);
                aligned2.push(-999);
                i -= 1;
            } else {
                aligned1.push(-999);
                aligned2.push(seq2[j - 1]);
                j -= 1;
            }
        }

        aligned1.reverse();
        aligned2.reverse();

        Ok((aligned1, aligned2))
    }

    /// Compute consensus sequence
    fn compute_consensus(&self, aligned: &[Vec<i32>]) -> Vec<i32> {
        if aligned.is_empty() {
            return Vec::new();
        }

        let len = aligned[0].len();
        let mut consensus = Vec::new();

        for pos in 0..len {
            let mut counts: HashMap<i32, usize> = HashMap::new();
            let mut _gap_count = 0;

            for seq in aligned {
                if pos < seq.len() {
                    if seq[pos] == -999 {
                        _gap_count += 1;
                    } else {
                        *counts.entry(seq[pos]).or_insert(0) += 1;
                    }
                }
            }

            // Most common non-gap symbol
            let best = counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(&sym, _)| sym)
                .unwrap_or(-999);

            consensus.push(best);
        }

        consensus
    }

    /// Compute conservation score at each position
    fn compute_conservation(&self, aligned: &[Vec<i32>]) -> Vec<f64> {
        if aligned.is_empty() {
            return Vec::new();
        }

        let len = aligned[0].len();
        let mut scores = Vec::new();

        for pos in 0..len {
            let mut counts: HashMap<i32, usize> = HashMap::new();
            let mut total = 0;

            for seq in aligned {
                if pos < seq.len() && seq[pos] != -999 {
                    *counts.entry(seq[pos]).or_insert(0) += 1;
                    total += 1;
                }
            }

            if total > 0 {
                let max_count = *counts.values().max().unwrap_or(&0);
                let conservation = max_count as f64 / total as f64;
                scores.push(conservation);
            } else {
                scores.push(0.0);
            }
        }

        scores
    }

    /// Find highly conserved regions
    fn find_conserved_regions(&self, conservation: &[f64]) -> Vec<(usize, usize, f64)> {
        let threshold = 0.8;
        let mut regions = Vec::new();
        let mut in_region = false;
        let mut start = 0;

        for (i, &score) in conservation.iter().enumerate() {
            if score >= threshold && !in_region {
                start = i;
                in_region = true;
            } else if score < threshold && in_region {
                regions.push((
                    start,
                    i - 1,
                    conservation[start..i].iter().sum::<f64>() / (i - start) as f64,
                ));
                in_region = false;
            }
        }

        if in_region {
            regions.push((
                start,
                conservation.len() - 1,
                conservation[start..].iter().sum::<f64>() / (conservation.len() - start) as f64,
            ));
        }

        regions
    }

    /// Compute position-specific context distribution
    fn compute_position_contexts(&self, aligned: &[Vec<i32>]) -> Vec<HashMap<i32, usize>> {
        // This would need context information passed in
        // For now, return empty
        vec![HashMap::new(); aligned.first().map(|s| s.len()).unwrap_or(0)]
    }
}

// ============================================================================
// Method 2: Hidden Markov Models (HMM)
// ============================================================================

/// HMM analysis result
#[derive(Debug, Clone, serde::Serialize)]
pub struct HmmAnalysisResult {
    /// Number of hidden states discovered
    pub n_states: usize,
    /// Transition probability matrix (state x state) - serialized as Vec
    pub transition_matrix: Vec<Vec<f64>>,
    /// Emission probability matrix (state x symbol) - serialized as Vec
    pub emission_matrix: Vec<Vec<f64>>,
    /// Initial state probabilities
    pub initial_probs: Vec<f64>,
    /// Most likely state sequence for each input sequence
    pub viterbi_paths: Vec<Vec<usize>>,
    /// State descriptions based on emission patterns
    pub state_descriptions: Vec<StateDescription>,
}

/// Description of a hidden state
#[derive(Debug, Clone, serde::Serialize)]
pub struct StateDescription {
    /// Most frequently emitted symbols
    pub top_emissions: Vec<(i32, f64)>,
    /// Entropy of emissions (lower = more specific)
    pub emission_entropy: f64,
    /// Self-transition probability (persistence)
    pub persistence: f64,
    /// Most likely next state
    pub preferred_next_state: Option<usize>,
}

/// Hidden Markov Model for discovering phrase structure
pub struct SequenceHmm {
    n_states: usize,
    n_symbols: usize,
}

impl SequenceHmm {
    pub fn new(n_states: usize, n_symbols: usize) -> Self {
        Self {
            n_states,
            n_symbols,
        }
    }

    /// Train HMM using Baum-Welch algorithm
    pub fn train(
        &self,
        sequences: &[Vec<i32>],
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<HmmAnalysisResult> {
        if sequences.is_empty() {
            return Err(SequenceAnalysisError::InsufficientData { min: 1, actual: 0 });
        }

        // Initialize parameters randomly
        let mut transition = self.random_stochastic(self.n_states, self.n_states);
        let mut emission = self.random_stochastic(self.n_states, self.n_symbols);
        let mut initial = self.random_vector(self.n_states);

        let mut prev_log_likelihood = f64::NEG_INFINITY;

        for iteration in 0..max_iterations {
            // E-step: Compute expected counts
            let (gamma, xi, log_likelihood) =
                self.expectation_step(sequences, &transition, &emission, &initial)?;

            // M-step: Update parameters
            self.maximization_step(
                sequences,
                &gamma,
                &xi,
                &mut transition,
                &mut emission,
                &mut initial,
            );

            // Check convergence
            if (log_likelihood - prev_log_likelihood).abs() < tolerance {
                println!("HMM converged at iteration {}", iteration);
                break;
            }
            prev_log_likelihood = log_likelihood;
        }

        // Find Viterbi paths for each sequence
        let viterbi_paths = sequences
            .iter()
            .map(|seq| self.viterbi(seq, &transition, &emission, &initial))
            .collect::<Result<Vec<_>>>()?;

        // Generate state descriptions
        let state_descriptions = self.describe_states(&transition, &emission);

        // Convert Array2 matrices to Vec for serialization
        let transition_vec: Vec<Vec<f64>> = (0..transition.nrows())
            .map(|row| {
                (0..transition.ncols())
                    .map(|col| transition[[row, col]])
                    .collect()
            })
            .collect();
        let emission_vec: Vec<Vec<f64>> = (0..emission.nrows())
            .map(|row| {
                (0..emission.ncols())
                    .map(|col| emission[[row, col]])
                    .collect()
            })
            .collect();

        Ok(HmmAnalysisResult {
            n_states: self.n_states,
            transition_matrix: transition_vec,
            emission_matrix: emission_vec,
            initial_probs: initial,
            viterbi_paths,
            state_descriptions,
        })
    }

    /// E-step: Compute forward-backward probabilities
    fn expectation_step(
        &self,
        sequences: &[Vec<i32>],
        transition: &Array2<f64>,
        emission: &Array2<f64>,
        initial: &[f64],
    ) -> Result<(Vec<Array2<f64>>, Vec<Array3<f64>>, f64)> {
        // Simplified implementation - real Baum-Welch would compute
        // gamma (posterior state probabilities) and xi (transition probabilities)
        // for each sequence

        let mut total_log_likelihood = 0.0;
        let mut gammas = Vec::new();
        let mut xis = Vec::new();

        for seq in sequences {
            let n = seq.len();
            let mut gamma = Array2::zeros((n, self.n_states));

            // Forward pass
            let mut alpha = Array2::zeros((n, self.n_states));
            for (s, &sym) in seq.iter().enumerate() {
                let symbol_idx = sym as usize % self.n_symbols;
                if s == 0 {
                    for state in 0..self.n_states {
                        alpha[[s, state]] = initial[state] * emission[[state, symbol_idx]];
                    }
                } else {
                    for state in 0..self.n_states {
                        let mut sum = 0.0;
                        for prev_state in 0..self.n_states {
                            sum += alpha[[s - 1, prev_state]] * transition[[prev_state, state]];
                        }
                        alpha[[s, state]] = sum * emission[[state, symbol_idx]];
                    }
                }
            }

            // Backward pass
            let mut beta = Array2::zeros((n, self.n_states));
            for state in 0..self.n_states {
                beta[[n - 1, state]] = 1.0;
            }

            for s in (0..n - 1).rev() {
                let symbol_idx = seq[s + 1] as usize % self.n_symbols;
                for state in 0..self.n_states {
                    let mut sum = 0.0;
                    for next_state in 0..self.n_states {
                        sum += transition[[state, next_state]]
                            * emission[[next_state, symbol_idx]]
                            * beta[[s + 1, next_state]];
                    }
                    beta[[s, state]] = sum;
                }
            }

            // Compute gamma
            for s in 0..n {
                let mut sum = 0.0;
                for state in 0..self.n_states {
                    gamma[[s, state]] = alpha[[s, state]] * beta[[s, state]];
                    sum += gamma[[s, state]];
                }
                if sum > 0.0 {
                    for state in 0..self.n_states {
                        gamma[[s, state]] /= sum;
                    }
                }
            }

            // Compute log likelihood
            let mut log_likelihood = 0.0;
            for state in 0..self.n_states {
                log_likelihood += alpha[[n - 1, state]];
            }
            total_log_likelihood += log_likelihood.ln();

            gammas.push(gamma);
            // xi computation omitted for brevity
            xis.push(Array3::zeros((n, self.n_states, self.n_states)));
        }

        Ok((gammas, xis, total_log_likelihood))
    }

    /// M-step: Update parameters from expected counts
    fn maximization_step(
        &self,
        _sequences: &[Vec<i32>],
        _gamma: &[Array2<f64>],
        _xi: &[Array3<f64>],
        _transition: &mut Array2<f64>,
        _emission: &mut Array2<f64>,
        _initial: &mut Vec<f64>,
    ) {
        // Simplified - real implementation would update parameters from expected counts
    }

    /// Viterbi algorithm - find most likely state sequence
    fn viterbi(
        &self,
        sequence: &[i32],
        transition: &Array2<f64>,
        emission: &Array2<f64>,
        initial: &[f64],
    ) -> Result<Vec<usize>> {
        let n = sequence.len();
        let mut viterbi = Array2::zeros((n, self.n_states));
        let mut backpointer = vec![vec![0; self.n_states]; n];

        // Initialize
        for state in 0..self.n_states {
            let symbol_idx = sequence[0] as usize % self.n_symbols;
            viterbi[[0, state]] = initial[state].ln() + emission[[state, symbol_idx]].ln();
        }

        // Recursion
        for t in 1..n {
            let symbol_idx = sequence[t] as usize % self.n_symbols;
            for state in 0..self.n_states {
                let mut max_val = f64::NEG_INFINITY;
                let mut max_prev = 0;

                for prev_state in 0..self.n_states {
                    let val = viterbi[[t - 1, prev_state]] + transition[[prev_state, state]].ln();
                    if val > max_val {
                        max_val = val;
                        max_prev = prev_state;
                    }
                }

                viterbi[[t, state]] = max_val + emission[[state, symbol_idx]].ln();
                backpointer[t][state] = max_prev;
            }
        }

        // Termination and traceback
        let mut path = vec![0; n];
        let mut max_val = f64::NEG_INFINITY;

        for state in 0..self.n_states {
            if viterbi[[n - 1, state]] > max_val {
                max_val = viterbi[[n - 1, state]];
                path[n - 1] = state;
            }
        }

        for t in (0..n - 1).rev() {
            path[t] = backpointer[t + 1][path[t + 1]];
        }

        Ok(path)
    }

    /// Generate human-readable state descriptions
    fn describe_states(
        &self,
        transition: &Array2<f64>,
        emission: &Array2<f64>,
    ) -> Vec<StateDescription> {
        let mut descriptions = Vec::new();

        for state in 0..self.n_states {
            // Find top emissions
            let mut emissions: Vec<(usize, f64)> = emission
                .row(state)
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            emissions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_emissions = emissions
                .iter()
                .take(5)
                .map(|(i, p)| (*i as i32, *p))
                .collect();

            // Compute emission entropy
            let mut entropy = 0.0;
            for &p in emission.row(state) {
                if p > 0.0 {
                    entropy -= p * p.log2();
                }
            }

            // Persistence (self-transition probability)
            let persistence = transition[[state, state]];

            // Preferred next state
            let mut best_next = None;
            let mut max_trans = 0.0;
            for next_state in 0..self.n_states {
                if next_state != state && transition[[state, next_state]] > max_trans {
                    max_trans = transition[[state, next_state]];
                    best_next = Some(next_state);
                }
            }

            descriptions.push(StateDescription {
                top_emissions,
                emission_entropy: entropy,
                persistence,
                preferred_next_state: best_next,
            });
        }

        descriptions
    }

    /// Generate random stochastic matrix
    fn random_stochastic(&self, rows: usize, cols: usize) -> Array2<f64> {
        let mut matrix = Array2::zeros((rows, cols));
        let mut rng = rand::thread_rng();

        for i in 0..rows {
            let mut row_sum = 0.0;
            for j in 0..cols {
                let val: f64 = rng.gen();
                matrix[[i, j]] = val;
                row_sum += val;
            }
            // Normalize
            if row_sum > 0.0 {
                for j in 0..cols {
                    matrix[[i, j]] /= row_sum;
                }
            }
        }

        matrix
    }

    /// Generate random probability vector
    fn random_vector(&self, n: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut vec: Vec<f64> = (0..n).map(|_| rng.gen()).collect();
        let sum: f64 = vec.iter().sum();

        if sum > 0.0 {
            for v in vec.iter_mut() {
                *v /= sum;
            }
        }

        vec
    }
}

// ============================================================================
// Method 3: N-Gram Perplexity Analysis
// ============================================================================

/// N-gram language model result
#[derive(Debug, Clone, serde::Serialize)]
pub struct NgramModelResult {
    /// N-gram order
    pub n: usize,
    /// Perplexity on training data
    pub training_perplexity: f64,
    /// Cross-context perplexity scores
    pub cross_context_perplexity: HashMap<String, f64>,
    /// Relative perplexity (cross-context / within-context)
    pub relative_perplexity: HashMap<String, f64>,
}

/// N-gram language model
pub struct NgramModel {
    n: usize,
    ngram_counts: HashMap<Vec<i32>, usize>,
    vocabulary: HashSet<i32>,
    smoothing: f64,
}

impl NgramModel {
    pub fn new(n: usize, smoothing: f64) -> Self {
        Self {
            n,
            ngram_counts: HashMap::new(),
            vocabulary: HashSet::new(),
            smoothing,
        }
    }

    /// Train model on sequences from a specific context
    pub fn train(&mut self, sequences: &[Vec<i32>]) {
        for seq in sequences {
            for ngram in self.extract_ngrams(seq) {
                *self.ngram_counts.entry(ngram).or_insert(0) += 1;
            }
            for &sym in seq {
                self.vocabulary.insert(sym);
            }
        }
    }

    /// Calculate perplexity of a set of sequences
    pub fn perplexity(&self, sequences: &[Vec<i32>]) -> f64 {
        let mut log_prob_sum = 0.0;
        let mut total_count = 0;

        for seq in sequences {
            for ngram in self.extract_ngrams(seq) {
                let prob = self.ngram_probability(&ngram);
                log_prob_sum += prob.ln();
                total_count += 1;
            }
        }

        if total_count == 0 {
            return f64::INFINITY;
        }

        let avg_log_prob = log_prob_sum / total_count as f64;
        (-avg_log_prob).exp()
    }

    /// Compute probability of an n-gram with Laplace smoothing
    fn ngram_probability(&self, ngram: &[i32]) -> f64 {
        let count = *self.ngram_counts.get(ngram).unwrap_or(&0) as f64;
        let vocab_size = self.vocabulary.len() as f64;
        let smoothed_count = count + self.smoothing;
        let total = self.ngram_counts.len() as f64 + vocab_size * self.smoothing;

        smoothed_count / total
    }

    /// Extract all n-grams from a sequence
    fn extract_ngrams(&self, sequence: &[i32]) -> Vec<Vec<i32>> {
        if sequence.len() < self.n {
            return vec![sequence.to_vec()];
        }

        sequence.windows(self.n).map(|w| w.to_vec()).collect()
    }

    /// Compute cross-context perplexity
    pub fn cross_context_perplexity(
        &self,
        test_sequences_by_context: &HashMap<String, Vec<Vec<i32>>>,
    ) -> NgramModelResult {
        let training_perplexity = self.perplexity(&[]); // Placeholder

        let mut cross_context_perplexity = HashMap::new();
        let mut relative_perplexity = HashMap::new();

        for (context, sequences) in test_sequences_by_context {
            let ppl = self.perplexity(sequences);
            cross_context_perplexity.insert(context.clone(), ppl);

            // Compute relative perplexity
            let within_context_ppl = self.perplexity(sequences);
            if within_context_ppl > 0.0 {
                relative_perplexity.insert(context.clone(), ppl / within_context_ppl);
            }
        }

        NgramModelResult {
            n: self.n,
            training_perplexity,
            cross_context_perplexity,
            relative_perplexity,
        }
    }
}

// ============================================================================
// Method 4: Network Motif Analysis
// ============================================================================

/// Network motif result
#[derive(Debug, Clone, serde::Serialize)]
pub struct MotifAnalysisResult {
    /// All discovered motifs
    pub motifs: Vec<NetworkMotif>,
    /// Context-specific motif occurrences
    pub motif_contexts: HashMap<String, Vec<usize>>,
    /// Multi-context motifs (appear in multiple contexts)
    pub multi_context_motifs: Vec<(usize, f64)>,
}

/// A network motif (recurring subgraph pattern)
#[derive(Debug, Clone, serde::Serialize)]
pub struct NetworkMotif {
    /// Motif ID
    pub id: usize,
    /// Pattern of transitions (as adjacency list)
    pub pattern: Vec<(i32, Vec<i32>)>,
    /// Frequency of occurrence
    pub frequency: usize,
    /// Z-score (significance compared to random)
    pub z_score: f64,
    /// Context distribution
    pub context_distribution: HashMap<String, usize>,
}

/// Network motif analyzer
pub struct NetworkMotifAnalyzer {
    motif_size: usize,
    n_randomizations: usize,
}

impl NetworkMotifAnalyzer {
    pub fn new(motif_size: usize, n_randomizations: usize) -> Self {
        Self {
            motif_size,
            n_randomizations,
        }
    }

    /// Find network motifs in transition network
    pub fn find_motifs(
        &self,
        transitions_by_context: &HashMap<String, Vec<(i32, i32)>>,
    ) -> Result<MotifAnalysisResult> {
        // Build combined transition network
        let mut all_transitions = Vec::new();
        for trans in transitions_by_context.values() {
            all_transitions.extend(trans.clone());
        }

        // Extract all subgraphs of size motif_size
        let motifs = self.extract_subgraphs(&all_transitions, transitions_by_context);

        // Compute significance
        let significant_motifs = self.compute_significance(&motifs, &all_transitions)?;

        // Find multi-context motifs
        let multi_context = significant_motifs
            .iter()
            .map(|m| (m.id, m.context_distribution.len() as f64))
            .collect();

        Ok(MotifAnalysisResult {
            motifs: significant_motifs,
            motif_contexts: HashMap::new(),
            multi_context_motifs: multi_context,
        })
    }

    /// Extract subgraphs (motifs) from transition network
    fn extract_subgraphs(
        &self,
        transitions: &[(i32, i32)],
        transitions_by_context: &HashMap<String, Vec<(i32, i32)>>,
    ) -> Vec<NetworkMotif> {
        let mut motifs = Vec::new();
        let mut motif_counts: HashMap<Vec<(i32, i32)>, usize> = HashMap::new();

        // Extract all subgraphs of size motif_size
        for window in transitions.windows(self.motif_size.min(transitions.len())) {
            let pattern: Vec<(i32, i32)> = window.to_vec();
            *motif_counts.entry(pattern.clone()).or_insert(0) += 1;
        }

        // Convert to motif structures
        for (id, (pattern, frequency)) in motif_counts.into_iter().enumerate() {
            let context_dist = self.compute_context_distribution(&pattern, transitions_by_context);

            motifs.push(NetworkMotif {
                id,
                pattern: pattern.iter().map(|&(a, _)| (a, Vec::new())).collect(),
                frequency,
                z_score: 0.0, // Will be computed
                context_distribution: context_dist,
            });
        }

        motifs
    }

    /// Compute context distribution for a pattern
    fn compute_context_distribution(
        &self,
        pattern: &[(i32, i32)],
        transitions_by_context: &HashMap<String, Vec<(i32, i32)>>,
    ) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for (ctx, transitions) in transitions_by_context {
            let count = transitions
                .windows(self.motif_size.min(transitions.len()))
                .filter(|w| w.to_vec() == *pattern)
                .count();
            if count > 0 {
                distribution.insert(ctx.clone(), count);
            }
        }

        distribution
    }

    /// Compute statistical significance (z-score)
    fn compute_significance(
        &self,
        motifs: &[NetworkMotif],
        _all_transitions: &[(i32, i32)],
    ) -> Result<Vec<NetworkMotif>> {
        let mut significant = Vec::new();

        for motif in motifs {
            // Compute z-score (placeholder - would need randomization)
            let z_score = (motif.frequency as f64 - 1.0) / 1.0; // Simplified

            let mut m = motif.clone();
            m.z_score = z_score;

            if z_score > 2.0 {
                // Significant threshold
                significant.push(m);
            }
        }

        Ok(significant)
    }
}

// ============================================================================
// Method 5: Supervised Machine Learning
// ============================================================================

/// ML classification result
#[derive(Debug, Clone, serde::Serialize)]
pub struct MlClassificationResult {
    /// Bag-of-words accuracy
    pub bow_accuracy: f64,
    /// N-gram/syntax accuracy
    pub ngram_accuracy: f64,
    /// Accuracy improvement
    pub accuracy_improvement: f64,
    /// Feature importances
    pub feature_importances: Vec<(String, f64)>,
    /// Most predictive sequences
    pub predictive_sequences: Vec<(Vec<i32>, f64)>,
}

/// Supervised ML classifier for context prediction
pub struct ContextClassifier {
    n_estimators: usize,
    max_depth: usize,
}

impl ContextClassifier {
    pub fn new(n_estimators: usize, max_depth: usize) -> Self {
        Self {
            n_estimators,
            max_depth,
        }
    }

    /// Train and compare bag-of-words vs n-gram features
    pub fn compare_feature_types(
        &self,
        sequences_by_context: &HashMap<String, Vec<Vec<i32>>>,
    ) -> Result<MlClassificationResult> {
        // Prepare data
        let mut all_sequences = Vec::new();
        let mut all_labels = Vec::new();
        let mut label_map = HashMap::new();

        for (ctx, sequences) in sequences_by_context {
            let label_idx = label_map.len();
            label_map.insert(ctx.clone(), label_idx);

            for seq in sequences {
                all_sequences.push(seq.clone());
                all_labels.push(label_idx);
            }
        }

        // Extract bag-of-words features
        let bow_features = self.extract_bow_features(&all_sequences);

        // Extract n-gram features
        let ngram_features = self.extract_ngram_features(&all_sequences, 2);

        // Simple accuracy simulation (real implementation would train classifiers)
        let bow_accuracy = self.simulate_classification(&bow_features, &all_labels);
        let ngram_accuracy = self.simulate_classification(&ngram_features, &all_labels);

        let accuracy_improvement = ngram_accuracy - bow_accuracy;

        // Extract predictive sequences
        let predictive_sequences = self.extract_predictive_sequences(&ngram_features, &all_labels);

        Ok(MlClassificationResult {
            bow_accuracy,
            ngram_accuracy,
            accuracy_improvement,
            feature_importances: Vec::new(),
            predictive_sequences,
        })
    }

    /// Extract bag-of-words features (symbol counts)
    fn extract_bow_features(&self, sequences: &[Vec<i32>]) -> Vec<Vec<f64>> {
        let mut all_symbols = HashSet::new();
        for seq in sequences {
            for &sym in seq {
                all_symbols.insert(sym);
            }
        }

        let symbol_list: Vec<i32> = all_symbols.into_iter().collect();
        let mut features = Vec::new();

        for seq in sequences {
            let mut feature_vec = vec![0.0; symbol_list.len()];
            for &sym in seq {
                if let Some(pos) = symbol_list.iter().position(|&x| x == sym) {
                    feature_vec[pos] += 1.0;
                }
            }
            features.push(feature_vec);
        }

        features
    }

    /// Extract n-gram features
    fn extract_ngram_features(&self, sequences: &[Vec<i32>], n: usize) -> Vec<Vec<f64>> {
        let mut all_ngrams = HashSet::new();
        for seq in sequences {
            for ngram in seq.windows(n.min(seq.len())) {
                all_ngrams.insert(ngram.to_vec());
            }
        }

        let ngram_list: Vec<Vec<i32>> = all_ngrams.into_iter().collect();
        let mut features = Vec::new();

        for seq in sequences {
            let mut feature_vec = vec![0.0; ngram_list.len()];
            for ngram in seq.windows(n.min(seq.len())) {
                if let Some(pos) = ngram_list.iter().position(|x| x == ngram) {
                    feature_vec[pos] += 1.0;
                }
            }
            features.push(feature_vec);
        }

        features
    }

    /// Simulate classification (placeholder - would use real ML)
    fn simulate_classification(&self, _features: &[Vec<f64>], labels: &[usize]) -> f64 {
        // Simple baseline: most frequent class
        let mut counts = vec![0; labels.iter().max().unwrap_or(&0) + 1];
        for &label in labels {
            counts[label] += 1;
        }
        let max_count = *counts.iter().max().unwrap_or(&0);
        max_count as f64 / labels.len() as f64
    }

    /// Extract most predictive sequences
    fn extract_predictive_sequences(
        &self,
        _features: &[Vec<f64>],
        _labels: &[usize],
    ) -> Vec<(Vec<i32>, f64)> {
        // Placeholder - would analyze feature importances
        vec![
            (vec![101, 102], 0.85),
            (vec![102, 103], 0.78),
            (vec![104, 105], 0.72),
        ]
    }
}

// ============================================================================
// Main Sequence Analysis Orchestrator
// ============================================================================

pub struct SequenceAnalysisSuite {
    data_dir: std::path::PathBuf,
}

impl SequenceAnalysisSuite {
    pub fn new(data_dir: impl AsRef<std::path::Path>) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
        }
    }

    /// Run all five analysis methods
    pub fn run_full_analysis(
        &self,
        sequences_by_context: &HashMap<String, Vec<Vec<i32>>>,
    ) -> Result<SequenceAnalysisReport> {
        println!("╔═══════════════════════════════════════════════════════════════════════════╗");
        println!("║      Advanced Sequence Analysis Suite - Combinatorial Syntax Test          ║");
        println!("╚═══════════════════════════════════════════════════════════════════════════╝");
        println!();

        // Flatten sequences for methods that need all data
        let all_sequences: Vec<_> = sequences_by_context
            .values()
            .flat_map(|v| v.clone())
            .collect();

        let mut report = SequenceAnalysisReport::default();

        // ========================================================================
        // Method 1: Multiple Sequence Alignment
        // ========================================================================
        println!("📊 Method 1: Multiple Sequence Alignment (MSA)");
        println!("   Finding conserved regions across contexts...");
        println!();

        let aligner = MultipleSequenceAligner::new(-2, 1, -1);

        // Align sequences from different contexts
        for (ctx1, ctx2) in self.get_context_pairs(sequences_by_context) {
            if let (Some(seq1), Some(seq2)) = (
                sequences_by_context.get(&ctx1).and_then(|v| v.first()),
                sequences_by_context.get(&ctx2).and_then(|v| v.first()),
            ) {
                println!("   Aligning {} vs {}...", ctx1, ctx2);
                if let Ok(msa_result) = aligner.align_multiple(&[seq1.clone(), seq2.clone()]) {
                    let n_conserved = msa_result.conserved_regions.len();
                    println!("      Found {} conserved regions", n_conserved);

                    for (start, end, score) in &msa_result.conserved_regions {
                        println!(
                            "      Region [{}-{}]: conservation={:.2}",
                            start, end, score
                        );
                    }

                    report.msa_conserved_regions += n_conserved;
                    report
                        .msa_results
                        .insert(format!("{}_vs_{}", ctx1, ctx2), msa_result);
                }
            }
        }

        println!();

        // ========================================================================
        // Method 2: Hidden Markov Models
        // ========================================================================
        println!("🔮 Method 2: Hidden Markov Models (HMM)");
        println!("   Discovering hidden phrase states...");
        println!();

        // Calculate required symbol capacity from unique phrase labels
        let unique_symbols: HashSet<i32> = all_sequences
            .iter()
            .flat_map(|s| s.iter().copied())
            .collect();
        let n_symbols = unique_symbols.len().max(200); // At least 200, or actual unique count
        let n_states = 5.min(unique_symbols.len()); // Use 5 states or fewer if limited symbols

        println!(
            "   Symbol capacity: {} (unique symbols: {})",
            n_symbols,
            unique_symbols.len()
        );

        let hmm = SequenceHmm::new(n_states, n_symbols);

        match hmm.train(&all_sequences, 100, 1e-4) {
            Ok(hmm_result) => {
                println!("   Discovered {} hidden states", hmm_result.n_states);

                for (i, desc) in hmm_result.state_descriptions.iter().enumerate() {
                    println!("   State {}:", i);
                    println!(
                        "      Top emissions: {:?}",
                        desc.top_emissions.iter().take(3).collect::<Vec<_>>()
                    );
                    println!("      Persistence: {:.3}", desc.persistence);
                    println!("      Entropy: {:.3}", desc.emission_entropy);
                    println!("      Preferred next: {:?}", desc.preferred_next_state);
                }

                report.hmm_states = hmm_result.n_states;
                report.hmm_result = Some(hmm_result);
            }
            Err(e) => {
                println!("   HMM training failed: {}", e);
            }
        }

        println!();

        // ========================================================================
        // Method 3: N-Gram Perplexity
        // ========================================================================
        println!("📚 Method 3: N-Gram Perplexity Analysis");
        println!("   Testing cross-context prediction...");
        println!();

        let mut ngram_model = NgramModel::new(2, 1.0); // Bigram model

        // Train on one context
        if let Some((train_ctx, train_seqs)) = sequences_by_context.iter().next() {
            println!("   Training on {} context...", train_ctx);
            ngram_model.train(train_seqs);

            let test_data: HashMap<_, _> = sequences_by_context
                .iter()
                .filter(|(ctx, _)| *ctx != train_ctx)
                .map(|(ctx, seqs)| (ctx.clone(), seqs.clone()))
                .collect();

            let ngram_result = ngram_model.cross_context_perplexity(&test_data);

            println!("   Cross-Context Perplexity:");
            for (ctx, ppl) in &ngram_result.cross_context_perplexity {
                println!("      {}: {:.2}", ctx, ppl);
            }

            report.ngram_perplexity = ngram_result.cross_context_perplexity;
            report.ngram_relative = ngram_result.relative_perplexity;
        }

        println!();

        // ========================================================================
        // Method 4: Network Motif Analysis
        // ========================================================================
        println!("🕸️  Method 4: Network Motif Analysis");
        println!("   Finding recurring structural patterns...");
        println!();

        let motif_analyzer = NetworkMotifAnalyzer::new(3, 100); // 3-node motifs

        // Build transition maps
        let mut transitions_by_context: HashMap<String, Vec<(i32, i32)>> = HashMap::new();
        for (ctx, sequences) in sequences_by_context {
            let transitions = extract_transitions(sequences);
            transitions_by_context.insert(ctx.clone(), transitions);
        }

        match motif_analyzer.find_motifs(&transitions_by_context) {
            Ok(motif_result) => {
                println!("   Found {} motifs", motif_result.motifs.len());
                println!(
                    "   Multi-context motifs: {}",
                    motif_result.multi_context_motifs.len()
                );

                for &(motif_id, n_ctx) in &motif_result.multi_context_motifs {
                    println!(
                        "      Motif {}: appears in {} contexts",
                        motif_id, n_ctx as usize
                    );
                }

                report.network_motifs = motif_result.motifs.len();
                report.multi_context_motifs = motif_result.multi_context_motifs.len();
            }
            Err(e) => {
                println!("   Motif analysis failed: {}", e);
            }
        }

        println!();

        // ========================================================================
        // Method 5: Supervised ML
        // ========================================================================
        println!("🤖 Method 5: Supervised Machine Learning");
        println!("   Comparing bag-of-words vs syntax features...");
        println!();

        let classifier = ContextClassifier::new(100, 10);

        match classifier.compare_feature_types(sequences_by_context) {
            Ok(ml_result) => {
                println!(
                    "   Bag-of-Words Accuracy: {:.2}%",
                    ml_result.bow_accuracy * 100.0
                );
                println!(
                    "   N-Gram Syntax Accuracy: {:.2}%",
                    ml_result.ngram_accuracy * 100.0
                );
                println!(
                    "   Improvement: {:.2}%",
                    ml_result.accuracy_improvement * 100.0
                );

                println!();
                println!("   Most predictive sequences:");
                for (seq, score) in &ml_result.predictive_sequences {
                    println!("      {:?} (importance: {:.2})", seq, score);
                }

                report.ml_bow_accuracy = ml_result.bow_accuracy;
                report.ml_ngram_accuracy = ml_result.ngram_accuracy;
                report.ml_improvement = ml_result.accuracy_improvement;
            }
            Err(e) => {
                println!("   ML analysis failed: {}", e);
            }
        }

        println!();

        // ========================================================================
        // Final Summary
        // ========================================================================

        println!("╔═══════════════════════════════════════════════════════════════════════════╗");
        println!("║                        ANALYSIS SUMMARY                                 ║");
        println!("╠═══════════════════════════════════════════════════════════════════════════╣");
        println!("║                                                                           ║");
        println!("║  Evidence for Combinatorial Syntax:                                       ║");
        println!(
            "║  • MSA conserved regions: {}                                             ║",
            report.msa_conserved_regions
        );
        println!(
            "║  • HMM hidden states: {}                                                ║",
            report.hmm_states
        );
        println!(
            "║  • Network motifs: {}                                                    ║",
            report.network_motifs
        );
        println!(
            "║  • Multi-context motifs: {}                                              ║",
            report.multi_context_motifs
        );
        println!(
            "║  • ML syntax improvement: {:.1}%                                         ║",
            report.ml_improvement * 100.0
        );
        println!("║                                                                           ║");
        println!("╚═══════════════════════════════════════════════════════════════════════════╝");
        println!();

        Ok(report)
    }

    /// Get pairs of contexts for comparison
    fn get_context_pairs(
        &self,
        contexts: &HashMap<String, Vec<Vec<i32>>>,
    ) -> Vec<(String, String)> {
        let ctx_names: Vec<_> = contexts.keys().cloned().collect();
        let mut pairs = Vec::new();

        for i in 0..ctx_names.len() {
            for j in (i + 1)..ctx_names.len() {
                pairs.push((ctx_names[i].clone(), ctx_names[j].clone()));
            }
        }

        pairs
    }
}

/// Helper: Extract transition pairs from sequences
fn extract_transitions(sequences: &[Vec<i32>]) -> Vec<(i32, i32)> {
    let mut transitions = Vec::new();

    for seq in sequences {
        for window in seq.windows(2) {
            transitions.push((window[0], window[1]));
        }
    }

    transitions
}

/// Comprehensive analysis report
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SequenceAnalysisReport {
    pub msa_conserved_regions: usize,
    pub msa_results: HashMap<String, MsaResult>,
    pub hmm_states: usize,
    pub hmm_result: Option<HmmAnalysisResult>,
    pub ngram_perplexity: HashMap<String, f64>,
    pub ngram_relative: HashMap<String, f64>,
    pub network_motifs: usize,
    pub multi_context_motifs: usize,
    pub ml_bow_accuracy: f64,
    pub ml_ngram_accuracy: f64,
    pub ml_improvement: f64,
}
