//! Voting Ensemble for Species Classification
//!
//! Architecture:
//!   INPUT: 105D Feature Vector
//!       │
//! ┌─────┴─────┐
//! ▼           ▼
//! [NN 105D]   [RF 105D]
//! │           │
//! ▼           ▼
//! Top-5       Probability
//! Candidates  Distribution
//! │           │
//! └─────┬─────┘
//!       ▼
//! [Ensemble Voter]
//!       │
//!       ▼
//! FINAL PREDICTION
//!
//! Logic:
//! 1. NN generates Top-5 shortlist (excellent at neighborhood finding)
//! 2. RF evaluates those specific candidates (better calibrated)
//! 3. Weighted vote: NN 40%, RF 60%

use anyhow::Result;
use std::collections::HashMap;

// ============================================================================
// Data Structures
// ============================================================================

/// A candidate prediction with confidence score
#[derive(Debug, Clone)]
pub struct Candidate {
    pub class_idx: usize,
    pub class_name: String,
    pub confidence: f32,
}

/// Ensemble prediction result
#[derive(Debug, Clone)]
pub struct EnsemblePrediction {
    pub predicted_class: String,
    pub predicted_idx: usize,
    pub confidence: f32,
    pub nn_rank: usize, // Where NN ranked this candidate
    pub nn_confidence: f32,
    pub rf_probability: f32,
    pub candidates: Vec<CandidateScore>,
}

/// Score for a single candidate in the ensemble
#[derive(Debug, Clone)]
pub struct CandidateScore {
    pub class_name: String,
    pub class_idx: usize,
    pub nn_confidence: f32,
    pub rf_probability: f32,
    pub combined_score: f32,
}

/// Configuration for the ensemble
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// Weight for NN predictions (0.0 to 1.0)
    pub nn_weight: f32,
    /// Weight for RF predictions (0.0 to 1.0)
    pub rf_weight: f32,
    /// Number of candidates to consider from NN
    pub top_k: usize,
    /// Minimum confidence threshold
    pub min_confidence: f32,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            nn_weight: 0.4,
            rf_weight: 0.6,
            top_k: 5,
            min_confidence: 0.0,
        }
    }
}

// ============================================================================
// Ensemble Voter
// ============================================================================

pub struct EnsembleVoter {
    config: EnsembleConfig,
}

impl EnsembleVoter {
    pub fn new(config: EnsembleConfig) -> Self {
        Self { config }
    }

    /// Combine NN Top-K candidates with RF probabilities
    ///
    /// # Arguments
    /// * `nn_candidates` - Top-K candidates from neural network
    /// * `rf_probabilities` - Probability distribution from random forest
    /// * `class_names` - Mapping from class index to class name
    ///
    /// # Returns
    /// * Ensemble prediction with combined scores
    pub fn vote(
        &self,
        nn_candidates: Vec<Candidate>,
        rf_probabilities: &[f32],
        class_names: &[String],
    ) -> EnsemblePrediction {
        let mut scored_candidates = Vec::new();
        let mut best_score = f32::NEG_INFINITY;
        let mut best_candidate: Option<CandidateScore> = None;

        for (rank, nn_cand) in nn_candidates.iter().enumerate() {
            // Get RF probability for this class
            let rf_prob = if nn_cand.class_idx < rf_probabilities.len() {
                rf_probabilities[nn_cand.class_idx]
            } else {
                0.0
            };

            // Combine scores with weights
            let combined =
                (nn_cand.confidence * self.config.nn_weight) + (rf_prob * self.config.rf_weight);

            let scored = CandidateScore {
                class_name: nn_cand.class_name.clone(),
                class_idx: nn_cand.class_idx,
                nn_confidence: nn_cand.confidence,
                rf_probability: rf_prob,
                combined_score: combined,
            };

            if combined > best_score {
                best_score = combined;
                best_candidate = Some(scored.clone());
            }

            scored_candidates.push(scored);
        }

        // Sort by combined score descending
        scored_candidates.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = best_candidate.unwrap_or_else(|| CandidateScore {
            class_name: "unknown".to_string(),
            class_idx: 0,
            nn_confidence: 0.0,
            rf_probability: 0.0,
            combined_score: 0.0,
        });

        // Find NN rank
        let nn_rank = nn_candidates
            .iter()
            .position(|c| c.class_idx == best.class_idx)
            .unwrap_or(99);

        EnsemblePrediction {
            predicted_class: best.class_name,
            predicted_idx: best.class_idx,
            confidence: best.combined_score,
            nn_rank,
            nn_confidence: best.nn_confidence,
            rf_probability: best.rf_probability,
            candidates: scored_candidates,
        }
    }

    /// Check if prediction matches ground truth
    pub fn is_correct(&self, prediction: &EnsemblePrediction, ground_truth: &str) -> bool {
        // Exact match
        if prediction.predicted_class.to_lowercase() == ground_truth.to_lowercase() {
            return true;
        }
        // Partial match (one contains the other)
        let pred_lower = prediction.predicted_class.to_lowercase();
        let truth_lower = ground_truth.to_lowercase();
        pred_lower.contains(&truth_lower) || truth_lower.contains(&pred_lower)
    }

    /// Check if ground truth is in Top-K candidates
    pub fn is_in_top_k(&self, prediction: &EnsemblePrediction, ground_truth: &str) -> bool {
        let truth_lower = ground_truth.to_lowercase();
        prediction
            .candidates
            .iter()
            .take(self.config.top_k)
            .any(|c| {
                let c_lower = c.class_name.to_lowercase();
                c_lower == truth_lower
                    || c_lower.contains(&truth_lower)
                    || truth_lower.contains(&c_lower)
            })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(idx: usize, name: &str, conf: f32) -> Candidate {
        Candidate {
            class_idx: idx,
            class_name: name.to_string(),
            confidence: conf,
        }
    }

    #[test]
    fn test_ensemble_voter_basic() {
        let voter = EnsembleVoter::new(EnsembleConfig::default());

        // NN says: "Blue Jay" (0.8), "Cardinal" (0.15), ...
        let nn_candidates = vec![
            make_candidate(0, "Blue Jay", 0.8),
            make_candidate(1, "Northern Cardinal", 0.15),
            make_candidate(2, "American Robin", 0.03),
            make_candidate(3, "House Sparrow", 0.01),
            make_candidate(4, "Dark-eyed Junco", 0.01),
        ];

        // RF says: "Cardinal" has highest probability
        let rf_probs = vec![0.3, 0.5, 0.1, 0.05, 0.05];
        let class_names = vec![
            "Blue Jay".to_string(),
            "Northern Cardinal".to_string(),
            "American Robin".to_string(),
            "House Sparrow".to_string(),
            "Dark-eyed Junco".to_string(),
        ];

        let result = voter.vote(nn_candidates, &rf_probs, &class_names);

        // Combined score for Blue Jay: 0.8 * 0.4 + 0.3 * 0.6 = 0.32 + 0.18 = 0.50
        // Combined score for Cardinal: 0.15 * 0.4 + 0.5 * 0.6 = 0.06 + 0.30 = 0.36
        // Blue Jay should win
        assert_eq!(result.predicted_class, "Blue Jay");
        assert_eq!(result.nn_rank, 0);
    }

    #[test]
    fn test_ensemble_rf_overrides_nn() {
        let voter = EnsembleVoter::new(EnsembleConfig {
            nn_weight: 0.3,
            rf_weight: 0.7,
            top_k: 5,
            min_confidence: 0.0,
        });

        // NN says: "Robin" (0.4), "Blue Jay" (0.35)
        let nn_candidates = vec![
            make_candidate(0, "American Robin", 0.4),
            make_candidate(1, "Blue Jay", 0.35),
            make_candidate(2, "Cardinal", 0.15),
            make_candidate(3, "Sparrow", 0.05),
            make_candidate(4, "Finch", 0.05),
        ];

        // RF strongly says "Blue Jay" (0.7)
        let rf_probs = vec![0.2, 0.7, 0.05, 0.03, 0.02];
        let class_names = vec![
            "American Robin".to_string(),
            "Blue Jay".to_string(),
            "Cardinal".to_string(),
            "Sparrow".to_string(),
            "Finch".to_string(),
        ];

        let result = voter.vote(nn_candidates, &rf_probs, &class_names);

        // Combined for Robin: 0.4 * 0.3 + 0.2 * 0.7 = 0.12 + 0.14 = 0.26
        // Combined for Blue Jay: 0.35 * 0.3 + 0.7 * 0.7 = 0.105 + 0.49 = 0.595
        // Blue Jay should win because RF is weighted higher
        assert_eq!(result.predicted_class, "Blue Jay");
    }

    #[test]
    fn test_ensemble_top_k_check() {
        let voter = EnsembleVoter::new(EnsembleConfig::default());

        let nn_candidates = vec![
            make_candidate(0, "Species A", 0.5),
            make_candidate(1, "Species B", 0.3),
            make_candidate(2, "Species C", 0.1),
            make_candidate(3, "Species D", 0.05),
            make_candidate(4, "Species E", 0.05),
        ];

        let rf_probs = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        let class_names = vec![
            "Species A".to_string(),
            "Species B".to_string(),
            "Species C".to_string(),
            "Species D".to_string(),
            "Species E".to_string(),
        ];

        let result = voter.vote(nn_candidates, &rf_probs, &class_names);

        // Check Top-5
        assert!(voter.is_in_top_k(&result, "Species A"));
        assert!(voter.is_in_top_k(&result, "Species C"));
        assert!(!voter.is_in_top_k(&result, "Species Z"));
    }

    #[test]
    fn test_ensemble_correctness_check() {
        let voter = EnsembleVoter::new(EnsembleConfig::default());

        let nn_candidates = vec![make_candidate(0, "Blue Jay", 0.8)];

        let rf_probs = vec![0.9];
        let class_names = vec!["Blue Jay".to_string()];

        let result = voter.vote(nn_candidates, &rf_probs, &class_names);

        // Exact match
        assert!(voter.is_correct(&result, "Blue Jay"));
        assert!(voter.is_correct(&result, "blue jay")); // case insensitive

        // Partial match
        assert!(voter.is_correct(&result, "Blue"));
    }

    #[test]
    fn test_ensemble_candidates_sorted_by_score() {
        let voter = EnsembleVoter::new(EnsembleConfig::default());

        let nn_candidates = vec![
            make_candidate(0, "A", 0.1),
            make_candidate(1, "B", 0.5),
            make_candidate(2, "C", 0.2),
        ];

        let rf_probs = vec![0.3, 0.3, 0.3];
        let class_names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = voter.vote(nn_candidates, &rf_probs, &class_names);

        // Should be sorted by combined score (B has highest NN confidence)
        assert!(result.candidates[0].combined_score >= result.candidates[1].combined_score);
        assert!(result.candidates[1].combined_score >= result.candidates[2].combined_score);
    }
}
