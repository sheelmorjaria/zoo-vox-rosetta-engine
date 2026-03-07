// =============================================================================
// Sequence Analysis Module - N-gram and Motif Analysis
// =============================================================================
//
// Analyzes phrase sequences for species with combinatorial syntax (zebra finch, orcas).
// Provides n-gram analysis, motif detection, and perplexity calculation.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Detected motif in a sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Motif {
    /// Pattern (sequence of phrase types)
    pub pattern: Vec<i32>,

    /// Number of occurrences
    pub occurrences: usize,

    /// Starting positions in the sequence
    pub positions: Vec<usize>,
}

/// N-gram statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramStats {
    /// Number of unique bigrams
    pub unique_bigrams: usize,

    /// Number of unique trigrams
    pub unique_trigrams: usize,

    /// Most common bigram
    pub most_common_bigram: (i32, i32),

    /// Most common trigram
    pub most_common_trigram: (i32, i32, i32),

    /// Bigram entropy
    pub bigram_entropy: f64,

    /// Trigram entropy
    pub trigram_entropy: f64,
}

impl Default for NgramStats {
    fn default() -> Self {
        Self {
            unique_bigrams: 0,
            unique_trigrams: 0,
            most_common_bigram: (0, 0),
            most_common_trigram: (0, 0, 0),
            bigram_entropy: 0.0,
            trigram_entropy: 0.0,
        }
    }
}

/// Result of sequence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceAnalysis {
    /// Original phrase sequence
    pub sequence: Vec<i32>,

    /// Detected motifs
    pub motifs: Vec<Motif>,

    /// N-gram statistics
    pub ngram_stats: NgramStats,

    /// Perplexity score
    pub perplexity: f64,

    /// Transition probability matrix
    pub transition_matrix: HashMap<i32, HashMap<i32, f64>>,

    /// Predicted context
    pub predicted_context: Option<String>,
}

/// Sequence analysis module for n-gram and motif analysis
pub struct SequenceModule {
    /// Maximum n-gram order to analyze
    max_ngram_order: usize,

    /// Minimum occurrence count for motif detection
    min_occurrence: usize,
}

impl SequenceModule {
    /// Create a new sequence module
    pub fn new(max_ngram_order: usize) -> Self {
        Self {
            max_ngram_order,
            min_occurrence: 2,
        }
    }

    /// Get maximum n-gram order
    pub fn max_ngram_order(&self) -> usize {
        self.max_ngram_order
    }

    /// Get minimum occurrence threshold
    pub fn min_occurrence(&self) -> usize {
        self.min_occurrence
    }

    /// Analyze a phrase sequence
    pub fn analyze(&self, sequence: &[i32]) -> SequenceAnalysis {
        if sequence.is_empty() {
            return SequenceAnalysis {
                sequence: Vec::new(),
                motifs: Vec::new(),
                ngram_stats: NgramStats::default(),
                perplexity: 1.0,
                transition_matrix: HashMap::new(),
                predicted_context: None,
            };
        }

        let motifs = self.find_motifs(sequence);
        let ngram_stats = self.compute_ngram_stats(sequence);
        let perplexity = self.compute_perplexity(sequence);
        let transition_matrix = self.compute_transition_matrix(sequence);

        SequenceAnalysis {
            sequence: sequence.to_vec(),
            motifs,
            ngram_stats,
            perplexity,
            transition_matrix,
            predicted_context: None,
        }
    }

    /// Find repeated motifs in the sequence
    pub fn find_motifs(&self, sequence: &[i32]) -> Vec<Motif> {
        if sequence.len() < 4 {
            return Vec::new();
        }

        let mut motif_map: HashMap<Vec<i32>, Vec<usize>> = HashMap::new();

        // Search for motifs of length 2 to max_ngram_order
        for motif_len in 2..=self.max_ngram_order.min(sequence.len() / 2) {
            for i in 0..=(sequence.len() - motif_len) {
                let pattern: Vec<i32> = sequence[i..i + motif_len].to_vec();
                motif_map.entry(pattern).or_default().push(i);
            }
        }

        // Convert to Motif structs and filter by minimum occurrence
        let mut motifs: Vec<Motif> = motif_map
            .into_iter()
            .filter(|(_, positions)| positions.len() >= self.min_occurrence)
            .map(|(pattern, positions)| Motif {
                pattern,
                occurrences: positions.len(),
                positions,
            })
            .collect();

        // Sort by occurrences (descending)
        motifs.sort_by(|a, b| b.occurrences.cmp(&a.occurrences));

        motifs
    }

    /// Compute n-gram statistics
    pub fn compute_ngram_stats(&self, sequence: &[i32]) -> NgramStats {
        if sequence.len() < 2 {
            return NgramStats::default();
        }

        // Count bigrams
        let mut bigram_counts: HashMap<(i32, i32), usize> = HashMap::new();
        for window in sequence.windows(2) {
            *bigram_counts.entry((window[0], window[1])).or_default() += 1;
        }

        // Count trigrams
        let mut trigram_counts: HashMap<(i32, i32, i32), usize> = HashMap::new();
        if sequence.len() >= 3 {
            for window in sequence.windows(3) {
                *trigram_counts.entry((window[0], window[1], window[2])).or_default() += 1;
            }
        }

        // Find most common bigram
        let most_common_bigram = bigram_counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|((a, b), _)| (*a, *b))
            .unwrap_or((0, 0));

        // Find most common trigram
        let most_common_trigram = trigram_counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|((a, b, c), _)| (*a, *b, *c))
            .unwrap_or((0, 0, 0));

        // Compute bigram entropy
        let total_bigrams: usize = bigram_counts.values().sum();
        let bigram_entropy = if total_bigrams > 0 {
            bigram_counts
                .values()
                .map(|&c| {
                    let p = c as f64 / total_bigrams as f64;
                    if p > 0.0 {
                        -p * p.log2()
                    } else {
                        0.0
                    }
                })
                .sum()
        } else {
            0.0
        };

        // Compute trigram entropy
        let total_trigrams: usize = trigram_counts.values().sum();
        let trigram_entropy = if total_trigrams > 0 {
            trigram_counts
                .values()
                .map(|&c| {
                    let p = c as f64 / total_trigrams as f64;
                    if p > 0.0 {
                        -p * p.log2()
                    } else {
                        0.0
                    }
                })
                .sum()
        } else {
            0.0
        };

        NgramStats {
            unique_bigrams: bigram_counts.len(),
            unique_trigrams: trigram_counts.len(),
            most_common_bigram,
            most_common_trigram,
            bigram_entropy,
            trigram_entropy,
        }
    }

    /// Compute perplexity of the sequence
    pub fn compute_perplexity(&self, sequence: &[i32]) -> f64 {
        if sequence.is_empty() {
            return 1.0;
        }

        // Count type frequencies
        let mut type_counts: HashMap<i32, usize> = HashMap::new();
        for &t in sequence {
            *type_counts.entry(t).or_default() += 1;
        }

        let n_types = type_counts.len();
        if n_types == 0 {
            return 1.0;
        }

        // Compute entropy
        let total = sequence.len() as f64;
        let entropy: f64 = type_counts
            .values()
            .map(|&c| {
                let p = c as f64 / total;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();

        // Perplexity = 2^entropy
        2.0_f64.powf(entropy)
    }

    /// Compute transition probability matrix
    pub fn compute_transition_matrix(&self, sequence: &[i32]) -> HashMap<i32, HashMap<i32, f64>> {
        if sequence.len() < 2 {
            return HashMap::new();
        }

        // Count transitions
        let mut transition_counts: HashMap<i32, HashMap<i32, usize>> = HashMap::new();
        let mut from_counts: HashMap<i32, usize> = HashMap::new();

        for window in sequence.windows(2) {
            let from = window[0];
            let to = window[1];

            *transition_counts.entry(from).or_default().entry(to).or_default() += 1;
            *from_counts.entry(from).or_default() += 1;
        }

        // Convert to probabilities
        let mut transition_matrix: HashMap<i32, HashMap<i32, f64>> = HashMap::new();

        for (from, to_counts) in transition_counts {
            let total = from_counts.get(&from).copied().unwrap_or(1) as f64;
            let probs: HashMap<i32, f64> = to_counts
                .into_iter()
                .map(|(to, count)| (to, count as f64 / total))
                .collect();
            transition_matrix.insert(from, probs);
        }

        transition_matrix
    }
}

impl Default for SequenceModule {
    fn default() -> Self {
        Self::new(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_module_creation() {
        let module = SequenceModule::new(3);

        assert_eq!(module.max_ngram_order(), 3);
        assert_eq!(module.min_occurrence(), 2);
    }

    #[test]
    fn test_find_motifs() {
        let module = SequenceModule::new(3);
        let sequence = vec![0, 1, 2, 3, 0, 1, 2, 4, 0, 1, 2];

        let motifs = module.find_motifs(&sequence);

        // Should find [0, 1, 2] pattern appearing 3 times
        assert!(motifs.iter().any(|m| m.pattern == vec![0, 1, 2] && m.occurrences >= 3));
    }

    #[test]
    fn test_compute_perplexity_low() {
        let module = SequenceModule::new(3);
        let sequence = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

        let perplexity = module.compute_perplexity(&sequence);

        // Low perplexity for repetitive sequence
        assert!(perplexity < 2.0);
    }

    #[test]
    fn test_compute_perplexity_high() {
        let module = SequenceModule::new(3);
        let sequence = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

        let perplexity = module.compute_perplexity(&sequence);

        // High perplexity for diverse sequence
        assert!(perplexity > 3.0);
    }

    #[test]
    fn test_compute_transition_matrix() {
        let module = SequenceModule::new(3);
        let sequence = vec![0, 1, 0, 1, 2, 0, 1];

        let transitions = module.compute_transition_matrix(&sequence);

        // Type 0 should always transition to type 1
        assert!(transitions.contains_key(&0));
        assert_eq!(transitions[&0].get(&1), Some(&1.0));
    }
}
