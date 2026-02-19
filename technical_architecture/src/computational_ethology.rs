//! Computational Ethology Module
//! ==============================
//!
//! Statistical validation of discovered linguistic structure in animal vocalizations.
//! This module implements the "Corpus Linguistics" approach to validate that
//! discovered phrases follow the statistical hallmarks of language:
//!
//! - **Syntax**: Non-random ordering (measured via perplexity, sequence similarity)
//! - **Semantics**: Non-random context association (measured via PMI)
//! - **Structure**: Language-like distributions (measured via Zipf's Law)
//!
//! # Key Metrics
//!
//! 1. **Reuse Ratio**: Total occurrences / Unique types (higher = better)
//! 2. **Singleton Rate**: % of types with only 1 occurrence (lower = better)
//! 3. **Zipf Compliance**: Correlation with ideal Zipf distribution
//! 4. **Perplexity**: How predictable the sequence is (lower = more structure)
//! 5. **PMI**: Phrase-Context associations (higher = stronger semantics)
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use anyhow::Result;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// =============================================================================
// Core Data Structures
// =============================================================================

/// A discovered phrase type with its statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseType {
    /// Unique identifier for this phrase type
    pub id: String,
    /// Human-readable label (if known)
    pub label: Option<String>,
    /// Number of times this phrase occurs in the corpus
    pub occurrence_count: usize,
    /// 45D centroid features
    pub centroid: Vec<f64>,
    /// Contexts this phrase appears in (metadata tags)
    pub contexts: HashMap<String, usize>,
}

/// A sequence of phrase IDs representing a vocalization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseSequence {
    /// Source file or recording ID
    pub source_id: String,
    /// Ordered list of phrase IDs
    pub phrases: Vec<String>,
    /// Metadata tags (e.g., "alarm", "contact", "food")
    pub metadata_tags: Vec<String>,
}

/// Results of linguistic structure validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Zipf's Law compliance (correlation coefficient, -1 to 1)
    pub zipf_correlation: f64,
    /// Whether Zipf correlation indicates language-like structure (> 0.8)
    pub is_zipfian: bool,
    /// Reuse ratio (occurrences / types)
    pub reuse_ratio: f64,
    /// Percentage of singleton phrase types
    pub singleton_rate: f64,
    /// Perplexity of real sequences
    pub real_perplexity: f64,
    /// Perplexity of shuffled baseline
    pub random_perplexity: f64,
    /// Whether perplexity indicates syntax (real < 0.8 * random)
    pub has_syntax: bool,
    /// PMI scores for phrase-context pairs
    pub pmi_scores: HashMap<String, HashMap<String, f64>>,
    /// Overall validation score (0.0 to 1.0)
    pub validation_score: f64,
}

/// Configuration for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Minimum occurrences for a phrase to be considered valid
    pub min_occurrences: usize,
    /// Zipf correlation threshold for "language-like"
    pub zipf_threshold: f64,
    /// Perplexity reduction ratio threshold
    pub perplexity_ratio_threshold: f64,
    /// Maximum singleton rate considered acceptable
    pub max_singleton_rate: f64,
    /// N-gram size for perplexity calculation
    pub ngram_size: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            min_occurrences: 2,
            zipf_threshold: 0.8,
            perplexity_ratio_threshold: 0.8,
            max_singleton_rate: 0.3,
            ngram_size: 2, // Bigrams
        }
    }
}

// =============================================================================
// Zipf's Law Validation
// =============================================================================

/// Calculate Zipf's Law compliance
///
/// Zipf's Law states that frequency is inversely proportional to rank:
/// f(r) ∝ 1/r, or log(f) = log(C) - s * log(r)
///
/// Returns the absolute value of the correlation coefficient.
/// A correlation (absolute value) > 0.8 indicates language-like structure.
/// Note: The raw correlation is negative (higher rank = lower frequency),
/// so we return the absolute value for easier interpretation.
pub fn calculate_zipf_correlation(phrase_types: &[PhraseType]) -> f64 {
    if phrase_types.is_empty() {
        return 0.0;
    }

    // Extract frequencies and sort by frequency (descending)
    let mut frequencies: Vec<usize> = phrase_types
        .iter()
        .map(|p| p.occurrence_count)
        .filter(|&f| f > 0)
        .collect();

    frequencies.sort_by(|a, b| b.cmp(a));

    if frequencies.len() < 3 {
        return 0.0;
    }

    // Calculate log(rank) and log(frequency)
    let n = frequencies.len();
    let log_ranks: Vec<f64> = (1..=n).map(|r| (r as f64).ln()).collect();
    let log_freqs: Vec<f64> = frequencies.iter().map(|f| (*f as f64).ln()).collect();

    // Calculate Pearson correlation
    // For Zipf's law, this should be strongly negative (~-1.0)
    // We return the absolute value for easier interpretation
    let correlation = pearson_correlation(&log_ranks, &log_freqs);
    correlation.abs()
}

/// Calculate Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|a| a * a).sum();
    let sum_y2: f64 = y.iter().map(|b| b * b).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator == 0.0 {
        return 0.0;
    }

    numerator / denominator
}

// =============================================================================
// Reuse Metrics
// =============================================================================

/// Calculate reuse ratio
///
/// Reuse Ratio = Total Occurrences / Number of Unique Types
///
/// Higher values indicate the system is finding "common words" rather than
/// fragmenting into thousands of unique tokens.
pub fn calculate_reuse_ratio(phrase_types: &[PhraseType]) -> f64 {
    if phrase_types.is_empty() {
        return 0.0;
    }

    let total_occurrences: usize = phrase_types.iter().map(|p| p.occurrence_count).sum();
    let num_types = phrase_types.len();

    total_occurrences as f64 / num_types as f64
}

/// Calculate singleton rate
///
/// Singleton Rate = Types with count = 1 / Total Types
///
/// Lower values indicate better clustering. Good systems have < 30% singletons.
pub fn calculate_singleton_rate(phrase_types: &[PhraseType]) -> f64 {
    if phrase_types.is_empty() {
        return 0.0;
    }

    let singleton_count = phrase_types
        .iter()
        .filter(|p| p.occurrence_count == 1)
        .count();

    singleton_count as f64 / phrase_types.len() as f64
}

// =============================================================================
// Perplexity (Syntax Detection)
// =============================================================================

/// Calculate perplexity of phrase sequences
///
/// Perplexity measures how "surprised" a language model is by a sequence.
/// Lower perplexity = more predictable = more structure (syntax).
///
/// PP(W) = exp(-1/N * sum(log P(w_i|context)))
pub fn calculate_perplexity(sequences: &[PhraseSequence], ngram_size: usize) -> f64 {
    if sequences.is_empty() {
        return f64::INFINITY;
    }

    // Build n-gram counts
    let mut ngram_counts: HashMap<Vec<String>, usize> = HashMap::new();
    let mut context_counts: HashMap<Vec<String>, usize> = HashMap::new();
    let mut total_ngrams = 0usize;

    for seq in sequences {
        if seq.phrases.len() < ngram_size {
            continue;
        }

        for i in 0..=seq.phrases.len() - ngram_size {
            let ngram: Vec<String> = seq.phrases[i..i + ngram_size].to_vec();
            let context: Vec<String> = seq.phrases[i..i + ngram_size - 1].to_vec();

            *ngram_counts.entry(ngram.clone()).or_insert(0) += 1;
            *context_counts.entry(context).or_insert(0) += 1;
            total_ngrams += 1;
        }
    }

    if total_ngrams == 0 {
        return f64::INFINITY;
    }

    // Calculate log probability
    let vocab_size = sequences
        .iter()
        .flat_map(|s| s.phrases.iter())
        .collect::<HashSet<_>>()
        .len();

    let smoothing = 1.0 / (vocab_size as f64 + 1.0);
    let mut log_prob_sum = 0.0;

    for (ngram, count) in &ngram_counts {
        let context: Vec<String> = ngram[..ngram_size - 1].to_vec();
        let context_count = context_counts.get(&context).copied().unwrap_or(0) as f64;

        let prob = if context_count > 0.0 {
            (*count as f64 / context_count) + smoothing
        } else {
            smoothing
        };

        log_prob_sum += (*count as f64) * prob.ln();
    }

    // Perplexity = exp(-1/N * sum(log P))
    let avg_log_prob = log_prob_sum / total_ngrams as f64;
    (-avg_log_prob).exp()
}

/// Shuffle sequences for baseline comparison
pub fn shuffle_sequences(sequences: &[PhraseSequence]) -> Vec<PhraseSequence> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    let mut shuffled = sequences.to_vec();
    let mut rng = thread_rng();

    for seq in &mut shuffled {
        seq.phrases.shuffle(&mut rng);
    }

    shuffled
}

// =============================================================================
// PMI (Semantics Detection)
// =============================================================================

/// Calculate Pointwise Mutual Information for phrase-context pairs
///
/// PMI(phrase, context) = log(P(phrase, context) / (P(phrase) * P(context)))
///
/// High PMI indicates the phrase is strongly associated with that context.
pub fn calculate_pmi(
    phrase_types: &[PhraseType],
    sequences: &[PhraseSequence],
) -> HashMap<String, HashMap<String, f64>> {
    let mut pmi_scores: HashMap<String, HashMap<String, f64>> = HashMap::new();

    // Count total phrase occurrences
    let total_occurrences: usize = phrase_types.iter().map(|p| p.occurrence_count).sum();
    if total_occurrences == 0 {
        return pmi_scores;
    }

    // Build phrase-to-sequences mapping
    let mut phrase_seq_counts: HashMap<String, usize> = HashMap::new();
    let mut phrase_context_counts: HashMap<(String, String), usize> = HashMap::new();

    for seq in sequences {
        // Count unique phrases in this sequence
        let unique_phrases: HashSet<_> = seq.phrases.iter().collect();
        for phrase_id in unique_phrases {
            *phrase_seq_counts.entry(phrase_id.clone()).or_insert(0) += 1;

            // Associate with each context tag
            for tag in &seq.metadata_tags {
                *phrase_context_counts
                    .entry((phrase_id.clone(), tag.clone()))
                    .or_insert(0) += 1;
            }
        }
    }

    // Count total sequences
    let total_sequences = sequences.len();
    if total_sequences == 0 {
        return pmi_scores;
    }

    // Count context occurrences (per sequence)
    let mut context_seq_counts: HashMap<String, usize> = HashMap::new();
    for seq in sequences {
        for tag in &seq.metadata_tags {
            *context_seq_counts.entry(tag.clone()).or_insert(0) += 1;
        }
    }

    // Calculate PMI for each phrase-context pair
    for ((phrase_id, context), joint_count) in &phrase_context_counts {
        let phrase_prob =
            *phrase_seq_counts.get(phrase_id).unwrap_or(&0) as f64 / total_sequences as f64;
        let context_prob =
            *context_seq_counts.get(context).unwrap_or(&0) as f64 / total_sequences as f64;
        let joint_prob = *joint_count as f64 / total_sequences as f64;

        if phrase_prob > 0.0 && context_prob > 0.0 && joint_prob > 0.0 {
            let pmi = (joint_prob / (phrase_prob * context_prob)).ln();

            if pmi.is_finite() {
                pmi_scores
                    .entry(phrase_id.clone())
                    .or_insert_with(HashMap::new)
                    .insert(context.clone(), pmi);
            }
        }
    }

    pmi_scores
}

// =============================================================================
// Sequence Similarity (Levenshtein)
// =============================================================================

/// Calculate Levenshtein distance between two phrase sequences
pub fn levenshtein_distance(seq1: &[String], seq2: &[String]) -> usize {
    let len1 = seq1.len();
    let len2 = seq2.len();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if seq1[i - 1] == seq2[j - 1] { 0 } else { 1 };
            matrix[i][j] = (matrix[i - 1][j] + 1) // deletion
                .min(matrix[i][j - 1] + 1) // insertion
                .min(matrix[i - 1][j - 1] + cost); // substitution
        }
    }

    matrix[len1][len2]
}

/// Calculate normalized Levenshtein similarity (0.0 to 1.0)
pub fn sequence_similarity(seq1: &[String], seq2: &[String]) -> f64 {
    let max_len = seq1.len().max(seq2.len());
    if max_len == 0 {
        return 1.0;
    }

    let distance = levenshtein_distance(seq1, seq2);
    1.0 - (distance as f64 / max_len as f64)
}

/// Calculate Jaccard index for phrase sets
pub fn jaccard_index(set1: &HashSet<String>, set2: &HashSet<String>) -> f64 {
    if set1.is_empty() && set2.is_empty() {
        return 1.0;
    }

    let intersection: usize = set1.intersection(set2).count();
    let union: usize = set1.union(set2).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f64 / union as f64
}

// =============================================================================
// Main Validation
// =============================================================================

/// Validate linguistic structure in discovered phrases
pub fn validate_linguistic_structure(
    phrase_types: &[PhraseType],
    sequences: &[PhraseSequence],
    config: &ValidationConfig,
) -> Result<ValidationResult> {
    info!("Validating linguistic structure...");

    // 1. Zipf's Law compliance
    let zipf_correlation = calculate_zipf_correlation(phrase_types);
    let is_zipfian = zipf_correlation > config.zipf_threshold;
    info!(
        "Zipf correlation: {:.3} ({})",
        zipf_correlation,
        if is_zipfian { "PASS" } else { "FAIL" }
    );

    // 2. Reuse metrics
    let reuse_ratio = calculate_reuse_ratio(phrase_types);
    let singleton_rate = calculate_singleton_rate(phrase_types);
    info!(
        "Reuse ratio: {:.2}, Singleton rate: {:.2}%",
        reuse_ratio,
        singleton_rate * 100.0
    );

    // 3. Perplexity (syntax)
    let real_perplexity = calculate_perplexity(sequences, config.ngram_size);
    let shuffled = shuffle_sequences(sequences);
    let random_perplexity = calculate_perplexity(&shuffled, config.ngram_size);

    let perplexity_ratio = if random_perplexity.is_finite() && random_perplexity > 0.0 {
        real_perplexity / random_perplexity
    } else {
        1.0
    };
    let has_syntax = perplexity_ratio < config.perplexity_ratio_threshold;
    info!(
        "Perplexity: real={:.2}, random={:.2}, ratio={:.2} ({})",
        real_perplexity,
        random_perplexity,
        perplexity_ratio,
        if has_syntax {
            "SYNTAX DETECTED"
        } else {
            "NO SYNTAX"
        }
    );

    // 4. PMI (semantics)
    let pmi_scores = calculate_pmi(phrase_types, sequences);

    // 5. Calculate overall validation score
    let mut validation_score = 0.0;

    // Zipf component (25%)
    if is_zipfian {
        validation_score += 0.25 * zipf_correlation.min(1.0);
    }

    // Singleton component (25%)
    if singleton_rate < config.max_singleton_rate {
        validation_score += 0.25 * (1.0 - singleton_rate / config.max_singleton_rate);
    }

    // Perplexity component (25%)
    if has_syntax {
        validation_score += 0.25 * (1.0 - perplexity_ratio);
    }

    // Reuse component (25%)
    if reuse_ratio >= 2.0 {
        validation_score += 0.25 * (reuse_ratio / 10.0).min(1.0);
    }

    info!("Overall validation score: {:.2}", validation_score);

    Ok(ValidationResult {
        zipf_correlation,
        is_zipfian,
        reuse_ratio,
        singleton_rate,
        real_perplexity,
        random_perplexity,
        has_syntax,
        pmi_scores,
        validation_score,
    })
}

/// Compare two configurations (e.g., Unified vs Species-Specific weights)
pub fn compare_configurations(
    phrase_types_a: &[PhraseType],
    sequences_a: &[PhraseSequence],
    phrase_types_b: &[PhraseType],
    sequences_b: &[PhraseSequence],
    config: &ValidationConfig,
) -> Result<ComparisonResult> {
    let result_a = validate_linguistic_structure(phrase_types_a, sequences_a, config)?;
    let result_b = validate_linguistic_structure(phrase_types_b, sequences_b, config)?;

    Ok(ComparisonResult {
        config_a_score: result_a.validation_score,
        config_b_score: result_b.validation_score,
        zipf_improvement: result_b.zipf_correlation - result_a.zipf_correlation,
        reuse_improvement: result_b.reuse_ratio - result_a.reuse_ratio,
        singleton_improvement: result_a.singleton_rate - result_b.singleton_rate, // Lower is better
        perplexity_improvement: result_a.real_perplexity - result_b.real_perplexity, // Lower is better
        winner: if result_b.validation_score > result_a.validation_score {
            "B"
        } else {
            "A"
        }
        .to_string(),
    })
}

/// Result of comparing two configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub config_a_score: f64,
    pub config_b_score: f64,
    pub zipf_improvement: f64,
    pub reuse_improvement: f64,
    pub singleton_improvement: f64,
    pub perplexity_improvement: f64,
    pub winner: String,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Zipf's Law Tests
    // =========================================================================

    #[test]
    fn test_zipf_correlation_perfect_zipfian() {
        // Create a perfect Zipf distribution: f(r) = 100/r
        let phrase_types: Vec<PhraseType> = (1..=100)
            .map(|r| PhraseType {
                id: format!("type_{}", r),
                label: None,
                occurrence_count: 100 / r,
                centroid: vec![],
                contexts: HashMap::new(),
            })
            .collect();

        let correlation = calculate_zipf_correlation(&phrase_types);
        assert!(
            correlation > 0.95,
            "Perfect Zipf should have correlation > 0.95, got {}",
            correlation
        );
    }

    #[test]
    fn test_zipf_correlation_random_distribution() {
        // Create a random distribution (not Zipfian)
        let phrase_types: Vec<PhraseType> = (1..=100)
            .map(|i| PhraseType {
                id: format!("type_{}", i),
                label: None,
                occurrence_count: 50 + (i * 7) % 50, // Not Zipfian
                centroid: vec![],
                contexts: HashMap::new(),
            })
            .collect();

        let correlation = calculate_zipf_correlation(&phrase_types);
        // Random distribution should have lower correlation
        assert!(
            correlation < 0.9,
            "Random distribution should have correlation < 0.9, got {}",
            correlation
        );
    }

    #[test]
    fn test_zipf_correlation_empty_input() {
        let phrase_types: Vec<PhraseType> = vec![];
        let correlation = calculate_zipf_correlation(&phrase_types);
        assert_eq!(correlation, 0.0);
    }

    #[test]
    fn test_zipf_correlation_single_phrase() {
        let phrase_types = vec![PhraseType {
            id: "only_one".to_string(),
            label: None,
            occurrence_count: 100,
            centroid: vec![],
            contexts: HashMap::new(),
        }];
        let correlation = calculate_zipf_correlation(&phrase_types);
        assert_eq!(correlation, 0.0); // Need at least 3 points
    }

    // =========================================================================
    // Reuse Metrics Tests
    // =========================================================================

    #[test]
    fn test_reuse_ratio_high_reuse() {
        // 10 types, 100 occurrences each = 1000 total / 10 types = ratio of 100
        let phrase_types: Vec<PhraseType> = (1..=10)
            .map(|i| PhraseType {
                id: format!("type_{}", i),
                label: None,
                occurrence_count: 100,
                centroid: vec![],
                contexts: HashMap::new(),
            })
            .collect();

        let ratio = calculate_reuse_ratio(&phrase_types);
        assert_eq!(ratio, 100.0); // 10 types × 100 = 1000 total, 1000/10 = 100
    }

    #[test]
    fn test_reuse_ratio_low_reuse() {
        // Each type appears only once
        let phrase_types: Vec<PhraseType> = (1..=100)
            .map(|i| PhraseType {
                id: format!("type_{}", i),
                label: None,
                occurrence_count: 1,
                centroid: vec![],
                contexts: HashMap::new(),
            })
            .collect();

        let ratio = calculate_reuse_ratio(&phrase_types);
        assert_eq!(ratio, 1.0);
    }

    #[test]
    fn test_singleton_rate_zero() {
        // No singletons
        let phrase_types: Vec<PhraseType> = (1..=10)
            .map(|i| PhraseType {
                id: format!("type_{}", i),
                label: None,
                occurrence_count: 10,
                centroid: vec![],
                contexts: HashMap::new(),
            })
            .collect();

        let rate = calculate_singleton_rate(&phrase_types);
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_singleton_rate_all_singletons() {
        let phrase_types: Vec<PhraseType> = (1..=100)
            .map(|i| PhraseType {
                id: format!("type_{}", i),
                label: None,
                occurrence_count: 1,
                centroid: vec![],
                contexts: HashMap::new(),
            })
            .collect();

        let rate = calculate_singleton_rate(&phrase_types);
        assert_eq!(rate, 1.0);
    }

    #[test]
    fn test_singleton_rate_mixed() {
        let phrase_types = vec![
            PhraseType {
                id: "a".to_string(),
                label: None,
                occurrence_count: 10,
                centroid: vec![],
                contexts: HashMap::new(),
            },
            PhraseType {
                id: "b".to_string(),
                label: None,
                occurrence_count: 1,
                centroid: vec![],
                contexts: HashMap::new(),
            },
            PhraseType {
                id: "c".to_string(),
                label: None,
                occurrence_count: 5,
                centroid: vec![],
                contexts: HashMap::new(),
            },
            PhraseType {
                id: "d".to_string(),
                label: None,
                occurrence_count: 1,
                centroid: vec![],
                contexts: HashMap::new(),
            },
        ];

        let rate = calculate_singleton_rate(&phrase_types);
        assert!((rate - 0.5).abs() < 0.01); // 2 singletons out of 4
    }

    // =========================================================================
    // Perplexity Tests
    // =========================================================================

    #[test]
    fn test_perplexity_perfectly_predictable() {
        // Same sequence repeated = low perplexity
        let sequences = vec![
            PhraseSequence {
                source_id: "s1".to_string(),
                phrases: vec!["A".to_string(), "B".to_string(), "C".to_string()],
                metadata_tags: vec![],
            },
            PhraseSequence {
                source_id: "s2".to_string(),
                phrases: vec!["A".to_string(), "B".to_string(), "C".to_string()],
                metadata_tags: vec![],
            },
            PhraseSequence {
                source_id: "s3".to_string(),
                phrases: vec!["A".to_string(), "B".to_string(), "C".to_string()],
                metadata_tags: vec![],
            },
        ];

        let perplexity = calculate_perplexity(&sequences, 2);
        assert!(
            perplexity < 2.0,
            "Predictable sequences should have low perplexity, got {}",
            perplexity
        );
    }

    #[test]
    fn test_perplexity_random_sequences() {
        // Random sequences = higher perplexity than predictable
        let sequences = vec![
            PhraseSequence {
                source_id: "s1".to_string(),
                phrases: vec!["A".to_string(), "X".to_string(), "Q".to_string()],
                metadata_tags: vec![],
            },
            PhraseSequence {
                source_id: "s2".to_string(),
                phrases: vec!["Z".to_string(), "B".to_string(), "M".to_string()],
                metadata_tags: vec![],
            },
            PhraseSequence {
                source_id: "s3".to_string(),
                phrases: vec!["P".to_string(), "K".to_string(), "C".to_string()],
                metadata_tags: vec![],
            },
        ];

        let perplexity = calculate_perplexity(&sequences, 2);
        // Random sequences have finite perplexity
        assert!(perplexity.is_finite(), "Perplexity should be finite");
    }

    #[test]
    fn test_perplexity_empty_sequences() {
        let sequences: Vec<PhraseSequence> = vec![];
        let perplexity = calculate_perplexity(&sequences, 2);
        assert!(perplexity.is_infinite());
    }

    #[test]
    fn test_shuffled_sequences_different() {
        let original = vec![PhraseSequence {
            source_id: "s1".to_string(),
            phrases: vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
            ],
            metadata_tags: vec![],
        }];

        let shuffled = shuffle_sequences(&original);

        // Shuffled should have same elements but different order (usually)
        assert_eq!(shuffled[0].phrases.len(), original[0].phrases.len());
        let same_set: HashSet<_> = shuffled[0].phrases.iter().collect();
        let orig_set: HashSet<_> = original[0].phrases.iter().collect();
        assert_eq!(same_set, orig_set);
    }

    // =========================================================================
    // PMI Tests
    // =========================================================================

    #[test]
    fn test_pmi_strong_association() {
        // Create sequences where "alarm_phrase" always appears with "alarm" context
        let sequences: Vec<PhraseSequence> = (0..90)
            .map(|i| PhraseSequence {
                source_id: format!("s{}", i),
                phrases: vec!["alarm_phrase".to_string()],
                metadata_tags: vec!["alarm".to_string()],
            })
            .chain((0..10).map(|i| PhraseSequence {
                source_id: format!("other_{}", i),
                phrases: vec!["other_phrase".to_string()],
                metadata_tags: vec!["other".to_string()],
            }))
            .collect();

        let phrase_types = vec![
            PhraseType {
                id: "alarm_phrase".to_string(),
                label: None,
                occurrence_count: 90,
                centroid: vec![],
                contexts: HashMap::new(),
            },
            PhraseType {
                id: "other_phrase".to_string(),
                label: None,
                occurrence_count: 10,
                centroid: vec![],
                contexts: HashMap::new(),
            },
        ];

        let pmi = calculate_pmi(&phrase_types, &sequences);

        // alarm_phrase should have positive PMI with alarm context
        assert!(
            pmi.contains_key("alarm_phrase"),
            "Should have PMI for alarm_phrase"
        );
        assert!(
            pmi["alarm_phrase"].contains_key("alarm"),
            "Should have PMI for alarm context"
        );
        assert!(
            pmi["alarm_phrase"]["alarm"] > 0.0,
            "Strong association should have positive PMI, got {}",
            pmi["alarm_phrase"]["alarm"]
        );
    }

    #[test]
    fn test_pmi_no_association() {
        let phrase_types = vec![PhraseType {
            id: "phrase_a".to_string(),
            label: None,
            occurrence_count: 50,
            centroid: vec![],
            contexts: [("context_x".to_string(), 25), ("context_y".to_string(), 25)]
                .into_iter()
                .collect(),
        }];

        let sequences: Vec<PhraseSequence> = (0..50)
            .map(|i| PhraseSequence {
                source_id: format!("s{}", i),
                phrases: vec!["phrase_a".to_string()],
                metadata_tags: vec![if i % 2 == 0 { "context_x" } else { "context_y" }.to_string()],
            })
            .collect();

        let pmi = calculate_pmi(&phrase_types, &sequences);
        // PMI should be near 0 for independent events
        if let Some(contexts) = pmi.get("phrase_a") {
            if let Some(&pmi_val) = contexts.get("context_x") {
                assert!(
                    pmi_val.abs() < 0.5,
                    "No association should have PMI near 0, got {}",
                    pmi_val
                );
            }
        }
    }

    // =========================================================================
    // Sequence Similarity Tests
    // =========================================================================

    #[test]
    fn test_levenshtein_identical() {
        let seq1 = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let seq2 = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        assert_eq!(levenshtein_distance(&seq1, &seq2), 0);
    }

    #[test]
    fn test_levenshtein_one_substitution() {
        let seq1 = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let seq2 = vec!["A".to_string(), "X".to_string(), "C".to_string()];
        assert_eq!(levenshtein_distance(&seq1, &seq2), 1);
    }

    #[test]
    fn test_levenshtein_one_insertion() {
        let seq1 = vec!["A".to_string(), "B".to_string()];
        let seq2 = vec!["A".to_string(), "X".to_string(), "B".to_string()];
        assert_eq!(levenshtein_distance(&seq1, &seq2), 1);
    }

    #[test]
    fn test_levenshtein_one_deletion() {
        let seq1 = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let seq2 = vec!["A".to_string(), "C".to_string()];
        assert_eq!(levenshtein_distance(&seq1, &seq2), 1);
    }

    #[test]
    fn test_sequence_similarity_identical() {
        let seq = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        assert!((sequence_similarity(&seq, &seq) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sequence_similarity_half_different() {
        let seq1 = vec!["A".to_string(), "B".to_string()];
        let seq2 = vec!["C".to_string(), "D".to_string()];
        assert!((sequence_similarity(&seq1, &seq2) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_jaccard_index_identical() {
        let set1: HashSet<String> = ["A".to_string(), "B".to_string()].into_iter().collect();
        let set2: HashSet<String> = ["A".to_string(), "B".to_string()].into_iter().collect();
        assert!((jaccard_index(&set1, &set2) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_jaccard_index_no_overlap() {
        let set1: HashSet<String> = ["A".to_string(), "B".to_string()].into_iter().collect();
        let set2: HashSet<String> = ["C".to_string(), "D".to_string()].into_iter().collect();
        assert!((jaccard_index(&set1, &set2) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_jaccard_index_half_overlap() {
        let set1: HashSet<String> = ["A".to_string(), "B".to_string()].into_iter().collect();
        let set2: HashSet<String> = ["B".to_string(), "C".to_string()].into_iter().collect();
        // Intersection = {B}, Union = {A, B, C}, Jaccard = 1/3
        assert!((jaccard_index(&set1, &set2) - 0.333).abs() < 0.01);
    }

    // =========================================================================
    // Full Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_linguistic_structure_good_corpus() {
        // Create a corpus with good linguistic properties
        let phrase_types: Vec<PhraseType> = (1..=20)
            .map(|r| PhraseType {
                id: format!("type_{}", r),
                label: None,
                occurrence_count: 200 / r, // Zipfian
                centroid: vec![],
                contexts: HashMap::new(),
            })
            .collect();

        // Create sequences with repeating patterns
        let sequences: Vec<PhraseSequence> = (0..50)
            .map(|i| PhraseSequence {
                source_id: format!("s{}", i),
                phrases: vec![
                    "type_1".to_string(),
                    "type_2".to_string(),
                    "type_3".to_string(),
                ],
                metadata_tags: vec![],
            })
            .collect();

        let config = ValidationConfig::default();
        let result = validate_linguistic_structure(&phrase_types, &sequences, &config).unwrap();

        assert!(
            result.zipf_correlation > 0.8,
            "Good corpus should be Zipfian"
        );
        assert!(result.has_syntax, "Repeating patterns should show syntax");
        assert!(
            result.singleton_rate < 0.3,
            "Good corpus should have few singletons"
        );
        assert!(
            result.validation_score > 0.5,
            "Good corpus should score well"
        );
    }

    #[test]
    fn test_validate_linguistic_structure_poor_corpus() {
        // Create a corpus with poor linguistic properties
        let phrase_types: Vec<PhraseType> = (1..=100)
            .map(|i| PhraseType {
                id: format!("type_{}", i),
                label: None,
                occurrence_count: 1, // All singletons
                centroid: vec![],
                contexts: HashMap::new(),
            })
            .collect();

        // Random sequences
        let sequences: Vec<PhraseSequence> = (0..10)
            .map(|i| PhraseSequence {
                source_id: format!("s{}", i),
                phrases: (0..10).map(|j| format!("type_{}", i * 10 + j)).collect(),
                metadata_tags: vec![],
            })
            .collect();

        let config = ValidationConfig::default();
        let result = validate_linguistic_structure(&phrase_types, &sequences, &config).unwrap();

        assert_eq!(
            result.singleton_rate, 1.0,
            "Poor corpus should have all singletons"
        );
        assert_eq!(
            result.reuse_ratio, 1.0,
            "Poor corpus should have reuse ratio of 1"
        );
        assert!(
            result.validation_score < 0.5,
            "Poor corpus should score poorly"
        );
    }

    #[test]
    fn test_compare_configurations() {
        // Config A: Poor
        let phrase_types_a: Vec<PhraseType> = (1..=100)
            .map(|i| PhraseType {
                id: format!("type_{}", i),
                label: None,
                occurrence_count: 1,
                centroid: vec![],
                contexts: HashMap::new(),
            })
            .collect();

        let sequences_a: Vec<PhraseSequence> = vec![];

        // Config B: Good (Zipfian)
        let phrase_types_b: Vec<PhraseType> = (1..=20)
            .map(|r| PhraseType {
                id: format!("type_{}", r),
                label: None,
                occurrence_count: 200 / r,
                centroid: vec![],
                contexts: HashMap::new(),
            })
            .collect();

        let sequences_b: Vec<PhraseSequence> = (0..10)
            .map(|i| PhraseSequence {
                source_id: format!("s{}", i),
                phrases: vec!["type_1".to_string(), "type_2".to_string()],
                metadata_tags: vec![],
            })
            .collect();

        let config = ValidationConfig::default();
        let comparison = compare_configurations(
            &phrase_types_a,
            &sequences_a,
            &phrase_types_b,
            &sequences_b,
            &config,
        )
        .unwrap();

        assert_eq!(comparison.winner, "B", "Better config should win");
        assert!(comparison.zipf_improvement > 0.0);
        assert!(comparison.reuse_improvement > 0.0);
    }

    // =========================================================================
    // Pearson Correlation Tests
    // =========================================================================

    #[test]
    fn test_pearson_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x
        let corr = pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // y = -x + 6
        let corr = pearson_correlation(&x, &y);
        assert!((corr - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_pearson_no_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 1.0, 4.0, 1.0, 5.0]; // No linear relationship
        let corr = pearson_correlation(&x, &y);
        assert!(corr.abs() < 0.5);
    }
}
