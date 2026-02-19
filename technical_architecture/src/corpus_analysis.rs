// Corpus Analysis Module
//
// Implements "Phrase X" discovery: Finding linguistic units with
// rigid internal structure but flexible external connections.
//
// This is the Method 1 approach: Information-Theoretic N-Gram Analysis
// using Pointwise Mutual Information (PMI) for internal rigidity and
// suffix entropy for external flexibility.
//
// Reference: Universal Rosetta Stone methodology for cross-species
// communication analysis.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum CorpusError {
    #[error("Empty corpus: no sequences provided")]
    EmptyCorpus,

    #[error("Sequence too short: {len} (minimum {min})")]
    SequenceTooShort { len: usize, min: usize },

    #[error("Invalid N-gram size: {n} (must be between 2 and 6)")]
    InvalidNGramSize { n: usize },

    #[error("Insufficient data for PMI calculation")]
    InsufficientDataForPMI,

    #[error("Symbol not found in corpus: {0}")]
    SymbolNotFound(usize),
}

pub type Result<T> = std::result::Result<T, CorpusError>;

// =============================================================================
// N-Gram Representation
// =============================================================================

/// An N-gram is a sequence of symbols (Cluster IDs)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NGram {
    /// The sequence of symbols in the n-gram
    pub symbols: Vec<usize>,
    /// The length of the n-gram
    pub n: usize,
}

impl NGram {
    /// Create a new n-gram from a sequence of symbols
    pub fn new(symbols: Vec<usize>) -> Result<Self> {
        let n = symbols.len();
        if !(2..=6).contains(&n) {
            return Err(CorpusError::InvalidNGramSize { n });
        }
        Ok(Self { symbols, n })
    }

    /// Create a 2-gram (bigram)
    pub fn bigram(a: usize, b: usize) -> Self {
        Self {
            symbols: vec![a, b],
            n: 2,
        }
    }

    /// Create a 3-gram (trigram)
    pub fn trigram(a: usize, b: usize, c: usize) -> Self {
        Self {
            symbols: vec![a, b, c],
            n: 3,
        }
    }

    /// Get the first n-1 symbols (prefix)
    pub fn prefix(&self) -> &[usize] {
        &self.symbols[..self.n - 1]
    }

    /// Get the last symbol (for suffix analysis)
    pub fn last(&self) -> usize {
        *self
            .symbols
            .last()
            .expect("n-gram should have at least 2 symbols")
    }

    /// Get all internal transitions (for PMI calculation)
    pub fn transitions(&self) -> Vec<(usize, usize)> {
        let mut transitions = Vec::with_capacity(self.n - 1);
        for i in 0..self.n - 1 {
            transitions.push((self.symbols[i], self.symbols[i + 1]));
        }
        transitions
    }
}

impl std::fmt::Display for NGram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let symbols: Vec<String> = self.symbols.iter().map(|s| s.to_string()).collect();
        write!(f, "[{}]", symbols.join(" → "))
    }
}

// =============================================================================
// Phrase X Candidate
// =============================================================================

/// A "Phrase X" candidate with rigid internal structure and flexible external connections
#[derive(Debug, Clone)]
pub struct PhraseX {
    /// The n-gram representing this phrase
    pub ngram: NGram,
    /// Internal rigidity score (average PMI of transitions)
    pub rigidity_score: f64,
    /// External flexibility score (entropy of following symbols)
    pub flexibility_score: f64,
    /// Number of times this phrase appears in the corpus
    pub frequency: usize,
    /// The symbols that follow this phrase (with counts)
    pub suffix_distribution: HashMap<usize, usize>,
}

impl PhraseX {
    /// Check if this phrase meets the thresholds for "Phrase X"
    pub fn is_phrase_x(&self, rigidity_threshold: f64, flexibility_threshold: f64) -> bool {
        self.rigidity_score >= rigidity_threshold && self.flexibility_score >= flexibility_threshold
    }

    /// Get the most common following symbol
    pub fn most_common_suffix(&self) -> Option<usize> {
        self.suffix_distribution
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&symbol, _)| symbol)
    }

    /// Get the diversity of suffixes (number of unique following symbols)
    pub fn suffix_diversity(&self) -> usize {
        self.suffix_distribution.len()
    }
}

impl std::fmt::Display for PhraseX {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PhraseX({}: rigidity={:.3}, flexibility={:.3}, freq={})",
            self.ngram, self.rigidity_score, self.flexibility_score, self.frequency
        )
    }
}

// =============================================================================
// N-Gram Miner
// =============================================================================

/// Mines n-grams from symbolic sequences
#[derive(Debug, Clone)]
pub struct NGramMiner {
    /// Maximum n-gram size to mine
    max_n: usize,
}

impl NGramMiner {
    /// Create a new n-gram miner
    ///
    /// # Arguments
    /// * `max_n` - Maximum n-gram size (default: 6)
    pub fn new(max_n: usize) -> Result<Self> {
        if !(2..=6).contains(&max_n) {
            return Err(CorpusError::InvalidNGramSize { n: max_n });
        }
        Ok(Self { max_n })
    }

    /// Create with default max_n = 6
    pub fn default() -> Self {
        Self { max_n: 6 }
    }

    /// Extract all n-grams from a single sequence
    pub fn extract_from_sequence(&self, sequence: &[usize]) -> Vec<NGram> {
        let mut ngrams = Vec::new();

        if sequence.len() < 2 {
            return ngrams;
        }

        // Extract n-grams of all sizes from 2 to max_n
        for n in 2..=self.max_n.min(sequence.len()) {
            for i in 0..=sequence.len() - n {
                let symbols = sequence[i..i + n].to_vec();
                if let Ok(ngram) = NGram::new(symbols) {
                    ngrams.push(ngram);
                }
            }
        }

        ngrams
    }

    /// Extract all n-grams from multiple sequences
    pub fn extract_from_corpus(&self, sequences: &[Vec<usize>]) -> Vec<NGram> {
        let mut all_ngrams = Vec::new();

        for sequence in sequences {
            all_ngrams.extend(self.extract_from_sequence(sequence));
        }

        all_ngrams
    }

    /// Count n-gram frequencies across the corpus
    pub fn count_ngrams(&self, sequences: &[Vec<usize>]) -> HashMap<NGram, usize> {
        let mut counts = HashMap::new();

        for ngram in self.extract_from_corpus(sequences) {
            *counts.entry(ngram).or_insert(0) += 1;
        }

        counts
    }
}

// =============================================================================
// PMI Calculator (Pointwise Mutual Information)
// =============================================================================

/// Calculates Pointwise Mutual Information for measuring internal rigidity
#[derive(Debug, Clone)]
pub struct PMICalculator {
    /// Unigram probabilities: P(symbol)
    unigram_probs: HashMap<usize, f64>,
    /// Bigram probabilities: P(symbol_a, symbol_b)
    bigram_probs: HashMap<(usize, usize), f64>,
    /// Total number of symbols in corpus
    total_symbols: usize,
    /// Total number of bigrams in corpus
    total_bigrams: usize,
}

impl PMICalculator {
    /// Create a new PMI calculator from corpus sequences
    pub fn from_corpus(sequences: &[Vec<usize>]) -> Result<Self> {
        if sequences.is_empty() {
            return Err(CorpusError::EmptyCorpus);
        }

        let mut unigram_counts: HashMap<usize, usize> = HashMap::new();
        let mut bigram_counts: HashMap<(usize, usize), usize> = HashMap::new();
        let mut total_symbols = 0;
        let mut total_bigrams = 0;

        for sequence in sequences {
            if sequence.is_empty() {
                continue;
            }

            // Count unigrams
            for &symbol in sequence {
                *unigram_counts.entry(symbol).or_insert(0) += 1;
                total_symbols += 1;
            }

            // Count bigrams
            for i in 0..sequence.len().saturating_sub(1) {
                let bigram = (sequence[i], sequence[i + 1]);
                *bigram_counts.entry(bigram).or_insert(0) += 1;
                total_bigrams += 1;
            }
        }

        if total_symbols == 0 || total_bigrams == 0 {
            return Err(CorpusError::InsufficientDataForPMI);
        }

        // Convert to probabilities
        let unigram_probs: HashMap<usize, f64> = unigram_counts
            .into_iter()
            .map(|(symbol, count)| (symbol, count as f64 / total_symbols as f64))
            .collect();

        let bigram_probs: HashMap<(usize, usize), f64> = bigram_counts
            .into_iter()
            .map(|(bigram, count)| (bigram, count as f64 / total_bigrams as f64))
            .collect();

        Ok(Self {
            unigram_probs,
            bigram_probs,
            total_symbols,
            total_bigrams,
        })
    }

    /// Calculate PMI for a single transition: PMI(a, b) = log(P(a,b) / (P(a) * P(b)))
    pub fn pmi(&self, a: usize, b: usize) -> Result<f64> {
        let p_a = self
            .unigram_probs
            .get(&a)
            .ok_or(CorpusError::SymbolNotFound(a))?;
        let p_b = self
            .unigram_probs
            .get(&b)
            .ok_or(CorpusError::SymbolNotFound(b))?;
        let p_ab = self.bigram_probs.get(&(a, b)).unwrap_or(&0.0);

        if *p_a == 0.0 || *p_b == 0.0 || *p_ab == 0.0 {
            return Ok(f64::NEG_INFINITY); // No association
        }

        let pmi = (p_ab / (p_a * p_b)).ln();
        Ok(pmi)
    }

    /// Calculate average PMI for all transitions in an n-gram
    ///
    /// This is the "rigidity score" - high values indicate the symbols
    /// are strongly associated and tend to appear together.
    pub fn average_pmi(&self, ngram: &NGram) -> Result<f64> {
        let transitions = ngram.transitions();

        if transitions.is_empty() {
            return Ok(0.0);
        }

        let mut total_pmi = 0.0;
        let mut valid_transitions = 0;

        for (a, b) in transitions {
            match self.pmi(a, b) {
                Ok(pmi) if pmi.is_finite() => {
                    total_pmi += pmi;
                    valid_transitions += 1;
                }
                _ => {
                    // Skip infinite/NaN PMI values
                }
            }
        }

        if valid_transitions == 0 {
            return Ok(0.0);
        }

        Ok(total_pmi / valid_transitions as f64)
    }
}

// =============================================================================
// Suffix Entropy Calculator
// =============================================================================

/// Calculates entropy of following symbols for measuring external flexibility
#[derive(Debug, Clone)]
pub struct SuffixEntropyCalculator {
    /// For each n-gram, the count of each following symbol
    suffix_counts: HashMap<NGram, HashMap<usize, usize>>,
    /// For each n-gram, the total count
    ngram_counts: HashMap<NGram, usize>,
}

impl SuffixEntropyCalculator {
    /// Create a new suffix entropy calculator from corpus sequences
    pub fn from_corpus(sequences: &[Vec<usize>]) -> Result<Self> {
        if sequences.is_empty() {
            return Err(CorpusError::EmptyCorpus);
        }

        let mut suffix_counts: HashMap<NGram, HashMap<usize, usize>> = HashMap::new();
        let mut ngram_counts: HashMap<NGram, usize> = HashMap::new();

        for sequence in sequences {
            if sequence.len() < 3 {
                continue; // Need at least n-gram + 1 symbol
            }

            // Extract n-grams of size 2 to 6
            for n in 2..=6.min(sequence.len()) {
                for i in 0..sequence.len().saturating_sub(n) {
                    let ngram_symbols = sequence[i..i + n].to_vec();
                    let ngram = NGram::new(ngram_symbols);

                    if let Ok(ng) = ngram {
                        // Check we have room for suffix
                        if i + n < sequence.len() {
                            let suffix = sequence[i + n];

                            *ngram_counts.entry(ng.clone()).or_insert(0) += 1;
                            *suffix_counts
                                .entry(ng)
                                .or_default()
                                .entry(suffix)
                                .or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        Ok(Self {
            suffix_counts,
            ngram_counts,
        })
    }

    /// Calculate the entropy of the suffix distribution for an n-gram
    ///
    /// H(Next | Phrase) = -sum P(x | Phrase) * log(P(x | Phrase))
    ///
    /// High entropy = flexible external connections (many different following symbols)
    /// Low entropy = rigid chain (always follows the same pattern)
    pub fn suffix_entropy(&self, ngram: &NGram) -> f64 {
        let suffix_dist = self.suffix_counts.get(ngram);
        let total_count = self.ngram_counts.get(ngram).copied().unwrap_or(0);

        if total_count == 0 {
            return 0.0;
        }

        let dist = match suffix_dist {
            Some(d) if !d.is_empty() => d,
            _ => return 0.0,
        };

        let mut entropy = 0.0;
        for &count in dist.values() {
            let p = count as f64 / total_count as f64;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        entropy
    }

    /// Get the suffix distribution for an n-gram
    pub fn suffix_distribution(&self, ngram: &NGram) -> HashMap<usize, usize> {
        self.suffix_counts.get(ngram).cloned().unwrap_or_default()
    }
}

// =============================================================================
// Phrase X Discovery Engine
// =============================================================================

/// Discovers "Phrase X" units with rigid internal structure and flexible external connections
#[derive(Debug, Clone)]
pub struct PhraseXDiscoveryEngine {
    /// PMI calculator for rigidity scoring
    pmi_calculator: PMICalculator,
    /// Suffix entropy calculator for flexibility scoring
    entropy_calculator: SuffixEntropyCalculator,
    /// N-gram miner for extraction
    ngram_miner: NGramMiner,
    /// Minimum frequency threshold for considering n-grams
    min_frequency: usize,
    /// Rigidity threshold (PMI)
    rigidity_threshold: f64,
    /// Flexibility threshold (entropy)
    flexibility_threshold: f64,
}

impl PhraseXDiscoveryEngine {
    /// Create a new Phrase X discovery engine
    ///
    /// # Arguments
    /// * `sequences` - Corpus of symbolic sequences (Cluster IDs)
    /// * `min_frequency` - Minimum occurrences for an n-gram to be considered
    /// * `rigidity_threshold` - Minimum average PMI score (default: 2.5)
    /// * `flexibility_threshold` - Minimum suffix entropy (default: 1.5)
    pub fn new(
        sequences: &[Vec<usize>],
        min_frequency: usize,
        rigidity_threshold: f64,
        flexibility_threshold: f64,
    ) -> Result<Self> {
        if sequences.is_empty() {
            return Err(CorpusError::EmptyCorpus);
        }

        let pmi_calculator = PMICalculator::from_corpus(sequences)?;
        let entropy_calculator = SuffixEntropyCalculator::from_corpus(sequences)?;
        let ngram_miner = NGramMiner::default();

        Ok(Self {
            pmi_calculator,
            entropy_calculator,
            ngram_miner,
            min_frequency,
            rigidity_threshold,
            flexibility_threshold,
        })
    }

    /// Create with default thresholds (rigidity=2.5, flexibility=1.5)
    pub fn with_defaults(sequences: &[Vec<usize>]) -> Result<Self> {
        Self::new(sequences, 2, 2.5, 1.5)
    }

    /// Discover all Phrase X candidates in the corpus
    pub fn discover(&self) -> Result<Vec<PhraseX>> {
        let _ngram_counts = self.ngram_miner.count_ngrams(&[]); // Will be recalculated

        // We need to extract n-grams and their frequencies
        let mut phrases_x = Vec::new();

        // Get all unique n-grams from the entropy calculator
        let unique_ngrams: HashSet<NGram> = self
            .entropy_calculator
            .suffix_counts
            .keys()
            .cloned()
            .collect();

        for ngram in unique_ngrams {
            // Check minimum frequency
            let frequency = self
                .entropy_calculator
                .ngram_counts
                .get(&ngram)
                .copied()
                .unwrap_or(0);

            if frequency < self.min_frequency {
                continue;
            }

            // Calculate rigidity score (average PMI)
            let rigidity_score = self.pmi_calculator.average_pmi(&ngram).unwrap_or(0.0);

            // Calculate flexibility score (suffix entropy)
            let flexibility_score = self.entropy_calculator.suffix_entropy(&ngram);

            // Get suffix distribution
            let suffix_distribution = self.entropy_calculator.suffix_distribution(&ngram);

            let phrase = PhraseX {
                ngram: ngram.clone(),
                rigidity_score,
                flexibility_score,
                frequency,
                suffix_distribution,
            };

            phrases_x.push(phrase);
        }

        // Sort by combined score (rigidity + flexibility)
        phrases_x.sort_by(|a, b| {
            let score_a = a.rigidity_score + a.flexibility_score;
            let score_b = b.rigidity_score + b.flexibility_score;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(phrases_x)
    }

    /// Filter phrases that meet the Phrase X thresholds
    pub fn filter_phrases_x(&self, phrases: &[PhraseX]) -> Vec<PhraseX> {
        phrases
            .iter()
            .filter(|p| p.is_phrase_x(self.rigidity_threshold, self.flexibility_threshold))
            .cloned()
            .collect()
    }

    /// Get top N phrases by combined score
    pub fn top_n(&self, phrases: &[PhraseX], n: usize) -> Vec<PhraseX> {
        phrases.iter().take(n).cloned().collect()
    }

    /// Analyze context variability for a phrase
    ///
    /// Returns examples of different contexts where this phrase appears
    pub fn analyze_context_variability(
        &self,
        phrase: &PhraseX,
        sequences: &[Vec<usize>],
    ) -> Vec<Vec<usize>> {
        let mut contexts = Vec::new();

        for sequence in sequences {
            let n = phrase.ngram.n;

            for i in 0..=sequence.len().saturating_sub(n + 1) {
                let ngram_symbols = sequence[i..i + n].to_vec();
                if let Ok(ngram) = NGram::new(ngram_symbols) {
                    if ngram == phrase.ngram {
                        // Found the phrase, capture context
                        let context_start = i.saturating_sub(2);
                        let context_end = (i + n + 2).min(sequence.len());
                        contexts.push(sequence[context_start..context_end].to_vec());
                    }
                }
            }

            if contexts.len() >= 10 {
                // Limit to 10 examples
                break;
            }
        }

        contexts
    }
}

// =============================================================================
// Statistics and Analysis
// =============================================================================

/// Corpus-wide statistics
#[derive(Debug, Clone)]
pub struct CorpusStatistics {
    /// Total number of sequences
    pub total_sequences: usize,
    /// Total number of symbols
    pub total_symbols: usize,
    /// Vocabulary size (unique symbols)
    pub vocabulary_size: usize,
    /// Average sequence length
    pub avg_sequence_length: f64,
    /// Number of unique n-grams found
    pub unique_ngrams: usize,
}

impl CorpusStatistics {
    /// Calculate statistics for a corpus
    pub fn from_corpus(sequences: &[Vec<usize>]) -> Result<Self> {
        if sequences.is_empty() {
            return Err(CorpusError::EmptyCorpus);
        }

        let total_sequences = sequences.len();
        let total_symbols: usize = sequences.iter().map(|s| s.len()).sum();

        let vocabulary: HashSet<usize> = sequences.iter().flatten().cloned().collect();
        let vocabulary_size = vocabulary.len();

        let avg_sequence_length = if total_sequences > 0 {
            total_symbols as f64 / total_sequences as f64
        } else {
            0.0
        };

        // Count unique n-grams
        let miner = NGramMiner::default();
        let _unique_ngrams = miner.extract_from_corpus(sequences).len();
        // We need to deduplicate
        let unique_ngrams_set: HashSet<_> =
            miner.extract_from_corpus(sequences).into_iter().collect();

        Ok(Self {
            total_sequences,
            total_symbols,
            vocabulary_size,
            avg_sequence_length,
            unique_ngrams: unique_ngrams_set.len(),
        })
    }
}

// =============================================================================
// Tests (TDD Approach)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple test corpus with clear Phrase X patterns
    ///
    /// Pattern: [A, B] always appears together (rigid internal)
    /// But followed by many different symbols (flexible external)
    fn create_test_corpus() -> Vec<Vec<usize>> {
        vec![
            // [0, 1] is our "Phrase X" - rigid internal, flexible external
            vec![0, 1, 2, 0, 1, 3, 0, 1, 4],
            vec![0, 1, 5, 2, 3, 2, 3, 2], // [2, 3] is rigid chain (always followed by 2)
            vec![0, 1, 6, 4, 5, 4, 5, 4], // [4, 5] is another rigid chain
            vec![0, 1, 2, 0, 1, 7, 0, 1, 8],
            vec![2, 3, 2, 4, 5, 4, 0, 1, 9],
        ]
    }

    #[test]
    fn test_ngram_creation() {
        let ngram = NGram::new(vec![1, 2, 3]).unwrap();
        assert_eq!(ngram.n, 3);
        assert_eq!(ngram.symbols, vec![1, 2, 3]);
    }

    #[test]
    fn test_ngram_transitions() {
        let ngram = NGram::new(vec![1, 2, 3]).unwrap();
        let transitions = ngram.transitions();
        assert_eq!(transitions, vec![(1, 2), (2, 3)]);
    }

    #[test]
    fn test_ngram_miner_extract() {
        let miner = NGramMiner::default();
        let sequence = vec![1, 2, 3, 4];
        let ngrams = miner.extract_from_sequence(&sequence);

        // Should extract: [1,2], [2,3], [3,4], [1,2,3], [2,3,4], [1,2,3,4]
        assert!(ngrams.len() >= 4);
    }

    #[test]
    fn test_pmi_calculator() {
        let corpus = vec![vec![0, 1, 0, 1, 0, 1]];
        let calc = PMICalculator::from_corpus(&corpus).unwrap();

        // [0, 1] always appears together, should have high PMI
        let pmi = calc.pmi(0, 1).unwrap();
        assert!(pmi > 0.0);
    }

    #[test]
    fn test_suffix_entropy() {
        let corpus = vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 1, 4]];
        let calc = SuffixEntropyCalculator::from_corpus(&corpus).unwrap();

        let ngram = NGram::bigram(0, 1);
        let entropy = calc.suffix_entropy(&ngram);

        // Should have high entropy (followed by 2, 3, 4)
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_phrase_x_discovery() {
        let corpus = create_test_corpus();
        // Use lower thresholds for the small test corpus
        let engine = PhraseXDiscoveryEngine::new(&corpus, 2, 0.1, 0.1).unwrap();

        let phrases = engine.discover().unwrap();
        let phrases_x = engine.filter_phrases_x(&phrases);

        // Should find at least one Phrase X
        assert!(!phrases_x.is_empty());

        // The [0, 1] pattern should be identified as Phrase X
        let phrase_01 = phrases_x.iter().find(|p| p.ngram.symbols == vec![0, 1]);
        assert!(phrase_01.is_some());

        if let Some(p) = phrase_01 {
            // Should have high flexibility (followed by many symbols)
            assert!(p.flexibility_score > 0.5);
            // Should have at least 3 different suffixes
            assert!(p.suffix_diversity() >= 3);
        }
    }

    #[test]
    fn test_corpus_statistics() {
        let corpus = create_test_corpus();
        let stats = CorpusStatistics::from_corpus(&corpus).unwrap();

        assert_eq!(stats.total_sequences, 5);
        assert!(stats.total_symbols > 0);
        assert!(stats.vocabulary_size > 0);
        assert!(stats.avg_sequence_length > 0.0);
    }

    #[test]
    fn test_empty_corpus_error() {
        let corpus: Vec<Vec<usize>> = vec![];
        let result = PhraseXDiscoveryEngine::with_defaults(&corpus);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_ngram_size() {
        let result = NGram::new(vec![1]); // Too short
        assert!(result.is_err());

        let result = NGram::new(vec![1, 2, 3, 4, 5, 6, 7]); // Too long
        assert!(result.is_err());
    }

    #[test]
    fn test_rigid_chain_vs_phrase_x() {
        // Create corpus with both rigid chains and Phrase X
        let corpus = vec![
            // [0, 1] is Phrase X (rigid internal, flexible external)
            vec![0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5],
            // [2, 3, 2] is rigid chain (predictable pattern)
            vec![2, 3, 2, 3, 2, 3, 2],
        ];

        let engine = PhraseXDiscoveryEngine::with_defaults(&corpus).unwrap();
        let phrases = engine.discover().unwrap();

        // Find [0, 1]
        let phrase_01 = phrases.iter().find(|p| p.ngram.symbols == vec![0, 1]);
        // Find [2, 3]
        let phrase_23 = phrases.iter().find(|p| p.ngram.symbols == vec![2, 3]);

        if let Some(p01) = phrase_01 {
            if let Some(p23) = phrase_23 {
                // [0, 1] should have higher flexibility than [2, 3]
                assert!(p01.flexibility_score > p23.flexibility_score);
            }
        }
    }

    #[test]
    fn test_context_variability() {
        let corpus = create_test_corpus();
        let engine = PhraseXDiscoveryEngine::with_defaults(&corpus).unwrap();
        let phrases = engine.discover().unwrap();

        if let Some(phrase) = phrases.first() {
            let contexts = engine.analyze_context_variability(phrase, &corpus);
            assert!(!contexts.is_empty());
        }
    }
}
