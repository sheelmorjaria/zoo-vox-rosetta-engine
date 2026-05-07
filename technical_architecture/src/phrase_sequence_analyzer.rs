// Phrase Sequence Analysis: Syntactic Structure Discovery
// ========================================================
//
// This module analyzes within-vocalization phrase sequences to discover:
// 1. Phrase "vocabulary" - clustering similar phrases into word types
// 2. PMI (Pointwise Mutual Information) - measuring word transition probabilities
// 3. Recurring patterns - finding common phrase sequences
// 4. Syntactic rules - discovering grammatical structure
//
// Research Goal: Prove that bat vocalizations follow fixed word order patterns
// (syntax) rather than random phrase combinations.

use std::collections::HashMap;
use thiserror::Error;

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, Error)]
pub enum PhraseSequenceError {
    #[error("No vocalizations provided for analysis")]
    NoVocalizations,

    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    #[error("Clustering failed: {0}")]
    ClusteringFailed(String),
}

pub type Result<T> = std::result::Result<T, PhraseSequenceError>;

// =============================================================================
// Phrase Representation
// =============================================================================

/// A detected phrase within a vocalization with acoustic features
#[derive(Debug, Clone)]
pub struct Phrase {
    /// Unique phrase ID
    pub id: usize,

    /// Start time in milliseconds
    pub start_ms: f64,

    /// Duration in milliseconds
    pub duration_ms: f64,

    /// Acoustic features for this phrase
    pub features: PhraseFeatures,
}

/// Acoustic features that define a phrase
#[derive(Debug, Clone, PartialEq)]
pub struct PhraseFeatures {
    /// Average F0 (fundamental frequency) in Hz
    pub f0_mean: f64,

    /// F0 standard deviation
    pub f0_std: f64,

    /// Spectral centroid in Hz
    pub spectral_centroid: f64,

    /// Spectral rolloff (85% energy point) in Hz
    pub spectral_rolloff: f64,

    /// Energy (RMS amplitude)
    pub energy: f64,

    /// Duration in milliseconds
    pub duration_ms: f64,
}

/// Word type - a cluster of similar phrases
#[derive(Debug, Clone)]
pub struct WordType {
    /// Unique word ID
    pub id: usize,

    /// Number of phrases assigned to this word
    pub count: usize,

    /// Representative features (centroid of cluster)
    pub features: PhraseFeatures,
}

/// A sequence of word types from a single vocalization
#[derive(Debug, Clone)]
pub struct WordSequence {
    /// Source vocalization ID
    pub vocalization_id: usize,

    /// Sequence of word IDs
    pub words: Vec<usize>,

    /// Original phrase boundaries (for reference)
    pub phrase_boundaries: Vec<f64>,
}

// =============================================================================
// Phrase Sequence Analyzer
// =============================================================================

pub struct PhraseSequenceAnalyzer {
    /// Clustering threshold for phrase similarity (cosine distance)
    similarity_threshold: f64,

    /// Minimum phrases to form a word type
    min_word_count: usize,
}

impl Default for PhraseSequenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl PhraseSequenceAnalyzer {
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.15, // Cosine distance threshold
            min_word_count: 5,          // Minimum occurrences to be a word type
        }
    }

    pub fn with_threshold(similarity_threshold: f64) -> Self {
        Self {
            similarity_threshold,
            min_word_count: 5,
        }
    }

    /// Extract phrases from a vocalization
    pub fn extract_phrases(&self, audio: &[f32], sample_rate: u32) -> Result<Vec<Phrase>> {
        use crate::within_vocalization_analyzer::{WithinVocalizationAnalyzer, WithinVocalizationConfig};

        // Use within-vocalization analyzer to detect phrase boundaries
        let config = WithinVocalizationConfig {
            min_phrase_duration_ms: 5.0,
            min_pause_duration_ms: 2.0,
            min_f0_change_hz: 1500.0,
            sample_rate,
            frame_size_ms: 2.0,
            hop_size_ms: 1.0,
            pause_energy_threshold: 0.15,
            require_consensus: false,
            max_phrases: 10,
        };

        let analyzer = WithinVocalizationAnalyzer::new(config.clone());
        let segmentation = analyzer
            .analyze_vocalization(audio, None)
            .map_err(|e| PhraseSequenceError::ClusteringFailed(e.to_string()))?;

        // Convert segmentation to phrases with features
        let mut phrases = Vec::new();
        let _hop_size_samples = (config.hop_size_ms * sample_rate as f64 / 1000.0) as usize;

        for (i, &start_ms) in segmentation.phrase_starts_ms.iter().enumerate() {
            let end_ms = start_ms + segmentation.phrase_durations_ms[i];
            let start_sample = (start_ms * sample_rate as f64 / 1000.0) as usize;
            let end_sample = ((end_ms * sample_rate as f64 / 1000.0) as usize).min(audio.len());

            if end_sample > start_sample {
                let phrase_audio = &audio[start_sample..end_sample];

                // Extract features
                let features = self.extract_phrase_features(phrase_audio, sample_rate);

                phrases.push(Phrase {
                    id: i,
                    start_ms,
                    duration_ms: segmentation.phrase_durations_ms[i],
                    features,
                });
            }
        }

        Ok(phrases)
    }

    /// Extract acoustic features from a phrase (simplified, no FFT)
    fn extract_phrase_features(&self, audio: &[f32], sample_rate: u32) -> PhraseFeatures {
        // Calculate energy (RMS)
        let energy: f64 = audio.iter().map(|&x| (x * x) as f64).sum::<f64>() / audio.len() as f64;
        let energy = energy.sqrt();

        // Calculate zero-crossing rate (simple frequency proxy)
        let zero_crossings = audio.windows(2).filter(|w| w[0] * w[1] < 0.0).count() as f64;
        let zcr = zero_crossings / audio.len() as f64;

        // Estimate F0 from ZCR (rough approximation for bats: 20-100 kHz)
        let f0_mean = zcr * sample_rate as f64 / 2.0;
        let f0_mean = f0_mean.clamp(10000.0, 100000.0); // Clamp to bat range

        // Spectral centroid approximated from ZCR (higher ZCR = higher frequency)
        let spectral_centroid = f0_mean * 1.2; // Rough approximation
        let spectral_rolloff = f0_mean * 1.5; // Rough approximation

        // Simple std dev approximation
        let f0_std = f0_mean * 0.1; // 10% variation estimate

        PhraseFeatures {
            f0_mean,
            f0_std,
            spectral_centroid,
            spectral_rolloff,
            energy,
            duration_ms: 0.0, // Set by caller
        }
    }

    /// Cluster similar phrases into word types
    pub fn discover_vocabulary(&self, all_phrases: Vec<Vec<Phrase>>) -> Result<Vec<WordType>> {
        let mut word_types: Vec<WordType> = Vec::new();
        let mut phrase_to_word: HashMap<(usize, usize), usize> = HashMap::new(); // (vocalization_id, phrase_id) -> word_id

        // Flatten all phrases
        let mut flat_phrases: Vec<(usize, usize, Phrase)> = Vec::new();
        for (voc_id, phrases) in all_phrases.iter().enumerate() {
            for phrase in phrases.iter() {
                flat_phrases.push((voc_id, phrase.id, phrase.clone()));
            }
        }

        // Cluster phrases by similarity
        let mut word_id_counter = 0;

        for (voc_id, phrase_id, phrase) in &flat_phrases {
            // Check if this phrase matches an existing word type
            let mut matched_word = None;

            for word in &word_types {
                let similarity = self.compute_similarity(&phrase.features, &word.features);
                if similarity > (1.0 - self.similarity_threshold) {
                    matched_word = Some(word.id);
                    break;
                }
            }

            if let Some(word_id) = matched_word {
                phrase_to_word.insert((*voc_id, *phrase_id), word_id);
                word_types[word_id].count += 1;
            } else {
                // Create new word type
                let new_word_id = word_id_counter;
                word_id_counter += 1;

                word_types.push(WordType {
                    id: new_word_id,
                    count: 1,
                    features: phrase.features.clone(),
                });

                phrase_to_word.insert((*voc_id, *phrase_id), new_word_id);
            }
        }

        // Filter by minimum count
        word_types.retain(|w| w.count >= self.min_word_count);

        // Reassign word IDs
        for (i, word) in word_types.iter_mut().enumerate() {
            word.id = i;
        }

        if word_types.is_empty() {
            return Err(PhraseSequenceError::InsufficientData(
                "No word types discovered - phrases may be too diverse".to_string(),
            ));
        }

        Ok(word_types)
    }

    /// Compute cosine similarity between two feature vectors
    fn compute_similarity(&self, f1: &PhraseFeatures, f2: &PhraseFeatures) -> f64 {
        // Normalize features to comparable ranges
        let v1 = self.normalize_features(f1);
        let v2 = self.normalize_features(f2);

        // Cosine similarity
        let dot_product = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum::<f64>();
        let norm1 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }

    fn normalize_features(&self, f: &PhraseFeatures) -> Vec<f64> {
        // Normalize to 0-1 range (approximate for bat vocalization ranges)
        vec![
            (f.f0_mean.log10() - 4.0) / 2.0,           // F0: 10^4 to 10^6 Hz
            (f.spectral_centroid.log10() - 4.0) / 2.0, // Centroid: same range
            f.spectral_rolloff / 50000.0,              // Rolloff: 0 to 50kHz
            f.energy.sqrt().min(1.0),                  // Energy: 0 to 1
        ]
    }

    /// Extract word sequences from vocalizations
    pub fn extract_sequences(
        &self,
        all_phrases: Vec<Vec<Phrase>>,
        word_types: &[WordType],
    ) -> Result<Vec<WordSequence>> {
        let mut sequences = Vec::new();

        for (voc_id, phrases) in all_phrases.iter().enumerate() {
            let mut words = Vec::new();
            let mut boundaries = Vec::new();

            for phrase in phrases {
                // Find matching word type for this phrase
                let mut word_id = None;

                for word in word_types {
                    let similarity = self.compute_similarity(&phrase.features, &word.features);
                    if similarity > (1.0 - self.similarity_threshold) {
                        word_id = Some(word.id);
                        break;
                    }
                }

                if let Some(wid) = word_id {
                    words.push(wid);
                    boundaries.push(phrase.start_ms);
                }
            }

            if !words.is_empty() {
                sequences.push(WordSequence {
                    vocalization_id: voc_id,
                    words,
                    phrase_boundaries: boundaries,
                });
            }
        }

        Ok(sequences)
    }

    /// Calculate PMI (Pointwise Mutual Information) for word transitions
    pub fn calculate_pmi(&self, sequences: &[WordSequence]) -> PMIAnalysis {
        let mut word_count: HashMap<usize, usize> = HashMap::new();
        let mut bigram_count: HashMap<(usize, usize), usize> = HashMap::new();
        let mut total_words = 0;
        let mut total_bigrams = 0;

        // Count unigrams and bigrams
        for seq in sequences {
            for (i, &word) in seq.words.iter().enumerate() {
                *word_count.entry(word).or_insert(0) += 1;
                total_words += 1;

                if i + 1 < seq.words.len() {
                    let bigram = (word, seq.words[i + 1]);
                    *bigram_count.entry(bigram).or_insert(0) += 1;
                    total_bigrams += 1;
                }
            }
        }

        // Calculate PMI for each bigram
        let mut pmi_scores: HashMap<(usize, usize), f64> = HashMap::new();

        for (bigram, &count) in &bigram_count {
            let (w1, w2) = *bigram;

            let p_w1 = word_count[&w1] as f64 / total_words as f64;
            let p_w2 = word_count[&w2] as f64 / total_words as f64;
            let p_w1_w2 = count as f64 / total_bigrams as f64;

            // PMI = log(P(w1,w2) / (P(w1) * P(w2)))
            let pmi = (p_w1_w2 / (p_w1 * p_w2)).max(1e-10).ln();
            pmi_scores.insert(*bigram, pmi);
        }

        // Calculate statistics
        let max_pmi = pmi_scores.values().fold(0.0_f64, |a, b| a.max(*b));
        let avg_pmi = if !pmi_scores.is_empty() {
            pmi_scores.values().sum::<f64>() / pmi_scores.len() as f64
        } else {
            0.0
        };

        // Find high-PMI transitions (indicating fixed word order)
        let mut high_pmi_transitions: Vec<(usize, usize, f64)> = pmi_scores
            .iter()
            .map(|(&(w1, w2), &pmi)| (w1, w2, pmi))
            .filter(|(_, _, pmi)| *pmi > 2.0) // PMI > 2.0 indicates strong association
            .collect::<Vec<_>>();

        high_pmi_transitions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        let vocabulary_size = word_count.len();

        PMIAnalysis {
            word_count,
            bigram_count,
            pmi_scores,
            max_pmi,
            avg_pmi,
            high_pmi_transitions,
            total_words,
            total_bigrams,
            vocabulary_size,
        }
    }

    /// Identify recurring n-gram patterns
    pub fn find_patterns(&self, sequences: &[WordSequence], max_n: usize) -> Vec<NGramPattern> {
        let mut ngram_counts: HashMap<Vec<usize>, usize> = HashMap::new();

        for seq in sequences {
            for n in 2..=max_n.min(seq.words.len()) {
                for window in seq.words.windows(n) {
                    *ngram_counts.entry(window.to_vec()).or_insert(0) += 1;
                }
            }
        }

        // Convert to patterns and sort by frequency
        let mut patterns: Vec<NGramPattern> = ngram_counts
            .into_iter()
            .map(|(words, count)| NGramPattern { words, count })
            .collect();

        patterns.sort_by_key(|b| std::cmp::Reverse(b.count));

        patterns
    }

    /// Discover syntactic rules
    pub fn discover_rules(&self, sequences: &[WordSequence]) -> SyntaxRules {
        let pmi = self.calculate_pmi(sequences);

        // Find common word positions (positional grammar)
        let mut position_words: HashMap<usize, Vec<usize>> = HashMap::new();

        for seq in sequences {
            for (pos, &word) in seq.words.iter().enumerate() {
                position_words.entry(pos).or_default().push(word);
            }
        }

        // Find most common word at each position
        let positional_grammar: HashMap<usize, usize> = position_words
            .into_iter()
            .filter_map(|(pos, words)| {
                // Find mode
                let mut counts: HashMap<usize, usize> = HashMap::new();
                for word in words {
                    *counts.entry(word).or_insert(0) += 1;
                }
                let most_common = counts.into_iter().max_by_key(|(_, count)| *count);
                most_common.map(|(word, _)| (pos, word))
            })
            .collect();

        // Identify "part of speech" categories by transition patterns
        let mut transition_patterns: HashMap<usize, Vec<usize>> = HashMap::new();

        for seq in sequences {
            for (i, &word) in seq.words.iter().enumerate() {
                if i + 1 < seq.words.len() {
                    transition_patterns.entry(word).or_default().push(seq.words[i + 1]);
                }
            }
        }

        SyntaxRules {
            vocabulary_size: pmi.vocabulary_size,
            fixed_transitions: pmi.high_pmi_transitions,
            positional_grammar,
            avg_sentence_length: if !sequences.is_empty() {
                sequences.iter().map(|s| s.words.len()).sum::<usize>() as f64 / sequences.len() as f64
            } else {
                0.0
            },
        }
    }
}

// =============================================================================
// Analysis Results
// =============================================================================

/// PMI analysis results
#[derive(Debug, Clone)]
pub struct PMIAnalysis {
    /// Count of each word
    pub word_count: HashMap<usize, usize>,

    /// Count of each word bigram
    pub bigram_count: HashMap<(usize, usize), usize>,

    /// PMI score for each bigram
    pub pmi_scores: HashMap<(usize, usize), f64>,

    /// Maximum PMI score
    pub max_pmi: f64,

    /// Average PMI score
    pub avg_pmi: f64,

    /// Transitions with PMI > 2.0 (strong associations)
    pub high_pmi_transitions: Vec<(usize, usize, f64)>,

    /// Total word tokens
    pub total_words: usize,

    /// Total bigram tokens
    pub total_bigrams: usize,

    /// Vocabulary size (unique words)
    pub vocabulary_size: usize,
}

/// A recurring n-gram pattern
#[derive(Debug, Clone)]
pub struct NGramPattern {
    /// Word sequence
    pub words: Vec<usize>,

    /// Occurrence count
    pub count: usize,
}

/// Discovered syntactic rules
#[derive(Debug, Clone)]
pub struct SyntaxRules {
    /// Number of unique words
    pub vocabulary_size: usize,

    /// Fixed word order transitions (high PMI)
    pub fixed_transitions: Vec<(usize, usize, f64)>,

    /// Positional grammar (most common word at each position)
    pub positional_grammar: HashMap<usize, usize>,

    /// Average sentence length in words
    pub avg_sentence_length: f64,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_computation() {
        let analyzer = PhraseSequenceAnalyzer::new();

        let features1 = PhraseFeatures {
            f0_mean: 20000.0,
            f0_std: 1000.0,
            spectral_centroid: 25000.0,
            spectral_rolloff: 40000.0,
            energy: 0.5,
            duration_ms: 20.0,
        };

        let features2 = PhraseFeatures {
            f0_mean: 21000.0, // Slightly different
            f0_std: 1100.0,
            spectral_centroid: 26000.0,
            spectral_rolloff: 41000.0,
            energy: 0.52,
            duration_ms: 20.0,
        };

        let similarity = analyzer.compute_similarity(&features1, &features2);

        // Similar phrases should have high similarity
        assert!(similarity > 0.8, "Similar phrases should have high similarity");
    }

    #[test]
    fn test_pmi_calculation() {
        let analyzer = PhraseSequenceAnalyzer::new();

        // Create test sequences
        let sequences = vec![
            WordSequence {
                vocalization_id: 0,
                words: vec![0, 1, 2, 0, 1], // Pattern: 0->1, 1->2, 2->0, 0->1
                phrase_boundaries: vec![0.0, 20.0, 40.0, 60.0, 80.0],
            },
            WordSequence {
                vocalization_id: 1,
                words: vec![0, 1, 2], // Same pattern
                phrase_boundaries: vec![0.0, 20.0, 40.0],
            },
            WordSequence {
                vocalization_id: 2,
                words: vec![1, 0, 1], // Different pattern
                phrase_boundaries: vec![0.0, 20.0, 40.0],
            },
        ];

        let pmi = analyzer.calculate_pmi(&sequences);

        // Check that PMI was calculated
        assert!(pmi.vocabulary_size > 0);
        assert!(pmi.total_words > 0);

        // Check that some transitions have PMI
        assert!(!pmi.pmi_scores.is_empty());
    }

    #[test]
    fn test_pattern_discovery() {
        let analyzer = PhraseSequenceAnalyzer::new();

        let sequences = vec![
            WordSequence {
                vocalization_id: 0,
                words: vec![0, 1, 2],
                phrase_boundaries: vec![0.0, 20.0, 40.0],
            },
            WordSequence {
                vocalization_id: 1,
                words: vec![0, 1, 2], // Same pattern
                phrase_boundaries: vec![0.0, 20.0, 40.0],
            },
            WordSequence {
                vocalization_id: 2,
                words: vec![0, 1, 3], // Different ending
                phrase_boundaries: vec![0.0, 20.0, 40.0],
            },
        ];

        let patterns = analyzer.find_patterns(&sequences, 3);

        // Should find [0, 1] as a common pattern
        let has_01_pattern = patterns.iter().any(|p| p.words == vec![0, 1]);
        assert!(has_01_pattern, "Should find [0, 1] pattern");
    }
}
