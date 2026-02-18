//! Within-Call Phrase Discovery: Acoustic Similarity Engine
//!
//! This module implements phrase discovery within single vocalizations using
//! acoustic similarity rather than clustering. This approach recognizes that
//! vocalizations exist on CONTINUOUS ACOUSTIC MANIFOLDS, not discrete islands.
//!
//! Key Insight:
//! ─────────────
//! Animal vocalizations form continuous gradients:
//!   Phee ←───────→ Trill ←───────→ Twitter ←──────→ Tsik
//!      (continuous acoustic transitions, not separate clusters)
//!
//! Uses weighted cosine similarity on 30D features for phrase typing.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::zoo_vox_data_models::{AcousticFeatures30D, PhrasePrototype};
use crate::acoustic_similarity::{AcousticSimilarityEngine, DistanceMetric};
use crate::species::FeatureWeights;

// =============================================================================
// Within-Call Analysis Configuration
// =============================================================================

/// Configuration for within-call phrase analysis
#[derive(Debug, Clone)]
pub struct WithinCallConfig {
    /// Similarity threshold for phrase typing (0.85 = 85% similar)
    pub similarity_threshold: f64,

    /// Distance threshold for merging phrases (lower = stricter)
    pub distance_threshold: f64,

    /// Minimum phrase duration in ms
    pub min_phrase_duration_ms: f64,

    /// Whether to use weighted features
    pub use_weighted_features: bool,

    /// Distance metric to use
    pub distance_metric: DistanceMetric,
}

impl Default for WithinCallConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            distance_threshold: 0.3,
            min_phrase_duration_ms: 10.0,
            use_weighted_features: true,
            distance_metric: DistanceMetric::Cosine,
        }
    }
}

impl WithinCallConfig {
    /// Create config for specific species
    pub fn for_species(species: &str) -> Self {
        let mut config = Self::default();

        // Adjust thresholds based on species encoding strategy
        match species {
            "sperm_whale" => {
                // Coda-type: short clicks, use stricter threshold
                config.similarity_threshold = 0.90;
                config.distance_threshold = 0.2;
            }
            "dolphin" | "orca" => {
                // Frequency-modulated: continuous contours, use looser threshold
                config.similarity_threshold = 0.80;
                config.distance_threshold = 0.4;
            }
            "zebra_finch" | "egyptian_bat" => {
                // Combinatorial: discrete syllables
                config.similarity_threshold = 0.85;
                config.distance_threshold = 0.3;
            }
            "marmoset" => {
                // Harmonic calls with continuous variation
                config.similarity_threshold = 0.85;
                config.distance_threshold = 0.3;
            }
            _ => {}
        }

        config
    }
}

// =============================================================================
// Phrase Type Discovery Result
// =============================================================================

/// A discovered phrase type from within-call analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredPhraseType {
    /// Unique phrase type identifier
    pub type_id: String,

    /// Human-readable phrase key
    pub phrase_key: String,

    /// Representative (centroid) features
    pub centroid_features: AcousticFeatures30D,

    /// All instances of this phrase type
    pub instances: Vec<PhraseInstance>,

    /// Intra-type variability (average pairwise distance)
    pub intra_variability: f64,

    /// Number of occurrences
    pub occurrence_count: usize,

    /// Associated contexts (if available)
    pub contexts: Vec<String>,
}

/// A single instance of a phrase within a call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseInstance {
    /// Source file or call ID
    pub source_id: String,

    /// Position in the call (0.0 to 1.0)
    pub position: f64,

    /// Start sample index
    pub start_sample: usize,

    /// End sample index
    pub end_sample: usize,

    /// 30D features
    pub features: AcousticFeatures30D,

    /// Distance to centroid
    pub distance_to_centroid: f64,
}

// =============================================================================
// Within-Call Analysis Result
// =============================================================================

/// Results from within-call phrase analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithinCallAnalysisResult {
    /// Original call/vocalization ID
    pub call_id: String,

    /// Species
    pub species: String,

    /// Discovered phrase types
    pub phrase_types: Vec<DiscoveredPhraseType>,

    /// Phrase sequence (ordered by position)
    pub phrase_sequence: Vec<String>,

    /// Transition counts between phrase types
    pub transition_matrix: HashMap<String, HashMap<String, usize>>,

    /// Overall call statistics
    pub total_phrases: usize,
    pub unique_types: usize,
    pub type_entropy: f64,

    /// Similarity statistics
    pub avg_within_type_similarity: f64,
    pub avg_between_type_distance: f64,
}

// =============================================================================
// Within-Call Analyzer
// =============================================================================

/// Analyzes single vocalizations to discover phrase types using acoustic similarity
pub struct WithinCallAnalyzer {
    config: WithinCallConfig,
    similarity_engine: AcousticSimilarityEngine,
}

impl WithinCallAnalyzer {
    /// Create new analyzer with default configuration
    pub fn new() -> Self {
        let config = WithinCallConfig::default();
        let similarity_engine = AcousticSimilarityEngine::with_metric(
            30,
            config.distance_metric.clone(),
        );

        Self {
            config,
            similarity_engine,
        }
    }

    /// Create analyzer for specific species with optimized weights
    ///
    /// This implements the "Router then Analyzer" pattern:
    /// - Phase 1 (Species ID): Uses unified 45D space (handled elsewhere)
    /// - Phase 2 (Phrase Analysis): Uses species-specific weights for
    ///   enhanced within-species discrimination
    ///
    /// The weights are applied to improve phrase type discrimination
    /// within a single species' vocal repertoire.
    pub fn for_species(species: &str) -> Self {
        let config = WithinCallConfig::for_species(species);
        let mut similarity_engine = AcousticSimilarityEngine::with_metric(
            30,
            config.distance_metric.clone(),
        );

        // Apply species-specific weights for Phase 2 analysis
        // This is the CORRECT use of species weights - within-species
        // discrimination, not cross-species comparison
        let weights = get_species_weights_30d(species);
        if config.use_weighted_features {
            similarity_engine.set_feature_weights(&weights);
        }

        Self {
            config,
            similarity_engine,
        }
    }

    /// Create analyzer for species with explicit weights
    pub fn for_species_with_weights(species: &str, weights: &FeatureWeights) -> Self {
        let config = WithinCallConfig::for_species(species);
        let mut similarity_engine = AcousticSimilarityEngine::with_metric(
            30,
            config.distance_metric.clone(),
        );

        // Convert 45D weights to 30D (first 6 feature groups)
        let weights_30d = weights.to_weight_vector_30d();
        if config.use_weighted_features {
            similarity_engine.set_feature_weights(&weights_30d);
        }

        Self {
            config,
            similarity_engine,
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: WithinCallConfig) -> Self {
        let similarity_engine = AcousticSimilarityEngine::with_metric(
            30,
            config.distance_metric.clone(),
        );

        Self {
            config,
            similarity_engine,
        }
    }

    /// Discover phrase types within a single vocalization
    ///
    /// This uses similarity-based grouping instead of clustering,
    /// which is more appropriate for continuous acoustic manifolds.
    pub fn discover_phrases(
        &mut self,
        phrase_candidates: Vec<PhrasePrototype>,
        call_id: &str,
        species: &str,
    ) -> WithinCallAnalysisResult {
        if phrase_candidates.is_empty() {
            return WithinCallAnalysisResult {
                call_id: call_id.to_string(),
                species: species.to_string(),
                phrase_types: Vec::new(),
                phrase_sequence: Vec::new(),
                transition_matrix: HashMap::new(),
                total_phrases: 0,
                unique_types: 0,
                type_entropy: 0.0,
                avg_within_type_similarity: 0.0,
                avg_between_type_distance: 0.0,
            };
        }

        // Convert to feature vectors
        let feature_vectors: Vec<ndarray::Array1<f64>> = phrase_candidates.iter()
            .map(|p| ndarray::Array1::from_vec(p.features_30d.to_vector().to_vec()))
            .collect();

        // Fit normalization
        let n = feature_vectors.len();
        let n_features = 30;
        let mut feature_matrix = ndarray::Array2::zeros((n, n_features));
        for (i, vec) in feature_vectors.iter().enumerate() {
            for j in 0..n_features.min(vec.len()) {
                feature_matrix[[i, j]] = vec[j];
            }
        }
        self.similarity_engine.fit_normalization(&feature_matrix);

        // Compute pairwise similarity matrix
        let similarity_matrix = self.compute_similarity_matrix(&feature_vectors);

        // Group phrases by similarity
        let phrase_types = self.group_by_similarity(
            &phrase_candidates,
            &similarity_matrix,
            &feature_vectors,
        );

        // Build phrase sequence
        let phrase_sequence: Vec<String> = phrase_types.iter()
            .flat_map(|pt| std::iter::repeat(pt.type_id.clone()).take(pt.instances.len()))
            .collect();

        // Compute transition matrix
        let transition_matrix = self.compute_transition_matrix(&phrase_sequence);

        // Compute statistics
        let total_phrases = phrase_candidates.len();
        let unique_types = phrase_types.len();

        let total_occurrences: usize = phrase_types.iter()
            .map(|pt| pt.occurrence_count)
            .sum();

        let type_entropy = if total_occurrences > 0 {
            phrase_types.iter()
                .map(|pt| {
                    let p = pt.occurrence_count as f64 / total_occurrences as f64;
                    if p > 0.0 { -p * p.log2() } else { 0.0 }
                })
                .sum()
        } else {
            0.0
        };

        // Compute similarity statistics
        let (avg_within, avg_between) = self.compute_similarity_stats(&phrase_types);

        WithinCallAnalysisResult {
            call_id: call_id.to_string(),
            species: species.to_string(),
            phrase_types,
            phrase_sequence,
            transition_matrix,
            total_phrases,
            unique_types,
            type_entropy,
            avg_within_type_similarity: avg_within,
            avg_between_type_distance: avg_between,
        }
    }

    /// Compute pairwise similarity matrix
    fn compute_similarity_matrix(&self, features: &[ndarray::Array1<f64>]) -> Vec<Vec<f64>> {
        let n = features.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i..n {
                let sim = self.similarity_engine.similarity(&features[i], &features[j]);
                matrix[i][j] = sim;
                matrix[j][i] = sim;
            }
        }

        matrix
    }

    /// Group phrases by acoustic similarity
    fn group_by_similarity(
        &self,
        phrases: &[PhrasePrototype],
        similarity_matrix: &[Vec<f64>],
        feature_vectors: &[ndarray::Array1<f64>],
    ) -> Vec<DiscoveredPhraseType> {
        let n = phrases.len();
        let mut assigned = vec![false; n];
        let mut phrase_types: Vec<DiscoveredPhraseType> = Vec::new();

        for i in 0..n {
            if assigned[i] {
                continue;
            }

            // Find all phrases similar to this one
            let mut group_indices = vec![i];
            assigned[i] = true;

            for j in (i + 1)..n {
                if assigned[j] {
                    continue;
                }

                let sim = similarity_matrix[i][j];
                if sim >= self.config.similarity_threshold {
                    group_indices.push(j);
                    assigned[j] = true;
                }
            }

            // Compute centroid features
            let centroid = self.compute_centroid(&group_indices, feature_vectors);

            // Create phrase instances
            let instances: Vec<PhraseInstance> = group_indices.iter()
                .map(|&idx| {
                    let phrase = &phrases[idx];
                    PhraseInstance {
                        source_id: phrase.source_file.clone().unwrap_or_default(),
                        position: phrase.typical_position as f64 / n as f64,
                        start_sample: 0, // Would be filled from actual segmentation
                        end_sample: 0,
                        features: phrase.features_30d.clone(),
                        distance_to_centroid: self.similarity_engine.distance(
                            &feature_vectors[idx],
                            &centroid,
                        ),
                    }
                })
                .collect();

            // Compute intra-type variability
            let intra_variability = if instances.len() > 1 {
                let mut total_dist = 0.0;
                let mut count = 0;
                for a in &instances {
                    for b in &instances {
                        if a.source_id != b.source_id {
                            let a_vec = ndarray::Array1::from_vec(a.features.to_vector().to_vec());
                            let b_vec = ndarray::Array1::from_vec(b.features.to_vector().to_vec());
                            total_dist += self.similarity_engine.distance(&a_vec, &b_vec);
                            count += 1;
                        }
                    }
                }
                if count > 0 { total_dist / count as f64 } else { 0.0 }
            } else {
                0.0
            };

            // Collect contexts
            let contexts: Vec<String> = phrases.iter()
                .enumerate()
                .filter(|(idx, _)| group_indices.contains(idx))
                .filter_map(|(_, p)| p.primary_context.clone())
                .collect();

            let template = &phrases[i];
            let centroid_features = AcousticFeatures30D::from_vector(
                centroid.into_raw_vec().try_into().unwrap_or([0.0; 30])
            );

            phrase_types.push(DiscoveredPhraseType {
                type_id: format!("{}_type_{}", template.species, phrase_types.len()),
                phrase_key: template.phrase_key.clone(),
                centroid_features,
                instances,
                intra_variability,
                occurrence_count: group_indices.len(),
                contexts,
            });
        }

        // Sort by occurrence count descending
        phrase_types.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));

        phrase_types
    }

    /// Compute centroid of a group of feature vectors
    fn compute_centroid(
        &self,
        indices: &[usize],
        feature_vectors: &[ndarray::Array1<f64>],
    ) -> ndarray::Array1<f64> {
        if indices.is_empty() {
            return ndarray::Array1::zeros(30);
        }

        let mut sum = ndarray::Array1::zeros(30);
        for &idx in indices {
            sum = sum + &feature_vectors[idx];
        }

        sum / indices.len() as f64
    }

    /// Compute transition matrix from phrase sequence
    fn compute_transition_matrix(
        &self,
        sequence: &[String],
    ) -> HashMap<String, HashMap<String, usize>> {
        let mut matrix: HashMap<String, HashMap<String, usize>> = HashMap::new();

        for window in sequence.windows(2) {
            let from = &window[0];
            let to = &window[1];

            *matrix.entry(from.clone())
                .or_insert_with(HashMap::new)
                .entry(to.clone())
                .or_insert(0) += 1;
        }

        matrix
    }

    /// Compute within-type and between-type similarity statistics
    fn compute_similarity_stats(
        &self,
        phrase_types: &[DiscoveredPhraseType],
    ) -> (f64, f64) {
        let mut within_similarities = Vec::new();
        let mut between_distances = Vec::new();

        // Within-type: average similarity of instances to centroid
        for pt in phrase_types {
            for instance in &pt.instances {
                within_similarities.push(1.0 - instance.distance_to_centroid);
            }
        }

        // Between-type: average distance between centroids
        for i in 0..phrase_types.len() {
            for j in (i + 1)..phrase_types.len() {
                let a = ndarray::Array1::from_vec(
                    phrase_types[i].centroid_features.to_vector().to_vec()
                );
                let b = ndarray::Array1::from_vec(
                    phrase_types[j].centroid_features.to_vector().to_vec()
                );
                between_distances.push(self.similarity_engine.distance(&a, &b));
            }
        }

        let avg_within = if !within_similarities.is_empty() {
            within_similarities.iter().sum::<f64>() / within_similarities.len() as f64
        } else {
            0.0
        };

        let avg_between = if !between_distances.is_empty() {
            between_distances.iter().sum::<f64>() / between_distances.len() as f64
        } else {
            0.0
        };

        (avg_within, avg_between)
    }

    /// Analyze phrase motifs (recurring patterns)
    pub fn find_motifs(
        &self,
        result: &WithinCallAnalysisResult,
        min_length: usize,
        min_occurrences: usize,
    ) -> Vec<PhraseMotif> {
        let sequence = &result.phrase_sequence;
        if sequence.len() < min_length {
            return Vec::new();
        }

        let mut motif_counts: HashMap<Vec<String>, usize> = HashMap::new();
        let mut motif_positions: HashMap<Vec<String>, Vec<usize>> = HashMap::new();

        // Find all subsequences
        for length in min_length..=sequence.len().min(5) {
            for start in 0..=(sequence.len() - length) {
                let subseq: Vec<String> = sequence[start..start + length].to_vec();
                *motif_counts.entry(subseq.clone()).or_insert(0) += 1;
                motif_positions.entry(subseq).or_insert_with(Vec::new).push(start);
            }
        }

        // Filter and create motifs
        motif_counts.into_iter()
            .filter(|(_, count)| *count >= min_occurrences)
            .map(|(pattern, count)| {
                PhraseMotif {
                    pattern: pattern.clone(),
                    occurrence_count: count,
                    positions: motif_positions.get(&pattern).cloned().unwrap_or_default(),
                    length: pattern.len(),
                }
            })
            .collect()
    }
}

impl Default for WithinCallAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Phrase Motif
// =============================================================================

/// A recurring pattern of phrases within calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseMotif {
    /// Sequence of phrase type IDs
    pub pattern: Vec<String>,

    /// Number of occurrences
    pub occurrence_count: usize,

    /// Starting positions in the call
    pub positions: Vec<usize>,

    /// Pattern length
    pub length: usize,
}

// =============================================================================
// Integration with Zoo Vox Library Builder
// =============================================================================

/// Enhanced library builder that uses acoustic similarity for phrase typing
pub struct SimilarityBasedLibraryBuilder {
    config: WithinCallConfig,
}

impl SimilarityBasedLibraryBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: WithinCallConfig::default(),
        }
    }

    /// Create for specific species
    pub fn for_species(species: &str) -> Self {
        Self {
            config: WithinCallConfig::for_species(species),
        }
    }

    /// Build library using acoustic similarity for phrase typing
    pub fn build_library(
        &self,
        phrases: Vec<PhrasePrototype>,
        species: &str,
    ) -> Result<crate::zoo_vox_data_models::SpeciesPhraseLibrary, crate::zoo_vox_library::LibraryError> {
        let mut analyzer = WithinCallAnalyzer::for_species(species);

        // Use call_id from first phrase or generate one
        let call_id = phrases.first()
            .map(|p| p.source_file.clone().unwrap_or_else(|| "unknown_call".to_string()))
            .unwrap_or_else(|| "empty_call".to_string());

        // Discover phrase types using similarity
        let result = analyzer.discover_phrases(phrases.clone(), &call_id, species);

        // Convert discovered types back to PhrasePrototypes
        let typed_phrases: Vec<PhrasePrototype> = result.phrase_types.into_iter()
            .map(|pt| {
                // Create merged phrase prototype
                let mut phrase = PhrasePrototype::new(
                    pt.type_id.clone(),
                    pt.phrase_key,
                    species.to_string(),
                );

                // Use centroid features
                phrase.features_30d = pt.centroid_features;
                phrase.occurrence_count = pt.occurrence_count as u32;

                // Merge contexts
                if let Some(ctx) = pt.contexts.first() {
                    phrase.primary_context = Some(ctx.clone());
                }

                phrase
            })
            .collect();

        // Use standard builder for rest of processing
        let builder = crate::zoo_vox_library::ZooVoxLibraryBuilder::new()
            .with_similarity_threshold(self.config.similarity_threshold);

        builder.build_library(typed_phrases, species, None)
    }
}

impl Default for SimilarityBasedLibraryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Species-Specific Weight Helper
// =============================================================================

/// Get species-specific 30D feature weights for within-species analysis
///
/// This implements Phase 2 of the "Router then Analyzer" pattern:
/// - These weights are for WITHIN-species phrase discrimination
/// - They should NOT be used for cross-species comparison (Phase 1)
///
/// The 30D features cover:
/// - D0-D4: Spectral (centroid, spread, skewness, kurtosis, tilt)
/// - D5-D9: Harmonic (f0, harmonicity, harmonic_ratio, inharmonicity, noise_ratio)
/// - D10-D14: Temporal (rms, zcr, attack, decay, sustain)
/// - D15-D19: Modulation (am_rate, am_depth, fm_rate, fm_slope, spectral_flux)
/// - D20-D24: Cepstral (mfcc_1-5)
/// - D25-D29: Formant (f1, f2, f3, b1, b2)
fn get_species_weights_30d(species: &str) -> Vec<f32> {
    let weights = match species {
        "dolphin" | "orca" => FeatureWeights::dolphin(),
        "sperm_whale" => FeatureWeights::sperm_whale(),
        "zebra_finch" => FeatureWeights::zebra_finch(),
        "marmoset" => FeatureWeights::marmoset(),
        "egyptian_bat" => FeatureWeights::bat(),
        "macaque" => FeatureWeights::macaque(),
        "meerkat" => FeatureWeights::meerkat(),
        _ => FeatureWeights::default(),
    };

    weights.to_weight_vector_30d()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_phrases() -> Vec<PhrasePrototype> {
        let mut phrases = Vec::new();

        // Create two groups of similar phrases
        for i in 0..3 {
            let mut p = PhrasePrototype::new(
                format!("phrase_{}", i),
                format!("F0_6800_DUR_{}", 60 + i * 5),
                "marmoset",
            );
            p.features_30d = AcousticFeatures30D {
                mean_f0_hz: 6800.0 + i as f64 * 50.0,
                duration_ms: 60.0 + i as f64 * 5.0,
                ..Default::default()
            };
            p.typical_position = i as u32;
            phrases.push(p);
        }

        for i in 3..5 {
            let mut p = PhrasePrototype::new(
                format!("phrase_{}", i),
                format!("F0_8500_DUR_{}", 40 + (i - 3) * 5),
                "marmoset",
            );
            p.features_30d = AcousticFeatures30D {
                mean_f0_hz: 8500.0 + (i - 3) as f64 * 50.0,
                duration_ms: 40.0 + (i - 3) as f64 * 5.0,
                ..Default::default()
            };
            p.typical_position = i as u32;
            phrases.push(p);
        }

        phrases
    }

    #[test]
    fn test_within_call_config_default() {
        let config = WithinCallConfig::default();
        assert!((config.similarity_threshold - 0.85).abs() < 1e-10);
        assert!((config.distance_threshold - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_within_call_config_species() {
        let config = WithinCallConfig::for_species("sperm_whale");
        assert!(config.similarity_threshold > 0.85); // Stricter for codas
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = WithinCallAnalyzer::new();
        assert_eq!(analyzer.config.similarity_threshold, 0.85);
    }

    #[test]
    fn test_analyzer_for_species() {
        let analyzer = WithinCallAnalyzer::for_species("dolphin");
        assert!(analyzer.config.similarity_threshold <= 0.85); // Looser for FM
    }

    #[test]
    fn test_discover_phrases_empty() {
        let mut analyzer = WithinCallAnalyzer::new();
        let result = analyzer.discover_phrases(Vec::new(), "test", "marmoset");

        assert_eq!(result.total_phrases, 0);
        assert_eq!(result.unique_types, 0);
    }

    #[test]
    fn test_discover_phrases_basic() {
        let mut analyzer = WithinCallAnalyzer::new();
        let phrases = create_test_phrases();
        let result = analyzer.discover_phrases(phrases, "test_call", "marmoset");

        assert_eq!(result.total_phrases, 5);
        assert!(result.unique_types >= 1);
        assert!(result.type_entropy >= 0.0);
    }

    #[test]
    fn test_transition_matrix() {
        let mut analyzer = WithinCallAnalyzer::new();
        let phrases = create_test_phrases();
        let result = analyzer.discover_phrases(phrases, "test", "marmoset");

        // Should have some transitions
        assert!(!result.transition_matrix.is_empty() || result.phrase_sequence.len() < 2);
    }

    #[test]
    fn test_find_motifs() {
        let analyzer = WithinCallAnalyzer::new();
        let result = WithinCallAnalysisResult {
            call_id: "test".to_string(),
            species: "marmoset".to_string(),
            phrase_types: Vec::new(),
            phrase_sequence: vec!["A".to_string(), "B".to_string(), "A".to_string(), "B".to_string()],
            transition_matrix: HashMap::new(),
            total_phrases: 4,
            unique_types: 2,
            type_entropy: 1.0,
            avg_within_type_similarity: 0.9,
            avg_between_type_distance: 0.5,
        };

        let motifs = analyzer.find_motifs(&result, 2, 2);

        // Should find AB pattern occurring twice
        assert!(!motifs.is_empty());
    }

    #[test]
    fn test_similarity_based_library_builder() {
        let builder = SimilarityBasedLibraryBuilder::for_species("marmoset");
        let phrases = create_test_phrases();
        let library = builder.build_library(phrases, "marmoset");

        assert!(library.is_ok());
        let lib = library.unwrap();
        assert!(lib.total_phrases > 0);
    }
}
