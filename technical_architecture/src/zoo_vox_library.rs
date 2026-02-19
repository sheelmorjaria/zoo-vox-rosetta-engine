//! Phrase Library Builder for Zoo Vox Rosetta Engine 2.0
//!
//! Builds and manages phrase libraries from extracted phrases,
//! handling phrase typing, deduplication, and context association.

use chrono::Utc;
use std::collections::HashMap;

use crate::species::SpeciesConfigFactory;
use crate::zoo_vox_data_models::{
    AcousticFeatures30D, ContextAssociation, CrossSpeciesPhraseDatabase, PhrasePrototype,
    SpeciesPhraseLibrary,
};

/// Zoo Vox Rosetta library error type
#[derive(Debug)]
pub enum LibraryError {
    /// Build error
    BuildError(String),
    /// IO error
    IoError(String),
    /// Serialization error
    SerializationError(String),
}

impl std::fmt::Display for LibraryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LibraryError::BuildError(msg) => write!(f, "Build error: {}", msg),
            LibraryError::IoError(msg) => write!(f, "IO error: {}", msg),
            LibraryError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for LibraryError {}

/// Builder for creating phrase libraries
pub struct ZooVoxLibraryBuilder {
    similarity_threshold: f64,
}

impl ZooVoxLibraryBuilder {
    /// Create new library builder
    pub fn new() -> Self {
        Self {
            similarity_threshold: 0.85,
        }
    }

    /// Set similarity threshold for phrase typing
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Build a species phrase library from extracted phrases
    pub fn build_library(
        &self,
        phrases: Vec<PhrasePrototype>,
        species: &str,
        dataset_info: Option<HashMap<String, String>>,
    ) -> Result<SpeciesPhraseLibrary, LibraryError> {
        let config = SpeciesConfigFactory::create(species);

        // Type and deduplicate phrases
        let typed_phrases = self.type_phrases(phrases);

        // Calculate statistics
        let total_occurrences: u64 = typed_phrases
            .iter()
            .map(|p| p.occurrence_count as u64)
            .sum();

        // Calculate entropy
        let type_entropy = if total_occurrences > 0 {
            let total = total_occurrences as f64;
            typed_phrases
                .iter()
                .map(|p| {
                    let p_prob = p.occurrence_count as f64 / total;
                    if p_prob > 0.0 {
                        -p_prob * p_prob.log2()
                    } else {
                        0.0
                    }
                })
                .sum()
        } else {
            0.0
        };

        // Get frequency and duration ranges
        let f0_values: Vec<f64> = typed_phrases
            .iter()
            .filter(|p| p.features_30d.mean_f0_hz > 0.0)
            .map(|p| p.features_30d.mean_f0_hz)
            .collect();

        let frequency_range_hz = if !f0_values.is_empty() {
            let min = f0_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = f0_values.iter().fold(0.0_f64, |a, &b| a.max(b));
            (min, max)
        } else {
            (0.0, 0.0)
        };

        let dur_values: Vec<f64> = typed_phrases
            .iter()
            .map(|p| p.features_30d.duration_ms)
            .collect();

        let typical_duration_ms = if !dur_values.is_empty() {
            let min = dur_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = dur_values.iter().fold(0.0_f64, |a, &b| a.max(b));
            (min, max)
        } else {
            (0.0, 0.0)
        };

        // Collect context labels
        let context_labels: Vec<String> = typed_phrases
            .iter()
            .filter_map(|p| p.primary_context.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Build library
        let mut library = SpeciesPhraseLibrary {
            species: species.to_string(),
            encoding_strategy: config.encoding_strategy(),
            encoding_modality: config.modality(),
            total_phrases: typed_phrases.len(),
            total_occurrences,
            type_entropy,
            phrases_per_file_avg: 1.0,
            phrases: typed_phrases,
            context_labels,
            frequency_range_hz,
            typical_duration_ms,
            dataset_info: dataset_info.unwrap_or_default(),
            extraction_timestamp: Utc::now(),
        };

        library.recalculate_statistics();

        Ok(library)
    }

    /// Type phrases by similarity and merge duplicates
    fn type_phrases(&self, phrases: Vec<PhrasePrototype>) -> Vec<PhrasePrototype> {
        if phrases.is_empty() {
            return Vec::new();
        }

        // Group by phrase_key first
        let mut key_groups: HashMap<String, Vec<PhrasePrototype>> = HashMap::new();
        for phrase in phrases {
            key_groups
                .entry(phrase.phrase_key.clone())
                .or_default()
                .push(phrase);
        }

        // Merge phrases with same key
        let mut typed_phrases = Vec::new();
        for (_key, group) in key_groups {
            if group.len() == 1 {
                typed_phrases.push(group.into_iter().next().unwrap());
            } else {
                // Merge: average features, sum occurrences
                let merged = self.merge_phrase_group(&group);
                typed_phrases.push(merged);
            }
        }

        // Sort by occurrence count descending
        typed_phrases.sort_by(|a, b| b.occurrence_count.cmp(&a.occurrence_count));

        typed_phrases
    }

    /// Merge multiple phrases of the same type
    fn merge_phrase_group(&self, phrases: &[PhrasePrototype]) -> PhrasePrototype {
        // Average the 30D features
        let mut feature_vec = [0.0; 30];
        for phrase in phrases {
            let vec = phrase.features_30d.to_vector();
            for i in 0..30 {
                feature_vec[i] += vec[i];
            }
        }
        for i in 0..30 {
            feature_vec[i] /= phrases.len() as f64;
        }
        let mean_features = AcousticFeatures30D::from_vector(feature_vec);

        // Merge contexts
        let mut context_counts: HashMap<String, u32> = HashMap::new();
        for phrase in phrases {
            for ctx in &phrase.contexts {
                *context_counts.entry(ctx.context_label.clone()).or_insert(0) +=
                    ctx.occurrence_count;
            }
        }

        let merged_contexts: Vec<ContextAssociation> = context_counts
            .iter()
            .map(|(label, &count)| ContextAssociation {
                context_label: label.clone(),
                context_category: "merged".to_string(),
                occurrence_count: count,
                ..Default::default()
            })
            .collect();

        // Find primary context (most occurrences)
        let primary_context = context_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(label, _)| label.clone());

        // Average typical position
        let typical_position = phrases
            .iter()
            .map(|p| p.typical_position as f64)
            .sum::<f64>()
            / phrases.len() as f64;

        // Collect co-occurring phrases
        let co_occurring: Vec<String> = phrases
            .iter()
            .flat_map(|p| p.co_occurring_phrases.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Use first phrase as template
        let template = &phrases[0];

        PhrasePrototype {
            phrase_id: template.phrase_id.clone(),
            phrase_key: template.phrase_key.clone(),
            species: template.species.clone(),
            source_file: Some("merged".to_string()),
            source_dataset: template.source_dataset.clone(),
            encoding_strategy: template.encoding_strategy,
            encoding_modality: template.encoding_modality,
            phrase_type: template.phrase_type.clone(),
            features_30d: mean_features,
            contexts: merged_contexts,
            primary_context,
            typical_position: typical_position.round() as u32,
            co_occurring_phrases: co_occurring,
            occurrence_count: phrases.iter().map(|p| p.occurrence_count).sum(),
            entropy_contribution: phrases.iter().map(|p| p.entropy_contribution).sum(),
            signal_to_noise_ratio: phrases.iter().map(|p| p.signal_to_noise_ratio).sum::<f64>()
                / phrases.len() as f64,
            extraction_confidence: phrases.iter().map(|p| p.extraction_confidence).sum::<f64>()
                / phrases.len() as f64,
            created_at: Utc::now(),
            notes: None,
        }
    }
}

impl Default for ZooVoxLibraryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Create sample phrase libraries for all 10 species
pub fn create_sample_libraries() -> Result<CrossSpeciesPhraseDatabase, LibraryError> {
    let mut database = CrossSpeciesPhraseDatabase::new();
    let builder = ZooVoxLibraryBuilder::new();

    let species_list = [
        "sperm_whale",
        "zebra_finch",
        "meerkat",
        "dolphin",
        "orca",
        "egyptian_bat",
        "marmoset",
        "giant_otter",
        "macaque",
    ];

    for species in species_list {
        let phrases = create_sample_phrases_for_species(species);
        let library = builder.build_library(phrases, species, None)?;
        database.add_library(library);
    }

    Ok(database)
}

/// Create sample phrases for a species
fn create_sample_phrases_for_species(species: &str) -> Vec<PhrasePrototype> {
    use rand::Rng;

    let config = SpeciesConfigFactory::create(species);
    let params = config.feature_params();

    let mut rng = rand::thread_rng();
    let mut phrases = Vec::new();

    let contexts = config.context_labels();

    let freq_min = 1000.0;
    let freq_max = 15000.0;
    let dur_min = params.phrase_min_ms;
    let dur_max = params.phrase_max_ms;

    // Create 3-5 sample phrase types
    let n_types = rng.gen_range(3..=5);

    for i in 0..n_types {
        // Generate representative features
        let mean_f0 = freq_min + (freq_max - freq_min) * (i as f64 / n_types as f64);
        let duration = dur_min + (dur_max - dur_min) * rng.gen_range(0.3..0.7);

        let features = AcousticFeatures30D {
            mean_f0_hz: mean_f0.max(100.0),
            duration_ms: duration.max(10.0),
            f0_range_hz: rng.gen_range(50.0..300.0),
            harmonic_to_noise_ratio: rng.gen_range(10.0..25.0),
            spectral_flatness: rng.gen_range(0.1..0.4),
            harmonicity: rng.gen_range(0.5..0.9),
            attack_time_ms: rng.gen_range(5.0..20.0),
            decay_time_ms: rng.gen_range(10.0..30.0),
            sustain_level: rng.gen_range(0.5..0.8),
            jitter: rng.gen_range(0.01..0.05),
            shimmer: rng.gen_range(0.01..0.05),
            ..Default::default()
        };

        // Associate with context
        let context_label = &contexts[i % contexts.len()];
        let context = ContextAssociation::new(context_label, "sample");

        let phrase_key = format!("F0_{:.0}_DUR_{:.0}", mean_f0, duration);

        let mut phrase = PhrasePrototype::new(
            format!("{}_{}_sample", species.replace(" ", "_").to_lowercase(), i),
            phrase_key,
            species.to_string(),
        );

        phrase.encoding_strategy = config.encoding_strategy();
        phrase.encoding_modality = config.modality();
        phrase.phrase_type = Some(format!("type_{}", i));
        phrase.features_30d = features;
        phrase.contexts = vec![context];
        phrase.primary_context = Some(context_label.clone());
        phrase.occurrence_count = rng.gen_range(10..200);

        phrases.push(phrase);
    }

    phrases
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_builder_creation() {
        let builder = ZooVoxLibraryBuilder::new();
        assert_eq!(builder.similarity_threshold, 0.85);
    }

    #[test]
    fn test_build_empty_library() {
        let builder = ZooVoxLibraryBuilder::new();
        let library = builder.build_library(Vec::new(), "marmoset", None).unwrap();

        assert_eq!(library.species, "marmoset");
        assert_eq!(library.total_phrases, 0);
    }

    #[test]
    fn test_build_library_with_phrases() {
        let builder = ZooVoxLibraryBuilder::new();

        let mut phrase = PhrasePrototype::new("marmoset_001", "F0_6800_DUR_65", "marmoset");
        phrase.occurrence_count = 100;
        phrase.primary_context = Some("contact".to_string());

        let library = builder
            .build_library(vec![phrase], "marmoset", None)
            .unwrap();

        assert_eq!(library.total_phrases, 1);
        assert_eq!(library.total_occurrences, 100);
        assert!(library.context_labels.contains(&"contact".to_string()));
    }

    #[test]
    fn test_merge_phrase_group() {
        let builder = ZooVoxLibraryBuilder::new();

        let mut p1 = PhrasePrototype::new("marmoset_001", "F0_6800_DUR_65", "marmoset");
        p1.occurrence_count = 1;
        let mut p2 = PhrasePrototype::new("marmoset_002", "F0_6800_DUR_65", "marmoset");
        p2.occurrence_count = 1;

        let merged = builder.merge_phrase_group(&[p1, p2]);

        assert_eq!(merged.occurrence_count, 2);
    }

    #[test]
    fn test_create_sample_libraries() {
        let database = create_sample_libraries().unwrap();

        assert!(database.species_libraries.len() >= 9);
        assert!(database.total_phrases() > 0);
    }
}
