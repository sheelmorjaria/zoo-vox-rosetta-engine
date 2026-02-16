// =============================================================================
// Species Configuration Module - Species-Specific Adaptation Layer
// =============================================================================
//
// Provides species-specific configurations for the Zoo Vox Rosetta Engine.
// Each species has different encoding strategies, modalities, and required modules.

use serde::{Deserialize, Serialize};

/// Encoding strategy for context decoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingStrategy {
    /// Combinatorial syntax (zebra finch, orcas)
    Combinatorial,

    /// Quantitative encoding by count (meerkats)
    Quantitative,

    /// Coda-type encoding (sperm whales)
    CodaType,

    /// Frequency-modulated contours (dolphins)
    FrequencyModulated,

    /// Duration-mediated encoding (bats)
    DurationMediated,

    /// Phrase type selection (marmosets)
    PhraseType,

    /// Minimal encoding (macaques, giant otters)
    Minimal,
}

impl std::fmt::Display for EncodingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodingStrategy::Combinatorial => write!(f, "Combinatorial"),
            EncodingStrategy::Quantitative => write!(f, "Quantitative"),
            EncodingStrategy::CodaType => write!(f, "Coda-Type"),
            EncodingStrategy::FrequencyModulated => write!(f, "Frequency-Modulated"),
            EncodingStrategy::DurationMediated => write!(f, "Duration-Mediated"),
            EncodingStrategy::PhraseType => write!(f, "Phrase-Type"),
            EncodingStrategy::Minimal => write!(f, "Minimal"),
        }
    }
}

/// Primary analysis modality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisModality {
    /// Temporal phrase analysis
    Temporal,

    /// Spectral frequency analysis
    Spectral,

    /// Hybrid temporal + spectral
    Hybrid,
}

impl std::fmt::Display for AnalysisModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnalysisModality::Temporal => write!(f, "Temporal"),
            AnalysisModality::Spectral => write!(f, "Spectral"),
            AnalysisModality::Hybrid => write!(f, "Hybrid"),
        }
    }
}

/// Required analysis module
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnalysisModule {
    /// Temporal phrase analysis
    Temporal,

    /// Spectral frequency analysis
    Spectral,

    /// N-gram sequence analysis
    Sequence,

    /// Duration-based analysis
    Duration,

    /// Phrase count analysis
    Count,
}

/// Feature extraction parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureParams {
    /// Minimum phrase duration in ms
    pub phrase_min_ms: f64,

    /// Maximum phrase duration in ms
    pub phrase_max_ms: f64,

    /// Similarity threshold for phrase typing
    pub similarity_threshold: f64,

    /// Feature dimension
    pub feature_dim: usize,
}

impl Default for FeatureParams {
    fn default() -> Self {
        Self {
            phrase_min_ms: 30.0,
            phrase_max_ms: 500.0,
            similarity_threshold: 0.75,
            feature_dim: 30,
        }
    }
}

/// Decoding method for context prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecodingMethod {
    /// Context = which phrase type selected
    PhraseTypeSelection,

    /// Context = number of phrases
    PhraseCount,

    /// Context = duration threshold
    DurationThreshold,

    /// Context = frequency contour shape
    ContourShape,

    /// Context = phrase sequence pattern
    SequencePattern,
}

/// Context decoding rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextRules {
    /// Decoding method
    pub decoding_method: DecodingMethod,

    /// Context labels
    pub context_labels: Vec<String>,
}

impl Default for ContextRules {
    fn default() -> Self {
        Self {
            decoding_method: DecodingMethod::PhraseTypeSelection,
            context_labels: Vec::new(),
        }
    }
}

/// Species-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesConfig {
    /// Species name
    species: String,

    /// Encoding strategy
    encoding_strategy: EncodingStrategy,

    /// Primary modality
    modality: AnalysisModality,

    /// Required analysis modules
    required_modules: Vec<AnalysisModule>,

    /// Feature extraction parameters
    feature_params: FeatureParams,

    /// Context decoding rules
    context_rules: ContextRules,
}

impl SpeciesConfig {
    /// Get species name
    pub fn species(&self) -> &str {
        &self.species
    }

    /// Get encoding strategy
    pub fn encoding_strategy(&self) -> EncodingStrategy {
        self.encoding_strategy
    }

    /// Get modality
    pub fn modality(&self) -> AnalysisModality {
        self.modality
    }

    /// Get required modules
    pub fn required_modules(&self) -> &[AnalysisModule] {
        &self.required_modules
    }

    /// Get feature parameters
    pub fn feature_params(&self) -> &FeatureParams {
        &self.feature_params
    }

    /// Get context labels
    pub fn context_labels(&self) -> &[String] {
        &self.context_rules.context_labels
    }

    /// Get decoding method
    pub fn decoding_method(&self) -> &DecodingMethod {
        &self.context_rules.decoding_method
    }

    /// Check if a module is required
    pub fn requires_module(&self, module: AnalysisModule) -> bool {
        self.required_modules.contains(&module)
    }
}

/// Factory for creating species-specific configurations
pub struct SpeciesConfigFactory;

impl SpeciesConfigFactory {
    /// Create configuration for a species
    pub fn create(species: &str) -> SpeciesConfig {
        match species.to_lowercase().as_str() {
            "sperm_whale" | "dominica" | "spermwhale" => Self::sperm_whale_config(),
            "meerkat" | "meerkats" => Self::meerkat_config(),
            "zebra_finch" | "zebrafinch" | "finch" => Self::zebra_finch_config(),
            "dolphin" | "dolphins" | "whistle_signals" | "bottlenose" => Self::dolphin_config(),
            "bat" | "egyptian_bat" | "egyptianbat" | "fruit_bat" => Self::bat_config(),
            "orca" | "orcas" | "killer_whale" => Self::orca_config(),
            "marmoset" | "marmosets" | "common_marmoset" => Self::marmoset_config(),
            "macaque" | "macaques" => Self::macaque_config(),
            "giant_otter" | "giantotter" | "otter" => Self::giant_otter_config(),
            _ => Self::default_config(),
        }
    }

    /// Sperm whale configuration
    fn sperm_whale_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Sperm Whale".to_string(),
            encoding_strategy: EncodingStrategy::CodaType,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal],
            feature_params: FeatureParams {
                phrase_min_ms: 10.0,
                phrase_max_ms: 100.0,
                similarity_threshold: 0.80,
                feature_dim: 30,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::PhraseTypeSelection,
                context_labels: vec![
                    "foraging".to_string(),
                    "social".to_string(),
                    "communication".to_string(),
                ],
            },
        }
    }

    /// Meerkat configuration
    fn meerkat_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Meerkat".to_string(),
            encoding_strategy: EncodingStrategy::Quantitative,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal, AnalysisModule::Count],
            feature_params: FeatureParams {
                phrase_min_ms: 30.0,
                phrase_max_ms: 500.0,
                similarity_threshold: 0.75,
                feature_dim: 30,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::PhraseCount,
                context_labels: vec![
                    "close_call".to_string(),
                    "alarm".to_string(),
                    "social".to_string(),
                    "sentinel".to_string(),
                ],
            },
        }
    }

    /// Zebra finch configuration
    fn zebra_finch_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Zebra Finch".to_string(),
            encoding_strategy: EncodingStrategy::Combinatorial,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal, AnalysisModule::Sequence],
            feature_params: FeatureParams {
                phrase_min_ms: 20.0,
                phrase_max_ms: 200.0,
                similarity_threshold: 0.75,
                feature_dim: 30,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::SequencePattern,
                context_labels: vec!["directed".to_string(), "undirected".to_string()],
            },
        }
    }

    /// Dolphin configuration
    fn dolphin_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Dolphin".to_string(),
            encoding_strategy: EncodingStrategy::FrequencyModulated,
            modality: AnalysisModality::Spectral,
            required_modules: vec![AnalysisModule::Spectral],
            feature_params: FeatureParams {
                phrase_min_ms: 500.0,
                phrase_max_ms: 2000.0,
                similarity_threshold: 0.70,
                feature_dim: 56,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::ContourShape,
                context_labels: vec![
                    "signature".to_string(),
                    "social".to_string(),
                    "food".to_string(),
                    "alarm".to_string(),
                ],
            },
        }
    }

    /// Egyptian fruit bat configuration
    fn bat_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Egyptian Fruit Bat".to_string(),
            encoding_strategy: EncodingStrategy::DurationMediated,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal, AnalysisModule::Duration],
            feature_params: FeatureParams {
                phrase_min_ms: 30.0,
                phrase_max_ms: 500.0,
                similarity_threshold: 0.75,
                feature_dim: 30,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::DurationThreshold,
                context_labels: vec![
                    "feeding".to_string(),
                    "mating".to_string(),
                    "landing".to_string(),
                ],
            },
        }
    }

    /// Orca configuration
    fn orca_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Orca".to_string(),
            encoding_strategy: EncodingStrategy::Combinatorial,
            modality: AnalysisModality::Hybrid,
            required_modules: vec![
                AnalysisModule::Temporal,
                AnalysisModule::Sequence,
                AnalysisModule::Spectral,
            ],
            feature_params: FeatureParams {
                phrase_min_ms: 50.0,
                phrase_max_ms: 1000.0,
                similarity_threshold: 0.75,
                feature_dim: 56,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::SequencePattern,
                context_labels: vec![
                    "hunting".to_string(),
                    "social".to_string(),
                    "travel".to_string(),
                ],
            },
        }
    }

    /// Marmoset configuration
    fn marmoset_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Common Marmoset".to_string(),
            encoding_strategy: EncodingStrategy::PhraseType,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal],
            feature_params: FeatureParams {
                phrase_min_ms: 50.0,
                phrase_max_ms: 500.0,
                similarity_threshold: 0.75,
                feature_dim: 30,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::PhraseTypeSelection,
                context_labels: vec!["phee".to_string(), "tsik".to_string(), "trill".to_string()],
            },
        }
    }

    /// Macaque configuration
    fn macaque_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Macaque".to_string(),
            encoding_strategy: EncodingStrategy::Minimal,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal, AnalysisModule::Spectral],
            feature_params: FeatureParams {
                phrase_min_ms: 100.0,
                phrase_max_ms: 500.0,
                similarity_threshold: 0.90,
                feature_dim: 56,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::PhraseTypeSelection,
                context_labels: vec!["coo".to_string()],
            },
        }
    }

    /// Giant otter configuration
    fn giant_otter_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Giant Otter".to_string(),
            encoding_strategy: EncodingStrategy::Minimal,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal, AnalysisModule::Spectral],
            feature_params: FeatureParams {
                phrase_min_ms: 50.0,
                phrase_max_ms: 500.0,
                similarity_threshold: 0.80,
                feature_dim: 56,
            },
            context_rules: ContextRules {
                decoding_method: DecodingMethod::PhraseTypeSelection,
                context_labels: vec!["contact".to_string(), "alarm".to_string()],
            },
        }
    }

    /// Default configuration for unknown species
    fn default_config() -> SpeciesConfig {
        SpeciesConfig {
            species: "Unknown".to_string(),
            encoding_strategy: EncodingStrategy::PhraseType,
            modality: AnalysisModality::Temporal,
            required_modules: vec![AnalysisModule::Temporal],
            feature_params: FeatureParams::default(),
            context_rules: ContextRules::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sperm_whale_config() {
        let config = SpeciesConfigFactory::create("sperm_whale");

        assert_eq!(config.species(), "Sperm Whale");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::CodaType);
        assert_eq!(config.modality(), AnalysisModality::Temporal);
        assert!(config.requires_module(AnalysisModule::Temporal));
    }

    #[test]
    fn test_dolphin_config() {
        let config = SpeciesConfigFactory::create("dolphin");

        assert_eq!(config.species(), "Dolphin");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::FrequencyModulated);
        assert_eq!(config.modality(), AnalysisModality::Spectral);
        assert!(config.requires_module(AnalysisModule::Spectral));
    }

    #[test]
    fn test_zebra_finch_config() {
        let config = SpeciesConfigFactory::create("zebra_finch");

        assert_eq!(config.species(), "Zebra Finch");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::Combinatorial);
        assert!(config.requires_module(AnalysisModule::Sequence));
    }

    #[test]
    fn test_case_insensitive() {
        let config1 = SpeciesConfigFactory::create("SPERM_WHALE");
        let config2 = SpeciesConfigFactory::create("sperm_whale");

        assert_eq!(config1.species(), config2.species());
    }

    #[test]
    fn test_unknown_species_defaults() {
        let config = SpeciesConfigFactory::create("unknown_species_xyz");

        assert_eq!(config.species(), "Unknown");
        assert_eq!(config.encoding_strategy(), EncodingStrategy::PhraseType);
    }
}
