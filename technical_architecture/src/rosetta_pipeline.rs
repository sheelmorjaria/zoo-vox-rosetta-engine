//! Rosetta Pipeline - Integrated Zoo Vox Rosetta Engine
//!
//! This module implements the complete Zoo Vox Rosetta pipeline:
//!
//! **Phase 1: Global Species Identification**
//! - Uses unified weights for cross-species comparison
//! - Dynamic segmentation and 45D feature extraction
//!
//! **Phase 2a: Semantic Grounding (Human-Guided)**
//! - Matches phrases against pre-seeded semantic dictionaries
//! - Maps acoustic types to semantic labels (e.g., "Type_1" -> "Alarm")
//!
//! **Phase 2b: Contextual Enrichment (Environmental/Syntax)**
//! - Refines interpretation based on sensor data and syntax analysis
//!
//! **Phase 3: Bundle Serialization**
//! - Packages dictionaries, weights, and models into deployable artifact

use crate::acoustic_similarity::{AcousticSimilarityEngine, DistanceMetric};
use crate::species::FeatureWeights;
use crate::zoo_vox_features::ZooVoxFeatureExtractor;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

// =============================================================================
// Constants
// =============================================================================

pub const FEATURE_DIM: usize = 45;
pub const SIMILARITY_THRESHOLD: f64 = 0.85;

// =============================================================================
// Semantic Dictionary (Human-Anchored)
// =============================================================================

/// Semantic phrase dictionary mapping phrase types to label probabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticPhraseDictionary {
    /// Species this dictionary belongs to
    pub species: String,

    /// Mapping from phrase type ID (e.g., "Type_1") to label probabilities
    pub type_to_labels: HashMap<String, HashMap<String, f32>>,

    /// Centroids for each phrase type (for matching)
    pub type_centroids: HashMap<String, Vec<f32>>,

    /// Total number of phrases used to build this dictionary
    pub total_phrases: usize,

    /// Number of distinct phrase types
    pub num_types: usize,
}

impl SemanticPhraseDictionary {
    /// Load dictionary from JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let dict: Self = serde_json::from_reader(reader)?;
        Ok(dict)
    }

    /// Save dictionary to JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Look up the most likely label for a phrase type
    pub fn get_primary_label(&self, type_id: &str) -> Option<(&String, f32)> {
        self.type_to_labels
            .get(type_id)
            .and_then(|labels| {
                labels
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            })
            .map(|(label, prob)| (label, *prob))
    }

    /// Find best matching type for a feature vector
    pub fn match_features(
        &self,
        features: &[f32],
        engine: &AcousticSimilarityEngine,
    ) -> Option<(String, f64, String, f32)> {
        let query: Array1<f64> = Array1::from_vec(features.iter().map(|&f| f as f64).collect());

        let mut best_match = None;
        let mut best_sim = 0.0;

        for (type_id, centroid) in &self.type_centroids {
            let proto: Array1<f64> = Array1::from_vec(centroid.iter().map(|&f| f as f64).collect());
            let dist = engine.distance(&query, &proto);
            let sim = 1.0 - dist;

            if sim > SIMILARITY_THRESHOLD && sim > best_sim {
                best_sim = sim;
                if let Some((label, prob)) = self.get_primary_label(type_id) {
                    best_match = Some((type_id.clone(), sim, label.clone(), prob));
                }
            }
        }

        best_match
    }
}

// =============================================================================
// Context Enriched Phrase
// =============================================================================

/// Environmental state from sensors
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum EnvState {
    Quiet,
    Wind,
    Rain,
    Storm,
    #[default]
    Unknown,
}

/// Syntax role in a sequence
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum SyntaxRole {
    Initiator,
    Reply,
    Solo,
    Chorus,
    #[default]
    Unknown,
}

/// Complete context-enriched phrase output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEnrichedPhrase {
    // Layer 1: Acoustic Identity
    /// Unique phrase type identifier (e.g., "Type_1")
    pub phrase_type_id: String,

    /// Grading score (0.0 = discrete, 1.0 = graded)
    pub grading_score: f32,

    /// Acoustic match confidence
    pub acoustic_confidence: f64,

    // Layer 2: Semantic Identity (From Human Annotations)
    /// Semantic label (e.g., "Phee_Call")
    pub semantic_label: String,

    /// Label confidence from dictionary
    pub label_confidence: f32,

    // Layer 3: Pragmatic Context (From Environment/Syntax)
    /// Role in syntax sequence
    pub syntax_role: SyntaxRole,

    /// Current environmental state
    pub environmental_state: EnvState,

    /// Inferred intent (e.g., "Territorial_Defense")
    pub inferred_intent: String,

    // Layer 4: Timing
    /// Start time in milliseconds
    pub start_ms: f64,

    /// Duration in milliseconds
    pub duration_ms: f64,
}

// =============================================================================
// Rosetta Bundle (Deployable Artifact)
// =============================================================================

/// Complete deployable bundle for a species
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosettaBundle {
    /// Bundle version
    pub version: String,

    /// Species identifier
    pub species: String,

    /// Species-specific feature weights
    pub feature_weights: FeatureWeights,

    /// Semantic phrase dictionary
    pub semantic_dictionary: SemanticPhraseDictionary,

    /// Global weights for species identification
    pub global_weights: FeatureWeights,

    /// Creation timestamp
    pub created_at: String,
}

impl RosettaBundle {
    /// Create a new bundle for a species
    pub fn new(
        species: &str,
        feature_weights: FeatureWeights,
        semantic_dictionary: SemanticPhraseDictionary,
        global_weights: FeatureWeights,
    ) -> Self {
        Self {
            version: "1.0.0".to_string(),
            species: species.to_string(),
            feature_weights,
            semantic_dictionary,
            global_weights,
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Load bundle from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let bundle: Self = serde_json::from_reader(reader)?;
        Ok(bundle)
    }

    /// Save bundle to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Load bundle from binary (compressed)
    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut compressed = Vec::new();
        reader.read_to_end(&mut compressed)?;

        let decompressed = zstd::decode_all(&compressed[..])?;
        let bundle: Self = bincode::deserialize(&decompressed)?;
        Ok(bundle)
    }

    /// Save bundle to binary (compressed)
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = bincode::serialize(self)?;
        let compressed = zstd::encode_all(&serialized[..], 3)?;

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&compressed)?;
        Ok(())
    }
}

// =============================================================================
// Rosetta Pipeline
// =============================================================================

/// Configuration for the Rosetta Pipeline
#[derive(Debug, Clone)]
pub struct RosettaConfig {
    /// Sample rate for audio processing
    pub sample_rate: u32,

    /// Similarity threshold for phrase matching
    pub similarity_threshold: f64,

    /// Minimum segment duration in ms
    pub min_segment_ms: f64,
}

impl Default for RosettaConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            similarity_threshold: SIMILARITY_THRESHOLD,
            min_segment_ms: 50.0,
        }
    }
}

/// Complete Zoo Vox Rosetta Pipeline
pub struct RosettaPipeline {
    config: RosettaConfig,

    // Phase 1: Global identification
    global_engine: AcousticSimilarityEngine,

    // Phase 2: Species-specific
    species_bundles: HashMap<String, RosettaBundle>,
    species_engines: HashMap<String, AcousticSimilarityEngine>,
}

/// Result from processing an audio stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RosettaResult {
    /// Identified species
    pub species: String,

    /// Species confidence (0.0 - 1.0)
    pub species_confidence: f64,

    /// Context-enriched phrases
    pub phrases: Vec<ContextEnrichedPhrase>,

    /// Processing time in ms
    pub processing_time_ms: f64,
}

impl RosettaPipeline {
    /// Create a new pipeline with default configuration
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_config(RosettaConfig::default())
    }

    /// Create a new pipeline with custom configuration
    pub fn with_config(config: RosettaConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Unified weights for global species identification
        let global_weights = FeatureWeights::unified();

        // Create global similarity engine
        let mut global_engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, DistanceMetric::Cosine);
        global_engine.set_feature_weights(&global_weights.to_weight_vector());

        Ok(Self {
            config,
            global_engine,
            species_bundles: HashMap::new(),
            species_engines: HashMap::new(),
        })
    }

    /// Load a species bundle into the pipeline
    pub fn load_bundle(&mut self, bundle: RosettaBundle) {
        let species = bundle.species.clone();

        // Create species-specific similarity engine
        let mut engine = AcousticSimilarityEngine::with_metric(FEATURE_DIM, DistanceMetric::Cosine);
        engine.set_feature_weights(&bundle.feature_weights.to_weight_vector());

        // Fit normalization on dictionary centroids
        let n_samples = bundle.semantic_dictionary.type_centroids.len();
        if n_samples > 0 {
            let mut matrix = Array2::<f64>::zeros((n_samples, FEATURE_DIM));
            for (i, (_, centroid)) in bundle.semantic_dictionary.type_centroids.iter().enumerate() {
                for (j, &val) in centroid.iter().enumerate() {
                    if j < FEATURE_DIM {
                        matrix[[i, j]] = val as f64;
                    }
                }
            }
            engine.fit_normalization(&matrix);
        }

        self.species_engines.insert(species.clone(), engine);
        self.species_bundles.insert(species, bundle);
    }

    /// Load bundle from file
    pub fn load_bundle_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let bundle = RosettaBundle::load(path)?;
        self.load_bundle(bundle);
        Ok(())
    }

    /// Get list of loaded species
    pub fn loaded_species(&self) -> Vec<&String> {
        self.species_bundles.keys().collect()
    }

    /// Process an audio stream
    pub fn process_stream(
        &self,
        audio: &[f32],
        _env_state: EnvState,
    ) -> Result<RosettaResult, Box<dyn std::error::Error>> {
        let start = std::time::Instant::now();

        // PHASE 1: GLOBAL SPECIES IDENTIFICATION
        // Extract features for entire audio (convert f32 to f64)
        let audio_f64: Vec<f64> = audio.iter().map(|&f| f as f64).collect();
        let mut feature_extractor = ZooVoxFeatureExtractor::new(self.config.sample_rate);
        let features = feature_extractor.extract_45d(&audio_f64)?;
        let feature_vec: Vec<f32> = features.to_vector().iter().map(|&f| f as f32).collect();

        // Identify species by comparing to all loaded species
        let (species, confidence) = self.identify_species(&feature_vec)?;

        // PHASE 2a: SEMANTIC GROUNDING
        let phrases = if let Some(bundle) = self.species_bundles.get(&species) {
            if let Some(engine) = self.species_engines.get(&species) {
                self.match_to_dictionary(&feature_vec, bundle, engine)?
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        // PHASE 2b: CONTEXTUAL ENRICHMENT
        let enriched_phrases: Vec<ContextEnrichedPhrase> = phrases
            .into_iter()
            .map(|p| self.enrich_with_context(p, &_env_state))
            .collect();

        let processing_time = start.elapsed().as_secs_f64() * 1000.0;

        Ok(RosettaResult {
            species,
            species_confidence: confidence,
            phrases: enriched_phrases,
            processing_time_ms: processing_time,
        })
    }

    /// Identify species by comparing to all loaded species prototypes
    fn identify_species(&self, features: &[f32]) -> Result<(String, f64), Box<dyn std::error::Error>> {
        if self.species_bundles.is_empty() {
            return Ok(("unknown".to_string(), 0.0));
        }

        let query: Array1<f64> = Array1::from_vec(features.iter().map(|&f| f as f64).collect());

        let mut best_species = "unknown".to_string();
        let mut best_sim = 0.0;

        // Compare to each species' dictionary centroids
        for (species_name, bundle) in &self.species_bundles {
            // Compare to all centroids in this species' dictionary
            for centroid in bundle.semantic_dictionary.type_centroids.values() {
                let proto: Array1<f64> = Array1::from_vec(centroid.iter().map(|&f| f as f64).collect());
                let dist = self.global_engine.distance(&query, &proto);
                let sim = 1.0 - dist;

                if sim > best_sim {
                    best_sim = sim;
                    best_species = species_name.clone();
                }
            }
        }

        Ok((best_species, best_sim))
    }

    /// Match features to semantic dictionary
    fn match_to_dictionary(
        &self,
        features: &[f32],
        bundle: &RosettaBundle,
        engine: &AcousticSimilarityEngine,
    ) -> Result<Vec<ContextEnrichedPhrase>, Box<dyn std::error::Error>> {
        let dict = &bundle.semantic_dictionary;

        if let Some((type_id, sim, label, label_conf)) = dict.match_features(features, engine) {
            // Calculate grading score based on distance from centroid
            let grading_score = self.calculate_grading_score(features, &type_id, dict, engine);

            Ok(vec![ContextEnrichedPhrase {
                phrase_type_id: type_id,
                grading_score,
                acoustic_confidence: sim,
                semantic_label: label,
                label_confidence: label_conf,
                syntax_role: SyntaxRole::Unknown,
                environmental_state: EnvState::Unknown,
                inferred_intent: String::new(),
                start_ms: 0.0,
                duration_ms: 0.0,
            }])
        } else {
            // Novel phrase - not in dictionary
            Ok(vec![ContextEnrichedPhrase {
                phrase_type_id: "Novel".to_string(),
                grading_score: 0.5,
                acoustic_confidence: 0.0,
                semantic_label: "Unknown".to_string(),
                label_confidence: 0.0,
                syntax_role: SyntaxRole::Unknown,
                environmental_state: EnvState::Unknown,
                inferred_intent: String::new(),
                start_ms: 0.0,
                duration_ms: 0.0,
            }])
        }
    }

    /// Calculate grading score (discrete vs graded vocalization)
    fn calculate_grading_score(
        &self,
        features: &[f32],
        type_id: &str,
        dict: &SemanticPhraseDictionary,
        engine: &AcousticSimilarityEngine,
    ) -> f32 {
        if let Some(centroid) = dict.type_centroids.get(type_id) {
            let query: Array1<f64> = Array1::from_vec(features.iter().map(|&f| f as f64).collect());
            let proto: Array1<f64> = Array1::from_vec(centroid.iter().map(|&f| f as f64).collect());
            let dist = engine.distance(&query, &proto);

            // Higher distance = more graded (deviates from typical exemplar)
            // Lower distance = more discrete (close to typical exemplar)
            (1.0 - dist).min(1.0).max(0.0) as f32
        } else {
            0.5
        }
    }

    /// Enrich phrase with contextual information
    fn enrich_with_context(&self, mut phrase: ContextEnrichedPhrase, env_state: &EnvState) -> ContextEnrichedPhrase {
        // Set environmental state
        phrase.environmental_state = env_state.clone();

        // Infer intent based on semantic label and environment
        phrase.inferred_intent = self.infer_intent(&phrase.semantic_label, env_state);

        phrase
    }

    /// Infer intent from semantic label and environmental state
    fn infer_intent(&self, label: &str, env_state: &EnvState) -> String {
        // Simple rule-based intent inference
        match label {
            "Phee" | "Phee_Call" => match env_state {
                EnvState::Wind => "Long_Range_Contact".to_string(),
                EnvState::Quiet => "Social_Contact".to_string(),
                _ => "Contact".to_string(),
            },
            "Tsik" | "Alarm" => match env_state {
                EnvState::Storm => "Emergency_Alert".to_string(),
                _ => "Warning".to_string(),
            },
            "Twitter" => "Social_Bonding".to_string(),
            "Trill" => "Affiliative".to_string(),
            "Infant_cry" => "Solicitation".to_string(),
            "Fighting" => "Aggression".to_string(),
            "Mating" => "Reproductive".to_string(),
            "Grooming" => "Social_Bonding".to_string(),
            _ => "Unknown".to_string(),
        }
    }
}

impl Default for RosettaPipeline {
    fn default() -> Self {
        // SAFETY: RosettaPipeline::new() with default config cannot fail:
        // - FeatureWeights::unified() is infallible
        // - AcousticSimilarityEngine creation is infallible
        // - DynamicSegmenter creation is infallible
        #[allow(clippy::expect_used)]
        Self::new().expect("default config cannot fail")
    }
}

// =============================================================================
// FeatureWeights Extension
// =============================================================================

impl FeatureWeights {
    /// Create unified weights for global species identification
    pub fn unified() -> Self {
        Self {
            spectral: 1.2,
            harmonic: 1.0,
            temporal: 1.2,
            modulation: 1.2,
            cepstral: 1.0,
            formant: 1.0,
            micro_dynamics: 1.0,
            psychoacoustic: 1.0,
            tfs: 1.0,
            overrides: vec![],
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_phrase_dictionary() {
        let mut type_to_labels = HashMap::new();
        let mut labels = HashMap::new();
        labels.insert("Phee".to_string(), 0.95);
        labels.insert("Unknown".to_string(), 0.05);
        type_to_labels.insert("Type_0".to_string(), labels);

        let mut type_centroids = HashMap::new();
        type_centroids.insert("Type_0".to_string(), vec![0.5; FEATURE_DIM]);

        let dict = SemanticPhraseDictionary {
            species: "marmoset".to_string(),
            type_to_labels,
            type_centroids,
            total_phrases: 100,
            num_types: 1,
        };

        let (label, prob) = dict.get_primary_label("Type_0").unwrap();
        assert_eq!(label, "Phee");
        assert!((prob - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_env_state_default() {
        let state = EnvState::default();
        assert_eq!(state, EnvState::Unknown);
    }

    #[test]
    fn test_syntax_role_default() {
        let role = SyntaxRole::default();
        assert_eq!(role, SyntaxRole::Unknown);
    }

    #[test]
    fn test_unified_weights() {
        let weights = FeatureWeights::unified();
        assert!((weights.spectral - 1.2).abs() < 0.01);
        assert!((weights.harmonic - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rosetta_pipeline_creation() {
        let pipeline = RosettaPipeline::new();
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_rosetta_bundle_serialization() {
        let dict = SemanticPhraseDictionary {
            species: "marmoset".to_string(),
            type_to_labels: HashMap::new(),
            type_centroids: HashMap::new(),
            total_phrases: 0,
            num_types: 0,
        };

        let bundle = RosettaBundle::new("marmoset", FeatureWeights::marmoset(), dict, FeatureWeights::unified());

        // Test serialization/deserialization
        let json = serde_json::to_string(&bundle).unwrap();
        let deserialized: RosettaBundle = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.species, "marmoset");
    }

    #[test]
    fn test_infer_intent() {
        let pipeline = RosettaPipeline::new().unwrap();

        assert_eq!(pipeline.infer_intent("Phee", &EnvState::Wind), "Long_Range_Contact");
        assert_eq!(pipeline.infer_intent("Tsik", &EnvState::Storm), "Emergency_Alert");
        assert_eq!(pipeline.infer_intent("Twitter", &EnvState::Quiet), "Social_Bonding");
    }
}
