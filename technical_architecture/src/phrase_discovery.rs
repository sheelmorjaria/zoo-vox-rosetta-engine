//! Phrase Discovery Pipeline
//! =========================
//!
//! A unified pipeline that chains Dynamic Segmentation with Acoustic Similarity
//! to discover atomic phrase types from animal vocalizations.
//!
//! ## Architecture
//!
//! ```text
//! Audio Stream
//!     |
//!     v
//! DynamicSegmenter (CPD) --- "Where does one sound end and the next begin?"
//!     |
//!     v
//! FeatureExtractor (45D Vector per candidate)
//!     |
//!     v
//! AcousticSimilarityEngine --- "What is this sound, and have we seen it before?"
//!     |
//!     v
//! Atomic Phrase Types
//!     |
//!     v
//! Syntax Analysis (Markov Chains) --- "How do phrases combine?"
//! ```
//!
//! ## Key Insight: Phase Alignment
//!
//! Dynamic segmentation creates **phase-aligned** candidates, meaning the Similarity
//! Engine compares "whole apples to whole apples" instead of "apple slices to apple cores."
//!
//! This solves the problem where fixed-window segmentation creates artificial "phases"
//! of the same sound (Start, Mid1, Mid2, etc.) that appear as different phrase types.

use crate::dynamic_segmenter::{DynamicPhraseCandidate, DynamicSegmenter, DynamicSegmenterConfig};
use crate::species::{AtomicGranularity, HierarchicalThresholds};
use crate::zoo_vox_data_models::{AcousticFeatures30D, PhrasePrototype};
use crate::zoo_vox_features::ZooVoxFeatureExtractor;
use crate::zoo_vox_within_call::{WithinCallAnalyzer, WithinCallAnalysisResult};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for the phrase discovery pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseDiscoveryConfig {
    /// Sample rate for audio processing
    pub sample_rate: u32,
    /// Hierarchical thresholds for segmentation
    pub hierarchical_thresholds: HierarchicalThresholds,
    /// Which level carries semantic meaning
    pub atomic_granularity: AtomicGranularity,
    /// Similarity threshold for clustering (0.0 - 1.0)
    pub similarity_threshold: f32,
    /// Minimum occurrences for a phrase type
    pub min_occurrences: usize,
    /// Species name for context
    pub species: String,
}

impl Default for PhraseDiscoveryConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            hierarchical_thresholds: HierarchicalThresholds::default_thresholds(),
            atomic_granularity: AtomicGranularity::Syllable,
            similarity_threshold: 0.30,
            min_occurrences: 2,
            species: "unknown".to_string(),
        }
    }
}

impl PhraseDiscoveryConfig {
    /// Create config for marmoset analysis
    pub fn marmoset() -> Self {
        Self {
            sample_rate: 44100,
            hierarchical_thresholds: HierarchicalThresholds::marmoset(),
            atomic_granularity: AtomicGranularity::Syllable,
            similarity_threshold: 0.30,
            min_occurrences: 2,
            species: "marmoset".to_string(),
        }
    }

    /// Create config for zebra finch analysis
    pub fn zebra_finch() -> Self {
        Self {
            sample_rate: 44100,
            hierarchical_thresholds: HierarchicalThresholds::zebra_finch(),
            atomic_granularity: AtomicGranularity::Motif,
            similarity_threshold: 0.30,
            min_occurrences: 2,
            species: "zebra_finch".to_string(),
        }
    }

    /// Create config for Egyptian fruit bat analysis
    pub fn bat() -> Self {
        Self {
            sample_rate: 250000,
            hierarchical_thresholds: HierarchicalThresholds::bat(),
            atomic_granularity: AtomicGranularity::Syllable,
            similarity_threshold: 0.35,
            min_occurrences: 2,
            species: "egyptian_bat".to_string(),
        }
    }

    /// Create config for dolphin analysis
    pub fn dolphin() -> Self {
        Self {
            sample_rate: 44100,
            hierarchical_thresholds: HierarchicalThresholds::dolphin(),
            atomic_granularity: AtomicGranularity::Contour,
            similarity_threshold: 0.30,
            min_occurrences: 2,
            species: "dolphin".to_string(),
        }
    }
}

/// Result of the phrase discovery pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseDiscoveryResult {
    /// Number of audio files processed
    pub files_processed: usize,
    /// Total phrase candidates extracted
    pub total_candidates: usize,
    /// Unique phrase types discovered
    pub phrase_types: Vec<PipelinePhraseType>,
    /// Symbol sequence for syntax analysis
    pub symbol_sequence: Vec<String>,
    /// Transition frequencies
    pub transitions: HashMap<(String, String), usize>,
    /// Pipeline statistics
    pub stats: PipelineStats,
}

/// A discovered phrase type with metadata (pipeline output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelinePhraseType {
    /// Unique type identifier (e.g., "Type_A", "Type_B")
    pub type_id: String,
    /// Number of instances found
    pub instance_count: usize,
    /// Average duration in ms
    pub avg_duration_ms: f64,
    /// Duration range
    pub duration_range_ms: (f64, f64),
    /// Representative feature vector (centroid)
    pub centroid_features: Vec<f64>,
    /// Source files containing this type
    pub source_files: Vec<String>,
}

/// Pipeline statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    /// Segmentation time in ms
    pub segmentation_time_ms: f64,
    /// Clustering time in ms
    pub clustering_time_ms: f64,
    /// Total time in ms
    pub total_time_ms: f64,
    /// Vocabulary reduction ratio (candidates / types)
    pub vocabulary_reduction: f64,
    /// Type entropy in bits
    pub type_entropy: f64,
}

/// The unified phrase discovery pipeline
pub struct PhraseDiscoveryPipeline {
    config: PhraseDiscoveryConfig,
    segmenter: DynamicSegmenter,
    analyzer: WithinCallAnalyzer,
    extractor: ZooVoxFeatureExtractor,
}

impl PhraseDiscoveryPipeline {
    /// Create a new pipeline with the given configuration
    pub fn new(config: PhraseDiscoveryConfig) -> Self {
        let segmenter_config = match config.atomic_granularity {
            AtomicGranularity::Motif => DynamicSegmenterConfig::for_motif_level(&config.hierarchical_thresholds),
            AtomicGranularity::Syllable => DynamicSegmenterConfig::for_syllable_level(&config.hierarchical_thresholds),
            AtomicGranularity::Note => DynamicSegmenterConfig::for_note_level(&config.hierarchical_thresholds),
            AtomicGranularity::Contour => DynamicSegmenterConfig::for_motif_level(&config.hierarchical_thresholds),
        };

        let segmenter = DynamicSegmenter::new(segmenter_config, config.sample_rate);
        let analyzer = WithinCallAnalyzer::new();
        let extractor = ZooVoxFeatureExtractor::new(config.sample_rate);

        Self {
            config,
            segmenter,
            analyzer,
            extractor,
        }
    }

    /// Create a pipeline configured for marmoset analysis
    pub fn for_marmoset() -> Self {
        Self::new(PhraseDiscoveryConfig::marmoset())
    }

    /// Create a pipeline configured for zebra finch analysis
    pub fn for_zebra_finch() -> Self {
        Self::new(PhraseDiscoveryConfig::zebra_finch())
    }

    /// Create a pipeline configured for bat analysis
    pub fn for_bat() -> Self {
        Self::new(PhraseDiscoveryConfig::bat())
    }

    /// Create a pipeline configured for dolphin analysis
    pub fn for_dolphin() -> Self {
        Self::new(PhraseDiscoveryConfig::dolphin())
    }

    /// Discover phrase types from audio files
    ///
    /// This implements the complete pipeline:
    /// 1. **Segmentation**: Dynamic segmentation finds acoustic boundaries
    /// 2. **Feature Extraction**: 45D vectors for each candidate
    /// 3. **Similarity Clustering**: Groups similar candidates into types
    /// 4. **Syntax Analysis**: Builds transition model
    pub fn discover_phrases(
        &mut self,
        audio_segments: &[(Vec<f32>, String)], // (audio samples, source file name)
    ) -> PhraseDiscoveryResult {
        let total_start = std::time::Instant::now();
        let mut seg_time = 0u128;
        let mut cluster_time = 0u128;

        // Step 1: Dynamic Segmentation (Find Boundaries)
        let seg_start = std::time::Instant::now();
        let mut all_candidates: Vec<(DynamicPhraseCandidate, String)> = Vec::new();

        for (audio, source_file) in audio_segments {
            // Use Arc<Mutex<>> to allow Fn closure (as in the example)
            let extractor = Arc::new(std::sync::Mutex::new(
                ZooVoxFeatureExtractor::new(self.config.sample_rate)
            ));

            let extract_fn = |frame: &[f32], _sr: u32| {
                let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
                let mut ext = extractor.lock().unwrap();
                ext.extract_45d(&frame_f64).ok().map(|f| f.to_vector().to_vec())
            };

            let result = self.segmenter.segment(audio, extract_fn, source_file);
            for candidate in result.candidates {
                all_candidates.push((candidate, source_file.clone()));
            }
        }
        seg_time = seg_start.elapsed().as_millis();

        if all_candidates.is_empty() {
            return PhraseDiscoveryResult {
                files_processed: audio_segments.len(),
                total_candidates: 0,
                phrase_types: Vec::new(),
                symbol_sequence: Vec::new(),
                transitions: HashMap::new(),
                stats: PipelineStats {
                    segmentation_time_ms: seg_time as f64,
                    clustering_time_ms: 0.0,
                    total_time_ms: total_start.elapsed().as_millis() as f64,
                    vocabulary_reduction: 0.0,
                    type_entropy: 0.0,
                },
            };
        }

        // Step 2: Convert to PhrasePrototypes (45D -> 30D)
        let prototypes: Vec<PhrasePrototype> = all_candidates.iter()
            .map(|(candidate, source_file)| {
                self.candidate_to_prototype(candidate, source_file)
            })
            .collect();

        // Step 3: Acoustic Similarity Clustering (Discover Types)
        let cluster_start = std::time::Instant::now();
        let analysis_result = self.analyzer.discover_phrases(
            prototypes,
            "pipeline",
            &self.config.species,
        );
        cluster_time = cluster_start.elapsed().as_millis();

        // Step 4: Build output
        let phrase_types: Vec<PipelinePhraseType> = analysis_result.phrase_types.iter()
            .map(|pt| {
                // Calculate durations from sample indices
                let sample_rate = self.config.sample_rate as f64;
                let durations: Vec<f64> = pt.instances.iter()
                    .map(|inst| {
                        let samples = (inst.end_sample - inst.start_sample) as f64;
                        samples / sample_rate * 1000.0 // Convert to ms
                    })
                    .collect();
                let avg_dur = if durations.is_empty() { 0.0 } else {
                    durations.iter().sum::<f64>() / durations.len() as f64
                };
                let min_dur = durations.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_dur = durations.iter().cloned().fold(0.0, f64::max);

                // Convert centroid features to Vec<f64>
                let centroid = pt.centroid_features.to_vector().to_vec();

                PipelinePhraseType {
                    type_id: pt.type_id.clone(),
                    instance_count: pt.occurrence_count,
                    avg_duration_ms: avg_dur,
                    duration_range_ms: (min_dur, max_dur),
                    centroid_features: centroid,
                    source_files: pt.instances.iter()
                        .map(|i| i.source_id.clone())
                        .collect(),
                }
            })
            .collect();

        // Build transition map
        let transitions = self.build_transitions(&analysis_result.phrase_sequence);

        // Calculate statistics
        let total_time = total_start.elapsed().as_millis() as f64;
        let vocab_reduction = if analysis_result.unique_types > 0 {
            analysis_result.total_phrases as f64 / analysis_result.unique_types as f64
        } else {
            0.0
        };

        PhraseDiscoveryResult {
            files_processed: audio_segments.len(),
            total_candidates: all_candidates.len(),
            phrase_types,
            symbol_sequence: analysis_result.phrase_sequence,
            transitions,
            stats: PipelineStats {
                segmentation_time_ms: seg_time as f64,
                clustering_time_ms: cluster_time as f64,
                total_time_ms: total_time,
                vocabulary_reduction: vocab_reduction,
                type_entropy: analysis_result.type_entropy,
            },
        }
    }

    /// Convert DynamicPhraseCandidate (45D) to PhrasePrototype (30D)
    fn candidate_to_prototype(
        &self,
        candidate: &DynamicPhraseCandidate,
        source_file: &str,
    ) -> PhrasePrototype {
        // Convert 45D features to 30D by taking the first 30 dimensions
        // This is a simplification - in practice you'd want a more sophisticated mapping
        let mut arr = [0.0f64; 30];
        for (i, &val) in candidate.features.iter().take(30).enumerate() {
            arr[i] = val;
        }
        let features_30d = crate::zoo_vox_data_models::AcousticFeatures30D::from_vector(arr);

        PhrasePrototype {
            phrase_id: candidate.id.clone(),
            phrase_key: format!("DUR_{:.0}", candidate.duration_ms),
            species: self.config.species.clone(),
            source_file: Some(source_file.to_string()),
            source_dataset: None,
            encoding_strategy: crate::species::EncodingStrategy::PhraseType,
            encoding_modality: crate::species::AnalysisModality::Temporal,
            phrase_type: None,
            features_30d,
            contexts: Vec::new(),
            primary_context: None,
            typical_position: 0,
            co_occurring_phrases: Vec::new(),
            occurrence_count: 1,
            entropy_contribution: 0.0,
            signal_to_noise_ratio: 1.0 / (1.0 + candidate.internal_variance as f64),
            extraction_confidence: 1.0,
            created_at: chrono::Utc::now(),
            notes: None,
        }
    }

    /// Build transition frequency map from symbol sequence
    fn build_transitions(&self, sequence: &[String]) -> HashMap<(String, String), usize> {
        let mut transitions = HashMap::new();
        for window in sequence.windows(2) {
            *transitions.entry((window[0].clone(), window[1].clone())).or_insert(0) += 1;
        }
        transitions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let config = PhraseDiscoveryConfig::marmoset();
        let pipeline = PhraseDiscoveryPipeline::new(config);
        assert_eq!(pipeline.config.species, "marmoset");
    }

    #[test]
    fn test_empty_audio() {
        let mut pipeline = PhraseDiscoveryPipeline::for_marmoset();
        let result = pipeline.discover_phrases(&[]);
        assert_eq!(result.files_processed, 0);
        assert_eq!(result.total_candidates, 0);
    }

    #[test]
    fn test_config_defaults() {
        let config = PhraseDiscoveryConfig::default();
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.similarity_threshold, 0.30);
    }
}
