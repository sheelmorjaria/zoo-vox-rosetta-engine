//! Metadata-First Synthesis Engine (Rust Implementation)
//! ===================================================
//!
//! Replaces Python-based persona routing with direct 30D vector space queries.
//!
//! **Architecture Migration:**
//! - OLD: Intent → PersonaRouter → Single Buffer → Synthesis
//! - NEW: Intent → Vector Query → Multi-Buffer Selection → Granular Morphing
//!
//! **Key Features:**
//! - Direct 30D vector space queries using SIMD-optimized operations
//! - Multi-source interpolation for "Ghost Word" synthesis
//! - Integration with island_hopping navigation engine
//! - PyO3 bindings for Python cognitive layer integration
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use crate::island_hopping::Vector30D;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

// ============================================================================
// Core Data Structures
// ============================================================================

/// Query for the 30D acoustic vector space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataQuery {
    /// Target fundamental frequency (Hz)
    pub target_f0_hz: f32,

    /// Target duration (ms)
    pub target_duration_ms: f32,

    /// F0 tolerance for scoring (Hz)
    pub f0_tolerance_hz: f32,

    /// Duration tolerance for scoring (ms)
    pub duration_tolerance_ms: f32,

    /// Preferred contexts (soft constraints, score modifiers)
    pub preferred_contexts: Vec<String>,

    /// Avoided contexts (soft constraints, negative score modifiers)
    pub avoided_contexts: Vec<String>,

    /// Scoring weights
    pub acoustic_weight: f32,
    pub context_weight: f32,
    pub novelty_weight: f32,
}

impl Default for MetadataQuery {
    fn default() -> Self {
        Self {
            target_f0_hz: 7000.0,
            target_duration_ms: 50.0,
            f0_tolerance_hz: 500.0,
            duration_tolerance_ms: 20.0,
            preferred_contexts: Vec::new(),
            avoided_contexts: Vec::new(),
            acoustic_weight: 1.0,
            context_weight: 0.5,
            novelty_weight: 0.3,
        }
    }
}

/// A phrase candidate with 30D metadata and scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseCandidate {
    /// Unique phrase identifier
    pub phrase_id: String,

    /// Species identifier
    pub species: String,

    /// Cluster ID (for persona grouping)
    pub cluster_id: i32,

    /// Semantic context label
    pub context: String,

    /// 30D feature vector (acoustic signature)
    pub features: Vector30D,

    /// Audio buffer (samples)
    #[serde(skip)]
    pub audio_buffer: Option<Vec<f32>>,

    /// Sample rate for audio buffer
    pub sample_rate: u32,

    /// Computed scores
    pub acoustic_score: f32,
    pub context_score: f32,
    pub novelty_score: f32,
    pub total_score: f32,
}

impl PhraseCandidate {
    /// Create a new phrase candidate from 30D features
    pub fn new(
        phrase_id: String,
        species: String,
        cluster_id: i32,
        context: String,
        features: Vector30D,
        sample_rate: u32,
    ) -> Self {
        Self {
            phrase_id,
            species,
            cluster_id,
            context,
            features,
            audio_buffer: None,
            sample_rate,
            acoustic_score: 0.0,
            context_score: 0.0,
            novelty_score: 0.0,
            total_score: 0.0,
        }
    }

    /// Create from metadata dictionary (for Python integration)
    pub fn from_metadata(
        phrase_id: String,
        species: String,
        cluster_id: i32,
        context: String,
        metadata: HashMap<String, f32>,
        sample_rate: u32,
    ) -> Result<Self> {
        // Build 30D feature vector from metadata
        let features = Vector30D {
            // Fundamental (3)
            mean_f0_hz: metadata.get("mean_f0_hz").copied().unwrap_or(0.0),
            duration_ms: metadata.get("duration_ms").copied().unwrap_or(0.0),
            f0_range_hz: metadata.get("f0_range_hz").copied().unwrap_or(0.0),
            // Grit Factors (3)
            harmonic_to_noise_ratio: metadata.get("harmonic_to_noise_ratio").copied().unwrap_or(0.0),
            spectral_flatness: metadata.get("spectral_flatness").copied().unwrap_or(0.0),
            harmonicity: metadata.get("harmonicity").copied().unwrap_or(0.0),
            // Motion Factors (7)
            attack_time_ms: metadata.get("attack_time_ms").copied().unwrap_or(0.0),
            decay_time_ms: metadata.get("decay_time_ms").copied().unwrap_or(0.0),
            sustain_level: metadata.get("sustain_level").copied().unwrap_or(0.0),
            vibrato_rate_hz: metadata.get("vibrato_rate_hz").copied().unwrap_or(0.0),
            vibrato_depth: metadata.get("vibrato_depth").copied().unwrap_or(0.0),
            jitter: metadata.get("jitter").copied().unwrap_or(0.0),
            shimmer: metadata.get("shimmer").copied().unwrap_or(0.0),
            // Fingerprint Factors (14)
            mfcc_1: metadata.get("mfcc_1").copied().unwrap_or(0.0),
            mfcc_2: metadata.get("mfcc_2").copied().unwrap_or(0.0),
            mfcc_3: metadata.get("mfcc_3").copied().unwrap_or(0.0),
            mfcc_4: metadata.get("mfcc_4").copied().unwrap_or(0.0),
            mfcc_5: metadata.get("mfcc_5").copied().unwrap_or(0.0),
            mfcc_6: metadata.get("mfcc_6").copied().unwrap_or(0.0),
            mfcc_7: metadata.get("mfcc_7").copied().unwrap_or(0.0),
            mfcc_8: metadata.get("mfcc_8").copied().unwrap_or(0.0),
            mfcc_9: metadata.get("mfcc_9").copied().unwrap_or(0.0),
            mfcc_10: metadata.get("mfcc_10").copied().unwrap_or(0.0),
            mfcc_11: metadata.get("mfcc_11").copied().unwrap_or(0.0),
            mfcc_12: metadata.get("mfcc_12").copied().unwrap_or(0.0),
            mfcc_13: metadata.get("mfcc_13").copied().unwrap_or(0.0),
            spectral_flux: metadata.get("spectral_flux").copied().unwrap_or(0.0),
            // Rhythm Factors (3)
            median_ici_ms: metadata.get("median_ici_ms").copied().unwrap_or(0.0),
            onset_rate_hz: metadata.get("onset_rate_hz").copied().unwrap_or(0.0),
            ici_coefficient_of_variation: metadata.get("ici_coefficient_of_variation").copied().unwrap_or(0.0),
        };

        Ok(Self::new(phrase_id, species, cluster_id, context, features, sample_rate))
    }

    /// Get key acoustic features (backward compatibility)
    pub fn f0_hz(&self) -> f32 {
        self.features.mean_f0_hz
    }

    pub fn harmonicity(&self) -> f32 {
        self.features.harmonicity
    }
}

/// Synthesis recipe with multiple source buffers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisRecipe {
    /// Source phrases with interpolation weights
    pub sources: Vec<(PhraseCandidate, f32)>,

    /// Interpolated 30D target parameters
    pub target_params: SynthesisTarget,

    /// Synthesis mode: morph, crossfade, alternate
    pub synthesis_mode: String,

    /// Is this cross-persona synthesis?
    pub is_cross_persona: bool,

    /// Discovery potential (0-1, how novel is this?)
    pub discovery_potential: f32,

    /// Human-readable reasoning
    pub reasoning: String,
}

/// Target synthesis parameters (30D)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisTarget {
    /// Full 30D target vector
    pub target_vector_30d: Vector30D,

    /// Key acoustic parameters (for convenience)
    pub mean_f0_hz: f32,
    pub duration_ms: f32,
    pub f0_range_hz: f32,
}

impl SynthesisTarget {
    /// Create from weighted interpolation of sources
    pub fn from_interpolation(sources: &[(PhraseCandidate, f32)]) -> Self {
        let mut target_vector_30d = Vector30D::default();

        // Weighted average of feature vectors
        for (candidate, weight) in sources {
            let weighted = candidate.features * *weight;
            target_vector_30d = target_vector_30d + weighted;
        }

        Self {
            mean_f0_hz: target_vector_30d.mean_f0_hz,
            duration_ms: target_vector_30d.duration_ms,
            f0_range_hz: target_vector_30d.f0_range_hz,
            target_vector_30d,
        }
    }
}

// ============================================================================
// Vector Space Query Engine
// ============================================================================

/// Queries the 30D acoustic vector space directly
pub struct VectorSpaceQueryEngine {
    /// All phrases in the database
    phrases: Vec<PhraseCandidate>,

    /// Index by species
    species_index: HashMap<String, Vec<usize>>,

    /// Index by cluster ID
    cluster_index: HashMap<i32, Vec<usize>>,

    /// Feature statistics for normalization
    feature_means: Vector30D,
    feature_stds: Vector30D,
}

impl VectorSpaceQueryEngine {
    /// Create a new query engine
    pub fn new() -> Self {
        Self {
            phrases: Vec::new(),
            species_index: HashMap::new(),
            cluster_index: HashMap::new(),
            feature_means: Vector30D::default(),
            feature_stds: Vector30D::default(),
        }
    }

    /// Load phrases from a vector of candidates
    pub fn load_phrases(&mut self, phrases: Vec<PhraseCandidate>) -> Result<()> {
        self.phrases = phrases;

        // Build indexes
        for (idx, phrase) in self.phrases.iter().enumerate() {
            // Species index
            self.species_index
                .entry(phrase.species.clone())
                .or_insert_with(Vec::new)
                .push(idx);

            // Cluster index
            self.cluster_index
                .entry(phrase.cluster_id)
                .or_insert_with(Vec::new)
                .push(idx);
        }

        // Calculate feature statistics
        self.calculate_statistics();

        Ok(())
    }

    /// Calculate mean and std for each dimension
    fn calculate_statistics(&mut self) {
        if self.phrases.is_empty() {
            return;
        }

        let n = self.phrases.len() as f32;

        // Calculate means
        let mut sum = Vector30D::default();
        for phrase in &self.phrases {
            sum = sum + phrase.features;
        }
        self.feature_means = sum * (1.0 / n);

        // Calculate stds
        let mut variance = Vector30D::default();
        for phrase in &self.phrases {
            let diff = phrase.features - self.feature_means;
            // Manually multiply element-wise
            variance = Vector30D {
                mean_f0_hz: variance.mean_f0_hz + diff.mean_f0_hz * diff.mean_f0_hz,
                f0_range_hz: variance.f0_range_hz + diff.f0_range_hz * diff.f0_range_hz,
                duration_ms: variance.duration_ms + diff.duration_ms * diff.duration_ms,
                harmonic_to_noise_ratio: variance.harmonic_to_noise_ratio + diff.harmonic_to_noise_ratio * diff.harmonic_to_noise_ratio,
                spectral_flatness: variance.spectral_flatness + diff.spectral_flatness * diff.spectral_flatness,
                harmonicity: variance.harmonicity + diff.harmonicity * diff.harmonicity,
                attack_time_ms: variance.attack_time_ms + diff.attack_time_ms * diff.attack_time_ms,
                decay_time_ms: variance.decay_time_ms + diff.decay_time_ms * diff.decay_time_ms,
                sustain_level: variance.sustain_level + diff.sustain_level * diff.sustain_level,
                vibrato_rate_hz: variance.vibrato_rate_hz + diff.vibrato_rate_hz * diff.vibrato_rate_hz,
                vibrato_depth: variance.vibrato_depth + diff.vibrato_depth * diff.vibrato_depth,
                jitter: variance.jitter + diff.jitter * diff.jitter,
                shimmer: variance.shimmer + diff.shimmer * diff.shimmer,
                mfcc_1: variance.mfcc_1 + diff.mfcc_1 * diff.mfcc_1,
                mfcc_2: variance.mfcc_2 + diff.mfcc_2 * diff.mfcc_2,
                mfcc_3: variance.mfcc_3 + diff.mfcc_3 * diff.mfcc_3,
                mfcc_4: variance.mfcc_4 + diff.mfcc_4 * diff.mfcc_4,
                mfcc_5: variance.mfcc_5 + diff.mfcc_5 * diff.mfcc_5,
                mfcc_6: variance.mfcc_6 + diff.mfcc_6 * diff.mfcc_6,
                mfcc_7: variance.mfcc_7 + diff.mfcc_7 * diff.mfcc_7,
                mfcc_8: variance.mfcc_8 + diff.mfcc_8 * diff.mfcc_8,
                mfcc_9: variance.mfcc_9 + diff.mfcc_9 * diff.mfcc_9,
                mfcc_10: variance.mfcc_10 + diff.mfcc_10 * diff.mfcc_10,
                mfcc_11: variance.mfcc_11 + diff.mfcc_11 * diff.mfcc_11,
                mfcc_12: variance.mfcc_12 + diff.mfcc_12 * diff.mfcc_12,
                mfcc_13: variance.mfcc_13 + diff.mfcc_13 * diff.mfcc_13,
                spectral_flux: variance.spectral_flux + diff.spectral_flux * diff.spectral_flux,
                median_ici_ms: variance.median_ici_ms + diff.median_ici_ms * diff.median_ici_ms,
                onset_rate_hz: variance.onset_rate_hz + diff.onset_rate_hz * diff.onset_rate_hz,
                ici_coefficient_of_variation: variance.ici_coefficient_of_variation + diff.ici_coefficient_of_variation * diff.ici_coefficient_of_variation,
            };
        }
        variance = variance * (1.0 / n);

        // Sqrt for std (add small epsilon to avoid division by zero)
        self.feature_stds = Vector30D {
            mean_f0_hz: variance.mean_f0_hz.sqrt() + 1e-6,
            f0_range_hz: variance.f0_range_hz.sqrt() + 1e-6,
            duration_ms: variance.duration_ms.sqrt() + 1e-6,
            harmonic_to_noise_ratio: variance.harmonic_to_noise_ratio.sqrt() + 1e-6,
            spectral_flatness: variance.spectral_flatness.sqrt() + 1e-6,
            harmonicity: variance.harmonicity.sqrt() + 1e-6,
            attack_time_ms: variance.attack_time_ms.sqrt() + 1e-6,
            decay_time_ms: variance.decay_time_ms.sqrt() + 1e-6,
            sustain_level: variance.sustain_level.sqrt() + 1e-6,
            vibrato_rate_hz: variance.vibrato_rate_hz.sqrt() + 1e-6,
            vibrato_depth: variance.vibrato_depth.sqrt() + 1e-6,
            jitter: variance.jitter.sqrt() + 1e-6,
            shimmer: variance.shimmer.sqrt() + 1e-6,
            mfcc_1: variance.mfcc_1.sqrt() + 1e-6,
            mfcc_2: variance.mfcc_2.sqrt() + 1e-6,
            mfcc_3: variance.mfcc_3.sqrt() + 1e-6,
            mfcc_4: variance.mfcc_4.sqrt() + 1e-6,
            mfcc_5: variance.mfcc_5.sqrt() + 1e-6,
            mfcc_6: variance.mfcc_6.sqrt() + 1e-6,
            mfcc_7: variance.mfcc_7.sqrt() + 1e-6,
            mfcc_8: variance.mfcc_8.sqrt() + 1e-6,
            mfcc_9: variance.mfcc_9.sqrt() + 1e-6,
            mfcc_10: variance.mfcc_10.sqrt() + 1e-6,
            mfcc_11: variance.mfcc_11.sqrt() + 1e-6,
            mfcc_12: variance.mfcc_12.sqrt() + 1e-6,
            mfcc_13: variance.mfcc_13.sqrt() + 1e-6,
            spectral_flux: variance.spectral_flux.sqrt() + 1e-6,
            median_ici_ms: variance.median_ici_ms.sqrt() + 1e-6,
            onset_rate_hz: variance.onset_rate_hz.sqrt() + 1e-6,
            ici_coefficient_of_variation: variance.ici_coefficient_of_variation.sqrt() + 1e-6,
        };
    }

    /// Query for nearest neighbors in 30D vector space
    pub fn query_nearest(
        &self,
        query: &MetadataQuery,
        species_filter: Option<&str>,
        top_k: usize,
    ) -> Vec<PhraseCandidate> {
        let mut candidates = Vec::new();

        // Filter by species if specified
        let search_space: Vec<usize> = if let Some(species) = species_filter {
            self.species_index.get(species).cloned().unwrap_or_default()
        } else {
            (0..self.phrases.len()).collect()
        };

        // Build target vector (use query values where specified, means otherwise)
        let mut target_vector = self.feature_means;
        target_vector.mean_f0_hz = query.target_f0_hz;
        target_vector.duration_ms = query.target_duration_ms;

        // Score each candidate
        for &idx in &search_space {
            let mut phrase = self.phrases[idx].clone();

            // Calculate 30D Euclidean distance (manually normalized)
            let diff = phrase.features - target_vector;
            // Manual element-wise division for normalization
            let normalized_diff = Vector30D {
                mean_f0_hz: diff.mean_f0_hz / self.feature_stds.mean_f0_hz,
                f0_range_hz: diff.f0_range_hz / self.feature_stds.f0_range_hz,
                duration_ms: diff.duration_ms / self.feature_stds.duration_ms,
                harmonic_to_noise_ratio: diff.harmonic_to_noise_ratio / self.feature_stds.harmonic_to_noise_ratio,
                spectral_flatness: diff.spectral_flatness / self.feature_stds.spectral_flatness,
                harmonicity: diff.harmonicity / self.feature_stds.harmonicity,
                attack_time_ms: diff.attack_time_ms / self.feature_stds.attack_time_ms,
                decay_time_ms: diff.decay_time_ms / self.feature_stds.decay_time_ms,
                sustain_level: diff.sustain_level / self.feature_stds.sustain_level,
                vibrato_rate_hz: diff.vibrato_rate_hz / self.feature_stds.vibrato_rate_hz,
                vibrato_depth: diff.vibrato_depth / self.feature_stds.vibrato_depth,
                jitter: diff.jitter / self.feature_stds.jitter,
                shimmer: diff.shimmer / self.feature_stds.shimmer,
                mfcc_1: diff.mfcc_1 / self.feature_stds.mfcc_1,
                mfcc_2: diff.mfcc_2 / self.feature_stds.mfcc_2,
                mfcc_3: diff.mfcc_3 / self.feature_stds.mfcc_3,
                mfcc_4: diff.mfcc_4 / self.feature_stds.mfcc_4,
                mfcc_5: diff.mfcc_5 / self.feature_stds.mfcc_5,
                mfcc_6: diff.mfcc_6 / self.feature_stds.mfcc_6,
                mfcc_7: diff.mfcc_7 / self.feature_stds.mfcc_7,
                mfcc_8: diff.mfcc_8 / self.feature_stds.mfcc_8,
                mfcc_9: diff.mfcc_9 / self.feature_stds.mfcc_9,
                mfcc_10: diff.mfcc_10 / self.feature_stds.mfcc_10,
                mfcc_11: diff.mfcc_11 / self.feature_stds.mfcc_11,
                mfcc_12: diff.mfcc_12 / self.feature_stds.mfcc_12,
                mfcc_13: diff.mfcc_13 / self.feature_stds.mfcc_13,
                spectral_flux: diff.spectral_flux / self.feature_stds.spectral_flux,
                median_ici_ms: diff.median_ici_ms / self.feature_stds.median_ici_ms,
                onset_rate_hz: diff.onset_rate_hz / self.feature_stds.onset_rate_hz,
                ici_coefficient_of_variation: diff.ici_coefficient_of_variation / self.feature_stds.ici_coefficient_of_variation,
            };

            // Calculate Euclidean distance manually
            let arr = normalized_diff.to_array();
            let sum_squared: f32 = arr.iter().map(|&x| x * x).sum();
            let normalized_distance = sum_squared.sqrt();

            // Convert distance to score (closer = higher score)
            phrase.acoustic_score = (-normalized_distance / 10.0).exp();

            // Additional scoring for F0 and duration proximity
            let f0_distance = (phrase.features.mean_f0_hz - query.target_f0_hz).abs();
            let duration_distance =
                (phrase.features.duration_ms - query.target_duration_ms).abs();

            let f0_score = 1.0 / (1.0 + f0_distance / query.f0_tolerance_hz);
            let duration_score = 1.0 / (1.0 + duration_distance / query.duration_tolerance_ms);

            // Combine scores
            phrase.acoustic_score = 0.5 * phrase.acoustic_score + 0.25 * (f0_score + duration_score);

            // Context score (soft constraint)
            if query.preferred_contexts.contains(&phrase.context) {
                phrase.context_score += query.context_weight;
            }
            if query.avoided_contexts.contains(&phrase.context) {
                phrase.context_score -= query.context_weight * 0.5;
            }

            // Novelty score (reward exploration)
            let cluster_usage = self.cluster_index.get(&phrase.cluster_id).map(|v| v.len()).unwrap_or(1);
            phrase.novelty_score = query.novelty_weight * (1.0 / cluster_usage as f32);

            // Total score
            phrase.total_score = query.acoustic_weight * phrase.acoustic_score
                + phrase.context_score
                + phrase.novelty_score;

            candidates.push(phrase);
        }

        // Sort by total score and return top_k
        candidates.sort_by(|a, b| b.total_score.partial_cmp(&a.total_score).unwrap());
        candidates.truncate(top_k);
        candidates
    }

    /// Query for interpolation targets (multi-source morphing)
    pub fn query_interpolation_targets(
        &self,
        query: &MetadataQuery,
        num_sources: usize,
        species_filter: Option<&str>,
    ) -> SynthesisRecipe {
        let candidates = self.query_nearest(query, species_filter, num_sources * 2);

        // Select diverse sources
        let mut sources = Vec::new();
        let mut used_clusters = std::collections::HashSet::new();

        for candidate in candidates {
            if candidate.cluster_id != -1 && !used_clusters.contains(&candidate.cluster_id)
                || sources.len() < num_sources
            {
                let weight = candidate.total_score;
                let cluster_id = candidate.cluster_id;
                sources.push((candidate, weight));
                used_clusters.insert(cluster_id);

                if sources.len() >= num_sources {
                    break;
                }
            }
        }

        // Normalize weights
        let total_weight: f32 = sources.iter().map(|(_, w)| w).sum();
        for (_, weight) in &mut sources {
            *weight /= total_weight;
        }

        // Calculate 30D target parameters
        let target_params = SynthesisTarget::from_interpolation(&sources);

        // Determine if cross-persona
        let clusters_used: Vec<i32> = sources.iter().map(|(c, _)| c.cluster_id).collect();
        let is_cross_persona = clusters_used.iter().collect::<std::collections::HashSet<_>>().len() > 1;

        // Calculate discovery potential
        let discovery_potential = if is_cross_persona && sources.len() >= 2 {
            let cluster_distance = (sources[0].0.cluster_id - sources[1].0.cluster_id).abs() as f32;
            (cluster_distance / 5.0).min(1.0)
        } else {
            0.0
        };

        // Generate reasoning
        let source_names: Vec<String> = sources.iter().map(|(c, _)| c.phrase_id.clone()).collect();
        let clusters_str: Vec<String> = sources.iter().map(|(c, _)| format!("C{}", c.cluster_id)).collect();
        let reasoning = format!(
            "Interpolating {} sources: {} (clusters: {})",
            sources.len(),
            source_names.join(", "),
            clusters_str.join(", ")
        );

        SynthesisRecipe {
            sources,
            target_params,
            synthesis_mode: "morph".to_string(),
            is_cross_persona,
            discovery_potential,
            reasoning,
        }
    }
}

impl Default for VectorSpaceQueryEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Metadata-First Synthesizer
// ============================================================================

/// Metadata-first synthesis engine (Rust implementation)
pub struct MetadataSynthesizer {
    query_engine: VectorSpaceQueryEngine,
    sample_rate: u32,
}

impl MetadataSynthesizer {
    /// Create a new metadata synthesizer
    pub fn new(sample_rate: u32) -> Self {
        Self {
            query_engine: VectorSpaceQueryEngine::new(),
            sample_rate,
        }
    }

    /// Load phrases from a vector
    pub fn load_phrases(&mut self, phrases: Vec<PhraseCandidate>) -> Result<()> {
        self.query_engine.load_phrases(phrases)
    }

    /// Synthesize by targeting acoustic coordinates
    pub fn synthesize_by_target(
        &self,
        target_f0_hz: f32,
        target_duration_ms: f32,
        species_filter: Option<&str>,
        preferred_contexts: Vec<String>,
    ) -> Result<(SynthesisRecipe, Vec<f32>)> {
        let query = MetadataQuery {
            target_f0_hz,
            target_duration_ms,
            preferred_contexts,
            ..Default::default()
        };

        let recipe = self
            .query_engine
            .query_interpolation_targets(&query, 2, species_filter);

        // Generate placeholder audio (actual synthesis would call granular engine)
        let audio = self.generate_placeholder_audio(&recipe, 200.0);

        Ok((recipe, audio))
    }

    /// Synthesize a "Ghost Word" between two clusters
    pub fn synthesize_ghost_word(
        &self,
        cluster_a_id: i32,
        cluster_b_id: i32,
        blend_ratio: f32,
        _species_filter: Option<&str>,
    ) -> Result<(SynthesisRecipe, Vec<f32>)> {
        // Get phrases from both clusters
        let phrases_a: Vec<_> = self
            .query_engine
            .cluster_index
            .get(&cluster_a_id)
            .map(|indices| indices.iter().map(|&idx| self.query_engine.phrases[idx].clone()).collect())
            .unwrap_or_default();

        let phrases_b: Vec<_> = self
            .query_engine
            .cluster_index
            .get(&cluster_b_id)
            .map(|indices| indices.iter().map(|&idx| self.query_engine.phrases[idx].clone()).collect())
            .unwrap_or_default();

        if phrases_a.is_empty() || phrases_b.is_empty() {
            anyhow::bail!("Cannot find phrases for clusters {} and {}", cluster_a_id, cluster_b_id);
        }

        // Select best from each cluster (by harmonicity)
        let best_a = phrases_a
            .iter()
            .max_by(|a, b| a.features.harmonicity.partial_cmp(&b.features.harmonicity).unwrap())
            .unwrap();
        let best_b = phrases_b
            .iter()
            .max_by(|a, b| a.features.harmonicity.partial_cmp(&b.features.harmonicity).unwrap())
            .unwrap();

        // Create sources
        let sources = vec![
            (best_a.clone(), 1.0 - blend_ratio),
            (best_b.clone(), blend_ratio),
        ];

        // Calculate target
        let target_params = SynthesisTarget::from_interpolation(&sources);

        let reasoning = format!(
            "Ghost word: Cluster {} + Cluster {} @ {:.1}% ratio",
            cluster_a_id,
            cluster_b_id,
            blend_ratio * 100.0
        );

        let recipe = SynthesisRecipe {
            sources,
            target_params,
            synthesis_mode: "morph".to_string(),
            is_cross_persona: true,
            discovery_potential: 1.0,
            reasoning,
        };

        let audio = self.generate_placeholder_audio(&recipe, 200.0);

        Ok((recipe, audio))
    }

    /// Generate placeholder audio (actual implementation would use granular engine)
    fn generate_placeholder_audio(&self, _recipe: &SynthesisRecipe, duration_ms: f32) -> Vec<f32> {
        let num_samples = (duration_ms / 1000.0 * self.sample_rate as f32) as usize;
        vec![0.0; num_samples]
    }
}

// ============================================================================
// PyO3 Bindings (Python Integration)
// ============================================================================

#[cfg(feature = "python-bindings")]
pub mod python_bindings {
    use super::*;
    use pyo3::types::PyDict;

    /// PyO3 wrapper for MetadataQuery
    #[pyclass(name = "MetadataQuery")]
    #[derive(Clone)]
    pub struct PyMetadataQuery(pub MetadataQuery);

    #[pymethods]
    impl PyMetadataQuery {
        #[new]
        #[pyo3(signature = (
            target_f0_hz=7000.0,
            target_duration_ms=50.0,
            f0_tolerance_hz=500.0,
            duration_tolerance_ms=20.0,
            preferred_contexts=None,
            avoided_contexts=None,
            acoustic_weight=1.0,
            context_weight=0.5,
            novelty_weight=0.3
        ))]
        fn new(
            target_f0_hz: f32,
            target_duration_ms: f32,
            f0_tolerance_hz: f32,
            duration_tolerance_ms: f32,
            preferred_contexts: Option<Vec<String>>,
            avoided_contexts: Option<Vec<String>>,
            acoustic_weight: f32,
            context_weight: f32,
            novelty_weight: f32,
        ) -> Self {
            Self(MetadataQuery {
                target_f0_hz,
                target_duration_ms,
                f0_tolerance_hz,
                duration_tolerance_ms,
                preferred_contexts: preferred_contexts.unwrap_or_default(),
                avoided_contexts: avoided_contexts.unwrap_or_default(),
                acoustic_weight,
                context_weight,
                novelty_weight,
            })
        }
    }

    /// PyO3 wrapper for PhraseCandidate
    #[pyclass(name = "PhraseCandidate")]
    #[derive(Clone)]
    pub struct PyPhraseCandidate(pub PhraseCandidate);

    #[pymethods]
    impl PyPhraseCandidate {
        #[getter]
        fn phrase_id(&self) -> String {
            self.0.phrase_id.clone()
        }

        #[getter]
        fn species(&self) -> String {
            self.0.species.clone()
        }

        #[getter]
        fn cluster_id(&self) -> i32 {
            self.0.cluster_id
        }

        #[getter]
        fn context(&self) -> String {
            self.0.context.clone()
        }

        #[getter]
        fn f0_hz(&self) -> f32 {
            self.0.f0_hz()
        }

        #[getter]
        fn harmonicity(&self) -> f32 {
            self.0.harmonicity()
        }

        #[getter]
        fn total_score(&self) -> f32 {
            self.0.total_score
        }

        /// Get 30D feature vector as dictionary
        fn get_feature_vector(&self, py: Python) -> Py<PyDict> {
            let f = &self.0.features;
            let dict = PyDict::new(py);
            dict.set_item("mean_f0_hz", f.mean_f0_hz).unwrap();
            dict.set_item("duration_ms", f.duration_ms).unwrap();
            dict.set_item("f0_range_hz", f.f0_range_hz).unwrap();
            dict.set_item("harmonic_to_noise_ratio", f.harmonic_to_noise_ratio).unwrap();
            dict.set_item("spectral_flatness", f.spectral_flatness).unwrap();
            dict.set_item("harmonicity", f.harmonicity).unwrap();
            dict.set_item("attack_time_ms", f.attack_time_ms).unwrap();
            dict.set_item("decay_time_ms", f.decay_time_ms).unwrap();
            dict.set_item("sustain_level", f.sustain_level).unwrap();
            dict.set_item("vibrato_rate_hz", f.vibrato_rate_hz).unwrap();
            dict.set_item("vibrato_depth", f.vibrato_depth).unwrap();
            dict.set_item("jitter", f.jitter).unwrap();
            dict.set_item("shimmer", f.shimmer).unwrap();
            // ... (would add all 30 dimensions)
            dict.into()
        }
    }

    /// PyO3 wrapper for SynthesisRecipe
    #[pyclass(name = "SynthesisRecipe")]
    #[derive(Clone)]
    pub struct PySynthesisRecipe(pub SynthesisRecipe);

    #[pymethods]
    impl PySynthesisRecipe {
        #[getter]
        fn reasoning(&self) -> String {
            self.0.reasoning.clone()
        }

        #[getter]
        fn is_cross_persona(&self) -> bool {
            self.0.is_cross_persona
        }

        #[getter]
        fn discovery_potential(&self) -> f32 {
            self.0.discovery_potential
        }

        #[getter]
        fn target_params(&self, py: Python) -> Py<PyDict> {
            let dict = PyDict::new(py);
            let t = &self.0.target_params;
            dict.set_item("mean_f0_hz", t.mean_f0_hz).unwrap();
            dict.set_item("duration_ms", t.duration_ms).unwrap();
            dict.set_item("f0_range_hz", t.f0_range_hz).unwrap();
            dict.into()
        }

        fn sources(&self) -> Vec<(PyPhraseCandidate, f32)> {
            self.0
                .sources
                .iter()
                .map(|(c, w)| (PyPhraseCandidate(c.clone()), *w))
                .collect()
        }
    }

    /// PyO3 wrapper for MetadataSynthesizer
    #[pyclass(name = "MetadataSynthesizer")]
    pub struct PyMetadataSynthesizer(pub MetadataSynthesizer);

    #[pymethods]
    impl PyMetadataSynthesizer {
        #[new]
        #[pyo3(signature = (sample_rate=48000))]
        fn new(sample_rate: u32) -> Self {
            Self(MetadataSynthesizer::new(sample_rate))
        }

        fn load_phrases(&mut self, phrases: Vec<PyPhraseCandidate>) -> PyResult<()> {
            let rust_phrases: Vec<PhraseCandidate> = phrases.into_iter().map(|p| p.0).collect();
            self.0.load_phrases(rust_phrases).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to load phrases: {}", e))
            })
        }

        fn synthesize_by_target(
            &self,
            target_f0_hz: f32,
            target_duration_ms: f32,
            species: Option<String>,
            preferred_contexts: Vec<String>,
        ) -> PyResult<(PySynthesisRecipe, Vec<f32>)> {
            self.0
                .synthesize_by_target(target_f0_hz, target_duration_ms, species.as_deref(), preferred_contexts)
                .map(|(recipe, audio)| (PySynthesisRecipe(recipe), audio))
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Synthesis failed: {}", e))
                })
        }

        fn synthesize_ghost_word(
            &self,
            cluster_a_id: i32,
            cluster_b_id: i32,
            blend_ratio: f32,
            species: Option<String>,
        ) -> PyResult<(PySynthesisRecipe, Vec<f32>)> {
            self.0
                .synthesize_ghost_word(cluster_a_id, cluster_b_id, blend_ratio, species.as_deref())
                .map(|(recipe, audio)| (PySynthesisRecipe(recipe), audio))
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("Ghost word synthesis failed: {}", e))
                })
        }
    }
}

#[cfg(feature = "python-bindings")]
pub use python_bindings::{PyMetadataQuery, PyMetadataSynthesizer, PyPhraseCandidate, PySynthesisRecipe};
