//! Semantic Reconstruction Pipeline - Stage 4 of Synthesis Pipeline
//! ================================================================
//!
//! This module implements the semantic reconstruction layer that bridges
//! corpus analysis with granular synthesis using 112D RosettaFeatures.
//!
//! **Components:**
//! - `SourceMetadata112D`: 112D feature metadata for synthesis (matches RosettaFeatures)
//! - `ExemplarManager`: Stores best audio per cluster ID
//! - `CachedGranularSynthesizer`: Synthesizes audio from timelines
//! - `SynthesisTimeline`: Timeline of semantic events for synthesis
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use crate::micro_dynamics_extractor::RosettaFeatures;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// 112D SOURCE METADATA - Wraps RosettaFeatures for Synthesis
// =============================================================================

/// 112D Source Metadata for synthesis control
///
/// This structure wraps `RosettaFeatures` with additional synthesis metadata
/// like cluster_id for corpus tracking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceMetadata112D {
    /// The underlying 112D features from RosettaFeatures
    pub features: RosettaFeatures,
    /// Cluster ID from corpus analysis (Stage 3)
    pub cluster_id: Option<u32>,
}

impl Default for SourceMetadata112D {
    fn default() -> Self {
        Self {
            features: RosettaFeatures::default(),
            cluster_id: None,
        }
    }
}

impl SourceMetadata112D {
    /// Create SourceMetadata112D from RosettaFeatures
    pub fn from_features(features: &RosettaFeatures) -> Self {
        Self {
            features: features.clone(),
            cluster_id: None,
        }
    }

    /// Create SourceMetadata112D from RosettaFeatures with cluster ID
    pub fn from_features_with_cluster(features: &RosettaFeatures, cluster_id: u32) -> Self {
        Self {
            features: features.clone(),
            cluster_id: Some(cluster_id),
        }
    }

    /// Convert to 112D array for ML use
    pub fn to_array_112d(&self) -> [f32; 112] {
        self.features.to_array()
    }

    /// Compute quality score for exemplar selection
    ///
    /// Higher quality = better exemplar candidate.
    /// Factors: RMS energy, HNR, low jitter/shimmer.
    pub fn quality_score(&self) -> f32 {
        // Normalize RMS energy (0-1)
        let energy_score = self.features.rms_energy.clamp(0.0, 1.0);

        // Normalize HNR (typically 0-40 dB, higher is better)
        let hnr_score = (self.features.harmonic_to_noise_ratio / 40.0).min(1.0);

        // Penalize high jitter (typically 0-0.1, lower is better)
        let jitter_penalty = (self.features.jitter / 0.1).min(1.0);

        // Penalize high shimmer (typically 0-0.1, lower is better)
        let shimmer_penalty = (self.features.shimmer / 0.1).min(1.0);

        // Combine scores with weights
        let quality = (energy_score * 0.3)
            + (hnr_score * 0.4)
            + ((1.0 - jitter_penalty) * 0.15)
            + ((1.0 - shimmer_penalty) * 0.15);

        quality.clamp(0.0, 1.0)
    }

    // Delegate accessors to underlying features for convenience
    pub fn mean_f0_hz(&self) -> f32 {
        self.features.mean_f0_hz
    }
    pub fn duration_ms(&self) -> f32 {
        self.features.duration_ms
    }
    pub fn rms_energy(&self) -> f32 {
        self.features.rms_energy
    }
}

// =============================================================================
// EXEMPLAR ENTRY - Audio + Metadata per Cluster
// =============================================================================

/// Entry in the ExemplarManager containing audio and metadata for a cluster
#[derive(Debug, Clone)]
pub struct ExemplarEntry {
    /// Cluster ID from corpus analysis
    pub cluster_id: u32,
    /// Audio samples for this exemplar
    pub audio: Vec<f32>,
    /// 112D feature metadata
    pub metadata: SourceMetadata112D,
}

// =============================================================================
// EXEMPLAR MANAGER - Best Audio per Cluster
// =============================================================================

/// Manages exemplars (best audio samples) for each cluster ID
///
/// When multiple audio samples are registered for the same cluster,
/// the one with the highest quality score is kept.
#[derive(Debug, Clone, Default)]
pub struct ExemplarManager {
    /// Map from cluster ID to exemplar entry
    exemplars: HashMap<u32, ExemplarEntry>,
}

impl ExemplarManager {
    /// Create a new empty ExemplarManager
    pub fn new() -> Self {
        Self {
            exemplars: HashMap::new(),
        }
    }

    /// Register an exemplar for a cluster
    ///
    /// If an exemplar already exists for this cluster, the one with
    /// higher quality score is kept.
    pub fn register_exemplar(&mut self, cluster_id: u32, audio: Vec<f32>, features: RosettaFeatures) {
        let metadata = SourceMetadata112D::from_features_with_cluster(&features, cluster_id);
        let quality = metadata.quality_score();

        // Check if we should replace existing exemplar
        if let Some(existing) = self.exemplars.get(&cluster_id) {
            if existing.metadata.quality_score() >= quality {
                // Keep existing, it's higher quality
                return;
            }
        }

        // Insert or replace with new exemplar
        self.exemplars.insert(
            cluster_id,
            ExemplarEntry {
                cluster_id,
                audio,
                metadata,
            },
        );
    }

    /// Get an exemplar by cluster ID
    pub fn get_exemplar(&self, cluster_id: u32) -> Option<&ExemplarEntry> {
        self.exemplars.get(&cluster_id)
    }

    /// Get the number of stored exemplars
    pub fn len(&self) -> usize {
        self.exemplars.len()
    }

    /// Check if the manager is empty
    pub fn is_empty(&self) -> bool {
        self.exemplars.is_empty()
    }

    /// Clear all exemplars
    pub fn clear(&mut self) {
        self.exemplars.clear();
    }

    /// Get all cluster IDs
    pub fn cluster_ids(&self) -> Vec<u32> {
        self.exemplars.keys().copied().collect()
    }
}

// =============================================================================
// SYNTHESIS TIMELINE - Events for Synthesis
// =============================================================================

/// A single event in the synthesis timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticTimelineEvent {
    /// Cluster ID to synthesize
    pub cluster_id: u32,
    /// Start time in milliseconds
    pub start_time_ms: f64,
    /// Duration in milliseconds
    pub duration_ms: f64,
    /// Amplitude (0.0-1.0)
    pub amplitude: f32,
}

/// Timeline of semantic events for synthesis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SynthesisTimeline {
    /// Ordered list of timeline events
    events: Vec<SemanticTimelineEvent>,
}

impl SynthesisTimeline {
    /// Create a new empty timeline
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Add an event to the timeline
    pub fn add_event(&mut self, event: SemanticTimelineEvent) {
        self.events.push(event);
        // Sort by start time
        self.events.sort_by(|a, b| {
            a.start_time_ms
                .partial_cmp(&b.start_time_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get the number of events
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if the timeline is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get the total duration in milliseconds
    pub fn total_duration_ms(&self) -> f64 {
        self.events
            .iter()
            .map(|e| e.start_time_ms + e.duration_ms)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    /// Get all events
    pub fn events(&self) -> &[SemanticTimelineEvent] {
        &self.events
    }

    /// Get events in a time range [start_ms, end_ms)
    pub fn get_events_in_range(&self, start_ms: f64, end_ms: f64) -> Vec<&SemanticTimelineEvent> {
        self.events
            .iter()
            .filter(|e| e.start_time_ms < end_ms && (e.start_time_ms + e.duration_ms) > start_ms)
            .collect()
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

// =============================================================================
// SYNTHESIS CONFIG
// =============================================================================

/// Configuration for the CachedGranularSynthesizer
#[derive(Debug, Clone)]
pub struct SynthesisConfig112D {
    /// Sample rate for synthesis output
    pub sample_rate: u32,
    /// Crossfade duration between grains in milliseconds
    pub crossfade_ms: f32,
    /// Maximum concurrent grains
    pub max_grains: usize,
}

impl Default for SynthesisConfig112D {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            crossfade_ms: 10.0,
            max_grains: 32,
        }
    }
}

// =============================================================================
// CACHED GRANULAR SYNTHESIZER
// =============================================================================

/// Source entry for the synthesizer
#[derive(Debug, Clone)]
struct SourceEntry {
    audio: Vec<f32>,
    metadata: SourceMetadata112D,
}

/// Cached Granular Synthesizer for semantic reconstruction
///
/// Registers audio sources with 112D metadata and synthesizes output
/// from synthesis timelines.
pub struct CachedGranularSynthesizer {
    config: SynthesisConfig112D,
    sources: HashMap<u32, SourceEntry>,
}

impl CachedGranularSynthesizer {
    /// Create a new synthesizer with the given configuration
    pub fn new(config: SynthesisConfig112D) -> Self {
        Self {
            config,
            sources: HashMap::new(),
        }
    }

    /// Register a source with 112D metadata
    pub fn register_source(&mut self, cluster_id: u32, audio: Vec<f32>, metadata: SourceMetadata112D) {
        self.sources.insert(cluster_id, SourceEntry { audio, metadata });
    }

    /// Get the number of registered sources
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Clear all sources
    pub fn clear_sources(&mut self) {
        self.sources.clear();
    }

    /// Synthesize audio from a timeline
    ///
    /// This is an async method that produces audio from the timeline events.
    /// Each event triggers playback of the corresponding source audio.
    pub async fn synthesize_timeline(&self, timeline: &SynthesisTimeline) -> anyhow::Result<Vec<f32>> {
        if timeline.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate total samples needed
        let total_duration_ms = timeline.total_duration_ms();
        let total_samples = ((total_duration_ms / 1000.0) * self.config.sample_rate as f64) as usize;

        // Create output buffer with small padding for last event
        let mut output = vec![0.0f32; total_samples + 4800];

        // Process each event
        for event in timeline.events() {
            // Get source audio
            let source = match self.sources.get(&event.cluster_id) {
                Some(s) => s,
                None => continue, // Skip missing sources gracefully
            };

            // Calculate sample positions
            let start_sample = ((event.start_time_ms / 1000.0) * self.config.sample_rate as f64) as usize;
            let duration_samples = ((event.duration_ms / 1000.0) * self.config.sample_rate as f64) as usize;

            // Copy source audio to output (with stretching/padding as needed)
            let copy_len = duration_samples
                .min(source.audio.len())
                .min(output.len() - start_sample);

            for i in 0..copy_len {
                output[start_sample + i] += source.audio[i % source.audio.len()] * event.amplitude;
            }
        }

        Ok(output)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_metadata_default() {
        let metadata = SourceMetadata112D::default();
        assert_eq!(metadata.cluster_id, None);
        assert!(metadata.quality_score() >= 0.0);
    }

    #[test]
    fn test_source_metadata_112d_array() {
        let metadata = SourceMetadata112D::default();
        let arr = metadata.to_array_112d();
        assert_eq!(arr.len(), 112);
    }

    #[test]
    fn test_source_metadata_from_features() {
        let features = RosettaFeatures::default();
        let metadata = SourceMetadata112D::from_features(&features);
        assert_eq!(metadata.mean_f0_hz(), features.mean_f0_hz);
        assert_eq!(metadata.duration_ms(), features.duration_ms);
    }

    #[test]
    fn test_source_metadata_with_cluster() {
        let features = RosettaFeatures::default();
        let metadata = SourceMetadata112D::from_features_with_cluster(&features, 42);
        assert_eq!(metadata.cluster_id, Some(42));
    }

    #[test]
    fn test_quality_score_high_quality() {
        let mut features = RosettaFeatures::default();
        features.rms_energy = 0.8;
        features.harmonic_to_noise_ratio = 30.0;
        features.jitter = 0.01;
        features.shimmer = 0.01;
        let metadata = SourceMetadata112D::from_features(&features);
        assert!(metadata.quality_score() > 0.7);
    }

    #[test]
    fn test_quality_score_low_quality() {
        let mut features = RosettaFeatures::default();
        features.rms_energy = 0.2;
        features.harmonic_to_noise_ratio = 5.0;
        features.jitter = 0.1;
        features.shimmer = 0.1;
        let metadata = SourceMetadata112D::from_features(&features);
        assert!(metadata.quality_score() < 0.5);
    }

    #[test]
    fn test_exemplar_manager_creation() {
        let manager = ExemplarManager::new();
        assert_eq!(manager.len(), 0);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_exemplar_manager_register() {
        let mut manager = ExemplarManager::new();
        let audio = vec![0.5; 100];
        let features = RosettaFeatures::default();
        manager.register_exemplar(1, audio, features);
        assert_eq!(manager.len(), 1);
        assert!(manager.get_exemplar(1).is_some());
    }

    #[test]
    fn test_exemplar_manager_quality_selection() {
        let mut manager = ExemplarManager::new();

        // Register low quality first
        let audio_low = vec![0.1; 100];
        let mut features_low = RosettaFeatures::default();
        features_low.rms_energy = 0.2;
        features_low.harmonic_to_noise_ratio = 5.0;
        manager.register_exemplar(1, audio_low.clone(), features_low);

        // Register high quality - should replace
        let audio_high = vec![0.9; 100];
        let mut features_high = RosettaFeatures::default();
        features_high.rms_energy = 0.8;
        features_high.harmonic_to_noise_ratio = 30.0;
        manager.register_exemplar(1, audio_high.clone(), features_high);

        let entry = manager.get_exemplar(1).unwrap();
        assert_eq!(entry.audio, audio_high);
    }

    #[test]
    fn test_synthesis_timeline_creation() {
        let timeline = SynthesisTimeline::new();
        assert!(timeline.is_empty());
        assert_eq!(timeline.len(), 0);
    }

    #[test]
    fn test_synthesis_timeline_add_event() {
        let mut timeline = SynthesisTimeline::new();
        timeline.add_event(SemanticTimelineEvent {
            cluster_id: 1,
            start_time_ms: 0.0,
            duration_ms: 100.0,
            amplitude: 1.0,
        });
        assert_eq!(timeline.len(), 1);
        assert!(!timeline.is_empty());
    }

    #[test]
    fn test_synthesis_timeline_duration() {
        let mut timeline = SynthesisTimeline::new();
        timeline.add_event(SemanticTimelineEvent {
            cluster_id: 1,
            start_time_ms: 0.0,
            duration_ms: 100.0,
            amplitude: 1.0,
        });
        timeline.add_event(SemanticTimelineEvent {
            cluster_id: 2,
            start_time_ms: 100.0,
            duration_ms: 50.0,
            amplitude: 1.0,
        });
        assert_eq!(timeline.total_duration_ms(), 150.0);
    }

    #[test]
    fn test_cached_granular_synthesizer_creation() {
        let config = SynthesisConfig112D::default();
        let synth = CachedGranularSynthesizer::new(config);
        assert_eq!(synth.source_count(), 0);
    }

    #[test]
    fn test_cached_granular_synthesizer_register() {
        let config = SynthesisConfig112D::default();
        let mut synth = CachedGranularSynthesizer::new(config);
        let audio = vec![0.5; 100];
        let metadata = SourceMetadata112D::default();
        synth.register_source(1, audio, metadata);
        assert_eq!(synth.source_count(), 1);
    }

    #[tokio::test]
    async fn test_synthesize_timeline() {
        let config = SynthesisConfig112D::default();
        let mut synth = CachedGranularSynthesizer::new(config);

        // Register sources
        let audio = vec![0.5; 4800];
        let metadata = SourceMetadata112D::default();
        synth.register_source(1, audio.clone(), metadata.clone());
        synth.register_source(2, audio.clone(), metadata);

        // Create timeline
        let mut timeline = SynthesisTimeline::new();
        timeline.add_event(SemanticTimelineEvent {
            cluster_id: 1,
            start_time_ms: 0.0,
            duration_ms: 100.0,
            amplitude: 1.0,
        });
        timeline.add_event(SemanticTimelineEvent {
            cluster_id: 2,
            start_time_ms: 100.0,
            duration_ms: 100.0,
            amplitude: 1.0,
        });

        let result = synth.synthesize_timeline(&timeline).await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(!output.is_empty());
    }
}
