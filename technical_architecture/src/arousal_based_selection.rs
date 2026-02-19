//! Arousal-Based Source Selection Module
//!
//! Matches synthesis phrase selection to the emotional intensity (arousal)
//! of the communication context. This ensures synthesized responses match
//! not just the content but also the emotional urgency.
//!
//! Key Concept: High-arousal calls (alarm, distress) should be synthesized
//! with high-arousal source material, not calm neutral phrases.
//!
//! Arousal Tags:
//! - Low: Resting, relaxed states (e.g., contact calls while foraging)
//! - Neutral: Normal social interaction
//! - High: Alert, excited states (e.g., food discovery, territorial)
//! - Urgent: Emergency, distress (e.g., predator alarm, infant distress)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// Arousal level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArousalLevel {
    /// Resting, relaxed states (e.g., contact calls while foraging)
    Low,
    /// Normal social interaction
    Neutral,
    /// Alert, excited states (e.g., food discovery, territorial)
    High,
    /// Emergency, distress (e.g., predator alarm, infant distress)
    Urgent,
}

impl Default for ArousalLevel {
    fn default() -> Self {
        ArousalLevel::Neutral
    }
}

impl ArousalLevel {
    /// Get numeric value for arousal level (0.0 - 1.0)
    pub fn to_value(&self) -> f64 {
        match self {
            ArousalLevel::Low => 0.2,
            ArousalLevel::Neutral => 0.4,
            ArousalLevel::High => 0.7,
            ArousalLevel::Urgent => 1.0,
        }
    }

    /// Convert from numeric value to arousal level
    pub fn from_value(value: f64) -> Self {
        if value < 0.3 {
            ArousalLevel::Low
        } else if value < 0.55 {
            ArousalLevel::Neutral
        } else if value < 0.85 {
            ArousalLevel::High
        } else {
            ArousalLevel::Urgent
        }
    }
}

/// Source phrase with arousal metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArousalTaggedSource {
    /// Unique identifier for the source
    pub source_id: String,
    /// Species this source belongs to
    pub species: String,
    /// Call type (e.g., "phee", "trill", "tsik")
    pub call_type: String,
    /// Arousal level tag
    pub arousal: ArousalLevel,
    /// Acoustic features for similarity matching
    pub features: Vec<f64>,
    /// Optional confidence score for the arousal tag
    pub arousal_confidence: Option<f64>,
}

/// Inferred arousal state from input analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredArousalState {
    /// Current arousal level
    pub level: ArousalLevel,
    /// Confidence in the inference (0-1)
    pub confidence: f64,
    /// Acoustic features that drove the inference
    pub evidence: ArousalEvidence,
    /// Timestamp of inference
    pub timestamp_ms: u64,
}

/// Acoustic evidence for arousal inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArousalEvidence {
    /// F0 mean (higher = more aroused)
    pub f0_mean: f64,
    /// F0 variability (higher variance = more aroused)
    pub f0_variability: f64,
    /// Call rate (faster = more aroused)
    pub call_rate: f64,
    /// Duration (shorter bursts = more urgent)
    pub duration_ms: f64,
    /// Energy/amplitude (louder = more aroused)
    pub energy: f64,
    /// Spectral bandwidth (wider = more aroused)
    pub spectral_bandwidth: f64,
}

impl Default for ArousalEvidence {
    fn default() -> Self {
        Self {
            f0_mean: 6000.0,
            f0_variability: 500.0,
            call_rate: 1.0,
            duration_ms: 200.0,
            energy: 0.5,
            spectral_bandwidth: 2000.0,
        }
    }
}

/// Result of source selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSelectionResult {
    /// Selected source
    pub source: ArousalTaggedSource,
    /// Arousal match score (0-1, higher = better match)
    pub arousal_match_score: f64,
    /// Feature similarity score (0-1, higher = more similar)
    pub feature_similarity: f64,
    /// Combined score (weighted average)
    pub combined_score: f64,
    /// Whether the arousal levels match exactly
    pub arousal_exact_match: bool,
}

/// Configuration for arousal-based selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArousalSelectionConfig {
    /// Weight for arousal matching in combined score
    pub arousal_weight: f64,
    /// Weight for feature similarity in combined score
    pub similarity_weight: f64,
    /// Whether to allow selection from adjacent arousal levels
    pub allow_adjacent_arousal: bool,
    /// Minimum arousal match score to consider a match
    pub min_arousal_score: f64,
    /// Species-specific arousal thresholds
    pub species_thresholds: HashMap<String, ArousalThresholds>,
}

impl Default for ArousalSelectionConfig {
    fn default() -> Self {
        let mut species_thresholds = HashMap::new();

        // Marmoset thresholds (based on literature)
        species_thresholds.insert(
            "marmoset".to_string(),
            ArousalThresholds {
                f0_low: 5000.0,
                f0_high: 10000.0,
                f0_urgent: 14000.0,
                energy_low: 0.3,
                energy_high: 0.6,
                energy_urgent: 0.85,
                rate_low: 0.5,
                rate_high: 2.0,
                rate_urgent: 4.0,
            },
        );

        // Bat thresholds
        species_thresholds.insert(
            "bat".to_string(),
            ArousalThresholds {
                f0_low: 20000.0,
                f0_high: 45000.0,
                f0_urgent: 70000.0,
                energy_low: 0.25,
                energy_high: 0.55,
                energy_urgent: 0.8,
                rate_low: 0.3,
                rate_high: 1.5,
                rate_urgent: 3.0,
            },
        );

        Self {
            arousal_weight: 0.6,
            similarity_weight: 0.4,
            allow_adjacent_arousal: true,
            min_arousal_score: 0.3,
            species_thresholds,
        }
    }
}

/// Species-specific arousal thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArousalThresholds {
    /// F0 threshold for low arousal
    pub f0_low: f64,
    /// F0 threshold for high arousal
    pub f0_high: f64,
    /// F0 threshold for urgent arousal
    pub f0_urgent: f64,
    /// Energy threshold for low arousal
    pub energy_low: f64,
    /// Energy threshold for high arousal
    pub energy_high: f64,
    /// Energy threshold for urgent arousal
    pub energy_urgent: f64,
    /// Call rate threshold for low arousal
    pub rate_low: f64,
    /// Call rate threshold for high arousal
    pub rate_high: f64,
    /// Call rate threshold for urgent arousal
    pub rate_urgent: f64,
}

// ============================================================================
// AROUSAL INFERRER
// ============================================================================

/// Infers arousal state from acoustic features
pub struct ArousalInferrer {
    config: ArousalSelectionConfig,
}

impl ArousalInferrer {
    /// Create new inferrer with default config
    pub fn new() -> Self {
        Self::with_config(ArousalSelectionConfig::default())
    }

    /// Create new inferrer with custom config
    pub fn with_config(config: ArousalSelectionConfig) -> Self {
        Self { config }
    }

    /// Infer arousal state from acoustic evidence
    pub fn infer_arousal(
        &self,
        species: &str,
        evidence: &ArousalEvidence,
        timestamp_ms: u64,
    ) -> InferredArousalState {
        let thresholds = self
            .config
            .species_thresholds
            .get(species)
            .cloned()
            .unwrap_or_else(|| ArousalThresholds {
                f0_low: 4000.0,
                f0_high: 8000.0,
                f0_urgent: 12000.0,
                energy_low: 0.25,
                energy_high: 0.5,
                energy_urgent: 0.75,
                rate_low: 0.5,
                rate_high: 1.5,
                rate_urgent: 3.0,
            });

        // Calculate arousal scores for each dimension
        let f0_score = self.score_f0(evidence.f0_mean, &thresholds);
        let variability_score = self.score_variability(evidence.f0_variability);
        let rate_score = self.score_rate(evidence.call_rate, &thresholds);
        let duration_score = self.score_duration(evidence.duration_ms);
        let energy_score = self.score_energy(evidence.energy, &thresholds);
        let bandwidth_score = self.score_bandwidth(evidence.spectral_bandwidth);

        // Weighted average of scores
        let weights = [0.25, 0.15, 0.20, 0.10, 0.20, 0.10];
        let scores = [
            f0_score,
            variability_score,
            rate_score,
            duration_score,
            energy_score,
            bandwidth_score,
        ];

        let combined_score: f64 = weights.iter().zip(scores.iter()).map(|(w, s)| w * s).sum();

        // Map to arousal level
        let level = ArousalLevel::from_value(combined_score);

        // Calculate confidence based on consistency of evidence
        let variance = self.calculate_evidence_variance(&scores);
        let confidence = 1.0 / (1.0 + variance); // Lower variance = higher confidence

        InferredArousalState {
            level,
            confidence,
            evidence: evidence.clone(),
            timestamp_ms,
        }
    }

    /// Score F0 mean
    fn score_f0(&self, f0: f64, thresholds: &ArousalThresholds) -> f64 {
        if f0 < thresholds.f0_low {
            0.2
        } else if f0 < thresholds.f0_high {
            0.3 + 0.2 * (f0 - thresholds.f0_low) / (thresholds.f0_high - thresholds.f0_low)
        } else if f0 < thresholds.f0_urgent {
            0.5 + 0.3 * (f0 - thresholds.f0_high) / (thresholds.f0_urgent - thresholds.f0_high)
        } else {
            0.8 + 0.2 * ((f0 - thresholds.f0_urgent) / thresholds.f0_urgent).min(1.0)
        }
    }

    /// Score F0 variability
    fn score_variability(&self, variability: f64) -> f64 {
        // Higher variability often indicates higher arousal
        (variability / 1000.0).min(1.0)
    }

    /// Score call rate
    fn score_rate(&self, rate: f64, thresholds: &ArousalThresholds) -> f64 {
        if rate < thresholds.rate_low {
            0.2
        } else if rate < thresholds.rate_high {
            0.3 + 0.2 * (rate - thresholds.rate_low) / (thresholds.rate_high - thresholds.rate_low)
        } else if rate < thresholds.rate_urgent {
            0.5 + 0.3 * (rate - thresholds.rate_high)
                / (thresholds.rate_urgent - thresholds.rate_high)
        } else {
            0.8 + 0.2 * ((rate - thresholds.rate_urgent) / thresholds.rate_urgent).min(1.0)
        }
    }

    /// Score duration (shorter = more urgent)
    fn score_duration(&self, duration_ms: f64) -> f64 {
        if duration_ms > 500.0 {
            0.2 // Long calls = low arousal
        } else if duration_ms > 200.0 {
            0.3 + 0.2 * (500.0 - duration_ms) / 300.0
        } else if duration_ms > 100.0 {
            0.5 + 0.3 * (200.0 - duration_ms) / 100.0
        } else {
            0.8 + 0.2 * ((100.0 - duration_ms) / 100.0).max(0.0)
        }
    }

    /// Score energy
    fn score_energy(&self, energy: f64, thresholds: &ArousalThresholds) -> f64 {
        if energy < thresholds.energy_low {
            0.2
        } else if energy < thresholds.energy_high {
            0.3 + 0.2 * (energy - thresholds.energy_low)
                / (thresholds.energy_high - thresholds.energy_low)
        } else if energy < thresholds.energy_urgent {
            0.5 + 0.3 * (energy - thresholds.energy_high)
                / (thresholds.energy_urgent - thresholds.energy_high)
        } else {
            0.8 + 0.2
                * ((energy - thresholds.energy_urgent) / (1.0 - thresholds.energy_urgent)).min(1.0)
        }
    }

    /// Score spectral bandwidth
    fn score_bandwidth(&self, bandwidth: f64) -> f64 {
        // Wider bandwidth often indicates higher arousal
        (bandwidth / 4000.0).min(1.0)
    }

    /// Calculate variance of evidence scores (for confidence)
    fn calculate_evidence_variance(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        variance
    }
}

impl Default for ArousalInferrer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// AROUSAL-BASED SOURCE SELECTOR
// ============================================================================

/// Selects source phrases based on arousal matching
pub struct ArousalBasedSelector {
    config: ArousalSelectionConfig,
    sources: Vec<ArousalTaggedSource>,
    sources_by_arousal: HashMap<ArousalLevel, Vec<usize>>,
    inferrer: ArousalInferrer,
}

impl ArousalBasedSelector {
    /// Create new selector with default config
    pub fn new() -> Self {
        Self::with_config(ArousalSelectionConfig::default())
    }

    /// Create new selector with custom config
    pub fn with_config(config: ArousalSelectionConfig) -> Self {
        let inferrer = ArousalInferrer::with_config(config.clone());
        Self {
            config,
            sources: Vec::new(),
            sources_by_arousal: HashMap::new(),
            inferrer,
        }
    }

    /// Add a source to the library
    pub fn add_source(&mut self, source: ArousalTaggedSource) {
        let idx = self.sources.len();
        let arousal = source.arousal;

        self.sources_by_arousal
            .entry(arousal)
            .or_insert_with(Vec::new)
            .push(idx);

        self.sources.push(source);
    }

    /// Get the number of sources in the library
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Get sources by arousal level
    pub fn sources_by_arousal(&self, level: ArousalLevel) -> Vec<&ArousalTaggedSource> {
        self.sources_by_arousal
            .get(&level)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&i| self.sources.get(i))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Select best source matching the inferred arousal state
    pub fn select_source(
        &self,
        target_features: &[f64],
        inferred_state: &InferredArousalState,
    ) -> Option<SourceSelectionResult> {
        let target_arousal = inferred_state.level;

        // Get candidate sources
        let candidates = self.get_candidate_sources(target_arousal);

        if candidates.is_empty() {
            return None;
        }

        // Score each candidate
        let mut best_result: Option<SourceSelectionResult> = None;
        let mut best_score = 0.0;

        for source in candidates {
            let result = self.score_source(source, target_features, target_arousal);

            if result.combined_score > best_score {
                best_score = result.combined_score;
                best_result = Some(result);
            }
        }

        best_result
    }

    /// Get candidate sources considering adjacent arousal levels
    fn get_candidate_sources(&self, target: ArousalLevel) -> Vec<&ArousalTaggedSource> {
        let mut candidates = Vec::new();

        // Always include exact matches
        if let Some(indices) = self.sources_by_arousal.get(&target) {
            for &i in indices {
                candidates.push(&self.sources[i]);
            }
        }

        // Include adjacent arousal levels if allowed
        if self.config.allow_adjacent_arousal {
            for adjacent in self.get_adjacent_arousal_levels(target) {
                if let Some(indices) = self.sources_by_arousal.get(&adjacent) {
                    for &i in indices {
                        candidates.push(&self.sources[i]);
                    }
                }
            }
        }

        candidates
    }

    /// Get adjacent arousal levels
    fn get_adjacent_arousal_levels(&self, level: ArousalLevel) -> Vec<ArousalLevel> {
        match level {
            ArousalLevel::Low => vec![ArousalLevel::Neutral],
            ArousalLevel::Neutral => vec![ArousalLevel::Low, ArousalLevel::High],
            ArousalLevel::High => vec![ArousalLevel::Neutral, ArousalLevel::Urgent],
            ArousalLevel::Urgent => vec![ArousalLevel::High],
        }
    }

    /// Score a source against target features and arousal
    fn score_source(
        &self,
        source: &ArousalTaggedSource,
        target_features: &[f64],
        target_arousal: ArousalLevel,
    ) -> SourceSelectionResult {
        // Calculate arousal match score
        let arousal_match_score = self.calculate_arousal_match(source.arousal, target_arousal);

        // Calculate feature similarity (cosine similarity)
        let feature_similarity = self.cosine_similarity(&source.features, target_features);

        // Combined score
        let combined_score = self.config.arousal_weight * arousal_match_score
            + self.config.similarity_weight * feature_similarity;

        SourceSelectionResult {
            source: source.clone(),
            arousal_match_score,
            feature_similarity,
            combined_score,
            arousal_exact_match: source.arousal == target_arousal,
        }
    }

    /// Calculate arousal match score
    fn calculate_arousal_match(
        &self,
        source_arousal: ArousalLevel,
        target_arousal: ArousalLevel,
    ) -> f64 {
        if source_arousal == target_arousal {
            1.0
        } else {
            // Penalize based on distance from target
            let distance = (source_arousal.to_value() - target_arousal.to_value()).abs();
            1.0 - distance * 0.5
        }
    }

    /// Cosine similarity between feature vectors
    fn cosine_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.is_empty() || b.is_empty() || a.len() != b.len() {
            return 0.0;
        }

        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            (dot / (norm_a * norm_b)).max(0.0)
        } else {
            0.0
        }
    }

    /// Get the inferrer for external use
    pub fn inferrer(&self) -> &ArousalInferrer {
        &self.inferrer
    }

    /// Get the configuration
    pub fn config(&self) -> &ArousalSelectionConfig {
        &self.config
    }

    /// Clear all sources
    pub fn clear_sources(&mut self) {
        self.sources.clear();
        self.sources_by_arousal.clear();
    }
}

impl Default for ArousalBasedSelector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arousal_level_values() {
        assert!((ArousalLevel::Low.to_value() - 0.2).abs() < 0.01);
        assert!((ArousalLevel::Neutral.to_value() - 0.4).abs() < 0.01);
        assert!((ArousalLevel::High.to_value() - 0.7).abs() < 0.01);
        assert!((ArousalLevel::Urgent.to_value() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_arousal_level_from_value() {
        assert_eq!(ArousalLevel::from_value(0.1), ArousalLevel::Low);
        assert_eq!(ArousalLevel::from_value(0.4), ArousalLevel::Neutral);
        assert_eq!(ArousalLevel::from_value(0.7), ArousalLevel::High);
        assert_eq!(ArousalLevel::from_value(0.95), ArousalLevel::Urgent);
    }

    #[test]
    fn test_arousal_inferrer_creation() {
        let inferrer = ArousalInferrer::new();
        assert!(inferrer.config.species_thresholds.contains_key("marmoset"));
    }

    #[test]
    fn test_infer_low_arousal() {
        let inferrer = ArousalInferrer::new();
        let evidence = ArousalEvidence {
            f0_mean: 4000.0,            // Low F0
            f0_variability: 200.0,      // Low variability
            call_rate: 0.3,             // Slow rate
            duration_ms: 600.0,         // Long duration
            energy: 0.2,                // Low energy
            spectral_bandwidth: 1000.0, // Narrow bandwidth
        };

        let state = inferrer.infer_arousal("marmoset", &evidence, 0);
        assert_eq!(state.level, ArousalLevel::Low);
    }

    #[test]
    fn test_infer_urgent_arousal() {
        let inferrer = ArousalInferrer::new();
        let evidence = ArousalEvidence {
            f0_mean: 16000.0,           // Very high F0
            f0_variability: 2000.0,     // High variability
            call_rate: 5.0,             // Very fast rate
            duration_ms: 50.0,          // Short duration
            energy: 0.95,               // High energy
            spectral_bandwidth: 5000.0, // Wide bandwidth
        };

        let state = inferrer.infer_arousal("marmoset", &evidence, 0);
        assert_eq!(state.level, ArousalLevel::Urgent);
    }

    #[test]
    fn test_infer_neutral_arousal() {
        let inferrer = ArousalInferrer::new();
        let evidence = ArousalEvidence {
            f0_mean: 7000.0, // Medium F0
            f0_variability: 500.0,
            call_rate: 1.0,
            duration_ms: 300.0,
            energy: 0.45, // Medium energy
            spectral_bandwidth: 2500.0,
        };

        let state = inferrer.infer_arousal("marmoset", &evidence, 0);
        assert_eq!(state.level, ArousalLevel::Neutral);
    }

    #[test]
    fn test_infer_high_arousal() {
        let inferrer = ArousalInferrer::new();
        let evidence = ArousalEvidence {
            f0_mean: 12000.0, // High F0
            f0_variability: 800.0,
            call_rate: 2.5,
            duration_ms: 150.0,
            energy: 0.65,
            spectral_bandwidth: 3500.0,
        };

        let state = inferrer.infer_arousal("marmoset", &evidence, 0);
        assert_eq!(state.level, ArousalLevel::High);
    }

    #[test]
    fn test_selector_creation() {
        let selector = ArousalBasedSelector::new();
        assert_eq!(selector.source_count(), 0);
    }

    #[test]
    fn test_add_sources() {
        let mut selector = ArousalBasedSelector::new();

        selector.add_source(ArousalTaggedSource {
            source_id: "s1".to_string(),
            species: "marmoset".to_string(),
            call_type: "phee".to_string(),
            arousal: ArousalLevel::Neutral,
            features: vec![1.0, 0.0, 0.0],
            arousal_confidence: Some(0.9),
        });

        selector.add_source(ArousalTaggedSource {
            source_id: "s2".to_string(),
            species: "marmoset".to_string(),
            call_type: "tsik".to_string(),
            arousal: ArousalLevel::High,
            features: vec![0.0, 1.0, 0.0],
            arousal_confidence: Some(0.85),
        });

        assert_eq!(selector.source_count(), 2);
        assert_eq!(selector.sources_by_arousal(ArousalLevel::Neutral).len(), 1);
        assert_eq!(selector.sources_by_arousal(ArousalLevel::High).len(), 1);
    }

    #[test]
    fn test_select_matching_arousal() {
        let mut selector = ArousalBasedSelector::new();

        // Add sources with different arousal levels
        selector.add_source(ArousalTaggedSource {
            source_id: "low_1".to_string(),
            species: "marmoset".to_string(),
            call_type: "trill".to_string(),
            arousal: ArousalLevel::Low,
            features: vec![0.1, 0.9, 0.0],
            arousal_confidence: None,
        });

        selector.add_source(ArousalTaggedSource {
            source_id: "high_1".to_string(),
            species: "marmoset".to_string(),
            call_type: "tsik".to_string(),
            arousal: ArousalLevel::High,
            features: vec![0.9, 0.1, 0.0],
            arousal_confidence: None,
        });

        // Request high arousal - should get high_1
        let inferred = InferredArousalState {
            level: ArousalLevel::High,
            confidence: 0.9,
            evidence: ArousalEvidence::default(),
            timestamp_ms: 0,
        };

        let target_features = vec![0.95, 0.05, 0.0];
        let result = selector.select_source(&target_features, &inferred);

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.source.source_id, "high_1");
        assert!(result.arousal_exact_match);
        assert!(result.arousal_match_score > 0.8);
    }

    #[test]
    fn test_select_with_feature_similarity() {
        let mut selector = ArousalBasedSelector::new();

        // Add two sources with same arousal but different features
        selector.add_source(ArousalTaggedSource {
            source_id: "high_similar".to_string(),
            species: "marmoset".to_string(),
            call_type: "tsik".to_string(),
            arousal: ArousalLevel::High,
            features: vec![0.9, 0.1, 0.0],
            arousal_confidence: None,
        });

        selector.add_source(ArousalTaggedSource {
            source_id: "high_different".to_string(),
            species: "marmoset".to_string(),
            call_type: "tsik".to_string(),
            arousal: ArousalLevel::High,
            features: vec![0.1, 0.9, 0.0],
            arousal_confidence: None,
        });

        let inferred = InferredArousalState {
            level: ArousalLevel::High,
            confidence: 0.9,
            evidence: ArousalEvidence::default(),
            timestamp_ms: 0,
        };

        // Target features match first source
        let target_features = vec![0.95, 0.05, 0.0];
        let result = selector.select_source(&target_features, &inferred);

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.source.source_id, "high_similar");
        assert!(result.feature_similarity > 0.9);
    }

    #[test]
    fn test_adjacent_arousal_selection() {
        let mut selector = ArousalBasedSelector::new();

        // Add only neutral source
        selector.add_source(ArousalTaggedSource {
            source_id: "neutral_1".to_string(),
            species: "marmoset".to_string(),
            call_type: "phee".to_string(),
            arousal: ArousalLevel::Neutral,
            features: vec![1.0, 0.0, 0.0],
            arousal_confidence: None,
        });

        // Request high arousal - with adjacent allowed, should get neutral
        let inferred = InferredArousalState {
            level: ArousalLevel::High,
            confidence: 0.9,
            evidence: ArousalEvidence::default(),
            timestamp_ms: 0,
        };

        let target_features = vec![1.0, 0.0, 0.0];
        let result = selector.select_source(&target_features, &inferred);

        assert!(result.is_some());
        assert!(!result.unwrap().arousal_exact_match);
    }

    #[test]
    fn test_empty_library() {
        let selector = ArousalBasedSelector::new();

        let inferred = InferredArousalState {
            level: ArousalLevel::Neutral,
            confidence: 0.9,
            evidence: ArousalEvidence::default(),
            timestamp_ms: 0,
        };

        let result = selector.select_source(&[1.0, 0.0], &inferred);
        assert!(result.is_none());
    }

    #[test]
    fn test_clear_sources() {
        let mut selector = ArousalBasedSelector::new();

        selector.add_source(ArousalTaggedSource {
            source_id: "s1".to_string(),
            species: "marmoset".to_string(),
            call_type: "phee".to_string(),
            arousal: ArousalLevel::Neutral,
            features: vec![1.0],
            arousal_confidence: None,
        });

        assert_eq!(selector.source_count(), 1);
        selector.clear_sources();
        assert_eq!(selector.source_count(), 0);
    }

    #[test]
    fn test_serialization() {
        let source = ArousalTaggedSource {
            source_id: "test".to_string(),
            species: "marmoset".to_string(),
            call_type: "phee".to_string(),
            arousal: ArousalLevel::High,
            features: vec![1.0, 2.0, 3.0],
            arousal_confidence: Some(0.85),
        };

        let json = serde_json::to_string(&source).unwrap();
        let decoded: ArousalTaggedSource = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.source_id, "test");
        assert_eq!(decoded.arousal, ArousalLevel::High);
    }

    #[test]
    fn test_inferred_state_serialization() {
        let state = InferredArousalState {
            level: ArousalLevel::Urgent,
            confidence: 0.95,
            evidence: ArousalEvidence {
                f0_mean: 15000.0,
                f0_variability: 1000.0,
                call_rate: 4.0,
                duration_ms: 80.0,
                energy: 0.9,
                spectral_bandwidth: 4500.0,
            },
            timestamp_ms: 12345,
        };

        let json = serde_json::to_string(&state).unwrap();
        let decoded: InferredArousalState = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.level, ArousalLevel::Urgent);
        assert!((decoded.confidence - 0.95).abs() < 0.001);
        assert_eq!(decoded.timestamp_ms, 12345);
    }

    #[test]
    fn test_confidence_calculation() {
        let inferrer = ArousalInferrer::new();

        // Consistent evidence should give high confidence
        let consistent_evidence = ArousalEvidence {
            f0_mean: 15000.0, // All point to high arousal
            f0_variability: 1500.0,
            call_rate: 4.0,
            duration_ms: 80.0,
            energy: 0.9,
            spectral_bandwidth: 4500.0,
        };

        let consistent_state = inferrer.infer_arousal("marmoset", &consistent_evidence, 0);
        assert!(consistent_state.confidence > 0.5);

        // Mixed evidence should give lower confidence
        let mixed_evidence = ArousalEvidence {
            f0_mean: 5000.0,            // Low
            f0_variability: 1500.0,     // High
            call_rate: 0.3,             // Low
            duration_ms: 80.0,          // Urgent
            energy: 0.9,                // High
            spectral_bandwidth: 4500.0, // High
        };

        let mixed_state = inferrer.infer_arousal("marmoset", &mixed_evidence, 0);
        assert!(mixed_state.confidence < consistent_state.confidence);
    }

    #[test]
    fn test_bat_arousal_inference() {
        let inferrer = ArousalInferrer::new();

        // Bat high arousal should use bat thresholds
        let evidence = ArousalEvidence {
            f0_mean: 50000.0, // Mid-high for bat (20-100kHz range)
            f0_variability: 1000.0,
            call_rate: 2.0, // High rate
            duration_ms: 150.0,
            energy: 0.6, // Medium-high energy
            spectral_bandwidth: 5000.0,
        };

        let state = inferrer.infer_arousal("bat", &evidence, 0);
        assert_eq!(state.level, ArousalLevel::High);
    }
}
