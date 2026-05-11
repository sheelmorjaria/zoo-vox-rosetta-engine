//! Ethological Validation Framework
//! ================================
//!
//! Multi-factor validation system for evaluating agent responses in
//! ethological contexts. Computes MFAS (Multi-Factor Acceptance Score)
//! from behavioral metrics and statistical analysis.
//!
//! Features:
//! - MFAS computation with weighted factors
//! - Behavioral metrics tracking
//! - Response appropriateness scoring
//! - Statistical significance testing
//! - Longitudinal analysis
//!
//! Author: Zoo Vox Research Team
//! License: CC BY-ND 4.0 International

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// CORE DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/// Multi-Factor Acceptance Score (MFAS) for response validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiFactorAcceptanceScore {
    /// Overall composite score (0-1)
    pub overall_score: f32,
    /// Temporal appropriateness (response timing)
    pub temporal_score: f32,
    /// Acoustic appropriateness (sound characteristics)
    pub acoustic_score: f32,
    /// Social appropriateness (context awareness)
    pub social_score: f32,
    /// Ethological consistency (species-typical behavior)
    pub ethological_score: f32,
    /// Confidence in the assessment (0-1)
    pub confidence: f32,
    /// Number of factors that contributed
    pub factors_count: usize,
}

impl MultiFactorAcceptanceScore {
    pub fn new() -> Self {
        Self {
            overall_score: 0.5,
            temporal_score: 0.5,
            acoustic_score: 0.5,
            social_score: 0.5,
            ethological_score: 0.5,
            confidence: 0.0,
            factors_count: 0,
        }
    }

    /// Compute weighted overall score
    pub fn compute_overall(&mut self, weights: &MFASWeights) {
        let total_weight = weights.temporal
            + weights.acoustic
            + weights.social
            + weights.ethological;

        if total_weight > 0.0 {
            self.overall_score = (self.temporal_score * weights.temporal
                + self.acoustic_score * weights.acoustic
                + self.social_score * weights.social
                + self.ethological_score * weights.ethological)
                / total_weight;
        }

        self.confidence = (self.factors_count as f32 / 4.0).min(1.0);
    }

    /// Check if response is acceptable (score above threshold)
    pub fn is_acceptable(&self, threshold: f32) -> bool {
        self.overall_score >= threshold && self.confidence >= 0.5
    }
}

impl Default for MultiFactorAcceptanceScore {
    fn default() -> Self {
        Self::new()
    }
}

/// Weights for MFAS components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MFASWeights {
    pub temporal: f32,
    pub acoustic: f32,
    pub social: f32,
    pub ethological: f32,
}

impl Default for MFASWeights {
    fn default() -> Self {
        Self {
            temporal: 1.0,
            acoustic: 1.0,
            social: 1.0,
            ethological: 1.5, // Higher weight for ethological consistency
        }
    }
}

/// Behavioral metrics for a single interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralMetrics {
    /// Interaction timestamp
    pub timestamp: f64,
    /// Subject ID (who the response was directed at)
    pub subject_id: String,
    /// Agent ID (who generated the response)
    pub agent_id: String,
    /// Stimulus call type
    pub stimulus_type: String,
    /// Response call type
    pub response_type: String,
    /// Response latency (ms)
    pub response_latency_ms: f32,
    /// Response duration (ms)
    pub response_duration_ms: f32,
    /// F0 of response (Hz)
    pub response_f0: f32,
    /// Whether response was spatially directed
    pub spatial_directed: bool,
    /// Social context at time of response
    pub social_context: SocialContext,
}

/// Social context classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SocialContext {
    Isolation,
    Dyadic,
    SmallGroup,
    LargeGroup,
}

/// Validation session for tracking multiple interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSession {
    /// Session identifier
    pub session_id: String,
    /// Species being tested
    pub species: String,
    /// Start timestamp
    pub start_time: f64,
    /// End timestamp
    pub end_time: Option<f64>,
    /// All interaction metrics
    pub interactions: Vec<BehavioralMetrics>,
    /// Computed MFAS scores
    pub mfas_scores: Vec<MultiFactorAcceptanceScore>,
    /// Session-level statistics
    pub statistics: SessionStatistics,
}

/// Session-level aggregated statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatistics {
    /// Number of interactions
    pub interaction_count: usize,
    /// Mean MFAS score
    pub mean_mfas: f32,
    /// Standard deviation of MFAS
    pub std_mfas: f32,
    /// Acceptance rate (fraction above threshold)
    pub acceptance_rate: f32,
    /// Mean response latency
    pub mean_latency_ms: f32,
    /// Temporal consistency (0-1)
    pub temporal_consistency: f32,
}

/// Statistical test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    /// Test type
    pub test_type: StatisticalTestType,
    /// Test statistic value
    pub statistic: f32,
    /// P-value
    pub p_value: f32,
    /// Is result significant at alpha=0.05?
    pub is_significant: bool,
    /// Effect size (Cohen's d or similar)
    pub effect_size: f32,
}

/// Types of statistical tests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatisticalTestType {
    /// One-sample t-test
    OneSampleTTest,
    /// Two-sample t-test
    TwoSampleTTest,
    /// Paired t-test
    PairedTTest,
    /// Wilcoxon rank-sum
    WilcoxonRankSum,
    /// ANOVA
    ANOVA,
}

// ═══════════════════════════════════════════════════════════════════════════════
// ETHOLOGICAL VALIDATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Main validator for ethological assessment
pub struct EthologicalValidator {
    /// MFAS weights
    weights: MFASWeights,
    /// Acceptance threshold
    acceptance_threshold: f32,
    /// Species-specific parameters
    species_params: HashMap<String, SpeciesParameters>,
    /// Baseline statistics for comparison
    baselines: HashMap<String, BaselineStats>,
}

/// Species-specific validation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesParameters {
    /// Expected response latency range (ms)
    pub latency_range: (f32, f32),
    /// Expected F0 range (Hz)
    pub f0_range: (f32, f32),
    /// Expected call duration range (ms)
    pub duration_range: (f32, f32),
    /// Species-typical response patterns
    pub typical_patterns: Vec<String>,
}

/// Baseline statistics for "normal" behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineStats {
    /// Mean response latency
    pub mean_latency_ms: f32,
    /// Std dev of latency
    pub std_latency_ms: f32,
    /// Mean call duration
    pub mean_duration_ms: f32,
    /// Mean F0
    pub mean_f0: f32,
    /// Sample size
    pub n: usize,
}

impl EthologicalValidator {
    pub fn new() -> Self {
        Self {
            weights: MFASWeights::default(),
            acceptance_threshold: 0.7,
            species_params: HashMap::new(),
            baselines: HashMap::new(),
        }
    }

    /// Set MFAS weights
    pub fn with_weights(mut self, weights: MFASWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Set acceptance threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.acceptance_threshold = threshold;
        self
    }

    /// Register species parameters
    pub fn register_species(&mut self, species: String, params: SpeciesParameters) {
        self.species_params.insert(species, params);
    }

    /// Register baseline statistics
    pub fn register_baseline(&mut self, species: String, stats: BaselineStats) {
        self.baselines.insert(species, stats);
    }

    /// Validate a single interaction and compute MFAS
    pub fn validate_interaction(
        &self,
        metrics: &BehavioralMetrics,
        species: &str,
    ) -> MultiFactorAcceptanceScore {
        let species_params = self.species_params.get(species);
        let baseline = self.baselines.get(species);

        let mut mfas = MultiFactorAcceptanceScore::new();

        // Temporal score: response latency appropriateness
        mfas.temporal_score = self.compute_temporal_score(metrics, species_params);
        mfas.factors_count += 1;

        // Acoustic score: F0 and duration appropriateness
        mfas.acoustic_score = self.compute_acoustic_score(metrics, species_params);
        mfas.factors_count += 1;

        // Social score: context-appropriate response
        mfas.social_score = self.compute_social_score(metrics);
        mfas.factors_count += 1;

        // Ethological score: species-typical behavior
        mfas.ethological_score = self.compute_ethological_score(metrics, species_params, baseline);
        mfas.factors_count += 1;

        // Compute overall weighted score
        let mut mfas_copy = mfas.clone();
        mfas_copy.compute_overall(&self.weights);
        mfas_copy

    }

    /// Compute temporal appropriateness score
    fn compute_temporal_score(
        &self,
        metrics: &BehavioralMetrics,
        params: Option<&SpeciesParameters>,
    ) -> f32 {
        let expected_range = params
            .map(|p| p.latency_range)
            .unwrap_or((100.0, 500.0)); // Default 100-500ms

        // Score based on how close latency is to expected range
        if metrics.response_latency_ms >= expected_range.0
            && metrics.response_latency_ms <= expected_range.1
        {
            1.0
        } else if metrics.response_latency_ms < expected_range.0 {
            // Too fast - penalize
            1.0 - (expected_range.0 - metrics.response_latency_ms) / expected_range.0
        } else {
            // Too slow - penalize
            (1.0 - (metrics.response_latency_ms - expected_range.1) / expected_range.1).max(0.0)
        }
    }

    /// Compute acoustic appropriateness score
    fn compute_acoustic_score(
        &self,
        metrics: &BehavioralMetrics,
        params: Option<&SpeciesParameters>,
    ) -> f32 {
        let expected_f0_range = params
            .map(|p| p.f0_range)
            .unwrap_or((1000.0, 15000.0));

        let expected_duration_range = params
            .map(|p| p.duration_range)
            .unwrap_or((50.0, 500.0));

        // F0 score
        let f0_score = if metrics.response_f0 >= expected_f0_range.0
            && metrics.response_f0 <= expected_f0_range.1
        {
            1.0
        } else {
            0.5
        };

        // Duration score
        let duration_score = if metrics.response_duration_ms >= expected_duration_range.0
            && metrics.response_duration_ms <= expected_duration_range.1
        {
            1.0
        } else {
            0.5
        };

        (f0_score + duration_score) / 2.0
    }

    /// Compute social appropriateness score
    fn compute_social_score(&self, metrics: &BehavioralMetrics) -> f32 {
        match metrics.social_context {
            SocialContext::Isolation => {
                // In isolation, any response is appropriate
                0.8
            }
            SocialContext::Dyadic => {
                // Direct response expected
                if metrics.spatial_directed {
                    1.0
                } else {
                    0.6
                }
            }
            SocialContext::SmallGroup => {
                // Spatial targeting important
                if metrics.spatial_directed {
                    0.9
                } else {
                    0.5
                }
            }
            SocialContext::LargeGroup => {
                // More tolerant of non-directed responses
                0.7
            }
        }
    }

    /// Compute ethological consistency score
    fn compute_ethological_score(
        &self,
        metrics: &BehavioralMetrics,
        params: Option<&SpeciesParameters>,
        baseline: Option<&BaselineStats>,
    ) -> f32 {
        let mut score = 0.5;

        // Check if response type is species-typical
        if let Some(params) = params {
            if params.typical_patterns.contains(&metrics.response_type) {
                score += 0.3;
            }
        }

        // Compare with baseline statistics
        if let Some(baseline) = baseline {
            // Z-score for latency
            let latency_z = if baseline.std_latency_ms > 0.0 {
                ((metrics.response_latency_ms - baseline.mean_latency_ms) / baseline.std_latency_ms)
                    .abs()
            } else {
                0.0
            };

            // Penalize outliers (|z| > 2)
            let z_score = if latency_z < 2.0 {
                1.0 - (latency_z / 2.0)
            } else {
                0.0
            };

            score = (score + z_score) / 2.0;
        }

        score.min(1.0).max(0.0)
    }

    /// Process an entire validation session
    pub fn validate_session(&self, session: &ValidationSession) -> SessionStatistics {
        if session.interactions.is_empty() {
            return SessionStatistics {
                interaction_count: 0,
                mean_mfas: 0.0,
                std_mfas: 0.0,
                acceptance_rate: 0.0,
                mean_latency_ms: 0.0,
                temporal_consistency: 0.0,
            };
        }

        let scores: Vec<f32> = session
            .interactions
            .iter()
            .map(|m| {
                let mfas = self.validate_interaction(m, &session.species);
                mfas.overall_score
            })
            .collect();

        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores
            .iter()
            .map(|s| (s - mean).powi(2))
            .sum::<f32>()
            / scores.len() as f32;
        let std = variance.sqrt();

        let acceptance_count = scores
            .iter()
            .filter(|&&s| s >= self.acceptance_threshold)
            .count();

        let mean_latency = session
            .interactions
            .iter()
            .map(|m| m.response_latency_ms)
            .sum::<f32>()
            / session.interactions.len() as f32;

        // Temporal consistency: inverse of coefficient of variation
        let temporal_consistency = if mean > 0.0 {
            1.0 - (std / mean).min(1.0)
        } else {
            0.0
        };

        SessionStatistics {
            interaction_count: session.interactions.len(),
            mean_mfas: mean,
            std_mfas: std,
            acceptance_rate: acceptance_count as f32 / session.interactions.len() as f32,
            mean_latency_ms: mean_latency,
            temporal_consistency,
        }
    }

    /// Perform statistical test comparing session to baseline
    pub fn statistical_test(
        &self,
        session: &ValidationSession,
        test_type: StatisticalTestType,
    ) -> Option<StatisticalTest> {
        let baseline = self.baselines.get(&session.species)?;
        if session.interactions.is_empty() {
            return None;
        }

        let latencies: Vec<f32> = session
            .interactions
            .iter()
            .map(|m| m.response_latency_ms)
            .collect();

        let (statistic, p_value, effect_size) = match test_type {
            StatisticalTestType::OneSampleTTest => {
                // One-sample t-test against baseline mean
                let n = latencies.len() as f32;
                let sample_mean = latencies.iter().sum::<f32>() / n;
                let sample_var = latencies
                    .iter()
                    .map(|x| (x - sample_mean).powi(2))
                    .sum::<f32>()
                    / (n - 1.0);
                let sample_std = sample_var.sqrt();

                let t_stat = (sample_mean - baseline.mean_latency_ms)
                    / (sample_std / n.sqrt());

                // Approximate p-value (would use proper t-distribution in production)
                let p_value = if t_stat.abs() > 1.96 { 0.05 } else { 0.5 };

                // Cohen's d
                let effect_size = (sample_mean - baseline.mean_latency_ms) / baseline.std_latency_ms;

                (t_stat, p_value, effect_size)
            }
            _ => return None, // Other test types not implemented
        };

        Some(StatisticalTest {
            test_type,
            statistic,
            p_value,
            is_significant: p_value < 0.05,
            effect_size,
        })
    }
}

impl Default for EthologicalValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PREDEFINED SPECIES PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════════

impl EthologicalValidator {
    /// Get default parameters for common species
    pub fn get_species_parameters(species: &str) -> Option<SpeciesParameters> {
        match species {
            "marmoset" => Some(SpeciesParameters {
                latency_range: (150.0, 400.0),
                f0_range: (7000.0, 12000.0),
                duration_range: (100.0, 300.0),
                typical_patterns: vec![
                    "phee".to_string(),
                    "twitter".to_string(),
                    "trill".to_string(),
                ],
            }),
            "bat" => Some(SpeciesParameters {
                latency_range: (50.0, 200.0),
                f0_range: (15000.0, 60000.0),
                duration_range: (5.0, 50.0),
                typical_patterns: vec![
                    "search".to_string(),
                    "approach".to_string(),
                    "social".to_string(),
                ],
            }),
            "dolphin" => Some(SpeciesParameters {
                latency_range: (200.0, 800.0),
                f0_range: (2000.0, 15000.0),
                duration_range: (200.0, 2000.0),
                typical_patterns: vec![
                    "signature_whistle".to_string(),
                    "burst".to_string(),
                    "click".to_string(),
                ],
            }),
            _ => None,
        }
    }

    /// Initialize validator with all default species
    pub fn with_default_species(mut self) -> Self {
        for species in ["marmoset", "bat", "dolphin"] {
            if let Some(params) = Self::get_species_parameters(species) {
                self.register_species(species.to_string(), params);
            }
        }
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mfas_new() {
        let mfas = MultiFactorAcceptanceScore::new();
        assert_eq!(mfas.overall_score, 0.5);
        assert_eq!(mfas.factors_count, 0);
    }

    #[test]
    fn test_mfas_acceptable() {
        let mut mfas = MultiFactorAcceptanceScore::new();
        mfas.overall_score = 0.8;
        mfas.confidence = 0.7;
        assert!(mfas.is_acceptable(0.7));

        mfas.overall_score = 0.6;
        assert!(!mfas.is_acceptable(0.7));
    }

    #[test]
    fn test_validator_new() {
        let validator = EthologicalValidator::new();
        assert_eq!(validator.acceptance_threshold, 0.7);
    }

    #[test]
    fn test_validate_interaction() {
        let validator = EthologicalValidator::new()
            .with_default_species();

        let metrics = BehavioralMetrics {
            timestamp: 0.0,
            subject_id: "test_subject".to_string(),
            agent_id: "test_agent".to_string(),
            stimulus_type: "phee".to_string(),
            response_type: "phee".to_string(),
            response_latency_ms: 200.0,
            response_duration_ms: 150.0,
            response_f0: 9000.0,
            spatial_directed: true,
            social_context: SocialContext::Dyadic,
        };

        let mfas = validator.validate_interaction(&metrics, "marmoset");

        assert!(mfas.overall_score > 0.0);
        assert!(mfas.overall_score <= 1.0);
        assert_eq!(mfas.factors_count, 4);
    }

    #[test]
    fn test_temporal_score() {
        let validator = EthologicalValidator::new();

        let metrics = BehavioralMetrics {
            timestamp: 0.0,
            subject_id: "test".to_string(),
            agent_id: "test".to_string(),
            stimulus_type: "test".to_string(),
            response_type: "test".to_string(),
            response_latency_ms: 200.0,
            response_duration_ms: 100.0,
            response_f0: 1000.0,
            spatial_directed: false,
            social_context: SocialContext::Isolation,
        };

        // Within default range should score high
        let score = validator.compute_temporal_score(&metrics, None);
        assert!(score > 0.5);
    }

    #[test]
    fn test_session_statistics() {
        let validator = EthologicalValidator::new();

        let session = ValidationSession {
            session_id: "test".to_string(),
            species: "marmoset".to_string(),
            start_time: 0.0,
            end_time: None,
            interactions: vec![],
            mfas_scores: vec![],
            statistics: SessionStatistics {
                interaction_count: 0,
                mean_mfas: 0.0,
                std_mfas: 0.0,
                acceptance_rate: 0.0,
                mean_latency_ms: 0.0,
                temporal_consistency: 0.0,
            },
        };

        let stats = validator.validate_session(&session);
        assert_eq!(stats.interaction_count, 0);
    }
}
