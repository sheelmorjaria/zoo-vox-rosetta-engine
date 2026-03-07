//! Rhythm Sequencer Module for Inter-Phrase Intervals (IPIs)
//!
//! Treats IPIs as first-class objects, enabling storage, recognition, and
//! generation of species-typical rhythm patterns.
//!
//! Key Concept: Different species have characteristic timing patterns:
//! - Marmoset duets: Alternating calls with ~500ms intervals
//! - Bat echolocation: Rapid bursts with 20-50ms intervals
//! - Dolphin signature whistles: Variable 200-800ms intervals
//!
//! First-Class IPI Object:
//! - Duration (ms)
//! - Variability (jitter)
//! - Position in sequence
//! - Context (pre/post phrase types)
//! - Species association

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/// First-class Inter-Phrase Interval object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterPhraseInterval {
    /// Unique identifier
    pub id: String,
    /// Duration in milliseconds
    pub duration_ms: f64,
    /// Jitter/variability (standard deviation around mean)
    pub jitter_ms: f64,
    /// Position in the sequence (0-based)
    pub sequence_position: usize,
    /// Phrase type that precedes this interval
    pub preceding_phrase_type: Option<String>,
    /// Phrase type that follows this interval
    pub following_phrase_type: Option<String>,
    /// Species this IPI is associated with
    pub species: String,
    /// Context tag (e.g., "duet", "alarm_sequence", "social")
    pub context_tag: Option<String>,
    /// Confidence score if derived from analysis
    pub confidence: Option<f64>,
}

impl InterPhraseInterval {
    /// Create a new IPI
    pub fn new(id: &str, duration_ms: f64, species: &str) -> Self {
        Self {
            id: id.to_string(),
            duration_ms,
            jitter_ms: 0.0,
            sequence_position: 0,
            preceding_phrase_type: None,
            following_phrase_type: None,
            species: species.to_string(),
            context_tag: None,
            confidence: None,
        }
    }

    /// Create with full context
    pub fn with_context(
        id: &str,
        duration_ms: f64,
        jitter_ms: f64,
        position: usize,
        preceding: Option<&str>,
        following: Option<&str>,
        species: &str,
        context: Option<&str>,
    ) -> Self {
        Self {
            id: id.to_string(),
            duration_ms,
            jitter_ms,
            sequence_position: position,
            preceding_phrase_type: preceding.map(|s| s.to_string()),
            following_phrase_type: following.map(|s| s.to_string()),
            species: species.to_string(),
            context_tag: context.map(|s| s.to_string()),
            confidence: None,
        }
    }

    /// Sample a duration from this IPI distribution
    pub fn sample_duration(&self) -> f64 {
        if self.jitter_ms == 0.0 {
            return self.duration_ms;
        }
        // Simple normal-ish sampling using jitter
        // In production, use proper RNG
        self.duration_ms + (self.jitter_ms * 0.5 * (self.id.len() as f64 % 2.0 - 1.0))
    }
}

/// A rhythm pattern consisting of multiple IPIs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmPattern {
    /// Unique identifier
    pub id: String,
    /// Pattern name (e.g., "marmoset_duet", "bat_hunting_burst")
    pub name: String,
    /// Species this pattern belongs to
    pub species: String,
    /// Ordered list of IPIs in the pattern
    pub intervals: Vec<InterPhraseInterval>,
    /// Total pattern duration
    pub total_duration_ms: f64,
    /// Pattern variability (how much the whole pattern can stretch/compress)
    pub pattern_variability: f64,
    /// Context this pattern is used in
    pub context: String,
    /// Usage count (how often this pattern has been observed)
    pub usage_count: usize,
}

impl RhythmPattern {
    /// Create a new rhythm pattern
    pub fn new(id: &str, name: &str, species: &str, context: &str) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            species: species.to_string(),
            intervals: Vec::new(),
            total_duration_ms: 0.0,
            pattern_variability: 0.1,
            context: context.to_string(),
            usage_count: 0,
        }
    }

    /// Add an IPI to the pattern
    pub fn add_interval(&mut self, mut interval: InterPhraseInterval) {
        self.total_duration_ms += interval.duration_ms;
        interval.sequence_position = self.intervals.len();
        self.intervals.push(interval);
    }

    /// Get the mean IPI duration
    pub fn mean_interval(&self) -> f64 {
        if self.intervals.is_empty() {
            return 0.0;
        }
        self.total_duration_ms / self.intervals.len() as f64
    }

    /// Get the standard deviation of IPIs
    pub fn std_interval(&self) -> f64 {
        if self.intervals.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_interval();
        let variance: f64 = self
            .intervals
            .iter()
            .map(|i| (i.duration_ms - mean).powi(2))
            .sum::<f64>()
            / self.intervals.len() as f64;
        variance.sqrt()
    }

    /// Get the IPI coefficient of variation
    pub fn cv_interval(&self) -> f64 {
        let mean = self.mean_interval();
        if mean == 0.0 {
            return 0.0;
        }
        self.std_interval() / mean
    }

    /// Generate timing sequence for synthesis
    pub fn generate_timing(&self) -> Vec<f64> {
        self.intervals.iter().map(|i| i.sample_duration()).collect()
    }

    /// Increment usage count
    pub fn record_usage(&mut self) {
        self.usage_count += 1;
    }
}

/// Result of rhythm pattern recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmRecognitionResult {
    /// Recognized pattern (if any)
    pub pattern: Option<RhythmPattern>,
    /// Match confidence (0-1)
    pub confidence: f64,
    /// Similarity scores to known patterns
    pub similarity_scores: HashMap<String, f64>,
    /// Whether this is a novel (unseen) pattern
    pub is_novel: bool,
    /// Extracted features from the observed rhythm
    pub features: RhythmFeatures,
}

/// Features extracted from a rhythm sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmFeatures {
    /// Mean interval duration
    pub mean_interval_ms: f64,
    /// Standard deviation of intervals
    pub std_interval_ms: f64,
    /// Coefficient of variation
    pub cv: f64,
    /// Number of intervals
    pub interval_count: usize,
    /// Total duration
    pub total_duration_ms: f64,
    /// Tempo estimate (intervals per second)
    pub tempo_ips: f64,
    /// Rhythmic regularity score (0-1, higher = more regular)
    pub regularity: f64,
    /// Accelerando/ritardando trend (positive = speeding up, negative = slowing down)
    pub tempo_trend: f64,
}

impl Default for RhythmFeatures {
    fn default() -> Self {
        Self {
            mean_interval_ms: 0.0,
            std_interval_ms: 0.0,
            cv: 0.0,
            interval_count: 0,
            total_duration_ms: 0.0,
            tempo_ips: 0.0,
            regularity: 1.0,
            tempo_trend: 0.0,
        }
    }
}

impl RhythmFeatures {
    /// Extract features from a list of interval durations
    pub fn from_intervals(intervals: &[f64]) -> Self {
        if intervals.is_empty() {
            return Self::default();
        }

        let interval_count = intervals.len();
        let total_duration_ms: f64 = intervals.iter().sum();
        let mean_interval_ms = total_duration_ms / interval_count as f64;

        let std_interval_ms = if interval_count > 1 {
            let variance: f64 =
                intervals.iter().map(|&d| (d - mean_interval_ms).powi(2)).sum::<f64>() / (interval_count - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let cv = if mean_interval_ms > 0.0 {
            std_interval_ms / mean_interval_ms
        } else {
            0.0
        };

        let tempo_ips = if total_duration_ms > 0.0 {
            (interval_count as f64) / (total_duration_ms / 1000.0)
        } else {
            0.0
        };

        // Regularity: inverse of CV, capped at 1.0
        let regularity = if cv > 0.0 { (1.0 / (1.0 + cv)).min(1.0) } else { 1.0 };

        // Tempo trend: linear regression slope of intervals
        let tempo_trend = if interval_count > 1 {
            let n = interval_count as f64;
            let sum_x: f64 = (0..interval_count).map(|i| i as f64).sum();
            let sum_y: f64 = intervals.iter().sum();
            let sum_xy: f64 = intervals.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
            let sum_xx: f64 = (0..interval_count).map(|i| (i as f64).powi(2)).sum();

            let denominator = n * sum_xx - sum_x * sum_x;
            if denominator != 0.0 {
                (n * sum_xy - sum_x * sum_y) / denominator
            } else {
                0.0
            }
        } else {
            0.0
        };

        Self {
            mean_interval_ms,
            std_interval_ms,
            cv,
            interval_count,
            total_duration_ms,
            tempo_ips,
            regularity,
            tempo_trend,
        }
    }
}

/// Configuration for the rhythm sequencer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmSequencerConfig {
    /// Minimum similarity threshold to recognize a pattern
    pub recognition_threshold: f64,
    /// Whether to learn new patterns automatically
    pub auto_learn: bool,
    /// Maximum number of patterns to store per species
    pub max_patterns_per_species: usize,
    /// Tolerance for timing comparison (ms)
    pub timing_tolerance_ms: f64,
}

impl Default for RhythmSequencerConfig {
    fn default() -> Self {
        Self {
            recognition_threshold: 0.7,
            auto_learn: true,
            max_patterns_per_species: 100,
            timing_tolerance_ms: 50.0,
        }
    }
}

// ============================================================================
// RHYTHM SEQUENCER
// ============================================================================

/// Main rhythm sequencer that stores, recognizes, and generates patterns
pub struct RhythmSequencer {
    config: RhythmSequencerConfig,
    /// Known patterns by species
    patterns_by_species: HashMap<String, Vec<RhythmPattern>>,
    /// Pattern templates for species-typical rhythms
    species_templates: HashMap<String, Vec<RhythmTemplate>>,
}

/// Template for generating species-typical rhythms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhythmTemplate {
    /// Template name
    pub name: String,
    /// Species
    pub species: String,
    /// Mean interval duration
    pub mean_interval_ms: f64,
    /// Interval variability
    pub interval_std_ms: f64,
    /// Typical interval count
    pub typical_count: usize,
    /// Context (e.g., "duet", "alarm", "foraging")
    pub context: String,
}

impl RhythmSequencer {
    /// Create new sequencer with default config
    pub fn new() -> Self {
        Self::with_config(RhythmSequencerConfig::default())
    }

    /// Create new sequencer with custom config
    pub fn with_config(config: RhythmSequencerConfig) -> Self {
        let mut sequencer = Self {
            config,
            patterns_by_species: HashMap::new(),
            species_templates: HashMap::new(),
        };

        // Initialize species templates
        sequencer.initialize_templates();
        sequencer
    }

    /// Initialize species-typical rhythm templates
    fn initialize_templates(&mut self) {
        // Marmoset templates
        self.species_templates.insert(
            "marmoset".to_string(),
            vec![
                RhythmTemplate {
                    name: "duet_alternating".to_string(),
                    species: "marmoset".to_string(),
                    mean_interval_ms: 500.0,
                    interval_std_ms: 100.0,
                    typical_count: 6,
                    context: "duet".to_string(),
                },
                RhythmTemplate {
                    name: "food_call_sequence".to_string(),
                    species: "marmoset".to_string(),
                    mean_interval_ms: 300.0,
                    interval_std_ms: 50.0,
                    typical_count: 4,
                    context: "food_discovery".to_string(),
                },
                RhythmTemplate {
                    name: "alarm_sequence".to_string(),
                    species: "marmoset".to_string(),
                    mean_interval_ms: 150.0,
                    interval_std_ms: 30.0,
                    typical_count: 8,
                    context: "alarm".to_string(),
                },
            ],
        );

        // Bat templates
        self.species_templates.insert(
            "bat".to_string(),
            vec![
                RhythmTemplate {
                    name: "echolocation_burst".to_string(),
                    species: "bat".to_string(),
                    mean_interval_ms: 35.0,
                    interval_std_ms: 10.0,
                    typical_count: 10,
                    context: "navigation".to_string(),
                },
                RhythmTemplate {
                    name: "social_call_sequence".to_string(),
                    species: "bat".to_string(),
                    mean_interval_ms: 200.0,
                    interval_std_ms: 50.0,
                    typical_count: 5,
                    context: "social".to_string(),
                },
            ],
        );

        // Dolphin templates
        self.species_templates.insert(
            "dolphin".to_string(),
            vec![
                RhythmTemplate {
                    name: "signature_whistle_exchange".to_string(),
                    species: "dolphin".to_string(),
                    mean_interval_ms: 500.0,
                    interval_std_ms: 200.0,
                    typical_count: 3,
                    context: "identification".to_string(),
                },
                RhythmTemplate {
                    name: "hunting_sequence".to_string(),
                    species: "dolphin".to_string(),
                    mean_interval_ms: 100.0,
                    interval_std_ms: 30.0,
                    typical_count: 8,
                    context: "foraging".to_string(),
                },
            ],
        );

        // Zebra finch templates
        self.species_templates.insert(
            "zebra_finch".to_string(),
            vec![
                RhythmTemplate {
                    name: "song_motif".to_string(),
                    species: "zebra_finch".to_string(),
                    mean_interval_ms: 50.0,
                    interval_std_ms: 15.0,
                    typical_count: 7,
                    context: "song".to_string(),
                },
                RhythmTemplate {
                    name: "distance_call".to_string(),
                    species: "zebra_finch".to_string(),
                    mean_interval_ms: 800.0,
                    interval_std_ms: 200.0,
                    typical_count: 2,
                    context: "contact".to_string(),
                },
            ],
        );
    }

    /// Add a known pattern
    pub fn add_pattern(&mut self, pattern: RhythmPattern) {
        let species_patterns = self.patterns_by_species.entry(pattern.species.clone()).or_default();

        // Check if we should replace an existing pattern
        if species_patterns.len() >= self.config.max_patterns_per_species {
            // Remove least used pattern
            species_patterns.sort_by_key(|p| std::cmp::Reverse(p.usage_count));
            species_patterns.pop();
        }

        species_patterns.push(pattern);
    }

    /// Get patterns for a species
    pub fn get_patterns(&self, species: &str) -> Vec<&RhythmPattern> {
        self.patterns_by_species
            .get(species)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Get templates for a species
    pub fn get_templates(&self, species: &str) -> Vec<&RhythmTemplate> {
        self.species_templates
            .get(species)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Recognize a rhythm pattern from observed intervals
    pub fn recognize_pattern(&mut self, species: &str, intervals: &[f64]) -> RhythmRecognitionResult {
        let features = RhythmFeatures::from_intervals(intervals);
        let mut similarity_scores: HashMap<String, f64> = HashMap::new();

        // Compare against known patterns
        if let Some(patterns) = self.patterns_by_species.get(species) {
            for pattern in patterns {
                let similarity = self.compute_pattern_similarity(&features, pattern);
                similarity_scores.insert(pattern.id.clone(), similarity);
            }
        }

        // Compare against templates
        if let Some(templates) = self.species_templates.get(species) {
            for template in templates {
                let similarity = self.compute_template_similarity(&features, template);
                similarity_scores.insert(format!("template_{}", template.name), similarity);
            }
        }

        // Find best match
        let (best_id, best_score) = similarity_scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, &v)| (k.clone(), v))
            .unwrap_or(("".to_string(), 0.0));

        // Determine if recognized
        let is_novel = best_score < self.config.recognition_threshold;

        let pattern = if !is_novel && !best_id.starts_with("template_") {
            // Find and return the matched pattern, incrementing usage
            if let Some(patterns) = self.patterns_by_species.get_mut(species) {
                if let Some(pattern) = patterns.iter_mut().find(|p| p.id == best_id) {
                    pattern.record_usage();
                    Some(pattern.clone())
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        RhythmRecognitionResult {
            pattern,
            confidence: best_score,
            similarity_scores,
            is_novel,
            features,
        }
    }

    /// Compute similarity between features and a pattern
    fn compute_pattern_similarity(&self, features: &RhythmFeatures, pattern: &RhythmPattern) -> f64 {
        let pattern_features =
            RhythmFeatures::from_intervals(&pattern.intervals.iter().map(|i| i.duration_ms).collect::<Vec<_>>());

        self.compute_feature_similarity(features, &pattern_features)
    }

    /// Compute similarity between features and a template
    fn compute_template_similarity(&self, features: &RhythmFeatures, template: &RhythmTemplate) -> f64 {
        // Create synthetic features from template
        let template_features = RhythmFeatures {
            mean_interval_ms: template.mean_interval_ms,
            std_interval_ms: template.interval_std_ms,
            cv: if template.mean_interval_ms > 0.0 {
                template.interval_std_ms / template.mean_interval_ms
            } else {
                0.0
            },
            interval_count: template.typical_count,
            total_duration_ms: template.mean_interval_ms * template.typical_count as f64,
            tempo_ips: 1000.0 / template.mean_interval_ms,
            regularity: 0.8,
            tempo_trend: 0.0,
        };

        self.compute_feature_similarity(features, &template_features)
    }

    /// Compute similarity between two feature sets
    fn compute_feature_similarity(&self, a: &RhythmFeatures, b: &RhythmFeatures) -> f64 {
        // Weighted combination of feature similarities
        let mean_sim = self.gaussian_similarity(a.mean_interval_ms, b.mean_interval_ms, 200.0);
        let cv_sim = self.gaussian_similarity(a.cv, b.cv, 0.3);
        let count_sim = self.gaussian_similarity(a.interval_count as f64, b.interval_count as f64, 3.0);
        let regularity_sim = self.gaussian_similarity(a.regularity, b.regularity, 0.2);

        // Weighted average
        0.35 * mean_sim + 0.25 * cv_sim + 0.20 * count_sim + 0.20 * regularity_sim
    }

    /// Gaussian similarity function
    fn gaussian_similarity(&self, a: f64, b: f64, sigma: f64) -> f64 {
        (-(a - b).powi(2) / (2.0 * sigma.powi(2))).exp()
    }

    /// Generate a rhythm sequence from a template
    pub fn generate_from_template(
        &self,
        species: &str,
        template_name: &str,
        num_intervals: Option<usize>,
    ) -> Option<Vec<f64>> {
        let templates = self.species_templates.get(species)?;

        let template = templates.iter().find(|t| t.name == template_name)?;

        let count = num_intervals.unwrap_or(template.typical_count);

        // Generate intervals with some variability
        let intervals: Vec<f64> = (0..count)
            .map(|i| {
                let base = template.mean_interval_ms;
                let jitter = template.interval_std_ms * ((i as f64 % 2.0) - 0.5);
                (base + jitter).max(10.0)
            })
            .collect();

        Some(intervals)
    }

    /// Learn a new pattern from observed intervals
    pub fn learn_pattern(&mut self, species: &str, name: &str, intervals: &[f64], context: &str) -> RhythmPattern {
        let id = format!("{}_{}", species, name);
        let mut pattern = RhythmPattern::new(&id, name, species, context);

        for (i, &duration) in intervals.iter().enumerate() {
            let ipi = InterPhraseInterval::with_context(
                &format!("{}_ipi_{}", id, i),
                duration,
                0.0, // Will be updated
                i,
                None,
                None,
                species,
                Some(context),
            );
            pattern.add_interval(ipi);
        }

        // Update jitter based on pattern variability
        let std = pattern.std_interval();
        for ipi in &mut pattern.intervals {
            ipi.jitter_ms = std;
        }

        if self.config.auto_learn {
            self.add_pattern(pattern.clone());
        }

        pattern
    }

    /// Get the configuration
    pub fn config(&self) -> &RhythmSequencerConfig {
        &self.config
    }

    /// Get total pattern count
    pub fn pattern_count(&self) -> usize {
        self.patterns_by_species.values().map(|v| v.len()).sum()
    }

    /// Clear all learned patterns
    pub fn clear_patterns(&mut self) {
        self.patterns_by_species.clear();
    }
}

impl Default for RhythmSequencer {
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
    fn test_ipi_creation() {
        let ipi = InterPhraseInterval::new("test_ipi", 500.0, "marmoset");
        assert_eq!(ipi.duration_ms, 500.0);
        assert_eq!(ipi.species, "marmoset");
        assert_eq!(ipi.jitter_ms, 0.0);
    }

    #[test]
    fn test_ipi_with_context() {
        let ipi = InterPhraseInterval::with_context(
            "ctx_ipi",
            300.0,
            50.0,
            1,
            Some("phee"),
            Some("trill"),
            "marmoset",
            Some("duet"),
        );
        assert_eq!(ipi.duration_ms, 300.0);
        assert_eq!(ipi.jitter_ms, 50.0);
        assert_eq!(ipi.preceding_phrase_type, Some("phee".to_string()));
        assert_eq!(ipi.following_phrase_type, Some("trill".to_string()));
    }

    #[test]
    fn test_ipi_sample_duration() {
        let mut ipi = InterPhraseInterval::new("test_ipi", 500.0, "marmoset");
        ipi.jitter_ms = 0.0;

        // With zero jitter, should return exact duration
        let sample = ipi.sample_duration();
        assert_eq!(sample, 500.0);
    }

    #[test]
    fn test_rhythm_pattern_creation() {
        let pattern = RhythmPattern::new("p1", "test_pattern", "marmoset", "duet");
        assert_eq!(pattern.name, "test_pattern");
        assert_eq!(pattern.species, "marmoset");
        assert_eq!(pattern.intervals.len(), 0);
    }

    #[test]
    fn test_rhythm_pattern_add_interval() {
        let mut pattern = RhythmPattern::new("p1", "test", "marmoset", "duet");
        let ipi = InterPhraseInterval::new("i1", 500.0, "marmoset");

        pattern.add_interval(ipi);

        assert_eq!(pattern.intervals.len(), 1);
        assert_eq!(pattern.total_duration_ms, 500.0);
    }

    #[test]
    fn test_rhythm_pattern_statistics() {
        let mut pattern = RhythmPattern::new("p1", "test", "marmoset", "duet");

        // Add intervals: 200, 300, 400 (mean = 300)
        pattern.add_interval(InterPhraseInterval::new("i1", 200.0, "marmoset"));
        pattern.add_interval(InterPhraseInterval::new("i2", 300.0, "marmoset"));
        pattern.add_interval(InterPhraseInterval::new("i3", 400.0, "marmoset"));

        assert_eq!(pattern.mean_interval(), 300.0);
        assert!(pattern.std_interval() > 0.0);
        assert!(pattern.cv_interval() > 0.0);
    }

    #[test]
    fn test_rhythm_features_extraction() {
        let intervals = vec![200.0, 300.0, 400.0, 300.0];
        let features = RhythmFeatures::from_intervals(&intervals);

        assert_eq!(features.interval_count, 4);
        assert_eq!(features.total_duration_ms, 1200.0);
        assert_eq!(features.mean_interval_ms, 300.0);
        assert!(features.std_interval_ms > 0.0);
        assert!(features.tempo_ips > 0.0);
    }

    #[test]
    fn test_rhythm_features_empty() {
        let features = RhythmFeatures::from_intervals(&[]);
        assert_eq!(features.interval_count, 0);
        assert_eq!(features.mean_interval_ms, 0.0);
    }

    #[test]
    fn test_rhythm_features_regularity() {
        // Regular pattern
        let regular = vec![500.0, 500.0, 500.0, 500.0];
        let features_regular = RhythmFeatures::from_intervals(&regular);
        assert!(features_regular.regularity > 0.9);

        // Irregular pattern
        let irregular = vec![100.0, 900.0, 200.0, 800.0];
        let features_irregular = RhythmFeatures::from_intervals(&irregular);
        assert!(features_irregular.regularity < features_regular.regularity);
    }

    #[test]
    fn test_rhythm_features_tempo_trend() {
        // Accelerando (speeding up - intervals getting shorter)
        let accel = vec![400.0, 300.0, 200.0, 100.0];
        let features_accel = RhythmFeatures::from_intervals(&accel);
        assert!(features_accel.tempo_trend < 0.0);

        // Ritardando (slowing down - intervals getting longer)
        let rit = vec![100.0, 200.0, 300.0, 400.0];
        let features_rit = RhythmFeatures::from_intervals(&rit);
        assert!(features_rit.tempo_trend > 0.0);
    }

    #[test]
    fn test_sequencer_creation() {
        let sequencer = RhythmSequencer::new();
        assert_eq!(sequencer.pattern_count(), 0);
    }

    #[test]
    fn test_sequencer_has_templates() {
        let sequencer = RhythmSequencer::new();

        // Should have marmoset templates
        let marmoset_templates = sequencer.get_templates("marmoset");
        assert!(!marmoset_templates.is_empty());

        // Should have bat templates
        let bat_templates = sequencer.get_templates("bat");
        assert!(!bat_templates.is_empty());
    }

    #[test]
    fn test_sequencer_add_pattern() {
        let mut sequencer = RhythmSequencer::new();
        let mut pattern = RhythmPattern::new("p1", "test", "marmoset", "duet");
        pattern.add_interval(InterPhraseInterval::new("i1", 500.0, "marmoset"));

        sequencer.add_pattern(pattern);

        assert_eq!(sequencer.pattern_count(), 1);
        let patterns = sequencer.get_patterns("marmoset");
        assert_eq!(patterns.len(), 1);
    }

    #[test]
    fn test_sequencer_recognize_known_pattern() {
        let mut sequencer = RhythmSequencer::new();

        // Learn a pattern
        let intervals = vec![500.0, 500.0, 500.0, 500.0, 500.0, 500.0];
        sequencer.learn_pattern("marmoset", "test_duet", &intervals, "duet");

        // Recognize similar pattern
        let similar = vec![480.0, 520.0, 500.0, 510.0, 490.0, 500.0];
        let result = sequencer.recognize_pattern("marmoset", &similar);

        assert!(!result.is_novel);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_sequencer_recognize_novel_pattern() {
        let mut sequencer = RhythmSequencer::new();

        // Unusual pattern
        let unusual = vec![1000.0, 10.0, 2000.0, 5.0, 3000.0];
        let result = sequencer.recognize_pattern("marmoset", &unusual);

        assert!(result.is_novel);
    }

    #[test]
    fn test_sequencer_generate_from_template() {
        let sequencer = RhythmSequencer::new();

        let intervals = sequencer.generate_from_template("marmoset", "duet_alternating", Some(4));

        assert!(intervals.is_some());
        let intervals = intervals.unwrap();
        assert_eq!(intervals.len(), 4);

        // Check intervals are roughly around the template mean (500ms)
        for &i in &intervals {
            assert!(i > 300.0 && i < 700.0, "Interval {} outside expected range", i);
        }
    }

    #[test]
    fn test_sequencer_generate_unknown_template() {
        let sequencer = RhythmSequencer::new();

        let intervals = sequencer.generate_from_template("marmoset", "nonexistent", None);
        assert!(intervals.is_none());
    }

    #[test]
    fn test_sequencer_learn_pattern() {
        let mut sequencer = RhythmSequencer::new();

        let intervals = vec![300.0, 400.0, 300.0, 400.0];
        let pattern = sequencer.learn_pattern("marmoset", "learned_pattern", &intervals, "test");

        assert_eq!(pattern.intervals.len(), 4);
        assert_eq!(sequencer.pattern_count(), 1);
    }

    #[test]
    fn test_pattern_usage_tracking() {
        let mut pattern = RhythmPattern::new("p1", "test", "marmoset", "duet");

        assert_eq!(pattern.usage_count, 0);

        pattern.record_usage();
        pattern.record_usage();

        assert_eq!(pattern.usage_count, 2);
    }

    #[test]
    fn test_serialization_ipi() {
        let ipi = InterPhraseInterval::with_context(
            "test",
            500.0,
            50.0,
            0,
            Some("phee"),
            Some("trill"),
            "marmoset",
            Some("duet"),
        );

        let json = serde_json::to_string(&ipi).unwrap();
        let decoded: InterPhraseInterval = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.id, "test");
        assert_eq!(decoded.duration_ms, 500.0);
        assert_eq!(decoded.preceding_phrase_type, Some("phee".to_string()));
    }

    #[test]
    fn test_serialization_pattern() {
        let mut pattern = RhythmPattern::new("p1", "test", "marmoset", "duet");
        pattern.add_interval(InterPhraseInterval::new("i1", 500.0, "marmoset"));

        let json = serde_json::to_string(&pattern).unwrap();
        let decoded: RhythmPattern = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.id, "p1");
        assert_eq!(decoded.intervals.len(), 1);
    }

    #[test]
    fn test_species_templates() {
        let sequencer = RhythmSequencer::new();

        // Marmoset should have duet template
        let marmoset = sequencer.get_templates("marmoset");
        assert!(marmoset.iter().any(|t| t.name == "duet_alternating"));

        // Bat should have echolocation template
        let bat = sequencer.get_templates("bat");
        assert!(bat.iter().any(|t| t.name == "echolocation_burst"));

        // Dolphin should have signature whistle template
        let dolphin = sequencer.get_templates("dolphin");
        assert!(dolphin.iter().any(|t| t.name == "signature_whistle_exchange"));
    }

    #[test]
    fn test_sequencer_clear_patterns() {
        let mut sequencer = RhythmSequencer::new();

        let intervals = vec![500.0, 500.0];
        sequencer.learn_pattern("marmoset", "test", &intervals, "duet");

        assert_eq!(sequencer.pattern_count(), 1);

        sequencer.clear_patterns();

        assert_eq!(sequencer.pattern_count(), 0);
    }

    #[test]
    fn test_gaussian_similarity() {
        let sequencer = RhythmSequencer::new();

        // Same values should give high similarity
        let sim_same = sequencer.gaussian_similarity(500.0, 500.0, 100.0);
        assert!(sim_same > 0.99);

        // Different values should give lower similarity
        let sim_diff = sequencer.gaussian_similarity(500.0, 700.0, 100.0);
        assert!(sim_diff < sim_same);
    }
}
