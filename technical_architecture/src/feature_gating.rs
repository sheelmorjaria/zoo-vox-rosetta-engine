//! Taxonomic-Aware Feature Gating for Species Classification
//! ==========================================================
//!
//! Implements "Dynamic Feature Reweighting" or "Attention-Based Input Gating"
//! that merges the Hierarchical Gatekeeper with the Voting Ensemble.
//!
//! # Architecture
//! ```text
//!       INPUT: 112D Feature Vector
//!             │
//!             ▼
//!    ┌─────────────────────┐
//!    │ FAST GATEKEEPER RF  │ -> Predicts Taxonomic Group (Bird/Mammal/etc.)
//!    │     (76D Physics)   │
//!    └─────────┬───────────┘
//!              │
//!              ▼
//!    ┌─────────────────────┐
//!    │  FEATURE MASKING    │ -> Applies Taxonomic Mask (Boost/Ssuppress)
//!    │  (Taxonomic Priors) │
//!    └─────────┬───────────┘
//!              │
//!              ▼
//!       WEIGHTED 112D Vector
//!              │
//!        ┌─────┴─────┐
//!        ▼           ▼
//!    [NN 112D]    [RF 112D]
//!        │           │
//!        └─────┬─────┘
//!              ▼
//!       [Ensemble Voter]
//!              │
//!              ▼
//!       FINAL PREDICTION
//! ```
//!
//! # Scientific Rationale
//! Different taxonomic groups rely on different acoustic dimensions:
//! - **Bats:** FM Slope, ICI (Inter-Call-Interval)
//! - **Whales:** Duration, Harmonic Density
//! - **Insects:** Tempo, Carrier Frequency
//!
//! By weighting features BEFORE the Species Expert sees them, we increase
//! the Signal-to-Noise Ratio for taxonomic-specific classification.
//!
//! # Usage
//! ```rust
//! use technical_architecture::feature_gating::{FeatureGate, FeatureGateConfig};
//!
//! let config = FeatureGateConfig::default();
//! let gate = FeatureGate::new(config);
//!
//! // Apply taxonomic-aware weighting
//! let (weighted_features, taxon) = gate.apply_gating(&features_112d, &gatekeeper_rf);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::taxonomic_router::{
    ConsolidatedTaxon, FEATURE_DIM, GATEKEEPER_DIM,
    feature_indices, idx_to_consolidated_taxon,
};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for Taxonomic-Aware Feature Gating
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FeatureGateConfig {
    /// Whether to enable feature gating (vs passthrough)
    pub enabled: bool,
    /// Minimum confidence to apply gating (below this, use default weights)
    pub min_confidence: f32,
    /// Boost factor for emphasized features
    pub boost_factor: f32,
    /// Suppress factor for de-emphasized features
    pub suppress_factor: f32,
    /// Whether to use soft gating (interpolate between masks)
    pub soft_gating: bool,
}

impl Default for FeatureGateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_confidence: 0.5,
            boost_factor: 2.5,
            suppress_factor: 0.5,
            soft_gating: true,
        }
    }
}

impl FeatureGateConfig {
    /// Create conservative config (less aggressive gating)
    pub fn conservative() -> Self {
        Self {
            enabled: true,
            min_confidence: 0.7,
            boost_factor: 1.5,
            suppress_factor: 0.7,
            soft_gating: true,
        }
    }

    /// Create aggressive config (strong gating)
    pub fn aggressive() -> Self {
        Self {
            enabled: true,
            min_confidence: 0.3,
            boost_factor: 3.5,
            suppress_factor: 0.3,
            soft_gating: false,
        }
    }
}

// =============================================================================
// Taxonomic Mask Definitions
// =============================================================================

/// Feature mask for a specific taxonomic group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxonomicMask {
    /// The taxonomic group this mask applies to
    pub taxon: ConsolidatedTaxon,
    /// Feature weights (112D vector)
    pub weights: Vec<f32>,
    /// Description of the mask strategy
    pub description: String,
}

impl TaxonomicMask {
    /// Create a new mask with default weights
    pub fn new(taxon: ConsolidatedTaxon) -> Self {
        Self {
            taxon,
            weights: vec![1.0; FEATURE_DIM],
            description: String::new(),
        }
    }

    /// Create a mask with custom weights
    pub fn with_weights(taxon: ConsolidatedTaxon, weights: Vec<f32>) -> Self {
        assert_eq!(weights.len(), FEATURE_DIM, "Weights must be {}D", FEATURE_DIM);
        Self {
            taxon,
            weights,
            description: String::new(),
        }
    }

    /// Boost a range of features
    pub fn boost_range(&mut self, start: usize, end: usize, factor: f32) -> &mut Self {
        for i in start..end.min(FEATURE_DIM) {
            self.weights[i] *= factor;
        }
        self
    }

    /// Suppress a range of features
    pub fn suppress_range(&mut self, start: usize, end: usize, factor: f32) -> &mut Self {
        for i in start..end.min(FEATURE_DIM) {
            self.weights[i] *= factor;
        }
        self
    }

    /// Set weight for a specific feature
    pub fn set_weight(&mut self, idx: usize, weight: f32) -> &mut Self {
        if idx < FEATURE_DIM {
            self.weights[idx] = weight;
        }
        self
    }

    /// Apply the mask to a feature vector
    pub fn apply(&self, features: &[f32]) -> Vec<f32> {
        assert_eq!(features.len(), FEATURE_DIM, "Features must be {}D", FEATURE_DIM);
        features.iter()
            .zip(self.weights.iter())
            .map(|(f, w)| f * w)
            .collect()
    }
}

// =============================================================================
// Feature Gate
// =============================================================================

/// Taxonomic-Aware Feature Gate
///
/// Uses Gatekeeper RF prediction to dynamically weight features
/// before passing to species classification models.
#[derive(Debug, Clone)]
pub struct FeatureGate {
    config: FeatureGateConfig,
    masks: HashMap<ConsolidatedTaxon, TaxonomicMask>,
}

impl FeatureGate {
    /// Create a new FeatureGate with default configuration
    pub fn new(config: FeatureGateConfig) -> Self {
        let mut gate = Self {
            config,
            masks: HashMap::new(),
        };
        gate.initialize_masks();
        gate
    }

    /// Initialize masks for all taxonomic groups
    fn initialize_masks(&mut self) {
        use feature_indices::*;

        // Bird: F0, Harmonics, Spectral complexity
        let mut bird_mask = TaxonomicMask::new(ConsolidatedTaxon::Bird);
        bird_mask.description = "Bird: F0, Harmonics, Pitch Geometry".to_string();
        bird_mask
            .boost_range(F0, F0 + 1, self.config.boost_factor)
            .boost_range(HARMONICITY, HARMONICITY + 2, 1.8)
            .boost_range(HARMONIC_TEXTURE_START, HARMONIC_TEXTURE_END, 1.5)
            .boost_range(PITCH_GEOMETRY_START, PITCH_GEOMETRY_END, 1.5);
        self.masks.insert(ConsolidatedTaxon::Bird, bird_mask);

        // Mammal (Bats/Primates): FM Slope, ICI, Formants
        let mut mammal_mask = TaxonomicMask::new(ConsolidatedTaxon::Mammal);
        mammal_mask.description = "Mammal: FM Slope, ICI, Formants".to_string();
        mammal_mask
            .boost_range(FM_BINS_START, FM_BINS_END, self.config.boost_factor)
            .boost_range(ICI_BINS_START, ICI_BINS_END, 3.0)
            .boost_range(28, 45, 1.8); // Formants
        self.masks.insert(ConsolidatedTaxon::Mammal, mammal_mask);

        // Marine Mammal (Dolphins/Whales): Duration, Harmonics, ICI
        let mut marine_mask = TaxonomicMask::new(ConsolidatedTaxon::MarineMammal);
        marine_mask.description = "Marine Mammal: Duration, Harmonics, ICI".to_string();
        marine_mask
            .boost_range(DURATION, DURATION + 1, self.config.boost_factor)
            .boost_range(HARMONIC_TEXTURE_START, HARMONIC_TEXTURE_END, 2.5)
            .boost_range(ICI_BINS_START, ICI_BINS_END, 3.0)
            .boost_range(FM_BINS_START, FM_BINS_END, 2.0);
        self.masks.insert(ConsolidatedTaxon::MarineMammal, marine_mask);

        // Insect: Tempo, Centroid, AM patterns
        let mut insect_mask = TaxonomicMask::new(ConsolidatedTaxon::Insect);
        insect_mask.description = "Insect: Tempo, Centroid, Dynamics".to_string();
        insect_mask
            .boost_range(RHYTHM_START, RHYTHM_END, self.config.boost_factor)
            .boost_range(SPECTRAL_CENTROID, SPECTRAL_CENTROID + 1, 2.5)
            .boost_range(DYNAMICS_BINS_START, DYNAMICS_BINS_END, 2.0);
        self.masks.insert(ConsolidatedTaxon::Insect, insect_mask);

        // Amphibian: AM Pulse, Tempo, F0
        let mut amphibian_mask = TaxonomicMask::new(ConsolidatedTaxon::Amphibian);
        amphibian_mask.description = "Amphibian: Dynamics, Rhythm, F0".to_string();
        amphibian_mask
            .boost_range(DYNAMICS_BINS_START, DYNAMICS_BINS_END, self.config.boost_factor)
            .boost_range(RHYTHM_START, RHYTHM_END, 2.5)
            .boost_range(F0, F0 + 1, 2.0);
        self.masks.insert(ConsolidatedTaxon::Amphibian, amphibian_mask);

        // Unknown: Passthrough (no modification)
        let unknown_mask = TaxonomicMask::new(ConsolidatedTaxon::Unknown);
        self.masks.insert(ConsolidatedTaxon::Unknown, unknown_mask);
    }

    /// Get the mask for a specific taxon
    pub fn get_mask(&self, taxon: ConsolidatedTaxon) -> &TaxonomicMask {
        self.masks.get(&taxon)
            .unwrap_or_else(|| self.masks.get(&ConsolidatedTaxon::Unknown).unwrap())
    }

    /// Apply gating based on Gatekeeper RF prediction
    ///
    /// # Arguments
    /// * `features_112d` - Full 112D feature vector
    /// * `gatekeeper_probs` - Probability output from Gatekeeper RF (6 classes)
    ///
    /// # Returns
    /// * (weighted_features, predicted_taxon, confidence)
    pub fn apply_gating(
        &self,
        features_112d: &[f32],
        gatekeeper_probs: &[f32],
    ) -> (Vec<f32>, ConsolidatedTaxon, f32) {
        assert_eq!(features_112d.len(), FEATURE_DIM, "Features must be {}D", FEATURE_DIM);
        assert_eq!(gatekeeper_probs.len(), 6, "Gatekeeper probs must be 6D (consolidated taxa)");

        // Find predicted taxon and confidence
        let (taxon_idx, &confidence) = gatekeeper_probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((5, &0.0));

        let predicted_taxon = idx_to_consolidated_taxon(taxon_idx);

        // If disabled or low confidence, return unmodified
        if !self.config.enabled || confidence < self.config.min_confidence {
            return (features_112d.to_vec(), predicted_taxon, confidence);
        }

        // Get the mask
        let mask = self.get_mask(predicted_taxon);

        // Apply mask
        let weighted = mask.apply(features_112d);

        (weighted, predicted_taxon, confidence)
    }

    /// Apply soft gating using probability-weighted masks
    ///
    /// Instead of hard gating (winner-take-all), interpolate between
    /// all masks weighted by gatekeeper probabilities.
    pub fn apply_soft_gating(
        &self,
        features_112d: &[f32],
        gatekeeper_probs: &[f32],
    ) -> (Vec<f32>, ConsolidatedTaxon, f32) {
        assert_eq!(features_112d.len(), FEATURE_DIM, "Features must be {}D", FEATURE_DIM);
        assert_eq!(gatekeeper_probs.len(), 6, "Gatekeeper probs must be 6D");

        // Find predicted taxon and confidence
        let (taxon_idx, &confidence) = gatekeeper_probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((5, &0.0));

        let predicted_taxon = idx_to_consolidated_taxon(taxon_idx);

        // If disabled or low confidence, return unmodified
        if !self.config.enabled || confidence < self.config.min_confidence {
            return (features_112d.to_vec(), predicted_taxon, confidence);
        }

        // Compute probability-weighted mask
        let mut combined_weights = vec![0.0f32; FEATURE_DIM];

        for (idx, &prob) in gatekeeper_probs.iter().enumerate() {
            let taxon = idx_to_consolidated_taxon(idx);
            let mask = self.get_mask(taxon);

            for (i, &w) in mask.weights.iter().enumerate() {
                combined_weights[i] += prob * w;
            }
        }

        // Apply combined weights
        let weighted: Vec<f32> = features_112d.iter()
            .zip(combined_weights.iter())
            .map(|(f, w)| f * w)
            .collect();

        (weighted, predicted_taxon, confidence)
    }

    /// Extract 76D Gatekeeper features from 112D vector
    ///
    /// Gatekeeper uses: Base Physics (0-45) + Macro Texture (46-75)
    pub fn extract_gatekeeper_features(features_112d: &[f32]) -> Vec<f32> {
        assert_eq!(features_112d.len(), FEATURE_DIM, "Features must be {}D", FEATURE_DIM);

        let mut gatekeeper_features = Vec::with_capacity(GATEKEEPER_DIM);

        // Layer 1: Base Physics (indices 0-45)
        gatekeeper_features.extend_from_slice(&features_112d[0..46]);

        // Layer 2: Macro Texture (indices 46-75)
        gatekeeper_features.extend_from_slice(&features_112d[46..76]);

        gatekeeper_features
    }

    /// Get configuration
    pub fn config(&self) -> &FeatureGateConfig {
        &self.config
    }

    /// Update configuration and reinitialize masks
    pub fn update_config(&mut self, config: FeatureGateConfig) {
        self.config = config;
        self.initialize_masks();
    }
}

// =============================================================================
// Gated Ensemble Input
// =============================================================================

/// Input for Gated Ensemble combining gatekeeper prediction with species features
#[derive(Debug, Clone)]
pub struct GatedEnsembleInput {
    /// Original 112D features (unmodified)
    pub original_features: Vec<f32>,
    /// Weighted 112D features (after gating)
    pub weighted_features: Vec<f32>,
    /// Predicted taxonomic group
    pub taxon: ConsolidatedTaxon,
    /// Confidence of taxonomic prediction
    pub taxon_confidence: f32,
    /// True species label
    pub true_label: usize,
}

impl GatedEnsembleInput {
    /// Create a new GatedEnsembleInput
    pub fn new(
        original_features: Vec<f32>,
        weighted_features: Vec<f32>,
        taxon: ConsolidatedTaxon,
        taxon_confidence: f32,
        true_label: usize,
    ) -> Self {
        Self {
            original_features,
            weighted_features,
            taxon,
            taxon_confidence,
            true_label,
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
    fn test_feature_gate_config_default() {
        let config = FeatureGateConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_confidence, 0.5);
        assert_eq!(config.boost_factor, 2.5);
        assert_eq!(config.suppress_factor, 0.5);
        assert!(config.soft_gating);
    }

    #[test]
    fn test_feature_gate_config_presets() {
        let conservative = FeatureGateConfig::conservative();
        assert_eq!(conservative.min_confidence, 0.7);
        assert_eq!(conservative.boost_factor, 1.5);

        let aggressive = FeatureGateConfig::aggressive();
        assert_eq!(aggressive.min_confidence, 0.3);
        assert_eq!(aggressive.boost_factor, 3.5);
        assert!(!aggressive.soft_gating);
    }

    #[test]
    fn test_taxonomic_mask_creation() {
        let mask = TaxonomicMask::new(ConsolidatedTaxon::Bird);
        assert_eq!(mask.taxon, ConsolidatedTaxon::Bird);
        assert_eq!(mask.weights.len(), FEATURE_DIM);

        // All weights should be 1.0 by default
        for w in &mask.weights {
            assert!((w - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_taxonomic_mask_boost_range() {
        let mut mask = TaxonomicMask::new(ConsolidatedTaxon::Mammal);
        mask.boost_range(0, 10, 2.0);

        // First 10 features should be boosted
        for i in 0..10 {
            assert!((mask.weights[i] - 2.0).abs() < 1e-6);
        }
        // Rest should be 1.0
        for i in 10..FEATURE_DIM {
            assert!((mask.weights[i] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_taxonomic_mask_apply() {
        let mut mask = TaxonomicMask::new(ConsolidatedTaxon::Insect);
        mask.boost_range(0, 5, 2.0);

        let features = vec![1.0; FEATURE_DIM];
        let weighted = mask.apply(&features);

        // First 5 features should be 2.0
        for i in 0..5 {
            assert!((weighted[i] - 2.0).abs() < 1e-6);
        }
        // Rest should be 1.0
        for i in 5..FEATURE_DIM {
            assert!((weighted[i] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_feature_gate_creation() {
        let config = FeatureGateConfig::default();
        let gate = FeatureGate::new(config);

        // Should have masks for all taxa
        assert!(gate.masks.contains_key(&ConsolidatedTaxon::Bird));
        assert!(gate.masks.contains_key(&ConsolidatedTaxon::Mammal));
        assert!(gate.masks.contains_key(&ConsolidatedTaxon::MarineMammal));
        assert!(gate.masks.contains_key(&ConsolidatedTaxon::Insect));
        assert!(gate.masks.contains_key(&ConsolidatedTaxon::Amphibian));
        assert!(gate.masks.contains_key(&ConsolidatedTaxon::Unknown));
    }

    #[test]
    fn test_apply_gating_high_confidence() {
        let config = FeatureGateConfig::default();
        let gate = FeatureGate::new(config);

        let features = vec![1.0; FEATURE_DIM];

        // High confidence prediction (Bird)
        let probs = vec![0.9, 0.05, 0.02, 0.01, 0.01, 0.01];
        let (weighted, taxon, conf) = gate.apply_gating(&features, &probs);

        assert_eq!(taxon, ConsolidatedTaxon::Bird);
        assert!((conf - 0.9).abs() < 1e-6);

        // Features should be modified (Bird mask boosts F0)
        assert!(weighted[0] != features[0]); // F0 is boosted
    }

    #[test]
    fn test_apply_gating_low_confidence() {
        let config = FeatureGateConfig::default();
        let gate = FeatureGate::new(config);

        let features = vec![1.0; FEATURE_DIM];

        // Low confidence prediction
        let probs = vec![0.3, 0.25, 0.2, 0.15, 0.05, 0.05];
        let (weighted, taxon, conf) = gate.apply_gating(&features, &probs);

        assert_eq!(taxon, ConsolidatedTaxon::Bird);
        assert!(conf < config.min_confidence);

        // Features should NOT be modified (low confidence)
        for i in 0..FEATURE_DIM {
            assert!((weighted[i] - features[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_gating_disabled() {
        let config = FeatureGateConfig {
            enabled: false,
            ..Default::default()
        };
        let gate = FeatureGate::new(config);

        let features = vec![1.0; FEATURE_DIM];
        let probs = vec![0.9, 0.05, 0.02, 0.01, 0.01, 0.01];
        let (weighted, _, _) = gate.apply_gating(&features, &probs);

        // Features should NOT be modified (disabled)
        for i in 0..FEATURE_DIM {
            assert!((weighted[i] - features[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_soft_gating() {
        let config = FeatureGateConfig {
            soft_gating: true,
            ..Default::default()
        };
        let gate = FeatureGate::new(config);

        let features = vec![1.0; FEATURE_DIM];

        // Mixed probabilities
        let probs = vec![0.5, 0.3, 0.1, 0.05, 0.03, 0.02];
        let (weighted, taxon, conf) = gate.apply_soft_gating(&features, &probs);

        assert_eq!(taxon, ConsolidatedTaxon::Bird);
        assert!((conf - 0.5).abs() < 1e-6);

        // Features should be modified (weighted combination of masks)
        // Not all weights are 1.0 anymore
        let has_modification = weighted.iter().zip(features.iter())
            .any(|(w, f)| (w - f).abs() > 1e-6);
        assert!(has_modification);
    }

    #[test]
    fn test_extract_gatekeeper_features() {
        let features_112d = vec![1.0; FEATURE_DIM];
        let gatekeeper_features = FeatureGate::extract_gatekeeper_features(&features_112d);

        // Should be 76D (46 + 30)
        assert_eq!(gatekeeper_features.len(), GATEKEEPER_DIM);
        assert_eq!(gatekeeper_features.len(), 76);
    }

    #[test]
    fn test_mammal_mask_fm_boost() {
        let config = FeatureGateConfig::default();
        let gate = FeatureGate::new(config);

        let mask = gate.get_mask(ConsolidatedTaxon::Mammal);

        // FM bins (81-86) should be boosted for mammals
        for i in feature_indices::FM_BINS_START..feature_indices::FM_BINS_END {
            assert!(mask.weights[i] > 1.0, "FM bin {} should be boosted for mammals", i);
        }
    }

    #[test]
    fn test_insect_mask_rhythm_boost() {
        let config = FeatureGateConfig::default();
        let gate = FeatureGate::new(config);

        let mask = gate.get_mask(ConsolidatedTaxon::Insect);

        // Rhythm (96-106) should be boosted for insects
        for i in feature_indices::RHYTHM_START..feature_indices::RHYTHM_END.min(FEATURE_DIM) {
            assert!(mask.weights[i] > 1.0, "Rhythm feature {} should be boosted for insects", i);
        }
    }

    #[test]
    fn test_marine_mammal_mask_duration_boost() {
        let config = FeatureGateConfig::default();
        let gate = FeatureGate::new(config);

        let mask = gate.get_mask(ConsolidatedTaxon::MarineMammal);

        // Duration should be boosted for marine mammals
        assert!(mask.weights[feature_indices::DURATION] > 1.0);
    }

    #[test]
    fn test_gated_ensemble_input_creation() {
        let original = vec![1.0; FEATURE_DIM];
        let weighted = vec![2.0; FEATURE_DIM];

        let input = GatedEnsembleInput::new(
            original.clone(),
            weighted.clone(),
            ConsolidatedTaxon::Bird,
            0.85,
            42,
        );

        assert_eq!(input.original_features, original);
        assert_eq!(input.weighted_features, weighted);
        assert_eq!(input.taxon, ConsolidatedTaxon::Bird);
        assert!((input.taxon_confidence - 0.85).abs() < 1e-6);
        assert_eq!(input.true_label, 42);
    }

    #[test]
    fn test_mask_safety_bounds() {
        let mut mask = TaxonomicMask::new(ConsolidatedTaxon::Bird);

        // Try to boost out of bounds - should not panic
        mask.boost_range(100, 150, 2.0);

        // Only features 100-111 should be modified (capped at FEATURE_DIM)
        for i in 0..100 {
            assert!((mask.weights[i] - 1.0).abs() < 1e-6);
        }
        for i in 100..FEATURE_DIM {
            assert!((mask.weights[i] - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_update_config_reinitializes_masks() {
        let config = FeatureGateConfig::default();
        let mut gate = FeatureGate::new(config);

        // Get original boost factor
        let original_boost = gate.get_mask(ConsolidatedTaxon::Bird).weights[0];

        // Update config with different boost factor
        let new_config = FeatureGateConfig {
            boost_factor: 5.0,
            ..Default::default()
        };
        gate.update_config(new_config);

        // Mask should be reinitialized with new boost
        let new_boost = gate.get_mask(ConsolidatedTaxon::Bird).weights[0];
        assert!(new_boost > original_boost);
    }
}
