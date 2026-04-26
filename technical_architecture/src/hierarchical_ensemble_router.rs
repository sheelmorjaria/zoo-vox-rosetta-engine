//! Hierarchical Ensemble Router - Unified Two-Stage Classification
//! ================================================================
//!
//! This module implements the ultimate architecture for bioacoustic classification:
//! a two-stage hierarchical ensemble that combines Curriculum Learning (NN) with
//! Feature Stacking (RF) to solve the "Resolution Paradox."
//!
//! # Architecture Overview
//!
//! ```text
//!        INPUT: 112D Feature Vector
//!             │
//!             ▼
//!    ┌────────────────────────────────────┐
//!    │     STAGE 1: GROUP DETECTION       │
//!    │      (Taxonomy / Context)          │
//!    ├────────────────────────────────────┤
//!    │  ┌──────────────────┐ ┌───────────┐│
//!    │  │ RF Gatekeeper    │ │ NN Block 1││
//!    │  │ (Physics 76D)    │ │ (Physics) ││
//!    │  └────────┬─────────┘ └─────┬─────┘│
//!    │           │                 │      │
//!    │           └───────┬─────────┘      │
//!    │                   ▼                │
//!    │          [Ensemble Voter]          │
//!    │           (5% NN / 95% RF)         │
//!    └───────────────┬────────────────────┘
//!                    │
//!            PREDICTION: "Bat"
//!                    │
//!       ┌────────────┴────────────┐
//!       │     FEATURE REWEIGHTING │
//!       │  (Boost FM/ICI for Bat) │
//!       └────────────┬────────────┘
//!                    │
//!                    ▼
//!    ┌────────────────────────────────────┐
//!    │   STAGE 2: SPECIES DISCRIMINATION  │
//!    │        (Specialist Models)         │
//!    ├────────────────────────────────────┤
//!    │  ┌──────────────────┐ ┌───────────┐│
//!    │  │ RF Specialist    │ │ NN Block 2││
//!    │  │ (Bat Expert)     │ │ (Unfreeze)││
//!    │  └────────┬─────────┘ └─────┬─────┘│
//!    │           │                 │      │
//!    │           └───────┬─────────┘      │
//!    │                   ▼                │
//!    │          [Ensemble Voter]          │
//!    │           (50% NN / 50% RF)        │
//!    └───────────────┬────────────────────┘
//!                    │
//!                    ▼
//!            FINAL PREDICTION
//!          "Species: Bat #42"
//! ```
//!
//! # Key Concepts
//!
//! 1. **Resolution Paradox**: Using high-resolution micro-texture features for
//!    coarse taxonomy (distinguishing "Mouse" from "Whale") is wasteful. Physics
//!    features suffice for that. Micro-texture should only be used for fine
//!    species discrimination within a known group.
//!
//! 2. **Curriculum Learning**: The NN is trained in phases, learning physics first,
//!    then progressively adding macro and micro texture. This prevents overfitting
//!    to high-variance micro features.
//!
//! 3. **Feature Reweighting**: After Stage 1 predicts a group, we apply taxonomic
//!    priors to boost features known to be discriminative for that group (e.g.,
//!    ICI for cetaceans, rhythm for insects).
//!
//! 4. **Specialist Models**: Stage 2 uses group-specific RFs trained only on
//!    samples from that taxonomic group. This allows finding subtle splits that
//!    would be washed out in a global model.
//!
//! # Usage
//!
//! ```rust
//! use technical_architecture::hierarchical_ensemble_router::{
//!     HierarchicalEnsembleRouter, RouterConfig, RouterResult
//! };
//!
//! // Create router with default configuration
//! let router = HierarchicalEnsembleRouter::new()?;
//!
//! // Classify a sample
//! let features = vec![0.0; 112]; // 112D feature vector
//! let result = router.classify(&features)?;
//!
//! println!("Species: {}", result.species);
//! println!("Confidence: {:.2}%", result.confidence * 100.0);
//! println!("Group: {:?}", result.predicted_group);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::rf_stacking_ensemble::{
    FeatureStackingEnsemble, RFModel, StackingConfig, FULL_DIM as RF_FULL_DIM, PHYSICS_DIM as RF_PHYSICS_DIM,
};
use crate::taxonomic_router::{
    apply_taxonomic_mask, consolidate_taxon, consolidated_taxon_to_idx, get_taxonomic_weights,
    idx_to_consolidated_taxon, map_species_to_taxon, slice_gatekeeper_input, slice_species_expert_input,
    ConsolidatedTaxon, Taxon, CONSOLIDATED_TAXON_COUNT, FEATURE_DIM, GATEKEEPER_DIM, MACRO_TEXTURE_DIM,
    MICRO_TEXTURE_DIM, PHYSICS_DIM, SPECIES_EXPERT_DIM,
};

// =============================================================================
// Constants
// =============================================================================

/// Default Stage 1 RF weight (95% trust in RF for taxonomy)
pub const STAGE1_RF_WEIGHT: f32 = 0.95;
/// Default Stage 1 NN weight (5% NN contribution)
pub const STAGE1_NN_WEIGHT: f32 = 0.05;

/// Default Stage 2 RF weight (50% trust in specialist RF)
pub const STAGE2_RF_WEIGHT: f32 = 0.50;
/// Default Stage 2 NN weight (50% NN contribution)
pub const STAGE2_NN_WEIGHT: f32 = 0.50;

/// Minimum Stage 1 confidence to proceed to Stage 2
pub const MIN_STAGE1_CONFIDENCE: f32 = 0.3;

/// Maximum number of species candidates to consider in Stage 2
pub const MAX_CANDIDATES: usize = 10;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the Hierarchical Ensemble Router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Weight given to RF in Stage 1 (Group Detection)
    pub stage1_rf_weight: f32,
    /// Weight given to NN in Stage 1 (Group Detection)
    pub stage1_nn_weight: f32,
    /// Weight given to specialist RF in Stage 2 (Species Discrimination)
    pub stage2_rf_weight: f32,
    /// Weight given to NN in Stage 2 (Species Discrimination)
    pub stage2_nn_weight: f32,
    /// Minimum Stage 1 confidence to proceed to Stage 2
    pub min_stage1_confidence: f32,
    /// Maximum candidates to consider from Stage 2
    pub max_candidates: usize,
    /// Apply feature reweighting between stages
    pub apply_feature_reweighting: bool,
    /// Enable NN contributions (can disable for RF-only mode)
    pub enable_nn: bool,
    /// Fallback taxon when confidence is too low
    pub fallback_taxon: ConsolidatedTaxon,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            stage1_rf_weight: STAGE1_RF_WEIGHT,
            stage1_nn_weight: STAGE1_NN_WEIGHT,
            stage2_rf_weight: STAGE2_RF_WEIGHT,
            stage2_nn_weight: STAGE2_NN_WEIGHT,
            min_stage1_confidence: MIN_STAGE1_CONFIDENCE,
            max_candidates: MAX_CANDIDATES,
            apply_feature_reweighting: true,
            enable_nn: true,
            fallback_taxon: ConsolidatedTaxon::Unknown,
        }
    }
}

impl RouterConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if (self.stage1_rf_weight + self.stage1_nn_weight - 1.0).abs() > 1e-6 {
            return Err("Stage 1 weights must sum to 1.0".to_string());
        }
        if (self.stage2_rf_weight + self.stage2_nn_weight - 1.0).abs() > 1e-6 {
            return Err("Stage 2 weights must sum to 1.0".to_string());
        }
        if self.stage1_rf_weight < 0.0 || self.stage1_nn_weight < 0.0 {
            return Err("Stage 1 weights must be non-negative".to_string());
        }
        if self.stage2_rf_weight < 0.0 || self.stage2_nn_weight < 0.0 {
            return Err("Stage 2 weights must be non-negative".to_string());
        }
        Ok(())
    }

    /// Create config for RF-only mode (no NN)
    pub fn rf_only() -> Self {
        Self {
            stage1_rf_weight: 1.0,
            stage1_nn_weight: 0.0,
            stage2_rf_weight: 1.0,
            stage2_nn_weight: 0.0,
            enable_nn: false,
            ..Default::default()
        }
    }

    /// Create config for NN-only mode (no RF)
    pub fn nn_only() -> Self {
        Self {
            stage1_rf_weight: 0.0,
            stage1_nn_weight: 1.0,
            stage2_rf_weight: 0.0,
            stage2_nn_weight: 1.0,
            enable_nn: true,
            ..Default::default()
        }
    }

    /// Create config with balanced RF/NN weights
    pub fn balanced() -> Self {
        Self {
            stage1_rf_weight: 0.5,
            stage1_nn_weight: 0.5,
            stage2_rf_weight: 0.5,
            stage2_nn_weight: 0.5,
            ..Default::default()
        }
    }
}

// =============================================================================
// Results
// =============================================================================

/// Result of Stage 1 (Group Detection)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage1Result {
    /// Predicted taxonomic group
    pub predicted_group: ConsolidatedTaxon,
    /// Confidence of group prediction (0.0 - 1.0)
    pub confidence: f32,
    /// RF's group prediction
    pub rf_prediction: ConsolidatedTaxon,
    /// RF's confidence
    pub rf_confidence: f32,
    /// RF's full probability distribution
    pub rf_proba: Vec<f32>,
    /// NN's group prediction (if enabled)
    pub nn_prediction: Option<ConsolidatedTaxon>,
    /// NN's confidence (if enabled)
    pub nn_confidence: Option<f32>,
    /// NN's full probability distribution (if enabled)
    pub nn_proba: Option<Vec<f32>>,
    /// Effective RF weight used
    pub rf_weight: f32,
    /// Effective NN weight used
    pub nn_weight: f32,
}

/// Result of Stage 2 (Species Discrimination)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage2Result {
    /// Final species prediction
    pub species: String,
    /// Confidence of species prediction (0.0 - 1.0)
    pub confidence: f32,
    /// Taxonomic group of predicted species
    pub taxon: Taxon,
    /// Top-N candidate species with confidence scores
    pub candidates: Vec<SpeciesCandidate>,
    /// Specialist RF's prediction
    pub rf_prediction: String,
    /// Specialist RF's confidence
    pub rf_confidence: f32,
    /// NN's prediction (if enabled)
    pub nn_prediction: Option<String>,
    /// NN's confidence (if enabled)
    pub nn_confidence: Option<f32>,
    /// Effective RF weight used
    pub rf_weight: f32,
    /// Effective NN weight used
    pub nn_weight: f32,
    /// Whether feature reweighting was applied
    pub reweighting_applied: bool,
}

/// A species candidate with confidence score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesCandidate {
    /// Species label
    pub label: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Taxonomic group
    pub taxon: Taxon,
    /// Rank in the candidate list (1 = top choice)
    pub rank: usize,
}

/// Complete result of the Hierarchical Ensemble Router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterResult {
    /// Final species prediction
    pub species: String,
    /// Final confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Predicted taxonomic group (from Stage 1)
    pub predicted_group: ConsolidatedTaxon,
    /// Detailed taxonomic group (from species)
    pub detailed_taxon: Taxon,
    /// Stage 1 (Group Detection) results
    pub stage1: Stage1Result,
    /// Stage 2 (Species Discrimination) results
    pub stage2: Stage2Result,
    /// Total processing time in microseconds
    pub processing_time_us: u64,
    /// Whether the prediction is considered reliable
    pub is_reliable: bool,
    /// Warning messages (if any)
    pub warnings: Vec<String>,
}

// =============================================================================
// Specialist RF Registry
// =============================================================================

/// Registry for taxonomic specialist RF models
///
/// Uses the full Taxon enum for fine-grained specialists:
/// - Each major clade gets its own specialist (Cetacean, Songbird, etc.)
/// - ConsolidatedTaxon groups are resolved to their best specialist
#[derive(Debug, Clone, Default)]
pub struct SpecialistRegistry {
    /// Specialist RF for Cetacean classification (toothed whales)
    pub rf_cetacean: Option<RFModel>,
    /// Specialist RF for Mysticete classification (baleen whales)
    pub rf_mysticete: Option<RFModel>,
    /// Specialist RF for Songbird classification (passerines)
    pub rf_songbird: Option<RFModel>,
    /// Specialist RF for Non-Passerine bird classification (parrots, owls)
    pub rf_non_passerine: Option<RFModel>,
    /// Specialist RF for Amphibian classification (frogs, toads)
    pub rf_amphibian: Option<RFModel>,
    /// Specialist RF for Pinniped classification (seals, sea lions)
    pub rf_pinniped: Option<RFModel>,
    /// Specialist RF for Insect classification
    pub rf_insect: Option<RFModel>,
    /// Specialist RF for Mammal classification (bats, primates)
    pub rf_mammal: Option<RFModel>,
    /// Fallback RF (trained on all data)
    pub rf_fallback: Option<RFModel>,
}

impl SpecialistRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a specialist is available for a detailed taxonomic group
    pub fn has_specialist(&self, taxon: Taxon) -> bool {
        match taxon {
            Taxon::Cetacean => self.rf_cetacean.is_some(),
            Taxon::Mysticete => self.rf_mysticete.is_some(),
            Taxon::Songbird => self.rf_songbird.is_some(),
            Taxon::NonPasserine => self.rf_non_passerine.is_some(),
            Taxon::Amphibian => self.rf_amphibian.is_some(),
            Taxon::Pinniped => self.rf_pinniped.is_some(),
            Taxon::Insect => self.rf_insect.is_some(),
            Taxon::Mammal => self.rf_mammal.is_some(),
            Taxon::Unknown => self.rf_fallback.is_some(),
        }
    }

    /// Check if any specialist is available for a consolidated group
    pub fn has_specialist_for_consolidated(&self, taxon: ConsolidatedTaxon) -> bool {
        self.get_best_specialist_for_consolidated(taxon).is_some()
    }

    /// Get the specialist RF for a detailed taxonomic group
    pub fn get_specialist(&self, taxon: Taxon) -> Option<&RFModel> {
        match taxon {
            Taxon::Cetacean => self.rf_cetacean.as_ref(),
            Taxon::Mysticete => self.rf_mysticete.as_ref(),
            Taxon::Songbird => self.rf_songbird.as_ref(),
            Taxon::NonPasserine => self.rf_non_passerine.as_ref(),
            Taxon::Amphibian => self.rf_amphibian.as_ref(),
            Taxon::Pinniped => self.rf_pinniped.as_ref(),
            Taxon::Insect => self.rf_insect.as_ref(),
            Taxon::Mammal => self.rf_mammal.as_ref(),
            Taxon::Unknown => self.rf_fallback.as_ref(),
        }
    }

    /// Get the best available specialist for a consolidated taxonomic group
    ///
    /// Tries to find the most specific specialist available:
    /// - For Bird: tries Songbird first, then NonPasserine
    /// - For MarineMammal: tries Cetacean first, then Mysticete, then Pinniped
    /// - For Mammal: uses the Mammal specialist
    pub fn get_best_specialist_for_consolidated(&self, taxon: ConsolidatedTaxon) -> Option<&RFModel> {
        match taxon {
            ConsolidatedTaxon::Bird => {
                // Try Songbird first (most common), then NonPasserine
                self.rf_songbird.as_ref().or(self.rf_non_passerine.as_ref())
            }
            ConsolidatedTaxon::MarineMammal => {
                // Try Cetacean first, then Mysticete, then Pinniped
                self.rf_cetacean
                    .as_ref()
                    .or(self.rf_mysticete.as_ref())
                    .or(self.rf_pinniped.as_ref())
            }
            ConsolidatedTaxon::Mammal => self.rf_mammal.as_ref(),
            ConsolidatedTaxon::Insect => self.rf_insect.as_ref(),
            ConsolidatedTaxon::Amphibian => self.rf_amphibian.as_ref(),
            ConsolidatedTaxon::Unknown => self.rf_fallback.as_ref(),
        }
    }

    /// Load a specialist RF for a detailed taxonomic group
    pub fn load_specialist(&mut self, taxon: Taxon, model: RFModel) {
        match taxon {
            Taxon::Cetacean => self.rf_cetacean = Some(model),
            Taxon::Mysticete => self.rf_mysticete = Some(model),
            Taxon::Songbird => self.rf_songbird = Some(model),
            Taxon::NonPasserine => self.rf_non_passerine = Some(model),
            Taxon::Amphibian => self.rf_amphibian = Some(model),
            Taxon::Pinniped => self.rf_pinniped = Some(model),
            Taxon::Insect => self.rf_insect = Some(model),
            Taxon::Mammal => self.rf_mammal = Some(model),
            Taxon::Unknown => self.rf_fallback = Some(model),
        }
    }

    /// Get the number of registered specialists (excluding fallback)
    pub fn count(&self) -> usize {
        [
            self.rf_cetacean.is_some(),
            self.rf_mysticete.is_some(),
            self.rf_songbird.is_some(),
            self.rf_non_passerine.is_some(),
            self.rf_amphibian.is_some(),
            self.rf_pinniped.is_some(),
            self.rf_insect.is_some(),
            self.rf_mammal.is_some(),
        ]
        .iter()
        .filter(|&&x| x)
        .count()
    }

    /// Check if registry has any specialists
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// List all available specialists
    pub fn available_specialists(&self) -> Vec<Taxon> {
        let mut available = Vec::new();
        if self.rf_cetacean.is_some() {
            available.push(Taxon::Cetacean);
        }
        if self.rf_mysticete.is_some() {
            available.push(Taxon::Mysticete);
        }
        if self.rf_songbird.is_some() {
            available.push(Taxon::Songbird);
        }
        if self.rf_non_passerine.is_some() {
            available.push(Taxon::NonPasserine);
        }
        if self.rf_amphibian.is_some() {
            available.push(Taxon::Amphibian);
        }
        if self.rf_pinniped.is_some() {
            available.push(Taxon::Pinniped);
        }
        if self.rf_insect.is_some() {
            available.push(Taxon::Insect);
        }
        if self.rf_mammal.is_some() {
            available.push(Taxon::Mammal);
        }
        available
    }
}

// =============================================================================
// NN Model Interface (for Curriculum NN)
// =============================================================================

/// Interface for Neural Network models (Curriculum NN)
///
/// This trait defines the interface expected from the Curriculum NN.
/// The actual implementation is in `train_curriculum_nn_112d.rs`.
pub trait NeuralNetworkModel: Send + Sync {
    /// Predict species from 112D features
    fn predict(&self, features: &[f32]) -> (usize, String);

    /// Get probability distribution over all classes
    fn predict_proba(&self, features: &[f32]) -> Vec<f32>;

    /// Get class labels
    fn class_labels(&self) -> &[String];

    /// Get number of classes
    fn n_classes(&self) -> usize;

    /// Predict group probabilities (Physics Block only - Stage 1)
    fn predict_group_proba(&self, features: &[f32]) -> Vec<f32>;
}

// =============================================================================
// Hierarchical Ensemble Router
// =============================================================================

/// The main Hierarchical Ensemble Router
///
/// Combines RF Gatekeeper, Specialist RFs, and Curriculum NN into a
/// unified two-stage classification pipeline.
pub struct HierarchicalEnsembleRouter {
    /// Configuration
    pub config: RouterConfig,
    /// RF Gatekeeper for Stage 1 (trained on 76D gatekeeper input)
    pub gatekeeper_rf: Option<RFModel>,
    /// Specialist RFs for Stage 2
    pub specialists: SpecialistRegistry,
    /// Neural Network model (optional)
    pub nn_model: Option<Box<dyn NeuralNetworkModel>>,
    /// Mapping from species labels to taxonomic groups
    pub species_to_taxon: HashMap<String, Taxon>,
    /// Mapping from species to consolidated taxonomic groups
    pub species_to_consolidated: HashMap<String, ConsolidatedTaxon>,
}

impl HierarchicalEnsembleRouter {
    /// Create a new router with default configuration
    pub fn new() -> Result<Self, String> {
        let config = RouterConfig::default();
        config.validate()?;
        Ok(Self {
            config,
            gatekeeper_rf: None,
            specialists: SpecialistRegistry::new(),
            nn_model: None,
            species_to_taxon: HashMap::new(),
            species_to_consolidated: HashMap::new(),
        })
    }

    /// Create a router with custom configuration
    pub fn with_config(config: RouterConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            gatekeeper_rf: None,
            specialists: SpecialistRegistry::new(),
            nn_model: None,
            species_to_taxon: HashMap::new(),
            species_to_consolidated: HashMap::new(),
        })
    }

    /// Load the gatekeeper RF model
    pub fn load_gatekeeper(&mut self, model: RFModel) {
        self.gatekeeper_rf = Some(model);
    }

    /// Load a specialist RF model for a detailed taxonomic group
    pub fn load_specialist(&mut self, taxon: Taxon, model: RFModel) {
        self.specialists.load_specialist(taxon, model);
    }

    /// Load a specialist RF model for a consolidated group (resolves to best specialist)
    pub fn load_specialist_for_consolidated(&mut self, taxon: ConsolidatedTaxon, model: RFModel) {
        // Map consolidated taxon to preferred detailed taxon for storage
        let detailed_taxon = match taxon {
            ConsolidatedTaxon::Bird => Taxon::Songbird,         // Default to songbird
            ConsolidatedTaxon::MarineMammal => Taxon::Cetacean, // Default to cetacean
            ConsolidatedTaxon::Mammal => Taxon::Mammal,
            ConsolidatedTaxon::Insect => Taxon::Insect,
            ConsolidatedTaxon::Amphibian => Taxon::Amphibian,
            ConsolidatedTaxon::Unknown => Taxon::Unknown,
        };
        self.specialists.load_specialist(detailed_taxon, model);
    }

    /// Load the NN model
    pub fn load_nn(&mut self, model: Box<dyn NeuralNetworkModel>) {
        self.nn_model = Some(model);
        self.build_species_mappings();
    }

    /// Build species-to-taxon mappings from NN model
    fn build_species_mappings(&mut self) {
        if let Some(nn) = &self.nn_model {
            for label in nn.class_labels() {
                let taxon = map_species_to_taxon(label);
                let consolidated = consolidate_taxon(taxon);
                self.species_to_taxon.insert(label.clone(), taxon);
                self.species_to_consolidated.insert(label.clone(), consolidated);
            }
        }
    }

    /// Check if the router is ready for classification
    pub fn is_ready(&self) -> bool {
        self.gatekeeper_rf.is_some() || self.nn_model.is_some()
    }

    /// Classify a 112D feature vector
    pub fn classify(&self, features: &[f32]) -> Result<RouterResult, String> {
        if features.len() != FEATURE_DIM {
            return Err(format!(
                "Invalid feature dimension: expected {}, got {}",
                FEATURE_DIM,
                features.len()
            ));
        }

        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();

        // === STAGE 1: GROUP DETECTION ===
        let stage1_result = self.stage1_classify(features)?;

        // Check confidence threshold
        if stage1_result.confidence < self.config.min_stage1_confidence {
            warnings.push(format!(
                "Stage 1 confidence ({:.2}%) below threshold ({:.2}%)",
                stage1_result.confidence * 100.0,
                self.config.min_stage1_confidence * 100.0
            ));
        }

        // === FEATURE REWEIGHTING ===
        let reweighted_features = if self.config.apply_feature_reweighting {
            // Convert ConsolidatedTaxon to Taxon for weight lookup
            let taxon = self.consolidated_to_taxon(stage1_result.predicted_group);
            apply_taxonomic_mask(features, taxon)
        } else {
            features.to_vec()
        };

        // === STAGE 2: SPECIES DISCRIMINATION ===
        let stage2_result = self.stage2_classify(
            &reweighted_features,
            stage1_result.predicted_group,
            self.config.apply_feature_reweighting,
        )?;

        // Determine reliability
        let is_reliable =
            stage1_result.confidence >= self.config.min_stage1_confidence && stage2_result.confidence >= 0.3;

        let processing_time_us = start_time.elapsed().as_micros() as u64;

        // Get detailed taxon from species
        let detailed_taxon = map_species_to_taxon(&stage2_result.species);

        Ok(RouterResult {
            species: stage2_result.species.clone(),
            confidence: stage2_result.confidence,
            predicted_group: stage1_result.predicted_group,
            detailed_taxon,
            stage1: stage1_result,
            stage2: stage2_result,
            processing_time_us,
            is_reliable,
            warnings,
        })
    }

    /// Stage 1: Group Detection
    fn stage1_classify(&self, features: &[f32]) -> Result<Stage1Result, String> {
        // Extract gatekeeper input (76D)
        let gatekeeper_features = slice_gatekeeper_input(features);

        // Get RF prediction
        let (rf_pred, rf_conf, rf_proba) = if let Some(rf) = &self.gatekeeper_rf {
            let proba = rf.predict_proba(&gatekeeper_features);
            let pred_idx = proba
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let conf = proba.get(pred_idx).copied().unwrap_or(0.0);
            let pred = idx_to_consolidated_taxon(pred_idx);
            (pred, conf, proba)
        } else {
            // No RF - use fallback
            (ConsolidatedTaxon::Unknown, 0.0, vec![0.0; CONSOLIDATED_TAXON_COUNT])
        };

        // Get NN prediction (if enabled)
        let (nn_pred, nn_conf, nn_proba) = if self.config.enable_nn {
            if let Some(nn) = &self.nn_model {
                let proba = nn.predict_group_proba(features);
                let pred_idx = proba
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let conf = proba.get(pred_idx).copied().unwrap_or(0.0);
                let pred = idx_to_consolidated_taxon(pred_idx);
                (Some(pred), Some(conf), Some(proba))
            } else {
                (None, None, None)
            }
        } else {
            (None, None, None)
        };

        // Ensemble voting
        let (final_pred, final_conf) =
            if let (Some(nn_p), Some(_nn_pred), Some(_nn_conf)) = (&nn_proba, &nn_pred, &nn_conf) {
                // Both RF and NN available - combine
                let mut combined_proba = [0.0; CONSOLIDATED_TAXON_COUNT];

                for i in 0..CONSOLIDATED_TAXON_COUNT {
                    combined_proba[i] = self.config.stage1_rf_weight * rf_proba.get(i).copied().unwrap_or(0.0)
                        + self.config.stage1_nn_weight * nn_p.get(i).copied().unwrap_or(0.0);
                }

                let pred_idx = combined_proba
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                (
                    idx_to_consolidated_taxon(pred_idx),
                    combined_proba.get(pred_idx).copied().unwrap_or(0.0),
                )
            } else {
                // Only RF available
                (rf_pred, rf_conf)
            };

        Ok(Stage1Result {
            predicted_group: final_pred,
            confidence: final_conf,
            rf_prediction: rf_pred,
            rf_confidence: rf_conf,
            rf_proba,
            nn_prediction: nn_pred,
            nn_confidence: nn_conf,
            nn_proba,
            rf_weight: self.config.stage1_rf_weight,
            nn_weight: self.config.stage1_nn_weight,
        })
    }

    /// Stage 2: Species Discrimination
    fn stage2_classify(
        &self,
        features: &[f32],
        predicted_group: ConsolidatedTaxon,
        reweighting_applied: bool,
    ) -> Result<Stage2Result, String> {
        // Get specialist RF prediction using best available specialist
        let (rf_pred, rf_conf) =
            if let Some(specialist) = self.specialists.get_best_specialist_for_consolidated(predicted_group) {
                let pred_label = specialist.predict_label(features);
                let conf = specialist.confidence(features);
                (pred_label.to_string(), conf)
            } else {
                // No specialist available - try fallback
                if let Some(fallback) = &self.specialists.rf_fallback {
                    let pred_label = fallback.predict_label(features);
                    let conf = fallback.confidence(features);
                    (pred_label.to_string(), conf)
                } else {
                    ("<unknown>".to_string(), 0.0)
                }
            };

        // Get NN prediction (if enabled)
        let (nn_pred, nn_conf) = if self.config.enable_nn {
            if let Some(nn) = &self.nn_model {
                // Apply taxonomic masking to NN logits
                let raw_logits = nn.predict_proba(features);
                let masked_logits =
                    self.apply_taxonomic_mask_to_logits(&raw_logits, predicted_group, nn.class_labels());

                // Get top prediction from masked logits
                let pred_idx = masked_logits
                    .iter()
                    .enumerate()
                    .filter(|(_, &l)| l > f32::NEG_INFINITY)
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i);

                if let Some(idx) = pred_idx {
                    let label = nn.class_labels().get(idx).cloned().unwrap_or_default();
                    // Recompute softmax over valid logits
                    let conf = self.softmax_confidence(&masked_logits, idx);
                    (Some(label), Some(conf))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // Ensemble voting for species
        let (final_species, final_conf, candidates) =
            self.ensemble_species_prediction(&rf_pred, rf_conf, nn_pred.as_deref(), nn_conf, predicted_group);

        // Build candidates list
        let taxon = map_species_to_taxon(&final_species);

        Ok(Stage2Result {
            species: final_species,
            confidence: final_conf,
            taxon,
            candidates,
            rf_prediction: rf_pred,
            rf_confidence: rf_conf,
            nn_prediction: nn_pred,
            nn_confidence: nn_conf,
            rf_weight: self.config.stage2_rf_weight,
            nn_weight: self.config.stage2_nn_weight,
            reweighting_applied,
        })
    }

    /// Apply taxonomic mask to NN logits
    fn apply_taxonomic_mask_to_logits(
        &self,
        logits: &[f32],
        predicted_group: ConsolidatedTaxon,
        labels: &[String],
    ) -> Vec<f32> {
        logits
            .iter()
            .enumerate()
            .map(|(i, &logit)| {
                let species_group = self
                    .species_to_consolidated
                    .get(labels.get(i).unwrap_or(&String::new()))
                    .copied()
                    .unwrap_or(ConsolidatedTaxon::Unknown);

                if species_group == predicted_group || predicted_group == ConsolidatedTaxon::Unknown {
                    logit
                } else {
                    f32::NEG_INFINITY // Mask out non-matching species
                }
            })
            .collect()
    }

    /// Compute softmax confidence for a specific index
    fn softmax_confidence(&self, logits: &[f32], target_idx: usize) -> f32 {
        let valid_logits: Vec<f32> = logits.iter().filter(|&&l| l > f32::NEG_INFINITY).copied().collect();

        if valid_logits.is_empty() {
            return 0.0;
        }

        let max_logit = valid_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = valid_logits.iter().map(|&l| (l - max_logit).exp()).sum();

        if let Some(&target_logit) = logits.get(target_idx) {
            if target_logit > f32::NEG_INFINITY {
                return (target_logit - max_logit).exp() / exp_sum;
            }
        }
        0.0
    }

    /// Ensemble species prediction from RF and NN
    fn ensemble_species_prediction(
        &self,
        rf_pred: &str,
        rf_conf: f32,
        nn_pred: Option<&str>,
        nn_conf: Option<f32>,
        _predicted_group: ConsolidatedTaxon,
    ) -> (String, f32, Vec<SpeciesCandidate>) {
        // If only RF available, use RF prediction
        if nn_pred.is_none() || nn_conf.is_none() {
            return (
                rf_pred.to_string(),
                rf_conf,
                vec![SpeciesCandidate {
                    label: rf_pred.to_string(),
                    confidence: rf_conf,
                    taxon: map_species_to_taxon(rf_pred),
                    rank: 1,
                }],
            );
        }

        let nn_p = nn_pred.unwrap();
        let nn_c = nn_conf.unwrap();

        // Simple weighted average for two models
        // If predictions agree, boost confidence
        if rf_pred == nn_p {
            // Agreement - boost confidence
            let combined_conf = (rf_conf + nn_c) / 2.0 * 1.1; // 10% boost
            return (
                rf_pred.to_string(),
                combined_conf.min(1.0),
                vec![SpeciesCandidate {
                    label: rf_pred.to_string(),
                    confidence: combined_conf.min(1.0),
                    taxon: map_species_to_taxon(rf_pred),
                    rank: 1,
                }],
            );
        }

        // Disagreement - use weighted selection
        let rf_score = self.config.stage2_rf_weight * rf_conf;
        let nn_score = self.config.stage2_nn_weight * nn_c;

        if rf_score >= nn_score {
            (
                rf_pred.to_string(),
                rf_conf,
                vec![
                    SpeciesCandidate {
                        label: rf_pred.to_string(),
                        confidence: rf_conf,
                        taxon: map_species_to_taxon(rf_pred),
                        rank: 1,
                    },
                    SpeciesCandidate {
                        label: nn_p.to_string(),
                        confidence: nn_c,
                        taxon: map_species_to_taxon(nn_p),
                        rank: 2,
                    },
                ],
            )
        } else {
            (
                nn_p.to_string(),
                nn_c,
                vec![
                    SpeciesCandidate {
                        label: nn_p.to_string(),
                        confidence: nn_c,
                        taxon: map_species_to_taxon(nn_p),
                        rank: 1,
                    },
                    SpeciesCandidate {
                        label: rf_pred.to_string(),
                        confidence: rf_conf,
                        taxon: map_species_to_taxon(rf_pred),
                        rank: 2,
                    },
                ],
            )
        }
    }

    /// Convert ConsolidatedTaxon to Taxon for weight lookup
    fn consolidated_to_taxon(&self, taxon: ConsolidatedTaxon) -> Taxon {
        match taxon {
            ConsolidatedTaxon::Bird => Taxon::Songbird, // Default to songbird for weights
            ConsolidatedTaxon::Mammal => Taxon::Mammal,
            ConsolidatedTaxon::MarineMammal => Taxon::Cetacean, // Default to cetacean
            ConsolidatedTaxon::Insect => Taxon::Insect,
            ConsolidatedTaxon::Amphibian => Taxon::Amphibian,
            ConsolidatedTaxon::Unknown => Taxon::Unknown,
        }
    }

    /// Batch classification
    pub fn classify_batch(&self, features_batch: &[Vec<f32>]) -> Vec<Result<RouterResult, String>> {
        features_batch.iter().map(|features| self.classify(features)).collect()
    }

    /// Evaluate accuracy on a test set
    pub fn evaluate(&self, features_batch: &[Vec<f32>], labels: &[String]) -> RouterMetrics {
        let mut correct_species = 0usize;
        let mut correct_group = 0usize;
        let mut total = 0usize;
        let mut reliable_count = 0usize;
        let mut reliable_correct = 0usize;
        let mut rf_only_count = 0usize;
        let mut nn_only_count = 0usize;
        let mut agreement_count = 0usize;

        for (features, true_label) in features_batch.iter().zip(labels.iter()) {
            if let Ok(result) = self.classify(features) {
                total += 1;

                if result.species == *true_label {
                    correct_species += 1;
                }

                let true_group = self
                    .species_to_consolidated
                    .get(true_label)
                    .copied()
                    .unwrap_or_else(|| consolidate_taxon(map_species_to_taxon(true_label)));

                if result.predicted_group == true_group {
                    correct_group += 1;
                }

                if result.is_reliable {
                    reliable_count += 1;
                    if result.species == *true_label {
                        reliable_correct += 1;
                    }
                }

                // Track model usage
                if result.stage2.nn_prediction.is_none() {
                    rf_only_count += 1;
                } else if result.stage2.rf_confidence < 0.01 {
                    nn_only_count += 1;
                }

                // Track agreement
                if result.stage2.rf_prediction.as_str() == result.stage2.nn_prediction.as_deref().unwrap_or("") {
                    agreement_count += 1;
                }
            }
        }

        RouterMetrics {
            species_accuracy: if total > 0 {
                correct_species as f32 / total as f32 * 100.0
            } else {
                0.0
            },
            group_accuracy: if total > 0 {
                correct_group as f32 / total as f32 * 100.0
            } else {
                0.0
            },
            total_samples: total,
            correct_species,
            correct_group,
            reliable_samples: reliable_count,
            reliable_accuracy: if reliable_count > 0 {
                reliable_correct as f32 / reliable_count as f32 * 100.0
            } else {
                0.0
            },
            rf_only_count,
            nn_only_count,
            agreement_count,
            agreement_rate: if total > 0 {
                agreement_count as f32 / total as f32 * 100.0
            } else {
                0.0
            },
        }
    }
}

// =============================================================================
// Metrics
// =============================================================================

/// Metrics for evaluating the Hierarchical Ensemble Router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterMetrics {
    /// Species classification accuracy (%)
    pub species_accuracy: f32,
    /// Taxonomic group accuracy (%)
    pub group_accuracy: f32,
    /// Total samples evaluated
    pub total_samples: usize,
    /// Number of correct species predictions
    pub correct_species: usize,
    /// Number of correct group predictions
    pub correct_group: usize,
    /// Number of reliable predictions
    pub reliable_samples: usize,
    /// Accuracy on reliable predictions only (%)
    pub reliable_accuracy: f32,
    /// Number of predictions using RF only
    pub rf_only_count: usize,
    /// Number of predictions using NN only
    pub nn_only_count: usize,
    /// Number of times RF and NN agreed
    pub agreement_count: usize,
    /// Agreement rate between RF and NN (%)
    pub agreement_rate: f32,
}

impl RouterMetrics {
    /// Calculate improvement of species accuracy over group accuracy
    pub fn species_over_group_improvement(&self) -> f32 {
        self.species_accuracy - self.group_accuracy
    }

    /// Calculate reliability rate (percentage of reliable predictions)
    pub fn reliability_rate(&self) -> f32 {
        if self.total_samples > 0 {
            self.reliable_samples as f32 / self.total_samples as f32 * 100.0
        } else {
            0.0
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // TDD Test Suite: Configuration
    // =========================================================================

    #[test]
    fn test_default_config_valid() {
        let config = RouterConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_stage1_weights_sum_to_one() {
        let config = RouterConfig::default();
        assert!((config.stage1_rf_weight + config.stage1_nn_weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_stage2_weights_sum_to_one() {
        let config = RouterConfig::default();
        assert!((config.stage2_rf_weight + config.stage2_nn_weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_invalid_weights_rejected() {
        let config = RouterConfig {
            stage1_rf_weight: 0.8,
            stage1_nn_weight: 0.3, // Sum = 1.1
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_negative_weights_rejected() {
        let config = RouterConfig {
            stage1_rf_weight: -0.1,
            stage1_nn_weight: 1.1,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_rf_only() {
        let config = RouterConfig::rf_only();
        assert!(config.validate().is_ok());
        assert_eq!(config.stage1_rf_weight, 1.0);
        assert_eq!(config.stage1_nn_weight, 0.0);
        assert!(!config.enable_nn);
    }

    #[test]
    fn test_config_nn_only() {
        let config = RouterConfig::nn_only();
        assert!(config.validate().is_ok());
        assert_eq!(config.stage1_rf_weight, 0.0);
        assert_eq!(config.stage1_nn_weight, 1.0);
        assert!(config.enable_nn);
    }

    #[test]
    fn test_config_balanced() {
        let config = RouterConfig::balanced();
        assert!(config.validate().is_ok());
        assert_eq!(config.stage1_rf_weight, 0.5);
        assert_eq!(config.stage1_nn_weight, 0.5);
    }

    // =========================================================================
    // TDD Test Suite: Router Creation
    // =========================================================================

    #[test]
    fn test_router_creation() {
        let router = HierarchicalEnsembleRouter::new();
        assert!(router.is_ok());
    }

    #[test]
    fn test_router_not_ready_without_models() {
        let router = HierarchicalEnsembleRouter::new().unwrap();
        assert!(!router.is_ready());
    }

    #[test]
    fn test_router_custom_config() {
        let config = RouterConfig::balanced();
        let router = HierarchicalEnsembleRouter::with_config(config);
        assert!(router.is_ok());
        assert_eq!(router.unwrap().config.stage1_rf_weight, 0.5);
    }

    // =========================================================================
    // TDD Test Suite: Specialist Registry
    // =========================================================================

    #[test]
    fn test_specialist_registry_empty() {
        let registry = SpecialistRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_specialist_registry_has_specialist() {
        let mut registry = SpecialistRegistry::new();
        registry.rf_songbird = Some(RFModel::new(FEATURE_DIM));
        assert!(registry.has_specialist(Taxon::Songbird));
        assert!(!registry.has_specialist(Taxon::Mammal));
    }

    #[test]
    fn test_specialist_registry_has_specialist_for_consolidated() {
        let mut registry = SpecialistRegistry::new();
        // Only songbird specialist available
        registry.rf_songbird = Some(RFModel::new(FEATURE_DIM));

        // Should find songbird specialist for Bird group
        assert!(registry.has_specialist_for_consolidated(ConsolidatedTaxon::Bird));

        // Should not find specialist for Mammal group
        assert!(!registry.has_specialist_for_consolidated(ConsolidatedTaxon::Mammal));
    }

    #[test]
    fn test_specialist_registry_get_specialist() {
        let mut registry = SpecialistRegistry::new();
        let model = RFModel::new(FEATURE_DIM);
        registry.rf_mammal = Some(model);
        assert!(registry.get_specialist(Taxon::Mammal).is_some());
        assert!(registry.get_specialist(Taxon::Songbird).is_none());
    }

    #[test]
    fn test_specialist_registry_get_best_for_consolidated_bird() {
        let mut registry = SpecialistRegistry::new();
        registry.rf_songbird = Some(RFModel::new(FEATURE_DIM));

        // Should find songbird for Bird group
        let specialist = registry.get_best_specialist_for_consolidated(ConsolidatedTaxon::Bird);
        assert!(specialist.is_some());
    }

    #[test]
    fn test_specialist_registry_get_best_for_consolidated_marine() {
        let mut registry = SpecialistRegistry::new();

        // Only pinniped available - should fallback to pinniped for MarineMammal
        registry.rf_pinniped = Some(RFModel::new(FEATURE_DIM));
        let specialist = registry.get_best_specialist_for_consolidated(ConsolidatedTaxon::MarineMammal);
        assert!(specialist.is_some());

        // Now add cetacean - should prefer cetacean
        registry.rf_cetacean = Some(RFModel::new(FEATURE_DIM));
        // Both available, cetacean is preferred
        assert!(registry
            .get_best_specialist_for_consolidated(ConsolidatedTaxon::MarineMammal)
            .is_some());
    }

    #[test]
    fn test_specialist_registry_load() {
        let mut registry = SpecialistRegistry::new();
        registry.load_specialist(Taxon::Songbird, RFModel::new(FEATURE_DIM));
        assert!(registry.rf_songbird.is_some());

        registry.load_specialist(Taxon::Cetacean, RFModel::new(FEATURE_DIM));
        assert!(registry.rf_cetacean.is_some());
    }

    #[test]
    fn test_specialist_registry_count() {
        let mut registry = SpecialistRegistry::new();
        assert_eq!(registry.count(), 0);

        registry.rf_songbird = Some(RFModel::new(FEATURE_DIM));
        assert_eq!(registry.count(), 1);

        registry.rf_mammal = Some(RFModel::new(FEATURE_DIM));
        assert_eq!(registry.count(), 2);

        // Add all 8 specialists
        registry.rf_cetacean = Some(RFModel::new(FEATURE_DIM));
        registry.rf_mysticete = Some(RFModel::new(FEATURE_DIM));
        registry.rf_non_passerine = Some(RFModel::new(FEATURE_DIM));
        registry.rf_amphibian = Some(RFModel::new(FEATURE_DIM));
        registry.rf_pinniped = Some(RFModel::new(FEATURE_DIM));
        registry.rf_insect = Some(RFModel::new(FEATURE_DIM));
        assert_eq!(registry.count(), 8);
    }

    #[test]
    fn test_specialist_registry_available_specialists() {
        let mut registry = SpecialistRegistry::new();
        registry.rf_songbird = Some(RFModel::new(FEATURE_DIM));
        registry.rf_cetacean = Some(RFModel::new(FEATURE_DIM));

        let available = registry.available_specialists();
        assert_eq!(available.len(), 2);
        assert!(available.contains(&Taxon::Songbird));
        assert!(available.contains(&Taxon::Cetacean));
    }

    // =========================================================================
    // TDD Test Suite: Classification Errors
    // =========================================================================

    #[test]
    fn test_classify_wrong_dimension() {
        let router = HierarchicalEnsembleRouter::new().unwrap();
        let features = vec![0.0; 50]; // Wrong dimension
        let result = router.classify(&features);
        assert!(result.is_err());
    }

    #[test]
    fn test_classify_correct_dimension() {
        let router = HierarchicalEnsembleRouter::new().unwrap();
        let features = vec![0.0; FEATURE_DIM];
        // Even without models, should not panic on dimension check
        // (will fail later but that's expected)
        let _ = router.classify(&features);
    }

    // =========================================================================
    // TDD Test Suite: Species Candidate
    // =========================================================================

    #[test]
    fn test_species_candidate_creation() {
        let candidate = SpeciesCandidate {
            label: "Eastern Towhee".to_string(),
            confidence: 0.85,
            taxon: Taxon::Songbird,
            rank: 1,
        };
        assert_eq!(candidate.label, "Eastern Towhee");
        assert!((candidate.confidence - 0.85).abs() < 1e-6);
        assert_eq!(candidate.taxon, Taxon::Songbird);
        assert_eq!(candidate.rank, 1);
    }

    // =========================================================================
    // TDD Test Suite: Stage 1 Result
    // =========================================================================

    #[test]
    fn test_stage1_result_creation() {
        let result = Stage1Result {
            predicted_group: ConsolidatedTaxon::Bird,
            confidence: 0.92,
            rf_prediction: ConsolidatedTaxon::Bird,
            rf_confidence: 0.95,
            rf_proba: vec![0.95, 0.02, 0.01, 0.01, 0.005, 0.005],
            nn_prediction: Some(ConsolidatedTaxon::Bird),
            nn_confidence: Some(0.88),
            nn_proba: Some(vec![0.88, 0.05, 0.03, 0.02, 0.01, 0.01]),
            rf_weight: 0.95,
            nn_weight: 0.05,
        };
        assert_eq!(result.predicted_group, ConsolidatedTaxon::Bird);
        assert!((result.confidence - 0.92).abs() < 1e-6);
    }

    // =========================================================================
    // TDD Test Suite: Stage 2 Result
    // =========================================================================

    #[test]
    fn test_stage2_result_creation() {
        let result = Stage2Result {
            species: "Eastern Towhee".to_string(),
            confidence: 0.87,
            taxon: Taxon::Songbird,
            candidates: vec![SpeciesCandidate {
                label: "Eastern Towhee".to_string(),
                confidence: 0.87,
                taxon: Taxon::Songbird,
                rank: 1,
            }],
            rf_prediction: "Eastern Towhee".to_string(),
            rf_confidence: 0.90,
            nn_prediction: Some("Eastern Towhee".to_string()),
            nn_confidence: Some(0.84),
            rf_weight: 0.50,
            nn_weight: 0.50,
            reweighting_applied: true,
        };
        assert_eq!(result.species, "Eastern Towhee");
        assert!(result.reweighting_applied);
    }

    // =========================================================================
    // TDD Test Suite: Router Result
    // =========================================================================

    #[test]
    fn test_router_result_is_reliable() {
        let mut warnings = Vec::new();
        warnings.push("Low confidence".to_string());

        let result = RouterResult {
            species: "Bat".to_string(),
            confidence: 0.45,
            predicted_group: ConsolidatedTaxon::Mammal,
            detailed_taxon: Taxon::Mammal,
            stage1: Stage1Result {
                predicted_group: ConsolidatedTaxon::Mammal,
                confidence: 0.50,
                rf_prediction: ConsolidatedTaxon::Mammal,
                rf_confidence: 0.55,
                rf_proba: vec![0.0; 6],
                nn_prediction: None,
                nn_confidence: None,
                nn_proba: None,
                rf_weight: 1.0,
                nn_weight: 0.0,
            },
            stage2: Stage2Result {
                species: "Bat".to_string(),
                confidence: 0.45,
                taxon: Taxon::Mammal,
                candidates: vec![],
                rf_prediction: "Bat".to_string(),
                rf_confidence: 0.50,
                nn_prediction: None,
                nn_confidence: None,
                rf_weight: 1.0,
                nn_weight: 0.0,
                reweighting_applied: false,
            },
            processing_time_us: 150,
            is_reliable: false,
            warnings,
        };
        assert!(!result.is_reliable);
        assert_eq!(result.warnings.len(), 1);
    }

    // =========================================================================
    // TDD Test Suite: Metrics
    // =========================================================================

    #[test]
    fn test_router_metrics_creation() {
        let metrics = RouterMetrics {
            species_accuracy: 75.0,
            group_accuracy: 95.0,
            total_samples: 100,
            correct_species: 75,
            correct_group: 95,
            reliable_samples: 80,
            reliable_accuracy: 85.0,
            rf_only_count: 10,
            nn_only_count: 5,
            agreement_count: 70,
            agreement_rate: 70.0,
        };
        assert!((metrics.species_accuracy - 75.0).abs() < 1e-6);
        assert!((metrics.group_accuracy - 95.0).abs() < 1e-6);
    }

    #[test]
    fn test_metrics_improvement_calculation() {
        let metrics = RouterMetrics {
            species_accuracy: 75.0,
            group_accuracy: 95.0,
            total_samples: 100,
            correct_species: 75,
            correct_group: 95,
            reliable_samples: 80,
            reliable_accuracy: 85.0,
            rf_only_count: 10,
            nn_only_count: 5,
            agreement_count: 70,
            agreement_rate: 70.0,
        };
        // Species - Group = 75 - 95 = -20
        assert!((metrics.species_over_group_improvement() - (-20.0)).abs() < 1e-6);
    }

    #[test]
    fn test_metrics_reliability_rate() {
        let metrics = RouterMetrics {
            species_accuracy: 75.0,
            group_accuracy: 95.0,
            total_samples: 100,
            correct_species: 75,
            correct_group: 95,
            reliable_samples: 80,
            reliable_accuracy: 85.0,
            rf_only_count: 10,
            nn_only_count: 5,
            agreement_count: 70,
            agreement_rate: 70.0,
        };
        // 80 / 100 = 80%
        assert!((metrics.reliability_rate() - 80.0).abs() < 1e-6);
    }

    // =========================================================================
    // TDD Test Suite: Softmax Confidence
    // =========================================================================

    #[test]
    fn test_softmax_confidence_single_value() {
        let router = HierarchicalEnsembleRouter::new().unwrap();
        let logits = vec![1.0, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let conf = router.softmax_confidence(&logits, 0);
        assert!((conf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_confidence_equal_values() {
        let router = HierarchicalEnsembleRouter::new().unwrap();
        let logits = vec![1.0, 1.0, 1.0];
        let conf = router.softmax_confidence(&logits, 0);
        // With 3 equal values, each should get ~33.3%
        assert!((conf - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_softmax_confidence_masked() {
        let router = HierarchicalEnsembleRouter::new().unwrap();
        let logits = vec![f32::NEG_INFINITY, 2.0, 3.0];
        let conf = router.softmax_confidence(&logits, 1);
        // Only indices 1 and 2 are valid
        assert!(conf > 0.0 && conf < 1.0);
    }

    // =========================================================================
    // TDD Test Suite: Load Specialists
    // =========================================================================

    #[test]
    fn test_load_specialist_songbird() {
        let mut router = HierarchicalEnsembleRouter::new().unwrap();
        let model = RFModel::new(FEATURE_DIM);
        router.load_specialist(Taxon::Songbird, model);
        assert!(router.specialists.rf_songbird.is_some());
    }

    #[test]
    fn test_load_specialist_cetacean() {
        let mut router = HierarchicalEnsembleRouter::new().unwrap();
        let model = RFModel::new(FEATURE_DIM);
        router.load_specialist(Taxon::Cetacean, model);
        assert!(router.specialists.rf_cetacean.is_some());
    }

    #[test]
    fn test_load_specialist_all() {
        let mut router = HierarchicalEnsembleRouter::new().unwrap();

        for taxon in [
            Taxon::Cetacean,
            Taxon::Mysticete,
            Taxon::Songbird,
            Taxon::NonPasserine,
            Taxon::Amphibian,
            Taxon::Pinniped,
            Taxon::Insect,
            Taxon::Mammal,
        ] {
            router.load_specialist(taxon, RFModel::new(FEATURE_DIM));
        }

        assert_eq!(router.specialists.count(), 8);
    }

    #[test]
    fn test_load_specialist_for_consolidated_bird() {
        let mut router = HierarchicalEnsembleRouter::new().unwrap();
        let model = RFModel::new(FEATURE_DIM);
        router.load_specialist_for_consolidated(ConsolidatedTaxon::Bird, model);
        // Should default to Songbird storage
        assert!(router.specialists.rf_songbird.is_some());
    }

    #[test]
    fn test_load_specialist_for_consolidated_marine() {
        let mut router = HierarchicalEnsembleRouter::new().unwrap();
        let model = RFModel::new(FEATURE_DIM);
        router.load_specialist_for_consolidated(ConsolidatedTaxon::MarineMammal, model);
        // Should default to Cetacean storage
        assert!(router.specialists.rf_cetacean.is_some());
    }

    // =========================================================================
    // TDD Test Suite: Gatekeeper Loading
    // =========================================================================

    #[test]
    fn test_load_gatekeeper() {
        let mut router = HierarchicalEnsembleRouter::new().unwrap();
        let model = RFModel::new(GATEKEEPER_DIM);
        router.load_gatekeeper(model);
        assert!(router.gatekeeper_rf.is_some());
        assert!(router.is_ready());
    }

    // =========================================================================
    // TDD Test Suite: Feature Dimension Constants
    // =========================================================================

    #[test]
    fn test_feature_dimensions() {
        assert_eq!(FEATURE_DIM, 112);
        assert_eq!(PHYSICS_DIM, 46);
        assert_eq!(MACRO_TEXTURE_DIM, 30);
        assert_eq!(MICRO_TEXTURE_DIM, 36);
        assert_eq!(GATEKEEPER_DIM, 76); // 46 + 30
        assert_eq!(SPECIES_EXPERT_DIM, 82); // 46 + 36
    }

    #[test]
    fn test_dimensions_sum_correctly() {
        assert_eq!(PHYSICS_DIM + MACRO_TEXTURE_DIM + MICRO_TEXTURE_DIM, FEATURE_DIM);
        assert_eq!(PHYSICS_DIM + MACRO_TEXTURE_DIM, GATEKEEPER_DIM);
        assert_eq!(PHYSICS_DIM + MICRO_TEXTURE_DIM, SPECIES_EXPERT_DIM);
    }

    // =========================================================================
    // TDD Test Suite: Weight Constants
    // =========================================================================

    #[test]
    fn test_stage1_weights() {
        assert!((STAGE1_RF_WEIGHT - 0.95).abs() < 1e-6);
        assert!((STAGE1_NN_WEIGHT - 0.05).abs() < 1e-6);
        assert!((STAGE1_RF_WEIGHT + STAGE1_NN_WEIGHT - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stage2_weights() {
        assert!((STAGE2_RF_WEIGHT - 0.50).abs() < 1e-6);
        assert!((STAGE2_NN_WEIGHT - 0.50).abs() < 1e-6);
        assert!((STAGE2_RF_WEIGHT + STAGE2_NN_WEIGHT - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_min_stage1_confidence() {
        assert!((MIN_STAGE1_CONFIDENCE - 0.3).abs() < 1e-6);
    }

    // =========================================================================
    // TDD Test Suite: Batch Prediction
    // =========================================================================

    #[test]
    fn test_predict_batch_empty() {
        let router = HierarchicalEnsembleRouter::new().unwrap();
        let batch: Vec<Vec<f32>> = vec![];
        let results = router.classify_batch(&batch);
        assert!(results.is_empty());
    }

    // =========================================================================
    // TDD Test Suite: Species Candidate Ordering
    // =========================================================================

    #[test]
    fn test_species_candidate_confidence_ordering() {
        let candidates = vec![
            SpeciesCandidate {
                label: "SpeciesC".to_string(),
                confidence: 0.5,
                taxon: Taxon::Mammal,
                rank: 3,
            },
            SpeciesCandidate {
                label: "SpeciesA".to_string(),
                confidence: 0.9,
                taxon: Taxon::Songbird,
                rank: 1,
            },
            SpeciesCandidate {
                label: "SpeciesB".to_string(),
                confidence: 0.7,
                taxon: Taxon::Cetacean,
                rank: 2,
            },
        ];

        // Verify candidates can be sorted by confidence
        let mut sorted = candidates.clone();
        sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        assert_eq!(sorted[0].label, "SpeciesA");
        assert_eq!(sorted[0].confidence, 0.9);
        assert_eq!(sorted[1].label, "SpeciesB");
        assert_eq!(sorted[2].label, "SpeciesC");
    }

    // =========================================================================
    // TDD Test Suite: Stage1Result with Unknown Taxon
    // =========================================================================

    #[test]
    fn test_stage1_result_with_unknown_taxon() {
        let result = Stage1Result {
            predicted_group: ConsolidatedTaxon::Unknown,
            confidence: 0.15,
            rf_prediction: ConsolidatedTaxon::Unknown,
            rf_confidence: 0.2,
            rf_proba: vec![0.2, 0.15, 0.15, 0.15, 0.15, 0.2],
            nn_prediction: None,
            nn_confidence: None,
            nn_proba: None,
            rf_weight: 1.0,
            nn_weight: 0.0,
        };

        assert_eq!(result.predicted_group, ConsolidatedTaxon::Unknown);
        assert!(result.confidence < 0.3);
        assert!(result.nn_prediction.is_none());
        assert!(result.nn_confidence.is_none());
        assert!(result.nn_proba.is_none());
    }
}
