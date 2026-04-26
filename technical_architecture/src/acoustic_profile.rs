//! Acoustic Profile Module - Strategy Pattern for Acoustic Processing
//! ================================================================
//!
//! This module implements the Strategy Pattern for species-specific acoustic
//! processing in the Rust Execution Layer.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    AcousticProfile Trait                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ + position_weights() -> &PositionWeights                       │
//! │ + transition_strictness() -> f32                               │
//! │ + is_valid_transition(from, to) -> bool                        │
//! │ + name() -> &str                                                │
//! └─────────────────────────────────────────────────────────────────┘
//!                              ▲
//!              ┌───────────────┴───────────────┐
//!              │                               │
//!  ┌─────────────────────┐       ┌─────────────────────┐
//!  │   GeneralProfile    │       │     BatProfile      │
//!  │   (default)         │       │  (bat-specific)     │
//!  ├─────────────────────┤       ├─────────────────────┤
//!  │ strictness: 0.0     │       │ strictness: 0.98    │
//!  │ all transitions OK  │       │ only 50 bigrams OK  │
//!  │ uniform weights     │       │ position-weighted   │
//!  └─────────────────────┘       └─────────────────────┘
//! ```
//!
//! # Background Research (Egyptian Fruit Bat Phase 2/3)
//!
//! From Phase 2:
//! - Only 0.02% of possible bigrams are used (extremely restrictive grammar)
//! - 50 valid bigrams out of 260,100 possible
//! - No function words detected (all segments have <5 unique transitions)
//!
//! From Phase 3:
//! - Openers: Shorter duration (~31.6ms), higher energy, lower HNR
//! - Closers: Longer duration (~58.0ms), lower energy, higher HNR
//! - Position determines role, not acoustics
//!
//! # Usage
//!
//! ```rust
//! use technical_architecture::{AcousticProfile, GeneralProfile, BatProfile};
//!
//! // Create general profile (default)
//! let general = GeneralProfile::default();
//! assert!(general.is_valid_transition(999, 888)); // All transitions valid
//!
//! // Create bat profile
//! let bat = BatProfile::default();
//! assert!(!bat.is_valid_transition(999, 888)); // Only known bigrams valid
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::species::FeatureWeights;

// ============================================================================
// Export Types (for IPC to Python Logic Layer)
// ============================================================================

/// Serializable rigid idiom for IPC export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigidIdiomExport {
    /// Segment sequence that forms the idiom
    pub segments: Vec<usize>,
    /// Semantic label for the idiom
    pub meaning: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
}

/// Serializable acoustic profile for IPC export to Python
///
/// This struct contains all acoustic grammar data that the Python
/// Logic Layer needs, eliminating the need for Python to maintain
/// its own copy of this data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticProfileExport {
    /// Profile name (e.g., "bat", "general")
    pub profile_name: String,
    /// Transition strictness (0.0 = permissive, 1.0 = strict)
    pub transition_strictness: f32,
    /// Valid bigrams as (from, to) pairs
    pub valid_bigrams: Vec<(usize, usize)>,
    /// Segments that commonly appear at position 0 (openers)
    pub openers: Vec<usize>,
    /// Segments that commonly appear at position 1 (closers)
    pub closers: Vec<usize>,
    /// Rigid idioms (unbreakable patterns)
    pub rigid_idioms: Vec<RigidIdiomExport>,
    /// Position weights serialized as JSON-friendly struct
    pub position_weights: PositionWeights,
}

// ============================================================================
// Position Weights
// ============================================================================

/// Position-specific feature weights for acoustic analysis
///
/// Different positions in a vocalization sequence may require different
/// feature weighting schemes for optimal classification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PositionWeights {
    /// Weights for opener position (position 0)
    /// For bats: emphasize temporal and micro_dynamics (staccato alerts)
    pub opener: FeatureWeights,

    /// Weights for closer position (position 1)
    /// For bats: emphasize harmonic and spectral (clean termination)
    pub closer: FeatureWeights,

    /// Weights for content positions (position 2+)
    pub content: FeatureWeights,
}

// ============================================================================
// Acoustic Profile Trait
// ============================================================================

/// Acoustic profile for species-specific processing
///
/// This trait defines the interface for species-specific acoustic processing
/// strategies. Implementations provide position-weighted feature logic and
/// transition validation.
pub trait AcousticProfile: Send + Sync {
    /// Get position-specific feature weights
    fn position_weights(&self) -> &PositionWeights;

    /// Get transition strictness (0.0 = permissive, 1.0 = strict)
    ///
    /// - 0.0: All transitions are valid (general mode)
    /// - 0.98: Only known bigrams are valid (bat mode, 0.02% of possible)
    fn transition_strictness(&self) -> f32;

    /// Check if a transition from one segment to another is valid
    ///
    /// In general mode, all transitions are valid.
    /// In bat mode, only the 50 observed bigrams from Phase 2 research are valid.
    fn is_valid_transition(&self, from: usize, to: usize) -> bool;

    /// Get profile name
    fn name(&self) -> &str;

    /// Get weights for a specific position
    ///
    /// Returns position-appropriate weights based on the index
    fn weights_for_position(&self, position: usize) -> &FeatureWeights {
        match position {
            0 => &self.position_weights().opener,
            1 => &self.position_weights().closer,
            _ => &self.position_weights().content,
        }
    }

    /// Get all valid transitions from a given segment
    ///
    /// Returns a list of segments that can follow the given segment
    fn get_valid_successors(&self, _from: usize) -> Vec<usize> {
        Vec::new() // Default: no restrictions
    }

    /// Export profile data for IPC to Python Logic Layer
    fn to_export(&self) -> AcousticProfileExport;
}

// ============================================================================
// General Profile (Default)
// ============================================================================

/// General-purpose profile (original behavior)
///
/// This profile preserves the original behavior:
/// - All transitions are valid
/// - Uniform feature weights across positions
/// - Permissive routing logic
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeneralProfile {
    weights: PositionWeights,
}

impl GeneralProfile {
    /// Create a new general profile with custom weights
    pub fn new(weights: PositionWeights) -> Self {
        Self { weights }
    }
}

impl AcousticProfile for GeneralProfile {
    fn position_weights(&self) -> &PositionWeights {
        &self.weights
    }

    fn transition_strictness(&self) -> f32 {
        0.0 // Permissive: all transitions valid
    }

    fn is_valid_transition(&self, _from: usize, _to: usize) -> bool {
        true // All transitions are valid in general mode
    }

    fn name(&self) -> &str {
        "general"
    }

    fn get_valid_successors(&self, _from: usize) -> Vec<usize> {
        Vec::new() // No restrictions in general mode
    }

    fn to_export(&self) -> AcousticProfileExport {
        AcousticProfileExport {
            profile_name: "general".to_string(),
            transition_strictness: 0.0,
            valid_bigrams: Vec::new(),
            openers: Vec::new(),
            closers: Vec::new(),
            rigid_idioms: Vec::new(),
            position_weights: self.weights.clone(),
        }
    }
}

// ============================================================================
// Bat Profile (Bat-Specific)
// ============================================================================

/// Bat-specific profile with position-weighted logic
///
/// This profile implements the findings from Phase 2/3 research:
///
/// **Transition Constraints (Phase 2):**
/// - Only 50 valid bigrams out of 260,100 possible (0.02%)
/// - Extremely restrictive grammar
/// - No function words detected
///
/// **Position-Weighted Features (Phase 3):**
/// - Openers: Shorter, higher energy, lower HNR (staccato alerts)
/// - Closers: Longer, lower energy, higher HNR (clean termination)
/// - Content: Standard bat feature weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatProfile {
    weights: PositionWeights,
    valid_bigrams: HashSet<(usize, usize)>,
    bigram_list: Vec<(usize, usize)>,
    /// Segments that commonly appear at position 0 (openers)
    openers: Vec<usize>,
    /// Segments that commonly appear at position 1 (closers)
    closers: Vec<usize>,
    /// Rigid idioms: unbreakable patterns (LRN-6, etc.)
    rigid_idioms: Vec<RigidIdiomExport>,
}

impl Default for BatProfile {
    fn default() -> Self {
        // Top 50 valid bigrams from Phase 2 research
        // These represent only 0.02% of possible bigrams
        let bigram_list: Vec<(usize, usize)> = vec![
            (764, 304),
            (534, 434),
            (304, 394),
            (514, 504),
            (384, 464),
            (574, 324),
            (444, 544),
            (1014, 684),
            (384, 44),
            (154, 204),
            (264, 44),
            (764, 464),
            (514, 304),
            (574, 684),
            (434, 504),
            (304, 404),
            (394, 404),
            (544, 504),
            (684, 504),
            (324, 394),
            // LRN-6 internal transitions (the rigid idiom)
            (114, 464),
            (464, 604),
            (604, 324),
            (324, 94),
            (94, 714),
            // Additional valid transitions
            (384, 514),
            (534, 304),
            (264, 384),
            (444, 304),
            (304, 324),
            (394, 304),
            (514, 574),
            (574, 444),
            (1014, 444),
            (154, 264),
            (764, 1014),
            (534, 514),
            (434, 514),
            (684, 324),
            (504, 324),
            (324, 504),
            (94, 464),
            (714, 114),
            (464, 94),
            (604, 94),
            (114, 604),
            (384, 574),
            (574, 514),
            (444, 394),
        ];

        let valid_bigrams: HashSet<(usize, usize)> = bigram_list.iter().copied().collect();

        // Compute openers: segments that appear as 'from' in bigrams but rarely as 'to'
        // These are segments that initiate vocalization sequences
        let from_segments: HashSet<usize> = bigram_list.iter().map(|(f, _)| *f).collect();
        let to_segments: HashSet<usize> = bigram_list.iter().map(|(_, t)| *t).collect();

        // Phase 3 research: openers appear >70% at position 0
        // These are segments that commonly start sequences (in 'from' but rarely 'to')
        let openers: Vec<usize> = from_segments
            .iter()
            .filter(|s| !to_segments.contains(s))
            .copied()
            .collect();

        // Phase 3 research: closers appear >70% at position 1
        // These are segments that commonly end sequences (in 'to' but rarely 'from')
        let closers: Vec<usize> = to_segments
            .iter()
            .filter(|s| !from_segments.contains(s))
            .copied()
            .collect();

        // Rigid idioms from Phase 2 research
        let rigid_idioms = vec![RigidIdiomExport {
            segments: vec![114, 464, 604, 324, 94, 714],
            meaning: "LRN-6_IDIOM".to_string(),
            confidence: 0.98,
        }];

        Self {
            weights: PositionWeights {
                // Openers: Short duration, high energy, low HNR
                // Phase 3: Openers are "staccato alerts" (~31.6ms, high energy, noisy)
                opener: FeatureWeights {
                    temporal: 2.0,       // Duration critical for identifying short bursts
                    micro_dynamics: 2.5, // Energy/attack critical for alert detection
                    harmonic: 0.8,       // HNR less important (openers are noisy)
                    modulation: 1.5,     // FM patterns may indicate urgency
                    spectral: 1.2,       // Some spectral content
                    cepstral: 0.8,
                    formant: 0.5, // Not relevant for ultrasonic
                    psychoacoustic: 1.0,
                    tfs: 1.2,
                    overrides: vec![
                        (10, 2.0), // D10: rms - energy critical
                        (12, 2.0), // D12: attack - onset critical
                        (13, 1.8), // D13: decay - termination
                        (30, 2.0), // D30: onset_rate - burst detection
                    ],
                },

                // Closers: Long duration, low energy, high HNR
                // Phase 3: Closers are "clean termination" (~58.0ms, low energy, harmonic)
                closer: FeatureWeights {
                    temporal: 1.5,       // Duration important but less critical
                    micro_dynamics: 1.0, // Energy less important
                    harmonic: 2.5,       // HNR CRITICAL for clean tones
                    modulation: 1.2,     // Some FM
                    spectral: 1.8,       // Higher frequency important
                    cepstral: 1.0,
                    formant: 0.5,
                    psychoacoustic: 1.2, // Pitch perception
                    tfs: 1.5,
                    overrides: vec![
                        (6, 2.5),  // D6: harmonicity - CRITICAL
                        (7, 2.0),  // D7: harmonic_to_noise - CRITICAL
                        (0, 1.8),  // D0: spectral_centroid - frequency tracking
                        (14, 1.5), // D14: sustain - duration
                    ],
                },

                // Content: Standard bat weights from species.rs
                content: FeatureWeights::bat(),
            },
            valid_bigrams,
            bigram_list,
            openers,
            closers,
            rigid_idioms,
        }
    }
}

impl BatProfile {
    /// Create a new bat profile with custom weights and bigrams
    pub fn new(weights: PositionWeights, valid_bigrams: Vec<(usize, usize)>) -> Self {
        let bigram_list = valid_bigrams.clone();
        let valid_bigrams_set: HashSet<(usize, usize)> = valid_bigrams.into_iter().collect();

        // Compute openers/closers from bigrams
        let from_segments: HashSet<usize> = bigram_list.iter().map(|(f, _)| *f).collect();
        let to_segments: HashSet<usize> = bigram_list.iter().map(|(_, t)| *t).collect();
        let openers: Vec<usize> = from_segments
            .iter()
            .filter(|s| !to_segments.contains(s))
            .copied()
            .collect();
        let closers: Vec<usize> = to_segments
            .iter()
            .filter(|s| !from_segments.contains(s))
            .copied()
            .collect();

        Self {
            weights,
            valid_bigrams: valid_bigrams_set,
            bigram_list,
            openers,
            closers,
            rigid_idioms: Vec::new(),
        }
    }

    /// Add a valid bigram to the profile
    pub fn add_valid_bigram(&mut self, from: usize, to: usize) {
        self.valid_bigrams.insert((from, to));
        self.bigram_list.push((from, to));
    }

    /// Get the number of valid bigrams
    pub fn valid_bigram_count(&self) -> usize {
        self.valid_bigrams.len()
    }

    /// Calculate combinatorial ratio (percentage of possible bigrams that are valid)
    ///
    /// Phase 2 research found this to be ~0.02% for bats
    pub fn combinatorial_ratio(&self, vocabulary_size: usize) -> f64 {
        let possible_bigrams = vocabulary_size * vocabulary_size;
        if possible_bigrams == 0 {
            return 0.0;
        }
        (self.valid_bigrams.len() as f64) / (possible_bigrams as f64)
    }
}

impl AcousticProfile for BatProfile {
    fn position_weights(&self) -> &PositionWeights {
        &self.weights
    }

    fn transition_strictness(&self) -> f32 {
        0.98 // Restrictive: only ~0.02% of transitions valid
    }

    fn is_valid_transition(&self, from: usize, to: usize) -> bool {
        self.valid_bigrams.contains(&(from, to))
    }

    fn name(&self) -> &str {
        "bat"
    }

    fn get_valid_successors(&self, from: usize) -> Vec<usize> {
        self.bigram_list
            .iter()
            .filter(|(f, _)| *f == from)
            .map(|(_, t)| *t)
            .collect()
    }

    fn to_export(&self) -> AcousticProfileExport {
        AcousticProfileExport {
            profile_name: "bat".to_string(),
            transition_strictness: self.transition_strictness(),
            valid_bigrams: self.bigram_list.clone(),
            openers: self.openers.clone(),
            closers: self.closers.clone(),
            rigid_idioms: self.rigid_idioms.clone(),
            position_weights: self.weights.clone(),
        }
    }
}

// ============================================================================
// Profile Factory
// ============================================================================

/// Factory for creating acoustic profiles based on species
pub struct AcousticProfileFactory;

impl AcousticProfileFactory {
    /// Create an acoustic profile for a given species
    ///
    /// # Arguments
    /// * `species` - Species name (case-insensitive)
    ///
    /// # Returns
    /// * `Box<dyn AcousticProfile>` - Appropriate profile for the species
    pub fn create(species: &str) -> Box<dyn AcousticProfile> {
        match species.to_lowercase().as_str() {
            "egyptian fruit bat" | "egyptian_fruit_bat" | "fruit_bat" | "bat" => Box::new(BatProfile::default()),
            _ => Box::new(GeneralProfile::default()),
        }
    }

    /// Create an acoustic profile from domain mode string
    ///
    /// # Arguments
    /// * `domain_mode` - "general", "bat", or "holophrastic"
    ///
    /// # Returns
    /// * `Box<dyn AcousticProfile>` - Appropriate profile
    pub fn from_domain_mode(domain_mode: &str) -> Box<dyn AcousticProfile> {
        match domain_mode.to_lowercase().as_str() {
            "bat" | "holophrastic" => Box::new(BatProfile::default()),
            _ => Box::new(GeneralProfile::default()),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Sprint 2.1: General Profile Tests (Backwards Compatibility)
    // =========================================================================

    #[test]
    fn test_general_profile_execution() {
        // Test Case 2.1.1: General Profile Execution
        let profile = GeneralProfile::default();

        // All transitions should be valid
        assert!(profile.is_valid_transition(999, 888));
        assert!(profile.is_valid_transition(0, 1));
        assert!(profile.is_valid_transition(12345, 67890));

        // Strictness should be 0 (permissive)
        assert_eq!(profile.transition_strictness(), 0.0);

        // Name should be "general"
        assert_eq!(profile.name(), "general");
    }

    #[test]
    fn test_general_profile_uniform_weights() {
        let profile = GeneralProfile::default();
        let weights = profile.position_weights();

        // All positions should have default (uniform) weights
        assert_eq!(weights.opener.temporal, 1.0);
        assert_eq!(weights.closer.temporal, 1.0);
        assert_eq!(weights.content.temporal, 1.0);
    }

    #[test]
    fn test_general_profile_weights_for_position() {
        let profile = GeneralProfile::default();

        // Position 0 should return opener weights
        let w0 = profile.weights_for_position(0);
        assert_eq!(w0.temporal, 1.0);

        // Position 1 should return closer weights
        let w1 = profile.weights_for_position(1);
        assert_eq!(w1.temporal, 1.0);

        // Position 2+ should return content weights
        let w2 = profile.weights_for_position(2);
        assert_eq!(w2.temporal, 1.0);
    }

    // =========================================================================
    // Sprint 2.2: Bat Profile Tests (Position-Weighted Logic)
    // =========================================================================

    #[test]
    fn test_bat_profile_positional_weighting() {
        // Test Case 2.2.1: Bat Profile Positional Weighting
        let profile = BatProfile::default();
        let weights = profile.position_weights();

        // Openers should emphasize temporal and micro_dynamics
        // (Short duration, high energy - "staccato alerts")
        assert!(weights.opener.temporal > 1.5);
        assert!(weights.opener.micro_dynamics > 2.0);

        // Closers should emphasize harmonic (HNR)
        // (Long duration, clean tones - "termination signals")
        assert!(weights.closer.harmonic > 2.0);
        assert!(weights.closer.spectral > 1.5);

        // Content should use standard bat weights
        assert!(weights.content.micro_dynamics > 1.5); // From FeatureWeights::bat()
    }

    #[test]
    fn test_bat_profile_transition_strictness() {
        let profile = BatProfile::default();

        // Strictness should be very high (0.98)
        assert!(profile.transition_strictness() > 0.9);
        assert_eq!(profile.name(), "bat");
    }

    #[test]
    fn test_bat_profile_valid_bigrams() {
        let profile = BatProfile::default();

        // Known valid bigrams from Phase 2 research
        assert!(profile.is_valid_transition(764, 304));
        assert!(profile.is_valid_transition(534, 434));
        assert!(profile.is_valid_transition(114, 464)); // LRN-6 start

        // Should have a reasonable number of valid bigrams
        assert!(profile.valid_bigram_count() > 40);
        assert!(profile.valid_bigram_count() < 100); // ~50 from research
    }

    #[test]
    fn test_bat_profile_combinatorial_ratio() {
        let profile = BatProfile::default();

        // With 510 segments (from Phase 2), ratio should be ~0.02%
        let ratio = profile.combinatorial_ratio(510);
        assert!(ratio < 0.001); // Less than 0.1%
        assert!(ratio > 0.0001); // But more than 0.01%
    }

    // =========================================================================
    // Sprint 2.3: Transition Validation Tests
    // =========================================================================

    #[test]
    fn test_permissive_vs_restrictive_transitions() {
        // Test Case 2.3.1: Permissive vs Restrictive
        let general = GeneralProfile::default();
        let bat = BatProfile::default();

        // Illegal bigram [999, 888] - should be accepted by general, rejected by bat
        assert!(general.is_valid_transition(999, 888)); // Valid in general
        assert!(!bat.is_valid_transition(999, 888)); // Rejected in bat

        // Valid bigram [764, 304] - should be accepted by both
        assert!(general.is_valid_transition(764, 304));
        assert!(bat.is_valid_transition(764, 304));
    }

    #[test]
    fn test_bat_profile_get_valid_successors() {
        let profile = BatProfile::default();

        // Get successors for a known segment
        let successors = profile.get_valid_successors(114);

        // Should include 464 (LRN-6 continuation)
        assert!(successors.contains(&464));
    }

    #[test]
    fn test_bat_profile_no_successors_for_unknown() {
        let profile = BatProfile::default();

        // Unknown segment should have no valid successors
        let successors = profile.get_valid_successors(99999);
        assert!(successors.is_empty());
    }

    // =========================================================================
    // Factory Tests
    // =========================================================================

    #[test]
    fn test_factory_creates_general_by_default() {
        let profile = AcousticProfileFactory::create("unknown_species");
        assert_eq!(profile.name(), "general");
    }

    #[test]
    fn test_factory_creates_bat_for_bat_species() {
        let profile = AcousticProfileFactory::create("egyptian fruit bat");
        assert_eq!(profile.name(), "bat");

        let profile2 = AcousticProfileFactory::create("bat");
        assert_eq!(profile2.name(), "bat");
    }

    #[test]
    fn test_factory_from_domain_mode() {
        let general = AcousticProfileFactory::from_domain_mode("general");
        assert_eq!(general.name(), "general");

        let bat = AcousticProfileFactory::from_domain_mode("bat");
        assert_eq!(bat.name(), "bat");

        let holophrastic = AcousticProfileFactory::from_domain_mode("holophrastic");
        assert_eq!(holophrastic.name(), "bat");
    }

    // =========================================================================
    // Serialization Tests
    // =========================================================================

    #[test]
    fn test_position_weights_serialization() {
        let weights = PositionWeights::default();
        let json = serde_json::to_string(&weights).unwrap();
        let decoded: PositionWeights = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.opener.temporal, weights.opener.temporal);
    }

    #[test]
    fn test_bat_profile_serialization() {
        let profile = BatProfile::default();
        let json = serde_json::to_string(&profile).unwrap();
        let decoded: BatProfile = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.name(), "bat");
        assert_eq!(decoded.valid_bigram_count(), profile.valid_bigram_count());
    }

    // =========================================================================
    // Backwards Compatibility Tests
    // =========================================================================

    #[test]
    fn test_backwards_compatibility_general_is_default() {
        // Default factory should create general profile
        let profile = AcousticProfileFactory::create("marmoset");
        assert_eq!(profile.name(), "general");

        // General profile should accept all transitions (original behavior)
        assert!(profile.is_valid_transition(0, 1));
        assert!(profile.is_valid_transition(999, 888));
    }

    #[test]
    fn test_opener_weight_overrides() {
        let profile = BatProfile::default();
        let opener = &profile.position_weights().opener;

        // Check that overrides are applied correctly
        // D10 (rms) should have override weight 2.0
        let rms_weight = opener.get_weight(10);
        assert!(rms_weight >= 2.0);
    }

    #[test]
    fn test_closer_weight_overrides() {
        let profile = BatProfile::default();
        let closer = &profile.position_weights().closer;

        // Check that overrides are applied correctly
        // D6 (harmonicity) should have override weight 2.5
        let harmonicity_weight = closer.get_weight(6);
        assert!(harmonicity_weight >= 2.5);
    }

    // =========================================================================
    // Export Tests
    // =========================================================================

    #[test]
    fn test_bat_profile_export() {
        let profile = BatProfile::default();
        let export = profile.to_export();

        assert_eq!(export.profile_name, "bat");
        assert!((export.transition_strictness - 0.98).abs() < 0.01);
        assert_eq!(export.valid_bigrams.len(), profile.valid_bigram_count());
        assert!(!export.openers.is_empty());
        assert!(!export.closers.is_empty());
        assert!(!export.rigid_idioms.is_empty());
    }

    #[test]
    fn test_bat_profile_export_roundtrip() {
        let profile = BatProfile::default();
        let export = profile.to_export();

        let json = serde_json::to_string(&export).unwrap();
        let decoded: AcousticProfileExport = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded.profile_name, "bat");
        assert_eq!(decoded.valid_bigrams.len(), export.valid_bigrams.len());
    }

    #[test]
    fn test_general_profile_export() {
        let profile = GeneralProfile::default();
        let export = profile.to_export();

        assert_eq!(export.profile_name, "general");
        assert!(export.valid_bigrams.is_empty());
        assert!(export.openers.is_empty());
        assert!(export.closers.is_empty());
        assert!(export.rigid_idioms.is_empty());
    }

    #[test]
    fn test_bat_profile_export_contains_lrn6() {
        let profile = BatProfile::default();
        let export = profile.to_export();

        let lrn6 = export.rigid_idioms.iter().find(|idiom| idiom.meaning == "LRN-6_IDIOM");
        assert!(lrn6.is_some());
        let lrn6 = lrn6.unwrap();
        assert_eq!(lrn6.segments, vec![114, 464, 604, 324, 94, 714]);
        assert!(lrn6.confidence > 0.9);
    }
}
