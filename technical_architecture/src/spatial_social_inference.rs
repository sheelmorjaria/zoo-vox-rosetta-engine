//! Spatial-Social Inference Module (Level 3)
//! ==========================================
//!
//! Combines spatial audio localization with social cognition for
//! multi-agent interaction. Infers speaker identity, social hierarchy,
//! and interaction dynamics from vocalization patterns.
//!
//! Features:
//! - Sound source localization (VBAP-based)
//! - Speaker diarization (individual identification)
//! - Social hierarchy inference (dominance relationships)
//! - Turn-taking detection
//! - Proximity-based social mapping
//!
//! Author: Zoo Vox Research Team
//! License: CC BY-ND 4.0 International

use crate::spatial_audio::{Position3D, SpatialAudioRenderer};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::f32::consts::PI;

// ═══════════════════════════════════════════════════════════════════════════════
// CORE DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/// Identified individual in the colony
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    /// Unique identifier (e.g., RFID tag, visual ID)
    pub id: String,
    /// Name/label for the individual
    pub name: String,
    /// Last known position
    pub last_position: Option<Position3D>,
    /// Last position update timestamp
    pub last_position_time: Option<f64>,
    /// Dominance rank (0 = lowest, 1 = highest)
    pub dominance_rank: f32,
    /// Confidence in dominance estimate
    pub dominance_confidence: f32,
    /// Typical vocalization characteristics
    pub vocal_profile: VocalProfile,
    /// Social relationship strengths (individual_id -> strength 0-1)
    pub social_bonds: HashMap<String, f32>,
}

impl Individual {
    pub fn new(id: String, name: String) -> Self {
        Self {
            id,
            name,
            last_position: None,
            last_position_time: None,
            dominance_rank: 0.5, // Default to middle rank
            dominance_confidence: 0.0,
            vocal_profile: VocalProfile::default(),
            social_bonds: HashMap::new(),
        }
    }

    /// Update position with timestamp
    pub fn update_position(&mut self, position: Position3D, timestamp: f64) {
        self.last_position = Some(position);
        self.last_position_time = Some(timestamp);
    }

    /// Calculate age of position estimate in seconds
    pub fn position_age(&self, current_time: f64) -> Option<f64> {
        self.last_position_time.map(|t| current_time - t)
    }

    /// Check if position estimate is stale (> 5 seconds old)
    pub fn is_position_stale(&self, current_time: f64) -> bool {
        match self.position_age(current_time) {
            Some(age) => age > 5.0,
            None => true,
        }
    }
}

/// Vocal profile for individual identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocalProfile {
    /// Typical F0 range (min, max) in Hz
    pub f0_range: (f32, f32),
    /// Typical spectral centroid range
    pub spectral_centroid_range: (f32, f32),
    /// Typical call duration range in seconds
    pub duration_range: (f32, f32),
    /// Unique vocal characteristics
    pub characteristics: HashMap<String, f32>,
}

impl Default for VocalProfile {
    fn default() -> Self {
        Self {
            f0_range: (1000.0, 15000.0),
            spectral_centroid_range: (2000.0, 10000.0),
            duration_range: (0.05, 0.5),
            characteristics: HashMap::new(),
        }
    }
}

/// Detected vocalization event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocalizationEvent {
    /// Event timestamp
    pub timestamp: f64,
    /// Estimated source position
    pub source_position: Position3D,
    /// Confidence in position estimate
    pub position_confidence: f32,
    /// Detected speaker ID (if diarization successful)
    pub speaker_id: Option<String>,
    /// Confidence in speaker identification
    pub speaker_confidence: f32,
    /// Acoustic features
    pub acoustic_features: AcousticFeatures,
    /// Call type classification
    pub call_type: Option<String>,
}

/// Acoustic features for diarization and classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticFeatures {
    /// Fundamental frequency (Hz)
    pub f0: f32,
    /// F0 variance (jitter measure)
    pub f0_variance: f32,
    /// Spectral centroid (Hz)
    pub spectral_centroid: f32,
    /// Spectral bandwidth (Hz)
    pub spectral_bandwidth: f32,
    /// RMS amplitude
    pub rms: f32,
    /// Duration (seconds)
    pub duration: f32,
    /// MFCC-like features
    pub mfcc: Vec<f32>,
}

/// Social hierarchy relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialRelationship {
    /// Higher-ranking individual
    pub dominant_id: String,
    /// Lower-ranking individual
    pub subordinate_id: String,
    /// Relationship strength (0-1)
    pub strength: f32,
    /// Confidence in relationship
    pub confidence: f32,
    /// Type of relationship
    pub relationship_type: RelationshipType,
}

/// Type of social relationship
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Clear dominance hierarchy
    Dominance,
    /// Mutual affiliation
    Affiliation,
    /// Agonistic (conflict) relationship
    Agonistic,
    /// Unknown/uncertain
    Unknown,
}

/// Turn-taking state for interaction tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnTakingState {
    /// Current speaker
    pub current_speaker: Option<String>,
    /// Turn start time
    pub turn_start_time: Option<f64>,
    /// Recent turn history (oldest first)
    pub turn_history: Vec<Turn>,
    /// Overlap detection count
    pub overlap_count: usize,
}

/// Single turn in conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub speaker_id: String,
    pub start_time: f64,
    pub end_time: f32,
    pub was_interrupted: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// SPATIAL-SOCIAL INFERENCE ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

/// Main spatial-social inference engine
pub struct SpatialSocialInference {
    /// Known individuals in the colony
    individuals: HashMap<String, Individual>,
    /// Current positions of all individuals
    positions: HashMap<String, Position3D>,
    /// Turn-taking state
    turn_taking: TurnTakingState,
    /// Inferred social relationships
    relationships: Vec<SocialRelationship>,
    /// Current timestamp
    current_time: f64,
    /// Position uncertainty threshold (meters)
    position_uncertainty_threshold: f32,
    /// Speaker diarization confidence threshold
    diarization_threshold: f32,
}

impl SpatialSocialInference {
    pub fn new() -> Self {
        Self {
            individuals: HashMap::new(),
            positions: HashMap::new(),
            turn_taking: TurnTakingState {
                current_speaker: None,
                turn_start_time: None,
                turn_history: Vec::new(),
                overlap_count: 0,
            },
            relationships: Vec::new(),
            current_time: 0.0,
            position_uncertainty_threshold: 0.5,
            diarization_threshold: 0.7,
        }
    }

    /// Register a new individual
    pub fn register_individual(&mut self, individual: Individual) {
        let id = individual.id.clone();
        self.individuals.insert(id.clone(), individual);
        log::info!("Registered individual: {}", id);
    }

    /// Update time and prune stale position data
    pub fn update_time(&mut self, timestamp: f64) {
        self.current_time = timestamp;

        // Mark stale positions
        for individual in self.individuals.values_mut() {
            if individual.is_position_stale(timestamp) {
                individual.last_position = None;
                individual.last_position_time = None;
            }
        }
    }

    /// Process a vocalization event and infer social context
    pub fn process_vocalization(&mut self, event: VocalizationEvent) -> InferenceResult {
        // Update position estimate
        self.update_position_from_vocalization(&event);

        // Diarize (identify speaker)
        let speaker_id = self.diarize(&event);

        // Update turn-taking state
        self.update_turn_taking(&speaker_id, event.timestamp);

        // Infer social dynamics
        let social_context = self.infer_social_context(&speaker_id, &event);

        InferenceResult {
            speaker_id: speaker_id.clone(),
            position_confidence: event.position_confidence,
            speaker_confidence: event.speaker_confidence,
            social_context,
            suggested_response: self.compute_suggested_response(&speaker_id, &social_context),
        }
    }

    /// Update position estimate from vocalization
    fn update_position_from_vocalization(&mut self, event: &VocalizationEvent) {
        if let Some(ref speaker_id) = event.speaker_id {
            if event.position_confidence > self.diarization_threshold {
                self.positions.insert(speaker_id.clone(), event.source_position);

                if let Some(individual) = self.individuals.get_mut(speaker_id) {
                    individual.update_position(event.source_position, event.timestamp);
                }
            }
        }
    }

    /// Speaker diarization - identify who is vocalizing
    fn diarize(&self, event: &VocalizationEvent) -> Option<String> {
        // If already has speaker ID with high confidence, use it
        if let Some(ref speaker_id) = event.speaker_id {
            if event.speaker_confidence > self.diarization_threshold {
                return Some(speaker_id.clone());
            }
        }

        // Otherwise, match based on position and vocal profile
        let mut best_match: Option<(String, f32)> = None;

        for (id, individual) in &self.individuals {
            if let Some(last_pos) = individual.last_position {
                let distance = last_pos.distance_to(&event.source_position);

                // Check if position is consistent
                if distance < self.position_uncertainty_threshold {
                    // Check vocal profile match
                    let profile_match = self.vocal_profile_match(
                        &event.acoustic_features,
                        &individual.vocal_profile,
                    );

                    let combined_score = profile_match * (1.0 - distance / self.position_uncertainty_threshold);

                    match &best_match {
                        Some((_, current_score)) if combined_score > *current_score => {
                            best_match = Some((id.clone(), combined_score));
                        }
                        None => {
                            best_match = Some((id.clone(), combined_score));
                        }
                        _ => {}
                    }
                }
            }
        }

        best_match.map(|(id, _)| id)
    }

    /// Calculate vocal profile match score
    fn vocal_profile_match(&self, features: &AcousticFeatures, profile: &VocalProfile) -> f32 {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // F0 range check
        let f0_score = if features.f0 >= profile.f0_range.0 && features.f0 <= profile.f0_range.1 {
            1.0
        } else {
            let dist = (features.f0 - profile.f0_range.0).min(features.f0 - profile.f0_range.1);
            (1.0 - dist / 5000.0).max(0.0)
        };
        score += f0_score * 2.0;
        weight_sum += 2.0;

        // Duration check
        let duration_score = if features.duration >= profile.duration_range.0
            && features.duration <= profile.duration_range.1
        {
            1.0
        } else {
            0.5
        };
        score += duration_score * 1.0;
        weight_sum += 1.0;

        if weight_sum > 0.0 {
            score / weight_sum
        } else {
            0.5
        }
    }

    /// Update turn-taking state
    fn update_turn_taking(&mut self, speaker_id: &Option<String>, timestamp: f64) {
        match (&self.turn_taking.current_speaker, speaker_id) {
            (None, Some(id)) => {
                // Start of new turn
                self.turn_taking.current_speaker = Some(id.clone());
                self.turn_taking.turn_start_time = Some(timestamp);
            }
            (Some(current), Some(id)) if current == id => {
                // Continuing turn - do nothing
            }
            (Some(current), Some(new_id)) => {
                // Turn change
                if let Some(start_time) = self.turn_taking.turn_start_time {
                    let turn = Turn {
                        speaker_id: current.clone(),
                        start_time,
                        end_time: (timestamp - start_time) as f32,
                        was_interrupted: false, // TODO: detect overlaps
                    };
                    self.turn_taking.turn_history.push(turn);

                    // Keep only recent history
                    if self.turn_taking.turn_history.len() > 100 {
                        self.turn_taking.turn_history.remove(0);
                    }
                }

                self.turn_taking.current_speaker = Some(new_id.clone());
                self.turn_taking.turn_start_time = Some(timestamp);
            }
            _ => {}
        }
    }

    /// Infer social context from current state
    fn infer_social_context(&self, speaker_id: &Option<String>, _event: &VocalizationEvent) -> SocialContext {
        let speaker_dominance = speaker_id
            .as_ref()
            .and_then(|id| self.individuals.get(id))
            .map(|ind| ind.dominance_rank)
            .unwrap_or(0.5);

        // Calculate proximity to other individuals
        let nearby_individuals = self.get_nearby_individuals(speaker_id, 2.0);

        // Detect turn-taking patterns
        let turn_regular = self.analyze_turn_regularity();

        SocialContext {
            speaker_dominance,
            nearby_count: nearby_individuals.len(),
            is_group_context: nearby_individuals.len() > 2,
            turn_taking_regularity: turn_regular,
            interaction_phase: self.infer_interaction_phase(),
        }
    }

    /// Get individuals within radius meters
    fn get_nearby_individuals(&self, speaker_id: &Option<String>, radius: f32) -> Vec<String> {
        let speaker_pos = speaker_id
            .as_ref()
            .and_then(|id| self.positions.get(id))
            .cloned();

        match speaker_pos {
            Some(pos) => self
                .positions
                .iter()
                .filter(|(id, _)| speaker_id.as_ref().map_or(true, |sid| sid != *id))
                .filter(|(_, other_pos)| pos.distance_to(other_pos) < radius)
                .map(|(id, _)| id.clone())
                .collect(),
            None => Vec::new(),
        }
    }

    /// Analyze turn-taking regularity
    fn analyze_turn_regularity(&self) -> f32 {
        if self.turn_taking.turn_history.len() < 5 {
            return 0.0; // Not enough data
        }

        let recent: Vec<_> = self.turn_taking.turn_history.iter().rev().take(10).collect();

        // Check for alternating pattern
        let mut alternations = 0;
        for window in recent.windows(2) {
            if window[0].speaker_id != window[1].speaker_id {
                alternations += 1;
            }
        }

        (alternations as f32) / (recent.len() - 1) as f32
    }

    /// Infer current interaction phase
    fn infer_interaction_phase(&self) -> InteractionPhase {
        if self.turn_taking.turn_history.len() < 3 {
            return InteractionPhase::Initiation;
        }

        let recent: Vec<_> = self.turn_taking.turn_history.iter().rev().take(5).collect();
        let unique_speakers: HashSet<_> = recent.iter().map(|t| &t.speaker_id).collect();

        match unique_speakers.len() {
            1 => InteractionPhase::Solo,
            2 => InteractionPhase::Dyadic,
            _ => InteractionPhase::Group,
        }
    }

    /// Compute suggested response based on social context
    fn compute_suggested_response(&self, speaker_id: &Option<String>, context: &SocialContext) -> SuggestedResponse {
        let speaker_rank = speaker_id
            .as_ref()
            .and_then(|id| self.individuals.get(id))
            .map(|ind| ind.dominance_rank)
            .unwrap_or(0.5);

        // Response strategy depends on social context
        let response_type = match context.interaction_phase {
            InteractionPhase::Initiation => ResponseType::Acknowledge,
            InteractionPhase::Solo => ResponseType::Observe,
            InteractionPhase::Dyadic => {
                if speaker_rank > 0.7 {
                    ResponseType::Defer
                } else {
                    ResponseType::Match
                }
            }
            InteractionPhase::Group => ResponseType::Observe,
        };

        // Suggested delay based on dominance
        let delay_ms = if speaker_rank > 0.7 {
            200.0 // Defer to high-ranking individual
        } else if speaker_rank < 0.3 {
            100.0 // Respond quickly to low-ranking
        } else {
            150.0 // Standard response time
        };

        SuggestedResponse {
            response_type,
            delay_ms,
            target_speaker: speaker_id.clone(),
            spatial_mode: context.is_group_context,
        }
    }

    /// Learn social relationship from interaction pattern
    pub fn learn_relationship(&mut self, id1: &str, id2: &str, outcome: InteractionOutcome) {
        // Update relationship strengths based on interaction
        let relationship = SocialRelationship {
            dominant_id: if outcome == InteractionOutcome::Id1Dominant {
                id1.to_string()
            } else {
                id2.to_string()
            },
            subordinate_id: if outcome == InteractionOutcome::Id1Dominant {
                id2.to_string()
            } else {
                id1.to_string()
            },
            strength: 0.5,
            confidence: 0.3,
            relationship_type: RelationshipType::Dominance,
        };

        // Update or add relationship
        if let Some(existing) = self.relationships.iter_mut().find(|r| {
            (r.dominant_id == relationship.dominant_id && r.subordinate_id == relationship.subordinate_id)
                || (r.dominant_id == relationship.subordinate_id && r.subordinate_id == relationship.dominant_id)
        }) {
            // Strengthen existing relationship
            existing.confidence = (existing.confidence + 0.1).min(1.0);
            existing.strength = (existing.strength + 0.1).min(1.0);
        } else {
            self.relationships.push(relationship);
        }
    }

    /// Get dominance rank for individual
    pub fn get_dominance_rank(&self, id: &str) -> Option<f32> {
        self.individuals.get(id).map(|ind| ind.dominance_rank)
    }

    /// Set dominance rank for individual
    pub fn set_dominance_rank(&mut self, id: &str, rank: f32) {
        if let Some(individual) = self.individuals.get_mut(id) {
            individual.dominance_rank = rank.clamp(0.0, 1.0);
            individual.dominance_confidence = 1.0;
        }
    }

    /// Get current positions of all individuals
    pub fn get_positions(&self) -> &HashMap<String, Position3D> {
        &self.positions
    }

    /// Get turn-taking state
    pub fn get_turn_taking(&self) -> &TurnTakingState {
        &self.turn_taking
    }
}

impl Default for SpatialSocialInference {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESULT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of spatial-social inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Identified speaker
    pub speaker_id: Option<String>,
    /// Position confidence
    pub position_confidence: f32,
    /// Speaker confidence
    pub speaker_confidence: f32,
    /// Inferred social context
    pub social_context: SocialContext,
    /// Suggested response
    pub suggested_response: SuggestedResponse,
}

/// Social context of vocalization
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SocialContext {
    /// Dominance rank of speaker
    pub speaker_dominance: f32,
    /// Number of nearby individuals
    pub nearby_count: usize,
    /// Is this a group context?
    pub is_group_context: bool,
    /// Turn-taking regularity (0-1)
    pub turn_taking_regularity: f32,
    /// Current interaction phase
    pub interaction_phase: InteractionPhase,
}

/// Interaction phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionPhase {
    /// Interaction initiation
    Initiation,
    /// Single individual vocalizing
    Solo,
    /// Two-way interaction
    Dyadic,
    /// Multi-individual group interaction
    Group,
}

/// Suggested response strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedResponse {
    /// Type of response to make
    pub response_type: ResponseType,
    /// Delay before response (ms)
    pub delay_ms: f32,
    /// Target speaker (if responding to specific individual)
    pub target_speaker: Option<String>,
    /// Use spatial rendering?
    pub spatial_mode: bool,
}

/// Response type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseType {
    /// No response (observe)
    Observe,
    /// Acknowledge presence
    Acknowledge,
    /// Match caller's vocalization
    Match,
    /// Defer to higher-ranking individual
    Defer,
    /// Redirect to different target
    Redirect,
}

/// Outcome of social interaction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionOutcome {
    Id1Dominant,
    Id2Dominant,
    Mutual,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_individual_new() {
        let ind = Individual::new("test_id".to_string(), "Test".to_string());
        assert_eq!(ind.id, "test_id");
        assert_eq!(ind.dominance_rank, 0.5);
        assert!(ind.last_position.is_none());
    }

    #[test]
    fn test_position_update() {
        let mut ind = Individual::new("test_id".to_string(), "Test".to_string());
        ind.update_position(Position3D::new(1.0, 2.0, 3.0), 100.0);

        assert!(ind.last_position.is_some());
        assert_eq!(ind.position_age(105.0), Some(5.0));
        assert!(!ind.is_position_stale(103.0));
        assert!(ind.is_position_stale(110.0));
    }

    #[test]
    fn test_register_individual() {
        let mut inference = SpatialSocialInference::new();
        let ind = Individual::new("test".to_string(), "Test".to_string());
        inference.register_individual(ind);

        assert!(inference.individuals.contains_key("test"));
    }

    #[test]
    fn test_dominance_rank() {
        let mut inference = SpatialSocialInference::new();
        let ind = Individual::new("test".to_string(), "Test".to_string());
        inference.register_individual(ind);

        inference.set_dominance_rank("test", 0.8);
        assert_eq!(inference.get_dominance_rank("test"), Some(0.8));
    }

    #[test]
    fn test_turn_regularity_analysis() {
        let mut inference = SpatialSocialInference::new();

        // Create alternating turns
        for i in 0..10 {
            let speaker_id = if i % 2 == 0 { "A" } else { "B" }.to_string();
            let turn = Turn {
                speaker_id,
                start_time: i as f64,
                end_time: 1.0,
                was_interrupted: false,
            };
            inference.turn_taking.turn_history.push(turn);
        }

        let regularity = inference.analyze_turn_regularity();
        assert!(regularity > 0.8); // High alternation
    }

    #[test]
    fn test_interaction_phase_inference() {
        let mut inference = SpatialSocialInference::new();

        // Solo phase
        assert_eq!(
            inference.infer_interaction_phase(),
            InteractionPhase::Initiation
        );

        // Add solo turns
        for _ in 0..3 {
            inference.turn_taking.turn_history.push(Turn {
                speaker_id: "A".to_string(),
                start_time: 0.0,
                end_time: 1.0,
                was_interrupted: false,
            });
        }

        assert_eq!(
            inference.infer_interaction_phase(),
            InteractionPhase::Solo
        );

        // Add dyadic turns
        inference.turn_taking.turn_history.push(Turn {
            speaker_id: "B".to_string(),
            start_time: 1.0,
            end_time: 1.0,
            was_interrupted: false,
        });

        assert_eq!(
            inference.infer_interaction_phase(),
            InteractionPhase::Dyadic
        );
    }
}
