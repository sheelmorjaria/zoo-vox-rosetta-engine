//! Bio-Acoustic Interaction Agent
//!
//! Bridges the RosettaPipeline (understanding) with Granular Synthesis (response).
//!
//! **Integration Points:**
//! 1. Acoustic Inventory - Semantic Dictionary with audio prototypes
//! 2. Context-to-Delta Mapping - Environmental adaptation
//! 3. Formant Barrier Validation - Physical synthesis constraints
//! 4. Response Strategy - Intent-based synthesis planning

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Constants
// =============================================================================

/// Maximum allowed spectral flatness change (Formant Barrier)
pub const MAX_SPECTRAL_FLATNESS_DELTA: f32 = 0.4;

/// Maximum allowed HNR change (Formant Barrier)
pub const MAX_HNR_DELTA: f32 = 15.0;

// =============================================================================
// Source Metadata (45D Feature Vector for Synthesis)
// =============================================================================

/// 45D Source Metadata for synthesis control
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SourceMetadata {
    // === Fundamental (5 features) ===
    pub mean_f0_hz: f32,
    pub duration_ms: f32,
    pub f0_range_hz: f32,
    pub f0_contour_slope: f32,
    pub pitch_stability: f32,

    // === Harmonic (5 features) ===
    pub harmonic_to_noise_ratio: f32,
    pub inharmonicity: f32,
    pub harmonic_1: f32,
    pub harmonic_2: f32,
    pub harmonic_3: f32,

    // === Temporal (5 features) ===
    pub attack_time_ms: f32,
    pub decay_time_ms: f32,
    pub sustain_level: f32,
    pub release_time_ms: f32,
    pub rms_energy: f32,

    // === Modulation (5 features) ===
    pub fm_rate_hz: f32,
    pub fm_depth_hz: f32,
    pub am_rate_hz: f32,
    pub am_depth: f32,
    pub tremolo_rate: f32,

    // === Cepstral (5 features) ===
    pub mfcc_1: f32,
    pub mfcc_2: f32,
    pub mfcc_3: f32,
    pub mfcc_4: f32,
    pub mfcc_5: f32,

    // === Formant (5 features) ===
    pub formant_1_hz: f32,
    pub formant_2_hz: f32,
    pub formant_3_hz: f32,
    pub bandwidth_1: f32,
    pub bandwidth_2: f32,

    // === Micro-Dynamics (5 features) ===
    pub jitter: f32,
    pub shimmer: f32,
    pub hnr_variation: f32,
    pub cpp: f32,
    pub entropy: f32,

    // === Psychoacoustic (5 features) ===
    pub loudness: f32,
    pub sharpness: f32,
    pub roughness: f32,
    pub tonality: f32,
    pub fluctuation_strength: f32,

    // === TFS (5 features) ===
    pub acf_peak: f32,
    pub acf_strength: f32,
    pub sfm: f32,
    pub periodicity: f32,
    pub tfs_entropy: f32,
}

impl SourceMetadata {
    /// Create from 45D feature vector
    pub fn from_vector(features: &[f32]) -> Self {
        let mut meta = Self::default();
        if features.len() >= 45 {
            meta.mean_f0_hz = features[0];
            meta.duration_ms = features[1];
            meta.f0_range_hz = features[2];
            meta.f0_contour_slope = features[3];
            meta.pitch_stability = features[4];
            meta.harmonic_to_noise_ratio = features[5];
            meta.inharmonicity = features[6];
            meta.harmonic_1 = features[7];
            meta.harmonic_2 = features[8];
            meta.harmonic_3 = features[9];
            meta.attack_time_ms = features[10];
            meta.decay_time_ms = features[11];
            meta.sustain_level = features[12];
            meta.release_time_ms = features[13];
            meta.rms_energy = features[14];
            meta.fm_rate_hz = features[15];
            meta.fm_depth_hz = features[16];
            meta.am_rate_hz = features[17];
            meta.am_depth = features[18];
            meta.tremolo_rate = features[19];
            meta.mfcc_1 = features[20];
            meta.mfcc_2 = features[21];
            meta.mfcc_3 = features[22];
            meta.mfcc_4 = features[23];
            meta.mfcc_5 = features[24];
            meta.formant_1_hz = features[25];
            meta.formant_2_hz = features[26];
            meta.formant_3_hz = features[27];
            meta.bandwidth_1 = features[28];
            meta.bandwidth_2 = features[29];
            meta.jitter = features[30];
            meta.shimmer = features[31];
            meta.hnr_variation = features[32];
            meta.cpp = features[33];
            meta.entropy = features[34];
            meta.loudness = features[35];
            meta.sharpness = features[36];
            meta.roughness = features[37];
            meta.tonality = features[38];
            meta.fluctuation_strength = features[39];
            meta.acf_peak = features[40];
            meta.acf_strength = features[41];
            meta.sfm = features[42];
            meta.periodicity = features[43];
            meta.tfs_entropy = features[44];
        }
        meta
    }

    /// Convert to vector
    pub fn to_vector(&self) -> Vec<f32> {
        vec![
            self.mean_f0_hz,
            self.duration_ms,
            self.f0_range_hz,
            self.f0_contour_slope,
            self.pitch_stability,
            self.harmonic_to_noise_ratio,
            self.inharmonicity,
            self.harmonic_1,
            self.harmonic_2,
            self.harmonic_3,
            self.attack_time_ms,
            self.decay_time_ms,
            self.sustain_level,
            self.release_time_ms,
            self.rms_energy,
            self.fm_rate_hz,
            self.fm_depth_hz,
            self.am_rate_hz,
            self.am_depth,
            self.tremolo_rate,
            self.mfcc_1,
            self.mfcc_2,
            self.mfcc_3,
            self.mfcc_4,
            self.mfcc_5,
            self.formant_1_hz,
            self.formant_2_hz,
            self.formant_3_hz,
            self.bandwidth_1,
            self.bandwidth_2,
            self.jitter,
            self.shimmer,
            self.hnr_variation,
            self.cpp,
            self.entropy,
            self.loudness,
            self.sharpness,
            self.roughness,
            self.tonality,
            self.fluctuation_strength,
            self.acf_peak,
            self.acf_strength,
            self.sfm,
            self.periodicity,
            self.tfs_entropy,
        ]
    }
}

// =============================================================================
// Acoustic Inventory (Upgraded Semantic Dictionary)
// =============================================================================

/// Acoustic prototype with audio buffer and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcousticPrototype {
    /// Semantic label (e.g., "Phee", "Tsik")
    pub label: String,

    /// Golden sample audio buffer (normalized f32)
    pub audio_buffer: Vec<f32>,

    /// Sample rate of the audio
    pub sample_rate: u32,

    /// 45D metadata for synthesis control
    pub metadata: SourceMetadata,

    /// Number of examples this prototype represents
    pub sample_count: usize,

    /// Modality classification
    pub modality: AcousticModality,
}

/// Acoustic modality for Formant Barrier checks
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AcousticModality {
    /// Tonal, harmonic (e.g., Phee, whistles)
    Harmonic,
    /// Short, transient (e.g., Tsik, clicks)
    Transient,
    /// Mixed (e.g., trills, warbles)
    Mixed,
}

impl AcousticModality {
    /// Determine modality from metadata
    pub fn from_metadata(meta: &SourceMetadata) -> Self {
        // High HNR + Low spectral flatness = Harmonic
        if meta.harmonic_to_noise_ratio > 15.0 && meta.entropy < 0.3 {
            Self::Harmonic
        }
        // Low HNR + High spectral flatness + Short duration = Transient
        else if meta.harmonic_to_noise_ratio < 10.0 && meta.entropy > 0.5 && meta.duration_ms < 100.0 {
            Self::Transient
        } else {
            Self::Mixed
        }
    }
}

/// Upgraded semantic dictionary with acoustic prototypes
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AcousticInventory {
    /// Species this inventory belongs to
    pub species: String,

    /// Mapping from semantic label to acoustic prototype
    pub prototypes: HashMap<String, AcousticPrototype>,

    /// Response strategies: input_label -> recommended_response_label
    pub response_strategies: HashMap<String, String>,

    /// Total audio samples in inventory
    pub total_samples: usize,
}

impl AcousticInventory {
    /// Create new empty inventory
    pub fn new(species: &str) -> Self {
        Self {
            species: species.to_string(),
            prototypes: HashMap::new(),
            response_strategies: HashMap::new(),
            total_samples: 0,
        }
    }

    /// Add a prototype to the inventory
    pub fn add_prototype(&mut self, prototype: AcousticPrototype) {
        self.total_samples += prototype.audio_buffer.len();
        self.prototypes.insert(prototype.label.clone(), prototype);
    }

    /// Get prototype by semantic label
    pub fn get_prototype(&self, label: &str) -> Option<&AcousticPrototype> {
        self.prototypes.get(label)
    }

    /// Get all available labels
    pub fn available_labels(&self) -> Vec<&String> {
        self.prototypes.keys().collect()
    }

    /// Set default response strategy
    pub fn set_response_strategy(&mut self, input_label: &str, response_label: &str) {
        self.response_strategies
            .insert(input_label.to_string(), response_label.to_string());
    }

    /// Get recommended response for an input
    pub fn get_response_label(&self, input_label: &str) -> Option<&String> {
        self.response_strategies.get(input_label)
    }

    /// Load from JSON file
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let inventory: Self = serde_json::from_reader(reader)?;
        Ok(inventory)
    }

    /// Save to JSON file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }
}

// =============================================================================
// Micro-Dynamics Delta (Synthesis Transformation)
// =============================================================================

/// Delta transformation for synthesis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MicroDynamicsDelta {
    // Fundamental deltas
    pub delta_mean_f0_hz: f32,
    pub delta_duration_ms: f32,
    pub delta_f0_range_hz: f32,

    // Harmonic deltas
    pub delta_harmonic_to_noise_ratio: f32,
    pub delta_inharmonicity: f32,

    // Temporal deltas
    pub delta_attack_time_ms: f32,
    pub delta_sustain_level: f32,

    // Modulation deltas
    pub delta_fm_depth_hz: f32,
    pub delta_am_depth: f32,

    // Micro-dynamics deltas (emotional intensity)
    pub delta_jitter: f32,
    pub delta_shimmer: f32,
    pub delta_entropy: f32,

    // Psychoacoustic deltas
    pub delta_loudness: f32,
    pub delta_sharpness: f32,
}

impl MicroDynamicsDelta {
    /// Create zero delta (no transformation)
    pub fn zero() -> Self {
        Self::default()
    }

    /// Apply delta to source metadata
    pub fn apply_to(&self, source: &SourceMetadata) -> SourceMetadata {
        let mut result = source.clone();
        result.mean_f0_hz += self.delta_mean_f0_hz;
        result.duration_ms += self.delta_duration_ms;
        result.f0_range_hz += self.delta_f0_range_hz;
        result.harmonic_to_noise_ratio += self.delta_harmonic_to_noise_ratio;
        result.inharmonicity += self.delta_inharmonicity;
        result.attack_time_ms += self.delta_attack_time_ms;
        result.sustain_level = (result.sustain_level + self.delta_sustain_level).clamp(0.0, 1.0);
        result.fm_depth_hz += self.delta_fm_depth_hz;
        result.am_depth = (result.am_depth + self.delta_am_depth).clamp(0.0, 1.0);
        result.jitter = (result.jitter + self.delta_jitter).clamp(0.0, 1.0);
        result.shimmer = (result.shimmer + self.delta_shimmer).clamp(0.0, 1.0);
        result.entropy = (result.entropy + self.delta_entropy).clamp(0.0, 1.0);
        result.loudness = (result.loudness + self.delta_loudness).clamp(0.0, 1.0);
        result.sharpness += self.delta_sharpness;
        result
    }
}

// =============================================================================
// Context Delta Calculator (Acoustic Algebra)
// =============================================================================

/// Environmental state (from RosettaPipeline)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum EnvState {
    Quiet,
    Wind,
    Rain,
    Storm,
    #[default]
    Unknown,
}

/// Interaction context for response generation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InteractionContext {
    /// Initiating a call
    Initiator,
    /// Replying to a call
    Reply,
    /// Solo vocalization
    Solo,
    /// Chorus participation
    Chorus,
}

/// Calculator for context-to-delta mapping (Acoustic Algebra)
pub struct ContextDeltaCalculator;

impl ContextDeltaCalculator {
    /// Calculate delta based on environmental state
    ///
    /// # Acoustic Algebra
    /// | Context | Delta | Result |
    /// |---------|-------|--------|
    /// | High Wind | +Pitch, +Loudness | Long_Range_Contact |
    /// | Storm | +Entropy, +Flatness | Broader band signal |
    /// | Agitation | +Jitter, +Shimmer | High urgency |
    /// | Reply | -Pitch (identity) | Individual signature |
    pub fn calculate(env: EnvState, context: InteractionContext) -> MicroDynamicsDelta {
        let mut delta = MicroDynamicsDelta::zero();

        // Environmental adaptations
        match env {
            EnvState::Wind => {
                // "Long_Range_Contact" - Cut through noise
                delta.delta_mean_f0_hz = 200.0; // Pitch up for propagation
                delta.delta_sustain_level = 0.2; // Louder
                delta.delta_loudness = 0.15; // More energy
            }
            EnvState::Rain => {
                // Moderate adaptation
                delta.delta_mean_f0_hz = 100.0;
                delta.delta_loudness = 0.1;
            }
            EnvState::Storm => {
                // Emergency signal - broader band
                delta.delta_entropy = 0.2; // More noise-like (penetrates)
                delta.delta_loudness = 0.25; // Much louder
                delta.delta_sharpness = 0.3; // More cutting
            }
            EnvState::Quiet | EnvState::Unknown => {
                // No environmental adaptation needed
            }
        }

        // Interaction context adaptations
        match context {
            InteractionContext::Reply => {
                // Individual identity marker
                delta.delta_mean_f0_hz -= 150.0; // Slightly lower pitch (identity)
            }
            InteractionContext::Initiator => {
                // Clear, strong signal
                delta.delta_sustain_level += 0.1;
            }
            InteractionContext::Solo | InteractionContext::Chorus => {
                // Standard emission
            }
        }

        delta
    }

    /// Calculate delta for emotional intensity (grading score)
    pub fn calculate_for_grading(grading_score: f32) -> MicroDynamicsDelta {
        let mut delta = MicroDynamicsDelta::zero();

        // High grading score = more emotional/volatile
        if grading_score > 0.7 {
            // "High_Urgency_Alarm"
            delta.delta_jitter = 0.15;
            delta.delta_shimmer = 0.1;
            delta.delta_entropy = 0.1;
        } else if grading_score < 0.3 {
            // Discrete, stable call
            delta.delta_jitter = -0.05;
            delta.delta_shimmer = -0.05;
        }

        delta
    }

    /// Combine multiple deltas
    pub fn combine(deltas: &[MicroDynamicsDelta]) -> MicroDynamicsDelta {
        let mut combined = MicroDynamicsDelta::zero();
        for delta in deltas {
            combined.delta_mean_f0_hz += delta.delta_mean_f0_hz;
            combined.delta_duration_ms += delta.delta_duration_ms;
            combined.delta_f0_range_hz += delta.delta_f0_range_hz;
            combined.delta_harmonic_to_noise_ratio += delta.delta_harmonic_to_noise_ratio;
            combined.delta_inharmonicity += delta.delta_inharmonicity;
            combined.delta_attack_time_ms += delta.delta_attack_time_ms;
            combined.delta_sustain_level += delta.delta_sustain_level;
            combined.delta_fm_depth_hz += delta.delta_fm_depth_hz;
            combined.delta_am_depth += delta.delta_am_depth;
            combined.delta_jitter += delta.delta_jitter;
            combined.delta_shimmer += delta.delta_shimmer;
            combined.delta_entropy += delta.delta_entropy;
            combined.delta_loudness += delta.delta_loudness;
            combined.delta_sharpness += delta.delta_sharpness;
        }
        combined
    }
}

// =============================================================================
// Formant Barrier Validation
// =============================================================================

/// Result of formant barrier validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub violations: Vec<String>,
    pub recommended_action: String,
}

/// Formant Barrier Validator
///
/// Prevents "Semantic Violations" - attempting synthesis beyond physical limits.
/// The key rule: cannot cross from Harmonic to Transient via warping alone.
pub struct FormantBarrierValidator;

impl FormantBarrierValidator {
    /// Validate if synthesis from source to target is physically possible
    pub fn validate(source: &SourceMetadata, target: &SourceMetadata) -> ValidationResult {
        let mut violations = Vec::new();

        // Check HNR change (harmonic structure)
        let hnr_delta = (target.harmonic_to_noise_ratio - source.harmonic_to_noise_ratio).abs();
        if hnr_delta > MAX_HNR_DELTA {
            violations.push(format!(
                "HNR change too large: {:.1}dB (max {:.1}dB) - would distort harmonic structure",
                hnr_delta, MAX_HNR_DELTA
            ));
        }

        // Check spectral flatness change (noise vs tonal)
        let flatness_delta = (target.entropy - source.entropy).abs();
        if flatness_delta > MAX_SPECTRAL_FLATNESS_DELTA {
            violations.push(format!(
                "Spectral flatness change too large: {:.2} (max {:.2}) - would cross modality barrier",
                flatness_delta, MAX_SPECTRAL_FLATNESS_DELTA
            ));
        }

        // Check modality crossing
        let source_modality = AcousticModality::from_metadata(source);
        let target_modality = AcousticModality::from_metadata(target);

        if source_modality != target_modality {
            if source_modality == AcousticModality::Harmonic && target_modality == AcousticModality::Transient {
                violations.push(
                    "FORMANT BARRIER VIOLATION: Cannot create a Transient (click) from a Harmonic (tone) via warping"
                        .to_string(),
                );
            } else if source_modality == AcousticModality::Transient && target_modality == AcousticModality::Harmonic {
                violations.push(
                    "FORMANT BARRIER VIOLATION: Cannot create a Harmonic (tone) from a Transient (click) via warping"
                        .to_string(),
                );
            }
        }

        // Determine recommended action
        let recommended_action = if violations.is_empty() {
            "Proceed with synthesis".to_string()
        } else if violations.iter().any(|v| v.contains("FORMANT BARRIER")) {
            "Switch source buffer to match target modality - do NOT warp".to_string()
        } else {
            "Reduce delta magnitude or split into multiple smaller transformations".to_string()
        };

        ValidationResult {
            is_valid: violations.is_empty(),
            violations,
            recommended_action,
        }
    }

    /// Quick check if modality crossing would occur
    pub fn would_cross_barrier(source: &SourceMetadata, delta: &MicroDynamicsDelta) -> bool {
        let target = delta.apply_to(source);
        let source_modality = AcousticModality::from_metadata(source);
        let target_modality = AcousticModality::from_metadata(&target);

        matches!(
            (source_modality, target_modality),
            (AcousticModality::Harmonic, AcousticModality::Transient)
                | (AcousticModality::Transient, AcousticModality::Harmonic)
        )
    }
}

// =============================================================================
// Synthesis Request
// =============================================================================

/// A synthesis request from the interaction agent
#[derive(Debug, Clone)]
pub struct SynthesisRequest {
    /// Semantic label to synthesize
    pub label: String,

    /// Environmental state
    pub environment: EnvState,

    /// Interaction context
    pub context: InteractionContext,

    /// Emotional intensity (0.0 = calm, 1.0 = urgent)
    pub grading_override: Option<f32>,

    /// Custom pitch offset (Hz)
    pub pitch_offset_hz: Option<f32>,
}

impl SynthesisRequest {
    /// Create new request for a label
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            environment: EnvState::Unknown,
            context: InteractionContext::Solo,
            grading_override: None,
            pitch_offset_hz: None,
        }
    }

    /// Set environment
    pub fn with_environment(mut self, env: EnvState) -> Self {
        self.environment = env;
        self
    }

    /// Set context
    pub fn with_context(mut self, context: InteractionContext) -> Self {
        self.context = context;
        self
    }

    /// Set grading override
    pub fn with_grading(mut self, grading: f32) -> Self {
        self.grading_override = Some(grading);
        self
    }

    /// Set pitch offset
    pub fn with_pitch_offset(mut self, offset: f32) -> Self {
        self.pitch_offset_hz = Some(offset);
        self
    }
}

// =============================================================================
// Synthesis Plan (Output of Agent)
// =============================================================================

/// Complete synthesis plan for execution
#[derive(Debug, Clone)]
pub struct SynthesisPlan {
    /// Source prototype to use
    pub source_label: String,

    /// Source audio buffer
    pub source_audio: Vec<f32>,

    /// Source metadata
    pub source_metadata: SourceMetadata,

    /// Calculated delta to apply
    pub delta: MicroDynamicsDelta,

    /// Target metadata (after delta)
    pub target_metadata: SourceMetadata,

    /// Validation result
    pub validation: ValidationResult,

    /// Description of the plan
    pub description: String,
}

// =============================================================================
// Bio-Acoustic Interaction Agent
// =============================================================================

/// The complete Bio-Acoustic Interaction Agent
///
/// Bridges the RosettaPipeline (understanding) with Granular Synthesis (response).
pub struct BioAcousticAgent {
    /// Acoustic inventory with prototypes
    inventory: AcousticInventory,
}

impl BioAcousticAgent {
    /// Create new agent with inventory
    pub fn new(inventory: AcousticInventory, _sample_rate: u32) -> Self {
        Self { inventory }
    }

    /// Load agent from inventory file
    pub fn load<P: AsRef<std::path::Path>>(path: P, sample_rate: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let inventory = AcousticInventory::load(path)?;
        Ok(Self::new(inventory, sample_rate))
    }

    /// Get the acoustic inventory
    pub fn inventory(&self) -> &AcousticInventory {
        &self.inventory
    }

    /// Get mutable inventory
    pub fn inventory_mut(&mut self) -> &mut AcousticInventory {
        &mut self.inventory
    }

    /// Plan synthesis for a request
    ///
    /// This is the main entry point that:
    /// 1. Selects source prototype from semantic label
    /// 2. Calculates context-based deltas
    /// 3. Validates against Formant Barrier
    /// 4. Returns complete synthesis plan
    pub fn plan_synthesis(&self, request: SynthesisRequest) -> Result<SynthesisPlan, String> {
        // Step 1: Retrieve prototype
        let prototype = self
            .inventory
            .get_prototype(&request.label)
            .ok_or_else(|| format!("No prototype found for label: {}", request.label))?;

        let source_metadata = prototype.metadata.clone();
        let source_audio = prototype.audio_buffer.clone();

        // Step 2: Calculate deltas
        let mut deltas = vec![ContextDeltaCalculator::calculate(request.environment, request.context)];

        // Add grading delta if specified
        if let Some(grading) = request.grading_override {
            deltas.push(ContextDeltaCalculator::calculate_for_grading(grading));
        }

        let mut combined_delta = ContextDeltaCalculator::combine(&deltas);

        // Add custom pitch offset
        if let Some(offset) = request.pitch_offset_hz {
            combined_delta.delta_mean_f0_hz += offset;
        }

        // Step 3: Calculate target metadata
        let target_metadata = combined_delta.apply_to(&source_metadata);

        // Step 4: Validate against Formant Barrier
        let validation = FormantBarrierValidator::validate(&source_metadata, &target_metadata);

        // Step 5: Build description
        let description = format!(
            "Synthesize '{}' with {:?} environment, {:?} context. {}",
            request.label,
            request.environment,
            request.context,
            if validation.is_valid {
                "Valid transformation.".to_string()
            } else {
                format!("WARNING: {}", validation.recommended_action)
            }
        );

        Ok(SynthesisPlan {
            source_label: request.label.clone(),
            source_audio,
            source_metadata,
            delta: combined_delta,
            target_metadata,
            validation,
            description,
        })
    }

    /// Select response label for an input semantic label
    pub fn select_response(&self, input_label: &str) -> Option<&String> {
        self.inventory.get_response_label(input_label)
    }

    /// Quick synthesis: label + environment -> plan
    pub fn quick_synthesize(&self, label: &str, environment: EnvState) -> Result<SynthesisPlan, String> {
        let request = SynthesisRequest::new(label)
            .with_environment(environment)
            .with_context(InteractionContext::Reply);

        self.plan_synthesis(request)
    }

    /// Get available semantic labels
    pub fn available_labels(&self) -> Vec<&String> {
        self.inventory.available_labels()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_metadata_vector_conversion() {
        let features = vec![0.5; 45];
        let meta = SourceMetadata::from_vector(&features);
        let vec = meta.to_vector();
        assert_eq!(vec.len(), 45);
        assert!((vec[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_acoustic_modality_classification() {
        let harmonic = SourceMetadata {
            harmonic_to_noise_ratio: 20.0,
            entropy: 0.2,
            ..Default::default()
        };
        assert_eq!(AcousticModality::from_metadata(&harmonic), AcousticModality::Harmonic);

        let transient = SourceMetadata {
            harmonic_to_noise_ratio: 5.0,
            entropy: 0.7,
            duration_ms: 50.0,
            ..Default::default()
        };
        assert_eq!(AcousticModality::from_metadata(&transient), AcousticModality::Transient);
    }

    #[test]
    fn test_context_delta_calculation() {
        let delta = ContextDeltaCalculator::calculate(EnvState::Wind, InteractionContext::Reply);
        assert!(delta.delta_mean_f0_hz > 0.0); // Should pitch up for wind
    }

    #[test]
    fn test_delta_application() {
        let source = SourceMetadata::default();
        let delta = MicroDynamicsDelta {
            delta_mean_f0_hz: 100.0,
            delta_loudness: 0.1,
            ..Default::default()
        };
        let target = delta.apply_to(&source);
        assert!((target.mean_f0_hz - 100.0).abs() < 0.01);
        assert!((target.loudness - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_formant_barrier_validation() {
        let source = SourceMetadata {
            harmonic_to_noise_ratio: 25.0,
            entropy: 0.1,
            ..Default::default()
        };

        // Valid small change
        let target_small = SourceMetadata {
            harmonic_to_noise_ratio: 30.0,
            ..source.clone()
        };
        let result = FormantBarrierValidator::validate(&source, &target_small);
        assert!(result.is_valid);

        // Invalid large change
        let target_large = SourceMetadata {
            harmonic_to_noise_ratio: 50.0,
            ..source.clone()
        };
        let result = FormantBarrierValidator::validate(&source, &target_large);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_acoustic_inventory() {
        let mut inventory = AcousticInventory::new("marmoset");

        let prototype = AcousticPrototype {
            label: "Phee".to_string(),
            audio_buffer: vec![0.1; 1000],
            sample_rate: 48000,
            metadata: SourceMetadata::default(),
            sample_count: 1,
            modality: AcousticModality::Harmonic,
        };

        inventory.add_prototype(prototype);
        assert!(inventory.get_prototype("Phee").is_some());
        assert_eq!(inventory.available_labels().len(), 1);
    }

    #[test]
    fn test_synthesis_request_builder() {
        let request = SynthesisRequest::new("Phee")
            .with_environment(EnvState::Wind)
            .with_context(InteractionContext::Reply)
            .with_grading(0.8)
            .with_pitch_offset(150.0);

        assert_eq!(request.label, "Phee");
        assert_eq!(request.environment, EnvState::Wind);
        assert_eq!(request.context, InteractionContext::Reply);
        assert_eq!(request.grading_override, Some(0.8));
        assert_eq!(request.pitch_offset_hz, Some(150.0));
    }

    #[test]
    fn test_agent_synthesis_plan() {
        let mut inventory = AcousticInventory::new("marmoset");

        let prototype = AcousticPrototype {
            label: "Phee".to_string(),
            audio_buffer: vec![0.1; 1000],
            sample_rate: 48000,
            metadata: SourceMetadata {
                mean_f0_hz: 7000.0,
                harmonic_to_noise_ratio: 20.0,
                entropy: 0.2,
                ..Default::default()
            },
            sample_count: 1,
            modality: AcousticModality::Harmonic,
        };

        inventory.add_prototype(prototype);

        let agent = BioAcousticAgent::new(inventory, 48000);
        let plan = agent.quick_synthesize("Phee", EnvState::Wind).unwrap();

        assert_eq!(plan.source_label, "Phee");
        assert!(plan.delta.delta_mean_f0_hz > 0.0); // Should pitch up for wind
        assert!(plan.validation.is_valid);
    }
}
