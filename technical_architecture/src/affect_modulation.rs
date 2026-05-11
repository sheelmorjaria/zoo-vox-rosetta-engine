//! Affect Modulation Mapping (Module 4)
//! ===================================
//!
//! Explicit mathematical mapping from 16D affect vector to DDSP synthesis parameters.
//!
//! Mapping Specification:
//! - Dimension 0 (Arousal): Harmonic-to-Noise Ratio (HNR) scaling
//! - Dimension 1 (Valence): Jitter/Shimmer injection
//! - Dimension 2 (Pitch Variation): Vibrato depth scaling
//! - Dimensions 3-15: Reserved for future expansion
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use crate::synthesis::DynamicMicroharmonicParams;
use serde::{Deserialize, Serialize};

/// Affective latent dimensions (16D β-VAE output)
///
/// Each dimension corresponds to a biologically-meaningful trait
/// extracted from the β-VAE trained on affective features.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AffectiveLatent {
    /// Dimension 0: Arousal (0-1) - activation level
    /// High arousal = increased energy, faster tempo
    pub arousal: f32,

    /// Dimension 1: Valence (-1 to 1) - pleasantness/unpleasantness
    /// Negative valence = harsh, aggressive; Positive = friendly
    pub valence: f32,

    /// Dimension 2: Pitch Variation (0-1) - melodic expressiveness
    /// Higher = more pitch modulation, wider F0 range
    pub pitch_variation: f32,

    /// Dimensions 3-15: Reserved for future affective dimensions
    /// Potential expansions: tension, certainty, sociality, etc.
    pub reserved: [f32; 13],
}

impl AffectiveLatent {
    /// Create a new affective latent vector from 16D array.
    ///
    /// # Arguments
    /// * `vector` - 16D array from β-VAE encoder
    ///
    /// # Returns
    /// * `AffectiveLatent` if input has valid length
    /// * `None` if input length != 16
    pub fn from_vector(vector: &[f32]) -> Option<Self> {
        if vector.len() != 16 {
            return None;
        }

        let mut reserved = [0.0; 13];
        reserved.copy_from_slice(&vector[3..16]);

        Some(Self {
            arousal: vector[0],
            valence: vector[1],
            pitch_variation: vector[2],
            reserved,
        })
    }

    /// Create affective latent from individual components.
    pub fn new(arousal: f32, valence: f32, pitch_variation: f32) -> Self {
        Self {
            arousal: arousal.clamp(0.0, 1.0),
            valence: valence.clamp(-1.0, 1.0),
            pitch_variation: pitch_variation.clamp(0.0, 1.0),
            reserved: [0.0; 13],
        }
    }

    /// Get as 16D array for serialization/transmission.
    pub fn to_vector(&self) -> [f32; 16] {
        let mut vector = [0.0; 16];
        vector[0] = self.arousal;
        vector[1] = self.valence;
        vector[2] = self.pitch_variation;
        vector[3..16].copy_from_slice(&self.reserved);
        vector
    }
}

/// Affect modulation parameters mapped to acoustic controls.
///
/// These parameters are applied to the base DDSP parameters
/// to generate affectively-modulated synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AffectModulation {
    /// HNR scaling factor (0.5 to 2.0)
    /// Arousal mapping: higher arousal → lower HNR (more noise/chaos)
    pub hnr_scaling: f32,

    /// Jitter factor (0.0 to 1.0)
    /// Valence mapping: negative valence → more jitter (instability)
    pub jitter_factor: f32,

    /// Shimmer factor (0.0 to 1.0)
    /// Valence mapping: negative valence → more shimmer (amplitude instability)
    pub shimmer_factor: f32,

    /// Vibrato depth in Hz (0 to 100)
    /// Pitch variation mapping: higher → more vibrato
    pub vibrato_depth_hz: f32,

    /// Vibrato rate in Hz (0 to 20)
    /// Arousal mapping: higher arousal → faster vibrato
    pub vibrato_rate_hz: f32,

    /// Spectral tilt offset in dB (-6 to +6)
    /// Arousal mapping: higher arousal → brighter (less tilt)
    pub spectral_tilt_offset_db: f32,

    /// Attack time scaling (0.5 to 2.0)
    /// Arousal mapping: higher arousal → sharper attack
    pub attack_scaling: f32,

    /// Reserved for future modulation parameters
    pub reserved: [f32; 8],
}

impl Default for AffectModulation {
    fn default() -> Self {
        Self {
            hnr_scaling: 1.0,
            jitter_factor: 0.0,
            shimmer_factor: 0.0,
            vibrato_depth_hz: 25.0,
            vibrato_rate_hz: 7.0,
            spectral_tilt_offset_db: 0.0,
            attack_scaling: 1.0,
            reserved: [0.0; 8],
        }
    }
}

/// Affect Modulation Mapper
///
/// Maps 16D affective latent vectors to DDSP synthesis modulation parameters.
pub struct AffectModulationMapper {
    /// Configuration for modulation sensitivity
    config: ModulationConfig,
}

/// Configuration for affect modulation mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModulationConfig {
    /// Maximum HNR scaling (for arousal = 0)
    pub max_hnr_scaling: f32,

    /// Minimum HNR scaling (for arousal = 1)
    pub min_hnr_scaling: f32,

    /// Maximum jitter injection (for valence = -1)
    pub max_jitter: f32,

    /// Maximum shimmer injection (for valence = -1)
    pub max_shimmer: f32,

    /// Maximum vibrato depth (for pitch_variation = 1)
    pub max_vibrato_depth: f32,

    /// Base vibrato rate (Hz)
    pub base_vibrato_rate: f32,

    /// Maximum vibrato rate (Hz, for arousal = 1)
    pub max_vibrato_rate: f32,

    /// Maximum spectral tilt boost (dB, for arousal = 1)
    pub max_tilt_boost: f32,

    /// Maximum attack time acceleration (for arousal = 1)
    pub max_attack_accel: f32,
}

impl Default for ModulationConfig {
    fn default() -> Self {
        Self {
            // HNR: arousal 0 → 2.0x (very clean), arousal 1 → 0.5x (noisy)
            max_hnr_scaling: 2.0,
            min_hnr_scaling: 0.5,

            // Jitter/Shimmer: valence -1 → max, valence 1 → 0
            max_jitter: 0.08,
            max_shimmer: 0.05,

            // Vibrato depth: pitch_variation 0 → 10Hz, 1 → 80Hz
            max_vibrato_depth: 80.0,

            // Vibrato rate: arousal 0 → 5Hz, 1 → 15Hz
            base_vibrato_rate: 5.0,
            max_vibrato_rate: 15.0,

            // Spectral tilt: arousal 1 → boost by +3dB (brighter)
            max_tilt_boost: 3.0,

            // Attack: arousal 1 → 2x faster (sharper)
            max_attack_accel: 2.0,
        }
    }
}

impl Default for AffectModulationMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl AffectModulationMapper {
    /// Create a new mapper with default configuration.
    pub fn new() -> Self {
        Self {
            config: ModulationConfig::default(),
        }
    }

    /// Create a mapper with custom configuration.
    pub fn with_config(config: ModulationConfig) -> Self {
        Self { config }
    }

    /// Map affective latent to modulation parameters.
    ///
    /// # Mathematical Mapping
    ///
    /// **HNR Scaling (Arousal → Noise)**
    /// ```text
    /// hnr_scaling = max_hnr - arousal * (max_hnr - min_hnr)
    /// ```
    /// High arousal → low HNR → more noise (chaotic/urgent)
    ///
    /// **Jitter/Shimmer (Valence → Stability)**
    /// ```text
    /// jitter_factor = max(0, -valence) * max_jitter
    /// shimmer_factor = max(0, -valence) * max_shimmer
    /// ```
    /// Negative valence → instability (harsh/aggressive)
    ///
    /// **Vibrato Depth (Pitch Variation → Expressiveness)**
    /// ```text
    /// vibrato_depth_hz = 10 + pitch_variation * (max_depth - 10)
    /// ```
    /// Higher pitch variation → more vibrato depth
    ///
    /// **Vibrato Rate (Arousal → Tempo)**
    /// ```text
    /// vibrato_rate_hz = base_rate + arousal * (max_rate - base_rate)
    /// ```
    /// High arousal → faster vibrato (higher tempo)
    ///
    /// **Spectral Tilt (Arousal → Brightness)**
    /// ```text
    /// spectral_tilt_offset = arousal * max_tilt_boost
    /// ```
    /// High arousal → brighter (less high-freq rolloff)
    ///
    /// **Attack Scaling (Arousal → Sharpness)**
    /// ```text
    /// attack_scaling = 1.0 + arousal * (max_accel - 1.0)
    /// ```
    /// High arousal → sharper attack (more percussive)
    pub fn map(&self, affect: &AffectiveLatent) -> AffectModulation {
        // HNR Scaling: arousal 0 → clean (2x), arousal 1 → noisy (0.5x)
        let hnr_scaling = self.config.max_hnr_scaling
            - affect.arousal * (self.config.max_hnr_scaling - self.config.min_hnr_scaling);

        // Jitter: only for negative valence (harshness)
        let valence_negativity = (-affect.valence).max(0.0);
        let jitter_factor = valence_negativity * self.config.max_jitter;

        // Shimmer: only for negative valence (harshness)
        let shimmer_factor = valence_negativity * self.config.max_shimmer;

        // Vibrato depth: controlled by pitch variation
        let vibrato_depth_hz = 10.0 + affect.pitch_variation * (self.config.max_vibrato_depth - 10.0);

        // Vibrato rate: controlled by arousal (tempo)
        let vibrato_rate_hz = self.config.base_vibrato_rate
            + affect.arousal * (self.config.max_vibrato_rate - self.config.base_vibrato_rate);

        // Spectral tilt offset: high arousal → brighter
        let spectral_tilt_offset_db = affect.arousal * self.config.max_tilt_boost;

        // Attack scaling: high arousal → sharper
        let attack_scaling = 1.0 + affect.arousal * (self.config.max_attack_accel - 1.0);

        AffectModulation {
            hnr_scaling: hnr_scaling.clamp(0.1, 3.0),
            jitter_factor: jitter_factor.clamp(0.0, 0.2),
            shimmer_factor: shimmer_factor.clamp(0.0, 0.2),
            vibrato_depth_hz: vibrato_depth_hz.clamp(0.0, 100.0),
            vibrato_rate_hz: vibrato_rate_hz.clamp(0.0, 20.0),
            spectral_tilt_offset_db: spectral_tilt_offset_db.clamp(-6.0, 6.0),
            attack_scaling: attack_scaling.clamp(0.5, 2.0),
            reserved: [0.0; 8],
        }
    }

    /// Apply modulation to base DDSP parameters.
    ///
    /// This method modifies the base synthesis parameters with the
    /// affective modulation, producing the final parameters for synthesis.
    pub fn apply_to_params(
        &self,
        affect: &AffectiveLatent,
        base_params: &DynamicMicroharmonicParams,
    ) -> DynamicMicroharmonicParams {
        let modulation = self.map(affect);

        let mut params = base_params.clone();

        // Apply HNR scaling (logarithmic domain)
        params.hnr_db = (params.hnr_db + modulation.hnr_scaling.log10() * 10.0).clamp(0.0, 40.0);

        // Apply jitter and shimmer
        params.jitter_amount = (params.jitter_amount + modulation.jitter_factor).clamp(0.0, 0.1);
        params.shimmer_amount = (params.shimmer_amount + modulation.shimmer_factor).clamp(0.0, 0.1);

        // Apply vibrato depth (convert Hz to cents)
        let base_f0 = params.f0_base;
        let vibrato_depth_cents = (modulation.vibrato_depth_hz / base_f0 * 1200.0).log2() * 1200.0;
        params.vibrato_depth_cents = vibrato_depth_cents.clamp(0.0, 100.0);

        // Apply vibrato rate
        params.vibrato_rate_hz = modulation.vibrato_rate_hz;

        // Apply spectral tilt offset
        params.spectral_tilt = (params.spectral_tilt + modulation.spectral_tilt_offset_db).clamp(-12.0, 0.0);

        // Apply attack scaling
        params.attack_ms = (params.attack_ms / modulation.attack_scaling).clamp(0.0, 100.0);

        params
    }

    /// Get the current configuration.
    pub fn config(&self) -> &ModulationConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: ModulationConfig) {
        self.config = config;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affective_latent_from_vector() {
        let vector = vec![
            0.7,   // arousal
            -0.3,  // valence
            0.5,   // pitch_variation
            // 13 reserved values
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.1, 0.2,
        ];

        let affect = AffectiveLatent::from_vector(&vector).unwrap();
        assert_eq!(affect.arousal, 0.7);
        assert_eq!(affect.valence, -0.3);
        assert_eq!(affect.pitch_variation, 0.5);
    }

    #[test]
    fn test_affective_latent_invalid_length() {
        let vector = vec![0.1, 0.2, 0.3]; // Too short
        assert!(AffectiveLatent::from_vector(&vector).is_none());
    }

    #[test]
    fn test_affective_latent_to_vector_roundtrip() {
        let original = AffectiveLatent::new(0.5, -0.2, 0.8);
        let vector = original.to_vector();
        let restored = AffectiveLatent::from_vector(&vector).unwrap();

        assert_eq!(restored.arousal, original.arousal);
        assert_eq!(restored.valence, original.valence);
        assert_eq!(restored.pitch_variation, original.pitch_variation);
    }

    #[test]
    fn test_affective_latent_clamping() {
        let affect = AffectiveLatent::new(1.5, -2.0, 1.5);
        assert_eq!(affect.arousal, 1.0);  // Clamped to 1
        assert_eq!(affect.valence, -1.0); // Clamped to -1
        assert_eq!(affect.pitch_variation, 1.0); // Clamped to 1
    }

    #[test]
    fn test_hnr_scaling_arousal_mapping() {
        let mapper = AffectModulationMapper::new();

        // Low arousal → high HNR (clean)
        let low_arousal = AffectiveLatent::new(0.0, 0.0, 0.5);
        let mod_low = mapper.map(&low_arousal);
        assert!(mod_low.hnr_scaling > 1.5);

        // High arousal → low HNR (noisy)
        let high_arousal = AffectiveLatent::new(1.0, 0.0, 0.5);
        let mod_high = mapper.map(&high_arousal);
        assert!(mod_high.hnr_scaling < 0.8);

        assert!(mod_low.hnr_scaling > mod_high.hnr_scaling);
    }

    #[test]
    fn test_jitter_shimmer_valence_mapping() {
        let mapper = AffectModulationMapper::new();

        // Positive valence → no jitter/shimmer
        let positive_valence = AffectiveLatent::new(0.5, 1.0, 0.5);
        let mod_pos = mapper.map(&positive_valence);
        assert_eq!(mod_pos.jitter_factor, 0.0);
        assert_eq!(mod_pos.shimmer_factor, 0.0);

        // Negative valence → jitter/shimmer
        let negative_valence = AffectiveLatent::new(0.5, -1.0, 0.5);
        let mod_neg = mapper.map(&negative_valence);
        assert!(mod_neg.jitter_factor > 0.0);
        assert!(mod_neg.shimmer_factor > 0.0);
    }

    #[test]
    fn test_vibrato_depth_pitch_variation_mapping() {
        let mapper = AffectModulationMapper::new();

        // Low pitch variation → shallow vibrato
        let low_pv = AffectiveLatent::new(0.5, 0.0, 0.0);
        let mod_low = mapper.map(&low_pv);
        assert!(mod_low.vibrato_depth_hz < 30.0);

        // High pitch variation → deep vibrato
        let high_pv = AffectiveLatent::new(0.5, 0.0, 1.0);
        let mod_high = mapper.map(&high_pv);
        assert!(mod_high.vibrato_depth_hz > 50.0);
    }

    #[test]
    fn test_vibrato_rate_arousal_mapping() {
        let mapper = AffectModulationMapper::new();

        // Low arousal → slow vibrato
        let low_arousal = AffectiveLatent::new(0.0, 0.0, 0.5);
        let mod_low = mapper.map(&low_arousal);
        assert_eq!(mod_low.vibrato_rate_hz, 5.0);

        // High arousal → fast vibrato
        let high_arousal = AffectiveLatent::new(1.0, 0.0, 0.5);
        let mod_high = mapper.map(&high_arousal);
        assert_eq!(mod_high.vibrato_rate_hz, 15.0);
    }

    #[test]
    fn test_spectral_tilt_arousal_mapping() {
        let mapper = AffectModulationMapper::new();

        // Low arousal → no tilt offset (darker)
        let low_arousal = AffectiveLatent::new(0.0, 0.0, 0.5);
        let mod_low = mapper.map(&low_arousal);
        assert_eq!(mod_low.spectral_tilt_offset_db, 0.0);

        // High arousal → positive offset (brighter)
        let high_arousal = AffectiveLatent::new(1.0, 0.0, 0.5);
        let mod_high = mapper.map(&high_arousal);
        assert_eq!(mod_high.spectral_tilt_offset_db, 3.0);
    }

    #[test]
    fn test_attack_scaling_arousal_mapping() {
        let mapper = AffectModulationMapper::new();

        // Low arousal → normal attack
        let low_arousal = AffectiveLatent::new(0.0, 0.0, 0.5);
        let mod_low = mapper.map(&low_arousal);
        assert_eq!(mod_low.attack_scaling, 1.0);

        // High arousal → faster attack
        let high_arousal = AffectiveLatent::new(1.0, 0.0, 0.5);
        let mod_high = mapper.map(&high_arousal);
        assert_eq!(mod_high.attack_scaling, 2.0);
    }

    #[test]
    fn test_apply_to_params_hnr() {
        let mapper = AffectModulationMapper::new();

        let affect = AffectiveLatent::new(1.0, 0.0, 0.5); // High arousal
        let base_params = DynamicMicroharmonicParams {
            hnr_db: 20.0,
            ..Default::default()
        };

        let modulated = mapper.apply_to_params(&affect, &base_params);

        // HNR should be reduced due to high arousal
        assert!(modulated.hnr_db < base_params.hnr_db);
    }

    #[test]
    fn test_apply_to_params_vibrato() {
        let mapper = AffectModulationMapper::new();

        let affect = AffectiveLatent::new(0.5, 0.0, 1.0); // High pitch variation
        let base_params = DynamicMicroharmonicParams::default();

        let modulated = mapper.apply_to_params(&affect, &base_params);

        // Vibrato rate should be set by arousal (0.5)
        assert_eq!(modulated.vibrato_rate_hz, 10.0);
        // Vibrato depth should be increased by pitch variation
        assert!(modulated.vibrato_depth_cents > 10.0);
    }

    #[test]
    fn test_apply_to_params_jitter_shimmer() {
        let mapper = AffectModulationMapper::new();

        let affect = AffectiveLatent::new(0.5, -1.0, 0.5); // Negative valence
        let base_params = DynamicMicroharmonicParams {
            jitter_amount: 0.0,
            shimmer_amount: 0.0,
            ..Default::default()
        };

        let modulated = mapper.apply_to_params(&affect, &base_params);

        // Jitter and shimmer should be added
        assert!(modulated.jitter_amount > 0.0);
        assert!(modulated.shimmer_amount > 0.0);
    }

    #[test]
    fn test_custom_config() {
        let config = ModulationConfig {
            max_hnr_scaling: 3.0,
            min_hnr_scaling: 0.3,
            ..Default::default()
        };

        let mapper = AffectModulationMapper::with_config(config);

        let affect = AffectiveLatent::new(0.0, 0.0, 0.5);
        let modulation = mapper.map(&affect);

        assert_eq!(modulation.hnr_scaling, 3.0);
    }

    #[test]
    fn test_modulation_clamping() {
        let mapper = AffectModulationMapper::new();

        let affect = AffectiveLatent::new(1.0, -1.0, 1.0);
        let modulation = mapper.map(&affect);

        // All values should be clamped to valid ranges
        assert!(modulation.hnr_scaling >= 0.1 && modulation.hnr_scaling <= 3.0);
        assert!(modulation.jitter_factor >= 0.0 && modulation.jitter_factor <= 0.2);
        assert!(modulation.shimmer_factor >= 0.0 && modulation.shimmer_factor <= 0.2);
        assert!(modulation.vibrato_depth_hz >= 0.0 && modulation.vibrato_depth_hz <= 100.0);
        assert!(modulation.vibrato_rate_hz >= 0.0 && modulation.vibrato_rate_hz <= 20.0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTEGRATION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    #[ignore = "Requires full DDSP pipeline"]
    fn test_full_modulation_pipeline() {
        // This test would verify the complete pipeline:
        // 1. Receive 16D affect vector from β-VAE
        // 2. Map to modulation parameters
        // 3. Apply to base DDSP parameters
        // 4. Generate audio with perceptible affective variation
    }

    /// Verify de-escalation behavior: high arousal → reduced HNR (calmer sound)
    #[test]
    fn test_deescalation_reduces_arousal_characteristics() {
        let mapper = AffectModulationMapper::new();

        // High arousal state
        let high_arousal = AffectiveLatent::new(0.9, 0.0, 0.5);
        let modulation_high = mapper.map(&high_arousal);

        // After de-escalation to 0.6
        let deescalated = AffectiveLatent::new(0.6, 0.0, 0.5);
        let modulation_deescalated = mapper.map(&deescalated);

        // De-escalated should have higher HNR (cleaner sound)
        assert!(modulation_deescalated.hnr_scaling > modulation_high.hnr_scaling);

        // De-escalated should have slower tempo (vibrato rate)
        assert!(modulation_deescalated.vibrato_rate_hz < modulation_high.vibrato_rate_hz);
    }
}
