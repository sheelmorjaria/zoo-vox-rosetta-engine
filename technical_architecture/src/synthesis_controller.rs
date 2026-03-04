//! Synthesis Controller - The "Virtual Vocalizer"
//! ==============================================
//!
//! This module enables Rosetta-Net to generate and modify 45D acoustic vectors
//! for driving the Rust Granular Synthesizer. It acts as the "Brain" that imagines
//! target sounds, while the synthesizer is the "Voice" that produces them.
//!
//! ## Integration with Corpus Analyzer
//!
//! Uses `RosettaConfig` for empirically-discovered vocabulary parameters:
//! - Vocabulary k=1020 (Peak SVS)
//! - Syntactic Depth N=6 (LRN)

use std::collections::HashMap;

use crate::corpus_analyzer::RosettaConfig;

/// Semantic attributes for vocalization modification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VocalAttribute {
    /// Agitated/aroused state
    Agitated,
    /// Calm/relaxed state
    Calm,
    /// Grumpy/annoyed state (lower HNR, faster attack)
    Grumpy,
    /// Playful state
    Playful,
    /// Alarm/distress state
    Alarm,
    /// Social/contact state
    Social,
    /// Territorial state
    Territorial,
    /// Juvenile/young state
    Juvenile,
}

/// Configuration for the synthesis controller
#[derive(Debug, Clone)]
pub struct SynthesisControllerConfig {
    pub latent_dim: usize,
    pub output_dim: usize,
    pub default_smoothness: f32,
    /// Vocabulary configuration from corpus analysis
    /// Uses empirically discovered k=1020, N=6
    pub vocab_config: RosettaConfig,
}

impl Default for SynthesisControllerConfig {
    fn default() -> Self {
        Self {
            latent_dim: 128,
            output_dim: 45,
            default_smoothness: 0.5,
            vocab_config: RosettaConfig::default(), // k=1020, N=6
        }
    }
}

/// The Synthesis Controller - generates and modifies 45D acoustic vectors
#[derive(Debug, Clone)]
pub struct SynthesisController {
    config: SynthesisControllerConfig,
    attribute_modifiers: HashMap<VocalAttribute, [f32; 45]>,
    species_prototypes: HashMap<String, [f32; 45]>,
    seed: u64,
}

impl SynthesisController {
    pub fn new() -> Self {
        Self::with_config(SynthesisControllerConfig::default())
    }

    pub fn with_config(config: SynthesisControllerConfig) -> Self {
        let mut controller = Self {
            config,
            attribute_modifiers: HashMap::new(),
            species_prototypes: HashMap::new(),
            seed: 42,
        };
        controller.initialize_attribute_modifiers();
        controller
    }

    fn initialize_attribute_modifiers(&mut self) {
        // 45D: Fundamental(3)+Grit(3)+Motion(7)+MFCCs(14)+Rhythm(3)+Resonance(6)+Spectral(4)+Modulation(3)+NonLinear(2)=45
        let grumpy: [f32; 45] = [
            -100.0, 0.0, 0.0, // Fundamental (3)
            -3.0, 0.1, -0.1, // Grit (3)
            -5.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Motion (7)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, // MFCCs (14)
            0.0, 0.0, 0.0, // Rhythm (3)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Resonance (6)
            0.0, 0.0, 0.0, 0.0, // Spectral (4)
            0.0, 0.0, 0.0, // Modulation (3)
            0.0, 0.0, // Non-Linear (2)
        ];
        self.attribute_modifiers
            .insert(VocalAttribute::Grumpy, grumpy);

        let agitated: [f32; 45] = [
            500.0, -50.0, 200.0, // Fundamental (3)
            2.0, -0.1, 0.1, // Grit (3)
            -10.0, 5.0, 0.1, 0.0, 0.0, 0.0, 0.0, // Motion (7)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, // MFCCs (14)
            0.0, 0.0, 0.0, // Rhythm (3)
            100.0, 200.0, 0.0, 0.0, 0.0, 0.0, // Resonance (6)
            0.0, 0.0, 0.0, 0.0, // Spectral (4)
            0.0, 0.0, 0.0, // Modulation (3)
            0.0, 0.0, // Non-Linear (2)
        ];
        self.attribute_modifiers
            .insert(VocalAttribute::Agitated, agitated);

        let calm: [f32; 45] = [
            -200.0, 100.0, -50.0, // Fundamental (3)
            3.0, 0.05, 0.05, // Grit (3)
            10.0, 20.0, 0.1, 0.0, 0.0, 0.0, 0.0, // Motion (7)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, // MFCCs (14)
            0.0, 0.0, 0.0, // Rhythm (3)
            -50.0, -100.0, 0.0, 0.0, 0.0, 0.0, // Resonance (6)
            0.0, 0.0, 0.0, 0.0, // Spectral (4)
            0.0, 0.0, 0.0, // Modulation (3)
            0.0, 0.0, // Non-Linear (2)
        ];
        self.attribute_modifiers.insert(VocalAttribute::Calm, calm);

        let alarm: [f32; 45] = [
            1000.0, -100.0, 500.0, // Fundamental (3)
            -5.0, 0.2, -0.2, // Grit (3)
            -15.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Motion (7)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, // MFCCs (14)
            0.0, 0.0, 0.0, // Rhythm (3)
            200.0, 400.0, 0.0, 0.0, 0.0, 0.0, // Resonance (6)
            0.0, 0.0, 0.0, 0.0, // Spectral (4)
            0.0, 0.0, 0.0, // Modulation (3)
            0.0, 0.0, // Non-Linear (2)
        ];
        self.attribute_modifiers
            .insert(VocalAttribute::Alarm, alarm);
    }

    pub fn generate_45d(&self, latent: &[f32]) -> [f32; 45] {
        let mut result = [0.0f32; 45];
        let len = 45.min(latent.len());
        result[..len].copy_from_slice(&latent[..len]);
        result
    }

    pub fn apply_attribute(&self, vector: &[f32; 45], attribute: VocalAttribute) -> [f32; 45] {
        let modifier = self
            .attribute_modifiers
            .get(&attribute)
            .copied()
            .unwrap_or([0.0; 45]);
        let mut result = *vector;
        for i in 0..45 {
            result[i] += modifier[i];
        }
        result
    }

    pub fn apply_attributes(
        &self,
        vector: &[f32; 45],
        attributes: &[(VocalAttribute, f32)],
    ) -> [f32; 45] {
        let mut result = *vector;
        for (attr, weight) in attributes {
            if let Some(modifier) = self.attribute_modifiers.get(attr) {
                for i in 0..45 {
                    result[i] += modifier[i] * weight;
                }
            }
        }
        result
    }

    pub fn interpolate(&self, v1: &[f32; 45], v2: &[f32; 45], alpha: f32) -> [f32; 45] {
        let alpha = alpha.clamp(0.0, 1.0);
        let mut result = [0.0f32; 45];
        for i in 0..45 {
            result[i] = v1[i] * (1.0 - alpha) + v2[i] * alpha;
        }
        result
    }

    pub fn smooth_interpolate(&self, v1: &[f32; 45], v2: &[f32; 45], t: f32) -> [f32; 45] {
        let t = t.clamp(0.0, 1.0);
        let smooth_t = t * t * (3.0 - 2.0 * t);
        self.interpolate(v1, v2, smooth_t)
    }

    pub fn generate_species_vector(&self, species: &str) -> Option<[f32; 45]> {
        self.species_prototypes.get(species).copied()
    }

    pub fn register_species_prototype(&mut self, species: &str, examples: &[[f32; 45]]) {
        if examples.is_empty() {
            return;
        }
        let mut prototype = [0.0f32; 45];
        for example in examples {
            for i in 0..45 {
                prototype[i] += example[i];
            }
        }
        for i in 0..45 {
            prototype[i] /= examples.len() as f32;
        }
        self.species_prototypes
            .insert(species.to_string(), prototype);
    }

    pub fn generate_random(&mut self) -> [f32; 45] {
        let mut result = [0.0f32; 45];
        for i in 0..45 {
            self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
            result[i] = ((self.seed >> 16) as f32 / 65536.0 - 0.5) * 2.0;
        }
        result
    }

    pub fn extrapolate(&self, vector: &[f32; 45], direction: &[f32; 45], factor: f32) -> [f32; 45] {
        let mut result = [0.0f32; 45];
        for i in 0..45 {
            result[i] = vector[i] + direction[i] * factor;
        }
        result
    }

    pub fn compute_delta(&self, from: &[f32; 45], to: &[f32; 45]) -> [f32; 45] {
        let mut delta = [0.0f32; 45];
        for i in 0..45 {
            delta[i] = to[i] - from[i];
        }
        delta
    }

    pub fn normalize(&self, vector: &[f32; 45]) -> [f32; 45] {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            return *vector;
        }
        let mut result = [0.0f32; 45];
        for i in 0..45 {
            result[i] = vector[i] / norm;
        }
        result
    }

    pub fn available_attributes(&self) -> Vec<VocalAttribute> {
        self.attribute_modifiers.keys().copied().collect()
    }

    pub fn registered_species(&self) -> Vec<String> {
        self.species_prototypes.keys().cloned().collect()
    }
}

impl Default for SynthesisController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_creation() {
        let controller = SynthesisController::new();
        assert_eq!(controller.config.latent_dim, 128);
    }

    #[test]
    fn test_generate_45d() {
        let controller = SynthesisController::new();
        let latent = vec![0.5f32; 128];
        let vector = controller.generate_45d(&latent);
        assert_eq!(vector.len(), 45);
    }

    #[test]
    fn test_apply_attribute() {
        let controller = SynthesisController::new();
        let base = [1000.0f32; 45];
        let modified = controller.apply_attribute(&base, VocalAttribute::Grumpy);
        assert!(modified[0] < base[0]);
    }

    #[test]
    fn test_interpolate() {
        let controller = SynthesisController::new();
        let v1 = [0.0f32; 45];
        let v2 = [100.0f32; 45];
        let result = controller.interpolate(&v1, &v2, 0.5);
        for &v in &result {
            assert!((v - 50.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_species_prototype() {
        let mut controller = SynthesisController::new();
        controller.register_species_prototype("test", &[[1000.0f32; 45]]);
        assert!(controller.generate_species_vector("test").is_some());
    }

    #[test]
    fn test_normalize() {
        let controller = SynthesisController::new();
        let vector = [3.0f32; 45];
        let normalized = controller.normalize(&vector);
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_random() {
        let mut controller = SynthesisController::new();
        let v1 = controller.generate_random();
        let v2 = controller.generate_random();
        let diff: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0);
    }
}
