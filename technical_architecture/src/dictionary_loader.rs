//! Dictionary Loader for Field Deployment
//!
//! Loads AcousticInventory from Human-Guided Context Discovery outputs:
//! - semantic_dictionary.json (Type -> Label mapping with probabilities)
//! - type_centroids.json (Type -> 45D centroid features)
//!
//! This enables the Bio-Acoustic Agent to operate with learned dictionaries
//! from the discovery pipeline.

use crate::bio_acoustic_agent::{
    AcousticInventory, AcousticModality, AcousticPrototype, SourceMetadata,
};
use std::collections::HashMap;
use std::path::Path;

// =============================================================================
// Discovery Output Formats
// =============================================================================

/// Semantic label mapping from discovery (Type -> {Label: Probability})
pub type SemanticDictionary = HashMap<String, HashMap<String, f64>>;

/// Type centroids from discovery (Type -> 45D feature vector)
pub type TypeCentroids = HashMap<String, Vec<f64>>;

/// Loaded dictionary data
#[derive(Debug, Clone)]
pub struct LoadedDictionary {
    /// Semantic dictionary (Type -> Labels)
    pub semantic: SemanticDictionary,

    /// Type centroids (Type -> 45D features)
    pub centroids: TypeCentroids,
}

// =============================================================================
// Feature Mapping (45D zoo_vox -> SourceMetadata)
// =============================================================================

/// Feature indices in the 45D zoo_vox feature vector
pub mod feature_indices {
    // Base 30D features
    pub const MEAN_F0_HZ: usize = 0;
    pub const DURATION_MS: usize = 1;
    pub const F0_RANGE_HZ: usize = 2;
    pub const HARMONIC_TO_NOISE_RATIO: usize = 3;
    pub const SPECTRAL_FLATNESS: usize = 4; // -> entropy
    pub const HARMONICITY: usize = 5;
    pub const ATTACK_TIME_MS: usize = 6;
    pub const DECAY_TIME_MS: usize = 7;
    pub const SUSTAIN_LEVEL: usize = 8;
    pub const VIBRATO_RATE_HZ: usize = 9; // -> fm_rate_hz
    pub const VIBRATO_DEPTH: usize = 10; // -> fm_depth_hz
    pub const JITTER: usize = 11;
    pub const SHIMMER: usize = 12;
    pub const MFCC_1: usize = 13;
    pub const MFCC_2: usize = 14;
    pub const MFCC_3: usize = 15;
    pub const MFCC_4: usize = 16;
    pub const MFCC_5: usize = 17;
    pub const MFCC_6: usize = 18;
    pub const MFCC_7: usize = 19;
    pub const MFCC_8: usize = 20;
    pub const MFCC_9: usize = 21;
    pub const MFCC_10: usize = 22;
    pub const MFCC_11: usize = 23;
    pub const MFCC_12: usize = 24;
    pub const MFCC_13: usize = 25;
    pub const SPECTRAL_FLUX: usize = 26;
    pub const MEDIAN_ICI_MS: usize = 27;
    pub const ONSET_RATE_HZ: usize = 28;
    pub const ICI_CV: usize = 29;

    // Extended 15D features
    pub const FORMANT_1_HZ: usize = 30;
    pub const FORMANT_2_HZ: usize = 31;
    pub const FORMANT_3_HZ: usize = 32;
    pub const FORMANT_1_BANDWIDTH: usize = 33;
    pub const FORMANT_2_BANDWIDTH: usize = 34;
    pub const FORMANT_DISPERSION: usize = 35;
    pub const SPECTRAL_CENTROID: usize = 36;
    pub const SPECTRAL_SPREAD: usize = 37;
    pub const SPECTRAL_SKEWNESS: usize = 38;
    pub const SPECTRAL_KURTOSIS: usize = 39;
    pub const SPECTRAL_TILT: usize = 40;
    pub const FM_SLOPE: usize = 41;
    pub const AM_DEPTH: usize = 42;
    pub const SUBHARMONIC_RATIO: usize = 43;
    pub const SPECTRAL_ENTROPY: usize = 44;
}

/// Convert 45D zoo_vox features to SourceMetadata
pub fn features_to_metadata(features: &[f64]) -> SourceMetadata {
    use feature_indices::*;

    if features.len() < 45 {
        return SourceMetadata::default();
    }

    SourceMetadata {
        // Fundamental
        mean_f0_hz: features[MEAN_F0_HZ] as f32,
        duration_ms: features[DURATION_MS] as f32,
        f0_range_hz: features[F0_RANGE_HZ] as f32,
        f0_contour_slope: features[FM_SLOPE] as f32,
        pitch_stability: 1.0 - features[JITTER] as f32, // Inverse of jitter

        // Harmonic
        harmonic_to_noise_ratio: features[HARMONIC_TO_NOISE_RATIO] as f32,
        inharmonicity: 1.0 - features[HARMONICITY] as f32,
        harmonic_1: features[FORMANT_1_HZ] as f32 / 1000.0, // Normalize
        harmonic_2: features[FORMANT_2_HZ] as f32 / 1000.0,
        harmonic_3: features[FORMANT_3_HZ] as f32 / 1000.0,

        // Temporal
        attack_time_ms: features[ATTACK_TIME_MS] as f32,
        decay_time_ms: features[DECAY_TIME_MS] as f32,
        sustain_level: features[SUSTAIN_LEVEL] as f32,
        release_time_ms: features[DECAY_TIME_MS] as f32 * 0.5, // Estimate
        rms_energy: 0.5,                                       // Default, would need actual audio

        // Modulation
        fm_rate_hz: features[VIBRATO_RATE_HZ] as f32,
        fm_depth_hz: features[VIBRATO_DEPTH] as f32,
        am_rate_hz: 5.0, // Default
        am_depth: features[AM_DEPTH] as f32,
        tremolo_rate: features[VIBRATO_RATE_HZ] as f32,

        // Cepstral
        mfcc_1: features[MFCC_1] as f32,
        mfcc_2: features[MFCC_2] as f32,
        mfcc_3: features[MFCC_3] as f32,
        mfcc_4: features[MFCC_4] as f32,
        mfcc_5: features[MFCC_5] as f32,

        // Formant
        formant_1_hz: features[FORMANT_1_HZ] as f32,
        formant_2_hz: features[FORMANT_2_HZ] as f32,
        formant_3_hz: features[FORMANT_3_HZ] as f32,
        bandwidth_1: features[FORMANT_1_BANDWIDTH] as f32,
        bandwidth_2: features[FORMANT_2_BANDWIDTH] as f32,

        // Micro-Dynamics
        jitter: features[JITTER] as f32,
        shimmer: features[SHIMMER] as f32,
        hnr_variation: 0.1, // Default
        cpp: 0.5,           // Default
        entropy: features[SPECTRAL_FLATNESS] as f32,

        // Psychoacoustic
        loudness: 0.5, // Default, would need actual audio
        sharpness: features[SPECTRAL_CENTROID] as f32 / 10000.0, // Normalize
        roughness: features[SPECTRAL_FLUX] as f32 / 10.0,
        tonality: features[HARMONICITY] as f32,
        fluctuation_strength: features[AM_DEPTH] as f32,

        // TFS
        acf_peak: features[HARMONICITY] as f32,
        acf_strength: 0.5,
        sfm: features[SPECTRAL_FLATNESS] as f32,
        periodicity: features[HARMONICITY] as f32,
        tfs_entropy: features[SPECTRAL_ENTROPY] as f32,
    }
}

// =============================================================================
// Dictionary Loader
// =============================================================================

/// Dictionary loader for building AcousticInventory from discovery outputs
pub struct DictionaryLoader {
    /// Base path for dictionary files
    base_path: std::path::PathBuf,
}

impl DictionaryLoader {
    /// Create a new loader with the given base path
    pub fn new<P: AsRef<Path>>(base_path: P) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
        }
    }

    /// Load semantic dictionary from JSON
    pub fn load_semantic_dictionary(&self, species: &str) -> Result<SemanticDictionary, LoadError> {
        let path = self.base_path.join(format!(
            "{}_guided_results/{}_semantic_dictionary.json",
            species, species
        ));

        if !path.exists() {
            return Err(LoadError::FileNotFound(path.display().to_string()));
        }

        let content = std::fs::read_to_string(&path)
            .map_err(|e| LoadError::IoError(path.display().to_string(), e))?;

        serde_json::from_str(&content)
            .map_err(|e| LoadError::ParseError(path.display().to_string(), e))
    }

    /// Load type centroids from JSON
    pub fn load_type_centroids(&self, species: &str) -> Result<TypeCentroids, LoadError> {
        let path = self.base_path.join(format!(
            "{}_guided_results/{}_type_centroids.json",
            species, species
        ));

        if !path.exists() {
            return Err(LoadError::FileNotFound(path.display().to_string()));
        }

        let content = std::fs::read_to_string(&path)
            .map_err(|e| LoadError::IoError(path.display().to_string(), e))?;

        serde_json::from_str(&content)
            .map_err(|e| LoadError::ParseError(path.display().to_string(), e))
    }

    /// Load both dictionaries
    pub fn load_all(&self, species: &str) -> Result<LoadedDictionary, LoadError> {
        Ok(LoadedDictionary {
            semantic: self.load_semantic_dictionary(species)?,
            centroids: self.load_type_centroids(species)?,
        })
    }

    /// Build AcousticInventory from loaded dictionaries
    ///
    /// This creates prototypes by:
    /// 1. Grouping types by their primary semantic label
    /// 2. Averaging centroids for types with the same label
    /// 3. Creating AcousticPrototype entries with placeholder audio
    pub fn build_inventory(&self, species: &str) -> Result<AcousticInventory, LoadError> {
        let dict = self.load_all(species)?;
        let mut inventory = AcousticInventory::new(species);

        // Group types by primary label
        let mut label_types: HashMap<String, Vec<(String, Vec<f64>)>> = HashMap::new();

        for (type_id, labels) in &dict.semantic {
            // Find primary label (highest probability)
            let primary_label = labels
                .iter()
                .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(l, _)| l.clone())
                .unwrap_or_else(|| "Unknown".to_string());

            // Get centroid for this type
            if let Some(centroid) = dict.centroids.get(type_id) {
                label_types
                    .entry(primary_label)
                    .or_insert_with(Vec::new)
                    .push((type_id.clone(), centroid.clone()));
            }
        }

        // Create prototypes for each label
        for (label, types) in label_types {
            if types.is_empty() {
                continue;
            }

            // Average centroids
            let n = types.len() as f64;
            let avg_features: Vec<f64> = (0..45)
                .map(|i| {
                    types
                        .iter()
                        .map(|(_, c)| c.get(i).copied().unwrap_or(0.0))
                        .sum::<f64>()
                        / n
                })
                .collect();

            // Convert to metadata
            let metadata = features_to_metadata(&avg_features);

            // Determine modality from metadata
            let modality = AcousticModality::from_metadata(&metadata);

            // Create prototype with placeholder audio
            // In production, this would load actual audio samples
            let duration_samples = (metadata.duration_ms / 1000.0 * 48000.0) as usize;
            let placeholder_audio = vec![0.0f32; duration_samples.max(4800)]; // At least 100ms

            let prototype = AcousticPrototype {
                label: label.clone(),
                audio_buffer: placeholder_audio,
                sample_rate: 48000,
                metadata,
                sample_count: types.len(),
                modality,
            };

            inventory.add_prototype(prototype);
        }

        // Set default response strategies
        Self::set_default_strategies(&mut inventory);

        Ok(inventory)
    }

    /// Set default response strategies based on common marmoset behavior
    fn set_default_strategies(inventory: &mut AcousticInventory) {
        // Contact calls
        if inventory.get_prototype("Phee").is_some() {
            inventory.set_response_strategy("Phee", "Phee"); // Reply to contact
        }

        // Alarm calls - respond with calming contact
        if inventory.get_prototype("Tsik").is_some() && inventory.get_prototype("Phee").is_some() {
            inventory.set_response_strategy("Tsik", "Phee");
        }

        // Social calls - echo
        if inventory.get_prototype("Twitter").is_some() {
            inventory.set_response_strategy("Twitter", "Twitter");
        }

        // Trills
        if inventory.get_prototype("Trill").is_some() {
            inventory.set_response_strategy("Trill", "Trill");
        }

        // Generic fallback
        inventory.set_response_strategy("Vocalization", "Phee");
    }
}

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug)]
pub enum LoadError {
    FileNotFound(String),
    IoError(String, std::io::Error),
    ParseError(String, serde_json::Error),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::FileNotFound(path) => write!(f, "File not found: {}", path),
            LoadError::IoError(path, e) => write!(f, "IO error reading {}: {}", path, e),
            LoadError::ParseError(path, e) => write!(f, "Parse error in {}: {}", path, e),
        }
    }
}

impl std::error::Error for LoadError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_features_to_metadata() {
        // Create a sample 45D feature vector
        let mut features = vec![0.0f64; 45];
        features[feature_indices::MEAN_F0_HZ] = 7500.0;
        features[feature_indices::DURATION_MS] = 250.0;
        features[feature_indices::HARMONIC_TO_NOISE_RATIO] = 18.0;
        features[feature_indices::JITTER] = 0.03;

        let meta = features_to_metadata(&features);

        assert!((meta.mean_f0_hz - 7500.0).abs() < 1.0);
        assert!((meta.duration_ms - 250.0).abs() < 1.0);
        assert!((meta.harmonic_to_noise_ratio - 18.0).abs() < 1.0);
        assert!((meta.jitter - 0.03).abs() < 0.01);
    }

    #[test]
    fn test_modality_classification() {
        // Harmonic: high HNR, low entropy
        let mut features = vec![0.0f64; 45];
        features[feature_indices::HARMONIC_TO_NOISE_RATIO] = 20.0;
        features[feature_indices::SPECTRAL_FLATNESS] = 0.2;
        features[feature_indices::DURATION_MS] = 200.0;

        let meta = features_to_metadata(&features);
        let modality = AcousticModality::from_metadata(&meta);
        assert_eq!(modality, AcousticModality::Harmonic);

        // Transient: low HNR, high entropy, short
        features[feature_indices::HARMONIC_TO_NOISE_RATIO] = 5.0;
        features[feature_indices::SPECTRAL_FLATNESS] = 0.7;
        features[feature_indices::DURATION_MS] = 50.0;

        let meta = features_to_metadata(&features);
        let modality = AcousticModality::from_metadata(&meta);
        assert_eq!(modality, AcousticModality::Transient);
    }

    #[test]
    fn test_loader_construction() {
        let loader = DictionaryLoader::new("test_path");
        assert!(loader.base_path.to_str().unwrap().contains("test_path"));
    }
}
