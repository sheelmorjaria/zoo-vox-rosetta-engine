//! Active Learning Module for Passive Acoustic Monitoring
//! =======================================================
//!
//! Provides uncertainty-based sample flagging for expert labeling.
//! Samples with confidence scores in the marginal range (1.4-1.5) are
//! flagged for active learning, enabling continuous model improvement.
//!
//! # Key Concepts
//!
//! - **Uncertainty Range**: Confidence scores between 1.4 and 1.5 are uncertain
//! - **Detection Payload**: JSON-serializable output for downstream systems
//! - **Sample Collection**: Uncertain samples are saved for expert labeling
//!
//! # Usage
//!
//! ```rust
//! use technical_architecture::active_learning::{
//!     ActiveLearningConfig, DetectionPayload, flag_for_active_learning
//! };
//!
//! let config = ActiveLearningConfig::default();
//! let confidence = 1.45; // In uncertain range
//!
//! if flag_for_active_learning(confidence, &config) {
//!     println!("Sample flagged for expert labeling");
//! }
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

// =============================================================================
// Active Learning Configuration
// =============================================================================

/// Configuration for active learning uncertainty range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveLearningConfig {
    /// Lower bound of uncertainty range (inclusive)
    pub margin_low: f32,
    /// Upper bound of uncertainty range (exclusive)
    pub margin_high: f32,
    /// Whether to save uncertain samples to disk
    pub save_uncertain_samples: bool,
    /// Directory for storing uncertain samples
    pub uncertain_samples_dir: PathBuf,
}

impl Default for ActiveLearningConfig {
    fn default() -> Self {
        Self {
            margin_low: 1.4,
            margin_high: 1.5,
            save_uncertain_samples: true,
            uncertain_samples_dir: PathBuf::from("uncertain_samples"),
        }
    }
}

impl ActiveLearningConfig {
    /// Create a new configuration with custom thresholds
    pub fn new(margin_low: f32, margin_high: f32) -> Self {
        Self {
            margin_low,
            margin_high,
            ..Default::default()
        }
    }

    /// Check if a confidence score falls in the uncertainty range
    pub fn is_uncertain(&self, confidence: f32) -> bool {
        confidence >= self.margin_low && confidence < self.margin_high
    }
}

// =============================================================================
// Detection Payload
// =============================================================================

/// JSON-serializable detection payload for downstream systems
///
/// This is the main output format for PAM detections, designed to be
/// easily consumed by downstream systems and stored in databases.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DetectionPayload {
    /// Unix timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Species label (canonical form)
    pub species: String,
    /// Confidence score
    pub confidence: f32,
    /// Acoustic group
    pub acoustic_group: String,
    /// Taxonomic group
    pub taxon: String,
    /// Inference time in microseconds
    pub inference_time_us: u64,
    /// Whether flagged for active learning
    pub active_learning: bool,
    /// Path to saved sample (if flagged for active learning)
    pub uncertain_sample_path: Option<String>,
}

impl DetectionPayload {
    /// Create a new detection payload
    pub fn new(
        timestamp_ms: u64,
        species: String,
        confidence: f32,
        acoustic_group: String,
        taxon: String,
        inference_time_us: u64,
    ) -> Self {
        Self {
            timestamp_ms,
            species,
            confidence,
            acoustic_group,
            taxon,
            inference_time_us,
            active_learning: false,
            uncertain_sample_path: None,
        }
    }

    /// Flag this detection for active learning
    pub fn flag_for_learning(&mut self, sample_path: Option<String>) {
        self.active_learning = true;
        self.uncertain_sample_path = sample_path;
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .with_context(|| "Failed to serialize DetectionPayload to JSON")
    }

    /// Parse from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .with_context(|| "Failed to deserialize DetectionPayload from JSON")
    }
}

// =============================================================================
// Active Learning Functions
// =============================================================================

/// Check if a confidence score should be flagged for active learning
///
/// Returns true if the confidence falls in the uncertainty range.
pub fn flag_for_active_learning(confidence: f32, config: &ActiveLearningConfig) -> bool {
    config.is_uncertain(confidence)
}

/// Generate a file path for saving an uncertain sample
///
/// The path is based on timestamp and species name for easy identification.
pub fn generate_sample_path(
    species: &str,
    timestamp_ms: u64,
    config: &ActiveLearningConfig,
) -> PathBuf {
    // Sanitize species name for filesystem
    let safe_species: String = species
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();

    config
        .uncertain_samples_dir
        .join(format!("{}_{}.bin", safe_species, timestamp_ms))
}

/// Ensure the uncertain samples directory exists
pub fn ensure_samples_dir(config: &ActiveLearningConfig) -> Result<()> {
    if config.save_uncertain_samples && !config.uncertain_samples_dir.exists() {
        std::fs::create_dir_all(&config.uncertain_samples_dir).with_context(|| {
            format!(
                "Failed to create uncertain samples directory: {:?}",
                config.uncertain_samples_dir
            )
        })?;
    }
    Ok(())
}

/// Save audio samples for active learning
///
/// Saves the raw audio samples to disk for later expert labeling.
pub fn save_uncertain_sample(
    audio: &[f32],
    species: &str,
    timestamp_ms: u64,
    config: &ActiveLearningConfig,
) -> Result<PathBuf> {
    if !config.save_uncertain_samples {
        return Ok(PathBuf::new());
    }

    ensure_samples_dir(config)?;

    let path = generate_sample_path(species, timestamp_ms, config);

    // Save as raw f32 samples (simple format)
    let bytes: Vec<u8> = audio
        .iter()
        .flat_map(|&sample| sample.to_le_bytes())
        .collect();

    std::fs::write(&path, &bytes)
        .with_context(|| format!("Failed to save uncertain sample to {:?}", path))?;

    Ok(path)
}

/// Build a canonical label map from species aliases
///
/// This function canonicalizes species names to a standard format.
pub fn build_label_canonical_map(species_list: &[&str]) -> HashMap<String, String> {
    let mut map = HashMap::new();

    for &species in species_list {
        // Canonicalize: lowercase, replace underscores with spaces
        let canonical = species.to_lowercase().replace('_', " ");
        map.insert(species.to_lowercase(), canonical);
    }

    map
}

use std::collections::HashMap;

// =============================================================================
// Tests (TDD: Red Phase)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_learning_config_default() {
        let config = ActiveLearningConfig::default();
        assert!((config.margin_low - 1.4).abs() < 0.01);
        assert!((config.margin_high - 1.5).abs() < 0.01);
        assert!(config.save_uncertain_samples);
    }

    #[test]
    fn test_is_uncertain_in_range() {
        let config = ActiveLearningConfig::default();

        // In uncertain range
        assert!(config.is_uncertain(1.4));
        assert!(config.is_uncertain(1.45));
        assert!(config.is_uncertain(1.499));

        // Outside uncertain range
        assert!(!config.is_uncertain(1.3));
        assert!(!config.is_uncertain(1.5)); // Upper bound is exclusive
        assert!(!config.is_uncertain(2.0));
    }

    /// Phase 4 TDD Test: Marginal confidence should be flagged for active learning
    #[test]
    fn test_marginal_confidence_flagged_for_active_learning() {
        let config = ActiveLearningConfig::default();

        // Just below 1.5 should be flagged
        let confidence = 1.48;
        let should_flag = flag_for_active_learning(confidence, &config);

        assert!(
            should_flag,
            "Confidence {} should be flagged for active learning",
            confidence
        );

        // Create a detection and flag it
        let mut detection = DetectionPayload::new(
            1234,
            "Tursiops truncatus".to_string(),
            confidence,
            "MarineWhistle".to_string(),
            "Cetacea".to_string(),
            500,
        );

        detection.flag_for_learning(Some("samples/test.bin".to_string()));

        assert!(detection.active_learning);
        assert!(detection.uncertain_sample_path.is_some());
    }

    /// Phase 4 TDD Test: JSON output format
    #[test]
    fn test_detection_json_format() {
        let detection = DetectionPayload {
            timestamp_ms: 1234567890,
            species: "Tursiops truncatus".to_string(),
            confidence: 0.85,
            acoustic_group: "MarineWhistle".to_string(),
            taxon: "Cetacea".to_string(),
            inference_time_us: 500,
            active_learning: false,
            uncertain_sample_path: None,
        };

        let json = detection.to_json().expect("Should serialize to JSON");

        // Verify JSON contains expected fields
        assert!(json.contains("\"timestamp_ms\":"));
        assert!(json.contains("\"species\":"));
        assert!(json.contains("\"confidence\":"));
        assert!(json.contains("\"acoustic_group\":"));
        assert!(json.contains("\"taxon\":"));
        assert!(json.contains("\"inference_time_us\":"));
        assert!(json.contains("\"active_learning\":"));

        // Parse back
        let parsed = DetectionPayload::from_json(&json).expect("Should parse from JSON");
        assert_eq!(parsed, detection);
    }

    #[test]
    fn test_generate_sample_path() {
        let config = ActiveLearningConfig::default();
        let path = generate_sample_path("Tursiops truncatus", 1234567890, &config);

        assert!(path.to_str().unwrap().contains("Tursiops_truncatus"));
        assert!(path.to_str().unwrap().contains("1234567890"));
        assert!(path.to_str().unwrap().ends_with(".bin"));
    }

    #[test]
    fn test_build_label_canonical_map() {
        let species_list = vec![
            "Tursiops_truncatus",
            "Delphinus_delphis",
            "Bottlenose_Dolphin",
        ];

        let map = build_label_canonical_map(&species_list);

        // Verify canonicalization
        assert_eq!(
            map.get("tursiops_truncatus"),
            Some(&"tursiops truncatus".to_string())
        );
        assert_eq!(
            map.get("bottlenose_dolphin"),
            Some(&"bottlenose dolphin".to_string())
        );
    }

    #[test]
    fn test_detection_payload_new() {
        let detection = DetectionPayload::new(
            1000,
            "Test species".to_string(),
            0.9,
            "TestGroup".to_string(),
            "TestTaxon".to_string(),
            100,
        );

        assert_eq!(detection.timestamp_ms, 1000);
        assert_eq!(detection.species, "Test species");
        assert!((detection.confidence - 0.9).abs() < 0.01);
        assert!(!detection.active_learning);
        assert!(detection.uncertain_sample_path.is_none());
    }

    #[test]
    fn test_active_learning_config_custom() {
        let config = ActiveLearningConfig::new(1.0, 1.2);

        assert!((config.margin_low - 1.0).abs() < 0.01);
        assert!((config.margin_high - 1.2).abs() < 0.01);
        assert!(config.save_uncertain_samples); // Default preserved
    }
}
