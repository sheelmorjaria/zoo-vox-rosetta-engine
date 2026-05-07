//! Species Vocab Config Module - Direction 1: Adaptive Vocabulary
//! =============================================================
//!
//! This module provides species-specific vocabulary configuration storage
//! and retrieval for the Rust Execution Layer.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   SpeciesVocabConfig                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ + species: String                                              │
//! │ + optimal_k: usize                                              │
//! │ + svs_score: f64                                                │
//! │ + discovery_timestamp: i64                                      │
//! └─────────────────────────────────────────────────────────────────┘
//!                              ▲
//!                              │
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   SpeciesVocabRegistry                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ + configs: HashMap<String, SpeciesVocabConfig>                  │
//! │ + register(config)                                              │
//! │ + get(species) -> Option<Config>                                │
//! │ + export_to_json() -> Result<String>                            │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust
//! use technical_architecture::species_vocab_config::{SpeciesVocabConfig, SpeciesVocabRegistry};
//!
//! // Create a config
//! let config = SpeciesVocabConfig::new("bat", 1020, 0.45);
//!
//! // Register in registry
//! let mut registry = SpeciesVocabRegistry::new();
//! registry.register(config);
//!
//! // Retrieve optimal k
//! let k = registry.get_optimal_k("bat", 1020);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum VocabConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Config not found for species: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, VocabConfigError>;

// ============================================================================
// Species Vocab Config
// ============================================================================

/// Species-specific vocabulary configuration
///
/// Stores the optimal vocabulary size (k) and Silhouette Validation Score (SVS)
/// for a species, along with the timestamp of discovery.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeciesVocabConfig {
    /// Species name (e.g., "egyptian_fruit_bat", "marmoset")
    pub species: String,

    /// Optimal vocabulary size (k) for this species
    pub optimal_k: usize,

    /// Silhouette Validation Score (0.0 - 1.0, higher is better)
    pub svs_score: f64,

    /// Unix timestamp of when this config was discovered
    pub discovery_timestamp: i64,
}

impl SpeciesVocabConfig {
    /// Create a new vocabulary configuration
    ///
    /// # Arguments
    /// * `species` - Species name
    /// * `optimal_k` - Optimal vocabulary size
    /// * `svs_score` - Silhouette Validation Score
    pub fn new(species: impl Into<String>, optimal_k: usize, svs_score: f64) -> Self {
        Self {
            species: species.into(),
            optimal_k,
            svs_score,
            discovery_timestamp: chrono::Utc::now().timestamp(),
        }
    }

    /// Create a config with a specific timestamp (useful for testing)
    pub fn with_timestamp(species: impl Into<String>, optimal_k: usize, svs_score: f64, timestamp: i64) -> Self {
        Self {
            species: species.into(),
            optimal_k,
            svs_score,
            discovery_timestamp: timestamp,
        }
    }

    /// Check if this configuration is recent (within last 30 days)
    pub fn is_recent(&self) -> bool {
        let now = chrono::Utc::now().timestamp();
        let thirty_days = 30 * 24 * 60 * 60;
        now - self.discovery_timestamp < thirty_days
    }

    /// Check if SVS score is good (> 0.4)
    pub fn is_good_quality(&self) -> bool {
        self.svs_score > 0.4
    }
}

// ============================================================================
// Species Vocab Registry
// ============================================================================

/// Registry for species-specific vocabulary configurations
///
/// Stores and retrieves `SpeciesVocabConfig` entries, with JSON export
/// for IPC to the Python Logic Layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesVocabRegistry {
    /// Map of species name to configuration
    #[serde(flatten)]
    configs: HashMap<String, SpeciesVocabConfig>,
}

impl Default for SpeciesVocabRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeciesVocabRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
        }
    }

    /// Register a vocabulary configuration
    ///
    /// # Arguments
    /// * `config` - Configuration to register
    pub fn register(&mut self, config: SpeciesVocabConfig) {
        let species = config.species.clone();
        let optimal_k = config.optimal_k;
        self.configs.insert(species.clone(), config);
        log::info!("Registered vocab config for {}: k={}", species, optimal_k);
    }

    /// Get configuration for a species
    ///
    /// # Arguments
    /// * `species` - Species name
    ///
    /// # Returns
    /// * `Some(config)` if found, `None` otherwise
    pub fn get(&self, species: &str) -> Option<&SpeciesVocabConfig> {
        self.configs.get(species)
    }

    /// Get optimal k for a species, with default fallback
    ///
    /// # Arguments
    /// * `species` - Species name
    /// * `default` - Default k if not found
    ///
    /// # Returns
    /// Optimal k value
    pub fn get_optimal_k(&self, species: &str, default: usize) -> usize {
        self.get(species).map(|config| config.optimal_k).unwrap_or(default)
    }

    /// Check if registry has configuration for a species
    ///
    /// # Arguments
    /// * `species` - Species name
    pub fn has_species(&self, species: &str) -> bool {
        self.configs.contains_key(species)
    }

    /// List all species in the registry
    ///
    /// # Returns
    /// Vector of species names
    pub fn list_species(&self) -> Vec<String> {
        let mut species: Vec<_> = self.configs.keys().cloned().collect();
        species.sort();
        species
    }

    /// Get the number of registered species
    pub fn len(&self) -> usize {
        self.configs.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.configs.is_empty()
    }

    /// Remove a species configuration
    ///
    /// # Arguments
    /// * `species` - Species name to remove
    ///
    /// # Returns
    /// * `Some(config)` if removed, `None` if not found
    pub fn remove(&mut self, species: &str) -> Option<SpeciesVocabConfig> {
        self.configs.remove(species)
    }

    /// Export registry to JSON string
    ///
    /// # Returns
    /// JSON string representation
    pub fn export_to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(Into::into)
    }

    /// Import registry from JSON string
    ///
    /// # Arguments
    /// * `json` - JSON string to parse
    ///
    /// # Returns
    /// New registry instance
    pub fn import_from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(Into::into)
    }

    /// Save registry to a JSON file
    ///
    /// # Arguments
    /// * `path` - File path to save to
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = self.export_to_json()?;
        let path_ref = path.as_ref();
        std::fs::write(path_ref, json)?;
        log::info!("Saved registry with {} species to {:?}", self.len(), path_ref);
        Ok(())
    }

    /// Load registry from a JSON file
    ///
    /// # Arguments
    /// * `path` - File path to load from
    ///
    /// # Returns
    /// New registry instance
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        let mut file = File::open(path_ref)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let registry = Self::import_from_json(&contents)?;
        log::info!("Loaded registry with {} species from {:?}", registry.len(), path_ref);
        Ok(registry)
    }

    /// Merge another registry into this one
    ///
    /// Existing configurations are preserved, new ones are added.
    ///
    /// # Arguments
    /// * `other` - Registry to merge from
    pub fn merge(&mut self, other: SpeciesVocabRegistry) {
        for (species, config) in other.configs {
            self.configs.entry(species).or_insert(config);
        }
    }

    /// Get all configurations as a slice
    pub fn all_configs(&self) -> Vec<&SpeciesVocabConfig> {
        self.configs.values().collect()
    }

    /// Filter configurations by quality (SVS score threshold)
    ///
    /// # Arguments
    /// * `min_svs` - Minimum SVS score
    ///
    /// # Returns
    /// Vector of configs meeting the quality threshold
    pub fn filter_by_quality(&self, min_svs: f64) -> Vec<&SpeciesVocabConfig> {
        self.configs
            .values()
            .filter(|config| config.svs_score >= min_svs)
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Sprint 1.3: SpeciesVocabConfig Tests
    // =========================================================================

    #[test]
    fn test_create_vocab_config() {
        let config = SpeciesVocabConfig::new("egyptian_fruit_bat", 1020, 0.45);

        assert_eq!(config.species, "egyptian_fruit_bat");
        assert_eq!(config.optimal_k, 1020);
        assert_eq!(config.svs_score, 0.45);
        assert!(config.discovery_timestamp > 0);
    }

    #[test]
    fn test_vocab_config_with_timestamp() {
        let config = SpeciesVocabConfig::with_timestamp("test", 100, 0.5, 1234567890);

        assert_eq!(config.discovery_timestamp, 1234567890);
        assert_eq!(config.species, "test");
        assert_eq!(config.optimal_k, 100);
        assert_eq!(config.svs_score, 0.5);
    }

    #[test]
    fn test_vocab_config_is_recent() {
        let recent = SpeciesVocabConfig::new("test", 100, 0.5);
        assert!(recent.is_recent());

        let old = SpeciesVocabConfig::with_timestamp("test", 100, 0.5, 1234567890);
        assert!(!old.is_recent());
    }

    #[test]
    fn test_vocab_config_is_good_quality() {
        let good = SpeciesVocabConfig::new("test", 100, 0.5);
        assert!(good.is_good_quality());

        let bad = SpeciesVocabConfig::new("test", 100, 0.3);
        assert!(!bad.is_good_quality());
    }

    #[test]
    fn test_vocab_config_equality() {
        let config1 = SpeciesVocabConfig::with_timestamp("bat", 100, 0.5, 100);
        let config2 = SpeciesVocabConfig::with_timestamp("bat", 100, 0.5, 100);
        let config3 = SpeciesVocabConfig::with_timestamp("bat", 200, 0.5, 100);

        assert!(config1 == config2); // Same values
        assert!(config1 != config3); // Different optimal_k
    }

    // =========================================================================
    // Registry Tests
    // =========================================================================

    #[test]
    fn test_registry_new_is_empty() {
        let registry = SpeciesVocabRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_register_vocab_config() {
        let mut registry = SpeciesVocabRegistry::new();
        let config = SpeciesVocabConfig::new("bat", 1020, 0.45);

        registry.register(config);

        assert_eq!(registry.len(), 1);
        assert!(registry.has_species("bat"));
    }

    #[test]
    fn test_get_vocab_config() {
        let mut registry = SpeciesVocabRegistry::new();
        let config = SpeciesVocabConfig::new("bat", 1020, 0.45);

        registry.register(config.clone());

        let retrieved = registry.get("bat");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().optimal_k, 1020);
    }

    #[test]
    fn test_get_nonexistent_species() {
        let registry = SpeciesVocabRegistry::new();
        assert!(registry.get("unknown").is_none());
    }

    #[test]
    fn test_get_optimal_k_with_default() {
        let mut registry = SpeciesVocabRegistry::new();
        registry.register(SpeciesVocabConfig::new("bat", 1020, 0.45));

        assert_eq!(registry.get_optimal_k("bat", 500), 1020);
        assert_eq!(registry.get_optimal_k("unknown", 500), 500);
    }

    #[test]
    fn test_list_species() {
        let mut registry = SpeciesVocabRegistry::new();
        registry.register(SpeciesVocabConfig::new("zebra_finch", 500, 0.3));
        registry.register(SpeciesVocabConfig::new("bat", 1020, 0.45));
        registry.register(SpeciesVocabConfig::new("marmoset", 450, 0.5));

        let species = registry.list_species();
        assert_eq!(species.len(), 3);
        // Should be sorted
        assert_eq!(species, vec!["bat", "marmoset", "zebra_finch"]);
    }

    #[test]
    fn test_remove_species() {
        let mut registry = SpeciesVocabRegistry::new();
        registry.register(SpeciesVocabConfig::new("bat", 1020, 0.45));

        let removed = registry.remove("bat");
        assert!(removed.is_some());
        assert!(!registry.has_species("bat"));

        let removed_again = registry.remove("bat");
        assert!(removed_again.is_none());
    }

    #[test]
    fn test_export_to_json() {
        let mut registry = SpeciesVocabRegistry::new();
        registry.register(SpeciesVocabConfig::new("bat", 1020, 0.45));

        let json = registry.export_to_json().unwrap();
        assert!(json.contains("bat"));
        assert!(json.contains("1020"));
        assert!(json.contains("0.45"));
    }

    #[test]
    fn test_import_from_json() {
        let json = r#"{
            "bat": {
                "species": "bat",
                "optimal_k": 1020,
                "svs_score": 0.45,
                "discovery_timestamp": 1234567890
            }
        }"#;

        let registry = SpeciesVocabRegistry::import_from_json(json).unwrap();
        assert!(registry.has_species("bat"));
        assert_eq!(registry.get_optimal_k("bat", 500), 1020);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut registry1 = SpeciesVocabRegistry::new();
        registry1.register(SpeciesVocabConfig::new("bat", 1020, 0.45));
        registry1.register(SpeciesVocabConfig::new("marmoset", 450, 0.5));

        let json = registry1.export_to_json().unwrap();
        let registry2 = SpeciesVocabRegistry::import_from_json(&json).unwrap();

        assert_eq!(registry2.len(), 2);
        assert_eq!(registry2.get_optimal_k("bat", 0), 1020);
        assert_eq!(registry2.get_optimal_k("marmoset", 0), 450);
    }

    #[test]
    fn test_merge_registries() {
        let mut registry1 = SpeciesVocabRegistry::new();
        registry1.register(SpeciesVocabConfig::new("bat", 1020, 0.45));

        let mut registry2 = SpeciesVocabRegistry::new();
        registry2.register(SpeciesVocabConfig::new("marmoset", 450, 0.5));
        registry2.register(SpeciesVocabConfig::new("bat", 999, 0.3)); // Duplicate, should be ignored

        registry1.merge(registry2);

        assert_eq!(registry1.len(), 2);
        assert_eq!(registry1.get_optimal_k("bat", 0), 1020); // Original preserved
        assert_eq!(registry1.get_optimal_k("marmoset", 0), 450);
    }

    #[test]
    fn test_filter_by_quality() {
        let mut registry = SpeciesVocabRegistry::new();
        registry.register(SpeciesVocabConfig::new("good1", 100, 0.6));
        registry.register(SpeciesVocabConfig::new("good2", 200, 0.5));
        registry.register(SpeciesVocabConfig::new("bad", 300, 0.3));

        let good_configs = registry.filter_by_quality(0.5);
        assert_eq!(good_configs.len(), 2);
    }

    #[test]
    fn test_all_configs() {
        let mut registry = SpeciesVocabRegistry::new();
        registry.register(SpeciesVocabConfig::new("bat", 1020, 0.45));
        registry.register(SpeciesVocabConfig::new("marmoset", 450, 0.5));

        let configs = registry.all_configs();
        assert_eq!(configs.len(), 2);
    }
}
