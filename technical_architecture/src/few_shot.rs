//! Few-Shot Species Adaptation - The "Rosetta Transfer"
//! =====================================================
//!
//! Enables instant adaptation to new species using Prototypical Networks.
//! Because Rosetta-Net is pre-trained on 45D "Physics of Sound," it doesn't
//! need to learn what "Harmonicity" or "ICI" is from scratch.
//!
//! ## Key Insight
//! Calculate the "Prototype" (mean latent vector) of a few new recordings,
//! and use this as the new class center for instant adaptation.
//!
//! ## Usage
//! ```rust
//! use technical_architecture::PrototypicalAdapter;
//!
//! let adapter = PrototypicalAdapter::new(128);
//!
//! // Add new species with just 3 examples
//! adapter.add_species_prototype("gibbon", &[
//!     &latent_1, &latent_2, &latent_3
//! ]);
//!
//! // Now classify using prototype distance
//! let species = adapter.classify_few_shot(&new_latent);
//! ```

use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Configuration for few-shot adaptation
#[derive(Debug, Clone)]
pub struct FewShotConfig {
    /// Dimension of latent space
    pub latent_dim: usize,
    /// Distance metric to use
    pub distance_metric: DistanceMetric,
    /// Minimum examples to create prototype
    pub min_examples: usize,
    /// Temperature for softmax (lower = sharper)
    pub temperature: f32,
}

impl Default for FewShotConfig {
    fn default() -> Self {
        Self {
            latent_dim: 128,
            distance_metric: DistanceMetric::Euclidean,
            min_examples: 1,
            temperature: 1.0,
        }
    }
}

/// Distance metric for prototype comparison
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Standard Euclidean distance
    Euclidean,
    /// Cosine similarity (1 - cos(theta))
    Cosine,
    /// Squared Euclidean (faster, no sqrt)
    SquaredEuclidean,
}

/// A species prototype (mean latent vector)
#[derive(Debug, Clone)]
pub struct SpeciesPrototype {
    /// Species name
    pub name: String,
    /// Mean latent vector (prototype)
    pub prototype: Array1<f32>,
    /// Number of examples used
    pub num_examples: usize,
    /// Variance of examples (for uncertainty)
    pub variance: f32,
    /// All embeddings (for detailed analysis)
    pub embeddings: Vec<Array1<f32>>,
}

/// Result of few-shot classification
#[derive(Debug, Clone)]
pub struct FewShotResult {
    /// Predicted species
    pub species: String,
    /// Confidence (0.0-1.0)
    pub confidence: f32,
    /// Distance to predicted prototype
    pub distance: f32,
    /// All class probabilities
    pub probabilities: HashMap<String, f32>,
    /// Whether this is a confident prediction
    pub is_confident: bool,
}

/// Few-Shot Prototypical Adapter
///
/// Enables rapid learning of new species from just a few examples.
#[derive(Debug, Clone)]
pub struct PrototypicalAdapter {
    config: FewShotConfig,
    /// Species prototypes
    prototypes: HashMap<String, SpeciesPrototype>,
    /// Total number of samples
    total_samples: usize,
}

impl PrototypicalAdapter {
    /// Create a new prototypical adapter with default configuration
    pub fn new(latent_dim: usize) -> Self {
        Self::with_config(FewShotConfig {
            latent_dim,
            ..Default::default()
        })
    }

    /// Create an adapter with custom configuration
    pub fn with_config(config: FewShotConfig) -> Self {
        Self {
            config,
            prototypes: HashMap::new(),
            total_samples: 0,
        }
    }

    /// Add a species prototype from a few examples
    pub fn add_species_prototype(&mut self, species: &str, examples: &[Vec<f32>]) {
        if examples.is_empty() {
            return;
        }

        let dim = self.config.latent_dim;

        // Convert to Array1
        let embeddings: Vec<Array1<f32>> = examples
            .iter()
            .map(|e| {
                let mut arr = Array1::zeros(dim);
                for (i, &val) in e.iter().take(dim).enumerate() {
                    arr[i] = val;
                }
                arr
            })
            .collect();

        // Compute prototype (mean)
        let mut prototype = Array1::zeros(dim);
        for embedding in &embeddings {
            prototype += embedding;
        }
        prototype /= embeddings.len() as f32;

        // Compute variance
        let variance = if embeddings.len() > 1 {
            let var_sum: f32 = embeddings
                .iter()
                .map(|e| {
                    let diff = e - &prototype;
                    diff.mapv(|x| x * x).sum()
                })
                .sum();
            var_sum / embeddings.len() as f32
        } else {
            0.0
        };

        self.prototypes.insert(
            species.to_string(),
            SpeciesPrototype {
                name: species.to_string(),
                prototype,
                num_examples: examples.len(),
                variance,
                embeddings,
            },
        );

        self.total_samples += examples.len();
    }

    /// Update existing prototype with new examples
    pub fn update_prototype(&mut self, species: &str, new_examples: &[Vec<f32>]) {
        if let Some(existing) = self.prototypes.get_mut(species) {
            let dim = self.config.latent_dim;
            let old_count = existing.num_examples;
            let old_prototype = existing.prototype.clone();

            // Add new embeddings
            for example in new_examples {
                let mut arr = Array1::zeros(dim);
                for (i, &val) in example.iter().take(dim).enumerate() {
                    arr[i] = val;
                }
                existing.embeddings.push(arr);
            }

            // Update prototype with weighted average
            let new_count = existing.embeddings.len();
            let mut new_prototype = Array1::zeros(dim);

            for embedding in &existing.embeddings {
                new_prototype += embedding;
            }
            new_prototype /= new_count as f32;

            existing.prototype = new_prototype;
            existing.num_examples = new_count;

            self.total_samples += new_examples.len();
        } else {
            // Species doesn't exist, create new
            self.add_species_prototype(species, new_examples);
        }
    }

    /// Classify a latent vector using prototype distance
    pub fn classify_few_shot(&self, latent: &[f32]) -> FewShotResult {
        if self.prototypes.is_empty() {
            return FewShotResult {
                species: "unknown".to_string(),
                confidence: 0.0,
                distance: f32::INFINITY,
                probabilities: HashMap::new(),
                is_confident: false,
            };
        }

        let dim = self.config.latent_dim;
        let embedding = {
            let mut arr = Array1::zeros(dim);
            for (i, &val) in latent.iter().take(dim).enumerate() {
                arr[i] = val;
            }
            arr
        };

        // Compute distances to all prototypes
        let mut distances: Vec<(String, f32)> = self
            .prototypes
            .iter()
            .map(|(name, proto)| {
                let dist = self.compute_distance(&embedding, &proto.prototype);
                (name.clone(), dist)
            })
            .collect();

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Compute softmax probabilities
        let probabilities = self.compute_probabilities(&distances);

        // Get best match
        let (best_species, best_distance) = distances[0].clone();
        let best_prob = probabilities.get(&best_species).copied().unwrap_or(0.0);

        // Determine confidence
        let is_confident = best_prob > 0.5 && self.prototypes.len() > 1;

        FewShotResult {
            species: best_species,
            confidence: best_prob,
            distance: best_distance,
            probabilities,
            is_confident,
        }
    }

    /// Get top-k species predictions
    pub fn top_k(&self, latent: &[f32], k: usize) -> Vec<(String, f32)> {
        let result = self.classify_few_shot(latent);

        let mut probs: Vec<(String, f32)> = result.probabilities.into_iter().collect();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        probs.into_iter().take(k).collect()
    }

    /// Check if a latent vector is close to a specific species
    pub fn is_species(&self, latent: &[f32], species: &str, threshold: f32) -> bool {
        if let Some(proto) = self.prototypes.get(species) {
            let dim = self.config.latent_dim;
            let embedding = {
                let mut arr = Array1::zeros(dim);
                for (i, &val) in latent.iter().take(dim).enumerate() {
                    arr[i] = val;
                }
                arr
            };

            let dist = self.compute_distance(&embedding, &proto.prototype);
            dist < threshold
        } else {
            false
        }
    }

    /// Get all registered species
    pub fn registered_species(&self) -> Vec<String> {
        self.prototypes.keys().cloned().collect()
    }

    /// Get number of registered species
    pub fn num_species(&self) -> usize {
        self.prototypes.len()
    }

    /// Get total number of samples
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Get prototype for a species
    pub fn get_prototype(&self, species: &str) -> Option<&SpeciesPrototype> {
        self.prototypes.get(species)
    }

    /// Remove a species prototype
    pub fn remove_species(&mut self, species: &str) -> bool {
        if let Some(proto) = self.prototypes.remove(species) {
            self.total_samples -= proto.num_examples;
            true
        } else {
            false
        }
    }

    /// Clear all prototypes
    pub fn clear(&mut self) {
        self.prototypes.clear();
        self.total_samples = 0;
    }

    /// Compute distance between two vectors
    fn compute_distance(&self, a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        match self.config.distance_metric {
            DistanceMetric::Euclidean => (a - b).mapv(|x| x * x).sum().sqrt(),
            DistanceMetric::SquaredEuclidean => (a - b).mapv(|x| x * x).sum(),
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

                if norm_a > 1e-10 && norm_b > 1e-10 {
                    1.0 - dot / (norm_a * norm_b)
                } else {
                    1.0
                }
            }
        }
    }

    /// Compute softmax probabilities from distances
    fn compute_probabilities(&self, distances: &[(String, f32)]) -> HashMap<String, f32> {
        let temp = self.config.temperature;

        // Convert distances to similarities (negative distance)
        let similarities: Vec<(String, f32)> = distances
            .iter()
            .map(|(name, dist)| (name.clone(), -dist / temp))
            .collect();

        // Softmax
        let max_sim = similarities
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);

        let exp_sum: f32 = similarities.iter().map(|(_, s)| (s - max_sim).exp()).sum();

        let mut probs = HashMap::new();
        for (name, sim) in &similarities {
            let prob = (sim - max_sim).exp() / exp_sum;
            probs.insert(name.clone(), prob);
        }

        probs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = PrototypicalAdapter::new(128);
        assert_eq!(adapter.config.latent_dim, 128);
        assert_eq!(adapter.num_species(), 0);
    }

    #[test]
    fn test_add_species_prototype() {
        let mut adapter = PrototypicalAdapter::new(4);

        adapter.add_species_prototype(
            "species_a",
            &[vec![1.0, 0.0, 0.0, 0.0], vec![1.1, 0.1, 0.0, 0.0]],
        );

        assert_eq!(adapter.num_species(), 1);
        assert_eq!(adapter.total_samples(), 2);
    }

    #[test]
    fn test_add_empty_prototype() {
        let mut adapter = PrototypicalAdapter::new(4);

        adapter.add_species_prototype("empty", &[]);

        assert_eq!(adapter.num_species(), 0);
    }

    #[test]
    fn test_classify_few_shot_single() {
        let mut adapter = PrototypicalAdapter::new(4);
        adapter.add_species_prototype("species_a", &[vec![1.0, 0.0, 0.0, 0.0]]);

        let result = adapter.classify_few_shot(&[1.0, 0.0, 0.0, 0.0]);

        assert_eq!(result.species, "species_a");
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_classify_few_shot_multiple() {
        let mut adapter = PrototypicalAdapter::new(4);

        adapter.add_species_prototype("species_a", &[vec![1.0, 0.0, 0.0, 0.0]]);

        adapter.add_species_prototype("species_b", &[vec![0.0, 1.0, 0.0, 0.0]]);

        let result_a = adapter.classify_few_shot(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(result_a.species, "species_a");

        let result_b = adapter.classify_few_shot(&[0.0, 1.0, 0.0, 0.0]);
        assert_eq!(result_b.species, "species_b");
    }

    #[test]
    fn test_classify_empty_adapter() {
        let adapter = PrototypicalAdapter::new(4);
        let result = adapter.classify_few_shot(&[1.0, 0.0, 0.0, 0.0]);

        assert_eq!(result.species, "unknown");
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_top_k_predictions() {
        let mut adapter = PrototypicalAdapter::new(4);

        adapter.add_species_prototype("a", &[vec![1.0, 0.0, 0.0, 0.0]]);
        adapter.add_species_prototype("b", &[vec![0.0, 1.0, 0.0, 0.0]]);
        adapter.add_species_prototype("c", &[vec![0.0, 0.0, 1.0, 0.0]]);

        let top_2 = adapter.top_k(&[0.9, 0.1, 0.0, 0.0], 2);

        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0].0, "a"); // Closest to species_a
    }

    #[test]
    fn test_is_species() {
        let mut adapter = PrototypicalAdapter::new(4);
        adapter.add_species_prototype("species_a", &[vec![1.0, 0.0, 0.0, 0.0]]);

        assert!(adapter.is_species(&[1.0, 0.0, 0.0, 0.0], "species_a", 1.0));
        assert!(!adapter.is_species(&[100.0, 0.0, 0.0, 0.0], "species_a", 1.0));
        assert!(!adapter.is_species(&[1.0, 0.0, 0.0, 0.0], "unknown", 1.0));
    }

    #[test]
    fn test_update_prototype() {
        let mut adapter = PrototypicalAdapter::new(4);
        adapter.add_species_prototype("species_a", &[vec![1.0, 0.0, 0.0, 0.0]]);

        adapter.update_prototype("species_a", &[vec![1.0, 0.0, 0.0, 0.0]]);

        let proto = adapter.get_prototype("species_a").unwrap();
        assert_eq!(proto.num_examples, 2);
    }

    #[test]
    fn test_update_new_species() {
        let mut adapter = PrototypicalAdapter::new(4);
        adapter.add_species_prototype("species_a", &[vec![1.0, 0.0, 0.0, 0.0]]);

        // Update non-existent species creates it
        adapter.update_prototype("species_b", &[vec![0.0, 1.0, 0.0, 0.0]]);

        assert_eq!(adapter.num_species(), 2);
    }

    #[test]
    fn test_remove_species() {
        let mut adapter = PrototypicalAdapter::new(4);
        adapter.add_species_prototype("species_a", &[vec![1.0, 0.0, 0.0, 0.0]]);

        let removed = adapter.remove_species("species_a");
        assert!(removed);
        assert_eq!(adapter.num_species(), 0);

        let not_removed = adapter.remove_species("unknown");
        assert!(!not_removed);
    }

    #[test]
    fn test_clear() {
        let mut adapter = PrototypicalAdapter::new(4);
        adapter.add_species_prototype("a", &[vec![1.0, 0.0, 0.0, 0.0]]);
        adapter.add_species_prototype("b", &[vec![0.0, 1.0, 0.0, 0.0]]);

        adapter.clear();

        assert_eq!(adapter.num_species(), 0);
        assert_eq!(adapter.total_samples(), 0);
    }

    #[test]
    fn test_prototype_mean() {
        let mut adapter = PrototypicalAdapter::new(4);
        adapter.add_species_prototype(
            "test",
            &[vec![1.0, 0.0, 0.0, 0.0], vec![3.0, 0.0, 0.0, 0.0]],
        );

        let proto = adapter.get_prototype("test").unwrap();

        // Mean should be [2.0, 0.0, 0.0, 0.0]
        assert!((proto.prototype[0] - 2.0).abs() < 0.01);
        assert!((proto.prototype[1] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_distance_metrics() {
        let mut adapter = PrototypicalAdapter::with_config(FewShotConfig {
            latent_dim: 4,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        });

        adapter.add_species_prototype("a", &[vec![1.0, 0.0, 0.0, 0.0]]);

        let result = adapter.classify_few_shot(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(result.species, "a");

        // Same direction, different magnitude - cosine should match
        let result = adapter.classify_few_shot(&[10.0, 0.0, 0.0, 0.0]);
        assert_eq!(result.species, "a");
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_confidence_in_predictions() {
        let mut adapter = PrototypicalAdapter::new(4);

        // Single species
        adapter.add_species_prototype("only", &[vec![1.0, 0.0, 0.0, 0.0]]);

        let result = adapter.classify_few_shot(&[1.0, 0.0, 0.0, 0.0]);
        // With only one species, can't be confident
        assert!(!result.is_confident || result.confidence > 0.5);

        // Add second species
        adapter.add_species_prototype("other", &[vec![0.0, 0.0, 0.0, 1.0]]);

        let result = adapter.classify_few_shot(&[1.0, 0.0, 0.0, 0.0]);
        // Clear match to first species
        assert!(result.is_confident);
    }

    #[test]
    fn test_registered_species() {
        let mut adapter = PrototypicalAdapter::new(4);
        adapter.add_species_prototype("a", &[vec![1.0, 0.0, 0.0, 0.0]]);
        adapter.add_species_prototype("b", &[vec![0.0, 1.0, 0.0, 0.0]]);

        let species = adapter.registered_species();
        assert_eq!(species.len(), 2);
        assert!(species.contains(&"a".to_string()));
        assert!(species.contains(&"b".to_string()));
    }

    #[test]
    fn test_variance_computation() {
        let mut adapter = PrototypicalAdapter::new(4);

        // Low variance examples
        adapter.add_species_prototype(
            "low_var",
            &[
                vec![1.0, 0.0, 0.0, 0.0],
                vec![1.1, 0.0, 0.0, 0.0],
                vec![0.9, 0.0, 0.0, 0.0],
            ],
        );

        // High variance examples
        adapter.add_species_prototype(
            "high_var",
            &[
                vec![1.0, 0.0, 0.0, 0.0],
                vec![10.0, 0.0, 0.0, 0.0],
                vec![-5.0, 0.0, 0.0, 0.0],
            ],
        );

        let low_var_proto = adapter.get_prototype("low_var").unwrap();
        let high_var_proto = adapter.get_prototype("high_var").unwrap();

        assert!(high_var_proto.variance > low_var_proto.variance);
    }

    #[test]
    fn test_few_shot_with_3_examples() {
        let mut adapter = PrototypicalAdapter::new(4);

        // Simulate "gibbon" with just 3 examples
        adapter.add_species_prototype(
            "gibbon",
            &[
                vec![2.0, 1.0, 0.5, 0.0],
                vec![2.1, 1.1, 0.4, 0.1],
                vec![1.9, 0.9, 0.6, 0.0],
            ],
        );

        // Add existing species
        adapter.add_species_prototype("marmoset", &[vec![8.0, 7.0, 6.0, 5.0]]);

        // Test classification
        let result = adapter.classify_few_shot(&[2.0, 1.0, 0.5, 0.0]);

        assert_eq!(result.species, "gibbon");
        assert!(result.confidence > 0.5);
    }
}
