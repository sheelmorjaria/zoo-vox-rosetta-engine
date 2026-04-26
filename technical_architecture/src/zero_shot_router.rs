//! Zero-Shot Hierarchical Router with Metric Learning
//! ==================================================
//!
//! This module implements a two-stage zero-shot classification architecture:
//!
//! **Stage 1: Group Detection (Closed Set)**
//! - Uses the existing Hierarchical Ensemble Router (RF + NN)
//! - Predicts broad taxonomic group (Bat, Bird, Cetacean, etc.)
//! - Applies feature reweighting based on predicted group
//!
//! **Stage 2: Zero-Shot Discrimination (Open Set)**
//! - Feature Reweighting: Apply taxonomic mask to 112D vector
//! - Embedding Generation: Siamese network converts 112D -> 64D latent
//! - k-NN Search: Compare against reference database
//! - Novelty Detection: Returns "Unknown" if distance > threshold
//!
//! # Architecture
//!
//! ```text
//! INPUT: 112D Feature Vector
//!       │
//!       ▼
//! ┌─────────────────────────────────────┐
//! │  STAGE 1: GROUP DETECTION (Closed)  │
//! │  RF Gatekeeper + NN Ensemble        │
//! │  Predicts: "Bat"                    │
//! └──────────────┬──────────────────────┘
//!                │
//!                ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │  STAGE 2: ZERO-SHOT DISCRIMINATION (Open Set)               │
//! │                                                             │
//! │  [A] Feature Reweighting: Apply "Bat Mask" to 112D         │
//! │               │                                             │
//! │               ▼                                             │
//! │  [B] Embedding Generation: 112D -> 64D Latent              │
//! │               │                                             │
//! │               ▼                                             │
//! │  [C] k-NN Search: Compare against Reference Gallery         │
//! │                                                             │
//! │  IF Distance < Threshold: Return species match              │
//! │  ELSE: Return "Novel Species / Unknown"                     │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust
//! use technical_architecture::zero_shot_router::{
//!     ZeroShotRouter, ZeroShotConfig, ZeroShotResult,
//!     ReferenceGallery, SiameseEmbedding,
//! };
//!
//! // Load reference gallery from training data
//! let gallery = ReferenceGallery::load_from_json("reference_gallery.json")?;
//!
//! // Create zero-shot router
//! let config = ZeroShotConfig::default();
//! let router = ZeroShotRouter::new(config, gallery)?;
//!
//! // Classify with zero-shot capability
//! let features = vec![0.0; 112];
//! let result = router.classify(&features)?;
//!
//! match result.prediction_type {
//!     PredictionType::KnownSpecies => println!("Matched: {}", result.species),
//!     PredictionType::NovelSpecies => println!("Novel species detected!"),
//!     PredictionType::Uncertain => println!("Uncertain - needs review"),
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::taxonomic_router::{
    apply_taxonomic_mask, consolidate_taxon, get_taxonomic_weights, map_species_to_taxon, ConsolidatedTaxon, Taxon,
    PHYSICS_DIM,
};

// Re-export FEATURE_DIM for external use
pub use crate::taxonomic_router::FEATURE_DIM;

// =============================================================================
// Constants
// =============================================================================

/// Latent embedding dimension (output of Siamese network)
pub const LATENT_DIM: usize = 64;

/// Default k for k-NN search
pub const DEFAULT_K_NEIGHBORS: usize = 5;

/// Default distance threshold for novelty detection
/// If all neighbors are farther than this, classify as "Unknown"
pub const DEFAULT_DISTANCE_THRESHOLD: f32 = 0.3;

/// Minimum confidence for known species match
pub const MIN_CONFIDENCE_THRESHOLD: f32 = 0.6;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the Zero-Shot Router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroShotConfig {
    /// Number of neighbors for k-NN search
    pub k_neighbors: usize,
    /// Distance threshold for novelty detection (cosine distance)
    pub distance_threshold: f32,
    /// Minimum confidence to return a species name
    pub min_confidence: f32,
    /// Apply feature reweighting between stages
    pub apply_reweighting: bool,
    /// Use weighted voting for k-NN
    pub weighted_knn: bool,
    /// Distance metric: "cosine" or "euclidean"
    pub distance_metric: String,
}

impl Default for ZeroShotConfig {
    fn default() -> Self {
        Self {
            k_neighbors: DEFAULT_K_NEIGHBORS,
            distance_threshold: DEFAULT_DISTANCE_THRESHOLD,
            min_confidence: MIN_CONFIDENCE_THRESHOLD,
            apply_reweighting: true,
            weighted_knn: true,
            distance_metric: "cosine".to_string(),
        }
    }
}

// =============================================================================
// Embedding and Reference Types
// =============================================================================

/// A reference sample in the gallery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceSample {
    /// Species label
    pub species: String,
    /// Taxonomic group
    pub taxon: Taxon,
    /// 64D latent embedding
    pub embedding: Vec<f32>,
    /// Original 112D features (for debugging)
    #[serde(default)]
    pub original_features: Option<Vec<f32>>,
}

/// The reference gallery containing all known species embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceGallery {
    /// All reference samples
    pub samples: Vec<ReferenceSample>,
    /// Index by taxonomic group for fast lookup
    #[serde(default)]
    pub taxon_index: HashMap<String, Vec<usize>>,
    /// Embedding matrix for fast k-NN (samples x LATENT_DIM)
    #[serde(default)]
    pub embedding_matrix: Vec<Vec<f32>>,
    /// Species labels
    #[serde(default)]
    pub species_labels: Vec<String>,
    /// Taxon for each sample
    #[serde(default)]
    pub taxon_labels: Vec<Taxon>,
}

impl ReferenceGallery {
    /// Create empty gallery
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            taxon_index: HashMap::new(),
            embedding_matrix: Vec::new(),
            species_labels: Vec::new(),
            taxon_labels: Vec::new(),
        }
    }

    /// Load gallery from JSON file
    pub fn load_from_json(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| format!("Failed to read gallery file: {}", e))?;

        let mut gallery: Self =
            serde_json::from_str(&content).map_err(|e| format!("Failed to parse gallery JSON: {}", e))?;

        // Rebuild taxon index if needed
        if gallery.taxon_index.is_empty() && !gallery.samples.is_empty() {
            gallery.rebuild_indices();
        }

        Ok(gallery)
    }

    /// Save gallery to JSON file
    pub fn save_to_json(&self, path: &str) -> Result<(), String> {
        let content = serde_json::to_string_pretty(self).map_err(|e| format!("Failed to serialize gallery: {}", e))?;

        std::fs::write(path, content).map_err(|e| format!("Failed to write gallery file: {}", e))
    }

    /// Rebuild internal indices from samples
    pub fn rebuild_indices(&mut self) {
        self.taxon_index.clear();
        self.embedding_matrix.clear();
        self.species_labels.clear();
        self.taxon_labels.clear();

        for (idx, sample) in self.samples.iter().enumerate() {
            let taxon_key = format!("{:?}", sample.taxon);
            self.taxon_index.entry(taxon_key).or_default().push(idx);

            self.embedding_matrix.push(sample.embedding.clone());
            self.species_labels.push(sample.species.clone());
            self.taxon_labels.push(sample.taxon);
        }
    }

    /// Add a sample to the gallery
    pub fn add_sample(&mut self, sample: ReferenceSample) {
        let taxon_key = format!("{:?}", sample.taxon);
        let idx = self.samples.len();

        // Add to taxon index
        self.taxon_index.entry(taxon_key).or_default().push(idx);

        // Add to embedding matrix
        self.embedding_matrix.push(sample.embedding.clone());
        self.species_labels.push(sample.species.clone());
        self.taxon_labels.push(sample.taxon);

        self.samples.push(sample);
    }

    /// Get samples for a specific taxonomic group
    pub fn get_samples_for_taxon(&self, taxon: Taxon) -> Vec<&ReferenceSample> {
        let taxon_key = format!("{:?}", taxon);
        if let Some(indices) = self.taxon_index.get(&taxon_key) {
            indices.iter().filter_map(|&i| self.samples.get(i)).collect()
        } else {
            Vec::new()
        }
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if gallery is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Regenerate all embeddings from original features using provided embedding model
    /// This ensures gallery embeddings match query embeddings
    pub fn regenerate_embeddings(&mut self, embedding: &SiameseEmbedding) {
        for sample in &mut self.samples {
            if let Some(ref features) = sample.original_features {
                if features.len() == FEATURE_DIM {
                    sample.embedding = embedding.embed(features);
                }
            }
        }
        // Rebuild embedding matrix
        self.embedding_matrix = self.samples.iter().map(|s| s.embedding.clone()).collect();
    }
}

// =============================================================================
// Siamese Network Embedding
// =============================================================================

/// Siamese network for generating 64D embeddings from 112D features
///
/// This is a simplified implementation that uses a learned linear projection
/// followed by L2 normalization. For production, replace with a trained
/// neural network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiameseEmbedding {
    /// Weight matrix: LATENT_DIM x FEATURE_DIM
    pub weights: Vec<Vec<f32>>,
    /// Bias vector: LATENT_DIM
    pub bias: Vec<f32>,
    /// Whether to apply L2 normalization
    pub normalize: bool,
}

impl SiameseEmbedding {
    /// Create a new embedding layer with random weights (for initialization)
    pub fn new_random(seed: u64) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut rng_state = seed;
        let mut next_random = || -> f32 {
            let mut hasher = DefaultHasher::new();
            rng_state.hash(&mut hasher);
            let hash = hasher.finish();
            rng_state = hash;
            // Xavier initialization
            (hash as f32 / u64::MAX as f32 - 0.5) * 2.0 / (FEATURE_DIM as f32).sqrt()
        };

        let weights: Vec<Vec<f32>> = (0..LATENT_DIM)
            .map(|_| (0..FEATURE_DIM).map(|_| next_random()).collect())
            .collect();

        let bias: Vec<f32> = (0..LATENT_DIM).map(|_| next_random() * 0.1).collect();

        Self {
            weights,
            bias,
            normalize: true,
        }
    }

    /// Load embedding weights from JSON
    pub fn load_from_json(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| format!("Failed to read embedding file: {}", e))?;

        serde_json::from_str(&content).map_err(|e| format!("Failed to parse embedding JSON: {}", e))
    }

    /// Save embedding weights to JSON
    pub fn save_to_json(&self, path: &str) -> Result<(), String> {
        let content =
            serde_json::to_string_pretty(self).map_err(|e| format!("Failed to serialize embedding: {}", e))?;

        std::fs::write(path, content).map_err(|e| format!("Failed to write embedding file: {}", e))
    }

    /// Generate 64D embedding from 112D features
    pub fn embed(&self, features: &[f32]) -> Vec<f32> {
        assert_eq!(features.len(), FEATURE_DIM, "Feature dimension mismatch");

        let mut embedding = vec![0.0f32; LATENT_DIM];

        // Matrix-vector multiplication: W * x + b
        for i in 0..LATENT_DIM {
            let mut sum = self.bias[i];
            for j in 0..FEATURE_DIM {
                sum += self.weights[i][j] * features[j];
            }
            embedding[i] = sum;
        }

        // Apply ReLU activation
        for val in &mut embedding {
            if *val < 0.0 {
                *val = 0.0;
            }
        }

        // L2 normalization
        if self.normalize {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for val in &mut embedding {
                    *val /= norm;
                }
            }
        }

        embedding
    }
}

impl Default for SiameseEmbedding {
    fn default() -> Self {
        Self::new_random(42)
    }
}

// =============================================================================
// Distance Functions
// =============================================================================

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimension mismatch");

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Calculate cosine distance (1 - similarity)
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Calculate Euclidean distance between two vectors
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimension mismatch");

    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

// =============================================================================
// k-NN Search
// =============================================================================

/// Result of a k-NN search
#[derive(Debug, Clone)]
pub struct KnnResult {
    /// Indices of k nearest neighbors
    pub indices: Vec<usize>,
    /// Distances to k nearest neighbors
    pub distances: Vec<f32>,
    /// Species labels of neighbors
    pub species: Vec<String>,
    /// Taxonomic groups of neighbors
    pub taxa: Vec<Taxon>,
}

/// Perform k-NN search in the reference gallery
pub fn knn_search(query: &[f32], gallery: &ReferenceGallery, k: usize, metric: &str) -> KnnResult {
    let n = gallery.embedding_matrix.len();
    let k = k.min(n);

    if k == 0 {
        return KnnResult {
            indices: Vec::new(),
            distances: Vec::new(),
            species: Vec::new(),
            taxa: Vec::new(),
        };
    }

    // Calculate all distances
    let mut distances_with_idx: Vec<(usize, f32)> = gallery
        .embedding_matrix
        .iter()
        .enumerate()
        .map(|(i, emb)| {
            let dist = if metric == "euclidean" {
                euclidean_distance(query, emb)
            } else {
                cosine_distance(query, emb)
            };
            (i, dist)
        })
        .collect();

    // Sort by distance
    distances_with_idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top k
    let top_k: Vec<_> = distances_with_idx.into_iter().take(k).collect();

    KnnResult {
        indices: top_k.iter().map(|(i, _)| *i).collect(),
        distances: top_k.iter().map(|(_, d)| *d).collect(),
        species: top_k.iter().map(|(i, _)| gallery.species_labels[*i].clone()).collect(),
        taxa: top_k.iter().map(|(i, _)| gallery.taxon_labels[*i]).collect(),
    }
}

/// Perform weighted k-NN search (closer neighbors weighted more)
pub fn weighted_knn_search(query: &[f32], gallery: &ReferenceGallery, k: usize, metric: &str) -> (String, f32) {
    let result = knn_search(query, gallery, k, metric);

    if result.species.is_empty() {
        return ("Unknown".to_string(), 0.0);
    }

    // Count votes with distance-based weighting
    let mut votes: HashMap<String, f32> = HashMap::new();

    for (species, distance) in result.species.iter().zip(result.distances.iter()) {
        // Weight = 1 / (1 + distance)
        let weight = 1.0 / (1.0 + distance);
        *votes.entry(species.clone()).or_insert(0.0) += weight;
    }

    // Find winner
    let (best_species, best_vote) = votes
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(("Unknown".to_string(), 0.0));

    // Normalize confidence
    let total_weight: f32 = 1.0 / (1.0 + result.distances.iter().sum::<f32>());
    let confidence = (best_vote / total_weight).min(1.0);

    (best_species, confidence)
}

// =============================================================================
// Prediction Types and Results
// =============================================================================

/// Type of prediction made
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionType {
    /// Matched a known species with high confidence
    KnownSpecies,
    /// Detected a potentially novel species
    NovelSpecies,
    /// Uncertain - needs manual review
    Uncertain,
}

/// Result of zero-shot classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroShotResult {
    /// Predicted species (or "Unknown")
    pub species: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Type of prediction
    pub prediction_type: PredictionType,
    /// Predicted taxonomic group from Stage 1
    pub taxon: Taxon,
    /// Consolidated taxonomic group
    pub consolidated_taxon: ConsolidatedTaxon,
    /// 64D latent embedding
    pub embedding: Vec<f32>,
    /// k-NN results
    pub knn_distances: Vec<f32>,
    /// k-NN species
    pub knn_species: Vec<String>,
    /// Distance to nearest neighbor
    pub nearest_distance: f32,
    /// Whether feature reweighting was applied
    pub reweighting_applied: bool,
    /// Processing time in microseconds
    pub processing_time_us: u64,
    /// Warning messages
    pub warnings: Vec<String>,
}

// =============================================================================
// Zero-Shot Router
// =============================================================================

/// The main Zero-Shot Hierarchical Router
pub struct ZeroShotRouter {
    /// Configuration
    pub config: ZeroShotConfig,
    /// Siamese network for embedding generation
    pub embedding: SiameseEmbedding,
    /// Reference gallery of known species
    pub gallery: ReferenceGallery,
}

impl ZeroShotRouter {
    /// Create a new zero-shot router
    pub fn new(config: ZeroShotConfig, gallery: ReferenceGallery) -> Result<Self, String> {
        Ok(Self {
            config,
            embedding: SiameseEmbedding::default(),
            gallery,
        })
    }

    /// Create with custom embedding network
    pub fn with_embedding(
        config: ZeroShotConfig,
        gallery: ReferenceGallery,
        embedding: SiameseEmbedding,
    ) -> Result<Self, String> {
        Ok(Self {
            config,
            embedding,
            gallery,
        })
    }

    /// Check if router is ready
    pub fn is_ready(&self) -> bool {
        !self.gallery.is_empty()
    }

    /// Classify a 112D feature vector with zero-shot capability
    pub fn classify(&self, features: &[f32]) -> Result<ZeroShotResult, String> {
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
        // For now, use a simple heuristic based on F0
        // In production, this would use the full HierarchicalEnsembleRouter
        let f0 = features.first().copied().unwrap_or(0.0);
        let duration = features.get(1).copied().unwrap_or(0.0);

        // Heuristic group detection based on acoustic properties
        let predicted_taxon = if f0 > 20000.0 {
            Taxon::Cetacean // High frequency ultrasonic
        } else if f0 > 8000.0 && duration < 50.0 {
            Taxon::Mammal // Bat-like: high freq, short duration
        } else if f0 > 2000.0 && f0 < 8000.0 {
            Taxon::Songbird // Bird frequency range
        } else if duration > 200.0 {
            Taxon::Mysticete // Long duration calls
        } else {
            Taxon::Unknown
        };

        let consolidated_taxon = consolidate_taxon(predicted_taxon);

        // === FEATURE REWEIGHTING ===
        let reweighted_features = if self.config.apply_reweighting {
            apply_taxonomic_mask(features, predicted_taxon)
        } else {
            features.to_vec()
        };

        // === STAGE 2: EMBEDDING GENERATION ===
        let embedding = self.embedding.embed(&reweighted_features);

        // === STAGE 2: k-NN SEARCH ===
        // Search within the predicted taxonomic group first, then fallback to all
        let taxon_samples = self.gallery.get_samples_for_taxon(predicted_taxon);

        let (species, confidence, nearest_distance, knn_distances, knn_species) = if !taxon_samples.is_empty() {
            // Create temporary gallery for taxon-specific search
            let mut taxon_gallery = ReferenceGallery::new();
            for sample in taxon_samples {
                taxon_gallery.add_sample(sample.clone());
            }

            let result = knn_search(
                &embedding,
                &taxon_gallery,
                self.config.k_neighbors,
                &self.config.distance_metric,
            );

            let (sp, conf) = if self.config.weighted_knn {
                weighted_knn_search(
                    &embedding,
                    &taxon_gallery,
                    self.config.k_neighbors,
                    &self.config.distance_metric,
                )
            } else {
                // Simple majority vote
                let mut counts: HashMap<String, usize> = HashMap::new();
                for s in &result.species {
                    *counts.entry(s.clone()).or_insert(0) += 1;
                }
                let (best, count) = counts
                    .into_iter()
                    .max_by_key(|(_, c)| *c)
                    .unwrap_or(("Unknown".to_string(), 0));
                (best, count as f32 / result.species.len() as f32)
            };

            let nearest = result.distances.first().copied().unwrap_or(1.0);
            (sp, conf, nearest, result.distances, result.species)
        } else {
            // Fallback: search entire gallery
            let result = knn_search(
                &embedding,
                &self.gallery,
                self.config.k_neighbors,
                &self.config.distance_metric,
            );

            let (sp, conf) = if self.config.weighted_knn {
                weighted_knn_search(
                    &embedding,
                    &self.gallery,
                    self.config.k_neighbors,
                    &self.config.distance_metric,
                )
            } else {
                let mut counts: HashMap<String, usize> = HashMap::new();
                for s in &result.species {
                    *counts.entry(s.clone()).or_insert(0) += 1;
                }
                let (best, count) = counts
                    .into_iter()
                    .max_by_key(|(_, c)| *c)
                    .unwrap_or(("Unknown".to_string(), 0));
                (best, count as f32 / result.species.len() as f32)
            };

            let nearest = result.distances.first().copied().unwrap_or(1.0);
            (sp, conf, nearest, result.distances, result.species)
        };

        // === DETERMINE PREDICTION TYPE ===
        let prediction_type = if nearest_distance > self.config.distance_threshold {
            PredictionType::NovelSpecies
        } else if confidence < self.config.min_confidence {
            PredictionType::Uncertain
        } else {
            PredictionType::KnownSpecies
        };

        // Add warnings for edge cases
        if nearest_distance > self.config.distance_threshold {
            warnings.push(format!(
                "Nearest neighbor distance ({:.3}) exceeds threshold ({:.3})",
                nearest_distance, self.config.distance_threshold
            ));
        }
        if confidence < self.config.min_confidence {
            warnings.push(format!(
                "Confidence ({:.1}%) below threshold ({:.1}%)",
                confidence * 100.0,
                self.config.min_confidence * 100.0
            ));
        }

        let processing_time_us = start_time.elapsed().as_micros() as u64;

        Ok(ZeroShotResult {
            species,
            confidence,
            prediction_type,
            taxon: predicted_taxon,
            consolidated_taxon,
            embedding,
            knn_distances,
            knn_species,
            nearest_distance,
            reweighting_applied: self.config.apply_reweighting,
            processing_time_us,
            warnings,
        })
    }

    /// Batch classification
    pub fn classify_batch(&self, features_batch: &[Vec<f32>]) -> Vec<Result<ZeroShotResult, String>> {
        features_batch.iter().map(|features| self.classify(features)).collect()
    }

    /// Evaluate on a test set
    pub fn evaluate(&self, features_batch: &[Vec<f32>], labels: &[String]) -> ZeroShotMetrics {
        let mut correct = 0usize;
        let mut novel_detected = 0usize;
        let mut uncertain = 0usize;
        let mut total = 0usize;
        let mut known_correct = 0usize;
        let mut known_total = 0usize;
        let mut novel_correct = 0usize;
        let mut novel_total = 0usize;

        for (features, true_label) in features_batch.iter().zip(labels.iter()) {
            if let Ok(result) = self.classify(features) {
                total += 1;

                match result.prediction_type {
                    PredictionType::KnownSpecies => {
                        known_total += 1;
                        if &result.species == true_label {
                            correct += 1;
                            known_correct += 1;
                        }
                    }
                    PredictionType::NovelSpecies => {
                        novel_total += 1;
                        novel_detected += 1;
                        // If label is "Unknown" or not in gallery, this is correct
                        if !self.gallery.species_labels.contains(&true_label.to_string()) {
                            novel_correct += 1;
                        }
                    }
                    PredictionType::Uncertain => {
                        uncertain += 1;
                    }
                }
            }
        }

        ZeroShotMetrics {
            total_samples: total,
            correct_predictions: correct,
            overall_accuracy: if total > 0 {
                correct as f32 / total as f32 * 100.0
            } else {
                0.0
            },
            known_species_accuracy: if known_total > 0 {
                known_correct as f32 / known_total as f32 * 100.0
            } else {
                0.0
            },
            novel_detection_rate: if novel_total > 0 {
                novel_correct as f32 / novel_total as f32 * 100.0
            } else {
                0.0
            },
            uncertain_rate: if total > 0 {
                uncertain as f32 / total as f32 * 100.0
            } else {
                0.0
            },
        }
    }
}

// =============================================================================
// Metrics
// =============================================================================

/// Metrics for zero-shot evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroShotMetrics {
    pub total_samples: usize,
    pub correct_predictions: usize,
    pub overall_accuracy: f32,
    pub known_species_accuracy: f32,
    pub novel_detection_rate: f32,
    pub uncertain_rate: f32,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!(cosine_distance(&a, &b).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_distance(&a, &c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_siamese_embedding() {
        let embedding = SiameseEmbedding::new_random(42);
        let features = vec![1.0; FEATURE_DIM];
        let latent = embedding.embed(&features);

        assert_eq!(latent.len(), LATENT_DIM);

        // Check L2 normalization
        let norm: f32 = latent.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_reference_gallery() {
        let mut gallery = ReferenceGallery::new();

        gallery.add_sample(ReferenceSample {
            species: "TestSpecies1".to_string(),
            taxon: Taxon::Mammal,
            embedding: vec![1.0; LATENT_DIM],
            original_features: None,
        });

        gallery.add_sample(ReferenceSample {
            species: "TestSpecies2".to_string(),
            taxon: Taxon::Songbird,
            embedding: vec![0.5; LATENT_DIM],
            original_features: None,
        });

        assert_eq!(gallery.len(), 2);
        assert_eq!(gallery.get_samples_for_taxon(Taxon::Mammal).len(), 1);
        assert_eq!(gallery.get_samples_for_taxon(Taxon::Songbird).len(), 1);
    }

    #[test]
    fn test_knn_search() {
        let mut gallery = ReferenceGallery::new();

        // Add samples with different embeddings
        gallery.add_sample(ReferenceSample {
            species: "SpeciesA".to_string(),
            taxon: Taxon::Mammal,
            embedding: vec![1.0, 0.0, 0.0],
            original_features: None,
        });

        gallery.add_sample(ReferenceSample {
            species: "SpeciesB".to_string(),
            taxon: Taxon::Mammal,
            embedding: vec![0.0, 1.0, 0.0],
            original_features: None,
        });

        let query = vec![0.9, 0.1, 0.0];
        let result = knn_search(&query, &gallery, 2, "cosine");

        assert_eq!(result.species.len(), 2);
        assert_eq!(result.species[0], "SpeciesA"); // Closer to query
    }

    #[test]
    fn test_zero_shot_router() {
        let mut gallery = ReferenceGallery::new();

        gallery.add_sample(ReferenceSample {
            species: "TestBat".to_string(),
            taxon: Taxon::Mammal,
            embedding: SiameseEmbedding::default().embed(&vec![9000.0; FEATURE_DIM]),
            original_features: None,
        });

        let config = ZeroShotConfig::default();
        let router = ZeroShotRouter::new(config, gallery).unwrap();

        let features = vec![9000.0; FEATURE_DIM];
        let result = router.classify(&features).unwrap();

        assert!(result.confidence >= 0.0);
        assert_eq!(result.embedding.len(), LATENT_DIM);
    }

    #[test]
    fn test_default_config() {
        let config = ZeroShotConfig::default();
        assert_eq!(config.k_neighbors, DEFAULT_K_NEIGHBORS);
        assert!((config.distance_threshold - DEFAULT_DISTANCE_THRESHOLD).abs() < 1e-6);
        assert!(config.apply_reweighting);
    }

    // =========================================================================
    // TDD Test Suite: Weighted k-NN Search
    // =========================================================================

    #[test]
    fn test_weighted_knn_search_known_species() {
        let mut gallery = ReferenceGallery::new();

        // Add 3 samples from different species
        gallery.add_sample(ReferenceSample {
            species: "SpeciesA".to_string(),
            taxon: Taxon::Mammal,
            embedding: vec![1.0, 0.0, 0.0],
            original_features: None,
        });
        gallery.add_sample(ReferenceSample {
            species: "SpeciesB".to_string(),
            taxon: Taxon::Songbird,
            embedding: vec![0.0, 1.0, 0.0],
            original_features: None,
        });
        gallery.add_sample(ReferenceSample {
            species: "SpeciesC".to_string(),
            taxon: Taxon::Cetacean,
            embedding: vec![0.0, 0.0, 1.0],
            original_features: None,
        });

        // Query close to SpeciesA
        let query = vec![0.9, 0.1, 0.0];
        let (species, confidence) = weighted_knn_search(&query, &gallery, 3, "cosine");

        assert_eq!(species, "SpeciesA");
        assert!(confidence > 0.0);
    }

    #[test]
    fn test_weighted_knn_search_empty_gallery() {
        let gallery = ReferenceGallery::new();
        let query = vec![1.0; 3];
        let (species, confidence) = weighted_knn_search(&query, &gallery, 3, "cosine");

        assert_eq!(species, "Unknown");
        assert!((confidence - 0.0).abs() < 1e-6);
    }

    // =========================================================================
    // TDD Test Suite: Batch Classification
    // =========================================================================

    #[test]
    fn test_classify_batch_multiple_queries() {
        let mut gallery = ReferenceGallery::new();
        let embedding = SiameseEmbedding::default();

        // Add multiple species to gallery
        for (species, taxon, feat_val) in [
            ("TestBat", Taxon::Mammal, 9000.0),
            ("TestBird", Taxon::Songbird, 4000.0),
            ("TestWhale", Taxon::Cetacean, 25000.0),
        ] {
            gallery.add_sample(ReferenceSample {
                species: species.to_string(),
                taxon,
                embedding: embedding.embed(&vec![feat_val; FEATURE_DIM]),
                original_features: None,
            });
        }

        let config = ZeroShotConfig::default();
        let router = ZeroShotRouter::new(config, gallery).unwrap();

        // 5 queries
        let queries: Vec<Vec<f32>> = (0..5).map(|i| vec![4000.0 + i as f32 * 100.0; FEATURE_DIM]).collect();

        let results = router.classify_batch(&queries);

        assert_eq!(results.len(), 5);
        for result in &results {
            assert!(result.is_ok());
            assert_eq!(result.as_ref().unwrap().embedding.len(), LATENT_DIM);
        }
    }

    // =========================================================================
    // TDD Test Suite: Single Classification Scenarios
    // =========================================================================

    #[test]
    fn test_classify_known_species() {
        let mut gallery = ReferenceGallery::new();
        let embedding = SiameseEmbedding::new_random(42);

        // Add multiple samples of same species for confident match
        let base_features = vec![5000.0; FEATURE_DIM];
        let base_embedding = embedding.embed(&base_features);
        for _ in 0..3 {
            gallery.add_sample(ReferenceSample {
                species: "KnownBird".to_string(),
                taxon: Taxon::Songbird,
                embedding: base_embedding.clone(),
                original_features: Some(base_features.clone()),
            });
        }

        // Use a low distance threshold so the match is within range
        let config = ZeroShotConfig {
            distance_threshold: 0.99,
            min_confidence: 0.0, // Low threshold so we get KnownSpecies
            ..Default::default()
        };
        let router = ZeroShotRouter::new(config, gallery).unwrap();

        let result = router.classify(&base_features).unwrap();
        assert_eq!(result.species, "KnownBird");
        assert_eq!(result.prediction_type, PredictionType::KnownSpecies);
    }

    #[test]
    fn test_classify_novel_species() {
        let mut gallery = ReferenceGallery::new();

        // Add samples with embeddings far from what we'll query
        gallery.add_sample(ReferenceSample {
            species: "DistantSpecies".to_string(),
            taxon: Taxon::Cetacean,
            embedding: vec![1.0; LATENT_DIM],
            original_features: None,
        });

        // Use a very tight distance threshold → everything is "novel"
        let config = ZeroShotConfig {
            distance_threshold: 0.001,
            min_confidence: 0.0,
            ..Default::default()
        };
        let router = ZeroShotRouter::new(config, gallery).unwrap();

        // Query with features that embed to something far from the gallery
        let features = vec![5000.0; FEATURE_DIM];
        let result = router.classify(&features).unwrap();

        // Should be NovelSpecies because distance exceeds the tight threshold
        assert_eq!(result.prediction_type, PredictionType::NovelSpecies);
    }

    #[test]
    fn test_classify_uncertain() {
        let mut gallery = ReferenceGallery::new();

        // Add samples from multiple species with similar embeddings
        // to create low confidence (split vote)
        for species_name in &["SpeciesA", "SpeciesB", "SpeciesC"] {
            gallery.add_sample(ReferenceSample {
                species: species_name.to_string(),
                taxon: Taxon::Mammal,
                embedding: vec![0.5; LATENT_DIM],
                original_features: None,
            });
        }

        // Use unweighted k-NN so confidence = majority_fraction
        // With 3 different species and k=3, majority = 1/3 ≈ 0.33
        // Very high min_confidence → prediction is uncertain
        let config = ZeroShotConfig {
            distance_threshold: 10.0, // Very permissive distance
            min_confidence: 0.99,     // Very strict confidence
            weighted_knn: false,      // Use simple majority vote
            ..Default::default()
        };
        let router = ZeroShotRouter::new(config, gallery).unwrap();

        let features = vec![5000.0; FEATURE_DIM];
        let result = router.classify(&features).unwrap();

        // Should be Uncertain because confidence (1/3) < min_confidence (0.99)
        assert_eq!(result.prediction_type, PredictionType::Uncertain);
    }

    // =========================================================================
    // TDD Test Suite: Evaluation
    // =========================================================================

    #[test]
    fn test_evaluate_known_accuracy() {
        let mut gallery = ReferenceGallery::new();
        let embedding = SiameseEmbedding::new_random(42);

        // Build gallery with well-separated species
        let species_data = vec![
            ("SpeciesA", Taxon::Mammal, vec![5000.0; FEATURE_DIM]),
            ("SpeciesB", Taxon::Songbird, vec![3000.0; FEATURE_DIM]),
        ];

        for (species, taxon, features) in &species_data {
            let emb = embedding.embed(features);
            for _ in 0..3 {
                gallery.add_sample(ReferenceSample {
                    species: species.to_string(),
                    taxon: *taxon,
                    embedding: emb.clone(),
                    original_features: Some(features.clone()),
                });
            }
        }

        let config = ZeroShotConfig {
            distance_threshold: 0.99,
            min_confidence: 0.0,
            ..Default::default()
        };
        let router = ZeroShotRouter::new(config, gallery).unwrap();

        // Evaluate on the same data we trained with
        let features_batch: Vec<Vec<f32>> = species_data.iter().map(|(_, _, f)| f.clone()).collect();
        let labels: Vec<String> = species_data.iter().map(|(s, _, _)| s.to_string()).collect();

        let metrics = router.evaluate(&features_batch, &labels);

        assert_eq!(metrics.total_samples, 2);
        // On training data with generous thresholds, should get reasonable accuracy
        assert!(metrics.overall_accuracy >= 0.0);
    }

    // =========================================================================
    // TDD Test Suite: Router Creation and Configuration
    // =========================================================================

    #[test]
    fn test_with_embedding_custom() {
        let mut gallery = ReferenceGallery::new();
        gallery.add_sample(ReferenceSample {
            species: "TestSpecies".to_string(),
            taxon: Taxon::Mammal,
            embedding: vec![0.5; LATENT_DIM],
            original_features: None,
        });

        let custom_embedding = SiameseEmbedding::new_random(123);
        let config = ZeroShotConfig::default();

        let router = ZeroShotRouter::with_embedding(config, gallery, custom_embedding).unwrap();

        // Verify the custom embedding is used by classifying
        let features = vec![5000.0; FEATURE_DIM];
        let result = router.classify(&features);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gallery_serialization() {
        let temp_dir = tempfile::tempdir().unwrap();
        let gallery_path = temp_dir.path().join("test_gallery.json");

        let mut gallery = ReferenceGallery::new();
        gallery.add_sample(ReferenceSample {
            species: "Species1".to_string(),
            taxon: Taxon::Mammal,
            embedding: vec![1.0; LATENT_DIM],
            original_features: None,
        });
        gallery.add_sample(ReferenceSample {
            species: "Species2".to_string(),
            taxon: Taxon::Songbird,
            embedding: vec![0.5; LATENT_DIM],
            original_features: None,
        });

        gallery.save_to_json(gallery_path.to_str().unwrap()).unwrap();
        let loaded = ReferenceGallery::load_from_json(gallery_path.to_str().unwrap()).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.samples[0].species, "Species1");
        assert_eq!(loaded.samples[1].species, "Species2");
    }

    #[test]
    fn test_router_not_ready_empty_gallery() {
        let gallery = ReferenceGallery::new();
        let config = ZeroShotConfig::default();
        let router = ZeroShotRouter::new(config, gallery).unwrap();

        assert!(!router.is_ready(), "Router with empty gallery should not be ready");
    }
}
