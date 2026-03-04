//! BEANS-Zero Zero-Shot Bioacoustic Evaluation System
//! ===================================================
//!
//! This module implements zero-shot learning for the BEANS-Zero benchmark
//! using the 45D acoustic feature space. No training required - just
//! feature extraction and k-NN search.
//!
//! **Key Insight:**
//! Traditional ML requires training (backpropagation, weight updates).
//! This system uses instance-based learning (k-NN) - just store examples
//! and search for nearest neighbors at inference time.
//!
//! **Three BEANS-Zero Tasks:**
//! 1. **Audio Classification**: Species identification via k-NN voting
//! 2. **Audio Detection**: Presence/absence via prototype matching
//! 3. **Audio Captioning**: Natural language via feature-to-semantic mapping
//!
//! **Architecture:**
//! ```
//! Reference DB (45D features + labels) ──► k-NN Search ──► Inference
//!                                            │
//!                    No training required ───┘
//! ```
//!
//! Author: Sheel Morjaria (sheelmorjaria@gmail.com)
//! License: CC BY-ND 4.0 International

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::acoustic_algebra_45d::Vector45D;
use crate::acoustic_similarity::{AcousticSimilarityEngine, KnnResult};

// ============================================================================
// Reference Database Structures
// ============================================================================

/// Metadata for a single audio sample in the reference database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleMetadata {
    /// Unique sample identifier
    pub sample_id: String,
    /// Species label (e.g., "Song Sparrow", "Hainan Gibbon")
    pub species: String,
    /// Optional subspecies or population
    pub subspecies: Option<String>,
    /// Original file path
    pub file_path: String,
    /// Dataset source (e.g., "xenocanto", "macaulay", "beans")
    pub dataset: String,
    /// Recording quality (1-5)
    pub quality_score: Option<f32>,
    /// Geographic location
    pub location: Option<GeoLocation>,
    /// Recording timestamp
    pub timestamp: Option<String>,
    /// Human-provided caption (for captioning task)
    pub caption: Option<String>,
    /// Additional tags
    pub tags: Vec<String>,
}

/// Geographic location metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoLocation {
    pub latitude: f32,
    pub longitude: f32,
    pub country: Option<String>,
    pub region: Option<String>,
}

/// A single reference sample with features and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceSample {
    /// 45D acoustic feature vector
    pub features: Vector45D,
    /// Sample metadata
    pub metadata: SampleMetadata,
}

/// Species prototype (mean/median features for a species)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesPrototype {
    /// Species name
    pub species: String,
    /// Mean 45D vector for this species
    pub mean_vector: Vector45D,
    /// Standard deviation per dimension
    pub std_vector: Vector45D,
    /// Number of samples used to build prototype
    pub sample_count: usize,
    /// Intra-species variance (average distance to mean)
    pub intra_variance: f32,
}

/// Reference database for zero-shot inference
#[derive(Debug, Clone)]
pub struct ReferenceDatabase {
    /// All reference samples
    samples: Vec<ReferenceSample>,
    /// Species prototypes (computed from samples)
    prototypes: HashMap<String, SpeciesPrototype>,
    /// Similarity engine for k-NN search
    similarity_engine: AcousticSimilarityEngine,
    /// Index: species name -> sample indices
    species_index: HashMap<String, Vec<usize>>,
    /// Index: dataset name -> sample indices
    dataset_index: HashMap<String, Vec<usize>>,
    /// Total feature dimension (45)
    feature_dim: usize,
}

impl Default for ReferenceDatabase {
    fn default() -> Self {
        Self::new()
    }
}

impl ReferenceDatabase {
    /// Create an empty reference database
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            prototypes: HashMap::new(),
            similarity_engine: AcousticSimilarityEngine::new_45d(),
            species_index: HashMap::new(),
            dataset_index: HashMap::new(),
            feature_dim: 45,
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            prototypes: HashMap::new(),
            similarity_engine: AcousticSimilarityEngine::new_45d(),
            species_index: HashMap::new(),
            dataset_index: HashMap::new(),
            feature_dim: 45,
        }
    }

    /// Add a sample to the database
    pub fn add_sample(&mut self, sample: ReferenceSample) {
        let idx = self.samples.len();
        let species = sample.metadata.species.clone();
        let dataset = sample.metadata.dataset.clone();

        self.samples.push(sample);

        // Update indices
        self.species_index.entry(species).or_default().push(idx);
        self.dataset_index.entry(dataset).or_default().push(idx);
    }

    /// Add multiple samples
    pub fn add_samples(&mut self, samples: Vec<ReferenceSample>) {
        for sample in samples {
            self.add_sample(sample);
        }
    }

    /// Build species prototypes from current samples
    ///
    /// This computes the mean and std vector for each species,
    /// which is used for detection and prototype-based matching.
    pub fn build_prototypes(&mut self) {
        self.prototypes.clear();

        for (species, indices) in &self.species_index {
            if indices.is_empty() {
                continue;
            }

            // Collect all vectors for this species
            let vectors: Vec<&Vector45D> = indices
                .iter()
                .map(|&idx| &self.samples[idx].features)
                .collect();

            // Compute mean vector
            let mean = self.compute_mean_vector(&vectors);

            // Compute std vector
            let std = self.compute_std_vector(&vectors, &mean);

            // Compute intra-species variance
            let intra_variance = self.compute_intra_variance(&vectors, &mean);

            self.prototypes.insert(
                species.clone(),
                SpeciesPrototype {
                    species: species.clone(),
                    mean_vector: mean,
                    std_vector: std,
                    sample_count: vectors.len(),
                    intra_variance,
                },
            );
        }
    }

    /// Compute mean vector from multiple vectors
    fn compute_mean_vector(&self, vectors: &[&Vector45D]) -> Vector45D {
        if vectors.is_empty() {
            return Vector45D::default();
        }

        let mut sum = [0.0f32; 45];
        for v in vectors {
            let arr = v.to_array();
            for i in 0..45 {
                sum[i] += arr[i];
            }
        }

        let count = vectors.len() as f32;
        let mut mean = [0.0f32; 45];
        for i in 0..45 {
            mean[i] = sum[i] / count;
        }

        Vector45D::from_array(mean)
    }

    /// Compute standard deviation vector
    fn compute_std_vector(&self, vectors: &[&Vector45D], mean: &Vector45D) -> Vector45D {
        if vectors.len() < 2 {
            return Vector45D::default();
        }

        let mean_arr = mean.to_array();
        let mut sum_sq = [0.0f32; 45];

        for v in vectors {
            let arr = v.to_array();
            for i in 0..45 {
                let diff = arr[i] - mean_arr[i];
                sum_sq[i] += diff * diff;
            }
        }

        let count = vectors.len() as f32;
        let mut std = [0.0f32; 45];
        for i in 0..45 {
            std[i] = (sum_sq[i] / count).sqrt();
        }

        Vector45D::from_array(std)
    }

    /// Compute intra-species variance (average distance to mean)
    fn compute_intra_variance(&self, vectors: &[&Vector45D], mean: &Vector45D) -> f32 {
        if vectors.is_empty() {
            return 0.0;
        }

        let total_dist: f32 = vectors.iter().map(|v| v.distance_to(mean)).sum();
        total_dist / vectors.len() as f32
    }

    /// Fit the similarity engine on current samples
    ///
    /// This normalizes features for better distance computation.
    pub fn fit_similarity_engine(&mut self) {
        // Convert samples to ndarray for fitting
        let n = self.samples.len();
        if n == 0 {
            return;
        }

        let mut features = ndarray::Array2::<f64>::zeros((n, 45));
        for (i, sample) in self.samples.iter().enumerate() {
            let arr = sample.features.to_array();
            for j in 0..45 {
                features[[i, j]] = arr[j] as f64;
            }
        }

        self.similarity_engine.fit_normalization(&features);
    }

    // ========================================================================
    // Query Operations
    // ========================================================================

    /// k-NN search: find k nearest neighbors to query vector
    pub fn knn_search(&self, query: &Vector45D, k: usize) -> Vec<KnnSearchResult> {
        let query_arr = query.to_array();
        let query_nd = ndarray::Array1::from_vec(query_arr.iter().map(|&x| x as f64).collect());

        let mut distances: Vec<(usize, f64)> = self
            .samples
            .iter()
            .enumerate()
            .map(|(idx, sample)| {
                let sample_arr = sample.features.to_array();
                let sample_nd =
                    ndarray::Array1::from_vec(sample_arr.iter().map(|&x| x as f64).collect());
                let dist = self.similarity_engine.distance(&query_nd, &sample_nd);
                (idx, dist)
            })
            .collect();

        // Sort by distance (ascending)
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Return top k results
        distances
            .into_iter()
            .take(k)
            .map(|(idx, dist)| KnnSearchResult {
                sample_idx: idx,
                sample: self.samples[idx].clone(),
                distance: dist as f32,
                similarity: 1.0 - (-dist).exp() as f32,
            })
            .collect()
    }

    /// Get species prototype by name
    pub fn get_prototype(&self, species: &str) -> Option<&SpeciesPrototype> {
        self.prototypes.get(species)
    }

    /// Get all species names in the database
    pub fn species_list(&self) -> Vec<&String> {
        self.species_index.keys().collect()
    }

    /// Get sample count
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get sample count per species
    pub fn species_counts(&self) -> HashMap<&String, usize> {
        self.species_index
            .iter()
            .map(|(species, indices)| (species, indices.len()))
            .collect()
    }

    /// Get all samples for a species
    pub fn get_samples_by_species(&self, species: &str) -> Vec<&ReferenceSample> {
        self.species_index
            .get(species)
            .map(|indices| indices.iter().map(|&idx| &self.samples[idx]).collect())
            .unwrap_or_default()
    }

    /// Get samples by dataset
    pub fn get_samples_by_dataset(&self, dataset: &str) -> Vec<&ReferenceSample> {
        self.dataset_index
            .get(dataset)
            .map(|indices| indices.iter().map(|&idx| &self.samples[idx]).collect())
            .unwrap_or_default()
    }

    /// Compute similarity to a species prototype
    pub fn similarity_to_prototype(&self, query: &Vector45D, species: &str) -> Option<f32> {
        let prototype = self.prototypes.get(species)?;
        let distance = query.distance_to(&prototype.mean_vector);
        Some(1.0 - (-distance).exp())
    }

    /// Find most similar species prototype
    pub fn find_most_similar_species(&self, query: &Vector45D) -> Option<(&String, f32)> {
        self.prototypes
            .iter()
            .map(|(species, proto)| {
                let distance = query.distance_to(&proto.mean_vector);
                let similarity = 1.0 - (-distance).exp();
                (species, similarity)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }
}

/// Result from k-NN search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnnSearchResult {
    /// Index in reference database
    pub sample_idx: usize,
    /// The reference sample
    pub sample: ReferenceSample,
    /// Distance to query
    pub distance: f32,
    /// Similarity score (0-1)
    pub similarity: f32,
}

// ============================================================================
// Zero-Shot Classification
// ============================================================================

/// Zero-shot classifier using k-NN voting
#[derive(Debug, Clone)]
pub struct ZeroShotClassifier {
    /// Reference database
    reference_db: ReferenceDatabase,
    /// Number of neighbors to consider
    k: usize,
    /// Minimum confidence threshold
    min_confidence: f32,
}

/// Classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// Predicted species
    pub predicted_species: String,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Top-k predictions with scores
    pub top_predictions: Vec<SpeciesScore>,
    /// Nearest neighbors used for voting
    pub neighbors: Vec<KnnSearchResult>,
}

/// Species prediction with score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesScore {
    pub species: String,
    pub score: f32,
}

impl ZeroShotClassifier {
    /// Create a new zero-shot classifier
    pub fn new(reference_db: ReferenceDatabase) -> Self {
        Self {
            reference_db,
            k: 10,
            min_confidence: 0.5,
        }
    }

    /// Set number of neighbors
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set minimum confidence threshold
    pub fn with_min_confidence(mut self, threshold: f32) -> Self {
        self.min_confidence = threshold;
        self
    }

    /// Classify audio sample using k-NN voting
    ///
    /// # Arguments
    /// * `query` - 45D feature vector of the query audio
    ///
    /// # Returns
    /// * Classification result with predicted species and confidence
    pub fn classify(&self, query: &Vector45D) -> ClassificationResult {
        // Find k nearest neighbors
        let neighbors = self.reference_db.knn_search(query, self.k);

        // Weighted voting by similarity
        let mut votes: HashMap<String, f32> = HashMap::new();
        for neighbor in &neighbors {
            let species = &neighbor.sample.metadata.species;
            *votes.entry(species.clone()).or_insert(0.0) += neighbor.similarity;
        }

        // Sort by vote weight
        let mut top_predictions: Vec<SpeciesScore> = votes
            .into_iter()
            .map(|(species, score)| SpeciesScore { species, score })
            .collect();
        top_predictions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Get top prediction
        let (predicted_species, confidence) = top_predictions
            .first()
            .map(|p| (p.species.clone(), p.score / neighbors.len() as f32))
            .unwrap_or(("Unknown".to_string(), 0.0));

        ClassificationResult {
            predicted_species,
            confidence,
            top_predictions,
            neighbors,
        }
    }

    /// Classify with rejection (return None if confidence below threshold)
    pub fn classify_with_rejection(&self, query: &Vector45D) -> Option<ClassificationResult> {
        let result = self.classify(query);
        if result.confidence >= self.min_confidence {
            Some(result)
        } else {
            None
        }
    }

    /// Get reference to underlying database
    pub fn reference_db(&self) -> &ReferenceDatabase {
        &self.reference_db
    }
}

// ============================================================================
// Zero-Shot Detection
// ============================================================================

/// Zero-shot detector for presence/absence in long recordings
#[derive(Debug, Clone)]
pub struct ZeroShotDetector {
    /// Reference database
    reference_db: ReferenceDatabase,
    /// Detection threshold (similarity)
    threshold: f32,
    /// Window size in seconds
    window_size_sec: f32,
    /// Window overlap (0.0 to 1.0)
    window_overlap: f32,
}

/// Detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Whether target species is present
    pub present: bool,
    /// Detection events with timestamps
    pub events: Vec<BioacousticDetectionEvent>,
    /// Total detection confidence
    pub overall_confidence: f32,
    /// Number of positive windows
    pub positive_windows: usize,
    /// Total windows analyzed
    pub total_windows: usize,
}

/// Single detection event (bioacoustic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioacousticDetectionEvent {
    /// Start time in seconds
    pub start_time_sec: f32,
    /// End time in seconds
    pub end_time_sec: f32,
    /// Center time of detection
    pub center_time_sec: f32,
    /// Similarity score to target
    pub similarity: f32,
    /// Window index
    pub window_idx: usize,
}

impl ZeroShotDetector {
    /// Create a new zero-shot detector
    pub fn new(reference_db: ReferenceDatabase) -> Self {
        Self {
            reference_db,
            threshold: 0.7,
            window_size_sec: 1.0,
            window_overlap: 0.5,
        }
    }

    /// Set detection threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set window parameters
    pub fn with_windows(mut self, size_sec: f32, overlap: f32) -> Self {
        self.window_size_sec = size_sec;
        self.window_overlap = overlap;
        self
    }

    /// Detect target species in pre-extracted window features
    ///
    /// For actual audio, you would first extract 45D features for each window,
    /// then call this method with the feature vectors.
    ///
    /// # Arguments
    /// * `window_features` - 45D features for each sliding window
    /// * `target_species` - Species to detect
    ///
    /// # Returns
    /// * Detection result with events and confidence
    pub fn detect(&self, window_features: &[Vector45D], target_species: &str) -> DetectionResult {
        // Get species prototype
        let prototype = match self.reference_db.get_prototype(target_species) {
            Some(p) => p,
            None => {
                return DetectionResult {
                    present: false,
                    events: Vec::new(),
                    overall_confidence: 0.0,
                    positive_windows: 0,
                    total_windows: window_features.len(),
                };
            }
        };

        let mut events = Vec::new();
        let step = self.window_size_sec * (1.0 - self.window_overlap);

        for (idx, features) in window_features.iter().enumerate() {
            let similarity = 1.0 - (-features.distance_to(&prototype.mean_vector)).exp();

            if similarity >= self.threshold {
                let start_time = idx as f32 * step;
                let end_time = start_time + self.window_size_sec;

                events.push(BioacousticDetectionEvent {
                    start_time_sec: start_time,
                    end_time_sec: end_time,
                    center_time_sec: (start_time + end_time) / 2.0,
                    similarity,
                    window_idx: idx,
                });
            }
        }

        let positive_windows = events.len();
        let total_windows = window_features.len();
        let present = !events.is_empty();
        let overall_confidence = if !events.is_empty() {
            events.iter().map(|e| e.similarity).sum::<f32>() / events.len() as f32
        } else {
            0.0
        };

        DetectionResult {
            present,
            events,
            overall_confidence,
            positive_windows,
            total_windows,
        }
    }

    /// Detect using k-NN instead of prototype matching
    ///
    /// This is more robust for species with high intra-class variability.
    pub fn detect_knn(
        &self,
        window_features: &[Vector45D],
        target_species: &str,
        k: usize,
    ) -> DetectionResult {
        let step = self.window_size_sec * (1.0 - self.window_overlap);
        let mut events = Vec::new();

        for (idx, features) in window_features.iter().enumerate() {
            // Find k nearest neighbors
            let neighbors = self.reference_db.knn_search(features, k);

            // Count how many neighbors are the target species
            let target_count = neighbors
                .iter()
                .filter(|n| n.sample.metadata.species == target_species)
                .count();

            let target_weight: f32 = neighbors
                .iter()
                .filter(|n| n.sample.metadata.species == target_species)
                .map(|n| n.similarity)
                .sum();

            let similarity = target_weight / k as f32;

            if similarity >= self.threshold && target_count >= k / 2 {
                let start_time = idx as f32 * step;
                let end_time = start_time + self.window_size_sec;

                events.push(BioacousticDetectionEvent {
                    start_time_sec: start_time,
                    end_time_sec: end_time,
                    center_time_sec: (start_time + end_time) / 2.0,
                    similarity,
                    window_idx: idx,
                });
            }
        }

        let positive_windows = events.len();
        let total_windows = window_features.len();
        let present = !events.is_empty();
        let overall_confidence = if !events.is_empty() {
            events.iter().map(|e| e.similarity).sum::<f32>() / events.len() as f32
        } else {
            0.0
        };

        DetectionResult {
            present,
            events,
            overall_confidence,
            positive_windows,
            total_windows,
        }
    }

    /// Get reference to underlying database
    pub fn reference_db(&self) -> &ReferenceDatabase {
        &self.reference_db
    }
}

// ============================================================================
// Zero-Shot Captioning
// ============================================================================

/// Zero-shot captioner using feature-to-semantic mapping
#[derive(Debug, Clone)]
pub struct ZeroShotCaptioner {
    /// Reference database
    reference_db: ReferenceDatabase,
    /// Number of captioned samples to retrieve
    k: usize,
    /// Semantic descriptor generator
    semantic_gen: SemanticDescriptorGenerator,
}

/// Caption result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionResult {
    /// Generated caption
    pub caption: String,
    /// Semantic descriptors used
    pub descriptors: SemanticDescriptors,
    /// Similar captioned samples
    pub similar_samples: Vec<CaptionedSample>,
    /// Overall confidence
    pub confidence: f32,
}

/// Similar captioned sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionedSample {
    pub species: String,
    pub caption: String,
    pub similarity: f32,
}

impl ZeroShotCaptioner {
    /// Create a new zero-shot captioner
    pub fn new(reference_db: ReferenceDatabase) -> Self {
        Self {
            reference_db,
            k: 5,
            semantic_gen: SemanticDescriptorGenerator::new(),
        }
    }

    /// Set number of samples to retrieve
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Generate a caption for an audio sample
    ///
    /// # Arguments
    /// * `query` - 45D feature vector of the query audio
    ///
    /// # Returns
    /// * Caption result with generated text and similar samples
    pub fn caption(&self, query: &Vector45D) -> CaptionResult {
        // 1. Generate semantic descriptors from features
        let descriptors = self.semantic_gen.generate(query);

        // 2. Find similar captioned samples
        let neighbors = self.reference_db.knn_search(query, self.k);

        let similar_samples: Vec<CaptionedSample> = neighbors
            .iter()
            .filter_map(|n| {
                n.sample
                    .metadata
                    .caption
                    .as_ref()
                    .map(|caption| CaptionedSample {
                        species: n.sample.metadata.species.clone(),
                        caption: caption.clone(),
                        similarity: n.similarity,
                    })
            })
            .collect();

        // 3. Generate caption from descriptors
        let caption = self.synthesize_caption(&descriptors, &similar_samples);

        // 4. Compute confidence
        let confidence = neighbors.first().map(|n| n.similarity).unwrap_or(0.0);

        CaptionResult {
            caption,
            descriptors,
            similar_samples,
            confidence,
        }
    }

    /// Synthesize caption from descriptors and similar samples
    fn synthesize_caption(
        &self,
        descriptors: &SemanticDescriptors,
        similar_samples: &[CaptionedSample],
    ) -> String {
        let mut parts = Vec::new();

        // Pitch description
        parts.push(format!("A {} vocalization", descriptors.pitch));

        // Quality description
        if descriptors.quality != "neutral" {
            parts.push(format!("with {} quality", descriptors.quality));
        }

        // Temporal description
        if descriptors.temporal != "neutral" {
            parts.push(format!("and {} envelope", descriptors.temporal));
        }

        // Modulation description
        if descriptors.modulation != "steady" {
            parts.push(format!("featuring {}", descriptors.modulation));
        }

        // Spectral description
        if descriptors.spectral != "neutral" {
            parts.push(format!(
                "with {} spectral characteristics",
                descriptors.spectral
            ));
        }

        // Similar species reference
        if let Some(similar) = similar_samples.first() {
            parts.push(format!("similar to {}", similar.species));
        }

        parts.join(", ")
    }

    /// Get reference to underlying database
    pub fn reference_db(&self) -> &ReferenceDatabase {
        &self.reference_db
    }
}

// ============================================================================
// Semantic Descriptors (Option C)
// ============================================================================

/// Semantic descriptors for caption generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticDescriptors {
    /// Pitch description (e.g., "high-pitched", "low", "ultrasonic")
    pub pitch: String,
    /// Quality description (e.g., "clear", "harsh", "melodic")
    pub quality: String,
    /// Temporal description (e.g., "gradual attack", "sharp onset")
    pub temporal: String,
    /// Modulation description (e.g., "rapid FM sweep", "steady tone")
    pub modulation: String,
    /// Spectral description (e.g., "bright", "warm", "complex")
    pub spectral: String,
    /// Rhythm description (e.g., "regular rhythm", "irregular pattern")
    pub rhythm: String,
    /// Formant description (e.g., "resonant", "flat")
    pub formants: String,
    /// Complexity description (e.g., "simple", "complex", "rich harmonics")
    pub complexity: String,
}

/// Generator for semantic descriptors from 45D features
#[derive(Debug, Clone)]
pub struct SemanticDescriptorGenerator {
    /// Pitch thresholds (Hz)
    pitch_thresholds: (f32, f32, f32, f32), // ultrasonic, high, medium, low
    /// HNR thresholds (dB)
    hnr_thresholds: (f32, f32),
    /// FM slope thresholds (Hz/ms)
    fm_thresholds: (f32, f32),
}

impl Default for SemanticDescriptorGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticDescriptorGenerator {
    /// Create a new semantic descriptor generator with default thresholds
    pub fn new() -> Self {
        Self {
            // Pitch thresholds based on bioacoustic ranges
            pitch_thresholds: (20000.0, 4000.0, 1000.0, 500.0), // ultrasonic > 20kHz, high > 4kHz, etc.
            // HNR thresholds
            hnr_thresholds: (25.0, 15.0), // clear > 25dB, harsh < 15dB
            // FM slope thresholds
            fm_thresholds: (50.0, 10.0), // rapid > 50 Hz/ms, moderate > 10 Hz/ms
        }
    }

    /// Generate all semantic descriptors from 45D features
    pub fn generate(&self, features: &Vector45D) -> SemanticDescriptors {
        SemanticDescriptors {
            pitch: self.describe_pitch(features),
            quality: self.describe_quality(features),
            temporal: self.describe_temporal(features),
            modulation: self.describe_modulation(features),
            spectral: self.describe_spectral(features),
            rhythm: self.describe_rhythm(features),
            formants: self.describe_formants(features),
            complexity: self.describe_complexity(features),
        }
    }

    /// Describe pitch from mean_f0_hz
    pub fn describe_pitch(&self, features: &Vector45D) -> String {
        let f0 = features.mean_f0_hz;

        if f0 >= self.pitch_thresholds.0 {
            "ultrasonic".to_string()
        } else if f0 >= self.pitch_thresholds.1 {
            "high-pitched".to_string()
        } else if f0 >= self.pitch_thresholds.2 {
            "medium-pitched".to_string()
        } else if f0 >= self.pitch_thresholds.3 {
            "low-pitched".to_string()
        } else {
            "very low-pitched".to_string()
        }
    }

    /// Describe quality from HNR, spectral_flatness, harmonicity
    pub fn describe_quality(&self, features: &Vector45D) -> String {
        let hnr = features.harmonic_to_noise_ratio;
        let flatness = features.spectral_flatness;
        let harmonicity = features.harmonicity;

        // High HNR + high harmonicity = clear/tonal
        if hnr >= self.hnr_thresholds.0 && harmonicity >= 0.8 {
            if flatness < 0.3 {
                "clear, tonal".to_string()
            } else {
                "clear".to_string()
            }
        }
        // Low HNR = harsh/rough
        else if hnr <= self.hnr_thresholds.1 {
            if flatness > 0.5 {
                "harsh, noisy".to_string()
            } else {
                "rough".to_string()
            }
        }
        // Medium quality
        else if harmonicity >= 0.6 {
            "melodic".to_string()
        } else if flatness > 0.4 {
            "breathy".to_string()
        } else {
            "neutral".to_string()
        }
    }

    /// Describe temporal envelope from attack, decay, sustain
    pub fn describe_temporal(&self, features: &Vector45D) -> String {
        let attack = features.attack_time_ms;
        let decay = features.decay_time_ms;
        let sustain = features.sustain_level;

        // Attack-based description
        let attack_desc = if attack <= 3.0 {
            "sharp, percussive onset"
        } else if attack <= 10.0 {
            "quick onset"
        } else if attack <= 30.0 {
            "gradual attack"
        } else {
            "slow, swelling attack"
        };

        // Sustain-based modifier
        let sustain_mod = if sustain >= 0.8 {
            ", sustained"
        } else if sustain <= 0.3 {
            ", brief"
        } else {
            ""
        };

        // Decay-based modifier
        let decay_mod = if decay >= 50.0 {
            ", long decay"
        } else if decay <= 10.0 {
            ", abrupt ending"
        } else {
            ""
        };

        format!("{}{}{}", attack_desc, sustain_mod, decay_mod)
    }

    /// Describe modulation from FM slope, AM depth, vibrato
    pub fn describe_modulation(&self, features: &Vector45D) -> String {
        let fm_slope = features.fm_slope.abs();
        let am_depth = features.am_depth;
        let vibrato_rate = features.vibrato_rate_hz;
        let vibrato_depth = features.vibrato_depth;

        let mut descriptors = Vec::new();

        // FM-based
        if fm_slope >= self.fm_thresholds.0 {
            descriptors.push("rapid FM sweep");
        } else if fm_slope >= self.fm_thresholds.1 {
            descriptors.push("moderate FM sweep");
        } else if fm_slope > 0.0 {
            descriptors.push("slow FM glide");
        }

        // Vibrato-based
        if vibrato_rate >= 15.0 && vibrato_depth >= 30.0 {
            descriptors.push("fast trill");
        } else if vibrato_rate >= 8.0 && vibrato_depth >= 20.0 {
            descriptors.push("vibrato");
        } else if vibrato_rate > 0.0 && vibrato_depth > 0.0 {
            descriptors.push("slight pitch wobble");
        }

        // AM-based
        if am_depth >= 0.7 {
            descriptors.push("strong amplitude modulation");
        } else if am_depth >= 0.4 {
            descriptors.push("moderate amplitude modulation");
        }

        if descriptors.is_empty() {
            "steady".to_string()
        } else {
            descriptors.join(" with ")
        }
    }

    /// Describe spectral characteristics from centroid, spread, tilt
    pub fn describe_spectral(&self, features: &Vector45D) -> String {
        let centroid = features.spectral_centroid;
        let spread = features.spectral_spread;
        let tilt = features.spectral_tilt;
        let kurtosis = features.spectral_kurtosis;

        let mut descriptors = Vec::new();

        // Brightness from centroid
        if centroid >= 8000.0 {
            descriptors.push("bright");
        } else if centroid >= 4000.0 {
            descriptors.push("moderately bright");
        } else if centroid <= 1500.0 {
            descriptors.push("warm");
        }

        // Richness from spread
        if spread >= 4000.0 {
            descriptors.push("rich");
        } else if spread <= 1000.0 {
            descriptors.push("focused");
        }

        // Tilt-based
        if tilt >= -3.0 {
            descriptors.push("flat spectrum");
        } else if tilt <= -9.0 {
            descriptors.push("bass-emphasized");
        }

        // Complexity from kurtosis
        if kurtosis >= 4.0 {
            descriptors.push("peaky");
        } else if kurtosis <= 2.5 {
            descriptors.push("smooth spectrum");
        }

        if descriptors.is_empty() {
            "neutral".to_string()
        } else {
            descriptors.join(", ")
        }
    }

    /// Describe rhythm from ICI, onset rate
    pub fn describe_rhythm(&self, features: &Vector45D) -> String {
        let ici = features.median_ici_ms;
        let onset_rate = features.onset_rate_hz;
        let ici_cv = features.ici_coefficient_of_variation;

        // Regularity
        let regularity = if ici_cv <= 0.2 {
            "regular"
        } else if ici_cv <= 0.5 {
            "somewhat regular"
        } else {
            "irregular"
        };

        // Tempo
        let tempo = if onset_rate >= 20.0 {
            "rapid"
        } else if onset_rate >= 10.0 {
            "moderate"
        } else if onset_rate >= 5.0 {
            "slow"
        } else {
            "very sparse"
        };

        if ici > 0.0 {
            format!("{} {} rhythm (ICI ~{:.0}ms)", tempo, regularity, ici)
        } else {
            format!("{} {} pattern", tempo, regularity)
        }
    }

    /// Describe formant characteristics
    pub fn describe_formants(&self, features: &Vector45D) -> String {
        let f1 = features.formant_1_hz;
        let f2 = features.formant_2_hz;
        let dispersion = features.formant_dispersion;

        if dispersion >= 2500.0 {
            "wide-spaced formants".to_string()
        } else if dispersion >= 1500.0 {
            "well-defined formants".to_string()
        } else if dispersion > 0.0 {
            "close-spaced formants".to_string()
        } else {
            "flat".to_string()
        }
    }

    /// Describe overall complexity
    pub fn describe_complexity(&self, features: &Vector45D) -> String {
        let entropy = features.spectral_entropy;
        let subharmonic = features.subharmonic_ratio;
        let jitter = features.jitter;
        let shimmer = features.shimmer;

        let mut complexity_score = 0.0;

        if entropy > 0.5 {
            complexity_score += 1.0;
        }
        if subharmonic > 0.1 {
            complexity_score += 1.0;
        }
        if jitter > 0.02 {
            complexity_score += 0.5;
        }
        if shimmer > 0.03 {
            complexity_score += 0.5;
        }

        match complexity_score as usize {
            0 => "simple, clean".to_string(),
            1 => "slightly complex".to_string(),
            2 => "moderately complex".to_string(),
            _ => "rich, complex harmonics".to_string(),
        }
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Quick classification function (creates classifier on the fly)
pub fn classify_zero_shot(
    query: &Vector45D,
    reference_db: &ReferenceDatabase,
    k: usize,
) -> ClassificationResult {
    let classifier = ZeroShotClassifier::new(reference_db.clone()).with_k(k);
    classifier.classify(query)
}

/// Quick detection function
pub fn detect_zero_shot(
    window_features: &[Vector45D],
    reference_db: &ReferenceDatabase,
    target_species: &str,
    threshold: f32,
) -> DetectionResult {
    let detector = ZeroShotDetector::new(reference_db.clone()).with_threshold(threshold);
    detector.detect(window_features, target_species)
}

/// Quick captioning function
pub fn caption_zero_shot(query: &Vector45D, reference_db: &ReferenceDatabase) -> CaptionResult {
    let captioner = ZeroShotCaptioner::new(reference_db.clone());
    captioner.caption(query)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sample(species: &str, f0: f32, sample_id: usize) -> ReferenceSample {
        let mut features = Vector45D::default();
        features.mean_f0_hz = f0;
        features.duration_ms = 100.0;

        ReferenceSample {
            features,
            metadata: SampleMetadata {
                sample_id: format!("test_{:04}", sample_id),
                species: species.to_string(),
                subspecies: None,
                file_path: format!("/test/{}.wav", sample_id),
                dataset: "test".to_string(),
                quality_score: Some(4.0),
                location: None,
                timestamp: None,
                caption: Some(format!("A {} vocalization", species)),
                tags: vec!["test".to_string()],
            },
        }
    }

    #[test]
    fn test_reference_database_creation() {
        let db = ReferenceDatabase::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_add_samples() {
        let mut db = ReferenceDatabase::new();
        db.add_sample(create_test_sample("Marmoset", 8000.0, 1));
        db.add_sample(create_test_sample("Marmoset", 8500.0, 2));
        db.add_sample(create_test_sample("Bat", 45000.0, 3));

        assert_eq!(db.len(), 3);
        assert_eq!(db.species_list().len(), 2);
    }

    #[test]
    fn test_build_prototypes() {
        let mut db = ReferenceDatabase::new();
        db.add_sample(create_test_sample("Marmoset", 8000.0, 1));
        db.add_sample(create_test_sample("Marmoset", 8500.0, 2));
        db.add_sample(create_test_sample("Bat", 45000.0, 3));

        db.build_prototypes();

        let proto = db.get_prototype("Marmoset").unwrap();
        assert_eq!(proto.sample_count, 2);
        // Mean should be around 8250 Hz
        assert!(proto.mean_vector.mean_f0_hz > 8000.0);
        assert!(proto.mean_vector.mean_f0_hz < 8500.0);
    }

    #[test]
    fn test_knn_search() {
        let mut db = ReferenceDatabase::new();
        db.add_sample(create_test_sample("Marmoset", 8000.0, 1));
        db.add_sample(create_test_sample("Marmoset", 8100.0, 2));
        db.add_sample(create_test_sample("Bat", 45000.0, 3));

        let query = Vector45D {
            mean_f0_hz: 8050.0,
            ..Default::default()
        };

        let results = db.knn_search(&query, 2);
        assert_eq!(results.len(), 2);
        // Should find the two marmoset samples
        assert!(results[0].sample.metadata.species.contains("Marmoset"));
    }

    #[test]
    fn test_zero_shot_classifier() {
        let mut db = ReferenceDatabase::new();
        for i in 0..10 {
            db.add_sample(create_test_sample("Marmoset", 8000.0 + i as f32 * 50.0, i));
        }
        for i in 0..10 {
            db.add_sample(create_test_sample(
                "Bat",
                45000.0 + i as f32 * 500.0,
                i + 10,
            ));
        }
        db.build_prototypes();

        let classifier = ZeroShotClassifier::new(db).with_k(5);

        // Query similar to marmoset
        let marmoset_query = Vector45D {
            mean_f0_hz: 8200.0,
            ..Default::default()
        };
        let result = classifier.classify(&marmoset_query);
        assert_eq!(result.predicted_species, "Marmoset");
        assert!(result.confidence > 0.5);

        // Query similar to bat
        let bat_query = Vector45D {
            mean_f0_hz: 46000.0,
            ..Default::default()
        };
        let result = classifier.classify(&bat_query);
        assert_eq!(result.predicted_species, "Bat");
    }

    #[test]
    fn test_zero_shot_detector() {
        let mut db = ReferenceDatabase::new();
        for i in 0..5 {
            db.add_sample(create_test_sample("Marmoset", 8000.0 + i as f32 * 100.0, i));
        }
        db.build_prototypes();

        let detector = ZeroShotDetector::new(db).with_threshold(0.5);

        // Create window features - some similar to marmoset
        let windows: Vec<Vector45D> = (0..10)
            .map(|i| Vector45D {
                mean_f0_hz: if i >= 5 && i <= 7 { 8100.0 } else { 4000.0 },
                ..Default::default()
            })
            .collect();

        let result = detector.detect(&windows, "Marmoset");
        assert!(result.present);
        assert!(!result.events.is_empty());
    }

    #[test]
    fn test_semantic_descriptor_generator() {
        let gen = SemanticDescriptorGenerator::new();

        // High-pitched, clear sample
        let features = Vector45D {
            mean_f0_hz: 10000.0,
            harmonic_to_noise_ratio: 28.0,
            harmonicity: 0.9,
            spectral_flatness: 0.2,
            attack_time_ms: 5.0,
            decay_time_ms: 30.0,
            sustain_level: 0.8,
            fm_slope: 60.0,
            vibrato_rate_hz: 15.0,
            vibrato_depth: 40.0,
            ..Default::default()
        };

        let desc = gen.generate(&features);
        assert_eq!(desc.pitch, "high-pitched");
        assert!(desc.quality.contains("clear"));
        assert!(desc.modulation.contains("rapid FM sweep"));
    }

    #[test]
    fn test_zero_shot_captioner() {
        let mut db = ReferenceDatabase::new();
        db.add_sample(create_test_sample("Marmoset", 8000.0, 1));
        db.add_sample(create_test_sample("Bat", 45000.0, 2));
        db.build_prototypes();

        let captioner = ZeroShotCaptioner::new(db);

        let query = Vector45D {
            mean_f0_hz: 8100.0,
            harmonic_to_noise_ratio: 25.0,
            harmonicity: 0.85,
            ..Default::default()
        };

        let result = captioner.caption(&query);
        assert!(!result.caption.is_empty());
        assert!(result.caption.contains("vocalization"));
    }

    #[test]
    fn test_similarity_to_prototype() {
        let mut db = ReferenceDatabase::new();
        db.add_sample(create_test_sample("Marmoset", 8000.0, 1));
        db.add_sample(create_test_sample("Marmoset", 8200.0, 2));
        db.build_prototypes();

        let query = Vector45D {
            mean_f0_hz: 8100.0,
            ..Default::default()
        };

        let sim = db.similarity_to_prototype(&query, "Marmoset");
        assert!(sim.is_some());
        // Similarity should be positive (higher than a very different species)
        assert!(sim.unwrap() > 0.0);
    }

    #[test]
    fn test_species_counts() {
        let mut db = ReferenceDatabase::new();
        db.add_sample(create_test_sample("Marmoset", 8000.0, 1));
        db.add_sample(create_test_sample("Marmoset", 8200.0, 2));
        db.add_sample(create_test_sample("Bat", 45000.0, 3));
        db.add_sample(create_test_sample("Dolphin", 12000.0, 4));

        let counts = db.species_counts();
        assert_eq!(*counts.get(&"Marmoset".to_string()).unwrap(), 2);
        assert_eq!(*counts.get(&"Bat".to_string()).unwrap(), 1);
        assert_eq!(*counts.get(&"Dolphin".to_string()).unwrap(), 1);
    }
}

// ============================================================================
// PYTHON BINDINGS (PyO3)
// ============================================================================

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

/// Python wrapper for ReferenceDatabase
#[cfg(feature = "python-bindings")]
#[pyclass(name = "ReferenceDatabase")]
#[derive(Clone)]
pub struct PyReferenceDatabase {
    pub inner: ReferenceDatabase,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyReferenceDatabase {
    #[new]
    fn new() -> Self {
        Self {
            inner: ReferenceDatabase::new(),
        }
    }

    /// Add a sample to the database
    fn add_sample(
        &mut self,
        features: Vec<f32>,
        species: String,
        sample_id: String,
    ) -> PyResult<()> {
        if features.len() != 45 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Features must have 45 elements, got {}",
                features.len()
            )));
        }

        let mut arr = [0.0f32; 45];
        arr.copy_from_slice(&features);

        let sample = ReferenceSample {
            features: Vector45D::from_array(arr),
            metadata: SampleMetadata {
                sample_id,
                species,
                subspecies: None,
                file_path: String::new(),
                dataset: "python".to_string(),
                quality_score: None,
                location: None,
                timestamp: None,
                caption: None,
                tags: Vec::new(),
            },
        };

        self.inner.add_sample(sample);
        Ok(())
    }

    /// Build species prototypes
    fn build_prototypes(&mut self) {
        self.inner.build_prototypes();
    }

    /// Get number of samples
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Get species list
    fn species_list(&self) -> Vec<String> {
        self.inner.species_list().into_iter().cloned().collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ReferenceDatabase({} samples, {} species)",
            self.inner.len(),
            self.inner.species_list().len()
        )
    }
}

/// Python wrapper for ZeroShotClassifier
#[cfg(feature = "python-bindings")]
#[pyclass(name = "ZeroShotClassifier")]
pub struct PyZeroShotClassifier {
    inner: ZeroShotClassifier,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyZeroShotClassifier {
    #[new]
    fn new(db: PyReferenceDatabase) -> Self {
        Self {
            inner: ZeroShotClassifier::new(db.inner),
        }
    }

    /// Classify a sample
    fn classify(&self, features: Vec<f32>) -> PyResult<PyClassificationResult> {
        if features.len() != 45 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Features must have 45 elements, got {}",
                features.len()
            )));
        }

        let mut arr = [0.0f32; 45];
        arr.copy_from_slice(&features);
        let query = Vector45D::from_array(arr);

        let result = self.inner.classify(&query);
        Ok(PyClassificationResult { inner: result })
    }
}

/// Python wrapper for ClassificationResult
#[cfg(feature = "python-bindings")]
#[pyclass(name = "ClassificationResult")]
pub struct PyClassificationResult {
    inner: ClassificationResult,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyClassificationResult {
    #[getter]
    fn predicted_species(&self) -> &str {
        &self.inner.predicted_species
    }

    #[getter]
    fn confidence(&self) -> f32 {
        self.inner.confidence
    }

    fn __repr__(&self) -> String {
        format!(
            "ClassificationResult({} @ {:.2}%)",
            self.inner.predicted_species,
            self.inner.confidence * 100.0
        )
    }
}

/// Python wrapper for SemanticDescriptors
#[cfg(feature = "python-bindings")]
#[pyclass(name = "SemanticDescriptors")]
pub struct PySemanticDescriptors {
    inner: SemanticDescriptors,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PySemanticDescriptors {
    /// Generate semantic descriptors from 45D features
    #[staticmethod]
    fn from_features(features: Vec<f32>) -> PyResult<Self> {
        if features.len() != 45 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Features must have 45 elements, got {}",
                features.len()
            )));
        }

        let mut arr = [0.0f32; 45];
        arr.copy_from_slice(&features);
        let vector = Vector45D::from_array(arr);

        let gen = SemanticDescriptorGenerator::new();
        Ok(Self {
            inner: gen.generate(&vector),
        })
    }

    #[getter]
    fn pitch(&self) -> &str {
        &self.inner.pitch
    }

    #[getter]
    fn quality(&self) -> &str {
        &self.inner.quality
    }

    #[getter]
    fn temporal(&self) -> &str {
        &self.inner.temporal
    }

    #[getter]
    fn modulation(&self) -> &str {
        &self.inner.modulation
    }

    #[getter]
    fn spectral(&self) -> &str {
        &self.inner.spectral
    }

    fn __repr__(&self) -> String {
        format!(
            "SemanticDescriptors(pitch={}, quality={}, modulation={})",
            self.inner.pitch, self.inner.quality, self.inner.modulation
        )
    }
}
