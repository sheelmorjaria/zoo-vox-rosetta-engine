// =============================================================================
// Acoustic Similarity Engine
// =============================================================================
//
// This module implements pairwise acoustic similarity analysis for animal
// vocalizations. Instead of forcing discrete clusters (which HDBSCAN expects),
// this approach recognizes that vocalizations exist on CONTINUOUS ACOUSTIC
// MANIFOLDS.
//
// Key Insight:
// ─────────────
// Animal vocalizations form continuous gradients, not discrete islands:
//
//   Phee ←───────→ Trill ←───────→ Twitter ←──────→ Tsik
//      (continuous acoustic transitions, not separate clusters)
//
// HDBSCAN expects ISLANDS. You have a CONTINENT.
//
// Use Cases:
// ──────────
// - Find phrases acoustically similar to a query
// - Measure within-call-type vs between-call-type similarity
// - Detect gradual acoustic drift over time
// - Identify phrase "dialects" between individuals/colonies
// - k-NN classification for call type prediction

use std::collections::HashMap;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

// =============================================================================
// Similarity Metrics
// =============================================================================

/// Distance metric type for similarity computation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum DistanceMetric {
    /// Weighted Euclidean distance
    #[default]
    WeightedEuclidean,

    /// Cosine similarity (1 - cosine = distance)
    Cosine,

    /// Manhattan distance (L1 norm)
    Manhattan,

    /// Chebyshev distance (L-inf norm)
    Chebyshev,
}

// =============================================================================
// Acoustic Similarity Engine
// =============================================================================

/// Computes acoustic similarity between phrase feature vectors
///
/// This engine uses weighted distance metrics with feature importance
/// weights optimized for bioacoustic similarity.
#[derive(Debug, Clone)]
pub struct AcousticSimilarityEngine {
    /// Feature weights (learned or heuristically determined)
    weights: Array1<f64>,

    /// Distance metric type
    metric: DistanceMetric,

    /// Normalization parameters
    feature_means: Array1<f64>,
    feature_stds: Array1<f64>,

    /// Whether normalization has been fitted
    fitted: bool,
}

impl AcousticSimilarityEngine {
    /// Create engine with default weights based on feature importance
    pub fn new(feature_dim: usize) -> Self {
        let weights = Self::default_feature_weights(feature_dim);

        Self {
            weights,
            metric: DistanceMetric::WeightedEuclidean,
            feature_means: Array1::zeros(feature_dim),
            feature_stds: Array1::ones(feature_dim),
            fitted: false,
        }
    }

    /// Create engine with specific distance metric
    pub fn with_metric(feature_dim: usize, metric: DistanceMetric) -> Self {
        let mut engine = Self::new(feature_dim);
        engine.metric = metric;
        engine
    }

    /// Feature weights optimized for bioacoustic similarity (30D features)
    ///
    /// Micro-Dynamics Feature Groups (30D):
    /// - [0-2]: Temporal (attack, decay, sustain) - HIGH importance for envelope shape
    /// - [3-4]: Modulation (vibrato rate, depth) - HIGH for trill vs phee
    /// - [5-6]: Perturbation (jitter, shimmer) - MEDIUM importance
    /// - [7-9]: Timbre (harmonicity, flatness, HNR) - HIGH importance for call type
    /// - [10-22]: MFCCs (13 coefficients) - HIGH importance
    /// - [23]: Spectral flux - MEDIUM importance
    /// - [24-26]: Rhythm (ICI, onset rate, ICI CV) - MEDIUM importance
    ///
    /// Note: For different feature dimensions, weights are adjusted proportionally
    fn default_feature_weights(feature_dim: usize) -> Array1<f64> {
        let mut weights = Array1::ones(feature_dim);

        // 30D layout (from MicroDynamicsFeatures):
        // [0-2]: attack_time_ms, decay_time_ms, sustain_level
        // [3-4]: vibrato_rate_hz, vibrato_depth
        // [5-6]: jitter, shimmer
        // [7-9]: harmonicity, spectral_flatness, harmonic_to_noise_ratio
        // [10-22]: mfcc[0-12]
        // [23]: spectral_flux
        // [24-26]: median_ici_ms, onset_rate_hz, ici_coefficient_of_variation

        if feature_dim >= 27 {
            // Temporal envelope (0-2)
            weights[0] = 1.8; // Attack - phee has slow attack, tsik fast
            weights[1] = 1.5; // Decay
            weights[2] = 1.3; // Sustain level

            // Modulation (3-4) - CRITICAL for trill identification
            weights[3] = 2.5; // Vibrato rate - trill has high rate
            weights[4] = 2.2; // Vibrato depth

            // Perturbation (5-6)
            weights[5] = 1.0; // Jitter
            weights[6] = 1.0; // Shimmer

            // Timbre (7-9) - HIGH importance for call type discrimination
            weights[7] = 1.8; // Harmonicity - distinguishes tonal vs noisy
            weights[8] = 1.5; // Spectral flatness - Wiener entropy
            weights[9] = 1.8; // HNR - critical for call quality

            // MFCCs (10-22) - HIGH importance
            for i in 10..=22 {
                if i < feature_dim {
                    weights[i] = match i {
                        10 => 2.0, // MFCC 0 - energy/brightness, very discriminative
                        11 => 1.8, // MFCC 1 - spectral shape
                        12 => 1.5, // MFCC 2
                        _ => 1.3,  // MFCC 3-12
                    };
                }
            }

            // Spectral flux (23)
            if feature_dim > 23 {
                weights[23] = 1.5;
            }

            // Rhythm (24-26)
            if feature_dim > 24 {
                weights[24] = 1.3; // Median ICI
            }
            if feature_dim > 25 {
                weights[25] = 1.3; // Onset rate
            }
            if feature_dim > 26 {
                weights[26] = 1.2; // ICI CV
            }
        } else {
            // For smaller feature sets, scale weights proportionally
            let scale = 30.0 / feature_dim as f64;
            for i in 0..feature_dim {
                weights[i] = 1.5 * scale.min(2.0);
            }
        }

        weights
    }

    /// Fit normalization parameters from dataset
    pub fn fit_normalization(&mut self, features: &Array2<f64>) {
        let n_features = features.ncols();

        for j in 0..n_features {
            let col = features.column(j);
            self.feature_means[j] = col.mean().unwrap_or(0.0);

            let variance = col.var(1.0);
            self.feature_stds[j] = variance.sqrt().max(1e-10);
        }

        self.fitted = true;
    }

    /// Set species-specific feature weights
    ///
    /// This allows the similarity engine to use domain knowledge about
    /// which features are important for each species:
    /// - Sperm Whale: Rhythm (ICI, onset rate) weighted high
    /// - Dolphin: FM slope weighted high for whistle contours
    /// - Macaque: Spectral kurtosis/tilt for voice quality
    /// - Bat: FM slope and micro-dynamics for echolocation
    pub fn set_feature_weights(&mut self, weights: &[f32]) {
        let min_len = weights.len().min(self.weights.len());
        for (i, &w) in weights.iter().take(min_len).enumerate() {
            self.weights[i] = w as f64;
        }
    }

    /// Get current feature weights
    pub fn feature_weights(&self) -> &[f64] {
        self.weights.as_slice().unwrap_or(&[])
    }

    /// Check if normalization has been fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Normalize a single feature vector
    fn normalize(&self, features: &Array1<f64>) -> Array1<f64> {
        if !self.fitted {
            return features.clone();
        }
        (features - &self.feature_means) / &self.feature_stds
    }

    /// Normalize a feature matrix
    fn normalize_matrix(&self, features: &Array2<f64>) -> Array2<f64> {
        if !self.fitted {
            return features.clone();
        }

        let mut normalized = features.clone();
        for j in 0..features.ncols() {
            let m = self.feature_means[j];
            let s = self.feature_stds[j];
            normalized.column_mut(j).mapv_inplace(|x| (x - m) / s);
        }
        normalized
    }

    /// Compute similarity between two phrases
    ///
    /// Returns: 0.0 = identical, 1.0 = maximally different
    pub fn similarity(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let distance = self.distance(a, b);

        // Convert distance to similarity using exponential decay
        // This maps [0, inf) -> [1, 0)
        1.0 - (-distance).exp()
    }

    /// Raw distance (lower = more similar)
    pub fn distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let a_norm = self.normalize(a);
        let b_norm = self.normalize(b);

        match self.metric {
            DistanceMetric::WeightedEuclidean => self.weighted_euclidean(&a_norm, &b_norm),
            DistanceMetric::Cosine => self.cosine_distance(&a_norm, &b_norm),
            DistanceMetric::Manhattan => self.manhattan_distance(&a_norm, &b_norm),
            DistanceMetric::Chebyshev => self.chebyshev_distance(&a_norm, &b_norm),
        }
    }

    fn weighted_euclidean(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let min_len = a.len().min(b.len()).min(self.weights.len());
        a.iter()
            .zip(b.iter())
            .zip(self.weights.iter())
            .take(min_len)
            .map(|((x, y), w)| w * (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn cosine_distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let min_len = a.len().min(b.len()).min(self.weights.len());

        let dot: f64 = a
            .iter()
            .zip(b.iter())
            .zip(self.weights.iter())
            .take(min_len)
            .map(|((x, y), w)| w * x * y)
            .sum();

        let norm_a: f64 = a
            .iter()
            .zip(self.weights.iter())
            .take(min_len)
            .map(|(x, w)| w * x.powi(2))
            .sum::<f64>()
            .sqrt();

        let norm_b: f64 = b
            .iter()
            .zip(self.weights.iter())
            .take(min_len)
            .map(|(y, w)| w * y.powi(2))
            .sum::<f64>()
            .sqrt();

        let cosine = dot / (norm_a * norm_b + 1e-10);
        1.0 - cosine // Convert similarity to distance
    }

    fn manhattan_distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let min_len = a.len().min(b.len()).min(self.weights.len());
        a.iter()
            .zip(b.iter())
            .zip(self.weights.iter())
            .take(min_len)
            .map(|((x, y), w)| w * (x - y).abs())
            .sum()
    }

    fn chebyshev_distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let min_len = a.len().min(b.len()).min(self.weights.len());
        a.iter()
            .zip(b.iter())
            .zip(self.weights.iter())
            .take(min_len)
            .map(|((x, y), w)| w * (x - y).abs())
            .fold(0.0_f64, |max, val| max.max(val))
    }

    /// Find k most similar phrases to query
    pub fn find_similar(&self, query: &Array1<f64>, candidates: &[Array1<f64>], k: usize) -> Vec<(usize, f64)> {
        let mut distances: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.distance(query, c)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);

        distances
    }

    /// Get feature weights
    pub fn weights(&self) -> &Array1<f64> {
        &self.weights
    }

    /// Set custom feature weights
    pub fn set_weights(&mut self, weights: Array1<f64>) {
        self.weights = weights;
    }
}

// =============================================================================
// Similarity Analysis Results
// =============================================================================

/// A pair of files with their distance/similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilePair {
    pub file_a: String,
    pub file_b: String,
    pub score: f64,
}

/// Feature discrimination score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDiscrimination {
    pub feature_name: String,
    pub f_ratio: f64,
}

/// Between-type distance entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetweenTypeDistance {
    pub type_a: String,
    pub type_b: String,
    pub distance: f64,
}

/// Results from comprehensive similarity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityAnalysis {
    /// Within-call-type average similarity
    pub within_type_similarity: HashMap<String, f64>,

    /// Between-call-type average distance (using string keys for JSON compatibility)
    pub between_type_distance: HashMap<String, f64>,

    /// Between-type distances as list for easier consumption
    pub between_type_list: Vec<BetweenTypeDistance>,

    /// Most similar pair across different types (concerning if distance is low)
    pub most_similar_cross_type: Option<FilePair>,

    /// Least similar pair within same type (indicates high within-type variance)
    pub least_similar_within_type: Option<FilePair>,

    /// Per-feature discrimination scores (F-ratio: higher = more discriminative)
    pub feature_discrimination: Vec<FeatureDiscrimination>,

    /// Average within-type distance
    pub avg_within_distance: f64,

    /// Average between-type distance
    pub avg_between_distance: f64,

    /// Separation ratio (between / within, higher = better)
    pub separation_ratio: f64,
}

impl SimilarityAnalysis {
    /// Compute comprehensive similarity analysis
    pub fn analyze(
        features: &Array2<f64>,
        call_types: &[String],
        file_names: &[String],
        feature_names: &[&str],
    ) -> Self {
        let _n_samples = features.nrows();
        let n_features = features.ncols();

        // Build similarity engine
        let mut engine = AcousticSimilarityEngine::new(n_features);
        engine.fit_normalization(features);

        // Group samples by call type
        let mut type_indices: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, ct) in call_types.iter().enumerate() {
            type_indices.entry(ct.clone()).or_default().push(i);
        }

        // Compute within-type similarities
        let mut within_type_similarity = HashMap::new();
        let mut least_similar_within: Option<(String, String, f64)> = None;
        let mut all_within_distances = Vec::new();

        for (ct, indices) in &type_indices {
            if indices.len() < 2 {
                continue;
            }

            let mut distances = Vec::new();
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    let a = features.row(indices[i]).to_owned();
                    let b = features.row(indices[j]).to_owned();
                    let dist = engine.distance(&a, &b);
                    distances.push(dist);
                    all_within_distances.push(dist);

                    // Track least similar within type (highest distance)
                    if least_similar_within.is_none()
                        || dist > least_similar_within.as_ref().map(|(_, _, d)| *d).unwrap_or(0.0)
                    {
                        least_similar_within =
                            Some((file_names[indices[i]].clone(), file_names[indices[j]].clone(), dist));
                    }
                }
            }

            if !distances.is_empty() {
                let avg_sim = 1.0 - (-distances.iter().sum::<f64>() / distances.len() as f64).exp();
                within_type_similarity.insert(ct.clone(), avg_sim);
            }
        }

        // Compute between-type distances
        let mut between_type_distance = HashMap::new();
        let mut most_similar_cross: Option<(String, String, f64)> = None;
        let mut all_between_distances = Vec::new();

        let types: Vec<_> = type_indices.keys().cloned().collect();
        for i in 0..types.len() {
            for j in (i + 1)..types.len() {
                let type_a = &types[i];
                let type_b = &types[j];
                let indices_a = &type_indices[type_a];
                let indices_b = &type_indices[type_b];

                let mut distances = Vec::new();
                for &idx_a in indices_a {
                    for &idx_b in indices_b {
                        let a = features.row(idx_a).to_owned();
                        let b = features.row(idx_b).to_owned();
                        let dist = engine.distance(&a, &b);
                        distances.push(dist);
                        all_between_distances.push(dist);

                        // Track most similar cross-type pair (lowest distance)
                        if most_similar_cross.is_none()
                            || dist < most_similar_cross.as_ref().map(|(_, _, d)| *d).unwrap_or(f64::MAX)
                        {
                            most_similar_cross = Some((file_names[idx_a].clone(), file_names[idx_b].clone(), dist));
                        }
                    }
                }

                if !distances.is_empty() {
                    let avg_dist = distances.iter().sum::<f64>() / distances.len() as f64;
                    between_type_distance.insert((type_a.clone(), type_b.clone()), avg_dist);
                }
            }
        }

        // Compute per-feature discrimination scores
        let mut feature_discrimination = Vec::new();
        for f in 0..n_features {
            let col = features.column(f);

            // Within-type variance
            let within_var: f64 = type_indices
                .values()
                .map(|indices| {
                    let values: Vec<f64> = indices.iter().map(|&i| col[i]).collect();
                    if values.len() > 1 {
                        let mean = values.iter().sum::<f64>() / values.len() as f64;
                        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
                / type_indices.len().max(1) as f64;

            // Between-type variance
            let type_means: Vec<f64> = type_indices
                .values()
                .map(|indices| indices.iter().map(|&i| col[i]).sum::<f64>() / indices.len().max(1) as f64)
                .collect();

            let grand_mean = if !type_means.is_empty() {
                type_means.iter().sum::<f64>() / type_means.len() as f64
            } else {
                0.0
            };

            let between_var = if !type_means.is_empty() {
                type_means.iter().map(|m| (m - grand_mean).powi(2)).sum::<f64>() / type_means.len() as f64
            } else {
                0.0
            };

            // F-ratio (higher = more discriminative)
            let f_ratio = if within_var > 1e-10 {
                between_var / within_var
            } else if between_var > 1e-10 {
                f64::INFINITY
            } else {
                0.0
            };

            let name = feature_names.get(f).unwrap_or(&"unknown");
            feature_discrimination.push((name.to_string(), f_ratio));
        }

        // Sort by discrimination score
        feature_discrimination.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Calculate overall statistics
        let avg_within = if !all_within_distances.is_empty() {
            all_within_distances.iter().sum::<f64>() / all_within_distances.len() as f64
        } else {
            0.0
        };

        let avg_between = if !all_between_distances.is_empty() {
            all_between_distances.iter().sum::<f64>() / all_between_distances.len() as f64
        } else {
            0.0
        };

        let separation_ratio = if avg_within > 1e-10 {
            avg_between / avg_within
        } else {
            f64::INFINITY
        };

        // Convert between_type_distance to string keys for JSON compatibility
        let between_type_distance_str: HashMap<String, f64> = between_type_distance
            .iter()
            .map(|((a, b), dist)| (format!("{}|{}", a, b), *dist))
            .collect();

        let between_type_list: Vec<BetweenTypeDistance> = between_type_distance
            .into_iter()
            .map(|((type_a, type_b), distance)| BetweenTypeDistance {
                type_a,
                type_b,
                distance,
            })
            .collect();

        // Convert feature_discrimination to struct format
        let feature_discrimination_struct: Vec<FeatureDiscrimination> = feature_discrimination
            .into_iter()
            .map(|(feature_name, f_ratio)| FeatureDiscrimination { feature_name, f_ratio })
            .collect();

        Self {
            within_type_similarity,
            between_type_distance: between_type_distance_str,
            between_type_list,
            most_similar_cross_type: most_similar_cross.map(|(file_a, file_b, score)| FilePair {
                file_a,
                file_b,
                score,
            }),
            least_similar_within_type: least_similar_within.map(|(file_a, file_b, score)| FilePair {
                file_a,
                file_b,
                score,
            }),
            feature_discrimination: feature_discrimination_struct,
            avg_within_distance: avg_within,
            avg_between_distance: avg_between,
            separation_ratio,
        }
    }

    /// Print analysis summary
    pub fn print_summary(&self) {
        println!();
        println!("╔═══════════════════════════════════════════════════════════════╗");
        println!("║         Acoustic Similarity Analysis Summary                   ║");
        println!("╚═══════════════════════════════════════════════════════════════╝");

        println!("\n📊 Overall Separation Metrics:");
        println!("   • Average within-type distance: {:.4}", self.avg_within_distance);
        println!("   • Average between-type distance: {:.4}", self.avg_between_distance);
        println!("   • Separation ratio: {:.2}x (higher = better)", self.separation_ratio);

        println!("\n📊 Within-Call-Type Similarity (higher = more cohesive):");
        let mut within: Vec<_> = self.within_type_similarity.iter().collect();
        within.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (ct, sim) in within {
            println!("   • {}: {:.4}", ct, sim);
        }

        println!("\n📊 Between-Call-Type Distance (higher = more separated):");
        let mut between: Vec<_> = self.between_type_list.iter().collect();
        between.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap());
        for entry in between.iter().take(10) {
            println!("   • {} ↔ {}: {:.4}", entry.type_a, entry.type_b, entry.distance);
        }

        if let Some(pair) = &self.most_similar_cross_type {
            println!("\n⚠️  Most similar cross-type pair:");
            println!("   {} ↔ {} (distance: {:.4})", pair.file_a, pair.file_b, pair.score);
            if pair.score < 0.5 {
                println!("   ⚠️  WARNING: Low distance suggests potential call type confusion");
            }
        }

        if let Some(pair) = &self.least_similar_within_type {
            println!("\n⚠️  Least similar within-type pair:");
            println!("   {} ↔ {} (distance: {:.4})", pair.file_a, pair.file_b, pair.score);
            if pair.score > 2.0 {
                println!("   ⚠️  WARNING: High within-type variance detected");
            }
        }

        println!("\n📊 Feature Discrimination (F-ratio, higher = more discriminative):");
        for feat in self.feature_discrimination.iter().take(10) {
            let stars = if feat.f_ratio > 2.0 {
                "★★★"
            } else if feat.f_ratio > 1.0 {
                "★★"
            } else {
                "★"
            };
            println!("   • {}: {:.2} {}", feat.feature_name, feat.f_ratio, stars);
        }
    }
}

// =============================================================================
// K-NN Classifier
// =============================================================================

/// k-Nearest Neighbors classifier for acoustic features
///
/// This is preferred over clustering for continuous acoustic manifolds
/// because it provides:
/// - Soft classification with confidence scores
/// - Query-dependent neighborhoods
/// - No forced discrete assignments
#[derive(Debug, Clone)]
pub struct KnnClassifier {
    /// Training features
    train_features: Array2<f64>,

    /// Training labels
    train_labels: Vec<String>,

    /// File names for training samples
    train_files: Vec<String>,

    /// Normalization parameters
    feature_means: Array1<f64>,
    feature_stds: Array1<f64>,

    /// Feature weights
    weights: Array1<f64>,
}

/// Result from k-NN classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnnResult {
    /// Predicted call type
    pub predicted_type: String,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,

    /// Top k predictions with distances
    pub neighbors: Vec<KnnNeighbor>,

    /// Distribution of votes across call types
    pub vote_distribution: HashMap<String, f64>,
}

/// A single neighbor in k-NN result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnnNeighbor {
    pub file_name: String,
    pub call_type: String,
    pub distance: f64,
    pub similarity: f64,
}

/// Confusion matrix entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionEntry {
    pub predicted: String,
    pub actual: String,
    pub count: usize,
}

/// Cross-validation results for k-NN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnnCvResults {
    /// Mean accuracy across folds
    pub mean_accuracy: f64,

    /// Accuracy per fold
    pub fold_accuracies: Vec<f64>,

    /// Per-class accuracy
    pub per_class_accuracy: HashMap<String, f64>,

    /// Confusion matrix as list for JSON compatibility
    pub confusion_matrix: Vec<ConfusionEntry>,

    /// Optimal k value found
    pub optimal_k: usize,
}

impl KnnClassifier {
    /// Create a new k-NN classifier from training data
    pub fn new(features: Array2<f64>, labels: Vec<String>, files: Vec<String>) -> Self {
        let n_features = features.ncols();

        // Compute normalization
        let mut means = Array1::zeros(n_features);
        let mut stds = Array1::ones(n_features);

        for j in 0..n_features {
            let col = features.column(j);
            means[j] = col.mean().unwrap_or(0.0);
            stds[j] = col.var(1.0).sqrt().max(1e-10);
        }

        Self {
            train_features: features,
            train_labels: labels,
            train_files: files,
            feature_means: means,
            feature_stds: stds,
            weights: AcousticSimilarityEngine::default_feature_weights(n_features),
        }
    }

    fn normalize(&self, features: &Array2<f64>) -> Array2<f64> {
        let mut normalized = features.clone();
        for j in 0..features.ncols() {
            let m = self.feature_means[j];
            let s = self.feature_stds[j];
            normalized.column_mut(j).mapv_inplace(|x| (x - m) / s);
        }
        normalized
    }

    fn normalize_vector(&self, query: &Array1<f64>) -> Array1<f64> {
        (query - &self.feature_means) / &self.feature_stds
    }

    /// Classify a single sample using k-NN
    pub fn classify(&self, query: &Array1<f64>, k: usize) -> KnnResult {
        let query_norm = self.normalize_vector(query);
        let train_norm = self.normalize(&self.train_features);

        // Compute distances to all training samples
        let mut distances: Vec<(usize, f64)> = train_norm
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                let min_len = row.len().min(self.weights.len());
                let dist: f64 = row
                    .iter()
                    .zip(query_norm.iter())
                    .zip(self.weights.iter())
                    .take(min_len)
                    .map(|((x, y), w)| w * (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (i, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Distance-weighted voting
        let mut votes: HashMap<String, f64> = HashMap::new();
        let total_weight: f64 = distances.iter().take(k).map(|(_, d)| 1.0 / (d + 1e-10)).sum();

        for (idx, dist) in distances.iter().take(k) {
            let label = &self.train_labels[*idx];
            let weight = 1.0 / (dist + 1e-10);
            *votes.entry(label.clone()).or_default() += weight / total_weight;
        }

        // Find winner
        let (predicted, confidence) = votes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(l, &w)| (l.clone(), w))
            .unwrap_or(("Unknown".to_string(), 0.0));

        // Build neighbor list
        let neighbors: Vec<KnnNeighbor> = distances
            .iter()
            .take(k)
            .map(|(idx, dist)| KnnNeighbor {
                file_name: self.train_files[*idx].clone(),
                call_type: self.train_labels[*idx].clone(),
                distance: *dist,
                similarity: 1.0 - (-dist).exp(),
            })
            .collect();

        KnnResult {
            predicted_type: predicted,
            confidence,
            neighbors,
            vote_distribution: votes,
        }
    }

    /// Find optimal k using cross-validation
    pub fn find_optimal_k(&self, k_values: &[usize], n_folds: usize) -> KnnCvResults {
        let _n_samples = self.train_features.nrows();

        // Try each k value
        let mut best_k = k_values[0];
        let mut best_accuracy = 0.0;

        for &k in k_values {
            let accuracy = self.cross_validate_k(k, n_folds);
            if accuracy > best_accuracy {
                best_accuracy = accuracy;
                best_k = k;
            }
        }

        // Run final CV with best k
        self.cross_validate_full(best_k, n_folds)
    }

    fn cross_validate_k(&self, k: usize, n_folds: usize) -> f64 {
        let n_samples = self.train_features.nrows();
        let fold_size = n_samples / n_folds;

        let mut total_correct = 0;
        let mut total_tested = 0;

        for fold in 0..n_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            for i in test_start..test_end {
                let query = self.train_features.row(i).to_owned();
                let true_label = &self.train_labels[i];

                let result = self.classify(&query, k);

                if &result.predicted_type == true_label {
                    total_correct += 1;
                }
                total_tested += 1;
            }
        }

        if total_tested > 0 {
            total_correct as f64 / total_tested as f64
        } else {
            0.0
        }
    }

    fn cross_validate_full(&self, k: usize, n_folds: usize) -> KnnCvResults {
        let n_samples = self.train_features.nrows();
        let fold_size = n_samples / n_folds;

        let mut fold_accuracies = Vec::new();
        let mut per_class_correct: HashMap<String, (usize, usize)> = HashMap::new();
        let mut confusion_matrix: HashMap<(String, String), usize> = HashMap::new();

        for fold in 0..n_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let mut fold_correct = 0;

            for i in test_start..test_end {
                let query = self.train_features.row(i).to_owned();
                let true_label = &self.train_labels[i];

                let result = self.classify(&query, k);

                let entry = per_class_correct.entry(true_label.clone()).or_default();
                entry.1 += 1;

                *confusion_matrix
                    .entry((result.predicted_type.clone(), true_label.clone()))
                    .or_default() += 1;

                if &result.predicted_type == true_label {
                    fold_correct += 1;
                    entry.0 += 1;
                }
            }

            fold_accuracies.push(fold_correct as f64 / (test_end - test_start) as f64);
        }

        let mean_accuracy = fold_accuracies.iter().sum::<f64>() / fold_accuracies.len() as f64;

        let per_class_accuracy: HashMap<String, f64> = per_class_correct
            .into_iter()
            .filter_map(|(label, (correct, total))| {
                if total > 0 {
                    Some((label, correct as f64 / total as f64))
                } else {
                    None
                }
            })
            .collect();

        // Convert confusion matrix to list format for JSON compatibility
        let confusion_list: Vec<ConfusionEntry> = confusion_matrix
            .into_iter()
            .map(|((predicted, actual), count)| ConfusionEntry {
                predicted,
                actual,
                count,
            })
            .collect();

        KnnCvResults {
            mean_accuracy,
            fold_accuracies,
            per_class_accuracy,
            confusion_matrix: confusion_list,
            optimal_k: k,
        }
    }

    /// Full cross-validation with confusion matrix
    pub fn cross_validate(&self, k: usize, n_folds: usize) -> KnnCvResults {
        self.cross_validate_full(k, n_folds)
    }
}

// =============================================================================
// Similarity Search Index
// =============================================================================

/// Index for fast similarity search
pub struct SimilarityIndex {
    features: Array2<f64>,
    file_names: Vec<String>,
    call_types: Vec<String>,
    engine: AcousticSimilarityEngine,
}

/// Result from similarity search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_name: String,
    pub call_type: String,
    pub distance: f64,
    pub similarity: f64,
    pub index: usize,
}

/// Analysis of a query's acoustic neighborhood
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborhoodAnalysis {
    pub n_neighbors: usize,
    pub radius: f64,
    pub type_distribution: HashMap<String, usize>,
    pub dominant_type: Option<String>,
    pub type_purity: f64,
}

impl SimilarityIndex {
    /// Create a new similarity index
    pub fn new(features: Array2<f64>, file_names: Vec<String>, call_types: Vec<String>) -> Self {
        let n_features = features.ncols();
        let mut engine = AcousticSimilarityEngine::new(n_features);
        engine.fit_normalization(&features);

        Self {
            features,
            file_names,
            call_types,
            engine,
        }
    }

    /// Find phrases most similar to query
    pub fn search(&self, query: &Array1<f64>, k: usize) -> Vec<SearchResult> {
        let candidates: Vec<_> = self.features.rows().into_iter().map(|r| r.to_owned()).collect();

        let distances = self.engine.find_similar(query, &candidates, k);

        let mut results: Vec<SearchResult> = distances
            .into_iter()
            .map(|(idx, dist)| {
                // Convert distance to similarity: exp(-dist) gives higher values for smaller distances
                let sim = (-dist).exp();
                SearchResult {
                    file_name: self.file_names[idx].clone(),
                    call_type: self.call_types[idx].clone(),
                    distance: dist,
                    similarity: sim,
                    index: idx,
                }
            })
            .collect();

        // Sort by similarity descending (highest similarity first)
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results
    }

    /// Find phrases similar to a specific file
    pub fn search_by_file(&self, file_name: &str, k: usize) -> Option<Vec<SearchResult>> {
        let idx = self.file_names.iter().position(|f| f == file_name)?;
        let query = self.features.row(idx).to_owned();

        let mut results = self.search(&query, k + 1);

        // Remove self from results
        results.retain(|r| r.file_name != file_name);
        results.truncate(k);

        Some(results)
    }

    /// Analyze a single phrase's acoustic neighborhood
    pub fn analyze_neighborhood(&self, query: &Array1<f64>, radius: f64) -> NeighborhoodAnalysis {
        let neighbors: Vec<_> = self
            .features
            .rows()
            .into_iter()
            .enumerate()
            .filter_map(|(i, row)| {
                let dist = self.engine.distance(&row.to_owned(), query);
                if dist <= radius {
                    Some((i, dist))
                } else {
                    None
                }
            })
            .collect();

        let n_neighbors = neighbors.len();

        let type_counts: HashMap<String, usize> =
            neighbors
                .iter()
                .map(|(i, _)| self.call_types[*i].clone())
                .fold(HashMap::new(), |mut acc, ct| {
                    *acc.entry(ct).or_default() += 1;
                    acc
                });

        let dominant_type = type_counts.iter().max_by_key(|(_, c)| *c).map(|(t, _)| t.clone());

        let type_purity = if n_neighbors > 0 {
            type_counts.values().max().copied().unwrap_or(0) as f64 / n_neighbors as f64
        } else {
            0.0
        };

        NeighborhoodAnalysis {
            n_neighbors,
            radius,
            type_distribution: type_counts,
            dominant_type,
            type_purity,
        }
    }

    /// Get reference to underlying similarity engine
    pub fn engine(&self) -> &AcousticSimilarityEngine {
        &self.engine
    }

    /// Get number of samples in index
    pub fn len(&self) -> usize {
        self.features.nrows()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.features.nrows() == 0
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_features() -> Array2<f64> {
        // Create simple test features: 4 samples, 5 dimensions
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, // Sample 0
            1.1, 2.1, 3.1, 4.1, 5.1, // Sample 1 (similar to 0)
            10.0, 20.0, 30.0, 40.0, 50.0, // Sample 2 (different)
            10.1, 20.1, 30.1, 40.1, 50.1, // Sample 3 (similar to 2)
        ];
        Array2::from_shape_vec((4, 5), data).unwrap()
    }

    #[test]
    fn test_similarity_engine_creation() {
        let engine = AcousticSimilarityEngine::new(30);
        assert_eq!(engine.weights().len(), 30);
        assert!(!engine.is_fitted());
    }

    #[test]
    fn test_normalization() {
        let features = create_test_features();
        let mut engine = AcousticSimilarityEngine::new(5);
        engine.fit_normalization(&features);
        assert!(engine.is_fitted());
    }

    #[test]
    fn test_distance_identical() {
        let engine = AcousticSimilarityEngine::new(5);
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let dist = engine.distance(&a, &b);
        assert!((dist - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_similar() {
        let features = create_test_features();
        let mut engine = AcousticSimilarityEngine::new(5);
        engine.fit_normalization(&features);

        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = Array1::from_vec(vec![1.1, 2.1, 3.1, 4.1, 5.1]);

        let dist_similar = engine.distance(&a, &b);

        let c = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
        let dist_different = engine.distance(&a, &c);

        assert!(dist_similar < dist_different);
    }

    #[test]
    fn test_similarity_values() {
        let engine = AcousticSimilarityEngine::new(5);
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let sim = engine.similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-10); // Identical = 0
    }

    #[test]
    fn test_find_similar() {
        let features = create_test_features();
        let mut engine = AcousticSimilarityEngine::new(5);
        engine.fit_normalization(&features);

        let query = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let candidates: Vec<_> = features.rows().into_iter().map(|r| r.to_owned()).collect();

        let similar = engine.find_similar(&query, &candidates, 2);

        assert_eq!(similar.len(), 2);
        assert_eq!(similar[0].0, 0); // First should be self (distance 0)
    }

    #[test]
    fn test_knn_classifier() {
        let features = create_test_features();
        let labels = vec!["A".to_string(), "A".to_string(), "B".to_string(), "B".to_string()];
        let files = vec!["f1".to_string(), "f2".to_string(), "f3".to_string(), "f4".to_string()];

        let classifier = KnnClassifier::new(features, labels, files);

        let query = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = classifier.classify(&query, 2);

        assert_eq!(result.predicted_type, "A");
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_similarity_index() {
        let features = create_test_features();
        let files = vec!["f1".to_string(), "f2".to_string(), "f3".to_string(), "f4".to_string()];
        let types = vec!["A".to_string(), "A".to_string(), "B".to_string(), "B".to_string()];

        let index = SimilarityIndex::new(features, files, types);

        let query = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let results = index.search(&query, 3);

        assert_eq!(results.len(), 3);
        assert!(results[0].similarity > results[1].similarity);
    }

    #[test]
    fn test_similarity_analysis() {
        let features = create_test_features();
        let labels = vec!["A".to_string(), "A".to_string(), "B".to_string(), "B".to_string()];
        let files = vec!["f1".to_string(), "f2".to_string(), "f3".to_string(), "f4".to_string()];
        let feature_names = vec!["f0", "f1", "f2", "f3", "f4"];

        let analysis = SimilarityAnalysis::analyze(&features, &labels, &files, &feature_names);

        // Between-type distance should be larger than within-type
        assert!(analysis.separation_ratio > 1.0);
    }
}

// ============================================================================
// PYTHON BINDINGS (PyO3)
// ============================================================================

#[cfg(feature = "python-bindings")]
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

/// Python wrapper for AcousticSimilarityEngine
#[cfg(feature = "python-bindings")]
#[pyclass(name = "AcousticSimilarityEngine")]
pub struct PyAcousticSimilarityEngine {
    inner: AcousticSimilarityEngine,
    feature_dim: usize,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl PyAcousticSimilarityEngine {
    #[new]
    #[args(feature_dim = 45)]
    /// Create a new AcousticSimilarityEngine
    ///
    /// Args:
    ///     feature_dim: Feature dimension (default: 45 for 45D features)
    fn new(feature_dim: usize) -> Self {
        Self {
            inner: AcousticSimilarityEngine::new(feature_dim),
            feature_dim,
        }
    }

    /// Create engine with specific distance metric
    ///
    /// Args:
    ///     feature_dim: Feature dimension
    ///     metric: Distance metric name ("WeightedEuclidean", "Cosine", "Manhattan", "Chebyshev")
    #[staticmethod]
    fn with_metric(feature_dim: usize, metric: &str) -> PyResult<Self> {
        let distance_metric = match metric {
            "WeightedEuclidean" | "weighted_euclidean" => DistanceMetric::WeightedEuclidean,
            "Cosine" | "cosine" => DistanceMetric::Cosine,
            "Manhattan" | "manhattan" => DistanceMetric::Manhattan,
            "Chebyshev" | "chebyshev" => DistanceMetric::Chebyshev,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown metric: {}. Valid options: WeightedEuclidean, Cosine, Manhattan, Chebyshev",
                    metric
                )))
            }
        };
        Ok(Self {
            inner: AcousticSimilarityEngine::with_metric(feature_dim, distance_metric),
            feature_dim,
        })
    }

    /// Fit normalization parameters from dataset
    ///
    /// Args:
    ///     features: 2D numpy array of shape (n_samples, n_features)
    fn fit_normalization(&mut self, features: PyReadonlyArray2<f64>) -> PyResult<()> {
        let array = features.as_array();
        let nrows = array.nrows();
        let ncols = array.ncols();

        let mut matrix = Array2::<f64>::zeros((nrows, ncols));
        for i in 0..nrows {
            for j in 0..ncols {
                matrix[[i, j]] = array[[i, j]];
            }
        }

        self.inner.fit_normalization(&matrix);
        Ok(())
    }

    /// Check if normalization has been fitted
    fn is_fitted(&self) -> bool {
        self.inner.is_fitted()
    }

    /// Compute similarity between two feature vectors
    ///
    /// Returns: 0.0 = identical, 1.0 = maximally different
    fn similarity(&self, a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let a_slice = a.as_slice()?;
        let b_slice = b.as_slice()?;
        let a_arr = Array1::from_vec(a_slice.to_vec());
        let b_arr = Array1::from_vec(b_slice.to_vec());
        Ok(self.inner.similarity(&a_arr, &b_arr))
    }

    /// Compute distance between two feature vectors
    ///
    /// Returns: Raw distance (lower = more similar)
    fn distance(&self, a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let a_slice = a.as_slice()?;
        let b_slice = b.as_slice()?;
        let a_arr = Array1::from_vec(a_slice.to_vec());
        let b_arr = Array1::from_vec(b_slice.to_vec());
        Ok(self.inner.distance(&a_arr, &b_arr))
    }

    /// Find k most similar candidates to query
    ///
    /// Args:
    ///     query: Query feature vector
    ///     candidates: 2D array of candidate features (n_candidates, n_features)
    ///     k: Number of nearest neighbors to return
    ///
    /// Returns: List of (index, distance) tuples sorted by distance
    fn find_similar<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f64>,
        candidates: PyReadonlyArray2<f64>,
        k: usize,
    ) -> PyResult<Vec<(usize, f64)>> {
        let query_slice = query.as_slice()?;
        let query_arr = Array1::from_vec(query_slice.to_vec());

        let candidates_array = candidates.as_array();
        let nrows = candidates_array.nrows();

        let candidates_vec: Vec<Array1<f64>> = (0..nrows)
            .map(|i| {
                let mut row = Vec::with_capacity(self.feature_dim);
                for j in 0..self.feature_dim.min(candidates_array.ncols()) {
                    row.push(candidates_array[[i, j]]);
                }
                Array1::from_vec(row)
            })
            .collect();

        let results = self.inner.find_similar(&query_arr, &candidates_vec, k);
        Ok(results)
    }

    /// Get feature weights
    fn get_weights<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f64>>> {
        let weights = self.inner.weights();
        Ok(PyArray1::from_vec(py, weights.to_vec()).into_py(py))
    }

    /// Get feature dimension
    fn feature_dim(&self) -> usize {
        self.feature_dim
    }
}
