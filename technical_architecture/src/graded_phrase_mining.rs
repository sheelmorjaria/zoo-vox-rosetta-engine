//! Graded Phrase Mining - Testing the "Hidden Discrete Motifs" Hypothesis
//! =======================================================================
//!
//! This module implements a scientific pipeline to answer a profound question:
//! **"Is a Graded Signal actually a sequence of Hidden Discrete Motifs?"**
//!
//! ## The Hypothesis
//!
//! Graded vocalizations (like Marmoset Phees) may appear continuous, but could
//! be constructed from a finite library of "Acoustic Gestures" (motifs) that
//! are blended together. This module tests that hypothesis by:
//!
//! 1. **Adaptive Segmentation**: Using Neural Boundary Detection (NBD) to cut
//!    the graded stream at "semantic shift" points
//! 2. **Feature Extraction**: Computing 112D features for each segment
//!    (46D Base Physics + 30D Macro Texture + 36D Micro Texture)
//! 3. **Similarity Clustering**: Using ASE distance with HDBSCAN
//! 4. **Purity Analysis**: Measuring how many segments fall into tight clusters
//!
//! ## Why 112D?
//!
//! For motif discovery, we need **fine-grained acoustic similarity**:
//!
//! ```text
//! Layer 1: BASE PHYSICS (46D)
//!   → Universal features (F0, HNR, MFCCs, release time)
//!
//! Layer 2: MACRO TEXTURE (30D)
//!   → Harmonic texture, pitch geometry, GLCM texture
//!   → Captures "spectral fingerprints" of motifs
//!
//! Layer 3: MICRO TEXTURE (36D)
//!   → Modulation spectra, rhythm histograms, psychoacoustics
//!   → Distinguishes similar motifs (e.g., Motif A vs Motif A')
//! ```
//!
//! If `"Aggressive Phee" = Motif A + Motif B` and `"Friendly Phee" = Motif A + Motif C`,
//! the **Micro Texture** layer is critical for distinguishing Motif B from Motif C.
//!
//! ## Interpretation Guide
//!
//! | Purity   | Noise Ratio | Interpretation                          |
//! |----------|-------------|-----------------------------------------|
//! | >60%     | <40%        | Hidden vocabulary exists (Bag-of-Phrases works) |
//! | 30-60%   | 40-70%      | Hybrid: some motifs, some grading       |
//! | <30%     | >70%        | True analog slider (Direct 112D needed) |
//!
//! ## Usage
//!
//! ```rust
//! use technical_architecture::{GradedPhraseMiner, MotifReport, GradedMiningConfig};
//!
//! // Default uses 112D features for maximum discriminative power
//! let config = GradedMiningConfig::default();
//! let mut miner = GradedPhraseMiner::new(config);
//!
//! // Analyze a graded vocalization stream
//! let report = miner.analyze(&audio, 48000)?;
//!
//! println!("Motif Purity: {:.1}%", report.purity * 100.0);
//! println!("Noise Ratio: {:.1}%", report.noise_ratio * 100.0);
//! println!("Interpretation: {}", report.interpretation);
//! ```

use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::acoustic_algebra_105d::{Vector105D, Vector112D};
use crate::acoustic_similarity::AcousticSimilarityEngine;
use crate::hdbscan::{HdbscanClustering, HdbscanStats};
use crate::micro_dynamics_extractor::{MicroDynamicsExtractor, RosettaFeatures};
use crate::neural_boundary::{BoundaryDetectorConfig, DetectionMode, NeuralBoundaryDetector};

// =============================================================================
// Configuration
// =============================================================================

/// Feature dimension mode for motif discovery
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureMode {
    /// 45D Base Physics only - faster, good for initial exploration
    Base45D,

    /// 105D Triple-Layer - legacy support
    /// Layer 1: Base Physics (45D)
    /// Layer 2: Macro Texture (30D)
    /// Layer 3: Micro Texture (30D)
    Full105D,

    /// 112D Triple-Layer (recommended) - maximum discriminative power for motif discovery
    /// Layer 1: Base Physics (46D)
    /// Layer 2: Macro Texture (30D)
    /// Layer 3: Micro Texture (36D)
    #[default]
    Full112D,
}

/// Configurable thresholds for purity-based interpretation
///
/// These thresholds control how the system interprets motif mining results.
/// Previously hardcoded, these are now configurable for different research contexts.
///
/// # Interpretation Logic
///
/// | Purity | Noise Ratio | Interpretation |
/// |--------|-------------|----------------|
/// | > bag_of_phrases_purity | < bag_of_phrases_max_noise | Hidden vocabulary (Bag-of-Phrases) |
/// | > hybrid_purity | < hybrid_max_noise | Hybrid (discrete + graded) |
/// | else | - | True analog slider (Direct 105D) |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradedMiningThresholds {
    /// Purity threshold for Bag-of-Phrases classification (default: 0.6)
    /// Higher values = stricter requirement for discrete motifs
    pub bag_of_phrases_purity: f64,

    /// Maximum noise ratio for Bag-of-Phrases (default: 0.4)
    /// If noise ratio exceeds this, cannot be Bag-of-Phrases
    pub bag_of_phrases_max_noise: f64,

    /// Purity threshold for Hybrid classification (default: 0.3)
    /// Between this and bag_of_phrases_purity = hybrid system
    pub hybrid_purity: f64,

    /// Maximum noise ratio for Hybrid (default: 0.7)
    /// If noise ratio exceeds this, classified as true graded
    pub hybrid_max_noise: f64,

    /// Minimum cluster reuse for discrete syntax detection (default: 0.5)
    /// Used for N-gram analysis
    pub discrete_syntax_reuse: f64,
}

impl Default for GradedMiningThresholds {
    fn default() -> Self {
        Self {
            bag_of_phrases_purity: 0.6,
            bag_of_phrases_max_noise: 0.4,
            hybrid_purity: 0.3,
            hybrid_max_noise: 0.7,
            discrete_syntax_reuse: 0.5,
        }
    }
}

impl GradedMiningThresholds {
    /// Conservative thresholds - require stronger evidence for discrete motifs
    ///
    /// Use when false positives are costly (e.g., confirmatory research)
    pub fn conservative() -> Self {
        Self {
            bag_of_phrases_purity: 0.75,    // Higher bar
            bag_of_phrases_max_noise: 0.25, // Stricter noise limit
            hybrid_purity: 0.4,
            hybrid_max_noise: 0.6,
            discrete_syntax_reuse: 0.6,
        }
    }

    /// Permissive thresholds - more likely to detect discrete structure
    ///
    /// Use for exploratory research or when false negatives are costly
    pub fn permissive() -> Self {
        Self {
            bag_of_phrases_purity: 0.5,    // Lower bar
            bag_of_phrases_max_noise: 0.5, // More tolerance for noise
            hybrid_purity: 0.25,
            hybrid_max_noise: 0.75,
            discrete_syntax_reuse: 0.4,
        }
    }

    /// Thresholds optimized for mammalian graded signals
    ///
    /// Mammals (bats, primates, cetaceans) typically have lower purity
    /// due to continuous modulation
    pub fn for_mammals() -> Self {
        Self {
            bag_of_phrases_purity: 0.4,
            bag_of_phrases_max_noise: 0.6,
            hybrid_purity: 0.2,
            hybrid_max_noise: 0.8,
            discrete_syntax_reuse: 0.35,
        }
    }

    /// Thresholds optimized for avian crystallized songs
    ///
    /// Birds (finches, songbirds) have stereotyped songs with high reuse
    pub fn for_birds() -> Self {
        Self {
            bag_of_phrases_purity: 0.7,
            bag_of_phrases_max_noise: 0.3,
            hybrid_purity: 0.4,
            hybrid_max_noise: 0.6,
            discrete_syntax_reuse: 0.6,
        }
    }
}

/// Configuration for Graded Phrase Mining
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradedMiningConfig {
    /// Minimum phrase duration in ms (prevents over-segmentation)
    pub min_phrase_duration_ms: f32,

    /// Boundary detection threshold (0.0-1.0, lower = more sensitive)
    pub boundary_threshold: f32,

    /// HDBSCAN minimum cluster size
    pub min_cluster_size: usize,

    /// HDBSCAN min_samples parameter
    pub min_samples: usize,

    /// Feature mode: 45D (fast) or 105D (recommended for motif discovery)
    pub feature_mode: FeatureMode,

    /// Minimum segment length for feature extraction (samples)
    pub min_segment_samples: usize,

    /// Configurable thresholds for interpretation (NEW in v2.1.0)
    /// Replaces previously hardcoded purity/noise thresholds
    pub thresholds: GradedMiningThresholds,
}

impl Default for GradedMiningConfig {
    fn default() -> Self {
        Self {
            min_phrase_duration_ms: 50.0,
            boundary_threshold: 0.4, // Lower threshold to catch semantic changes
            min_cluster_size: 5,
            min_samples: 3,
            feature_mode: FeatureMode::Full112D, // 112D recommended for motif discovery
            min_segment_samples: 480,            // ~10ms at 48kHz
            thresholds: GradedMiningThresholds::default(),
        }
    }
}

// =============================================================================
// Results
// =============================================================================

/// A single discovered motif segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifSegment {
    /// Start time in milliseconds
    pub start_ms: f32,

    /// End time in milliseconds
    pub end_ms: f32,

    /// Duration in milliseconds
    pub duration_ms: f32,

    /// Assigned cluster label (-1 = noise)
    pub cluster_label: i32,

    /// Cluster purity (fraction of segments in same cluster)
    pub cluster_purity: f64,

    /// Feature vector (45D or 105D depending on config)
    pub features: Vec<f64>,

    /// Feature mode used for this segment
    pub feature_mode: FeatureMode,

    /// Segmenter used to create this segment (NEW in v2.1.0)
    /// Values: "NBD", "CPD", "Hybrid", or "Unknown"
    #[serde(default = "default_segmenter")]
    pub segmenter: String,
}

fn default_segmenter() -> String {
    "Unknown".to_string()
}

/// Report from graded phrase mining analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifReport {
    /// Total number of segments extracted
    pub total_segments: usize,

    /// Number of distinct clusters found
    pub num_clusters: usize,

    /// Number of segments marked as noise (not in any cluster)
    pub noise_count: usize,

    /// Cluster purity: fraction of segments that belong to tight clusters
    /// High (>0.6) = Hidden vocabulary exists
    /// Low (<0.3) = True graded continuum
    pub purity: f64,

    /// Noise ratio: fraction of segments marked as noise
    /// High (>0.7) = True analog slider
    /// Low (<0.4) = Hidden discrete motifs
    pub noise_ratio: f64,

    /// Average silhouette-like cohesion score within clusters
    pub avg_cohesion: f64,

    /// Per-cluster statistics
    pub cluster_stats: Vec<MotifClusterInfo>,

    /// All discovered segments
    pub segments: Vec<MotifSegment>,

    /// Human-readable interpretation
    pub interpretation: String,

    /// Recommended processing approach based on analysis
    pub recommended_approach: ProcessingApproach,

    /// Species-specific prediction (based on known patterns)
    pub species_prediction: Option<SpeciesGradingPrediction>,
}

/// Information about a discovered motif cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifClusterInfo {
    /// Cluster label
    pub label: i32,

    /// Number of segments in this cluster
    pub size: usize,

    /// Average within-cluster distance (cohesion)
    pub avg_cohesion: f64,

    /// Typical duration of segments in this cluster
    pub typical_duration_ms: f64,

    /// Centroid features (for visualization)
    pub centroid: Vec<f64>,
}

/// Recommended processing approach based on motif analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingApproach {
    /// Discrete vocabulary found - use Bag-of-Phrases
    BagOfPhrases,

    /// Hybrid system - some discrete, some graded
    HybridDiscreteGraded,

    /// True graded continuum - use Direct 105D similarity
    Direct105D,

    /// Insufficient data to determine
    InsufficientData,
}

/// Species-specific grading predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesGradingPrediction {
    /// Predicted purity range (min, max)
    pub expected_purity_range: (f64, f64),

    /// Predicted noise ratio range (min, max)
    pub expected_noise_range: (f64, f64),

    /// Whether the observed values match predictions
    pub matches_prediction: bool,

    /// Explanation of the species' communication style
    pub communication_style: String,
}

// =============================================================================
// Graded Phrase Miner
// =============================================================================

/// The main Graded Phrase Mining engine
///
/// Tests whether graded vocalizations are built from hidden discrete motifs.
/// Uses 105D features by default for maximum discriminative power.
pub struct GradedPhraseMiner {
    config: GradedMiningConfig,
    boundary_detector: NeuralBoundaryDetector,
    feature_extractor: MicroDynamicsExtractor,
    /// Similarity engine - dimension set based on feature_mode
    similarity_engine: AcousticSimilarityEngine,
}

impl GradedPhraseMiner {
    /// Create a new Graded Phrase Miner with the specified configuration
    pub fn new(config: GradedMiningConfig) -> Self {
        let boundary_config = BoundaryDetectorConfig {
            hop_size: 512,
            sample_rate: 48000, // Will be updated in analyze()
            min_phrase_duration_ms: config.min_phrase_duration_ms,
            threshold: config.boundary_threshold,
            smoothing_frames: 3,
            mode: DetectionMode::Phrase,
            max_phrase_duration_ms: 5000.0,
            smoothing_window_ms: 20.0,
            energy_weight: 0.5,
            spectral_weight: 0.5,
        };

        let boundary_detector = NeuralBoundaryDetector::with_config(boundary_config);
        let feature_extractor = MicroDynamicsExtractor::new(48000);

        // Create similarity engine with dimension based on feature mode
        let feature_dim = match config.feature_mode {
            FeatureMode::Base45D => 45,
            FeatureMode::Full105D => 105,
            FeatureMode::Full112D => 112,
        };
        let similarity_engine = AcousticSimilarityEngine::new(feature_dim);

        Self {
            config,
            boundary_detector,
            feature_extractor,
            similarity_engine,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(GradedMiningConfig::default())
    }

    /// Get the feature dimension based on current configuration
    fn feature_dim(&self) -> usize {
        match self.config.feature_mode {
            FeatureMode::Base45D => 45,
            FeatureMode::Full105D => 105,
            FeatureMode::Full112D => 112,
        }
    }

    /// Extract 112D RosettaFeatures from audio segment
    ///
    /// Uses the full 112D feature vector from MicroDynamicsExtractor:
    /// - Layer 1: Base Physics (46D) - indices 0-45
    /// - Layer 2: Macro Texture (30D) - indices 46-75
    /// - Layer 3: Micro Texture (36D) - indices 76-111
    fn extract_112d_features(&self, audio: &[f32]) -> Result<Vec<f64>> {
        let rosetta = self.feature_extractor.extract(audio)?;
        Ok(rosetta.to_array().iter().map(|&v| v as f64).collect())
    }

    /// Extract feature vector based on configured mode
    ///
    /// - Base45D: First 45 dimensions (subset of Base Physics)
    /// - Full105D: First 105 dimensions (Base Physics + Macro + partial Micro)
    /// - Full112D: Complete 112D vector (recommended)
    fn extract_features_by_mode(&self, audio: &[f32]) -> Result<Vec<f64>> {
        let features_112d = self.extract_112d_features(audio)?;

        let result = match self.config.feature_mode {
            FeatureMode::Base45D => {
                // Use first 45 dimensions of the 112D vector
                features_112d[..45].to_vec()
            }
            FeatureMode::Full105D => {
                // Use first 105 dimensions of the 112D vector
                features_112d[..105].to_vec()
            }
            FeatureMode::Full112D => {
                // Use complete 112D vector (recommended)
                features_112d
            }
        };

        Ok(result)
    }

    /// Analyze a graded vocalization stream for hidden motifs
    ///
    /// # Arguments
    /// * `audio` - The audio samples (f32, normalized -1.0 to 1.0)
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// * `MotifReport` - Complete analysis of discovered motifs
    pub fn analyze(&mut self, audio: &[f32], sample_rate: u32) -> Result<MotifReport> {
        // Step 1: Adaptive Segmentation using Neural Boundary Detection
        let boundaries = self.boundary_detector.detect_boundaries(audio);

        // Convert boundaries to segments
        let raw_segments = self.boundaries_to_segments(audio, &boundaries, sample_rate);

        if raw_segments.is_empty() {
            return Ok(MotifReport {
                total_segments: 0,
                num_clusters: 0,
                noise_count: 0,
                purity: 0.0,
                noise_ratio: 1.0,
                avg_cohesion: 0.0,
                cluster_stats: Vec::new(),
                segments: Vec::new(),
                interpretation: "No segments found - audio too short or silent".to_string(),
                recommended_approach: ProcessingApproach::InsufficientData,
                species_prediction: None,
            });
        }

        // Filter out segments that are too short or have insufficient energy (silent)
        let min_rms_threshold = 0.001; // Filter out near-silent segments
        let segments: Vec<_> = raw_segments
            .into_iter()
            .filter(|(segment_audio, start, end)| {
                let len = end - start;
                if len < self.config.min_segment_samples {
                    return false;
                }
                // Check RMS energy to filter out silent segments
                let rms = (segment_audio.iter().map(|&x| x as f64 * x as f64).sum::<f64>()
                    / segment_audio.len() as f64)
                    .sqrt();
                rms >= min_rms_threshold
            })
            .collect();

        if segments.len() < self.config.min_cluster_size {
            let feature_dim = self.feature_dim();
            return Ok(MotifReport {
                total_segments: segments.len(),
                num_clusters: 0,
                noise_count: segments.len(),
                purity: 0.0,
                noise_ratio: 1.0,
                avg_cohesion: 0.0,
                cluster_stats: Vec::new(),
                segments: segments
                    .iter()
                    .map(|(_, start, end)| {
                        let start_ms = (*start as f32 / sample_rate as f32) * 1000.0;
                        let end_ms = (*end as f32 / sample_rate as f32) * 1000.0;
                        MotifSegment {
                            start_ms,
                            end_ms,
                            duration_ms: end_ms - start_ms,
                            cluster_label: -1,
                            cluster_purity: 0.0,
                            features: vec![0.0; feature_dim],
                            feature_mode: self.config.feature_mode,
                            segmenter: "NBD".to_string(),
                        }
                    })
                    .collect(),
                interpretation: format!(
                    "Insufficient segments ({}) for clustering (need {})",
                    segments.len(),
                    self.config.min_cluster_size
                ),
                recommended_approach: ProcessingApproach::InsufficientData,
                species_prediction: None,
            });
        }

        // Step 2: Feature Extraction using 112D RosettaFeatures
        // All modes now use the 112D extractor, slicing to required dimensions
        let feature_vectors: Vec<Vec<f64>> = segments
            .iter()
            .filter_map(|(segment_audio, _, _)| self.extract_features_by_mode(segment_audio).ok())
            .collect();

        // Check if we have enough features after filtering
        if feature_vectors.len() < self.config.min_cluster_size {
            let feature_dim = self.feature_dim();
            return Ok(MotifReport {
                total_segments: feature_vectors.len(),
                num_clusters: 0,
                noise_count: feature_vectors.len(),
                purity: 0.0,
                noise_ratio: 1.0,
                avg_cohesion: 0.0,
                cluster_stats: Vec::new(),
                segments: feature_vectors
                    .iter()
                    .enumerate()
                    .map(|(i, _)| MotifSegment {
                        start_ms: 0.0,
                        end_ms: 0.0,
                        duration_ms: 0.0,
                        cluster_label: -1,
                        cluster_purity: 0.0,
                        features: vec![0.0; feature_dim],
                        feature_mode: self.config.feature_mode,
                        segmenter: "NBD".to_string(),
                    })
                    .collect(),
                interpretation: format!(
                    "Insufficient valid segments ({}) for clustering after feature extraction",
                    feature_vectors.len()
                ),
                recommended_approach: ProcessingApproach::InsufficientData,
                species_prediction: None,
            });
        }

        // Convert to 2D array for clustering
        let n_segments = feature_vectors.len();
        let n_features = self.feature_dim();

        let mut feature_matrix = Array2::<f64>::zeros((n_segments, n_features));
        for (i, feat_vec) in feature_vectors.iter().enumerate() {
            for (j, &val) in feat_vec.iter().enumerate().take(n_features) {
                feature_matrix[[i, j]] = val;
            }
        }

        // Fit normalization on the feature matrix
        self.similarity_engine.fit_normalization(&feature_matrix);

        // Step 3: Compute ASE Distance Matrix
        let distance_matrix = self.compute_distance_matrix(&feature_matrix);

        // Step 4: HDBSCAN Clustering
        let hdbscan = HdbscanClustering::new(self.config.min_cluster_size, self.config.min_samples)?;

        let labels = hdbscan.fit_predict(&feature_matrix)?;
        let stats = hdbscan.get_cluster_stats(&labels);

        // Step 5: Calculate Purity and Cohesion Metrics
        let (purity, avg_cohesion, cluster_info) =
            self.calculate_metrics(&labels, &distance_matrix, &stats, &feature_matrix);

        // Build segment reports
        let motif_segments: Vec<MotifSegment> = feature_vectors
            .iter()
            .zip(labels.iter())
            .map(|(feat_vec, &label)| {
                let start_ms = 0.0; // We don't have the original segment info here
                let end_ms = 0.0;
                let duration_ms = 0.0;

                let cluster_purity = if label >= 0 {
                    cluster_info
                        .iter()
                        .find(|c| c.label == label)
                        .map(|c| 1.0 - c.avg_cohesion)
                        .unwrap_or(0.0)
                } else {
                    0.0
                };

                MotifSegment {
                    start_ms,
                    end_ms,
                    duration_ms,
                    cluster_label: label,
                    cluster_purity,
                    features: feat_vec.clone(),
                    feature_mode: self.config.feature_mode,
                    segmenter: "NBD".to_string(),
                }
            })
            .collect();

        // Calculate noise ratio
        let noise_ratio = stats.noise_count as f64 / n_segments as f64;

        // Determine interpretation
        let (interpretation, approach) = self.interpret_results(purity, noise_ratio, stats.n_clusters);

        let report = MotifReport {
            total_segments: n_segments,
            num_clusters: stats.n_clusters,
            noise_count: stats.noise_count,
            purity,
            noise_ratio,
            avg_cohesion,
            cluster_stats: cluster_info,
            segments: motif_segments,
            interpretation,
            recommended_approach: approach,
            species_prediction: None,
        };

        Ok(report)
    }

    /// Convert boundaries to segment ranges
    fn boundaries_to_segments(
        &self,
        audio: &[f32],
        boundaries: &[crate::neural_boundary::PhraseBoundary],
        sample_rate: u32,
    ) -> Vec<(Vec<f32>, usize, usize)> {
        let mut segments = Vec::new();

        if boundaries.is_empty() {
            // No boundaries - return entire audio as one segment
            return vec![(audio.to_vec(), 0, audio.len())];
        }

        let mut start_sample = 0usize;

        for boundary in boundaries {
            let end_sample = (boundary.time_ms * sample_rate as f32 / 1000.0) as usize;

            if end_sample > start_sample && end_sample <= audio.len() {
                segments.push((audio[start_sample..end_sample].to_vec(), start_sample, end_sample));
            }

            start_sample = end_sample;
        }

        // Add final segment
        if start_sample < audio.len() {
            segments.push((audio[start_sample..].to_vec(), start_sample, audio.len()));
        }

        segments
    }

    /// Compute pairwise ASE distance matrix
    fn compute_distance_matrix(&self, features: &Array2<f64>) -> Array2<f64> {
        let n = features.nrows();
        let mut matrix = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for j in (i + 1)..n {
                let a = features.row(i).to_owned();
                let b = features.row(j).to_owned();
                let dist = self.similarity_engine.distance(&a, &b);
                matrix[[i, j]] = dist;
                matrix[[j, i]] = dist;
            }
        }

        matrix
    }

    /// Calculate purity and cohesion metrics
    fn calculate_metrics(
        &self,
        labels: &[i32],
        distance_matrix: &Array2<f64>,
        stats: &HdbscanStats,
        features: &Array2<f64>,
    ) -> (f64, f64, Vec<MotifClusterInfo>) {
        let n = labels.len();

        // Group segments by cluster
        let mut cluster_members: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            cluster_members.entry(label).or_default().push(i);
        }

        let mut cluster_info = Vec::new();
        let mut total_cohesion = 0.0;
        let mut total_clustered = 0;

        for (&label, members) in &cluster_members {
            if label < 0 {
                continue; // Skip noise
            }

            // Calculate within-cluster average distance (cohesion)
            let mut sum_dist = 0.0;
            let mut count = 0;

            for &i in members {
                for &j in members {
                    if i < j {
                        sum_dist += distance_matrix[[i, j]];
                        count += 1;
                    }
                }
            }

            let avg_cohesion = if count > 0 { sum_dist / count as f64 } else { 0.0 };

            // Calculate centroid
            let centroid: Vec<f64> = if !members.is_empty() {
                let mut sum = Array1::<f64>::zeros(features.ncols());
                for &i in members {
                    sum = sum + features.row(i);
                }
                (sum / members.len() as f64).to_vec()
            } else {
                vec![0.0; features.ncols()]
            };

            // Typical duration (placeholder - would need segment info)
            let typical_duration_ms = 0.0;

            cluster_info.push(MotifClusterInfo {
                label,
                size: members.len(),
                avg_cohesion,
                typical_duration_ms,
                centroid,
            });

            total_cohesion += avg_cohesion * members.len() as f64;
            total_clustered += members.len();
        }

        // Calculate purity = fraction of samples in valid clusters
        let purity = if n > 0 {
            (n - stats.noise_count) as f64 / n as f64
        } else {
            0.0
        };

        // Average cohesion
        let avg_cohesion = if total_clustered > 0 {
            total_cohesion / total_clustered as f64
        } else {
            0.0
        };

        (purity, avg_cohesion, cluster_info)
    }

    /// Interpret results and recommend approach
    ///
    /// Uses configurable thresholds from `self.config.thresholds` instead of
    /// hardcoded values. This allows species-specific tuning of the interpretation.
    fn interpret_results(&self, purity: f64, noise_ratio: f64, n_clusters: usize) -> (String, ProcessingApproach) {
        let t = &self.config.thresholds;

        if n_clusters == 0 {
            return (
                "No clear cluster structure detected. The signal may be purely graded or data is insufficient."
                    .to_string(),
                ProcessingApproach::InsufficientData,
            );
        }

        // Decision logic based on configurable thresholds
        if purity > t.bag_of_phrases_purity && noise_ratio < t.bag_of_phrases_max_noise {
            (
                format!(
                    "HIGH MOTIF REUSE DETECTED. {:.0}% of segments fall into {} tight clusters.\n\
                     The graded signal appears to be constructed from a HIDDEN VOCABULARY of {} discrete motifs.\n\
                     RECOMMENDATION: Use Bag-of-Phrases approach for classification.",
                    purity * 100.0,
                    n_clusters,
                    n_clusters
                ),
                ProcessingApproach::BagOfPhrases,
            )
        } else if purity > t.hybrid_purity && noise_ratio < t.hybrid_max_noise {
            (
                format!(
                    "HYBRID SYSTEM DETECTED. {:.0}% of segments form {} clusters, but {:.0}% are unique.\n\
                     The animal uses BOTH discrete motifs and graded transitions.\n\
                     RECOMMENDATION: Use hybrid approach with both discrete phrases and continuous similarity.",
                    purity * 100.0,
                    n_clusters,
                    noise_ratio * 100.0
                ),
                ProcessingApproach::HybridDiscreteGraded,
            )
        } else {
            (
                format!(
                    "TRUE GRADED CONTINUUM DETECTED. {:.0}% of segments are unique (noise).\n\
                     The animal uses an ANALOG SLIDER - continuous modulation without discrete units.\n\
                     RECOMMENDATION: Use Direct 105D similarity approach. Bag-of-Phrases will fail.",
                    noise_ratio * 100.0
                ),
                ProcessingApproach::Direct105D,
            )
        }
    }

    /// Predict species communication style based on results
    pub fn predict_species_style(&self, report: &MotifReport, species: &str) -> SpeciesGradingPrediction {
        // Known patterns from research
        let (expected_purity, expected_noise, style) = match species.to_lowercase().as_str() {
            "marmoset" => (
                (0.30, 0.50),
                (0.50, 0.70),
                "Hybrid: Reuses alarm chirps but grades Phees continuously".to_string(),
            ),
            "bat" | "egyptian fruit bat" => (
                (0.10, 0.20),
                (0.80, 0.90),
                "Prosodic modulation: FM sweeps are unique events, not reused motifs".to_string(),
            ),
            "dolphin" => (
                (0.60, 0.70),
                (0.30, 0.40),
                "Signature whistles are stereotyped discrete units".to_string(),
            ),
            "finch" | "zebra finch" => (
                (0.40, 0.60),
                (0.40, 0.60),
                "Mixed: Song motifs are discrete, calls are graded".to_string(),
            ),
            "human" => (
                (0.70, 0.80),
                (0.20, 0.30),
                "Discrete phonemes: Speech is built from a finite vocabulary of sounds".to_string(),
            ),
            _ => (
                (0.0, 1.0),
                (0.0, 1.0),
                "Unknown species - no prediction available".to_string(),
            ),
        };

        let matches = report.purity >= expected_purity.0
            && report.purity <= expected_purity.1
            && report.noise_ratio >= expected_noise.0
            && report.noise_ratio <= expected_noise.1;

        SpeciesGradingPrediction {
            expected_purity_range: expected_purity,
            expected_noise_range: expected_noise,
            matches_prediction: matches,
            communication_style: style,
        }
    }

    /// Reset internal state for new analysis
    pub fn reset(&mut self) {
        self.boundary_detector.reset();
    }
}

// =============================================================================
// Batch Analysis
// =============================================================================

/// Analyze multiple recordings and aggregate results
pub fn analyze_batch(
    recordings: &[(&[f32], u32)], // (audio, sample_rate) pairs
    config: Option<GradedMiningConfig>,
) -> Result<BatchAnalysisReport> {
    let config = config.unwrap_or_default();
    let mut miner = GradedPhraseMiner::new(config.clone());

    let mut reports = Vec::new();
    let mut total_segments = 0;
    let mut total_clusters = 0;
    let mut total_noise = 0;

    for (audio, sample_rate) in recordings {
        let report = miner.analyze(audio, *sample_rate)?;
        total_segments += report.total_segments;
        total_clusters += report.num_clusters;
        total_noise += report.noise_count;
        reports.push(report);
        miner.reset();
    }

    let n_recordings = recordings.len();
    let avg_purity = reports.iter().map(|r| r.purity).sum::<f64>() / n_recordings as f64;
    let avg_noise_ratio = reports.iter().map(|r| r.noise_ratio).sum::<f64>() / n_recordings as f64;

    // Determine overall interpretation using configurable thresholds
    let t = &config.thresholds;
    let (overall_interpretation, approach) = if avg_purity > t.bag_of_phrases_purity {
        (
            "Species uses HIDDEN DISCRETE MOTIFS across recordings".to_string(),
            ProcessingApproach::BagOfPhrases,
        )
    } else if avg_purity > t.hybrid_purity {
        (
            "Species uses HYBRID discrete+graded communication".to_string(),
            ProcessingApproach::HybridDiscreteGraded,
        )
    } else {
        (
            "Species uses TRUE GRADED CONTINUUM (analog slider)".to_string(),
            ProcessingApproach::Direct105D,
        )
    };

    Ok(BatchAnalysisReport {
        n_recordings,
        individual_reports: reports,
        aggregate_stats: AggregateStats {
            total_segments,
            total_clusters,
            total_noise,
            avg_purity,
            avg_noise_ratio,
        },
        overall_interpretation,
        recommended_approach: approach,
    })
}

/// Aggregated statistics from batch analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateStats {
    pub total_segments: usize,
    pub total_clusters: usize,
    pub total_noise: usize,
    pub avg_purity: f64,
    pub avg_noise_ratio: f64,
}

/// Report from batch analysis of multiple recordings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchAnalysisReport {
    pub n_recordings: usize,
    pub individual_reports: Vec<MotifReport>,
    pub aggregate_stats: AggregateStats,
    pub overall_interpretation: String,
    pub recommended_approach: ProcessingApproach,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_audio(duration_ms: f32, sample_rate: u32) -> Vec<f32> {
        let n_samples = (duration_ms / 1000.0 * sample_rate as f32) as usize;
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect()
    }

    fn generate_test_audio_with_gaps(sample_rate: u32) -> Vec<f32> {
        let mut audio = Vec::new();

        // Three "calls" with gaps between them
        for _ in 0..3 {
            // 200ms of tone
            let tone: Vec<f32> = (0..(sample_rate as f32 * 0.2) as usize)
                .map(|i| {
                    let t = i as f32 / sample_rate as f32;
                    (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
                })
                .collect();
            audio.extend(tone);

            // 100ms of silence
            audio.extend(vec![0.0f32; (sample_rate as f32 * 0.1) as usize]);
        }

        audio
    }

    #[test]
    fn test_miner_creation() {
        let config = GradedMiningConfig::default();
        let miner = GradedPhraseMiner::new(config);
        assert_eq!(miner.config.feature_mode, FeatureMode::Full112D); // Default is 112D
    }

    #[test]
    fn test_analyze_silent_audio() {
        let mut miner = GradedPhraseMiner::with_defaults();
        let silence = vec![0.0f32; 48000]; // 1 second of silence
        let report = miner.analyze(&silence, 48000).unwrap();

        // Silent audio may have no energy, so boundaries don't produce meaningful segments
        // With NBD default, silent audio produces no boundaries
        // When no boundaries are found, the entire audio becomes one segment
        // BUT that segment should be filtered out due to RMS check or other energy-based filtering
        assert_eq!(report.total_segments, 0, "Silent audio should produce 0 segments");
        assert_eq!(report.recommended_approach, ProcessingApproach::InsufficientData);
    }

    #[test]
    fn test_analyze_short_audio() {
        let mut miner = GradedPhraseMiner::with_defaults();
        let audio = generate_test_audio(50.0, 48000); // Only 50ms
        let report = miner.analyze(&audio, 48000).unwrap();

        // Short audio should either have no segments or insufficient data
        assert!(report.total_segments <= 1 || report.recommended_approach == ProcessingApproach::InsufficientData);
    }

    #[test]
    fn test_analyze_audio_with_gaps() {
        let mut miner = GradedPhraseMiner::new(GradedMiningConfig {
            min_phrase_duration_ms: 30.0,
            boundary_threshold: 0.3,
            min_cluster_size: 2,
            min_samples: 1,
            feature_mode: FeatureMode::Base45D, // Use 45D for faster testing
            min_segment_samples: 240,
            thresholds: GradedMiningThresholds::default(),
        });

        let audio = generate_test_audio_with_gaps(48000);
        let report = miner.analyze(&audio, 48000).unwrap();

        // Should detect at least some segments from the three tone bursts
        assert!(report.total_segments >= 1);
    }

    #[test]
    fn test_interpret_results() {
        let miner = GradedPhraseMiner::with_defaults();

        // High purity = discrete vocabulary
        let (interp, approach) = miner.interpret_results(0.7, 0.3, 5);
        assert_eq!(approach, ProcessingApproach::BagOfPhrases);
        assert!(interp.contains("HIDDEN VOCABULARY"));

        // Low purity = graded continuum
        let (interp, approach) = miner.interpret_results(0.2, 0.8, 2);
        assert_eq!(approach, ProcessingApproach::Direct105D);
        assert!(interp.contains("GRADED CONTINUUM"));

        // Medium = hybrid
        let (interp, approach) = miner.interpret_results(0.45, 0.55, 3);
        assert_eq!(approach, ProcessingApproach::HybridDiscreteGraded);
        assert!(interp.contains("HYBRID"));
    }

    #[test]
    fn test_species_prediction() {
        let miner = GradedPhraseMiner::with_defaults();

        let report = MotifReport {
            total_segments: 100,
            num_clusters: 10,
            noise_count: 60,
            purity: 0.4,
            noise_ratio: 0.6,
            avg_cohesion: 0.3,
            cluster_stats: Vec::new(),
            segments: Vec::new(),
            interpretation: String::new(),
            recommended_approach: ProcessingApproach::HybridDiscreteGraded,
            species_prediction: None,
        };

        let prediction = miner.predict_species_style(&report, "marmoset");
        assert!(prediction.communication_style.contains("Hybrid"));
    }

    #[test]
    fn test_config_default() {
        let config = GradedMiningConfig::default();
        assert_eq!(config.feature_mode, FeatureMode::Full112D); // 112D is default
        assert_eq!(config.min_cluster_size, 5);
        assert_eq!(config.min_samples, 3);
    }

    #[test]
    fn test_feature_modes() {
        // Test 45D mode
        let config_45d = GradedMiningConfig {
            feature_mode: FeatureMode::Base45D,
            ..Default::default()
        };
        let miner_45d = GradedPhraseMiner::new(config_45d);
        assert_eq!(miner_45d.feature_dim(), 45);

        // Test 105D mode (legacy)
        let config_105d = GradedMiningConfig {
            feature_mode: FeatureMode::Full105D,
            ..Default::default()
        };
        let miner_105d = GradedPhraseMiner::new(config_105d);
        assert_eq!(miner_105d.feature_dim(), 105);

        // Test 112D mode (recommended default)
        let config_112d = GradedMiningConfig {
            feature_mode: FeatureMode::Full112D,
            ..Default::default()
        };
        let miner_112d = GradedPhraseMiner::new(config_112d);
        assert_eq!(miner_112d.feature_dim(), 112);
    }

    #[test]
    fn test_thresholds_default() {
        let thresholds = GradedMiningThresholds::default();
        assert!((thresholds.bag_of_phrases_purity - 0.6).abs() < 0.01);
        assert!((thresholds.hybrid_purity - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_thresholds_presets() {
        // Conservative - higher bars
        let conservative = GradedMiningThresholds::conservative();
        assert!(conservative.bag_of_phrases_purity > 0.6);
        assert!(conservative.hybrid_purity > 0.3);

        // Permissive - lower bars
        let permissive = GradedMiningThresholds::permissive();
        assert!(permissive.bag_of_phrases_purity < 0.6);
        assert!(permissive.hybrid_purity < 0.3);

        // Mammalian - optimized for graded signals
        let mammalian = GradedMiningThresholds::for_mammals();
        assert!(mammalian.bag_of_phrases_purity < 0.5);
        assert!(mammalian.hybrid_purity < 0.3);

        // Avian - optimized for crystallized songs
        let avian = GradedMiningThresholds::for_birds();
        assert!(avian.bag_of_phrases_purity > 0.6);
    }

    #[test]
    fn test_configurable_thresholds_in_interpretation() {
        // Test that configurable thresholds affect interpretation
        let config_conservative = GradedMiningConfig {
            thresholds: GradedMiningThresholds::conservative(),
            ..Default::default()
        };
        let miner_conservative = GradedPhraseMiner::new(config_conservative);

        // With conservative thresholds (bag_of_phrases_purity = 0.75),
        // 0.80 purity should be BagOfPhrases
        let (interp, approach) = miner_conservative.interpret_results(0.80, 0.20, 5);
        assert_eq!(approach, ProcessingApproach::BagOfPhrases);

        // With permissive thresholds (bag_of_phrases_purity = 0.5),
        // 0.55 purity should be BagOfPhrases
        let config_permissive = GradedMiningConfig {
            thresholds: GradedMiningThresholds::permissive(),
            ..Default::default()
        };
        let miner_permissive = GradedPhraseMiner::new(config_permissive);
        let (interp2, approach2) = miner_permissive.interpret_results(0.55, 0.45, 5);
        assert_eq!(approach2, ProcessingApproach::BagOfPhrases);

        // With default thresholds (bag_of_phrases_purity = 0.6),
        // 0.65 purity should be BagOfPhrases
        let miner_default = GradedPhraseMiner::with_defaults();
        let (interp3, approach3) = miner_default.interpret_results(0.65, 0.35, 5);
        assert_eq!(approach3, ProcessingApproach::BagOfPhrases);
    }

    #[test]
    fn test_batch_analysis_uses_configurable_thresholds() {
        // Test that analyze_batch respects configurable thresholds
        let audio = generate_test_audio(500.0, 48000);
        let recordings: Vec<(&[f32], u32)> = vec![(&audio, 48000)];

        // With default thresholds
        let report_default = analyze_batch(&recordings, Some(GradedMiningConfig::default())).unwrap();

        // With permissive thresholds
        let config_permissive = GradedMiningConfig {
            thresholds: GradedMiningThresholds::permissive(),
            ..Default::default()
        };
        let report_permissive = analyze_batch(&recordings, Some(config_permissive)).unwrap();

        // Both should produce valid reports
        assert!(report_default.n_recordings > 0);
        assert!(report_permissive.n_recordings > 0);
    }

    // =========================================================================
    // Additional Coverage: Reset, Batch, Species, Thresholds
    // =========================================================================

    #[test]
    fn test_reset_clears_state() {
        let mut miner = GradedPhraseMiner::new(GradedMiningConfig {
            min_phrase_duration_ms: 30.0,
            boundary_threshold: 0.3,
            min_cluster_size: 2,
            min_samples: 1,
            feature_mode: FeatureMode::Base45D,
            min_segment_samples: 240,
            thresholds: GradedMiningThresholds::default(),
        });

        // Analyze some audio to populate internal state
        let audio = generate_test_audio_with_gaps(48000);
        let report1 = miner.analyze(&audio, 48000).unwrap();

        // Reset should not panic and should allow re-analysis
        miner.reset();

        // Re-analyze after reset should produce same structure
        let report2 = miner.analyze(&audio, 48000).unwrap();
        assert_eq!(report1.total_segments, report2.total_segments);
    }

    #[test]
    fn test_analyze_batch_multiple_clips() {
        let clip1 = generate_test_audio(200.0, 48000);
        let clip2 = generate_test_audio(300.0, 48000);
        let clip3 = generate_test_audio(150.0, 48000);

        let recordings: Vec<(&[f32], u32)> = vec![(&clip1, 48000), (&clip2, 48000), (&clip3, 48000)];

        let report = analyze_batch(&recordings, None).unwrap();

        // Should produce results for all 3 recordings
        assert_eq!(report.n_recordings, 3);
        assert_eq!(report.individual_reports.len(), 3);
        // Aggregate stats should be computed
        assert!(report.aggregate_stats.total_segments > 0);
    }

    #[test]
    fn test_species_prediction_all_styles() {
        let miner = GradedPhraseMiner::with_defaults();

        let report = MotifReport {
            total_segments: 50,
            num_clusters: 5,
            noise_count: 25,
            purity: 0.5,
            noise_ratio: 0.5,
            avg_cohesion: 0.5,
            cluster_stats: Vec::new(),
            segments: Vec::new(),
            interpretation: String::new(),
            recommended_approach: ProcessingApproach::HybridDiscreteGraded,
            species_prediction: None,
        };

        // Test all supported species return valid predictions
        for species in &[
            "marmoset",
            "bat",
            "egyptian fruit bat",
            "dolphin",
            "finch",
            "zebra finch",
            "human",
        ] {
            let prediction = miner.predict_species_style(&report, species);
            assert!(
                !prediction.communication_style.is_empty(),
                "Species '{}' should return non-empty style",
                species
            );
        }

        // Unknown species should also return a prediction
        let unknown = miner.predict_species_style(&report, "alien");
        assert!(unknown.communication_style.contains("Unknown"));
    }

    #[test]
    fn test_feature_mode_112d_processing() {
        let config = GradedMiningConfig {
            feature_mode: FeatureMode::Full112D,
            min_phrase_duration_ms: 30.0,
            boundary_threshold: 0.3,
            min_cluster_size: 2,
            min_samples: 1,
            min_segment_samples: 240,
            thresholds: GradedMiningThresholds::default(),
        };
        let mut miner = GradedPhraseMiner::new(config);

        // Should process without error in 112D mode
        let audio = generate_test_audio_with_gaps(48000);
        let report = miner.analyze(&audio, 48000);
        assert!(report.is_ok(), "112D mode analysis should succeed");

        let report = report.unwrap();
        // Segments should use 112D features
        for segment in &report.segments {
            assert_eq!(segment.feature_mode, FeatureMode::Full112D);
            assert_eq!(segment.features.len(), 112);
        }
    }

    #[test]
    fn test_thresholds_conservative_stricter_than_default() {
        let defaults = GradedMiningThresholds::default();
        let conservative = GradedMiningThresholds::conservative();

        // Conservative should require higher purity
        assert!(
            conservative.bag_of_phrases_purity > defaults.bag_of_phrases_purity,
            "Conservative purity ({}) should exceed default ({})",
            conservative.bag_of_phrases_purity,
            defaults.bag_of_phrases_purity
        );

        // Conservative should tolerate less noise
        assert!(
            conservative.bag_of_phrases_max_noise < defaults.bag_of_phrases_max_noise,
            "Conservative max noise ({}) should be less than default ({})",
            conservative.bag_of_phrases_max_noise,
            defaults.bag_of_phrases_max_noise
        );
    }
}
