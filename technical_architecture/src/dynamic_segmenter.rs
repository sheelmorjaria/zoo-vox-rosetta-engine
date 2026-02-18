//! Dynamic-Size Phrase Segmentation using Change Point Detection (CPD)
//!
//! This module implements acoustic boundary detection for discovering atomic phrases
//! in animal vocalizations. Instead of fixed-size windowing, it treats vocalizations
//! as continuous landscapes where boundaries are defined by acoustic change.
//!
//! Pipeline:
//! 1. Micro-Frame Extraction: Generate 45D vectors at high frame rate (100Hz)
//! 2. Distance Calculation: Compute acoustic distance between consecutive frames
//! 3. Change Point Detection: Identify peaks in distance curve = phrase boundaries
//! 4. Segment Aggregation: Average features within boundaries to create Phrase Candidates
//! 5. Atomic Discovery: Cluster candidates to find reusable "Atomic Phrases"
//!
//! # Species-Dependent Atomic Granularity
//!
//! Different species encode meaning at different hierarchical levels:
//! - **Zebra Finch**: Motifs (~350ms) are the carrier of meaning (song patterns)
//! - **Egyptian Bat**: Syllables (~32ms) are the carrier of meaning (chirp types)
//! - **Dolphin**: Contours (~500ms+) are the carrier of meaning (whistle shapes)

use crate::species::HierarchicalThresholds;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for dynamic segmentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicSegmenterConfig {
    /// Micro-frame duration in ms (default: 10ms for 100Hz extraction)
    pub frame_duration_ms: f32,
    /// Minimum phrase duration in ms (filters out noise/artifacts)
    pub min_phrase_duration_ms: f32,
    /// Maximum phrase duration in ms (splits overly long segments)
    pub max_phrase_duration_ms: f32,
    /// Sensitivity for detecting changes (0.0 - 1.0)
    /// Higher values = fewer segments (only big changes detected)
    pub change_threshold: f32,
    /// Smoothing window for distance calculation (in frames)
    pub smoothing_window: usize,
    /// Minimum peak prominence for boundary detection
    pub peak_prominence: f32,
    /// Feature dimension (default: 45)
    pub feature_dim: usize,
}

impl Default for DynamicSegmenterConfig {
    fn default() -> Self {
        Self {
            frame_duration_ms: 10.0,
            min_phrase_duration_ms: 30.0,
            max_phrase_duration_ms: 2000.0,
            change_threshold: 0.25,
            smoothing_window: 3,
            peak_prominence: 0.05,
            feature_dim: 45,
        }
    }
}

impl DynamicSegmenterConfig {
    /// Create config optimized for zebra finch songs
    pub fn zebra_finch() -> Self {
        Self {
            frame_duration_ms: 10.0,
            min_phrase_duration_ms: 20.0,   // Short syllables
            max_phrase_duration_ms: 500.0,  // Song motifs
            change_threshold: 0.30,
            smoothing_window: 3,
            peak_prominence: 0.08,
            feature_dim: 45,
        }
    }

    /// Create config optimized for dolphin whistles
    pub fn dolphin() -> Self {
        Self {
            frame_duration_ms: 10.0,
            min_phrase_duration_ms: 100.0,
            max_phrase_duration_ms: 3000.0, // Long whistles
            change_threshold: 0.20,         // Lower threshold for continuous changes
            smoothing_window: 5,
            peak_prominence: 0.03,
            feature_dim: 45,
        }
    }

    /// Create config optimized for marmoset calls
    pub fn marmoset() -> Self {
        Self {
            frame_duration_ms: 10.0,
            min_phrase_duration_ms: 30.0,
            max_phrase_duration_ms: 800.0,
            change_threshold: 0.35,
            smoothing_window: 3,
            peak_prominence: 0.06,
            feature_dim: 45,
        }
    }

    /// Create config optimized for bat vocalizations
    pub fn bat() -> Self {
        Self {
            frame_duration_ms: 5.0,          // Higher time resolution for FM sweeps
            min_phrase_duration_ms: 15.0,
            max_phrase_duration_ms: 400.0,
            change_threshold: 0.40,
            smoothing_window: 2,
            peak_prominence: 0.10,
            feature_dim: 45,
        }
    }

    /// Create MOTIF-level config from species-specific hierarchical thresholds
    ///
    /// Motifs are the highest level of organization (complete songs, call sequences).
    /// For songbirds like zebra finch, this is the ATOMIC level (carrier of meaning).
    pub fn for_motif_level(thresholds: &HierarchicalThresholds) -> Self {
        Self {
            frame_duration_ms: 10.0 * thresholds.tempo_factor,
            min_phrase_duration_ms: thresholds.motif_min_ms,
            max_phrase_duration_ms: thresholds.motif_max_ms,
            change_threshold: thresholds.motif_change_threshold,
            smoothing_window: if thresholds.tempo_factor < 0.5 { 2 } else { 3 },
            peak_prominence: 0.08 * thresholds.tempo_factor,
            feature_dim: 45,
        }
    }

    /// Create SYLLABLE-level config from species-specific hierarchical thresholds
    ///
    /// Syllables are discrete acoustic units within motifs.
    /// For bats and many mammals, this is the ATOMIC level (carrier of meaning).
    pub fn for_syllable_level(thresholds: &HierarchicalThresholds) -> Self {
        Self {
            frame_duration_ms: 10.0 * thresholds.tempo_factor,
            min_phrase_duration_ms: thresholds.syllable_min_ms,
            max_phrase_duration_ms: thresholds.syllable_max_ms,
            change_threshold: thresholds.syllable_change_threshold,
            smoothing_window: if thresholds.tempo_factor < 0.5 { 2 } else { 3 },
            peak_prominence: 0.06 * thresholds.tempo_factor,
            feature_dim: 45,
        }
    }

    /// Create NOTE-level config from species-specific hierarchical thresholds
    ///
    /// Notes are the smallest acoustic units (individual sound elements).
    /// For some species with simple calls, this may be the atomic level.
    pub fn for_note_level(thresholds: &HierarchicalThresholds) -> Self {
        Self {
            frame_duration_ms: 5.0 * thresholds.tempo_factor,
            min_phrase_duration_ms: thresholds.note_min_ms,
            max_phrase_duration_ms: thresholds.note_max_ms,
            change_threshold: thresholds.note_change_threshold,
            smoothing_window: if thresholds.tempo_factor < 0.5 { 1 } else { 2 },
            peak_prominence: 0.04 * thresholds.tempo_factor,
            feature_dim: 45,
        }
    }
}

/// Represents a single segmented phrase candidate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicPhraseCandidate {
    /// Unique identifier
    pub id: String,
    /// Start time in milliseconds
    pub start_ms: f32,
    /// End time in milliseconds
    pub end_ms: f32,
    /// Duration in milliseconds
    pub duration_ms: f32,
    /// The averaged 45D feature vector for this specific segment
    pub features: Vec<f64>,
    /// Source file this came from
    pub source_file: String,
    /// Index in the sequence of phrases from this source
    pub phrase_index: usize,
    /// Number of micro-frames in this phrase
    pub num_frames: usize,
    /// Variance of features within this phrase (internal coherence indicator)
    pub internal_variance: f32,
}

/// A phrase candidate with type assignment and grading information
///
/// This is the output of the clustering step, where raw candidates are
/// assigned to phrase types and annotated with grading scores.
///
/// # Grading System
///
/// For species with graded vocalizations (like marmosets), the phrase_type_id
/// alone may not capture the full meaning. The grading_score indicates how
/// far this instance is from the "perfect" example of its type.
///
/// - **Discrete Path** (low grading_score): Emit type ID only
/// - **Continuous Path** (high grading_score): Emit type ID + 45D vector
///
/// The Python Cognitive Agent can track grading_score trajectories to detect
/// emotional state changes even when phrase_type_id remains constant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedPhraseCandidate {
    /// Original phrase candidate
    pub candidate: DynamicPhraseCandidate,
    /// Assigned phrase type ID (e.g., "Type_1", "Type_2")
    pub phrase_type_id: String,
    /// Index of the assigned type in the type list
    pub type_index: usize,
    /// Distance from this instance to the type centroid (grading score)
    /// Low score = typical example, High score = outlier/graded instance
    pub grading_score: f32,
    /// Intra-type variance of the assigned type (from variance analysis)
    pub intra_type_variance: f32,
    /// Whether this instance is considered "graded" (outlier within its type)
    /// True if grading_score > 0.05 (threshold based on variance analysis)
    pub is_graded: bool,
    /// Confidence that this type assignment is correct
    pub assignment_confidence: f32,
}

impl TypedPhraseCandidate {
    /// Threshold for determining if a call is "graded"
    /// Based on variance analysis: avg intra-cluster distance was ~0.01 for discrete,
    /// ~0.07 for graded types
    pub const GRADING_THRESHOLD: f32 = 0.05;

    /// Create a new typed candidate from a raw candidate and type assignment
    pub fn new(
        candidate: DynamicPhraseCandidate,
        phrase_type_id: String,
        type_index: usize,
        centroid: &[f64],
        intra_type_variance: f32,
    ) -> Self {
        // Calculate distance from centroid (grading score)
        let grading_score = Self::cosine_distance(&candidate.features, centroid) as f32;

        // Determine if this is a graded instance
        let is_graded = grading_score > Self::GRADING_THRESHOLD;

        // Assignment confidence is inverse of grading score
        let assignment_confidence = (1.0 - grading_score.min(1.0)).max(0.0);

        Self {
            candidate,
            phrase_type_id,
            type_index,
            grading_score,
            intra_type_variance,
            is_graded,
            assignment_confidence,
        }
    }

    /// Calculate cosine distance between two vectors
    fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 1.0;
        }

        let similarity = dot / (mag_a * mag_b);
        1.0 - similarity
    }

    /// Check if the assigned type is a "discrete" type (low variance)
    pub fn is_discrete_type(&self) -> bool {
        self.intra_type_variance < Self::GRADING_THRESHOLD
    }

    /// Get the emission strategy for this candidate
    ///
    /// Returns what should be emitted to the Python Cognitive Agent:
    /// - `EmissionStrategy::Discrete`: Type ID only (sufficient for discrete types)
    /// - `EmissionStrategy::Continuous`: Type ID + 45D vector (needed for graded types)
    pub fn emission_strategy(&self) -> EmissionStrategy {
        if self.is_discrete_type() && !self.is_graded {
            EmissionStrategy::Discrete
        } else {
            EmissionStrategy::Continuous
        }
    }
}

/// Strategy for emitting phrase information to the Python Cognitive Agent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmissionStrategy {
    /// Emit type ID only (sufficient for discrete types with low variance)
    /// Output: Event { type: "Type_1", confidence: 0.999 }
    Discrete,
    /// Emit type ID + 45D vector (needed for graded types to track trajectories)
    /// Output: Event { type: "Type_2", grading_score: 0.06, vector: [...] }
    Continuous,
}

/// Change point detected in the acoustic stream
#[derive(Debug, Clone)]
pub struct ChangePoint {
    /// Frame index where change occurs
    pub frame_idx: usize,
    /// Time in milliseconds
    pub time_ms: f32,
    /// Distance value at this change point
    pub distance: f32,
    /// Prominence of this change point
    pub prominence: f32,
}

/// Result of dynamic segmentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationResult {
    /// Phrase candidates extracted
    pub candidates: Vec<DynamicPhraseCandidate>,
    /// Change points detected
    pub num_change_points: usize,
    /// Average phrase duration
    pub avg_duration_ms: f32,
    /// Total audio duration processed
    pub total_duration_ms: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
}

/// Main segmenter implementation
pub struct DynamicSegmenter {
    config: DynamicSegmenterConfig,
    sample_rate: u32,
}

impl DynamicSegmenter {
    /// Create a new dynamic segmenter with the given configuration
    pub fn new(config: DynamicSegmenterConfig, sample_rate: u32) -> Self {
        Self { config, sample_rate }
    }

    /// Create with default configuration
    pub fn with_defaults(sample_rate: u32) -> Self {
        Self::new(DynamicSegmenterConfig::default(), sample_rate)
    }

    /// Create for zebra finch
    pub fn for_zebra_finch(sample_rate: u32) -> Self {
        Self::new(DynamicSegmenterConfig::zebra_finch(), sample_rate)
    }

    /// Main entry point: segments audio based on 45D feature dynamics
    ///
    /// # Arguments
    /// * `audio` - Audio samples
    /// * `extractor` - Function to extract 45D features from a frame
    /// * `source_id` - Identifier for the source file
    pub fn segment<F>(
        &self,
        audio: &[f32],
        extract_features: F,
        source_id: &str,
    ) -> SegmentationResult
    where
        F: Fn(&[f32], u32) -> Option<Vec<f64>>,
    {
        let start_time = std::time::Instant::now();

        // 1. Extract micro-frames (stream of 45D vectors)
        let (feature_stream, frame_times) = self.extract_feature_stream(audio, &extract_features);

        if feature_stream.is_empty() {
            return SegmentationResult {
                candidates: vec![],
                num_change_points: 0,
                avg_duration_ms: 0.0,
                total_duration_ms: (audio.len() as f32 / self.sample_rate as f32) * 1000.0,
                processing_time_ms: start_time.elapsed().as_millis() as f32,
            };
        }

        // 2. Calculate distance curve (Derivative of features)
        let distance_curve = self.compute_distance_curve(&feature_stream);

        // 3. Smooth the distance curve
        let smoothed_curve = self.smooth_distance_curve(&distance_curve);

        // 4. Find boundaries using Peak Detection
        let change_points = self.find_change_points(&smoothed_curve);

        // 5. Convert boundaries to Phrase Candidates
        let candidates = self.change_points_to_candidates(
            &change_points,
            &feature_stream,
            &frame_times,
            source_id,
        );

        // Calculate statistics
        let total_duration_ms = (audio.len() as f32 / self.sample_rate as f32) * 1000.0;
        let avg_duration_ms = if candidates.is_empty() {
            0.0
        } else {
            candidates.iter().map(|c| c.duration_ms).sum::<f32>() / candidates.len() as f32
        };

        SegmentationResult {
            candidates,
            num_change_points: change_points.len(),
            avg_duration_ms,
            total_duration_ms,
            processing_time_ms: start_time.elapsed().as_millis() as f32,
        }
    }

    /// Extract a stream of feature vectors at high frame rate
    fn extract_feature_stream<F>(
        &self,
        audio: &[f32],
        extract_features: F,
    ) -> (Vec<Vec<f64>>, Vec<f32>)
    where
        F: Fn(&[f32], u32) -> Option<Vec<f64>>,
    {
        let frame_samples = ((self.config.frame_duration_ms / 1000.0) * self.sample_rate as f32) as usize;
        let hop_samples = frame_samples / 2; // 50% overlap

        let mut features = Vec::new();
        let mut times = Vec::new();

        let mut frame_idx = 0;
        for start in (0..audio.len().saturating_sub(frame_samples)).step_by(hop_samples) {
            let end = (start + frame_samples).min(audio.len());
            let frame = &audio[start..end];

            if let Some(feat) = extract_features(frame, self.sample_rate) {
                if feat.len() == self.config.feature_dim {
                    let time_ms = (start as f32 / self.sample_rate as f32) * 1000.0;
                    features.push(feat);
                    times.push(time_ms);
                    frame_idx += 1;
                }
            }
        }

        (features, times)
    }

    /// Computes Cosine Distance between adjacent feature vectors
    fn compute_distance_curve(&self, features: &[Vec<f64>]) -> Vec<f32> {
        (0..features.len().saturating_sub(1))
            .map(|i| cosine_distance(&features[i], &features[i + 1]))
            .collect()
    }

    /// Apply smoothing to the distance curve
    fn smooth_distance_curve(&self, distances: &[f32]) -> Vec<f32> {
        let window = self.config.smoothing_window;
        if window == 0 || distances.len() <= window * 2 {
            return distances.to_vec();
        }

        let mut smoothed = Vec::with_capacity(distances.len());

        for i in 0..distances.len() {
            let start = i.saturating_sub(window);
            let end = (i + window + 1).min(distances.len());
            let sum: f32 = distances[start..end].iter().sum();
            smoothed.push(sum / (end - start) as f32);
        }

        smoothed
    }

    /// Detect peaks in the distance curve that exceed the threshold
    fn find_change_points(&self, distances: &[f32]) -> Vec<ChangePoint> {
        if distances.is_empty() {
            return vec![];
        }

        let threshold = self.config.change_threshold;
        let prominence = self.config.peak_prominence;
        let window = self.config.smoothing_window.max(1);

        let mut change_points = Vec::new();

        // Find local maxima that exceed threshold
        for i in window..distances.len().saturating_sub(window) {
            let is_local_max = distances[i] > distances[i - 1] && distances[i] >= distances[i + 1];
            let exceeds_threshold = distances[i] > threshold;

            // Calculate prominence (how much this peak stands out from neighbors)
            let local_min_left = distances[i.saturating_sub(window * 2)..i]
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            let local_min_right = distances[i + 1..distances.len().min(i + window * 2 + 1)]
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            let peak_prominence = distances[i] - local_min_left.min(local_min_right);

            let has_prominence = peak_prominence > prominence;

            if is_local_max && exceeds_threshold && has_prominence {
                let time_ms = (i as f32 + 0.5) * self.config.frame_duration_ms;
                change_points.push(ChangePoint {
                    frame_idx: i,
                    time_ms,
                    distance: distances[i],
                    prominence: peak_prominence,
                });
            }
        }

        change_points
    }

    /// Convert change points to phrase candidates
    fn change_points_to_candidates(
        &self,
        change_points: &[ChangePoint],
        features: &[Vec<f64>],
        frame_times: &[f32],
        source_id: &str,
    ) -> Vec<DynamicPhraseCandidate> {
        let mut candidates = Vec::new();

        // Create boundaries from change points
        let mut boundaries = vec![0];
        for cp in change_points {
            boundaries.push(cp.frame_idx);
        }
        boundaries.push(features.len().saturating_sub(1));

        // Create candidates from each boundary window
        let mut phrase_idx = 0;
        for window in boundaries.windows(2) {
            let start_frame = window[0];
            let end_frame = window[1];

            if start_frame >= end_frame || end_frame > features.len() {
                continue;
            }

            let start_ms = frame_times.get(start_frame).copied().unwrap_or(0.0);
            let end_ms = frame_times.get(end_frame - 1).copied().unwrap_or(start_ms + self.config.frame_duration_ms);
            let duration_ms = end_ms - start_ms + self.config.frame_duration_ms;

            // Filter by duration constraints
            if duration_ms < self.config.min_phrase_duration_ms {
                continue;
            }

            // Split overly long segments
            if duration_ms > self.config.max_phrase_duration_ms {
                // Split into multiple phrases
                let num_splits = (duration_ms / self.config.max_phrase_duration_ms).ceil() as usize;
                let frames_per_split = (end_frame - start_frame) / num_splits;

                for split_idx in 0..num_splits {
                    let split_start = start_frame + split_idx * frames_per_split;
                    let split_end = if split_idx == num_splits - 1 {
                        end_frame
                    } else {
                        start_frame + (split_idx + 1) * frames_per_split
                    };

                    if split_start >= split_end || split_end > features.len() {
                        continue;
                    }

                    let split_start_ms = frame_times.get(split_start).copied().unwrap_or(0.0);
                    let split_end_ms = frame_times.get(split_end - 1).copied().unwrap_or(split_start_ms);
                    let split_duration = split_end_ms - split_start_ms + self.config.frame_duration_ms;

                    if split_duration >= self.config.min_phrase_duration_ms {
                        if let Some(candidate) = self.create_candidate(
                            split_start,
                            split_end,
                            features,
                            frame_times,
                            source_id,
                            phrase_idx,
                        ) {
                            candidates.push(candidate);
                            phrase_idx += 1;
                        }
                    }
                }
            } else {
                // Create single candidate
                if let Some(candidate) = self.create_candidate(
                    start_frame,
                    end_frame,
                    features,
                    frame_times,
                    source_id,
                    phrase_idx,
                ) {
                    candidates.push(candidate);
                    phrase_idx += 1;
                }
            }
        }

        candidates
    }

    /// Create a single phrase candidate from a range of frames
    fn create_candidate(
        &self,
        start_frame: usize,
        end_frame: usize,
        features: &[Vec<f64>],
        frame_times: &[f32],
        source_id: &str,
        phrase_idx: usize,
    ) -> Option<DynamicPhraseCandidate> {
        if start_frame >= end_frame || end_frame > features.len() {
            return None;
        }

        let segment = &features[start_frame..end_frame];
        if segment.is_empty() {
            return None;
        }

        // Average features across the segment
        let avg_features = average_features(segment);

        // Calculate internal variance
        let variance = calculate_internal_variance(segment, &avg_features);

        let start_ms = frame_times.get(start_frame).copied().unwrap_or(0.0);
        let end_ms = frame_times.get(end_frame - 1).copied().unwrap_or(start_ms);
        let duration_ms = end_ms - start_ms + self.config.frame_duration_ms;

        Some(DynamicPhraseCandidate {
            id: format!("{}_phrase{}", source_id.replace(".wav", ""), phrase_idx),
            start_ms,
            end_ms,
            duration_ms,
            features: avg_features,
            source_file: source_id.to_string(),
            phrase_index: phrase_idx,
            num_frames: end_frame - start_frame,
            internal_variance: variance,
        })
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Compute cosine distance between two feature vectors
/// Returns 0.0 for identical vectors, up to 2.0 for opposite vectors
pub fn cosine_distance(v1: &[f64], v2: &[f64]) -> f32 {
    if v1.len() != v2.len() || v1.is_empty() {
        return 0.0;
    }

    let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let mag1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag1 < 1e-10 || mag2 < 1e-10 {
        return 0.0;
    }

    let similarity = dot / (mag1 * mag2);
    // Clamp to [-1, 1] to handle numerical errors
    let similarity = similarity.clamp(-1.0, 1.0);

    (1.0 - similarity) as f32 // Range [0, 2], where 0 is identical
}

/// Compute Euclidean distance between two feature vectors
pub fn euclidean_distance(v1: &[f64], v2: &[f64]) -> f32 {
    if v1.len() != v2.len() {
        return f32::MAX;
    }

    v1.iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt() as f32
}

/// Average features across multiple frames
fn average_features(features: &[Vec<f64>]) -> Vec<f64> {
    if features.is_empty() {
        return vec![0.0; 45];
    }

    let dim = features[0].len();
    let mut avg = vec![0.0; dim];

    for frame in features {
        for (i, &val) in frame.iter().enumerate() {
            if i < dim {
                avg[i] += val;
            }
        }
    }

    let n = features.len() as f64;
    for val in &mut avg {
        *val /= n;
    }

    avg
}

/// Calculate internal variance of features within a segment
fn calculate_internal_variance(features: &[Vec<f64>], avg: &[f64]) -> f32 {
    if features.len() <= 1 {
        return 0.0;
    }

    let dim = avg.len();
    let mut total_var = 0.0;

    for frame in features {
        for (i, &val) in frame.iter().enumerate() {
            if i < dim {
                total_var += (val - avg[i]).powi(2);
            }
        }
    }

    (total_var / (features.len() * dim) as f64) as f32
}

// ============================================================================
// ATOMIC PHRASE ANALYZER
// ============================================================================

/// Analyzer for discovering atomic phrase types from candidates
pub struct AtomicPhraseAnalyzer {
    /// Similarity threshold for clustering
    similarity_threshold: f32,
    /// Minimum occurrences to be considered atomic
    min_occurrences: usize,
    /// Feature dimension
    feature_dim: usize,
}

impl AtomicPhraseAnalyzer {
    /// Create new analyzer
    pub fn new(similarity_threshold: f32, min_occurrences: usize) -> Self {
        Self {
            similarity_threshold,
            min_occurrences,
            feature_dim: 45,
        }
    }

    /// Cluster candidates to discover atomic phrase types
    pub fn discover_atomic_types(
        &self,
        candidates: &[DynamicPhraseCandidate],
    ) -> Vec<AtomicPhraseType> {
        if candidates.is_empty() {
            return vec![];
        }

        // Compute pairwise distance matrix
        let dist_matrix = self.compute_pairwise_distances(candidates);

        // Cluster using threshold-based approach
        let clusters = self.cluster_by_threshold(&dist_matrix, candidates.len());

        // Convert clusters to AtomicPhraseTypes
        self.clusters_to_types(clusters, candidates)
    }

    /// Compute pairwise cosine distance matrix
    fn compute_pairwise_distances(&self, candidates: &[DynamicPhraseCandidate]) -> Vec<Vec<f32>> {
        let n = candidates.len();
        let mut matrix = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = cosine_distance(&candidates[i].features, &candidates[j].features);
                matrix[i][j] = dist;
                matrix[j][i] = dist;
            }
        }

        matrix
    }

    /// Cluster candidates using threshold-based approach
    fn cluster_by_threshold(&self, dist_matrix: &[Vec<f32>], n: usize) -> Vec<Vec<usize>> {
        let mut clusters: Vec<Vec<usize>> = vec![];
        let mut assigned = vec![false; n];

        for i in 0..n {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![i];
            assigned[i] = true;

            // Find all neighbors similar to i
            for j in (i + 1)..n {
                if !assigned[j] && dist_matrix[i][j] < self.similarity_threshold {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }

            // Only keep clusters meeting minimum occurrence requirement
            if cluster.len() >= self.min_occurrences {
                clusters.push(cluster);
            }
        }

        clusters
    }

    /// Convert clusters to atomic phrase types
    fn clusters_to_types(
        &self,
        clusters: Vec<Vec<usize>>,
        candidates: &[DynamicPhraseCandidate],
    ) -> Vec<AtomicPhraseType> {
        clusters
            .into_iter()
            .enumerate()
            .map(|(type_id, cluster)| {
                let members: Vec<DynamicPhraseCandidate> =
                    cluster.iter().map(|&idx| candidates[idx].clone()).collect();

                // Compute centroid
                let centroid = average_features(
                    &members.iter().map(|m| m.features.clone()).collect::<Vec<_>>(),
                );

                // Compute quality metrics
                let intra_similarity = compute_intra_cluster_similarity(&members, &centroid);
                let avg_duration = members.iter().map(|m| m.duration_ms).sum::<f32>() / members.len() as f32;

                AtomicPhraseType {
                    type_id,
                    centroid,
                    members,
                    intra_similarity,
                    avg_duration_ms: avg_duration,
                    occurrence_count: cluster.len(),
                }
            })
            .collect()
    }
}

/// Represents a discovered atomic phrase type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicPhraseType {
    /// Unique type identifier
    pub type_id: usize,
    /// Centroid feature vector
    pub centroid: Vec<f64>,
    /// Member phrases
    pub members: Vec<DynamicPhraseCandidate>,
    /// Internal coherence (average similarity of members to centroid)
    pub intra_similarity: f32,
    /// Average duration of phrases in this type
    pub avg_duration_ms: f32,
    /// Number of occurrences
    pub occurrence_count: usize,
}

/// Compute average similarity of cluster members to centroid
fn compute_intra_cluster_similarity(members: &[DynamicPhraseCandidate], centroid: &[f64]) -> f32 {
    if members.is_empty() {
        return 0.0;
    }

    let total_sim: f32 = members
        .iter()
        .map(|m| 1.0 - cosine_distance(&m.features, centroid))
        .sum();

    total_sim / members.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let dist = cosine_distance(&v, &v);
        assert!((dist - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];
        let dist = cosine_distance(&v1, &v2);
        assert!((dist - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![-1.0, 0.0];
        let dist = cosine_distance(&v1, &v2);
        assert!((dist - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_average_features() {
        let features = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let avg = average_features(&features);
        assert!((avg[0] - 3.0).abs() < 0.001);
        assert!((avg[1] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_dynamic_segmenter_config_defaults() {
        let config = DynamicSegmenterConfig::default();
        assert_eq!(config.frame_duration_ms, 10.0);
        assert_eq!(config.min_phrase_duration_ms, 30.0);
        assert_eq!(config.change_threshold, 0.25);
    }

    #[test]
    fn test_dynamic_segmenter_config_species() {
        let zf = DynamicSegmenterConfig::zebra_finch();
        assert!(zf.min_phrase_duration_ms < 50.0);

        let dol = DynamicSegmenterConfig::dolphin();
        assert!(dol.max_phrase_duration_ms > 2000.0);
    }
}
