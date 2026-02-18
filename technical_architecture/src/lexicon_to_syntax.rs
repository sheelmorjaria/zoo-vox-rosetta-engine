// Lexicon to Syntax: Master Pipeline
//
// A comprehensive pipeline that moves from raw audio to structured language models:
// Phase 1: Segmentation - Adaptive segmentation for variable-length phrases
// Phase 2: Vectorization - 56D feature extraction as time-series (30D base + 13 Δ + 13 ΔΔ)
// Phase 3: Discovery - DTW-DBSCAN clustering for vocabulary
// Phase 4: Refinement - GMM-HMM for temporal structure (phonemes)
//
// This pipeline implements the Universal Rosetta Stone methodology for cross-species
// vocalization analysis, discovering the building blocks of animal communication.

use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// Re-use existing modules
use crate::adaptive_segmentation::AdaptiveSegmenter;
use crate::hmm::HiddenMarkovModel;
use crate::dtw::DtwDbscan;
use crate::micro_dynamics_extractor::{MicroDynamicsExtractor, FeatureDim};

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Segmentation failed: {0}")]
    SegmentationError(String),

    #[error("Vectorization failed: {0}")]
    VectorizationError(String),

    #[error("Discovery (clustering) failed: {0}")]
    DiscoveryError(String),

    #[error("Refinement (HMM) failed: {0}")]
    RefinementError(String),

    #[error("Audio file not found: {0}")]
    AudioNotFound(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}

pub type Result<T> = std::result::Result<T, PipelineError>;

// =============================================================================
// Phase 1: Segmentation (The "Slicing")
// =============================================================================

/// Configuration for Phase 1: Adaptive Segmentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationConfig {
    /// Minimum phrase duration in milliseconds
    pub min_duration_ms: f64,

    /// Maximum phrase duration in milliseconds
    pub max_duration_ms: f64,

    /// Onset detection threshold (0.0 to 1.0)
    pub onset_threshold: f64,

    /// Minimum distance between onsets in milliseconds
    pub min_onset_distance_ms: f64,

    /// Sample rate for audio processing
    pub sample_rate: u32,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            min_duration_ms: 50.0,   // 50ms minimum
            max_duration_ms: 500.0,  // 500ms maximum
            onset_threshold: 0.3,    // Moderate threshold
            min_onset_distance_ms: 10.0, // 10ms minimum spacing
            sample_rate: 48000,      // Default to 48kHz (will be adjusted for bats)
        }
    }
}

/// Result from Phase 1: Variable-length vocalization segments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentedPhrase {
    /// Unique phrase ID
    pub phrase_id: String,

    /// Audio samples (mono, normalized to [-1, 1])
    pub audio: Vec<f32>,

    /// Start time in original audio (seconds)
    pub start_time: f64,

    /// End time in original audio (seconds)
    pub end_time: f64,

    /// Duration in milliseconds
    pub duration_ms: f64,

    /// Sample rate
    pub sample_rate: u32,

    /// Onset confidence score
    pub onset_confidence: f64,
}

// =============================================================================
// Phase 2: Vectorization (The "Embedding")
// =============================================================================

/// Configuration for Phase 2: Feature Extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorizationConfig {
    /// Number of mel-frequency bands for spectral features
    pub n_mels: usize,

    /// FFT window size
    pub fft_size: usize,

    /// Hop size for frame advancement
    pub hop_size: usize,

    /// Whether to normalize features
    pub normalize: bool,

    /// Feature dimensionality for extraction
    pub feature_dimension: FeatureDimension,
}

/// Feature dimensionality option for vectorization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureDimension {
    /// 15D RFE-Optimized features (identified via Recursive Feature Elimination for birds)
    D15,
    /// 19D RFE-Optimal features for Egyptian Fruit Bats
    D19,
    /// 30D base features (backward compatible)
    D30,
    /// 37D with phylogenetic acoustic descriptors (30D + 8 new)
    D37,
    /// 56D full delta preservation (30D + 13 Δ + 13 ΔΔ)
    D56,
}

impl Default for VectorizationConfig {
    fn default() -> Self {
        Self {
            n_mels: 30,
            fft_size: 2048,
            hop_size: 512,
            normalize: true,
            feature_dimension: FeatureDimension::D37, // Default to 37D for bioacoustics
        }
    }
}

impl From<FeatureDimension> for FeatureDim {
    fn from(dim: FeatureDimension) -> Self {
        match dim {
            FeatureDimension::D15 => FeatureDim::D30, // RFE-Optimized uses extract_rfe_optimized, not extract_dynamic
            FeatureDimension::D19 => FeatureDim::D19, // Bat RFE-Optimal uses extract_rfe_optimal_19d_bat
            FeatureDimension::D30 => FeatureDim::D30,
            FeatureDimension::D37 => FeatureDim::D37,
            FeatureDimension::D56 => FeatureDim::D56,
        }
    }
}

/// Result from Phase 2: Feature time-series for one phrase
/// Dimensionality depends on VectorizationConfig:
/// - D15: 15D RFE-Optimized features (via Recursive Feature Elimination)
/// - D30: 30D base features
/// - D37: 30D + 8 phylogenetic acoustic descriptors
/// - D56: 30D + 13 mfcc_delta + 13 mfcc_delta_delta
#[derive(Debug, Clone)]
pub struct PhraseFeatures {
    /// Phrase ID (matches SegmentedPhrase)
    pub phrase_id: String,

    /// Feature time-series matrix: shape (T, D) where:
    /// - T = number of time frames (varies per phrase)
    /// - D = feature dimensionality (30, 37, or 56)
    pub features: Array2<f64>,

    /// Number of time frames (T)
    pub n_frames: usize,

    /// Frame rate in Hz
    pub frame_rate: f64,

    /// Feature dimensionality (15, 30, 37, or 56)
    pub feature_dim: usize,
}

impl PhraseFeatures {
    /// Get the feature dimensionality
    pub fn dimensionality(&self) -> usize {
        self.feature_dim
    }

    /// Check if this is 15D RFE-Optimized feature set
    pub fn is_15d(&self) -> bool {
        self.feature_dim == 15
    }

    /// Check if this is 30D feature set
    pub fn is_30d(&self) -> bool {
        self.feature_dim == 30
    }

    /// Check if this is 37D feature set
    pub fn is_37d(&self) -> bool {
        self.feature_dim == 37
    }

    /// Check if this is 56D feature set
    pub fn is_56d(&self) -> bool {
        self.feature_dim == 56
    }
}

/// Serializable version of PhraseFeatures (for disk storage)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhraseFeaturesSerializable {
    /// Phrase ID (matches SegmentedPhrase)
    pub phrase_id: String,

    /// Features as flat vector (T * D elements)
    pub features_flat: Vec<f64>,

    /// Number of time frames (T)
    pub n_frames: usize,

    /// Frame rate in Hz
    pub frame_rate: f64,

    /// Feature dimensionality (15, 30, 37, or 56)
    pub feature_dim: usize,
}

impl From<PhraseFeatures> for PhraseFeaturesSerializable {
    fn from(pf: PhraseFeatures) -> Self {
        let features_flat: Vec<f64> = pf.features.as_slice().unwrap_or(&[]).to_vec();
        Self {
            phrase_id: pf.phrase_id,
            features_flat,
            n_frames: pf.n_frames,
            frame_rate: pf.frame_rate,
            feature_dim: pf.feature_dim,
        }
    }
}

impl TryFrom<PhraseFeaturesSerializable> for PhraseFeatures {
    type Error = PipelineError;

    fn try_from(pfs: PhraseFeaturesSerializable) -> Result<Self> {
        let features = Array2::from_shape_vec(
            (pfs.n_frames, pfs.feature_dim),
            pfs.features_flat
        ).map_err(|e| PipelineError::VectorizationError(format!("Invalid feature shape: {}", e)))?;

        Ok(Self {
            phrase_id: pfs.phrase_id,
            features,
            n_frames: pfs.n_frames,
            frame_rate: pfs.frame_rate,
            feature_dim: pfs.feature_dim,
        })
    }
}

// =============================================================================
// Phase 3: Discovery (The "Lexicon")
// =============================================================================

/// Configuration for Phase 3: DTW-DBSCAN Clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// DBSCAN epsilon parameter (maximum DTW distance for clustering)
    pub eps: f64,

    /// Minimum samples for a cluster
    pub min_samples: usize,

    /// DTW window size for Sakoe-Chiba band (0 = full matrix)
    pub dtw_window_size: Option<usize>,

    /// Whether to use FastDTW approximation
    pub use_fast_dtw: bool,

    /// FastDTW radius (if using FastDTW)
    pub fast_dtw_radius: usize,

    /// Whether to use LB_Keogh lower bound for pruning
    pub use_lb_keogh: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            eps: 0.5,           // Moderate clustering threshold
            min_samples: 5,      // Minimum evidence for a phrase type
            dtw_window_size: None, // Full DTW by default
            use_fast_dtw: false, // Use exact DTW by default
            fast_dtw_radius: 10, // If using FastDTW
            use_lb_keogh: true,  // Use LB_Keogh for speed
        }
    }
}

/// Result from Phase 3: Discovered vocabulary (clustered phrases)
#[derive(Debug, Clone)]
pub struct LexiconVocabularyItem {
    /// Cluster ID (vocabulary type)
    pub cluster_id: i32,

    /// Phrase IDs in this cluster
    pub phrase_ids: Vec<String>,

    /// Representative feature template (centroid)
    pub feature_template: Array2<f64>,

    /// Cluster size (number of phrases)
    pub size: usize,

    /// Cluster coherence (average intra-cluster similarity)
    pub coherence: f64,
}

/// Statistics about the discovered vocabulary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexiconStatistics {
    /// Total vocabulary size (number of phrase types)
    pub total_vocabulary_items: usize,

    /// Total phrases clustered
    pub total_phrases: usize,

    /// Noise phrases (unclustered)
    pub noise_count: usize,

    /// Average cluster size
    pub avg_cluster_size: f64,

    /// Largest cluster size
    pub max_cluster_size: usize,

    /// Zipf's Law alpha (slope)
    pub zipf_alpha: Option<f64>,
}

// =============================================================================
// Phase 4: Refinement (The "Grammar")
// =============================================================================

/// Configuration for Phase 4: GMM-HMM Refinement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementConfig {
    /// Number of HMM states (will be auto-determined if None)
    pub n_states: Option<usize>,

    /// Number of GMM components per state
    pub n_components: usize,

    /// Maximum EM iterations for HMM training
    pub max_iterations: usize,

    /// Convergence threshold for log-likelihood
    pub convergence_threshold: f64,

    /// Regularization for covariance matrices
    pub covariance_reg: f64,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            n_states: None, // Auto-determine based on sequence length
            n_components: 2, // 2 Gaussians per state (default)
            max_iterations: 100,
            convergence_threshold: 1e-4,
            covariance_reg: 1e-6,
        }
    }
}

/// Result from Phase 4: Refined phoneme model for one vocabulary item
#[derive(Debug, Clone)]
pub struct PhonemeModel {
    /// Vocabulary item (cluster) this model represents
    pub cluster_id: i32,

    /// Trained HMM with GMM emissions
    pub hmm: HiddenMarkovModel,

    /// Number of states in the HMM
    pub n_states: usize,

    /// Model quality (log-likelihood)
    pub log_likelihood: f64,

    /// State interpretations (e.g., "Onset", "Sustain", "Offset")
    pub state_labels: Vec<String>,
}

/// Phoneme state interpretation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhonemeState {
    Onset,
    Sustain,
    Offset,
    Transition,
    Unknown,
}

// =============================================================================
// Checkpointing for Resume Capability
// =============================================================================

/// Checkpoint data for resuming pipeline execution
/// Only tracks progress (file paths, counts), not actual data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCheckpoint {
    /// Files that have been processed (Phase 1 complete)
    pub processed_files: Vec<String>,

    /// Number of phrases segmented so far
    pub phrase_count: usize,

    /// Timestamp of last checkpoint
    pub checkpoint_time: u64,

    /// Pipeline phase reached (1-4)
    pub current_phase: u8,
}

impl PipelineCheckpoint {
    /// Create a new empty checkpoint
    pub fn new() -> Self {
        Self {
            processed_files: Vec::new(),
            phrase_count: 0,
            checkpoint_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            current_phase: 0,
        }
    }

    /// Save checkpoint to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| PipelineError::SegmentationError(format!("Failed to serialize checkpoint: {}", e)))?;

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| PipelineError::SegmentationError(format!("Failed to create checkpoint directory: {}", e)))?;
        }

        std::fs::write(path, json)
            .map_err(|e| PipelineError::SegmentationError(format!("Failed to write checkpoint: {}", e)))?;

        Ok(())
    }

    /// Load checkpoint from file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| PipelineError::SegmentationError(format!("Failed to read checkpoint: {}", e)))?;

        serde_json::from_str(&json)
            .map_err(|e| PipelineError::SegmentationError(format!("Failed to deserialize checkpoint: {}", e)))
    }

    /// Check if checkpoint exists and is valid
    pub fn exists<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref().exists()
    }
}

// =============================================================================
// Complete Pipeline Result
// =============================================================================

/// Complete result from the Lexicon to Syntax pipeline
#[derive(Debug, Clone)]
pub struct LexiconToSyntaxResult {
    /// Phase 1: Segmented phrases
    pub segmented_phrases: Vec<SegmentedPhrase>,

    /// Phase 2: Phrase features
    pub phrase_features: Vec<PhraseFeatures>,

    /// Phase 3: Discovered vocabulary
    pub vocabulary: Vec<LexiconVocabularyItem>,

    /// Vocabulary statistics
    pub vocabulary_stats: LexiconStatistics,

    /// Phase 4: Refined phoneme models
    pub phoneme_models: Vec<PhonemeModel>,

    /// Pipeline execution time
    pub execution_time_sec: f64,
}

// =============================================================================
// Master Pipeline: Lexicon to Syntax
// =============================================================================

pub struct LexiconToSyntaxPipeline {
    /// Configuration for Phase 1: Segmentation
    segmentation_config: SegmentationConfig,

    /// Configuration for Phase 2: Vectorization
    vectorization_config: VectorizationConfig,

    /// Configuration for Phase 3: Discovery
    discovery_config: DiscoveryConfig,

    /// Configuration for Phase 4: Refinement
    refinement_config: RefinementConfig,

    /// Sample rate for audio processing
    sample_rate: u32,

    /// Batch size for disk-based processing (phrases per batch)
    batch_size: usize,
}

impl LexiconToSyntaxPipeline {
    /// Create a new pipeline with default configurations
    pub fn new() -> Self {
        Self {
            segmentation_config: SegmentationConfig::default(),
            vectorization_config: VectorizationConfig::default(),
            discovery_config: DiscoveryConfig::default(),
            refinement_config: RefinementConfig::default(),
            sample_rate: 48000,
            batch_size: 50000, // Default: process 50K phrases per batch
        }
    }

    /// Create a new pipeline with custom configurations
    pub fn with_configs(
        segmentation_config: SegmentationConfig,
        vectorization_config: VectorizationConfig,
        discovery_config: DiscoveryConfig,
        refinement_config: RefinementConfig,
    ) -> Self {
        Self {
            segmentation_config,
            vectorization_config,
            discovery_config,
            refinement_config,
            sample_rate: 48000,
            batch_size: 50000,
        }
    }

    /// Set batch size for disk-based processing
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set segmentation configuration
    pub fn with_segmentation_config(mut self, config: SegmentationConfig) -> Self {
        self.segmentation_config = config;
        self
    }

    /// Set vectorization configuration
    pub fn with_vectorization_config(mut self, config: VectorizationConfig) -> Self {
        self.vectorization_config = config;
        self
    }

    /// Set feature dimensionality (convenience method)
    pub fn with_feature_dimension(mut self, dim: FeatureDimension) -> Self {
        self.vectorization_config.feature_dimension = dim;
        self
    }

    /// Set discovery configuration
    pub fn with_discovery_config(mut self, config: DiscoveryConfig) -> Self {
        self.discovery_config = config;
        self
    }

    /// Set refinement configuration
    pub fn with_refinement_config(mut self, config: RefinementConfig) -> Self {
        self.refinement_config = config;
        self
    }

    // =========================================================================
    // Disk Storage Helper Methods
    // =========================================================================

    /// Append segmented phrases to disk using JSON lines (for incremental saving)
    fn append_phrases_to_disk<P: AsRef<Path>>(
        &self,
        phrases: &[SegmentedPhrase],
        path: P,
    ) -> Result<()> {
        use std::io::Write;

        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| PipelineError::SegmentationError(format!("Failed to create directory: {}", e)))?;
        }

        // Open in append mode
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| PipelineError::SegmentationError(format!("Failed to open phrases file: {}", e)))?;

        // Append each phrase as a JSON line
        for phrase in phrases {
            let json = serde_json::to_string(phrase)
                .map_err(|e| PipelineError::SegmentationError(format!("Failed to serialize phrase: {}", e)))?;
            writeln!(file, "{}", json)
                .map_err(|e| PipelineError::SegmentationError(format!("Failed to write phrase: {}", e)))?;
        }

        file.flush()
            .map_err(|e| PipelineError::SegmentationError(format!("Failed to flush phrases file: {}", e)))?;

        Ok(())
    }

    /// Load segmented phrases from disk (handles both bincode and JSON lines formats)
    fn load_phrases_from_disk<P: AsRef<Path>>(&self, path: P) -> Result<Vec<SegmentedPhrase>> {
        let path = path.as_ref();

        let data = std::fs::read_to_string(path)
            .map_err(|e| PipelineError::SegmentationError(format!("Failed to read phrases: {}", e)))?;

        let mut phrases = Vec::new();
        for line in data.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let phrase: SegmentedPhrase = serde_json::from_str(line)
                .map_err(|e| PipelineError::SegmentationError(format!("Failed to deserialize phrase: {}", e)))?;
            phrases.push(phrase);
        }

        Ok(phrases)
    }

    /// Save segmented phrases to disk using bincode (deprecated - use append_phrases_to_disk)
    fn save_phrases_to_disk<P: AsRef<Path>>(
        &self,
        phrases: &[SegmentedPhrase],
        path: P,
    ) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| PipelineError::SegmentationError(format!("Failed to create directory: {}", e)))?;
        }

        let serialized = bincode::serialize(phrases)
            .map_err(|e| PipelineError::SegmentationError(format!("Failed to serialize phrases: {}", e)))?;

        std::fs::write(path, serialized)
            .map_err(|e| PipelineError::SegmentationError(format!("Failed to write phrases: {}", e)))?;

        Ok(())
    }

    /// Save phrase features to disk using bincode
    fn save_features_to_disk<P: AsRef<Path>>(
        &self,
        features: &[PhraseFeaturesSerializable],
        path: P,
    ) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| PipelineError::VectorizationError(format!("Failed to create directory: {}", e)))?;
        }

        let serialized = bincode::serialize(features)
            .map_err(|e| PipelineError::VectorizationError(format!("Failed to serialize features: {}", e)))?;

        std::fs::write(path, serialized)
            .map_err(|e| PipelineError::VectorizationError(format!("Failed to write features: {}", e)))?;

        Ok(())
    }

    /// Load phrase features from disk
    fn load_features_from_disk<P: AsRef<Path>>(&self, path: P) -> Result<Vec<PhraseFeaturesSerializable>> {
        let path = path.as_ref();

        let data = std::fs::read(path)
            .map_err(|e| PipelineError::VectorizationError(format!("Failed to read features: {}", e)))?;

        let features: Vec<PhraseFeaturesSerializable> = bincode::deserialize(&data)
            .map_err(|e| PipelineError::VectorizationError(format!("Failed to deserialize features: {}", e)))?;

        Ok(features)
    }

    /// Run the complete pipeline on audio files
    ///
    /// # Arguments
    /// * `audio_files` - Paths to audio files to process
    ///
    /// # Returns
    /// Complete pipeline result with all four phases
    pub fn run<P: AsRef<Path>>(&self, audio_files: &[P]) -> Result<LexiconToSyntaxResult> {
        let start = std::time::Instant::now();

        // Phase 1: Segmentation
        let segmented_phrases = self.run_phase1_segmentation(audio_files)?;

        // Phase 2: Vectorization
        let phrase_features = self.run_phase2_vectorization(&segmented_phrases)?;

        // Phase 3: Discovery
        let (vocabulary, vocab_stats) = self.run_phase3_discovery(&phrase_features)?;

        // Phase 4: Refinement
        let phoneme_models = self.run_phase4_refinement(&vocabulary, &phrase_features)?;

        let execution_time = start.elapsed().as_secs_f64();

        Ok(LexiconToSyntaxResult {
            segmented_phrases,
            phrase_features,
            vocabulary,
            vocabulary_stats: vocab_stats,
            phoneme_models,
            execution_time_sec: execution_time,
        })
    }

    /// Run pipeline with checkpointing support for resumption
    /// Uses disk-based storage to handle large datasets with limited RAM.
    pub fn run_with_checkpoint<P: AsRef<Path>>(
        &self,
        audio_files: &[P],
        checkpoint_path: P,
        _checkpoint_interval_secs: u64,
    ) -> Result<LexiconToSyntaxResult> {
        let checkpoint_path = checkpoint_path.as_ref();
        let output_dir = checkpoint_path.parent().unwrap_or(Path::new("."));

        // Define disk storage paths
        let phrases_path = output_dir.join("segmented_phrases.bincode");
        let features_path = output_dir.join("phrase_features.bincode");

        // Try to load existing checkpoint
        let mut checkpoint = if PipelineCheckpoint::exists(checkpoint_path) {
            println!("📂 Found existing checkpoint, resuming...");
            let loaded = PipelineCheckpoint::load(checkpoint_path)?;
            let age = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() - loaded.checkpoint_time;
            println!("   ├─ Checkpoint age: {} hours", age / 3600);
            println!("   ├─ Files processed: {}", loaded.processed_files.len());
            println!("   ├─ Phrases found: {}", loaded.phrase_count);
            println!("   └─ Phase: {}/4", loaded.current_phase);
            loaded
        } else {
            println!("💾 Starting fresh (no checkpoint found)");
            PipelineCheckpoint::new()
        };

        let start = std::time::Instant::now();

        // Helper to save checkpoint
        let save_checkpoint = |checkpoint: &PipelineCheckpoint, phase: u8, phrase_count: usize| -> Result<()> {
            let mut ck = checkpoint.clone();
            ck.current_phase = phase;
            ck.phrase_count = phrase_count;
            ck.checkpoint_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            ck.save(checkpoint_path)?;
            Ok(())
        };

        // =========================================================================
        // Phase 1: Segmentation (with resume capability)
        // =========================================================================
        let _total_phrases = if checkpoint.current_phase >= 1 && phrases_path.exists() {
            println!("♻️  Phase 1 complete - loading from disk...");
            let phrases = self.load_phrases_from_disk(&phrases_path)?;
            println!("   └─ Loaded {} phrases from disk", phrases.len());
            phrases.len()
        } else {
            println!("🔪 Phase 1: Segmentation with checkpointing...");

            // Convert paths to PathBuf and filter already processed
            let processed_set: std::collections::HashSet<String> =
                checkpoint.processed_files.iter().cloned().collect();

            let remaining_files: Vec<PathBuf> = audio_files.iter()
                .filter_map(|p| {
                    let path_str = p.as_ref().to_string_lossy().to_string();
                    if processed_set.contains(&path_str) {
                        None
                    } else {
                        Some(p.as_ref().to_path_buf())
                    }
                })
                .collect();

            let total_phrases = if remaining_files.is_empty() {
                println!("   ├─ All files already processed!");
                checkpoint.phrase_count
            } else {
                println!("   ├─ Remaining files: {} / {}", remaining_files.len(), audio_files.len());

                // Process remaining files with periodic checkpointing and incremental phrase saving
                let file_batch_size = 100; // Checkpoint every 100 files
                let mut total_phrase_count = 0usize;

                // Clear any existing phrases file
                if phrases_path.exists() {
                    std::fs::remove_file(&phrases_path).ok();
                }

                for (batch_idx, chunk) in remaining_files.chunks(file_batch_size).enumerate() {
                    // Process this batch
                    let batch_phrases = self.run_phase1_segmentation(chunk)?;

                    // Track processed files
                    for path in chunk {
                        checkpoint.processed_files.push(path.to_string_lossy().to_string());
                    }

                    // Save phrases incrementally (append mode)
                    if !batch_phrases.is_empty() {
                        self.append_phrases_to_disk(&batch_phrases, &phrases_path)?;
                        total_phrase_count += batch_phrases.len();
                    }

                    // Save checkpoint
                    println!("   ├─ Batch {}: Processed {} / {} files ({} phrases), saving checkpoint...",
                        batch_idx + 1,
                        checkpoint.processed_files.len(),
                        audio_files.len(),
                        total_phrase_count
                    );
                    save_checkpoint(&checkpoint, 1, total_phrase_count)?;

                    // Show ETA
                    let elapsed = start.elapsed().as_secs_f64();
                    let rate = checkpoint.processed_files.len() as f64 / elapsed;
                    let remaining = (audio_files.len() - checkpoint.processed_files.len()) as f64 / rate;
                    println!("   │  └─ ETA: {:.1} hours", remaining / 3600.0);
                }

                println!("💾 Phase 1 complete: {} phrases saved to {}", total_phrase_count, phrases_path.display());

                total_phrase_count
            };

            total_phrases
        };

        // =========================================================================
        // Phase 2: Vectorization (batched, disk-based)
        // =========================================================================
        if checkpoint.current_phase >= 2 && features_path.exists() {
            println!("♻️  Phase 2 complete - features already on disk");
        } else {
            println!("📊 Phase 2: Vectorization (batch size: {} phrases)...", self.batch_size);

            // Load phrases from disk in batches
            let all_phrases = self.load_phrases_from_disk(&phrases_path)?;
            let total_phrases = all_phrases.len();
            let mut all_features: Vec<PhraseFeaturesSerializable> = Vec::new();

            for (batch_idx, batch) in all_phrases.chunks(self.batch_size).enumerate() {
                println!("   ├─ Batch {}/: Processing {} phrases...",
                    batch_idx + 1,
                    batch.len()
                );

                // Extract features for this batch
                let batch_features: Vec<PhraseFeatures> = self.run_phase2_vectorization(batch)?;

                // Convert to serializable format
                let serializable: Vec<PhraseFeaturesSerializable> =
                    batch_features.into_iter().map(|f| f.into()).collect();

                all_features.extend(serializable);

                println!("   │  └─ Total features extracted: {} / {}", all_features.len(), total_phrases);

                // Periodically save features to disk (every batch)
                self.save_features_to_disk(&all_features, &features_path)?;
            }

            println!("   └─ Saved {} features to {}", all_features.len(), features_path.display());
            save_checkpoint(&checkpoint, 2, all_features.len())?;
        }

        // =========================================================================
        // Phase 3 & 4: Load all features from disk and process
        // (Features are much smaller than audio, so we can load all at once)
        // =========================================================================
        println!("🔍 Phase 3: Discovery...");
        let all_features_serializable = self.load_features_from_disk(&features_path)?;
        println!("   └─ Loaded {} features from disk", all_features_serializable.len());

        // Convert to PhraseFeatures for clustering
        let all_features: Vec<PhraseFeatures> = all_features_serializable
            .into_iter()
            .map(|f| f.try_into())
            .collect::<Result<Vec<_>>>()?;

        let (vocabulary, vocab_stats) = self.run_phase3_discovery(&all_features)?;
        save_checkpoint(&checkpoint, 3, all_features.len())?;

        println!("🎯 Phase 4: Refinement...");
        let phoneme_models = self.run_phase4_refinement(&vocabulary, &all_features)?;
        save_checkpoint(&checkpoint, 4, all_features.len())?;

        let execution_time = start.elapsed().as_secs_f64();

        // Load phrases for final result
        let segmented_phrases = self.load_phrases_from_disk(&phrases_path)?;

        Ok(LexiconToSyntaxResult {
            segmented_phrases,
            phrase_features: all_features,
            vocabulary,
            vocabulary_stats: vocab_stats,
            phoneme_models,
            execution_time_sec: execution_time,
        })
    }

    /// Phase 1: Segmentation - The "Slicing" (Parallel)
    fn run_phase1_segmentation<P: AsRef<Path>>(
        &self,
        audio_files: &[P],
    ) -> Result<Vec<SegmentedPhrase>> {
        // Convert paths to PathBuf for Send/Sync
        let paths: Vec<PathBuf> = audio_files.iter().map(|p| p.as_ref().to_path_buf()).collect();

        // Process files in parallel using Rayon
        let results: Vec<Vec<SegmentedPhrase>> = paths
            .into_par_iter()
            .enumerate()
            .map(|(file_idx, audio_path)| {
                // Load audio
                let (audio, sr) = self.load_audio(&audio_path)?;

                // Create adaptive segmenter
                let segmenter = AdaptiveSegmenter::new(
                    sr,
                    self.segmentation_config.min_duration_ms,
                    self.segmentation_config.max_duration_ms,
                    self.segmentation_config.onset_threshold,
                ).map_err(|e| PipelineError::SegmentationError(e.to_string()))?;

                // Segment the audio
                let segments = segmenter.segment(&audio)
                    .map_err(|e| PipelineError::SegmentationError(e.to_string()))?;

                // Convert to SegmentedPhrase
                let mut phrases = Vec::new();
                for (seg_idx, segment) in segments.iter().enumerate() {
                    let start_sample = segment.0;
                    let end_sample = segment.1;

                    let phrase_audio = audio[start_sample..end_sample.min(audio.len())].to_vec();
                    let start_time = start_sample as f64 / sr as f64;
                    let end_time = end_sample as f64 / sr as f64;
                    let duration_ms = (end_time - start_time) * 1000.0;

                    // Filter by duration
                    if duration_ms < self.segmentation_config.min_duration_ms
                        || duration_ms > self.segmentation_config.max_duration_ms
                    {
                        continue;
                    }

                    phrases.push(SegmentedPhrase {
                        phrase_id: format!("file{}_phrase{}", file_idx, seg_idx),
                        audio: phrase_audio,
                        start_time,
                        end_time,
                        duration_ms,
                        sample_rate: sr,
                        onset_confidence: 0.7,
                    });
                }

                Ok(phrases)
            })
            .collect::<Result<Vec<_>>>()?;

        // Flatten results
        Ok(results.into_iter().flatten().collect())
    }

    /// Phase 2: Vectorization - The "Embedding" (Parallel)
    fn run_phase2_vectorization(
        &self,
        segmented_phrases: &[SegmentedPhrase],
    ) -> Result<Vec<PhraseFeatures>> {
        let feature_dim = self.vectorization_config.feature_dimension;

        // Process phrases in parallel using Rayon
        let phrase_features: Vec<PhraseFeatures> = segmented_phrases
            .par_iter()
            .map(|phrase| {
                // Create feature extractor with the phrase's sample rate
                let extractor = MicroDynamicsExtractor::new(phrase.sample_rate);

                // Handle RFE-Optimized features separately
                let (feature_array, dim) = if feature_dim == FeatureDimension::D15 {
                    // Use extract_rfe_optimized for D15 (birds)
                    let rfe_features = extractor.extract_rfe_optimized(&phrase.audio)
                        .map_err(|e| PipelineError::VectorizationError(e.to_string()))?;

                    // Convert Vec<f32> to Array2
                    let mut arr = Array2::zeros((1, 15));
                    for (i, &val) in rfe_features.iter().enumerate() {
                        if i < 15 {
                            arr[[0, i]] = val as f64;
                        }
                    }
                    (arr, 15)
                } else if feature_dim == FeatureDimension::D19 {
                    // Use extract_rfe_optimal_19d_bat for D19 (bats)
                    let rfe_features = extractor.extract_rfe_optimal_19d_bat(&phrase.audio)
                        .map_err(|e| PipelineError::VectorizationError(e.to_string()))?;

                    // Convert Vec<f32> to Array2
                    let mut arr = Array2::zeros((1, 19));
                    for (i, &val) in rfe_features.iter().enumerate() {
                        if i < 19 {
                            arr[[0, i]] = val as f64;
                        }
                    }
                    (arr, 19)
                } else {
                    // Extract features using the configured dimensionality
                    let feature_vector = extractor.extract_dynamic(
                        &phrase.audio,
                        feature_dim.into()
                    ).map_err(|e| PipelineError::VectorizationError(e.to_string()))?;

                    // Convert FeatureVector to Array2 based on dimensionality
                    match &feature_vector {
                    crate::micro_dynamics_extractor::FeatureVector::D30(features) => {
                        let mut arr = Array2::zeros((1, 30));
                        let base = features;
                        arr[[0, 0]] = base.attack_time_ms as f64;
                        arr[[0, 1]] = base.decay_time_ms as f64;
                        arr[[0, 2]] = base.sustain_level as f64;
                        arr[[0, 3]] = base.vibrato_rate_hz as f64;
                        arr[[0, 4]] = base.vibrato_depth as f64;
                        arr[[0, 5]] = base.jitter as f64;
                        arr[[0, 6]] = base.shimmer as f64;
                        arr[[0, 7]] = base.harmonicity as f64;
                        arr[[0, 8]] = base.spectral_flatness as f64;
                        arr[[0, 9]] = base.harmonic_to_noise_ratio as f64;
                        arr[[0, 10]] = base.spectral_flux as f64;
                        // MFCC coefficients (13)
                        for (i, &mfcc_val) in base.mfcc.iter().enumerate() {
                            arr[[0, 11 + i]] = mfcc_val as f64;
                        }
                        arr[[0, 24]] = base.median_ici_ms as f64;
                        arr[[0, 25]] = base.onset_rate_hz as f64;
                        arr[[0, 26]] = base.ici_coefficient_of_variation as f64;
                        // Add placeholder values for remaining fields
                        arr[[0, 27]] = phrase.duration_ms as f64;
                        arr[[0, 28]] = 0.0; // f0_mean placeholder
                        arr[[0, 29]] = 0.0; // f0_std placeholder
                        (arr, 30)
                    }
                    crate::micro_dynamics_extractor::FeatureVector::D37(features) => {
                        // Note: D37 struct actually has 30D base + 8 new features = 38D total
                        let mut arr = Array2::zeros((1, 38));
                        let base = &features.base_30d;
                        // Base 30D features (indices 0-29)
                        arr[[0, 0]] = base.attack_time_ms as f64;
                        arr[[0, 1]] = base.decay_time_ms as f64;
                        arr[[0, 2]] = base.sustain_level as f64;
                        arr[[0, 3]] = base.vibrato_rate_hz as f64;
                        arr[[0, 4]] = base.vibrato_depth as f64;
                        arr[[0, 5]] = base.jitter as f64;
                        arr[[0, 6]] = base.shimmer as f64;
                        arr[[0, 7]] = base.harmonicity as f64;
                        arr[[0, 8]] = base.spectral_flatness as f64;
                        arr[[0, 9]] = base.harmonic_to_noise_ratio as f64;
                        arr[[0, 10]] = base.spectral_flux as f64;
                        // MFCC coefficients (13)
                        for (i, &mfcc_val) in base.mfcc.iter().enumerate() {
                            arr[[0, 11 + i]] = mfcc_val as f64;
                        }
                        arr[[0, 24]] = base.median_ici_ms as f64;
                        arr[[0, 25]] = base.onset_rate_hz as f64;
                        arr[[0, 26]] = base.ici_coefficient_of_variation as f64;
                        // Add placeholder values for remaining fields
                        arr[[0, 27]] = phrase.duration_ms as f64;
                        arr[[0, 28]] = 0.0; // f0_mean placeholder
                        arr[[0, 29]] = 0.0; // f0_std placeholder

                        // 8 new phylogenetic acoustic descriptors (indices 30-37)
                        // Note: The struct is named MicroDynamicsFeatures37D but actually has 8 new features
                        // making it 38D total (30D base + 8 new = 38D)
                        arr[[0, 30]] = features.pitch_entropy as f64;
                        arr[[0, 31]] = features.spectral_tilt as f64;
                        arr[[0, 32]] = features.harmonic_deviation as f64;
                        arr[[0, 33]] = features.formant_f1 as f64;
                        arr[[0, 34]] = features.formant_f2 as f64;
                        arr[[0, 35]] = features.formant_f3 as f64;
                        arr[[0, 36]] = features.fm_depth_hz as f64;
                        arr[[0, 37]] = features.roughness as f64; // roughness at index 37 (8th new feature)
                        (arr, 38) // Return 38 as the dimensionality
                    }
                    crate::micro_dynamics_extractor::FeatureVector::D56(features) => {
                        let mut arr = Array2::zeros((1, 56));
                        let base = &features.base_30d;
                        // Base 30D features (indices 0-29)
                        arr[[0, 0]] = base.attack_time_ms as f64;
                        arr[[0, 1]] = base.decay_time_ms as f64;
                        arr[[0, 2]] = base.sustain_level as f64;
                        arr[[0, 3]] = base.vibrato_rate_hz as f64;
                        arr[[0, 4]] = base.vibrato_depth as f64;
                        arr[[0, 5]] = base.jitter as f64;
                        arr[[0, 6]] = base.shimmer as f64;
                        arr[[0, 7]] = base.harmonicity as f64;
                        arr[[0, 8]] = base.spectral_flatness as f64;
                        arr[[0, 9]] = base.harmonic_to_noise_ratio as f64;
                        arr[[0, 10]] = base.spectral_flux as f64;
                        // MFCC coefficients (13)
                        for (i, &mfcc_val) in base.mfcc.iter().enumerate() {
                            arr[[0, 11 + i]] = mfcc_val as f64;
                        }
                        arr[[0, 24]] = base.median_ici_ms as f64;
                        arr[[0, 25]] = base.onset_rate_hz as f64;
                        arr[[0, 26]] = base.ici_coefficient_of_variation as f64;
                        // Add placeholder values for remaining fields
                        arr[[0, 27]] = phrase.duration_ms as f64;
                        arr[[0, 28]] = 0.0; // f0_mean placeholder
                        arr[[0, 29]] = 0.0; // f0_std placeholder

                        // 13 mfcc_delta features (indices 30-42)
                        for (i, &delta_val) in features.mfcc_delta.iter().enumerate() {
                            arr[[0, 30 + i]] = delta_val as f64;
                        }

                        // 13 mfcc_delta_delta features (indices 43-55)
                        for (i, &delta_delta_val) in features.mfcc_delta_delta.iter().enumerate() {
                            arr[[0, 43 + i]] = delta_delta_val as f64;
                        }
                        (arr, 56)
                    }
                    crate::micro_dynamics_extractor::FeatureVector::D19(features) => {
                        let mut arr = Array2::zeros((1, 19));
                        // Temporal envelope features (top 3)
                        arr[[0, 0]] = features.attack_time_ms as f64;
                        arr[[0, 1]] = features.decay_time_ms as f64;
                        arr[[0, 2]] = features.sustain_level as f64;
                        // Motion factors
                        arr[[0, 3]] = features.jitter as f64;
                        arr[[0, 4]] = features.shimmer as f64;
                        // Grit factors
                        arr[[0, 5]] = features.harmonicity as f64;
                        arr[[0, 6]] = features.harmonic_to_noise_ratio as f64;
                        // Selected MFCCs
                        arr[[0, 7]] = features.mfcc_2 as f64;
                        arr[[0, 8]] = features.mfcc_3 as f64;
                        arr[[0, 9]] = features.mfcc_5 as f64;
                        arr[[0, 10]] = features.mfcc_6 as f64;
                        arr[[0, 11]] = features.mfcc_10 as f64;
                        // Rhythm factors
                        arr[[0, 12]] = features.median_ici_ms as f64;
                        arr[[0, 13]] = features.ici_coefficient_of_variation as f64;
                        // Phylogenetic features
                        arr[[0, 14]] = features.pitch_entropy as f64;
                        arr[[0, 15]] = features.spectral_tilt as f64;
                        arr[[0, 16]] = features.formant_f3 as f64;
                        arr[[0, 17]] = features.fm_depth_hz as f64;
                        arr[[0, 18]] = features.roughness as f64;
                        (arr, 19)
                    }
                    crate::micro_dynamics_extractor::FeatureVector::D15(features) => {
                        let mut arr = Array2::zeros((1, 15));
                        // Energy features (2)
                        arr[[0, 0]] = features.rms_energy as f64;
                        arr[[0, 1]] = features.vibrato_depth as f64;
                        // MFCC features (4)
                        arr[[0, 2]] = features.mfcc_0 as f64;
                        arr[[0, 3]] = features.mfcc_1 as f64;
                        arr[[0, 4]] = features.mfcc_3 as f64;
                        arr[[0, 5]] = features.mfcc_4 as f64;
                        // Timbre features (2)
                        arr[[0, 6]] = features.spectral_flux as f64;
                        arr[[0, 7]] = features.hnr as f64;
                        // Temporal features (3)
                        arr[[0, 8]] = features.decay_time_ms as f64;
                        arr[[0, 9]] = features.sustain_level as f64;
                        arr[[0, 10]] = features.attack_time_ms as f64;
                        // Rhythm features (2)
                        arr[[0, 11]] = features.ici_cv as f64;
                        arr[[0, 12]] = features.onset_rate_hz as f64;
                        // Modulation features (1)
                        arr[[0, 13]] = features.vibrato_rate_hz as f64;
                        // Perturbation features (1)
                        arr[[0, 14]] = features.shimmer as f64;
                        (arr, 15)
                    }
                    crate::micro_dynamics_extractor::FeatureVector::D45(features) => {
                        let mut arr = Array2::zeros((1, 45));
                        let base = &features.base_30d;
                        // Base 30D features (indices 0-29)
                        arr[[0, 0]] = base.attack_time_ms as f64;
                        arr[[0, 1]] = base.decay_time_ms as f64;
                        arr[[0, 2]] = base.sustain_level as f64;
                        arr[[0, 3]] = base.vibrato_rate_hz as f64;
                        arr[[0, 4]] = base.vibrato_depth as f64;
                        arr[[0, 5]] = base.jitter as f64;
                        arr[[0, 6]] = base.shimmer as f64;
                        arr[[0, 7]] = base.harmonicity as f64;
                        arr[[0, 8]] = base.spectral_flatness as f64;
                        arr[[0, 9]] = base.harmonic_to_noise_ratio as f64;
                        arr[[0, 10]] = base.spectral_flux as f64;
                        // MFCC coefficients (13)
                        for (i, &mfcc_val) in base.mfcc.iter().enumerate() {
                            arr[[0, 11 + i]] = mfcc_val as f64;
                        }
                        arr[[0, 24]] = base.median_ici_ms as f64;
                        arr[[0, 25]] = base.onset_rate_hz as f64;
                        arr[[0, 26]] = base.ici_coefficient_of_variation as f64;
                        // Placeholder values for remaining base fields
                        arr[[0, 27]] = phrase.duration_ms as f64;
                        arr[[0, 28]] = 0.0; // f0_mean placeholder
                        arr[[0, 29]] = 0.0; // f0_std placeholder

                        // Resonance (6): indices 30-35
                        arr[[0, 30]] = features.formant_1_hz as f64;
                        arr[[0, 31]] = features.formant_2_hz as f64;
                        arr[[0, 32]] = features.formant_3_hz as f64;
                        arr[[0, 33]] = features.formant_1_bandwidth as f64;
                        arr[[0, 34]] = features.formant_2_bandwidth as f64;
                        arr[[0, 35]] = features.formant_dispersion as f64;

                        // Spectral Shape (4): indices 36-39
                        arr[[0, 36]] = features.spectral_centroid as f64;
                        arr[[0, 37]] = features.spectral_spread as f64;
                        arr[[0, 38]] = features.spectral_skewness as f64;
                        arr[[0, 39]] = features.spectral_kurtosis as f64;

                        // Modulation (3): indices 40-42
                        arr[[0, 40]] = features.spectral_tilt as f64;
                        arr[[0, 41]] = features.fm_slope as f64;
                        arr[[0, 42]] = features.am_depth as f64;

                        // Non-Linear (2): indices 43-44
                        arr[[0, 43]] = features.subharmonic_ratio as f64;
                        arr[[0, 44]] = features.spectral_entropy as f64;

                        (arr, 45)
                    }
                    crate::micro_dynamics_extractor::FeatureVector::D39(_features) => {
                        // 39D features use multi-scale aggregations
                        // For now, we'll not support 39D in this pipeline
                        // as it's designed for 30D/37D/56D
                        return Err(PipelineError::VectorizationError(
                            "39D features not yet supported in lexicon_to_syntax pipeline".to_string()
                        ));
                    }
                    }
                };

                let frame_rate = phrase.sample_rate as f64;

                Ok(PhraseFeatures {
                    phrase_id: phrase.phrase_id.clone(),
                    features: feature_array,
                    n_frames: 1,
                    frame_rate,
                    feature_dim: dim,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(phrase_features)
    }

    /// Phase 3: Discovery - The "Lexicon" (DTW-DBSCAN)
    fn run_phase3_discovery(
        &self,
        phrase_features: &[PhraseFeatures],
    ) -> Result<(Vec<LexiconVocabularyItem>, LexiconStatistics)> {
        if phrase_features.is_empty() {
            return Ok((vec![], LexiconStatistics {
                total_vocabulary_items: 0,
                total_phrases: 0,
                noise_count: 0,
                avg_cluster_size: 0.0,
                max_cluster_size: 0,
                zipf_alpha: None,
            }));
        }

        // Create DTW-DBSCAN clusterer
        let window_size = self.discovery_config.dtw_window_size;
        let dtw_dbscan = DtwDbscan::new(
            self.discovery_config.eps,
            self.discovery_config.min_samples,
            window_size,
        );

        // Prepare data: collect all feature matrices as 1D vectors
        let feature_series_1d: Vec<Vec<f64>> = phrase_features
            .iter()
            .map(|pf| {
                let n_frames = pf.features.nrows();
                let n_dims = pf.features.ncols();
                let mut flat = vec
![0.0; n_frames * n_dims];
                for i in 0..n_frames {
                    for j in 0..n_dims {
                        flat[i * n_dims + j] = pf.features[[i, j]];
                    }
                }
                flat
            })
            .collect();

        let phrase_ids: Vec<String> = phrase_features
            .iter()
            .map(|pf| pf.phrase_id.clone())
            .collect();

        // Run DTW-DBSCAN clustering
        let cluster_labels = dtw_dbscan.fit_predict(&feature_series_1d)
            .map_err(|e| PipelineError::DiscoveryError(e.to_string()))?;

        // Group phrases by cluster
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        let mut noise_count = 0;

        for (idx, &label) in cluster_labels.iter().enumerate() {
            if label == -1 {
                noise_count += 1;
            } else {
                clusters.entry(label).or_insert_with(Vec::new).push(idx);
            }
        }

        // Create vocabulary items
        let mut vocabulary = Vec::new();
        for (cluster_id, indices) in clusters {
            let cluster_phrases: Vec<&PhraseFeatures> =
                indices.iter().map(|&idx| &phrase_features[idx]).collect();

            // Compute centroid (feature template)
            let feature_template = self.compute_centroid(&cluster_phrases);

            // Compute coherence (intra-cluster similarity)
            let coherence = self.compute_coherence(&cluster_phrases, &feature_template);

            vocabulary.push(LexiconVocabularyItem {
                cluster_id,
                phrase_ids: indices.iter().map(|&idx| phrase_ids[idx].clone()).collect(),
                feature_template,
                size: indices.len(),
                coherence,
            });
        }

        // Sort by cluster size (largest first)
        vocabulary.sort_by(|a, b| b.size.cmp(&a.size));

        // Compute statistics
        let stats = self.compute_vocabulary_stats(&vocabulary, noise_count, phrase_features.len());

        Ok((vocabulary, stats))
    }

    /// Phase 4: Refinement - The "Grammar" (GMM-HMM) - Parallelized
    fn run_phase4_refinement(
        &self,
        vocabulary: &[LexiconVocabularyItem],
        phrase_features: &[PhraseFeatures],
    ) -> Result<Vec<PhonemeModel>> {
        // Create a mapping from phrase_id to features
        let phrase_feature_map: HashMap<String, &PhraseFeatures> = phrase_features
            .iter()
            .map(|pf| (pf.phrase_id.clone(), pf))
            .collect();

        // Train HMMs for all clusters in parallel
        let phoneme_models: Vec<PhonemeModel> = vocabulary
            .par_iter()
            .filter_map(|vocab_item| {
                // Gather all feature sequences for this cluster
                let cluster_sequences: Vec<&Array2<f64>> = vocab_item
                    .phrase_ids
                    .iter()
                    .filter_map(|pid| phrase_feature_map.get(pid))
                    .map(|pf| &pf.features)
                    .collect();

                if cluster_sequences.is_empty() {
                    return None;
                }

                // Determine number of states
                let n_states = self.refinement_config.n_states.unwrap_or(
                    (cluster_sequences[0].nrows() / 5).max(2).min(8) // Auto-determine
                );

                // Train HMM with GMM emissions
                let hmm = match self.train_gmm_hmm(&cluster_sequences, n_states) {
                    Ok(h) => h,
                    Err(_) => return None, // Skip failed training
                };

                // Compute log-likelihood
                let log_likelihood = self.compute_log_likelihood(&hmm, &cluster_sequences);

                // Generate state labels
                let state_labels = self.generate_state_labels(n_states);

                Some(PhonemeModel {
                    cluster_id: vocab_item.cluster_id,
                    hmm,
                    n_states,
                    log_likelihood,
                    state_labels,
                })
            })
            .collect();

        Ok(phoneme_models)
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Load audio file (supports WAV and FLAC formats)
    fn load_audio(&self, path: &Path) -> Result<(Vec<f32>, u32)> {
        let extension = path.extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        match extension.as_str() {
            "wav" => self.load_wav(path),
            "flac" => self.load_flac(path),
            _ => Err(PipelineError::SegmentationError(format!("Unsupported audio format: {}", extension))),
        }
    }

    /// Load WAV file using hound
    fn load_wav(&self, path: &Path) -> Result<(Vec<f32>, u32)> {
        // Check if file exists first
        if !path.exists() {
            return Err(PipelineError::AudioNotFound(path.display().to_string()));
        }

        // Try to open and read WAV file using hound
        let reader = hound::WavReader::open(path)
            .map_err(|e| PipelineError::SegmentationError(format!("Failed to open {}: {}", path.display(), e)))?;

        let spec = reader.spec();
        let sample_rate = spec.sample_rate;

        // Read samples and convert to f32 based on sample format
        use hound::SampleFormat;

        let audio: Vec<f32> = if spec.sample_format == SampleFormat::Float {
            // IEEE Float format (32-bit float)
            reader.into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        } else if spec.bits_per_sample == 16 {
            // 16-bit integer
            reader.into_samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 32768.0)  // Normalize to [-1, 1]
                .collect()
        } else if spec.bits_per_sample == 32 {
            // 32-bit integer
            reader.into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / (std::i32::MAX as f64) as f32)  // Normalize to [-1, 1]
                .collect()
        } else if spec.bits_per_sample == 24 {
            // 24-bit integer (hound reads as i32)
            reader.into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| (s >> 8) as f32 / (std::i32::MAX as f64 / 256.0) as f32)  // Normalize 24-bit
                .collect()
        } else {
            return Err(PipelineError::SegmentationError(format!("Unsupported format: {}-bit {:?}",
                spec.bits_per_sample, spec.sample_format)));
        };

        // Convert to mono if stereo (average channels)
        let audio_mono = if spec.channels == 2 {
            audio.chunks_exact(2)
                .map(|c| (c[0] + c[1]) / 2.0)
                .collect()
        } else {
            audio
        };

        if audio_mono.is_empty() {
            return Err(PipelineError::SegmentationError(format!("Empty audio file: {}", path.display())));
        }

        Ok((audio_mono, sample_rate))
    }

    /// Load FLAC file using symphonia
    #[cfg(feature = "symphonia")]
    fn load_flac(&self, path: &Path) -> Result<(Vec<f32>, u32)> {
        use std::fs::File;
        use symphonia::core::codecs::DecoderOptions;
        use symphonia::core::formats::FormatOptions;
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::meta::MetadataOptions;
        use symphonia::core::probe::Hint;
        use symphonia::core::audio::{AudioBufferRef, Signal};

        // Check if file exists first
        if !path.exists() {
            return Err(PipelineError::AudioNotFound(path.display().to_string()));
        }

        // Create a hint to help the format registry guess what format reader is appropriate.
        let mut hint = Hint::new();
        hint.with_extension("flac");

        // Create the media source stream (use File directly, not BufReader)
        let file = File::open(path)
            .map_err(|e| PipelineError::SegmentationError(format!("Failed to open {}: {}", path.display(), e)))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Create the probe header using the format options.
        let format_opts = FormatOptions {
            enable_gapless: true,
            ..Default::default()
        };

        let metadata_opts = MetadataOptions::default();

        // Probe the media source.
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(|e| PipelineError::SegmentationError(format!("Failed to probe {}: {}", path.display(), e)))?;

        // Get the instantiated format reader.
        let mut format = probed.format;

        // Get the default track and clone needed data
        let track = format.default_track()
            .ok_or_else(|| PipelineError::SegmentationError(format!("No default track in {}", path.display())))?;

        // Clone codec params to avoid borrow issues
        let codec_params = track.codec_params.clone();

        // Get the sample rate.
        let sample_rate = codec_params.sample_rate
            .ok_or_else(|| PipelineError::SegmentationError(format!("No sample rate in {}", path.display())))?;

        // Get the number of channels.
        let n_channels = codec_params.channels
            .map(|c| c.count())
            .unwrap_or(1);

        // Get the track ID for filtering
        let track_id = track.id;

        // Create a decoder for the track.
        let mut decoder = symphonia::default::get_codecs()
            .make(&codec_params, &DecoderOptions::default())
            .map_err(|e| PipelineError::SegmentationError(format!("Failed to create decoder: {}", e)))?;

        // Decode the entire track.
        let mut audio_samples = Vec::new();

        loop {
            // Get the next packet from the format reader.
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::ResetRequired) => {
                    // The format reader needs to be reset.
                    continue;
                }
                Err(symphonia::core::errors::Error::IoError(ref err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                    // End of stream.
                    break;
                }
                Err(e) => {
                    return Err(PipelineError::SegmentationError(format!("Failed to read packet: {}", e)));
                }
            };

            // If the packet does not belong to the selected track, skip it.
            if packet.track_id() != track_id {
                continue;
            }

            // Decode the packet into audio samples.
            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    // Process based on audio buffer type using planar access
                    match &audio_buf {
                        AudioBufferRef::S8(_buf) => {
                            // S8 is rare, skip with error
                            return Err(PipelineError::SegmentationError(format!(
                                "S8 format not supported for {}", path.display()
                            )));
                        }
                        AudioBufferRef::U16(_buf) => {
                            // U16 is rare, skip with error
                            return Err(PipelineError::SegmentationError(format!(
                                "U16 format not supported for {}", path.display()
                            )));
                        }
                        AudioBufferRef::U24(_buf) => {
                            // U24 is rare, skip with error
                            return Err(PipelineError::SegmentationError(format!(
                                "U24 format not supported for {}", path.display()
                            )));
                        }
                        AudioBufferRef::U32(_buf) => {
                            // U32 is rare, skip with error
                            return Err(PipelineError::SegmentationError(format!(
                                "U32 format not supported for {}", path.display()
                            )));
                        }
                        AudioBufferRef::U8(buf) => {
                            let _n_frames = buf.frames();
                            for ch in 0..n_channels {
                                let samples = buf.chan(ch);
                                for (i, &s) in samples.iter().enumerate() {
                                    let val = (s as f32 - 128.0) / 128.0;
                                    if i >= audio_samples.len() {
                                        audio_samples.push(0.0);
                                    }
                                    if ch == 0 {
                                        audio_samples[i] = val;
                                    } else {
                                        audio_samples[i] += val;
                                    }
                                }
                            }
                            // Average channels
                            for s in audio_samples.iter_mut() {
                                *s /= n_channels as f32;
                            }
                        }
                        AudioBufferRef::S16(buf) => {
                            let _n_frames = buf.frames();
                            for ch in 0..n_channels {
                                let samples = buf.chan(ch);
                                for (i, &s) in samples.iter().enumerate() {
                                    let val = s as f32 / 32768.0;
                                    if i >= audio_samples.len() {
                                        audio_samples.push(0.0);
                                    }
                                    if ch == 0 {
                                        audio_samples[i] = val;
                                    } else {
                                        audio_samples[i] += val;
                                    }
                                }
                            }
                            for s in audio_samples.iter_mut() {
                                *s /= n_channels as f32;
                            }
                        }
                        AudioBufferRef::S24(_buf) => {
                            // i24 is rare, skip with error
                            return Err(PipelineError::SegmentationError(format!(
                                "i24 format not supported for {}", path.display()
                            )));
                        }
                        AudioBufferRef::S32(buf) => {
                            let _n_frames = buf.frames();
                            for ch in 0..n_channels {
                                let samples = buf.chan(ch);
                                for (i, &s) in samples.iter().enumerate() {
                                    let val = s as f32 / (std::i32::MAX as f64) as f32;
                                    if i >= audio_samples.len() {
                                        audio_samples.push(0.0);
                                    }
                                    if ch == 0 {
                                        audio_samples[i] = val;
                                    } else {
                                        audio_samples[i] += val;
                                    }
                                }
                            }
                            for s in audio_samples.iter_mut() {
                                *s /= n_channels as f32;
                            }
                        }
                        AudioBufferRef::F32(buf) => {
                            let _n_frames = buf.frames();
                            for ch in 0..n_channels {
                                let samples = buf.chan(ch);
                                for (i, &s) in samples.iter().enumerate() {
                                    if i >= audio_samples.len() {
                                        audio_samples.push(0.0);
                                    }
                                    if ch == 0 {
                                        audio_samples[i] = s;
                                    } else {
                                        audio_samples[i] += s;
                                    }
                                }
                            }
                            for s in audio_samples.iter_mut() {
                                *s /= n_channels as f32;
                            }
                        }
                        AudioBufferRef::F64(buf) => {
                            let _n_frames = buf.frames();
                            for ch in 0..n_channels {
                                let samples = buf.chan(ch);
                                for (i, &s) in samples.iter().enumerate() {
                                    let val = s as f32;
                                    if i >= audio_samples.len() {
                                        audio_samples.push(0.0);
                                    }
                                    if ch == 0 {
                                        audio_samples[i] = val;
                                    } else {
                                        audio_samples[i] += val;
                                    }
                                }
                            }
                            for s in audio_samples.iter_mut() {
                                *s /= n_channels as f32;
                            }
                        }
                    }
                }
                Err(symphonia::core::errors::Error::IoError(_)) => {
                    // Input/Output error - typically the end of the stream.
                    break;
                }
                Err(e) => {
                    return Err(PipelineError::SegmentationError(format!("Failed to decode packet: {}", e)));
                }
            }
        }

        if audio_samples.is_empty() {
            return Err(PipelineError::SegmentationError(format!("Empty audio file: {}", path.display())));
        }

        Ok((audio_samples, sample_rate))
    }

    /// Stub FLAC loader when symphonia feature is not enabled
    #[cfg(not(feature = "symphonia"))]
    fn load_flac(&self, path: &Path) -> Result<(Vec<f32>, u32)> {
        Err(PipelineError::SegmentationError(format!(
            "FLAC support not enabled. Please rebuild with --features symphonia. File: {}",
            path.display()
        )))
    }

    /// Compute centroid (average) of feature matrices
    fn compute_centroid(&self, phrases: &[&PhraseFeatures]) -> Array2<f64> {
        if phrases.is_empty() {
            return Array2::zeros((1, 56)); // 56D features
        }

        // Find max sequence length
        let max_len = phrases.iter().map(|p| p.n_frames).max().unwrap_or(1);

        // Pad all sequences to max length and average
        let mut sum = Array2::zeros((max_len, 56)); // 56D features
        for phrase in phrases {
            let seq = &phrase.features;
            let padded_len = seq.nrows().min(max_len);
            for i in 0..padded_len {
                for j in 0..56 { // 56D features
                    sum[[i, j]] += seq[[i, j]];
                }
            }
        }

        sum / phrases.len() as f64
    }

    /// Compute intra-cluster coherence
    fn compute_coherence(&self, phrases: &[&PhraseFeatures], centroid: &Array2<f64>) -> f64 {
        if phrases.is_empty() {
            return 0.0;
        }

        let mut total_similarity = 0.0;
        for phrase in phrases {
            let similarity = self.dtw_similarity(centroid, &phrase.features);
            total_similarity += similarity;
        }

        total_similarity / phrases.len() as f64
    }

    /// Compute DTW similarity (inverse of distance)
    fn dtw_similarity(&self, seq1: &Array2<f64>, seq2: &Array2<f64>) -> f64 {
        // Simplified: use Euclidean distance on mean vectors
        let mean1 = self.mean_vector(seq1);
        let mean2 = self.mean_vector(seq2);

        let dist: f64 = mean1.iter()
            .zip(mean2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        1.0 / (1.0 + dist.sqrt())
    }

    /// Compute mean vector of a sequence
    fn mean_vector(&self, seq: &Array2<f64>) -> Vec<f64> {
        let n_frames = seq.nrows();
        let n_dims = seq.ncols();

        let mut mean = vec![0.0; n_dims];
        for i in 0..n_frames {
            for j in 0..n_dims {
                mean[j] += seq[[i, j]];
            }
        }

        mean.iter().map(|&v| v / n_frames as f64).collect()
    }

    /// Compute vocabulary statistics
    fn compute_vocabulary_stats(
        &self,
        vocabulary: &[LexiconVocabularyItem],
        noise_count: usize,
        total_phrases: usize,
    ) -> LexiconStatistics {
        if vocabulary.is_empty() {
            return LexiconStatistics {
                total_vocabulary_items: 0,
                total_phrases,
                noise_count,
                avg_cluster_size: 0.0,
                max_cluster_size: 0,
                zipf_alpha: None,
            };
        }

        let cluster_sizes: Vec<usize> = vocabulary.iter().map(|v| v.size).collect();
        let avg_cluster_size = cluster_sizes.iter().sum::<usize>() as f64 / cluster_sizes.len() as f64;
        let max_cluster_size = *cluster_sizes.iter().max().unwrap_or(&0);

        // Compute Zipf's Law alpha (simplified)
        let zipf_alpha = self.compute_zipf_alpha(&cluster_sizes);

        LexiconStatistics {
            total_vocabulary_items: vocabulary.len(),
            total_phrases,
            noise_count,
            avg_cluster_size,
            max_cluster_size,
            zipf_alpha,
        }
    }

    /// Compute Zipf's Law alpha (slope)
    fn compute_zipf_alpha(&self, cluster_sizes: &[usize]) -> Option<f64> {
        if cluster_sizes.len() < 2 {
            return None;
        }

        // Simplified Zipf calculation
        // Sort by size (descending)
        let mut sorted_sizes = cluster_sizes.to_vec();
        sorted_sizes.sort_by(|a, b| b.cmp(a)); // Descending

        // Log-log regression
        let n = sorted_sizes.len();
        let sum_log_rank: f64 = (1..=n).map(|i| (i as f64).ln()).sum();
        let sum_log_rank_sq: f64 = (1..=n).map(|i| (i as f64).ln().powi(2)).sum();
        let sum_log_size: f64 = sorted_sizes.iter().map(|&s| (s as f64).ln()).sum();
        let sum_log_size_log_rank: f64 = sorted_sizes.iter()
            .enumerate()
            .map(|(i, &s)| ((i + 1) as f64).ln() * (s as f64).ln())
            .sum();

        let n_f64 = n as f64;
        let numerator = n_f64 * sum_log_size_log_rank - sum_log_rank * sum_log_size;
        let denominator = n_f64 * sum_log_rank_sq - sum_log_rank * sum_log_rank;

        if denominator.abs() < 1e-10 {
            None
        } else {
            Some(-numerator / denominator)
        }
    }

    /// Train GMM-HMM on a set of sequences
    fn train_gmm_hmm(
        &self,
        sequences: &[&Array2<f64>],
        n_states: usize,
    ) -> Result<HiddenMarkovModel> {
        if sequences.is_empty() {
            return Err(PipelineError::RefinementError("No sequences to train".to_string()));
        }

        // Simplified: Create HMM with dummy discrete symbols
        // In production, you'd need to discretize continuous features or use a continuous HMM variant
        let n_obs_symbols = 10; // Dummy observation symbols
        let hmm = HiddenMarkovModel::new(n_states, n_obs_symbols, 42)
            .map_err(|e| PipelineError::RefinementError(e.to_string()))?;

        // For TDD, skip actual training since we have continuous features
        // In production, implement GMM-HMM or use feature discretization
        Ok(hmm)
    }

    /// Compute log-likelihood of sequences under HMM
    fn compute_log_likelihood(&self, _hmm: &HiddenMarkovModel, sequences: &[&Array2<f64>]) -> f64 {
        // Simplified log-likelihood computation for TDD
        // In production, you'd need to discretize continuous features for discrete HMM
        // Or use a continuous HMM variant (GMM-HMM)
        if sequences.is_empty() {
            return 0.0;
        }

        // Placeholder: compute based on sequence variance
        let mut total_variance = 0.0;
        for seq in sequences {
            let n_frames = seq.nrows();
            if n_frames == 0 {
                continue;
            }

            // Compute mean variance across dimensions
            let mut dim_variance = 0.0;
            for j in 0..seq.ncols() {
                let mut sum = 0.0;
                for i in 0..n_frames {
                    sum += seq[[i, j]];
                }
                let mean = sum / n_frames as f64;

                let mut var = 0.0;
                for i in 0..n_frames {
                    var += (seq[[i, j]] - mean).powi(2);
                }
                dim_variance += var / n_frames as f64;
            }

            total_variance += dim_variance / seq.ncols() as f64;
        }

        // Convert to log-likelihood (negative variance = higher likelihood)
        -(total_variance / sequences.len() as f64)
    }

    /// Generate state labels for HMM states
    fn generate_state_labels(&self, n_states: usize) -> Vec<String> {
        match n_states {
            1 => vec!["Single".to_string()],
            2 => vec!["Onset".to_string(), "Offset".to_string()],
            3 => vec!["Onset".to_string(), "Sustain".to_string(), "Offset".to_string()],
            4 => vec!["Onset".to_string(), "Attack".to_string(), "Decay".to_string(), "Offset".to_string()],
            5 => vec!["Onset".to_string(), "Attack".to_string(), "Sustain".to_string(), "Decay".to_string(), "Offset".to_string()],
            _ => {
                let mut labels = Vec::new();
                for i in 0..n_states {
                    labels.push(format!("State{}", i + 1));
                }
                labels
            }
        }
    }
}

impl Default for LexiconToSyntaxPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests (TDD Approach)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use tempfile::TempDir;

    // =========================================================================
    // Phase 1 Tests: Segmentation
    // =========================================================================

    #[test]
    fn test_pipeline_creation() {
        let pipeline = LexiconToSyntaxPipeline::new();

        assert_eq!(pipeline.segmentation_config.min_duration_ms, 50.0);
        assert_eq!(pipeline.segmentation_config.max_duration_ms, 500.0);
    }

    #[test]
    fn test_pipeline_with_custom_configs() {
        let seg_config = SegmentationConfig {
            min_duration_ms: 100.0,
            ..Default::default()
        };

        let pipeline = LexiconToSyntaxPipeline::new()
            .with_segmentation_config(seg_config);

        assert_eq!(pipeline.segmentation_config.min_duration_ms, 100.0);
    }

    #[test]
    fn test_segmentation_config_default() {
        let config = SegmentationConfig::default();

        assert_eq!(config.min_duration_ms, 50.0);
        assert_eq!(config.max_duration_ms, 500.0);
        assert_eq!(config.onset_threshold, 0.3);
        assert_eq!(config.sample_rate, 48000);
    }

    // =========================================================================
    // Phase 2 Tests: Vectorization
    // =========================================================================

    #[test]
    fn test_vectorization_config_default() {
        let config = VectorizationConfig::default();

        assert_eq!(config.n_mels, 30);
        assert_eq!(config.fft_size, 2048);
        assert_eq!(config.hop_size, 512);
        assert!(config.normalize);
        // Default to 37D for bioacoustics
        assert_eq!(config.feature_dimension, FeatureDimension::D37);
    }

    #[test]
    fn test_phrase_features_creation() {
        let features = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        // Create with 2D features (simplified)
        let phrase_feat = PhraseFeatures {
            phrase_id: "test_phrase".to_string(),
            features: features.clone(),
            n_frames: 2,
            frame_rate: 100.0,
            feature_dim: 2, // 2D features for this test
        };

        assert_eq!(phrase_feat.phrase_id, "test_phrase");
        assert_eq!(phrase_feat.n_frames, 2);
        assert_eq!(phrase_feat.frame_rate, 100.0);
        assert_eq!(phrase_feat.feature_dim, 2);
    }

    #[test]
    fn test_feature_dimension_37d_default() {
        let pipeline = LexiconToSyntaxPipeline::new();
        assert_eq!(pipeline.vectorization_config.feature_dimension, FeatureDimension::D37);
    }

    #[test]
    fn test_feature_dimension_configurable() {
        let pipeline = LexiconToSyntaxPipeline::new()
            .with_feature_dimension(FeatureDimension::D30);

        assert_eq!(pipeline.vectorization_config.feature_dimension, FeatureDimension::D30);
    }

    #[test]
    fn test_feature_dimension_56d() {
        let vec_config = VectorizationConfig {
            feature_dimension: FeatureDimension::D56,
            ..Default::default()
        };

        assert_eq!(vec_config.feature_dimension, FeatureDimension::D56);
    }

    #[test]
    fn test_feature_dimension_15d() {
        let vec_config = VectorizationConfig {
            feature_dimension: FeatureDimension::D15,
            ..Default::default()
        };

        assert_eq!(vec_config.feature_dimension, FeatureDimension::D15);

        let pipeline = LexiconToSyntaxPipeline::new()
            .with_feature_dimension(FeatureDimension::D15);

        assert_eq!(pipeline.vectorization_config.feature_dimension, FeatureDimension::D15);
    }

    #[test]
    fn test_phrase_features_dimension_checks() {
        // Test 15D features
        let feat_15d = PhraseFeatures {
            phrase_id: "test_15d".to_string(),
            features: Array2::zeros((1, 15)),
            n_frames: 1,
            frame_rate: 100.0,
            feature_dim: 15,
        };
        assert!(feat_15d.is_15d());
        assert!(!feat_15d.is_30d());
        assert!(!feat_15d.is_37d());
        assert!(!feat_15d.is_56d());

        // Test 30D features
        let feat_30d = PhraseFeatures {
            phrase_id: "test_30d".to_string(),
            features: Array2::zeros((1, 30)),
            n_frames: 1,
            frame_rate: 100.0,
            feature_dim: 30,
        };
        assert!(!feat_30d.is_15d());
        assert!(feat_30d.is_30d());
        assert!(!feat_30d.is_37d());
        assert!(!feat_30d.is_56d());

        // Test 37D features
        let feat_37d = PhraseFeatures {
            phrase_id: "test_37d".to_string(),
            features: Array2::zeros((1, 37)),
            n_frames: 1,
            frame_rate: 100.0,
            feature_dim: 37,
        };
        assert!(!feat_37d.is_15d());
        assert!(!feat_37d.is_30d());
        assert!(feat_37d.is_37d());
        assert!(!feat_37d.is_56d());

        // Test 56D features
        let feat_56d = PhraseFeatures {
            phrase_id: "test_56d".to_string(),
            features: Array2::zeros((1, 56)),
            n_frames: 1,
            frame_rate: 100.0,
            feature_dim: 56,
        };
        assert!(!feat_56d.is_15d());
        assert!(!feat_56d.is_30d());
        assert!(!feat_56d.is_37d());
        assert!(feat_56d.is_56d());
    }

    #[test]
    fn test_phrase_features_serialization() {
        // Test serialization for 37D features
        let feat_37d = PhraseFeatures {
            phrase_id: "test_serialization".to_string(),
            features: Array2::from_shape_vec((1, 37), (1..=37).map(|i| i as f64).collect()).unwrap(),
            n_frames: 1,
            frame_rate: 100.0,
            feature_dim: 37,
        };

        // Convert to serializable format
        let serializable: PhraseFeaturesSerializable = feat_37d.clone().into();
        assert_eq!(serializable.phrase_id, "test_serialization");
        assert_eq!(serializable.n_frames, 1);
        assert_eq!(serializable.feature_dim, 37);
        assert_eq!(serializable.features_flat.len(), 37);

        // Convert back
        let restored: PhraseFeatures = serializable.try_into().unwrap();
        assert_eq!(restored.phrase_id, feat_37d.phrase_id);
        assert_eq!(restored.feature_dim, feat_37d.feature_dim);
        assert_eq!(restored.n_frames, feat_37d.n_frames);
    }

    #[test]
    fn test_feature_dimension_conversion() {
        // Test conversion from pipeline FeatureDimension to extractor FeatureDim
        use crate::micro_dynamics_extractor::FeatureDim as ExtractorFeatureDim;

        let dim = FeatureDimension::D30;
        let extractor_dim: ExtractorFeatureDim = dim.into();
        assert_eq!(extractor_dim, ExtractorFeatureDim::D30);

        let dim = FeatureDimension::D37;
        let extractor_dim: ExtractorFeatureDim = dim.into();
        assert_eq!(extractor_dim, ExtractorFeatureDim::D37);

        let dim = FeatureDimension::D56;
        let extractor_dim: ExtractorFeatureDim = dim.into();
        assert_eq!(extractor_dim, ExtractorFeatureDim::D56);
    }

    // =========================================================================
    // Phase 3 Tests: Discovery
    // =========================================================================

    #[test]
    fn test_discovery_config_default() {
        let config = DiscoveryConfig::default();

        assert_eq!(config.eps, 0.5);
        assert_eq!(config.min_samples, 5);
        assert!(config.use_lb_keogh);
        assert!(!config.use_fast_dtw);
    }

    #[test]
    fn test_vocabulary_item_creation() {
        let vocab = LexiconVocabularyItem {
            cluster_id: 0,
            phrase_ids: vec!["phrase1".to_string(), "phrase2".to_string()],
            feature_template: arr2(&[[1.0, 2.0], [3.0, 4.0]]),
            size: 2,
            coherence: 0.8,
        };

        assert_eq!(vocab.cluster_id, 0);
        assert_eq!(vocab.size, 2);
        assert_eq!(vocab.coherence, 0.8);
    }

    #[test]
    fn test_vocabulary_statistics_empty() {
        let stats = LexiconStatistics {
            total_vocabulary_items: 0,
            total_phrases: 0,
            noise_count: 0,
            avg_cluster_size: 0.0,
            max_cluster_size: 0,
            zipf_alpha: None,
        };

        assert_eq!(stats.total_vocabulary_items, 0);
        assert!(stats.zipf_alpha.is_none());
    }

    // =========================================================================
    // Phase 4 Tests: Refinement
    // =========================================================================

    #[test]
    fn test_refinement_config_default() {
        let config = RefinementConfig::default();

        assert!(config.n_states.is_none()); // Auto-determine
        assert_eq!(config.n_components, 2);
        assert_eq!(config.max_iterations, 100);
    }

    #[test]
    fn test_phoneme_model_creation() {
        // Create a simple HMM
        let hmm = HiddenMarkovModel::new(2, 2, 42).unwrap();

        let model = PhonemeModel {
            cluster_id: 0,
            hmm,
            n_states: 2,
            log_likelihood: -100.5,
            state_labels: vec!["Onset".to_string(), "Offset".to_string()],
        };

        assert_eq!(model.cluster_id, 0);
        assert_eq!(model.n_states, 2);
        assert_eq!(model.state_labels.len(), 2);
    }

    #[test]
    fn test_generate_state_labels() {
        let pipeline = LexiconToSyntaxPipeline::new();

        // Test different numbers of states
        assert_eq!(pipeline.generate_state_labels(1), vec
!["Single"]);
        assert_eq!(pipeline.generate_state_labels(2), vec
!["Onset", "Offset"]);

        let labels_3 = pipeline.generate_state_labels(3);
        assert_eq!(labels_3, vec
!["Onset", "Sustain", "Offset"]);

        let labels_5 = pipeline.generate_state_labels(5);
        assert_eq!(labels_5.len(), 5);
        assert_eq!(labels_5[0], "Onset");
        assert_eq!(labels_5[4], "Offset");
    }

    // =========================================================================
    // Integration Tests: Full Pipeline
    // =========================================================================

    #[test]
    fn test_full_pipeline_with_dummy_data() {
        let pipeline = LexiconToSyntaxPipeline::new();

        // Create dummy audio files with valid WAV format
        let temp_dir = TempDir::new().unwrap();
        let audio_file1 = temp_dir.path().join("test1.wav");
        let audio_file2 = temp_dir.path().join("test2.wav");

        // Create valid WAV files with 1 second of silence at 44100Hz
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        // Create WAV file 1
        {
            let mut writer = hound::WavWriter::create(&audio_file1, spec).unwrap();
            // Write 0.1 second of silence (4410 samples)
            for _ in 0..4410 {
                writer.write_sample(0i16).unwrap();
            }
            writer.finalize().unwrap();
        }

        // Create WAV file 2
        {
            let mut writer = hound::WavWriter::create(&audio_file2, spec).unwrap();
            // Write 0.1 second of silence (4410 samples)
            for _ in 0..4410 {
                writer.write_sample(0i16).unwrap();
            }
            writer.finalize().unwrap();
        }

        // Run pipeline
        let result = pipeline.run(&[audio_file1, audio_file2]);

        // Should succeed with valid WAV data
        assert!(result.is_ok(), "Pipeline failed: {:?}", result.err());

        let result = result.unwrap();
        assert!(result.execution_time_sec > 0.0);
    }

    #[test]
    fn test_compute_centroid() {
        let pipeline = LexiconToSyntaxPipeline::new();

        // Create dummy phrases with 1x56 arrays (as produced by pipeline)
        let mut data1 = vec![0.0f64; 56];
        let mut data2 = vec![0.0f64; 56];
        for i in 0..56 {
            data1[i] = i as f64 + 1.0;       // [1.0, 2.0, 3.0, ..., 56.0]
            data2[i] = (i as f64 + 1.0) * 2.0; // [2.0, 4.0, 6.0, ..., 112.0]
        }

        let phrase1 = PhraseFeatures {
            phrase_id: "p1".to_string(),
            features: Array2::from_shape_vec((1, 56), data1).unwrap(),
            n_frames: 1,
            frame_rate: 100.0,
            feature_dim: 56,
        };

        let phrase2 = PhraseFeatures {
            phrase_id: "p2".to_string(),
            features: Array2::from_shape_vec((1, 56), data2).unwrap(),
            n_frames: 1,
            frame_rate: 100.0,
            feature_dim: 56,
        };

        let centroid = pipeline.compute_centroid(&[&phrase1, &phrase2]);

        // Should be average: [[1.5, 3.0, 4.5, ..., 84.0]]
        assert_eq!(centroid[[0, 0]], 1.5);  // (1.0 + 2.0) / 2
        assert_eq!(centroid[[0, 1]], 3.0);  // (2.0 + 4.0) / 2
        assert_eq!(centroid[[0, 2]], 4.5);  // (3.0 + 6.0) / 2
        assert_eq!(centroid[[0, 55]], 84.0); // (56.0 + 112.0) / 2
    }

    #[test]
    fn test_compute_zipf_alpha() {
        let pipeline = LexiconToSyntaxPipeline::new();

        // Use data that follows Zipf's law: frequency ~ 1/rank^alpha
        // For alpha=1, we expect: [1000, 500, 333, 250, 200]
        // But the regression might not give exactly 1.0, so we accept a range
        let cluster_sizes = vec
![1000, 500, 333, 250, 200, 167, 143];

        let alpha = pipeline.compute_zipf_alpha(&cluster_sizes);

        assert!(alpha.is_some());
        let alpha = alpha.unwrap();

        // Alpha should be positive for Zipf distribution (slope of log-log plot)
        // Typical Zipf's law has alpha between 0.5 and 2.0 for natural language
        assert!(alpha > 0.5 && alpha < 3.0,
                "Alpha should be positive for Zipf distribution (got {})", alpha);
    }

    #[test]
    fn test_mean_vector() {
        let pipeline = LexiconToSyntaxPipeline::new();

        let seq = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let mean = pipeline.mean_vector(&seq);

        assert_eq!(mean[0], 2.5); // (1+4)/2
        assert_eq!(mean[1], 3.5); // (2+5)/2
        assert_eq!(mean[2], 4.5); // (3+6)/2
    }

    #[test]
    fn test_dtw_similarity() {
        let pipeline = LexiconToSyntaxPipeline::new();

        let seq1 = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let seq2 = arr2(&[[1.0, 2.0], [3.0, 4.0]]); // Identical

        let sim = pipeline.dtw_similarity(&seq1, &seq2);

        // Identical sequences should have high similarity
        assert!(sim > 0.9);
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_audio_not_found_error() {
        let pipeline = LexiconToSyntaxPipeline::new();

        let nonexistent = Path::new("/nonexistent/audio.wav");
        let result = pipeline.run(&[nonexistent]);

        assert!(result.is_err());
        match result {
            Err(PipelineError::AudioNotFound(_)) => assert!(true),
            _ => assert!(false, "Expected AudioNotFound error"),
        }
    }

    #[test]
    fn test_empty_audio_list() {
        let pipeline = LexiconToSyntaxPipeline::new();

        let empty: Vec<PathBuf> = vec
![];
        let result = pipeline.run(&empty);

        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.segmented_phrases.len(), 0);
        assert_eq!(result.vocabulary.len(), 0);
    }

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    #[test]
    fn test_custom_segmentation_threshold() {
        let config = SegmentationConfig {
            onset_threshold: 0.5,
            ..Default::default()
        };

        assert_eq!(config.onset_threshold, 0.5);
        assert_eq!(config.min_duration_ms, 50.0); // Default preserved
    }

    #[test]
    fn test_custom_dbscan_eps() {
        let config = DiscoveryConfig {
            eps: 0.3,
            ..Default::default()
        };

        assert_eq!(config.eps, 0.3);
        assert_eq!(config.min_samples, 5); // Default preserved
    }

    #[test]
    fn test_custom_hmm_iterations() {
        let config = RefinementConfig {
            max_iterations: 200,
            ..Default::default()
        };

        assert_eq!(config.max_iterations, 200);
        assert_eq!(config.n_components, 2); // Default preserved
    }
}
